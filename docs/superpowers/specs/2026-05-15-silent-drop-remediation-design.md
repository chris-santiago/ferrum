# Silent-Drop Remediation Design Spec

**Date:** 2026-05-15 (revised same day — scope expanded)
**Scope:** Groups 2–4 from the 2026-05-15 code archaeology report — encoding fields never read by Rust, mark kwargs that raise at runtime, and encoding channels unrendered in SVG and WASM. Group 1 (mark-kwargs) handled separately.

---

## 1. Scope

Complete the deferred API surface for: (a) five `EncodingSpec` fields (`sort`, `stack`, `axis`, `format_type`, `impute`) that are accepted and serialized but silently ignored by Rust; (b) legend kwargs beyond `disabled`; (c) three mark desugar paths (`histogram multiple=`, `density multiple=`, `lmplot truncate=False`) that raise instead of rendering; (d) two data-routing gaps (`Chart(data=None)`, `Layer(data=...)` via `.layer()`); (e) four stroke/angle encoding channels unrendered in both SVG and WASM; and (f) `mark_raster(blend="additive")` WASM GPU path. Items are ordered static-SVG first, interactive-WASM second.

---

## 2. Goals

**Static SVG**
- `X("col", sort="descending")` / `sort="ascending"` / `sort=[...]` (explicit list) produces correctly ordered axis ticks and data.
- `Y("val", stack="normalize")` / `"center"` / `"zero"` produces stacked bar/area charts.
- `axis={"title": ..., "ticks": False, "label_angle": -45, ...}` dict properties flow through to the rendered axis.
- `format_type="number"` / `"time"` controls tick-label formatter selection.
- `Y("val", impute={"method": "value", "value": 0})` fills missing values before rendering.
- `Color("col", legend={"orient": "bottom", "title": "...", "format": ".2f", ...})` — the documented legend property set affects legend rendering.
- `mark_histogram(multiple="stack"/"fill"/"dodge")` and `mark_density(multiple="dodge")` route through Stack/Dodge transforms.
- `lmplot(truncate=False)` / `regplot(truncate=False)` extends the fit line to the plot-axis boundary.
- `Chart(data=None)` with per-layer `data=` and `Layer(data=df)` via `.layer()` are both accepted.
- `encode(stroke_width="col")`, `encode(stroke_opacity="col")`, `encode(stroke_dash="col")`, `encode(angle="col")` apply per-row values as SVG element attributes on all mark kinds that support them.

**Interactive WASM**
- The four stroke/angle channels are also applied per-row in the WASM GPU renderer.
- `mark_raster(blend="additive")` uses GPU additive compositing (src + dst) in the WASM renderer.

---

## 3. Non-goals

- Full `Axis(...)` Python value class — dict passthrough only (see §6).

---

## 4. System behavior

**`sort=`** Ordinal/nominal X/Y scales consult `EncodingSpec.sort` to order their domain. String values: `"ascending"`, `"descending"`, a field name (sort by another field's values), `"-field"` (descending by field). List value: explicit domain order — `["b", "a", "c"]` forces that category sequence. Quantitative scales ignore `sort=`.

**`stack=`** Inserts the Stack position-adjustment transform with the named strategy. `"zero"` starts from zero; `"normalize"` normalises each group to [0, 1]; `"center"` centres around zero. Valid only on bar and area marks — other marks raise `ValueError` at desugar time.

**`axis=`** Dict of display properties forwarded to the Rust axis layout engine. Only the keys listed in §6 are honoured; all others are silently accepted with no effect.

**`format_type=`** Selects the tick-label formatter family: `"number"` → numeric; `"time"` → temporal. If unset, inferred from the field's data type.

**`impute=`** Dict with `method` key and optional `value` key. Before the mark renderer runs, missing values in the encoded column are filled according to the method: `"value"` (constant `value`), `"mean"`, `"median"`, `"max"`, `"min"`. Operates on the column named by `field=` on the encoding. Primary use case: filling gaps in time-series lines.

**Legend kwargs** `Color("col", legend={...})` — a dict of display properties forwarded to the Rust legend renderer. Accepted keys listed in §6. Keys outside that set are silently accepted with no effect.

**`histogram(multiple=)` / `density(multiple=)`** `"layer"` (default) overlaps; `"stack"` inserts Stack on the count/density y-axis; `"fill"` inserts Stack with `normalize`; `"dodge"` inserts Dodge on the bin x-axis. Python desugar responsibility.

**`lmplot/regplot(truncate=False)`** `SmoothSpec`/`RobustSpec` accept `x_range: Option<[f64; 2]>`. When `truncate=False`, Python desugar sets `x_range` to the chart's x-scale domain. The Rust transform evaluates the fit line over that range instead of `[xs.min(), xs.max()]`.

**`Chart(data=None)` / `Layer(data=df)` via `.layer()`** `Chart(data=None)` is accepted; per-layer `data=` requirement is enforced at `to_spec()` time. `Chart.layer()` accepts `Layer` objects with `data=` attributes; produces the same internal representation as the `+` operator.

**Stroke/angle channels — SVG** `stroke_width`, `stroke_opacity`, `stroke_dash`, and `angle` field-driven encodings emit per-element SVG attributes (`stroke-width`, `stroke-opacity`, `stroke-dasharray`, `transform="rotate(N)"`) on the marks that support them. `stroke_dash` maps to a `stroke-dasharray` value from a fixed palette (see §6). `angle` is in degrees, applied as a rotation around the mark's anchor point.

**Stroke/angle channels — WASM** The same four channels are packed into per-instance GPU buffers for Circle and Rect mark kinds. `stroke_dash` uses a palette index (same palette as SVG). `angle` rotates around the instance anchor.

**`blend="additive"` WASM** The WASM render pipeline selects an additive blend state (`src + dst`, no alpha attenuation) for raster mark batches with `blend=additive`. SVG path already works via `mix-blend-mode:screen`.

---

## 5. Architecture

**`sort=`** Resolved in `scale_resolve.rs` ordinal domain builder: string values map to sort comparators; list values set the domain directly. `EncodingSpec.sort` is already deserialized.

**`stack=`** Read in `position.rs`; `EncodingSpec.stack` already deserialized. Stack strategy enum maps directly to existing `StackStrategy` variants.

**`axis=`, `format_type=`** `axis=` dict threaded from `EncodingSpec.axis` through `AxisLayout` to `marks/axis.rs`. `format_type=` selects the formatter branch in `render/format.rs`.

**`impute=`** New `Impute` Rust transform added to `crates/ferrum-core/src/transform/`. Inserted into the transform pipeline when `EncodingSpec.impute` is present, before mark rendering. `EncodingSpec.impute` is already deserialized as an opaque value; the pipeline detects and deserializes it to `ImputeSpec`.

**Legend kwargs** `LegendSpec` in `crates/ferrum-core/src/spec/` gains fields for the documented property set. The legend layout/render path in `marks/legend.rs` reads them.

**Histogram/density `multiple=`** Python desugar constructs `PositionAdjustment::Stack` or `::Dodge` and sets it on the layer's `position` field. Rust pipeline unchanged.

**`lmplot truncate=False`** `SmoothSpec`/`RobustSpec` gain `x_range: Option<[f64; 2]>`. Python sets it from the x-scale domain when `truncate=False`. Rust transform uses it as the evaluation domain.

**Data routing** `_coerce.py` relaxes `Chart(data=None)` to a deferred check. `Chart.layer()` extended to accept `Layer` with `data=`; mirrors `__add__` path.

**SVG stroke/angle channels** Mark renderers (`point.rs`, `bar.rs`, `line.rs`, `rule.rs`, etc.) read `stroke_width`/`stroke_opacity`/`stroke_dash`/`angle` columns from the batch. Per-element SVG attributes emitted inline. These channels are removed from `_SILENT_CHANNELS` when the SVG path is wired (they remain silent for mark kinds that don't support them, e.g. area fill).

**WASM stroke/angle + blend** `CircleInstance`/`RectInstance` gain four fields; `scene_load.rs` batch-builder populates them. WebGPU vertex shader consumes them. For `blend="additive"`, the render pipeline has a second pipeline state for additive blend; selected per-batch via `batch.blend_mode`.

---

## 6. Canonical interfaces / data contracts

**`axis=` accepted keys**
```
title: str | None
ticks: bool
tick_count: int
grid: bool
labels: bool
label_angle: float
orient: "top" | "bottom" | "left" | "right"
```

**`legend=` accepted keys**
```
title: str | None
orient: "left" | "right" | "top" | "bottom" | "top-left" | "top-right" | "bottom-left" | "bottom-right" | "none"
direction: "vertical" | "horizontal"
type: "symbol" | "gradient"
tick_count: int
values: list            # explicit tick values
format: str             # tick label format string
label_font_size: float
columns: int            # multi-column legend
gradient_length: float
gradient_thickness: float
```

**`ImputeSpec`**
```
method: "value" | "mean" | "median" | "max" | "min"
value: float | None     # required when method == "value"
```

**`sort=` accepted values**
`"ascending"`, `"descending"`, field name string, `"-field"` (descending by field), or a list of domain values. Any other value is `ValueError`.

**`SmoothSpec` / `RobustSpec` addition**
```rust
pub x_range: Option<[f64; 2]>,
```

**`stroke_dash` palette** (applies to both SVG `stroke-dasharray` and WASM palette index)
```
0 → solid        (no dash)
1 → dashed       "6,3"
2 → dotted       "2,3"
3 → dash-dot     "6,3,2,3"
```
Integer column values map to palette indices; out-of-range values clamp to nearest.

**`stack=` accepted values** `"zero"`, `"normalize"`, `"center"`, `None`. Any other value is `ValueError` at desugar time.

---

## 7. Invariants and constraints

- No silent drops after this work: every accepted kwarg either produces a visual effect or raises `ValueError` with an actionable message.
- `stack=` on a non-bar/non-area mark raises `ValueError` at desugar time.
- `impute=` with `method="value"` and no `value` key raises `ValueError`.
- `lmplot(truncate=False)` clips at the x-scale domain boundary — never extrapolates past it.
- `Chart(data=None)` raises `ValueError` at `to_spec()` time if any layer lacks a data source.
- WASM and SVG `stroke_dash` share the same four-entry palette so cross-renderer output is consistent.
- Stroke/angle channels remain in `_SILENT_CHANNELS` for mark kinds that don't emit a per-element stroke (e.g. area fill) — the silence is intentional, not a gap.

---

## 8. Key decisions and tradeoffs

**`sort=` resolved at scale time, not layout time.** Domain ordering is a scale property; both axes and marks see the sorted domain without duplicating logic.

**`impute=` is a pre-render transform, not a Python-side fill.** Imputation logic belongs in the Rust pipeline so the filled batch is used consistently by all layers and transforms that read the same field.

**`stack=`/`multiple=` desugar is Python's responsibility.** Rust position-adjustment pipeline is already correct; Python constructs the right spec.

**Legend kwargs as dict passthrough, not a Python `Legend(...)` class.** Same rationale as `axis=` — reversible, lower maintenance surface.

**Stroke/angle channels emit per-element SVG attributes, not inline `style=`.** Explicit attributes (`stroke-width="N"`) are more composable with CSS and more inspectable than an inline style string.

**`stroke_dash` shared palette for SVG and WASM.** Cross-renderer consistency is more important than per-renderer flexibility. Four patterns cover practical use cases.

**`blend="additive"` WASM uses a second render pipeline state, not a post-process pass.** WebGPU additive blending is a render-pipeline property, not a fragment shader operation. Creating a second pipeline state is the canonical GPU approach.

**`x_range` on Smooth/Robust, not a chart-level clip.** A chart-level clip would also clip the scatter layer.

---

## 9. Acceptance criteria

- `X("cat", sort=["b", "a", "c"])` → SVG axis ticks appear in that exact order.
- `Y("val", stack="normalize")` on a grouped bar chart → each bar group fills 0–1.
- `X("date", axis={"label_angle": -45})` → tick labels have `transform="rotate(-45)"`.
- `Y("val", format_type="number")` on a date-typed field → ticks formatted as numbers.
- `Y("y", impute={"method": "value", "value": 0})` on a sparse time series → no gaps in the rendered line.
- `Color("col", legend={"orient": "bottom", "direction": "horizontal"})` → legend appears at bottom with horizontal layout.
- `mark_histogram(multiple="dodge")` → bins are side-by-side; `multiple="stack"` → bins are stacked.
- `lmplot(truncate=False)` → fit-line path extends to the x-axis boundary.
- `Chart(data=None).layer(Layer(data=df1, ...), Layer(data=df2, ...))` → both layers render.
- `encode(stroke_width="col")` on a line chart → each segment has a distinct `stroke-width` attribute in SVG.
- `encode(stroke_dash="col")` with integer values 0–3 → elements use the corresponding `stroke-dasharray` values.
- WASM: `encode(stroke_opacity="col")` → `CircleInstance.stroke_opacity` populated per-row.
- WASM: `mark_raster(blend="additive")` → additive blend state selected in the render pipeline.

---

## 10. Validation strategy

Static SVG: Python integration tests calling `show_svg()` asserting SVG structure — tick order, bar heights, attribute presence/values.

`impute=`: assert no `null`/`NaN` elements appear in the rendered path after imputation.

Legend kwargs: assert rendered legend `<g>` transform or text attributes match the specified properties.

WASM stroke/angle: Rust unit tests asserting `CircleInstance` / `RectInstance` fields populated from column values.

WASM blend: Rust unit test asserting the correct `wgpu::BlendState` is selected for an additive-blend raster batch.

---

## 11. Open questions

- **`stroke_dash` palette:** The four entries in §6 are proposed. If the implementation finds these insufficient, the palette may be expanded to 8 without a spec revision — the contract is the index-to-pattern mapping, not the count.
