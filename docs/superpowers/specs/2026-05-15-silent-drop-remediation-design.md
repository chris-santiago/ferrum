# Silent-Drop Remediation Design Spec

**Date:** 2026-05-15
**Scope:** Groups 2–4 from the 2026-05-15 code archaeology report — encoding fields never read by Rust, mark kwargs that raise at runtime, and stroke encoding channels unrendered in both backends. Group 1 (11 silent mark-kwargs) is handled separately under TDD.

---

## 1. Scope

Complete the deferred API surface for: (a) five `EncodingSpec` fields (`sort`, `stack`, `axis`, `format_type`, `impute`) that are accepted and serialized but silently ignored by the Rust renderer; (b) three mark desugar paths (`histogram multiple=`, `density multiple=`, `lmplot truncate=False`) that raise `ValueError` instead of rendering; (c) two data-routing gaps (`Chart(data=None)`, `Layer(data=...)` via `.layer()`); and (d) four stroke/angle encoding channels (`stroke_opacity`, `stroke_width`, `stroke_dash`, `angle`) that are silent in both SVG and WASM renderers. Items are ordered static-SVG first, interactive-WASM second.

---

## 2. Goals

**Static SVG**
- `X("col", sort="descending")` / `sort="ascending"` / `sort="-field"` produces correctly ordered axis ticks and data in the rendered SVG.
- `X("col", stack="zero")` / `stack="normalize"` / `stack="center"` produces stacked bar/area charts; these are the only accepted values.
- `axis={"title": "...", "ticks": False, "grid": False, "label_angle": -45}` — a subset of axis display properties — flows through to the rendered axis.
- `format_type="number"` / `"time"` on X/Y encodings controls tick-label formatter selection.
- `mark_histogram(multiple="stack")` / `"fill"` / `"dodge"` routes through the existing Stack/Dodge position-adjustment transforms.
- `mark_density(multiple="dodge")` routes through the existing Dodge position-adjustment transform.
- `lmplot(truncate=False)` / `regplot(truncate=False)` extends the fit line to the plot boundaries rather than clipping at the observed data extent.
- `Chart(data=None)` with per-layer `data=` is accepted and routes each layer through its own data source.
- `Layer(data=df)` passed to `Chart.layer()` is accepted; it is equivalent to the existing `+` operator path.

**Interactive WASM**
- `stroke_opacity=`, `stroke_width=`, `stroke_dash=`, and `angle=` field-driven encodings are applied per data row in the WASM renderer for Circle and Rect mark kinds.

---

## 3. Non-goals

- `impute=` on encodings — data imputation requires a new Rust transform and is deferred.
- `mark_raster(blend="additive")` **SVG** — already implemented via `mix-blend-mode:screen` in `svg_walk.rs`; no action needed. The non-goal is the **WASM GPU pixel-blending path** specifically, which requires additive compositing in the fragment shader and is a separate effort.
- Full `Axis(...)` Python value class — only dict-passthrough of the named axis properties listed in §6 is in scope.
- Legend kwarg passthrough beyond `disabled` — separate effort.
- SVG rendering of `stroke_dash`/`stroke_opacity`/`stroke_width`/`angle` channels — only WASM.

---

## 4. System behavior

**`sort=`** Ordinal and nominal X/Y scales resolve their domain order after consulting `EncodingSpec.sort`. Values: `"ascending"`, `"descending"`, `"-field"` (sort descending by the encoded field), `"x"` / `"-x"` / `"y"` / `"-y"` (sort by another encoding's field). Quantitative scales ignore `sort=`.

**`stack=`** A non-`None` `stack=` on an encoding inserts the Stack position-adjustment transform with the named strategy before the mark renderer runs. `"normalize"` normalises each stack group to [0, 1]; `"center"` centres stacks around zero; `"zero"` (default stacked behavior) starts from zero. The mark's desugar layer is responsible for asserting `stack=` is only valid for bar and area marks.

**`axis=`** A dict of axis display properties forwarded to the Rust axis layout engine. Only the properties listed in §6 are honoured; all others are silently accepted but have no effect (pre-existing behavior).

**`format_type=`** Selects the tick-label formatter family. `"number"` → numeric formatter; `"time"` → temporal formatter. If unset, the formatter is inferred from the field's data type as today.

**`histogram(multiple=)`** `"layer"` (default, existing) overlaps; `"stack"` inserts Stack on the y-axis after binning; `"fill"` inserts Stack with `normalize`; `"dodge"` inserts Dodge on the x-axis after binning.

**`density(multiple=)`** `"layer"` (default) overlaps; `"dodge"` inserts Dodge on the x-axis after KDE.

**`lmplot/regplot(truncate=False)`** The Smooth/Robust transform accepts an optional `x_range: Option<[f64; 2]>` specifying the x-domain over which the fit line is evaluated. When `truncate=False`, `x_range` is set to the plot's x-scale domain (i.e. the axis extent), not the observed data range.

**`Chart(data=None)` / `Layer(data=df)` via `.layer()`** When `Chart(data=None)` is constructed, the chart-level batch is empty; each layer must supply its own `data=`. `Chart.layer()` is extended to accept `Layer` instances that carry a `data=` attribute. Both paths produce the same internal `_Layer` representation as the `+` operator today.

**WASM stroke/angle channels** `stroke_opacity`, `stroke_width`, `stroke_dash`, and `angle` field-driven values are packed into the per-instance GPU buffers for Circle and Rect. `stroke_dash` maps to a dash-pattern index selecting from a small fixed palette (solid, dashed, dotted, dash-dot). `angle` rotates the instance around its anchor point.

---

## 5. Architecture

**Static SVG — sort, stack, axis, format_type, histogram/density multiple**

`sort=` and `stack=` are resolved in `scale_resolve.rs` and `position.rs` respectively. The Rust `EncodingSpec` fields `sort` and `stack` are already deserialized; the renderers simply need to read them. `axis=` properties flow from `EncodingSpec.axis` through the `AxisLayout` struct to `marks/axis.rs`. `format_type=` selects the formatter branch in `render/format.rs`.

Histogram/density `multiple=` is a Python desugar responsibility: the desugar layer constructs a `PositionAdjustment::Stack` or `::Dodge` spec and sets it on the layer's `position` field. The Rust position-adjustment pipeline already handles these.

**Static SVG — lmplot truncate=False**

`SmoothSpec` and `RobustSpec` gain an `x_range: Option<[f64; 2]>` field. The Python desugar sets this when `truncate=False` by reading the chart's x-scale domain. The Rust transform evaluates the fit line over `x_range` instead of `[xs.min(), xs.max()]`.

**Static SVG — data routing**

`Chart(data=None)` validation in `_coerce.py` is relaxed: when `data` is `None`, the per-layer `data=` requirement is enforced at `to_spec()` time (each layer must have a source). `Chart.layer()` is extended to accept `Layer` objects; the existing `__add__` path is the reference implementation.

**Interactive WASM — stroke/angle channels**

`CircleInstance` and `RectInstance` in `scene_load.rs` gain four new fields: `stroke_opacity: f32`, `stroke_width: f32`, `stroke_dash: u8` (palette index), `angle: f32`. The `scene_load.rs` batch-building path reads the encoded columns and populates these fields. The WebGPU vertex shader and render pipeline are updated to consume the new per-instance attributes.

---

## 6. Canonical interfaces / data contracts

**`axis=` accepted properties (dict keys)**

```
title: str | None
ticks: bool          # show/hide tick marks
tick_count: int      # target number of ticks
grid: bool           # show/hide gridlines
labels: bool         # show/hide tick labels
label_angle: float   # tick label rotation in degrees
orient: "top"|"bottom"|"left"|"right"
```

All other keys are accepted without error and have no effect.

**`SmoothSpec` / `RobustSpec` addition**

```rust
pub x_range: Option<[f64; 2]>,   // if Some, evaluate fit over this domain
```

**`stack=` accepted values**

`"zero"`, `"normalize"`, `"center"`, `None` (no stacking). Any other value is a `ValueError` at desugar time.

**`sort=` accepted values**

`"ascending"`, `"descending"`, a field name string, or `None`. The `"-"` prefix inverts order. Passing a list (custom domain order) is out of scope.

---

## 7. Invariants and constraints

- No silent drops after this work: every accepted kwarg either produces a visual effect or raises `ValueError` with an actionable message.
- `stack=` on a non-bar/non-area mark raises `ValueError` at desugar time.
- `lmplot(truncate=False)` does not extrapolate beyond the x-scale domain — it clips at the axis boundary, not arbitrarily far.
- WASM stroke/angle channels are data-driven only (field-mapped); constant overrides remain in `mark_style` (pre-existing).
- `Chart(data=None)` raises `ValueError` at `to_spec()` time — not at construction time — if any layer lacks a data source.

---

## 8. Key decisions and tradeoffs

**`sort=` resolved at scale time, not layout time.** Domain ordering is a scale property. Resolving it in `scale_resolve.rs` means axes and marks both see the sorted domain without duplicating logic.

**`stack=`/`multiple=` desugar is Python's responsibility, not Rust's.** The Rust position-adjustment pipeline is already correct. Python's desugar layer constructs the right `PositionAdjustment` spec. This avoids adding histogram-specific logic to the Rust core.

**`x_range` on Smooth/Robust, not a chart-level clip.** The fit line domain is a transform property. A chart-level clip would also clip the scatter layer, which is wrong.

**WASM stroke channels as per-instance GPU attributes, not conditional encodings.** Conditional encodings change values in response to interaction. Stroke channels are data-driven constants per row. They use the existing instance-buffer path, not the conditional-encoding path.

**`stroke_dash` as palette index, not raw SVG dash array.** GPU instanced rendering does not support per-instance arbitrary dash arrays efficiently. A fixed palette (4–8 patterns) covers all practical use cases.

**`axis=` as dict passthrough, not a Python `Axis(...)` class.** A full value class is non-trivial to maintain in sync with the Rust layout engine. Dict passthrough with a documented allowed-key list is sufficient and reversible.

---

## 9. Acceptance criteria

- `Chart(df).mark_bar().encode(x=X("cat", sort="descending"), y="val").show_svg()` — bars appear right-to-left in descending category order.
- `Chart(df).mark_bar().encode(x="cat", y=Y("val", stack="normalize"), color="grp").show_svg()` — bars fill 0–1 per category.
- `Chart(df).mark_bar().encode(x=X("cat", axis={"label_angle": -45})).show_svg()` — tick labels are rotated.
- `mark_histogram(multiple="dodge")` produces side-by-side bins; `multiple="stack"` produces stacked bins.
- `lmplot(df, x="x", y="y", truncate=False).show_svg()` — fit line extends to the axis edge, not just to `x.min()`/`x.max()`.
- `Chart(data=None).layer(Layer(data=df1, mark="point", x="a", y="b"), Layer(data=df2, mark="line", x="a", y="b")).show_svg()` — both layers render from their respective sources.
- WASM: encoding `stroke_opacity` to a numeric field produces per-row varying opacity on Circle/Rect marks in the interactive renderer.

---

## 10. Validation strategy

Static SVG items: Python integration tests calling `show_svg()` and asserting SVG structure (sorted domain in axis tick text nodes, normalized bar heights, rotated label transforms, side-by-side bin positions).

WASM stroke channels: Rust unit tests asserting `CircleInstance.stroke_opacity` is populated from the column values; a WebGPU render test (if the test harness supports it) or a visual snapshot test.

`lmplot(truncate=False)`: assert the fit-line `<path>` in the SVG extends to the x-scale domain endpoints.

---

## 11. Open questions

- **`sort=` with list value (custom domain order):** Vega-Lite supports `sort=["a", "b", "c"]`. Deferred — the accepted-value contract above must raise `ValueError` for list values with an actionable message pointing to this deferral.
- **WASM `stroke_dash` palette definition:** The exact 4–8 dash patterns need to be chosen and documented in the implementation. The spec does not mandate specific patterns.
