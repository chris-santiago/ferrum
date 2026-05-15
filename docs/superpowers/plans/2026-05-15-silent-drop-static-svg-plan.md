# Silent-Drop Remediation — Static SVG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Wire eleven deferred static-SVG items so every accepted kwarg produces a visual effect instead of being silently dropped or raising at runtime.

## 2. Spec references

- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §4 System behavior`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §5 Architecture`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §6 Canonical interfaces`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §7 Invariants`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/scale_resolve.rs` | `sort=` string + list ordering for ordinal domains |
| Modify | `crates/ferrum-core/src/render/position.rs` | `stack=` strategy selection |
| Modify | `crates/ferrum-core/src/layout/mod.rs` (or `axis.rs`) | thread `axis=` dict into `AxisLayout` |
| Modify | `crates/ferrum-core/src/render/marks/axis.rs` | consume `AxisLayout` display properties |
| Modify | `crates/ferrum-core/src/render/format.rs` | `format_type=` formatter selection |
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | confirm all fields present and deserialised |
| Create | `crates/ferrum-core/src/transform/impute.rs` | new Impute transform (spec §5) |
| Modify | `crates/ferrum-core/src/transform/core.rs` | register Impute in transform dispatch |
| Modify | `crates/ferrum-core/src/spec/` | add `ImputeSpec` struct (spec §6) |
| Modify | `crates/ferrum-core/src/spec/legend.rs` (or `encoding.rs`) | add full `LegendSpec` fields (spec §6) |
| Modify | `crates/ferrum-core/src/render/marks/legend.rs` | consume new `LegendSpec` fields |
| Modify | `crates/ferrum-core/src/render/marks/point.rs` | emit stroke/angle per-element SVG attributes |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` | same for bar |
| Modify | `crates/ferrum-core/src/render/marks/line.rs` | same for line |
| Modify | `crates/ferrum-core/src/render/marks/rule.rs` | same for rule |
| Modify | `src/ferrum/chart.py` | remove stroke/angle from `_SILENT_CHANNELS` after SVG wired |
| Modify | `src/ferrum/marks/statistical.py` | histogram `multiple=` → Stack/Dodge |
| Modify | `src/ferrum/marks/composite.py` | density `multiple="dodge"` → Dodge |
| Modify | `src/ferrum/plots/regression.py` | `truncate=False` → set `x_range` |
| Modify | `crates/ferrum-core/src/transform/smooth.rs` | add `x_range: Option<[f64; 2]>` |
| Modify | `crates/ferrum-core/src/transform/robust.rs` | same |
| Modify | `src/ferrum/_coerce.py` | relax `Chart(data=None)` to deferred check |
| Modify | `src/ferrum/chart.py` | `Chart.layer()` accepts `Layer` with `data=` |
| Test | `tests/test_pipeline_regression.py` | sort, stack, impute, legend regression tests |
| Test | `tests/marks/test_statistical.py` | histogram/density multiple= tests |
| Test | `tests/plots/test_regression.py` | truncate=False extent test |
| Test | `tests/test_coerce.py` | Chart(data=None) + Layer(data=) tests |
| Test | `tests/test_encoding_channels.py` (new) | stroke/angle SVG attribute tests |

## 4. Constraints

- `sort=` list value sets the exact ordinal domain — no sorting applied to the list itself (spec §4).
- `stack=` only valid on bar/area — `ValueError` at desugar time for other marks (spec §7).
- `impute={"method": "value"}` without a `value` key raises `ValueError` (spec §7).
- `truncate=False` clips at x-scale domain boundary — never extrapolates beyond it (spec §7).
- `Chart(data=None)` raises at `to_spec()` time, not construction time (spec §7).
- `axis=` and `legend=` keys outside the documented sets are silently accepted with no effect (spec §6).
- SVG stroke/angle channels remain in `_SILENT_CHANNELS` for mark kinds that don't support per-element stroke (e.g. area fill) — intentional, not a gap (spec §7).
- `stroke_dash` palette is exactly the four entries in spec §6; integer column values clamp to nearest index.
- No matplotlib. No warn-fallbacks. No `NotImplementedError`.
- Before any commit touching `*.py`: dispatch `python-review-lite`. Before any commit touching `*.rs`: dispatch `rust-review-lite`.

## 5. Tasks

### Task 1: sort= — string and list values
- [ ] Read `EncodingSpec.sort` in `scale_resolve.rs` ordinal domain builder; apply ascending/descending/field ordering for strings; use list directly as domain for list values (spec §4, §6)
- [ ] Write failing tests: `sort="descending"` → reverse-alphabetical ticks; `sort=["b","a","c"]` → exact sequence in SVG
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "sort" -v`

### Task 2: stack= on X/Y encodings
- [ ] Read `EncodingSpec.stack` in `position.rs`; map to Stack strategy (`zero`/`normalize`/`center`) (spec §4, §6)
- [ ] `ValueError` in bar/area desugar for `stack=` on incompatible marks (spec §7)
- [ ] Write failing tests: `normalize` → bar heights sum to 1.0; `None` → heights unchanged
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "stack" -v`

### Task 3: axis= dict passthrough
- [ ] Extend `AxisLayout` to carry the seven accepted properties (spec §6)
- [ ] Thread from `EncodingSpec.axis` through layout → `marks/axis.rs`
- [ ] Write failing tests: `ticks: False` → no tick `<line>` elements; `label_angle: -45` → `rotate(-45)` on labels
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "axis" -v`

### Task 4: format_type= formatter selection
- [ ] Read `EncodingSpec.format_type` in `render/format.rs`; select formatter branch (spec §4)
- [ ] Write failing test: `format_type="number"` on date-typed column formats as number
- [ ] Implement; make test pass
- [ ] Verify: `uv run pytest tests/ -k "format_type" -v`

### Task 5: impute= transform
- [ ] Create `ImputeSpec` struct (spec §6) and `Impute` transform in `crates/ferrum-core/src/transform/impute.rs`
- [ ] Register in transform dispatch (`transform/core.rs`)
- [ ] Insert into pipeline when `EncodingSpec.impute` is present (spec §5)
- [ ] Write failing test: sparse time-series with `impute={"method":"value","value":0}` → no gaps in rendered line path
- [ ] Implement; make test pass
- [ ] Verify: `uv run pytest tests/ -k "impute" -v`

### Task 6: legend kwargs passthrough
- [ ] Add full `LegendSpec` field set to Rust spec (spec §6)
- [ ] Consume in `marks/legend.rs` for orient, direction, title, format, tick_count, columns, gradient dimensions
- [ ] Write failing tests: `orient="bottom"` → legend `<g>` positioned at bottom; `direction="horizontal"` → horizontal symbol layout
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "legend" -v`

### Task 7: histogram and density multiple=
- [ ] Extend `desugar_histogram` to construct `PositionAdjustment::Stack` or `::Dodge` for `multiple="stack"/"fill"/"dodge"` (spec §4, §5)
- [ ] Extend density desugar for `multiple="dodge"`; remove existing `ValueError` for these values
- [ ] Write failing tests: `dodge` → non-overlapping x ranges; `stack` → y values accumulate
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/marks/ -v`

### Task 8: lmplot/regplot truncate=False
- [ ] Add `x_range: Option<[f64; 2]>` to `SmoothSpec` and `RobustSpec` (spec §6)
- [ ] Python desugar: set `x_range` from x-scale domain when `truncate=False`; remove `ValueError`
- [ ] Write failing test: fit-line path extends beyond observed `x.min()`/`x.max()` to axis boundary
- [ ] Implement; make test pass
- [ ] Verify: `uv run pytest tests/plots/ -k "truncate" -v`

### Task 9: Chart(data=None) and Layer(data=) via .layer()
- [ ] Relax `_coerce.py`; enforce per-layer source at `to_spec()` time (spec §4, §7)
- [ ] Extend `Chart.layer()` to accept `Layer` with `data=`; mirrors `__add__` path (spec §4)
- [ ] Update stale "Phase 8a" error message
- [ ] Write failing tests: two-layer `Chart(data=None)` chart renders both layers correctly
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/test_coerce.py tests/test_composition.py -v`

### Task 10: SVG stroke/angle channels
- [ ] In mark renderers (`point.rs`, `bar.rs`, `line.rs`, `rule.rs`): read `stroke_width`/`stroke_opacity`/`stroke_dash`/`angle` columns from batch; emit per-element SVG attributes using the `stroke_dash` palette from spec §6 (spec §4, §5)
- [ ] Remove `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` from `_SILENT_CHANNELS` in `chart.py` (leave them silent for mark kinds that don't support per-element stroke, per spec §7)
- [ ] Write failing tests: `encode(stroke_width="col")` on a line chart → each element has distinct `stroke-width`; `encode(stroke_dash="col")` → correct `stroke-dasharray` values from palette
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "stroke" -v`

## 6. Acceptance checks

- `uv run pytest tests/ -x -q` — 1807+ pass, 0 new failures
- `source ~/.cargo/env && DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core --lib` — all pass
- Each acceptance criterion in spec §9 (static SVG items) produces the described SVG output

## 7. Open questions

- None — `sort=` list is now in scope (spec §4). The open question in the previous plan version is resolved.
