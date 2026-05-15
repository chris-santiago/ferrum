# Silent-Drop Remediation — Static SVG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Wire seven deferred static-SVG items — `sort=`, `stack=`, `axis=` dict passthrough, `format_type=`, histogram/density `multiple=`, `lmplot(truncate=False)`, and per-layer `data=` routing — so every accepted kwarg produces a visual effect instead of being silently dropped.

## 2. Spec references

- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §4 System behavior`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §5 Architecture`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §6 Canonical interfaces`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §7 Invariants`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/scale_resolve.rs` | read `EncodingSpec.sort` to order ordinal domains |
| Modify | `crates/ferrum-core/src/render/position.rs` | honour `EncodingSpec.stack` to select Stack strategy |
| Modify | `crates/ferrum-core/src/layout/mod.rs` (or `axis.rs`) | thread `axis=` dict properties into `AxisLayout` |
| Modify | `crates/ferrum-core/src/render/marks/axis.rs` | consume `AxisLayout` axis display properties |
| Modify | `crates/ferrum-core/src/render/format.rs` | select formatter from `EncodingSpec.format_type` |
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | confirm all five fields are present and deserialised |
| Modify | `src/ferrum/marks/statistical.py` | histogram `multiple=` → Stack/Dodge `PositionAdjustment` |
| Modify | `src/ferrum/marks/composite.py` (or density desugar) | density `multiple="dodge"` → Dodge |
| Modify | `src/ferrum/plots/regression.py` | `truncate=False` → set `x_range` on Smooth/RobustSpec |
| Modify | `crates/ferrum-core/src/transform/smooth.rs` | add `x_range: Option<[f64; 2]>` field, use it for fit-line domain |
| Modify | `crates/ferrum-core/src/transform/robust.rs` | same `x_range` addition |
| Modify | `src/ferrum/_coerce.py` | relax `Chart(data=None)` error to a `to_spec()`-time check |
| Modify | `src/ferrum/chart.py` | `Chart.layer()` accepts `Layer` objects with `data=` |
| Test | `tests/test_pipeline_regression.py` | sort= and stack= regression tests |
| Test | `tests/marks/test_statistical.py` (or new file) | histogram multiple= tests |
| Test | `tests/plots/test_regression.py` (or new file) | truncate=False fit-line extent test |
| Test | `tests/test_coerce.py` | Chart(data=None) + Layer(data=) routing tests |

## 4. Constraints

- `sort=` applies only to ordinal/nominal domains; quantitative scales ignore it (spec §4).
- `stack=` only valid on bar and area marks — raise `ValueError` at desugar time for others (spec §7).
- `stack=` accepted values: `"zero"`, `"normalize"`, `"center"`, `None`; all others are `ValueError` (spec §6).
- `truncate=False` clips at the x-scale domain boundary — does not extrapolate past the axis edge (spec §7).
- `Chart(data=None)` raises `ValueError` at `to_spec()` time, not construction time, if any layer lacks a source (spec §7).
- `axis=` keys outside the documented set are silently accepted with no effect — do not raise (spec §6).
- No matplotlib. No warn-fallbacks. No `NotImplementedError`.
- Before any `git commit` touching `*.py`: dispatch `python-review-lite`. Before any commit touching `*.rs`: dispatch `rust-review-lite`.

## 5. Tasks

### Task 1: sort= on ordinal/nominal X/Y encodings
- [ ] Read `EncodingSpec.sort` in `scale_resolve.rs` ordinal domain builder; apply ascending/descending/field-based ordering (spec §4)
- [ ] Write failing test: `X("cat", sort="descending")` → SVG axis ticks appear in reverse-alphabetical order
- [ ] Implement; make test pass
- [ ] Verify: `uv run pytest tests/ -k "sort" -v`

### Task 2: stack= on X/Y encodings
- [ ] Read `EncodingSpec.stack` in `position.rs`; map to Stack strategy enum (`zero`/`normalize`/`center`) (spec §4, §6)
- [ ] Add `ValueError` in bar/area desugar for `stack=` on incompatible marks
- [ ] Write failing tests: stacked bar heights sum to 1.0 for `normalize`; side-by-side heights unchanged for `None`
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "stack" -v`

### Task 3: axis= dict passthrough
- [ ] Extend `AxisLayout` struct to carry the seven accepted properties (spec §6)
- [ ] Thread from `EncodingSpec.axis` dict through layout → `marks/axis.rs` renderer
- [ ] Write failing test: `axis={"ticks": False}` → no `<line>` tick elements in SVG; `axis={"label_angle": -45}` → `transform="rotate(-45)"` on tick labels
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/ -k "axis" -v`

### Task 4: format_type= tick-label formatter selection
- [ ] Read `EncodingSpec.format_type` in `render/format.rs`; select formatter branch (spec §4)
- [ ] Write failing test: `format_type="number"` on a date-typed column still formats as number
- [ ] Implement; make test pass
- [ ] Verify: `uv run pytest tests/ -k "format_type" -v`

### Task 5: histogram and density multiple=
- [ ] Extend `desugar_histogram` in `statistical.py` to construct `PositionAdjustment::Stack` or `::Dodge` for `multiple="stack"/"fill"/"dodge"` (spec §4, §5)
- [ ] Extend density desugar similarly for `multiple="dodge"`
- [ ] Remove the existing `ValueError` for these values
- [ ] Write failing tests: `multiple="dodge"` bins are side-by-side (non-overlapping x ranges); `multiple="stack"` bins are stacked (y values accumulate)
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/marks/ -v`

### Task 6: lmplot/regplot truncate=False
- [ ] Add `x_range: Option<[f64; 2]>` to `SmoothSpec` and `RobustSpec`; use it as the fit-line evaluation domain (spec §4, §5, §6)
- [ ] In `regression.py` desugar: when `truncate=False`, set `x_range` from the chart's x-scale domain instead of raising `ValueError`
- [ ] Write failing test: fit-line `<path>` in SVG extends beyond observed `x.min()`/`x.max()` to the axis boundary
- [ ] Implement; make test pass
- [ ] Verify: `uv run pytest tests/plots/ -k "truncate" -v`

### Task 7: Chart(data=None) and Layer(data=) via .layer()
- [ ] Relax `_coerce.py` validation: `Chart(data=None)` is accepted; `to_spec()` raises if any layer has no source (spec §4, §7)
- [ ] Extend `Chart.layer()` to accept `Layer` instances with `data=`; produce same internal representation as `+` operator (spec §4)
- [ ] Update stale error message (currently references "Phase 8a")
- [ ] Write failing tests: two-layer chart via `Chart(data=None).layer(Layer(data=df1, ...), Layer(data=df2, ...))` renders both layers
- [ ] Implement; make tests pass
- [ ] Verify: `uv run pytest tests/test_coerce.py tests/test_composition.py -v`

## 6. Acceptance checks

- `uv run pytest tests/ -x -q` — 1806+ pass, 0 new failures
- `source ~/.cargo/env && DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core --lib` — all pass
- Each acceptance criterion in spec §9 produces the described SVG output

## 7. Open questions

- `sort=` with list value (custom domain order) — must raise `ValueError` with message pointing to spec deferral (spec §11)
