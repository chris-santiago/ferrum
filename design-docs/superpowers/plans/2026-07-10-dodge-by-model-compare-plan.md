# Dodge-by-Model `compare=` Layout Implementation Plan (GH #42)

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Switch `importance_chart`, `shap_bar_chart` (`per_class=False`), and `cv_scores_chart` (`kind="box"|"strip"`) under `compare=` from small-multiples panels to a single shared-axis panel dodged by model, adding text-mark position-offset consumption in Rust.

## 2. Spec references

- `design-docs/superpowers/specs/2026-07-10-dodge-by-model-compare-design.md` §4 (behavior), §6 (schemas/eligibility/domain rule), §7 (invariants), §8 (D1–D6), §9–10 (acceptance/validation)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/marks/text.rs` | consume `read_position_offsets` (mirror `tick.rs:24`) + Rust unit test |
| Modify | `src/ferrum/position.py` | `_DODGE_ELIGIBLE` += `importance`, `shap_bar`, `cv_scores`, `text` |
| Modify | `src/ferrum/plots/_helpers.py` | shared per-model frame-stacking helper (contract, Task 3) |
| Modify | `src/ferrum/plots/explanation.py` | importance + shap_bar compare dodge branches/builders; docstrings |
| Modify | `src/ferrum/plots/model_selection.py` | cv_scores compare dodge branch (box/strip); docstring |
| Modify | `src/ferrum/marks/diagnostic/_selection.py` | thread `color_field` into `kind="box"` boxplot desugar if not already |
| Modify | `tests/test_phase_9_position.py` | eligibility-matrix rows for new dodge-eligible marks |
| Modify | `tests/diagnostics/test_compare_exclusions.py` | rewrite importance/cv_scores small-multiples assertions to dodged-panel contract |
| Test | `tests/diagnostics/test_compare_dodge.py` | new behavior tests (all three charts, all flags) |
| Modify | `tests/diagnostics/test_compare_aggregate_goldens.py` | regen cv_scores compare golden; add dodge goldens |
| Modify | `ferrum-spec.md` | dated note amending the 2026-06-27 small-multiples contract |

## 4. Constraints

- Single-model output (`compare=None`/omitted) of all three charts stays byte-identical; existing `compare=None` byte-identity tests must pass unmodified.
- All other `compare=` diagnostics stay byte-identical; `_compose_compare` and `_resolve_source` in `src/ferrum/plots/_helpers.py` must not change.
- Text offset consumption must be zero-effect when `__pos_x_offset__`/`__pos_y_offset__` columns are absent — all existing text/SVG output byte-identical.
- The stamped column is literally `model: Utf8`; model order = compare registration order, `"base"` first. Dodge declaration: `color="model"` + `position=Dodge(by="model")` at the chart level (falls through to every desugared layer via Rust prepare).
- Horizontal importance under compare = vertical desugar form + `CoordFlip` (spec D2); never dodge an ordinal-y layout directly, never rewrite a quantitative x.
- `cv_scores_chart(kind="bar")` and `shap_bar_chart(per_class=True)` keep the existing small-multiples path (spec D3/D6). `kind="strip"` under dodge drops jitter (spec D3).
- Feature axis under compare: one global ranking across models, top-`top_k`/`max_display` shared set (spec D4, schemas spec §6); value-axis domain computed over the combined DataFrame with the single-model formula.
- Every regenerated/new golden goes through `tests/_snapshots.py::regen_and_verify` and the resulting PNG is Read and visually confirmed before commit (CLAUDE.md goldens rule).
- No matplotlib; no global mutable state; no warn-fallbacks/`NotImplementedError`.
- Rust changes require rebuild before Python tests: `unset CONDA_PREFIX && uv run --no-sync maturin develop`.

## 5. Tasks

### Task 1: Rust text-mark position offsets
- [ ] `text.rs` draw path reads `read_position_offsets` and adds per-row offsets to resolved glyph positions (mirror `tick.rs`)
- [ ] Rust unit test: text batch with offset columns shifts glyph x; without columns, output unchanged
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core`

### Task 2: Dodge eligibility
- [ ] Add `importance`, `shap_bar`, `cv_scores`, `text` to `_DODGE_ELIGIBLE` with the composite-mark comment pattern
- [ ] Extend the eligibility matrix test with the new marks
- [ ] Verify: `uv run pytest tests/test_phase_9_position.py -x -q` then `uv run pytest -n auto -q` (shared public contract — suite-wide)

### Task 3: importance_chart compare dodge
- Consumes: eligibility from Task 2
- [ ] Build shared helper in `plots/_helpers.py`: stack per-model frames — iterate `ComparedModelSource.items()`, apply a per-model frame callback, stamp `model`, concat (contract for Tasks 4–5)
- [ ] New compare builder: global mean-importance ranking → top-`top_k` set (spec §6 schema), single chart via `mark_importance` with `color_field="model"` + `Dodge(by="model")`; horizontal = vertical form + `CoordFlip`; `error_bars`/`show_values` layers included
- [ ] Rewrite the importance small-multiples test; add dodge tests: returns `Chart`, n_models bars per feature band, legend, both orients, dodged value labels/rules, determinism, `compare=None` byte-identity untouched
- [ ] Verify: `uv run pytest tests/diagnostics/test_compare_dodge.py tests/diagnostics/test_compare_exclusions.py tests/diagnostics/test_compare.py -x -q`

### Task 4: shap_bar_chart compare dodge
- Consumes: frame-stacking helper from Task 3 → `src/ferrum/plots/_helpers.py`
- [ ] Compare builder for `per_class=False`: pooled `_shap_order_features` ranking across models (spec §6 schema), dodged bars; `per_class=True` keeps `_compose_compare`
- [ ] Tests: `Chart` return, shared feature set ≤ `max_display`, per_class=True still `ConcatChart`, determinism
- [ ] Verify: `uv run pytest tests/diagnostics/test_compare_dodge.py tests/diagnostics/test_compare_exclusions.py -x -q`

### Task 5: cv_scores_chart compare dodge
- Consumes: frame-stacking helper from Task 3 → `src/ferrum/plots/_helpers.py`
- [ ] Compare branch for `kind="box"|"strip"`: combined frame, `x=split` kept, `color_field="model"`, `Dodge(by="model")`; thread `color_field` through `desugar_cv_scores`→`desugar_boxplot` groupby if missing; strip drops jitter; `kind="bar"` keeps `_compose_compare`
- [ ] Rewrite the cv_scores small-multiples test; add dodge tests: box/strip return `Chart`, n_models marks per split band, `split=` filter works, `kind="bar"` still `ConcatChart`, determinism
- [ ] Verify: `uv run pytest tests/diagnostics/test_compare_dodge.py tests/diagnostics/test_compare_exclusions.py tests/diagnostics/test_compare_aggregate_goldens.py -x -q` (golden failure expected → Task 6)

### Task 6: Goldens
- Consumes: builders from Tasks 3–5
- [ ] Regenerate the cv_scores compare golden and add importance/shap_bar dodge goldens via `regen_and_verify`
- [ ] Read every produced PNG; confirm dodged offsets, legends, labels-on-bars before staging
- [ ] Verify: `uv run pytest tests/diagnostics/test_compare_aggregate_goldens.py -x -q`

### Task 7: Spec note + docstrings
- [ ] `ferrum-spec.md`: dated 2026-07-10 note amending the 2026-06-27 small-multiples contract for the three charts (incl. kind="bar"/per_class=True carve-outs)
- [ ] Update the three public docstrings' compare= behavior description
- [ ] Verify: `uv run pytest -n auto -q` and `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core`

## 6. Acceptance checks

- `uv run pytest -n auto` — full suite green
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — green
- Spec §9 criteria hold: dodged single-panel `Chart` for the three charts, byte-identical single-model and other-compare output, dodged text, determinism, visually inspected goldens
- `nox -s lint` clean

## 7. Open questions

None.
