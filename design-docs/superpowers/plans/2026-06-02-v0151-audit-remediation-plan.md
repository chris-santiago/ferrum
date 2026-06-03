# v0.15.1 Audit-Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development to implement this plan task-by-task. `.py` → `python-coder`; `.rs` → `rust-coder`. Every task gets the full gate chain (spec review → quality review → review-lite). All render changes visually inspected per CLAUDE.md goldens rule.

## 1. Objective

Fix the ~13 confirmed bugs found by the post-v0.15.0 auditor sweep, grouped into 5 cohesive root-cause themes plus a lower bucket, then ship v0.15.1.

## 2. Spec references

- Findings source: the v0.15.0 Wave-1 + Wave-2 audit (this session) — recorded in `design-docs/superpowers/followups/2026-05-15-code-archaeology.md`.
- FA-16 (line ribbon under rescale) is OUT of scope — own spec: `design-docs/superpowers/specs/2026-06-02-wasm-relayout-rescale-design.md`.
- FA-15 (color-conditional builds no legend) remains tracked separately.

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/transform/data_window.rs` | T1: group_key migration |
| Modify | `crates/ferrum-core/src/transform/data_stack.rs` | T1: group_key migration |
| Modify | `crates/ferrum-core/src/transform/data_aggregate.rs` | T1: group_key + dtype materialization |
| Modify | `crates/ferrum-core/src/transform/join_aggregate.rs` | T1: group_key migration |
| Modify | `crates/ferrum-core/src/transform/pivot.rs` | T1: group_key + dtype materialization |
| Modify | `crates/ferrum-core/src/transform/group_key.rs` | T1: FA-9 null-key distinctness |
| Modify | `crates/ferrum-core/src/transform/impute.rs` | T1: dedup into numeric_util |
| Modify | `crates/ferrum-core/src/transform/{density_data,bin,swarm}.rs` | T1: accept int/bool groupby |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` | T2: build_polar channel parity |
| Modify | `crates/ferrum-core/src/render/marks/rect.rs` | T2: range-path channel parity |
| Modify | `crates/ferrum-core/src/render/marks/arc.rs` | T2: href/description passthrough |
| Modify | `src/ferrum/axis.py` | T3: title=None suppression |
| Modify | `src/ferrum/legend.py` | T3: title=None suppression |
| Modify | `src/ferrum/chart.py` | T4: add_params/add_selection validation + collision detection |
| Modify | `crates/ferrum-wasm/src/lib.rs` | T5: reproject brush before rescale affine |
| Modify | `crates/ferrum-wasm/src/param_runtime.rs` | T5: src≠tgt rescale unit test |
| Modify | `src/ferrum/_core.pyi` | T6: ChartSpec stub 15→35 args |
| Modify | `crates/ferrum-wasm/src/lib.rs` | T6: legend-toggle via add_params warn/work |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | T6: single-panel domain-param brush mode |
| Modify | `crates/ferrum-core/src/render/marks/{area,bar}.rs` | T6: x2 implement-or-raise |
| Test | `tests/test_bug_hunt_release_transforms.py` | existing red tests (T1) |
| Test | `tests/test_bug_hunt_reactive_params.py` | existing red tests (T4/T6) |
| Test | `crates/ferrum-core/tests/bug_hunt_release_transforms.rs` | existing red tests (T1) |

## 4. Constraints

- **No silent failure.** Every fix replaces silent-drop/collapse with correct behavior or a named error/warning. No NotImplementedError, no warn-fallback where a real fix is possible (CLAUDE.md no-defer rule).
- **Cohesive, paradigm-respecting fixes.** Migrate to the shared helper (`group_key`, `numeric_util`, `StrokeChannels::load`/`MetadataColumns`/`meta.build_metadata`); do not reimplement per-site. Push computation to Rust where it belongs.
- **Byte-stability for unaffected paths.** Existing goldens for Utf8/Float64 groupby and non-range marks must not change; if a golden legitimately changes, rasterize + visually inspect + bless per CLAUDE.md.
- **Pre-existing failing tests are the acceptance bar.** The three bug-hunt test files must go green; do not weaken their assertions.
- **`cargo test` must pass** before any Rust theme is considered done.
- **Group_key null contract:** a null groupby key must produce a distinct output key from a real 0/false/"" — choose one representation (e.g. a reserved null sentinel column-value) and apply it uniformly in `materialize_groupby_col`.

## 5. Tasks

Stages are file-footprint-disjoint for parallelism. Within a stage, dispatch in parallel; between stages, serialize.

### Stage A (parallel: T1, T2, T3, T4 touch disjoint files)

### Task 1: Finish group_key unification (Rust transforms)
- [ ] Migrate `extract_key`/`extract_key_str`/`extract_string_key` in data_window, data_stack, data_aggregate, join_aggregate, pivot to `group_key::groupby_key_at`.
- [ ] In data_aggregate + pivot, emit groupby output columns via `materialize_groupby_col` so declared dtype matches the actual array (fixes `RecordBatch::try_new` mismatch).
- [ ] Fix FA-9 in `group_key.rs`: null keys distinct from real 0/false/empty (see Constraints null contract).
- [ ] Replace `impute.rs::clean_values` with `numeric_util::clean_float64_values`.
- [ ] Make `density_data`, `bin`, `swarm` accept int/bool groupby via `group_key` (replace the loud Utf8-only rejection).
- [ ] Verify: `cargo test -p ferrum-core` (DYLD cmd) + `uv run pytest tests/test_bug_hunt_release_transforms.py -v` all green.

### Task 2: Mark channel-loading parity (Rust render)
- [ ] `bar.rs::build_polar`: load `StrokeChannels::load(ctx)` + `MetadataColumns::from_ctx(ctx)`; apply per-row opacity/stroke_width/stroke_opacity/stroke_dash/angle/fill_opacity and emit tooltips, matching the Cartesian bar paths.
- [ ] `rect.rs::build_quantitative_range` + `build_ordinal_range`: load fill_opacity + per-row stroke_width/stroke_opacity/stroke_dash to match `build_heatmap`.
- [ ] `arc.rs::build_nominal_theta` + `build_annular`: route href/description/tooltip through `meta.build_metadata(ctx)` instead of hardcoding `None`.
- [ ] Verify: `cargo test -p ferrum-core`; render minimal repros (polar bar opacity/stroke_width, range-rect fill_opacity, arc href) to SVG and visually confirm channels apply.

### Task 3: skip-None to_dict suppression (Python)
- [ ] `axis.py` + `legend.py`: on explicit `title=None`, forward `title=""` (suppress) like `encoding/base.py`; absent key keeps field-name default. Prefer a single shared helper over duplicating the convention twice.
- [ ] Verify: `uv run pytest -k "title" tests/` + a repro confirming `Axis(title=None)`, `Legend(title=None)`, and the channel-level forms all suppress consistently.

### Task 4: Param/selection namespace integrity (Python)
- [ ] `chart.py` `add_params`/`add_selection`: validate each arg is the expected type at the boundary; raise `TypeError` naming the bad argument (no silent drop, no late `AttributeError`).
- [ ] Detect cross-kind same-name collisions in `_collect_params` (~3517) and the `+` merge path (~2120-2131); warn or raise rather than letting a selection silently shadow a same-named `VariableParameter`.
- [ ] Verify: `uv run pytest tests/test_bug_hunt_reactive_params.py -v` green (the param-collection cases).

### Stage B (after A — T5 touches WASM lib.rs/param_runtime.rs, isolated)

### Task 5: Cross-panel reactive rescale (Rust WASM)
- [ ] `apply_reactive_rescale` (lib.rs): reproject the brush source-pixels → shared data domain → target-pixels before calling `rescale_affine`, mirroring `apply_crossfilter`/`reproject_extent`.
- [ ] Add a `param_runtime.rs` unit test with `src.plot_area != tgt.plot_area` asserting target marks land inside the target plot area.
- [ ] Verify: `cargo build -p ferrum-wasm --target wasm32-unknown-unknown` + `cargo test -p ferrum-wasm`; rebuild WASM (`wasm-pack build … --release`) and browser-validate an `hconcat(overview, detail)` rescale — detail marks zoom in-panel, not off-screen.

### Stage C (after B — T6 lower bucket; sub-items touch disjoint files, parallel where so)

### Task 6: Lower bucket
- [ ] Regenerate/update `_core.pyi` `ChartSpec` stub to the actual 35-arg constructor (incl. params/selections/conditionals/chart_description).
- [ ] `chart.py` (~3495): raise a legible ferrum error naming the offending param/bound when a `fm.param` domain contains Inf/NaN, before `json.dumps`.
- [ ] WASM legend-toggle via `add_params` (lib.rs:543-591 / chart.py): make it work or warn — no silent no-op.
- [ ] `ferrum-anywidget.js` (~536): a single-panel domain-param chart should enable the rescale brush affordance (don't silently default to inert pan mode).
- [ ] `area.rs`/`bar.rs`: `mark_area` `x2` and `mark_bar` `x2`+`y2` — implement the extent or raise a named error; no silent drop.
- [ ] Verify: `uv run pytest tests/test_bug_hunt_reactive_params.py -v` (Inf/NaN case) + the skeleton-check from CLAUDE.md for the `.pyi`.

### Stage D: Gate + release
- [ ] Update `design-docs/superpowers/followups/2026-05-15-code-archaeology.md`: mark resolved items, close action-list entries.
- [ ] `uv run nox` — all 5 sessions green.
- [ ] `/release patch` → v0.15.1.

## 6. Acceptance checks

- `cargo test -p ferrum-core` and `cargo test -p ferrum-wasm` — all pass.
- `cargo build -p ferrum-wasm --target wasm32-unknown-unknown` — succeeds.
- `uv run pytest tests/test_bug_hunt_release_transforms.py tests/test_bug_hunt_reactive_params.py -v` — all pass (the red tests go green).
- `cargo test` (bug_hunt_release_transforms.rs) — all pass.
- `uv run nox` — lint, test, cargo_test, build, docs all green.
- Browser: `hconcat(overview, detail)` reactive rescale zooms the detail panel in-bounds.
- No golden byte-changes except those visually inspected and blessed.

## 7. Open questions

- Group_key null representation: pick the reserved-null-sentinel scheme in T1 (must distinguish null from 0/false/""); if it forces a wire/JSON change, surface before implementing.
- `mark_area x2` / `mark_bar x2+y2`: implement vs. raise — default to implement if the extent is well-defined for the mark, else raise; confirm during T6 if the geometry is ambiguous.
