# Cohesion & Complexity Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task.

## 1. Objective

Address all 9 open recommendations from the 2026-05-24 cohesion/complexity audit — pure refactoring, no API changes, no new features.

## 2. Spec references

- `design-docs/superpowers/audits/2026-05-24-cohesion-complexity-audit.md` — full audit report with line numbers and rationale

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/mod.rs` | Extract shared render pipeline (T1) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | `LegendPreparedOverrides` sub-struct (T2) |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | Decompose `ThemeInputs` into sub-structs (T5) |
| Modify | `crates/ferrum-core/src/render/scale_resolve.rs` | Split into sub-modules (T8) |
| Create | `crates/ferrum-core/src/render/scale_resolve/` | Sub-modules: positional, color, auxiliary, domain (T8) |
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | Factor `ContinuousScaleCommon`, unify `inherit_*` (T10, T10b) |
| Modify | `src/ferrum/chart.py` | Auto-clone, extract mark mixins, extract `to_spec` helpers (T3, T4, T9) |
| Create | `src/ferrum/_marks_statistical.py` | Statistical mark mixin (T4) |
| Create | `src/ferrum/_marks_diagnostic.py` | Diagnostic + model-selection mark mixin (T4) |
| Modify | `src/ferrum/composition.py` | Shared configure mixin (T7) |
| Create | `src/ferrum/_configure_mixin.py` | Unified `configure_*` methods (T7) |
| Test | All existing test suites | No regressions |

## 4. Constraints

- Zero public API changes — every user-facing method signature, return type, and import path must remain identical
- Each task must leave all tests green before the next begins — `uv run pytest -n auto` and `cargo test` (with DYLD_LIBRARY_PATH)
- `theme_from_dict` (audit item 6) is already serde-based (7 lines) — skip it
- Do not touch files being modified by other agents on the current branch — check `git status` before each task
- Mixin extraction (T4, T7): use multiple inheritance; `Chart` and `_ChartLike` inherit the mixin, existing method resolution order must be preserved
- `scale_resolve.rs` split (T8): keep `mod.rs` re-exporting all public items so no other file's `use` paths change

## 5. Tasks

### Task 1: Extract shared Rust render pipeline [Rust]
- [ ] Create `prepare_and_layout(spec, viewport, format) -> (PreparedInputs, LayoutResult, ThemeInputs, Vec<RenderWarning>)` in `render/mod.rs` containing the shared override-layering pipeline (~70 lines duplicated between `render_svg` at 569-673 and `render_scene_json` at 702-769)
- [ ] Include the secondary-Y padding block (lines 617-640) — it's currently missing from `render_scene_json`, which is a bug
- [ ] Rewrite `render_svg` and `render_scene_json` to call the new function, diverging only at SVG walk vs JSON serialization
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test`

### Task 2: Group legend overrides into sub-struct [Rust]
- [ ] Create `LegendPreparedOverrides` struct with the 11 legend override fields from `PreparedInputs` (lines 180-204: orient, title, title_font_size, columns, tick_count, label_font_size, gradient_length, gradient_thickness, direction, values, type_)
- [ ] Replace those 11 fields with a single `legend_overrides: LegendPreparedOverrides` field
- [ ] Update `prepare_render_inputs` extraction block and `legend_overrides_from_prep` to use the sub-struct
- [ ] Update consumers in `render/mod.rs` (now in `prepare_and_layout` from T1)
- [ ] Verify: `cargo test`

### Task 3: Auto-generate `Chart._clone` [Python]
- [ ] Replace the 27 manual slot assignments (chart.py:317-345) with a loop over `self.__slots__`, using `copy.copy()` for mutable containers (list, dict) and direct assignment for immutables
- [ ] Add a unit test asserting `set(Chart.__slots__) == set(vars(cloned))` to catch future drift
- [ ] Verify: `uv run pytest tests/ -n auto`

### Task 4: Split `chart.py` into mark-method mixins [Python]
- [ ] Create `_marks_statistical.py` with a `StatisticalMarksMixin` class containing: `mark_density`, `mark_histogram`, `mark_smooth`, `mark_boxplot`, `mark_boxen`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`, `mark_contour`, `mark_violin`, `mark_qq`, `mark_raster`, `mark_hex`, `mark_swarm`, `mark_function` (lines 972-1919, ~950 lines)
- [ ] Create `_marks_diagnostic.py` with a `DiagnosticMarksMixin` class containing: all diagnostic marks (lines 1920-2976, ~1,060 lines) and model-selection/clustering marks (lines 2977-3681, ~700 lines) — total ~1,760 lines
- [ ] `Chart` inherits from both mixins. Mixins reference `self._set_composite_mark`, `self._clone`, `self._resolve_pending` etc. via `self` (duck-typed, no circular import)
- [ ] Verify no import changes needed in `__init__.py` — `Chart` still lives in `chart.py`
- [ ] Verify: `uv run pytest tests/ -n auto`

### Task 5: Decompose `ThemeInputs` into sub-structs [Rust]
- [ ] Create sub-structs: `ThemePadding` (~10 fields, lines 117-129), `ThemeRenderSizes` (~11 fields, lines 132-142), `ThemeColors` (~7 fields, lines 145-151), `ThemeTypography` (~9 fields, lines 162-170), `ThemeLegend` (~3 fields, lines 189-195)
- [ ] Replace flat fields in `ThemeInputs` with sub-struct fields
- [ ] Refactor `apply_chart_config` (render/mod.rs:196-326) to delegate to per-sub-struct `apply_overrides` methods
- [ ] Update all consumers: `compute_layout`, scene builders, mark renderers
- [ ] Verify: `cargo test`

### Task 6: Unify `configure_*` via shared mixin [Python]
- [ ] Create `_configure_mixin.py` with `ConfigureMixin` defining `configure_axis`, `configure_legend`, `configure_title`, `configure_grid`, `configure_padding`, `configure_color`, and `configure` — each constructs the config object and calls `self._append_configure(config)` (abstract method)
- [ ] `Chart` implements `_append_configure` to append to `self._configure`; `_ChartLike` implements it to append to `self._configure_layers`
- [ ] Both classes inherit `ConfigureMixin`, removing ~500 lines of duplicated methods
- [ ] Verify: `uv run pytest tests/ -n auto`

### Task 7: Split `scale_resolve.rs` into sub-modules [Rust]
- [ ] Create `render/scale_resolve/` directory with `mod.rs`
- [ ] Extract: `positional.rs` (build_axis_scale, axis_pixel_range, resolve_continuous_domain_and_range — lines 598-1016), `color.rs` (build_color_scale — lines 1017-1104), `auxiliary.rs` (build_size_scale, build_shape_scale, build_opacity_scale — lines 1105-1194), `domain.rs` (numeric_domain_union, apply_sort_to_domain, locate_field — lines 664-852)
- [ ] Keep top-level types, `resolve_scales`, `resolve_scales_with_outputs`, and `dispatch_all!` in `mod.rs`
- [ ] Re-export all public items from `mod.rs` so no external `use` paths change
- [ ] Move tests into sub-module test files or a `tests.rs` submodule
- [ ] Verify: `cargo test`

### Task 8: Extract `to_spec()` internals [Python]
- [ ] Extract from `to_spec` (chart.py:5282-5535):
  - `_build_encoding_specs(self) -> dict` — channel aliasing, aggregate field remapping, unrecognized-channel warnings
  - `_resolve_polar_remapping(self, encoding_specs) -> dict` — CoordPolar x↔angle, y↔radius
  - `_resolve_pending_aggregates(self, encoding_specs) -> dict` — pending aggregate resolution
- [ ] `to_spec()` becomes a ~50-line orchestrator calling these helpers
- [ ] Verify: `uv run pytest tests/ -n auto`

### Task 9: Factor `ContinuousScaleCommon` [Rust]
- [ ] Create `ContinuousScaleCommon { domain: Option<Vec<f64>>, range: Option<Vec<f64>>, clamp: bool, padding: Option<f64> }`
- [ ] Refactor 7 `ScaleSpec` variants (Linear, Log, Time, Symlog, Pow, Sqrt, Utc) to embed `#[serde(flatten)] common: ContinuousScaleCommon` plus their variant-specific fields
- [ ] Unify `Encoding::inherit_from` and `inherit_non_positional`: extract the inner `inherit` closure to a module-level function; use a const array `POSITIONAL_CHANNELS` to drive the skip logic in `inherit_non_positional`
- [ ] Verify: `cargo test`

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass
- `chart.py` line count reduced by ~2,700+ lines
- `composition.py` `configure_*` methods removed (~240 lines)
- `scale_resolve.rs` split into 4+ sub-modules
- `PreparedInputs` legend fields reduced from 11 to 1
- `render_svg`/`render_scene_json` share a single pipeline function
- No public API changes — `from ferrum import Chart` and all method signatures unchanged
