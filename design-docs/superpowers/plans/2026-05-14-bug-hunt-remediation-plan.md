# Bug Hunt Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Fix 14 bugs across 8 subsystems identified in the 2026-05-14 bug hunt run, grouped into four independent themes: renderer robustness (zero-row + degenerate domain), Python coercion safety, Rust math correctness, and structural fixes (FacetSpec row/col + wasm compilation).

## 2. Spec references

- `.claude/skills/bug-hunt/output/BUG_REPORT.md` — all bug descriptions, root causes, and proposed fixes
- `tests/test_bug_hunt_coerce_transport.py` — failing test: `test_regular_dict_with_type_key_not_geojson`
- `tests/test_bug_hunt_composition_facet.py` — failing tests: `test_joint_chart_with_marginals_renders`, `test_facet_grid_mode_spec_round_trip`
- `tests/test_bug_hunt_figure_api.py` — failing tests: three zero-row variants
- `tests/test_bug_hunt_marks_rendering.py` — failing test: `test_point_single_row_emits_one_circle`
- `tests/test_bug_hunt_scale_stat.py` — failing tests: `test_single_row_no_explicit_scale_renders_no_nan`, `test_all_equal_values_no_nan_in_svg`
- `crates/ferrum-core/tests/bug_hunt_projection.rs` — failing test: `test_natural_earth_high_latitude_round_trip`
- `crates/ferrum-core/tests/bug_hunt_stats_transforms.rs` — failing test: `test_shapiro_w_n5_inner_loop_runs_once`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/prepare.rs` | zero-row guard + degenerate domain expansion |
| Modify | `crates/ferrum-core/src/render/mod.rs` | propagate empty-batch as empty-axes SVG, not error |
| Modify | `src/ferrum/_coerce.py` | GeoJSON type-key isinstance guard |
| Modify | `crates/ferrum-core/src/transform/bin.rs` (or `stat_bin`) | cast Int64 → Float64 before binning |
| Modify | `crates/ferrum-core/src/layout/facet.rs` | add `row_field: Option<String>`, keep `field` as col |
| Modify | `crates/ferrum-core/src/spec/chart.rs` | update FacetSpec construction in tests |
| Modify | `src/ferrum/chart.py` | `_build_facet_dict()` — emit `row_field` key |
| Modify | `crates/ferrum-core/src/projection.rs` | fix Natural Earth inverse Jacobian (product rule) |
| Modify | `crates/ferrum-core/src/transform/stats.rs` | guard `eps.max(0.0).sqrt()` in `shapiro_w_scalar` |
| Modify | `crates/ferrum-wasm/src/gpu.rs` | gate `SurfaceTarget::Canvas` with `#[cfg(target_arch = "wasm32")]` |

## 4. Constraints

- No matplotlib. No global mutable state.
- `cargo test` must pass before marking done; use `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")` prefix on macOS.
- Zero-row guard must produce a valid, renderable empty-axes SVG — not an empty string and not an error.
- Degenerate domain expansion must only fire for auto-inferred domains, not for explicit user-supplied `LinearScale(domain=[v, v])`.
- `FacetSpec` wire format change is additive: `row_field` is `Option<String>` — existing single-dimension facets (wrap mode) still serialize without it.
- The `gpu.rs` fix must not break the wasm build (`wasm-pack build crates/ferrum-wasm --target web`).
- All failing bug-hunt tests must pass after their task; do not delete or skip them.

## 5. Tasks

### Task 1: Renderer robustness — zero-row DataFrame
- [ ] In `prepare.rs`, when `batch.num_rows() == 0`, skip stat transforms and scale inference; produce a zero-row `PreparedScene` with empty axes
- [ ] In `render/mod.rs`, handle the zero-row path by returning an SVG with axes and an empty plot area instead of propagating `RenderError::EmptyBatch` to Python
- [ ] Verify: `uv run pytest tests/test_bug_hunt_figure_api.py tests/test_bug_hunt_phase_11_interactive.py -k "zero_row" --tb=short`

### Task 2: Renderer robustness — degenerate auto-domain
- [ ] In `prepare.rs`, where continuous domain is inferred from column min/max, detect `lo == hi` and expand to `[lo - 0.5, hi + 0.5]` (or `[lo - 1, lo + 1]` if `lo == 0`)
- [ ] Guard must only fire when domain was inferred, not when user supplied explicit scale domain
- [ ] Verify: `uv run pytest tests/test_bug_hunt_marks_rendering.py tests/test_bug_hunt_scale_stat.py -k "single_row or equal_values" --tb=short`

### Task 3: Python coerce — GeoJSON type-key guard
- [ ] In `_coerce.py`, add `isinstance(data.get("type"), str)` before the frozenset membership test in both `_is_geojson_geometry_root` and `_is_geojson_feature_collection`
- [ ] Verify: `uv run pytest tests/test_bug_hunt_coerce_transport.py --tb=short`

### Task 4: stat_bin Int64 auto-cast
- [ ] In the `Bin` / `stat_bin` Rust transform, cast any integer (`Int8`–`Int64`, `UInt8`–`UInt64`) input column to `Float64` before running the binning logic
- [ ] Verify: `uv run pytest tests/test_bug_hunt_composition_facet.py -k "marginals" --tb=short`

### Task 5: FacetSpec row + col fields
- [ ] In `layout/facet.rs`, add `pub row_field: Option<String>` to `FacetSpec`; existing `field` becomes the column field
- [ ] In `layout/facet.rs` and the facet renderer, use `row_field` for row-axis grouping when present
- [ ] In `src/ferrum/chart.py` `_build_facet_dict()`, emit `"row_field": row_col_name` alongside `"field"` when both row and col are specified
- [ ] Update any `FacetSpec { field: ..., mode: ... }` construction in Rust tests to compile
- [ ] Verify: `uv run pytest tests/test_bug_hunt_composition_facet.py -k "grid_mode" --tb=short`

### Task 6: Natural Earth inverse Jacobian
- [ ] In `projection.rs`, fix the Newton-Raphson Jacobian for the Natural Earth inverse: the derivative of `phi * ne_poly(NE_B, phi)` must apply the product rule — `ne_poly(NE_B, phi) + phi * ne_poly_deriv(NE_B, phi)`
- [ ] Verify: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core --tests -- bug_hunt_projection --nocapture`

### Task 7: shapiro_w negative eps guard
- [ ] In `transform/stats.rs` `shapiro_w_scalar`, replace `eps.sqrt()` with `eps.max(0.0).sqrt()` at the coefficient computation step; add a comment noting the guard prevents NaN propagation at n=5 with linear input
- [ ] Verify: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core --tests -- bug_hunt_stats_transforms --nocapture`

### Task 8: ferrum-wasm gpu.rs compilation fix
- [ ] In `gpu.rs`, wrap the `wgpu::SurfaceTarget::Canvas` usage (line 36 and any surrounding canvas-specific code) with `#[cfg(target_arch = "wasm32")]`; provide a stub or compile error for non-wasm builds
- [ ] Verify wasm build still works: `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`
- [ ] Verify host tests now compile: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-wasm -- bug_hunt --nocapture`

## 6. Acceptance checks

- `uv run pytest tests/test_bug_hunt_*.py --tb=short -q` — all pass
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core --tests -- bug_hunt --nocapture` — all pass
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-wasm -- bug_hunt --nocapture` — all pass
- `uv run pytest --tb=short -q` — full suite still passes (no regressions)

## 7. Open questions

- Task 2 (degenerate domain): should expansion be `±0.5` unconditionally, or `±(abs(v) * 0.1)` for non-zero values? The bug reports suggest `±0.5`; verify this looks reasonable for non-unit data (e.g. v=10000).
- Task 5 (FacetSpec): does the Rust facet renderer in `layout/facet.rs` use `FacetSpec.field` directly for both row and col iteration, or does it have separate row/col paths already? Check before adding `row_field` to avoid duplicating logic.
- Task 8 (gpu.rs): confirm whether a non-wasm stub for `init_gpu` is needed for tests, or whether the whole `gpu` module should be `#[cfg(target_arch = "wasm32")]`-gated at the module level.
