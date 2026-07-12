# Band/Point Range — Band-Geometry Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Make the resolved ordinal scale the single source of truth for band geometry (mark widths, heatmap cells, tick extents, categorical axis-tick placement) when an explicit pixel range is set, byte-identical otherwise — GH #39 phase 2.

## 2. Spec references

- `design-docs/superpowers/specs/2026-07-11-band-point-range-geometry-design.md` — all sections; §6 (interface contract), §7 (invariants), §9 (acceptance) are binding.
- Phase 1 already on branch `fix/band-point-scale-range` (uncommitted): `ScaleSpec::Band/Point.range`, resolver `band_point_pixel_range`, RED regression tests in `tests/test_regression_band_point_range.py`.

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/scale/ordinal.rs` | record range-explicitness on `OrdinalScale` |
| Modify | `crates/ferrum-core/src/render/scale_resolve/mod.rs` | `ScaleKind::explicit_band_extent()` accessor |
| Modify | `crates/ferrum-core/src/render/scale_resolve/positional.rs` | explicit-range arms (Band/Point/Ordinal) mark the resolved scale explicit |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` | bar w (`:312`) / h (`:431`) from explicit extent |
| Modify | `crates/ferrum-core/src/render/marks/rect.rs` | box w/h (`:224/:302`), heatmap cell w/h (`:396/:397`) |
| Modify | `crates/ferrum-core/src/render/marks/tick.rs` | tick half-extent (`:70/136/176/217`) |
| Modify | `crates/ferrum-core/src/render/prepare/mod.rs` + `crates/ferrum-core/src/layout/axis.rs` | categorical tick placement via scale band centers when explicit (uniform_center sites `axis.rs:874/:1349`; fraction seam `prepare/mod.rs:1365`, `scale_resolve/mod.rs:257 tick_fractions`) |
| Test | `tests/test_regression_band_point_range.py` | extend with spec §9 behavioral checks |
| Test | in-file Rust `mod tests` of each touched module | contract + geometry units |

## 4. Constraints

- **Byte-identity without explicit range (spec §7):** the no-range path must run today's exact arithmetic (`panel.w` / `panel.h` expressions unchanged); consumer pattern is `scale.explicit_band_extent().map(f64::abs).unwrap_or(panel_extent)`. No golden may change; goldens are never regenerated in this work.
- **Explicitness is recorded at construction by the resolver, never inferred by float comparison** (spec §8). The panel-extent fallback range yields `None` from `explicit_band_extent()`.
- **Mark width formulas keep their shape factors** (0.8 ratio, `band_size`, `/ n_groups`); only the extent term changes (spec §8). Do not switch widths to `bandwidth()`.
- **Band, Point, and positional Ordinal scales behave identically** — none special-cased (spec §8).
- **Polar arms (`tau / n_cats`) untouched** (spec §3).
- Rebuild before any pytest: `unset CONDA_PREFIX && uv run --no-sync maturin develop`.
- Rust tests: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core` (polluted shells also need `PYTHONHOME`/`PYO3_PYTHON`/`RUSTFLAGS` per memory).
- Clippy judged by delta only (~166–167 pre-existing failures on baseline).
- Coding dispatch: `rust-coder` for `.rs`, `python-coder` for `.py`; lite-review gates before any commit.

## 5. Tasks

### Task 1: Explicitness contract on the resolved scale
- [ ] `OrdinalScale` records whether its pixel range was user-supplied (resolver-set at construction; default false so every existing constructor call is unchanged).
- [ ] `ScaleKind::explicit_band_extent() -> Option<f64>` per spec §6 (signed `r1 − r0`; `Some` only for explicit ordinal positional scales).
- [ ] `build_from_scale_spec` Band/Point arms and the Ordinal arm's `ordinal_pixel_range` path mark the scale explicit exactly when the spec carried a usable (≥2-entry numeric) range.
- [ ] Rust unit tests: explicit → `Some(extent)` incl. reversed range (negative extent); fallback → `None`; non-ordinal → `None`.
- [ ] Verify: `cargo test -p ferrum-core` (env per §4) — full crate green.

### Task 2: Behavioral tests first (RED)
- Consumes: nothing new — runs against the current build, where phase 1 is live but geometry consumers are not.
- [ ] Extend `tests/test_regression_band_point_range.py` with spec §9 checks: ordinal-y horizontal bar (heights + y-tick alignment), heatmap cell extent = `|range|/n` on the ranged axis, tick-mark half-extent, x-axis tick-label centers == mark band centers, dodged bars non-overlapping within range.
- [ ] Confirm each new test FAILS against the current build (discriminating RED); the existing `test_band_scale_range_constrains_bar_positions` stays RED.
- [ ] Verify: `uv run pytest tests/test_regression_band_point_range.py -q` — new tests fail for geometry reasons only (no import/setup errors); wire-level tests still pass.

### Task 3: Mark builders consume the explicit extent
- Consumes: `explicit_band_extent()` from Task 1 (spec §6 consumer contract).
- [ ] Apply the consumer pattern at `bar.rs:312/:431`, `rect.rs:224/:302/:396/:397`, `tick.rs:70/136/176/217`, matching each site's axis (x → `panel.w`, y → `panel.h`).
- [ ] Rust unit tests in each mark's `mod tests`: explicit-range scale → geometry scales with `|extent|`; no-range → identical values to before.
- [ ] Verify: `cargo test -p ferrum-core` full crate; then rebuild + `uv run pytest tests/test_regression_band_point_range.py -q` — bar-width and heatmap/tick tests green; axis-alignment test may remain RED (Task 4).

### Task 4: Categorical axis ticks through the scale
- Consumes: `explicit_band_extent()` / band centers from Task 1; uniform_center + fraction seams per §3 Files row.
- [ ] When the categorical axis's resolved scale is explicit, tick label/grid positions = scale band centers (same pixels marks get from `to_pixel_str`), both channels; ordinal-y non-reversal semantics preserved (spec §5).
- [ ] No explicit range → `uniform_center` path byte-identical (spec §7).
- [ ] Rust unit tests at the chosen seam (x and y, explicit vs not); keep SPINE-08 x/y parity tests green.
- [ ] Verify: `cargo test -p ferrum-core` full crate; rebuild + `uv run pytest tests/test_regression_band_point_range.py -q` — ALL tests green including axis alignment and phase-1 bar-positions.

### Task 5: Whole-change verification
- [ ] `uv run pytest -n auto` — full suite, zero failures (goldens byte-identical).
- [ ] `cargo test` (env per §4) — full workspace green.
- [ ] Render one explicit-range band chart + one explicit-range heatmap to SVG, rasterize via `tests/_snapshots.py` helpers, Read the PNGs, confirm visually coherent (marks within range, ticks aligned) — orchestrator inspects, not a subagent claim.
- [ ] Verify: commands above; visual confirmation recorded in task notes.

## 6. Acceptance checks

- `uv run pytest tests/test_regression_band_point_range.py tests/test_migration_compat.py tests/test_scale_spec_parity.py -q` — all pass.
- `uv run pytest -n auto` — full suite green, no golden regenerated.
- `cargo test` — green.
- Spec §9 criteria all observable: marks and axis ticks within/aligned to `[a, b]`; no-range output byte-identical.

## 7. Open questions

None.
