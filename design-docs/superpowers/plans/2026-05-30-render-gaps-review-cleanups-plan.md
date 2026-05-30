# Render-gaps review cleanups (R1–R4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Apply four behavior-preserving cohesion/cosmetic cleanups (R1–R4) from the rust-review of the Grid/render-gaps work, with every test and golden staying byte-identical.

## 2. Spec references

- `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` — "Review findings" section, rows R1–R4 (severity, file, disposition)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/scale_resolve/mod.rs` | R1: route `minor_tick_fractions` through `project_values_to_fractions` |
| Modify | `crates/ferrum-core/src/layout/axis.rs` | R2: group projection fields into `Option<TickProjection>`; R3 helper lives nearby if shared (else see below) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | R2: construct the grouped projection type |
| Modify | `crates/ferrum-core/src/render/marks/axis.rs` | R3: extract one `emit_gridlines` helper, call 4× |
| Modify | `crates/ferrum-core/src/render/binding.rs` | R4: replace stale `theme` docstring key-list with a pointer to `ThemeOverridesSpec` |

## 4. Constraints

- **Behavior-preserving refactors only.** No new behavior, no Python-API change. The definition of done is: all existing tests pass AND every golden is byte-identical (no diffs under `tests/goldens/**` or `crates/ferrum-core/tests/golden/**`).
- rust-coder only; never general-purpose.
- **R2 invariant:** categorical axes (`None` projection) keep byte-identical uniform-slot output; continuous axes keep byte-identical projected output; minor-off byte-identical. Confirm `AxisInput` has no `#[pyclass]`/`FromPyObject` (it's internal) — no Python churn.
- **R2 gate reconciliation:** the `theme.grid.minor` gate in `prepare.rs` still decides whether minor fractions are populated; dropping the separate `include_minor` bool means deriving "minor enabled" from the grouped type's presence/non-emptiness — keep the theme gate as the source of truth.
- **R1 policy:** major callers (`tick_fractions`, `value_fractions`) must keep all-or-nothing non-finite handling (index-aligned with labels); minor keeps per-element drop. Express the difference as a named policy, not a duplicated loop.
- **R3:** node order identical (minors before majors); per-level baseline-skip (`<0.5`) and `show_grid` filter preserved.
- Rust env (conda/pdm-safe): `unset CONDA_PREFIX PYTHONPATH; export PYO3_PYTHON="$PWD/.venv/bin/python"; LIB=$(.venv/bin/python -c "import sys; print(sys.base_prefix + '/lib')"); export PYTHONHOME=$(.venv/bin/python -c "import sys; print(sys.base_prefix)"); export RUSTFLAGS="-L $LIB"; export DYLD_LIBRARY_PATH="$LIB"`.
- After all four, invoke `/regression-test` (here it confirms the refactors added no behavior — goldens byte-identical is the evidence).

## 5. Tasks

Order: R4 → R1 → R3 → R2 (safest/most-mechanical first; R2 highest radius last). Each independently committable.

### Task 1 (R4): Fix stale binding docstrings
- [ ] In `render_svg`/`render_png`, replace the abbreviated/stale enumerated `theme` key list with a one-line pointer to `ThemeOverridesSpec` (the authoritative, `deny_unknown_fields` list). Doc-only.
- [ ] Verify: `cargo build -p ferrum-core` (env above); no golden/test changes.

### Task 2 (R1): Unify fraction projection
- [ ] Route `minor_tick_fractions` through the existing private `project_values_to_fractions`, adding an explicit non-finite policy (named enum or two named wrappers): major → all-or-nothing (RejectAll), minor → per-element drop (DropOne). Remove the duplicated `(px - r0)/span` loop.
- [ ] Verify: `cargo test -p ferrum-core --lib scale_resolve::` green, incl. `fractions_on_zero_span_domain_are_empty_not_nan`; goldens unchanged.

### Task 3 (R3): Collapse build_grid emission loops
- [ ] Extract one local helper (e.g. `emit_gridlines(nodes, ticks, orient, plot_area, baseline_coord, style)`); call it for x-minor, y-minor, x-major, y-major. Preserve node order, baseline-skip, `show_grid` filter.
- [ ] Verify: `cargo test -p ferrum-core --lib` green; in-crate goldens byte-identical.

### Task 4 (R2): Group AxisInput projection fields
- [ ] Replace `include_minor` + `minor_tick_positions` + `projected_tick_fractions` + `scale_padding_frac` with one `Option<TickProjection>` (`{ padding_frac, major, minor }`; `None` = categorical/uniform-slot). Derive "minor enabled" from the grouped value + theme gate; delete the standalone `include_minor` bool.
- [ ] Update construction in `prepare.rs` and consumption in `layout/axis.rs` (`project_tick_positions`, `build_minor_ticks`, `layout_x_axis`, `layout_y_axis`).
- [ ] Confirm `AxisInput` has no `#[pyclass]`/`FromPyObject`.
- [ ] Verify: full `cargo test -p ferrum-core --lib` green; all axis-layout unit tests pass; goldens byte-identical.

### Task 5: Final gate
- [ ] `cargo test` (env above), `unset CONDA_PREFIX && uv run --no-sync maturin develop`, `uv run pytest -n auto`, `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — all green.
- [ ] Confirm zero golden diffs (`git status` shows no `*.svg` / `*.sha256` changes).
- [ ] `/regression-test`.

## 6. Acceptance checks

- `cargo test` (env above) — all pass (1038 lib + integration)
- `uv run pytest -n auto` — all pass (4665)
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- **Zero golden diffs** — no changes under `tests/goldens/**` or `crates/ferrum-core/tests/golden/**` (the behavior-preserving proof)
- `minor_tick_fractions` no longer duplicates the projection loop; `AxisInput` projection fields grouped; `build_grid` emits via one helper; binding docstrings point to `ThemeOverridesSpec`

## 7. Open questions

- None. (R5 WASM id-collision is a separate immediate fix; R6/R7 deferred per the archaeology Review-findings section.)
