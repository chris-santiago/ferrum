# Continuous-axis scale-projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Place continuous-axis (linear/time/log/symlog/pow/sqrt) major ticks and gridlines by scale projection — the same domain→pixel mapping that positions data marks — while categorical/discretizing axes keep uniform-slot placement; prerequisite that aligns continuous gridlines with data and unblocks item 18 minor ticks.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-30-continuous-axis-scale-projection-design.md` — all decisions locked (§4 behavior, §5 architecture, §6 contract, §7 invariants, §8 decisions)
- `design-docs/superpowers/specs/2026-05-30-grid-minor-ticks-design.md` — downstream consumer (item 18); minors already scale-projected

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/layout/axis.rs` | AxisInput carries projected tick pixels; layout_x/y place at them when present; cascade uses real gaps |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | supply per-tick projected pixels (continuous) / none (categorical) into AxisInput |
| Modify | `crates/ferrum-core/src/render/scale_resolve/mod.rs` | expose resolved-scale projected tick pixels if not already reachable (`tick_data`) |
| Test | `crates/ferrum-core` unit tests | continuous placement, categorical-unchanged, cascade non-uniform gaps |
| Test | `tests/` (Python/SVG) | tick↔mark coincidence regression; log non-uniform; categorical byte-identical |
| Modify | `tests/goldens/**` (continuous-axis only) | regenerate + visually inspect; categorical untouched |

## 4. Constraints

- Coding agents only: `.rs` → rust-coder, `.py` → python-coder; never general-purpose.
- **Categorical/discretizing byte-identity.** Ordinal/band/point/quantile/threshold/bin-ordinal axes must produce byte-identical SVG. They supply NO projected tick pixels; layout keeps the uniform-slot formula for them.
- **Tick ↔ mark coincidence.** Continuous tick pixels MUST come from the resolved positional scale via `tick_data` using the SAME 8px-capped inset (`inset_pixel_range`) that places marks — so a tick at value `v` and a mark at value `v` share a pixel. Do NOT introduce a new projection path or inset/padding constant.
- **Cascade safety.** The x-axis collision cascade must use ACTUAL per-tick gaps (or the min gap) for continuous axes; uniform-slot cascade stays for categorical. Non-uniform (log/pow/symlog) spacing must not regress label legibility.
- **Item-18 gate stays OFF.** `include_minor` remains false; this change does not enable minor rendering. Once majors are scale-projected, the already-committed minor path aligns with no item-18 change.
- **Goldens are the central risk** and are NOT blessed until visually inspected: regenerate changed continuous goldens via `scripts/snapshot-goldens.py` (or `tests/_snapshots.py` `regen_and_verify`/`rasterize_svg`), `Read` each PNG, confirm gridlines pass through data and no blank/misdrawn panels, BEFORE committing.
- After code changes invoke `/regression-test`.
- Rust env (conda/pdm-safe): `unset CONDA_PREFIX PYTHONPATH; export PYO3_PYTHON="$PWD/.venv/bin/python"; LIB=$(.venv/bin/python -c "import sys; print(sys.base_prefix + '/lib')"); export PYTHONHOME=$(.venv/bin/python -c "import sys; print(sys.base_prefix)"); export RUSTFLAGS="-L $LIB"; export DYLD_LIBRARY_PATH="$LIB"`.

## 5. Tasks

Sequence is locked: 1 → 2 → 3. Tasks 1+2 may be done together (same files) at the implementer's judgment.

### Task 1: Thread projected tick pixels into AxisInput (rust-coder)
- [ ] Add an optional per-axis carrier on `AxisInput` for scale-projected tick pixel positions (one per tick label, same order); absent for categorical (spec §5, §6).
- [ ] `prepare.rs`: for continuous axes, supply the projected pixels from the resolved positional scale via `tick_data` (same inset as marks); for categorical/discretizing axes supply none. Reuse `scale_resolve` projection — no new path/inset (Constraints).
- [ ] Verify: `cargo build -p ferrum-core` (env above) compiles; categorical path supplies nothing.

### Task 2: Place at projected pixels + adapt cascade (rust-coder)
- [ ] `layout_x_axis` / `layout_y_axis`: set `TickLayout.position` to the supplied projected pixel when present (continuous); else uniform-slot fallback (categorical) unchanged (spec §4, §5; decision 3).
- [ ] Adapt the x collision cascade to use real per-tick gaps (or min gap) for continuous; keep uniform-slot cascade for categorical (decision 4).
- [ ] Rust unit tests: continuous ticks placed at supplied projected pixels; categorical placement byte-identical (uniform slots); cascade exercised with non-uniform gaps.
- [ ] Verify: `cargo test -p ferrum-core --lib` (env above).

### Task 3: Behavioral + regression tests + golden regeneration (python-coder; rust-coder if Rust test files)
- [ ] Regression test: on a linear axis, gridline x-positions == data-mark cx-positions for shared values (pins tick↔mark coincidence; the audit's 64.9/320.5/576.0 marks now match gridlines).
- [ ] Behavioral: log/pow/symlog axis shows non-uniform gridlines at projected pixels; a categorical-axis chart's SVG is byte-identical.
- [ ] Rebuild extension (`unset CONDA_PREFIX && uv run --no-sync maturin develop`); run full `uv run pytest -n auto`; identify every changed continuous-axis golden.
- [ ] Regenerate changed continuous goldens; `Read` each PNG and confirm correct render (gridlines through data, no blank/misdrawn panels) per CLAUDE.md. Confirm categorical goldens byte-identical.
- [ ] `/regression-test`.
- [ ] Verify: `cargo test` (env above) + `uv run pytest -n auto` + `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` all green.

## 6. Acceptance checks

- `cargo test` (env above) — all pass (continuous placement, categorical-unchanged, cascade non-uniform)
- `uv run pytest -n auto` — all pass (tick↔mark coincidence regression, log non-uniform, categorical byte-identical)
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- Linear gridline pixels == scale-projected tick pixels == mark pixels for shared values
- Log/pow/symlog/time: non-uniform gridlines at projected pixels
- Categorical/discretizing goldens byte-identical; all changed continuous goldens visually inspected
- `include_minor` gate still off; item-18 minor path needs no change to align

## 7. Open questions

- None blocking. (Out of scope, follow-up: reconcile the interactive `TickLevel`/`tick_data` zoom-tick grid with the static projected grid.)
