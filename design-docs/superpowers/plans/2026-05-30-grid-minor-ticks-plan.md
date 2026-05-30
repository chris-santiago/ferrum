# ferrum.Grid + minor-tick subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Implement the `ferrum.Grid` theme-level value class (§3.19) backed by a real minor-tick generation subsystem, on `feat/render-gaps-17-19-21`.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-30-grid-minor-ticks-design.md` — all decisions locked (§4 behavior, §5 architecture, §6 contracts, §7 invariants, §8 decisions)
- `ferrum-spec.md §3.19` — `Grid` constructor (L1607-1611), Theme shorthand example (L1001)
- `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` — item 18 row to update on completion

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/scale/ticks.rs` + per-scale `scale/{linear,log,time,pow,sqrt,symlog}.rs` | `Tick` type + minor generation |
| Modify | `crates/ferrum-core/src/layout/axis.rs` | `TickLayout.is_major`; thread minor ticks |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | `ThemeGrid`/`ThemeColors`/`ThemeRenderSizes` per-level fields + builtin minor defaults |
| Modify | `crates/ferrum-core/src/render/marks/axis.rs` | `build_grid()` two-level emission |
| Modify | `crates/ferrum-core/src/render/binding.rs` | `ThemeOverridesSpec` per-level keys + `apply_theme_overrides` |
| Create | `src/ferrum/grid.py` | `Grid` frozen dataclass + `to_spec_dict()` |
| Modify | `src/ferrum/themes/__init__.py` | `Theme.to_spec_dict()` value-object ingestion of `grid` |
| Modify | `src/ferrum/__init__.py` | export `Grid` in `__all__` |
| Modify | `ferrum-spec.md` §3.19 | dated note: bare shorthand = both-levels fallback |
| Test | `crates/ferrum-core` unit tests, `tests/` (Grid + minor goldens) | per-task coverage |

## 4. Constraints

- Coding agents only: `.rs` → rust-coder, `.py` → python-coder; never general-purpose.
- **Byte-identical non-minor output.** Charts that do not enable minor render byte-identical SVG; minor emission gated on minor enabled. Existing goldens unchanged.
- **Major tick positions unchanged** across every scale — the `Tick` refactor preserves today's major output exactly.
- **GridConfig API unchanged** (now formally the major level); no `major_*`/`minor_*` on `GridConfig`.
- **Categorical/discretizing minor = no-op, not error.** ordinal/band/point/quantile/threshold/bin-ordinal render no minor lines and do not raise.
- **Shared key contract (tasks 3 ↔ 4 must agree exactly):** `Grid.to_spec_dict()` emits per-level keys `major_color`/`minor_color`, `major_width`/`minor_width`, `major_dash`/`minor_dash`, `major_opacity`/`minor_opacity`, plus `major`/`minor` booleans; never the bare shorthand. The Rust `ThemeOverridesSpec` per-level key names must match these exactly. Decide the spelling once in Task 3 and reuse verbatim in Task 4.
- Minor algorithm (spec §8): default = subdivide major intervals in transformed space; **log override = 2-9 intra-decade**; continuous only.
- Rust env (conda/pdm-safe): `unset CONDA_PREFIX PYTHONPATH; export PYO3_PYTHON="$PWD/.venv/bin/python"; LIB=$(.venv/bin/python -c "import sys; print(sys.base_prefix + '/lib')"); export PYTHONHOME=$(.venv/bin/python -c "import sys; print(sys.base_prefix)"); export RUSTFLAGS="-L $LIB"; export DYLD_LIBRARY_PATH="$LIB"`.
- After code changes invoke `/regression-test`. New `minor=True` golden must be rasterized + visually inspected (CLAUDE.md) before commit.

## 5. Tasks

Build order is locked: 1 → 2 → 3 → 4 → 5. (Task 4 Python may proceed in parallel with 1-3 against the Task 3 key contract, but the key spelling must be fixed in Task 3 first.)

### Task 1: Rust — `Tick` type + minor generation (rust-coder)
- [ ] Introduce `Tick { position: f64, is_major: bool }` at the scale tick-generation boundary; major output identical to today (same positions, `is_major=true`) (spec §5, §6).
- [ ] Add minor generation: default subdivision in transformed space for linear/pow/sqrt/symlog/time; **log override = 2-9 per decade**; categorical (ordinal/band/point) + discretizing (quantile/threshold/bin-ordinal) return no minors (spec §8).
- [ ] Tests: per-scale minor generation (linear subdivision counts, log 2-9 placement, time subdivision, categorical→empty); major-positions-unchanged for every scale.
- [ ] Verify: `cargo test` (env above).

### Task 2: Rust — thread minor ticks through layout (rust-coder)
- [ ] `TickLayout` gains `is_major: bool`; layout carries minor ticks to render only when minor rendering is enabled; major `TickLayout` output unchanged.
- [ ] Tests: minor ticks present in layout only when enabled; major layout unchanged.
- [ ] Verify: `cargo test` (env above).

### Task 3: Rust — per-level styling + `build_grid()` emission (rust-coder)
- [ ] `ThemeGrid`/`ThemeColors`/`ThemeRenderSizes` gain `major_*`/`minor_*` grid fields; builtin theme sets derived lighter/thinner minor defaults (spec §8).
- [ ] `ThemeOverridesSpec` gains matching per-level keys; `apply_theme_overrides` wires them. **Fix the per-level key spelling here (shared contract, Constraints).**
- [ ] `build_grid()` emits major + minor `SceneNode::Line` batches, each styled from its level, minor drawn first (under major), minor emission gated on minor enabled.
- [ ] Tests: two-level emission/styling; non-minor output byte-identical.
- [ ] Verify: `cargo test` (env above) + `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`.

### Task 4: Python — `Grid` value class + Theme ingestion (python-coder)
- [ ] Create `src/ferrum/grid.py`: frozen dataclass `Grid` (§3.19 signature + bare `color`/`width`/`dash`/`opacity`). Mirror `src/ferrum/title.py`.
- [ ] `to_spec_dict()`: resolve shorthand → per-level (bare sets both; explicit per-level wins); emit only non-None per-level keys + `major`/`minor` booleans; never emit the bare key. Keys must match the Task 3 contract exactly.
- [ ] Export `Grid` in `src/ferrum/__init__.py __all__`.
- [ ] `Theme.to_spec_dict()`: if the `grid` prop has `.to_spec_dict()`, call it before serialization.
- [ ] Tests: construction per §3.19; shorthand resolution (bare→both, per-level override); Theme ingestion; `minor=True` reaches Rust.
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync maturin develop` then `uv run pytest -n auto -k "grid or theme"`.

### Task 5: Goldens + docs (python-coder)
- [ ] Add a new `minor=True` continuous-scale golden; rasterize + visually inspect per CLAUDE.md; confirm existing goldens byte-identical.
- [ ] Add dated note to `ferrum-spec.md §3.19`: bare `color=`/`width=`/`dash=`/`opacity=` shorthand is a both-levels fallback (full signature omits it).
- [ ] Update archaeology doc item-18 row to fixed.
- [ ] `/regression-test`.

## 6. Acceptance checks

- `cargo test` (env above) — all pass (Tick/minor generation, layout threading, two-level emission)
- `uv run pytest -n auto` — all pass (Grid construction/shorthand/Theme ingestion, minor golden)
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- Existing SVG goldens byte-identical; new `minor=True` golden inspected
- `ferrum.Grid` in `__all__`; `Theme.update(grid=fr.Grid(...))` ingests it; log minors at 2-9, linear/time minors subdivide, categorical/discrete produce none
- `ferrum-spec.md §3.19` note added; archaeology doc item-18 updated

## 7. Open questions

- None. (Per-level theme-key spelling is an implementation detail resolved in Task 3 and reused verbatim in Task 4 — see Constraints shared-key contract.)
