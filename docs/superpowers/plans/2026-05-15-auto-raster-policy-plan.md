# Auto-Raster Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Wire the auto-raster policy so charts exceeding `raster_threshold` marks transparently substitute `mark_raster`, producing compact SVG output without user intervention.

## 2. Spec references

- `docs/superpowers/specs/2026-05-15-auto-raster-policy-design.md` — full spec
- `ferrum-spec.md §3.3` — auto-raster substitution rules and color-encoding guard
- `ferrum-spec.md §3.17` — backend selection table and RenderConfig fields

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `src/ferrum/render_config.py` | `RenderConfig` dataclass |
| Modify | `src/ferrum/chart.py` | Add `_render_config` slot, `_apply_auto_raster()`, update `_render_inputs()` and `properties()` |
| Modify | `src/ferrum/__init__.py` | Export `RenderConfig` |
| Modify | `crates/ferrum-core/src/render/config.rs` | Add `raster_threshold` and `raster_behavior` fields (forward compat) |
| Test | `tests/test_scale_rendering.py` | Add `TestAutoRaster` class; update 1M assertions |

## 4. Constraints

- **No silent data loss.** Color encoding present → auto-raster must NOT fire. Emit guidance warning instead. (Spec §3.3, design spec §4 condition 3.)
- **Eligible mark types only:** `point`, `bar`, `rect`, `tick`, `rule`, `segment`. Line/area/hex/raster are excluded.
- **Requires both x and y quantitative.** Raster needs a 2D numeric field.
- **No new dependencies.** Reuse existing `desugar_raster()` from `marks/heavy_stat.py`.
- **Backward compatible.** 500k default threshold is above all existing test data sizes.
- **Docstrings** on `show_svg()`, `show()`, `save()`, and `RenderConfig` must document auto-raster and the `raster_threshold=None` escape hatch.

## 5. Tasks

### Task 1: RenderConfig dataclass
- [ ] Create `src/ferrum/render_config.py` with the `RenderConfig` dataclass (spec §6 interface)
- [ ] Export from `src/ferrum/__init__.py`
- [ ] Add `_render_config` slot to `Chart.__slots__`, default `None`
- [ ] Wire `properties(render_config=...)` to store it; propagate through `_clone()`
- [ ] Verify: `uv run python -c "import ferrum as fm; rc = fm.RenderConfig(); print(rc)"`

### Task 2: _apply_auto_raster method
- [ ] Add `Chart._apply_auto_raster()` — returns `self` or a substituted chart
- [ ] Mark counting: row count of resolved `_data` for per-element marks
- [ ] Eligibility checks: mark type, color encoding absence, x+y quantitative (spec §4 conditions 1–4)
- [ ] When eligible: clone chart, apply `mark_raster(aggregate=..., cmap=...)`, emit `UserWarning` per `raster_behavior`
- [ ] When ineligible but over threshold: emit guidance warning
- [ ] When `raster_behavior="error"`: raise `ValueError`
- [ ] Idempotent: no-op on mark_raster/mark_hex/mark_image charts
- [ ] Verify: `uv run pytest tests/test_scale_rendering.py -m slow -k "auto_raster" -v`

### Task 3: Wire into _render_inputs
- [ ] Call `_apply_auto_raster()` on the resolved chart inside `_render_inputs()`, between `_resolve_pending()` and `to_spec()`
- [ ] Update `show_svg()`, `show()`, `save()` docstrings to document auto-raster and `raster_threshold=None`
- [ ] Verify: `uv run python -c "import ferrum as fm, polars as pl, numpy as np; rng=np.random.default_rng(42); df=pl.DataFrame({'x':rng.normal(0,1,1_000_000).tolist(),'y':rng.normal(0,1,1_000_000).tolist()}); svg=fm.Chart(df).mark_point().encode(x='x:Q',y='y:Q').show_svg(); print(len(svg)//1024,'KB')"`

### Task 4: Rust RenderConfig forward compat
- [ ] Add `raster_threshold: Option<u64>` and `raster_behavior: Option<String>` to `crates/ferrum-core/src/render/config.rs` with defaults
- [ ] Verify: `source ~/.cargo/env && cargo clippy -p ferrum-core -- -D warnings 2>&1 | tail -3`

### Task 5: Tests
- [ ] Add `TestAutoRaster` class to `tests/test_scale_rendering.py` covering all 9 acceptance criteria from spec §9
- [ ] Update `test_scatter_1m_svg_size` assertion: <2MB (was <100MB)
- [ ] Verify: `uv run pytest tests/test_scale_rendering.py -m slow -v`
- [ ] Verify: `uv run pytest -x -q` (all existing tests still pass)

## 6. Acceptance checks

- `uv run pytest tests/test_scale_rendering.py -m slow -v` — all pass including new auto-raster tests
- `uv run pytest -x -q` — all existing tests unaffected
- 1M scatter `show_svg()` produces <2MB SVG and emits `UserWarning`
- 1M scatter with `raster_threshold=None` produces ~57MB SVG, no warning
