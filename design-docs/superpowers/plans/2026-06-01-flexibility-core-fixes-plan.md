# Flexibility Core Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task. Per project CLAUDE.md, dispatch Rust edits to `rust-coder` and Python edits to `python-coder`; run `/regression-test` before declaring any task done.

## 1. Objective

Fix the five renderer/coercion-side defects (D1–D5) that silently break charts across the grammar, per the design spec.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-01-flexibility-core-fixes-design.md` — full spec
- §4 System behavior; §6 Canonical interfaces; §8 Key decisions; §9 Acceptance criteria

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/scale_resolve/color.rs` | resolve `Sequential`/`Diverging` scale specs (D1) |
| Modify | `crates/ferrum-core/src/render/color/continuous.rs` | named-scheme ramp + domain/clamp (D1) |
| Modify | `crates/ferrum-core/src/render/palette.rs` | shared named-palette table for resolver + `cmap` (D1) |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` | honor `y2`/`x2` extent (D3) |
| Modify | `crates/ferrum-core/src/render/marks/area.rs` | honor `y2` extent (D3) |
| Modify | `src/ferrum/chart.py` | `mark_bar(zero=)` escape + conditional zero-anchor (D3); facet cardinality inference, per-partition transform routing, layer preservation, facet scale-resolve (D2); warn on dropped kwargs (D5) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | partition on both row+col fields; re-run transform per partition (D2) |
| Modify | `src/ferrum/_coerce.py` | normalize integer/nominal columns once at the transport boundary (D4) |
| Modify | `crates/ferrum-core/src/transform/data_window.rs` | correct `frame` sign to documented convention (D5) |
| Modify | `crates/ferrum-core/src/render/annotation.rs` | stop dropping `label_position`; warn if unhonored (D5) |
| Test | `tests/test_flexibility_core/` | per-defect regression tests (D1–D5) |
| Modify | `tests/goldens/**`, `tests/test_phase_9_e2e/goldens/**` | regen affected goldens (color, bar/area, facet) |

## 4. Constraints

- **Goldens are not blessed until visually inspected** — rasterize every regenerated SVG via `scripts/snapshot-goldens.py`, Read each PNG, confirm correct render before committing. Color and bar/area changes move many goldens.
- `cargo test` must pass before the phase is marked done.
- No matplotlib; no global mutable state.
- `mark_bar` `zero=` default stays `True`; charts that render correctly today stay byte-stable except where they were silently wrong.
- Diagnostics are warnings, not errors — except the window-frame sign, which is corrected outright (changelog note required; pipelines using the inverted sign will shift).
- D1 must not regress the categorical color cycle or the working `mark_raster`/`mark_hex` `cmap` path; D7-era polar work is out of scope.
- Integer → quantitative by default; explicit `:O`/`:N` opts into ordinal/nominal.

## 5. Tasks

### Task 1: D1 — continuous-color scale resolver
- [ ] Write failing tests: viridis/magma/reds/greens/oranges/purples render their palette; `DivergingScale(domain=[-1,0,1], clamp=True)` centers + clamps (spec §9)
- [ ] Resolve `Sequential`/`Diverging` in `color.rs` from the shared palette table; wire `continuous.rs` ramp + domain/clamp
- [ ] Regenerate color goldens; rasterize + visually inspect each PNG
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_core/test_d1_color.py`

### Task 2: D3 — bar/area second extent + zero escape
- [ ] Failing tests: candlestick bodies span open→close; diverging Likert keeps negative segments; floating gap bar off baseline (spec §9)
- [ ] `bar.rs`/`area.rs` honor `y2`/`x2`; `chart.py` adds `zero=` and makes zero-anchor conditional on `zero`/`y2`-bound
- [ ] Regenerate bar/area goldens; inspect PNGs
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_core/test_d3_bar_extent.py`

### Task 3: D2 — faceting grammar
- [ ] Failing tests: 3×3 `facet(row=,col=)` → 9 populated panels with headers; `mark_density().facet(col=)` populated; `share_scale(y="independent")` vs `"shared"` change domains; faceted multi-DataFrame layer preserved (spec §4, §9)
- [ ] `chart.py`: infer `nrows/ncols`, route per-partition transform, preserve layers, add facet scale-resolve
- [ ] `prepare.rs`: include row+col in partition keys; re-run transform per partition
- [ ] Regenerate facet goldens; inspect PNGs
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_core/test_d2_facet.py`

### Task 4: D4 — dtype coercion at boundary
- [ ] Failing tests: integer-keyed and string-keyed heatmaps render all cells; `mark_bar(y="cat:N")` renders; `Stack` accepts integer columns (spec §4, §9)
- [ ] `_coerce.py`: normalize integer/nominal once; integer→quantitative unless `:O`/`:N`
- [ ] Verify: `uv run pytest tests/test_flexibility_core/test_d4_dtype.py`

### Task 5: D5 — surface silent failures + frame sign
- [ ] Failing tests: `frame=(-13,0)` → non-null rolling mean; unsupported kwarg warns; empty partition warns; `label_position` honored or warns (spec §4, §8)
- [ ] `data_window.rs`: fix frame sign; `annotation.rs`: honor/warn `label_position`; `chart.py`: warn on dropped `Y(stack=)`/kwargs
- [ ] Changelog note for the frame-sign break
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_core/test_d5_diagnostics.py`

### Task 6: Cross-cutting verification
- [ ] Re-run flexibility-audit categories tied to D1–D5; confirm flagged designs now render
- [ ] Golden inspection sweep across all regenerated goldens

## 6. Acceptance checks

- `uv run pytest tests/test_flexibility_core/ -v` — all pass
- `uv run pytest -n auto` — full suite green
- `cargo test` — all pass
- All regenerated goldens rasterized and visually confirmed correct
- Spec §9 designs render on inspection; param-free / unaffected goldens byte-stable

## 7. Open questions

None — resolved in spec §11.
