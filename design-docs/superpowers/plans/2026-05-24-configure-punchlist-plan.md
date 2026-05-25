# Configure + Bug Hunt — Unified Fix Plan

> Fix every silently-dropped config field, every broken recipe PNG, and all 31 bugs from the 2026-05-24 bug hunt.

## Inventory: 43 items

### A. Configure punchlist (11 items) — tests in `test_configure_integration.py`

| # | Field | Test assertion |
|---|---|---|
| A1 | `configure_axis(label_format="currency")` | `$` in tick labels |
| A2 | `configure_axis(label_format_raw=".0%")` | `%` in tick labels |
| A3 | `configure_axis(tick_values=[10,30,50])` | exact tick positions |
| A4 | `configure_axis(title_font_size=20)` | SVG changes |
| A5 | `configure_axis(title_color="#ff0000")` | hex in SVG |
| A6 | `configure_axis(title_padding=30)` | SVG changes |
| A7 | `configure_color(range=[...])` | custom hex in SVG |
| A8 | `configure_color(domain=[0,100])` | SVG changes |
| A9 | `configure_legend(gradient_length=200)` | SVG changes |
| A10 | `configure_grid(band_colors=[...])` | band hex in SVG |
| A11 | Recipe PNGs render correctly | visual verification |
| A12 | `ColorScale::Categorical.palette` accepts user colors | Change `&'static [Color]` → `Cow<'static, [Color]>` |

### B. Bug hunt — confirmed (21 items) — tests already exist, DO NOT modify

| # | Subsystem | Bug | Test file | Fix file |
|---|---|---|---|---|
| B1 | coerce-transport | LazyFrame bypasses Date cast | `test_bug_hunt_coerce_transport.py` | `src/ferrum/_coerce.py` |
| B2 | coerce-transport | LazyFrame bypasses Categorical cast | same | same |
| B3 | coerce-transport | LazyFrame bypasses Duration cast | same | same |
| B4 | figure-api | catplot/box rejects integer columns | `test_bug_hunt_figure_api.py` | `src/ferrum/marks/heavy_stat.py` or `_resolve_pending` |
| B5 | figure-api | catplot/box integer (horizontal) | same | same |
| B6 | figure-api | catplot/box integer (with configure) | same | same |
| B7 | figure-api | Rust color parser rejects `#fff` | same | `render/color/categorical.rs` |
| B8 | interactive | annotate_hline dropped in interactive | `test_bug_hunt_phase_11_interactive.py` | `src/ferrum/_interactive.py` or scene pipeline |
| B9 | interactive | annotate_vline dropped | same | same |
| B10 | interactive | annotate_text dropped | same | same |
| B11 | interactive | annotate_rect dropped | same | same |
| B12 | interactive | multiple annotations dropped | same | same |
| B13 | interactive | annotate_abline dropped | same | same |
| B14 | projection | Natural Earth poly drops 5th coeff | `bug_hunt_projection.rs` | `projection.rs` |
| B15 | projection | Natural Earth y at pole wrong | same | same |
| B16 | projection | Natural Earth deriv drops 5th term | same | same |
| B17 | projection | Orthographic inverse unstable near poles | same | `projection.rs` |
| B18 | stats | Cook's distance negative leverage | `bug_hunt_stats_transforms.rs` | `stats.rs` |
| B19 | stats | Cook's distance p_eff=0 div by zero | same | same |
| B20 | stats | Shapiro-Wilk W exceeds 1.0 | same | same |
| B21 | stats | variance_rank_vec NaN at n=0 | same | same |

### C. Bug hunt — latent (10 items) — tests already exist, DO NOT modify

| # | Subsystem | Bug | Fix file |
|---|---|---|---|
| C1 | model-diag | Binary ROC AUC crash on single-class | `src/ferrum/_diagnostics/sources/_classification.py` |
| C2 | model-diag | Dead validation in ModelSource | same |
| C3 | model-diag | Dead validation in PrecomputedSource | `src/ferrum/_diagnostics/precomputed.py` |
| C4 | draw | `format_numeric(NaN)` → "NaN" string | `render/format.rs` |
| C5 | draw | NaN opacity → silent transparency | `render/color/categorical.rs` |
| C6 | draw | `format_ordinal_number(i64::MIN)` panics | `render/format.rs` |
| C7 | draw | Annotation image src not XML-escaped | `render/annotation.rs` |
| C8 | draw | Callout text width uses byte count | `render/annotation.rs` |
| C9 | draw | `to_scene_stroke` diverges from `parse_stroke_cap` | `render/draw.rs` |
| C10 | draw | Null bytes pass through XML escaping | `render/svg.rs` |

## Execution stages

### Stage 1: Python fixes (B1–B6, C1–C3)

No Rust rebuild needed. 3 parallel agents by file footprint:

| Agent | Items | Files |
|---|---|---|
| python-coder #1 | B1–B3 | `src/ferrum/_coerce.py` |
| python-coder #2 | B4–B6 | `src/ferrum/marks/heavy_stat.py` (or boxplot desugar) |
| python-coder #3 | C1–C3 | `src/ferrum/_diagnostics/sources/_classification.py`, `precomputed.py` |

Verify: `uv run pytest tests/test_bug_hunt_coerce_transport.py tests/test_bug_hunt_figure_api.py tests/test_bug_hunt_model_diagnostics.py -v`

### Stage 2: Rust fixes — rendering + draw (A1–A10, B7, C4–C10)

Depends on Stage 1 agent for config wiring (already running). Additional Rust agent for:

| Agent | Items | Files |
|---|---|---|
| rust-coder #1 (running) | A1–A10 | `prepare.rs`, `scale_resolve.rs`, `marks/axis.rs`, `mod.rs` |
| rust-coder #2 | B7, C4–C10 | `color/categorical.rs`, `format.rs`, `annotation.rs`, `draw.rs`, `svg.rs` |

Verify: `cargo test -p ferrum-core --lib`

### Stage 3: Rust fixes — projection + stats (B14–B21)

Independent from Stage 2. Parallel agent:

| Agent | Items | Files |
|---|---|---|
| rust-coder #3 | B14–B17 | `projection.rs` |
| rust-coder #4 | B18–B21 | `stats.rs` |

Verify: `cargo test -p ferrum-core`

### Stage 4: Interactive annotations (B8–B13)

Depends on understanding how `annotate_hline` etc. flow through the interactive path. Likely requires wiring the existing annotation system (`Chart._annotation_lines` etc.) into `_render_scene` → `chart_config.annotations`.

| Agent | Items | Files |
|---|---|---|
| python-coder #4 | B8–B13 | `src/ferrum/_interactive.py`, `src/ferrum/chart.py` |

Verify: `uv run pytest tests/test_bug_hunt_phase_11_interactive.py -v`

### Stage 5: Rebuild + PNGs + visual verification (A11)

1. `maturin develop`
2. `uv run pytest tests/test_configure_integration.py -v` — all 12 pass
3. `uv run python scripts/render-recipe-pngs.py` — all 12 render
4. Visually verify each PNG
5. Update goldens if needed

### Stage 6: Gate

1. `uv run pytest -n auto` — ALL pass (including bug hunt tests)
2. `cargo test -p ferrum-core` — ALL pass (including bug hunt tests)
3. Review-lite on all changed `.py` and `.rs`
4. Single commit
