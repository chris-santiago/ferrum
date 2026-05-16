# Changelog

All notable changes to Ferrum are documented here.

## Unreleased

*No unreleased changes.*

## 0.8.2

### Added

- `/test-sweep` skill — iterative combinatorial test-and-fix campaign
- `python-coder` and `rust-coder` coding agents for language-specific dispatch
- Plotly added to scatter benchmark suite
- Social card generation script with iris lmplot

### Fixed

- **Per-instance channel rendering** — `opacity`, `stroke_width`, `fill_opacity`, `size`, and `angle` encoding channels now affect SVG output for all 9 primitive marks (previously only `mark_point` had full wiring)
- **All-null data handling** — charts with entirely null/NaN columns now render an empty chart gracefully instead of crashing with `ValueError`
- **Histogram auto-groupby** — `displot(hue=)` and `jointplot(hue=)` no longer crash; the Bin transform now preserves the color-groupby column
- **RepeatPlaceholder guard** — `pairplot` mark overrides no longer crash when encodings contain unresolved repeat placeholders
- **Chained layer column-overlap** — `mark_rule` layering with overlapping column names resolved
- **Scale constructors** — `range` parameter now optional on all scale constructors
- **License packaging** — NOTICE file declared in `license-files` for sdist inclusion

### Changed

- Benchmark tables updated with Plotly results (4-library comparison)
- Documentation expanded: 25 new-user gaps addressed, API reflinks, stale gallery PNGs regenerated

## 0.8.1

### Fixed

- **Scale `range` now optional** — `LinearScale`, `LogScale`, `SymlogScale`, `TimeScale`, and `OrdinalScale` no longer require `range=` in their constructors. The renderer auto-fills from the plot-area dimensions, so users can write `fm.LogScale(domain=[100, 100000], base=10)` without pixel math.
- **`mark_rule` layering with overlapping columns** — layering a rule on a scatter (`scatter + hline`) no longer draws a line per scatter row when both DataFrames share a column name (e.g. `"y"`).
- **Chained `+` layering** — `scatter + hline + vline + label` now renders correctly. Added `Identity` transform and `inherit_non_positional` to prevent chart-level positional encoding from polluting routed layers.
- **Docs code examples** — fixed `CoordFlip` encoding order, `stroke_dash` string→list, `Stack` Int64→Float64, `to_theme_inputs_dict()`→`to_spec_dict()`, `chart.render_config()`→`chart.properties(render_config=)`, `feature_importances_chart`→`importance_chart`, `ferrum[...]`→`ferrum-viz[...]`.

### Added

- **25 new documentation sections** — type suffix explanation, position adjustments (Dodge/Stack/Jitter), axis customization (log scale, limits, reversed), smooth method table, legend control, palette cycling, shared scales, regplot, multi-model comparison, precomputed scores, output methods summary, DPI note, PDF note, interactive compatibility, CoordFlip recipe, annotation recipe, chart sizing recipe, category sorting recipe, time-series recipe.
- **13 guide PNGs** — visuals for every new section (position adjustments, axis customization, legend suppression, shared scales, regplot, multi-model ROC, CoordFlip, annotations, sizing, category order, time-series).
- **12-theme visual grid** — side-by-side comparison of all built-in themes on the same chart.
- **Complete 54-mark table** — all mark methods listed with reflinks in marks-encodings guide.
- **Consistent API reflinks** — every public symbol mentioned in prose now links to its API reference across all doc pages.
- **`Identity` transform** — Rust pass-through transform for named-output routing in layered compositions.
- **11 regression tests** — covering scale range, rule layering, and chained layer fixes.
- **PyPI metadata** — keywords, classifiers, and project URLs added to `pyproject.toml`.

### Changed

- **README** — updated install section with `ferrum-viz[all]`, added docs link, interactive rendering feature, corrected dev commands.

## 0.8.0

Grammar-of-graphics core with Rust rendering engine.

### Added

- **12 built-in themes** — Paper Ink (default), Slate Citrus, Arctic Signal, Observable, Minimal, Dark, Publication, Economist, FiveThirtyEight, Solarized Light, Solarized Dark, plus explicit `paper_ink` for derivation.
- **3 original categorical palettes** — `paper_ink`, `slate_citrus`, `arctic_signal` (8 colors each).
- **9 continuous color schemes** — `cool_blue`, `warm_ochre`, `night_blue`, `electric_lime`, `signal_blue`, `ember_orange`, `blue_to_red`, `cyan_to_amber`, `blue_to_violet`.
- **[`fm.hconcat()`][ferrum.hconcat] / [`fm.vconcat()`][ferrum.vconcat]** — top-level convenience functions for multi-chart concatenation.
- **[`Chart.axis()`][ferrum.Chart.axis]** — method for suppressing or restoring axis decorations.
- **Facet-before-transform** — faceting now partitions data before statistical transforms run, so each panel gets its own transform subset.
- **Grouped smooth** — `mark_smooth` supports group-by; no explicit per-group layering needed.
- **Continuous colorbar** — rendered alongside heatmaps and continuous-color charts.
- **Full Theme key wiring** — all `ferrum-spec.md` §3.13 keys plumbed end-to-end from Python through Rust renderer. Unknown keys raise `ValueError` at construction.
- **Model diagnostics (Phase 10)** — [`ModelSource`][ferrum.ModelSource], [`ComparedModelSource`][ferrum.ComparedModelSource], 32 visualizer classes, 31 figure-level helpers covering classification, regression, feature explanation, model selection, and clustering.
- **9 figure-level helpers** — [`displot`][ferrum.displot], [`catplot`][ferrum.catplot], [`relplot`][ferrum.relplot], [`lmplot`][ferrum.lmplot], [`residplot`][ferrum.residplot], [`pairplot`][ferrum.pairplot], [`heatmap`][ferrum.heatmap], [`clustermap`][ferrum.clustermap], [`jointplot`][ferrum.jointplot].
- **54 mark methods** — primitives, statistical, distribution, uncertainty, scale-aware, and diagnostic marks.
- **DataFrame pluralism** — polars, pandas, modin, cuDF, dask, ibis, and pyarrow all accepted via `Chart(data)`.
- **Title rendering** — `Chart.properties(title="...")` rendered in the SVG with theme-controlled typography.
- **Grid lines** — theme-controlled grid rendering with configurable color, width, dash, and opacity.
- **Legend** — categorical and continuous legends with theme-controlled positioning and styling.
- **Interactive rendering (Phase 11)** — [`Chart.interactive()`][ferrum.Chart.interactive] switches to a GPU-backed WASM renderer with selections, zoom/pan, linked views, and tooltips. Backed by `anywidget` for Jupyter integration.
- **Selection API** — [`selection_point`][ferrum.selection_point], [`selection_interval`][ferrum.selection_interval], [`selection_single`][ferrum.selection_single], [`selection_multi`][ferrum.selection_multi] for declaring interactive state. Conditional encodings via `sel.when(...).otherwise(...)`.
- **[`SelectionMark`][ferrum.SelectionMark]** — configurable brush styling for interval selections.
- **[`InteractiveChart`][ferrum.InteractiveChart]** — anywidget-based Jupyter widget with `on_selection_change` callback and self-contained HTML export via `.save()`.
- **Scene graph renderer** — Rust-side `render_interactive` produces a SceneGraph JSON consumed by the WASM GPU renderer.
- **`compose_svg_horizontal` / `compose_svg_vertical` / `compose_svg_grid`** — low-level Rust SVG composition helpers.
- **t-SNE and UMAP in pure Rust** — [`ManifoldVisualizer`][ferrum.ManifoldVisualizer] runs both via `manifolds-rs`, no Python `umap-learn` dependency.
- **7 new diagnostic helpers** — [`classification_report_chart`][ferrum.classification_report_chart], [`class_balance_chart`][ferrum.class_balance_chart], [`cooks_distance_chart`][ferrum.cooks_distance_chart], [`prediction_error_chart`][ferrum.prediction_error_chart], [`silhouette_chart`][ferrum.silhouette_chart], [`elbow_chart`][ferrum.elbow_chart], [`manifold_chart`][ferrum.manifold_chart].
- **[`mark_label`][ferrum.Chart.mark_label]** — text labels with collision avoidance (`avoid_overlap=True`).
- **[`mark_image`][ferrum.Chart.mark_image]** — image tiles from URL fields.
- **[`RenderConfig`][ferrum.RenderConfig]** — per-chart auto-raster policy configuration (threshold, behavior, aggregate, colormap).
- **`raster=` keyword** — one-off auto-raster override on `.show()`, `.save()`, `.show_svg()`, `.show_png()`.
- **`score()` on visualizers** — all Group A visualizers implement `.score()` for sklearn-protocol compatibility.
- **`width=` on [`mark_boxplot`][ferrum.Chart.mark_boxplot]** — API symmetry with other marks.
- **`stroke`, `angle`, `fill_opacity` channels** — wired end-to-end to SVG attribute emission and WASM renderer.
- **Legend `format=` and `columns=`** — kwargs now wired through to legend rendering.
- **Packed tooltips** — field-level tooltip content sent via binary buffer for interactive performance.
- **Binary instance bridge** — GPU data bypasses JSON serialization in interactive rendering.

### Fixed

- **Facet encoding** — `facet=`, `facet_col=`, `facet_row=` now correctly partition data into panels.
- **`mark_tick` y-rug** — single-axis tick marks render correctly in both x and y orientations.
- **CoordFlip rendering** — no longer drops violin paths or boxplot rects under coordinate flip.
- **Histogram `multiple='stack'`/`'fill'`** — bin edges now align across groups.
- **[`calibration_chart`][ferrum.calibration_chart] rendering** — layering wiring gap resolved; renders correctly via `.show_svg()`.
- **Chart decomposition** — `chart.py` split into rendering, encoding helpers, and composition helpers for maintainability.

### Changed

- **`+` operator always layers** — no longer ambiguous between layering and concatenation. Use `|`, `&`, [`fm.hconcat()`][ferrum.hconcat], or [`fm.vconcat()`][ferrum.vconcat] for concatenation.
- **Default theme identity is Paper Ink** — warm cream background, blue marks, warm grid. Previously was Observable-style white background.
- **Default categorical palette is `paper_ink`** — previously `okabe_ito`.
