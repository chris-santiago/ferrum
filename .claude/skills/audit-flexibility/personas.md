# Flexibility-audit categories

Eight plot categories, each a distinct visualization tradition. One `viz-power-user` agent per category. The agent's system prompt already carries the shared "power-user" context (how ferrum works, render-and-inspect discipline, rules, deliverable format). This file holds the **per-category brief** to paste into each dispatch: the category name, the incumbent libraries to compare against, the scratch slug, and the specific ambitious chart designs to attempt.

Each brief tells the agent to push the **most ambitious ~4–5** designs hard rather than skim all of them. Add or swap designs freely as ferrum's surface grows — the categories are stable, the target lists are living.

Scratch convention: each agent works in `/tmp/ferrum-ux-audit/<slug>/` and writes its report to `/tmp/ferrum-ux-audit/<slug>.md`.

---

## 1. distributions — Statistical / distributional plots
**Compare against:** seaborn, Altair, matplotlib.
**Targets:**
- Raincloud plot (half-violin + boxplot + jittered strip, per category).
- Ridgeline / joyplot (stacked, slightly overlapping KDEs across many categories).
- Violin + inner box + strip overlay, split by a hue.
- Beeswarm / swarm plot: non-overlapping points placed by a force/swarm layout against a categorical axis (not random jitter), plus a sina plot where jitter width tracks local density. Tests whether a layout can move points to avoid collisions, not just encode them.
- Layered ECDF curves comparing groups, with a reference band.
- Joint distribution: scatter + marginal histograms/KDEs (jointplot), plus a bivariate KDE/contour with rug marks.
**Push:** per-group colors, custom bandwidths, category ordering, faceting by a second variable, direct labels on distributions.

## 2. explanatory — Annotation-heavy editorial charts
**Compare against:** matplotlib, Altair, d3 (NYT/Economist/FT/Pew register).
**Targets:**
- "Highlight one series" line chart: many gray background lines, 1–2 colored + directly labeled at endpoints, active sentence-style title + deck, source footnote.
- Annotated time-series with event callouts: arrows + text at specific (x,y) data points, a shaded region band, a horizontal reference line with inline label.
- Slope / dumbbell before→after chart with endpoint value labels and category labels (no y-axis).
- A chart mixing data coordinates and pixel/figure coordinates for annotation placement (legend-as-text, callout boxes).
- Small-multiples panel where each facet has its own highlighted point and annotation.
**Push:** text styling (font/weight/color), label anchoring/collision, layering annotations above/below marks, headline+deck+footnote, color-to-draw-attention.

## 3. timeseries — Time-series & financial charts
**Compare against:** mplfinance / matplotlib, Altair.
**Targets:**
- Candlestick or OHLC chart (up/down colored bodies + wicks); compose by hand if no native mark.
- Price + volume panel: shared x-axis, price on top, volume bars below, aligned.
- Dual-axis chart: two series, very different units, left/right y-axes (`fm.SecondaryY`).
- Time series with rolling-mean overlay (window transform), confidence band (area between columns), region shading, event annotations.
- Horizon chart: layered banded areas where each band recycles one shared color scale at stacked value offsets (cubism.js). The densest small-multiple time-series form; tests whether a fill can reuse a scale across offset bands.
- Streamgraph / ThemeRiver: many series as stacked areas around a centered "wiggle" baseline instead of a zero baseline. Tests whether the stack position-adjustment supports a non-zero, data-driven baseline.
- Calendar heatmap (GitHub-contributions style): derive a week×weekday grid from a date column and color each cell by value. Tests deriving a 2D categorical grid from a datetime and laying out one cell per day.
- Log-scale returns; a broken axis (`fm.BreakAxis`) or inset zoom (`fm.Inset`).
**Push:** temporal axis tick formatting (month/year), multi-panel alignment, secondary axes, log scales, band/ribbon fills, wiggle/centered baselines, layered banded fills, date-derived day/week grids, weekend gaps, real datetime parsing.

## 4. faceting — Small multiples / composed dashboards
**Compare against:** seaborn (FacetGrid/PairGrid), Altair.
**Targets:**
- Trellis: scatter faceted by row×column of two categoricals, per-facet regression line, shared x but **independent** y scales (then try shared — verify the resolve actually changes).
- SPLOM / pairplot: scatter matrix, hist/KDE diagonal, hue by class. Try the figure helper **and** by-hand `Repeat`.
- Scatter with marginal histograms (jointplot), marginals sized/aligned to the joint panel.
- Correlation heatmap with annotated cell values; a dendrogram-clustered version (clustermap).
- Heterogeneous dashboard: vconcat/hconcat of **different** chart types into one figure with a shared super-title.
- **Grand challenge — Minard's Napoleon march:** a flow ribbon whose width encodes army size positioned over geographic x/y, with a temperature line panel below sharing the same x-axis. The hardest cross-coordinate composition test in the suite: flow geometry, spatial positioning, and aligned multi-panel layout all at once. Report exactly which of the three breaks first.
**Push:** independent vs shared scales, per-facet layers, facet headers/spacing/sorting, mixing chart types, cross-coordinate composition, shared-x panel alignment, controlling panel sizes, column wrapping.

## 5. multivariate — Encoding-rich multivariate plots
**Compare against:** Altair, d3.
**Targets:**
- Gapminder-style bubble chart: x, y, size (area scale + size legend), color, maybe shape — 4–5 simultaneous encodings, **a legend for each**.
- Connected scatterplot (path through time over 2D space) with directional ordering and point labels.
- Slope graph and a bump/rank chart over time (many categories, ranked lines).
- Parallel coordinates over 5+ dimensions, colored by class, per-axis scaling. As a stretch, make it interactive: brush a range on one axis and have it filter/highlight polylines across all N axes (N-axis linked brushing — the canonical high-dim linked view).
- Diverging/sequential continuous color encoding with a custom domain, custom palette, and a formatted continuous colorbar.
**Push:** multiple simultaneous legends, continuous color domains/clamping, custom size ranges, shape palettes, legend placement/merging, ordinal vs nominal handling, custom color scales.

## 6. scientific — Scientific / technical figures
**Compare against:** matplotlib (the scientific gold standard), Altair.
**Targets:**
- Filled contour of a 2D function z=f(x,y) with contour lines + colorbar (`contourf` workhorse); also a bivariate KDE contour over scattered data.
- Hexbin density of a large 2D scatter with a count colorbar.
- Error-band figure: line + shaded CI (mean ± CI), plus discrete error bars on points.
- Vector/quiver field or streamplot; if no quiver mark, attempt via `mark_segment`/annotate and report the gap.
- Log-log / semilog with proper log ticks; a polar plot (`CoordPolar`) of a radial function; scientific/SI tick formatting.
- Wind rose / Nightingale coxcomb: stacked bars under a polar coordinate transform (angular axis = direction or category, radius = value). Tests whether bar geometry and stacking survive a polar transform, not just the radial line above.
**Push:** continuous colorbars, contour level control, log scales & tick formatting, error/band marks, polar coords, stacked bars in polar, large-N density.

## 7. categorical — Categorical / part-to-whole / ranking
**Compare against:** Altair, matplotlib (Economist/FT/Datawrapper register).
**Targets:**
- 100% (normalized) stacked bar with segment value labels; grouped (dodged) bars with value labels above bars — verify normalization and dodging.
- Diverging stacked bar (Likert: agree/disagree from a center baseline).
- Marimekko / mosaic plot: stacked bars whose column widths are proportional to each category's total (2D variable-width position adjustment), with a value label per tile. Tests whether bar width can be data-driven, not uniform.
- Sankey / alluvial flow: weighted ribbons between stages, node boxes sized by throughput, ribbon thickness encoding flow volume. Tests whether the engine can express flow geometry at all.
- Sunburst / icicle / treemap: a hierarchical part-to-whole layout where nested categories subdivide a parent's area (rectangles for treemap and icicle, radial wedges for sunburst). Tests whether the engine can express a recursive hierarchical layout at all, not just flat categories.
- Horizontal lollipop or Cleveland dot plot, sorted descending, with value labels.
- Dumbbell / connected-dot chart comparing two time points per category, sorted by gap.
- Pie or donut (`CoordPolar`) and a radial/polar bar — note whether you'd actually recommend them.
**Push:** stack normalization to percent, sorting by value/aggregate, value labels on/above/inside bars, horizontal orientation, diverging baselines, variable-width columns, flow geometry, hierarchical/recursive layout, category ordering, top-k filtering.

## 8. interactive — Interactive & linked views
**Compare against:** Altair/Vega-Lite, d3/Observable, Plotly, Bokeh.
**Targets:**
- Tooltips on a scatter showing multiple fields on hover — verify the tooltip data is actually in the export.
- Interval brush on one panel that filters/highlights a second linked panel (linked brushing / crossfilter): `selection_interval` + `transform_filter`.
- Range selector / overview+detail (focus+context): a small overview panel showing the full series with a draggable, resizable brush window, plus a large detail panel that zooms to the brushed range. Tests whether a selection can drive another panel's *scale domain* (reactive rescaling), not just filter or highlight rows — distinct from the crossfilter target above. Verify the emitted HTML actually wires the brush extent to the detail panel's domain.
- Interactive legend toggling series visibility on click (`selection_point` bound to a legend).
- Pan/zoom on a scatter (`.interactive()`); a conditional encoding that grays out unselected points.
- Dashboard combining 2+ linked interactive charts exported to a single self-contained HTML file; inspect file size and whether data/WASM embed correctly.
**Inspect by reading emitted HTML/JS** (you can't click in a browser): verify selection wiring, tooltip markup, and embedded data/WASM are present and correct. A selection that silently does nothing, a missing tooltip, or an export that errors all count.
**Note the known deferred limits** (repo CLAUDE.md: W4 node-type offsets, W5 JointChart flat layout) — observe them as a user would rather than just repeating them.
