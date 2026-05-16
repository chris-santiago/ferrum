# One chart model

Ferrum's first design decision is that there is only one kind of chart.

A scatter plot is a chart. A faceted histogram is a chart. A ROC curve is a chart. A SHAP beeswarm is a chart. All of them are built from the same conceptual primitives, accept the same operations, and live inside the same composition system. If you understand how to assemble one Ferrum chart, you can read and write every other Ferrum chart.

This is the foundation that every other choice in the library rests on.

## The primitives

Every visualization in Ferrum is a composition of the same building blocks:

- **Data** — a tabular source. Pandas, Polars, Arrow tables, NumPy arrays, or any dataframe Narwhals can interpret. Data is always columnar and always referenced by field name from the rest of the spec.
- **Encodings** — typed mappings from data fields to visual variables. [`X`][ferrum.X] and [`Y`][ferrum.Y] are encodings; so are [`Color`][ferrum.Color], [`Size`][ferrum.Size], [`Shape`][ferrum.Shape], [`Facet`][ferrum.Facet], and the error and angle channels. Encodings declare *what* a field means in the chart, not *how* it should look.
- **Marks** — the geometric primitives that visualize encoded data: points, lines, bars, areas, rectangles, ticks, text, ribbons, contours, swarms, density curves, and more. A mark is the smallest visible unit of a chart.
- **Scales** — the functions that turn data domains into visual ranges. Linear, log, symlog, time, quantile, threshold, ordinal. Scales are explicit; Ferrum does not silently choose a scale on your behalf when the choice would be ambiguous.
- **Coordinate systems** — Cartesian by default, with explicit `CoordFlip`, `CoordPolar`, `CoordGeo`, and `CoordFixed` for the cases where the geometry needs to change rather than the data.
- **Transforms** — statistical operations applied inside the chart specification, before rendering. Aggregation, binning, smoothing, kernel density estimation, quantile computation, and similar transforms are part of the chart, not preprocessing that happens in user code.
- **Composition** — the operators that combine charts: layering one mark on another, concatenating side-by-side or stacked, faceting across small multiples, joint plots with marginals, repeat grids, and clustered heatmaps. Composition is structural, not visual: the operators describe how charts relate, not how they are styled.
- **Themes** — value objects (not mutable global state) that carry style decisions. A theme can be passed to one chart, set as a process-wide default, or scoped to a `with` block.

That list is the full conceptual surface. There is no second universe of "ML diagnostic objects," no separate "interactive chart" type, no "convenience plot" parallel API. Everything in Ferrum builds from these primitives, and every figure-level helper compiles back to them.

## Grammar first, convenience second

Ferrum is grammar-first, but not grammar-only.

Figure-level helpers such as [`displot`][ferrum.displot], [`lmplot`][ferrum.lmplot], [`catplot`][ferrum.catplot], and the model-diagnostic chart helpers exist because speed matters and because seaborn-style ergonomics are genuinely useful. But they are sugar over the chart system rather than a parallel API with different rules.

That distinction is structural, not stylistic. When you call `displot`, the helper builds a [`Chart`][ferrum.Chart] and returns it. The result is themeable, composable, savable, and faceted in exactly the same way as a chart you wrote from primitives. There is no helper-object trap and no point at which switching from the helper to the grammar requires rewriting your plot.

This principle prevents feature silos. A convenience plot should not lock you into an object that behaves differently from the rest of the library, and a chart created through a helper should remain themeable, composable, and extensible in the same way as one written from first principles.

## Why one chart model

The motivation is the cost of fragmentation. Most Python visualization workflows still split one practice across multiple mental models — a layered statistical library for exploration, a convenience library for distributions, an interactive library for linked views, and a diagnostics library for ROC curves and SHAP summaries. Every transition introduces a new object type, a new set of defaults, and a new set of limitations.

Ferrum's bet is that one large coherent system is better than several smaller incompatible ones stitched together at the application layer. The cost of a broad API is paid once, in library size and learning surface. The cost of API fragmentation is paid every time the task changes.

The consequences of this bet show up everywhere. Convenience helpers cannot become separate worlds. Interactivity cannot require a different chart object. Model diagnostics cannot be treated as bespoke artifacts that stop being composable the moment they appear. Each of these is a separate page in this Concepts section.

## What "one model" lets you do

The practical consequence of a single chart model is that operations compose freely:

- A figure-level `displot` can be faceted, themed, and concatenated next to a regression-line `lmplot` because both return charts.
- A ROC curve from the model-diagnostic helpers can be horizontally concatenated with a confusion matrix and a calibration plot in the same compound view, with a single shared theme.
- A static chart can be switched to interactive output by changing the renderer, not by rewriting the chart spec.
- A theme written for an ordinary scatter plot applies cleanly to a SHAP beeswarm because the SHAP plot is a chart of the same kind.

These compositions are not special-cased. They fall out of the model because there is nothing about a "diagnostic chart" or a "convenience chart" or an "interactive chart" that puts it in a different object hierarchy.

## Where to go next

- [Stats in the rendering pipeline](stats-pipeline.md) explains why transforms like KDE, binning, and bootstrap CIs are part of the chart spec rather than user-side preprocessing.
- [Dataframe pluralism](dataframe-pluralism.md) covers how the same chart sits on top of pandas, Polars, modin, cuDF, dask, ibis, and Arrow inputs.
- [Interactivity is a renderer, not a rewrite](interactivity.md) covers how static and interactive output share one chart object.
- [Model outputs are data](model-outputs-as-data.md) explains why ROC, calibration, confusion, and SHAP plots are charts in the same sense as scatter plots.
- [Performance & scale](performance-scale.md) covers the Python/Rust/Arrow architecture that lets the same chart spec work at very different data sizes.
