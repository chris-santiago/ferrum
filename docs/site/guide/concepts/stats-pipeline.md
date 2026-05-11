# Stats in the rendering pipeline

In most Python visualization workflows, statistical computation lives outside the plotting library. You call SciPy to estimate a kernel density. You write a loop to bootstrap a confidence interval. You bin a column with NumPy or Polars. You fit a LOESS curve with statsmodels. Then you hand the result to a plotting function and ask it to draw what you computed.

The plot is the easy step. The pre-processing dominates the code.

Ferrum reverses that arrangement. Statistical transforms are first-class operations inside the chart specification, executed by the engine before rendering. You declare intent — *bin this field*, *smooth this series*, *show a 95% confidence band* — and the engine computes the result in Rust as part of the build.

## Why this is a structural choice

The placement of statistics is not a convenience decision. It changes what the chart specification can honestly describe.

When a confidence band is computed in user code, the chart spec sees only the endpoints of that band. The spec knows it is drawing two lines, not what those lines represent. The statistical assumption — the bootstrap procedure, the resampling count, the confidence level — lives in a script somewhere, separate from the artifact it produced. Re-running the plot on new data means re-running the script first.

When the same band is declared in the chart spec, the statistical intent is part of the chart's identity. The chart says "draw a 95% bootstrap CI on this field," and the engine computes it. The intent is reviewable, themeable, and reproducible together with the visual it produces. Re-running on new data is one operation, not two.

This is what "statistically literate" means in Ferrum's design vocabulary. The library is not just a renderer that happens to ship with some statistical helpers. It is a system in which statistical transforms are declarative chart operations on equal footing with marks and encodings.

## What ferrum computes for you

The Ferrum engine includes statistical transforms covering the common needs of exploratory and explanatory plotting:

- **Aggregation** — sum, mean, median, count, min, max, quantile, and related summaries computed per group.
- **Binning** — both 1-D histogram bins and 2-D bivariate bins, with automatic bin-width policies for both numeric and time-valued fields.
- **Kernel density** — 1-D KDE and 2-D KDE with bandwidth selection.
- **Smoothing and regression** — LOESS, GLM, logistic regression, and other curve-fitting routines used by `Smooth`-style marks.
- **Boxplot statistics** — quartiles, whiskers, outliers, and letter-value boxplot variants.
- **Confidence intervals and error extents** — bootstrap CIs, standard error, and configurable error-bar geometry.
- **Quantile transforms** — quantile-quantile plots and quantile scales.
- **Contours and rasters** — for high-cardinality 2-D data where individual marks would obscure structure.
- **Reordering and aggregation utilities** — for ordinal scales whose order is derived from data rather than declared.

Each of these is named, declarable, and computed once per chart build. The output of a transform is data that the rendering pipeline consumes, not a side effect that lives in your script.

## What this lets you stop doing

The most visible practical effect is that the size of a Ferrum plotting script shrinks dramatically compared to a `precompute + plot` workflow. The steps that used to be data preparation become declarations on the chart.

You stop:

- Maintaining a separate stats utility module alongside your plotting code.
- Re-running bootstrap loops when you tweak a chart parameter.
- Writing glue code to keep computed CIs in sync with the line they belong to.
- Translating between dataframe schemas for the stat step and the plot step.
- Losing the statistical context when you share or revisit the chart later.

You start treating statistical assumptions as visible, declared properties of the chart artifact.

## Why "in the rendering pipeline" specifically

Ferrum's engine runs transforms in Rust, before the rendering backend draws anything. This placement has a few consequences worth being explicit about.

**Statistics compose with the rest of the spec.** Because transforms run before rendering, a binned histogram can be faceted, themed, layered with a smoothed density, or composed with a marginal plot — exactly the same way as any other chart. The transform does not stand apart from the rest of the grammar.

**Computation is single-pass.** The engine sees the full spec before it computes, so it can plan a single execution rather than re-running passes for each layer. This is a piece of why the same chart spec can scale up to large datasets without changing shape.

**Transforms run on real data structures.** The engine consumes Arrow columns directly. When the input is Polars, the data crosses the Python-Rust boundary through Arrow CDI with no row-level copying. The statistical work happens on the same data layout the rendering layer uses.

**Statistical defaults can be honest.** Because transforms are part of the chart, the library can apply sensible defaults — perceptually-uniform palettes, accessible binning floors, descriptive errors when inference is ambiguous — without forcing the user to undo defaults from a separate stat package.

## What this doesn't mean

Ferrum is not trying to replace SciPy, statsmodels, scikit-learn, or any other statistical computing library. The transforms in the rendering pipeline are the ones a plot needs in order to be a plot — the operations that turn data into the geometry the chart wants to show.

If you need a fitted regression model as an object — for prediction, interpretation, or downstream pipelines — you reach for scikit-learn or statsmodels. If you need a hypothesis test, you reach for the appropriate library. Ferrum's job is to compute the smoothing curve the chart is drawing, not to substitute for the statistical workflow that surrounds the chart.

The boundary is "what does the chart need to show?" — and the answer to that question lives in the chart spec.

## Where to go next

- [One chart model](one-chart-model.md) for the broader context that makes pipeline-statistics possible.
- [Performance & scale](performance-scale.md) for how the Rust-backed pipeline keeps the same chart spec usable at very different data sizes.
- [Model outputs are data](model-outputs-as-data.md) for the related principle that ROC curves, calibration plots, and SHAP summaries are also data — and therefore also chart transforms.
