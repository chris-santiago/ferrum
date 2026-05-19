# Why Ferrum

Most Python visualization workflows still fracture across multiple mental models. One library handles layered statistical graphics, another handles interactivity, another handles convenience plots, and another handles machine-learning diagnostics. Users keep switching abstractions as soon as the task changes.

Ferrum is designed to remove that boundary. The same chart system supports a simple scatter plot, a faceted distribution analysis, a linked interactive view, and a model-diagnostic suite without forcing users into separate object types or unrelated APIs.

## What makes Ferrum different

There are excellent Python plotting libraries today. Each is best at one thing. Ferrum is built around three properties that no single existing library delivers together:

### 1. One grammar that scales to production data size

Every existing library breaks at some point — plotnine and seaborn (both matplotlib-bound) around 100,000 marks, Altair around 5,000 rows, plotly around 500,000. The library closest to Ferrum's grammar philosophy — plotnine, a faithful ggplot2 port — inherits matplotlib's rendering ceiling, so the grammar stops scaling before the data does. (See [benchmarks](../guide/concepts/performance-scale.md) for methodology and numbers.)

Ferrum keeps the same chart spec working at 100 rows and at 10,000,000 rows. Auto-raster and GPU rendering happen transparently behind the same spec — you don't author for one scale and rewrite for another.

### 2. Model diagnostics as first-class grammar objects, not a parallel API

Yellowbrick and scikit-plot are separate universes from your charting library. They use different object models, different styling, and don't compose with the rest of your plots.

In Ferrum a confusion matrix and a ROC curve can be [`hconcat`][ferrum.hconcat]'d, themed, and saved with the same code as any other chart. They are charts, not special objects. A diagnostic plot is the same kind of artifact as a scatter plot — it accepts the same theme, lives in the same composition operators, and exports through the same renderer.

### 3. Statistical computation in the render pipeline, not in userspace

Every existing library makes you precompute: call SciPy, build the KDE yourself, bootstrap your confidence intervals manually, then hand the result to the plotter. Your plotting code spends most of its lines on data preparation that has nothing to do with the visual you want.

Ferrum declares intent and computes in Rust before rendering. KDE, LOESS, bootstrap CIs, binning, calibration curves, smoothing, and similar transforms are declarative chart operations. The library is statistically literate, not just a renderer.

## At a glance

**Grammar of Graphics, without the ceiling.** Declarative, composable, layered — like Altair or plotnine — but no row limits and no API switch when data grows.

**Stat transforms in the pipeline.** Declared in the chart spec, computed in Rust before rendering. You stop preprocessing data before plotting.

**Model diagnostics that compose.** ROC curves, SHAP beeswarm, residuals, calibration — same grammar, same theme, same `.save()`. [`fm.hconcat()`][ferrum.hconcat] just works.

**Handles every dataframe API.** Polars, pandas, modin, cuDF, dask, and ibis all flow through the same [`Chart`][ferrum.Chart] constructor — internally normalized to Arrow once, then routed through the Rust core unchanged. No per-framework adapters in user code; no special-case ingestion paths in ferrum.

**Zero system dependencies.** Ships in a wheel. No Cairo, no X11, no display server. Renders in Kubernetes, CI, SSH sessions, containers.

**SHAP and ICE at full sample size.** The plots that matter most for understanding models at scale — the ones existing tools sample or crash on — render in full because the rasterization is in Rust and the interactivity is GPU-backed.

## What Ferrum takes from prior art

Ferrum is not built in opposition to prior libraries so much as in response to the seams between them. Its closest ancestor is plotnine — the grammar-of-graphics layering, explicit scales, and faceting philosophy come directly from plotnine and ggplot2. But plotnine is bound to matplotlib for rendering and pandas for data, which limits its scale ceiling, its interactivity story, and its composition model (layers and facets, but no arbitrary chart concatenation). Ferrum takes that grammar foundation and rebuilds it on a Rust engine with Arrow-based data transport, then extends it with typed encodings and selection ideas from Altair, statistical vocabulary and figure-level helpers from Seaborn, interactive output ideas from Plotly, and diagnostic vocabulary from Yellowbrick and scikit-plot.

What Ferrum rejects is not the value of those libraries, but the fractures between their strengths. You should not have to choose between a grammar library, an interactive library, and a diagnostics library depending on the day's task.

## What success looks like

Ferrum succeeds if you can stay inside one mental model from first exploration to final diagnostic review. A simple distribution plot, a polished publication graphic, a linked interactive view, and a threshold-tuning chart for a classifier should feel like variations of the same language rather than migrations across tools.

The goal is not maximal novelty, but a plotting system that is broad, fast, statistically honest, operationally simple, and conceptually unified enough that you stop paying the tax of switching abstractions every time your question changes.

## Where to go next

- Read [Install](install.md) and [your first plot](first-plot.md) if you want to start with code.
- Read the [Concepts](../guide/concepts/one-chart-model.md) pages if you want the design rationale behind the choices above.
- Browse the [Gallery](../gallery/index.md) if you prefer to see what Ferrum looks like in practice.
