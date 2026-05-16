---
hide:
  - navigation
  - toc
---

# Ferrum

**Grammar-of-graphics statistical visualization for Python, with a Rust core.**
One chart model for statistical graphics, interactive views, and ML diagnostics.

[Get started :material-arrow-right:](getting-started/install.md){ .md-button .md-button--primary }
[Why Ferrum](getting-started/why-ferrum.md){ .md-button }

---

## One mental model, from scatter plot to SHAP summary

Ferrum is a statistical visualization library built around one idea: every chart should follow the same mental model. A scatter plot, a faceted histogram, a ROC curve, and a SHAP beeswarm are all charts, so Ferrum builds them from the same grammar of data, encodings, marks, scales, coordinates, and statistical transforms.

A practitioner moving from Altair, Seaborn, or Yellowbrick should be able to reach for the charts they need without falling back to matplotlib — and without paying the tax of switching abstractions every time the question changes.

## What Ferrum is for

<div class="grid cards" markdown>

-   __Grammar of graphics, without the ceiling__

    ---

    Declarative, composable, layered — like Altair or plotnine — but no row limits and no API switch when data grows. Auto-raster and GPU rendering happen transparently behind the same spec.

-   __Stat transforms in the pipeline__

    ---

    KDE, LOESS, bootstrap CIs, binning — declared in the chart, computed in Rust before rendering. You stop preprocessing data before plotting.

-   __Diagnostics that compose__

    ---

    ROC curves, SHAP beeswarm, residuals, calibration — same grammar, same theme, same [`.save()`][ferrum.Chart.save]. [`fm.hconcat()`][ferrum.hconcat] just works.

-   __Zero system dependencies__

    ---

    Ships in a wheel. No Cairo, no X11, no display server. Renders in Kubernetes, CI, SSH sessions. `pip install` is the entire setup.

-   __SHAP and ICE at full sample size__

    ---

    The plots that matter most for understanding models at scale — the ones existing tools sample or crash on — render in full because the rasterization is in Rust and the interactivity is GPU-backed.

-   __Handles every dataframe API__

    ---

    Polars, pandas, modin, cuDF, dask, and ibis all flow through the same [`Chart`][ferrum.Chart] constructor. Narwhals normalizes the input to Arrow once; the Rust core sees one shape. No per-framework adapters in your code, no special-case ingestion paths in ferrum.

</div>

## What you read next

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __New to Ferrum?__

    ---

    [Install](getting-started/install.md) it, render [your first plot](getting-started/first-plot.md), then read [Why Ferrum](getting-started/why-ferrum.md) to see what makes it different from seaborn, Altair, or Yellowbrick.

-   :material-book-open-variant:{ .lg .middle } __Want the design rationale?__

    ---

    The [Concepts](guide/concepts/one-chart-model.md) pages explain the core beliefs — one chart model, stats in the rendering pipeline, model outputs as data, and performance as a public-API concern.

-   :material-image-multiple:{ .lg .middle } __Prefer to see it work?__

    ---

    The [Gallery](gallery/index.md) walks through hand-crafted examples, each one teaching a technique rather than just rendering a figure.

-   :material-api:{ .lg .middle } __Looking up a specific symbol?__

    ---

    The [API Reference](api/ferrum.md) covers every public class and function — from [`Chart`][ferrum.Chart] and [`encoding`](api/encoding.md) to [`plots`](api/plots.md), [`themes`](api/themes.md), [`selection`](api/selection.md), and [more](api/ferrum.md).

</div>
