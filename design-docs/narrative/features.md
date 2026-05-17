# What does Ferrum bring over existing plotting libraries?

Three things none of them have individually:

**1. One grammar that scales to production data size.** Every existing library breaks at some point — Altair at 5k rows, seaborn/matplotlib at ~100k marks, plotly at ~500k. You don't swap tools or APIs as data grows. The same chart spec works at 100 rows and 10M rows.

**2. Model diagnostics as first-class grammar objects, not a parallel API.** Yellowbrick and scikit-plot are separate universes — different objects, different styling, non-composable. In Ferrum a confusion matrix and a ROC curve can be hconcatted, themed, and saved with the same code as any other chart. They're charts, not special objects.

**3. Stat computation in the render pipeline, not in userspace.** Every existing library makes you precompute — call SciPy, build the KDE yourself, bootstrap CIs manually, then hand the result to the plotter. Ferrum declares intent and computes in Rust before rendering. The library is statistically literate, not just a renderer.

The secondary wins — zero-copy Arrow ingestion, no matplotlib dependency, headless rendering without a display server, GPU-backed interactivity — are real but incremental. The three above don't exist in combination anywhere today.

# Key Features

**The headline:**

> One grammar. Any data size. Model diagnostics included.

**Five features worth leading with:**

**Grammar of Graphics, without the ceiling.** Declarative, composable, layered — like Altair or plotnine — but no row limits, no API switch when data grows. Auto-raster and GPU rendering happen transparently behind the same spec.

**Stat transforms in the pipeline.** KDE, LOESS, bootstrap CIs, binning — declared in the chart, computed in Rust before rendering. You stop preprocessing data before plotting.

**Model diagnostics that compose.** ROC curves, SHAP beeswarm, residuals, calibration — same grammar, same theme, same `.save()`. `fr.hconcat(roc_chart, confusion_chart)` just works.

**Zero system dependencies.** Ships in a wheel. No Cairo, no X11, no display server. Renders in Kubernetes, CI, SSH sessions. `pip install ferrum` is the entire setup.

**SHAP and ICE at full sample size.** The plots that matter most for understanding models at scale — the ones existing tools sample or crash on — render in full because the rasterization is in Rust and the interactivity is GPU-backed.

## Incorporated during design implementation

**Handles multiple dataframe APIs via `Narwhals`.** Polars, pandas, modin, cuDF, dask, and ibis all flow through the same `Chart(data)` constructor — internally normalized to Arrow once, then routed through the Rust core unchanged. No per-framework adapters in user code; no special-case ingestion paths in ferrum.

**Diagnostics parametrized by intent, not by estimator class.** Yellowbrick's tuning visualizers — `AlphaSelection`, `ValidationCurve`, `LearningCurve`, `CVScores` — each require specific estimator wrappers or argument shapes. `AlphaSelection` only accepts a `*CV` regressor (`RidgeCV` / `LassoCV` / `ElasticNetCV`); `ValidationCurve` wants a particular `param_name` / `param_range` pair on its own class; `LearningCurve` is its own visualizer class with its own fit/score/finalize lifecycle. Ferrum's equivalents — `alpha_selection_chart`, `validation_curve_chart`, `learning_curve_chart`, `cv_scores_chart` — all accept *any* estimator instance plus the sweep / CV configuration as plain keyword arguments. The regressor you already trained on your model line is the same object you pass on your plot line; no parallel CV-wrapper or `*Curve` class to instantiate.

The same pattern propagates through the classification diagnostics (`roc_chart(model, X, y)`, `pr_chart(model, X, y)`, `confusion_matrix_chart(model, X, y)`, `calibration_chart(model, X, y)`, `importance_chart(model, X, y)`, `pdp_chart(model, X, y, features=[...])`): one function per chart kind, parametrized by what the user wants to *see*, not by what shape of estimator wrapper the library forces them into. This is the practical payoff of feature #2 above — "model diagnostics as first-class grammar objects, not a parallel API." When the diagnostic is a chart and the chart's input is the same estimator the rest of the codebase already uses, the parallel-universe boundary that yellowbrick draws around its visualizers simply does not exist.
