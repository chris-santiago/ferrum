# Model outputs are data

A confusion matrix is a table.

A ROC curve is derived tabular data from predicted scores. A calibration plot is derived tabular data. A SHAP explanation is a set of columns, one per feature. A learning curve is a small dataframe of training-size-vs-score.

There is no philosophical reason these objects should require a separate visualization universe. Yet existing Python tooling has historically built one anyway — Yellowbrick objects, scikit-plot figures, custom matplotlib wrappers — each with its own object model, its own styling, and its own non-composable conventions.

Ferrum's position is that model outputs are data, and the natural way to plot them is the same way as everything else.

## What this principle gets you

Treating diagnostics as data has direct, practical consequences for what you can do with them. The diagnostic plots return `Chart` objects (or compound views like `JointChart` / `RepeatChart` / `ClusterMapChart`) — they are not foreign artifacts that stop being composable the moment they appear.

That means a ROC curve participates in the rest of the grammar:

- **It composes.** `fm.hconcat(roc_chart, confusion_chart, calibration_chart)` puts three diagnostics in a row. The same `hconcat` you use for scatter plots, with the same `|` operator.
- **It themes.** Set a process-wide theme via `set_default_theme()`, or pass a theme to one chart with `.theme()`, and the ROC curve picks it up identically to a faceted distribution plot.
- **It saves.** `.save("roc.svg")` works exactly the same as for any other chart.
- **It facets.** When the underlying data supports it, the same encoding channels (`Facet`, `FacetRow`, `FacetCol`) apply to diagnostics.
- **It renders.** Static SVG, CPU raster, GPU interactive — same chart spec, same renderer choices. (Interactive output is described in [Interactivity is a renderer](interactivity.md).)

These compositions are not special-cased. They fall out of the chart model because a ROC chart and a scatter chart are the same kind of object.

## How the surface is organized

Ferrum exposes the model-output side of the library through three coordinated layers:

**`ModelSource`** is the data interface. It wraps a fitted model and a held-out dataset and exposes the derived tables that diagnostics build on: predicted probabilities, predicted classes, residuals, ROC curve points, PR curve points, calibration bins, confusion-matrix counts, learning-curve samples, validation-curve samples, SHAP values, partial dependence grids, and related outputs. The point of `ModelSource` is that you compute those derived tables once, then the various diagnostic plots reuse the same data.

**Figure-level diagnostic helpers** are the convenience layer: `roc_chart`, `pr_chart`, `calibration_chart`, `confusion_matrix_chart`, `residuals_chart`, `shap_chart`, `learning_curve_chart`, `validation_curve_chart`, `feature_importances_chart`, `pdp_chart`, and similar functions. They take a fitted model + data (or a `ModelSource`) and return a `Chart`. This is the layer you reach for when you want a diagnostic in one line.

**sklearn-protocol visualizers** — `ROCVisualizer`, `CalibrationVisualizer`, `ConfusionMatrixVisualizer`, `ResidualsVisualizer`, `SHAPVisualizer`, `LearningCurveVisualizer`, and others — provide an object-oriented interface that mirrors yellowbrick's pattern. Each visualizer has a `.fit()` / `.score()` / `.show()` flow and ultimately returns a `Chart`. These are useful when you want lifecycle control or when the visualizer needs to manage CV-fold state.

All three layers produce the same kind of chart object. You can mix them freely: take a `roc_chart()` helper result and `hconcat` it with a `ROCVisualizer().show()` result.

## What this lets you stop doing

The most visible practical effect is that you stop maintaining a separate object hierarchy for diagnostics:

- You stop installing two plotting libraries — one for exploration, one for evaluation.
- You stop writing helper functions that translate between yellowbrick figures and your "real" plots.
- You stop maintaining theme parallels — one set of style settings for seaborn-style plots, another for diagnostics.
- You stop choosing between "interactive plot for EDA" and "static plot for the model report"; the same chart can do both depending on the renderer.

You start treating model-evaluation plots as one more thing you can do with the chart grammar you already know.

## Why this is "part of the contract," not a feature

The commitment to model-outputs-as-data is structural, not cosmetic. If diagnostics were a parallel API, every other principle in this library would have asterisks attached:

- ["One chart model"](one-chart-model.md) would mean "one chart model, except for diagnostics."
- ["Stats in the rendering pipeline"](stats-pipeline.md) would mean "stats in the pipeline, unless you want a calibration curve, in which case you precompute the bin counts."
- ["Performance & scale"](performance-scale.md) would mean "the architecture lets the same spec scale, unless you want SHAP at full sample size, in which case you switch libraries."
- ["Dataframe pluralism"](dataframe-pluralism.md) would mean "any dataframe goes through Chart(data), except your diagnostic data, which needs a different ingestion path."

None of those qualifiers exist in Ferrum because diagnostics are not a parallel system. They live inside the same grammar, the same composition operators, the same rendering pipeline, and the same dataframe ingestion path.

That is the load-bearing claim. Every other choice in the library either follows from it (the visualizer base class returning `Chart`, the diagnostic helpers in `ferrum.figure`, the encoding channels working uniformly across diagnostic and non-diagnostic plots) or is consistent with it.

## What this does not mean

This principle is about the visualization layer, not about model training, evaluation logic, or statistical inference. Ferrum is not a model-evaluation framework. It does not select metrics for you, it does not warn you about model overfitting, and it does not replace scikit-learn / statsmodels / your gradient-boosting library.

What Ferrum does is take the *outputs* those frameworks produce — fitted models, predictions, residuals, importance scores, SHAP values — and turn them into charts that compose, theme, save, and render through the same pipeline as the rest of the library.

The boundary is the same as it is for stats in the pipeline ([described elsewhere](stats-pipeline.md)): Ferrum's job is to be a coherent visualization system. The model lives outside; the chart lives inside; `ModelSource` and the diagnostic helpers are the bridge.

## Where to go next

- [One chart model](one-chart-model.md) for the broader grammar that diagnostics participate in.
- [Stats in the rendering pipeline](stats-pipeline.md) for why diagnostic-style derivations (ROC curve construction, calibration bin counts, etc.) belong inside the chart spec rather than outside it.
- [Dataframe pluralism](dataframe-pluralism.md) for how the same chart sits on top of any dataframe input — including the derived tables `ModelSource` produces.
- [Performance & scale](performance-scale.md) for the architecture that lets SHAP and ICE plots render at full sample size.
- The [Model diagnostics](../model-diagnostics.md) Guide page is where this principle becomes specific — what the helpers are, when to use each, and how to compose them.
