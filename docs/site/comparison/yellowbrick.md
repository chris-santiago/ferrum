# Migrating from yellowbrick

Yellowbrick pioneered the idea of "visual diagnostics" — sklearn-protocol objects with `.fit()` / `.score()` / `.show()` that wrap matplotlib. Ferrum's visualizer classes follow the same lifecycle pattern, but produce grammar-of-graphics chart objects instead of matplotlib figures.

## Visualizer mapping

| yellowbrick | Ferrum visualizer | Ferrum helper |
|---|---|---|
| `ROCAUC(model)` | [`fm.ROCVisualizer(model)`][ferrum.ROCVisualizer] | [`fm.roc_chart(model, X, y)`][ferrum.roc_chart] |
| `PrecisionRecallCurve(model)` | [`fm.PRVisualizer(model)`][ferrum.PRVisualizer] | [`fm.pr_chart(model, X, y)`][ferrum.pr_chart] |
| `ConfusionMatrix(model)` | [`fm.ConfusionMatrixVisualizer(model)`][ferrum.ConfusionMatrixVisualizer] | [`fm.confusion_matrix_chart(model, X, y)`][ferrum.confusion_matrix_chart] |
| `ClassificationReport(model)` | [`fm.ClassificationReportVisualizer(model)`][ferrum.ClassificationReportVisualizer] | — |
| `ClassPredictionError(model)` | [`fm.ClassPredictionErrorVisualizer(model)`][ferrum.ClassPredictionErrorVisualizer] | [`fm.class_prediction_error_chart(model, X, y)`][ferrum.class_prediction_error_chart] |
| `DiscriminationThreshold(model)` | [`fm.DiscriminationThresholdVisualizer(model)`][ferrum.DiscriminationThresholdVisualizer] | [`fm.discrimination_threshold_chart(model, X, y)`][ferrum.discrimination_threshold_chart] |
| `ResidualsPlot(model)` | [`fm.ResidualsVisualizer(model)`][ferrum.ResidualsVisualizer] | [`fm.residuals_chart(model, X, y)`][ferrum.residuals_chart] |
| `PredictionError(model)` | [`fm.PredictionErrorVisualizer(model)`][ferrum.PredictionErrorVisualizer] | — |
| `CooksDistance(model)` | [`fm.CooksDistanceVisualizer(model)`][ferrum.CooksDistanceVisualizer] | — |
| `FeatureImportances(model)` | [`fm.FeatureImportancesVisualizer(model)`][ferrum.FeatureImportancesVisualizer] | [`fm.importance_chart(model, X, y)`][ferrum.importance_chart] |
| `LearningCurve(model)` | [`fm.LearningCurveVisualizer(model)`][ferrum.LearningCurveVisualizer] | [`fm.learning_curve_chart(model, X, y)`][ferrum.learning_curve_chart] |
| `ValidationCurve(model)` | [`fm.ValidationCurveVisualizer(model)`][ferrum.ValidationCurveVisualizer] | [`fm.validation_curve_chart(model, X, y)`][ferrum.validation_curve_chart] |
| `CVScores(model)` | [`fm.CVScoresVisualizer(model)`][ferrum.CVScoresVisualizer] | [`fm.cv_scores_chart(model, X, y)`][ferrum.cv_scores_chart] |
| `SilhouetteVisualizer(model)` | [`fm.SilhouetteVisualizer(model)`][ferrum.SilhouetteVisualizer] | — |
| `KElbowVisualizer(model)` | [`fm.ElbowVisualizer(model_class)`][ferrum.ElbowVisualizer] | — |
| `InterclusterDistance(model)` | [`fm.InterclusterDistanceVisualizer(model)`][ferrum.InterclusterDistanceVisualizer] | [`fm.intercluster_distance_chart(model, X)`][ferrum.intercluster_distance_chart] |
| `Manifold(model)` | [`fm.ManifoldVisualizer(model)`][ferrum.ManifoldVisualizer] | — |
| `ClassBalance(labels)` | [`fm.ClassBalanceVisualizer(model)`][ferrum.ClassBalanceVisualizer] | — |
| `FeatureCorrelation` | [`fm.Rank1DVisualizer`][ferrum.Rank1DVisualizer] / [`fm.Rank2DVisualizer`][ferrum.Rank2DVisualizer] | [`fm.rank1d_chart`][ferrum.rank1d_chart] / [`fm.rank2d_chart`][ferrum.rank2d_chart] |

## The lifecycle pattern

Yellowbrick and Ferrum visualizers follow the same sklearn-style protocol:

<!--pytest.mark.skip-->
```python
# yellowbrick
from yellowbrick.classifier import ROCAUC
viz = ROCAUC(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()  # renders to matplotlib

# Ferrum
import ferrum as fm
viz = fm.ROCVisualizer(model)
viz.fit(X_train, y_train).score(X_test, y_test)
chart = viz.show()  # returns a Chart
chart.save("roc.svg")
```

The difference is what `.show()` returns: yellowbrick renders to a matplotlib axes and calls `plt.show()`. Ferrum returns a [`Chart`][ferrum.Chart] that you can theme, compose, and save.

## Key differences

### No matplotlib dependency

Yellowbrick requires matplotlib. Ferrum renders SVG directly from Rust. There is no `plt.show()`, no `fig.savefig()`, no `ax` parameter.

### Output is a composable chart

Yellowbrick visualizers produce matplotlib figures that are difficult to combine. Ferrum visualizers produce regular chart objects that compose with the same operators as any other chart:

<!--pytest.mark.skip-->
```python
roc = fm.roc_chart(model, X_test, y_test)
cm = fm.confusion_matrix_chart(model, X_test, y_test)
importances = fm.importance_chart(model, X_test, y_test)
report = (roc | cm) & importances
report.save("model_report.svg")
```

### Figure-level helpers as a fast path

Every yellowbrick visualizer also has a one-line helper in Ferrum (`roc_chart`, `confusion_matrix_chart`, etc.) that skips the `.fit()` / `.score()` ceremony. Pass a fitted model and held-out data, get a chart back.

### ModelSource for efficiency

When computing multiple diagnostics on the same model, build a [`ModelSource`][ferrum.ModelSource] once and pass it to each helper — predicted probabilities and derived tables are computed once and shared:

<!--pytest.mark.skip-->
```python
source = fm.ModelSource(model, X_test, y_test)
roc = fm.roc_chart(source)
cm = fm.confusion_matrix_chart(source)
```

### Themes instead of rcParams

Yellowbrick inherits matplotlib's `rcParams` for styling. Ferrum uses immutable theme values:

<!--pytest.mark.skip-->
```python
roc = fm.roc_chart(model, X_test, y_test)
roc.theme(fm.themes.publication).save("roc_pub.svg")
```

## Coverage comparison

| Category | yellowbrick | Ferrum |
|---|---|---|
| Classification | `ROCAUC`, `PrecisionRecallCurve`, `ConfusionMatrix`, `ClassificationReport`, `ClassPredictionError`, `DiscriminationThreshold` | All covered — plus gain, lift, and multi-model calibration via [`ComparedModelSource`][ferrum.ComparedModelSource]. |
| Regression | `ResidualsPlot`, `PredictionError`, `CooksDistance` | All covered. |
| Feature analysis | `FeatureImportances`, `Rank1D`, `Rank2D` | All covered — plus SHAP (beeswarm, bar, waterfall) and partial dependence. |
| Model selection | `LearningCurve`, `ValidationCurve`, `CVScores` | All covered — plus alpha selection. |
| Clustering | `KElbow`, `Silhouette`, `InterclusterDistance` | All covered — plus parallel coordinates and decision boundary. |
| Manifold | `Manifold` (wraps sklearn TSNE/Isomap, optional UMAP) | [`ManifoldVisualizer`][ferrum.ManifoldVisualizer] — t-SNE and UMAP run in pure Rust via `manifolds-rs`, no optional Python dependency. Also PCA. |
| Target | `ClassBalance`, `FeatureCorrelation` | All covered — [`ClassBalanceVisualizer`][ferrum.ClassBalanceVisualizer], plus [`rank1d_chart`][ferrum.rank1d_chart] / [`rank2d_chart`][ferrum.rank2d_chart] for feature correlation. |
| Text | `FreqDistVisualizer`, `TSNEVisualizer` for text data | Not in scope. t-SNE is available via [`ManifoldVisualizer`][ferrum.ManifoldVisualizer] for general dimensionality reduction. |

## Where to go next

- [Model diagnostics](../guide/model-diagnostics.md) for the full diagnostic surface.
- [Figure-level helpers](../guide/figure-helpers.md) for the one-line helper pattern.
- [Themes](../guide/themes.md) for styling diagnostics.
