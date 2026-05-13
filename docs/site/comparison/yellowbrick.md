# Migrating from yellowbrick

Yellowbrick pioneered the idea of "visual diagnostics" — sklearn-protocol objects with `.fit()` / `.score()` / `.show()` that wrap matplotlib. Ferrum's visualizer classes follow the same lifecycle pattern, but produce grammar-of-graphics chart objects instead of matplotlib figures.

## Visualizer mapping

| yellowbrick | Ferrum visualizer | Ferrum helper |
|---|---|---|
| `ROCAUC(model)` | `fm.ROCVisualizer(model)` | `fm.roc_chart(model, X, y)` |
| `PrecisionRecallCurve(model)` | `fm.PRVisualizer(model)` | `fm.pr_chart(model, X, y)` |
| `ConfusionMatrix(model)` | `fm.ConfusionMatrixVisualizer(model)` | `fm.confusion_matrix_chart(model, X, y)` |
| `ClassificationReport(model)` | `fm.ClassificationReportVisualizer(model)` | — |
| `ClassPredictionError(model)` | `fm.ClassPredictionErrorVisualizer(model)` | `fm.class_prediction_error_chart(model, X, y)` |
| `DiscriminationThreshold(model)` | `fm.DiscriminationThresholdVisualizer(model)` | `fm.discrimination_threshold_chart(model, X, y)` |
| `ResidualsPlot(model)` | `fm.ResidualsVisualizer(model)` | `fm.residuals_chart(model, X, y)` |
| `PredictionError(model)` | `fm.PredictionErrorVisualizer(model)` | `fm.prediction_error_chart(model, X, y)` |
| `CooksDistance(model)` | `fm.CooksDistanceVisualizer(model)` | `fm.cooks_distance_chart(model, X, y)` |
| `FeatureImportances(model)` | `fm.FeatureImportancesVisualizer(model)` | `fm.importance_chart(model, X, y)` |
| `LearningCurve(model)` | `fm.LearningCurveVisualizer(model)` | `fm.learning_curve_chart(model, X, y)` |
| `ValidationCurve(model)` | `fm.ValidationCurveVisualizer(model)` | `fm.validation_curve_chart(model, X, y)` |
| `CVScores(model)` | `fm.CVScoresVisualizer(model)` | `fm.cv_scores_chart(model, X, y)` |
| `SilhouetteVisualizer(model)` | `fm.SilhouetteVisualizer(model)` | `fm.silhouette_chart(model, X)` |
| `KElbowVisualizer(model)` | `fm.ElbowVisualizer(model)` | — |
| `InterclusterDistance(model)` | `fm.InterclusterDistanceVisualizer(model)` | `fm.intercluster_distance_chart(model, X)` |

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

The difference is what `.show()` returns: yellowbrick renders to a matplotlib axes and calls `plt.show()`. Ferrum returns a `Chart` that you can theme, compose, and save.

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

When computing multiple diagnostics on the same model, build a `ModelSource` once and pass it to each helper — predicted probabilities and derived tables are computed once and shared:

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

## What yellowbrick has that Ferrum does not (yet)

- **`ManifoldVisualizer`** — yellowbrick wraps TSNE/UMAP with a visual diagnostic. Ferrum has `ManifoldVisualizer` but the UMAP dependency is optional (`pip install ferrum[umap]`).
- **Target visualizers** — `ClassBalance`, `FeatureCorrelation`. Ferrum covers `ClassBalanceVisualizer` and has `rank1d_chart` / `rank2d_chart` for feature correlation.
- **Text visualizers** — `FreqDistVisualizer`, `TSNEVisualizer` for text data. Not in Ferrum's scope.

## Where to go next

- [Model diagnostics](../guide/model-diagnostics.md) for the full diagnostic surface.
- [Figure-level helpers](../guide/figure-helpers.md) for the one-line helper pattern.
- [Themes](../guide/themes.md) for styling diagnostics.
