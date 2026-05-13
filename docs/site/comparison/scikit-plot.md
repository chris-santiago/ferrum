# Migrating from scikit-plot

scikit-plot (`skplt`) provides one-function plotting for sklearn model evaluation — `plot_roc`, `plot_confusion_matrix`, `plot_precision_recall`, etc. Ferrum's `*_chart` helpers follow the same pattern: pass a fitted model and held-out data, get a chart back.

## Function mapping

| scikit-plot | Ferrum | Notes |
|---|---|---|
| `skplt.metrics.plot_roc(y_test, y_probas)` | `fm.roc_chart(model, X_test, y_test)` | Ferrum takes the model directly; computes probabilities internally. |
| `skplt.metrics.plot_precision_recall(y_test, y_probas)` | `fm.pr_chart(model, X_test, y_test)` | Same. |
| `skplt.metrics.plot_confusion_matrix(y_test, y_pred)` | `fm.confusion_matrix_chart(model, X_test, y_test)` | `normalize=` supported. |
| `skplt.metrics.plot_calibration_curve(y_test, probas_list)` | `fm.calibration_chart(model, X_test, y_test)` | Single model or `ComparedModelSource` for multi-model. |
| `skplt.metrics.plot_cumulative_gain(y_test, y_probas)` | `fm.gain_chart(model, X_test, y_test)` | — |
| `skplt.metrics.plot_lift_curve(y_test, y_probas)` | `fm.lift_chart(model, X_test, y_test)` | — |
| `skplt.estimators.plot_feature_importances(model)` | `fm.importance_chart(model, X_test, y_test)` | Handles `feature_importances_`, permutation, and coefficients. |
| `skplt.estimators.plot_learning_curve(model, X, y)` | `fm.learning_curve_chart(model, X, y)` | — |
| `skplt.cluster.plot_silhouette(X, labels)` | `fm.silhouette_chart(model, X)` | Takes the fitted clusterer directly. |
| `skplt.cluster.plot_elbow_curve(model, X)` | `fm.ElbowVisualizer(model)` | Visualizer-style (`.fit()` / `.show()`). |
| `skplt.decomposition.plot_pca_component_variance(pca)` | `fm.pca_scree_chart(model, X)` | — |

## Key differences

### Model-in, chart-out

scikit-plot typically requires you to pre-compute predictions (`y_pred`) or probabilities (`y_probas`) and pass arrays. Ferrum takes the fitted model directly and computes everything internally via `ModelSource`:

<!--pytest.mark.skip-->
```python
# scikit-plot
y_probas = model.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_probas)
plt.savefig("roc.svg")

# Ferrum
chart = fm.roc_chart(model, X_test, y_test)
chart.save("roc.svg")
```

### No matplotlib

scikit-plot renders to matplotlib and requires `plt.show()` or `plt.savefig()`. Ferrum returns a `Chart` — same object you get from `fm.Chart(data).mark_point().encode(...)`.

### Charts compose

scikit-plot functions are standalone — combining multiple plots means managing matplotlib subplots. In Ferrum, diagnostic charts compose with the same operators as any other chart:

<!--pytest.mark.skip-->
```python
roc = fm.roc_chart(model, X_test, y_test)
cm = fm.confusion_matrix_chart(model, X_test, y_test)
gain = fm.gain_chart(model, X_test, y_test)
report = (roc | cm) & gain
report.save("model_report.svg")
```

### ModelSource for efficiency

When running multiple diagnostics, build a `ModelSource` once:

<!--pytest.mark.skip-->
```python
source = fm.ModelSource(model, X_test, y_test)
roc = fm.roc_chart(source)
cm = fm.confusion_matrix_chart(source)
pr = fm.pr_chart(source)
```

Predicted probabilities, confusion counts, and other derived tables are computed once and shared.

### Themes

scikit-plot inherits matplotlib styling. Ferrum charts accept theme values:

<!--pytest.mark.skip-->
```python
fm.roc_chart(model, X_test, y_test).theme(fm.themes.publication).save("roc.svg")
```

## Coverage comparison

Ferrum's diagnostic surface is a superset of scikit-plot's:

| Category | scikit-plot | Ferrum |
|---|---|---|
| Classification metrics | ROC, PR, confusion matrix, calibration, gain, lift | All of scikit-plot's + class prediction error, discrimination threshold |
| Regression | — | Residuals, prediction error, Cook's distance |
| Feature explanation | Feature importances | Importances + SHAP (beeswarm, bar, waterfall) + partial dependence |
| Model selection | Learning curve | Learning curve + validation curve + CV scores + alpha selection |
| Clustering | Silhouette, elbow, PCA variance | All of scikit-plot's + intercluster distance, decision boundary, parallel coordinates |

## Where to go next

- [Model diagnostics](../guide/model-diagnostics.md) for the full diagnostic surface.
- [Figure-level helpers](../guide/figure-helpers.md) for the one-line helper pattern.
- [Composition](../guide/composition.md) for building multi-panel model reports.
