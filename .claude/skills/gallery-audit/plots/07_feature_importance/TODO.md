# Row 07 — Feature importance

**ferrum API:** NOT YET IMPLEMENTED. `ferrum._diagnostics.source.py` does extract `feature_importances_` / `coef_` from sklearn estimators (see line 21), but no figure-level function exposes them as a horizontal-bar chart yet.

## Needed in ferrum

`ferrum.feature_importance_chart(model, *, top_n=None, sort=True)` — horizontal bar, sorted descending by importance, optional top-N truncation.

When that lands, update `ferrum_status = "READY"` and add `ferrum` to `panels`.

## Comparator panels

- `yellowbrick_panel.py` — `yellowbrick.model_selection.FeatureImportances(model, ax=ax); fit; finalize`
- `skp_panel.py` — `scikitplot.estimators.plot_feature_importances(model, feature_names=names, ax=ax)`

**Watch for:** B8 (importance value on each bar — yellowbrick shows relative %; scikit-plot does not).
