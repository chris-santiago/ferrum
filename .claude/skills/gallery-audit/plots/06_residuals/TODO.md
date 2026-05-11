# Row 06 — Residuals plot

**ferrum API:** `ferrum.residuals_chart(model, X, y)` — available (Phase 10a).

## Panels to write

- `ferrum_panel.py` — `ferrum.residuals_chart(model, Xte, yte).properties(width=W, height=H).show_svg()`
- `sklearn_panel.py` — `sklearn.metrics.PredictionErrorDisplay.from_estimator(model, Xte, yte, ax=ax)`
- `seaborn_panel.py` — `seaborn.residplot(x=ypred, y=yte - ypred, ax=ax)` (PEP 723 inline deps: seaborn, scikit-learn, matplotlib)
- `yellowbrick_panel.py` — `yellowbrick.regressor.ResidualsPlot(model, ax=ax); fit/score/finalize`

**Watch for:** C4 (y=0 reference line), B5 (R²/MAE/RMSE annotation). yellowbrick annotates train/test R² by default.
