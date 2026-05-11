# Row 05 — Learning curve

**ferrum API:** NOT YET IMPLEMENTED. Phase 10 (model diagnostics) currently exposes ROC, PR, calibration, confusion matrix, residuals, prediction error, Cook's distance, class balance, class prediction error, discrimination threshold — but not learning curve.

## Needed in ferrum

A figure-level function like `ferrum.learning_curve_chart(model, X, y, *, cv=5, train_sizes=...)` that:
- Computes train and validation scores at increasing train-set fractions (sklearn's `learning_curve` is the reference).
- Renders one line per series with a ribbon for ±std (or CI) — `Smooth` + `mark_ribbon` from Phase 9.
- Adds the two series to the legend.

This row stays BLOCKED until that lands. When it does:
1. Update `ferrum_status = "READY"` and add ferrum to `panels`.
2. Write `ferrum_panel.py`.

## Comparator panels (write now, blocked from running until ferrum lands)

- `sklearn_panel.py` — `sklearn.model_selection.LearningCurveDisplay.from_estimator(model, X, y, ax=ax)`
- `yellowbrick_panel.py` — `yellowbrick.model_selection.LearningCurve(model, ax=ax); fit; finalize`

**Watch for:** D1 (±std shaded band), C6 (train vs val distinguishable), E1+E2 (legend with meaningful names).
