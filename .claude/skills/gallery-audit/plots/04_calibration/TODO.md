# Row 04 — Calibration curve

**ferrum API:** `ferrum.calibration_chart(model, X, y)` — available.

## Panels to write

- `ferrum_panel.py` — `ferrum.calibration_chart(model, Xte, yte).properties(width=W, height=H).show_svg()`
- `sklearn_panel.py` — `sklearn.calibration.CalibrationDisplay.from_estimator(model, Xte, yte, ax=ax)`
- `skp_panel.py` — `scikitplot.metrics.plot_calibration_curve(yte, [probas], clf_names=["LR"], ax=ax)`

(No yellowbrick equivalent.)

**Watch for:** B4 (Brier / ECE annotation) and C3 (y=x perfect-calibration line). sklearn shows the diagonal but not Brier; scikit-plot shows the diagonal.
