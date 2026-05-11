# Row 03 — Confusion matrix

**ferrum API:** `ferrum.confusion_matrix_chart(model, X, y)` — available (Phase 10).
Square aspect ratio (500×500) — overrides the default 640×480.

## Panels to write

- `ferrum_panel.py` — `ferrum.confusion_matrix_chart(model, Xte, yte).properties(width=W, height=H).show_svg()`
- `sklearn_panel.py` — `sklearn.metrics.ConfusionMatrixDisplay.from_estimator(model, Xte, yte, ax=ax)`
- `yellowbrick_panel.py` — `yellowbrick.classifier.ConfusionMatrix(model, ax=ax); fit/score/finalize`
- `skp_panel.py` — `scikitplot.metrics.plot_confusion_matrix(yte, ypred, ax=ax)`

**Watch for:** per-cell count overlay (B3 in rubric). sklearn and yellowbrick both show this by default. If ferrum doesn't, that's a HIGH-severity finding.
