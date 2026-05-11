# Row 02 — Precision-recall curve

**ferrum API:** `ferrum.pr_chart(model, X, y)` — available (Phase 10b).
Note: `annotate_ap=True` and `iso_lines=True` are reserved for Phase 10h.

## Panels to write

- `ferrum_panel.py` — `ferrum.pr_chart(model, Xte, yte).properties(width=W, height=H).show_svg()`
- `sklearn_panel.py` — `sklearn.metrics.PrecisionRecallDisplay.from_estimator(model, Xte, yte, ax=ax)`
- `yellowbrick_panel.py` — `yellowbrick.classifier.PrecisionRecallCurve(model, ax=ax); viz.fit; viz.score; viz.finalize`
- `skp_panel.py` — `scikitplot.metrics.plot_precision_recall(yte, probas, ax=ax)`

Update `panels = ["ferrum", "sklearn", "yellowbrick", "skp"]` in config.toml when scripts are written.
Copy `../01_roc/*_panel.py` and adapt — the dataset, env vars, and figure boilerplate are identical.
