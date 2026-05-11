# Row 13 — Partial dependence plot

**ferrum API:** `ferrum.pdp_chart(model, X, y, features=[...])` — available (Task 23 / Phase 10d, commit `4679e86`).

`features=` is a required argument (no default); both panels must pass the same list. Default `kind="average"`, `grid_resolution=100`, `ice_alpha=0.2`, `center=False`.

## Panels

- `ferrum_panel.py` — `ferrum.pdp_chart(model, X, y, features=[0, 1, 2, 3]).properties(...).show_svg()`
- `sklearn_panel.py` — `sklearn.inspection.PartialDependenceDisplay.from_estimator(model, X, features=[0, 1, 2, 3])`

No yellowbrick / scikit-plot equivalent — only sklearn ships a default PDP display.

## Watch for

- **Grid layout**: sklearn arranges PDPs in a 2×N grid by default. Does ferrum match this convention?
- **B5 axis labels**: feature name on x, "partial dependence" on y per subplot — sklearn does this. Does ferrum?
- **C-category reference**: PDPs often show a horizontal y=0 line or training-data mean. Either library?
- **Rugplot at bottom of each subplot**: sklearn adds this by default (shows training-data distribution). ferrum?
- **`kind="average"` vs `"individual"` (ICE)**: both libraries default to average. Worth flagging if ICE looks different.
