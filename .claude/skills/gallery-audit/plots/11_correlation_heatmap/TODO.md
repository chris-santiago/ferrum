# Row 11 — Correlation heatmap

**ferrum API:** `ferrum.heatmap(corr_df, annot=True)` — available (Phase 9e). Check whether `annot=True` is the default; if not, the audit will surface that as a B7 finding.

## Panels to write

- `ferrum_panel.py` — load breast_cancer as DataFrame, take first 10 numeric cols, `.corr()`, then `ferrum.heatmap(corr).properties(...).show_svg()`. Do NOT pass `annot=True` — defaults only.
- `seaborn_panel.py` — `seaborn.heatmap(corr, annot=True, ax=ax)` (seaborn's default is `annot=False`, but adding it makes the comparison fair vs yellowbrick which shows values)

Wait — defaults only. So `seaborn.heatmap(corr, ax=ax)` (no annot). Document this in the verdict notes so the judge sees both libraries' actual default behavior.

- `yellowbrick_panel.py` — `yellowbrick.features.Rank2D(algorithm='pearson', ax=ax); fit; finalize` (shows numeric values? check)

**Watch for:** B7 (cell value overlay), F2 (diverging cmap for signed correlations — both red and blue are common defaults).
