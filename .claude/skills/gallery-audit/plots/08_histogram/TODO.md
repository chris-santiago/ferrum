# Row 08 — Histogram

**ferrum API:** `ferrum.displot(data, x="total_bill", kde=True)` — available (Phase 9e).

## Panels to write

- `ferrum_panel.py` — load tips, call `ferrum.displot(df, x="total_bill", kde=True).properties(...).show_svg()`
- `seaborn_panel.py` — `seaborn.histplot(df["total_bill"], kde=True, ax=ax)`

(No sklearn/yellowbrick/skp equivalents — this is pure EDA.)

**Watch for:** A2/A3 (x and y labels: "total_bill" vs "Count"), F3 (saturation of the histogram fill).
