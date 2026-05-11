# Row 09 — Boxplot

**ferrum API:** `ferrum.catplot(data, x="day", y="total_bill", kind="box")` — available.

## Panels to write

- `ferrum_panel.py` — `ferrum.catplot(df, x="day", y="total_bill", kind="box").properties(...).show_svg()`
- `seaborn_panel.py` — `seaborn.boxplot(data=df, x="day", y="total_bill", ax=ax)`

**Watch for:** outlier markers (seaborn shows them as diamond fliers by default; does ferrum?), median line visibility, whisker-cap rendering.
