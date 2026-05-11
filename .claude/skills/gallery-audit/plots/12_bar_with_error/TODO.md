# Row 12 — Bar chart with error bars

**ferrum API:** `ferrum.catplot(data, x="day", y="total_bill", kind="bar")` — available, but check whether the default aggregate is `mean` and whether error bars are shown by default. Seaborn's `barplot` shows mean + 95% CI by default; this is a strong default that ferrum should match.

## Panels to write

- `ferrum_panel.py` — `ferrum.catplot(df, x="day", y="total_bill", kind="bar").properties(...).show_svg()`. Inspect the output: does it show error bars by default?
- `seaborn_panel.py` — `seaborn.barplot(data=df, x="day", y="total_bill", ax=ax)`

**Watch for:** D3 (error bars by default — seaborn does this without being asked; this is a HIGH-value default if ferrum does *not* show them).
