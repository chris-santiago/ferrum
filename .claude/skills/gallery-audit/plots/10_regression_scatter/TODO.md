# Row 10 — Regression scatter

**ferrum API:** `ferrum.lmplot(data, x="total_bill", y="tip")` — available (Phase 9e).

## Panels to write

- `ferrum_panel.py` — `ferrum.lmplot(df, x="total_bill", y="tip").properties(...).show_svg()`
- `seaborn_panel.py` — `seaborn.regplot(data=df, x="total_bill", y="tip", ax=ax)`

**Watch for:** D2 (CI band around fit line — seaborn shows ±95% by default), B6 (R² annotation — seaborn does NOT show this; this is a place ferrum could be *better* than seaborn's defaults if it added it).
