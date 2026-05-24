"""Recipe: Currency-formatted Y-axis.

Demonstrates using the "currency" format preset to display monetary values
on the y-axis with proper dollar and thousands-separator formatting.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "revenue": [1_240_000, 1_580_000, 1_410_000, 1_920_000],
})

chart = (
    fm.Chart(df)
    .mark_bar()
    .encode(
        x=fm.X("quarter:N", sort=None),
        y="revenue:Q",
    )
    .configure_axis(y=True, x=False, label_format="currency")
    .labs(title="Quarterly Revenue", y="Revenue")
)

# chart.save("currency_axis.svg")
