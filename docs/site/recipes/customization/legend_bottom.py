"""Recipe: Legend at the bottom with horizontal layout.

Demonstrates moving the legend below the chart and arranging its items
horizontally — useful for charts with limited vertical space.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"] * 3,
    "region": ["North"] * 6 + ["South"] * 6 + ["West"] * 6,
    "sales": [
        120, 145, 132, 168, 181, 175,
        98, 112, 105, 130, 142, 138,
        85, 94, 91, 107, 118, 115,
    ],
})

chart = (
    fm.Chart(df)
    .mark_line()
    .encode(
        x="month:N",
        y="sales:Q",
        color="region:N",
    )
    .configure_legend(orient="bottom", direction="horizontal")
    .labs(title="Regional Monthly Sales", x=None, y="Sales (units)")
)

# chart.save("legend_bottom.svg")
