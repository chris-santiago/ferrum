"""Recipe: Dual y-axes with different scales.

Demonstrates SecondaryY to overlay two series that have incompatible units
(e.g. revenue in dollars and conversion rate as a percentage), each with
its own independent y axis.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "revenue": [125_000, 138_500, 112_000, 161_000, 183_000, 172_000],
    "conversion_rate": [0.032, 0.038, 0.029, 0.041, 0.045, 0.043],
})

chart = (
    fm.Chart(df)
    .mark_bar(opacity=0.7, color="#1e40af")
    .encode(x="month:N", y="revenue:Q")
    .configure(
        axis_y=fm.AxisConfig(label_format="currency", title_color="#1e40af"),
        axis_y2=fm.AxisConfig(label_format="percent", title_color="#dc2626"),
    )
    .configure_padding(right=80)
    .labs(title="Revenue and Conversion Rate", y="Revenue")
    + fm.SecondaryY(
        field="conversion_rate",
        mark="line",
        color="#dc2626",
        axis=fm.Axis(title="Conversion Rate"),
    )
)

# chart.save("dual_axis.svg")
