"""Recipe: Custom tick values and formatting.

Demonstrates using tick_values to place ticks at specific data positions
and tick_count to control tick density. Useful when the data has meaningful
threshold values that should always appear as labeled ticks.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "score": [45, 52, 58, 61, 67, 71, 74, 78, 82, 89, 93, 97],
    "count": [3, 7, 12, 18, 24, 31, 28, 22, 17, 11, 6, 2],
})

# Grade thresholds: 60 = D/C boundary, 70 = C/B, 80 = B/A
chart = (
    fm.Chart(df)
    .mark_bar()
    .encode(x="score:Q", y="count:Q")
    .configure(
        axis_x=fm.AxisConfig(tick_values=[0, 60, 70, 80, 90, 100], label_font_size=11),
        axis_y=fm.AxisConfig(tick_count=5, label_format="integer"),
    )
    .labs(title="Score Distribution", x="Score", y="Students")
)

# chart.save("custom_ticks.svg")
