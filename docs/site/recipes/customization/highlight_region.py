"""Recipe: Highlight a region with a rectangle annotation.

Demonstrates using annotation.span to shade a meaningful range of values
along an axis, with an optional label inside the band.
"""
import polars as pl
import ferrum as fm
import ferrum.annotation as ann

df = pl.DataFrame({
    "week": list(range(1, 25)),
    "score": [
        62, 65, 68, 70, 73, 72, 69, 75, 78, 80,
        82, 85, 84, 86, 88, 87, 90, 91, 89, 93,
        95, 94, 96, 98,
    ],
})

# Highlight the target range [80, 100] on the y-axis
chart = (
    fm.Chart(df)
    .mark_line(stroke_width=2)
    .encode(x="week:Q", y="score:Q")
    .configure(
        axis_y=fm.AxisConfig(domain_min=55, domain_max=105),
        axis_x=fm.AxisConfig(tick_count=12),
    )
    + ann.rect(fm.norm(0.0), 80, fm.norm(1.0), fm.norm(0.0), fill="#059669", opacity=0.15)
    + ann.text(fm.norm(0.02), 90, "Target zone", color="#059669", font_size=11, anchor="start")
    + ann.line(
        fm.norm(0.0), 80, fm.norm(1.0), 80,
        stroke="#16a34a", stroke_width=1, dash=[4, 4],
    )
)

# chart.save("highlight_region.svg")
