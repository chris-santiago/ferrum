"""Recipe: Rotated X-axis labels for long category names.

Demonstrates using configure_axis(label_angle=...) to rotate tick labels
when category names are long enough to collide at default orientation.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "quarter": [
        "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024",
        "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025",
    ],
    "revenue": [125, 138, 112, 161, 183, 172, 195, 210],
})

chart = (
    fm.Chart(df)
    .mark_bar()
    .encode(
        x=fm.X("quarter:N", axis=fm.Axis(label_angle=-45)),
        y="revenue:Q",
    )
    .labs(title="Quarterly Revenue", x=None, y="Revenue ($k)")
)

# chart.save("rotated_labels.svg")
