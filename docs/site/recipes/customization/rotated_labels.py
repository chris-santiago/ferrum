"""Recipe: Rotated X-axis labels for long category names.

Demonstrates using configure_axis(label_angle=...) to rotate tick labels
when category names are long enough to collide at default orientation.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "department": [
        "Engineering",
        "Product Management",
        "Sales & Marketing",
        "Customer Success",
        "Research & Development",
    ],
    "headcount": [42, 18, 31, 24, 15],
})

chart = (
    fm.Chart(df)
    .mark_bar()
    .encode(
        x=fm.X("department:N", sort="-y"),
        y="headcount:Q",
    )
    .configure_axis(x=True, y=False, label_angle=-40)
    .configure_padding(bottom=80)
    .labs(title="Department Headcount", x=None, y="Headcount")
)

# chart.save("rotated_labels.svg")
