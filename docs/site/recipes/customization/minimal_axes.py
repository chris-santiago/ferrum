"""Recipe: Strip chart to minimal axis elements.

Demonstrates removing decorative axis elements — domain lines, tick marks,
grid lines — to produce a maximally clean chart. Useful for small multiples
or dashboard tiles where axis chrome competes with the data.
"""
import polars as pl
import ferrum as fm

df = pl.DataFrame({
    "category": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "value": [0.42, 0.68, 0.55, 0.81, 0.37],
})

chart = (
    fm.Chart(df)
    .mark_bar(corner_radius_top_left=3, corner_radius_top_right=3)
    .encode(
        x=fm.X("category:N", sort="-y"),
        y=fm.Y("value:Q", axis=fm.Axis(label_format=".0%")),
    )
    .configure_axis(
        domain=False,     # no axis lines
        tick_size=0,      # no tick marks
        grid=False,       # no grid
        label_font_size=11,
    )
    .configure_padding(top=10, right=10, bottom=10, left=10)
    .labs(title="Completion Rate by Category", x=None, y=None)
)

# chart.save("minimal_axes.svg")
