"""Recipe: Apply brand colors via configure and theme.

Demonstrates combining a custom Theme (brand identity: background, typography,
palette) with configure_color (explicit color range) and configure_title to
produce a chart that matches a hypothetical brand guide.
"""
import polars as pl
import ferrum as fm

# Hypothetical brand palette: deep navy, teal, coral, gold
BRAND_COLORS = ["#1a3a5c", "#2a9d8f", "#e76f51", "#e9c46a"]

brand_theme = fm.Theme(
    background="#f8f6f1",
    mark_color=BRAND_COLORS[0],
    font_color="#2d2d2d",
    grid_color="#e8e4da",
    grid=True,
    axis_line=False,
    title_font_weight="bold",
    title_color="#1a3a5c",
)

df = pl.DataFrame({
    "product": ["Core", "Pro", "Enterprise", "Platform"],
    "revenue": [3_200_000, 5_800_000, 4_100_000, 2_700_000],
    "growth": [0.12, 0.28, 0.09, 0.35],
})

chart = (
    fm.Chart(df)
    .mark_bar(corner_radius=3)
    .encode(
        x=fm.X("product:N", sort="-y"),
        y="revenue:Q",
        color="product:N",
    )
    .theme(brand_theme)
    .configure_color(range=BRAND_COLORS)
    .configure_axis(y=True, x=False, label_format="currency")
    .configure_title(anchor="start")
    .configure_legend(orient="none")
    .labs(title="Revenue by Product Line", subtitle="FY 2026", x=None, y="Revenue")
)

# chart.save("branded_chart.svg")
