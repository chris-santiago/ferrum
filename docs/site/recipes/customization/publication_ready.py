"""Recipe: Publication-ready chart.

Demonstrates combining the "publication" built-in theme with configure calls
to produce a clean, print-ready figure: no grid, no axis lines, minimal
decoration, left-aligned title, and a source note annotation.
"""
import polars as pl
import ferrum as fm
import ferrum.annotation as ann

df = pl.DataFrame({
    "year": [str(y) for y in range(2015, 2025)],
    "gdp_growth": [3.1, 2.9, 2.4, 2.9, 2.3, -3.4, 5.9, 2.1, 2.5, 2.8],
})

chart = (
    fm.Chart(df)
    .mark_bar()
    .encode(
        x=fm.X("year:N", axis=fm.Axis(label_angle=0)),
        y=fm.Y("gdp_growth:Q", axis=fm.Axis(title="GDP Growth (%)")),
        color=fm.Color(
            "gdp_growth:Q",
            scale=fm.DivergingScale(scheme="rdbu", domain=[-4, 4]),
            legend=None,
        ),
    )
    .theme(fm.themes.publication)
    .configure_axis(domain=False, tick_size=0)
    .configure_title(anchor="start", font_size=14)
    .configure_legend(orient="none")
    .labs(title="U.S. GDP Growth Rate, 2015–2024")
    + ann.text(
        fm.norm(0.0), fm.norm(1.03),
        "Source: Bureau of Economic Analysis",
        font_size=9,
        color="#666",
        anchor="start",
    )
)

# chart.save("publication_ready.svg")
