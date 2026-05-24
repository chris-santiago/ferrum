"""Recipe: Break axis to handle outlier values.

Demonstrates using BreakAxis to keep a large outlier visible without
collapsing the rest of the data into an unreadable band at the bottom.
The break indicator marks the omitted range visually.
"""
import polars as pl
import ferrum as fm
import ferrum.annotation as ann

df = pl.DataFrame({
    "server": ["web-01", "web-02", "web-03", "web-04", "db-01"],
    "response_ms": [42, 38, 45, 1240, 51],
})

chart = (
    fm.Chart(df)
    .mark_bar()
    .encode(
        x=fm.X("server:N", sort=None),
        y="response_ms:Q",
        color=fm.Color(
            "response_ms:Q",
            scale=fm.SequentialScale(scheme="oranges"),
            legend=None,
        ),
    )
    .labs(title="Server Response Times", y="Response Time (ms)")
    + fm.BreakAxis(axis="y", gap=(80, 1180), break_style="zigzag", break_size=14)
    + ann.text(
        fm.norm(0.98), fm.norm(0.98),
        "Note: scale break at 80–1,180 ms",
        font_size=9,
        color="#666",
        anchor="end",
    )
)

# chart.save("break_axis_outlier.svg")
