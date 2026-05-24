"""Recipe: Annotate an outlier point with arrow and text.

Demonstrates using annotation.text and annotation.arrow to call attention
to an unusual data point, with the text positioned offset from the data
coordinate and the arrow connecting text to point.
"""
import polars as pl
import ferrum as fm
import ferrum.annotation as ann

df = pl.DataFrame({
    "x": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    "y": [2.1, 2.8, 3.2, 3.6, 8.9, 4.2, 4.8, 5.1, 5.5, 5.9],
})

# The outlier is at (3.0, 8.9)
chart = (
    fm.Chart(df)
    .mark_point(size=60)
    .encode(x="x:Q", y="y:Q")
    + ann.text(4.0, 7.0, "Sensor fault", color="#c0392b", font_size=14, anchor="start")
    + ann.arrow(3.9, 7.5, 3.1, 8.8, stroke="#c0392b", stroke_width=1.5)
)

# chart.save("annotated_outlier.svg")
