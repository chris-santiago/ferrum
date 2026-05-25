"""Recipe: Main chart with inset detail panel.

Demonstrates using Inset to embed a zoomed-in view of a data cluster
over the parent scatter plot. The inset has its own independent axes
and scales; a connector line links it to the source region.
"""
import polars as pl
import ferrum as fm

rng_state = 42

# 100-point scatter with a dense cluster near (2.0, 3.0)
import random
random.seed(rng_state)

xs = [random.gauss(3.0, 1.5) for _ in range(90)] + [random.gauss(2.0, 0.15) for _ in range(10)]
ys = [random.gauss(3.0, 1.2) for _ in range(90)] + [random.gauss(3.0, 0.12) for _ in range(10)]

df = pl.DataFrame({"x": xs, "y": ys})
cluster_df = df.filter(
    pl.col("x").is_between(1.6, 2.4) & pl.col("y").is_between(2.6, 3.4)
)

zoom = (
    fm.Chart(cluster_df)
    .mark_point(size=80, opacity=0.9)
    .encode(x="x:Q", y="y:Q")
    .configure_axis(label_font_size=9, tick_count=4)
    .configure_padding(top=6, right=6, bottom=6, left=6, auto=False)
)

chart = (
    fm.Chart(df)
    .mark_point(size=40, opacity=0.5)
    .encode(x="x:Q", y="y:Q")
    .labs(title="Scatter with Cluster Detail")
    + fm.Inset(
        chart=zoom,
        bounds=(fm.norm(0.6), fm.norm(0.0), fm.norm(1.0), fm.norm(0.42)),
        connect_to=(2.0, 3.0),
        connect_style="lines",
        shadow=True,
    )
)

# chart.save("dashboard_inset.svg")
