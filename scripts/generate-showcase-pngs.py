"""Generate showcase PNGs for the "What You Can Build" gallery page.

Each function builds one chart and saves a PNG to docs/site/gallery/showcase_img/.
These charts demonstrate designs unlocked by the flexibility campaign (D1–D9 fixes,
size/shape legends, and temporal auto-inference).

Usage:
    unset CONDA_PREFIX && uv run --no-sync python scripts/generate-showcase-pngs.py
"""

from __future__ import annotations

import datetime
import math
import random
import traceback
from pathlib import Path

import polars as pl

import ferrum as fm

RANDOM_SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "site" / "gallery" / "showcase_img"


def _save(chart: fm.Chart, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    path.write_bytes(chart.show_png())
    print(f"  [OK]  {name}.png ({path.stat().st_size} bytes)")


def build_gapminder_bubble() -> fm.Chart:
    """Gapminder-style bubble: 4 simultaneous encodings — x, y, size, color."""
    df = pl.DataFrame(
        {
            "gdp_per_capita": [
                820.0,
                1500.0,
                2800.0,
                5200.0,
                9800.0,
                14500.0,
                22000.0,
                31000.0,
                42000.0,
                52000.0,
                11000.0,
                7300.0,
            ],
            "life_expectancy": [
                52.0,
                58.0,
                62.0,
                66.0,
                70.0,
                73.0,
                75.0,
                77.0,
                79.0,
                81.0,
                67.0,
                64.0,
            ],
            "population": [
                800.0,
                210.0,
                150.0,
                95.0,
                60.0,
                45.0,
                80.0,
                35.0,
                28.0,
                15.0,
                55.0,
                120.0,
            ],
            "region": [
                "Africa",
                "Africa",
                "Asia",
                "Asia",
                "Asia",
                "Latin America",
                "Europe",
                "Europe",
                "N. America",
                "N. America",
                "Latin America",
                "Asia",
            ],
        }
    )
    return (
        fm.Chart(df)
        .mark_point(opacity=0.85)
        .encode(
            x=fm.X("gdp_per_capita", axis=fm.Axis(label_format="$~s")),
            y=fm.Y("life_expectancy", axis=fm.Axis(label_format="~g")),
            size=fm.Size("population"),
            color=fm.Color("region"),
        )
        .properties(
            title=fm.Title(
                text="Gapminder Bubble Chart",
                subtitle="GDP per capita vs life expectancy — size encodes population, color encodes region",
            ),
            width=520,
            height=360,
        )
    )


def build_highlight_line() -> fm.Chart:
    """Highlight line: many gray context lines + one accent series."""
    rng = random.Random(RANDOM_SEED)
    n = 24
    context_rows: dict[str, list] = {"x": [], "y": [], "series": []}
    highlight_rows: dict[str, list] = {"x": [], "y": []}

    for s in range(9):
        phase = rng.uniform(0, math.pi)
        amp = rng.uniform(0.6, 1.4)
        for i in range(n):
            context_rows["x"].append(i)
            context_rows["y"].append(amp * math.sin(i * 0.28 + phase) + s * 0.15)
            context_rows["series"].append(f"s{s}")

    # Highlighted series: clear uptrend
    for i in range(n):
        highlight_rows["x"].append(i)
        highlight_rows["y"].append(0.04 * i + math.sin(i * 0.28) * 0.4 + 1.2)

    df_ctx = pl.DataFrame(context_rows)
    df_hi = pl.DataFrame(highlight_rows)

    ctx = (
        fm.Chart(df_ctx)
        .mark_line(stroke="#cccccc", opacity=0.55, stroke_width=1)
        .encode(x="x", y="y", detail="series")
    )
    highlight = fm.Chart(df_hi).mark_line(stroke="#e4572e", stroke_width=2.5).encode(x="x", y="y")
    return (ctx + highlight).properties(
        title=fm.Title(
            text="Highlight Line Chart",
            subtitle="Context series faded to gray; one series drawn in accent color to direct attention",
        ),
        width=520,
        height=320,
    )


def build_dumbbell() -> fm.Chart:
    """Dumbbell / connected dot plot: before-and-after per category."""
    categories = [
        "Cloud infrastructure",
        "Data engineering",
        "Machine learning",
        "SQL & analytics",
        "Python / Spark",
        "Visualization",
        "Communication",
    ]
    before = [48, 62, 55, 74, 80, 59, 71]
    after = [67, 58, 72, 69, 85, 74, 68]
    gap = [b - a for a, b in zip(before, after)]
    sorted_cats = [cat for _, cat in sorted(zip(gap, categories))]

    # Both layers use a long-format 'score' column to keep the axis label clean
    df_seg = pl.DataFrame({"category": categories, "score_before": before, "score_after": after})
    df_pts = pl.concat(
        [
            pl.DataFrame(
                {
                    "category": categories,
                    "score": [float(v) for v in before],
                    "period": ["Before"] * len(categories),
                }
            ),
            pl.DataFrame(
                {
                    "category": categories,
                    "score": [float(v) for v in after],
                    "period": ["After"] * len(categories),
                }
            ),
        ]
    )

    ordinal_scale = fm.OrdinalScale(domain=sorted_cats)
    color_scale = fm.OrdinalScale(
        domain=["Before", "After"],
        range=["#4e79a7", "#e4572e"],
    )

    connector = (
        fm.Chart(df_seg)
        .mark_rule(stroke="#cccccc", stroke_width=2)
        .encode(
            x=fm.X("score_before", axis=fm.Axis(title="score")),
            x2=fm.X2("score_after"),
            y=fm.Y("category", scale=ordinal_scale),
        )
    )
    dots = (
        fm.Chart(df_pts)
        .mark_point(size=90, opacity=1.0)
        .encode(
            x="score",
            y=fm.Y("category", scale=ordinal_scale),
            color=fm.Color("period", scale=color_scale),
        )
    )
    return (connector + dots).properties(
        title=fm.Title(
            text="Dumbbell Chart: Skill Scores Before vs After Training",
            subtitle="Each dumbbell shows the change per skill; blue = before, orange = after",
        ),
        width=500,
        height=300,
    )


def build_time_series_events() -> fm.Chart:
    """Time series with formatted date axis, event annotation, and shaded region."""
    months = [datetime.date(2023, m, 1) for m in range(1, 13)]
    sales = [82.0, 78.0, 85.0, 91.0, 94.0, 102.0, 118.0, 115.0, 98.0, 93.0, 87.0, 105.0]
    df = pl.DataFrame({"date": months, "sales": sales})

    line = (
        fm.Chart(df)
        .mark_line(stroke="#2563eb", stroke_width=2)
        .encode(
            x=fm.X("date", axis=fm.Axis(label_format="%b")),
            y=fm.Y("sales"),
        )
    )
    dots = (
        fm.Chart(df)
        .mark_point(fill="#2563eb", size=40)
        .encode(
            x=fm.X("date", axis=fm.Axis(label_format="%b")),
            y="sales",
        )
    )
    # annotate_rect accepts date-string boundaries (D7 + annotate fix)
    anno_band = fm.annotate_rect(
        x1="2023-06-01",
        x2="2023-09-01",
        y1=70.0,
        y2=125.0,
        label="Peak season",
        fill="#fbbf24",
        opacity=0.25,
    )
    anno_line = fm.annotate_vline(x="2023-09-01", label="Product launch")

    return (anno_band + line + dots + anno_line).properties(
        title=fm.Title(
            text="Monthly Sales: 2023",
            subtitle="Yellow band = peak season; vertical rule marks product launch — temporal axis auto-inferred",
        ),
        width=520,
        height=320,
    )


def build_cleveland_dot() -> fm.Chart:
    """Cleveland dot plot (lollipop) sorted by value."""
    skills = [
        "Python",
        "SQL",
        "Machine learning",
        "Statistics",
        "Data engineering",
        "Visualization",
        "Cloud",
        "Communication",
    ]
    scores = [88, 81, 72, 58, 65, 63, 54, 76]
    sorted_domain = [s for _, s in sorted(zip(scores, skills))]

    # Keep both values in a single 'score' column and use a literal for x2=0
    df = pl.DataFrame({"skill": skills, "score": [float(v) for v in scores]})
    # Rule layer: x from zero literal via a helper column named 'score_zero'
    df_rule = df.with_columns(pl.lit(0.0).alias("score_zero"))

    ordinal = fm.OrdinalScale(domain=sorted_domain)

    stem = (
        fm.Chart(df_rule)
        .mark_rule(stroke="#d1d5db", stroke_width=1.5)
        .encode(
            x=fm.X("score_zero", axis=fm.Axis(title="score")),
            x2=fm.X2("score"),
            y=fm.Y("skill", scale=ordinal),
        )
    )
    dot = (
        fm.Chart(df)
        .mark_point(fill="#3b82f6", size=80, opacity=1.0)
        .encode(
            x="score",
            y=fm.Y("skill", scale=ordinal),
        )
    )
    return (stem + dot).properties(
        title=fm.Title(
            text="Cleveland Dot Plot: Survey Skill Scores",
            subtitle="Sorted ascending — each dot is a skill; stem anchors to zero",
        ),
        width=480,
        height=300,
    )


def build_stacked_bar() -> fm.Chart:
    """Stacked bar with categorical color range (D1)."""
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"]
    segments = ["Enterprise", "Mid-market", "SMB", "Consumer"]
    # Revenue shares that sum to 100 per quarter
    shares = [
        [40.0, 30.0, 20.0, 10.0],
        [38.0, 33.0, 22.0, 7.0],
        [35.0, 31.0, 25.0, 9.0],
        [42.0, 28.0, 21.0, 9.0],
    ]
    data: dict[str, list] = {"quarter": [], "segment": [], "share": []}
    for i, q in enumerate(quarters):
        for j, s in enumerate(segments):
            data["quarter"].append(q)
            data["segment"].append(s)
            data["share"].append(shares[i][j])

    df = pl.DataFrame(data)
    colors = ["#1e3a5f", "#2563eb", "#60a5fa", "#bfdbfe"]

    return (
        fm.Chart(df)
        .mark_bar(position=fm.Stack())
        .encode(
            x=fm.X("quarter", scale=fm.OrdinalScale(domain=quarters)),
            y=fm.Y("share"),
            color=fm.Color("segment", scale=fm.OrdinalScale(domain=segments, range=colors)),
        )
        .properties(
            title=fm.Title(
                text="Revenue Mix by Quarter",
                subtitle="Stacked bars with a four-color categorical range — enterprise dominates each quarter",
            ),
            width=440,
            height=320,
        )
    )


def build_correlation_heatmap() -> fm.Chart:
    """Correlation heatmap with RdBu diverging scheme."""
    features = ["Price", "Volume", "PE_Ratio", "ROE", "Debt_Eq"]
    corr = {
        "feature": features,
        "Price": [1.0, 0.7, -0.3, 0.4, -0.6],
        "Volume": [0.7, 1.0, -0.2, 0.3, -0.5],
        "PE_Ratio": [-0.3, -0.2, 1.0, -0.6, 0.2],
        "ROE": [0.4, 0.3, -0.6, 1.0, -0.7],
        "Debt_Eq": [-0.6, -0.5, 0.2, -0.7, 1.0],
    }
    df = pl.DataFrame(corr)
    return fm.heatmap(df, cmap="rdbu", center=0, annot=True, fmt=".2f").properties(
        title=fm.Title(
            text="Correlation Heatmap",
            subtitle="RdBu diverging colormap — blue = positive, red = negative; values annotated in each cell",
        ),
        width=420,
        height=360,
    )


def build_ridgeline() -> fm.Chart:
    """Build grouped KDE density chart: overlapping fills per group."""
    rng = random.Random(RANDOM_SEED)
    groups = ["Control", "Treatment A", "Treatment B", "Treatment C"]
    means = [5.0, 6.5, 4.2, 7.8]
    stds = [1.2, 1.0, 1.5, 0.9]

    rows: dict[str, list] = {"group": [], "value": []}
    for g, mu, sigma in zip(groups, means, stds):
        for _ in range(300):
            rows["group"].append(g)
            rows["value"].append(mu + rng.gauss(0, sigma))

    df = pl.DataFrame(rows)
    # groupby= on mark_density computes a per-group KDE — each group gets its
    # own distribution, clearly separated along the x axis.
    return (
        fm.Chart(df)
        .mark_density(groupby="group")
        .encode(
            x=fm.X("value"),
            color=fm.Color("group"),
        )
        .properties(
            title=fm.Title(
                text="Distribution Comparison — Grouped KDE",
                subtitle="Per-group KDE fills show shift in location and spread across four conditions",
            ),
            width=520,
            height=340,
        )
    )


def build_hexbin() -> fm.Chart:
    """Bivariate density via hex binning."""
    rng = random.Random(RANDOM_SEED)
    rows: dict[str, list[float]] = {"x": [], "y": []}
    for _ in range(800):
        x = rng.gauss(0, 1)
        y = 0.65 * x + rng.gauss(0, 0.75)
        rows["x"].append(x)
        rows["y"].append(y)

    df = pl.DataFrame(rows)
    return (
        fm.Chart(df)
        .mark_hex()
        .encode(x="x", y="y")
        .properties(
            title=fm.Title(
                text="Bivariate Density — Hex Binning",
                subtitle="800 observations; hex bins aggregate point density — no overplotting at scale",
            ),
            width=420,
            height=360,
        )
    )


def build_faceted_scatter() -> fm.Chart:
    """Faceted small-multiples scatter for species-level comparison."""
    rng = random.Random(RANDOM_SEED)
    species_order = ["Setosa", "Versicolor", "Virginica"]
    species_params = {
        "Setosa": (1.5, 3.4, 0.25, 0.35),
        "Versicolor": (4.3, 2.8, 0.45, 0.28),
        "Virginica": (5.5, 3.0, 0.55, 0.32),
    }
    rows: dict[str, list] = {"petal_length": [], "sepal_width": [], "species": []}
    for sp, (ml, mw, sl, sw) in species_params.items():
        for _ in range(50):
            rows["petal_length"].append(ml + rng.gauss(0, sl))
            rows["sepal_width"].append(mw + rng.gauss(0, sw))
            rows["species"].append(sp)

    df = pl.DataFrame(rows)
    # Explicit OrdinalScale domain+range pins each species to a consistent color
    # across all facet panels; without it, per-panel color scale collapses to
    # the first theme color for each single-category subset.
    colors = ["#4e79a7", "#e15759", "#76b7b2"]
    color_scale = fm.OrdinalScale(domain=species_order, range=colors)

    return (
        fm.Chart(df)
        .mark_point(opacity=0.65, size=45)
        .encode(
            x="petal_length",
            y="sepal_width",
            color=fm.Color("species", scale=color_scale),
        )
        .facet(col="species")
        .properties(
            title=fm.Title(
                text="Iris: Petal Length vs Sepal Width by Species",
                subtitle="Small multiples — explicit color scale keeps each species consistently colored across panels",
            ),
        )
    )


CHARTS: list[tuple[str, object]] = [
    ("showcase_gapminder_bubble", build_gapminder_bubble),
    ("showcase_highlight_line", build_highlight_line),
    ("showcase_dumbbell", build_dumbbell),
    ("showcase_time_series", build_time_series_events),
    ("showcase_cleveland_dot", build_cleveland_dot),
    ("showcase_stacked_bar", build_stacked_bar),
    ("showcase_correlation_heatmap", build_correlation_heatmap),
    ("showcase_ridgeline", build_ridgeline),
    ("showcase_hexbin", build_hexbin),
    ("showcase_faceted_scatter", build_faceted_scatter),
]


def main() -> None:
    """Generate all showcase PNGs and report success/failure counts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    total = len(CHARTS)

    print(f"\nGenerating {total} showcase PNGs → {OUT_DIR}\n")
    for name, builder in CHARTS:
        try:
            chart = builder()
            _save(chart, name)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    print(f"\n{ok}/{total} showcase PNGs generated successfully.")


if __name__ == "__main__":
    main()
