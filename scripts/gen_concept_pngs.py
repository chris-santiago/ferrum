"""Generate PNG visuals for all docs concept pages."""

import traceback
import ferrum as fm
import ferrum.annotation as ann
import polars as pl


def save_png(chart, path):
    """Render chart to PNG using ferrum's built-in renderer at 2x scale."""
    png_data = chart.show_png(scale=2.0)
    with open(path, "wb") as f:
        f.write(png_data)
    print(f"  OK: {path} ({len(png_data)} bytes)")


BASE = "/Users/chrissantiago/Dropbox/GitHub/ferrum/docs/site/assets/concepts"
RECIPES = "/Users/chrissantiago/Dropbox/GitHub/ferrum/docs/site/assets/recipes"


def gen_customizing_cascade():
    """customizing-charts.md: config cascade example (month/revenue bar)."""
    df = pl.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "revenue": [12000, 15400, 11200, 18600, 21000, 19500],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(
            x="month:N",
            y=fm.Y("revenue:Q", axis=fm.Axis(label_format="$,.0f")),
        )
        .configure_axis(label_angle=-30)
    )
    save_png(chart, f"{BASE}/customizing_cascade.png")


def gen_customizing_annotations_example():
    """customizing-charts.md: annotations example with text+arrow+span."""
    df = pl.DataFrame(
        {
            "x": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "y": [2.1, 3.0, 3.5, 4.0, 4.5, 8.2, 5.0, 5.5, 6.0, 6.5],
        }
    )
    chart = (
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        + ann.text(3.5, 8.2, "Anomaly", color="#c0392b", font_size=13)
        + ann.arrow(3.5, 8.0, 3.5, 6.5)
        + ann.span("x", 3.0, 4.5, fill="#fee2e2", opacity=0.2, label="Anomalous region")
    )
    save_png(chart, f"{BASE}/customizing_annotations_example.png")


def gen_customizing_secondaryy():
    """customizing-charts.md: SecondaryY example."""
    df = pl.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "revenue": [125000, 138500, 112000, 161000, 183000, 172000],
            "growth_rate": [0.0, 0.107, -0.191, 0.438, 0.137, -0.066],
        }
    )
    chart = fm.Chart(df).mark_bar().encode(x="month:N", y="revenue:Q") + fm.SecondaryY(
        field="growth_rate", mark="line", color="#e74c3c"
    )
    save_png(chart, f"{BASE}/customizing_secondaryy.png")


def gen_customizing_breakaxis():
    """customizing-charts.md: BreakAxis example."""
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [72, 68, 75, 900, 81],
        }
    )
    chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q") + fm.BreakAxis(
        axis="y", gap=(150, 900)
    )
    save_png(chart, f"{BASE}/customizing_breakaxis.png")


def gen_customizing_inset():
    """customizing-charts.md: Inset example."""
    df = pl.DataFrame(
        {
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "y": [2.1, 2.8, 3.5, 4.1, 4.8, 5.2, 5.8, 6.3, 6.9, 7.4],
        }
    )
    zoom_df = df.filter(pl.col("x").is_between(1.0, 2.0))
    zoom = fm.Chart(zoom_df).mark_point(size=60).encode(x="x:Q", y="y:Q")
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q") + fm.Inset(
        chart=zoom,
        bounds=(fm.norm(0.6), fm.norm(0.0), fm.norm(1.0), fm.norm(0.45)),
    )
    save_png(chart, f"{BASE}/customizing_inset.png")


def gen_format_presets_revenue():
    """format-presets.md: Revenue chart with currency y and date x (FIXED).

    The docs code example uses :T (temporal) encoding. Temporal bar charts
    generate many auto-ticks that get elided. We render as :O (ordinal) with
    pre-formatted month labels so the visual is clean and representative.
    """
    df = pl.DataFrame(
        {
            "date": ["Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"],
            "revenue": [125000, 138500, 112000, 161000],
        }
    )
    chart = (
        fm.Chart(df, width=700, height=400)
        .mark_bar()
        .encode(
            x=fm.X("date:N", sort=None),
            y="revenue:Q",
        )
        .configure(
            axis_x=fm.AxisConfig(label_angle=0),
            axis_y=fm.AxisConfig(label_format="currency"),
        )
        .configure_padding(left=80, right=40, bottom=50)
        .labs(x="date", y="revenue")
    )
    save_png(chart, f"{BASE}/format_presets_revenue.png")


def gen_secondary_axes_basic():
    """secondary-axes.md: Basic usage example."""
    df = pl.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "revenue": [125000, 138500, 112000, 161000, 183000, 172000],
            "growth_rate": [0.0, 0.107, -0.191, 0.438, 0.137, -0.066],
        }
    )
    chart = fm.Chart(df).mark_bar().encode(x="month:N", y="revenue:Q").labs(
        title="Revenue and Month-over-Month Growth"
    ) + fm.SecondaryY(field="growth_rate", mark="line", color="#e74c3c")
    save_png(chart, f"{BASE}/secondary_axes_basic.png")


def gen_secondary_axes_color_coded():
    """secondary-axes.md: Color-coded axes example."""
    df = pl.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "revenue": [125000, 138500, 112000, 161000, 183000, 172000],
            "growth_rate": [0.0, 0.107, -0.191, 0.438, 0.137, -0.066],
        }
    )
    chart = fm.Chart(df).mark_bar(color="#1e40af", opacity=0.8).encode(
        x="month:N", y="revenue:Q"
    ).configure(
        axis_y=fm.AxisConfig(title_color="#1e40af"),
    ) + fm.SecondaryY(
        field="growth_rate",
        mark="line",
        color="#dc2626",
        axis=fm.Axis(title="Growth Rate", title_color="#dc2626"),
    )
    save_png(chart, f"{BASE}/secondary_axes_color_coded.png")


def gen_break_axes_basic():
    """break-axes.md: Basic usage example."""
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [72, 68, 75, 900, 81],
        }
    )
    chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q") + fm.BreakAxis(
        axis="y", gap=(150, 850)
    )
    save_png(chart, f"{BASE}/break_axes_basic.png")


def gen_break_axes_horizontal():
    """break-axes.md: Horizontal break axis example."""
    df = pl.DataFrame(
        {
            "group": ["Control"] * 10 + ["Treatment"] * 10,
            "measurement": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                1,
                2,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
            ],
        }
    )
    chart = fm.Chart(df).mark_point().encode(
        x="measurement:Q", y="group:N", color="group:N"
    ) + fm.BreakAxis(axis="x", gap=(12, 48))
    save_png(chart, f"{BASE}/break_axes_horizontal.png")


def gen_break_axes_comparative():
    """break-axes.md: Comparative bar chart with outlier suppression."""
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E", "F"],
            "value": [85, 92, 78, 510, 88, 95],
        }
    )
    chart = (
        fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q", color="category:N")
        + fm.BreakAxis(axis="y", gap=(120, 480), break_style="zigzag", break_size=16)
        + ann.text(
            fm.norm(0.98),
            fm.norm(0.98),
            "Note: scale break at 120–480",
            font_size=9,
            color="#666",
            anchor="end",
        )
    )
    save_png(chart, f"{BASE}/break_axes_comparative.png")


def gen_inset_detail_zoom():
    """inset-panels.md: Detail zoom pattern."""
    import random

    random.seed(42)
    xs = [random.gauss(5, 2) for _ in range(80)] + [random.gauss(2.5, 0.3) for _ in range(20)]
    ys = [random.gauss(50, 15) for _ in range(80)] + [random.gauss(50, 5) for _ in range(20)]
    df = pl.DataFrame({"x": xs, "y": ys})

    zoom_df = df.filter(pl.col("x").is_between(2.0, 3.0) & pl.col("y").is_between(40, 60))
    zoom = fm.Chart(zoom_df).mark_point(size=80, opacity=0.9).encode(x="x:Q", y="y:Q")

    chart = fm.Chart(df).mark_point(opacity=0.4).encode(x="x:Q", y="y:Q") + fm.Inset(
        chart=zoom,
        bounds=(fm.norm(0.55), fm.norm(0.0), fm.norm(1.0), fm.norm(0.42)),
        connect_to=(2.5, 50),
        connect_style="lines",
        shadow=True,
    )
    save_png(chart, f"{BASE}/inset_detail_zoom.png")


def gen_inset_marginal_hist():
    """inset-panels.md: Marginal histogram inset."""
    import random

    random.seed(99)
    xs = [random.gauss(5, 2) for _ in range(100)]
    ys = [random.gauss(10, 3) for _ in range(100)]
    df = pl.DataFrame({"x": xs, "y": ys})

    # Pre-compute the histogram for the inset since bin+count inside Inset
    # hits a transform issue with groupby column resolution.
    hist_df = (
        df.with_columns(pl.col("x").round(0).alias("x_bin"))
        .group_by("x_bin")
        .agg(pl.len().alias("count"))
        .sort("x_bin")
    )
    hist = (
        fm.Chart(hist_df)
        .mark_bar(opacity=0.6)
        .encode(x="x_bin:Q", y="count:Q")
        .configure_axis(domain=False, grid=False, tick_count=0)
        .configure_padding(top=2, right=2, bottom=2, left=2, auto=False)
    )

    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q") + fm.Inset(
        chart=hist,
        bounds=(fm.norm(0.0), fm.norm(0.0), fm.norm(1.0), fm.norm(0.22)),
        border=False,
        background=None,
    )
    save_png(chart, f"{BASE}/inset_marginal_hist.png")


def gen_inset_dashboard():
    """inset-panels.md: Dashboard card with summary inset (sparkline)."""
    import datetime

    dates = pl.date_range(datetime.date(2025, 1, 1), datetime.date(2025, 12, 1), "1mo", eager=True)
    metrics = [100, 112, 108, 125, 130, 128, 145, 150, 162, 170, 178, 190]
    df = pl.DataFrame({"date": dates, "metric": metrics})
    recent_df = df.tail(4)

    sparkline = (
        fm.Chart(recent_df)
        .mark_line(stroke_width=1.5, color="#16a34a")
        .encode(x="date:T", y="metric:Q")
        .configure_axis(domain=False, grid=False, tick_count=0)
        .configure_padding(top=2, right=2, bottom=2, left=2, auto=False)
    )

    chart = fm.Chart(df).mark_area(opacity=0.3).encode(x="date:T", y="metric:Q") + fm.Inset(
        chart=sparkline,
        bounds=(fm.norm(0.7), fm.norm(0.0), fm.norm(1.0), fm.norm(0.3)),
        border_dash=[3, 3],
        shadow=False,
    )
    save_png(chart, f"{BASE}/inset_dashboard.png")


GENERATORS = [
    ("customizing_cascade", gen_customizing_cascade),
    ("customizing_annotations_example", gen_customizing_annotations_example),
    ("customizing_secondaryy", gen_customizing_secondaryy),
    ("customizing_breakaxis", gen_customizing_breakaxis),
    ("customizing_inset", gen_customizing_inset),
    ("format_presets_revenue (FIX)", gen_format_presets_revenue),
    ("secondary_axes_basic", gen_secondary_axes_basic),
    ("secondary_axes_color_coded", gen_secondary_axes_color_coded),
    ("break_axes_basic", gen_break_axes_basic),
    ("break_axes_horizontal", gen_break_axes_horizontal),
    ("break_axes_comparative", gen_break_axes_comparative),
    ("inset_detail_zoom", gen_inset_detail_zoom),
    ("inset_marginal_hist", gen_inset_marginal_hist),
    ("inset_dashboard", gen_inset_dashboard),
]


def main():
    """Run all PNG generators and report results."""
    succeeded = []
    failed = []
    for name, gen_fn in GENERATORS:
        print(f"\n--- {name} ---")
        try:
            gen_fn()
            succeeded.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'=' * 60}")
    print(f"Succeeded: {len(succeeded)}/{len(GENERATORS)}")
    for name in succeeded:
        print(f"  OK  {name}")
    if failed:
        print(f"Failed: {len(failed)}/{len(GENERATORS)}")
        for name in failed:
            print(f"  FAIL  {name}")


if __name__ == "__main__":
    main()
