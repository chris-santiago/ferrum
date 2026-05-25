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
        + ann.span("x", 3.0, 4.5, fill="#fee2e2", opacity=0.08, label="Anomalous region")
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
        axis="y", gap=(100, 800)
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
    zoom_df = df.filter(pl.col("x").is_between(1.0, 3.0))
    zoom = fm.Chart(zoom_df).mark_point(size=120).encode(x="x:Q", y="y:Q")
    chart = fm.Chart(df).mark_point(size=50).encode(x="x:Q", y="y:Q") + fm.Inset(
        chart=zoom,
        bounds=(fm.norm(0.55), fm.norm(0.0), fm.norm(1.0), fm.norm(0.5)),
    )
    save_png(chart, f"{BASE}/customizing_inset.png")
    # Also write inset_basic.png — referenced by inset-panels.md basic usage
    save_png(chart, f"{BASE}/inset_basic.png")


def gen_format_presets_revenue():
    """format-presets.md: Revenue chart with currency y and date x.

    Matches the markdown code: ISO date strings parsed to Date, temporal
    encoding, month_year x-axis format, currency y-axis format.
    """
    df = pl.DataFrame(
        {
            "date": ["Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"],
            "revenue": [125000, 138500, 112000, 161000],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="date:N", y="revenue:Q")
        .configure(
            axis_y=fm.AxisConfig(label_format="currency"),
        )
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
    """secondary-axes.md: Color-coded axes example.

    Matches the markdown recipe: conversion_rate field, axis_y2 config with
    percent format, title and y label.
    """
    df = pl.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "revenue": [125_000, 138_500, 112_000, 161_000, 183_000, 172_000],
            "conversion_rate": [0.032, 0.038, 0.029, 0.041, 0.045, 0.043],
        }
    )
    chart = fm.Chart(df).mark_bar(opacity=0.7, color="#1e40af").encode(
        x="month:N", y="revenue:Q"
    ).configure(
        axis_y=fm.AxisConfig(label_format="currency", title_color="#1e40af"),
        axis_y2=fm.AxisConfig(label_format="percent", title_color="#dc2626"),
    ).labs(title="Revenue and Conversion Rate", y="Revenue") + fm.SecondaryY(
        field="conversion_rate",
        mark="line",
        color="#dc2626",
        axis=fm.Axis(title="Conversion Rate"),
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
    """inset-panels.md: Detail zoom pattern.

    Matches the markdown recipe: gauss(3.0, 1.5) + gauss(2.0, 0.15) for x,
    gauss(3.0, 1.2) + gauss(3.0, 0.12) for y, 90+10 samples, zoom filter
    [1.6, 2.4] x [2.6, 3.4], configure_axis and configure_padding on inset.
    """
    import random

    random.seed(42)
    xs = [random.gauss(3.0, 1.5) for _ in range(90)] + [random.gauss(2.0, 0.15) for _ in range(10)]
    ys = [random.gauss(3.0, 1.2) for _ in range(90)] + [random.gauss(3.0, 0.12) for _ in range(10)]
    df = pl.DataFrame({"x": xs, "y": ys})

    cluster_df = df.filter(pl.col("x").is_between(1.6, 2.4) & pl.col("y").is_between(2.6, 3.4))
    zoom = (
        fm.Chart(cluster_df)
        .mark_point(size=80, opacity=0.9)
        .encode(x="x:Q", y="y:Q")
        .configure_axis(label_font_size=9, tick_count=4)
        .configure_padding(top=6, right=6, bottom=6, left=6, auto=False)
    )

    chart = fm.Chart(df).mark_point(size=40, opacity=0.5).encode(x="x:Q", y="y:Q").labs(
        title="Scatter with Cluster Detail"
    ) + fm.Inset(
        chart=zoom,
        bounds=(fm.norm(0.6), fm.norm(0.0), fm.norm(1.0), fm.norm(0.42)),
        connect_to=(2.0, 3.0),
        connect_style="lines",
        shadow=True,
    )
    save_png(chart, f"{BASE}/inset_detail_zoom.png")


def gen_inset_marginal_hist():
    """inset-panels.md: Marginal histogram inset at the bottom edge."""
    import random

    random.seed(99)
    xs = [random.gauss(5, 2) for _ in range(100)]
    ys = [random.gauss(10, 3) for _ in range(100)]
    df = pl.DataFrame({"x": xs, "y": ys})

    hist_df = (
        df.with_columns((pl.col("x") * 2).round(0).truediv(2).alias("x_bin"))
        .group_by("x_bin")
        .agg(pl.len().alias("count"))
        .sort("x_bin")
    )
    hist = (
        fm.Chart(hist_df)
        .mark_bar(opacity=0.5)
        .encode(x="x_bin:Q", y="count:Q")
        .configure_axis(domain=False, grid=False, tick_count=0)
        .configure_padding(top=2, right=2, bottom=2, left=2, auto=False)
    )

    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q") + fm.Inset(
        chart=hist,
        bounds=(fm.norm(0.0), fm.norm(0.78), fm.norm(1.0), fm.norm(1.0)),
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


def gen_format_presets_percent():
    """format-presets.md: Percentage axis example."""
    df = pl.DataFrame(
        {
            "segment": ["Organic", "Paid", "Referral", "Direct"],
            "share": [0.42, 0.28, 0.18, 0.12],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="segment:N", y="share:Q")
        .configure_axis(y=True, x=False, label_format="percent")
    )
    save_png(chart, f"{BASE}/format_presets_percent.png")


def gen_secondary_axes_volume():
    """secondary-axes.md: Volume overlay pattern (price line + volume bars).

    Matches the markdown code with corrected API: label_format instead of
    format on Axis, and .configure() instead of .configure_axis() for axis_x.
    """
    df = pl.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "price": [142.5, 145.2, 138.1, 151.0, 158.3, 153.7],
            "volume": [2300000, 1850000, 3100000, 2650000, 1920000, 2480000],
        }
    )
    chart = fm.Chart(df).mark_line(stroke_width=2).encode(x="month:N", y="price:Q") + fm.SecondaryY(
        field="volume",
        mark="bar",
        color="#94a3b8",
        opacity=0.35,
        axis=fm.Axis(title="Volume", label_format=".2s"),
    )
    save_png(chart, f"{BASE}/secondary_axes_volume.png")


def gen_override_example():
    """override.md: Override example with x_axis_label_angle."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2.1, 3.8, 3.2, 5.1, 4.9],
        }
    )
    chart = (
        fm.Chart(df).mark_point(size=80).encode(x="x:Q", y="y:Q").override(x_axis_label_angle=-30)
    )
    save_png(chart, f"{BASE}/override_example.png")


GENERATORS = [
    ("customizing_cascade", gen_customizing_cascade),
    ("customizing_annotations_example", gen_customizing_annotations_example),
    ("customizing_secondaryy", gen_customizing_secondaryy),
    ("customizing_breakaxis", gen_customizing_breakaxis),
    ("customizing_inset", gen_customizing_inset),
    ("format_presets_revenue", gen_format_presets_revenue),
    ("format_presets_percent", gen_format_presets_percent),
    ("secondary_axes_basic", gen_secondary_axes_basic),
    ("secondary_axes_color_coded", gen_secondary_axes_color_coded),
    ("secondary_axes_volume", gen_secondary_axes_volume),
    ("break_axes_basic", gen_break_axes_basic),
    ("break_axes_horizontal", gen_break_axes_horizontal),
    ("break_axes_comparative", gen_break_axes_comparative),
    ("inset_detail_zoom", gen_inset_detail_zoom),
    ("inset_marginal_hist", gen_inset_marginal_hist),
    ("inset_dashboard", gen_inset_dashboard),
    ("override_example", gen_override_example),
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
