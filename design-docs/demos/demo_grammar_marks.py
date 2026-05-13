"""Grammar Marks & Compositions gallery — Paper Ink theme.

Renders every implemented grammar-level mark and composition stress tests,
saves each as a PNG, and generates a single HTML gallery document.

Usage:
    uv run design-docs/themes/demo_grammar_marks.py
"""

from __future__ import annotations

import base64
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import ferrum as fm

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PNG_DIR = HERE / "demo_grammar_pngs"
HTML_OUT = HERE / "demo_grammar_marks.html"
PNG_DIR.mkdir(parents=True, exist_ok=True)

THEME = fm.themes.paper_ink
W, H = 580, 400

# ── synthetic data ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# Multi-series time-series
n_months = 24
groups = ["Product A", "Product B", "Product C"]
rows_ts = []
for g in groups:
    base = rng.uniform(20, 80)
    noise = rng.normal(0, 4, n_months).cumsum()
    for i, v in enumerate(base + noise):
        rows_ts.append({"month": i, "value": float(v), "group": g})
df_lines = pl.DataFrame(rows_ts)

# Scatter 3-class
n_sc = 180
scatter_x = np.concatenate(
    [rng.normal(-1.5, 0.8, 60), rng.normal(1.5, 0.8, 60), rng.normal(0, 0.8, 60)]
)
scatter_y = np.concatenate(
    [rng.normal(-1.5, 0.8, 60), rng.normal(1.5, 0.8, 60), rng.normal(0, 0.8, 60)]
)
scatter_cls = ["Alpha"] * 60 + ["Beta"] * 60 + ["Gamma"] * 60
df_scatter = pl.DataFrame({"x": scatter_x, "y": scatter_y, "class": scatter_cls})

# Bar chart
categories = ["Q1", "Q2", "Q3", "Q4"]
bar_groups = ["North", "South", "West"]
rows_bar = [
    {"quarter": cat, "region": bg, "sales": float(rng.integers(40, 120))}
    for cat in categories
    for bg in bar_groups
]
df_bar = pl.DataFrame(rows_bar)

# Area data
n_area = 50
area_x = np.linspace(0, 10, n_area)
rows_area = []
for g in ["Series A", "Series B", "Series C"]:
    vals = rng.uniform(5, 30, n_area) + np.sin(area_x) * 5
    for x, v in zip(area_x, vals):
        rows_area.append({"x": float(x), "value": float(v), "series": g})
df_area = pl.DataFrame(rows_area)

# Distribution / boxplot groups
box_groups = ["Control", "Treatment A", "Treatment B", "Treatment C"]
rows_box = [
    {"group": g, "value": float(rng.normal(mu, 1.0))}
    for g, mu in zip(box_groups, [0.0, 0.8, -0.4, 1.2])
    for _ in range(80)
]
df_box = pl.DataFrame(rows_box)

# Hex / contour data
hex_x = rng.normal(0, 1, 500)
hex_y = hex_x * 0.7 + rng.normal(0, 0.5, 500)
df_hex = pl.DataFrame({"x": hex_x, "y": hex_y})

# Two-variable data for smooth / errorband
smooth_x = rng.uniform(0, 10, 120)
smooth_y = 2 * smooth_x + rng.normal(0, 3, 120)
df_smooth = pl.DataFrame({"x": smooth_x, "y": smooth_y})

# Two-variable data with groups (for grouped smooth)
rows_grouped = []
for g in ["Alpha", "Beta", "Gamma"]:
    slope = {"Alpha": 1.5, "Beta": 2.5, "Gamma": 0.8}[g]
    gx = rng.uniform(0, 10, 60)
    gy = slope * gx + rng.normal(0, 2, 60)
    for x, y in zip(gx, gy):
        rows_grouped.append({"x": float(x), "y": float(y), "group": g})
df_grouped = pl.DataFrame(rows_grouped)

# Text labels
df_text = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 1.5, 3.5, 2.5],
        "label": ["A", "B", "C", "D", "E"],
    }
)

# Tick / strip
df_tick = df_box

# Histogram groups
hist_groups = ["Group A", "Group B"]
rows_hist = [
    {"value": float(rng.normal(0.0 if g == "Group A" else 2.0, 1.0)), "group": g}
    for g in hist_groups
    for _ in range(150)
]
df_hist = pl.DataFrame(rows_hist)

# Errorbar data (pre-aggregated)
eb_groups = ["A", "B", "C", "D"]
eb_means = [2.5, 3.8, 1.9, 4.2]
eb_errs = [0.3, 0.5, 0.4, 0.6]
df_eb = pl.DataFrame(
    {
        "group": eb_groups,
        "mean": eb_means,
        "err": eb_errs,
        "lower": [m - e for m, e in zip(eb_means, eb_errs)],
        "upper": [m + e for m, e in zip(eb_means, eb_errs)],
    }
)

# Violin / KDE data
df_violin = df_box

# QQ data
qq_vals = rng.normal(0, 1, 200)
df_qq = pl.DataFrame({"value": qq_vals})

# Raster (2D density — concentrated bivariate normal)
raster_xy = np.vstack([
    rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 1.0]], 800),
    rng.multivariate_normal([3, 3], [[0.5, -0.3], [-0.3, 0.5]], 400),
])
df_raster = pl.DataFrame({"x": raster_xy[:, 0], "y": raster_xy[:, 1]})

# Swarm
df_swarm = pl.DataFrame(
    {
        "group": (["A"] * 40 + ["B"] * 40 + ["C"] * 40),
        "value": list(rng.normal(0, 1, 40))
        + list(rng.normal(1, 0.8, 40))
        + list(rng.normal(-0.5, 1.2, 40)),
    }
)

# Segment
df_segment = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.5, 1.5, 3.0],
        "x2": [1.5, 2.5, 3.5, 4.5],
        "y2": [2.0, 3.5, 2.5, 4.0],
    }
)

# Boxen
df_boxen = df_box

# Ribbon (continuous x, sin wave +/- 0.5)
x_ribbon = np.linspace(0, 4 * np.pi, 100)
df_ribbon = pl.DataFrame(
    {
        "x": x_ribbon,
        "lower": np.sin(x_ribbon) - 0.5,
        "upper": np.sin(x_ribbon) + 0.5,
    }
)

# Rect (5x5 heatmap-style grid)
rect_rows = [
    {"x": str(i), "y": str(j), "value": float(rng.uniform(0, 1))}
    for i in range(5)
    for j in range(5)
]
df_rect = pl.DataFrame(rect_rows)

# Facet data
facet_categories = ["Low", "Medium", "High"]
rows_facet = []
for fc in facet_categories:
    xv = rng.uniform(0, 10, 60)
    slope = {"Low": 0.3, "Medium": 0.7, "High": 1.2}[fc]
    yv = slope * xv + rng.normal(0, 0.8, 60)
    for x, y in zip(xv, yv):
        rows_facet.append({"exposure": float(x), "response": float(y), "dose": fc})
df_facet = pl.DataFrame(rows_facet)


# ── helpers ───────────────────────────────────────────────────────────────────


def try_render(name: str, fn) -> tuple[str, bytes | None, str | None]:
    """Run fn(), return (name, png_bytes_or_None, error_or_None)."""
    try:
        result = fn()
        if isinstance(result, bytes):
            png = result
        else:
            png = result.show_png()
        return name, png, None
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  [FAIL] {name}: {exc}")
        return name, None, f"{exc}\n{tb}"


# ── Grammar-level marks ──────────────────────────────────────────────────────


def build_mark_point():
    return (
        fm.Chart(df_scatter)
        .mark_point()
        .encode(
            x=fm.X("x", title="X"), y=fm.Y("y", title="Y"), color=fm.Color("class", title="Class")
        )
        .properties(title="mark_point() — Scatter", width=W, height=H)
        .theme(THEME)
    )


def build_mark_line():
    return (
        fm.Chart(df_lines)
        .mark_line()
        .encode(
            x=fm.X("month", title="Month"),
            y=fm.Y("value", title="Value"),
            color=fm.Color("group", title="Group"),
        )
        .properties(title="mark_line() — Multi-Series Line", width=W, height=H)
        .theme(THEME)
    )


def build_mark_bar():
    return (
        fm.Chart(df_bar)
        .mark_bar(position=fm.Dodge())
        .encode(
            x=fm.X("quarter", title="Quarter"),
            y=fm.Y("sales", title="Sales"),
            color=fm.Color("region", title="Region"),
        )
        .properties(title="mark_bar() — Grouped Bar", width=W, height=H)
        .theme(THEME)
    )


def build_mark_area():
    return (
        fm.Chart(df_area)
        .mark_area(position=fm.Stack())
        .encode(
            x=fm.X("x", title="X"),
            y=fm.Y("value", title="Value"),
            color=fm.Color("series", title="Series"),
        )
        .properties(title="mark_area() — Stacked Area", width=W, height=H)
        .theme(THEME)
    )


def build_mark_boxplot():
    return (
        fm.Chart(df_box)
        .mark_boxplot()
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="mark_boxplot() — Box Plot", width=W, height=H)
        .theme(THEME)
    )


def build_mark_violin():
    return (
        fm.Chart(df_violin)
        .mark_violin()
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="mark_violin() — Violin Plot", width=W, height=H)
        .theme(THEME)
    )


def build_mark_histogram():
    return (
        fm.Chart(df_hist)
        .mark_histogram(groupby="group")
        .encode(
            x=fm.X("value", title="Value"),
            y=fm.Y("count", title="Count"),
            color=fm.Color("group", title="Group"),
        )
        .properties(title="mark_histogram() — Histogram", width=W, height=H)
        .theme(THEME)
    )


def build_mark_density():
    return (
        fm.Chart(df_hist)
        .mark_density(groupby="group")
        .encode(x=fm.X("value", title="Value"), color=fm.Color("group", title="Group"))
        .properties(title="mark_density() — KDE Density", width=W, height=H)
        .theme(THEME)
    )


def build_mark_hex():
    return (
        fm.Chart(df_hex)
        .mark_hex()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="mark_hex() — Hexbin", width=W, height=H)
        .theme(THEME)
    )


def build_mark_text():
    return (
        fm.Chart(df_text)
        .mark_text()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"), text=fm.Text("label"))
        .properties(title="mark_text() — Text Labels", width=W, height=H)
        .theme(THEME)
    )


def build_mark_tick():
    return (
        fm.Chart(df_tick)
        .mark_tick()
        .encode(x=fm.X("value", title="Value"), y=fm.Y("group", title="Group"))
        .properties(title="mark_tick() — Strip/Tick Plot", width=W, height=H)
        .theme(THEME)
    )


def build_mark_errorbar():
    return (
        fm.Chart(df_box)
        .mark_errorbar(extent="stdev")
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="mark_errorbar() — Error Bars", width=W, height=H)
        .theme(THEME)
    )


def build_mark_smooth_ci():
    return (
        fm.Chart(df_smooth)
        .mark_smooth(ci=0.95)
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="mark_smooth(ci=0.95) — LOESS + CI Band", width=W, height=H)
        .theme(THEME)
    )


def build_mark_smooth_lm():
    return (
        fm.Chart(df_smooth)
        .mark_smooth(method="lm")
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="mark_smooth(method='lm') — Linear Smooth", width=W, height=H)
        .theme(THEME)
    )


def build_mark_contour():
    return (
        fm.Chart(df_hex)
        .mark_contour()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="mark_contour() — 2D Contour", width=W, height=H)
        .theme(THEME)
    )


def build_mark_raster():
    return (
        fm.Chart(df_raster)
        .mark_raster()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="mark_raster() — Raster Heatmap", width=W, height=H)
        .theme(THEME)
    )


def build_mark_swarm():
    return (
        fm.Chart(df_swarm)
        .mark_swarm()
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="mark_swarm() — Beeswarm Plot", width=W, height=H)
        .theme(THEME)
    )


def build_mark_boxen():
    return (
        fm.Chart(df_boxen)
        .mark_boxen()
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="mark_boxen() — Letter-Value (Boxen)", width=W, height=H)
        .theme(THEME)
    )


def build_mark_qq():
    return (
        fm.Chart(df_qq)
        .mark_qq()
        .encode(x=fm.X("value", title="Theoretical"))
        .properties(title="mark_qq() — QQ Plot", width=W, height=H)
        .theme(THEME)
    )


def build_mark_ribbon():
    return (
        fm.Chart(df_ribbon)
        .mark_ribbon()
        .encode(x=fm.X("x", title="X"), y=fm.Y("lower", title="Y"), y2=fm.Y2("upper"))
        .properties(title="mark_ribbon() — Ribbon Band", width=W, height=H)
        .theme(THEME)
    )


def build_mark_segment():
    return (
        fm.Chart(df_segment)
        .mark_segment()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"), x2=fm.X2("x2"), y2=fm.Y2("y2"))
        .properties(title="mark_segment() — Segment Lines", width=W, height=H)
        .theme(THEME)
    )


def build_mark_rule():
    from ferrum.layer import Layer

    mean_y = float(np.mean(smooth_y))
    df_rule = df_smooth.with_columns(pl.lit(mean_y).alias("_ref_y"))
    return (
        fm.Chart(df_rule)
        .mark_point(opacity=0.4)
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .layer(Layer(mark="rule", encoding={"y": "_ref_y"}, mark_kwargs={"stroke_dash": [4, 2]}))
        .properties(title="mark_rule() — Reference Line Overlay", width=W, height=H)
        .theme(THEME)
    )


def build_mark_rect():
    return (
        fm.Chart(df_rect)
        .mark_rect()
        .encode(x=fm.X("x"), y=fm.Y("y"), color=fm.Color("value"))
        .properties(title="mark_rect() — Heatmap Grid", width=W, height=H)
        .theme(THEME)
    )


# ── Compositions & Overlays ──────────────────────────────────────────────────


def build_comp_scatter_rule():
    """Scatter + Rule — mark_point scatter + mark_rule horizontal reference (same data)."""
    mean_y = float(np.mean(scatter_y))
    df_with_ref = df_scatter.with_columns(pl.lit(mean_y).alias("ref_y"))
    scatter = (
        fm.Chart(df_with_ref)
        .mark_point(opacity=0.5)
        .encode(
            x=fm.X("x", title="X"),
            y=fm.Y("y", title="Y"),
            color=fm.Color("class", title="Class"),
        )
        .properties(width=W, height=H)
        .theme(THEME)
    )
    rule = fm.Chart(df_with_ref).mark_rule().encode(y=fm.Y("ref_y")).theme(THEME)
    return (scatter + rule).properties(title="Scatter + Rule — reference at mean y")


def build_comp_scatter_smooth_ci():
    """Scatter + Smooth CI — prior_mark detection."""
    return (
        fm.Chart(df_smooth)
        .mark_point(opacity=0.3)
        .mark_smooth(ci=0.95)
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="Scatter + Smooth CI — prior_mark detection", width=W, height=H)
        .theme(THEME)
    )


def build_comp_grouped_smooth():
    """Grouped Smooth — mark_smooth(method='lm', groupby='group') with color grouping."""
    return (
        fm.Chart(df_grouped)
        .mark_smooth(method="lm", groupby="group")
        .encode(
            x=fm.X("x", title="X"),
            y=fm.Y("y", title="Y"),
            color=fm.Color("group", title="Group"),
        )
        .properties(title="Grouped Smooth — LM by color group", width=W, height=H)
        .theme(THEME)
    )


def build_comp_dodge_bar_errorbar():
    """Dodge Bar + Errorbar overlay."""
    bars = (
        fm.Chart(df_bar)
        .mark_bar(position=fm.Dodge())
        .encode(
            x=fm.X("quarter", title="Quarter"),
            y=fm.Y("sales", title="Sales"),
            color=fm.Color("region", title="Region"),
        )
        .properties(width=W, height=H)
        .theme(THEME)
    )
    errorbars = (
        fm.Chart(df_bar)
        .mark_errorbar(extent="stdev")
        .encode(x=fm.X("quarter"), y=fm.Y("sales"))
        .theme(THEME)
    )
    return (bars + errorbars).properties(title="Dodge Bar + Errorbar Overlay")


def build_comp_line_ribbon():
    """Line + Ribbon — sin wave with +/-0.5 band."""
    mid_vals = np.sin(x_ribbon)
    df_lr = pl.DataFrame(
        {
            "x": x_ribbon,
            "mid": mid_vals,
            "lower": mid_vals - 0.5,
            "upper": mid_vals + 0.5,
        }
    )
    ribbon = (
        fm.Chart(df_lr)
        .mark_ribbon()
        .encode(x=fm.X("x", title="X"), y=fm.Y("lower", title="Y"), y2=fm.Y2("upper"))
        .properties(width=W, height=H)
        .theme(THEME)
    )
    line = (
        fm.Chart(df_lr)
        .mark_line()
        .encode(x=fm.X("x"), y=fm.Y("mid"))
        .theme(THEME)
    )
    return (ribbon + line).properties(title="Line + Ribbon — sin wave band")


def build_comp_histogram_density():
    """Histogram + Density — same data, two layers."""
    hist = (
        fm.Chart(df_hist)
        .mark_histogram()
        .encode(x=fm.X("value", title="Value"), y=fm.Y("count", title="Count"))
        .properties(width=W, height=H)
        .theme(THEME)
    )
    density = (
        fm.Chart(df_hist)
        .mark_density()
        .encode(x=fm.X("value"))
        .theme(THEME)
    )
    return (hist + density).properties(title="Histogram + Density Overlay")


def build_comp_boxplot_swarm():
    """Boxplot + Swarm — boxplot with jittered scatter overlay."""
    box = (
        fm.Chart(df_swarm)
        .mark_boxplot()
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(width=W, height=H)
        .theme(THEME)
    )
    swarm = (
        fm.Chart(df_swarm)
        .mark_swarm()
        .encode(x=fm.X("group"), y=fm.Y("value"))
        .theme(THEME)
    )
    return (box + swarm).properties(title="Boxplot + Swarm Overlay")


def build_comp_faceted_smooth():
    """Faceted Smooth — mark_point().mark_smooth(groupby='dose') with facet(col='dose')."""
    return (
        fm.Chart(df_facet)
        .mark_point()
        .mark_smooth(groupby="dose")
        .encode(
            x=fm.X("exposure", title="Exposure"),
            y=fm.Y("response", title="Response"),
        )
        .facet(col="dose", ncols=3)
        .properties(title="Faceted Smooth — by dose", width=W, height=H)
        .theme(THEME)
    )


def build_comp_hconcat():
    """HConcat — two different chart types side by side."""
    c1 = (
        fm.Chart(df_hex)
        .mark_hex()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="Hexbin", width=260, height=H)
        .theme(THEME)
    )
    c2 = (
        fm.Chart(df_box)
        .mark_violin()
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="Violin", width=260, height=H)
        .theme(THEME)
    )
    return fm.hconcat(c1, c2)


def build_comp_triple_layer():
    """Triple Layer — scatter + smooth + rule (3 layers)."""
    mean_y = float(np.mean(smooth_y))
    df_with_ref = df_smooth.with_columns(pl.lit(mean_y).alias("ref_y"))
    base = (
        fm.Chart(df_with_ref)
        .mark_point(opacity=0.3)
        .mark_smooth(ci=0.95)
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(width=W, height=H)
        .theme(THEME)
    )
    rule = fm.Chart(df_with_ref).mark_rule().encode(y=fm.Y("ref_y")).theme(THEME)
    return (base + rule).properties(title="Triple Layer — scatter + smooth + rule")


# ── Chart registries ──────────────────────────────────────────────────────────

GRAMMAR_MARKS = [
    (
        "mark_point",
        "mark_point() — Scatter",
        build_mark_point,
        "fm.Chart(df).mark_point().encode(x='x', y='y', color='class')",
    ),
    (
        "mark_line",
        "mark_line() — Multi-Series Line",
        build_mark_line,
        "fm.Chart(df).mark_line().encode(x='month', y='value', color='group')",
    ),
    (
        "mark_bar",
        "mark_bar() — Grouped Bar",
        build_mark_bar,
        "fm.Chart(df).mark_bar(position=fm.Dodge()).encode(x='quarter', y='sales', color='region')",
    ),
    (
        "mark_area",
        "mark_area() — Stacked Area",
        build_mark_area,
        "fm.Chart(df).mark_area(position=fm.Stack()).encode(x='x', y='value', color='series')",
    ),
    (
        "mark_boxplot",
        "mark_boxplot() — Box Plot",
        build_mark_boxplot,
        "fm.Chart(df).mark_boxplot().encode(x='group', y='value')",
    ),
    (
        "mark_violin",
        "mark_violin() — Violin",
        build_mark_violin,
        "fm.Chart(df).mark_violin().encode(x='group', y='value')",
    ),
    (
        "mark_histogram",
        "mark_histogram() — Histogram",
        build_mark_histogram,
        "fm.Chart(df).mark_histogram(groupby='group').encode(x='value', y='count', color='group')",
    ),
    (
        "mark_density",
        "mark_density() — KDE",
        build_mark_density,
        "fm.Chart(df).mark_density(groupby='group').encode(x='value', color='group')",
    ),
    (
        "mark_hex",
        "mark_hex() — Hexbin",
        build_mark_hex,
        "fm.Chart(df).mark_hex().encode(x='x', y='y')",
    ),
    (
        "mark_text",
        "mark_text() — Text Labels",
        build_mark_text,
        "fm.Chart(df).mark_text().encode(x='x', y='y', text='label')",
    ),
    (
        "mark_tick",
        "mark_tick() — Strip/Tick",
        build_mark_tick,
        "fm.Chart(df).mark_tick().encode(x='value', y='group')",
    ),
    (
        "mark_errorbar",
        "mark_errorbar() — Error Bars",
        build_mark_errorbar,
        "fm.Chart(df).mark_errorbar(extent='stdev').encode(x='group', y='value')",
    ),
    (
        "mark_smooth_ci",
        "mark_smooth(ci=0.95) — LOESS + CI Band",
        build_mark_smooth_ci,
        "fm.Chart(df).mark_smooth(ci=0.95).encode(x='x', y='y')",
    ),
    (
        "mark_smooth_lm",
        "mark_smooth(method='lm') — Linear",
        build_mark_smooth_lm,
        "fm.Chart(df).mark_smooth(method='lm').encode(x='x', y='y')",
    ),
    (
        "mark_contour",
        "mark_contour() — 2D Contour",
        build_mark_contour,
        "fm.Chart(df).mark_contour().encode(x='x', y='y')",
    ),
    (
        "mark_raster",
        "mark_raster() — 2D Raster",
        build_mark_raster,
        "fm.Chart(df).mark_raster().encode(x='x', y='y')",
    ),
    (
        "mark_swarm",
        "mark_swarm() — Beeswarm",
        build_mark_swarm,
        "fm.Chart(df).mark_swarm().encode(x='group', y='value')",
    ),
    (
        "mark_boxen",
        "mark_boxen() — Letter-Value",
        build_mark_boxen,
        "fm.Chart(df).mark_boxen().encode(x='group', y='value')",
    ),
    (
        "mark_qq",
        "mark_qq() — QQ Plot",
        build_mark_qq,
        "fm.Chart(df).mark_qq().encode(x='value')",
    ),
    (
        "mark_ribbon",
        "mark_ribbon() — Ribbon Band",
        build_mark_ribbon,
        "fm.Chart(df).mark_ribbon().encode(x='x', y='lower', y2='upper')",
    ),
    (
        "mark_segment",
        "mark_segment() — Segments",
        build_mark_segment,
        "fm.Chart(df).mark_segment().encode(x='x', y='y', x2='x2', y2='y2')",
    ),
    (
        "mark_rule",
        "mark_rule() — Reference Line Overlay",
        build_mark_rule,
        "fm.Chart(df).mark_point() + fm.Chart(rule_df).mark_rule().encode(y='y')",
    ),
    (
        "mark_rect",
        "mark_rect() — Heatmap Grid",
        build_mark_rect,
        "fm.Chart(df).mark_rect().encode(x='x', y='y', color='value')",
    ),
]

COMPOSITION_EXAMPLES = [
    (
        "comp_scatter_rule",
        "Scatter + Rule",
        build_comp_scatter_rule,
        "fm.Chart(df).mark_point() + fm.Chart(rule_df).mark_rule()",
    ),
    (
        "comp_scatter_smooth_ci",
        "Scatter + Smooth CI",
        build_comp_scatter_smooth_ci,
        "fm.Chart(df).mark_point(opacity=0.3).mark_smooth(ci=0.95).encode(...)",
    ),
    (
        "comp_grouped_smooth",
        "Grouped Smooth",
        build_comp_grouped_smooth,
        "fm.Chart(df).mark_smooth(method='lm', groupby='group').encode(x, y, color='group')",
    ),
    (
        "comp_dodge_bar_errorbar",
        "Dodge Bar + Errorbar",
        build_comp_dodge_bar_errorbar,
        "fm.Chart(df).mark_bar(position=fm.Dodge()) + fm.Chart(df).mark_errorbar()",
    ),
    (
        "comp_line_ribbon",
        "Line + Ribbon",
        build_comp_line_ribbon,
        "fm.Chart(df).mark_ribbon() + fm.Chart(df).mark_line()",
    ),
    (
        "comp_histogram_density",
        "Histogram + Density",
        build_comp_histogram_density,
        "fm.Chart(df).mark_histogram() + fm.Chart(df).mark_density()",
    ),
    (
        "comp_boxplot_swarm",
        "Boxplot + Swarm",
        build_comp_boxplot_swarm,
        "fm.Chart(df).mark_boxplot() + fm.Chart(df).mark_swarm()",
    ),
    (
        "comp_faceted_smooth",
        "Faceted Smooth",
        build_comp_faceted_smooth,
        "fm.Chart(df).mark_point().mark_smooth(groupby='dose').encode(...).facet(col='dose')",
    ),
    (
        "comp_hconcat",
        "HConcat",
        build_comp_hconcat,
        "chart_hex + chart_violin",
    ),
    (
        "comp_triple_layer",
        "Triple Layer",
        build_comp_triple_layer,
        "scatter + smooth + rule",
    ),
]


# ── rendering ─────────────────────────────────────────────────────────────────


def render_section(
    entries: list,
    section_label: str,
) -> dict[str, dict]:
    """Render all entries in a section. Returns dict of slug -> result info."""
    results = {}
    for slug, label, builder, api_call in entries:
        out_path = PNG_DIR / f"{slug}.png"
        print(f"  [{section_label}] {label:50s} ...", end="", flush=True)
        _, png, error = try_render(label, builder)
        if png is not None:
            out_path.write_bytes(png)
            results[slug] = {
                "label": label,
                "api_call": api_call,
                "png_path": out_path,
                "error": None,
            }
            print(" OK")
        else:
            results[slug] = {"label": label, "api_call": api_call, "png_path": None, "error": error}
    return results


def main() -> None:
    print("Ferrum Grammar Marks & Compositions — Paper Ink Theme")
    print("=" * 60)

    print("\nGrammar-Level Marks:")
    grammar_results = render_section(GRAMMAR_MARKS, "GRAMMAR")

    print("\nCompositions & Overlays:")
    composition_results = render_section(COMPOSITION_EXAMPLES, "COMPOSE")

    all_results = {**grammar_results, **composition_results}
    total = len(all_results)
    passed = sum(1 for v in all_results.values() if v["png_path"] is not None)
    print(f"\n{passed}/{total} charts rendered successfully.")

    # ── Generate HTML ─────────────────────────────────────────────────────────
    html = _generate_html(grammar_results, composition_results)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"\nHTML gallery written to: {HTML_OUT}")


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #F5F3EE;
    color: #1F2937;
    padding: 40px 32px 80px;
    max-width: 1600px;
    margin: 0 auto;
}
h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
    color: #111827;
}
.subtitle {
    font-size: 1rem;
    color: #6B7280;
    margin-bottom: 48px;
    border-bottom: 2px solid #D6D3D1;
    padding-bottom: 16px;
}
h2 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1F2937;
    margin: 48px 0 20px;
    padding-left: 12px;
    border-left: 4px solid #2563EB;
    letter-spacing: -0.3px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 20px;
}
.card {
    background: white;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.card-header {
    padding: 12px 16px 8px;
    border-bottom: 1px solid #F3F4F6;
}
.card-header h3 {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1F2937;
    margin-bottom: 2px;
}
.card-img {
    display: block;
    width: 100%;
    height: auto;
    background: #FAF7F2;
}
.card-footer {
    padding: 8px 12px;
    background: #F9FAFB;
    border-top: 1px solid #F3F4F6;
}
.card-footer code {
    font-size: 0.73rem;
    color: #6B7280;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    word-break: break-all;
}
.card.failed {
    border-color: #FCA5A5;
    background: #FFF5F5;
}
.card.failed .card-header h3 { color: #DC2626; }
.error-detail {
    padding: 12px;
    font-size: 0.75rem;
    color: #B91C1C;
    font-family: monospace;
    white-space: pre-wrap;
    word-break: break-all;
    background: #FEF2F2;
    border-top: 1px solid #FECACA;
    max-height: 140px;
    overflow-y: auto;
}
.stats {
    font-size: 0.85rem;
    color: #6B7280;
    margin-bottom: 8px;
}
.stats strong { color: #059669; }
footer {
    margin-top: 64px;
    text-align: center;
    font-size: 0.78rem;
    color: #9CA3AF;
    border-top: 1px solid #E5E7EB;
    padding-top: 24px;
}
"""


def _card(slug: str, info: dict) -> str:
    label = info["label"]
    api_call = info["api_call"]
    png_path = info["png_path"]
    error = info["error"]

    if png_path is not None and png_path.exists():
        data = png_path.read_bytes()
        b64 = base64.b64encode(data).decode()
        img_tag = (
            f'<img class="card-img" src="data:image/png;base64,{b64}" alt="{label}" loading="lazy">'
        )
        return f"""
<div class="card">
  <div class="card-header"><h3>{label}</h3></div>
  {img_tag}
  <div class="card-footer"><code>{api_call}</code></div>
</div>"""
    else:
        short_err = (error or "unknown error").split("\n")[0][:200]
        return f"""
<div class="card failed">
  <div class="card-header"><h3>{label}</h3></div>
  <div class="error-detail">FAILED: {short_err}</div>
  <div class="card-footer"><code>{api_call}</code></div>
</div>"""


def _section(title: str, results: dict) -> str:
    cards = "\n".join(_card(slug, info) for slug, info in results.items())
    return f'<h2>{title}</h2>\n<div class="grid">\n{cards}\n</div>'


def _generate_html(grammar: dict, composition: dict) -> str:
    all_r = {**grammar, **composition}
    total = len(all_r)
    passed = sum(1 for v in all_r.values() if v["png_path"] is not None)

    sections = "\n".join(
        [
            _section("Grammar-Level Marks", grammar),
            _section("Compositions &amp; Overlays", composition),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grammar Marks &amp; Compositions — Paper Ink Theme</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>Grammar Marks &amp; Compositions — Paper Ink Theme</h1>
<p class="subtitle">
  Every implemented grammar-level mark plus composition stress tests, rendered with
  <code>fm.themes.paper_ink</code> for visual inspection. &nbsp;|&nbsp;
  <span class="stats"><strong>{passed}</strong> / {total} rendered successfully.</span>
</p>
{sections}
<footer>Generated by <code>design-docs/themes/demo_grammar_marks.py</code> &mdash; ferrum {fm.__version__}</footer>
</body>
</html>"""


if __name__ == "__main__":
    main()
