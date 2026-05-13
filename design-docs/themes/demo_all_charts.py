"""Comprehensive ferrum chart gallery — Paper Ink theme.

Renders every public-facing plot type from ferrum using the Paper Ink theme,
saves each as a PNG, and generates a single HTML gallery document.

Usage:
    uv run python design-docs/themes/demo_all_charts.py
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
PNG_DIR = HERE / "demo_all_pngs"
HTML_OUT = HERE / "demo_all_charts.html"
PNG_DIR.mkdir(parents=True, exist_ok=True)

THEME = fm.themes.paper_ink
W, H = 580, 400

# ── synthetic data ─────────────────────────────────────────────────────────────
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

# Simple bar for area
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

# Wide corr matrix
feature_names = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
cov_raw = rng.standard_normal((5, 50))
corr = np.corrcoef(cov_raw)
corr_rows = [
    {"feature": fn, **{cn: float(round(corr[i, j], 2)) for j, cn in enumerate(feature_names)}}
    for i, fn in enumerate(feature_names)
]
df_corr = pl.DataFrame(corr_rows)

# Hex / contour data
hex_x = rng.normal(0, 1, 500)
hex_y = hex_x * 0.7 + rng.normal(0, 0.5, 500)
df_hex = pl.DataFrame({"x": hex_x, "y": hex_y})

# Two-variable data for smooth / errorband
smooth_x = rng.uniform(0, 10, 120)
smooth_y = 2 * smooth_x + rng.normal(0, 3, 120)
df_smooth = pl.DataFrame({"x": smooth_x, "y": smooth_y})

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

# Function data for mark_function
fn_x = np.linspace(0, 2 * np.pi, 200)
rows_fn = []
for name, vals in [("sin", np.sin(fn_x)), ("cos", np.cos(fn_x))]:
    for x, y in zip(fn_x, vals):
        rows_fn.append({"x": float(x), "y": float(y), "fn": name})
df_fn = pl.DataFrame(rows_fn)

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

# Raster (2D density-like)
raster_x = rng.uniform(0, 10, 1000)
raster_y = raster_x * 0.5 + rng.normal(0, 1.5, 1000)
df_raster = pl.DataFrame({"x": raster_x, "y": raster_y})

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

# jointplot / pairplot data
iris_x = np.vstack(
    [
        rng.multivariate_normal([5.0, 3.5], [[0.4, 0.1], [0.1, 0.2]], 50),
        rng.multivariate_normal([6.0, 2.8], [[0.5, 0.2], [0.2, 0.3]], 50),
        rng.multivariate_normal([6.5, 3.0], [[0.3, 0.1], [0.1, 0.4]], 50),
    ]
)
df_iris = pl.DataFrame(
    {
        "sepal_length": iris_x[:, 0],
        "sepal_width": iris_x[:, 1],
        "petal_length": rng.normal(3.5, 1.5, 150),
        "petal_width": rng.normal(1.2, 0.7, 150),
        "species": ["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50,
    }
)

# clustermap data (wide)
cluster_data = rng.standard_normal((8, 6))
cluster_rows = [
    {"row": f"Sample {i + 1}", **{f"Gene {j + 1}": float(cluster_data[i, j]) for j in range(6)}}
    for i in range(8)
]
df_cluster = pl.DataFrame(cluster_rows)

# lmplot data
lm_x = rng.uniform(0, 10, 100)
lm_y = 1.5 * lm_x + rng.normal(0, 2, 100)
lm_grp = np.where(lm_x < 5, "Low", "High")
df_lm = pl.DataFrame({"x": lm_x, "y": lm_y, "group": lm_grp})

# residplot data (for figure.residplot, not diagnostic)
df_resid = df_lm

# Parallel coordinates — use iris
df_parallel = df_iris

# ── sklearn model setup ────────────────────────────────────────────────────────
try:
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_breast_cancer, load_iris, make_regression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import Lasso, LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    # Classification (multi-class)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    clf_rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_iris, y_iris)
    feature_names_iris = list(iris.feature_names)

    # Classification (binary)
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data[:, :8], bc.target
    clf_bc = LogisticRegression(max_iter=300, random_state=42).fit(X_bc, y_bc)
    feature_names_bc = list(bc.feature_names[:8])

    # Also try GBM on breast cancer for SHAP
    gbm_bc = GradientBoostingClassifier(n_estimators=50, random_state=42).fit(X_bc, y_bc)

    # Regression
    X_reg, y_reg = make_regression(n_samples=200, n_features=6, noise=10, random_state=42)
    reg = Ridge(alpha=1.0).fit(X_reg, y_reg)
    feature_names_reg = [f"f{i}" for i in range(6)]

    # PCA
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    pca = PCA(n_components=4).fit(X_iris_scaled)

    # KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_iris)

    # pandas DataFrames for diagnostics API
    df_iris_pd = pd.DataFrame(X_iris, columns=feature_names_iris)
    df_iris_pd["target"] = y_iris
    df_bc_pd = pd.DataFrame(X_bc, columns=feature_names_bc)
    df_bc_pd["target"] = y_bc
    df_reg_pd = pd.DataFrame(X_reg, columns=feature_names_reg)
    df_reg_pd["target"] = y_reg

    SKLEARN_AVAILABLE = True
except Exception as exc:
    SKLEARN_AVAILABLE = False
    print(f"[WARN] sklearn not available: {exc}")


# ── chart builders ─────────────────────────────────────────────────────────────


def try_render(name: str, fn) -> tuple[str, bytes | None, str | None]:
    """Run fn(), return (name, png_bytes_or_None, error_or_None)."""
    try:
        result = fn()
        if isinstance(result, bytes):
            png = result
        else:
            # It's a Chart/JointChart/etc. — call show_png()
            png = result.show_png()
        return name, png, None
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  [FAIL] {name}: {exc}")
        return name, None, f"{exc}\n{tb}"


# ── Grammar-level marks ───────────────────────────────────────────────────────


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


def build_mark_rule():
    scatter = (
        fm.Chart(df_smooth)
        .mark_point(opacity=0.4)
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(width=W, height=H)
        .theme(THEME)
    )
    rule_df = pl.DataFrame({"y": [float(np.mean(smooth_y))]})
    rule = fm.Chart(rule_df).mark_rule(stroke_dash="4,2").encode(y=fm.Y("y")).theme(THEME)
    return (scatter + rule).properties(title="mark_rule() — Reference Line overlay")


def build_mark_errorbar():
    # mark_errorbar uses an extent stat transform on raw data — no pre-aggregated yError needed
    return (
        fm.Chart(df_box)
        .mark_errorbar(extent="stdev")
        .encode(x=fm.X("group", title="Group"), y=fm.Y("value", title="Value"))
        .properties(title="mark_errorbar() — Error Bars", width=W, height=H)
        .theme(THEME)
    )


def build_mark_errorband():
    return (
        fm.Chart(df_smooth)
        .mark_smooth(ci=0.95)
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="mark_smooth(ci=0.95) — LOESS + CI Band", width=W, height=H)
        .theme(THEME)
    )


def build_mark_smooth():
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


def build_mark_function():
    return (
        fm.Chart(df_fn)
        .mark_line()
        .encode(
            x=fm.X("x", title="x"), y=fm.Y("y", title="y"), color=fm.Color("fn", title="Function")
        )
        .properties(title="mark_function() — Sin/Cos Lines", width=W, height=H)
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
    x_rb = np.linspace(0, 10, 50)
    df_rb = pl.DataFrame(
        {
            "x": x_rb,
            "lower": np.sin(x_rb) - 0.5,
            "upper": np.sin(x_rb) + 0.5,
        }
    )
    return (
        fm.Chart(df_rb)
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


# ── Composition examples ──────────────────────────────────────────────────────


def build_facet():
    return (
        fm.Chart(df_facet)
        .mark_point()
        .encode(x=fm.X("exposure", title="Exposure"), y=fm.Y("response", title="Response"))
        .facet(col="dose", ncols=3)
        .properties(title="Faceted Chart — by Dose", width=W, height=H)
        .theme(THEME)
    )


def build_layered():
    chart = (
        fm.Chart(df_smooth)
        .mark_point(opacity=0.5)
        .mark_smooth()
        .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        .properties(title="Layered Chart — Scatter + Smooth", width=W, height=H)
        .theme(THEME)
    )
    return chart


def build_hconcat():
    c1 = (
        fm.Chart(df_scatter)
        .mark_point()
        .encode(x=fm.X("x"), y=fm.Y("y"), color=fm.Color("class"))
        .properties(title="Panel A", width=260, height=H)
        .theme(THEME)
    )
    c2 = (
        fm.Chart(df_box)
        .mark_boxplot()
        .encode(x=fm.X("group"), y=fm.Y("value"))
        .properties(title="Panel B", width=260, height=H)
        .theme(THEME)
    )
    return c1 + c2


# ── Figure-level functions ────────────────────────────────────────────────────


def build_catplot_box():
    return fm.catplot(df_box, x="group", y="value", kind="box", theme=THEME).properties(
        title="catplot(kind='box')", width=W, height=H
    )


def build_catplot_violin():
    return fm.catplot(df_violin, x="group", y="value", kind="violin", theme=THEME).properties(
        title="catplot(kind='violin')", width=W, height=H
    )


def build_catplot_strip():
    return fm.catplot(df_box, x="group", y="value", kind="strip", theme=THEME).properties(
        title="catplot(kind='strip')", width=W, height=H
    )


def build_catplot_swarm():
    return fm.catplot(df_swarm, x="group", y="value", kind="swarm", theme=THEME).properties(
        title="catplot(kind='swarm')", width=W, height=H
    )


def build_catplot_bar():
    return fm.catplot(
        df_bar, x="quarter", y="sales", hue="region", kind="bar", theme=THEME
    ).properties(title="catplot(kind='bar')", width=W, height=H)


def build_displot_hist():
    # multiple="dodge" passes groupby so hue column is preserved in the transform output
    return fm.displot(
        df_hist, x="value", hue="group", kind="hist", multiple="dodge", theme=THEME
    ).properties(title="displot(kind='hist')", width=W, height=H)


def build_displot_kde():
    return fm.displot(df_hist, x="value", hue="group", kind="kde", theme=THEME).properties(
        title="displot(kind='kde')", width=W, height=H
    )


def build_displot_ecdf():
    # ecdf without hue to avoid group-column-not-found in transformed data
    return fm.displot(df_hist, x="value", kind="ecdf", theme=THEME).properties(
        title="displot(kind='ecdf')", width=W, height=H
    )


def build_lmplot():
    return fm.lmplot(df_lm, x="x", y="y", hue="group", theme=THEME).properties(
        title="lmplot() — Linear Model Plot", width=W, height=H
    )


def build_residplot():
    return fm.residplot(df_resid, x="x", y="y", theme=THEME).properties(
        title="residplot() — Residual Scatter", width=W, height=H
    )


def build_jointplot():
    # hue causes "unknown column" in the marginal transforms, use without hue
    return fm.jointplot(df_iris, x="sepal_length", y="sepal_width", theme=THEME)


def build_pairplot():
    return fm.pairplot(
        df_iris,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        theme=THEME,
    )


def build_heatmap():
    return fm.heatmap(df_corr, center=0.0, annot=True, fmt=".2f", theme=THEME).properties(
        title="heatmap() — Correlation Matrix", width=480, height=400
    )


def build_clustermap():
    return fm.clustermap(df_cluster, theme=THEME)


# ── Diagnostic charts (sklearn required) ─────────────────────────────────────


def build_roc_chart():
    return fm.roc_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="roc_chart() — ROC Curve", width=W, height=H
    )


def build_pr_chart():
    return fm.pr_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="pr_chart() — Precision-Recall Curve", width=W, height=H
    )


def build_confusion_matrix_chart():
    return fm.confusion_matrix_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="confusion_matrix_chart()", width=480, height=400
    )


def build_residuals_chart():
    return fm.residuals_chart(reg, X_reg, y_reg, theme=THEME)


def build_prediction_error_chart():
    src = fm.ModelSource(reg, X_reg, y_reg)
    return (
        fm.Chart(src.predictions())
        .mark_prediction_error()
        .properties(title="prediction_error_chart()", width=W, height=H)
        .theme(THEME)
    )


def build_learning_curve_chart():
    return fm.learning_curve_chart(
        LogisticRegression(max_iter=200), X_bc, y_bc, cv=3, theme=THEME
    ).properties(title="learning_curve_chart()", width=W, height=H)


def build_validation_curve_chart():
    return fm.validation_curve_chart(
        Ridge(), X_reg, y_reg, param="alpha", values=[0.01, 0.1, 1, 10, 100], cv=3, theme=THEME
    ).properties(title="validation_curve_chart()", width=W, height=H)


def build_cv_scores_chart():
    return fm.cv_scores_chart(
        LogisticRegression(max_iter=200), X_bc, y_bc, cv=5, theme=THEME
    ).properties(title="cv_scores_chart()", width=W, height=H)


def build_calibration_chart():
    return fm.calibration_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="calibration_chart()", width=W, height=H
    )


def build_importance_chart():
    return fm.importance_chart(clf_rf, X_iris, y_iris, theme=THEME).properties(
        title="importance_chart() — Feature Importance", width=W, height=H
    )


def build_gain_chart():
    return fm.gain_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="gain_chart() — Cumulative Gains", width=W, height=H
    )


def build_lift_chart():
    return fm.lift_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="lift_chart() — Lift Curve", width=W, height=H
    )


def build_discrimination_threshold_chart():
    return fm.discrimination_threshold_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="discrimination_threshold_chart()", width=W, height=H
    )


def build_alpha_selection_chart():
    alphas = list(np.logspace(-3, 2, 12))
    return fm.alpha_selection_chart(
        Ridge(), X_reg, y_reg, alphas=alphas, cv=3, theme=THEME
    ).properties(title="alpha_selection_chart()", width=W, height=H)


def build_pca_scree_chart():
    return fm.pca_scree_chart(pca, X_iris_scaled, theme=THEME).properties(
        title="pca_scree_chart()", width=W, height=H
    )


def build_intercluster_distance_chart():
    return fm.intercluster_distance_chart(km, X_iris, theme=THEME).properties(
        title="intercluster_distance_chart()", width=W, height=H
    )


def build_decision_boundary_chart():
    clf_2d = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_iris[:, :2], y_iris)
    return fm.decision_boundary_chart(
        clf_2d, X_iris[:, :2], y_iris, features=(0, 1), grid_resolution=80, theme=THEME
    ).properties(title="decision_boundary_chart()", width=W, height=H)


def build_parallel_coordinates_chart():
    return fm.parallel_coordinates_chart(
        df_iris.select(["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]),
        hue="species",
        theme=THEME,
    ).properties(title="parallel_coordinates_chart()", width=W, height=H)


def build_rank1d_chart():
    X_df = pl.DataFrame(
        {fn: list(X_iris[:, i].astype(float)) for i, fn in enumerate(feature_names_iris)}
    )
    return fm.rank1d_chart(X_df, theme=THEME).properties(
        title="rank1d_chart() — Univariate Ranking", width=W, height=H
    )


def build_rank2d_chart():
    X_df = pl.DataFrame(
        {fn: list(X_iris[:, i].astype(float)) for i, fn in enumerate(feature_names_iris)}
    )
    return fm.rank2d_chart(X_df, theme=THEME).properties(
        title="rank2d_chart() — Pairwise Correlation", width=480, height=400
    )


def build_shap_beeswarm_chart():
    return fm.shap_beeswarm_chart(gbm_bc, X_bc, y_bc, max_display=8, theme=THEME).properties(
        title="shap_beeswarm_chart()", width=W, height=H
    )


def build_shap_bar_chart():
    return fm.shap_bar_chart(gbm_bc, X_bc, y_bc, max_display=8, theme=THEME).properties(
        title="shap_bar_chart()", width=W, height=H
    )


def build_shap_waterfall_chart():
    return fm.shap_waterfall_chart(
        gbm_bc, X_bc, y_bc, sample_idx=0, max_display=8, theme=THEME
    ).properties(title="shap_waterfall_chart()", width=W, height=H)


def build_class_prediction_error_chart():
    return fm.class_prediction_error_chart(clf_bc, X_bc, y_bc, theme=THEME).properties(
        title="class_prediction_error_chart()", width=W, height=H
    )


def build_cluster_diagnostics():
    return fm.cluster_diagnostics(X_iris, ks=range(2, 7), scoring="both", theme=THEME)


def build_pdp_chart():
    return fm.pdp_chart(clf_rf, X_iris, y_iris, features=[0, 1], theme=THEME).properties(
        title="pdp_chart() — Partial Dependence", width=W, height=H
    )


# ── Chart registry ─────────────────────────────────────────────────────────────

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
        "fm.Chart(df).mark_errorbar().encode(x='group', y='mean', yError='err')",
    ),
    (
        "mark_smooth_ci",
        "mark_smooth(ci) — LOESS + CI Band",
        build_mark_errorband,
        "fm.Chart(df).mark_smooth(ci=0.95).encode(x='x', y='y')",
    ),
    (
        "mark_smooth_lm",
        "mark_smooth(lm) — Linear",
        build_mark_smooth,
        "fm.Chart(df).mark_smooth(method='lm').encode(x='x', y='y')",
    ),
    (
        "mark_contour",
        "mark_contour() — 2D Contour",
        build_mark_contour,
        "fm.Chart(df).mark_contour().encode(x='x', y='y')",
    ),
    (
        "mark_function",
        "mark_function() — Function Lines",
        build_mark_function,
        "fm.Chart(df).mark_line().encode(x='x', y='y', color='fn')",
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
    ("mark_qq", "mark_qq() — QQ Plot", build_mark_qq, "fm.Chart(df).mark_qq().encode(x='value')"),
    (
        "mark_ribbon",
        "mark_ribbon() — Error Band",
        build_mark_ribbon,
        "fm.Chart(df).mark_errorband().encode(x='group', y='lower', y2='upper')",
    ),
    (
        "mark_segment",
        "mark_segment() — Segments",
        build_mark_segment,
        "fm.Chart(df).mark_segment().encode(x='x', y='y', x2='x2', y2='y2')",
    ),
]

COMPOSITION_EXAMPLES = [
    (
        "facet",
        "Faceted Chart",
        build_facet,
        "fm.Chart(df).mark_point().encode(...).facet(col='dose', ncols=3)",
    ),
    ("layered", "Layered — Scatter+Smooth", build_layered, "scatter.layer(smooth).properties(...)"),
    ("hconcat", "Horizontal Concat (A+B)", build_hconcat, "chart1 + chart2"),
]

FIGURE_FUNCTIONS = [
    (
        "catplot_box",
        "catplot(kind='box')",
        build_catplot_box,
        "fm.catplot(df, x='group', y='value', kind='box')",
    ),
    (
        "catplot_violin",
        "catplot(kind='violin')",
        build_catplot_violin,
        "fm.catplot(df, x='group', y='value', kind='violin')",
    ),
    (
        "catplot_strip",
        "catplot(kind='strip')",
        build_catplot_strip,
        "fm.catplot(df, x='group', y='value', kind='strip')",
    ),
    (
        "catplot_swarm",
        "catplot(kind='swarm')",
        build_catplot_swarm,
        "fm.catplot(df, x='group', y='value', kind='swarm')",
    ),
    (
        "catplot_bar",
        "catplot(kind='bar')",
        build_catplot_bar,
        "fm.catplot(df, x='quarter', y='sales', hue='region', kind='bar')",
    ),
    (
        "displot_hist",
        "displot(kind='hist')",
        build_displot_hist,
        "fm.displot(df, x='value', hue='group', kind='hist')",
    ),
    (
        "displot_kde",
        "displot(kind='kde')",
        build_displot_kde,
        "fm.displot(df, x='value', hue='group', kind='kde')",
    ),
    (
        "displot_ecdf",
        "displot(kind='ecdf')",
        build_displot_ecdf,
        "fm.displot(df, x='value', kind='ecdf')",
    ),
    ("lmplot", "lmplot()", build_lmplot, "fm.lmplot(df, x='x', y='y', hue='group')"),
    ("residplot", "residplot()", build_residplot, "fm.residplot(df, x='x', y='y')"),
    (
        "jointplot",
        "jointplot()",
        build_jointplot,
        "fm.jointplot(df, x='sepal_length', y='sepal_width', hue='species')",
    ),
    ("pairplot", "pairplot()", build_pairplot, "fm.pairplot(df, vars=[...], hue='species')"),
    ("heatmap", "heatmap()", build_heatmap, "fm.heatmap(df_corr, center=0, annot=True)"),
    ("clustermap", "clustermap()", build_clustermap, "fm.clustermap(df_cluster)"),
]

DIAGNOSTIC_CHARTS = [
    ("roc_chart", "roc_chart()", build_roc_chart, "fm.roc_chart(clf, X, y)"),
    ("pr_chart", "pr_chart()", build_pr_chart, "fm.pr_chart(clf, X, y)"),
    (
        "confusion_matrix_chart",
        "confusion_matrix_chart()",
        build_confusion_matrix_chart,
        "fm.confusion_matrix_chart(clf, X, y)",
    ),
    (
        "residuals_chart",
        "residuals_chart() — 4 panels",
        build_residuals_chart,
        "fm.residuals_chart(reg, X, y)",
    ),
    (
        "prediction_error_chart",
        "prediction_error_chart()",
        build_prediction_error_chart,
        "fm.Chart(src.predictions()).mark_prediction_error()",
    ),
    (
        "learning_curve_chart",
        "learning_curve_chart()",
        build_learning_curve_chart,
        "fm.learning_curve_chart(clf, X, y, cv=3)",
    ),
    (
        "validation_curve_chart",
        "validation_curve_chart()",
        build_validation_curve_chart,
        "fm.validation_curve_chart(Ridge(), X, y, param='alpha', values=[...])",
    ),
    (
        "cv_scores_chart",
        "cv_scores_chart()",
        build_cv_scores_chart,
        "fm.cv_scores_chart(clf, X, y, cv=5)",
    ),
    (
        "calibration_chart",
        "calibration_chart()",
        build_calibration_chart,
        "fm.calibration_chart(clf, X, y)",
    ),
    (
        "importance_chart",
        "importance_chart()",
        build_importance_chart,
        "fm.importance_chart(rf, X, y)",
    ),
    ("gain_chart", "gain_chart()", build_gain_chart, "fm.gain_chart(clf, X, y)"),
    ("lift_chart", "lift_chart()", build_lift_chart, "fm.lift_chart(clf, X, y)"),
    (
        "discrimination_threshold",
        "discrimination_threshold_chart()",
        build_discrimination_threshold_chart,
        "fm.discrimination_threshold_chart(clf, X, y)",
    ),
    (
        "alpha_selection_chart",
        "alpha_selection_chart()",
        build_alpha_selection_chart,
        "fm.alpha_selection_chart(Ridge(), X, y, alphas=[...])",
    ),
    ("pca_scree_chart", "pca_scree_chart()", build_pca_scree_chart, "fm.pca_scree_chart(pca, X)"),
    (
        "intercluster_distance_chart",
        "intercluster_distance_chart()",
        build_intercluster_distance_chart,
        "fm.intercluster_distance_chart(km, X)",
    ),
    (
        "decision_boundary_chart",
        "decision_boundary_chart()",
        build_decision_boundary_chart,
        "fm.decision_boundary_chart(clf, X, y, features=(0,1))",
    ),
    (
        "parallel_coordinates_chart",
        "parallel_coordinates_chart()",
        build_parallel_coordinates_chart,
        "fm.parallel_coordinates_chart(df, hue='species')",
    ),
    ("rank1d_chart", "rank1d_chart()", build_rank1d_chart, "fm.rank1d_chart(X_df)"),
    ("rank2d_chart", "rank2d_chart()", build_rank2d_chart, "fm.rank2d_chart(X_df)"),
    (
        "shap_beeswarm_chart",
        "shap_beeswarm_chart()",
        build_shap_beeswarm_chart,
        "fm.shap_beeswarm_chart(gbm, X, y)",
    ),
    ("shap_bar_chart", "shap_bar_chart()", build_shap_bar_chart, "fm.shap_bar_chart(gbm, X, y)"),
    (
        "shap_waterfall_chart",
        "shap_waterfall_chart()",
        build_shap_waterfall_chart,
        "fm.shap_waterfall_chart(gbm, X, y, sample_idx=0)",
    ),
    (
        "class_prediction_error",
        "class_prediction_error_chart()",
        build_class_prediction_error_chart,
        "fm.class_prediction_error_chart(clf, X, y)",
    ),
    (
        "cluster_diagnostics",
        "cluster_diagnostics()",
        build_cluster_diagnostics,
        "fm.cluster_diagnostics(X, ks=range(2,7))",
    ),
    ("pdp_chart", "pdp_chart()", build_pdp_chart, "fm.pdp_chart(clf, X, y, features=[0, 1])"),
]


# ── rendering ──────────────────────────────────────────────────────────────────


def render_section(
    entries: list,
    section_label: str,
    skip_if_no_sklearn: bool = False,
) -> dict[str, dict]:
    """Render all entries in a section. Returns dict of slug -> result info."""
    results = {}
    for slug, label, builder, api_call in entries:
        if skip_if_no_sklearn and not SKLEARN_AVAILABLE:
            results[slug] = {
                "label": label,
                "api_call": api_call,
                "png_path": None,
                "error": "sklearn not available",
            }
            continue

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
    print("Ferrum Chart Gallery — Paper Ink Theme")
    print("=" * 60)

    print("\nGrammar-Level Marks:")
    grammar_results = render_section(GRAMMAR_MARKS, "GRAMMAR")

    print("\nComposition Examples:")
    composition_results = render_section(COMPOSITION_EXAMPLES, "COMPOSE")

    print("\nFigure-Level Functions:")
    figure_results = render_section(FIGURE_FUNCTIONS, "FIGURE")

    print("\nDiagnostic Charts:")
    diag_results = render_section(
        DIAGNOSTIC_CHARTS, "DIAG", skip_if_no_sklearn=not SKLEARN_AVAILABLE
    )

    all_results = {**grammar_results, **composition_results, **figure_results, **diag_results}
    total = len(all_results)
    passed = sum(1 for v in all_results.values() if v["png_path"] is not None)
    print(f"\n{passed}/{total} charts rendered successfully.")

    # ── Generate HTML ──────────────────────────────────────────────────────────
    html = _generate_html(grammar_results, composition_results, figure_results, diag_results)
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
        # Embed as base64 data URI so the HTML is self-contained
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


def _generate_html(grammar, composition, figure, diag) -> str:
    all_r = {**grammar, **composition, **figure, **diag}
    total = len(all_r)
    passed = sum(1 for v in all_r.values() if v["png_path"] is not None)

    sections = "\n".join(
        [
            _section("Grammar-Level Marks", grammar),
            _section("Composition Examples", composition),
            _section("Figure-Level Functions", figure),
            _section("Diagnostic &amp; Model Charts", diag),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ferrum Chart Gallery — Paper Ink Theme</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>Ferrum Chart Gallery — Paper Ink Theme</h1>
<p class="subtitle">
  Every public-facing chart type rendered with <code>fm.themes.paper_ink</code> for visual inspection. &nbsp;|&nbsp;
  <span class="stats"><strong>{passed}</strong> / {total} rendered successfully.</span>
</p>
{sections}
<footer>Generated by <code>design-docs/themes/demo_all_charts.py</code> &mdash; ferrum {fm.__version__}</footer>
</body>
</html>"""


if __name__ == "__main__":
    main()
