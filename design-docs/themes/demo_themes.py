"""Theme visual demo — Paper Ink, Slate Citrus, Arctic Signal.

Renders 6 diverse chart types, each in all three custom themes, then
generates a single HTML comparison grid at design-docs/themes/demo_themes.html.

Usage:
    uv run python design-docs/themes/demo_themes.py
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

import ferrum as fm

# ── paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PNG_DIR = HERE / "demo_pngs"
HTML_OUT = HERE / "demo_themes.html"
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ── themes under comparison ──────────────────────────────────────────────────
THEMES = [
    ("paper_ink",     fm.themes.paper_ink,     "Paper Ink"),
    ("slate_citrus",  fm.themes.slate_citrus,   "Slate Citrus"),
    ("arctic_signal", fm.themes.arctic_signal,  "Arctic Signal"),
]

# ── synthetic datasets ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# Multi-series time-series data (line chart)
n_months = 24
months = list(range(n_months))
groups = ["Product A", "Product B", "Product C", "Product D", "Product E"]
rows_ts = []
for g in groups:
    base = rng.uniform(20, 80)
    noise = rng.normal(0, 4, n_months).cumsum()
    for i, v in enumerate(base + noise):
        rows_ts.append({"month": i, "value": float(v), "group": g})
df_lines = pl.DataFrame(rows_ts)

# Scatter data with 3 classes
n_scatter = 200
scatter_x = np.concatenate([
    rng.normal(-1.5, 0.8, n_scatter // 3),
    rng.normal(1.5, 0.8, n_scatter // 3),
    rng.normal(0.0, 0.8, n_scatter - 2 * (n_scatter // 3)),
])
scatter_y = np.concatenate([
    rng.normal(-1.5, 0.8, n_scatter // 3),
    rng.normal(1.5, 0.8, n_scatter // 3),
    rng.normal(0.0, 0.8, n_scatter - 2 * (n_scatter // 3)),
])
scatter_cls = (
    ["Alpha"] * (n_scatter // 3)
    + ["Beta"] * (n_scatter // 3)
    + ["Gamma"] * (n_scatter - 2 * (n_scatter // 3))
)
df_scatter = pl.DataFrame({
    "x": scatter_x,
    "y": scatter_y,
    "class": scatter_cls,
})

# Bar chart data — category + value + group
categories = ["Q1", "Q2", "Q3", "Q4"]
bar_groups = ["North", "South", "West"]
rows_bar = []
for cat in categories:
    for bg in bar_groups:
        rows_bar.append({
            "quarter": cat,
            "region": bg,
            "sales": float(rng.integers(40, 120)),
        })
df_bar = pl.DataFrame(rows_bar)

# Correlation matrix for heatmap
feature_names = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
cov_raw = rng.standard_normal((5, 50))
corr = np.corrcoef(cov_raw)
corr_rows = []
for i, row_name in enumerate(feature_names):
    d = {"feature": row_name}
    for j, col_name in enumerate(feature_names):
        d[col_name] = float(round(corr[i, j], 2))
    corr_rows.append(d)
df_corr = pl.DataFrame(corr_rows)

# Boxplot / distribution data — 4 groups, 100 obs each
box_groups = ["Control", "Treatment A", "Treatment B", "Treatment C"]
rows_box = []
means = [0.0, 0.8, -0.4, 1.2]
for g, mu in zip(box_groups, means):
    vals = rng.normal(mu, 1.0, 100)
    for v in vals:
        rows_box.append({"group": g, "value": float(v)})
df_box = pl.DataFrame(rows_box)

# Faceted strip/scatter — 3 facet panels
facet_categories = ["Low", "Medium", "High"]
rows_facet = []
for fc in facet_categories:
    n = 60
    x_vals = rng.uniform(0, 10, n)
    if fc == "Low":
        y_vals = 0.3 * x_vals + rng.normal(0, 0.5, n)
    elif fc == "Medium":
        y_vals = 0.7 * x_vals + rng.normal(0, 0.8, n)
    else:
        y_vals = 1.2 * x_vals + rng.normal(0, 1.0, n)
    for xv, yv in zip(x_vals, y_vals):
        rows_facet.append({"exposure": float(xv), "response": float(yv), "dose": fc})
df_facet = pl.DataFrame(rows_facet)

# Histogram KDE overlay data
hist_groups = ["Group A", "Group B"]
rows_hist = []
for g in hist_groups:
    mu = 0.0 if g == "Group A" else 2.0
    vals = rng.normal(mu, 1.0, 150)
    for v in vals:
        rows_hist.append({"value": float(v), "group": g})
df_hist = pl.DataFrame(rows_hist)


# ── chart builder functions ───────────────────────────────────────────────────

def build_line(theme: Any) -> bytes:
    """Multi-series line chart — showcases categorical color cycle."""
    chart = (
        fm.Chart(df_lines)
        .mark_line()
        .encode(
            x=fm.X("month", title="Month"),
            y=fm.Y("value", title="Value"),
            color=fm.Color("group", title="Product"),
        )
        .properties(title="Multi-Series Trends", width=480, height=300)
        .theme(theme)
    )
    return chart.show_png()


def build_scatter(theme: Any) -> bytes:
    """Scatter plot with color-encoded classes."""
    chart = (
        fm.Chart(df_scatter)
        .mark_point()
        .encode(
            x=fm.X("x", title="Component 1"),
            y=fm.Y("y", title="Component 2"),
            color=fm.Color("class", title="Class"),
        )
        .properties(title="Cluster Scatter", width=480, height=300)
        .theme(theme)
    )
    return chart.show_png()


def build_bar(theme: Any) -> bytes:
    """Grouped bar chart — showcases mark color and grid contrast."""
    chart = (
        fm.Chart(df_bar)
        .mark_bar(position=fm.Dodge())
        .encode(
            x=fm.X("quarter", title="Quarter"),
            y=fm.Y("sales", title="Sales"),
            color=fm.Color("region", title="Region"),
        )
        .properties(title="Regional Sales by Quarter", width=480, height=300)
        .theme(theme)
    )
    return chart.show_png()


def build_heatmap(theme: Any) -> bytes:
    """Correlation heatmap — showcases diverging colormap."""
    chart = fm.heatmap(
        df_corr,
        cmap="rdbu",
        center=0.0,
        annot=True,
        fmt=".2f",
        theme=theme,
    ).properties(title="Feature Correlation Matrix", width=420, height=360)
    return chart.show_png()


def build_boxplot(theme: Any) -> bytes:
    """Boxplot — showcases composite mark styling."""
    chart = (
        fm.Chart(df_box)
        .mark_boxplot()
        .encode(
            x=fm.X("group", title="Group"),
            y=fm.Y("value", title="Measurement"),
        )
        .properties(title="Distribution by Group", width=480, height=300)
        .theme(theme)
    )
    return chart.show_png()


def build_faceted(theme: Any) -> bytes:
    """Faceted scatter — showcases strip backgrounds and panel layout."""
    chart = (
        fm.Chart(df_facet)
        .mark_point()
        .encode(
            x=fm.X("exposure", title="Exposure"),
            y=fm.Y("response", title="Response"),
        )
        .facet(col="dose", ncols=3)
        .properties(title="Dose-Response (Faceted)", width=480, height=300)
        .theme(theme)
    )
    return chart.show_png()


def build_histogram(theme: Any) -> bytes:
    """Histogram with grouped color — showcases fill and grid interaction."""
    chart = (
        fm.Chart(df_hist)
        .mark_histogram(groupby="group")
        .encode(
            x=fm.X("value", title="Value"),
            y=fm.Y("count", title="Count"),
            color=fm.Color("group", title="Group"),
        )
        .properties(title="Distribution Histogram", width=480, height=300)
        .theme(theme)
    )
    return chart.show_png()


# ── chart registry ────────────────────────────────────────────────────────────

CHART_TYPES: list[tuple[str, str, Callable]] = [
    ("line",       "Multi-Series Line",    build_line),
    ("scatter",    "Cluster Scatter",      build_scatter),
    ("bar",        "Grouped Bar",          build_bar),
    ("heatmap",    "Correlation Heatmap",  build_heatmap),
    ("boxplot",    "Box Plot",             build_boxplot),
    ("faceted",    "Faceted Scatter",      build_faceted),
    ("histogram",  "Histogram",            build_histogram),
]


# ── rendering ─────────────────────────────────────────────────────────────────

def render_all() -> dict[tuple[str, str], Path]:
    """Render all (chart_type, theme) combos; return mapping → png path."""
    results: dict[tuple[str, str], Path] = {}
    errors: list[str] = []

    for chart_key, chart_label, builder in CHART_TYPES:
        for theme_key, theme_obj, theme_label in THEMES:
            out_path = PNG_DIR / f"{chart_key}_{theme_key}.png"
            print(f"  {chart_label:25s} × {theme_label:15s} → {out_path.name} ...", end="", flush=True)
            try:
                png = builder(theme_obj)
                out_path.write_bytes(png)
                results[(chart_key, theme_key)] = out_path
                print(" OK")
            except Exception as exc:
                print(f" ERROR: {exc}")
                errors.append(f"{chart_label} × {theme_label}: {exc}")
                results[(chart_key, theme_key)] = None

    if errors:
        print("\nFailed renders:")
        for e in errors:
            print(f"  - {e}")

    return results


# ── HTML generation ───────────────────────────────────────────────────────────

_THEME_ACCENTS = {
    "paper_ink":     ("#FAF7F2", "#1F2937", "#2563EB"),   # bg, text, accent
    "slate_citrus":  ("#111827", "#E5E7EB", "#84CC16"),
    "arctic_signal": ("#F8FAFC", "#0F172A", "#0284C7"),
}

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #F5F5F5;
    color: #1a1a1a;
    padding: 32px 24px 64px;
}
h1 {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.subtitle {
    font-size: 0.95rem;
    color: #666;
    margin-bottom: 40px;
}
.grid {
    display: grid;
    gap: 0;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
    background: white;
}
.header-row {
    display: grid;
    grid-template-columns: 180px repeat(3, 1fr);
    background: #1F2937;
    color: white;
}
.header-cell {
    padding: 14px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.header-cell.theme-paper-ink   { background: #FAF7F2; color: #1F2937; border-left: 4px solid #2563EB; }
.header-cell.theme-slate        { background: #111827; color: #E5E7EB; border-left: 4px solid #84CC16; }
.header-cell.theme-arctic       { background: #F8FAFC; color: #0F172A; border-left: 4px solid #0284C7; }
.chart-row {
    display: grid;
    grid-template-columns: 180px repeat(3, 1fr);
    border-top: 1px solid #e8e8e8;
}
.chart-row:nth-child(even) .row-label { background: #f9f9f9; }
.chart-row:nth-child(odd)  .row-label { background: #f3f3f3; }
.row-label {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #444;
    letter-spacing: 0.01em;
    border-right: 1px solid #e8e8e8;
}
.chart-cell {
    padding: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-left: 1px solid #e8e8e8;
    background: #fafafa;
}
.chart-cell img {
    max-width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
}
.chart-cell.missing {
    color: #999;
    font-size: 0.8rem;
    font-style: italic;
    min-height: 120px;
}
.footer {
    margin-top: 32px;
    font-size: 0.78rem;
    color: #aaa;
    text-align: center;
}
"""


def _img_tag(png_path: Path | None, chart_label: str, theme_label: str) -> str:
    if png_path is None or not png_path.exists():
        return f'<div class="chart-cell missing">render failed</div>'
    rel = png_path.relative_to(HERE)
    alt = f"{chart_label} — {theme_label}"
    return f'<div class="chart-cell"><img src="{rel}" alt="{alt}" loading="lazy"></div>'


def generate_html(results: dict[tuple[str, str], Path]) -> str:
    theme_header_classes = {
        "paper_ink":     "theme-paper-ink",
        "slate_citrus":  "theme-slate",
        "arctic_signal": "theme-arctic",
    }

    header_cells = ['<div class="header-cell" style="background:#1F2937"></div>']
    for tk, _, tl in THEMES:
        cls = theme_header_classes[tk]
        header_cells.append(f'<div class="header-cell {cls}">{tl}</div>')

    chart_rows = []
    for chart_key, chart_label, _ in CHART_TYPES:
        cells = [f'<div class="row-label">{chart_label}</div>']
        for theme_key, _, theme_label in THEMES:
            png_path = results.get((chart_key, theme_key))
            cells.append(_img_tag(png_path, chart_label, theme_label))
        chart_rows.append(
            f'<div class="chart-row">{"".join(cells)}</div>'
        )

    grid_inner = (
        f'<div class="header-row">{"".join(header_cells)}</div>'
        + "".join(chart_rows)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ferrum Themes — Visual Demo</title>
<style>
{_HTML_CSS}
</style>
</head>
<body>
<h1>Ferrum Custom Themes</h1>
<p class="subtitle">
  Side-by-side comparison of <strong>Paper Ink</strong>, <strong>Slate Citrus</strong>,
  and <strong>Arctic Signal</strong> across 7 chart types.
  Each column renders the same data with a different theme applied.
</p>
<div class="grid">
{grid_inner}
</div>
<p class="footer">Generated by design-docs/themes/demo_themes.py · ferrum v{fm.__version__}</p>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"ferrum v{fm.__version__}")
    print(f"Output directory: {PNG_DIR}")
    print(f"Rendering {len(CHART_TYPES)} chart types × {len(THEMES)} themes ...\n")

    results = render_all()

    n_ok = sum(1 for v in results.values() if v is not None)
    n_total = len(results)
    print(f"\n{n_ok}/{n_total} renders succeeded.")

    print(f"\nGenerating HTML → {HTML_OUT} ...", end="", flush=True)
    html = generate_html(results)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(" done.")

    print(f"\nOpen in browser:\n  open {HTML_OUT}")


if __name__ == "__main__":
    main()
