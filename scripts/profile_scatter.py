"""Profile ferrum vs Altair vs seaborn vs Plotly vs plotnine: scatter with tooltips.

Runs at multiple scales (200k and 1M by default). Measures render time
(wall clock, median of 3 runs) and output file size for SVG, PNG, and
interactive HTML formats.

Usage:
    python scripts/profile_scatter.py                  # run all libraries
    python scripts/profile_scatter.py --only plotnine  # run one library
    python scripts/profile_scatter.py --only ferrum,plotnine  # run a subset
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

SCALES = [200_000, 1_000_000]
RUNS = 3
ALL_LIBS = ["ferrum", "altair", "seaborn", "plotly", "plotnine"]

tmp = Path(tempfile.mkdtemp(prefix="ferrum_profile_"))


def make_data(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "label": [f"pt {i}" for i in range(n)],
    })


# ── ferrum ──────────────────────────────────────────────────────────
def bench_ferrum(n: int, *, force_raster: bool = False) -> dict:
    import ferrum as fm
    from ferrum.render_config import RenderConfig

    df = make_data(n)
    tag = f"ferrum_{n}" + ("_raster" if force_raster else "")
    svg_path = tmp / f"{tag}.svg"
    png_path = tmp / f"{tag}.png"
    html_path = tmp / f"{tag}.html"

    rc = RenderConfig(raster_threshold=1, raster_behavior="silent") if force_raster else None

    def _chart():
        c = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                tooltip=fm.Tooltip(
                    fm.TooltipField("x", format=".2f"),
                    fm.TooltipField("y", format=".2f"),
                    "label",
                ),
            )
            .properties(width=500, height=400, title=f"Scatter — {n:,} points")
        )
        if rc:
            c = c.properties(render_config=rc)
        return c

    times_svg = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        svg = _chart().to_svg()
        times_svg.append(time.perf_counter() - t0)
    svg_path.write_text(svg)

    times_png = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        png_bytes = _chart().to_png()
        times_png.append(time.perf_counter() - t0)
    png_path.write_bytes(png_bytes)

    times_html = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _chart().interactive().save(str(html_path))
        times_html.append(time.perf_counter() - t0)

    return {
        "svg_time": sorted(times_svg)[RUNS // 2],
        "svg_size": svg_path.stat().st_size,
        "png_time": sorted(times_png)[RUNS // 2],
        "png_size": png_path.stat().st_size,
        "html_time": sorted(times_html)[RUNS // 2],
        "html_size": html_path.stat().st_size,
    }


# ── altair (isolated subprocess) ────────────────────────────────────
ALTAIR_SCRIPT = r'''
# /// script
# requires-python = ">=3.10"
# dependencies = ["altair>=5", "vl-convert-python", "polars", "numpy"]
# ///
import json, os, sys, time
import altair as alt, numpy as np, polars as pl
import vl_convert as vlc

alt.data_transformers.disable_max_rows()
N = int(sys.argv[2])
RUNS = 3
rng = np.random.default_rng(42)
df = pl.DataFrame({"x": rng.normal(0,1,N), "y": rng.normal(0,1,N),
                    "label": [f"pt {i}" for i in range(N)]})
tmp = sys.argv[1]

def _chart():
    return (alt.Chart(df).mark_point()
            .encode(x="x:Q", y="y:Q",
                    tooltip=[alt.Tooltip("x:Q",format=".2f"),
                             alt.Tooltip("y:Q",format=".2f"),"label:N"])
            .properties(width=500, height=400, title=f"Scatter — {N:,} points"))

# Spec gen
times_spec = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    spec = _chart().to_dict()
    times_spec.append(time.perf_counter() - t0)

# SVG
svg_time = svg_size = None
svg_path = f"{tmp}/altair_{N}.svg"
try:
    t0 = time.perf_counter()
    svg_bytes = vlc.vegalite_to_svg(spec)
    svg_time = time.perf_counter() - t0
    with open(svg_path, "w") as f: f.write(svg_bytes)
    svg_size = os.path.getsize(svg_path)
except Exception as e:
    print(f"SVG failed: {e}", file=sys.stderr)

# HTML
html_time = html_size = None
html_path = f"{tmp}/altair_{N}.html"
try:
    times_html = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _chart().interactive().save(html_path)
        times_html.append(time.perf_counter() - t0)
    html_time = sorted(times_html)[RUNS // 2]
    html_size = os.path.getsize(html_path)
except Exception as e:
    print(f"HTML failed: {e}", file=sys.stderr)

print(json.dumps({"spec_gen_time": sorted(times_spec)[RUNS//2],
                   "svg_time": svg_time, "svg_size": svg_size,
                   "html_time": html_time, "html_size": html_size}))
'''


def bench_altair(n: int) -> dict:
    script_path = tmp / "_altair_bench.py"
    script_path.write_text(ALTAIR_SCRIPT)
    result = subprocess.run(
        ["uv", "run", "--no-project", "--script", str(script_path), str(tmp), str(n)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  Altair exited {result.returncode}", file=sys.stderr)
        return {"crashed": True, "exit_code": result.returncode}
    for line in result.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON in Altair output:\n{result.stdout}")


# ── seaborn (isolated subprocess) ───────────────────────────────────
SEABORN_SCRIPT = r'''
# /// script
# requires-python = ">=3.10"
# dependencies = ["seaborn>=0.13", "matplotlib>=3.8", "numpy"]
# ///
import json, os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, seaborn as sns

N = int(sys.argv[2])
RUNS = 3
rng = np.random.default_rng(42)
x, y = rng.normal(0,1,N), rng.normal(0,1,N)
tmp = sys.argv[1]

# SVG
times_svg = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(6.25, 5))
    sns.scatterplot(x=x, y=y, s=3, alpha=0.3, ax=ax)
    ax.set_title(f"Scatter — {N:,} points")
    fig.savefig(f"{tmp}/seaborn_{N}.svg", format="svg")
    plt.close(fig)
    times_svg.append(time.perf_counter() - t0)

# PNG
times_png = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(6.25, 5))
    sns.scatterplot(x=x, y=y, s=3, alpha=0.3, ax=ax)
    ax.set_title(f"Scatter — {N:,} points")
    fig.savefig(f"{tmp}/seaborn_{N}.png", format="png", dpi=100)
    plt.close(fig)
    times_png.append(time.perf_counter() - t0)

print(json.dumps({
    "svg_time": sorted(times_svg)[RUNS//2],
    "svg_size": os.path.getsize(f"{tmp}/seaborn_{N}.svg"),
    "png_time": sorted(times_png)[RUNS//2],
    "png_size": os.path.getsize(f"{tmp}/seaborn_{N}.png"),
}))
'''


def bench_seaborn(n: int) -> dict:
    script_path = tmp / "_seaborn_bench.py"
    script_path.write_text(SEABORN_SCRIPT)
    result = subprocess.run(
        ["uv", "run", "--no-project", "--script", str(script_path), str(tmp), str(n)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  Seaborn exited {result.returncode}", file=sys.stderr)
        return {"crashed": True, "exit_code": result.returncode}
    for line in result.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON in seaborn output:\n{result.stdout}")


# ── plotly (isolated subprocess) ────────────────────────────────────
PLOTLY_SCRIPT = r'''
# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly>=5", "kaleido>=0.2", "numpy"]
# ///
import json, os, sys, time
import numpy as np, plotly.graph_objects as go

N = int(sys.argv[2])
RUNS = 3
rng = np.random.default_rng(42)
x, y = rng.normal(0,1,N), rng.normal(0,1,N)
tmp = sys.argv[1]

def _fig():
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers",
                                  marker=dict(size=3, opacity=0.3)))
    fig.update_layout(width=500, height=400, title=f"Scatter — {N:,} points")
    return fig

# SVG
svg_path = f"{tmp}/plotly_{N}.svg"
times_svg = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    _fig().write_image(svg_path, format="svg")
    times_svg.append(time.perf_counter() - t0)

# PNG
png_path = f"{tmp}/plotly_{N}.png"
times_png = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    _fig().write_image(png_path, format="png")
    times_png.append(time.perf_counter() - t0)

# HTML
html_path = f"{tmp}/plotly_{N}.html"
times_html = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    _fig().write_html(html_path)
    times_html.append(time.perf_counter() - t0)

print(json.dumps({
    "svg_time": sorted(times_svg)[RUNS//2],
    "svg_size": os.path.getsize(svg_path),
    "png_time": sorted(times_png)[RUNS//2],
    "png_size": os.path.getsize(png_path),
    "html_time": sorted(times_html)[RUNS//2],
    "html_size": os.path.getsize(html_path),
}))
'''


def bench_plotly(n: int) -> dict:
    script_path = tmp / "_plotly_bench.py"
    script_path.write_text(PLOTLY_SCRIPT)
    result = subprocess.run(
        ["uv", "run", "--no-project", "--script", str(script_path), str(tmp), str(n)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  Plotly exited {result.returncode}", file=sys.stderr)
        return {"crashed": True, "exit_code": result.returncode}
    for line in result.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON in Plotly output:\n{result.stdout}")


# ── plotnine (isolated subprocess) ─────────────────────────────────
PLOTNINE_SCRIPT = r'''
# /// script
# requires-python = ">=3.10"
# dependencies = ["plotnine>=0.13", "matplotlib>=3.8", "numpy", "pandas"]
# ///
import json, os, sys, time
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
from plotnine import ggplot, aes, geom_point, labs, theme, element_text

N = int(sys.argv[2])
RUNS = 3
rng = np.random.default_rng(42)
df = pd.DataFrame({"x": rng.normal(0,1,N), "y": rng.normal(0,1,N)})
tmp = sys.argv[1]

def _plot():
    return (ggplot(df, aes("x", "y"))
            + geom_point(size=0.5, alpha=0.3)
            + labs(title=f"Scatter — {N:,} points"))

# SVG
svg_path = f"{tmp}/plotnine_{N}.svg"
times_svg = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    _plot().save(svg_path, format="svg", width=6.25, height=5, dpi=100, verbose=False)
    times_svg.append(time.perf_counter() - t0)

# PNG
png_path = f"{tmp}/plotnine_{N}.png"
times_png = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    _plot().save(png_path, format="png", width=6.25, height=5, dpi=100, verbose=False)
    times_png.append(time.perf_counter() - t0)

print(json.dumps({
    "svg_time": sorted(times_svg)[RUNS//2],
    "svg_size": os.path.getsize(svg_path),
    "png_time": sorted(times_png)[RUNS//2],
    "png_size": os.path.getsize(png_path),
}))
'''


def bench_plotnine(n: int) -> dict:
    script_path = tmp / "_plotnine_bench.py"
    script_path.write_text(PLOTNINE_SCRIPT)
    result = subprocess.run(
        ["uv", "run", "--no-project", "--script", str(script_path), str(tmp), str(n)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  plotnine exited {result.returncode}", file=sys.stderr)
        return {"crashed": True, "exit_code": result.returncode}
    for line in result.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON in plotnine output:\n{result.stdout}")


# ── display ─────────────────────────────────────────────────────────
def fmt_size(b: int | None) -> str:
    if b is None:
        return "—"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"


def fmt_time(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 1:
        return f"{s * 1000:.0f} ms"
    return f"{s:.2f} s"


BENCH_FUNCS = {
    "ferrum": lambda n: bench_ferrum(n, force_raster=True),
    "altair": bench_altair,
    "seaborn": bench_seaborn,
    "plotly": bench_plotly,
    "plotnine": bench_plotnine,
}

DISPLAY_NAMES = {
    "ferrum": "Ferrum",
    "altair": "Altair",
    "seaborn": "Seaborn",
    "plotly": "Plotly",
    "plotnine": "plotnine",
}

METRICS = [
    ("SVG render time", "svg_time", fmt_time),
    ("SVG file size", "svg_size", fmt_size),
    ("PNG render time", "png_time", fmt_time),
    ("PNG file size", "png_size", fmt_size),
    ("HTML render+save", "html_time", fmt_time),
    ("HTML file size", "html_size", fmt_size),
]


def print_table(n: int, results: dict[str, dict], libs: list[str]):
    col_w = 15
    label_w = 32
    header_libs = [DISPLAY_NAMES[lib] for lib in libs]
    W = label_w + col_w * len(libs)
    print()
    print(f"  {n:,} points (median of {RUNS} runs)")
    print("=" * W)
    print(f"{'Metric':<{label_w}}" + "".join(f"{name:>{col_w}}" for name in header_libs))
    print("-" * W)

    for label, key, formatter in METRICS:
        vals = []
        for lib in libs:
            d = results.get(lib, {})
            if d.get("crashed"):
                vals.append("OOM")
            else:
                vals.append(formatter(d.get(key)))
        print(f"{label:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in vals))

    for lib in libs:
        d = results.get(lib, {})
        if d.get("crashed"):
            print(f"{DISPLAY_NAMES[lib] + ' status':<{label_w}}" +
                  "".join(f"{'OOM (exit ' + str(d['exit_code']) + ')' if l == lib else '—':>{col_w}}" for l in libs))

    print("=" * W)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark scatter plot rendering")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of libraries to benchmark (e.g. --only plotnine,ferrum)")
    args = parser.parse_args()

    if args.only:
        libs = [lib.strip() for lib in args.only.split(",")]
        for lib in libs:
            if lib not in BENCH_FUNCS:
                parser.error(f"Unknown library: {lib!r}. Choose from: {', '.join(ALL_LIBS)}")
    else:
        libs = ALL_LIBS

    print(f"Temp dir: {tmp}")
    print(f"Libraries: {', '.join(libs)}")

    for n in SCALES:
        print(f"\n--- {n:,} points ---")
        results: dict[str, dict] = {}
        for lib in libs:
            label = DISPLAY_NAMES[lib]
            print(f"  {label}...", end=" ", flush=True)
            result = BENCH_FUNCS[lib](n)
            crashed = result.get("crashed", False)
            print("done" if not crashed else f"OOM (exit {result['exit_code']})")
            results[lib] = result
        print_table(n, results, libs)

    print(f"\nAll files in {tmp}")
