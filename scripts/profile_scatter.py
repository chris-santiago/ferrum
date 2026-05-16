"""Profile ferrum vs Altair vs seaborn: scatter with tooltips.

Runs at multiple scales (200k and 1M by default). Measures render time
(wall clock, median of 3 runs) and output file size for SVG, PNG, and
interactive HTML formats.
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
        svg = _chart().show_svg()
        times_svg.append(time.perf_counter() - t0)
    svg_path.write_text(svg)

    times_png = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        png_bytes = _chart().show_png()
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


def print_table(n: int, fm: dict, alt: dict, sns: dict):
    crashed_alt = alt.get("crashed", False)
    W = 80
    print()
    print(f"  {n:,} points (median of {RUNS} runs)")
    print("=" * W)
    print(f"{'Metric':<32} {'Ferrum':>15} {'Altair':>15} {'Seaborn':>15}")
    print("-" * W)

    def t(d, k):
        return "OOM" if d.get("crashed") else fmt_time(d.get(k))

    def s(d, k):
        return "OOM" if d.get("crashed") else fmt_size(d.get(k))

    print(f"{'SVG render time':<32} {fmt_time(fm['svg_time']):>15} {t(alt, 'svg_time'):>15} {fmt_time(sns.get('svg_time')):>15}")
    print(f"{'SVG file size':<32} {fmt_size(fm['svg_size']):>15} {s(alt, 'svg_size'):>15} {fmt_size(sns.get('svg_size')):>15}")
    print(f"{'PNG render time':<32} {fmt_time(fm['png_time']):>15} {'—':>15} {fmt_time(sns.get('png_time')):>15}")
    print(f"{'PNG file size':<32} {fmt_size(fm['png_size']):>15} {'—':>15} {fmt_size(sns.get('png_size')):>15}")
    print(f"{'HTML render+save':<32} {fmt_time(fm['html_time']):>15} {t(alt, 'html_time'):>15} {'—':>15}")
    print(f"{'HTML file size':<32} {fmt_size(fm['html_size']):>15} {s(alt, 'html_size'):>15} {'—':>15}")
    if not crashed_alt:
        print(f"{'Altair spec gen':<32} {'—':>15} {fmt_time(alt.get('spec_gen_time')):>15} {'—':>15}")
    else:
        print(f"{'Altair status':<32} {'—':>15} {'OOM (exit ' + str(alt['exit_code']) + ')':>15} {'—':>15}")
    print("=" * W)


if __name__ == "__main__":
    print(f"Temp dir: {tmp}")

    for n in SCALES:
        print(f"\n--- {n:,} points ---")
        print("  ferrum (raster)...", end=" ", flush=True)
        fm = bench_ferrum(n, force_raster=True)
        print("done")
        print("  altair...", end=" ", flush=True)
        alt = bench_altair(n)
        print("done" if not alt.get("crashed") else f"OOM (exit {alt['exit_code']})")
        print("  seaborn...", end=" ", flush=True)
        sns = bench_seaborn(n)
        print("done" if not sns.get("crashed") else f"crashed (exit {sns['exit_code']})")
        print_table(n, fm, alt, sns)

    print(f"\nAll files in {tmp}")
