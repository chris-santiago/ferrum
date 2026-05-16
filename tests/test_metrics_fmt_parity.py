"""Schwabish C9 audit-rework (2026-05-12) — parity regression test.

The corner-metrics annotation can be emitted from two paths:

1. **Python**: ``ferrum._metrics_fmt.format_corner_metrics(r2, rmse, mae)``
   used by ``_inject_metrics_corner`` in the residuals chart builder.
2. **Rust**: ``residuals::append_metrics_columns`` in
   ``crates/ferrum-core/src/transform/residuals.rs``, used by the
   Smooth and Robust transforms when ``inject_metrics=True``.

The two implementations are deliberate near-duplicates because the format
spec lives on either side of the PyO3 FFI. This test pins them to the same
output on a known numeric fixture so any future format-string change is
caught loudly instead of producing a silent visual divergence between the
residplot path (Python) and the lmplot path (Rust).
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum
from ferrum import Smooth, Chart
from ferrum._metrics_fmt import format_corner_metrics


def _rust_metrics_text(seed: int = 0) -> str:
    """Render a Smooth-fitted chart with inject_metrics and extract the corner text."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 50)
    y = 0.5 * x + rng.normal(0, 0.5, 50)
    df = pl.DataFrame({"x": x, "y": y})
    line_layer = (
        Chart(df)
        .mark_line()
        .encode(x="x", y="y")
        .transform(Smooth(x="x", y="y", method="lm", ci=0.95, inject_metrics=True, output="fitted"))
    )
    text_layer = (
        Chart(df)
        .mark_text(align="right")
        .encode(x="x", y="_metrics_y", text="_metrics_text")
        .transform(Smooth(x="x", y="y", method="lm", ci=0.95, inject_metrics=True, output="fitted"))
    )
    svg = (line_layer + text_layer).show_svg()
    # With multiline tspan support, metric text may be split across <tspan>
    # elements. Extract both plain <text>content</text> and <text><tspan>...
    # </tspan></text> forms, joining tspan lines with \n.
    metrics = []
    for m in re.finditer(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL):
        inner = m.group(1)
        tspan_lines = re.findall(r"<tspan[^>]*>([^<]+)</tspan>", inner)
        text = "\n".join(tspan_lines) if tspan_lines else inner
        if "R" in text and "RMSE" in text and "MAE" in text:
            metrics.append(text)
    assert metrics, "Rust path did not emit a metric-text element"
    return metrics[0]


def _python_metrics_text(seed: int = 0) -> str:
    """Compute metrics in Python via the same fixture; return formatted text."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 50)
    y = 0.5 * x + rng.normal(0, 0.5, 50)
    # OLS fit via closed form (matches the Smooth Lm path).
    mean_x = x.mean()
    mean_y = y.mean()
    sxx = ((x - mean_x) ** 2).sum()
    sxy = ((x - mean_x) * (y - mean_y)).sum()
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x
    resid = y - (alpha + beta * x)
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - mean_y) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt((resid**2).mean()))
    mae = float(np.abs(resid).mean())
    return format_corner_metrics(r2, rmse, mae)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_rust_python_corner_metrics_byte_identical(seed):
    """Schwabish C9 regression — same input ⇒ same output bytes."""
    rust = _rust_metrics_text(seed)
    py = _python_metrics_text(seed)
    assert rust == py, (
        f"Format-string drift between Rust and Python corner-metrics paths!\n"
        f"  rust : {rust!r}\n"
        f"  python: {py!r}\n"
        f"Both should go through the same template "
        f"('R² {{r2:.3f}}\\nRMSE {{rmse:.3f}}\\nMAE {{mae:.3f}}'). "
        f"If you intentionally changed one side, change the other to match "
        f"and update format_corner_metrics in src/ferrum/_metrics_fmt.py."
    )


def test_format_corner_metrics_basic():
    """Sanity-check the formatter on hand-chosen inputs."""
    s = format_corner_metrics(0.701234, 0.928, 0.680)
    assert s == "R² 0.701\nRMSE 0.928\nMAE 0.680"
