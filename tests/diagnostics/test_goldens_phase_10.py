"""Phase 10 SVG golden tests.

Single tier — all goldens render at the renderer's default 3-decimal-place
quantization (``fmt_f`` in ``crates/ferrum-core/src/render/svg.rs``). The
original plan proposed a tiered byte-identical / quantized split; that was
collapsed once it became clear the renderer already quantizes everything
via ``FLOAT_PRECISION = 3``.

Regenerate with ``FERRUM_REGENERATE_GOLDENS=1 pytest tests/diagnostics/test_goldens_phase_10.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture

_GOLDEN_ROOT = Path(__file__).parent.parent / "goldens" / "phase_10"
_REGENERATE = bool(os.environ.get("FERRUM_REGENERATE_GOLDENS"))


def _check_golden(svg: str, name: str) -> None:
    path = _GOLDEN_ROOT / f"{name}.svg"
    if _REGENERATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        if not _REGENERATE:
            pytest.skip(f"created new golden at {path}; rerun to verify")
        return
    expected = path.read_text()
    assert svg == expected, (
        f"Golden mismatch for {name!r}. "
        f"Set FERRUM_REGENERATE_GOLDENS=1 to regenerate after intentional changes."
    )


# --- 10a goldens ---


def test_golden_residuals_chart_regression():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"])
    svg = chart.show_svg()
    _check_golden(svg, "residuals_chart_regression")


def test_golden_prediction_error_regression():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    chart = _prediction_error_chart_from_source(source)
    svg = chart.show_svg()
    _check_golden(svg, "prediction_error_regression")
