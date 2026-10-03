"""Test-suite-wide fixtures."""

from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def pairplot_df() -> pl.DataFrame:
    """Small hue-bearing DataFrame shared by the composite-shared-legend tests."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "z": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
            "grp": ["a", "b", "a", "b", "a", "b"],
        }
    )


@pytest.fixture
def curved_xy_df() -> pl.DataFrame:
    """Genuinely curved `(x, y)` data (a real quadratic term), shared by the
    batch-D T9 smooth/lmplot/residplot `degree`/`order` tests
    (`tests/test_smooth_methods.py`, `tests/test_lmplot_residplot_order.py`).
    Near-straight-line data makes a linear and a quadratic fit coincide to
    the point of byte-identical SVG output, which defeats any test relying
    on them differing."""
    xs = [i * 0.3 for i in range(20)]
    ys = [1.0 + 2.0 * x + 3.0 * x**2 for x in xs]
    return pl.DataFrame({"x": xs, "y": ys})


@pytest.fixture(scope="module")
def browser():
    """One headless system-Chrome per module for the ``wasm_pixels`` captures.

    Skips when Playwright cannot launch the system Chrome channel; checked
    here, lazily, so collecting the suite never launches a browser.
    """
    from tests import _pixel_capture as pc

    if not pc.chrome_available():
        pytest.skip("system Chrome is not available")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=pc.LAUNCH_ARGS)
        yield b
        b.close()


# umap removed from Python deps in Phase 11d — handled at Rust level (manifolds-rs).
_REQUIRED_FOR_PHASE_10 = ("sklearn", "shap", "scipy", "skops")


def pytest_configure(config):  # noqa: ARG001
    """Fail early at collection time (xdist-safe) if required extras are missing."""
    missing = []
    for mod in _REQUIRED_FOR_PHASE_10:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise pytest.UsageError(
            f"Phase 10 test suite requires: {', '.join(missing)}. "
            f"Install with `uv sync --extra ml-all` plus dev deps."
        )

    import sklearn

    pin_file = Path(__file__).parent / "fixtures" / "SKLEARN_VERSION"
    pinned = pin_file.read_text().strip()
    if sklearn.__version__ != pinned:
        raise pytest.UsageError(
            f"sklearn=={sklearn.__version__} installed but fixtures pinned to "
            f"sklearn=={pinned} (tests/fixtures/SKLEARN_VERSION). "
            f"Run `uv pip install scikit-learn=={pinned}` or bump the pin "
            f"and regenerate fixtures via `python tests/fixtures/build.py`."
        )
