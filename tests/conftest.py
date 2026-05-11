"""Test-suite-wide fixtures."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


_REQUIRED_FOR_PHASE_10 = ("sklearn", "shap", "umap", "scipy", "skops")


@pytest.fixture(scope="session", autouse=True)
def _require_phase_10_extras():
    """Phase 10 tests assume the full ml-all extras + dev deps are installed."""
    missing = []
    for mod in _REQUIRED_FOR_PHASE_10:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        pytest.exit(
            f"Phase 10 test suite requires: {', '.join(missing)}. "
            f"Install with `uv sync --extra ml-all` plus dev deps.",
            returncode=1,
        )

    # Verify sklearn version matches the fixture pin.
    import sklearn
    pin_file = Path(__file__).parent / "fixtures" / "SKLEARN_VERSION"
    pinned = pin_file.read_text().strip()
    if sklearn.__version__ != pinned:
        pytest.exit(
            f"sklearn=={sklearn.__version__} installed but fixtures pinned to "
            f"sklearn=={pinned} (tests/fixtures/SKLEARN_VERSION). "
            f"Run `uv pip install scikit-learn=={pinned}` or bump the pin "
            f"and regenerate fixtures via `python tests/fixtures/build.py`.",
            returncode=1,
        )
    yield
