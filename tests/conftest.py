"""Test-suite-wide fixtures."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


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
