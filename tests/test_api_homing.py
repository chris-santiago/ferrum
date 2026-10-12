"""Regression guard for API-reference page homing (MOD-08).

``ferrum.Grid`` is a sibling value class to ``Axis`` (axis.md) and ``Legend``
(legend.md), so it must own its own ``grid.md`` page rather than being filed
onto the ``structural`` page next to ``BreakAxis`` / ``Inset`` / ``SecondaryY``.
These tests assert both the runtime fact (``Grid.__module__``) and the
``scripts/gen_api_pages.py`` partition that drives the generated docs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

import ferrum as fm
from tests._script_loader import load_script

_GEN_API = Path(__file__).resolve().parent.parent / "scripts" / "gen_api_pages.py"


@pytest.fixture(scope="module")
def gen_api() -> ModuleType:
    """Import scripts/gen_api_pages.py as a module (it is a plain script)."""
    return load_script("gen_api_pages", _GEN_API)


def test_grid_defining_module() -> None:
    assert fm.Grid.__module__ == "ferrum.grid"


def test_grid_homes_to_grid_page_not_structural(gen_api: ModuleType) -> None:
    assert gen_api.page_for("Grid") == "grid"


def test_grid_page_exists_in_metadata(gen_api: ModuleType) -> None:
    assert "grid" in gen_api.PAGES


def test_structural_page_excludes_grid(gen_api: ModuleType) -> None:
    by_page, _ = gen_api.build_partition()
    assert "Grid" not in by_page.get("structural", [])
    assert by_page.get("grid") == ["Grid"]
