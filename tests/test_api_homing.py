"""Regression guard for API-reference page homing (MOD-08).

``ferrum.Grid`` is a sibling value class to ``Axis`` (axis.md) and ``Legend``
(legend.md), so it must own its own ``grid.md`` page rather than being filed
onto the ``structural`` page next to ``BreakAxis`` / ``Inset`` / ``SecondaryY``.
These tests assert both the runtime fact (``Grid.__module__``) and the
``scripts/gen_api_pages.py`` partition that drives the generated docs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import ferrum as fm

_GEN_API = Path(__file__).resolve().parent.parent / "scripts" / "gen_api_pages.py"


@pytest.fixture(scope="module")
def gen_api() -> ModuleType:
    """Import scripts/gen_api_pages.py as a module (it is a plain script)."""
    spec = importlib.util.spec_from_file_location("gen_api_pages", _GEN_API)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
