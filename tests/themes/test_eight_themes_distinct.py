"""All 8 builtin themes render the same chart visibly differently.

Each builtin overrides a different subset of theme keys (see
``src/ferrum/themes/builtins.py``); rendering the same chart under each
must produce 8 byte-distinct SVGs. The per-theme goldens in
``tests/goldens/theme_gallery/`` lock in the visual identity.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from ferrum.themes import (
    dark,
    default,
    economist,
    fivethirtyeight,
    minimal,
    publication,
    solarized_dark,
    solarized_light,
)

THEMES: dict[str, fm.Theme] = {
    "default": default,
    "minimal": minimal,
    "dark": dark,
    "publication": publication,
    "economist": economist,
    "fivethirtyeight": fivethirtyeight,
    "solarized_light": solarized_light,
    "solarized_dark": solarized_dark,
}

_GOLDEN_ROOT = Path(__file__).parent.parent / "goldens" / "theme_gallery"
_REGENERATE = bool(os.environ.get("FERRUM_UPDATE_GOLDENS")) or bool(
    os.environ.get("FERRUM_REGENERATE_GOLDENS")
)


def _base_chart() -> fm.Chart:
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 15.0, 25.0]})
    return fm.Chart(df).mark_bar().encode(x="cat", y="val", color="cat")


def test_all_eight_themes_produce_distinct_svgs() -> None:
    chart = _base_chart()
    svgs = {name: chart.theme(theme).to_svg() for name, theme in THEMES.items()}
    hashes = {name: hashlib.sha256(svg.encode()).hexdigest()[:16] for name, svg in svgs.items()}
    assert len(set(hashes.values())) == 8, (
        f"expected 8 distinct theme hashes, got duplicates: {hashes}"
    )


@pytest.mark.parametrize("name", list(THEMES.keys()))
def test_each_theme_golden(name: str) -> None:
    chart = _base_chart()
    theme = THEMES[name]
    svg = chart.theme(theme).to_svg()
    golden_path = _GOLDEN_ROOT / f"{name}.svg"
    if _REGENERATE or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(svg)
        if not _REGENERATE:
            pytest.skip(f"created new golden at {golden_path}; rerun to verify")
        return
    expected = golden_path.read_text()
    assert svg == expected, (
        f"golden mismatch for theme {name!r}. "
        f"Set FERRUM_UPDATE_GOLDENS=1 to regenerate after intentional changes."
    )
