"""A composition refuses two different definitions under one selection name.

One name is one selection across a composition, so linked views must declare
the same definition. The refusal reaches the public render call as a
``ValueError`` naming the selection, the declaring charts by their position
in the composition, and each differing field with both values as Python
literals.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 1.0]})

_ADVICE = (
    "One name is one selection across the composition: share one selection between "
    "the charts, or give each its own name."
)


def _declaring(selection: fm.Selection) -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y").add_selection(selection)


def test_hconcat_refuses_a_redefined_selection_naming_both_charts() -> None:
    first = _declaring(fm.selection_point(name="s"))
    second = _declaring(fm.selection_point(name="s", nearest=True))
    expected = (
        "hconcat: selection 's' is declared twice with different definitions: chart 1 of "
        "hconcat declares nearest=False, chart 2 of hconcat declares nearest=True. " + _ADVICE
    )
    with pytest.raises(ValueError) as err:
        (first | second).to_svg()
    assert str(err.value) == expected


def test_a_redefinition_inside_a_nested_layerchart_names_its_layer() -> None:
    plain = fm.Chart(_DF).mark_line().encode(x="x", y="y")
    layered = fm.LayerChart(plain, _declaring(fm.selection_point(name="s")))
    other = _declaring(fm.selection_interval(name="s"))
    expected = (
        "vconcat: selection 's' is declared twice with different definitions: layer 2 of "
        "overlay in chart 1 of vconcat declares type='point', chart 2 of vconcat declares "
        "type='interval'. " + _ADVICE
    )
    with pytest.raises(ValueError) as err:
        (layered & other).to_svg()
    assert str(err.value) == expected


def test_identical_declarations_link_without_refusal() -> None:
    shared = fm.selection_point(name="s")
    svg = (_declaring(shared) | _declaring(fm.selection_point(name="s"))).to_svg()
    assert svg.count("<circle") == 6
