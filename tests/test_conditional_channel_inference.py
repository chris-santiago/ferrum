"""A conditional declared without a channel takes it from its branches, not from one side.

`Chart.conditional(sel.when(a).otherwise(b))` names no channel. An encoding
channel on either branch names it; otherwise the first literal does (a number
is an opacity, anything else a color). The literal on each branch is then typed
by that channel: a number on a color conditional is refused, not re-typed as an
opacity. A channel no conditional can target is refused by name.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.selection import _resolve_encoding_value, selection_point, value

DF = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "v": [3.0, 4.0], "g": ["a", "b"]})


def _chart():
    return fm.Chart(DF).mark_point().encode(x="x", y="y")


def _conditional_of(chart) -> dict:
    (cond,) = chart.to_dict()["conditionals"]
    return cond


def test_channel_named_by_the_if_not_branch():
    sel = selection_point(name="s_size")
    chart = _chart().add_selection(sel).conditional(sel.when(value(20.0)).otherwise(fm.Size("v")))
    cond = _conditional_of(chart)
    assert cond["channel"] == "size"
    assert cond["if_selected"] == {"kind": "size", "value": 20.0}


def test_channel_named_by_the_if_selected_branch_still_wins_for_literals():
    sel = selection_point(name="s_op")
    chart = _chart().add_selection(sel).conditional(sel.when(value(1.0)).otherwise(value(0.2)))
    cond = _conditional_of(chart)
    assert cond["channel"] == "opacity"
    assert cond["if_not"] == {"kind": "opacity", "value": 0.2}


def test_number_on_a_color_conditional_is_refused():
    sel = selection_point(name="s_col")
    chart = _chart().add_selection(sel).conditional(sel.when(fm.Color("g")).otherwise(value(0.2)))
    with pytest.raises(ValueError) as exc:
        chart.to_dict()
    assert str(exc.value) == (
        "fm.value(0.2) for channel='color' needs a color string, got float: 0.2"
    )


def test_channel_no_conditional_can_target_is_refused_at_declaration():
    sel = selection_point(name="s_x")
    with pytest.raises(ValueError, match=r"'x'"):
        _chart().add_selection(sel).conditional(sel.when(fm.X("x")).otherwise(value(1.0)))


def test_literal_for_an_unknown_channel_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _resolve_encoding_value(value(1.0), channel="x")
    assert str(exc.value) == (
        "fm.value(1.0) for channel='x': no selection conditional can target 'x'"
    )


def test_channelless_literal_keeps_its_own_kind():
    assert _resolve_encoding_value(value(0.5)) == {"kind": "opacity", "value": 0.5}
    assert _resolve_encoding_value(value("red"))["kind"] == "color"
