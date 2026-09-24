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


def test_literal_for_a_non_conditional_channel_gets_the_vocabulary_refusal():
    with pytest.raises(
        ValueError, match=r"^condition= is not supported on the 'x' encoding channel"
    ):
        _resolve_encoding_value(value(1.0), channel="x")


def test_literal_on_an_alias_channel_types_as_its_target():
    assert _resolve_encoding_value(value("red"), channel="fill") == {
        "kind": "color",
        "value": {"r": 255, "g": 0, "b": 0, "a": 255},
    }


def test_unstamped_spec_to_spec_dict_resolves_an_alias_branch_to_its_wire_channel():
    """A spec built by `sel.when(...)` and serialized directly (no chart to stamp
    it) infers its channel through the conditional vocabulary: `Fill` -> `color`."""
    sel = selection_point(name="s_fill")
    spec = sel.when(fm.Fill("g")).otherwise(value("#cccccc")).to_spec_dict()
    assert spec["channel"] == "color"
    assert spec["if_not"] == {"kind": "color", "value": {"r": 204, "g": 204, "b": 204, "a": 255}}


def test_unstamped_spec_to_spec_dict_refuses_a_non_conditional_branch():
    sel = selection_point(name="s_shape")
    with pytest.raises(ValueError) as exc:
        sel.when(fm.Shape("g")).otherwise(fm.Shape("h")).to_spec_dict()
    assert str(exc.value).startswith(
        "condition= is not supported on the 'shape' encoding channel; a selection "
        "conditional can target only: angle, color, fill_opacity, opacity, size, "
        "stroke_dash, stroke_opacity, stroke_width, fill (-> color)."
    )


def test_channelless_literal_keeps_its_own_kind():
    assert _resolve_encoding_value(value(0.5)) == {"kind": "opacity", "value": 0.5}
    assert _resolve_encoding_value(value("red"))["kind"] == "color"


def test_literal_handler_table_covers_exactly_the_conditional_channels():
    """`_resolve_encoding_value` subscripts the table with `conditional_wire_channel`'s
    result, a member of `_CONDITIONAL_CHANNELS`; equal sets keep that lookup total."""
    from ferrum.encoding._channel_policy import _CONDITIONAL_CHANNELS
    from ferrum.selection import _CONDITIONAL_LITERAL_HANDLERS

    assert set(_CONDITIONAL_LITERAL_HANDLERS) == _CONDITIONAL_CHANNELS
