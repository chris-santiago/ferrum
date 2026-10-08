"""Every `stroke_dash=` door accepts one closed vocabulary and refuses the rest by name.

The doors (mark kwargs, the `annotate_*` reference lines, `fm.SelectionMark` and
the `stroke_dash` conditional literal) share `marks.base._normalize_stroke_dash`.
Its accepted forms are closed: a named linetype, a comma-separated numeric
string, a real number, or a list/tuple of real numbers
(`ferrum._validate.is_real_number`, so numpy scalars count and `bool` does not).
Anything else raises a typed `ValueError` at the door naming the door and the
value, instead of constructing and failing later with serde text or a raw
`TypeError`. Every accepted form reaches the wire as a list of floats.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.selection import selection_point, value

DF = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})


def _mark(v):
    return fm.Chart(DF).mark_line(stroke_dash=v).encode(x="x", y="y")


def _annotation(v):
    return fm.annotate_hline(1.0, stroke_dash=v)


def _selection_mark(v):
    return fm.SelectionMark(stroke_dash=v)


def _conditional(v):
    sel = selection_point(name="dash_door")
    cond = replace(sel.when(value([1.0, 1.0])).otherwise(value(v)), channel="stroke_dash")
    return cond.to_spec_dict()


DOORS = {
    "mark": (_mark, r"mark_line: stroke_dash="),
    "annotation": (_annotation, r"annotate_hline: stroke_dash="),
    "selection_mark": (_selection_mark, r"SelectionMark: stroke_dash="),
    "conditional": (_conditional, r"fm\.value\(.*\) for channel='stroke_dash'"),
}

MALFORMED = {
    "dict": {"a": 1},
    "object": object(),
    "set": {4.0, 2.0},
    "bytes": b"4,2",
    "ndarray": np.array([4.0, 2.0]),
    "bool": True,
    "np_bool": np.bool_(True),
    "timedelta": np.timedelta64(4, "s"),
}


@pytest.mark.parametrize("bad", sorted(MALFORMED))
@pytest.mark.parametrize("door", sorted(DOORS))
def test_malformed_value_refused_at_the_door(door, bad):
    call, prefix = DOORS[door]
    with pytest.raises(ValueError, match=rf"^{prefix}"):
        call(MALFORMED[bad])


def _mark_wire(v):
    return _mark(v)._mark_kwargs["stroke_dash"]


def _annotation_wire(v):
    return _annotation(v)._mark_kwargs["stroke_dash"]


def _selection_mark_wire(v):
    return _selection_mark(v).to_spec_dict()["stroke_dash"]


def _conditional_wire(v):
    return _conditional(v)["if_not"]["value"]


WIRES = {
    "mark": _mark_wire,
    "annotation": _annotation_wire,
    "selection_mark": _selection_mark_wire,
    "conditional": _conditional_wire,
}

ACCEPTED = {
    "scalar_int": (5, [5.0]),
    "np_int_scalar": (np.int64(5), [5.0]),
    "list_ints": ([4, 2], [4.0, 2.0]),
    "tuple": ((4.0, 2.0), [4.0, 2.0]),
    "np_elements": ([np.int64(4), np.float32(2.0)], [4.0, 2.0]),
    "named": ("dashed", [4.0, 2.0]),
    "comma": ("4, 2", [4.0, 2.0]),
}


@pytest.mark.parametrize("form", sorted(ACCEPTED))
@pytest.mark.parametrize("door", sorted(WIRES))
def test_accepted_value_reaches_the_wire_as_floats(door, form):
    raw, expected = ACCEPTED[form]
    wire = WIRES[door](raw)
    assert wire == expected
    assert all(type(x) is float for x in wire)


def test_scalar_dash_renders_on_a_mark():
    svg = _mark(5).to_svg()
    assert 'stroke-dasharray="5"' in svg


def test_numpy_dash_elements_render_on_a_mark():
    svg = _mark([np.int64(4), np.float32(2.0)]).to_svg()
    assert 'stroke-dasharray="4,2"' in svg


# ---------------------------------------------------------------------------
# `is_real_number` at the other numeric doors (F10): a numpy scalar is a
# number wherever a number is accepted, and reaches the wire as a float.
# ---------------------------------------------------------------------------


def test_inset_pixel_bound_accepts_a_numpy_integer():
    inner = fm.Chart(DF).mark_point().encode(x="x", y="y")
    inset = fm.Inset(inner, bounds=(fm.px(np.int64(10)), 0.1, 0.5, 0.5))
    from ferrum.structural import _inset_bound_kind_and_value

    assert [_inset_bound_kind_and_value(b) for b in inset.bounds] == [
        ("px", 10.0),
        ("norm", 0.1),
        ("norm", 0.5),
        ("norm", 0.5),
    ]
    (fm.Chart(DF).mark_point().encode(x="x", y="y") + inset).to_svg()


def test_padding_accepts_a_numpy_scalar_and_stores_a_float():
    padding = fm.PaddingConfig(top=np.int64(5), left=np.float32(2.0))
    assert padding.to_dict() == {"top": 5.0, "left": 2.0}
    assert type(padding.top) is float
    chart = fm.Chart(DF).mark_point().encode(x="x", y="y")
    chart.configure(padding=padding).to_svg()


@pytest.mark.parametrize("bad", [True, np.bool_(True), "5", np.timedelta64(5, "s")])
def test_padding_refuses_a_non_number(bad):
    with pytest.raises(ValueError, match=r"^padding\.top must be a numeric pixel value"):
        fm.PaddingConfig(top=bad)


def test_grid_style_accepts_numpy_scalars_and_stores_floats():
    from ferrum.configure import GridConfig

    grid = GridConfig(width=np.int64(2), opacity=np.float32(0.5))
    assert (grid.width, grid.opacity) == (2.0, 0.5)
    assert (type(grid.width), type(grid.opacity)) == (float, float)
    fm.Chart(DF).mark_point().encode(x="x", y="y").configure(grid=grid).to_svg()


def test_int_pixel_and_fraction_values_serialize_as_floats():
    """The validators return the float they checked, so an int the caller passed
    reaches `to_dict()` (and the wire) as a float: `5` becomes `5.0`."""
    from ferrum.configure import GridAxisConfig, GridConfig

    wires = [
        fm.PaddingConfig(top=5).to_dict(),
        GridConfig(width=2, opacity=1).to_dict(),
        GridAxisConfig(width=2).to_dict(),
    ]
    assert wires == [{"top": 5.0}, {"width": 2.0, "opacity": 1.0}, {"width": 2.0}]
    assert all(type(v) is float for w in wires for v in w.values())
