"""An encoding channel that aliases into mark style reaches every layer, like `color`.

`apply_channel_aliases` routes some encoding channels (today only `detail`) into
the mark-style bag rather than the wire encoding. On a flat chart that bag is the
chart's own `mark_style`. A layered chart emits no chart-level `mark_style`
(`#119`, spec §4.8: a layer's style is its own), but `detail` is an ENCODING, not
style, and `color` is the model: a layer inherits the bindings of its one parent
and never a sibling's (spec §4.6 V1, G1). A binding written on a composition
(`(a + b).encode(...)`) reaches every layer of it that binds none, whenever it
was declared; a binding written on one `+` operand stays on that operand's
layers, so a later operand does not inherit it (`(a + b).encode(detail="g") + c`
does not split `c`).

The matrix pins every shape against `color` as the control (same counts, static
and interactive). The population is derived by running `apply_channel_aliases`
over every encoding channel and keeping the keys it writes into the mark-style
bag.
"""

from __future__ import annotations

import json
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._overrides import _apply_overrides
from ferrum._scene import _render_scene
from ferrum._warn import reset_warnings
from ferrum.encoding import _channel_class_map
from ferrum.encoding._aliases import apply_channel_aliases

DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 1.0, 2.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "g": ["p", "p", "q", "q"],
        "one": ["k", "k", "k", "k"],
    }
)
XY = {"x": "x", "y": "y"}


def _mark_style_alias_keys() -> list[str]:
    enc = {}
    for name, cls in _channel_class_map().items():
        try:
            enc[name] = cls("g")
        except (TypeError, ValueError):
            continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, mk = apply_channel_aliases(enc, {}, "line")
    return sorted(mk)


ALIAS_KEYS = _mark_style_alias_keys()


def test_population_is_not_empty():
    assert ALIAS_KEYS == ["detail"]


def _line(**enc):
    return fm.Chart(DF).mark_line().encode(**XY, **enc)


def _static_count(chart) -> int:
    return len(re.findall(r"<polyline", chart.to_svg()))


def _interactive_count(chart) -> int:
    if hasattr(chart, "_render_interactive"):
        scene_json, _ = chart._render_interactive()
    else:
        scene_json, _ = _render_scene(chart)
    scene = json.loads(scene_json)
    return sum(len(m["nodes"]) for m in scene["panels"][0]["marks"] if m["kind"] == "line")


# Each shape: (builder(key) -> chart, expected polyline count). Two groups in `g`.
SHAPES = {
    "flat": (lambda k: _line(**{k: "g"}), 2),
    "plus_declared_after": (lambda k: (_line() + _line()).encode(**{k: "g"}), 4),
    "plus_three_declared_after": (
        lambda k: (_line() + _line() + _line()).encode(**{k: "g"}),
        6,
    ),
    "composite_line_point_declared_after": (
        lambda k: fm.Chart(DF).mark_line(point=True).encode(**XY, **{k: "g"}),
        2,
    ),
    # The member's composition-level binding stays with its own layers on
    # every route, the interactive merge included (V1, P8).
    "layerchart_layered_member": (
        lambda k: fm.LayerChart((_line() + _line()).encode(**{k: "g"}), _line()),
        5,
    ),
    "layerchart_flat_members": (
        lambda k: fm.LayerChart(_line(**{k: "g"}), _line(**{k: "g"})),
        4,
    ),
    # Parent rule: the binding written on `(a + b)` stays with that operand.
    "plus_declared_after_then_plus": (
        lambda k: (_line() + _line()).encode(**{k: "g"}) + _line(),
        5,
    ),
    # G1/V1: the left operand's binding never reaches the right operand.
    "declared_before_any_plus": (lambda k: _line(**{k: "g"}) + _line(), 3),
    # A layer's own value wins over the inherited chart-level one.
    "layer_own_differs_from_chart_level": (
        lambda k: _line(**{k: "g"}) + _line(**{k: "one"}),
        3,
    ),
    # A later declaration replaces the earlier one on every layer.
    "plus_redeclared": (
        lambda k: (_line() + _line()).encode(**{k: "g"}).encode(**{k: "one"}),
        2,
    ),
}
COUNTERS = {"static": _static_count, "interactive": _interactive_count}


@pytest.mark.parametrize("output", sorted(COUNTERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_color_control_has_the_same_counts(shape, output):
    """`color` is the model: a layer inherits its parent's binding, never a sibling's."""
    build, expected = SHAPES[shape]
    assert COUNTERS[output](build("color")) == expected


@pytest.mark.parametrize("output", sorted(COUNTERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("key", ALIAS_KEYS)
def test_alias_channel_reaches_the_layers_its_parent_owns(key, shape, output):
    build, expected = SHAPES[shape]
    assert COUNTERS[output](build(key)) == expected


@pytest.mark.parametrize("key", ALIAS_KEYS)
def test_layered_wire_carries_the_key_per_layer_not_chart_level(key):
    spec = json.loads((_line() + _line()).encode(**{key: "g"}).to_spec().to_json())
    assert "mark_style" not in spec
    assert [layer["mark_style"] for layer in spec["layers"]] == [{key: "g"}, {key: "g"}]


@pytest.mark.parametrize("key", ALIAS_KEYS)
def test_layer_own_key_wins_on_the_wire(key):
    spec = json.loads((_line(**{key: "g"}) + _line(**{key: "one"})).to_spec().to_json())
    assert [layer["mark_style"] for layer in spec["layers"]] == [{key: "g"}, {key: "one"}]


def test_layered_detail_on_non_consuming_marks_warns_once():
    chart = fm.Chart(DF).mark_point().encode(**XY) + fm.Chart(DF).mark_point().encode(**XY)
    chart = chart.encode(detail="g")
    reset_warnings()
    with pytest.warns(UserWarning, match=r"encode\(detail=\.\.\.\) is accepted but ignored"):
        chart.to_spec()


def test_detail_ignored_warning_names_the_marks_the_table_says_draw_it():
    """The warning's mark list is read live off Rust's style table, not a Python literal.

    Patches ``marks_drawing`` (the reader of the table) to a value the real
    table never returns, so a message built from a hand-copied literal
    would stay unchanged while the derived one tracks the patch.
    """
    import ferrum.encoding._aliases as aliases_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(aliases_mod, "marks_drawing", lambda field: ("bar", "rect"))
        reset_warnings()
        with pytest.warns(UserWarning) as record:
            fm.Chart(DF).mark_point().encode(**XY, detail="g").to_spec()
    assert str(record[0].message) == (
        "encode(detail=...) is accepted but ignored by mark_point; "
        "only mark_bar and mark_rect read the detail channel for per-group splitting."
    )


@pytest.mark.parametrize(
    ("drawers", "expected"),
    [
        (
            (),
            "encode(detail=...) is accepted but ignored by mark_point; "
            "no mark reads the detail channel for per-group splitting.",
        ),
        (
            ("bar",),
            "encode(detail=...) is accepted but ignored by mark_point; "
            "only mark_bar reads the detail channel for per-group splitting.",
        ),
        (
            ("bar", "rect", "line"),
            "encode(detail=...) is accepted but ignored by mark_point; "
            "only mark_bar, mark_rect, and mark_line read the detail channel "
            "for per-group splitting.",
        ),
    ],
)
def test_detail_ignored_warning_reads_grammatically_for_every_drawer_count(drawers, expected):
    """The drawer-list clause must read grammatically whether the table
    names zero, one, or three-or-more marks. ``_prose_list([])`` used to
    return ``""``, which would have read "only  read the detail channel" for
    the zero-drawer case; the two-drawer case is already pinned by
    ``test_detail_ignored_warning_names_the_marks_the_table_says_draw_it``
    above."""
    import ferrum.encoding._aliases as aliases_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(aliases_mod, "marks_drawing", lambda field: drawers)
        reset_warnings()
        with pytest.warns(UserWarning) as record:
            fm.Chart(DF).mark_point().encode(**XY, detail="g").to_spec()
    assert str(record[0].message) == expected


def test_layered_detail_with_a_consuming_layer_does_not_warn():
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chart = fm.Chart(DF).mark_line(point=True).encode(**XY, detail="g")
        chart.to_svg()


def test_mark_override_on_unnamed_layered_chart_styles_every_layer():
    chart = _apply_overrides(_line() + _line(), entry="figure", mark={"stroke_width": 7})
    widths = re.findall(r'<polyline[^>]*stroke-width="([^"]+)"', chart.to_svg())
    assert widths == ["7", "7"]


def test_mark_override_on_flat_chart_is_the_charts_own_style():
    chart = _apply_overrides(_line(), entry="figure", mark={"stroke_width": 7})
    widths = re.findall(r'<polyline[^>]*stroke-width="([^"]+)"', chart.to_svg())
    assert widths == ["7"]


# ---------------------------------------------------------------------------
# Readers of a layered chart's `_mark`/`_mark_kwargs` (layer 0's, copied up by
# the clone that built the chart) must not treat them as chart-wide.
# ---------------------------------------------------------------------------


def test_smooth_on_a_layered_chart_keeps_every_existing_layer_and_its_style():
    point = fm.Chart(DF).mark_point(opacity=0.3).encode(**XY)
    chart = (point + _line()).mark_smooth()
    spec = json.loads(chart.to_spec().to_json())
    assert [layer["mark"] for layer in spec["layers"]] == ["point", "line", "line"]
    assert [layer.get("mark_style") for layer in spec["layers"][:2]] == [{"opacity": 0.3}, None]
    assert _static_count(chart) == 2


def test_layered_smooth_with_a_ci_band_keeps_every_prior_layer_and_its_style():
    """The layered desugar branch (`ci=` adds a ribbon) keeps every existing layer too."""
    point = fm.Chart(DF).mark_point(fill="red").encode(**XY)
    line = fm.Chart(DF).mark_line(stroke="blue").encode(**XY)
    spec = json.loads((point + line).mark_smooth(ci=0.95).to_spec().to_json())
    assert [layer["mark"] for layer in spec["layers"]] == ["point", "line", "ribbon", "line"]
    assert [layer.get("mark_style") for layer in spec["layers"][:2]] == [
        {"fill": "red"},
        {"stroke": "blue"},
    ]
