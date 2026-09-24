"""An encoding channel that aliases into mark style reaches every layer, like `color`.

`apply_channel_aliases` routes some encoding channels (today only `detail`) into
the mark-style bag rather than the wire encoding. On a flat chart that bag is the
chart's own `mark_style`. A layered chart emits no chart-level `mark_style`
(`#119`, spec §4.8: a layer's style is its own), but `detail` is an ENCODING, not
style: user ruling Q3 settles that a combined chart's encoding is shared by every
layer, and `color` is the model. So every line/area/polygon layer without a
`detail` of its own inherits the chart-level one at serialization, by the same
lazy mechanism `color` uses: whenever it was declared, and whatever layers are
added afterwards.

The matrix pins every shape against `color` as the control (same counts, static
and interactive). The population is derived by running `apply_channel_aliases`
over every encoding channel and keeping the keys it writes into the mark-style
bag.

One cell is a known divergence shared by `color`: an interactive `LayerChart`
merges its members with `+`, so a layered member's chart-level encoding reaches
the members after it (6 polylines), while the static overlay renders each member
alone (5). It is pinned as a strict xfail for both channels.
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
    "layerchart_layered_member": (
        lambda k: fm.LayerChart((_line() + _line()).encode(**{k: "g"}), _line()),
        5,
    ),
    "layerchart_flat_members": (
        lambda k: fm.LayerChart(_line(**{k: "g"}), _line(**{k: "g"})),
        4,
    ),
    "plus_declared_after_then_plus": (
        lambda k: (_line() + _line()).encode(**{k: "g"}) + _line(),
        6,
    ),
    "declared_before_any_plus": (lambda k: _line(**{k: "g"}) + _line(), 4),
    # A later declaration replaces the earlier one on every layer.
    "plus_redeclared": (
        lambda k: (_line() + _line()).encode(**{k: "g"}).encode(**{k: "one"}),
        2,
    ),
}
COUNTERS = {"static": _static_count, "interactive": _interactive_count}


def _xfail_layerchart_interactive(request, shape: str, output: str) -> None:
    if shape == "layerchart_layered_member" and output == "interactive":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="interactive LayerChart merges members with `+`, so a layered "
                "member's chart-level encoding reaches later members (color and detail alike)",
            )
        )


@pytest.mark.parametrize("output", sorted(COUNTERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_color_control_has_the_same_counts(request, shape, output):
    """`color` is the model: a combined chart's encoding is shared by every layer."""
    _xfail_layerchart_interactive(request, shape, output)
    build, expected = SHAPES[shape]
    assert COUNTERS[output](build("color")) == expected


@pytest.mark.parametrize("output", sorted(COUNTERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("key", ALIAS_KEYS)
def test_alias_channel_reaches_every_layer(request, key, shape, output):
    _xfail_layerchart_interactive(request, shape, output)
    build, expected = SHAPES[shape]
    assert COUNTERS[output](build(key)) == expected


@pytest.mark.parametrize("key", ALIAS_KEYS)
def test_layered_wire_carries_the_key_per_layer_not_chart_level(key):
    spec = json.loads((_line() + _line()).encode(**{key: "g"}).to_spec().to_json())
    assert "mark_style" not in spec
    assert [layer["mark_style"] for layer in spec["layers"]] == [{key: "g"}, {key: "g"}]


def test_layered_detail_on_non_consuming_marks_warns_once():
    chart = fm.Chart(DF).mark_point().encode(**XY) + fm.Chart(DF).mark_point().encode(**XY)
    chart = chart.encode(detail="g")
    reset_warnings()
    with pytest.warns(UserWarning, match=r"encode\(detail=\.\.\.\) is accepted but ignored"):
        chart.to_spec()


def test_layered_detail_with_a_consuming_layer_does_not_warn():
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chart = fm.Chart(DF).mark_line(point=True).encode(**XY, detail="g")
        chart.to_svg()


def test_mark_override_on_unnamed_layered_chart_styles_every_layer():
    chart = _apply_overrides(_line() + _line(), mark={"stroke_width": 7})
    widths = re.findall(r'<polyline[^>]*stroke-width="([^"]+)"', chart.to_svg())
    assert widths == ["7", "7"]


def test_mark_override_on_flat_chart_is_the_charts_own_style():
    chart = _apply_overrides(_line(), mark={"stroke_width": 7})
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
