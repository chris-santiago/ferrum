"""The layer encoding model's rule table, cell by cell, on every door.

A layer's effective encoding is its own bindings merged over exactly one
parent's, channel by channel, and never over a sibling's: a host for a layer
attached with ``Chart.layer`` or ``SecondaryY``, what the caller wrote on the
composition itself for a ``+`` operand's layer, and nothing for a
``fm.LayerChart`` member. The one cross-sibling flow is the axis: a layer that
binds no x/y takes layer 0's written x/y.

Every case is keyed by a short behavior tag and a door code:

- F flat chart, L ``Chart.layer(fm.Layer(...))``, LD ``fm.Layer(data=...)``;
- ``+`` / ``+rev`` the composition and its operands reversed (``+s`` over one
  frame, ``+d`` over two, named in the case), ``+e`` a binding written on the
  composition, SY ``+ fm.SecondaryY(...)``;
- LCs / LCm / LCi / LCx ``fm.LayerChart`` static, ``_build_merged()``,
  ``resolve={"y": "independent"}`` and ``.interactive()``;
- Fc faceted.

A rendering case asserts the oracle census: each layer draws what it
draws alone, so the chart's marks equal the sum of its parts' marks rendered
standalone (geometry left out, every style attribute kept, since the panel's
shared axes move positions), its legend set equals the parts' (where the
case names one), and no axis or legend text is an internal column. It also
runs under I: the interactive scene's marks equal the parts' interactive
marks, and ``.interactive().to_html()`` renders. A refusal asserts the exact
string. The wire seam is read off every case: each layer's wire encoding
completed by the chart-level encoding names the fields its part draws alone,
and the chart-level encoding carries no binding written only on one operand
or layer.

Each test names the rejected alternative it discriminates against: the
grain-preserving leak (a sibling's binding still recolors or regroups it)
and the sibling-encoding leak (the left operand's encoding becomes the
composition's, and a later binding is promoted to the chart).
"""

from __future__ import annotations

import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._aux_scale_oracle import log_map, niced_log_domain, pow_map
from tests._legends import chrome_texts as _chrome_texts
from tests._legends import clip_groups as _clip_groups
from tests._legends import interactive_legend_set, legend_set
from tests._legends import svg_root as _svg_root
from tests._legends import tag as _tag
from tests._svg_extents import axis_tick_labels
from tests._svg_rects import THEME_CHROME_STROKES, mark_layer, svg_elements
from tests._wire_steps import all_steps, operand_branches

SY = fm.SecondaryY

# Fixtures: duplicate (g, y) rows on purpose (rows 0 and 2 share (a, 3.0)).
df = pl.DataFrame(
    {
        "v": [1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 6.0, 7.0],
        "y": [3.0, 1.0, 3.0, 1.5, 5.0, 9.0, 2.0, 6.0],
        "z": [10.0, 20.0, 10.0, 30.0, 25.0, 35.0, 40.0, 45.0],
        "r": [1.0, 2.0, 1.0, 3.0, 2.5, 3.5, 4.0, 4.5],
        "g": ["a", "b"] * 4,
        "c": ["p", "p", "q", "q"] * 2,
        "k": ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"],
    }
)
df2 = pl.DataFrame(
    {
        "v": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "y": [2.0, 3.0, 4.0, 2.5, 3.5, 1.0],
        "z": [5.0, 6.0, 7.0, 8.0, 6.5, 5.5],
        "g": ["a", "b", "c"] * 2,
        "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)
# df2 with no column name in common with df but `g`.
df3 = df2.rename({"v": "v3", "y": "y3", "z": "z3", "w": "w3"})


def mk(mark: str, data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    return getattr(fm.Chart(data), f"mark_{mark}")().encode(**encoding)


def P(data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    """The spec's ``P``: points at ``x="v", y="y"``."""
    return mk("point", data, **{"x": "v", "y": "y", **encoding})


def line(data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    return mk("line", data, **{"x": "v", "y": "y", **encoding})


def agg(mark: str, data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    """``mark`` at ``x="g", y="mean(y)"``: one mark per ``g`` unless a channel groups finer."""
    return mk(mark, data, **{"x": "g", "y": "mean(y)", **encoding})


def bar_mean(data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    return agg("bar", data, **encoding)


def layer(mark: str, **encoding: Any) -> fm.Layer:
    return fm.Layer(mark=mark, encoding=encoding)


def mean(cls: type, field: str = "z") -> Any:
    return cls(field, aggregate="mean")


# ---------------------------------------------------------------------------
# Census and oracle
# ---------------------------------------------------------------------------

_MARK_TAGS = ("circle", "rect", "path", "line", "polyline", "polygon", "text")
_GEOMETRY_ATTRS = frozenset(
    ("cx", "cy", "x", "y", "x1", "x2", "y1", "y2", "width", "height", "points", "d", "transform")
)
_FORBIDDEN_TITLES = ("bin_start", "bin_end", "agg_value")
# The theme's axis tick-label color (`tests/_svg_rects.THEME_CHROME_STROKES`).
_CHROME_TEXT = "#6b7280"


def _signature(element: ET.Element) -> tuple:
    """What a mark element draws, without where: its tag, style attributes and text."""
    attrs = tuple(sorted((k, v) for k, v in element.attrib.items() if k not in _GEOMETRY_ATTRS))
    return (_tag(element), attrs, (element.text or "").strip())


def _clip_rects(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    """Each clip path's rect as ``(x0, y0, x1, y1)``, by clip id."""
    out = {}
    for clip in root.iter():
        if _tag(clip) != "clipPath":
            continue
        rect = next((e for e in clip if _tag(e) == "rect"), None)
        if rect is not None:
            x, y = float(rect.get("x")), float(rect.get("y"))
            out[clip.get("id")] = (
                x,
                y,
                x + float(rect.get("width")),
                y + float(rect.get("height")),
            )
    return out


def panel_census(svg: str) -> list[Counter]:
    """Per plot clip group, the multiset of mark signatures it draws.

    A text mark draws above the clip, so each panel also counts the text
    elements outside every clip group that sit inside its plot area: the
    labels a text layer draws. Axis ticks, titles and legends sit outside the
    plot area, and a polar axis's tick labels inside it are drawn in the
    theme's chrome color, which no data label takes. A label inside several
    overlapping plot areas (a static overlay's leaves) counts once, for the
    first.
    """
    root = _svg_root(svg)
    groups = _clip_groups(root)
    rects = _clip_rects(root)
    inside = {id(e) for group in groups for e in group.iter()}
    loose = [
        e
        for e in root.iter()
        if _tag(e) == "text" and id(e) not in inside and e.get("fill") != _CHROME_TEXT
    ]
    out = []
    for group in groups:
        counts = Counter(_signature(e) for e in group.iter() if _tag(e) in _MARK_TAGS)
        clip_id = re.search(r"#([^)]+)", group.get("clip-path")).group(1)
        x0, y0, x1, y1 = rects[clip_id]
        claimed = []
        for e in loose:
            if x0 <= float(e.get("x", "-1")) <= x1 and y0 <= float(e.get("y", "-1")) <= y1:
                counts[_signature(e)] += 1
                claimed.append(e)
        loose = [e for e in loose if all(e is not c for c in claimed)]
        out.append(counts)
    return out


def census(svg: str) -> Counter:
    """The multiset of mark signatures over every plot clip group."""
    return sum(panel_census(svg), Counter())


def tag_counts(svg: str) -> dict[str, int]:
    counts = Counter(signature[0] for signature in census(svg).elements())
    return dict(counts)


def chrome_text(svg: str) -> list[str]:
    return ["".join(e.itertext()).strip() for e in _chrome_texts(svg)]


def axis_titles(svg: str) -> list[str]:
    return [
        "".join(e.itertext()).strip()
        for e in _chrome_texts(svg)
        if e.get("font-weight") == "600" and e.get("font-size") == "13"
    ]


_SCENE_GEOMETRY = frozenset(
    (
        "cx",
        "cy",
        "x",
        "y",
        "x1",
        "x2",
        "y1",
        "y2",
        "w",
        "h",
        "width",
        "height",
        "points",
        "d",
        "path",
        "commands",
    )
)


def _scene(chart: Any) -> dict:
    return json.loads(chart.interactive().scene_json)


def _scene_line_counts(scene: dict) -> list[int]:
    """Each panel's line-batch node count, in panel order; 0 for a panel drawing none."""
    return [
        sum(len(batch["nodes"]) for batch in panel["marks"] if batch["kind"] == "line")
        for panel in scene["panels"]
    ]


def interactive_census(chart: Any) -> Counter:
    """The multiset of interactive scene mark nodes, geometry left out."""
    out: Counter = Counter()
    for panel in _scene(chart)["panels"]:
        for batch in panel["marks"]:
            for node in batch["nodes"]:
                kept = {k: v for k, v in node.items() if k not in _SCENE_GEOMETRY}
                out[(batch["kind"], json.dumps(kept, sort_keys=True))] += 1
    return out


def interactive_tooltip_fields(chart: Any) -> list[set[str]]:
    """Per mark batch, the tooltip field names its marks carry."""
    out = []
    for panel in _scene(chart)["panels"]:
        for batch in panel["marks"]:
            names = {f["name"] for t in batch.get("tooltips") or [] for f in t["fields"]}
            out.append(names)
    return out


def _quiet(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One rendering cell on one door.

    ``parts`` are the standalone charts whose marks the case's marks equal,
    one per wire layer in layer order; ``marks`` pins tag counts where no
    oracle exists; ``legends`` the static legend set (``None``: not judged
    here); ``chart_level`` the non-positional channels the chart-level
    encoding may carry (``None``: not judged); ``static`` is False for LCx,
    which renders only interactively.
    """

    build: Callable[[], Any]
    parts: tuple[Callable[[], Any], ...] = ()
    marks: dict[str, int] | None = None
    legends: dict[str, tuple[str, ...]] | None = None
    chart_level: frozenset | None = frozenset()
    static: bool = True
    faceted: bool = False


def _parts_census(parts: tuple, faceted: bool) -> Any:
    if faceted:
        panels = [panel_census(_quiet(part).to_svg()) for part in parts]
        return [sum(column, Counter()) for column in zip(*panels, strict=True)]
    return sum((census(_quiet(part).to_svg()) for part in parts), Counter())


def _parts_legends(parts: tuple) -> dict:
    out: dict = {}
    for part in parts:
        out.update(legend_set(_quiet(part).to_svg()))
    return out


def check_rendering(case: Case) -> None:
    chart = _quiet(case.build)
    if case.static:
        svg = _quiet(chart.to_svg)
        for text in chrome_text(svg):
            assert text not in _FORBIDDEN_TITLES and "__rhs_" not in text, text
        if case.parts:
            got = panel_census(svg) if case.faceted else census(svg)
            assert got == _parts_census(case.parts, case.faceted)
        if case.marks is not None:
            assert tag_counts(svg) == case.marks
        if case.legends is not None:
            assert legend_set(svg) == case.legends
    if case.parts and not case.faceted:
        expected = sum((interactive_census(_quiet(p)) for p in case.parts), Counter())
        assert _quiet(lambda: interactive_census(chart)) == expected
    assert "<html" in _quiet(lambda: chart.interactive().to_html()).lower()


# The channels the seam compares: every non-positional channel a layer draws.
_SEAM_CHANNELS = (
    "color",
    "size",
    "shape",
    "opacity",
    "fill_opacity",
    "stroke_opacity",
    "stroke_width",
    "stroke_dash",
    "angle",
    "text",
    "tooltip",
    "href",
    "key",
    "description",
)
_POSITIONAL = ("x", "y", "x2", "y2")


def _drawn_fields(encoding: dict, mark_style: dict | None) -> dict[str, str]:
    fields = {
        ch: re.sub(r"__rhs_\d+$", "", encoding[ch]["field"])
        for ch in _SEAM_CHANNELS
        if ch in encoding and "field" in encoding[ch]
    }
    if mark_style and "detail" in mark_style:
        fields["detail"] = mark_style["detail"]
    return fields


def _wire(chart: Any) -> dict:
    merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    return json.loads(_quiet(lambda: merged.to_spec().to_json()))


def _steps_at_any_depth(chart: Any) -> list[dict]:
    """Every pipeline step of *chart*'s wire, each branch's own steps after it, in written order."""
    return all_steps(_wire(chart))


def check_wire_seam(case: Case) -> None:
    """Each layer's completed wire encoding draws what its part draws alone."""
    chart = _quiet(case.build)
    if not isinstance(chart, (fm.Chart, fm.LayerChart)):
        return
    wire = _wire(chart)
    chart_encoding = wire["encoding"]
    layers = wire.get("layers")
    if case.chart_level is not None and layers:
        assert set(chart_encoding) - set(_POSITIONAL) <= case.chart_level, chart_encoding
    if not case.parts or not layers or len(layers) != len(case.parts):
        return
    for i, (wire_layer, part) in enumerate(zip(layers, case.parts, strict=True)):
        inherited = {
            ch: v
            for ch, v in chart_encoding.items()
            if ch not in _POSITIONAL or "data_source" not in wire_layer
        }
        completed = {**inherited, **wire_layer["encoding"]}
        alone = _wire(_quiet(part))
        assert _drawn_fields(completed, wire_layer.get("mark_style")) == _drawn_fields(
            alone["encoding"], alone.get("mark_style")
        ), f"layer {i}"


def plus_doors(
    a: Callable[[], fm.Chart],
    b: Callable[[], fm.Chart],
    *,
    legends: dict | None = None,
    lcs: bool = True,
) -> dict[str, Case]:
    """``a + b`` on every door that composes two charts: both orders and every LayerChart route.

    The static overlay (LCs) is judged on marks only: its one-legend rule
    is the composite overlay's, pinned with that route.
    """
    doors = {
        "+": Case(lambda: a() + b(), (a, b), legends=legends),
        "+rev": Case(lambda: b() + a(), (b, a), legends=legends),
        "LCm": Case(lambda: fm.LayerChart(a(), b())._build_merged(), (a, b), legends=legends),
        "LCi": Case(
            lambda: fm.LayerChart(a(), b(), resolve={"y": "independent"}), (a, b), legends=legends
        ),
        "LCx": Case(lambda: fm.LayerChart(a(), b()), (a, b), static=False),
    }
    if lcs:
        doors["LCs"] = Case(lambda: fm.LayerChart(a(), b()), (a, b))
    return doors


def _cases(prefix: str, doors: dict[str, Case]) -> dict[str, Case]:
    return {f"{prefix}/{door}": case for door, case in doors.items()}


# ---------------------------------------------------------------------------
# A value channel's binding never crosses siblings
# ---------------------------------------------------------------------------

# Channel -> (bound field, the mark a binder draws it with).
_VALUE_CHANNELS: dict[str, tuple[str, str]] = {
    "color": ("c", "point"),
    "size": ("z", "point"),
    "opacity": ("z", "point"),
    "fill_opacity": ("z", "point"),
    "stroke_opacity": ("z", "point"),
    "stroke_width": ("z", "point"),
    "angle": ("z", "point"),
    "text": ("k", "text"),
    "tooltip": ("z", "point"),
}


def _binder(channel: str) -> Callable[[], fm.Chart]:
    value, mark = _VALUE_CHANNELS[channel]
    return lambda: mk(mark, **{"x": "v", "y": "y", channel: value})


def _value_binding_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for channel, (value, _mark) in _VALUE_CHANNELS.items():
        binder = _binder(channel)
        legends = legend_set(_quiet(lambda binder=binder: binder().to_svg()))
        cases |= _cases(
            f"value_binding/{channel}/binder-then-line", plus_doors(binder, line, legends=legends)
        )
        bound_line = lambda channel=channel, value=value: line(**{channel: value})  # noqa: E731
        bound_legends = legend_set(_quiet(lambda bound_line=bound_line: bound_line().to_svg()))
        cases |= _cases(
            f"value_binding/{channel}/point-then-bound-line",
            plus_doors(P, bound_line, legends=bound_legends),
        )
        cases[f"value_binding/{channel}/L-own"] = Case(
            lambda channel=channel, value=value: P().layer(layer("line", **{channel: value})),
            (P, bound_line),
            legends=bound_legends,
            chart_level=None,
        )
        cases[f"value_binding/{channel}/L-inherited"] = Case(
            lambda channel=channel, value=value: P(**{channel: value}).layer(layer("line")),
            (
                lambda channel=channel, value=value: P(**{channel: value}),
                lambda channel=channel, value=value: line(**{channel: value}),
            ),
            chart_level=None,
        )
    return cases


_VALUE_BINDING = _value_binding_cases()


@pytest.mark.parametrize("case", sorted(_VALUE_BINDING))
def test_a_value_binding_never_crosses_siblings(case):
    """``P(color="c") + line`` draws one polyline; the legend lists the binder's values only.

    Rejects the sibling-encoding leak: the left's color would split the line,
    and a right or attached binding promoted to the chart would recolor the
    points.
    """
    check_rendering(_VALUE_BINDING[case])


def test_the_left_operands_color_leaves_the_line_whole():
    """One polyline on every route (rejects the sibling-encoding leak: 2 polylines)."""
    for door, built in plus_doors(lambda: P(color="c"), line).items():
        chart = _quiet(built.build)
        if built.static:
            assert tag_counts(_quiet(chart.to_svg))["polyline"] == 1, door
        kinds = [kind for kind, _ in interactive_census(chart).elements()]
        assert kinds.count("line") == 1, door


@pytest.mark.parametrize("order", ["bound-first", "bound-later"])
def test_a_tooltip_reaches_only_its_own_layer_in_interactive_output(order):
    """Interactively, the line's tooltip field never becomes the points' tooltip.

    Rejects tooltip promotion: the points would report ``z``.
    """
    bound = line(tooltip="z")
    chart = bound + P() if order == "bound-first" else P() + bound
    tooltips = interactive_tooltip_fields(chart)
    points = tooltips[1] if order == "bound-first" else tooltips[0]
    assert "z" not in points
    assert points == interactive_tooltip_fields(P())[0]


# ---------------------------------------------------------------------------
# A composition's binding names each layer's own column
# ---------------------------------------------------------------------------

_ABC = {"g": ("a", "b", "c")}

_VALUE_BINDING_BY_NAME: dict[str, Case] = {
    "+e/overlap": Case(
        lambda: (P(df) + line(df2)).encode(color="g"),
        (lambda: P(df, color="g"), lambda: line(df2, color="g")),
        legends=_ABC,
        chart_level=frozenset({"color"}),
    ),
    "+e/overlap/rev": Case(
        lambda: (line(df2) + P(df)).encode(color="g"),
        (lambda: line(df2, color="g"), lambda: P(df, color="g")),
        legends=_ABC,
        chart_level=frozenset({"color"}),
    ),
    "+e/only-g-shared": Case(
        lambda: (P(df) + mk("line", df3, x="v3", y="y3")).encode(color="g"),
        (lambda: P(df, color="g"), lambda: mk("line", df3, x="v3", y="y3", color="g")),
        legends=_ABC,
        chart_level=frozenset({"color"}),
    ),
    **_cases(
        "LC",
        {
            door: c
            for door, c in plus_doors(
                lambda: P(df, color="g"), lambda: line(df2, color="g"), legends=_ABC, lcs=False
            ).items()
            if door.startswith("LC")
        },
    ),
}


@pytest.mark.parametrize("case", sorted(_VALUE_BINDING_BY_NAME))
def test_by_name_a_compositions_binding_colors_each_layer_by_its_own_column(case):
    """``(P(df) + line(df2)).encode(color="g")`` colors both by their own ``g``; one legend a, b, c.

    Rejects reading the left's column on the right's rows (the line would
    draw a null group).
    """
    check_rendering(_VALUE_BINDING_BY_NAME[case])


# ---------------------------------------------------------------------------
# Statistics keep their grain beside any sibling
# ---------------------------------------------------------------------------

_VALUE_AGGREGATE: dict[str, Case] = {
    "F": Case(lambda: agg("point", color=mean(fm.Color)), marks={"circle": 2}),
    "L-inherits-the-hosts-aggregate": Case(
        lambda: agg("point", color=mean(fm.Color)).layer(layer("tick")),
        (lambda: agg("point", color=mean(fm.Color)), lambda: agg("tick", color=mean(fm.Color))),
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_VALUE_AGGREGATE))
def test_a_value_aggregate_is_computed_per_layer(case):
    check_rendering(_VALUE_AGGREGATE[case])


_AGG_CLASSES = {
    "color": fm.Color,
    "size": fm.Size,
    "opacity": fm.Opacity,
    "fill_opacity": fm.FillOpacity,
    "stroke_opacity": fm.StrokeOpacity,
    "stroke_width": fm.StrokeWidth,
    "angle": fm.Angle,
    "text": fm.Text,
    "tooltip": fm.Tooltip,
}


def _sibling_grain_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {
        "L/bar_mean+point(color=mean(z))": Case(
            lambda: bar_mean().layer(layer("point", color=mean(fm.Color))),
            (bar_mean, lambda: agg("point", color=mean(fm.Color))),
            marks={"rect": 2, "circle": 2},
            chart_level=None,
        ),
        "Fc/(tick+point(color=mean(z))).facet(c)": Case(
            lambda: (agg("tick") + agg("point", color=mean(fm.Color))).facet("c"),
            (
                lambda: agg("tick").facet("c"),
                lambda: agg("point", color=mean(fm.Color)).facet("c"),
            ),
            faceted=True,
        ),
    }
    for channel, cls in _AGG_CLASSES.items():
        cases |= _cases(
            f"{channel}/point(mean)+tick",
            plus_doors(
                lambda cls=cls, channel=channel: agg("point", **{channel: mean(cls)}),
                lambda: agg("tick"),
            ),
        )
    cases |= _cases("P(x=g,color=c)+bar_mean", plus_doors(lambda: P(x="g", color="c"), bar_mean))
    return cases


_SIBLING_GRAIN = _sibling_grain_cases()


@pytest.mark.parametrize("case", sorted(_SIBLING_GRAIN))
def test_a_siblings_binding_never_changes_a_layers_grain(case):
    """Host 2 bars + 2 points; the tick keeps 2; the bar mean keeps 2; Fc 4 ticks.

    Rejects a grain-preserving leak that would still recolor the sibling,
    and a sibling-encoding leak where the sibling would group by the leaked
    field (7 bars).
    """
    check_rendering(_SIBLING_GRAIN[case])


def test_the_tick_beside_a_colored_mean_draws_two_ticks():
    """Named repro: ``point(g, mean(y), color=mean(z)) + tick(g, mean(y))`` keeps 2 ticks."""
    for channel, cls in _AGG_CLASSES.items():
        svg = _quiet(lambda: (agg("point", **{channel: mean(cls)}) + agg("tick")).to_svg())
        assert tag_counts(svg).get("line", 0) == 2, channel


def _different_frame_sibling_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for channel in ("color", "size", "opacity", "fill_opacity", "stroke_opacity"):
        cls = _AGG_CLASSES[channel]
        cases |= _cases(
            f"{channel}/tick(df)+point(df2,mean)",
            plus_doors(
                lambda: agg("tick"),
                lambda cls=cls, channel=channel: agg("point", df2, **{channel: mean(cls)}),
            ),
        )
    cases |= _cases(
        "bar_mean+P(df2,x=g,color=g)",
        plus_doors(bar_mean, lambda: P(df2, x="g", color="g")),
    )
    return cases


_DIFFERENT_FRAME_SIBLING = _different_frame_sibling_cases()


@pytest.mark.parametrize("case", sorted(_DIFFERENT_FRAME_SIBLING))
def test_a_different_frame_sibling_never_empties_or_regroups_a_layer(case):
    """Left 2 ticks, right 3 points; no build error; ``bar_mean`` keeps 2 bars.

    Rejects the sibling-encoding leak: the left would group by the right's
    renamed column and draw nothing, and the opacity family's renamed
    promotion copy would refuse its stamped title.
    """
    check_rendering(_DIFFERENT_FRAME_SIBLING[case])


_RIGHT_AGGREGATE = _cases(
    "P(x=g)+point(mean(y),color=mean(z))",
    plus_doors(
        lambda: P(x="g"),
        lambda: agg("point", color=mean(fm.Color)),
        legends={
            "mean_z": tuple(legend_set(agg("point", color=mean(fm.Color)).to_svg())["mean_z"])
        },
    ),
)


@pytest.mark.parametrize("case", sorted(_RIGHT_AGGREGATE))
def test_a_right_aggregate_leaves_the_left_raw_and_uncolored(case):
    """The left draws 8 raw uncolored points; one color legend from the right, titled ``mean_z``.

    Rejects the sibling-encoding leak: the left's raw z would draw through
    the promoted scale.
    """
    check_rendering(_RIGHT_AGGREGATE[case])


_ALIAS_SHADOW: dict[str, Case] = {
    "L/fill-shadows-nothing": Case(
        lambda: bar_mean().layer(layer("point", fill=mean(fm.Fill))),
        (bar_mean, lambda: agg("point", fill=mean(fm.Fill))),
        marks={"rect": 2, "circle": 2},
        chart_level=None,
    ),
    # The host's nominal color establishes the color scale, so the layer's
    # numeric color stays out of it; the grain is what these cells pin: 4
    # bars by (g, c), 2 points by g alone.
    "L/fill-shadows-the-hosts-color": Case(
        lambda: bar_mean(color="c").layer(layer("point", fill=mean(fm.Fill))),
        marks={"rect": 4, "circle": 2},
        chart_level=None,
    ),
    "L/stroke-shadows-the-hosts-color": Case(
        lambda: bar_mean(color="c").layer(layer("point", stroke=mean(fm.Stroke))),
        marks={"rect": 4, "circle": 2},
        chart_level=None,
    ),
    **_cases(
        "+d/bar_mean+point(df2,fill=mean(z))",
        plus_doors(bar_mean, lambda: agg("point", df2, fill=mean(fm.Fill))),
    ),
}


@pytest.mark.parametrize("case", sorted(_ALIAS_SHADOW))
def test_an_alias_shadows_the_parents_channel(case):
    """A layer's own ``fill``/``stroke`` is its color, so it groups by ``g`` alone (2 points).

    Rejects an alias-blind merge: the layer would also group by the
    parent's color field (7 points, or a vanished layer over two frames).
    """
    check_rendering(_ALIAS_SHADOW[case])


# ---------------------------------------------------------------------------
# A user transform's output, and one scale and legend over every drawing layer
# ---------------------------------------------------------------------------


def _mean_r_by_v() -> fm.Chart:
    return (
        fm.Chart(df)
        .transform(
            fm.transform_aggregate({"field": "r", "fn": "mean", "as": "mean_r"}, groupby=["v"])
        )
        .mark_point()
        .encode(x="v", y="mean_r")
    )


def _my_column() -> fm.Chart:
    return (
        fm.Chart(df)
        .transform(fm.transform_calculate("my", "datum.k"))
        .mark_point()
        .encode(x="v", y="y")
    )


_TRANSFORM_OUTPUT_BINDING: dict[str, Case] = {
    "SY(mean_r)": Case(lambda: _mean_r_by_v() + SY("mean_r"), marks={"circle": 8, "polyline": 1}),
    "L(text=my)": Case(
        lambda: _my_column().layer(layer("text", text="my")),
        marks={"circle": 8, "text": 8},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_TRANSFORM_OUTPUT_BINDING))
def test_a_user_transforms_output_binds_on_every_door(case):
    check_rendering(_TRANSFORM_OUTPUT_BINDING[case])


def test_the_attached_text_layer_draws_the_calculated_column():
    labels = set(chrome_text(_my_column().layer(layer("text", text="my")).to_svg()))
    assert set(df["k"]) <= labels


_SHARED_SCALE_LEGEND: dict[str, Case] = {
    **{
        k: v
        for k, v in _cases(
            "color-union",
            plus_doors(lambda: P(color="g"), lambda: line(df2, color="g"), legends=_ABC, lcs=False),
        ).items()
        if not k.endswith("LCx")
    },
}


@pytest.mark.parametrize("case", sorted(_SHARED_SCALE_LEGEND))
def test_one_scale_and_one_legend_over_every_drawing_layer(case):
    """A ``g`` color on both layers over two frames draws one legend a, b, c.

    Rejects first-layer domains: the legend would list a, b only.
    """
    case = _SHARED_SCALE_LEGEND[case]
    chart = _quiet(case.build)
    assert legend_set(_quiet(chart.to_svg)) == _ABC


@pytest.mark.parametrize(
    "build",
    [lambda: P(size="z") + P(df2, size="z"), lambda: P(df2, size="z") + P(size="z")],
    ids=["size", "size-rev"],
)
def test_a_size_on_both_layers_draws_one_legend_over_the_union(build):
    """One size legend titled ``z`` whose entries span both frames' z (5..45).

    Rejects a first-layer domain: the legend would list either frame's
    entries alone.
    """
    legends = legend_set(_quiet(build).to_svg())
    assert list(legends) == ["z"]
    entries = legends["z"]
    assert entries != legend_set(P(size="z").to_svg())["z"]
    assert entries != legend_set(P(df2, size="z").to_svg())["z"]
    assert float(entries[0]) <= df["z"].min() and float(entries[-1]) >= 0.8 * df["z"].max()


def test_an_opacity_on_both_layers_draws_one_legend():
    titles = list(legend_set(_quiet(lambda: P(opacity="z") + P(df2, opacity="z")).to_svg()))
    assert titles == ["z"]


def test_a_type_conflict_excludes_the_later_layer_with_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = (P(color="z") + line(color="g")).to_svg()
    messages = [str(w.message) for w in caught]
    assert any("stays out of that scale" in m for m in messages), messages
    assert list(legend_set(svg)) == ["z"]


# ---------------------------------------------------------------------------
# The opacity family and stroke_width draw one legend on every door
# ---------------------------------------------------------------------------
#
# A field bound to opacity, fill_opacity, stroke_opacity or stroke_width
# draws one legend per channel, titled, suppressed and scaled from Python as
# a size legend is. Every mark that draws a bound opacity draws it through
# the channel's scale, so its painted alpha is the one its legend shows.

_LEGEND_CHANNEL_CLASSES: dict[str, type] = {
    "opacity": fm.Opacity,
    "fill_opacity": fm.FillOpacity,
    "stroke_opacity": fm.StrokeOpacity,
    "stroke_width": fm.StrokeWidth,
}

# The explicit scale each channel is given: domain 0..50 onto this range.
_EXPLICIT_RANGE: dict[str, tuple[float, float]] = {
    "opacity": (0.2, 0.8),
    "fill_opacity": (0.2, 0.8),
    "stroke_opacity": (0.2, 0.8),
    "stroke_width": (0.5, 3.0),
}


def interactive_legend(
    chart: Any, channel: str | None = None
) -> dict[str, tuple[tuple[str, float], ...]]:
    """Each interactive legend's title and its (label, swatch) entries.

    A swatch is a line's width or a circle's opacity (its radius on a size
    legend): the value the scale maps that entry's label to.
    """
    legends: dict[str, list[tuple[str, float]]] = {}
    title: str | None = None
    swatch: float | None = None
    for node in _scene(chart)["legend"]:
        style = node.get("style") or {}
        if node["type"] == "text":
            if style.get("font_weight") == {"custom": "600"}:
                title = node["content"]
                legends[title] = []
            elif title is not None and swatch is not None:
                legends[title].append((node["content"], swatch))
                swatch = None
        elif node["type"] == "line":
            swatch = style["width"]
        elif node["type"] == "circle":
            swatch = node["r"] if channel == "size" else style["opacity"]
    return {t: tuple(entries) for t, entries in legends.items()}


def _legend_title_count(svg: str, title: str) -> int:
    return sum(
        1
        for e in _chrome_texts(svg)
        if "".join(e.itertext()).strip() == title
        and e.get("font-weight") == "600"
        and e.get("font-size") != "13"
    )


def _bound_value(channel: str, spelling: str) -> Any:
    cls = _LEGEND_CHANNEL_CLASSES[channel]
    if spelling == "string":
        return "z"
    if spelling == "titled":
        return cls("z", title="Mapped")
    lo, hi = _EXPLICIT_RANGE[channel]
    return cls("z", scale=fm.LinearScale(domain=[0.0, 50.0], range=[lo, hi]))


def _legend_doors(channel: str, spelling: str) -> dict[str, Callable[[], Any]]:
    """The doors a channel's binding is written on, with a point partner where one composes."""
    return _binding_doors(lambda: {channel: _bound_value(channel, spelling)})


def _binding_doors(bound: Callable[[], dict[str, Any]]) -> dict[str, Callable[[], Any]]:
    """Every door one binding is written on; the binding is built inside each door."""

    def binder() -> fm.Chart:
        return P(**bound())

    def partner() -> fm.Chart:
        return P(y="r")

    return {
        "F": binder,
        "L-own": lambda: P().layer(layer("point", y="r", **bound())),
        "L-inherited": lambda: binder().layer(layer("point", y="r")),
        "+": lambda: binder() + partner(),
        "+rev": lambda: partner() + binder(),
        "+e": lambda: (P() + partner()).encode(**bound()),
        "SY": lambda: binder() + SY("r"),
        "LCs": lambda: fm.LayerChart(binder(), partner()),
        "LCm": lambda: fm.LayerChart(binder(), partner())._build_merged(),
        "LCi": lambda: fm.LayerChart(binder(), partner(), resolve={"y": "independent"}),
        "Fc": lambda: binder().facet("c"),
        "Fc+": lambda: (binder() + partner()).facet("c"),
    }


_LEGEND_DOOR_CASES = {
    f"{channel}/{spelling}/{door}": (channel, spelling, door)
    for channel in _LEGEND_CHANNEL_CLASSES
    for spelling in ("string", "titled", "scaled")
    for door in _legend_doors(channel, spelling)
}


@pytest.mark.parametrize("case", sorted(_LEGEND_DOOR_CASES))
def test_an_opacity_or_width_binding_draws_one_legend_alike_on_every_door(case):
    """Every door draws the flat chart's one legend, static and interactive.

    The interactive legend's entries carry the swatch each label maps to,
    so a door that drew another scale (or another title) would differ.
    Rejects the legend-less channel: no legend, or ``title=``/``scale=``
    refused or dropped before the wire.
    """
    channel, spelling, door = _LEGEND_DOOR_CASES[case]
    reference = _quiet(lambda: interactive_legend(_legend_doors(channel, spelling)["F"]()))
    ((title, entries),) = reference.items()
    assert title == ("Mapped" if spelling == "titled" else "z")
    assert len(entries) >= 2

    chart = _quiet(_legend_doors(channel, spelling)[door])
    svg = _quiet(chart.to_svg)
    assert legend_set(svg) == {title: tuple(label for label, _ in entries)}
    assert _legend_title_count(svg, title) == 1
    assert _quiet(lambda: interactive_legend(chart)) == reference


@pytest.mark.parametrize("channel", sorted(_LEGEND_CHANNEL_CLASSES))
def test_an_explicit_domain_and_range_reach_the_wire_and_set_the_legend(channel):
    """The legend lists the explicit domain's values, each drawn at its linear map onto the range.

    Rejects a scale dropped before the wire (the data extent 10..45 and the
    default range would set the entries).
    """
    lo, hi = _EXPLICIT_RANGE[channel]
    chart = P(**{channel: _bound_value(channel, "scaled")})
    scale = _wire(chart)["encoding"][channel]["scale"]
    assert scale["domain"] == [0.0, 50.0] and scale["range"] == [lo, hi]
    ((title, entries),) = interactive_legend(chart).items()
    assert title == "z"
    labels = [float(label) for label, _ in entries]
    assert labels[0] == 0.0 and labels[-1] == 50.0
    assert [swatch for _, swatch in entries] == pytest.approx(
        [lo + (hi - lo) * v / 50.0 for v in labels]
    )


def _mark_values(chart: Any, channel: str) -> list:
    """Each interactive mark's drawn value for *channel*: a radius for size, else its style value.

    A style with no alpha of its own is opaque.
    """
    if channel == "size":
        return [
            node["r"]
            for panel in _scene(chart)["panels"]
            for batch in panel["marks"]
            for node in batch["nodes"]
        ]
    default = None if channel == "stroke_width" else 1.0
    return [
        (node.get("style") or {}).get(channel, default)
        for panel in _scene(chart)["panels"]
        for batch in panel["marks"]
        for node in batch["nodes"]
    ]


@pytest.mark.parametrize("door", ["F", "+", "LCs", "LCm", "Fc"])
@pytest.mark.parametrize("channel", sorted(_LEGEND_CHANNEL_CLASSES))
def test_legend_none_suppresses_the_legend_and_keeps_the_mapped_values(channel, door):
    """``legend=None`` draws no legend; every mark keeps the value the scale maps it to.

    Rejects ``legend=None`` dropped before the wire (the legend still
    draws) and a suppression that also drops the scale (the marks would
    draw one default value).
    """
    cls = _LEGEND_CHANNEL_CLASSES[channel]

    def build(value: Any) -> Callable[[], Any]:
        def binder() -> fm.Chart:
            return P(**{channel: value})

        return {
            "F": binder,
            "+": lambda: binder() + P(y="r"),
            "LCs": lambda: fm.LayerChart(binder(), P(y="r")),
            "LCm": lambda: fm.LayerChart(binder(), P(y="r"))._build_merged(),
            "Fc": lambda: binder().facet("c"),
        }[door]

    shown = _quiet(build(cls("z")))
    hidden = _quiet(build(cls("z", legend=None)))
    assert list(legend_set(_quiet(shown.to_svg))) == ["z"]
    hidden_svg = _quiet(hidden.to_svg)
    assert legend_set(hidden_svg) == {}
    assert _quiet(lambda: interactive_legend(hidden)) == {}
    assert census(hidden_svg) == census(_quiet(shown.to_svg))
    values = _quiet(lambda: _mark_values(hidden, channel))
    assert values == _quiet(lambda: _mark_values(shown, channel))
    assert len(set(v for v in values if v is not None)) > 1


def _union_layers(channel: str) -> tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]:
    # A stroke width reads off lines; the opacity family reads off points.
    first = line if channel == "stroke_width" else P
    return (lambda: first(**{channel: "r"}), lambda: first(df2, **{channel: "w"}))


@pytest.mark.parametrize("order", ["r-first", "w-first"])
@pytest.mark.parametrize("channel", sorted(_LEGEND_CHANNEL_CLASSES))
def test_two_layers_draw_one_legend_spanning_both_extents(channel, order):
    """``line(stroke_width="r") + line(df2, stroke_width="w")``: one legend over r and w together.

    ``r`` spans 1..4.5 and ``w`` 1..6. Rejects a first-layer domain: the
    legend would end at 4.5 (or start at w's own nice extent).
    """
    a, b = _union_layers(channel)
    chart = _quiet(lambda: a() + b() if order == "r-first" else b() + a())
    title = "r" if order == "r-first" else "w"
    legends = legend_set(_quiet(chart.to_svg))
    assert list(legends) == [title]
    values = [float(v) for v in legends[title]]
    assert values[0] <= 1.0 and values[-1] >= 6.0
    assert legends[title] != legend_set(_quiet(a().to_svg))["r"]
    ((_, entries),) = _quiet(lambda: interactive_legend(chart)).items()
    swatches = [s for _, s in entries]
    assert swatches == sorted(swatches) and swatches[0] < swatches[-1]


@pytest.mark.parametrize("channel", sorted(_LEGEND_CHANNEL_CLASSES))
def test_a_string_layer_is_excluded_and_one_legend_stays(channel):
    """``point(opacity="z") + point(opacity="g")`` warns the exclusion and draws layer 0's one legend."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = (P(**{channel: "z"}) + P(**{channel: "g"})).to_svg()
    excluded = [str(w.message) for w in caught if "stays out of that scale" in str(w.message)]
    assert excluded == [
        f"layer 1: its {channel} scale differs from the one layer 0 establishes the {channel} "
        "scale with, so it stays out of that scale; its marks draw through that scale and may "
        "fall outside its legend"
    ]
    assert list(legend_set(svg)) == ["z"]
    assert legend_set(svg) == legend_set(P(**{channel: "z"}).to_svg())


_ONE_ROW = pl.DataFrame({"v": [1.0], "y": [2.0], "z": [3.0]})
_CONSTANT = pl.DataFrame({"v": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0], "z": [0.3, 0.3, 0.3]})


@pytest.mark.parametrize("frame", ["one-row", "constant"])
@pytest.mark.parametrize("channel", sorted(_LEGEND_CHANNEL_CLASSES))
def test_a_degenerate_domain_draws_a_one_entry_legend_at_the_marks_value(channel, frame):
    """A one-row chart or a constant column draws one legend entry at the value its marks draw.

    A stroke width draws the midpoint of its 1-4 px band.
    """
    data = _ONE_ROW if frame == "one-row" else _CONSTANT
    chart = mk("point", data, x="v", y="y", **{channel: "z"})
    ((title, entries),) = _quiet(lambda: interactive_legend(chart)).items()
    assert title == "z"
    ((label, swatch),) = entries
    assert float(label) == data["z"][0]
    drawn = set(_quiet(lambda: _mark_values(chart, channel)))
    assert drawn == {swatch}
    if channel == "stroke_width":
        assert swatch == 2.5


_WITH_NULL = pl.DataFrame(
    {"v": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 3.0, 4.0, 1.0], "r": [1.0, None, 3.0, 4.0]}
)


@pytest.mark.parametrize("channel", sorted(_LEGEND_CHANNEL_CLASSES))
def test_a_null_value_draws_the_default_and_adds_no_legend_entry(channel):
    """The null row draws what an unbound mark draws; the legend is the null-free chart's.

    Rejects a null read as 0 (the row would draw the range's low end, and
    the legend would start at 0).
    """
    filled = channel not in ("stroke_opacity", "stroke_width")

    def chart(data: pl.DataFrame, **encoding: Any) -> fm.Chart:
        return fm.Chart(data).mark_point(filled=filled).encode(x="v", y="y", **encoding)

    bound = chart(_WITH_NULL, **{channel: "r"})
    unbound_values = _mark_values(chart(_WITH_NULL), channel)
    values = _mark_values(bound, channel)
    assert values[1] == unbound_values[1]
    assert len({values[0], values[2], values[3]}) == 3
    without_null = chart(_WITH_NULL.drop_nulls(), **{channel: "r"})
    assert list(legend_set(bound.to_svg())) == ["r"]
    assert legend_set(bound.to_svg()) == legend_set(without_null.to_svg())
    assert interactive_legend(bound) == interactive_legend(without_null)


def _painted_alpha(node: dict) -> float:
    """The alpha a line or bar node paints: its style opacity, paint alpha and fill opacity."""
    style = node["style"]
    paint = style.get("fill") or style.get("color")
    return style["opacity"] * (paint["a"] / 255) * style.get("fill_opacity", 1.0)


@pytest.mark.parametrize("mark", ["line", "bar"])
def test_a_line_or_bar_paints_the_alpha_its_opacity_legend_shows(mark):
    """A bound opacity on a line or a bar draws through the scale its legend states.

    Each bar draws its row's alpha, and the line draws its first row's; each
    equals the legend swatch at that row's ``z``. Rejects the raw clamped
    column value (``z`` is 10..45, so every raw alpha would be 1).
    """
    chart = line(opacity="z") if mark == "line" else mk("bar", x="k", y="y", opacity="z")
    ((_, entries),) = interactive_legend(chart).items()
    swatch = {float(label): alpha for label, alpha in entries}
    nodes = [n for p in _scene(chart)["panels"] for b in p["marks"] for n in b["nodes"]]
    rows = df["z"].to_list()[: len(nodes)]
    assert len(nodes) == (1 if mark == "line" else len(df))
    for node, z in zip(nodes, rows, strict=True):
        assert _painted_alpha(node) == pytest.approx(swatch[z], abs=0.005)
    assert min(swatch.values()) < 0.5


# ---------------------------------------------------------------------------
# The numeric auxiliary scale: what scale= each of the five channels honors
# ---------------------------------------------------------------------------
#
# Size, the opacity family and StrokeWidth share one linear scale, checked by
# one Rust gate at construction and at render. Each linear-scale key is
# either refused, honored (the marks and the legend change together and
# agree), or inert, and the table below names every key the linear scale
# accepts, so a key added later fails the coverage cell until it is placed.

_NUMERIC_AUX_CLASSES: dict[str, type] = {"size": fm.Size, **_LEGEND_CHANNEL_CLASSES}

# A range each channel can draw, away from its default band.
_ALT_RANGE: dict[str, list[float]] = {
    "size": [20.0, 400.0],
    "opacity": [0.2, 0.6],
    "fill_opacity": [0.2, 0.6],
    "stroke_opacity": [0.2, 0.6],
    "stroke_width": [0.5, 6.0],
}

# Key -> (disposition, the scale dict that sets it, the dict it is compared with).
_LINEAR_KEY_CASES: dict[str, tuple[str, Callable[[str], dict], Callable[[str], dict] | None]] = {
    "domain": ("honored", lambda ch: {"domain": [0.0, 100.0]}, lambda ch: {}),
    "range": ("honored", lambda ch: {"range": _ALT_RANGE[ch]}, lambda ch: {}),
    "reverse": ("honored", lambda ch: {"reverse": True}, lambda ch: {}),
    "nice": (
        "honored",
        lambda ch: {"domain": [3.3, 47.7], "nice": True},
        lambda ch: {"domain": [3.3, 47.7]},
    ),
    "zero": ("honored", lambda ch: {"zero": True}, lambda ch: {}),
    "domainParam": ("honored", lambda ch: {"domain": fm.param("d", value=[0.0, 100.0])}, None),
    "clamp": (
        "inert",
        lambda ch: {"domain": [15.0, 35.0], "clamp": False},
        lambda ch: {"domain": [15.0, 35.0], "clamp": True},
    ),
    "padding": ("refused", lambda ch: {"padding": 0.1}, None),
    "scheme": ("refused", lambda ch: {"scheme": "blues"}, None),
    # The positional resolver's flag for a scale it injects itself; the
    # Python door never writes it.
    "systemInjected": ("internal", lambda ch: {}, None),
}


def test_the_key_table_places_every_key_the_linear_scale_accepts():
    from ferrum._core import scale_accepted_keys

    assert set(_LINEAR_KEY_CASES) == set(scale_accepted_keys("linear"))


def _scaled_point(channel: str, scale: dict) -> fm.Chart:
    cls = _NUMERIC_AUX_CLASSES[channel]
    chart = P(**{channel: cls("z", scale=scale) if scale else cls("z")})
    domain = scale.get("domain")
    if isinstance(domain, fm.Parameter):
        chart = chart.add_params(domain)
    return chart


def _marks_and_legend(chart: Any, channel: str) -> tuple[list, tuple]:
    """The marks' drawn values, and the legend's entries, after checking they agree.

    Every legend entry whose label is some row's ``z`` has the swatch that
    row's mark draws.
    """
    values = _quiet(lambda: _mark_values(chart, channel))
    ((_, entries),) = _quiet(lambda: interactive_legend(chart, channel)).items()
    by_z: dict[float, set] = {}
    for z, value in zip(df["z"].to_list(), values, strict=True):
        by_z.setdefault(z, set()).add(round(value, 6))
    for label, swatch in entries:
        if float(label) in by_z:
            assert by_z[float(label)] == {round(swatch, 6)}, (label, swatch, by_z)
    return values, entries


_LINEAR_KEY_MATRIX = {
    f"{channel}/{key}": (channel, key)
    for channel in _NUMERIC_AUX_CLASSES
    for key in _LINEAR_KEY_CASES
    if _LINEAR_KEY_CASES[key][0] != "internal"
}


@pytest.mark.parametrize("case", sorted(_LINEAR_KEY_MATRIX))
def test_each_linear_scale_key_is_refused_or_drawn_by_marks_and_legend_together(case):
    """A refused key raises at construction; an honored key moves the marks and the legend
    together; ``clamp`` is inert, since these scales always clamp.

    Rejects a key dropped without a word (the honored case would draw the
    comparison's marks), and marks and legend that disagree.
    """
    channel, key = _LINEAR_KEY_MATRIX[case]
    disposition, scale, compared = _LINEAR_KEY_CASES[key]
    if disposition == "refused":
        with pytest.raises(ValueError, match=rf"^{channel} scale: {key} is not supported"):
            _scaled_point(channel, scale(channel))
        return
    chart = _scaled_point(channel, scale(channel))
    got = _marks_and_legend(chart, channel)
    if compared is None:
        # domainParam: the parameter's value draws as the literal domain does.
        assert got == _marks_and_legend(_scaled_point(channel, {"domain": [0.0, 100.0]}), channel)
        compared = lambda ch: {}  # noqa: E731
    base = _marks_and_legend(_scaled_point(channel, compared(channel)), channel)
    if disposition == "inert":
        assert got == base
    else:
        assert got[0] != base[0] and got[1] != base[1]


def test_size_reverse_swaps_the_areas_and_the_legend_with_them():
    """Reversed, the largest ``z`` draws the smallest dot and the legend says so."""
    values, entries = _marks_and_legend(_scaled_point("size", {"reverse": True}), "size")
    zs = df["z"].to_list()
    assert values[zs.index(max(zs))] < values[zs.index(min(zs))]
    swatches = [s for _, s in entries]
    assert swatches == sorted(swatches, reverse=True)


# Linear, Log, Pow and Sqrt each map through their own transform; "ordinal"
# takes the still-unsupported-type slot the table always needs one of.
_MALFORMED_DOOR_SCALES: dict[str, tuple[dict, str]] = {
    "ordinal": ({"type": "ordinal"}, "type 'ordinal' is not supported"),
    "padding": ({"padding": 0.1}, "padding is not supported"),
    "scheme": ({"scheme": "blues"}, "scheme is not supported"),
    "domain-3": ({"domain": [1.0, 2.0, 3.0]}, "domain must have length 2; got 3"),
    "range-1": ({"range": [1.0]}, "range must have length 2; got 1"),
}

# The ``zero`` row on a descending domain: the widened pair keeps its order.
# Case -> (the scale that sets ``zero``, the literal domain it draws as).
_ZERO_DESCENDING_CASES: dict[str, tuple[Callable[[], dict], list[float]]] = {
    "descending": (lambda: {"domain": [100.0, 5.0], "zero": True}, [100.0, 0.0]),
    "descending-negative": (lambda: {"domain": [-5.0, -20.0], "zero": True}, [0.0, -20.0]),
    "descending-param": (
        lambda: {"domain": fm.param("d", value=[100.0, 5.0]), "zero": True},
        [100.0, 0.0],
    ),
}


@pytest.mark.parametrize("case", sorted(_ZERO_DESCENDING_CASES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_zero_on_a_descending_domain_keeps_it_descending(channel, case):
    """``zero=True`` on a descending domain widens it to zero and keeps it descending.

    The marks and the legend draw exactly what the widened descending literal
    domain draws. Rejects widening the pair as if it ascended, which flattens
    ``[100, 5]`` into the reversal-free ``[0, 5]``.
    """
    scale, literal = _ZERO_DESCENDING_CASES[case]
    got = _marks_and_legend(_scaled_point(channel, scale()), channel)
    assert got == _marks_and_legend(_scaled_point(channel, {"domain": literal}), channel)


@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_a_channel_class_refuses_a_scale_it_cannot_draw_at_construction(channel):
    """Constructing the channel with a scale it cannot draw raises the gate's text.

    The refusal comes at construction, before any door builds a chart; the
    wire's render-time refusal is the next test.
    """
    cls = _NUMERIC_AUX_CLASSES[channel]
    for scale, reason in _MALFORMED_DOOR_SCALES.values():
        with pytest.raises(ValueError, match=rf"^{channel} scale: {re.escape(reason)}"):
            cls("z", scale=scale)


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("scale_id", sorted(_MALFORMED_DOOR_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_a_wire_spec_is_refused_at_render_with_the_construction_text(channel, scale_id, kind):
    """A spec that reaches the wire without a channel class refuses at render, word for word."""
    from ferrum._core import render_interactive, render_svg

    scale, _reason = _MALFORMED_DOOR_SCALES[scale_id]
    cls = _NUMERIC_AUX_CLASSES[channel]
    with pytest.raises(ValueError) as built:
        cls("z", scale=scale)
    wire = _wire(P(**{channel: "z"}))
    wire["encoding"][channel]["scale"] = {"type": "linear", **scale}
    renderer = render_svg if kind == "static" else render_interactive
    with pytest.raises(ValueError) as rendered:
        renderer(fm.ChartSpec.from_json(json.dumps(wire)), df.to_arrow(), viewport=(400.0, 300.0))
    assert str(rendered.value).endswith(str(built.value))


# ---------------------------------------------------------------------------
# Log, Pow and Sqrt map the five numeric auxiliary channels
# ---------------------------------------------------------------------------
#
# Size, the opacity family and stroke width take a ``LogScale``,
# ``PowScale`` or ``SqrtScale`` through the identical gate and transform the
# positional resolver gives x/y. Every numeric auxiliary channel clamps an
# out-of-domain value to the nearer range end on every scale kind (a
# positional axis defaults to not clamping): x/y and the auxiliary channels
# agree only on the transform itself and on a ``Log`` value's side -- a
# value on the far side of zero from the domain is left out of the channel
# (dropped on x/y, drawn at the channel default on an auxiliary channel),
# not clamped. ``tests._aux_scale_oracle`` mirrors ``LogScaleData::scale``/
# ``PowScaleData::scale``'s in-domain mapping in Python, so each cell
# asserts the exact mapped value and legend swatch -- never merely that
# the render changed.

_LOG_BASE = 3.0
_POW_EXPONENT = 3.0
_TRANSFORM_DOMAIN = (5.0, 50.0)  # positive and away from df["z"]'s 10..45 extent


# ``build_size_scale``/``build_stroke_width_scale`` nice a data-derived domain
# by default (``nice_data_domain=true``, auxiliary.rs); the opacity family's
# builder passes ``false``, so it never nices even under Log.
_AUX_NICES_DATA_DOMAIN = frozenset({"size", "stroke_width"})


def _data_derived_domain(channel: str, kind: str, lo: float, hi: float) -> tuple[float, float]:
    # A degenerate domain (a constant column) nices to itself: LogScaleData::
    # nice returns the domain unchanged when its two magnitudes are equal,
    # the same early return LinearScaleData::nice and PowScaleData::nice take.
    if lo == hi:
        return (lo, hi)
    if kind == "log" and channel in _AUX_NICES_DATA_DOMAIN:
        return niced_log_domain(lo, hi, _LOG_BASE)
    return (lo, hi)


_TRANSFORM_SCALES: dict[str, Callable[..., Any]] = {
    "log": lambda **kw: fm.LogScale(base=_LOG_BASE, **kw),
    "pow": lambda **kw: fm.PowScale(exponent=_POW_EXPONENT, **kw),
    "sqrt": fm.SqrtScale,
}

_TRANSFORM_MAP: dict[str, Callable[[float, tuple[float, float], tuple[float, float]], float]] = {
    "log": lambda x, d, r: log_map(x, d, r, base=_LOG_BASE),
    "pow": lambda x, d, r: pow_map(x, d, r, exponent=_POW_EXPONENT),
    "sqrt": lambda x, d, r: pow_map(x, d, r, exponent=0.5),
}


def _drawn_value(channel: str, mapped: float) -> float:
    """The literal drawn attribute for a range-space *mapped* value.

    Size maps onto an area; the mark draws its radius
    (``sqrt(area / pi)``, ``render/marks/point.rs``). Every other numeric
    auxiliary channel draws its mapped value directly.
    """
    if channel == "size":
        return math.sqrt(mapped / math.pi) if mapped > 0.0 else 0.0
    return mapped


def _default_band(channel: str) -> tuple[float, float]:
    """The channel's default output band: an explicit ``[0, 100]`` domain's two
    endpoints, read directly off two rows at ``z=0`` and ``z=100`` under
    ``LinearScale``'s unset range -- the theme band every numeric auxiliary
    channel defaults to, whatever kind later maps onto it.
    """
    cls = _NUMERIC_AUX_CLASSES[channel]
    probe = pl.DataFrame({"v": [1.0, 2.0], "y": [1.0, 2.0], "z": [0.0, 100.0]})
    chart = mk(
        "point",
        probe,
        x="v",
        y="y",
        **{channel: cls("z", scale=fm.LinearScale(domain=[0.0, 100.0]))},
    )
    values = _quiet(lambda: _mark_values(chart, channel))
    if channel == "size":
        values = [math.pi * v * v for v in values]
    return (values[0], values[1])


def _transform_doors(
    channel: str, kind: str, domain: tuple[float, float], range_: tuple[float, float]
) -> dict[str, Callable[[], Any]]:
    cls = _NUMERIC_AUX_CLASSES[channel]
    scale = _TRANSFORM_SCALES[kind](domain=list(domain), range=list(range_))
    return _binding_doors(lambda: {channel: cls("z", scale=scale)})


def _assert_transform_matches_oracle(
    chart: Any,
    channel: str,
    kind: str,
    domain: tuple[float, float],
    range_: tuple[float, float],
    zs: list[float],
) -> None:
    """Every mark and every legend entry maps through *kind*'s own oracle transform.

    Checks both surfaces against one independently computed mapping, not
    merely that the marks and the legend agree with each other.
    """
    mapper = _TRANSFORM_MAP[kind]
    expected = [_drawn_value(channel, mapper(z, domain, range_)) for z in zs]
    values = _quiet(lambda: _mark_values(chart, channel))
    assert values == pytest.approx(expected, abs=1e-6)
    ((_, entries),) = _quiet(lambda: interactive_legend(chart, channel)).items()
    for label, swatch in entries:
        expected_swatch = _drawn_value(channel, mapper(float(label), domain, range_))
        # The legend's displayed label is the tick's true position rounded
        # to display precision; re-deriving the oracle from that rounded
        # text recovers the swatch only to the same precision -- tighter
        # near a narrow niced domain's steep slope would false-fail on
        # rounding alone, not on a wrong mapping.
        assert swatch == pytest.approx(expected_swatch, rel=2e-3, abs=2e-3)


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_a_transform_scale_maps_each_row_and_legend_entry_through_its_own_math(channel, kind):
    """Log(base=3)/Pow(exponent=3)/Sqrt map size, stroke_width and the opacity
    family through the same per-kind transform the positional resolver gives
    x/y, not a linear fallback, under an explicit domain and range.

    Rejects a gate that resolves every kind as ``linear``: the oracle
    mirrors ``LogScaleData``/``PowScaleData``'s own ``scale()`` directly,
    so a row's drawn value, and every legend swatch (not only the ones
    a data row happens to land on), must equal that formula's output.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN), range=list(range_))
    chart = P(**{channel: _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)})
    _assert_transform_matches_oracle(
        chart, channel, kind, _TRANSFORM_DOMAIN, range_, df["z"].to_list()
    )


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_a_data_derived_domain_nices_for_log_and_stays_raw_for_pow_and_sqrt(channel, kind):
    """No ``domain=``: on size and stroke_width, Log nices the data extent to
    the enclosing power of its base, as a data-derived positional Log axis
    would; the opacity family's own builder never nices, Log included; Pow
    and Sqrt carry no ``nice`` parameter at all and always keep the raw
    10..45 extent.

    Rejects a data-derived domain resolved as if it were explicit (the
    unrounded 10..45 for a nicing channel's Log), and a Pow/Sqrt domain, or
    the opacity family's Log domain, silently niced to round numbers they
    never ask for.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    zs = df["z"].to_list()
    domain = _data_derived_domain(channel, kind, min(zs), max(zs))
    scale = _TRANSFORM_SCALES[kind](range=list(range_))
    chart = P(**{channel: _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)})
    _assert_transform_matches_oracle(chart, channel, kind, domain, range_, zs)


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_the_default_range_feeds_the_same_band_a_linear_scale_uses(channel, kind):
    """No ``range=``: Log/Pow/Sqrt map onto the exact same theme band
    ``LinearScale``'s default range would use for this channel.

    Rejects a kind-specific default band: the three kinds would draw
    differently-ranged legends from the same unset ``range=``.
    """
    band = _default_band(channel)
    scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN))
    chart = P(**{channel: _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)})
    _assert_transform_matches_oracle(
        chart, channel, kind, _TRANSFORM_DOMAIN, band, df["z"].to_list()
    )


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_reverse_swaps_the_mapped_values(channel, kind):
    """``reverse=True`` swaps the range endpoints before the transform maps
    through them, on every kind -- the largest ``z`` draws the smallest
    value, matching ``reverse``'s existing linear-scale contract
    (``test_size_reverse_swaps_the_areas_and_the_legend_with_them``).
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    scale = _TRANSFORM_SCALES[kind](
        domain=list(_TRANSFORM_DOMAIN), range=list(range_), reverse=True
    )
    chart = P(**{channel: _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)})
    _assert_transform_matches_oracle(
        chart, channel, kind, _TRANSFORM_DOMAIN, (range_[1], range_[0]), df["z"].to_list()
    )


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_a_constant_column_maps_to_the_range_midpoint(channel, kind):
    """A constant column is a zero-width domain: every kind maps it to the
    range midpoint, never NaN -- ``degenerate_ratio``
    resolves the same way under a non-linear transform as under Linear,
    and ``LogScaleData::nice`` leaves a degenerate domain's two equal
    magnitudes unchanged rather than widening them to a full decade.

    The domain must stay data-derived: ``LogScale``/``PowScale``/
    ``SqrtScale`` all refuse an explicit zero-width domain at construction
    (``domain endpoints must differ``), the same refusal a positional
    channel gets.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    domain = _data_derived_domain(channel, kind, 0.3, 0.3)
    scale = _TRANSFORM_SCALES[kind](range=list(range_))
    chart = mk(
        "point",
        _CONSTANT,
        x="v",
        y="y",
        **{channel: _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)},
    )
    _assert_transform_matches_oracle(chart, channel, kind, domain, range_, _CONSTANT["z"].to_list())


# Positive out-of-domain values for Log (a non-positive value is left out,
# not clamped -- its own cells below); for Pow/Sqrt, -5 is below the
# stated domain but still on the transform's own (sign-preserving) domain.
_OUT_OF_DOMAIN_Z: dict[str, list[float]] = {
    "log": [1.0, 20.0, 500.0],
    "pow": [-5.0, 20.0, 500.0],
    "sqrt": [-5.0, 20.0, 500.0],
}


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_an_out_of_domain_value_clamps_to_the_nearer_range_end(channel, kind):
    """A value below the domain clamps to the lower range end; one above
    clamps to the upper end -- the numeric auxiliary channels' own clamp
    contract: every scale kind clamps on these five channels, where a
    positional axis defaults to not clamping.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    out_of_domain = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0], "z": _OUT_OF_DOMAIN_Z[kind]}
    )
    scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN), range=list(range_))
    chart = mk(
        "point",
        out_of_domain,
        x="v",
        y="y",
        **{channel: _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)},
    )
    values = _quiet(lambda: _mark_values(chart, channel))
    assert len(values) == 3
    assert values[0] == pytest.approx(_drawn_value(channel, range_[0]), abs=1e-6)
    assert values[2] == pytest.approx(_drawn_value(channel, range_[1]), abs=1e-6)


_TRANSFORM_DOORS = ("+", "L-own", "LCm", "Fc")


@pytest.mark.parametrize("door", _TRANSFORM_DOORS)
@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_a_transform_scale_draws_one_legend_alike_on_every_door(channel, kind, door):
    """Log/Pow/Sqrt draw the flat chart's one legend on ``+``, ``Chart.layer``,
    ``LayerChart`` and a facet, static and interactive alike -- the same
    door coverage the linear scale is pinned on above.

    Rejects a door that resolves the auxiliary scale independently per
    layer or per panel: the legend or a mark's swatch would disagree with
    the flat chart's own oracle-verified mapping, pinned above.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    doors = _transform_doors(channel, kind, _TRANSFORM_DOMAIN, range_)
    reference = _quiet(lambda: interactive_legend(doors["F"](), channel))
    ((title, entries),) = reference.items()

    chart = _quiet(doors[door])
    svg = _quiet(chart.to_svg)
    assert legend_set(svg) == {title: tuple(label for label, _ in entries)}
    assert _legend_title_count(svg, title) == 1
    assert _quiet(lambda: interactive_legend(chart, channel)) == reference


def test_a_log_legends_ticks_are_the_scales_own_decades():
    """The size legend's labels under ``LogScale(base=3)`` are themselves
    powers of 3 within the domain, not ``LinearScale``'s evenly spaced
    ``nice_ticks`` over the same domain -- the legend reads the Log
    scale's own tick algorithm, not a second, scale-blind generator.
    """
    domain = [3.0, 243.0]  # 3**1 .. 3**5
    log_chart = P(size=fm.Size("z", scale=fm.LogScale(base=3.0, domain=domain)))
    ((_, log_entries),) = _quiet(lambda: interactive_legend(log_chart, "size")).items()
    log_labels = [float(label) for label, _ in log_entries]
    for label in log_labels:
        exponent = math.log(label, 3.0)
        assert abs(exponent - round(exponent)) < 1e-6, log_labels

    linear_chart = P(size=fm.Size("z", scale=fm.LinearScale(domain=domain)))
    ((_, linear_entries),) = _quiet(lambda: interactive_legend(linear_chart, "size")).items()
    linear_labels = [float(label) for label, _ in linear_entries]
    assert log_labels != linear_labels


def test_log_scale_ticks_follow_the_decade_then_multiple_then_linear_ladder():
    """``LogScale.ticks(count)`` feeds every axis and legend of a Log scale:
    whole powers of the base first, then 1-3/1-2-5/1-9-style multiples when
    the powers alone fall short, then linear ``nice_ticks`` for a span
    under one decade or a non-integer base. A degenerate domain's own
    single-value ticks are covered by the constant-column cells above.
    """
    assert fm.LogScale(domain=[1.0, 1000.0]).ticks(5) == [1.0, 10.0, 100.0, 1000.0]
    assert fm.LogScale(domain=[1.0, 1000.0]).ticks(10) == [
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    ]
    assert fm.LogScale(domain=[1.0, 50.0]).ticks(5) == [1.0, 3.0, 10.0, 30.0]
    assert fm.LogScale(base=3.0, domain=[3.0, 243.0]).ticks(5) == [3.0, 9.0, 27.0, 81.0, 243.0]
    assert fm.LogScale(domain=[-1000.0, -1.0]).ticks(5) == [-1000.0, -100.0, -10.0, -1.0]
    # A span under one decade has no decade structure to round to; it falls
    # to the same linear nice_ticks a LinearScale over the same domain uses.
    narrow_log = fm.LogScale(domain=[3.0, 7.0]).ticks(5)
    narrow_linear = fm.LinearScale(domain=[3.0, 7.0]).ticks(5)
    assert narrow_log == narrow_linear


# ---------------------------------------------------------------------------
# A Log scale maps only its admitted side; the other side is left out of
# the channel (dropped on x/y, drawn at the channel default on an
# auxiliary channel) and warned once, naming the row count. With no
# stated domain the side is always the positive one -- never inferred
# from the data. A channel with nothing left on its side refuses, naming
# the remedy; a stated negative domain admits the negative side instead.
# ---------------------------------------------------------------------------

_LOG_WARN_CHANNELS = ("x", "y", *sorted(_NUMERIC_AUX_CLASSES))

# Each scenario has exactly one off-side row among three on-side ones.
_LOG_LEFT_OUT_SCENARIOS: dict[str, list[float]] = {
    "negative": [-5.0, 1.0, 10.0, 100.0],
    "zero": [0.0, 1.0, 10.0, 100.0],
}


def _log_channel_chart(channel: str, data: pl.DataFrame, scale: Any) -> fm.Chart:
    """A point chart binding *scale* to *channel* ("x", "y", or a numeric
    auxiliary channel), reading the test column "v"."""
    if channel == "x":
        return fm.Chart(data).mark_point().encode(x=fm.X("v", scale=scale), y="y")
    if channel == "y":
        return fm.Chart(data).mark_point().encode(x="y", y=fm.Y("v", scale=scale))
    cls = _NUMERIC_AUX_CLASSES[channel]
    return fm.Chart(data).mark_point().encode(x="y", y="y", **{channel: cls("v", scale=scale)})


def _log_left_out_warning(channel: str, field: str, rows: int, side: str = "positive") -> str:
    """Mirrors ``RenderWarning::LogScaleRowsLeftOut``'s Display."""
    excluded = "zero or negative" if side == "positive" else "zero or positive"
    if channel in ("x", "y"):
        return (
            f"{channel} scale: {rows} row(s) of '{field}' are {excluded}, which a LogScale "
            "cannot place, so they are not drawn; filter them, or use fm.SymlogScale() for "
            "data that reaches zero"
        )
    return (
        f"{channel} scale: {rows} row(s) of '{field}' are {excluded}, which a LogScale cannot "
        f"map, so they are not drawn; filter them, or use fm.SqrtScale()"
    )


@pytest.mark.parametrize("scenario", sorted(_LOG_LEFT_OUT_SCENARIOS))
@pytest.mark.parametrize("channel", _LOG_WARN_CHANNELS)
def test_a_log_scale_leaves_an_off_side_row_out_and_warns_once(channel, scenario):
    """A value on the far side of zero from a Log scale's domain (here, no
    stated domain, so the positive side) is left out of the channel --
    not drawn, on x/y or on an auxiliary channel alike, static and
    interactive -- and the chart warns once, naming the row count.

    Rejects a row silently dropped with no warning, and a row drawn at
    the auxiliary channel's default as if it were merely out-of-domain
    rather than off-side.
    """
    vs = _LOG_LEFT_OUT_SCENARIOS[scenario]
    data = pl.DataFrame({"v": vs, "y": [1.0, 2.0, 3.0, 4.0]})
    range_ = None if channel in ("x", "y") else tuple(float(v) for v in _ALT_RANGE[channel])
    scale = fm.LogScale(**({} if range_ is None else {"range": list(range_)}))
    chart = _log_channel_chart(channel, data, scale)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    messages = [str(w.message) for w in caught]
    assert messages == [_log_left_out_warning(channel, "v", 1)]
    # Static: the off-side row builds no mark at all.
    assert tag_counts(svg).get("circle", 0) == 3

    if channel in ("x", "y"):
        return

    # Interactive: the remaining three rows map through the Log transform
    # onto the data-derived domain (the off-side row contributes nothing
    # to it), not merely "not the default".
    on_side = [v for v in vs if v > 0.0]
    domain = (min(on_side), max(on_side))
    expected = [_drawn_value(channel, log_map(v, domain, range_, base=10.0)) for v in on_side]
    values = _quiet(lambda: _mark_values(chart, channel))
    assert values == pytest.approx(expected, abs=1e-6)


def test_an_off_side_log_aux_row_keeps_every_other_rows_data_index_and_key():
    """A row the Log aux drop removes does not renumber the survivors: each
    surviving node's ``data_indices`` entry (and its ``key``) still names
    its own source row, exactly as an off-side x/y row's drop already
    does -- not its position in the filtered batch.

    Rejects data_indices renumbered by position in the filtered batch
    (``[0, 1, 2, 3]``), which would point a WASM selection or transition
    at the wrong row in a sibling layer.
    """
    data = pl.DataFrame(
        {
            "k": ["a", "b", "c", "d", "e"],
            "v": [1.0, -5.0, 10.0, None, 100.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    aux_chart = (
        fm.Chart(data)
        .mark_point()
        .encode(x="y", y="y", size=fm.Size("v", scale=fm.LogScale()), key="k")
    )
    x_chart = fm.Chart(data).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="y", key="k")

    aux_svg = _quiet(aux_chart.to_svg)
    assert tag_counts(aux_svg).get("circle", 0) == 4

    ((aux_batch,),) = [
        (b,) for p in _quiet(lambda: _scene(aux_chart))["panels"] for b in p["marks"]
    ]
    assert aux_batch["data_indices"] == [0, 2, 3, 4]
    assert aux_batch["keys"] == ["a", "c", "d", "e"]

    ((x_batch,),) = [(b,) for p in _quiet(lambda: _scene(x_chart))["panels"] for b in p["marks"]]
    assert x_batch["data_indices"] == [0, 2, 4]
    assert x_batch["keys"] == ["a", "c", "e"]


def test_a_layered_log_aux_drop_keeps_each_layers_source_rows():
    """A Log aux drop on one layer never renumbers a sibling layer that
    draws the same frame without dropping anything -- each layer's own
    ``data_indices`` names its own source rows, on both the ``+`` door
    and ``Chart.layer``.
    """
    data = pl.DataFrame({"v": [1.0, -5.0, 10.0, 100.0], "y": [1.0, 2.0, 3.0, 4.0]})

    def sized() -> fm.Chart:
        return (
            fm.Chart(data).mark_point().encode(x="y", y="y", size=fm.Size("v", scale=fm.LogScale()))
        )

    def ticked() -> fm.Chart:
        return fm.Chart(data).mark_tick().encode(x="v", y="y")

    plus_chart = _quiet(lambda: sized() + ticked())
    layer_chart = _quiet(lambda: sized().layer(layer("tick", x="v", y="y")))

    for chart in (plus_chart, layer_chart):
        batches = [b for p in _quiet(lambda: _scene(chart))["panels"] for b in p["marks"]]
        indices_by_kind = {b["kind"]: b["data_indices"] for b in batches}
        assert indices_by_kind["point"] == [0, 2, 3]
        assert indices_by_kind["tick"] == [0, 1, 2, 3]


_LOG_WARN_DOOR_CHANNELS = ("size", "x", "y")


def _log_warn_doors(channel: str, data: pl.DataFrame, scale: Any) -> dict[str, Callable[[], Any]]:
    """Every door that can carry one Log-scaled *channel* through one
    shared frame -- flat, ``+``, ``Chart.layer``, static and interactive
    ``LayerChart``, a faceted chart under shared and independent facet
    scale resolution, and a faceted ``Chart.layer``.

    For a positional channel, the host layer of the ``Chart.layer`` and
    faceted-``Chart.layer`` doors is a ``mark_rule`` bound only to the
    *other* position: ``mark_point`` refuses a layer with no positional
    channel of its own, so the host cannot simply omit *channel* the way
    an auxiliary channel's host omits size/opacity/stroke_width.
    """
    is_pos = channel in ("x", "y")

    def binder() -> fm.Chart:
        return _log_channel_chart(channel, data, scale)

    def partner() -> fm.Chart:
        # Draws through the scale *binder* establishes (a bare field, no
        # restated scale), reading the same column, so the whole-frame
        # tally counts it once rather than as a second distinct read.
        if channel == "x":
            return mk("point", data, x="v", y="y")
        if channel == "y":
            return mk("point", data, x="y", y="v")
        return mk("point", data, x="y", y="y")

    def host_and_layer() -> fm.Chart:
        if channel == "x":
            return mk("rule", data, y="y").layer(layer("point", x=fm.X("v", scale=scale), y="y"))
        if channel == "y":
            return mk("rule", data, x="y").layer(layer("point", x="y", y=fm.Y("v", scale=scale)))
        cls = _NUMERIC_AUX_CLASSES[channel]
        return mk("point", data, x="y", y="y").layer(
            layer("point", **{channel: cls("v", scale=scale)})
        )

    doors: dict[str, Callable[[], Any]] = {
        "F": binder,
        "+": lambda: binder() + partner(),
        "L": host_and_layer,
        "LCs": lambda: fm.LayerChart(binder(), partner()),
        "LCi": lambda: fm.LayerChart(binder(), partner()).interactive(),
        "LCm": lambda: fm.LayerChart(binder(), partner())._build_merged(),
        "Fc-shared": lambda: binder().facet("c"),
        "L-Fc": lambda: host_and_layer().facet("c"),
    }
    if is_pos:
        # Facet scale resolution has no "independent" mode for an
        # auxiliary channel (``Chart.share_scale`` takes only x/y): an
        # auxiliary scale always resolves over the whole frame regardless
        # of faceting, so this door exists only for x/y.
        doors["Fc-independent"] = lambda: (
            binder().facet("c").share_scale(**{channel: "independent"})
        )
    return doors


@pytest.mark.parametrize("channel", _LOG_WARN_DOOR_CHANNELS)
def test_a_log_channel_warns_once_for_the_whole_frame_on_every_door(channel):
    """A Log channel -- ``size`` or a positional ``x``/``y`` -- carried by
    one ``+`` operand, one ``Chart.layer`` layer, one static or interactive
    ``LayerChart`` member, or a faceted chart's shared or independent
    scale still warns exactly once, with the whole frame's row count --
    not once per layer and not once per facet panel.
    """
    data = pl.DataFrame(
        {"v": [-5.0, 1.0, 10.0, 100.0], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "p", "q", "q"]}
    )
    doors = _log_warn_doors(channel, data, fm.LogScale())
    expected = [_log_left_out_warning(channel, "v", 1)]
    for door, build in doors.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # LCi's ``.interactive()`` builds the scene eagerly at
            # construction, so *build* itself (not just the render call
            # that follows) must run inside the capture for its warning
            # to be caught.
            chart = build()
            if door == "LCi":
                json.loads(chart.scene_json)
            else:
                chart.to_svg()
        messages = [str(w.message) for w in caught if "LogScale cannot" in str(w.message)]
        assert messages == expected, f"{channel}/{door}: got {messages}"


def test_log_warns_identically_on_plus_and_layer_chart_for_two_frames_combined_count():
    """``b + v`` and ``fm.LayerChart(b, v)`` warn identically when ``b`` and
    ``v`` are two different frames, each contributing its own off-side row
    to the one whole-frame count -- parity between the ``+`` sugar and its
    ``LayerChart`` equivalent, not just within a single frame.
    """
    scale = fm.LogScale()
    b_data = pl.DataFrame({"v": [-5.0, 1.0, 10.0], "y": [1.0, 2.0, 3.0]})
    v_data = pl.DataFrame({"v": [-2.0, 4.0, 8.0], "y": [1.0, 2.0, 3.0]})

    def b() -> fm.Chart:
        return mk("point", b_data, x="y", y="y", size=fm.Size("v", scale=scale))

    def v() -> fm.Chart:
        return mk("point", v_data, x="y", y="y", size=fm.Size("v", scale=scale))

    expected = [_log_left_out_warning("size", "v", 2)]

    with warnings.catch_warnings(record=True) as caught_plus:
        warnings.simplefilter("always")
        (b() + v()).to_svg()
    assert [str(w.message) for w in caught_plus] == expected

    with warnings.catch_warnings(record=True) as caught_layer_chart:
        warnings.simplefilter("always")
        fm.LayerChart(b(), v()).to_svg()
    assert [str(w.message) for w in caught_layer_chart] == expected


@pytest.mark.parametrize("channel", ("x", "y"))
def test_an_all_off_side_member_beside_an_on_side_one_draws_and_warns_on_every_door(channel):
    """An overlay member whose Log channel has nothing on its side draws
    nothing beside a sibling that does, on every door -- it does not
    refuse the whole chart, and the single whole-frame warning carries
    the combined count.
    """
    pos = pl.DataFrame({"x": [1.0, 2.0, 3.0], "v": [1.0, 10.0, 100.0]})
    neg = pl.DataFrame({"x": [1.5, 2.5], "v": [-5.0, -7.0]})

    def chart(data: pl.DataFrame) -> fm.Chart:
        if channel == "x":
            return fm.Chart(data).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="x")
        return fm.Chart(data).mark_point().encode(x="x", y=fm.Y("v", scale=fm.LogScale()))

    def a() -> fm.Chart:
        return chart(pos)

    def b() -> fm.Chart:
        return chart(neg)

    expected = [_log_left_out_warning(channel, "v", 2)]

    static_doors: dict[str, Callable[[], fm.Chart]] = {
        "+": lambda: a() + b(),
        "+rev": lambda: b() + a(),
        "LCs": lambda: fm.LayerChart(a(), b()),
        "LCsrev": lambda: fm.LayerChart(b(), a()),
    }
    for door, build in static_doors.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = build().to_svg()
        assert tag_counts(svg).get("circle", 0) == 3, f"{channel}/{door}"
        assert [str(w.message) for w in caught] == expected, f"{channel}/{door}"

    with warnings.catch_warnings(record=True) as caught_interactive:
        warnings.simplefilter("always")
        json.loads(fm.LayerChart(a(), b()).interactive().scene_json)
    assert [str(w.message) for w in caught_interactive] == expected, f"{channel}/LCi"


@pytest.mark.parametrize("channel", ("x", "y"))
def test_a_stated_log_domain_warns_once_on_every_door(channel):
    """A stated Log domain's off-side row warns exactly once, on every
    door, not once per member that happens to read it.
    """
    one = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "v": [1.0, 10.0, -5.0, 100.0]})
    scale = fm.LogScale(domain=[1.0, 100.0])

    def a() -> fm.Chart:
        if channel == "x":
            return fm.Chart(one).mark_point().encode(x=fm.X("v", scale=scale), y="x")
        return fm.Chart(one).mark_point().encode(x="x", y=fm.Y("v", scale=scale))

    def b() -> fm.Chart:
        if channel == "x":
            return fm.Chart(one).mark_line().encode(x="v", y="x")
        return fm.Chart(one).mark_line().encode(x="x", y="v")

    expected = [_log_left_out_warning(channel, "v", 1)]

    static_doors: dict[str, Callable[[], fm.Chart]] = {
        "+": lambda: a() + b(),
        "L": lambda: a().layer(
            layer("line", **({"x": "v", "y": "x"} if channel == "x" else {"x": "x", "y": "v"}))
        ),
        "LCs": lambda: fm.LayerChart(a(), b()),
    }
    for door, build in static_doors.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build().to_svg()
        assert [str(w.message) for w in caught] == expected, f"{channel}/{door}"

    with warnings.catch_warnings(record=True) as caught_interactive:
        warnings.simplefilter("always")
        json.loads(fm.LayerChart(a(), b()).interactive().scene_json)
    assert [str(w.message) for w in caught_interactive] == expected, f"{channel}/LCi"


def _rust_float(v: float) -> str:
    """Formats *v* the way Rust's ``{}`` Display does for ``f64``: an
    integer-valued float prints with no trailing ``.0``."""
    return str(int(v)) if v == int(v) else repr(v)


def _log_no_value_reason(
    channel: str,
    field: str,
    rows: int,
    side: str,
    has_zero: bool,
    opposite: tuple[float, float] | None,
) -> str:
    """Mirrors ``log_no_value_on_side_reason`` (``crates/ferrum-core/src/
    render/mod.rs``), the ``RenderError::LogScaleNoValueOnSide`` text a
    channel with nothing left on its side refuses with."""
    maps_word = side
    sign_word = "negative" if side == "positive" else "positive"
    alt_scale = (
        "fm.SymlogScale()" if channel in ("x", "y") else "fm.SqrtScale() or fm.LinearScale()"
    )
    if not has_zero and opposite is not None:
        lo, hi = opposite
        reason = (
            f"every value of '{field}' ({rows} rows) is {sign_word}, and a LogScale with no "
            f"domain maps only {maps_word} values; pass an explicit {sign_word} domain, e.g. "
            f"fm.LogScale(domain=[{_rust_float(lo)}, {_rust_float(hi)}]), to draw a mirrored "
            "log scale"
        )
    else:
        excluded = "zero or negative" if side == "positive" else "zero or positive"
        if opposite is not None:
            lo, hi = opposite
            reason = (
                f"{rows} row(s) of '{field}' are {excluded}, and a LogScale with no domain "
                f"maps only {maps_word} values; use {alt_scale}, or pass an explicit "
                f"{sign_word} domain, e.g. fm.LogScale(domain=[{_rust_float(lo)}, {_rust_float(hi)}])"
            )
        else:
            reason = (
                f"{rows} row(s) of '{field}' are {excluded}, and a LogScale with no domain "
                f"maps only {maps_word} values; use {alt_scale}"
            )
    return f"{channel} scale: {reason}"


# Each scenario's ``v`` column, with the shape ``log_no_value_on_side_reason``
# reads from it: every non-null, finite value is off-side (none positive).
_LOG_NO_VALUE_SCENARIOS: dict[str, tuple[list[float], int, bool, tuple[float, float] | None]] = {
    "all-negative": ([-1000.0, -100.0, -10.0, -1.0], 4, False, (-1000.0, -1.0)),
    "all-zero": ([0.0, 0.0, 0.0, 0.0], 4, True, None),
    "zero-and-negative": ([-5.0, 0.0, -10.0], 3, True, (-10.0, -5.0)),
}


@pytest.mark.parametrize("scenario", sorted(_LOG_NO_VALUE_SCENARIOS))
@pytest.mark.parametrize("channel", _LOG_WARN_CHANNELS)
def test_a_log_channel_with_nothing_on_its_side_refuses_naming_the_remedy(channel, scenario):
    """A Log channel with no stated domain refuses when every one of its
    values is off the positive side -- naming the row count and, when any
    value is strictly negative, the explicit domain that would draw it
    mirrored.

    Rejects a mirrored negative axis drawn from data alone (x/y) and
    every mark drawn at one end with no word about it (an auxiliary
    channel) -- the side is never inferred from the data.
    """
    vs, rows, has_zero, opposite = _LOG_NO_VALUE_SCENARIOS[scenario]
    data = pl.DataFrame({"v": vs, "y": [float(i) for i in range(len(vs))]})
    chart = _log_channel_chart(channel, data, fm.LogScale())
    expected = _log_no_value_reason(channel, "v", rows, "positive", has_zero, opposite)
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == expected


def test_a_flipped_layer_charts_log_refusal_names_the_written_channel():
    """Under ``CoordFlip``, a ``LayerChart`` whose members' x channel has
    nothing on its Log side refuses naming ``x`` -- the channel the user
    actually wrote -- not the post-flip resolved axis.
    """
    data = pl.DataFrame({"v": [-5.0, -7.0], "y": [1.0, 2.0]})

    def flipped() -> fm.Chart:
        return (
            fm.Chart(data)
            .mark_point()
            .encode(x=fm.X("v", scale=fm.LogScale()), y="y")
            .coord(fm.CoordFlip())
        )

    with pytest.raises(ValueError) as refused:
        fm.LayerChart(flipped(), flipped()).to_svg()
    message = str(refused.value)
    assert "x scale" in message or "shared x domain" in message
    assert "shared y" not in message


_LOG_DOMAIN_REFUSALS: dict[str, tuple[list[float], str]] = {
    "zero": ([0.0, 100.0], "log scale domain must not contain 0"),
    "sign-mismatch": ([-10.0, 1000.0], "log scale domain endpoints must have the same sign"),
}


@pytest.mark.parametrize("case", sorted(_LOG_DOMAIN_REFUSALS))
def test_an_explicit_log_domain_touching_or_crossing_zero_refuses_on_x_at_render(case):
    """A wire spec that states a Log domain with a zero endpoint or
    endpoints of different signs refuses at render -- the same text
    ``fm.LogScale(domain=...)`` itself gives at construction -- even
    though this door reaches the render without a channel class to
    refuse it first.
    """
    domain, reason = _LOG_DOMAIN_REFUSALS[case]
    from ferrum._core import render_svg

    wire = _wire(P())
    wire["encoding"]["x"]["scale"] = {"type": "log", "domain": domain}
    with pytest.raises(ValueError, match=rf"^x scale: {re.escape(reason)}$"):
        render_svg(fm.ChartSpec.from_json(json.dumps(wire)), df.to_arrow(), viewport=(400.0, 300.0))


@pytest.mark.parametrize("case", sorted(_LOG_DOMAIN_REFUSALS))
@pytest.mark.parametrize("channel", sorted(_NUMERIC_AUX_CLASSES))
def test_an_explicit_log_domain_touching_or_crossing_zero_refuses_at_aux_construction(
    channel, case
):
    """Every numeric auxiliary channel refuses the same malformed Log
    domain at construction, through the dict door that bypasses
    ``fm.LogScale``'s own eager validation.
    """
    domain, reason = _LOG_DOMAIN_REFUSALS[case]
    cls = _NUMERIC_AUX_CLASSES[channel]
    with pytest.raises(ValueError, match=rf"^{channel} scale: {re.escape(reason)}$"):
        cls("v", scale={"type": "log", "domain": domain})


_NEGATIVE_DOMAIN: tuple[float, float] = (-1000.0, -1.0)


def _negative_domain_scale(channel: str) -> Any:
    range_ = None if channel in ("x", "y") else tuple(float(v) for v in _ALT_RANGE[channel])
    kwargs: dict[str, Any] = {"domain": list(_NEGATIVE_DOMAIN)}
    if range_ is not None:
        kwargs["range"] = list(range_)
    return fm.LogScale(**kwargs), range_


@pytest.mark.parametrize("channel", _LOG_WARN_CHANNELS)
def test_a_stated_negative_domain_renders_every_row_mirrored(channel):
    """A Log scale whose stated domain is strictly negative draws an
    all-negative column mirrored -- the domain the earlier refusal's
    remedy suggests actually renders -- on x, y, and every numeric
    auxiliary channel alike: mirrored ticks (or, on an auxiliary channel,
    mirrored mapped values and a mirrored legend), with no row left out.
    """
    vs = [-1000.0, -100.0, -10.0, -1.0]
    data = pl.DataFrame({"v": vs, "y": [1.0, 2.0, 3.0, 4.0]})
    scale, range_ = _negative_domain_scale(channel)
    chart = _log_channel_chart(channel, data, scale)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    assert caught == []
    assert tag_counts(svg).get("circle", 0) == 4

    if channel in ("x", "y"):
        labels = axis_tick_labels(svg, axis=channel)
        for decade in ("-1000", "-100", "-10", "-1"):
            assert decade in labels, f"{channel}: missing decade label {decade!r}: {labels}"
        return

    expected = [_drawn_value(channel, log_map(v, _NEGATIVE_DOMAIN, range_, base=10.0)) for v in vs]
    values = _quiet(lambda: _mark_values(chart, channel))
    assert values == pytest.approx(expected, abs=1e-6)

    static_tol = 0.005 if channel == "opacity" else 1e-3
    static_values = _static_mark_values(svg, channel)
    assert static_values == pytest.approx(expected, abs=static_tol)

    ((title, entries),) = _quiet(lambda: interactive_legend(chart, channel)).items()
    assert title == "v"
    assert sorted(float(label) for label, _ in entries) == sorted(vs)

    legend_labels = legend_set(svg)["v"]
    assert sorted(float(label) for label in legend_labels) == sorted(vs)


@pytest.mark.parametrize("channel", _LOG_WARN_CHANNELS)
def test_a_positive_row_under_a_stated_negative_domain_is_left_out_and_warned(channel):
    """Under a stated negative Log domain, a row on the positive side (or
    zero) is left out -- not drawn, per the off-side rule, on x, y, and
    every numeric auxiliary channel alike -- and the chart warns once,
    "zero or positive": the mirror image of the default positive side's
    "zero or negative".
    """
    vs = [-1000.0, -100.0, -10.0, 5.0]
    data = pl.DataFrame({"v": vs, "y": [1.0, 2.0, 3.0, 4.0]})
    scale, range_ = _negative_domain_scale(channel)
    chart = _log_channel_chart(channel, data, scale)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    messages = [str(w.message) for w in caught]
    assert messages == [_log_left_out_warning(channel, "v", 1, side="negative")]
    assert tag_counts(svg).get("circle", 0) == 3

    if channel in ("x", "y"):
        return

    on_side = [v for v in vs if v < 0.0]
    expected = [
        _drawn_value(channel, log_map(v, _NEGATIVE_DOMAIN, range_, base=10.0)) for v in on_side
    ]
    values = _quiet(lambda: _mark_values(chart, channel))
    assert values == pytest.approx(expected, abs=1e-6)

    static_tol = 0.005 if channel == "opacity" else 1e-3
    static_values = _static_mark_values(svg, channel)
    assert static_values == pytest.approx(expected, abs=static_tol)


# ---------------------------------------------------------------------------
# A transform scale maps every mark that draws stroke_width or the opacity
# family, not only a point -- the one table Mark::draws_channel answers
# from, read through ferrum._core.mark_drawn_channels so this matrix
# cannot drift from it.
# ---------------------------------------------------------------------------

# fill_opacity/stroke_opacity/stroke_width each have one literal style
# key across every mark that draws them. Plain "opacity" has none of its
# own: a mark bakes its mapped value into whichever paint it actually
# has -- its fill's alpha on a fill-painted mark with no fill_opacity of
# its own (point, area), a literal fill-opacity attribute where it
# shares that attribute with fill_opacity (bar), its stroke's alpha on a
# stroke-only mark with no stroke_opacity of its own (line, rule), or a
# literal stroke-opacity attribute where it shares that attribute with
# stroke_opacity (tick, segment) -- so this matrix checks it too, table-
# driven per mark rather than hand-excluded to a subset.
_AUX_STYLE_CHANNELS = ("stroke_width", "fill_opacity", "stroke_opacity", "opacity")

_CHANNEL_MARK_POSITIONS: dict[str, dict[str, str]] = {
    "point": {"x": "v", "y": "y"},
    "line": {"x": "v", "y": "y"},
    "area": {"x": "v", "y": "y"},
    "bar": {"x": "k", "y": "y"},
    "tick": {"x": "v", "y": "y"},
    "rule": {"y": "y"},
    "segment": {"x": "v", "y": "y", "x2": "v", "y2": "z"},
}

# line/area draw one node (a polyline/path) for the whole series, at the
# first row's resolved value.
_SINGLE_NODE_MARKS = frozenset({"line", "area"})

# Marks whose plain "opacity" bakes into a paint's 8-bit alpha byte,
# both in the interactive scene (_INTERACTIVE_OPACITY_READ) and in the
# static SVG's rgba() fallback (_color_alpha) -- unlike bar/tick/segment,
# which carry it as a literal float attribute.
_BYTE_QUANTIZED_OPACITY_MARKS = frozenset({"point", "line", "area", "rule"})

# A segment's fill_opacity fills its arrowheads, drawn only with
# arrow=True (Mark::draws_opacity_channel's doc) -- plain `mk("segment",
# ...)` carries no arrowhead to map it onto, so this one (mark, channel)
# pair needs its own chart, built by `_mark_chart` below.
_SEGMENT_ARROW_CHANNELS = frozenset({"fill_opacity"})


def _marks_drawing(channel: str) -> list[str]:
    """Every mark in ``_CHANNEL_MARK_POSITIONS`` that draws *channel*, per
    ``ferrum._core.mark_drawn_channels`` (``Mark::draws_channel``'s own answer)."""
    from ferrum._core import mark_drawn_channels

    return [m for m in _CHANNEL_MARK_POSITIONS if channel in mark_drawn_channels(m)]


def _mark_chart(
    mark: str,
    channel: str,
    value: Any,
    data: pl.DataFrame = df,
    positions: dict[str, str] | None = None,
) -> fm.Chart:
    """*mark* bound to *channel* = *value* (left unbound when *value* is
    ``None``) over *positions* (``_CHANNEL_MARK_POSITIONS[mark]`` by
    default) and *data* (the module's own ``df`` by default)."""
    if positions is None:
        positions = _CHANNEL_MARK_POSITIONS[mark]
    encoding = {} if value is None else {channel: value}
    if mark == "segment" and channel in _SEGMENT_ARROW_CHANNELS:
        return fm.Chart(data).mark_segment(arrow=True).encode(**positions, **encoding)
    return mk(mark, data, **positions, **encoding)


# Where plain "opacity" lands in the interactive scene, per mark: a
# literal `style["opacity"]` whenever the scene also writes the mapped
# value there (point, tick, segment -- each ALSO echoes it into the
# paint alpha or stroke_opacity, so reading `style["opacity"]` directly
# is the one read that is never double-applied); otherwise baked into
# the paint's own alpha, with `style["opacity"]` left inert at 1.0
# (line, rule: `color.a`; area: `fill.a`), or into a literal
# `fill_opacity` that shares bar's own fill_opacity slot.
_INTERACTIVE_OPACITY_READ: dict[str, Callable[[dict], float]] = {
    "point": lambda s: s["opacity"],
    "line": lambda s: (s.get("color") or {}).get("a", 255) / 255,
    "area": lambda s: (s.get("fill") or {}).get("a", 255) / 255,
    "bar": lambda s: s.get("fill_opacity", 1.0),
    "tick": lambda s: s["opacity"],
    "rule": lambda s: (s.get("color") or {}).get("a", 255) / 255,
    "segment": lambda s: s["opacity"],
}


def _style_values(chart: Any, mark: str, channel: str) -> list[Any]:
    """Each interactive node's drawn *channel* value, across every mark kind
    that draws it in one node per row.

    A circle/rect-family node keys its outline width ``stroke_width``; a
    line-family node (line/tick/rule/segment/area) has only one width,
    keyed ``width``. A node whose style carries no key for *channel* at
    all -- a segment's shaft, when *channel* is ``fill_opacity`` (only
    its arrowhead carries that key) -- is skipped rather than counted as
    a null row: the two are different node *kinds*, not one row with a
    missing value. Plain ``opacity`` routes through ``_INTERACTIVE_
    OPACITY_READ`` instead, since its value is not always under a
    literal ``"opacity"`` key.
    """
    out = []
    for panel in _scene(chart)["panels"]:
        for batch in panel["marks"]:
            for node in batch["nodes"]:
                style = node.get("style") or {}
                if channel == "opacity":
                    out.append(_INTERACTIVE_OPACITY_READ[mark](style))
                    continue
                if channel == "stroke_width":
                    if "stroke_width" not in style and "width" not in style:
                        continue
                    out.append(style.get("stroke_width", style.get("width")))
                    continue
                if channel not in style:
                    continue
                out.append(style[channel])
    return out


_RGBA_ALPHA_RE = re.compile(r"rgba\([\d.]+,[\d.]+,[\d.]+,([\d.]+)\)")


def _color_alpha(color: str | None) -> float:
    """The alpha fraction baked into a color attribute by
    ``render/color/primitive.rs``'s ``fmt_svg``: 1.0 for a fully opaque
    ``#rrggbb`` (or no color/``"none"`` at all), the parsed fraction for
    its ``rgba(r,g,b,a)`` fallback (alpha < 1.0)."""
    if not color or color == "none":
        return 1.0
    m = _RGBA_ALPHA_RE.match(color)
    return float(m.group(1)) if m else 1.0


_MARK_SVG_TAG = {
    "point": "circle",
    "line": "polyline",
    "area": "path",
    "bar": "rect",
    "tick": "line",
    "rule": "line",
    "segment": "line",
}

# fill-opacity/stroke-opacity/stroke-width each have their own SVG
# attribute (render/svg.rs's push_fill_stroke), read directly; absence
# means the unscaled default.
_STATIC_ATTR = {"fill_opacity": "fill-opacity", "stroke_opacity": "stroke-opacity"}
_STATIC_ATTR_DEFAULT = {"fill_opacity": 1.0, "stroke_opacity": 1.0}

# Where plain "opacity" lands statically, per mark: a literal attribute
# it shares with fill_opacity (bar) or stroke_opacity (tick, segment,
# via push_fill_stroke), or baked into the paint color's own alpha
# (render/color/primitive.rs's fmt_svg) on a mark with no opacity-family
# attribute of its own to share (point, area: fill; line, rule: stroke).
# This is Rust's own per-mark choice of representation, not a
# discriminator this test enforces.
_STATIC_OPACITY_READ: dict[str, Callable[[dict[str, str]], float]] = {
    "point": lambda a: _color_alpha(a.get("fill")),
    "line": lambda a: _color_alpha(a.get("stroke")),
    "area": lambda a: _color_alpha(a.get("fill")),
    "bar": lambda a: float(a.get("fill-opacity", "1.0")),
    "tick": lambda a: float(a.get("stroke-opacity", "1.0")),
    "rule": lambda a: _color_alpha(a.get("stroke")),
    "segment": lambda a: float(a.get("stroke-opacity", "1.0")),
}


def _static_style_values(svg: str, mark: str, channel: str) -> list[float]:
    """Each static mark element's drawn *channel* value, in document order
    within the plot's clip groups -- the static-SVG counterpart to
    ``_style_values``."""
    layer = mark_layer(svg)
    if mark == "segment" and channel in _SEGMENT_ARROW_CHANNELS:
        # fill_opacity paints the arrowhead, a filled triangle <path>;
        # the shaft stays a plain <line>, carrying no fill at all.
        elements = svg_elements(layer, "path", lambda a: True)
    else:
        tag = _MARK_SVG_TAG[mark]
        keep = (
            (lambda a: a.get("stroke") not in THEME_CHROME_STROKES)
            if tag == "line"
            else (lambda a: True)
        )
        elements = svg_elements(layer, tag, keep)
    if channel == "opacity":
        return [_STATIC_OPACITY_READ[mark](a) for a in elements]
    if channel == "stroke_width":
        return [float(a.get("stroke-width", "0.0")) for a in elements]
    attr, default = _STATIC_ATTR[channel], _STATIC_ATTR_DEFAULT[channel]
    return [float(a.get(attr, str(default))) for a in elements]


def _static_mark_values(svg: str, channel: str, mark: str = "point") -> list[float]:
    """The static SVG's drawn *channel* value per *mark* (a point, by
    default -- the stated-negative-domain cells' only caller shape),
    mirroring ``_mark_values``'s interactive reader: ``size`` reads each
    circle's own ``r`` attribute directly (``_static_style_values`` has
    no size rule -- a point's radius is not one of the literal/baked
    style channels that function covers); every other aux channel
    routes through ``_static_style_values``.
    """
    if channel == "size":
        return [float(a["r"]) for a in svg_elements(mark_layer(svg), "circle", lambda a: True)]
    return _static_style_values(svg, mark, channel)


_MARK_CHANNEL_PAIRS = [
    (mark, channel) for channel in _AUX_STYLE_CHANNELS for mark in _marks_drawing(channel)
]

_OFF_SIDE_MARK_CHANNEL_PAIRS = [*_MARK_CHANNEL_PAIRS, ("point", "size"), ("line", "size")]


def _off_side_data() -> pl.DataFrame:
    """The module's own ``df`` with row 0 forced negative (off-side) and
    row 1 forced null, over the ``z`` column every pair in this matrix
    binds its channel to."""
    zs = df["z"].to_list()
    return df.with_columns(pl.Series("z", [-5.0, None, *zs[2:]]))


def _off_side_positions(mark: str) -> dict[str, str]:
    """``_CHANNEL_MARK_POSITIONS[mark]``, with segment's ``y2`` moved off
    ``'z'``: segment is the one mark whose position encoding reuses the
    same field this matrix binds its channel to, which would conflate a
    dropped/defaulted AUX row with a dropped/defaulted POSITION row.
    """
    positions = dict(_CHANNEL_MARK_POSITIONS[mark])
    if positions.get("y2") == "z":
        positions["y2"] = "r"
    return positions


def _off_side_chart(mark: str, channel: str, value: Any, data: pl.DataFrame) -> fm.Chart:
    return _mark_chart(mark, channel, value, data=data, positions=_off_side_positions(mark))


def _off_side_unbound_chart(mark: str, channel: str, data: pl.DataFrame) -> fm.Chart:
    """The same chart *_off_side_chart* builds, with *channel* left unbound."""
    return _mark_chart(mark, channel, None, data=data, positions=_off_side_positions(mark))


def _unbound_default(mark: str, channel: str, data: pl.DataFrame) -> float:
    """The value *channel* draws when left unbound on *mark* -- the value
    a null row still draws at, interactively."""
    chart = _off_side_unbound_chart(mark, channel, data)
    if channel == "size" and mark in _SINGLE_NODE_MARKS:
        node = next(
            n
            for p in _quiet(lambda: _scene(chart))["panels"]
            for b in p["marks"]
            for n in b["nodes"]
        )
        return node["style"]["width"]
    return _off_side_interactive_values(chart, mark, channel)[0]


def _expected_off_side_value(mark: str, channel: str, mapped: float) -> float:
    """The drawn value *mapped* (a scale's raw range-space output) becomes
    on *mark*: size converts area to a point's radius or a line's
    diameter (twice the radius); every other channel is drawn directly.
    """
    if channel != "size":
        return mapped
    radius = _drawn_value("size", mapped)
    return 2.0 * radius if mark in _SINGLE_NODE_MARKS else radius


def _off_side_interactive_values(chart: Any, mark: str, channel: str) -> list[float]:
    """Each surviving row's drawn *channel* value, from the interactive
    scene.

    Unlike ``_style_values`` (which skips a node whose style carries no
    key for *channel* at all, to separate a segment's shaft from its
    arrowhead), a null row's node -- or an entirely unbound channel's --
    omits a literal ``fill_opacity``/``stroke_opacity`` key the same way,
    so this reader defaults a missing key to 1.0 instead of skipping the
    node. A segment's shaft and arrowhead are told apart structurally
    instead (a ``"fill"`` paint key, present only on the arrowhead),
    which holds whether or not *channel* happens to be bound.
    """
    nodes = [n for p in _scene(chart)["panels"] for b in p["marks"] for n in b["nodes"]]
    if mark == "segment" and channel in _SEGMENT_ARROW_CHANNELS:
        nodes = [n for n in nodes if "fill" in n["style"]]
    if channel == "size":
        if mark in _SINGLE_NODE_MARKS:
            return nodes[0]["widths"]
        return [n["r"] for n in nodes]
    styles = [n["style"] for n in nodes]
    if channel == "stroke_width":
        return [s.get("stroke_width", s.get("width")) for s in styles]
    if channel == "opacity":
        return [_INTERACTIVE_OPACITY_READ[mark](s) for s in styles]
    return [s.get(channel, 1.0) for s in styles]


def _unbound_default_static(mark: str, channel: str, data: pl.DataFrame) -> float:
    """The static-SVG counterpart to ``_unbound_default``."""
    chart = _off_side_unbound_chart(mark, channel, data)
    svg = _quiet(chart.to_svg)
    return _static_mark_values(svg, channel, mark)[0]


def _tapered_line_widths(d: str, n: int) -> list[float]:
    """Each of *n* vertices' stroke width, decoded from a tapered line's
    outline path geometry.

    The first ``n - 1`` ``M...Z`` subpaths are the per-segment quads
    between consecutive vertices: 4 points in order (this vertex's near
    offset, the next vertex's near offset, the next vertex's far
    offset, this vertex's far offset) -- the segment's own two ends,
    each as a pair of points straddling the centerline perpendicular to
    the segment direction. A vertex's width is the distance between its
    own pair of offset points: its near/far pair for the first vertex
    of the first segment, or the next-vertex pair of each segment in
    turn for every vertex after it.
    """
    segments = re.findall(r"M[^Z]*Z", d)[: n - 1]
    points = [
        [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
        for nums in (list(map(float, re.findall(r"-?[\d.]+", seg))) for seg in segments)
    ]

    def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    return [dist(points[0][0], points[0][3])] + [dist(seg[1], seg[2]) for seg in points]


@pytest.mark.parametrize("mark,channel", _OFF_SIDE_MARK_CHANNEL_PAIRS)
def test_a_log_off_side_row_is_absent_on_every_mark_that_draws_the_channel(mark, channel):
    """An off-side Log row is absent on every (mark, channel) pair this
    module's mark-coverage matrix covers, not only on a point: a drop
    that preserves row identity holds for every mark and every numeric
    channel, statically and interactively, with the surviving rows'
    values and the dropped and null rows' ``data_indices`` entries all
    correct.

    A single-node mark (line, area) draws one node for the whole series,
    representing only its FIRST surviving row's value for every channel
    but ``size`` (which alone tapers per vertex) -- in this fixture,
    that first survivor is the null row, so only the unbound-default
    check applies there; the "every other row maps" claim is only
    representable, and only checked, where there is more than one node
    (or, for ``size`` on a line, more than one tapered width) to read.

    Rejects a row silently drawn at the default (not dropped), a
    survivor renumbered by its position in the filtered batch rather
    than its source row, and a null row read as off-side instead of
    drawn at the unbound default.
    """
    data = _off_side_data()
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    scale = fm.LogScale(range=list(range_))
    value = _NUMERIC_AUX_CLASSES[channel]("z", scale=scale)
    chart = _off_side_chart(mark, channel, value, data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    assert [str(w.message) for w in caught] == [_log_left_out_warning(channel, "z", 1)]

    is_segment_arrow = mark == "segment" and channel in _SEGMENT_ARROW_CHANNELS
    is_single_node = mark in _SINGLE_NODE_MARKS
    # A single-node mark's static form only tapers per vertex for
    # `size` (an outline <path>); every other channel on it (and `size`
    # itself on every other mark) stays a literal/baked scalar this
    # test's generic static reader (_static_mark_values) covers. Size
    # on a line gets its own dedicated static check below instead.
    size_on_line = channel == "size" and mark in _SINGLE_NODE_MARKS

    batches = [b for p in _quiet(lambda: _scene(chart))["panels"] for b in p["marks"]]
    if is_single_node:
        assert sum(len(b["nodes"]) for b in batches) == 1
        assert [idx for b in batches for idx in b["data_indices"]] == [1]
    elif is_segment_arrow:
        # Each surviving row draws two nodes (shaft, arrowhead), both
        # naming that row's own source index.
        assert sum(len(b["nodes"]) for b in batches) == 14
        assert [idx for b in batches for idx in b["data_indices"]] == [
            i for i in range(1, 8) for _ in range(2)
        ]
    else:
        assert sum(len(b["nodes"]) for b in batches) == 7
        assert [idx for b in batches for idx in b["data_indices"]] == list(range(1, 8))

    # Static: the same row counts, read off the SVG directly.
    layer = mark_layer(svg)
    if size_on_line:
        # A tapered line's outline is one closed <path>, not a plain
        # <polyline>: each of the N-1 segments between consecutive
        # points is a 4-vertex (M + 3L) quad, and each of the N-2
        # interior points gets a 3-vertex (M + 2L) join triangle, so a
        # path over N points has M count 2N-3 and L count 5N-7 (derived
        # by rendering a known N and counting; confirmed for N=8: M=13,
        # L=33). For the 7 surviving rows here, that is M=11, L=28.
        elements = svg_elements(layer, "path", lambda a: True)
        assert len(elements) == 1
        assert elements[0]["d"].count("M") == 2 * 7 - 3
        assert elements[0]["d"].count("L") == 5 * 7 - 7
    elif is_single_node:
        tag = _MARK_SVG_TAG[mark]
        elements = svg_elements(layer, tag, lambda a: True)
        assert len(elements) == 1
        if tag == "polyline":
            assert len(elements[0]["points"].split()) == 7
        else:
            # An area's outline: M + (N-1) L's for the data (top)
            # edge, plus 2 more L's closing the fill to the baseline.
            assert 1 + elements[0]["d"].count("L") - 2 == 7
    elif is_segment_arrow:
        assert len(svg_elements(layer, "path", lambda a: True)) == 7
    else:
        tag = _MARK_SVG_TAG[mark]
        keep = (
            (lambda a: a.get("stroke") not in THEME_CHROME_STROKES)
            if tag == "line"
            else (lambda a: True)
        )
        assert len(svg_elements(layer, tag, keep)) == 7

    # The null row (source index 1, the first survivor) draws at the
    # same value the chart draws with *channel* unbound.
    default = _unbound_default(mark, channel, data)
    byte_quantized = channel == "opacity" and mark in _BYTE_QUANTIZED_OPACITY_MARKS
    interactive_tol = 0.005 if byte_quantized else 1e-6

    interactive_values = _off_side_interactive_values(chart, mark, channel)
    assert interactive_values[0] == pytest.approx(default, abs=interactive_tol)

    # Every other (on-side) row maps through the Log transform -- only
    # representable where there is more than one value to read. size
    # and stroke_width nice a data-derived Log domain to the enclosing
    # power of the base (niced_log_domain); the opacity family does not.
    if not (is_single_node and channel != "size"):
        on_side_zs = [10.0, 30.0, 25.0, 35.0, 40.0, 45.0]
        raw_domain = (min(on_side_zs), max(on_side_zs))
        domain = (
            niced_log_domain(*raw_domain, 10.0) if channel in _AUX_NICES_DATA_DOMAIN else raw_domain
        )
        expected = [default] + [
            _expected_off_side_value(mark, channel, log_map(z, domain, range_, base=10.0))
            for z in on_side_zs
        ]
        assert interactive_values == pytest.approx(expected, abs=interactive_tol)

    if size_on_line:
        # The null row's default and every on-side row's mapped value,
        # decoded directly from the tapered outline's own geometry
        # (each per-segment quad's two offset points at a vertex are
        # `width` apart) rather than reconstructed through a second
        # chart: a twin built from a 7-row subset of *data* shifts the
        # path's very first join (its start cap differs depending on
        # whether a dropped neighbour row ever existed positionally),
        # so comparing two charts' paths byte-for-byte is the wrong
        # tool here -- measuring the ONE path this chart drew is not.
        path_values = _tapered_line_widths(elements[0]["d"], n=7)
        assert path_values == pytest.approx(interactive_values, abs=1e-3)
    else:
        static_tol = 0.005 if byte_quantized else 1e-3
        static_default = _unbound_default_static(mark, channel, data)
        static_values = _static_mark_values(svg, channel, mark)
        assert static_values[0] == pytest.approx(static_default, abs=static_tol)
        if not (is_single_node and channel != "size"):
            static_expected = [static_default] + [
                _expected_off_side_value(mark, channel, log_map(z, domain, range_, base=10.0))
                for z in on_side_zs
            ]
            assert static_values == pytest.approx(static_expected, abs=static_tol)


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
@pytest.mark.parametrize("mark,channel", _MARK_CHANNEL_PAIRS)
def test_a_transform_scale_maps_every_mark_that_draws_the_channel(mark, channel, kind):
    """stroke_width and the opacity family (including plain ``opacity``)
    map through Log/Pow/Sqrt on every mark that draws them, not only a
    point, in both the interactive scene and the static SVG.

    Rejects a gate keyed on mark kind: the same per-kind math point cells
    already pin must hold for a line's width, a bar's fill, a rule's
    stroke and the rest; rejects a static/interactive divergence, since
    each representation must carry the same mapped value.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE[channel])
    scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN), range=list(range_))
    chart = _mark_chart(mark, channel, _NUMERIC_AUX_CLASSES[channel]("z", scale=scale))
    mapper = _TRANSFORM_MAP[kind]
    zs = df["z"].to_list()
    expected = (
        [mapper(zs[0], _TRANSFORM_DOMAIN, range_)]
        if mark in _SINGLE_NODE_MARKS
        else [mapper(z, _TRANSFORM_DOMAIN, range_) for z in zs]
    )
    # Plain "opacity" on point/line/area/rule bakes into a paint's 8-bit
    # alpha byte, round-tripping to the nearest 1/255 rather than the
    # exact fraction fill_opacity/stroke_opacity/stroke_width (and bar's/
    # tick's/segment's literal opacity attribute) carry. Every static
    # attribute, byte-quantized or not, also passes through the SVG
    # writer's 3-decimal-place formatting (render/mod.rs's
    # FLOAT_PRECISION, used by svg.rs's fmt_f/fmt_svg).
    byte_quantized = channel == "opacity" and mark in _BYTE_QUANTIZED_OPACITY_MARKS
    interactive_tol = 0.005 if byte_quantized else 1e-6
    static_tol = 0.005 if byte_quantized else 1e-3

    values = _quiet(lambda: _style_values(chart, mark, channel))
    assert values == pytest.approx(expected, abs=interactive_tol)

    static_values = _static_style_values(_quiet(chart.to_svg), mark, channel)
    assert static_values == pytest.approx(expected, abs=static_tol)


def _first_path_d(svg: str) -> str:
    match = re.search(r'<path[^>]*\bd="([^"]+)"', mark_layer(svg))
    assert match is not None, "no <path> in the plot's clip group"
    return match.group(1)


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
def test_a_transform_scale_maps_size_on_a_line_as_a_tapered_stroke_width(kind):
    """``size`` maps through Log/Pow/Sqrt on a ``line``, not only a point:
    a per-vertex stroke width, each the diameter of the circle the same
    mapped area would draw on a point (``_drawn_value``'s radius,
    doubled -- a line's thickness spans both sides of its center point).

    A varying line width is one tapered outline ``<path>`` whose
    thickness is baked into its boundary geometry rather than exposed as
    a per-vertex attribute -- decoding that geometry directly would mean
    re-deriving taper math this codebase already verifies through golden
    and pixel-capture tests (``test_golden_tapered_line.py``,
    ``test_tapered_line_pixels.py``). Instead, an identity-scale twin
    chart -- the SAME per-vertex widths, inverted back to the raw mapped
    area ``LinearScale(domain=range_, range=range_)`` leaves untouched,
    then re-mapped through the renderer's own area-to-diameter formula
    -- must draw the byte-identical outline path: if the renderer's
    taper geometry is a pure function of the per-vertex widths (and
    nothing else), feeding it the same widths through any scale must
    produce the same path. ``legend=None`` on both charts keeps a
    differently-formatted legend label from shifting the plot area's
    width by a few pixels, which would move every coordinate uniformly
    and mask the comparison this test exists to make.
    """
    range_ = tuple(float(v) for v in _ALT_RANGE["size"])
    scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN), range=list(range_))
    chart = mk("line", x="v", y="y", size=fm.Size("z", scale=scale, legend=None))
    mapper = _TRANSFORM_MAP[kind]
    zs = df["z"].to_list()
    expected = [2.0 * _drawn_value("size", mapper(z, _TRANSFORM_DOMAIN, range_)) for z in zs]

    nodes = [n for p in _scene(chart)["panels"] for b in p["marks"] for n in b["nodes"]]
    ((widths,),) = [(n["widths"],) for n in nodes]
    assert widths == pytest.approx(expected, abs=1e-6)

    mapped_area = [math.pi * (w / 2.0) ** 2 for w in widths]
    twin_data = df.with_columns(pl.Series("z", mapped_area))
    identity_scale = fm.LinearScale(domain=list(range_), range=list(range_))
    twin = mk(
        "line", data=twin_data, x="v", y="y", size=fm.Size("z", scale=identity_scale, legend=None)
    )
    assert _first_path_d(_quiet(chart.to_svg)) == _first_path_d(_quiet(twin.to_svg))


# ---------------------------------------------------------------------------
# An auxiliary channel's transform fraction, and its Log side rule, match
# a real positional render of the same column under the same scale --
# not merely a claim that they do.
# ---------------------------------------------------------------------------


def _x_pixel_fractions(chart: Any) -> list[float]:
    """Each drawn circle's x position as a fraction of the plot's own pixel
    width, in document order. The scale must carry no explicit ``range=``:
    on a positional channel, ``range=`` sets literal pixel coordinates, not
    a fraction of the plot -- the auto-filled full-width range is what
    makes this fraction comparable to an auxiliary channel's own fraction
    of *its* range.
    """
    svg = _quiet(chart.to_svg)
    root = _svg_root(svg)
    (x0, _y0, x1, _y1) = next(iter(_clip_rects(root).values()))
    return [(float(c.get("cx")) - x0) / (x1 - x0) for c in root.iter() if _tag(c) == "circle"]


@pytest.mark.parametrize("kind", sorted(_TRANSFORM_SCALES))
def test_an_aux_fraction_equals_the_positional_pixel_fraction_for_in_domain_rows(kind):
    """A column's fraction of its range under Log/Pow/Sqrt is the same
    number whether it is drawn on ``x`` (a pixel fraction of the plot) or
    on an auxiliary channel (a fraction of its own range) -- the
    transform parity the aux docstrings claim, shown against a real
    render of BOTH sides, not merely the oracle on one side. Mapping onto
    ``range=[0, 1]`` makes the oracle's own output literally that
    fraction.
    """
    scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN))
    zs = df["z"].to_list()

    x_chart = mk("point", x=fm.X("z", scale=scale), y="y")
    x_fractions = _x_pixel_fractions(x_chart)

    mapper = _TRANSFORM_MAP[kind]
    aux_fractions = [mapper(z, _TRANSFORM_DOMAIN, (0.0, 1.0)) for z in zs]
    assert x_fractions == pytest.approx(aux_fractions, abs=1e-3)

    aux_scale = _TRANSFORM_SCALES[kind](domain=list(_TRANSFORM_DOMAIN), range=[0.0, 1.0])
    aux_chart = mk("point", x="y", y="y", opacity=fm.Opacity("z", scale=aux_scale))
    aux_values = _quiet(lambda: _mark_values(aux_chart, "opacity"))
    assert aux_values == pytest.approx(x_fractions, abs=1e-3)

    static_values = _static_style_values(_quiet(aux_chart.to_svg), "point", "opacity")
    assert static_values == pytest.approx(x_fractions, abs=0.005)


def test_log_drops_the_same_rows_on_x_and_the_aux_channel_with_the_same_warning():
    """A mixed-sign column under the identical, undomained ``LogScale()``
    leaves out the same row on ``x`` and on ``size``, and both warn with
    the same row count.
    """
    data = pl.DataFrame({"v": [-5.0, 1.0, 10.0, 100.0], "y": [1.0, 2.0, 3.0, 4.0]})
    scale = fm.LogScale()

    x_chart = _log_channel_chart("x", data, scale)
    with warnings.catch_warnings(record=True) as caught_x:
        warnings.simplefilter("always")
        x_svg = x_chart.to_svg()
    assert [str(w.message) for w in caught_x] == [_log_left_out_warning("x", "v", 1)]
    assert tag_counts(x_svg).get("circle", 0) == 3

    size_chart = _log_channel_chart("size", data, scale)
    with warnings.catch_warnings(record=True) as caught_size:
        warnings.simplefilter("always")
        size_svg = size_chart.to_svg()
    assert [str(w.message) for w in caught_size] == [_log_left_out_warning("size", "v", 1)]
    assert tag_counts(size_svg).get("circle", 0) == 3
    size_values = _quiet(lambda: _mark_values(size_chart, "size"))
    assert len(size_values) == 3


@pytest.mark.parametrize("kind", ["pow", "sqrt"])
def test_pow_and_sqrt_map_a_negative_value_with_the_same_fraction_on_x_and_aux(kind):
    """Pow and Sqrt's sign-preserving power maps a negative value to the
    same fraction of its range on ``x`` and on an auxiliary channel --
    neither drops nor refuses it, unlike Log -- shown on a real aux
    render, and against a LinearScale twin that must NOT match (the
    sign-preserving power is actually in effect, not a linear fallback).
    """
    domain = (-50.0, 50.0)
    vs = [-40.0, -10.0, 0.0, 10.0, 40.0]
    data = pl.DataFrame({"v": vs, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scale = _TRANSFORM_SCALES[kind](domain=list(domain))

    x_chart = fm.Chart(data).mark_point().encode(x=fm.X("v", scale=scale), y="y")
    x_fractions = _x_pixel_fractions(x_chart)

    mapper = _TRANSFORM_MAP[kind]
    aux_fractions = [mapper(v, domain, (0.0, 1.0)) for v in vs]
    assert x_fractions == pytest.approx(aux_fractions, abs=1e-3)

    aux_scale = _TRANSFORM_SCALES[kind](domain=list(domain), range=[0.0, 1.0])
    aux_chart = (
        fm.Chart(data).mark_point().encode(x="y", y="y", opacity=fm.Opacity("v", scale=aux_scale))
    )
    aux_values = _quiet(lambda: _mark_values(aux_chart, "opacity"))
    assert aux_values == pytest.approx(x_fractions, abs=1e-3)

    linear_twin = (
        fm.Chart(data)
        .mark_point()
        .encode(
            x="y",
            y="y",
            opacity=fm.Opacity("v", scale=fm.LinearScale(domain=list(domain), range=[0.0, 1.0])),
        )
    )
    linear_values = _quiet(lambda: _mark_values(linear_twin, "opacity"))
    assert linear_values != pytest.approx(aux_values, abs=1e-3)


_AGGREGATE_LEGEND_TITLE: dict[str, Case] = {
    "same-frame/+": Case(
        lambda: P(x="g") + agg("point", color=mean(fm.Color)), chart_level=frozenset()
    ),
    "same-frame/+rev": Case(lambda: agg("point", color=mean(fm.Color)) + P(x="g")),
    "diff-frame/+": Case(lambda: P(x="g") + agg("point", df2, color=mean(fm.Color))),
    "diff-frame/+rev": Case(lambda: agg("point", df2, color=mean(fm.Color)) + P(x="g")),
    "LCm": Case(
        lambda: fm.LayerChart(P(x="g"), agg("point", color=mean(fm.Color)))._build_merged()
    ),
    "LCi": Case(
        lambda: fm.LayerChart(
            P(x="g"), agg("point", color=mean(fm.Color)), resolve={"y": "independent"}
        )
    ),
    "L": Case(
        lambda: P(x="g").layer(layer("point", y=mean(fm.Y, "y"), color=mean(fm.Color))),
        chart_level=None,
    ),
    "F": Case(lambda: agg("point", color=mean(fm.Color))),
}


@pytest.mark.parametrize("case", sorted(_AGGREGATE_LEGEND_TITLE))
def test_an_aggregated_legend_is_titled_by_its_output_column(case):
    """The color legend is titled ``mean_z`` on every door, same frame or not.

    Rejects the source-field title (``z``) a different-frame promotion copy drew.
    """
    case = _AGGREGATE_LEGEND_TITLE[case]
    check_rendering(case)
    assert list(legend_set(_quiet(_quiet(case.build).to_svg))) == ["mean_z"]


_LAYER_ENCODING_STRING: dict[str, Case] = {
    "string": Case(
        lambda: P().layer(fm.Layer(mark="point", encoding={"y": "r", "color": "c"})),
        (P, lambda: P(y="r", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "object": Case(
        lambda: P().layer(fm.Layer(mark="point", encoding={"y": "r", "color": fm.Color("c")})),
        (P, lambda: P(y="r", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_LAYER_ENCODING_STRING))
def test_a_string_in_layer_encoding_draws_with_a_legend(case):
    """``encoding={"color": "c"}`` draws colored with a legend, as ``Color("c")`` does.

    Rejects a scale built only from the chart-level encoding: the string
    spelling drew no color.
    """
    check_rendering(_LAYER_ENCODING_STRING[case])


# ---------------------------------------------------------------------------
# One channel vocabulary on every door
# ---------------------------------------------------------------------------
#
# Every door reads a binding the way ``encode()`` does and refuses the same
# malformed binding with the same text: a channel object, ``encode()``
# keyword arguments, shorthand, ``Layer(encoding=)`` (attached and with its
# own data) and a figure's ``encode=`` and ``layers=``.


def _channel_classes() -> dict[str, type]:
    from ferrum.encoding import _channel_class_map

    return dict(_channel_class_map())


# Channels whose bin= is honored: the positional bin (x, y) and the binned
# color scale (color and its aliases).
_BIN_HONORED = frozenset({"x", "y", "color", "fill", "stroke"})
_BIN_REFUSED = sorted(set(_channel_classes()) - _BIN_HONORED)

# The channels that refuse aggregate=: a grouping, link, facet or error
# channel draws no aggregated value.
_AGGREGATE_REFUSED = (
    "detail",
    "shape",
    "stroke_dash",
    "key",
    "href",
    "description",
    "url",
    "facet",
    "facet_row",
    "facet_col",
    "x_error",
    "y_error",
)


def _bin_refusal(channel: str, value: Any) -> str:
    return (
        f"{channel}(bin={value!r}) is not supported: {channel} draws no binned value. "
        "Bin the field with .transform(fm.Bin(...)) and bind its output column, or drop bin=."
    )


def _aggregate_refusal(channel: str, value: Any) -> str:
    return (
        f"{channel}(aggregate={value!r}) is not supported: {channel} draws no aggregated "
        "value. Compute the aggregate with .transform(fm.transform_aggregate(...)) and "
        "bind its output column, or drop aggregate=."
    )


def _vocabulary_doors(name: str, value: Callable[[], Any]) -> dict[str, Callable[[], Any]]:
    """Every door one binding of channel *name* is written on; *value* builds it inside the door."""
    return {
        "object": value,
        "encode": lambda: P().encode(**{name: value()}),
        "L": lambda: P().layer(fm.Layer(mark="point", encoding={name: value()})),
        "LD": lambda: P().layer(fm.Layer(data=df2, mark="point", encoding={name: value()})),
        "figure": lambda: fm.relplot(df, x="v", y="y", encode={name: value()}),
        "figure-layers": lambda: fm.relplot(
            df, x="v", y="y", layers=[fm.Layer(mark="point", encoding={name: value()})]
        ),
    }


_BIN_SPELLINGS: dict[str, Callable[[], Any]] = {
    "True": lambda: True,
    "Bin": lambda: fm.Bin("z", bin_count=3),
}

_BIN_REFUSAL_CASES = {
    f"{'value-channel' if name in _VALUE_CHANNELS else 'grouping-channel'}/{name}/{spelling}/{door}": (
        name,
        spelling,
        door,
    )
    for name in _BIN_REFUSED
    for spelling in _BIN_SPELLINGS
    for door in _vocabulary_doors(name, lambda: None)
}


@pytest.mark.parametrize("case", sorted(_BIN_REFUSAL_CASES))
def test_a_bin_on_a_channel_drawing_no_binned_value_is_refused_on_every_door(case):
    """``bin=`` on any channel but x, y and the color family refuses at construction.

    Rejects the warn-and-keep path, whose bin still lowered into a step that
    replaced the chart's rows: ``P(size=fm.Size("z", bin=True))`` drew 39
    circles from 8 rows.
    """
    name, spelling, door = _BIN_REFUSAL_CASES[case]
    cls = _channel_classes()[name]
    bin_value = _BIN_SPELLINGS[spelling]()
    build = _vocabulary_doors(name, lambda: cls("z", bin=bin_value))[door]
    with pytest.raises(ValueError) as refused:
        build()
    assert str(refused.value) == _bin_refusal(cls.__name__, bin_value)


_AGGREGATE_REFUSAL_CASES = {
    f"no_value_aggregate/{name}/{spelling}/{door}": (name, spelling, door)
    for name in _AGGREGATE_REFUSED
    for spelling in ("object", "shorthand")
    for door in _vocabulary_doors(name, lambda: None)
    if not (spelling == "shorthand" and door == "object")
}


@pytest.mark.parametrize("case", sorted(_AGGREGATE_REFUSAL_CASES))
def test_an_aggregate_on_a_channel_drawing_no_aggregated_value_is_refused_on_every_door(case):
    """``aggregate=`` on a grouping, link, facet or error channel refuses at construction.

    The shorthand cells reject reading ``Layer(encoding={"detail": "mean(z)"})``
    as a literal column named ``mean(z)``: a layer string means what
    ``encode()`` makes of it.
    """
    name, spelling, door = _AGGREGATE_REFUSAL_CASES[case]
    cls = _channel_classes()[name]
    value = (lambda: cls("z", aggregate="mean")) if spelling == "object" else (lambda: "mean(z)")
    with pytest.raises(ValueError) as refused:
        _vocabulary_doors(name, value)[door]()
    assert str(refused.value) == _aggregate_refusal(cls.__name__, "mean")


_UNSET_STATISTICS = {
    "aggregate=None": {"aggregate": None},
    "bin=None": {"bin": None},
    "bin=False": {"bin": False},
}


@pytest.mark.parametrize("unset", sorted(_UNSET_STATISTICS))
@pytest.mark.parametrize("name", sorted(_channel_classes()))
def test_an_unset_aggregate_or_bin_neither_refuses_nor_warns_on_any_channel(name, unset):
    """None is unset: ``aggregate=None``, ``bin=None`` and ``bin=False`` ask for nothing.

    Rejects the honored-kwarg warning ``Detail(aggregate=None)`` raised, and
    a refusal keyed on the kwarg's presence rather than its value.
    """
    from ferrum._warn import reset_warnings

    reset_warnings()
    cls = _channel_classes()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        channel = cls("z", **_UNSET_STATISTICS[unset])
    assert channel.to_implicit_transforms() == []


@pytest.mark.parametrize("door", ["F", "L"])
def test_an_unset_bin_or_aggregate_draws_what_the_bare_binding_draws(door):
    """An unset statistic changes nothing: the SVG equals the binding written without it."""
    unset = fm.Size("z", bin=False, aggregate=None)
    if door == "F":
        built, bare = P(size=unset), P(size="z")
    else:
        built = P().layer(layer("point", y="r", size=unset))
        bare = P().layer(layer("point", y="r", size="z"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert built.to_svg() == bare.to_svg()


_LAYER_ENCODING_SPELLING: dict[str, Case] = {
    "sum/L": Case(
        lambda: P().layer(fm.Layer(mark="point", encoding={"y": "sum(y)"})),
        (P, lambda: P(y="sum(y)")),
        chart_level=None,
    ),
    "sum/LD": Case(
        lambda: P().layer(fm.Layer(data=df2, mark="point", encoding={"y": "sum(y)"})),
        (P, lambda: P(df2, y="sum(y)")),
        chart_level=None,
    ),
    "count/L": Case(
        lambda: P().layer(fm.Layer(mark="point", encoding={"y": "count()"})),
        (P, lambda: P(y="count()")),
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_LAYER_ENCODING_SPELLING))
def test_a_layer_encoding_shorthand_aggregates_that_layer(case):
    """``Layer(encoding={"y": "sum(y)"})`` aggregates the layer, as ``encode(y="sum(y)")`` does.

    Rejects the literal reading: ``"sum(y)"`` was refused as an unknown
    column of that name.
    """
    check_rendering(_LAYER_ENCODING_SPELLING[case])


@pytest.mark.parametrize("type_", ["Q", "N", "O", "T"])
def test_a_layer_encoding_type_suffix_sets_the_type(type_):
    """``"r:Q"`` binds ``r`` with that type, on the layer and on its wire encoding.

    Rejects the literal reading, which bound a column named ``r:Q``.
    """
    built = fm.Layer(mark="point", encoding={"y": f"r:{type_}"})
    assert built.encoding["y"] == fm.Y("r", type=type_)
    wire_y = _wire(P().layer(built))["layers"][1]["encoding"]["y"]
    assert wire_y["field"] == "r"
    assert (
        wire_y["type"]
        == {"Q": "quantitative", "N": "nominal", "O": "ordinal", "T": "temporal"}[type_]
    )


@pytest.mark.parametrize("name", sorted(_channel_classes()))
def test_a_layer_string_equals_the_channel_object_spelling(name):
    """A string in ``Layer(encoding=)`` is the channel object ``encode()`` builds from it."""
    cls = _channel_classes()[name]
    assert fm.Layer(encoding={name: "z"}).encoding[name] == cls("z")
    assert fm.Layer(encoding={name: "z:O"}).encoding[name] == cls("z", type="O")
    assert (
        fm.Layer(encoding={name: "z"}).encoding[name] == P().encode(**{name: "z"})._encoding[name]
    )


# Channel -> (field, mark) a layer draws it with.
_DRAWN_STRING_CHANNELS = {
    **_VALUE_CHANNELS,
    "shape": ("c", "point"),
    "stroke_dash": ("c", "line"),
    "detail": ("c", "line"),
}


@pytest.mark.parametrize("name", sorted(_DRAWN_STRING_CHANNELS))
def test_a_layer_string_draws_what_the_channel_object_draws(name):
    """The string and the object spelling of one layer binding render the same bytes."""
    field, mark = _DRAWN_STRING_CHANNELS[name]
    cls = _channel_classes()[name]
    as_string = P().layer(layer(mark, y="r", **{name: field}))
    as_object = P().layer(layer(mark, y="r", **{name: cls(field)}))
    assert _quiet(as_string.to_svg) == _quiet(as_object.to_svg)


def test_an_unknown_op_in_a_layer_string_is_refused_as_encode_refuses_it():
    """``"foo(y)"`` names an unknown aggregate on every door, not an unknown column."""
    with pytest.raises(ValueError) as from_encode:
        P(y="foo(y)").to_svg()
    with pytest.raises(ValueError) as from_layer:
        P().layer(fm.Layer(mark="point", encoding={"y": "foo(y)"})).to_svg()
    assert str(from_layer.value) == str(from_encode.value)
    assert str(from_layer.value).startswith("AggregateOp: unknown fn 'foo'")


# Every value kind an encoding key can carry, built inside the test.
_ANY_VALUE: dict[str, Callable[[], Any]] = {
    "str": lambda: "y",
    "object": lambda: fm.Y("y"),
    "none": lambda: None,
    "conditional": lambda: _PICK_C.when(fm.value("#d62728")).otherwise(fm.value("#cccccc")),
    "repeat": lambda: fm.Repeat.column,
    "int": lambda: 3,
}


@pytest.mark.parametrize("value", sorted(_ANY_VALUE))
@pytest.mark.parametrize("name", ["bogus", "order", "colour"])
def test_an_unknown_channel_in_a_layer_encoding_is_refused_as_encode_refuses_it(name, value):
    """A name ``encode()`` does not take is refused at ``Layer`` construction, whatever its value.

    Rejects carrying it to the wire: ``bogus`` was warned and dropped, and
    the wire-only ``order`` reached the render as a channel ``encode()``
    never offers. The ``None`` cells reject skipping the name check for an
    unset value, which let ``encoding={"colour": None}`` render as if
    nothing were wrong; the conditional cells reject answering in the
    conditional vocabulary's voice instead of ``encode()``'s.
    """
    with pytest.raises(ValueError) as from_encode:
        P().encode(**{name: _ANY_VALUE[value]()})
    with pytest.raises(ValueError) as from_layer:
        fm.Layer(mark="point", encoding={name: _ANY_VALUE[value]()})
    assert str(from_layer.value) == str(from_encode.value) == f"unknown encoding channel: {name!r}"


@pytest.mark.parametrize("name", sorted(_channel_classes()))
def test_a_none_binding_binds_nothing_on_encode_and_layer_encoding(name):
    """None is unset on both doors: the SVG equals the chart written without the kwarg.

    Rejects ``encode(color=None)``'s ``TypeError``, which made ``encode()``
    refuse what ``Layer(encoding=)`` reads as unset, and a ``None`` that
    clears an earlier binding.
    """
    assert _quiet(P().encode(**{name: None}).to_svg) == _quiet(P().to_svg)
    assert _quiet(P(color="c").encode(**{name: None}).to_svg) == _quiet(P(color="c").to_svg)
    own = {"y": "r"} if name != "y" else {"x": "r"}
    with_none = P().layer(fm.Layer(mark="point", encoding={**own, name: None}))
    without = P().layer(fm.Layer(mark="point", encoding=own))
    assert _quiet(with_none.to_svg) == _quiet(without.to_svg)


@pytest.mark.parametrize("value", [3, 2.5, ["y"]], ids=["int", "float", "list"])
def test_a_layer_value_that_is_no_binding_is_refused_as_encode_refuses_it(value):
    """A value that is neither a string nor a channel object is a ``TypeError`` on both doors.

    Rejects passing it to the wire, where ``encoding={"y": 3}`` rendered.
    """
    kind = type(value).__name__
    with pytest.raises(TypeError) as from_encode:
        P().encode(y=value)
    with pytest.raises(TypeError) as from_layer:
        fm.Layer(mark="point", encoding={"y": value})
    assert str(from_encode.value) == (
        f"encode(y=...) expects str, Y instance, or Repeat placeholder; got {kind}"
    )
    assert str(from_layer.value) == (
        f"Layer(encoding={{'y': ...}}) expects str, Y instance, or Repeat placeholder; got {kind}"
    )


@pytest.mark.parametrize("name", sorted(_channel_classes()))
def test_a_statistic_rebinding_stamps_only_kwargs_its_channel_honors(name):
    """A copy re-pointed at a statistic's output keeps a ``title`` only where the channel honors it.

    Rejects stamping the source field as ``title=`` on every channel, which
    a tooltip or text label refuses at construction.
    """
    from ferrum._encode_lowering import _rebind_channel

    cls = _channel_classes()[name]
    honored = cls._honored_kwargs
    statistic = {"aggregate": "mean"} if "aggregate" in honored else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rebound = _rebind_channel(cls("z", **statistic), "mean_z")
    assert set(rebound._kwargs) <= honored
    assert ("title" in rebound._kwargs) == ("title" in honored)


# ---------------------------------------------------------------------------
# Grouping channels
# ---------------------------------------------------------------------------

_GROUPING_BINDING: dict[str, Case] = {
    **_cases("detail", plus_doors(lambda: line(detail="c"), lambda: line(y="r"))),
    **_cases("stroke_dash", plus_doors(lambda: line(stroke_dash="c"), lambda: line(y="r"))),
    **_cases("shape", plus_doors(lambda: P(shape="c"), lambda: P(y="r"))),
}


@pytest.mark.parametrize("case", sorted(_GROUPING_BINDING))
def test_a_grouping_binding_never_splits_a_sibling(case):
    """``line(detail=c) + line`` = ``line + line(detail=c)`` = 3 polylines.

    Rejects the sibling-encoding leak: the left's detail would split the
    right line into 2.
    """
    check_rendering(_GROUPING_BINDING[case])


def test_three_polylines_in_both_orders():
    for chart in (line(detail="c") + line(y="r"), line(y="r") + line(detail="c")):
        assert tag_counts(chart.to_svg())["polyline"] == 3


_INHERITED_GROUPING: dict[str, Case] = {
    # The shape scale's second glyph is a square `<rect>`.
    "F": Case(lambda: agg("point", shape="c"), marks={"circle": 2, "rect": 2}),
    "L-inherits-the-hosts-shape": Case(
        lambda: agg("point", shape="c").layer(layer("tick")),
        (lambda: agg("point", shape="c"), lambda: agg("tick", shape="c")),
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_INHERITED_GROUPING))
def test_a_raw_grouping_channel_groups_an_aggregating_layer(case):
    """Flat 4 groups; an attached layer inheriting the host's ``shape`` groups by it too."""
    check_rendering(_INHERITED_GROUPING[case])


# Channel -> (field, the raw binder's mark), as the probes draw them.
_GROUPING: dict[str, tuple[str, str]] = {
    "detail": ("c", "line"),
    "shape": ("c", "point"),
    "stroke_dash": ("c", "line"),
    "key": ("k", "point"),
    "href": ("k", "point"),
    "text": ("c", "text"),
    "tooltip": ("c", "point"),
}


def _aggregating_sibling_grain_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for channel, (value, mark) in _GROUPING.items():
        cases |= _cases(
            channel,
            plus_doors(
                lambda channel=channel, value=value, mark=mark: mk(
                    mark, **{"x": "g", "y": "y", channel: value}
                ),
                lambda: agg("tick"),
            ),
        )
    return cases


_AGGREGATING_SIBLING_GRAIN = _aggregating_sibling_grain_cases()


@pytest.mark.parametrize("case", sorted(_AGGREGATING_SIBLING_GRAIN))
def test_an_aggregating_sibling_keeps_its_grain_beside_a_raw_grouping_binding(case):
    """The tick keeps 2 marks beside a raw detail/shape/stroke_dash/key/href/text/tooltip.

    Rejects the sibling-encoding leak: the tick would group by the sibling's
    raw field.
    """
    check_rendering(_AGGREGATING_SIBLING_GRAIN[case])


_ATTACHED_GROUPING: dict[str, Case] = {
    "shape": Case(
        lambda: P().layer(fm.Layer(mark="point", encoding={"y": "r", "shape": "c"})),
        (P, lambda: P(y="r", shape="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "stroke_dash": Case(
        lambda: line().layer(fm.Layer(mark="line", encoding={"y": "r", "stroke_dash": "c"})),
        (line, lambda: line(y="r", stroke_dash="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_ATTACHED_GROUPING))
def test_an_attached_layers_own_grouping_channel_draws_with_a_legend(case):
    check_rendering(_ATTACHED_GROUPING[case])


# ---------------------------------------------------------------------------
# Parent rules
# ---------------------------------------------------------------------------

_TITLED = fm.Color("c", title="Group", scale=fm.OrdinalScale(domain=["q", "p"]))

_PARENT: dict[str, Case] = {
    "L-inherits-the-host-whole": Case(
        lambda: P(color=_TITLED).layer(layer("line")),
        (lambda: P(color=_TITLED), lambda: line(color=_TITLED)),
        legends=legend_set(P(color=_TITLED).to_svg()),
        chart_level=None,
    ),
    "L-inherits-the-hosts-aggregate": Case(
        lambda: bar_mean().layer(layer("point")),
        (bar_mean, lambda: agg("point")),
        marks={"rect": 2, "circle": 2},
        chart_level=None,
    ),
    "+e-reaches-every-layer-binding-none": Case(
        lambda: (P() + line(y="r")).encode(color="c"),
        (lambda: P(color="c"), lambda: line(y="r", color="c")),
        legends={"c": ("p", "q")},
        chart_level=frozenset({"color"}),
    ),
    "+e-detail-then-plus-leaves-c-whole": Case(
        lambda: (line() + line(y="r")).encode(detail="c") + line(y="z"),
        (lambda: line(detail="c"), lambda: line(y="r", detail="c"), lambda: line(y="z")),
        marks={"polyline": 5},
    ),
    # A binding written on a right operand that is itself a composition is
    # that operand's own: it reaches the operand's layers, never the left.
    "+e-on-a-right-operand-reaches-its-own-layers": Case(
        lambda: P() + (line() + line(y="r")).encode(detail="c"),
        (P, lambda: line(detail="c"), lambda: line(y="r", detail="c")),
        marks={"circle": 8, "polyline": 4},
    ),
    "LC-member-inherits-nothing": Case(
        lambda: fm.LayerChart((P() + line(y="r")).encode(color="c"), line(y="z"))._build_merged(),
        (lambda: P(color="c"), lambda: line(y="r", color="c"), lambda: line(y="z")),
    ),
    "+e-post-merge-aggregate-reaches-only-unbound-layers": Case(
        lambda: (P(x="g") + fm.Chart(df2).mark_point()).encode(y="mean(y)"),
        (lambda: P(x="g"), lambda: agg("point", df2)),
        marks={"circle": 11},
    ),
}


@pytest.mark.parametrize("case", sorted(_PARENT))
def test_parent_rules(case):
    """A layer inherits its one parent's bindings whole, and nothing from a sibling.

    Rejects the sibling-encoding leak: ``(a + b).encode(detail="g") + c``
    would split ``c``, and a LayerChart member would inherit an earlier
    member's composition color.
    """
    check_rendering(_PARENT[case])


# ---------------------------------------------------------------------------
# Axis fallback: the one cross-sibling flow
# ---------------------------------------------------------------------------


def _line_points() -> fm.Chart:
    """``mark_line(point=True)``: a line and a point layer, each binding no x/y of its own."""
    return fm.Chart(df).mark_line(point=True).encode(x="v", y="r")


_AXIS_FALLBACK: dict[str, Case] = {
    "L": Case(lambda: P().layer(layer("line")), (P, line), chart_level=None),
    "LD": Case(lambda: P().layer(fm.Layer(data=df2, mark="line")), (P, lambda: line(df2))),
    "+s": Case(lambda: P() + line(), (P, line)),
    "+d": Case(lambda: P() + line(df2), (P, lambda: line(df2))),
    "+d-right-binds-none": Case(lambda: P() + fm.Chart(df2).mark_line(), (P, lambda: line(df2))),
    "+d-rev-right-binds-none": Case(
        lambda: line(df2) + fm.Chart(df).mark_point(), (lambda: line(df2), P)
    ),
    # A layered right operand's layers bind no x/y of their own; they take
    # that operand's (its own bindings), not layer 0's.
    "+s-layered-right-operand": Case(
        lambda: P() + _line_points(), (P, _line_points), marks={"circle": 16, "polyline": 1}
    ),
    "+s-layered-left-operand": Case(
        lambda: _line_points() + P(), (_line_points, P), marks={"circle": 16, "polyline": 1}
    ),
    "SY": Case(lambda: P() + SY("r"), marks={"circle": 8, "polyline": 1}, chart_level=None),
    **{
        k: v
        for k, v in _cases("LC", plus_doors(P, lambda: line(df2), lcs=True)).items()
        if k != "LC/+" and k != "LC/+rev"
    },
    "Fc": Case(
        lambda: (P() + line()).facet("c"),
        (lambda: P().facet("c"), lambda: line().facet("c")),
        faceted=True,
    ),
}


@pytest.mark.parametrize("case", sorted(_AXIS_FALLBACK))
def test_a_layer_binding_no_axis_takes_layer_zeros_written_axis(case):
    check_rendering(_AXIS_FALLBACK[case])


@pytest.mark.parametrize("order", ["right", "left"])
def test_a_layered_operands_layers_draw_that_operands_written_axis(order):
    """``mark_line(point=True).encode(y="r")``'s two layers draw ``r`` beside ``P``, either order.

    Rejects the composition's axis reaching them: they bind no y of their
    own, and would draw layer 0's ``y``.
    """
    chart = P() + _line_points() if order == "right" else _line_points() + P()
    wire = _wire(chart)
    layers = wire["layers"][1:] if order == "right" else wire["layers"][:2]
    for wire_layer in layers:
        completed = {**wire["encoding"], **wire_layer["encoding"]}
        assert completed["y"]["field"] == "r", wire_layer


@pytest.mark.parametrize(
    ("build", "titles"),
    [
        (lambda: P() + fm.Chart(df2).mark_line(), ["y", "v"]),
        (lambda: bar_mean() + fm.Chart(df).mark_point(), ["mean_y", "g"]),
    ],
    ids=["+d", "+s-aggregate"],
)
def test_layer_zeros_written_axis_titles_the_axes(build, titles):
    assert axis_titles(build().to_svg()) == titles


def test_a_drawn_fallback_column_the_layers_rows_lack_is_refused():
    """``+d`` reversed: the points take layer 0's ``x="w"``, which df lacks, so the chart refuses.

    A point draws x, so it takes the fallback and reads ``w`` by name in its
    own rows. Rejects the padded column: the points drew nothing, silently.
    """
    with pytest.raises(ValueError) as refused:
        (line(df2, x="w") + fm.Chart(df).mark_point()).to_svg()
    assert str(refused.value).startswith(
        "layer 1: encoding 'x' references unknown column 'w' (not present in that layer's data)"
    )


# ---------------------------------------------------------------------------
# Regression net: consistent cells at their pinned counts
# ---------------------------------------------------------------------------


def _hist() -> fm.Chart:
    return fm.Chart(df).mark_histogram().encode(x="v")


def _binned(data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    return mk("bar", data, **{"x": fm.X("v", bin=True), "y": "count()", **encoding})


_NET: dict[str, Case] = {
    # An encoding aggregate, each layer over its own rows by its own groupby.
    "layer_aggregate/F": Case(bar_mean, marks={"rect": 2}),
    "layer_aggregate/L-own-sum": Case(
        lambda: bar_mean().layer(layer("point", y=fm.Y("y", aggregate="sum"))),
        (bar_mean, lambda: agg("point", y="sum(y)")),
        chart_level=None,
    ),
    "layer_aggregate/L-raw-y": Case(
        lambda: bar_mean().layer(layer("point", y="y")),
        (bar_mean, lambda: P(x="g")),
        chart_level=None,
    ),
    "layer_aggregate/L-finer-color": Case(
        lambda: bar_mean().layer(layer("point", color="c")),
        (bar_mean, lambda: agg("point", color="c")),
        chart_level=None,
    ),
    "layer_aggregate/+s-right-binds-none": Case(
        lambda: bar_mean() + fm.Chart(df).mark_point(), (bar_mean, lambda: agg("point"))
    ),
    "layer_aggregate/+d-right-binds-none": Case(
        lambda: bar_mean() + fm.Chart(df2).mark_point(), (bar_mean, lambda: agg("point", df2))
    ),
    "layer_aggregate/+r": Case(lambda: P(x="g") + bar_mean(), (lambda: P(x="g"), bar_mean)),
    "layer_aggregate/+r-diff": Case(
        lambda: P(df2, x="g") + bar_mean(), (lambda: P(df2, x="g"), bar_mean)
    ),
    "layer_aggregate/LCm": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g"))._build_merged(),
        (bar_mean, lambda: P(df2, x="g")),
    ),
    "layer_aggregate/Fc": Case(lambda: bar_mean().facet("c"), marks={"rect": 4}),
    # Composite statistics per layer; attached layers read the host's rows.
    "composite_layer/H-layer-rug": Case(
        lambda: _hist().layer(layer("tick")),
        (_hist, lambda: mk("tick", x="v")),
        marks={"rect": 4, "line": 8},
        chart_level=None,
    ),
    "composite_layer/H+rug": Case(
        lambda: _hist() + mk("tick", x="v"), marks={"rect": 4, "line": 8}
    ),
    "composite_layer/rug+H": Case(
        lambda: mk("tick", x="v") + _hist(), marks={"line": 8, "rect": 4}
    ),
    # A user transform; attached layers read the host chain's output.
    # `datum.v > 3` keeps 4 of df's rows.
    "transform_layer/L": Case(
        lambda: P().transform(fm.transform_filter("datum.v > 3")).layer(layer("tick")),
        marks={"circle": 4, "line": 4},
        chart_level=None,
    ),
    "transform_layer/LD": Case(
        lambda: (
            P().transform(fm.transform_filter("datum.v > 3")).layer(fm.Layer(data=df2, mark="tick"))
        ),
        marks={"circle": 4, "line": 6},
    ),
    "transform_layer/+e": Case(
        lambda: (P() + line(df2)).transform(fm.transform_filter("datum.v > 3")),
        marks={"circle": 4, "polyline": 1},
    ),
    # A span with no statistic, read per layer.
    "span_no_statistic/F": Case(lambda: mk("bar", x="g", y="y", y2="r"), marks={"rect": 8}),
    "span_no_statistic/L-inherits-y2": Case(
        lambda: mk("bar", x="v", y="y", y2="r").layer(layer("tick")),
        marks={"rect": 8, "line": 8},
        chart_level=None,
    ),
    # A binned channel gets its span; a user transform's span.
    "binned_span/F": Case(_binned, marks={"rect": 4}),
    "binned_span/L-inherits-the-bin": Case(
        lambda: _binned().layer(layer("point")),
        (_binned, lambda: mk("point", x=fm.X("v", bin=True), y="count()")),
        chart_level=None,
    ),
    "transform_span/F": Case(
        lambda: (
            fm.Chart(df)
            .transform(
                fm.transform_aggregate(
                    {"field": "v", "fn": "min", "as": "lo"},
                    {"field": "v", "fn": "max", "as": "hi"},
                    groupby=["g"],
                )
            )
            .mark_bar()
            .encode(y="g", x="lo", x2="hi")
        ),
        marks={"rect": 2},
    ),
    # Facets partition the source; every statistic groups by them.
    "facet_grain/L": Case(
        lambda: bar_mean().layer(layer("text", text=fm.Text("y", aggregate="mean"))).facet("c"),
        marks={"rect": 4, "text": 4},
        chart_level=None,
    ),
    "facet_grain/SY": Case(
        lambda: (bar_mean() + SY("r")).facet("c"),
        marks={"rect": 4, "polyline": 2},
        chart_level=None,
    ),
    # Polar.
    "polar_color/F": Case(
        lambda: mk("arc", theta="z", color="g").coord(fm.CoordPolar()), marks={"path": 8}
    ),
    "polar_transform_aggregate/F": Case(
        lambda: (
            fm.Chart(df)
            .transform(
                fm.transform_aggregate({"field": "z", "fn": "sum", "as": "sz"}, groupby=["g"])
            )
            .mark_arc()
            .encode(theta="sz", color="g")
            .coord(fm.CoordPolar())
        ),
        marks={"path": 2},
    ),
    # The errorbar composite, per layer.
    "span_composite/+point": Case(
        lambda: mk("errorbar", x="g", y="y") + agg("point"),
        (lambda: mk("errorbar", x="g", y="y"), lambda: agg("point")),
    ),
    "span_composite/L-point": Case(
        lambda: mk("errorbar", x="g", y="y").layer(layer("point")), chart_level=None
    ),
}


@pytest.mark.parametrize("case", sorted(_NET))
def test_regression_net(case):
    check_rendering(_NET[case])


# ---------------------------------------------------------------------------
# An aggregate on a span end or a polar channel, per layer
# ---------------------------------------------------------------------------


def _ranged_bar() -> fm.Chart:
    """One bar per ``g`` from its min ``v`` to its max ``v``."""
    return mk("bar", y="g", x="min(v)", x2="max(v)")


def _polar(mark: str, data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    return mk(mark, data, **encoding).coord(fm.CoordPolar())


def _radial_bar() -> fm.Chart:
    """One radial bar per ``g``, its radius the group's summed ``z``."""
    return _polar("bar", theta="g", radius="sum(z)")


def _radial_label() -> fm.Chart:
    return _polar("text", theta="g", radius="sum(z)", text="g")


_SPAN_POLAR: dict[str, Case] = {
    "span_aggregate/F": Case(_ranged_bar, marks={"rect": 2}),
    "span_aggregate/F-objects": Case(
        lambda: mk("bar", y="g", x=fm.X("v", aggregate="min"), x2=fm.X2("v", aggregate="max")),
        marks={"rect": 2},
    ),
    "span_aggregate/F-rule": Case(
        lambda: mk("rule", x="g", y="min(y)", y2="max(y)"), marks={"line": 2}
    ),
    # A raw `v` beside `max(v)` is a group key: one bar per distinct (g, v).
    "span_aggregate/F-raw-x": Case(lambda: mk("bar", y="g", x="v", x2="max(v)"), marks={"rect": 8}),
    "span_aggregate/L-inherits": Case(
        lambda: _ranged_bar().layer(layer("rule")),
        (_ranged_bar, lambda: mk("rule", y="g", x="min(v)", x2="max(v)")),
        chart_level=None,
    ),
    "span_aggregate/L-own-objects": Case(
        lambda: P(y="g").layer(
            layer("bar", x=fm.X("v", aggregate="min"), x2=fm.X2("v", aggregate="max"))
        ),
        (lambda: P(y="g"), _ranged_bar),
        chart_level=None,
    ),
    **_cases("span_aggregate", plus_doors(_ranged_bar, lambda: mk("point", y="g", x="mean(v)"))),
    "polar_aggregate/F": Case(lambda: _polar("arc", theta="sum(z)", color="g"), marks={"path": 2}),
    "polar_aggregate/F-object": Case(
        lambda: _polar("arc", theta=fm.Theta("z", aggregate="sum"), color="g"), marks={"path": 2}
    ),
    "polar_aggregate/F-radius": Case(
        lambda: _polar("arc", theta="sum(z)", radius="mean(z)", color="g"), marks={"path": 2}
    ),
    "polar_aggregate/F-no-color": Case(lambda: _polar("arc", theta="sum(z)"), marks={"path": 1}),
    "polar_aggregate/F-count": Case(
        lambda: _polar("arc", theta="count()", color="g"), marks={"path": 2}
    ),
    "polar_aggregate/F-span-ends": Case(
        lambda: _polar(
            "arc", theta="min(z)", theta2="max(z)", radius="min(r)", radius2="max(r)", color="g"
        ),
        marks={"path": 2},
    ),
    "polar_aggregate/L-inherits": Case(
        lambda: _radial_bar().layer(layer("text", text="g")),
        (_radial_bar, _radial_label),
        chart_level=None,
    ),
    "polar_aggregate/L-raw-host": Case(
        lambda: _polar("bar", theta="g", radius="z").layer(
            layer("text", radius="sum(z)", text="g")
        ),
        (lambda: _polar("bar", theta="g", radius="z"), _radial_label),
        chart_level=None,
    ),
    **_cases("polar_aggregate", plus_doors(_radial_bar, _radial_label)),
    # Layer(data=): the label takes the host's written theta and radius,
    # aggregating its own rows by them.
    "polar_aggregate/LD-same-data": Case(
        lambda: _radial_bar().layer(fm.Layer(data=df, mark="text", encoding={"text": "g"})),
        (_radial_bar, _radial_label),
        chart_level=None,
    ),
    "polar_aggregate/LD-other-data": Case(
        lambda: _radial_bar().layer(fm.Layer(data=df2, mark="text", encoding={"text": "g"})),
        (_radial_bar, lambda: _polar("text", df2, theta="g", radius="sum(z)", text="g")),
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_SPAN_POLAR))
def test_a_span_or_polar_aggregate_is_computed_per_layer(case):
    """``aggregate=`` on x2/y2 and theta/radius/theta2/radius2 renders on every door.

    Rejects refusing it as drawing no value, the field-keyed output remap
    that pointed ``x`` and ``x2`` at one column when both aggregate ``v``,
    and a layer reading its own statistic's output without the polar
    binding it inherits from its host.
    """
    check_rendering(_SPAN_POLAR[case])
    check_wire_seam(_SPAN_POLAR[case])


def _via_transform(
    mark: str, ops: list[tuple[str, str]], groupby: list[str], **encoding: Any
) -> fm.Chart:
    """*mark* over a ``transform_aggregate`` naming each output ``<op>_<field>``, as an encoding aggregate does."""
    step = fm.transform_aggregate(
        *[{"field": field, "fn": op, "as": f"{op}_{field}"} for field, op in ops], groupby=groupby
    )
    return getattr(fm.Chart(df).transform(step), f"mark_{mark}")().encode(**encoding)


_TRANSFORM_ORACLES: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
    "span_aggregate/F": (
        _ranged_bar,
        lambda: _via_transform(
            "bar", [("v", "min"), ("v", "max")], ["g"], y="g", x="min_v", x2="max_v"
        ),
    ),
    "span_aggregate/F-rule": (
        lambda: mk("rule", x="g", y="min(y)", y2="max(y)"),
        lambda: _via_transform(
            "rule", [("y", "min"), ("y", "max")], ["g"], x="g", y="min_y", y2="max_y"
        ),
    ),
    "span_aggregate/F-raw-x": (
        lambda: mk("bar", y="g", x="v", x2="max(v)"),
        lambda: _via_transform("bar", [("v", "max")], ["g", "v"], y="g", x="v", x2="max_v"),
    ),
    "polar_aggregate/F": (
        lambda: _polar("arc", theta="sum(z)", color="g"),
        lambda: _via_transform("arc", [("z", "sum")], ["g"], theta="sum_z", color="g").coord(
            fm.CoordPolar()
        ),
    ),
    "polar_aggregate/F-radius": (
        lambda: _polar("arc", theta="sum(z)", radius="mean(z)", color="g"),
        lambda: _via_transform(
            "arc", [("z", "sum"), ("z", "mean")], ["g"], theta="sum_z", radius="mean_z", color="g"
        ).coord(fm.CoordPolar()),
    ),
    "polar_aggregate/F-span-ends": (
        lambda: _polar(
            "arc", theta="min(z)", theta2="max(z)", radius="min(r)", radius2="max(r)", color="g"
        ),
        lambda: _via_transform(
            "arc",
            [("z", "min"), ("z", "max"), ("r", "min"), ("r", "max")],
            ["g"],
            theta="min_z",
            theta2="max_z",
            radius="min_r",
            radius2="max_r",
            color="g",
        ).coord(fm.CoordPolar()),
    ),
}


@pytest.mark.parametrize("case", sorted(_TRANSFORM_ORACLES))
def test_an_encoding_aggregate_draws_what_the_same_transform_draws(case):
    """Each span end and polar channel draws its own (field, op) output.

    The oracle is the chart spelled with ``transform_aggregate`` computing
    the same outputs, so the bars span min to max and the arcs sweep each
    group's sum. Rejects zero-width bars from both ends reading the last
    op's column, and a raw ``x="v"`` moved onto ``max_v``.
    """
    encoded, transformed = _TRANSFORM_ORACLES[case]
    assert _quiet(encoded).to_svg() == _quiet(transformed).to_svg()


def _aggregate_steps(chart: Any) -> list[dict]:
    steps = _wire(chart).get("transforms") or []
    return [step for step in steps if step["type"] == "aggregate"]


def test_one_aggregate_computes_both_ends_of_the_span():
    """One Aggregate per layer holds both ops, grouped like x/y, and each end reads its own."""
    ops = [
        {"field": "v", "fn": "min", "as": "min_v"},
        {"field": "v", "fn": "max", "as": "max_v"},
    ]
    flat = _wire(_ranged_bar())
    assert [(s["ops"], s["groupby"]) for s in _aggregate_steps(_ranged_bar())] == [(ops, ["g"])]
    assert (flat["encoding"]["x"]["field"], flat["encoding"]["x2"]["field"]) == ("min_v", "max_v")
    layered = _ranged_bar().layer(layer("rule"))
    steps = _aggregate_steps(layered)
    assert [(s["ops"], s["groupby"]) for s in steps] == [(ops, ["g"])] * 2
    for wire_layer in _wire(layered)["layers"]:
        ends = (wire_layer["encoding"]["x"]["field"], wire_layer["encoding"]["x2"]["field"])
        assert ends == ("min_v", "max_v")


def test_a_raw_binding_of_an_aggregated_field_stays_a_group_key_on_a_flat_chart():
    """On a flat chart, a channel naming ``y`` raw beside ``size="mean(y)"`` draws ``y``, as a layer does.

    Rejects re-pointing every channel whose field an aggregate reads at that
    aggregate's output, which drew the flat chart's y axis from ``mean_y``
    and titled it so, while the same binding on a layer drew ``y``.
    """
    flat = _wire(mk("point", x="g", y="y", size="mean(y)"))
    assert (flat["encoding"]["y"]["field"], flat["encoding"]["size"]["field"]) == ("y", "mean_y")
    assert "mean_y" not in axis_titles(_quiet(mk("point", x="g", y="y", size="mean(y)").to_svg))


def test_a_polar_aggregate_is_lowered_onto_the_channel_the_remap_produces():
    """The remap runs first, so the aggregate is x's and a layer reading its own output keeps its angle.

    The text layer's own statistic moves it onto a named output, where Rust
    passes no chart positional channel: its inherited ``theta`` reaches it as
    the chart's ``x``, listed as an inherited positional binding, as an
    inherited ``x`` would.
    """
    flat = _wire(_polar("arc", theta="sum(z)", color="g"))
    assert flat["encoding"]["x"]["field"] == "sum_z"
    assert "theta" not in flat["encoding"]
    layered = _wire(_polar("bar", theta="g", radius="z").layer(layer("text", text="sum(z)")))
    label = layered["layers"][1]
    assert label["encoding"]["x"] == {"field": "g"}
    assert label["inherited_positional"] == ["x", "y"]
    assert label["encoding"]["text"]["field"] == "sum_z"


# ---------------------------------------------------------------------------
# The labelled pie, every route, static and interactive
# ---------------------------------------------------------------------------


def _pie(theta: str = "z", **coord: Any) -> fm.Chart:
    return mk("arc", theta=theta, color="g").coord(fm.CoordPolar(**coord))


def _stacked(theta: str) -> fm.Theta:
    """``Theta(stack=True)`` over *theta*'s field, keeping a shorthand aggregate."""
    field, aggregate = (theta[4:-1], "sum") if theta.startswith("sum(") else (theta, None)
    return fm.Theta(field, aggregate=aggregate, stack=True)


def _pie_labels(theta: str = "z", data: pl.DataFrame = df, **coord: Any) -> fm.Chart:
    """Labels stacked into the pie the arc draws over *theta*, one per wedge."""
    return mk("text", data, theta=_stacked(theta), text="g").coord(fm.CoordPolar(**coord))


def _labelled_pie_doors(theta: str) -> dict[str, Case]:
    """The arc beside its stacked labels on every route that composes two charts."""
    arc, text = (lambda: _pie(theta)), (lambda: _pie_labels(theta))
    counts = (8, 8) if theta == "z" else (2, 2)
    return {
        **plus_doors(arc, text),
        # The attached text inherits no color (it draws only its own), while
        # the chart-level color the seam helper completes it from is the
        # host's, so its marks are pinned by count.
        "L": Case(
            lambda: arc().layer(layer("text", theta=_stacked(theta), text="g")),
            marks={"path": counts[0], "text": counts[1]},
            chart_level=None,
        ),
        # Layer(data=) over the host's rows, computing its own statistic.
        "LD-same-data": Case(
            lambda: arc().layer(
                fm.Layer(data=df, mark="text", encoding={"theta": _stacked(theta), "text": "g"})
            ),
            marks={"path": counts[0], "text": counts[1]},
            chart_level=None,
        ),
    }


_LABELLED_PIE: dict[str, Case] = {
    **_cases("pie_label", _labelled_pie_doors("z")),
    **_cases("pie_label/sum", _labelled_pie_doors("sum(z)")),
    "pie_label/F-text": Case(_pie_labels, marks={"text": 8}),
    "pie_label/F-band": Case(lambda: _pie("g"), marks={"path": 8}),
}


@pytest.mark.parametrize("case", sorted(_LABELLED_PIE))
def test_a_labelled_pie_renders_its_wedges_and_labels_on_every_route(case):
    """``arc + text`` over one theta draws each layer as it draws alone.

    Rejects refusing a theta-only text under ``CoordPolar`` (``encoding 'y'
    expected EncodingSpec, got None``) and a text aggregating at its
    sibling's grain on the ``sum(z)`` variant.
    """
    check_rendering(_LABELLED_PIE[case])
    check_wire_seam(_LABELLED_PIE[case])


def _pie_mid_angles(values: Any) -> np.ndarray:
    """Each wedge's mid-angle in radians, clockwise from the top: an independent cumulative sum."""
    v = np.asarray(values, dtype=float)
    return np.cumsum(v) / v.sum() * 2 * np.pi - v / v.sum() * np.pi


def _polar_point(x: float, y: float, cx: float, cy: float) -> tuple[float, float]:
    """``(angle, radius)`` of a pixel about the center, angle clockwise from the top."""
    return float(np.arctan2(x - cx, cy - y) % (2 * np.pi)), float(np.hypot(x - cx, y - cy))


_LABEL = re.compile(
    r'<text x="([-\d.]+)" y="([-\d.]+)" fill="([^"]*)" font-family="Inter" font-size="11" '
    r'text-anchor="middle">([^<]*)</text>'
)


def _static_labels(chart: Any) -> tuple[list[tuple[str, float, float]], float]:
    """The data labels inside the plot, as ``(content, angle, radius)``, and the outer radius."""
    svg = _quiet(chart.to_svg)
    x0, y0, x1, y1 = next(iter(_clip_rects(_svg_root(svg)).values()))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    labels = [
        (text, *_polar_point(float(x), float(y), cx, cy))
        for x, y, fill, text in _LABEL.findall(svg)
        if fill != _CHROME_TEXT and x0 <= float(x) <= x1 and y0 <= float(y) <= y1
    ]
    return labels, min(x1 - x0, y1 - y0) / 2


def _interactive_labels(chart: Any) -> tuple[list[tuple[str, float, float]], float]:
    panel = _quiet(lambda: _scene(chart))["panels"][0]
    area = panel["plot_area"]
    cx, cy = area["x"] + area["w"] / 2, area["y"] + area["h"] / 2
    labels = [
        (node["content"], *_polar_point(node["x"], node["y"], cx, cy))
        for batch in panel["marks"]
        if batch["kind"] == "text"
        for node in batch["nodes"]
    ]
    return labels, panel["coord"]["outer_radius"]


def _assert_labels(got: list, expected: list) -> None:
    got, expected = sorted(got), sorted(expected)
    assert [g[0] for g in got] == [e[0] for e in expected]
    flat = [float(v) for label in got for v in label[1:]]
    assert flat == pytest.approx([float(v) for label in expected for v in label[1:]], abs=1e-3)


_SUMS = df.group_by("g", maintain_order=True).agg(pl.col("z").sum())

#: Per route and variant: the chart, and each label's (content, mid-angle,
#: radius as a function of the outer radius). The default label radius is
#: mid-ring, half the outer radius on a pie.
_LABEL_ORACLES: dict[str, tuple[Callable[[], Any], Callable[[float], list]]] = {}
for _variant, _theta, _contents, _values in (
    ("z", "z", df["g"].to_list(), df["z"].to_numpy()),
    ("sum", "sum(z)", _SUMS["g"].to_list(), _SUMS["z"].to_numpy()),
):
    for _door, _case in _labelled_pie_doors(_theta).items():
        _LABEL_ORACLES[f"pie_label/{_variant}/{_door}"] = (
            _case.build,
            lambda outer, c=_contents, v=_values: [
                (t, a, outer / 2) for t, a in zip(c, _pie_mid_angles(v))
            ],
        )
_LABEL_ORACLES["pie_label/z/F-text"] = (
    _pie_labels,
    lambda outer: [(t, a, outer / 2) for t, a in zip(df["g"], _pie_mid_angles(df["z"]))],
)
# Layer(data=) over other rows: the labels stack their own rows' pie.
_SUMS2 = df2.group_by("g", maintain_order=True).agg(pl.col("z").sum())
for _variant, _theta, _contents, _values in (
    ("z", "z", df2["g"].to_list(), df2["z"].to_numpy()),
    ("sum", "sum(z)", _SUMS2["g"].to_list(), _SUMS2["z"].to_numpy()),
):
    _LABEL_ORACLES[f"pie_label/{_variant}/LD-other-data"] = (
        lambda t=_theta: _pie(t).layer(
            fm.Layer(data=df2, mark="text", encoding={"theta": _stacked(t), "text": "g"})
        ),
        lambda outer, c=_contents, v=_values: [
            (t, a, outer / 2) for t, a in zip(c, _pie_mid_angles(v))
        ],
    )


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_LABEL_ORACLES))
def test_each_label_sits_at_its_wedges_mid_angle_at_mid_ring(case, kind):
    """A ``Theta(stack=True)`` label's angle is its wedge's mid-angle from a cumulative sum.

    Rejects placing a stacked label by the linear theta projection (off its
    wedge) and a ``sum(z)`` label summing rows its wedge does not.
    """
    build, expected = _LABEL_ORACLES[case]
    chart = _quiet(build)
    if kind == "static" and case.endswith("/LCx"):
        pytest.skip("LCx renders only interactively")
    labels, outer = (_static_labels if kind == "static" else _interactive_labels)(chart)
    _assert_labels(labels, expected(outer))


_r = df["r"].to_numpy()
_g = df["g"].to_list()
_mid = _pie_mid_angles(df["z"])
_bands = {"a": 0.5 * np.pi, "b": 1.5 * np.pi}

#: Each label's radius off the default: a donut's mid-ring, a radius
#: binding, a radius= style value; and an untyped string theta's bands.
_LABEL_EXTRAS: dict[str, tuple[Callable[[], Any], Callable[[float], list]]] = {
    "donut": (
        lambda: _pie(inner_radius=50) + _pie_labels(inner_radius=50),
        lambda outer: [(t, a, (50 + outer) / 2) for t, a in zip(_g, _mid)],
    ),
    "radius-binding": (
        lambda: mk("text", theta=_stacked("z"), radius="r", text="g").coord(fm.CoordPolar()),
        lambda outer: [(t, a, r / _r.max() * outer) for t, a, r in zip(_g, _mid, _r)],
    ),
    "radius-style": (
        lambda: (
            fm.Chart(df)
            .mark_text(radius=60)
            .encode(theta=_stacked("z"), text="g")
            .coord(fm.CoordPolar())
        ),
        lambda outer: [(t, a, 60.0) for t, a in zip(_g, _mid)],
    ),
    "untyped-band": (
        lambda: _pie("g") + _pie_labels("g"),
        lambda outer: [(t, _bands[t], outer / 2) for t in _g],
    ),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_LABEL_EXTRAS))
def test_a_label_sits_at_its_radius_binding_style_or_mid_ring(case, kind):
    """The label radius is its radius binding, else ``radius=``, else mid-ring.

    The untyped string theta (``theta="g"``, no declared type) draws one
    band per category, and a label at each band's mid-angle; it drew no
    wedge before, reading the strings as pie numbers.
    """
    build, expected = _LABEL_EXTRAS[case]
    chart = _quiet(build)
    labels, outer = (_static_labels if kind == "static" else _interactive_labels)(chart)
    _assert_labels(labels, expected(outer))


def _polar_points() -> fm.Chart:
    return _polar("point", theta="v", radius="r")


def _point_labels() -> fm.Chart:
    """Unstacked labels: each sits on its own theta value, where its point draws."""
    return _polar("text", theta="v", radius="r", text="k")


_SCATTER_LABELS: dict[str, Callable[[], Any]] = {
    "+": lambda: _polar_points() + _point_labels(),
    "+rev": lambda: _point_labels() + _polar_points(),
    "L": lambda: _polar_points().layer(layer("text", text="k")),
    "LD-same-data": lambda: _polar_points().layer(
        fm.Layer(data=df, mark="text", encoding={"text": "k"})
    ),
    "LCm": lambda: fm.LayerChart(_polar_points(), _point_labels())._build_merged(),
    "LCs": lambda: fm.LayerChart(_polar_points(), _point_labels()),
    "LCx": lambda: fm.LayerChart(_polar_points(), _point_labels()),
}

_CIRCLE = re.compile(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"')


def _static_points_and_labels(chart: Any) -> tuple[list, list]:
    svg = _quiet(chart.to_svg)
    x0, y0, x1, y1 = next(iter(_clip_rects(_svg_root(svg)).values()))
    points = [
        (float(x), float(y))
        for x, y in _CIRCLE.findall(svg)
        if x0 <= float(x) <= x1 and y0 <= float(y) <= y1
    ]
    labels = [
        (float(x), float(y))
        for x, y, fill, _ in _LABEL.findall(svg)
        if fill != _CHROME_TEXT and x0 <= float(x) <= x1 and y0 <= float(y) <= y1
    ]
    return points, labels


def _interactive_points_and_labels(chart: Any) -> tuple[list, list]:
    marks = _quiet(lambda: _scene(chart))["panels"][0]["marks"]
    points = [(n["cx"], n["cy"]) for b in marks if b["kind"] == "point" for n in b["nodes"]]
    labels = [(n["x"], n["y"]) for b in marks if b["kind"] == "text" for n in b["nodes"]]
    return points, labels


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("route", sorted(_SCATTER_LABELS))
def test_an_unstacked_polar_label_sits_on_its_point(route, kind):
    """Without ``Theta(stack=True)`` a polar text sits on its own theta value.

    Each label of a polar scatter draws where its point draws, on every
    route, ``Layer(data=)`` included (it takes the host's written theta and
    radius). Rejects stacking every polar text into a pie, which moved the
    labels to pie mid-angles off their points.
    """
    if kind == "static" and route == "LCx":
        pytest.skip("LCx renders only interactively")
    chart = _quiet(_SCATTER_LABELS[route])
    read = _static_points_and_labels if kind == "static" else _interactive_points_and_labels
    points, labels = read(chart)
    assert len(points) == len(labels) == df.height
    assert sorted(labels) == pytest.approx(sorted(points), abs=1e-2)


def _polar_label_warnings(render: Callable[[], Any]) -> set[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render()
    return {str(w.message) for w in caught if "sit off its wedges" in str(w.message)}


_UNSTACKED_LABEL_WARNING = (
    "layer 1: mark_text: these labels sit at their own theta values, but an arc in the same "
    "panel stacks that theta into a pie, so they sit off its wedges; write stack=True on the "
    "text's theta (Theta(..., stack=True), on the text or on the theta it inherits) to place "
    "each label in its wedge"
)


def _unstacked_labels(field: str = "z") -> fm.Chart:
    return mk("text", theta=field, text="g").coord(fm.CoordPolar())


_UNSTACKED_BESIDE_A_PIE: dict[str, Callable[[], Any]] = {
    "+": lambda: (_pie() + _unstacked_labels()).to_svg(),
    "L": lambda: _pie().layer(layer("text", text="g")).to_svg(),
    "LCs": lambda: fm.LayerChart(_pie(), _unstacked_labels()).to_svg(),
    "I+": lambda: (_pie() + _unstacked_labels()).interactive().to_html(),
    "LCx": lambda: fm.LayerChart(_pie(), _unstacked_labels()).interactive().to_html(),
    # The warning names no lowered column (``sum_z``) the user never wrote.
    "+sum": lambda: (_pie("sum(z)") + _unstacked_labels("sum(z)")).to_svg(),
}


@pytest.mark.parametrize("route", sorted(_UNSTACKED_BESIDE_A_PIE))
def test_an_unstacked_label_beside_a_pie_on_its_field_warns(route):
    """A text on the field an arc stacks, itself unstacked, warns its labels sit off the wedges."""
    assert _polar_label_warnings(_UNSTACKED_BESIDE_A_PIE[route]) == {_UNSTACKED_LABEL_WARNING}


@pytest.mark.parametrize(
    "build",
    [
        lambda: _pie() + _pie_labels(),
        lambda: _pie() + _unstacked_labels("r"),
        _unstacked_labels,
    ],
    ids=["stacked", "other-field", "no-arc"],
)
def test_a_stacked_label_or_one_off_the_pies_field_does_not_warn(build):
    assert _polar_label_warnings(lambda: build().to_svg()) == set()


def test_theta_stack_on_a_polar_text_reaches_the_wire_and_is_not_warned():
    """``Theta(stack=True)`` on a polar text is honored, on its own and on an attached layer.

    Rejects stripping it with the warning that stacking is honored by bar
    and area only; a Cartesian text's ``stack=`` is still stripped and warned.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = json.loads(_pie_labels().to_spec().to_json())
        stacked = layer("text", theta=_stacked("z"), text="g")
        attached = json.loads(_pie().layer(stacked).to_spec().to_json())
    assert flat["encoding"]["x"] == {"field": "z", "stack": "zero"}
    assert attached["layers"][1]["encoding"]["x"] == {"field": "z", "stack": "zero"}
    from ferrum._warn import reset_warnings

    reset_warnings()
    with pytest.warns(UserWarning) as caught:
        spec = mk("text", x="g", y=fm.Y("z", stack=True), text="g").to_spec()
    cartesian = json.loads(spec.to_json())
    assert "stack" not in cartesian["encoding"]["y"]
    assert [str(w.message) for w in caught] == [
        "stack='zero' is not supported by mark_text and was ignored. Stacking is honored by "
        "mark_bar and mark_area, and by mark_arc and mark_text on the theta channel of a "
        "CoordPolar."
    ]


@pytest.mark.parametrize(
    "build",
    [
        lambda: fm.Chart(df).mark_text(radius=-10),
        lambda: fm.Layer(mark="text", mark_kwargs={"radius": -10}),
    ],
    ids=["mark_text", "Layer"],
)
def test_a_negative_label_radius_is_refused_at_construction(build):
    """A negative ``radius=`` would mirror each label through the center, so it refuses."""
    with pytest.raises(ValueError, match=r"\(radius=\.\.\.\) must be non-negative; received -10$"):
        build()


def test_a_zero_label_radius_places_every_label_at_the_center():
    chart = fm.Chart(df).mark_text(radius=0).encode(theta=_stacked("z"), text="g")
    labels, _ = _static_labels(chart.coord(fm.CoordPolar()))
    assert [radius for _, _, radius in labels] == pytest.approx([0.0] * df.height, abs=1e-3)


def _radius_style_labels() -> fm.Chart:
    return (
        fm.Chart(df)
        .mark_text(radius=60)
        .encode(theta=_stacked("z"), text="k")
        .coord(fm.CoordPolar())
    )


_RADIUS_OWNERSHIP: dict[str, Callable[[], Any]] = {
    # The attached text inherits the host's stacked theta but not its style.
    "L": lambda: _radius_style_labels().layer(layer("text", text="g")),
    "+": lambda: _radius_style_labels() + _pie_labels(),
    "+rev": lambda: _pie_labels() + _radius_style_labels(),
    "LCm": lambda: fm.LayerChart(_radius_style_labels(), _pie_labels())._build_merged(),
    "LCs": lambda: fm.LayerChart(_radius_style_labels(), _pie_labels()),
    "LCx": lambda: fm.LayerChart(_radius_style_labels(), _pie_labels()),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("route", sorted(_RADIUS_OWNERSHIP))
def test_a_radius_style_stays_on_its_own_text_layer(route, kind):
    """One operand's ``mark_text(radius=60)`` never moves a sibling text's labels.

    The ``k`` labels sit at 60 px, the ``g`` labels at mid-ring.
    """
    if kind == "static" and route == "LCx":
        pytest.skip("LCx renders only interactively")
    chart = _quiet(_RADIUS_OWNERSHIP[route])
    labels, outer = (_static_labels if kind == "static" else _interactive_labels)(chart)
    expected = [(k, a, 60.0) for k, a in zip(df["k"], _mid)] + [
        (g, a, outer / 2) for g, a in zip(_g, _mid)
    ]
    _assert_labels(labels, expected)


# A stack written once on a shared theta places both the arc and its labels.


def _shared_stacked_theta(data: pl.DataFrame = df) -> fm.Chart:
    return fm.Chart(data).encode(theta=fm.Theta("z", stack=True), color="g")


def _stacked_pie() -> fm.Chart:
    return _shared_stacked_theta().mark_arc().coord(fm.CoordPolar())


def _stacked_pie_text() -> fm.Chart:
    return _shared_stacked_theta().mark_text().encode(text="g").coord(fm.CoordPolar())


_SHARED_STACK: dict[str, Callable[[], Any]] = {
    "L": lambda: _stacked_pie().layer(layer("text", text="g")),
    "LD": lambda: _stacked_pie().layer(fm.Layer(data=df, mark="text", encoding={"text": "g"})),
    "+": lambda: _stacked_pie() + _stacked_pie_text(),
    "LCs": lambda: fm.LayerChart(_stacked_pie(), _stacked_pie_text()),
    "LCx": lambda: fm.LayerChart(_stacked_pie(), _stacked_pie_text()),
}


def _all_warnings(render: Callable[[], Any]) -> list[str]:
    from ferrum._warn import reset_warnings

    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render()
    return [str(w.message) for w in caught]


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("route", sorted(_SHARED_STACK))
def test_a_stack_written_once_on_the_shared_theta_labels_the_pie(route, kind):
    """``Theta('z', stack=True)`` on the base places the arc's wedges and the text's labels.

    Each label sits at its wedge's mid-angle, and nothing warns. Rejects
    judging the chart-level binding by the host mark, which stripped the
    stack the text inherits and warned that the arc ignored it.
    """
    if kind == "static" and route == "LCx":
        pytest.skip("LCx renders only interactively")
    chart = _quiet(_SHARED_STACK[route])
    render = chart.to_svg if kind == "static" else lambda: chart.interactive().to_html()
    assert _all_warnings(render) == []
    labels, outer = (_static_labels if kind == "static" else _interactive_labels)(chart)
    _assert_labels(labels, [(t, a, outer / 2) for t, a in zip(_g, _mid)])


def test_an_arcs_theta_stack_is_honored_without_a_warning():
    """An arc always sweeps its theta into a pie, so ``Theta(stack=True)`` draws the same pie."""
    stacked = _stacked_pie()
    assert _all_warnings(stacked.to_svg) == []
    assert stacked.to_svg() == _quiet(lambda: _pie().to_svg())
    assert json.loads(stacked.to_spec().to_json())["encoding"]["x"]["stack"] == "zero"


def test_a_point_host_warns_and_its_attached_text_keeps_the_stack():
    """A polar point cannot stack its theta, so it warns; the text drawing the same binding stacks it."""
    host = _polar("point", theta=fm.Theta("z", stack=True), radius="r")
    chart = host.layer(layer("text", text="g"))
    messages = _all_warnings(chart.to_svg)
    labels, outer = _static_labels(chart)
    _assert_labels(labels, [(t, a, r / _r.max() * outer) for t, a, r in zip(_g, _mid, _r)])
    assert messages == [
        "stack='zero' is not supported by mark_point and was ignored. Stacking is honored by "
        "mark_bar and mark_area, and by mark_arc and mark_text on the theta channel of a "
        "CoordPolar."
    ]


_STACK_ROWS = pl.DataFrame(
    {"g": ["a", "a", "b", "b"], "c": ["p", "q", "p", "q"], "v": [1.0, 2, 3, 4]}
)


def _stacked_y(mark: str) -> fm.Chart:
    chart = getattr(fm.Chart(_STACK_ROWS), f"mark_{mark}")()
    return chart.encode(x="g", y=fm.Y("v", stack=True), color="c")


def _stacked_segments(svg: str) -> bool:
    """Whether each x drawing two bars draws them end to end, one on top of the other."""
    from tests._svg_rects import data_bar_rects, mark_layer

    columns: dict[str, list[tuple[float, float]]] = {}
    for r in data_bar_rects(mark_layer(svg)):
        columns.setdefault(r["x"], []).append((float(r["y"]), float(r["height"])))
    pairs = [sorted(column) for column in columns.values() if len(column) == 2]
    return bool(pairs) and all(abs(c[0][0] + c[0][1] - c[1][0]) < 1e-2 for c in pairs)


def test_a_bar_attached_to_a_point_host_keeps_the_stack_the_point_ignores():
    """A stacked y the point host cannot honor still stacks the bars that inherit it.

    Each x's two bars sit one on top of the other, and the point warns once.
    Rejects stripping the chart-level stack by the host mark.
    """
    chart = _stacked_y("point").layer(layer("bar"))
    messages = _all_warnings(chart.to_svg)
    assert _stacked_segments(chart.to_svg())
    assert messages == [
        "stack='zero' is not supported by mark_point and was ignored. Stacking is honored by "
        "mark_bar and mark_area, and by mark_arc and mark_text on the theta channel of a "
        "CoordPolar."
    ]


def test_a_point_attached_to_a_bar_host_draws_every_row_unstacked():
    """A point inheriting a stacked y it cannot honor draws a stack-less copy, every row, and warns.

    Rejects handing it the stack: Rust's value-axis stack then drew 3 of the
    point's 4 rows with no warning.
    """
    chart = _stacked_y("bar").layer(layer("point"))
    messages = _all_warnings(chart.to_svg)
    from tests._svg_rects import mark_layer

    assert mark_layer(chart.to_svg()).count("<circle") == _STACK_ROWS.height
    point = _wire(chart)["layers"][1]
    assert point["encoding"]["y"] == {"field": "v"}
    assert point["inherited_positional"] == ["y"]
    assert messages == [
        "stack='zero' is not supported by mark_point and was ignored. Stacking is honored by "
        "mark_bar and mark_area, and by mark_arc and mark_text on the theta channel of a "
        "CoordPolar."
    ]


# A text draws a color only when it binds one: an inherited color groups its
# statistic, as detail does, and stays off its wire.


def _colored_mean_bars() -> fm.Chart:
    return mk("bar", x="g", y="mean(y)", color="c")


def _mean_label(**encoding: Any) -> dict:
    return {"text": fm.Text("y", aggregate="mean"), **encoding}


_TEXT_COLOR_DOORS: dict[str, Callable[..., Any]] = {
    "L": lambda **own: _colored_mean_bars().layer(layer("text", **_mean_label(**own))),
    "LD": lambda **own: _colored_mean_bars().layer(
        fm.Layer(data=df, mark="text", encoding=_mean_label(**own))
    ),
    "+e": lambda **own: (
        mk("bar", x="g", y="mean(y)") + mk("text", x="g", y="mean(y)", **_mean_label(**own))
    ).encode(color="c"),
}


def _label_fills_and_values(svg: str) -> list[tuple[str, float]]:
    x0, y0, x1, y1 = next(iter(_clip_rects(_svg_root(svg)).values()))
    return sorted(
        (fill, float(text))
        for x, y, fill, text in _LABEL.findall(svg)
        if fill != _CHROME_TEXT and x0 <= float(x) <= x1 and y0 <= float(y) <= y1
    )


@pytest.mark.parametrize("door", sorted(_TEXT_COLOR_DOORS))
def test_a_text_groups_by_an_inherited_color_without_drawing_or_carrying_it(door):
    """A text under a colored parent draws one label per colored bar, in its own color.

    The inherited color groups the label's mean as a grouping-only key, the
    wire's ``group``; it is neither the text layer's color nor drawn, and
    nothing warns.
    """
    chart = _TEXT_COLOR_DOORS[door]()
    assert _all_warnings(chart.to_svg) == []
    text = next(layer_ for layer_ in _wire(chart)["layers"] if layer_["mark"] == "text")
    assert "color" not in text["encoding"]
    assert "color" not in text.get("inherited_channels", [])
    assert "detail" not in (text.get("mark_style") or {})
    assert text["encoding"]["group"] == {"field": "c"}
    means = df.group_by(["g", "c"]).agg(pl.col("y").mean())["y"].to_list()
    labels = _label_fills_and_values(chart.to_svg())
    assert sorted(value for _, value in labels) == pytest.approx(sorted(means))
    assert {fill for fill, _ in labels} == {"#1f2937"}


@pytest.mark.parametrize("door", sorted(_TEXT_COLOR_DOORS))
def test_a_text_binding_its_own_color_draws_it(door):
    """A color the text binds itself still draws on its labels."""
    chart = _TEXT_COLOR_DOORS[door](color="c")
    labels = _label_fills_and_values(_quiet(chart.to_svg))
    assert len(labels) == 4
    assert "#1f2937" not in {fill for fill, _ in labels}


# Every door a text inherits a color through: the inherited color spellings
# group the text's statistic at the bars' grain, and a text's own color
# spelling shadows them all.

#: Two rows per (g, c, f) group, so each group's mean is its own.
_GROUP_ROWS = pl.DataFrame(
    {
        "g": ["a"] * 8 + ["b"] * 8,
        "c": (["p"] * 4 + ["q"] * 4) * 2,
        "f": (["u"] * 2 + ["v"] * 2) * 4,
        "y": [float(i) for i in range(1, 17)],
        "v": [1.1, 1.6, 2.2, 2.7, 3.1, 3.8, 4.4, 4.9] * 2,
    }
)
#: Unit bins over [1, 5], so each row's bin is the floor of its ``v``.
_UNIT_BINS = {"bin_width": 1.0, "extent": (1.0, 5.0), "nice": False}
_GROUP_ROWS_BINNED = _GROUP_ROWS.with_columns(pl.col("v").floor().alias("v_bin"))
#: The wire scale ``Color("v", bin=_UNIT_BINS)`` draws through.
_UNIT_BIN_SCALE = {
    "type": "bin-ordinal",
    "bin": {"bin_width": 1.0, "extent": [1.0, 5.0], "nice": False},
}


def _binned_color() -> fm.Color:
    return fm.Color("v", bin=_UNIT_BINS)


def _group_bars(**parent: Any) -> fm.Chart:
    return mk("bar", _GROUP_ROWS, x="g", y="mean(y)", **parent)


def _group_label(**own: Any) -> fm.Chart:
    return mk("text", _GROUP_ROWS, x="g", y="mean(y)", **_mean_label(**own))


def _group_rule() -> fm.Chart:
    return mk("rule", _GROUP_ROWS, y="y")


_GROUP_DOORS: dict[str, Callable[[dict, dict], Any]] = {
    "L": lambda parent, own: _group_bars(**parent).layer(layer("text", **_mean_label(**own))),
    "LD": lambda parent, own: _group_bars(**parent).layer(
        fm.Layer(data=_GROUP_ROWS, mark="text", encoding=_mean_label(**own))
    ),
    "+e": lambda parent, own: (_group_bars() + _group_label(**own)).encode(**parent),
    "+L": lambda parent, own: (
        _group_bars(**parent).layer(layer("text", **_mean_label(**own))) + _group_rule()
    ),
    "+e+": lambda parent, own: (
        (_group_bars() + _group_label(**own)).encode(**parent) + _group_rule()
    ),
    "LC": lambda parent, own: fm.LayerChart(
        _group_bars(**parent).layer(layer("text", **_mean_label(**own))), _group_rule()
    ),
}

#: Per binding shape: the parent's bindings, the text's own, and the fields
#: the label's mean groups by (the bars' grain, or the text's own color).
_GROUP_SHAPES: dict[str, tuple[dict, dict, list[str]]] = {
    "color+detail": ({"color": "c", "detail": "f"}, {}, ["g", "c", "f"]),
    "own-detail": ({"color": "c"}, {"detail": "f"}, ["g", "c", "f"]),
    "parent-fill": ({"fill": "c"}, {}, ["g", "c"]),
    "parent-stroke": ({"stroke": "c"}, {}, ["g", "c"]),
    "two-spellings": ({"color": "c", "fill": "f"}, {}, ["g", "c", "f"]),
    "own-color": ({"color": "c"}, {"color": "f"}, ["g", "f"]),
    "own-fill": ({"color": "c"}, {"fill": "f"}, ["g", "f"]),
    "own-stroke": ({"color": "c"}, {"stroke": "f"}, ["g", "f"]),
    "aggregated-color": ({"color": fm.Color("y", aggregate="sum")}, {}, ["g"]),
    "binned-color": ({"color": _binned_color()}, {}, ["g", "v_bin"]),
}

_GROUP_CASES = [
    pytest.param(shape, door, kind, id=f"{shape}/{door}/{kind}")
    for shape in _GROUP_SHAPES
    for door in _GROUP_DOORS
    for kind in ("static", "interactive")
]


def _interactive_label_values(chart: Any) -> list[float]:
    marks = _quiet(lambda: _scene(chart))["panels"][0]["marks"]
    return [float(n["content"]) for b in marks if b["kind"] == "text" for n in b["nodes"]]


def _label_values(chart: Any, kind: str) -> list[float]:
    if kind == "interactive":
        return sorted(_interactive_label_values(chart))
    return sorted(value for _, value in _label_fills_and_values(_quiet(chart.to_svg)))


@pytest.mark.parametrize(("shape", "door", "kind"), _GROUP_CASES)
def test_a_text_label_groups_by_every_inherited_color_spelling_its_own_color_shadows(
    shape, door, kind
):
    """A text's mean groups by each inherited color spelling and detail, at the bars' grain.

    A text's own color, fill or stroke shadows the parent's under any
    spelling, an aggregated color groups nothing, and a binned color groups
    by its bins, as the bars do. A host's or ``+`` operand's ``detail``
    groups the text's mean although a text draws no detail. Rejects
    lowering the inherited color onto the one ``detail`` slot, which
    dropped the color beside a detail, kept only the first of two
    spellings, and grouped an own ``color`` by the parent's too; grouping a
    binned color by its raw values; and handing a ``+`` operand's detail
    only to the layers whose mark draws it, which grouped its text by
    ``(g, c)`` beside bars grouped by ``(g, c, f)``.
    """
    parent, own, grain = _GROUP_SHAPES[shape]
    chart = _quiet(lambda: _GROUP_DOORS[door](parent, own))
    if kind == "interactive":
        chart = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    groups = _GROUP_ROWS_BINNED.group_by(grain).agg(pl.col("y").mean())["y"].to_list()
    assert _label_values(chart, kind) == pytest.approx(sorted(groups))


@pytest.mark.parametrize("door", sorted(_GROUP_DOORS))
def test_the_text_wire_carries_its_grouping_key_and_no_color(door):
    """The text layer's wire names the parent's color field as ``group`` and binds no color.

    Two spellings carry the one the parent draws as color (``color`` over
    ``fill``); an aggregated color carries no ``group``.
    """

    def text_layer(parent: dict) -> dict:
        chart = _quiet(lambda: _GROUP_DOORS[door](parent, {}))
        return next(lay for lay in _wire(chart)["layers"] if lay["mark"] == "text")

    two = text_layer({"color": "c", "fill": "f"})
    assert two["encoding"]["group"] == {"field": "c"}
    assert "color" not in two["encoding"]
    assert "group" not in two.get("inherited_channels", [])
    assert text_layer({"fill": "c"})["encoding"]["group"] == {"field": "c"}
    assert "group" not in text_layer({"color": fm.Color("y", aggregate="sum")})["encoding"]


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("own_text", ["mean(y)", "g"])
@pytest.mark.parametrize("door", sorted(_GROUP_DOORS))
def test_a_text_under_a_binned_color_draws_one_label_per_binned_bar(door, own_text, kind):
    """A text under a binned color bins with it, with or without its own aggregate.

    One label per binned bar, and its wire ``group`` carries the bars' binned
    color scale, so Rust groups its statistic, dodge and stack by those bins.
    Rejects grouping by the raw field, which drew one label per distinct value
    over fewer bars.
    """
    from tests._svg_rects import data_bar_rects, mark_layer

    own = {} if own_text == "mean(y)" else {"text": own_text}
    chart = _quiet(lambda: _GROUP_DOORS[door]({"color": _binned_color()}, own))
    bars = _GROUP_ROWS_BINNED.select(["g", "v_bin"]).n_unique()
    if kind == "static":
        svg = _quiet(chart.to_svg)
        assert len(data_bar_rects(mark_layer(svg))) == bars
        x0, y0, x1, y1 = next(iter(_clip_rects(_svg_root(svg)).values()))
        labels = [
            text
            for x, y, fill, text in _LABEL.findall(svg)
            if fill != _CHROME_TEXT and x0 <= float(x) <= x1 and y0 <= float(y) <= y1
        ]
    else:
        merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
        panel_marks = _quiet(lambda: _scene(merged))["panels"][0]["marks"]
        labels = [n for b in panel_marks if b["kind"] == "text" for n in b["nodes"]]
    assert len(labels) == bars
    merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    text = next(lay for lay in _wire(merged)["layers"] if lay["mark"] == "text")
    assert text["encoding"]["group"] == {"field": "v", "scale": _UNIT_BIN_SCALE}


# Dodged and stacked labels sit on their bars: Rust's position grouping
# reads the text's grouping key.

_DODGE_ROWS = pl.DataFrame(
    {"g": ["a", "a", "b", "b"], "c": ["q", "p", "q", "p"], "y": [1.0, 2, 3, 4]}
)


def _dodged_bars(**encoding: Any) -> fm.Chart:
    chart = fm.Chart(_DODGE_ROWS).mark_bar(position=fm.Dodge())
    return chart.encode(x="g", y="y", **encoding)


def _dodged_labels() -> fm.Chart:
    return fm.Chart(_DODGE_ROWS).mark_text(position=fm.Dodge()).encode(x="g", y="y", text="y")


_DODGE_ROUTES: dict[str, Callable[[], Any]] = {
    "+e": lambda: (_dodged_bars() + _dodged_labels()).encode(color="c"),
    "L-host-dodge": lambda: _dodged_bars(color="c").layer(layer("text", text="y")),
    "+e+rule": lambda: (
        (_dodged_bars() + _dodged_labels()).encode(color="c")
        + fm.Chart(_DODGE_ROWS).mark_rule().encode(y="y")
    ),
}


def _bar_centers_and_labels(svg: str) -> tuple[list[float], list[float]]:
    from tests._svg_rects import data_bar_rects, mark_layer

    marks = mark_layer(svg)
    centers = sorted(round(float(r["x"]) + float(r["width"]) / 2, 1) for r in data_bar_rects(marks))
    labels = sorted(
        round(float(x), 1) for x, _, fill, _ in _LABEL.findall(svg) if fill != _CHROME_TEXT
    )
    return centers, labels


@pytest.mark.parametrize("route", sorted(_DODGE_ROUTES))
def test_a_dodged_label_sits_on_its_own_bar(route):
    """Each dodged label sits at its bar's center, on every route.

    Rejects the nested ``+`` route dropping the text's grouping, which put
    both labels of a band at its center.
    """
    centers, labels = _bar_centers_and_labels(_quiet(lambda: _DODGE_ROUTES[route]().to_svg()))
    assert len(labels) == _DODGE_ROWS.height
    assert labels == centers


def test_a_stacked_label_sits_on_its_own_segment():
    """A label attached to a stacked bar stacks with the bars, one per segment, centered on it."""
    chart = (
        fm.Chart(_DODGE_ROWS)
        .mark_bar(position=fm.Stack())
        .encode(x="g", y="y", color="c")
        .layer(layer("text", text="y"))
    )
    from tests._svg_rects import data_bar_rects, mark_layer

    svg = _quiet(chart.to_svg)
    segments = sorted(
        (round(float(r["x"]) + float(r["width"]) / 2, 1), float(r["y"]), float(r["height"]))
        for r in data_bar_rects(mark_layer(svg))
    )
    labels = sorted(
        (round(float(x), 1), float(y))
        for x, y, fill, _ in _LABEL.findall(svg)
        if fill != _CHROME_TEXT
    )
    assert len(labels) == len(segments) == _DODGE_ROWS.height
    for (lx, ly), (sx, sy, sh) in zip(labels, segments, strict=True):
        assert lx == sx
        assert sy <= ly <= sy + sh


def _stacked_label_contents(svg: str) -> list[str]:
    return sorted(text for _x, _y, fill, text in _LABEL.findall(svg) if fill != _CHROME_TEXT)


#: In group ``b``, ``c``'s row order is p, q, but the categorical domain
#: stacks q before p, so a label pinned to its own segment (not sorted
#: insertion order) is a real discriminating check, not an accident of a
#: fixture where stack order and row order happen to coincide.
_STACK_ORDER_ROWS = pl.DataFrame(
    {"g": ["a", "a", "b", "b"], "c": ["q", "p", "p", "q"], "y": [1.0, 2.0, 3.0, 4.0]}
)


#: ``c``'s first-seen order in ``_STACK_ORDER_ROWS`` is q, p, so the default
#: categorical palette assigns q the first color, p the second -- stable
#: across every chart built from this fixture, static and interactive.
_STACK_ORDER_FILL_TO_C = {"#2563eb": "q", "#dc2626": "p"}
_STACK_ORDER_RGB_TO_C = {(37, 99, 235): "q", (220, 38, 38): "p"}


def _label_g_map(xs: list[float]) -> dict[float, str]:
    """The two distinct label x positions, in ascending order, as g='a' then g='b'."""
    ordered = sorted(set(xs))
    assert len(ordered) == 2
    return dict(zip(ordered, ("a", "b")))


def _alone_label_pairs(svg: str) -> set[tuple[str, str, str]]:
    """Each ``door='alone'`` label's (g, c, content): the text itself draws ``color='c'``."""
    labels = [
        (round(float(x), 1), fill, text)
        for x, _y, fill, text in _LABEL.findall(svg)
        if fill != _CHROME_TEXT
    ]
    g_of_x = _label_g_map([x for x, _fill, _text in labels])
    return {(g_of_x[x], _STACK_ORDER_FILL_TO_C[fill], text) for x, fill, text in labels}


def _interactive_alone_label_pairs(chart: Any) -> set[tuple[str, str, str]]:
    """Interactive twin of :func:`_alone_label_pairs`, reading the text nodes' own ``color``."""
    nodes = [
        (round(n["x"], 1), n["style"]["color"], n["content"])
        for panel in _quiet(lambda: _scene(chart))["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "text"
        for n in batch["nodes"]
        if "content" in n
    ]
    g_of_x = _label_g_map([x for x, _color, _content in nodes])
    return {
        (g_of_x[x], _STACK_ORDER_RGB_TO_C[(color["r"], color["g"], color["b"])], content)
        for x, color, content in nodes
    }


def _beside_bars_label_pairs(svg: str) -> set[tuple[str, str, str]]:
    """Each ``door='beside-bars'`` label's (g, c, content): a text layer beside the host bars

    inherits no color of its own, so c is read off the bar segment sitting
    at the label's exact position instead (center x, same y: an
    ``anchor='top'`` stacked label sits at its segment's top, matching the
    bar's own top).
    """
    from tests._svg_rects import data_bar_rects, mark_layer

    segments = {
        (round(float(r["x"]) + float(r["width"]) / 2, 1), round(float(r["y"]), 1)): r["fill"]
        for r in data_bar_rects(mark_layer(svg))
    }
    labels = [
        (round(float(x), 1), round(float(y), 1), text)
        for x, y, fill, text in _LABEL.findall(svg)
        if fill != _CHROME_TEXT
    ]
    g_of_x = _label_g_map([x for x, _y, _text in labels])
    return {(g_of_x[x], _STACK_ORDER_FILL_TO_C[segments[(x, y)]], text) for x, y, text in labels}


def _interactive_beside_bars_label_pairs(chart: Any) -> set[tuple[str, str, str]]:
    """Interactive twin of :func:`_beside_bars_label_pairs`, matching bar and text nodes by position."""
    bar_fills: dict[tuple[float, float], tuple[int, int, int]] = {}
    labels: list[tuple[float, float, str]] = []
    for panel in _quiet(lambda: _scene(chart))["panels"]:
        for batch in panel["marks"]:
            if batch["kind"] == "bar":
                for n in batch["nodes"]:
                    center = round(n["x"] + n["w"] / 2, 1)
                    fill = n["style"]["fill"]
                    bar_fills[(center, round(n["y"], 1))] = (fill["r"], fill["g"], fill["b"])
            elif batch["kind"] == "text":
                for n in batch["nodes"]:
                    if "content" in n:
                        labels.append((round(n["x"], 1), round(n["y"], 1), n["content"]))
    g_of_x = _label_g_map([x for x, _y, _content in labels])
    return {
        (g_of_x[x], _STACK_ORDER_RGB_TO_C[bar_fills[(x, y)]], content) for x, y, content in labels
    }


def _label_segment_scale_factors(svg: str) -> list[float]:
    """Each stacked label's content, paired to its own bar segment by position, as a height/value ratio.

    Every row shares one linear y scale from a zero baseline, so a
    segment's pixel height is proportional to its OWN row's value,
    regardless of what any other stacked row's value is. The ratio
    (height / label content) is the same constant for every row when the
    label prints its own value; it is NOT constant when a label prints the
    cumulative stack top instead (a different, value-dependent distortion
    per segment).
    """
    from tests._svg_rects import data_bar_rects, mark_layer

    segments = sorted(
        (round(float(r["x"]) + float(r["width"]) / 2, 1), float(r["y"]), float(r["height"]))
        for r in data_bar_rects(mark_layer(svg))
    )
    labels = sorted(
        (round(float(x), 1), float(y), text)
        for x, y, fill, text in _LABEL.findall(svg)
        if fill != _CHROME_TEXT
    )
    assert len(labels) == len(segments)
    return [
        sh / float(content)
        for (_lx, _ly, content), (_sx, _sy, sh) in zip(labels, segments, strict=True)
    ]


@pytest.mark.parametrize("door", ["alone", "beside-bars"])
def test_a_stacked_labels_text_prints_its_own_value_not_the_stack_top(door):
    """``mark_text(position=fm.Stack())`` prints each row's own value, as ggplot2's position_stack does.

    The stacked *position* is unchanged (pinned by
    ``test_a_stacked_label_sits_on_its_own_segment``); only the printed
    content differs. Static and interactive agree. Rejects printing the
    cumulative stack top, which would distort the height/content ratio
    differently per segment instead of leaving it constant, on a text mark
    carrying its own ``Stack()`` position and on one attached beside
    stacked bars.
    """
    text = (
        fm.Chart(_STACK_ORDER_ROWS)
        .mark_text(position=fm.Stack())
        .encode(x="g", y="y", text="y", color="c")
    )
    if door == "alone":
        chart = text
    else:
        bars = (
            fm.Chart(_STACK_ORDER_ROWS)
            .mark_bar(position=fm.Stack())
            .encode(x="g", y="y", color="c")
        )
        chart = bars.layer(fm.Layer(mark="text", encoding={"text": "y"}))
    svg = _quiet(chart.to_svg)
    assert _stacked_label_contents(svg) == ["1", "2", "3", "4"]

    #: Ground truth: row 0 (g=a, c=q, y=1), row 1 (g=a, c=p, y=2),
    #: row 2 (g=b, c=p, y=3), row 3 (g=b, c=q, y=4).
    expected_pairs = {("a", "q", "1"), ("a", "p", "2"), ("b", "p", "3"), ("b", "q", "4")}
    if door == "alone":
        assert _alone_label_pairs(svg) == expected_pairs
        assert _interactive_alone_label_pairs(chart) == expected_pairs
    else:
        scale_factors = _label_segment_scale_factors(svg)
        assert scale_factors == pytest.approx([scale_factors[0]] * len(scale_factors), rel=1e-3)
        assert _beside_bars_label_pairs(svg) == expected_pairs
        assert _interactive_beside_bars_label_pairs(chart) == expected_pairs


def _interactive_stacked_label_contents(chart: Any) -> list[str]:
    """Interactive twin of :func:`_stacked_label_contents`: every text node's own content, sorted."""
    return sorted(
        n["content"]
        for panel in _quiet(lambda: _scene(chart))["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "text"
        for n in batch["nodes"]
        if "content" in n
    )


def test_a_stacked_label_on_a_null_row_prints_nothing():
    """A stacked label's null row prints no content, on either door, matching its ``Dodge()`` twin.

    Static and interactive agree. Rejects mapping a null's own-value
    companion column to ``0`` (the cumulative-sum reading that would make
    a null row print ``"0"`` instead of staying blank).
    """
    rows = _STACK_ORDER_ROWS.with_columns(
        y=pl.Series([1.0, None, 3.0, 4.0]),
    )
    dodge_text = (
        fm.Chart(rows).mark_text(position=fm.Dodge()).encode(x="g", y="y", text="y", color="c")
    )
    stack_text = (
        fm.Chart(rows).mark_text(position=fm.Stack()).encode(x="g", y="y", text="y", color="c")
    )
    for door in ("alone", "beside-bars"):
        if door == "alone":
            dodge_chart, stack_chart = dodge_text, stack_text
        else:
            dodge_bars = (
                fm.Chart(rows).mark_bar(position=fm.Dodge()).encode(x="g", y="y", color="c")
            )
            stack_bars = (
                fm.Chart(rows).mark_bar(position=fm.Stack()).encode(x="g", y="y", color="c")
            )
            dodge_chart = dodge_bars.layer(fm.Layer(mark="text", encoding={"text": "y"}))
            stack_chart = stack_bars.layer(fm.Layer(mark="text", encoding={"text": "y"}))
        dodge_svg = _quiet(dodge_chart.to_svg)
        stack_svg = _quiet(stack_chart.to_svg)
        assert (
            _stacked_label_contents(stack_svg)
            == _stacked_label_contents(dodge_svg)
            == [
                "1",
                "3",
                "4",
            ]
        )
        assert (
            _interactive_stacked_label_contents(stack_chart)
            == _interactive_stacked_label_contents(dodge_chart)
            == ["1", "3", "4"]
        )


_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)


def _title_entries(svg: str) -> list[dict[str, str]]:
    """Each static ``<title>``'s fields as a name->value dict, one dict per title.

    A duplicate name inside one title (the dedup bug this reader guards
    against) raises here rather than silently collapsing into one key.
    """
    out = []
    for title in _TITLE_RE.findall(svg):
        fields: dict[str, str] = {}
        for entry in title.split(", "):
            name, _, value = entry.partition(": ")
            assert name not in fields, f"duplicate {name!r} entry in one <title>: {title!r}"
            fields[name] = value
        out.append(fields)
    return out


def _interactive_tooltip_entries(chart: Any) -> list[dict[str, str]]:
    """Each interactive tooltip's fields as a name->value dict, one dict per row.

    Tooltips are carried on ``batch["tooltips"]``, row-aligned with
    ``batch["nodes"]`` by index -- not on each node itself.
    """
    out = []
    for panel in _quiet(lambda: _scene(chart))["panels"]:
        for batch in panel["marks"]:
            for t in batch.get("tooltips") or []:
                fields: dict[str, str] = {}
                for f in t["fields"]:
                    assert f["name"] not in fields, f"duplicate {f['name']!r} entry: {t['fields']}"
                    fields[f["name"]] = f["value"]
                out.append(fields)
    return out


def _tooltip_values(chart: Any, field: str) -> list[str]:
    """Every *field* value *chart*'s tooltip entries hold, static and interactive, sorted and equal.

    A null row's value is blank (kept, as ``""``) or the row carries no
    entry for *field* at all (skipped); either way each entry holds *field*
    at most once, enforced by the two entry readers above.
    """
    static = sorted(d[field] for d in _title_entries(_quiet(chart.to_svg)) if field in d)
    interactive = sorted(d[field] for d in _interactive_tooltip_entries(chart) if field in d)
    assert static == interactive
    return static


def test_a_non_positional_channel_reads_the_rows_own_value_under_a_position_adjustment():
    """A tooltip or color channel bound to the adjusted field reads the row's own value.

    Compared against the ``fm.Dodge()`` twin, which never rewrites the
    field in place, so its tooltip/color values ARE each row's own value
    by construction. Covers ``Stack()`` (bar/point/text with ``tooltip``,
    and a bar's own ``color``) and data-space ``Jitter()`` (``tooltip``,
    on the same continuous field it jitters).
    """
    rows = _STACK_ORDER_ROWS.with_columns(y=pl.Series([1.0, 2.0, None, 4.0]))

    for mark in ("bar", "point", "text"):
        extra = {"text": "y"} if mark == "text" else {}
        stack_chart = getattr(fm.Chart(rows), f"mark_{mark}")(position=fm.Stack()).encode(
            x="g", y="y", color="c", tooltip="y", **extra
        )
        dodge_chart = getattr(fm.Chart(rows), f"mark_{mark}")(position=fm.Dodge()).encode(
            x="g", y="y", color="c", tooltip="y", **extra
        )
        # Blanks dropped before comparing: a null row's own value is blank
        # under both, but a stacked mark still draws a zero-height ghost for
        # it (one blank title) while a dodged mark draws none at all (no
        # title) -- a separate, already-pinned difference in whether a null
        # row's mark exists at all, not in what a present row's value is.
        non_null_stack = [v for v in _tooltip_values(stack_chart, "y") if v != ""]
        non_null_dodge = [v for v in _tooltip_values(dodge_chart, "y") if v != ""]
        assert non_null_stack == non_null_dodge == ["1", "2", "4"]

    # A bar's own color, bound to the stacked field: each rect's fill
    # follows its own row's value, giving 4 distinct fills (as the Dodge
    # twin's does), not the 2 cumulative stack tops it would give if color
    # read the overwritten column. No null here: a null row's own-color
    # ghost segment is a separate question from whether a non-null row's
    # color reads its own value.
    stack_color = (
        fm.Chart(_STACK_ORDER_ROWS)
        .mark_bar(position=fm.Stack())
        .encode(x="g", y="y", color="y", detail="c")
    )
    dodge_color = (
        fm.Chart(_STACK_ORDER_ROWS)
        .mark_bar(position=fm.Dodge())
        .encode(x="g", y="y", color="y", detail="c")
    )
    from tests._svg_rects import data_bar_rects, mark_layer

    stack_fills = {r["fill"] for r in data_bar_rects(mark_layer(_quiet(stack_color.to_svg)))}
    dodge_fills = {r["fill"] for r in data_bar_rects(mark_layer(_quiet(dodge_color.to_svg)))}
    assert stack_fills == dodge_fills
    assert len(stack_fills) == 4

    # Jitter's data-space branch only rewrites a CONTINUOUS axis in place
    # (a categorical axis jitters in pixel space, untouched by the own-value
    # retarget); the tooltip is bound to that same continuous field, so a
    # duplicate or a jittered reading would show here.
    jitter_chart = (
        fm.Chart(df)
        .mark_point(position=fm.Jitter(width=0.3, seed=2))
        .encode(x="v", y="y", tooltip="v")
    )
    unjittered_chart = fm.Chart(df).mark_point().encode(x="v", y="y", tooltip="v")
    assert _tooltip_values(jitter_chart, "v") == _tooltip_values(unjittered_chart, "v")


#: Two rows, one shared ``g``, two distinct ``v``/``c`` values: enough to
#: give a ``Stack``, a continuous ``Dodge`` or a data-space ``Jitter`` real
#: work to do, and small enough that each row's own value is unambiguous.
_SELECTION_DEDUP_ROWS = pl.DataFrame({"g": ["a", "a"], "v": [60.0, 40.0], "c": ["p", "q"]})

#: Each case's chart, built with ``x``/``y``/the position adjustment already
#: set; the test adds ``tooltip=``/the selection itself.
_SELECTION_DEDUP_CASES: dict[str, Callable[[], fm.Chart]] = {
    "stack": lambda: (
        fm.Chart(_SELECTION_DEDUP_ROWS)
        .mark_bar(position=fm.Stack(by="c"))
        .encode(x="g", y="v", color="c")
    ),
    "dodge-continuous": lambda: (
        fm.Chart(_SELECTION_DEDUP_ROWS)
        .mark_point(position=fm.Dodge(by="c"))
        .encode(x="v", y="v", color="c")
    ),
    "jitter-data-space": lambda: (
        fm.Chart(_SELECTION_DEDUP_ROWS)
        .mark_point(position=fm.Jitter(width=0.5, seed=1))
        .encode(x="v", y="v")
    ),
}


@pytest.mark.parametrize("case", sorted(_SELECTION_DEDUP_CASES))
def test_a_field_selection_dedupes_against_its_tooltip_and_reads_the_own_value(case):
    """A selection on an adjusted field never doubles its tooltip entry, with or without ``tooltip=``.

    Covers ``Stack``, continuous ``Dodge`` and data-space ``Jitter``,
    static and interactive: ``tooltip='v'`` plus a field selection on
    ``v`` holds exactly one ``v`` entry, the row's own value; a selection
    alone (no ``tooltip=``) gives the same one entry. Rejects a second,
    unretargeted entry (a duplicate '60'/'100'-style reading) and rejects
    the selection-only entry reading the adjusted position instead of the
    row's own value.
    """
    build = _SELECTION_DEDUP_CASES[case]

    with_tooltip = build().encode(tooltip="v").add_selection(fm.selection_point(fields=["v"]))
    selection_only = build().add_selection(fm.selection_point(fields=["v"]))

    for chart in (with_tooltip, selection_only):
        svg = _quiet(chart.to_svg)
        static_entries = _title_entries(svg)
        interactive_entries = _interactive_tooltip_entries(chart)
        assert len(static_entries) == len(interactive_entries) == 2
        # _title_entries/_interactive_tooltip_entries each raise on a
        # duplicate key within one entry, so reaching this point already
        # proves "exactly one" 'v' per row; this checks its VALUE too.
        assert sorted(d["v"] for d in static_entries) == ["40", "60"]
        assert sorted(d["v"] for d in interactive_entries) == ["40", "60"]


def test_a_stacked_keys_interactive_read_the_rows_own_value():
    """``key=`` on a stacked field reads each row's own value, not the cumulative stack top.

    A sibling's height changing a later row's cumulative position must not
    re-key an earlier row's mark (the whole point of a key: a stable row
    identity across data updates).
    """
    chart = (
        fm.Chart(_SELECTION_DEDUP_ROWS)
        .mark_bar(position=fm.Stack(by="c"))
        .encode(x="g", y="v", color="c", key="v")
    )
    keys = [
        key
        for panel in _quiet(lambda: _scene(chart))["panels"]
        for batch in panel["marks"]
        for key in (batch.get("keys") or [])
    ]
    assert sorted(keys) == ["40", "60"]


def _binned_mark(mark: str, position: Any, **encoding: Any) -> fm.Chart:
    chart = getattr(fm.Chart(_GROUP_ROWS), f"mark_{mark}")(position=position)
    return chart.encode(x="g", y="mean(y)", **encoding)


_BINNED_DODGE_ROUTES: dict[str, Callable[[], Any]] = {
    "+e": lambda: (
        _binned_mark("bar", fm.Dodge())
        + _binned_mark("text", fm.Dodge(), text=fm.Text("y", aggregate="mean"))
    ).encode(color=_binned_color()),
    "L-host-dodge": lambda: _binned_mark("bar", fm.Dodge(), color=_binned_color()).layer(
        layer("text", text=fm.Text("y", aggregate="mean"))
    ),
    "+e+rule": lambda: (
        (
            _binned_mark("bar", fm.Dodge())
            + _binned_mark("text", fm.Dodge(), text=fm.Text("y", aggregate="mean"))
        ).encode(color=_binned_color())
        + _group_rule()
    ),
}


@pytest.mark.parametrize("route", sorted(_BINNED_DODGE_ROUTES))
def test_a_dodged_label_under_a_binned_color_sits_on_its_bar(route):
    """Each label dodges by the bars' bins and sits at its own bar's center."""
    centers, labels = _bar_centers_and_labels(
        _quiet(lambda: _BINNED_DODGE_ROUTES[route]().to_svg())
    )
    assert len(labels) == _GROUP_ROWS_BINNED.select(["g", "v_bin"]).n_unique()
    assert labels == centers


def test_a_bar_stacked_by_a_binned_color_draws_one_segment_per_x_and_bin():
    """A mean bar stacked by a binned color: one segment per (x, bin), each its bin's mean.

    Each bar's segments split its height as that x's bin means split their
    total. Rejects grouping by the raw field, which stacked two segments per
    bin.
    """
    from tests._svg_rects import data_bar_rects, mark_layer

    svg = _quiet(_binned_mark("bar", fm.Stack(), color=_binned_color()).to_svg)
    bars: dict[float, list[float]] = {}
    for r in data_bar_rects(mark_layer(svg)):
        bars.setdefault(round(float(r["x"]), 1), []).append(float(r["height"]))
    means = _GROUP_ROWS_BINNED.group_by(["g", "v_bin"]).agg(pl.col("y").mean())
    expected = [sorted(means.filter(pl.col("g") == g)["y"].to_list()) for g in ("a", "b")]
    drawn = [sorted(bars[x]) for x in sorted(bars)]
    assert [len(h) for h in drawn] == [len(m) for m in expected] == [4, 4]
    for heights, bin_means in zip(drawn, expected, strict=True):
        assert [h / sum(heights) for h in heights] == pytest.approx(
            [m / sum(bin_means) for m in bin_means], abs=1e-3
        )


def test_a_stacked_label_under_a_binned_color_sits_on_its_segment():
    """A label attached to a bar stacked by a binned color stacks with it, one per segment."""
    from tests._svg_rects import data_bar_rects, mark_layer

    chart = _binned_mark("bar", fm.Stack(), color=_binned_color()).layer(
        layer("text", text=fm.Text("y", aggregate="mean"))
    )
    svg = _quiet(chart.to_svg)
    segments = sorted(
        (round(float(r["x"]) + float(r["width"]) / 2, 1), float(r["y"]), float(r["height"]))
        for r in data_bar_rects(mark_layer(svg))
    )
    labels = sorted(
        (round(float(x), 1), float(y))
        for x, y, fill, _ in _LABEL.findall(svg)
        if fill != _CHROME_TEXT
    )
    assert len(labels) == len(segments) == _GROUP_ROWS_BINNED.select(["g", "v_bin"]).n_unique()
    for (lx, ly), (sx, sy, sh) in zip(labels, segments, strict=True):
        assert lx == sx
        assert sy <= ly <= sy + sh


def _mean_text() -> fm.Text:
    return fm.Text("y", aggregate="mean")


#: Every door that attaches a mark to a stacked mean bar, by the color it stacks by.
_STACKED_BAR_DOORS: dict[str, Callable[[Any], Any]] = {
    "L": lambda color: _binned_mark("bar", fm.Stack(), color=color).layer(
        layer("text", text=_mean_text())
    ),
    "+e": lambda color: (
        _binned_mark("bar", fm.Stack()) + _binned_mark("text", fm.Stack(), text=_mean_text())
    ).encode(color=color),
    "LD": lambda color: _binned_mark("bar", fm.Stack(), color=color).layer(
        fm.Layer(
            data=_GROUP_ROWS, mark="text", encoding={"x": "g", "y": "mean(y)", "text": _mean_text()}
        )
    ),
    "+point": lambda color: (
        _binned_mark("bar", fm.Stack(), color=color) + mk("point", _GROUP_ROWS, x="g", y="y")
    ),
}
_STACK_COLORS: dict[str, tuple[Callable[[], Any], str]] = {
    "nominal": (lambda: "c", "c"),
    "binned": (_binned_color, "v_bin"),
}


@pytest.mark.parametrize("color", sorted(_STACK_COLORS))
@pytest.mark.parametrize("door", sorted(_STACKED_BAR_DOORS))
def test_a_mark_attached_to_a_stacked_statistic_removes_no_segment(door, color):
    """A label or point attached to a stacked mean bar keeps every segment; a label sits on its own.

    Rejects an axis spanning the bar output's unstacked means: the segments
    stacked above the largest mean fell past it and were dropped.
    """
    from tests._svg_rects import data_bar_rects, mark_layer

    build, key = _STACK_COLORS[color]
    svg = _quiet(_STACKED_BAR_DOORS[door](build()).to_svg)
    segments = sorted(
        (round(float(r["x"]) + float(r["width"]) / 2, 1), float(r["y"]), float(r["height"]))
        for r in data_bar_rects(mark_layer(svg))
    )
    assert len(segments) == _GROUP_ROWS_BINNED.select(["g", key]).n_unique()
    labels = sorted(
        (round(float(x), 1), float(y))
        for x, y, fill, _ in _LABEL.findall(svg)
        if fill != _CHROME_TEXT
    )
    if door == "+point":
        return
    assert len(labels) == len(segments)
    for (lx, ly), (sx, sy, sh) in zip(labels, segments, strict=True):
        assert lx == sx
        assert sy <= ly <= sy + sh


def test_stacked_operands_on_their_own_frames_draw_every_segment():
    """``+`` over two frames: the right operand stacks taller than the left's rows reach.

    Every segment of both operands draws, each band's stack as tall as its
    total, the right operand's reaching 15. Rejects reading each operand's branch output
    unstacked, which stopped the axis at 8 and dropped the right operand's
    upper segments.
    """
    from tests._svg_rects import data_bar_rects, mark_layer

    def stacked(rows: pl.DataFrame) -> fm.Chart:
        return fm.Chart(rows).mark_bar(position=fm.Stack()).encode(x="cat", y="val", color="grp")

    left = pl.DataFrame(
        {"cat": ["a", "a", "b", "b"], "grp": ["g1", "g2"] * 2, "val": [1.0, 2.0, 3.0, 4.0]}
    )
    right = pl.DataFrame(
        {"cat": ["c", "c", "d", "d"], "grp": ["g1", "g2"] * 2, "val": [5.0, 6.0, 7.0, 8.0]}
    )
    svg = _quiet((stacked(left) + stacked(right)).to_svg)
    rects = data_bar_rects(mark_layer(svg))
    assert len(rects) == 8
    stacks: dict[float, float] = {}
    for r in rects:
        band = round(float(r["x"]), 1)
        stacks[band] = stacks.get(band, 0.0) + float(r["height"])
    # Each band's stack height is its total (3, 7, 11, 15) on one scale.
    heights = [stacks[x] for x in sorted(stacks)]
    assert [h / heights[0] for h in heights] == pytest.approx([1.0, 7 / 3, 11 / 3, 5.0], abs=1e-3)


@pytest.mark.parametrize("route", ["flat", "+point"])
def test_an_area_stacked_by_the_stack_shorthand_draws_every_band(route):
    """``Y(stack="zero")`` stacks an area's bands on an axis that spans their tops.

    Rejects an axis spanning the unstacked values, which cut the top band.
    """
    rows = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0] * 3,
            "y": [float(i) for i in range(1, 10)],
            "c": ["p"] * 3 + ["q"] * 3 + ["r"] * 3,
        }
    )
    chart = mk("area", rows, x="x", y=fm.Y("y", stack="zero"), color="c")
    if route == "+point":
        chart = chart + mk("point", rows, x="x", y="y")
    stacked = _quiet(chart.to_svg)
    explicit = fm.Chart(rows).mark_area(position=fm.Stack()).encode(x="x", y="y", color="c")
    if route == "+point":
        explicit = explicit + mk("point", rows, x="x", y="y")
    assert stacked == _quiet(explicit.to_svg)


# The stack shorthand stacks the channel that declares it: x=X(..., stack=)
# stacks a horizontal mark's segments end to end along x, as y=Y(..., stack=)
# stacks a vertical one along y. A shorthand on both channels is refused.

_HSTACK_ROWS = pl.DataFrame(
    {
        "g": ["a", "a", "b", "b"],
        "z": [2.0, 6.0, 3.0, 1.0],
        "t": [0.0, 1.0, 0.0, 1.0],
        "c": ["u", "v", "u", "v"],
        "f": ["p", "p", "q", "q"],
    }
)


def _hbar(stack: Any = "zero") -> fm.Chart:
    """A native horizontal bar, value ``z`` on x, with ``stack`` on x (``None``: unstacked)."""
    return mk("bar", _HSTACK_ROWS, x=fm.X("z", stack=stack), y="g", color="c")


def _hbar_explicit(offset: str = "zero") -> fm.Chart:
    """The same bar under the explicit horizontal ``Stack``."""
    return (
        fm.Chart(_HSTACK_ROWS)
        .mark_bar(position=fm.Stack(offset=offset, value_axis="x"))
        .encode(x="z", y="g", color="c")
    )


def _hpoint() -> fm.Chart:
    return mk("point", _HSTACK_ROWS, x="z", y="g")


def _hbar_layer(stack: Any = None) -> fm.Layer:
    return fm.Layer(mark="bar", encoding={"x": fm.X("z", stack=stack), "y": "g", "color": "c"})


#: Door -> how it composes a horizontal bar built by the argument with a point.
_HSTACK_DOORS: dict[str, Callable[[Callable[[], fm.Chart]], Any]] = {
    "F": lambda bar: bar(),
    "Fc": lambda bar: bar().facet("f"),
    "+": lambda bar: bar() + _hpoint(),
    "+rev": lambda bar: _hpoint() + bar(),
    "+c": lambda bar: (bar() + _hpoint()).facet("f"),
    "LCs": lambda bar: fm.LayerChart(bar(), _hpoint()),
    "flip": lambda bar: bar().coord(fm.CoordFlip()),
}


def _hstack_door(door: str, which: str) -> Any:
    """``door``'s chart with the bar stacked by the x shorthand, the explicit horizontal stack, or unstacked.

    A ``Layer`` takes no ``position=``, so the attached door's explicit
    stack is the point composed with the explicitly stacked bar, which
    renders what the attached bar would.
    """
    if door == "L":
        if which == "explicit":
            return _hpoint() + _hbar_explicit()
        return _hpoint().layer(_hbar_layer("zero" if which == "shorthand" else None))
    bar = {"shorthand": _hbar, "explicit": _hbar_explicit, "unstacked": lambda: _hbar(None)}[which]
    return _HSTACK_DOORS[door](bar)


def _band_segments(svg: str) -> dict[float, list[tuple[float, float]]]:
    """Each y band's bar segments as ``(left, right)``, left to right."""
    from tests._svg_rects import data_bar_rects, mark_layer

    bands: dict[float, list[tuple[float, float]]] = {}
    for r in data_bar_rects(mark_layer(svg)):
        left = float(r["x"])
        bands.setdefault(round(float(r["y"]), 1), []).append((left, left + float(r["width"])))
    return {band: sorted(segments) for band, segments in bands.items()}


def test_a_horizontal_bar_stacks_its_segments_end_to_end_along_x():
    """``X("z", stack="zero")`` on a native horizontal bar draws each band's segments end to end.

    Each band's second segment starts where its first ends, and the band
    totals (8 and 4) read 2 to 1 against an axis spanning the stack. Rejects
    reading the shorthand from y on an unflipped chart, which drew every
    segment from 0, overlapping.
    """
    bands = _band_segments(_quiet(_hbar().to_svg))
    assert len(bands) == 2
    for first, second in bands.values():
        assert second[0] == pytest.approx(first[1], abs=1e-2)
    widths = [segments[-1][1] - segments[0][0] for _, segments in sorted(bands.items())]
    assert widths[0] / widths[1] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("door", sorted([*_HSTACK_DOORS, "L"]))
def test_the_x_stack_shorthand_draws_the_explicit_horizontal_stack_on_every_door(door):
    """The x shorthand renders what ``Stack(value_axis="x")`` renders, static and interactive.

    Rejects any door reading the shorthand from y, which drew the bars
    unstacked: the unstacked chart's render, which the shorthand must differ from.
    """

    def scene(which: str) -> dict:
        return _quiet(lambda: _scene(_hstack_door(door, which)))

    assert _quiet(_hstack_door(door, "shorthand").to_svg) == _quiet(
        _hstack_door(door, "explicit").to_svg
    )
    assert scene("shorthand") == scene("explicit")
    assert scene("shorthand") != scene("unstacked")


@pytest.mark.parametrize("offset", ["zero", "normalize", "center"])
def test_every_stack_offset_on_x_stacks_along_x(offset):
    """Each offset of the x shorthand draws the explicit horizontal stack with that offset.

    Rejects an x shorthand honored for ``"zero"`` only.
    """
    shorthand = _quiet(_hbar(offset).to_svg)
    assert shorthand == _quiet(_hbar_explicit(offset).to_svg)
    assert shorthand != _quiet(_hbar(None).to_svg)


@pytest.mark.parametrize("off", [None, False])
def test_stack_none_or_false_on_x_stacks_nothing(off):
    """``X(..., stack=None)`` and ``stack=False`` draw the plain, unstacked bar."""
    plain = mk("bar", _HSTACK_ROWS, x="z", y="g", color="c")
    assert _quiet(_hbar(off).to_svg) == _quiet(plain.to_svg)


@pytest.mark.parametrize("route", ["flat", "+point"])
def test_a_horizontal_area_stacked_by_the_x_shorthand_draws_the_explicit_stack(route):
    """``X("z", stack="zero")`` on an area over ``y="t"`` stacks its bands along x.

    Rejects the y-only reading, which drew the bands unstacked.
    """

    def area(position: Any = None, stack: Any = None) -> fm.Chart:
        chart = (
            fm.Chart(_HSTACK_ROWS)
            .mark_area(position=position)
            .encode(x=fm.X("z", stack=stack), y="t", color="c")
        )
        return chart + mk("point", _HSTACK_ROWS, x="z", y="t") if route == "+point" else chart

    shorthand = _quiet(area(stack="zero").to_svg)
    assert shorthand == _quiet(area(fm.Stack(value_axis="x")).to_svg)
    assert shorthand != _quiet(area().to_svg)
    assert _quiet(lambda: _scene(area(stack="zero"))) == _quiet(
        lambda: _scene(area(fm.Stack(value_axis="x")))
    )


def test_the_y_shorthand_and_the_flipped_shorthand_keep_the_default_stack():
    """``Y(stack=)``, flipped or not, draws the explicit ``Stack()`` it always drew."""
    vertical = mk("bar", _HSTACK_ROWS, x="g", y=fm.Y("z", stack="zero"), color="c")
    explicit = fm.Chart(_HSTACK_ROWS).mark_bar(position=fm.Stack()).encode(x="g", y="z", color="c")
    assert _quiet(vertical.to_svg) == _quiet(explicit.to_svg)
    flip = fm.CoordFlip()
    assert _quiet(vertical.coord(flip).to_svg) == _quiet(explicit.coord(flip).to_svg)


def _position_conflict(adjustment: str, channels: str) -> str:
    return (
        f"position=fm.{adjustment}(...) and stack= on {channels} are both set, but a mark draws "
        f"one position adjustment. Keep one: remove stack= from {channels}, or remove position=."
    )


def _positioned(position: Any, **encoding: Any) -> fm.Chart:
    return (
        fm.Chart(_HSTACK_ROWS)
        .mark_bar(position=position)
        .encode(**{"y": "g", "color": "c", **encoding})
    )


#: Door -> (render, the refusal it raises) for an explicit position beside a stack shorthand.
_POSITION_CONFLICT_DOORS: dict[str, tuple[Callable[[], Any], str]] = {
    "F": (
        lambda: _positioned(fm.Stack(value_axis="x"), x=fm.X("z", stack="zero")).to_svg(),
        f"mark_bar: {_position_conflict('Stack', 'x')}",
    ),
    "F-dodge": (
        lambda: _positioned(fm.Dodge(), x=fm.X("z", stack="zero")).to_svg(),
        f"mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
    "F-identity": (
        lambda: _positioned(fm.Identity(), x=fm.X("z", stack="zero")).to_svg(),
        f"mark_bar: {_position_conflict('Identity', 'x')}",
    ),
    "F-both": (
        lambda: _positioned(
            fm.Stack(), x=fm.X("z", stack="zero"), y=fm.Y("g", stack="zero")
        ).to_svg(),
        f"mark_bar: {_position_conflict('Stack', 'x and y')}",
    ),
    "flip": (
        lambda: (
            fm.Chart(_HSTACK_ROWS)
            .mark_bar(position=fm.Dodge())
            .encode(x="g", y=fm.Y("z", stack="zero"), color="c")
            .coord(fm.CoordFlip())
            .to_svg()
        ),
        f"mark_bar: {_position_conflict('Dodge', 'y')}",
    ),
    "Fc": (
        lambda: _positioned(fm.Dodge(), x=fm.X("z", stack="zero")).facet("f").to_svg(),
        f"mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
    "I": (
        lambda: _positioned(fm.Dodge(), x=fm.X("z", stack="zero")).interactive().to_html(),
        f"mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
    "+": (
        lambda: (_positioned(fm.Dodge(), x=fm.X("z", stack="zero")) + _hpoint()).to_svg(),
        f"layer 0: mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
    "+rev": (
        lambda: (_hpoint() + _positioned(fm.Dodge(), x=fm.X("z", stack="zero"))).to_svg(),
        f"layer 1: mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
    "LCs": (
        lambda: fm.LayerChart(
            _hpoint(), _positioned(fm.Dodge(), x=fm.X("z", stack="zero"))
        ).to_svg(),
        f"failed to render composite leaf leaf #1: layer 1: mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
    "LCx": (
        lambda: (
            fm.LayerChart(_hpoint(), _positioned(fm.Dodge(), x=fm.X("z", stack="zero")))
            .interactive()
            .to_html()
        ),
        f"layer 1: mark_bar: {_position_conflict('Dodge', 'x')}",
    ),
}


@pytest.mark.parametrize("door", sorted(_POSITION_CONFLICT_DOORS))
def test_an_explicit_position_beside_a_stack_shorthand_is_refused_on_every_door(door):
    """``position=`` and a ``stack=`` shorthand on one mark are refused, naming both and the remedy.

    Rejects the position silently winning, which drew the chart as though
    the shorthand were not written.
    """
    render, message = _POSITION_CONFLICT_DOORS[door]
    with pytest.raises(ValueError) as exc:
        _quiet(render)
    assert str(exc.value) == message


def _category_refusal(channel: str, value: str, field: str = "g") -> str:
    return (
        f"mark_bar: stack= is set on {channel}, whose field '{field}' is categorical, so there is no value "
        f"to stack along it. Put stack= on the value channel, {value}."
    )


#: Orientation -> (render, the refusal it raises) for a stack shorthand on the category channel.
_CATEGORY_STACK_DOORS: dict[str, tuple[Callable[[], Any], str]] = {
    "vertical": (
        lambda: mk("bar", _HSTACK_ROWS, x=fm.X("g", stack="zero"), y="z", color="c").to_svg(),
        _category_refusal("x", "y"),
    ),
    "horizontal": (
        lambda: mk("bar", _HSTACK_ROWS, x="z", y=fm.Y("g", stack="zero"), color="c").to_svg(),
        _category_refusal("y", "x"),
    ),
    "vertical-I": (
        lambda: (
            mk("bar", _HSTACK_ROWS, x=fm.X("g", stack="zero"), y="z", color="c")
            .interactive()
            .to_html()
        ),
        _category_refusal("x", "y"),
    ),
    "horizontal-flip": (
        lambda: (
            mk("bar", _HSTACK_ROWS, x="z", y=fm.Y("g", stack="zero"), color="c")
            .coord(fm.CoordFlip())
            .to_svg()
        ),
        _category_refusal("y", "x"),
    ),
    "nominal-number": (
        lambda: mk("bar", _HSTACK_ROWS, x=fm.X("t:N", stack="zero"), y="z", color="c").to_svg(),
        _category_refusal("x", "y", field="t"),
    ),
    "ordinal-number": (
        lambda: mk("bar", _HSTACK_ROWS, x=fm.X("t:O", stack="zero"), y="z", color="c").to_svg(),
        _category_refusal("x", "y", field="t"),
    ),
    "+": (
        lambda: (
            mk("point", _HSTACK_ROWS, x="g", y="z")
            + mk("bar", _HSTACK_ROWS, x=fm.X("g", stack="zero"), y="z")
        ).to_svg(),
        f"layer 1: {_category_refusal('x', 'y')}",
    ),
    "L": (
        lambda: (
            mk("point", _HSTACK_ROWS, x="g", y="z")
            .layer(fm.Layer(mark="bar", encoding={"x": fm.X("g", stack="zero"), "y": "z"}))
            .to_svg()
        ),
        f"layer 1: {_category_refusal('x', 'y')}",
    ),
}


@pytest.mark.parametrize("door", sorted(_CATEGORY_STACK_DOORS))
def test_a_stack_shorthand_on_the_category_channel_is_refused(door):
    """``stack=`` on a categorical channel names the channel, its field and the value channel.

    Rejects the stack's dtype error, which named no remedy.
    """
    render, message = _CATEGORY_STACK_DOORS[door]
    with pytest.raises(ValueError) as exc:
        _quiet(render)
    assert str(exc.value) == message


_STACK_ON_BOTH = (
    "stack= is set on both x and y, but a mark stacks along one channel: the value channel its "
    "segments run along. Keep stack= on x to stack them end to end horizontally, or on y to stack "
    "them vertically."
)


def _both_stacked() -> fm.Chart:
    return mk("bar", _HSTACK_ROWS, x=fm.X("z", stack="zero"), y=fm.Y("g", stack="zero"), color="c")


#: Door -> (render, the refusal it raises).
_STACK_ON_BOTH_DOORS: dict[str, tuple[Callable[[], Any], str]] = {
    "F": (lambda: _both_stacked().to_svg(), f"mark_bar: {_STACK_ON_BOTH}"),
    "Fc": (lambda: _both_stacked().facet("f").to_svg(), f"mark_bar: {_STACK_ON_BOTH}"),
    "flip": (lambda: _both_stacked().coord(fm.CoordFlip()).to_svg(), f"mark_bar: {_STACK_ON_BOTH}"),
    "I": (lambda: _both_stacked().interactive().to_html(), f"mark_bar: {_STACK_ON_BOTH}"),
    "+": (lambda: (_both_stacked() + _hpoint()).to_svg(), f"layer 0: mark_bar: {_STACK_ON_BOTH}"),
    "+rev": (
        lambda: (_hpoint() + _both_stacked()).to_svg(),
        f"layer 1: mark_bar: {_STACK_ON_BOTH}",
    ),
    "L": (
        lambda: (
            _hpoint()
            .layer(
                fm.Layer(
                    mark="bar",
                    encoding={"x": fm.X("z", stack="zero"), "y": fm.Y("g", stack="zero")},
                )
            )
            .to_svg()
        ),
        f"layer 1: mark_bar: {_STACK_ON_BOTH}",
    ),
    "LCs": (
        lambda: fm.LayerChart(_hpoint(), _both_stacked()).to_svg(),
        f"failed to render composite leaf leaf #1: layer 1: mark_bar: {_STACK_ON_BOTH}",
    ),
    "LCx": (
        lambda: fm.LayerChart(_hpoint(), _both_stacked()).interactive().to_html(),
        f"layer 1: mark_bar: {_STACK_ON_BOTH}",
    ),
    "+c": (
        lambda: (_both_stacked() + _hpoint()).facet("f").to_svg(),
        f"layer 0: mark_bar: {_STACK_ON_BOTH}",
    ),
    "LCs0": (
        lambda: fm.LayerChart(_both_stacked(), _hpoint()).to_svg(),
        f"failed to render composite leaf leaf #0: layer 0: mark_bar: {_STACK_ON_BOTH}",
    ),
    "LCx0": (
        lambda: fm.LayerChart(_both_stacked(), _hpoint()).interactive().to_html(),
        f"layer 0: mark_bar: {_STACK_ON_BOTH}",
    ),
    "IL": (
        lambda: (
            _hpoint()
            .layer(
                fm.Layer(
                    mark="bar",
                    encoding={"x": fm.X("z", stack="zero"), "y": fm.Y("g", stack="zero")},
                )
            )
            .interactive()
            .to_html()
        ),
        f"layer 1: mark_bar: {_STACK_ON_BOTH}",
    ),
}


@pytest.mark.parametrize("door", sorted(_STACK_ON_BOTH_DOORS))
def test_the_stack_shorthand_on_both_x_and_y_is_refused_on_every_door(door):
    """A shorthand on both channels is refused, naming both and the remedy.

    Rejects stacking silently along either channel.
    """
    render, message = _STACK_ON_BOTH_DOORS[door]
    with pytest.raises(ValueError) as exc:
        _quiet(render)
    assert str(exc.value) == message


# A + operand's position= adjusts that operand's own layers and no sibling's:
# the composition carries no position of its own. A layer attached to a host
# takes the host's position unless it writes its own stack= shorthand.

_PLACED_ROWS = pl.DataFrame(
    {
        "g": ["a", "a", "b", "b"],
        "c": ["p", "q", "p", "q"],
        "y": [1.0, 2.0, 3.0, 4.0],
        "f": ["r", "r", "s", "s"],
    }
)


def _placed_bars(position: Any) -> fm.Chart:
    return fm.Chart(_PLACED_ROWS).mark_bar(position=position).encode(x="g", y="y", color="c")


def _shorthand_bars(stack: Any = "zero") -> fm.Chart:
    return fm.Chart(_PLACED_ROWS).mark_bar().encode(x="g", y=fm.Y("y", stack=stack), color="c")


def _stacked_layer(stack: Any = "zero") -> fm.Layer:
    return fm.Layer(mark="bar", encoding={"y": fm.Y("y", stack=stack)})


def _placed_points(position: Any = None) -> fm.Chart:
    return fm.Chart(_PLACED_ROWS).mark_point(position=position).encode(x="g", y="y", color="c")


def _placed_box(position: Any = None) -> fm.Chart:
    return fm.Chart(_PLACED_ROWS).mark_boxplot(position=position).encode(x="g", y="y", color="c")


def _rect_columns(svg: str) -> list[tuple[float, float]]:
    """Every drawn rect's ``(x, width)``: a dodged bar and a full-band one differ here."""
    from tests._svg_rects import data_bar_rects, mark_layer

    return sorted(
        (round(float(r["x"]), 1), round(float(r["width"]), 1))
        for r in data_bar_rects(mark_layer(svg))
    )


def _scene_nodes(chart: Any) -> list[list[dict]]:
    """Each interactive mark batch's nodes: the geometry and paint, without the tooltips."""
    return [
        batch["nodes"]
        for panel in _quiet(lambda: _scene(chart))["panels"]
        for batch in panel["marks"]
    ]


def _circle_xs(svg: str) -> list[float]:
    from tests._svg_rects import mark_layer, svg_elements

    return sorted(
        round(float(c["cx"]), 1)
        for c in svg_elements(mark_layer(svg), "circle", lambda a: "cx" in a)
    )


#: Door -> how it composes the dodged bars with the stacked bars built by the argument.
_DODGE_BESIDE_STACK_DOORS: dict[str, Callable[[Callable[[], fm.Chart]], Any]] = {
    "+": lambda stacked: _placed_bars(fm.Dodge()) + stacked(),
    "+rev": lambda stacked: stacked() + _placed_bars(fm.Dodge()),
    "+c": lambda stacked: (_placed_bars(fm.Dodge()) + stacked()).facet("f"),
    "LCs": lambda stacked: fm.LayerChart(_placed_bars(fm.Dodge()), stacked()),
}


@pytest.mark.parametrize("door", sorted(_DODGE_BESIDE_STACK_DOORS))
def test_dodged_bars_beside_shorthand_stacked_bars_draw_each_as_written(door):
    """``dodge + stacked`` draws the dodge and the shorthand's stack, in either order, static and interactive.

    The oracle is the same door with the stacked operand written as the
    explicit ``position=fm.Stack()``, a position its own mark call writes,
    which no sibling's can reach. Rejects the composition handing operand
    0's Dodge to the stacked operand, which refused it for a position it
    never wrote, and the stacked bars drawn dodged.
    """
    compose = _DODGE_BESIDE_STACK_DOORS[door]
    shorthand = compose(_shorthand_bars)
    explicit = compose(lambda: _placed_bars(fm.Stack()))
    assert _quiet(shorthand.to_svg) == _quiet(explicit.to_svg)
    assert _quiet(lambda: _scene(shorthand)) == _quiet(lambda: _scene(explicit))
    assert _quiet(shorthand.to_svg) != _quiet(compose(lambda: _placed_bars(fm.Dodge())).to_svg)


def test_a_plus_composition_carries_no_position_of_its_own():
    """On the wire, ``dodge + stacked`` keeps Dodge on operand 0's layer and none at chart level.

    A single-mark operand's layer writes its own position, so it carries no
    ``inherited_position``; a composite operand's layers carry the operand's
    position marked inherited, and a ``SecondaryY`` on the composition
    inherits none. Rejects the composition keeping operand 0's position,
    which reached every sibling that wrote none.
    """
    wire = _wire(_placed_bars(fm.Dodge()) + _shorthand_bars())
    assert "position" not in wire
    assert [
        (l.get("position", {}).get("type"), "inherited_position" in l) for l in wire["layers"]
    ] == [
        ("dodge", False),
        (None, False),
    ]
    boxed = _wire(_placed_points() + _placed_box(fm.Dodge()))
    assert "position" not in boxed
    assert boxed["layers"][0].get("position") is None
    assert "inherited_position" not in boxed["layers"][0]
    box_layers = boxed["layers"][1:]
    assert box_layers and all(
        l["position"]["type"] == "dodge" and l["inherited_position"] is True for l in box_layers
    )
    secondary = _wire(_placed_bars(fm.Dodge()) + _placed_points() + fm.SecondaryY("y"))
    assert "position" not in secondary
    assert [l.get("position", {}).get("type") for l in secondary["layers"]] == ["dodge", None, None]


@pytest.mark.parametrize("order", ["box+point", "point+box"])
def test_a_dodged_boxplot_beside_points_draws_each_as_alone(order):
    """``box(Dodge) + point`` and ``point + box(Dodge)`` dodge the boxes and leave the points on their bands.

    Rejects the left box's Dodge reaching the points, and the right box's
    Dodge dropped with the right operand's chart.
    """
    box, points = _placed_box(fm.Dodge()), _placed_points()
    chart = box + points if order == "box+point" else points + box
    svg = _quiet(chart.to_svg)
    assert _rect_columns(svg) == _rect_columns(_quiet(box.to_svg))
    assert _rect_columns(svg) != _rect_columns(_quiet(_placed_box().to_svg))
    assert _circle_xs(svg) == _circle_xs(_quiet(points.to_svg))
    interactive = _quiet(lambda: _scene(chart))
    oracle = _placed_box(fm.Dodge()) + _placed_points(fm.Identity())
    if order == "point+box":
        oracle = _placed_points(fm.Identity()) + _placed_box(fm.Dodge())
    assert interactive == _quiet(lambda: _scene(oracle))


def test_an_unpositioned_label_beside_dodged_bars_sits_on_its_band():
    """A colored text operand with no position of its own sits at its band's center beside dodged bars.

    Rejects the bars' Dodge reaching the labels.
    """

    def labels(position: Any = None) -> fm.Chart:
        return (
            fm.Chart(_PLACED_ROWS)
            .mark_text(position=position)
            .encode(x="g", y="y", text="y", color="c")
        )

    chart = _placed_bars(fm.Dodge()) + labels()
    oracle = _placed_bars(fm.Dodge()) + labels(fm.Identity())
    assert _quiet(chart.to_svg) == _quiet(oracle.to_svg)
    assert _scene_nodes(chart) == _scene_nodes(oracle)
    dodged = _placed_bars(fm.Dodge()) + labels(fm.Dodge())
    assert _quiet(chart.to_svg) != _quiet(dodged.to_svg)


#: Door -> the dodged host with an attached shorthand-stacked bar, and the
#: same chart with the attached bar written as an explicitly stacked operand.
_ATTACHED_STACK_DOORS: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
    "L": (
        lambda: _placed_bars(fm.Dodge()).layer(_stacked_layer()),
        lambda: _placed_bars(fm.Dodge()) + _placed_bars(fm.Stack()),
    ),
    "+L": (
        lambda: _placed_points() + _placed_bars(fm.Dodge()).layer(_stacked_layer()),
        lambda: _placed_points() + _placed_bars(fm.Dodge()) + _placed_bars(fm.Stack()),
    ),
}


@pytest.mark.parametrize("door", sorted(_ATTACHED_STACK_DOORS))
def test_an_attached_layer_s_own_shorthand_wins_over_its_host_s_position(door):
    """An attached bar with ``Y(stack=)`` stacks under a dodged host, which stays dodged.

    The attached layer takes its host's Dodge only when it writes no stack
    of its own, also when the host is a ``+`` operand. Rejects the refusal
    blaming the attached layer for its host's Dodge, and the right
    operand's host losing its Dodge.
    """
    build, oracle = _ATTACHED_STACK_DOORS[door]
    assert _quiet(build().to_svg) == _quiet(oracle().to_svg)
    assert _scene_nodes(build()) == _scene_nodes(oracle())
    assert _rect_columns(_quiet(build().to_svg)) != _rect_columns(
        _quiet(_placed_bars(fm.Dodge()).layer(layer("bar")).to_svg)
    )


def test_identity_bars_beside_shorthand_stacked_bars_draw_the_stack():
    """``bar(position=Identity()) + Y(stack=)`` renders the shorthand's stack.

    Rejects operand 0's Identity reaching the stacked operand, which refused it.
    """
    chart = _placed_bars(fm.Identity()) + _shorthand_bars()
    explicit = _placed_bars(fm.Identity()) + _placed_bars(fm.Stack())
    assert _quiet(chart.to_svg) == _quiet(explicit.to_svg)
    assert _quiet(lambda: _scene(chart)) == _quiet(lambda: _scene(explicit))


def _declined_stack(channels: str) -> str:
    return (
        f"position=fm.Stack(...) and stack=False on {channels} are both set, but one stacks the mark "
        f"and the other declines it. Keep one: remove stack=False from {channels}, or remove position=."
    )


def _stacked_declining(**encoding: Any) -> fm.Chart:
    return (
        fm.Chart(_PLACED_ROWS)
        .mark_bar(position=fm.Stack())
        .encode(**{"x": "g", "y": fm.Y("y", stack=False), "color": "c", **encoding})
    )


#: Door -> (render, the refusal it raises) for position=Stack() beside stack=False.
_DECLINED_STACK_DOORS: dict[str, tuple[Callable[[], Any], str]] = {
    "F": (lambda: _stacked_declining().to_svg(), f"mark_bar: {_declined_stack('y')}"),
    "F-both": (
        lambda: _stacked_declining(x=fm.X("g", stack="none")).to_svg(),
        f"mark_bar: {_declined_stack('x and y')}",
    ),
    "flip": (
        lambda: _stacked_declining().coord(fm.CoordFlip()).to_svg(),
        f"mark_bar: {_declined_stack('y')}",
    ),
    "I": (
        lambda: _stacked_declining().interactive().to_html(),
        f"mark_bar: {_declined_stack('y')}",
    ),
    "+": (
        lambda: (_placed_points() + _stacked_declining()).to_svg(),
        f"layer 1: mark_bar: {_declined_stack('y')}",
    ),
    "LCs": (
        lambda: fm.LayerChart(_placed_points(), _stacked_declining()).to_svg(),
        f"failed to render composite leaf leaf #1: layer 1: mark_bar: {_declined_stack('y')}",
    ),
    "LCx": (
        lambda: fm.LayerChart(_placed_points(), _stacked_declining()).interactive().to_html(),
        f"layer 1: mark_bar: {_declined_stack('y')}",
    ),
}


@pytest.mark.parametrize("door", sorted(_DECLINED_STACK_DOORS))
def test_position_stack_beside_stack_false_is_refused_on_every_door(door):
    """``position=fm.Stack()`` and ``stack=False`` on one mark contradict, so they are refused.

    Rejects either silently winning.
    """
    render, message = _DECLINED_STACK_DOORS[door]
    with pytest.raises(ValueError) as exc:
        _quiet(render)
    assert str(exc.value) == message


@pytest.mark.parametrize("position", ["Dodge", "Identity"])
def test_stack_false_beside_a_position_that_does_not_stack_draws_that_position(position):
    """``stack=False`` agrees with a Dodge or Identity, so the chart draws the plain position."""
    declining = (
        fm.Chart(_PLACED_ROWS)
        .mark_bar(position=getattr(fm, position)())
        .encode(x="g", y=fm.Y("y", stack=False), color="c")
    )
    plain = _placed_bars(getattr(fm, position)())
    assert _quiet(declining.to_svg) == _quiet(plain.to_svg)
    assert _quiet(lambda: _scene(declining)) == _quiet(lambda: _scene(plain))


@pytest.mark.parametrize("door", ["L", "+"])
def test_stack_false_declines_a_stack_the_layer_did_not_write(door):
    """A bar writing ``Y(stack=False)`` draws unstacked beside a stacked host or a stacked operand.

    An attached layer's inherited Stack yields to its own ``stack=False``;
    a ``+`` operand never takes a sibling's. Rejects the refusal of a
    contradiction the layer never wrote.
    """
    if door == "L":
        chart = _placed_bars(fm.Stack()).layer(_stacked_layer(stack=False))
    else:
        chart = _placed_bars(fm.Stack()) + _shorthand_bars(stack=False)
    oracle = _placed_bars(fm.Stack()) + _placed_bars(fm.Identity())
    assert _quiet(chart.to_svg) == _quiet(oracle.to_svg)
    assert _scene_nodes(chart) == _scene_nodes(oracle)


def test_a_stack_offset_in_any_case_stacks():
    """``Y(stack="Zero")`` stacks as ``stack="zero"`` does.

    Rejects the offset reaching Rust in the caller's spelling, which it
    validated and then stacked nothing.
    """
    assert _quiet(_shorthand_bars("Zero").to_svg) == _quiet(_shorthand_bars("zero").to_svg)
    assert _quiet(_shorthand_bars("NORMALIZE").to_svg) == _quiet(
        _shorthand_bars("normalize").to_svg
    )
    assert _quiet(_shorthand_bars("Zero").to_svg) != _quiet(_shorthand_bars(None).to_svg)


def test_a_text_from_other_rows_adds_no_legend_entry_it_does_not_draw():
    """A text under a colored host draws no color, so its rows' values join no legend."""
    other = pl.DataFrame({"g": ["a", "b"], "c": ["zz", "zz"], "y": [5.0, 6.0]})
    chart = (
        fm.Chart(_DODGE_ROWS)
        .mark_point()
        .encode(x="g", y="y", color="c")
        .layer(fm.Layer(data=other, mark="text", encoding={"text": "y"}))
    )
    assert sorted(_quiet(lambda: legend_set(chart.to_svg()))["c"]) == ["p", "q"]


def _text_only(color_route: str) -> fm.Chart:
    def text(field: str) -> fm.Chart:
        return fm.Chart(_DODGE_ROWS).mark_text().encode(x="g", y="y", text=field)

    composed = (text("y") + text("c")).encode(color="c")
    if color_route == "nested":
        return composed + fm.Chart(_DODGE_ROWS).mark_rule().encode(y="y")
    return composed


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("color_route", ["+e", "nested"])
def test_a_color_only_texts_inherit_draws_no_legend_and_warns(color_route, kind):
    """A composition color no layer draws builds no scale or legend, and says so.

    Interactive output emits each render warning twice (a logged route-wide
    follow-up), so the distinct messages are compared.
    """
    chart = _text_only(color_route)
    render = chart.to_svg if kind == "static" else lambda: chart.interactive().to_html()
    assert set(_all_warnings(render)) == {
        "layer 0: mark_text: color 'c' has no effect: no layer's mark draws color (a text "
        "mark draws only a color it binds itself), so it gets no scale and no legend; it "
        "was ignored"
    }
    if kind == "static":
        assert legend_set(_quiet(chart.to_svg)) == {}


def test_naming_another_layers_statistic_is_refused_with_the_statistic_remedy():
    with pytest.raises(ValueError, match=r"statistic"):
        bar_mean().layer(layer("text", text="mean_y")).to_svg()
    with pytest.raises(ValueError, match=r"statistic the base chart's encoding computes"):
        bar_mean() + SY("mean_y")
    with pytest.raises(ValueError, match=r"statistic"):
        _hist().layer(layer("tick", y="count")).to_svg()
    with pytest.raises(ValueError, match=r"statistic the base chart's mark_histogram computes"):
        _hist() + SY("count")


def test_a_facet_on_a_right_operand_is_warned_and_ignored():
    with pytest.warns(UserWarning, match="secondary layer's theme/facet is ignored"):
        chart = P() + P(df2, y="w").facet("g")
    assert len(_clip_groups(_svg_root(chart.to_svg()))) == 1


def test_an_aggregate_on_a_facet_channel_is_refused():
    with pytest.raises(ValueError) as exc:
        fm.Facet("c", aggregate="mean")
    assert str(exc.value).startswith("Facet(aggregate='mean') is not supported")


def test_an_error_channel_on_encode_is_warned_and_omitted():
    with pytest.warns(UserWarning):
        svg = P(y_error="r").to_svg()
    assert tag_counts(svg) == {"circle": 8}


# ---------------------------------------------------------------------------
# A different-frame operand reads only its own rows
# ---------------------------------------------------------------------------
#
# A ``+`` operand or ``fm.Layer(data=...)`` over another frame reads its own
# rows: the columns the merge renamed are restored inside that operand, so
# no binding, groupby, legend, axis title or tooltip of any layer names an
# internal ``__rhs_`` column, and no layer groups by a column null on its rows.


def _null_group_warnings(build: Callable[[], Any]) -> list[str]:
    """The null-groupby-key warnings a chart raises rendering static and interactive."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg()
        chart.interactive().to_html()
    return [str(w.message) for w in caught if "null in its groupby key" in str(w.message)]


# df2 inside the host's panel ``p``: the facet field ``c`` names a column of
# both frames, which the merge renames on the data layer's rows.
_DF2_IN_P = df2.with_columns(c=pl.lit("p"))


def _f3(faceted: bool, data: pl.DataFrame = df2) -> fm.Chart:
    host = bar_mean().facet("c") if faceted else bar_mean()
    return host.layer(fm.Layer(data=data, mark="point"))


_DATA_LAYER_OWN_ROWS: dict[str, Case] = {
    "LD": Case(
        lambda: _f3(faceted=False),
        (bar_mean, lambda: agg("point", df2)),
        marks={"rect": 2, "circle": 3},
        chart_level=None,
    ),
    "LD-faceted": Case(
        lambda: _f3(faceted=True, data=_DF2_IN_P),
        marks={"rect": 4, "circle": 3},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_DATA_LAYER_OWN_ROWS))
def test_a_data_layer_inheriting_an_aggregate_draws_its_own_rows_means(case):
    """The data layer averages its own rows by ``g``: 4 bars and 3 points faceted, nothing warned.

    Rejects reading the combined rows: the faceted data layer grouped by
    the renamed ``g`` copy, null on every row of the host's frame, and each
    panel warned that every row had a null in its groupby key.
    """
    check_rendering(_DATA_LAYER_OWN_ROWS[case])
    assert _null_group_warnings(_DATA_LAYER_OWN_ROWS[case].build) == []


def test_the_faceted_data_layer_draws_its_means_in_its_own_panel():
    """Faceted: panel ``p`` draws the host's 2 bars and the data layer's 3 points; ``q`` the bars."""
    panels = panel_census(_quiet(lambda: _f3(faceted=True, data=_DF2_IN_P).to_svg()))
    kinds = [dict(Counter(signature[0] for signature in panel.elements())) for panel in panels]
    assert kinds == [{"rect": 2, "circle": 3}, {"rect": 2}]


def _kinds_per_panel(svg: str) -> list[dict[str, int]]:
    """Per plot clip group, the count of each mark element kind."""
    return [
        dict(Counter(signature[0] for signature in panel.elements())) for panel in panel_census(svg)
    ]


def _interactive_kinds(chart: Any) -> Counter:
    """The interactive scene's mark nodes per batch kind, over every panel."""
    return Counter(kind for kind, _ in _quiet(lambda: interactive_census(chart)).elements())


def _facet_warnings(build: Callable[[], Any]) -> list[str]:
    """The facet warnings a chart raises rendering static and interactive output."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg()
        chart.interactive().to_html()
    return [str(w.message) for w in caught if str(w.message).startswith("facet ")]


def test_a_data_layer_whose_frame_lacks_the_facet_field_repeats_in_every_panel():
    """Over df2, which has no ``c``: each panel draws 2 bars and 3 points, nothing warned.

    The data layer's ``mean(y)`` by ``g`` runs on its own 6 rows in every
    panel, as ggplot2 draws a layer whose data lacks the faceting variable.
    Rejects the silent drop: the 3 points were drawn in no panel.
    """
    assert _null_group_warnings(lambda: _f3(faceted=True)) == []
    assert _facet_warnings(lambda: _f3(faceted=True)) == []
    svg = _quiet(lambda: _f3(faceted=True).to_svg())
    assert _kinds_per_panel(svg) == [{"rect": 2, "circle": 3}, {"rect": 2, "circle": 3}]
    assert _interactive_kinds(_quiet(lambda: _f3(faceted=True))) == Counter({"bar": 4, "point": 6})


# Each faceted door over df and df2, which lacks ``c``: df2's rows repeat in
# every panel, df's partition by ``c``.
_FACET_REPEAT: dict[str, tuple[Callable[[], Any], list[dict[str, int]]]] = {
    "+/facet-after": (lambda: (P() + P(df2)).facet("c"), [{"circle": 10}, {"circle": 10}]),
    "+rev/facet-after": (lambda: (P(df2) + P()).facet("c"), [{"circle": 10}, {"circle": 10}]),
    "+/facet-before": (lambda: P().facet("c") + P(df2), [{"circle": 10}, {"circle": 10}]),
    "+/statistic": (
        lambda: (bar_mean() + agg("point", df2)).facet("c"),
        [{"rect": 2, "circle": 3}, {"rect": 2, "circle": 3}],
    ),
    "+rev/statistic": (
        lambda: (agg("point", df2) + bar_mean()).facet("c"),
        [{"circle": 3, "rect": 2}, {"circle": 3, "rect": 2}],
    ),
    # df2 has ``g`` but no ``c``: it partitions by column ``g`` and repeats down the rows.
    "grid/lacks-only-the-row-field": (
        lambda: (P() + P(df2)).facet(row="c", col="g"),
        [{"circle": n} for n in (4, 4, 2, 4, 4, 2)],
    ),
    "LCs": (lambda: fm.LayerChart(P().facet("c"), P(df2)), [{"circle": 10}, {"circle": 10}]),
    "LCm": (
        lambda: fm.LayerChart(P().facet("c"), P(df2))._build_merged(),
        [{"circle": 10}, {"circle": 10}],
    ),
}


@pytest.mark.parametrize("case", sorted(_FACET_REPEAT))
def test_a_frame_lacking_the_facet_field_draws_its_rows_in_every_panel(case):
    """A different-frame operand without the facet field repeats per panel on every door.

    Static and interactive draw the same marks, and no row-drop warning
    fires. Rejects the silent drop: df2's rows reached no panel, and the
    static overlay drew them in a stray third plot area.
    """
    build, panels = _FACET_REPEAT[case]
    assert _facet_warnings(build) == []
    assert _kinds_per_panel(_quiet(lambda: build().to_svg())) == panels
    expected = sum((Counter(panel) for panel in panels), Counter())
    names = {"circle": "point", "rect": "bar"}
    assert _interactive_kinds(_quiet(build)) == Counter(
        {names[kind]: n for kind, n in expected.items()}
    )


def test_the_static_overlay_of_a_faceted_member_draws_what_its_merged_chart_draws():
    """LCs = LCm = LCx: a LayerChart with a faceted member renders the merged chart's facet grid."""
    members = lambda: (P().facet("c"), P(df2))  # noqa: E731
    static = _quiet(lambda: fm.LayerChart(*members()).to_svg())
    merged = _quiet(lambda: fm.LayerChart(*members())._build_merged())
    assert static == _quiet(merged.to_svg)
    assert _quiet(lambda: fm.LayerChart(*members()).interactive().scene_json) == _quiet(
        lambda: merged.interactive().scene_json
    )


# A frame holding ``c`` with nulls: rows 7 and 8 of df, rows 2 and 4 of df2.
_DF_NULL_C = df.with_columns(c=pl.Series(["p", "p", "q", "q", "p", "p", None, None], dtype=pl.Utf8))
_DF2_NULL_C = df2.with_columns(c=pl.Series(["p", None, "q", None, "p", "q"], dtype=pl.Utf8))
_TWO_NULL_ROWS = (
    "facet 'c': 2 rows have a null 'c' and belong to no panel, so they were dropped; "
    "fill or filter the nulls in 'c'"
)

# Each door: the chart, its per-panel marks, and the one warning it raises.
_FACET_NULLS: dict[str, tuple[Callable[[], Any], list[dict[str, int]]]] = {
    "F": (lambda: P(_DF_NULL_C).facet("c"), [{"circle": 4}, {"circle": 2}]),
    "LD": (
        lambda: bar_mean().facet("c").layer(fm.Layer(data=_DF2_NULL_C, mark="point")),
        # The layer's mean(y) by g over each panel's rows: a, b in p; c twice in q.
        [{"rect": 2, "circle": 2}, {"rect": 2, "circle": 1}],
    ),
    "+": (lambda: (P() + P(_DF2_NULL_C)).facet("c"), [{"circle": 6}, {"circle": 6}]),
    "+rev": (lambda: (P(_DF2_NULL_C) + P()).facet("c"), [{"circle": 6}, {"circle": 6}]),
}


@pytest.mark.parametrize("case", sorted(_FACET_NULLS))
def test_a_null_facet_value_in_a_frame_with_the_field_is_dropped_with_a_warning(case):
    """A row whose own frame has ``c`` but whose value is null is dropped, and the count is named.

    Only rows of frames holding ``c`` count; a frame without ``c`` repeats.
    Rejects the silent drop (no warning) and a count of the padded rows of
    a frame lacking the field.
    """
    build, panels = _FACET_NULLS[case]
    assert _kinds_per_panel(_quiet(lambda: build().to_svg())) == panels
    assert set(_facet_warnings(build)) == {_TWO_NULL_ROWS}


# Each pair: a data layer, and the ``+`` it lowers to, binding on the right
# operand what the data layer inherits from its host.
_DATA_LAYER_PLUS_PAIRS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "bar-mean-point": (
        lambda: _f3(faceted=False),
        lambda: bar_mean() + fm.Chart(df2).mark_point(),
    ),
    "faceted-bar-mean-point": (
        lambda: _f3(faceted=True, data=_DF2_IN_P),
        lambda: bar_mean().facet("c") + fm.Chart(_DF2_IN_P).mark_point(),
    ),
    "point-line": (
        lambda: P().layer(fm.Layer(data=df2, mark="line")),
        lambda: P() + fm.Chart(df2).mark_line(),
    ),
    "colored-point-line": (
        lambda: P(color="g").layer(fm.Layer(data=df2, mark="line")),
        lambda: P(color="g") + line(df2, color="g"),
    ),
}


@pytest.mark.parametrize("case", sorted(_DATA_LAYER_PLUS_PAIRS))
def test_a_data_layer_draws_what_the_plus_it_lowers_to_draws(case):
    """``Layer(data=df2)`` over a different frame renders the bytes of its ``+``."""
    layered, plus = _DATA_LAYER_PLUS_PAIRS[case]
    assert _quiet(lambda: layered().to_svg()) == _quiet(lambda: plus().to_svg())


def _filtered(chart: fm.Chart) -> fm.Chart:
    return chart.transform(fm.transform_filter("datum.y > 1"))


# Operands over df and df2, which share v, y, z and g: each pair is composed
# in both orders, with and without a user chain on either side.
_OVERLAP_OPERANDS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "no-chain": (P, lambda: line(df2)),
    "chain-left": (lambda: _filtered(P()), lambda: line(df2)),
    "chain-right": (P, lambda: _filtered(line(df2))),
    "chain-both": (lambda: _filtered(P()), lambda: _filtered(line(df2))),
    "right-aggregates-by-g": (lambda: P(x="g"), lambda: agg("point", df2, color="g")),
    "right-sized-by-z": (P, lambda: P(df2, size="z")),
}


def _overlap_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for name, (a, b) in _OVERLAP_OPERANDS.items():
        cases[f"{name}/+"] = Case(lambda a=a, b=b: a() + b(), (a, b))
        cases[f"{name}/+rev"] = Case(lambda a=a, b=b: b() + a(), (b, a))
    cases["faceted/+"] = Case(
        lambda: (P() + line(df2)).facet("g"),
        marks={"circle": 8, "polyline": 3},
        chart_level=None,
    )
    cases["faceted/+rev"] = Case(
        lambda: (line(df2) + P()).facet("g"),
        marks={"circle": 8, "polyline": 3},
        chart_level=None,
    )
    cases["faceted-right-aggregates/+"] = Case(
        lambda: (P(x="g") + agg("point", df2)).facet("g"),
        marks={"circle": 11},
        chart_level=None,
    )
    cases["faceted-right-aggregates/+rev"] = Case(
        lambda: (agg("point", df2) + P(x="g")).facet("g"),
        marks={"circle": 11},
        chart_level=None,
    )
    return cases


_DIFFERENT_FRAME_OWN_ROWS = _overlap_cases()


def _wire_names(chart: Any) -> list[str]:
    """Every field and title a chart's wire binds, and every groupby column its steps name."""
    wire = _wire(chart)
    names: list[str] = []

    def bindings(encoding: dict) -> None:
        for value in encoding.values():
            for entry in value if isinstance(value, list) else [value]:
                if isinstance(entry, dict):
                    names.extend(str(entry[k]) for k in ("field", "title") if k in entry)

    def groupbys(node: Any) -> None:
        if isinstance(node, dict):
            names.extend(node.get("groupby") or [])
            for value in node.values():
                groupbys(value)
        elif isinstance(node, list):
            for value in node:
                groupbys(value)

    bindings(wire["encoding"])
    for layer_ in wire.get("layers") or []:
        bindings(layer_["encoding"])
    groupbys(wire["transforms"])
    return names


@pytest.mark.parametrize("case", sorted(_DIFFERENT_FRAME_OWN_ROWS))
def test_an_operand_sharing_column_names_reads_only_its_own_rows(case):
    """Each operand of ``P(df) + line(df2)`` draws its own rows alone, in both orders.

    No layer binding, groupby, legend, axis title or tooltip names a
    ``__rhs_`` column. Rejects reading the combined rows under renamed
    fields: the right layer bound ``v__rhs_0001``, and a faceted or
    aggregating right layer grouped by a column null on the left's rows.
    """
    built = _DIFFERENT_FRAME_OWN_ROWS[case]
    check_rendering(built)
    chart = _quiet(built.build)
    assert not [name for name in _wire_names(chart) if "__rhs_" in name]
    assert not [n for names in interactive_tooltip_fields(chart) for n in names if "__rhs_" in n]


# Channel -> a class binding z on it.
_RENAMED_CHANNELS: dict[str, type] = {
    "color": fm.Color,
    "size": fm.Size,
    "opacity": fm.Opacity,
    "fill_opacity": fm.FillOpacity,
    "stroke_opacity": fm.StrokeOpacity,
}


def _renamed_title_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for channel in _RENAMED_CHANNELS:
        bound = lambda channel=channel: P(df2, **{channel: "z"})  # noqa: E731
        legends = legend_set(_quiet(lambda: bound().to_svg()))
        cases[f"{channel}/+"] = Case(lambda bound=bound: P() + bound(), (P, bound), legends=legends)
        cases[f"{channel}/+rev"] = Case(
            lambda bound=bound: bound() + P(), (bound, P), legends=legends
        )
    return cases


_RENAMED_TITLES = _renamed_title_cases()


@pytest.mark.parametrize("case", sorted(_RENAMED_TITLES))
def test_a_right_binding_on_a_shared_column_name_is_titled_by_its_field(case):
    """``P(df) + P(df2, size="z")`` draws one legend titled ``z``, with no title on the wire.

    Rejects the rename title stamp: the right layer bound ``z__rhs_0001``
    with a stamped ``title="z"`` where its channel takes a title, and an
    opacity-family legend was titled ``z__rhs_0001`` where it takes none.
    """
    built = _RENAMED_TITLES[case]
    check_rendering(built)
    channel = case.split("/")[0]
    layers = _wire(_quiet(built.build))["layers"]
    (bound,) = [layer_["encoding"][channel] for layer_ in layers if channel in layer_["encoding"]]
    assert bound == {"field": "z"}


@pytest.mark.parametrize("order", ["+", "+rev"])
def test_an_explicit_tooltip_on_a_shared_column_name_shows_its_field(order):
    """An explicit ``Tooltip("z")`` on a right operand over df2 shows ``z``, as alone.

    Rejects reading the combined rows: the tooltip named ``z__rhs_0001``.
    """
    tipped = lambda: P(df2, tooltip=fm.Tooltip("z"))  # noqa: E731
    chart = _quiet(lambda: P() + tipped() if order == "+" else tipped() + P())
    (bound,) = [
        layer_["encoding"]["tooltip"]
        for layer_ in _wire(chart)["layers"]
        if "tooltip" in layer_["encoding"]
    ]
    assert bound == {"field": "z"}
    svg = _quiet(chart.to_svg)
    assert "__rhs_" not in svg
    assert "<title>z: 5</title>" in svg
    names = interactive_tooltip_fields(chart)
    assert (names[1] if order == "+" else names[0]) == interactive_tooltip_fields(tipped())[0]


def _tipped_layer() -> fm.Layer:
    return layer("point", y="r", tooltip=fm.Tooltip("z"))


# Each door draws a point layer of its own with ``Tooltip("z")``, last.
_OWN_TOOLTIP_DOORS: dict[str, Callable[[], fm.Chart]] = {
    "+s": lambda: P() + P(y="r", tooltip=fm.Tooltip("z")),
    "+d": lambda: P() + P(df2, tooltip=fm.Tooltip("z")),
    "L": lambda: P().layer(_tipped_layer()),
    "LD": lambda: P().layer(
        fm.Layer(data=df2, mark="point", encoding={"tooltip": fm.Tooltip("z")})
    ),
}


@pytest.mark.parametrize("door", sorted(_OWN_TOOLTIP_DOORS))
def test_a_layers_own_tooltip_shows_in_interactive_output_on_every_door(door):
    """A layer's explicit ``Tooltip("z")`` lists ``z`` on hover, as the flat chart does.

    Rejects completing the layer's tooltip pair from the chart-level auto
    fields: the layer listed ``v`` and ``y``.
    """
    assert interactive_tooltip_fields(_quiet(_OWN_TOOLTIP_DOORS[door]))[-1] == {"z"}


# df2 with no column name in common with df.
_DF4 = df2.rename({c: f"{c}4" for c in df2.columns})

# Charts the own-rows routing does not reach: same-frame ``+``, different
# frames sharing no column name, and every LayerChart route.
_OWN_ROWS_CONTROLS: dict[str, Callable[[], Any]] = {
    "+/no-shared-name": lambda: P(color="g") + mk("line", _DF4, x="v4", y="y4", color="g4"),
    "+/no-shared-name-rev": lambda: mk("line", _DF4, x="v4", y="y4", color="g4") + P(color="g"),
    "+/same-frame-mean": lambda: bar_mean() + agg("point", color=mean(fm.Color)),
    "LCs/shared-names": lambda: fm.LayerChart(P(color="g"), line(df2, color="g")),
    "LCm/shared-names": lambda: fm.LayerChart(P(color="g"), line(df2, color="g"))._build_merged(),
    "LCi/shared-names": lambda: fm.LayerChart(
        P(color="g"), line(df2, y="w", color="g"), resolve={"y": "independent"}
    ),
    "LCm/shared-names-mean": lambda: fm.LayerChart(
        bar_mean(), agg("point", df2, size=mean(fm.Size))
    )._build_merged(),
    "LCi/shared-names-mean": lambda: fm.LayerChart(
        bar_mean(), agg("point", df2, y="mean(w)"), resolve={"y": "independent"}
    ),
}

# The SVG bytes each control renders; routing every different-frame operand
# through its own rows leaves them unchanged.
_OWN_ROWS_CONTROL_DIGESTS = {
    "+/no-shared-name": "373d2a3b67c7526b251ffa8d91652fb0e29b9c27157c167a1a73bc54361e17a3",
    "+/no-shared-name-rev": "d745f517551755552412f8683ea65d44c7afcffd13cdb1389817bbb3f077c3c5",
    "+/same-frame-mean": "6195fa7a7201dcdd04850635d30432d855bb7c1cd2d7739cb52dd75b5e1d2520",
    "LCs/shared-names": "6b9a1a465974887656e16ac3c9502a757283bb971f363f8e19ffc692dbd4f5e1",
    "LCm/shared-names": "373d2a3b67c7526b251ffa8d91652fb0e29b9c27157c167a1a73bc54361e17a3",
    "LCi/shared-names": "fa3123bd3b4b02c0b2fa291a97e19f14b43087bdcfa873d12823737243a3b720",
    "LCm/shared-names-mean": "a348555fc978b946c57a01f30998b6f155b6179ff3ebbb192a4147f0b04eeea9",
    "LCi/shared-names-mean": "99c14bd0f45b1077763c25691cc254a931461cb29519b11b4bcca778e9a8274a",
}


@pytest.mark.parametrize("case", sorted(_OWN_ROWS_CONTROLS))
def test_an_own_rows_control_renders_its_pinned_bytes(case):
    import hashlib

    svg = _quiet(lambda: _OWN_ROWS_CONTROLS[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _OWN_ROWS_CONTROL_DIGESTS[case]


# ---------------------------------------------------------------------------
# A SecondaryY on a composition reads the left operand's frame
# ---------------------------------------------------------------------------

_SECONDARY_W_REFUSAL = (
    "SecondaryY: field 'w' is not a column of the base chart's data (SecondaryY draws "
    "from the same table as the base chart); available columns: "
    "['v', 'y', 'z', 'r', 'g', 'c', 'k']"
)


@pytest.mark.parametrize("left", [P, bar_mean], ids=["raw", "aggregating"])
def test_a_secondary_y_naming_a_right_only_column_is_refused_at_construction(left):
    """``(P(df) + P(df2, y="w")) + SY("w")`` raises the ``SecondaryY`` column message.

    Rejects reading the composition's combined frame, which holds the right
    operand's ``w``: the secondary layer drew nothing, silently.
    """
    composition = _quiet(lambda: left() + P(df2, y="w"))
    with pytest.raises(ValueError) as refused:
        composition + SY("w")
    assert str(refused.value) == _SECONDARY_W_REFUSAL


_W_ABSENT = "layer 2: encoding 'y' references unknown column 'w' (not present in that layer's data)"


def test_a_secondary_y_on_a_left_chain_lacking_the_column_is_refused_at_render():
    """A left operand with its own chain: the secondary layer's rows are the left's, which have no ``w``.

    Rejects keeping the other frame's padded ``w`` in the left's rows: the
    secondary axis drew nothing, silently.
    """
    chart = (_filtered(P()) + P(df2, y="w")) + SY("w")
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value).startswith(_W_ABSENT)


def test_a_layer_attached_to_a_composition_reads_only_the_left_operands_columns():
    """``(P(df) + P(df2)).layer(rule, y=mean(w))`` refuses: the composition's data is df's.

    Rejects reading the combined frame: the rule averaged nothing, silently.
    """
    chart = (P() + P(df2)).layer(fm.Layer(mark="rule", encoding={"y": fm.Y("w", aggregate="mean")}))
    with pytest.raises(ValueError) as refused:
        chart.to_spec()
    assert str(refused.value) == _W_ABSENT


def _polyline_vertices(svg: str) -> list[int]:
    """The vertex count of every polyline the chart draws."""
    return [
        len(e.get("points").split())
        for group in _clip_groups(_svg_root(svg))
        for e in group.iter()
        if _tag(e) == "polyline"
    ]


_SECONDARY_OPERAND_ROWS: dict[str, tuple[Case, list[int]]] = {
    "left-column": (
        Case(
            lambda: (P() + P(df2)) + SY("r"), marks={"circle": 14, "polyline": 1}, chart_level=None
        ),
        [8],
    ),
    "left-transform-creates-it": (
        Case(
            lambda: (
                (P().transform(fm.transform_calculate("w", "datum.r * 2")) + P(df2, y="w"))
                + SY("w")
            ),
            marks={"circle": 14, "polyline": 1},
            chart_level=None,
        ),
        [8],
    ),
    "left-is-the-w-frame": (
        Case(
            lambda: (P(df2) + P()) + SY("w"), marks={"circle": 14, "polyline": 1}, chart_level=None
        ),
        [6],
    ),
}


@pytest.mark.parametrize("case", sorted(_SECONDARY_OPERAND_ROWS))
def test_a_secondary_y_on_a_composition_draws_the_left_operands_rows(case):
    """The secondary line draws one vertex per row of the left operand's frame."""
    built, vertices = _SECONDARY_OPERAND_ROWS[case]
    check_rendering(built)
    assert _polyline_vertices(_quiet(lambda: built.build().to_svg())) == vertices


# ---------------------------------------------------------------------------
# Facet coalescing reads the merge's own copies, never a column's name
# ---------------------------------------------------------------------------

# A frame whose ``g`` has nulls beside a column named like a merge's copy of it.
_COPY_SHAPED = pl.DataFrame(
    {
        "v": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "g": ["a", "b", None, None],
        "g__rhs_0001": ["a", "a", "b", "b"],
    }
)


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: mk("point", frame, x="v", y="y").facet("g"),
        lambda frame: (mk("point", frame, x="v", y="y") + line(df3, x="v3", y="y3")).facet("g"),
    ],
    ids=["flat", "composed"],
)
def test_a_facet_never_fills_its_field_from_a_column_named_like_a_copy(build):
    """A row whose facet value is null stays out of every panel, whatever its other columns are named.

    The chart draws the panels it draws without the copy-named column.
    Rejects finding a merge's copies by the ``{field}__rhs_`` name: the two
    null-``g`` points were placed in a panel by the caller's own column.
    """
    plain = _COPY_SHAPED.drop("g__rhs_0001")
    expected = panel_census(_quiet(lambda: build(plain).to_svg()))
    assert panel_census(_quiet(lambda: build(_COPY_SHAPED).to_svg())) == expected


def test_a_facet_fills_its_field_from_every_copy_a_chain_of_merges_made():
    """A facet on ``g`` over ``P(df) + (line(df2) + tick(df3))`` draws every operand's rows.

    The right operand's own merge renamed df3's ``g``, and the outer merge
    renamed it again; both copies fill the facet field.
    """
    chart = (P() + (line(df2) + mk("tick", df3, x="v3", y="y3"))).facet("g")
    assert tag_counts(_quiet(chart.to_svg)) == {"circle": 8, "polyline": 3, "line": 6}


# ---------------------------------------------------------------------------
# On the static overlay: one legend per channel, as the merged chart draws
# ---------------------------------------------------------------------------

# Channel -> (bound field, the left operand's mark, the right operand's mark).
_OVERLAY_LEGEND_CHANNELS: dict[str, tuple[str, str, str]] = {
    "color": ("g", "point", "line"),
    "size": ("z", "point", "point"),
    "shape": ("g", "point", "point"),
    "opacity": ("z", "point", "point"),
    "stroke_dash": ("g", "line", "line"),
}


@pytest.mark.parametrize("channel", sorted(_OVERLAY_LEGEND_CHANNELS))
def test_the_static_overlay_draws_one_legend_over_both_members(channel):
    """On the static overlay, two members binding one channel over df and df2 draw one legend, the merged chart's.

    Rejects a legend per member: the title would be drawn twice and each
    legend would list one member's values.
    """
    field, left_mark, right_mark = _OVERLAY_LEGEND_CHANNELS[channel]

    def members() -> tuple[fm.Chart, fm.Chart]:
        return (
            mk(left_mark, x="v", y="y", **{channel: field}),
            mk(right_mark, df2, x="v", y="y", **{channel: field}),
        )

    static = _quiet(lambda: fm.LayerChart(*members()).to_svg())
    merged = _quiet(lambda: fm.LayerChart(*members())._build_merged().to_svg())
    assert legend_set(static) == legend_set(merged)
    assert list(legend_set(static)) == [field]
    assert _legend_title_count(static, field) == 1


# ---------------------------------------------------------------------------
# A layer takes an inherited positional channel only where its mark draws it
# ---------------------------------------------------------------------------
#
# A layer takes an inherited x, y, x2 or y2 only on a channel its mark draws
# (Rust's one per-mark table, ``ferrum._core.mark_inherited_positional``),
# and a drawn channel whose column the layer's rows lack is refused.


def test_a_drawn_inherited_channel_its_rows_lack_is_refused_on_a_data_layer():
    """A point data layer over df2 takes its host's ``x="r"``, which df2 lacks, and refuses.

    Rejects reading the padded column: the layer's points drew nothing.
    """
    chart = mk("point", x="r", y="y").layer(fm.Layer(data=df2, mark="point"))
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == (
        "layer 1: encoding 'x' references unknown column 'r' (not present in that layer's data)"
    )


def test_a_one_axis_count_rule_counts_every_row_beside_a_layer_with_its_own_x():
    """``P(x="z") + rule(y="count()")`` draws one rule at the total, 8, over the same frame.

    The y-only rule draws no x, so its count groups by nothing. Rejects
    the inherited x reaching the rule's grain: one rule per ``z`` value.
    """
    chart = P(x="z") + mk("rule", y="count()")
    rule = _wire(chart)["layers"][-1]
    assert "x" not in rule["encoding"]
    assert [s.get("groupby") for s in _steps_at_any_depth(chart) if s["type"] == "aggregate"] == [
        []
    ]
    assert tag_counts(_quiet(chart.to_svg)) == {"circle": 8, "line": 1}


@pytest.mark.parametrize(
    ("mark", "own", "offered", "taken"),
    [
        ("line", ["x", "y"], ["x2", "y2"], set()),
        ("segment", ["x", "y"], ["x2", "y2"], {"x2", "y2"}),
        ("tick", ["x"], ["y"], set()),
        ("rule", ["y"], ["x"], set()),
        ("point", [], ["x", "y"], {"x", "y"}),
    ],
)
def test_the_effective_encoding_takes_only_the_positional_channels_the_mark_draws(
    mark, own, offered, taken
):
    """``_effective_encoding`` reads the Rust table: no Python copy of which mark draws what."""
    from ferrum._layer import _Layer
    from ferrum._layer_transforms import _effective_encoding

    layer_ = _Layer(mark=mark, encoding={ch: ch for ch in own})
    parent = {ch: f"{ch}_field" for ch in offered}
    drawn = _effective_encoding(layer_, parent)
    assert {ch for ch in offered if ch in drawn} == taken


def _aggregate_groupbys(chart: Any) -> list[list[str]]:
    """The groupby of every aggregate step in a chart's wire, branches included, in order."""
    out: list[list[str]] = []

    def walk(steps: list) -> None:
        for step in steps:
            if step.get("type") == "aggregate":
                out.append(step.get("groupby") or [])
            walk(step.get("transforms") or [])

    walk(_wire(chart)["transforms"])
    return out


# Hosts writing a span channel a point does not draw.
_SPAN_HOSTS: dict[str, Callable[[], fm.Chart]] = {
    "bar-y2": lambda: mk("bar", x="g", y="y", y2="z"),
    "rect-x2": lambda: mk("rect", x="v", x2="r", y="y"),
}


def _mean_v_layer(mark: str | None, data: pl.DataFrame | None) -> fm.Layer:
    kwargs: dict[str, Any] = {} if mark is None else {"mark": mark}
    if data is not None:
        kwargs["data"] = data
    return fm.Layer(encoding={"y": fm.Y("v", aggregate="mean")}, **kwargs)


@pytest.mark.parametrize("data", [None, df2], ids=["chart-data", "data-source"])
@pytest.mark.parametrize("host", sorted(_SPAN_HOSTS))
def test_a_layer_with_no_mark_takes_what_a_point_takes(host, data):
    """``Layer()`` draws a point, so it takes no inherited span: it equals ``Layer(mark="point")``.

    Its mean groups by the host's x alone and it draws the same SVG.
    Rejects taking every offered channel: the mean grouped by the span
    field too (7 or 4 points instead of 2).
    """
    unmarked = _quiet(lambda: _SPAN_HOSTS[host]().layer(_mean_v_layer(None, data)))
    point = _quiet(lambda: _SPAN_HOSTS[host]().layer(_mean_v_layer("point", data)))
    assert _aggregate_groupbys(unmarked) == _aggregate_groupbys(point)
    assert _quiet(unmarked.to_svg) == _quiet(point.to_svg)


@pytest.mark.parametrize(
    "build",
    [
        lambda: mk("bar", x="v", x2="r", y="y") + mk("tick", x="v"),
        lambda: mk("bar", x="v", x2="r", y="y").layer(layer("tick", x="v")),
    ],
    ids=["+", "L"],
)
def test_a_tick_writing_x_beside_a_ranged_bar_takes_no_y_and_draws_a_rug(build):
    """``bar(x, x2, y) + tick(x)``: the tick takes no inherited y, so it draws as a rug alone does.

    A tick draws no span, so the bar's x2 does not make it a two-axis tick.
    Rejects counting an offered span the tick never draws: the tick took
    the bar's y.
    """
    chart = _quiet(build)
    tick = _wire(chart)["layers"][-1]
    assert set(tick["encoding"]) == {"x"}
    tick_marks = Counter({k: n for k, n in census(_quiet(chart.to_svg)).items() if k[0] == "line"})
    alone = Counter(
        {
            k: n
            for k, n in census(_quiet(lambda: mk("tick", x="v").to_svg())).items()
            if k[0] == "line"
        }
    )
    assert tick_marks == alone


@pytest.mark.parametrize(
    "build",
    [
        lambda: fm.LayerChart(P().facet("c"), P(df2)),
        lambda: fm.LayerChart(P().facet("c"), P(df2))._build_merged(),
        lambda: P().facet("c") + P(df2),
        lambda: (P() + P(df2)).facet("c"),
    ],
    ids=["LCs", "LCm", "+", "+-facet-after"],
)
def test_a_right_operand_with_no_facet_joins_the_left_facet_without_a_warning(build):
    """Nothing of a right operand without a facet is ignored, so nothing warns; its rows repeat."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg()
        chart.interactive().to_html()
    assert [str(w.message) for w in caught] == []


def test_a_right_operands_own_facet_is_still_warned_and_ignored():
    """``P + P(df2).facet("g")`` warns that the right operand's facet is ignored."""
    with pytest.warns(UserWarning, match="secondary layer's theme/facet is ignored"):
        (P() + P(df2).facet("g")).to_svg()


# ---------------------------------------------------------------------------
# Layer 0 is judged by the column gate like every later layer
# ---------------------------------------------------------------------------
#
# The first layer that draws a channel establishes its scale, axis and legend
# from its own rows, and no reader falls back to another layer's batch. So a
# layer 0 binding to a column only another operand's frame holds is refused in
# the gate's one message, as a later layer's is below, rather than drawing an
# axis or legend read from rows it does not draw.


def _layer_zero_absent(channel: str, field: str) -> str:
    return (
        f"layer 0: encoding '{channel}' references unknown column '{field}' "
        "(not present in that layer's data)"
    )


# Left operand (bound to a column only the right's frame holds), right
# operand, and the refused channel and field.
_LAYER_ZERO_OPERANDS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart], str, str]] = {
    "x-other-frame": (lambda: P(x="w"), lambda: P(df2), "x", "w"),
    "color-left": (lambda: P(color="w"), lambda: P(df2), "color", "w"),
    "color-left-over-df2": (lambda: P(df2, color="k"), P, "color", "k"),
    "size-left": (lambda: P(size="w"), lambda: P(df2), "size", "w"),
    "shape-left-over-df2": (lambda: P(df2, shape="k"), P, "shape", "k"),
}


def _layer_zero_refusals() -> dict[str, tuple[Callable[[], Any], tuple[str, ...], str]]:
    """Each operand pair on ``+``, LCm (both outputs) and LayerChart's interactive route."""
    cases: dict[str, tuple[Callable[[], Any], tuple[str, ...], str]] = {}
    both = ("static", "interactive")
    for name, (a, b, channel, field) in _LAYER_ZERO_OPERANDS.items():
        message = _layer_zero_absent(channel, field)
        cases[f"{name}/+"] = (lambda a=a, b=b: a() + b(), both, message)
        cases[f"{name}/LCm"] = (
            lambda a=a, b=b: fm.LayerChart(a(), b())._build_merged(),
            both,
            message,
        )
        cases[f"{name}/LCx"] = (lambda a=a, b=b: fm.LayerChart(a(), b()), ("interactive",), message)
    for channel in ("x", "color"):
        cases[f"{channel}-other-frame/LD"] = (
            lambda channel=channel: P(**{channel: "w"}).layer(fm.Layer(data=df2, mark="point")),
            both,
            _layer_zero_absent(channel, "w"),
        )
    return cases


_LAYER_ZERO_REFUSALS = _layer_zero_refusals()


@pytest.mark.parametrize(
    ("case", "kind"),
    [
        (case, kind)
        for case, (_, kinds, _) in sorted(_LAYER_ZERO_REFUSALS.items())
        for kind in kinds
    ],
)
def test_layer_zeros_binding_to_a_column_only_another_frame_holds_is_refused(case, kind):
    """Layer 0 is refused on the channel it draws from a column its own rows lack.

    Rejects reading the field outside layer 0's batch: ``P(x="w") + P(df2)``
    drew df2's 6 circles on a ``w`` axis, and ``P(color="w") + P(df2)``
    a ``w`` legend read from df2's rows, for points that draw neither.
    """
    build, _, message = _LAYER_ZERO_REFUSALS[case]
    with pytest.raises(ValueError) as refused:
        chart = build()
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == message


@pytest.mark.parametrize("case", sorted(_LAYER_ZERO_OPERANDS))
def test_the_static_overlay_refuses_the_member_with_its_leaf_message(case):
    """LCs renders each member as its own unlayered leaf, which keeps the flat chart's message."""
    a, b, _, field = _LAYER_ZERO_OPERANDS[case]
    with pytest.raises(ValueError) as refused:
        fm.LayerChart(a(), b()).to_svg()
    assert str(refused.value) == (
        f"failed to render composite leaf leaf #0: unknown column '{field}' referenced by an encoding"
    )


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("channel", ["x", "color", "size"])
def test_an_unlayered_charts_missing_column_keeps_its_message(channel, kind):
    """The control: a flat chart has no layer to name, so its refusal is unchanged."""
    with pytest.raises(ValueError) as refused:
        chart = P(**{channel: "w"})
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == "unknown column 'w' referenced by an encoding"


# ---------------------------------------------------------------------------
# The column gate covers every channel a layer's mark draws
# ---------------------------------------------------------------------------
#
# A layer reading rows the user supplied (a ``+`` operand's frame, a
# ``Layer(data=)`` frame) that names a column those rows lack is refused on
# every channel its mark draws, own, inherited or conditional, in the gate's
# one message. A binding on a channel the mark never draws is never judged.


def _absent(channel: str, field: str) -> str:
    return (
        f"layer 1: encoding '{channel}' references unknown column '{field}' "
        "(not present in that layer's data)"
    )


_PICK_C = fm.selection_point(fields=["c"], name="pick")


def _over_df2(host: fm.Chart) -> fm.Chart:
    """*host* with a line over ``df2``, which lacks ``c`` and ``k``."""
    return host.layer(fm.Layer(data=df2, mark="line"))


_COLUMN_GATE_REFUSALS: dict[str, tuple[Callable[[], Any], str]] = {
    "color-right": (lambda: P(df2) + P(color="w"), _absent("color", "w")),
    "color-other-frame": (lambda: P() + P(df2, color="k"), _absent("color", "k")),
    "host-color-over-a-data-layer": (lambda: _over_df2(P(color="c")), _absent("color", "c")),
    "host-field-conditional-over-a-data-layer": (
        lambda: _over_df2(
            P(color=_PICK_C.when(fm.Color("c")).otherwise(fm.value("#cccccc"))).add_selection(
                _PICK_C
            )
        ),
        _absent("color", "c"),
    ),
    **{
        f"own-{channel}": (
            (lambda channel=channel: P() + P(df2, **{channel: "k"})),
            _absent(channel, "k"),
        )
        for channel in ("tooltip", "shape", "stroke_dash", "key", "href", "description")
    },
    "own-text": (lambda: P() + mk("text", df2, x="v", y="y", text="k"), _absent("text", "k")),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_COLUMN_GATE_REFUSALS))
def test_a_drawn_binding_its_layers_rows_lack_is_refused(case, kind):
    """A later layer's drawn binding to a column its own rows lack names the layer and channel.

    The color cells reject reading the other frame's column (``P(df2) +
    P(color="w")`` drew 14 circles under a ``w`` legend); the host cells
    reject drawing the line uncolored under the host's legend; a host
    conditional is refused on its own channel, not on the selection's
    tooltip, because the selection field never reaches a layer whose rows
    lack it.
    """
    build, message = _COLUMN_GATE_REFUSALS[case]
    with pytest.raises(ValueError) as refused:
        chart = build()
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == message


_NON_POSITIONAL_COLUMN_REFUSALS: dict[str, tuple[Callable[[], Any], str]] = {
    "L": (
        lambda: P().layer(fm.Layer(mark="line", encoding={"color": "nope"})),
        _absent("color", "nope"),
    ),
    "+s": (lambda: P() + line(color="nope"), _absent("color", "nope")),
    "SY-host": (
        lambda: P().layer(fm.Layer(mark="line", encoding={"color": "nope"})) + SY("r"),
        _absent("color", "nope"),
    ),
    "SY-composed": (
        lambda: (P() + SY("r")).layer(fm.Layer(mark="line", encoding={"color": "nope"})),
        "layer 2: encoding 'color' references unknown column 'nope' "
        "(not present in that layer's data)",
    ),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_NON_POSITIONAL_COLUMN_REFUSALS))
def test_an_unknown_column_on_a_non_positional_channel_is_refused_with_the_plain_message(
    door, kind
):
    """A layer's color naming a column its rows lack refuses at render, plain message.

    Rejects drawing the layer uncolored under no legend.
    """
    build, message = _NON_POSITIONAL_COLUMN_REFUSALS[door]
    with pytest.raises(ValueError) as refused:
        chart = build()
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == message


# The SVG each rendering cell draws: the column gate covers every channel
# and refuses none of these cells.
_COLUMN_GATE_RENDER_DIGESTS = {
    "selection-only": "9d910acf614a7f9c5c5c000fc2d6d525a6a95d475dd1060ac1328bd928beb313",
    "value-conditional": "9d910acf614a7f9c5c5c000fc2d6d525a6a95d475dd1060ac1328bd928beb313",
    "selection-plus": "9d910acf614a7f9c5c5c000fc2d6d525a6a95d475dd1060ac1328bd928beb313",
    "undrawn-host-shape": "bbc7316b39bcbcb990493cf76ec22e5c99c53ec7f41c8decf05df5b6faeec9e8",
}

_COLUMN_GATE_RENDERS: dict[str, Callable[[], Any]] = {
    "selection-only": lambda: _over_df2(P().add_selection(_PICK_C)),
    "value-conditional": lambda: _over_df2(
        P(color=_PICK_C.when(fm.value("#d62728")).otherwise(fm.value("#cccccc"))).add_selection(
            _PICK_C
        )
    ),
    "selection-plus": lambda: P().add_selection(_PICK_C) + line(df2),
    "undrawn-host-shape": lambda: _over_df2(P(shape="c")),
}


@pytest.mark.parametrize("case", sorted(_COLUMN_GATE_RENDERS))
def test_a_layer_whose_rows_lack_only_an_unread_field_renders(case):
    """A selection field, or an undrawn binding, its rows lack never refuses a layer.

    A field selection's ``c`` joins the host's tooltip but not the df2
    line's, whose rows lack it, and a line never reads the host's shape. Each
    renders static and interactive, the SVG at its pinned bytes, and the
    line's interactive tooltip lists only its own drawn fields. Rejects carrying
    the selection field (or the undrawn shape's column) into the line's
    tooltip, which the gate refuses.
    """
    import hashlib

    svg = _COLUMN_GATE_RENDERS[case]().to_svg()
    assert hashlib.sha256(svg.encode()).hexdigest() == _COLUMN_GATE_RENDER_DIGESTS[case]
    host, data_line = interactive_tooltip_fields(_COLUMN_GATE_RENDERS[case]())
    assert data_line == {"v", "y"}
    assert "c" in host or case == "undrawn-host-shape"


def test_a_selection_field_is_never_written_into_a_tooltip_binding():
    """The spec names no selection field in any tooltip; the scene lists ``c`` on the host only.

    A field selection's fields are match keys the render's tooltip reader
    appends where the rows hold them, so the column gate never judges them.
    """
    chart = _COLUMN_GATE_RENDERS["selection-plus"]()
    wire = _wire(chart)
    assert "tooltip_fields" not in wire["encoding"]
    assert all("tooltip_fields" not in layer["encoding"] for layer in wire["layers"])
    host, data_line = interactive_tooltip_fields(chart)
    assert "c" in host and "c" not in data_line


def test_a_selection_field_reaches_the_frame_of_a_chain_of_merges_that_carries_it():
    """In ``(P + P(df2)) + P(df3-like)``, the selection's ``c`` stays off df2's layer only."""
    with_c = df2.with_columns(c=pl.Series(["p", "q"] * 3))
    chart = (P().add_selection(_PICK_C) + P(df2)) + P(with_c)
    assert [names for names in interactive_tooltip_fields(chart)] == [
        {"c", "v", "y"},
        {"v", "y"},
        {"c", "v", "y"},
    ]
    chart.to_svg()


def test_a_layer_keeps_the_selection_fields_its_rows_carry():
    """A selection on ``c`` and ``g``: the df2 line, which holds only ``g``, lists ``g`` beside its own."""
    both = fm.selection_point(fields=["c", "g"], name="both")
    chart = _over_df2(P().add_selection(both))
    assert interactive_tooltip_fields(chart) == [{"c", "g", "v", "y"}, {"g", "v", "y"}]


@pytest.mark.parametrize(
    ("build", "line_fields"),
    [
        (lambda: P().layer(layer("line", shape="c")), {"v", "y"}),
        (lambda: P() + line(shape="c"), {"v", "y"}),
        (lambda: P().layer(layer("line", shape="c", color="g")), {"g"}),
    ],
    ids=["L-own-shape", "+-own-shape", "L-own-shape-beside-own-color"],
)
def test_a_layers_own_undrawn_binding_stays_out_of_its_auto_tooltip(build, line_fields):
    """A line never draws ``shape``, so its own ``shape="c"`` is not listed in its interactive tooltip.

    Rejects listing ``c`` (the line lists ``{c}`` alone, or ``c`` beside its
    drawn fields): a field the mark never reads is not one it shows on hover.
    """
    assert interactive_tooltip_fields(_quiet(build))[1] == line_fields


# ---------------------------------------------------------------------------
# A field selection's fields are match keys, never bindings (GH #58)
# ---------------------------------------------------------------------------
#
# Each family below draws statistic sub-layers or layers over rows that lack
# the selection's ``c``. With the selection it renders in both kinds and
# raises exactly the warnings it raises without one; ``c`` sits in the
# scene tooltips of the batches whose rows carry it, and of no other.


def _sel_cat() -> pl.DataFrame:
    return _CAT_DATA.with_columns(c=pl.Series(["p", "q", "q", "p"] * 15))


# Per family: the chart, and the mark kinds of the batches whose rows carry ``c``.
_SELECTION_FAMILIES: dict[str, tuple[Callable[[], fm.Chart], list[str]]] = {
    "boxen": (lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z"), []),
    "boxplot": (lambda: fm.Chart(df).mark_boxplot().encode(x="g", y="z"), []),
    "violin": (lambda: fm.Chart(df).mark_violin().encode(x="g", y="z"), []),
    "mark_qq": (lambda: fm.Chart(df).mark_qq().encode(x="z"), []),
    "mark_qq-color": (lambda: fm.Chart(df).mark_qq().encode(x="z", color="c"), ["point", "rule"]),
    "catplot-point": (lambda: fm.catplot(_sel_cat(), x="cat", y="val", kind="point"), []),
    "catplot-boxen-hue": (
        lambda: fm.catplot(_sel_cat(), x="cat", y="val", kind="boxen", hue="grp"),
        [],
    ),
    "lmplot-color": (
        lambda: fm.lmplot(_sel_cat(), x="x", y="y", encode={"color": "c"}),
        ["line", "point", "ribbon"],
    ),
    "L-mean-point": (
        lambda: bar_mean().layer(layer("point", y=fm.Y("y", aggregate="mean"))),
        [],
    ),
    "plus-mean-point": (lambda: bar_mean() + agg("point"), []),
    "plus-filtered-operand": (
        lambda: P() + line(df2).transform(fm.transform_filter("datum.v > 0")),
        ["point"],
    ),
}


def _render_recording(chart: Any, kind: str) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = chart.to_svg() if kind == "static" else _scene(chart)
    return out, sorted({str(w.message) for w in caught})


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("family", sorted(_SELECTION_FAMILIES))
def test_a_field_selection_never_refuses_or_warns_on_a_layer_lacking_its_field(family, kind):
    """Rejects the selection's ``c`` reaching the column gate as a tooltip binding.

    Written into a tooltip, ``c`` refused a statistic sub-layer interactively
    (``layer 12: encoding 'tooltip' references unknown column 'c'``), warned
    ``tooltip 'c' was dropped`` per sub-layer statically, and refused a
    filtered operand's line in both kinds.
    """
    build, carrying = _SELECTION_FAMILIES[family]
    selected, warned = _render_recording(build().add_selection(_PICK_C), kind)
    _plain, warned_without = _render_recording(build(), kind)
    assert warned == warned_without
    if kind == "interactive":
        batches = [
            (batch["kind"], {f["name"] for t in batch.get("tooltips") or [] for f in t["fields"]})
            for panel in selected["panels"]
            for batch in panel["marks"]
        ]
        assert sorted(k for k, names in batches if "c" in names) == sorted(carrying)


def test_an_unlayered_chart_lists_its_auto_fields_then_the_selection_field():
    scene = _scene(P().add_selection(_PICK_C))
    (batch,) = [b for panel in scene["panels"] for b in panel["marks"]]
    assert [f["name"] for f in batch["tooltips"][0]["fields"]] == ["v", "y", "c"]


def test_a_layers_own_explicit_tooltip_lists_its_entries_then_the_selection_field():
    chart = P().add_selection(_PICK_C).layer(layer("line", tooltip=fm.Tooltip("v")))
    _host, line_batch = [b for panel in _scene(chart)["panels"] for b in panel["marks"]]
    assert [f["name"] for f in line_batch["tooltips"][0]["fields"]] == ["v", "c"]


@pytest.mark.parametrize("kind", ["static", "interactive"])
def test_a_user_written_tooltip_its_rows_lack_still_refuses_under_a_selection(kind):
    """The negative control: a tooltip the user wrote is a binding, and the gate refuses it."""
    chart = (
        P().add_selection(_PICK_C).layer(fm.Layer(data=df2, mark="line", encoding={"tooltip": "c"}))
    )
    with pytest.raises(ValueError) as refused:
        chart.to_svg() if kind == "static" else _scene(chart)
    assert str(refused.value) == _absent("tooltip", "c")


def test_a_parent_fallback_auto_tooltip_is_listed_as_inherited():
    """A layer binding no field lists its parent's drawn fields, marked inherited; an own list is not."""
    chart = P(shape="c").layer(fm.Layer(data=df2, mark="line"))
    spec, *_rest = chart._render_inputs(_auto_tooltips=True)
    host, data_line = json.loads(spec.to_json())["layers"]
    assert data_line["encoding"]["tooltip_fields"] == [{"field": "v"}, {"field": "y"}]
    assert data_line["inherited_channels"] == ["tooltip_fields"]
    assert host["encoding"]["tooltip_fields"] == [{"field": "v"}, {"field": "y"}, {"field": "c"}]
    assert "inherited_channels" not in host


_EXCLUDED_COLOR_SCALE_BUILDS: dict[str, Callable[[], Any]] = {
    "+": lambda: bar_mean(color=mean(fm.Color)) + mk("tick", df2, color="g"),
    "L": lambda: bar_mean(color=mean(fm.Color)).layer(
        fm.Layer(data=df2, mark="tick", encoding={"color": "g"})
    ),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_EXCLUDED_COLOR_SCALE_BUILDS))
def test_a_tick_left_out_of_a_numeric_color_scale_draws_with_one_warning(door, kind):
    """A string-colored tick beside a mean-colored bar stays out of the scale, as a point does.

    Rejects the tick's color reader refusing the column's dtype
    (``column 'g' has unsupported dtype: Utf8``).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = _EXCLUDED_COLOR_SCALE_BUILDS[door]()
        chart.to_svg() if kind == "static" else chart.interactive().scene_json
    excluded = [str(w.message) for w in caught if "stays out of that scale" in str(w.message)]
    assert excluded == [
        "layer 1: its color scale differs from the one layer 0 establishes the color scale "
        "with, so it stays out of that scale; its marks draw through that scale and may fall "
        "outside its legend"
    ]


# ---------------------------------------------------------------------------
# A statistic sub-layer drops an inherited channel its statistic does not
# carry, with a warning
# ---------------------------------------------------------------------------

_CAT_RNG = np.random.default_rng(0)
_CAT_DATA = pl.DataFrame(
    {
        "cat": ["a", "b", "c"] * 20,
        "grp": ["X"] * 30 + ["Y"] * 30,
        "val": _CAT_RNG.normal(0, 1, 60),
        "x": _CAT_RNG.normal(0, 1, 60),
        "y": _CAT_RNG.normal(0, 1, 60),
    }
)


def _dropped(entry: str, mark: str, channel: str, field: str) -> str:
    """The warning, voiced by the call that built the layer (its internal index is not named)."""
    return (
        f"{entry}: {channel} '{field}' was dropped: the statistic this {mark} "
        f"mark draws does not carry '{field}', so it draws without {channel}"
    )


_DROPPED_CHANNEL_CASES: dict[str, tuple[Callable[[], Any], list[str], str]] = {
    "catplot-point-tooltip": (
        lambda: fm.catplot(_CAT_DATA, x="cat", y="val", kind="point", tooltip="grp"),
        [
            _dropped("catplot", "rule", "tooltip", "grp"),
            _dropped("catplot", "point", "tooltip", "grp"),
        ],
        "d6002ccc9dd6882e00677cb808ef3fe957791e384381c9fab34961d2698a2dd6",
    ),
}


@pytest.mark.parametrize("case", sorted(_DROPPED_CHANNEL_CASES))
def test_a_statistic_layer_draws_without_an_inherited_channel_it_dropped(case):
    """Every statistic sub-layer, layer 0 included, warns and draws without the channel.

    Rejects refusing a call whose column is in the user's data (the
    statistic's output, not the user, dropped it), rejects dropping it
    silently, and rejects exempting layer 0 (catplot's error-bar rule) from
    the warning its later layers give.
    """
    import hashlib

    build, expected, digest = _DROPPED_CHANNEL_CASES[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
    assert [str(w.message) for w in caught if "was dropped" in str(w.message)] == expected
    assert hashlib.sha256(svg.encode()).hexdigest() == digest


def test_a_raw_rows_layer_still_refuses_an_inherited_column_its_rows_lack():
    """The control: a ``Layer(data=)`` line reads the user's rows, so the host's color refuses."""
    with pytest.raises(ValueError) as refused:
        _over_df2(P(color="c")).to_svg()
    assert str(refused.value) == _absent("color", "c")


@pytest.mark.parametrize("kind", ["static", "interactive"])
def test_a_column_only_another_operands_rows_hold_gets_the_plain_message(kind):
    """With a left chain: the right operand's own ``w`` is the user's rows, not a statistic.

    Rejects the statistic remedy, which told the caller to compute ``w`` in
    a transform when it is a column of the other operand's frame.
    """
    with pytest.raises(ValueError) as refused:
        chart = (_filtered(P()) + P(df2, y="w")) + SY("w")
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == _W_ABSENT


def test_a_statistic_another_layers_mark_computes_keeps_the_statistic_remedy():
    """The control: a histogram's ``count`` is a statistic output, so its remedy stays."""
    with pytest.raises(ValueError) as refused:
        _hist().layer(layer("tick", y="count")).to_svg()
    assert str(refused.value) == (
        "layer 1: encoding 'y' references unknown column 'count' (not present in that layer's "
        "data) — 'count' is a statistic another layer's mark computes for itself; compute it in "
        "the chart's .transform(...) (e.g. fm.Bin, fm.transform_aggregate) so every layer reads "
        "it, or draw it from its own chart composed with `+`"
    )


# A rule drawing only one axis beside a bar drawing both, per axis the rule
# leaves to the bar.
_ONE_AXIS_RULE_AND_BAR: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "x": (lambda: mk("rule", df2, y="w"), lambda: mk("bar", x="g", y="y")),
    "y": (lambda: mk("rule", df2, x="w"), lambda: mk("bar", y="g", x="y")),
}


@pytest.mark.parametrize("door", ["+", "LCm"])
@pytest.mark.parametrize("axis", sorted(_ONE_AXIS_RULE_AND_BAR))
def test_the_static_overlay_takes_an_axis_its_first_member_does_not_draw_from_the_drawing_member(
    axis, door
):
    """On the static overlay, the bar, the first member drawing the axis, establishes it.

    ``LayerChart(rule(df2, y="w"), bar(x="g"))`` draws x a, b titled ``g``
    and titles ``['w', 'g']``, as ``+`` and LCm do. Rejects taking the axis
    from the first member, which draws no x: it drew an untitled 0..1 axis
    describing no mark.
    """
    a, b = _ONE_AXIS_RULE_AND_BAR[axis]
    overlay = _quiet(lambda: fm.LayerChart(a(), b()).to_svg())
    other = {
        "+": lambda: a() + b(),
        "LCm": lambda: fm.LayerChart(a(), b())._build_merged(),
    }[door]
    svg = _quiet(lambda: other().to_svg())
    assert chrome_text(overlay) == chrome_text(svg)
    assert axis_titles(overlay) == axis_titles(svg) == (["w", "g"] if axis == "x" else ["g", "w"])
    assert tag_counts(overlay) == tag_counts(svg)
    if axis == "x":
        assert _x_tick_labels(overlay) == ["a", "b"]


def test_the_independent_y_layer_chart_draws_the_bars_x_axis_beside_a_rule_drawing_no_x():
    """LCi of the x cell: the shared x is the bar's a, b, titled ``g``; y gains its own axis."""
    a, b = _ONE_AXIS_RULE_AND_BAR["x"]
    svg = _quiet(lambda: fm.LayerChart(a(), b(), resolve={"y": "independent"}).to_svg())
    plus = _quiet(lambda: (a() + b()).to_svg())
    assert _x_tick_labels(svg) == ["a", "b"]
    assert axis_titles(svg) == ["w", "g", "y"]
    assert tag_counts(svg) == tag_counts(plus)


def test_the_static_overlay_keeps_refusing_a_category_on_the_axis_its_first_member_draws():
    """The rule draws x on ``w``, so it decides the continuous x; the bar's string ``g`` is refused."""
    with pytest.raises(ValueError) as refused:
        fm.LayerChart(mk("rule", df2, x="w"), mk("bar", x="g", y="y")).to_svg()
    assert str(refused.value) == (
        "composite scale resolution failed: failed to resolve shared x domain for field 'g': "
        "layer 1: column 'g' (Utf8) cannot be drawn on the shared x axis, which is continuous "
        "— a layered chart (fm.layer(...) / chart + chart) merges its members onto one panel. "
        'Cast the column, or give the layer its own scale with resolve={"x": "independent"}.'
    )


# The same pair under a caller's scale on the axis the first member does not
# draw: a log scale (no stated domain) and a stated domain.
_CALLER_SCALED_RULE_AND_POINT: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "x-log": (
        lambda: mk("rule", df2, y="w"),
        lambda: mk("point", x=fm.X("v", scale=fm.LogScale()), y="y"),
    ),
    "x-stated-domain": (
        lambda: mk("rule", df2, y="w"),
        lambda: mk("point", x=fm.X("v", scale=fm.LinearScale(domain=[0, 10])), y="y"),
    ),
    "y-log": (
        lambda: mk("rule", df2, x="w"),
        lambda: mk("point", x="v", y=fm.Y("y", scale=fm.LogScale())),
    ),
    "y-stated-domain": (
        lambda: mk("rule", df2, x="w"),
        lambda: mk("point", x="v", y=fm.Y("y", scale=fm.LinearScale(domain=[0, 10]))),
    ),
}


@pytest.mark.parametrize("door", ["+", "LCm"])
@pytest.mark.parametrize("case", sorted(_CALLER_SCALED_RULE_AND_POINT))
def test_the_static_overlay_takes_a_caller_scaled_axis_from_the_member_drawing_it(case, door):
    """On the static overlay under a caller's scale: the point, which draws the axis, establishes it.

    Rejects keeping the rule's untitled 0..1 axis because the caller's scale
    resolved the group on its own route: ``+`` and LCm draw the point's
    titled axis, log or stated domain.
    """
    a, b = _CALLER_SCALED_RULE_AND_POINT[case]
    overlay = _quiet(lambda: fm.LayerChart(a(), b()).to_svg())
    other = {
        "+": lambda: a() + b(),
        "LCm": lambda: fm.LayerChart(a(), b())._build_merged(),
    }[door]
    svg = _quiet(lambda: other().to_svg())
    assert chrome_text(overlay) == chrome_text(svg)
    assert axis_titles(overlay) == axis_titles(svg)
    assert axis_titles(overlay) == (["w", "v"] if case.startswith("x") else ["y", "w"])
    assert tag_counts(overlay) == tag_counts(svg)


# A rule drawing one axis beside a member whose own categorical scale draws
# the other, and the category order that member's rows give.
_CATEGORICAL_SCALED_RULE_AND_MARK: dict[
    str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart], list[str]]
] = {
    "x-band": (
        lambda: mk("rule", df2, y="w"),
        lambda: mk("bar", df2, x=fm.X("g", scale=fm.BandScale(padding_inner=0.2)), y="w"),
        ["a", "b", "c"],
    ),
    "x-point": (
        lambda: mk("rule", df2, y="w"),
        lambda: mk("point", df2, x=fm.X("g", scale=fm.PointScale()), y="w"),
        ["a", "b", "c"],
    ),
    "y-band": (
        lambda: mk("rule", df2, x="w"),
        lambda: mk("bar", df2, y=fm.Y("g", scale=fm.BandScale(padding_inner=0.2)), x="w"),
        ["a", "b", "c"],
    ),
    "x-band-sorted": (
        lambda: mk("rule", df2, y="w"),
        lambda: mk(
            "bar", df2, x=fm.X("g", scale=fm.BandScale(padding_inner=0.2), sort="-y"), y="w"
        ),
        ["c", "b", "a"],
    ),
}


@pytest.mark.parametrize("door", ["+", "LCm"])
@pytest.mark.parametrize("case", sorted(_CATEGORICAL_SCALED_RULE_AND_MARK))
def test_the_static_overlay_takes_a_categorical_axis_from_the_member_whose_scale_draws_it(
    case, door
):
    """On the static overlay under a band, point or ordinal scale: that member's categories are the axis.

    Equal to ``+`` and LCm on every tick, title and mark count, in the order
    the member's own sort gives over its own rows. Rejects keeping the
    rule's untitled 0..1 axis because a categorical scale takes neither the
    union nor the caller-scale route.
    """
    a, b, categories = _CATEGORICAL_SCALED_RULE_AND_MARK[case]
    overlay = _quiet(lambda: fm.LayerChart(a(), b()).to_svg())
    other = {
        "+": lambda: a() + b(),
        "LCm": lambda: fm.LayerChart(a(), b())._build_merged(),
    }[door]
    svg = _quiet(lambda: other().to_svg())
    assert chrome_text(overlay) == chrome_text(svg)
    assert axis_titles(overlay) == axis_titles(svg)
    assert tag_counts(overlay) == tag_counts(svg)
    labels = chrome_text(overlay)
    assert [label for label in labels if label in {"a", "b", "c"}] == categories


# A sibling's only y drawer marked independent, on both doors that mark it.
_LONE_INDEPENDENT_Y: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
    "LCi": (
        lambda: fm.LayerChart(
            mk("rule", df2, x="w"), mk("bar", y="g", x="y"), resolve={"y": "independent"}
        ),
        lambda: mk("rule", df2, x="w") + mk("bar", y="g", x="y"),
    ),
    "SY": (
        lambda: mk("rule", df2, x="w") + SY("z", mark="point"),
        lambda: mk("rule", df2, x="w") + mk("point", df2, x="w", y="z"),
    ),
}


@pytest.mark.parametrize("door", sorted(_LONE_INDEPENDENT_Y))
def test_the_only_y_drawer_owns_the_primary_axis_though_marked_independent(door):
    """With no shared layer drawing y, the independent layer establishes the left axis.

    The chart draws one y axis, on the left, as ``+`` does. Rejects an
    untitled 0..1 primary axis describing no mark beside a right-hand axis
    for the layer's own y.
    """
    independent, plus = _LONE_INDEPENDENT_Y[door]
    svg = _quiet(lambda: independent().to_svg())
    expected = _quiet(lambda: plus().to_svg())
    assert chrome_text(svg) == chrome_text(expected)
    assert axis_titles(svg) == axis_titles(expected)
    assert tag_counts(svg) == tag_counts(expected)


_SORT_ROWS = pl.DataFrame(
    {
        "g": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
        "y": [float(v) for v in (1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9)],
    }
)


def _SORTED_BAR() -> fm.Chart:
    """Bars over df2 sorted by their own ``w``: c, b, a."""
    return mk("bar", df2, x=fm.X("g", sort="-y"), y="w")


def _SORT_RULE() -> fm.Chart:
    """A rule over df, which holds ``g`` but orders it a, c, b by ``v``."""
    return mk("rule", y="v")


def _SORT_RULE_NO_G() -> fm.Chart:
    return mk("rule", df.drop("g"), y="v")


def _SORTED_BOX() -> fm.Chart:
    """Boxes sorted by their own raw ``y``: b, c, a."""
    return fm.Chart(_SORT_ROWS).mark_boxplot().encode(x=fm.X("g", sort="-y"), y="y")


# Each chart, and the category order its sorted layer's own rows give.
_SORT_CASES: dict[str, tuple[Callable[[], Any], list[str]]] = {
    "F/bar": (_SORTED_BAR, ["c", "b", "a"]),
    "+/rule-over-df-then-bar": (lambda: _SORT_RULE() + _SORTED_BAR(), ["c", "b", "a"]),
    "+/rule-without-g-then-bar": (lambda: _SORT_RULE_NO_G() + _SORTED_BAR(), ["c", "b", "a"]),
    "+rev/bar-then-rule": (lambda: _SORTED_BAR() + _SORT_RULE(), ["c", "b", "a"]),
    "LCs/bar-then-rule": (lambda: fm.LayerChart(_SORTED_BAR(), _SORT_RULE()), ["c", "b", "a"]),
    "LCs/rule-then-bar": (lambda: fm.LayerChart(_SORT_RULE(), _SORTED_BAR()), ["c", "b", "a"]),
    "LCm/rule-then-bar": (
        lambda: fm.LayerChart(_SORT_RULE(), _SORTED_BAR())._build_merged(),
        ["c", "b", "a"],
    ),
    "Fc/bar-shared-x": (
        lambda: mk(
            "bar",
            df2.with_columns(f=pl.Series(["m", "n"] * 3)),
            x=fm.X("g", sort="-y"),
            y="w",
        ).facet("f"),
        ["c", "b", "a"],
    ),
    "F/boxplot": (_SORTED_BOX, ["b", "c", "a"]),
    "+/boxplot-then-rule": (lambda: _SORTED_BOX() + _SORT_RULE(), ["b", "c", "a"]),
    "+/rule-then-boxplot": (lambda: _SORT_RULE() + _SORTED_BOX(), ["b", "c", "a"]),
    "LCs/rule-then-boxplot": (lambda: fm.LayerChart(_SORT_RULE(), _SORTED_BOX()), ["b", "c", "a"]),
}


@pytest.mark.parametrize("case", sorted(_SORT_CASES))
def test_a_data_aware_sort_reads_the_establishing_layers_own_rows(case):
    """``sort="-y"`` orders the categories by the sorted layer's own rows and its own y.

    The bar sorts by df2's ``w`` (c, b, a) and the boxplot by its raw ``y``
    (b, c, a) whatever stands beside them. Rejects sorting by another
    operand's column (a rule over df ordered the bar a, c, b), leaving the
    sort unapplied when the other frame lacks the category (a, b, c), and
    the static overlay dropping a member's sort (a, b, c).
    """
    build, order = _SORT_CASES[case]
    assert _x_tick_labels(_quiet(lambda: build().to_svg())) == order


#: g's groups differ in size (a: 6, b: 2, c: 4), so mean, sum and count each
#: give a different order: mean b(8) > c(5) > a(3); sum c(20) > a(18) > b(16);
#: count a(6) > c(4) > b(2). ``f`` alternates evenly within every group, so a
#: facet panel's per-group order matches the chart-wide one.
_AGG_SORT_ROWS = pl.DataFrame(
    {
        "g": ["a"] * 6 + ["b"] * 2 + ["c"] * 4,
        "y": [3.0] * 6 + [8.0] * 2 + [5.0] * 4,
        "f": ["m", "n"] * 6,
    }
)


def _agg_bar(sort: Any, rows: pl.DataFrame = _AGG_SORT_ROWS) -> fm.Chart:
    """A bar whose x sorts by the chart's own ``mean(y)``, not a raw column."""
    return fm.Chart(rows).mark_bar().encode(x=fm.X("g", sort=sort), y="mean(y)")


def _agg_rule(rows: pl.DataFrame = _AGG_SORT_ROWS) -> fm.Chart:
    """A sibling that draws no x, so it reads the bar's established order."""
    return fm.Chart(rows).mark_rule().encode(y="y")


#: Each door, building the chart for a given sort spec.
_AGG_SORT_DOORS: dict[str, Callable[[Any], fm.Chart]] = {
    "F": lambda sort: _agg_bar(sort),
    "+": lambda sort: _agg_bar(sort) + _agg_rule(),
    "L": lambda sort: _agg_bar(sort).layer(fm.Layer(mark="rule", encoding={"y": "y"})),
    "LCs": lambda sort: fm.LayerChart(_agg_bar(sort), _agg_rule()),
    "LCm": lambda sort: fm.LayerChart(_agg_bar(sort), _agg_rule())._build_merged(),
    "Fc": lambda sort: _agg_bar(sort).facet("f"),
    "Fc-layered": lambda sort: (_agg_bar(sort) + _agg_rule()).facet("f"),
}

#: Each sort form, and the category order it gives over the chart's own aggregate.
#: The channel shorthand (``"-y"``/``"y"``) reads the chart's own drawn
#: aggregate (``mean``); a sort-field object naming the raw column defaults
#: to ``sum`` (Vega-Lite's ``EncodingSortField`` default), independent of
#: the chart's own aggregate, unless it names its own ``op``.
_AGG_SORT_SPECS: dict[str, tuple[Any, list[str]]] = {
    "descending": ("-y", ["b", "c", "a"]),
    "ascending": ("y", ["a", "c", "b"]),
    "named-field": ({"field": "mean_y", "order": "descending"}, ["b", "c", "a"]),
    "raw-field-mean": ({"field": "y", "op": "mean", "order": "descending"}, ["b", "c", "a"]),
    "raw-field-default-op": ({"field": "y", "order": "descending"}, ["c", "a", "b"]),
    "raw-field-sum": ({"field": "y", "op": "sum", "order": "descending"}, ["c", "a", "b"]),
    "raw-field-count": ({"field": "y", "op": "count", "order": "descending"}, ["a", "c", "b"]),
}


def _interactive_x_tick_labels(chart: Any) -> list[str]:
    """The flat (or first facet panel's) bottom x axis tick labels, off the interactive scene."""
    panel = _scene(chart)["panels"][0]
    ticks = [
        n
        for n in panel["axes"]
        if n["type"] == "text"
        and n["style"].get("anchor") == "middle"
        and n["style"].get("angle", 0.0) == 0.0
        and n["style"].get("font_weight") == "normal"
    ]
    return [t["content"] for t in ticks]


@pytest.mark.parametrize("door", sorted(_AGG_SORT_DOORS))
@pytest.mark.parametrize("variant", sorted(_AGG_SORT_SPECS))
def test_a_data_aware_sort_reads_the_aggregate_the_establishing_layer_draws(variant, door):
    """``X("g", sort=...)`` beside ``y="mean(y)"`` sorts by the chart's own aggregate.

    Covers the channel shorthand (descending and ascending, which reads the
    chart's own drawn aggregate), a sort-field object naming the
    aggregate's own output column, and a sort-field object naming the raw
    field with its own op (mean, the default sum, and count, each giving a
    different order), on flat, ``+``, ``Chart.layer``, static and
    interactive ``LayerChart``, and flat and layered facet, with no
    fallback warning raised on any of them. Rejects a sort that reads only
    the pre-statistic rows, which lack the aggregate's own output column
    and have no per-field rule for a raw-field sort beside an aggregate,
    falling back to insertion order or miscounting a grouped op.
    """
    sort, order = _AGG_SORT_SPECS[variant]
    build = _AGG_SORT_DOORS[door]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build(sort)
        svg = chart.to_svg()
    assert caught == []
    assert _x_tick_labels(svg) == order
    with warnings.catch_warnings(record=True) as caught_interactive:
        warnings.simplefilter("always")
        interactive_order = _interactive_x_tick_labels(chart)
    assert caught_interactive == []
    assert interactive_order == order


#: A legend sort naming the raw field with its own ``op``, independent of
#: the chart's own ``mean(y)``.
_LEGEND_SORT = {"field": "y", "op": "mean", "order": "descending"}

#: Each (channel, mark) pair and door, encoding the legend sort beside the
#: chart's own aggregate: a colored bar, a shaped point, each beside a rule
#: sharing its rows for the ``+`` door.
_LEGEND_SORT_CASES: dict[tuple[str, str], Callable[[], fm.Chart]] = {
    ("color", "F"): lambda: mk(
        "bar", _AGG_SORT_ROWS, x="g", y="mean(y)", color=fm.Color("g", sort=_LEGEND_SORT)
    ),
    ("color", "+"): lambda: (
        mk("bar", _AGG_SORT_ROWS, x="g", y="mean(y)", color=fm.Color("g", sort=_LEGEND_SORT))
        + mk("rule", _AGG_SORT_ROWS, y="y")
    ),
    ("shape", "F"): lambda: mk(
        "point", _AGG_SORT_ROWS, x="g", y="mean(y)", shape=fm.Shape("g", sort=_LEGEND_SORT)
    ),
    ("shape", "+"): lambda: (
        mk("point", _AGG_SORT_ROWS, x="g", y="mean(y)", shape=fm.Shape("g", sort=_LEGEND_SORT))
        + mk("rule", _AGG_SORT_ROWS, y="y")
    ),
}


@pytest.mark.parametrize("channel,door", sorted(_LEGEND_SORT_CASES))
def test_a_legend_sort_reads_the_raw_field_beside_an_aggregate(channel, door):
    """A legend's ``sort={'field': ..., 'op': 'mean', ...}`` reads the raw rows, not the drawn aggregate.

    The chart's own y is ``mean(y)`` but the legend sort names the same raw
    field with its own op, giving the group order by that op's result
    (b, c, a by mean) rather than falling back to insertion order (a, b, c).
    """
    build = _LEGEND_SORT_CASES[(channel, door)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        svg = chart.to_svg()
    assert caught == []
    assert legend_set(svg)["g"] == ("b", "c", "a")
    with warnings.catch_warnings(record=True) as caught_interactive:
        warnings.simplefilter("always")
        interactive_legends = interactive_legend_set(chart)
    assert caught_interactive == []
    assert interactive_legends["g"] == ("b", "c", "a")


def test_a_layer_charts_reactive_name_collision_is_refused_alike_on_every_route():
    """A name that is a variable parameter on one member and a selection on another.

    LCs refuses as ``+`` and LCm do, in the merge's words. Rejects a second
    text for the same collision on the static overlay.
    """
    message = (
        "Reactive-parameter name collision (layer merge): 's' is a Selection on one side "
        "and a VariableParameter on the other. A name must resolve to a single "
        "reactive-object kind. Rename one of them."
    )
    a = lambda: P().add_params(fm.param(name="s", value=1))  # noqa: E731
    b = lambda: line().add_selection(fm.selection_point(name="s"))  # noqa: E731
    for build in (
        lambda: a() + b(),
        lambda: fm.LayerChart(a(), b())._build_merged(),
        lambda: fm.LayerChart(a(), b()),
    ):
        with pytest.raises(ValueError) as refused:
            build().to_svg()
        assert str(refused.value) == message


def _collision(context: str, first_kind: str) -> str:
    other = "VariableParameter" if first_kind == "Selection" else "Selection"
    return (
        f"Reactive-parameter name collision ({context}): 's' is a {first_kind} on one side "
        f"and a {other} on the other. A name must resolve to a single reactive-object kind. "
        "Rename one of them."
    )


def _own_collision() -> fm.Chart:
    """A chart declaring selection ``s`` and variable parameter ``s`` itself."""
    return P().add_selection(fm.selection_point(name="s")).add_params(fm.param(name="s", value=1))


@pytest.mark.parametrize(
    "door",
    ["F", "+left", "+right", "LCs", "LCm", "LCx"],
)
def test_a_charts_own_reactive_name_collision_is_refused_as_its_param_collection(door):
    """One chart's own selection and parameter sharing a name refuse at its spec build.

    On every door the operand carrying both is judged alone, as the chart
    alone is. Rejects refusing a right operand's or a later member's own
    collision as a merge between operands it never was.
    """
    build = {
        "F": _own_collision,
        "+left": lambda: _own_collision() + line(),
        "+right": lambda: line() + _own_collision(),
        "LCs": lambda: fm.LayerChart(line(), _own_collision()),
        "LCm": lambda: fm.LayerChart(line(), _own_collision())._build_merged(),
        "LCx": lambda: fm.LayerChart(line(), _own_collision()),
    }[door]
    with pytest.raises(ValueError) as refused:
        chart = build()
        chart.interactive().to_html() if door == "LCx" else chart.to_svg()
    assert str(refused.value) == _collision("param collection", "Selection")


def _declares_selection() -> fm.Chart:
    return P().add_selection(fm.selection_point(name="s"))


def _declares_parameter() -> fm.Chart:
    return line().add_params(fm.param(name="s", value=1))


@pytest.mark.parametrize("door", ["+", "LCs", "LCm"])
@pytest.mark.parametrize(
    ("order", "later_kind"),
    [("selection-first", "VariableParameter"), ("parameter-first", "Selection")],
)
def test_a_reactive_name_across_operands_is_refused_as_a_merge(order, later_kind, door):
    """A selection on one operand and a parameter of its name on another, in either order.

    The later operand is checked against the earlier ones, in the merge's
    words, on ``+``, LCs and LCm alike.
    """
    first, second = (
        (_declares_selection, _declares_parameter)
        if order == "selection-first"
        else (_declares_parameter, _declares_selection)
    )
    build = {
        "+": lambda: first() + second(),
        "LCs": lambda: fm.LayerChart(first(), second()),
        "LCm": lambda: fm.LayerChart(first(), second())._build_merged(),
    }[door]
    with pytest.raises(ValueError) as refused:
        build().to_svg()
    assert str(refused.value) == _collision("layer merge", later_kind)


# ---------------------------------------------------------------------------
# Each channel is established by the first layer that draws it, from the
# rows that layer draws
# ---------------------------------------------------------------------------

_RASTER_RNG = np.random.default_rng(3)
_RASTER_ROWS = pl.DataFrame(
    {
        "x": _RASTER_RNG.normal(0, 1, 400),
        "y": _RASTER_RNG.normal(0, 1, 400),
        "f": ["m", "n"] * 200,
    }
)


def _raster(**encoding: Any) -> fm.Chart:
    return fm.Chart(_RASTER_ROWS).mark_raster().encode(x="x", y="y", **encoding)


# Charts whose establishing layer draws every bound field from its own rows:
# a raster image draws x/y through its extent columns, an auxiliary channel is
# established by the first layer whose mark draws it (catplot's box draws
# size only on its outlier points), and an overlay whose first member draws
# both axes establishes them.
_ESTABLISHING_CONTROLS: dict[str, Callable[[], Any]] = {
    "F/raster": lambda: _raster(),
    "F/auto-raster": lambda: (
        fm.Chart(_RASTER_ROWS)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .properties(render_config=fm.RenderConfig(raster_threshold=0, raster_behavior="silent"))
    ),
    "Fc/raster": lambda: _raster().facet("f"),
    "F/catplot-box-size": lambda: fm.catplot(_CAT_DATA, x="cat", y="val", kind="box", size="val"),
    "F/contour-utf8-groupby": lambda: fm.Chart(df).mark_contour(groupby="g").encode(x="v", y="y"),
    "LCs/first-member-draws-both-axes": lambda: fm.LayerChart(bar_mean(), mk("rule", df2, y="w")),
}

# The SVG each control renders: its establishing layer draws every bound
# field from its own rows. A raster's cells map through a color scale over
# [0, max] of their non-negative values.
_ESTABLISHING_CONTROL_DIGESTS = {
    "F/raster": "f73507afe152bfdb98c902da73ac6b00aa136a7c4cf55b87f5118bea812a0034",
    "F/auto-raster": "8412beedc32d0d99c327e9a03d6e431a0bede1ba05906be168b21a25d591b6a8",
    "Fc/raster": "bf3950c5b0c510c151ca5793f67753c72d0177f841ed670bcb59cb023cb95846",
    "F/catplot-box-size": "cabbacf2dda7460c3a41d28940deac87f5a2ca9b86d093b4eb0bd14c76677cee",
    "F/contour-utf8-groupby": "d3ae503b8103f09448afe3cf190db052407db636f11db1bc55072f8e18de71dd",
    "LCs/first-member-draws-both-axes": (
        "828817e8a5935759f3bda63e6bc3035e50b154a57aa7b1ebde496265ee71f46e"
    ),
}


@pytest.mark.parametrize("case", sorted(_ESTABLISHING_CONTROLS))
def test_a_chart_whose_establishing_layer_draws_its_own_rows_renders_its_pinned_bytes(case):
    import hashlib

    svg = _quiet(lambda: _ESTABLISHING_CONTROLS[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _ESTABLISHING_CONTROL_DIGESTS[case]


def _axis_chrome(svg: str) -> list[str]:
    """The chart's chrome text without its axis titles: the tick labels, in document order."""
    titles = set(axis_titles(svg))
    return [text for text in chrome_text(svg) if text not in titles]


def test_a_raster_after_a_few_points_draws_what_the_reverse_order_draws():
    """The gate judges a raster image on the extent columns it draws, not its bound ``x``/``y``.

    ``P(5 rows) + raster`` renders the axes and marks ``raster + P(5 rows)``
    draws. Rejects judging the image's ``x`` binding against its one-row
    batch, which refused the chart. The x title stays layer 0's (``v``).
    """
    points = lambda: P(df.head(5))  # noqa: E731
    after = _quiet(lambda: (points() + _raster()).to_svg())
    before = _quiet(lambda: (_raster() + points()).to_svg())
    assert _axis_chrome(after) == _axis_chrome(before)
    assert tag_counts(after) == tag_counts(before) == {"circle": 5}
    assert after.count("<image") == before.count("<image") == 1
    assert axis_titles(after) == ["y", "v"]


def _not_drawn(entry: str, channel: str, field: str, mark: str) -> str:
    return (
        f"{entry}: {channel} '{field}' has no effect: no layer's mark draws {channel} "
        f"(this {mark} mark does not), so it gets no scale and no legend; it was ignored"
    )


# A channel bound where no layer's mark draws it: the bound chart, the same
# chart without the binding, and the one warning.
_UNDRAWN_CHANNELS: dict[str, tuple[Callable[[], Any], Callable[[], Any], str]] = {
    "F/line-shape": (
        lambda: line(shape="c"),
        line,
        _not_drawn("mark_line", "shape", "c", "line"),
    ),
    "F/raster-color": (
        lambda: _raster(color="f"),
        _raster,
        "mark_raster: color 'f' has no effect: the image colors its cells by their 'count' "
        "values (field=) and does not read color='f', so it gets no scale and no legend; "
        "it was ignored",
    ),
    "L/own-shape": (
        lambda: line().layer(layer("line", shape="c")),
        lambda: line().layer(layer("line")),
        _not_drawn("layer 1: mark_line", "shape", "c", "line"),
    ),
    "L/host-shape": (
        lambda: line(shape="c").layer(layer("line")),
        lambda: line().layer(layer("line")),
        _not_drawn("layer 0: mark_line", "shape", "c", "line"),
    ),
    "+/shape": (
        lambda: line(shape="c") + line(df2),
        lambda: line() + line(df2),
        _not_drawn("layer 0: mark_line", "shape", "c", "line"),
    ),
    "LCm/shape": (
        lambda: fm.LayerChart(line(shape="c"), line(df2))._build_merged(),
        lambda: fm.LayerChart(line(), line(df2))._build_merged(),
        _not_drawn("layer 0: mark_line", "shape", "c", "line"),
    ),
    "LCs/shape": (
        lambda: fm.LayerChart(line(shape="c"), line(df2)),
        lambda: fm.LayerChart(line(), line(df2)),
        _not_drawn("mark_line", "shape", "c", "line"),
    ),
}


def _recorded_svg(build: Callable[[], Any]) -> tuple[str, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
    return svg, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("case", sorted(_UNDRAWN_CHANNELS))
def test_a_channel_no_layers_mark_draws_builds_no_scale_and_no_legend(case):
    """A channel no layer's mark draws gets no scale and no legend, and warns once.

    The chart renders the bytes it renders without the binding. Rejects a
    shape legend beside a line that draws no glyph, and a color legend
    beside a raster image whose color names a field other than the value
    field its cells are colored by.
    """
    bound, unbound, message = _UNDRAWN_CHANNELS[case]
    svg, caught = _recorded_svg(bound)
    assert caught == [message]
    assert legend_set(svg) == {}
    assert svg == _quiet(lambda: unbound().to_svg())


@pytest.mark.parametrize("case", sorted(_UNDRAWN_CHANNELS))
def test_a_channel_no_layers_mark_draws_has_no_interactive_legend(case):
    bound, _, _ = _UNDRAWN_CHANNELS[case]
    assert _quiet(lambda: interactive_legend_set(bound())) == {}


_RASTER_VALUES = _RASTER_ROWS.with_columns(
    z=pl.Series(np.random.default_rng(5).normal(0, 1, len(_RASTER_ROWS)))
)


def _mean_raster(**encoding: Any) -> fm.Chart:
    return (
        fm.Chart(_RASTER_VALUES)
        .mark_raster(aggregate="mean", field="z")
        .encode(x="x", y="y", **encoding)
    )


def test_a_raster_color_naming_its_value_field_draws_a_colorbar_over_the_drawn_cells():
    """A raster image draws color through its value field: the colorbar spans the cell means.

    Rejects the colorless render with an undrawn-channel warning, and a
    colorbar over the source column whose colors the cells do not use.
    """
    import hashlib

    svg, caught = _recorded_svg(lambda: _mean_raster(color="z"))
    assert caught == []
    assert legend_set(svg) == {"z": ("-2.7", "-1.33", "0.04", "1.41", "2.77")}
    assert "linearGradient" in svg
    assert hashlib.sha256(svg.encode()).hexdigest() == (
        "8b9f15c3b480ff81ee78725fbde2076689d8f96aadf1e99cba304506818fb97b"
    )
    assert _quiet(lambda: interactive_legend_set(_mean_raster(color="z"))) != {}


def _few_raster_points() -> fm.Chart:
    return fm.Chart(_RASTER_ROWS.head(5)).mark_point().encode(x="x", y="y")


# Each raster door, and the tooltip field set of each interactive mark batch.
_RASTER_TOOLTIPS: dict[str, tuple[Callable[[], Any], list[set[str]]]] = {
    "F/raster": (_raster, [set()]),
    "F/auto-raster": (_ESTABLISHING_CONTROLS["F/auto-raster"], [set()]),
    "Fc/raster": (_ESTABLISHING_CONTROLS["Fc/raster"], [set(), set()]),
    "+/raster-then-points": (lambda: _raster() + _few_raster_points(), [set(), {"x", "y"}]),
    "+/points-then-raster": (lambda: _few_raster_points() + _raster(), [{"x", "y"}, set()]),
}


@pytest.mark.parametrize("case", sorted(_RASTER_TOOLTIPS))
def test_a_raster_image_lists_no_auto_tooltip_for_the_axes_it_draws_through_extents(case):
    """The auto tooltip lists only the fields a layer draws as its own columns.

    A raster image draws ``x``/``y`` through its cells' extent columns, so
    its auto tooltip lists neither and its interactive render succeeds; a
    point beside it keeps its own ``x``, ``y``. Rejects refusing every
    interactive raster on the bound ``x`` its rows lack, and rejects
    rewriting the entries onto the extent columns, which read as four
    constants on every cell.
    """
    build, expected = _RASTER_TOOLTIPS[case]
    assert "<html" in _quiet(lambda: build().interactive().to_html()).lower()
    assert _quiet(lambda: interactive_tooltip_fields(build())) == expected


@pytest.mark.parametrize("kind", ["static", "interactive"])
def test_a_raster_tooltip_naming_a_bound_axis_is_dropped_with_a_warning(kind):
    """A raster's ``tooltip="x"`` is the chart's binding its image sub-layer inherits.

    The image reads the raster statistic's cells, which carry no ``x``, so
    the inherited tooltip is dropped with a warning and the image carries
    no tooltip.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = _raster(tooltip="x")
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert _dropped("mark_raster", "image", "tooltip", "x") in {str(w.message) for w in caught}
    if kind == "interactive":
        assert _quiet(lambda: interactive_tooltip_fields(_raster(tooltip="x"))) == [set()]


def _raster_color_refusal(field: str, reason: str) -> str:
    return (
        f"mark_raster: encoding 'color' names the raster's cell values '{field}', which are "
        f"quantitative, but {reason}. Drop type= (or the categorical color binding), or draw "
        "categorical cells with mark_rect."
    )


def _grouped_points() -> fm.Chart:
    """Points colored by the categorical ``f``, over the raster's own rows."""
    return fm.Chart(_RASTER_VALUES.head(5)).mark_point().encode(x="x", y="y", color="f")


_COUNT_RASTER_BESIDE_CATEGORIES = (
    "the chart's color scale is categorical, established by layer 0 ('f')"
)

_BOTH_KINDS = ("static", "interactive")
_OTHER_LAYERS_SCALE = _raster_color_refusal("count", _COUNT_RASTER_BESIDE_CATEGORIES)

# Each door: its build, the output kinds it is judged on, and the exact refusal.
_RASTER_COLOR_REFUSALS: dict[str, tuple[Callable[[], Any], tuple[str, ...], str]] = {
    **{
        f"F/declared-{kind}": (
            lambda kind=kind: _mean_raster(color=fm.Color("z", type=kind)),
            _BOTH_KINDS,
            _raster_color_refusal("z", f"it is declared {kind}"),
        )
        for kind in ("nominal", "ordinal")
    },
    "+/categorical-scale-of-another-layer": (
        lambda: _grouped_points() + _raster(color="count"),
        _BOTH_KINDS,
        _OTHER_LAYERS_SCALE,
    ),
    "LCm/categorical-scale-of-another-layer": (
        lambda: fm.LayerChart(_grouped_points(), _raster(color="count"))._build_merged(),
        _BOTH_KINDS,
        _OTHER_LAYERS_SCALE,
    ),
    # The static overlay renders the image as its own leaf (the second
    # member, leaf #1), whose refusal the composite render wraps;
    # LayerChart's interactive route is LCm's.
    "LCs/categorical-scale-of-another-layer": (
        lambda: fm.LayerChart(_grouped_points(), _raster(color="count")),
        ("static",),
        f"failed to render composite leaf leaf #1: {_OTHER_LAYERS_SCALE}",
    ),
    "LCx/categorical-scale-of-another-layer": (
        lambda: fm.LayerChart(_grouped_points(), _raster(color="count")),
        ("interactive",),
        _OTHER_LAYERS_SCALE,
    ),
}


@pytest.mark.parametrize(
    ("case", "kind"),
    [
        (case, kind)
        for case, (_, kinds, _) in sorted(_RASTER_COLOR_REFUSALS.items())
        for kind in kinds
    ],
)
def test_a_categorical_color_over_a_rasters_cells_is_refused(case, kind):
    """A raster's cells are quantitative: a nominal or ordinal color over them is refused.

    Declared on the raster, or established by another layer's categorical
    scale the raster's value-field color would read. Rejects painting every
    cell transparent with no message, a categorical scale finding no
    category for a cell value.
    """
    build, _, message = _RASTER_COLOR_REFUSALS[case]
    with pytest.raises(ValueError) as refused:
        chart = build()
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == message


def _count_raster(**color: Any) -> fm.Chart:
    """A count raster coarse enough that its densest cell holds many rows."""
    return fm.Chart(_RASTER_ROWS).mark_raster(resolution=8).encode(x="x", y="y", **color)


def _colorbar(build: Callable[[], Any], kind: str) -> tuple[str, ...]:
    legends = (
        legend_set(_quiet(lambda: build().to_svg()))
        if kind == "static"
        else _quiet(lambda: interactive_legend_set(build()))
    )
    return legends["count"]


# The count raster on every door that composes it.
_COUNT_RASTER_DOORS: dict[str, Callable[[], Any]] = {
    "F": lambda: _count_raster(color="count"),
    "+": lambda: _count_raster(color="count") + _few_raster_points(),
    "LCs": lambda: fm.LayerChart(_count_raster(color="count"), _few_raster_points()),
    "LCm": lambda: fm.LayerChart(
        _count_raster(color="count"), _few_raster_points()
    )._build_merged(),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_COUNT_RASTER_DOORS))
def test_a_count_rasters_colorbar_reads_raw_counts_through_symlog(door, kind):
    """The cells keep raw counts and the color scale owns the log compression.

    The five labels sit at even steps of the symlog scale over [0, max]:
    ``expm1(t * log1p(max))``, the densest cell's count at the top. Rejects
    labels in log1p units (a top label of 3.76 for 42 rows).
    """
    import math

    labels = _colorbar(_COUNT_RASTER_DOORS[door], kind)
    top = float(labels[-1])
    assert top == 42
    expected = [math.expm1(t * math.log1p(top)) for t in (0, 0.25, 0.5, 0.75, 1)]
    assert [float(label) for label in labels] == pytest.approx(expected, abs=0.06)


def test_a_linear_color_scale_on_a_count_raster_labels_counts_linearly():
    """The caller's ``LinearScale`` replaces the raster's symlog default."""
    labels = _colorbar(
        lambda: _count_raster(color=fm.Color("count", scale=fm.LinearScale())), "static"
    )
    assert [float(label) for label in labels] == pytest.approx([0, 10.5, 21, 31.5, 42], abs=0.06)


def test_a_density_rasters_colorbar_reads_raw_density():
    """A density raster's labels are densities, the top one the densest cell's."""
    svg = _quiet(
        lambda: (
            fm.Chart(_RASTER_ROWS)
            .mark_raster(aggregate="density")
            .encode(x="x", y="y", color="density")
            .to_svg()
        )
    )
    labels = [float(label) for label in legend_set(svg)["density"]]
    assert labels[0] == 0
    assert labels[-1] > 10


def _symlog_labels(lo: float, hi: float) -> list[float]:
    """The five colorbar labels of a symlog scale over [lo, hi], at t = 0, .25, .5, .75, 1."""
    import math

    def forward(v: float) -> float:
        return math.copysign(math.log1p(abs(v)), v)

    def inverse(v: float) -> float:
        return math.copysign(math.expm1(abs(v)), v)

    return [inverse(forward(lo) + t * (forward(hi) - forward(lo))) for t in (0, 0.25, 0.5, 0.75, 1)]


def _mean_raster_8(**kwargs: Any) -> fm.Chart:
    return (
        fm.Chart(_RASTER_VALUES)
        .mark_raster(resolution=8, aggregate="mean", field="z", **kwargs)
        .encode(x="x", y="y", color="z")
    )


# Each log_scale value: the raster and its colorbar's expected labels.
_LOG_SCALE_CASES: dict[str, tuple[Callable[[], fm.Chart], str, list[float]]] = {
    "count/unset-is-symlog": (
        lambda: _count_raster(color="count"),
        "count",
        _symlog_labels(0, 42),
    ),
    "count/false-is-linear": (
        lambda: (
            fm.Chart(_RASTER_ROWS)
            .mark_raster(resolution=8, log_scale=False)
            .encode(x="x", y="y", color="count")
        ),
        "count",
        [0, 10.5, 21, 31.5, 42],
    ),
    "mean/unset-is-linear": (
        _mean_raster_8,
        "z",
        [-1.64, -0.82, 0, 0.82, 1.63],
    ),
    "mean/true-is-symlog": (
        lambda: _mean_raster_8(log_scale=True),
        "z",
        _symlog_labels(-1.64, 1.63),
    ),
}


@pytest.mark.parametrize("door", ["F", "+"])
@pytest.mark.parametrize("case", sorted(_LOG_SCALE_CASES))
def test_a_rasters_log_scale_picks_its_color_scale_on_any_aggregate(case, door):
    """``log_scale=None`` is symlog for count and density, linear otherwise; True/False are honored.

    Rejects reading an explicit ``False`` as unset, which kept a count
    raster log-compressed, and rejects ignoring ``True`` on a mean raster.
    """
    build, title, expected = _LOG_SCALE_CASES[case]
    chart = (lambda: build() + _few_raster_points()) if door == "+" else build
    labels = legend_set(_quiet(lambda: chart().to_svg()))[title]
    # Labels are rounded for display: to 0.1 at this range, 0.01 near zero.
    assert [float(label) for label in labels] == pytest.approx(expected, abs=0.06)


def test_a_rasters_log_scale_refuses_anything_but_a_truth_value_or_none():
    with pytest.raises(ValueError) as refused:
        fm.Chart(_RASTER_ROWS).mark_raster(log_scale="yes")
    assert str(refused.value) == "mark_raster: log_scale must be True or False; got 'yes'"


def test_a_rasters_cmap_colors_its_cells_and_colorbar_when_color_names_the_value_field():
    """``mark_raster(cmap="magma").encode(color="count")`` draws magma, cells and colorbar alike.

    The cells draw the image the uncolored magma raster draws, and the
    colorbar's gradient starts at magma's first color. Rejects the bound
    color scale replacing the mark's cmap with the theme's scheme.
    """

    def magma(**color: Any) -> fm.Chart:
        return (
            fm.Chart(_RASTER_ROWS)
            .mark_raster(resolution=8, cmap="magma")
            .encode(x="x", y="y", **color)
        )

    def images(svg: str) -> list[str]:
        return re.findall(r'href="(data:image/png;base64,[^"]+)"', svg)

    bound = _quiet(lambda: magma(color="count").to_svg())
    assert images(bound) == images(_quiet(lambda: magma().to_svg()))
    assert images(bound) != images(_quiet(lambda: _count_raster(color="count").to_svg()))
    assert re.findall(r'<stop[^>]*stop-color="([^"]+)"', bound)[0] == "#000004"


_STEPS = ("#ffffcc", "#fd8d3c", "#800026")


def _rgba(hex_color: str) -> tuple[int, int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16), 255)


# Each discretizing scale on the count raster's color: the scale, the
# legend's labels in raw counts, and the cell colors (None: any three).
_STEPPED_RASTERS: dict[str, tuple[Any, tuple[str, ...], set | None]] = {
    "quantize": (
        fm.QuantizeScale(domain=[0, 42], range=list(_STEPS)),
        ("0", "14", "28", "42"),
        {_rgba(color) for color in _STEPS},
    ),
    "threshold": (
        fm.ThresholdScale(domain=[5, 20], range=[0.0, 1.0, 2.0]),
        ("0", "5", "20", "42"),
        None,
    ),
}


@pytest.mark.parametrize("door", ["F", "+"])
@pytest.mark.parametrize("case", sorted(_STEPPED_RASTERS))
def test_a_discretizing_color_scale_steps_a_rasters_cells_by_bucket(case, door):
    """A quantize or threshold color on a count raster draws one color per bucket.

    The image's opaque cells take exactly as many colors as the scale has
    buckets (a quantize scale's own range colors), and the legend's bucket
    boundaries read raw counts. Rejects mapping the cells through a
    continuous ramp, which drew one color per distinct count.
    """
    scale, labels, colors = _STEPPED_RASTERS[case]

    def build() -> fm.Chart:
        chart = _count_raster(color=fm.Color("count", scale=scale))
        return chart + _few_raster_points() if door == "+" else chart

    svg = _quiet(lambda: build().to_svg())
    (cells,) = _panel_cell_colors(svg)
    assert len(cells) == 3
    if colors is not None:
        assert set(cells) == colors
    assert legend_set(svg) == {"count": labels}


# Points over a few of the raster's rows, carrying a ``count`` column of their
# own, so a sibling's color can name a raster's value field.
_SIBLING_POINT_ROWS = _RASTER_VALUES.head(5).with_columns(
    count=pl.Series([1.0, 5.0, 10.0, 20.0, 40.0])
)


def _sibling_points(field: str) -> fm.Chart:
    return fm.Chart(_SIBLING_POINT_ROWS).mark_point().encode(x="x", y="y", color=field)


def _unbound_count_raster() -> fm.Chart:
    return fm.Chart(_RASTER_ROWS).mark_raster(resolution=8).encode(x="x", y="y")


def _unbound_mean_raster() -> fm.Chart:
    return (
        fm.Chart(_RASTER_VALUES)
        .mark_raster(resolution=8, aggregate="mean", field="z")
        .encode(x="x", y="y")
    )


# Each raster beside points colored by the raster's value field.
_SIBLING_RASTERS: dict[str, tuple[Callable[[], fm.Chart], str]] = {
    "count": (_unbound_count_raster, "count"),
    "mean-of-z": (_unbound_mean_raster, "z"),
}


def _sibling_doors(raster: Callable[[], fm.Chart], field: str) -> dict[str, Callable[[], Any]]:
    return {
        "+/points-then-raster": lambda: _sibling_points(field) + raster(),
        "+/raster-then-points": lambda: raster() + _sibling_points(field),
        "LD/points-on-the-raster": lambda: raster().layer(
            fm.Layer(data=_SIBLING_POINT_ROWS, mark="point", encoding={"color": field})
        ),
        "LCm/points-first": lambda: fm.LayerChart(_sibling_points(field), raster())._build_merged(),
        "LCm/raster-first": lambda: fm.LayerChart(raster(), _sibling_points(field))._build_merged(),
        "LCs/points-first": lambda: fm.LayerChart(_sibling_points(field), raster()),
        "LCs/raster-first": lambda: fm.LayerChart(raster(), _sibling_points(field)),
    }


@pytest.mark.parametrize(
    ("raster", "door"),
    [
        (raster, door)
        for raster in sorted(_SIBLING_RASTERS)
        for door in sorted(_sibling_doors(*_SIBLING_RASTERS[raster]))
    ],
)
def test_a_raster_beside_points_colored_by_its_value_field_draws_its_own_cells(raster, door):
    """The points' color is theirs: the raster, binding no color, draws on its own cell scale.

    One image on every route, LayerChart's own included, equal to the
    raster drawn alone, beside the points' colorbar, and every route draws
    LayerChart's chrome, legend and marks. Rejects judging the image by the sibling's ``color="count"``,
    which dropped it from the cell scale and drew no image.
    """
    build_raster, field = _SIBLING_RASTERS[raster]
    svg = _quiet(lambda: _sibling_doors(build_raster, field)[door]().to_svg())
    layer_chart = _quiet(lambda: fm.LayerChart(_sibling_points(field), build_raster()).to_svg())

    def images(text: str) -> list[str]:
        return re.findall(r'href="(data:image/png;base64,[^"]+)"', text)

    assert images(svg) == images(_quiet(lambda: build_raster().to_svg()))
    assert len(images(svg)) == 1
    assert list(legend_set(svg)) == [field]
    assert legend_set(svg) == legend_set(layer_chart)
    assert chrome_text(svg) == chrome_text(layer_chart)
    assert tag_counts(svg) == tag_counts(layer_chart) == {"circle": 5}


_AREA_ROWS = _RASTER_VALUES.head(20).sort("x").with_columns(g=pl.Series(["a", "b"] * 10))


def _area(**color: Any) -> fm.Chart:
    return fm.Chart(_AREA_ROWS).mark_area().encode(x="x", y="y", **color)


def _area_points(**color: Any) -> fm.Chart:
    return fm.Chart(_AREA_ROWS).mark_point().encode(x="x", y="y", **color)


# Each pair, and the color legend it draws: "colorbar" (continuous) or
# "categorical" (one swatch per value).
_AREA_GROUPING: dict[str, tuple[Callable[[], fm.Chart], str, str]] = {
    "area-then-raster": (lambda: _area() + _count_raster(color="count"), "count", "colorbar"),
    "raster-then-area": (lambda: _count_raster(color="count") + _area(), "count", "colorbar"),
    "area-then-points-z": (lambda: _area() + _area_points(color="z"), "z", "colorbar"),
    "points-z-then-area": (lambda: _area_points(color="z") + _area(), "z", "colorbar"),
    "points-then-area-z": (lambda: _area_points() + _area(color="z"), "z", "categorical"),
    "area-z-then-points": (lambda: _area(color="z") + _area_points(), "z", "categorical"),
}


@pytest.mark.parametrize("case", sorted(_AREA_GROUPING))
def test_an_area_groups_color_by_category_only_when_it_draws_color(case):
    """An area beside a quantitative color leaves it continuous unless the area draws color.

    An area drawing no color makes no demand on the color scale, in either
    order: a raster's or a point's numeric color keeps its colorbar. An
    area drawing color groups by it in either order, so its ``z`` is
    categorical. Rejects keying the grouping on the chart's first mark,
    which refused ``area + raster(color="count")``, gave
    ``area + point(color="z")`` one swatch per value, and drew the area's
    own ``z`` as a colorbar behind a point layer 0.
    """
    from tests._hue_probe import has_colorbar

    build, title, kind = _AREA_GROUPING[case]
    svg = _quiet(lambda: build().to_svg())
    assert has_colorbar(svg) == (kind == "colorbar")
    entries = legend_set(svg)[title]
    if kind == "colorbar":
        assert len(entries) == 5
    else:
        assert len(entries) == _AREA_ROWS["z"].n_unique()


@pytest.mark.parametrize(("order", "layer"), [("area-then-raster", 0), ("raster-then-area", 1)])
def test_an_area_drawing_a_category_beside_a_rasters_cells_is_refused(order, layer):
    """``area(color="g")`` makes the color scale categorical, which a raster's cells cannot read."""
    build = (
        (lambda: _area(color="g") + _count_raster(color="count"))
        if order == "area-then-raster"
        else (lambda: _count_raster(color="count") + _area(color="g"))
    )
    with pytest.raises(ValueError) as refused:
        build().to_svg()
    assert str(refused.value) == _raster_color_refusal(
        "count", f"the chart's color scale is categorical, established by layer {layer} ('g')"
    )


_UNEVEN_FACETS_RNG = np.random.default_rng(7)
_UNEVEN_FACETS = pl.DataFrame(
    {
        "x": _UNEVEN_FACETS_RNG.normal(0, 1, 400),
        "y": _UNEVEN_FACETS_RNG.normal(0, 1, 400),
        "f": ["m"] * 300 + ["n"] * 100,
    }
)


def _panel_cell_colors(svg: str) -> list[Counter]:
    """Each panel's raster image: how many of its pixels draw each opaque color."""
    import base64

    from _snapshots import decode_png_rgba

    panels = []
    for encoded in re.findall(r'href="data:image/png;base64,([^"]+)"', svg):
        pixels, _, _ = decode_png_rgba(base64.b64decode(encoded))
        panels.append(Counter(p for row in pixels for p in row if p[3] > 0))
    return panels


def test_a_faceted_raster_drawing_no_color_maps_every_panel_through_one_cell_scale():
    """Panels whose densest cells differ (300 rows vs 100) share one color scale.

    A cell holding one row draws the same color in both panels; the denser
    panel's top colors appear in it alone. Rejects scaling each panel by its
    own maximum, which drew the sparse panel's single-row cells darker than
    the dense panel's.
    """
    svg = _quiet(
        lambda: (
            fm.Chart(_UNEVEN_FACETS)
            .mark_raster(resolution=12)
            .encode(x="x", y="y")
            .facet("f")
            .to_svg()
        )
    )
    dense, sparse = _panel_cell_colors(svg)
    assert dense.most_common(1)[0][0] == sparse.most_common(1)[0][0]
    assert set(sparse) < set(dense)


def test_a_rasters_cells_are_no_output_a_layer_can_read():
    """A layer naming ``raster__cells`` is refused as any unknown output is."""
    from ferrum._layer import _Layer

    chart = _raster().layer(
        _Layer(
            name="cells",
            mark="point",
            encoding={"x": "x", "y": "y"},
            data_source="raster__cells",
        )
    )
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == (
        "transform failed: layer 1 data_source 'raster__cells' not found in transform "
        "outputs; available keys: [__final__, raster]"
    )


def test_a_shape_one_layers_point_draws_is_established_by_that_point():
    """The point draws shape, so its rows alone give the shape legend.

    Rejects unioning the attached line's ``g`` values (df2's a, b, c) into
    the domain: the line inherits the binding but draws no glyph.
    """
    svg = _quiet(lambda: P(shape="g").layer(fm.Layer(data=df2, mark="line")).to_svg())
    assert legend_set(svg) == {"g": ("a", "b")}


# ---------------------------------------------------------------------------
# A field selection declared on one LayerChart member reaches every member
# ---------------------------------------------------------------------------

_PICK_G = fm.selection_point(fields=["g"], name="pick_g")

# Two members, each built fresh, carrying the selections each case declares.
_SELECTION_MEMBERS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "declared-on-the-second": (P, lambda: line().add_selection(_PICK_C)),
    "declared-on-the-first": (lambda: P().add_selection(_PICK_C), line),
    "other-member-lacks-the-field": (lambda: P().add_selection(_PICK_C), lambda: line(df2)),
    "one-on-each": (lambda: P().add_selection(_PICK_G), lambda: line().add_selection(_PICK_C)),
}


def _title_texts(svg: str) -> Counter:
    return Counter(re.findall(r"<title>([^<]*)</title>", svg))


@pytest.mark.parametrize("door", ["+", "LCm"])
@pytest.mark.parametrize("case", sorted(_SELECTION_MEMBERS))
def test_a_layer_chart_members_selection_reaches_every_member_on_the_static_overlay(case, door):
    """LCs lists a field selection's field on every member whose rows carry it, as ``+`` does.

    Rejects projecting a selection only onto the member that declared it:
    ``LayerChart(P(), line().add_selection(pick_c))`` drew 1 ``<title>``
    where ``+`` and LCm draw 9.
    """
    a, b = _SELECTION_MEMBERS[case]
    overlay = _quiet(lambda: fm.LayerChart(a(), b()).to_svg())
    other = {
        "+": lambda: a() + b(),
        "LCm": lambda: fm.LayerChart(a(), b())._build_merged(),
    }[door]
    expected = _title_texts(_quiet(lambda: other().to_svg()))
    assert _title_texts(overlay) == expected
    assert sum(expected.values()) > 1


#: Each case, and the selected fields ({"c"} or {"g"}) each member's
#: interactive tooltip carries once the selection has reached it.
_SELECTION_REACH_EXPECTED: dict[str, tuple[set[str], set[str]]] = {
    "declared-on-the-second": ({"c"}, {"c"}),
    "declared-on-the-first": ({"c"}, {"c"}),
    "other-member-lacks-the-field": ({"c"}, set()),
    "one-on-each": ({"c", "g"}, {"c", "g"}),
}


@pytest.mark.parametrize("case", sorted(_SELECTION_MEMBERS))
def test_a_layer_chart_members_selection_reaches_every_member_interactively(case):
    """LCx lists a field selection's field on every member's tooltip whose rows carry it.

    The same four cases as the static overlay, read off the merged
    interactive scene's tooltip fields rather than ``<title>`` text. A
    guard, not a regression test: rejects the field reaching a sibling
    whose rows lack it (``other-member-lacks-the-field``), and rejects it
    reaching only the member that declared it.
    """
    a, b = _SELECTION_MEMBERS[case]
    expected_a, expected_b = _SELECTION_REACH_EXPECTED[case]
    fields = _quiet(lambda: interactive_tooltip_fields(fm.LayerChart(a(), b())))
    assert len(fields) == 2
    assert fields[0] & {"c", "g"} == expected_a
    assert fields[1] & {"c", "g"} == expected_b


# ---------------------------------------------------------------------------
# The wire seam, read off every matrix case
# ---------------------------------------------------------------------------

_MATRIX: dict[str, Case] = {
    **{f"value_binding/{k}": v for k, v in _VALUE_BINDING.items()},
    **{f"value_binding_by_name/{k}": v for k, v in _VALUE_BINDING_BY_NAME.items()},
    **{f"value_aggregate/{k}": v for k, v in _VALUE_AGGREGATE.items()},
    **{f"sibling_grain/{k}": v for k, v in _SIBLING_GRAIN.items()},
    **{f"different_frame_sibling/{k}": v for k, v in _DIFFERENT_FRAME_SIBLING.items()},
    **{f"right_aggregate/{k}": v for k, v in _RIGHT_AGGREGATE.items()},
    **{f"alias_shadow/{k}": v for k, v in _ALIAS_SHADOW.items()},
    **{f"layer_encoding_string/{k}": v for k, v in _LAYER_ENCODING_STRING.items()},
    **{f"grouping_binding/{k}": v for k, v in _GROUPING_BINDING.items()},
    **{f"inherited_grouping/{k}": v for k, v in _INHERITED_GROUPING.items()},
    **{f"aggregating_sibling_grain/{k}": v for k, v in _AGGREGATING_SIBLING_GRAIN.items()},
    **{f"attached_grouping/{k}": v for k, v in _ATTACHED_GROUPING.items()},
    **{f"parent/{k}": v for k, v in _PARENT.items()},
    **{f"axis_fallback/{k}": v for k, v in _AXIS_FALLBACK.items()},
    **{f"net/{k}": v for k, v in _NET.items()},
    **{f"data_layer_own_rows/{k}": v for k, v in _DATA_LAYER_OWN_ROWS.items()},
    **{f"different_frame_own_rows/{k}": v for k, v in _DIFFERENT_FRAME_OWN_ROWS.items()},
    **{f"different_frame_sibling_titled/{k}": v for k, v in _RENAMED_TITLES.items()},
    **{f"secondary_operand_rows/{k}": v for k, (v, _vertices) in _SECONDARY_OPERAND_ROWS.items()},
}


@pytest.mark.parametrize("case", sorted(_MATRIX))
def test_each_layers_completed_wire_encoding_draws_what_it_draws_alone(case):
    """The wire layer encoding over the chart-level encoding = the layer's effective encoding.

    Rejects the sibling-encoding leak on the wire: the chart-level encoding
    would carry an operand's own binding (or a promoted copy), completing a
    sibling with it.
    """
    check_wire_seam(_MATRIX[case])


# Every matrix case reached through ``fm.LayerChart``: the prefix its four routes share.
_LAYER_CHART_CASES = sorted(key[: -len("LCs")] for key in _MATRIX if key.endswith("/LCs"))


@pytest.mark.parametrize("prefix", _LAYER_CHART_CASES)
def test_the_static_overlay_draws_the_legends_every_layer_chart_route_draws(prefix):
    """Route parity: LCs draws LCm's and LCi's legend set, and LCx shows it interactively.

    Rejects a static overlay that resolves each member's auxiliary scales
    alone: a legend per member, or one listing only the first member's values.
    """
    static = legend_set(_quiet(lambda: _MATRIX[prefix + "LCs"].build().to_svg()))
    for door in ("LCm", "LCi"):
        assert legend_set(_quiet(lambda: _MATRIX[prefix + door].build().to_svg())) == static, door
    assert _quiet(lambda: interactive_legend_set(_MATRIX[prefix + "LCx"].build())) == static


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def test_a_parent_with_no_bindings_leaves_every_layer_its_own():
    chart = mk("point", x="v", y="y").layer(layer("line", color="c"))
    wire = _wire(chart)
    assert "color" not in wire["encoding"]
    assert wire["layers"][1]["encoding"]["color"]["field"] == "c"


def test_a_layer_binding_every_channel_itself_ignores_its_parent():
    """The layer draws its own fields; the host's reach only the host (the shared scales union them)."""
    chart = P(color="c", size="z").layer(layer("point", y="r", color="g", size="r"))
    check_wire_seam(
        Case(
            lambda: chart,
            (lambda: P(color="c", size="z"), lambda: P(y="r", color="g", size="r")),
            chart_level=None,
        )
    )
    assert tag_counts(chart.to_svg()) == {"circle": 16}
    assert list(legend_set(chart.to_svg())) == ["c", "z"]


def test_none_channel_values_stay_unset_and_inherit_nothing_new():
    assert _quiet(P().encode(color=None).to_svg) == _quiet(P().to_svg)
    chart = (P() + line(y="r")).encode(color=None)
    assert "color" not in _wire(chart)["encoding"]


def test_a_none_in_layer_encoding_leaves_the_channel_to_the_parent():
    """A ``None`` channel value binds nothing: the parent's binding arrives whole.

    Python's effective encoding agrees with the wire Rust completes, so a
    host's aggregated color is lowered for the layer too. Rejects counting
    ``None`` as an own binding: the layer would group without the color and
    draw uncolored points while Rust reads the host's color.
    """
    lowered = bar_mean(color=mean(fm.Color)).layer(fm.Layer(mark="point", encoding={"color": None}))
    check_rendering(
        Case(
            lambda: lowered,
            (lambda: bar_mean(color=mean(fm.Color)), lambda: agg("point", color=mean(fm.Color))),
            marks={"rect": 2, "circle": 2},
            chart_level=None,
        )
    )

    from ferrum._layer_transforms import _effective_encoding

    chart = P(color="c").layer(fm.Layer(mark="line", encoding={"color": None}))
    attached = chart._layers[-1]
    assert _effective_encoding(attached, chart._encoding)["color"].field == "c"
    assert tag_counts(chart.to_svg()) == {"circle": 8, "polyline": 2}
    check_wire_seam(
        Case(lambda: chart, (lambda: P(color="c"), lambda: line(color="c")), chart_level=None)
    )


def test_an_alias_and_its_canonical_on_one_layer_keep_todays_resolution():
    """``color`` wins over ``stroke`` on one layer, with the existing warning."""
    from ferrum._warn import reset_warnings

    reset_warnings()
    with pytest.warns(UserWarning, match=r"encode\(stroke=\.\.\.\) is ignored when color="):
        chart = P().layer(layer("point", y="r", color="c", stroke="g"))
        svg = chart.to_svg()
    assert legend_set(svg) == {"c": ("p", "q")}


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: a + b,
        lambda a, b: b + a,
        lambda a, b: a.layer(fm.Layer(mark="line", encoding={"color": "c"})),
        lambda a, b: fm.LayerChart(a, b)._build_merged(),
        lambda a, b: (a + b).encode(detail="g"),
    ],
    ids=["+", "+rev", "L", "LCm", "+e"],
)
def test_the_callers_charts_are_never_mutated(build):
    a = bar_mean(color=mean(fm.Color))
    b = agg("tick", df2, size=mean(fm.Size))
    before = (a.to_json(), b.to_json(), dict(a._encoding), dict(b._encoding))
    _quiet(lambda: build(a, b).to_svg())
    assert (a.to_json(), b.to_json(), dict(a._encoding), dict(b._encoding)) == before


# ---------------------------------------------------------------------------
# Metric labels: one label per curve hue, written on the chart or its first coloring layer
# ---------------------------------------------------------------------------


def _roc_frame() -> pl.DataFrame:
    rows = []
    # `brier` is the per-sample score BrierLabel reads, broadcast per class.
    for name, lift, brier in (("c0", 1.0, 0.12), ("c1", 0.6, 0.21)):
        for fpr in (0.0, 0.25, 0.5, 0.75, 1.0):
            tpr = min(1.0, fpr + lift * (1 - fpr) * 0.5)
            rows.append({"fpr": fpr, "tpr": tpr, "class": name, "brier": brier})
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    "label", [fm.AUCLabel, fm.APLabel, fm.BrierLabel], ids=["AUC", "AP", "Brier"]
)
def test_a_metric_label_on_a_composition_labels_each_curve_hue(label):
    """The label groups by the curves' color although the composition carries none itself.

    The left operand's color stays on its own layer, so the label reads
    the first layer that draws a color. Rejects reading only the chart
    level: one pooled label would replace the two per-class labels.
    """
    roc = _roc_frame()
    curves = fm.Chart(roc).mark_line().encode(x="fpr", y="tpr", color="class")
    composed = curves + fm.Chart(roc).mark_point().encode(x="fpr", y="tpr")
    flat_labels = [t for t in chrome_text((curves + label()).to_svg()) if "=" in t]
    composed_labels = [t for t in chrome_text((composed + label()).to_svg()) if "=" in t]
    assert len(flat_labels) == 2
    assert composed_labels == flat_labels


# ---------------------------------------------------------------------------
# Byte-identical controls: charts the model does not reach
# ---------------------------------------------------------------------------

_CONTROLS: dict[str, Callable[[], Any]] = {
    "F/point-color": lambda: P(color="c"),
    "F/bar-mean-color": lambda: bar_mean(color="c"),
    # The opacity legend is suppressed: the marks draw the bytes they drew
    # before the opacity family drew a legend.
    "F/point-size-opacity": lambda: P(size="z", opacity=fm.Opacity("r", legend=None)),
    "L/host-color-line": lambda: P(color="c").layer(layer("line")),
    "L/host-bar-mean-label": lambda: bar_mean().layer(
        layer("text", text=fm.Text("y", aggregate="mean"))
    ),
    "+/both-color": lambda: P(color="c") + line(color="c"),
    "+/neither": lambda: P() + line(df2),
    "+/both-size": lambda: P(size="z") + line(size="z"),
    "LCs/neither": lambda: fm.LayerChart(P(), line(df2)),
    # Polar charts and aggregates with no span or polar aggregate: the
    # polar remap moved ahead of the statistics, and each channel reads its
    # own (field, op) output, without changing what these draw.
    "F/arc-theta": lambda: _polar("arc", theta="z", color="g"),
    "F/arc-theta-on-y": lambda: mk("arc", theta="z", color="g").coord(fm.CoordPolar(theta="y")),
    "F/radial-stack": lambda: _polar(
        "bar", theta="g", radius=fm.Radius("z", stack=True), color="c"
    ),
    "L/polar-bar-text": lambda: _polar("bar", theta="g", radius="z").layer(layer("text", text="g")),
    "+/polar-bar-text": lambda: (
        _polar("bar", theta="g", radius="z") + _polar("text", theta="g", radius="z", text="g")
    ),
    "LCs/polar-arcs": lambda: fm.LayerChart(
        _polar("arc", theta="z", color="g"), _polar("arc", theta="z", color="g")
    ),
    "F/count-bar": lambda: mk("bar", x="g", y="count()"),
    "F/two-ops": lambda: mk("point", x="g", y="mean(y)", size="sum(z)"),
}

# The SVG each control renders.
_CONTROL_DIGESTS = {
    "F/point-color": "a4a1d3b7746458ddbe4983e33cee3886e90b208e4f2e5be9c8c12482da5d5dff",
    "F/bar-mean-color": "3d61496ff5e8aba078879d090bafeb1a3ab31cfd72eeb86cb39ec6c617c73aa0",
    "F/point-size-opacity": "0f9930eab400f34f690bbbb5d8086dfc2284512b0ef1dda6b586ff4aa37d38fa",
    "L/host-color-line": "99a33743ce01c54b26f67dbfaa45dc58f34b9e2e9514e28cdf236bccdf267f78",
    "L/host-bar-mean-label": "4b0490b9eccca7c9c8973d5322b1e582068981407f9bc25e624f720df342506d",
    "+/both-color": "99a33743ce01c54b26f67dbfaa45dc58f34b9e2e9514e28cdf236bccdf267f78",
    "+/neither": "ab618fcb0f56aa5dc88e8f28c6c11214824166ef199be81771863ad707e4f386",
    "+/both-size": "356a7ea8ac422ec1379f30507987f35540a7f3933ec32ba8140799c0c2a6496e",
    "LCs/neither": "7d0f2c91ecb949cb1b20bb61543b06c1f7223390b381e767977fb5b53293bc24",
    "F/arc-theta": "5c5f1b01e3cdbac1ec0e6f2e3e86d88ab69f3534ff6b01d75b58b23066344d1b",
    "F/arc-theta-on-y": "6d975afe613277131bb547c5fa48984d4de13e97e56115175a88b6fe1c6bfc20",
    "F/radial-stack": "f3336e01b5844bd24a0f9ea845abf74a4805ce8af6a37128a7bab9a013a62997",
    "L/polar-bar-text": "bd839ed9c6026b660bcbdd15192e73016a45e26af9ee0935de00452c29faccbd",
    "+/polar-bar-text": "bd839ed9c6026b660bcbdd15192e73016a45e26af9ee0935de00452c29faccbd",
    "LCs/polar-arcs": "614ffef7c51cfb8411810444014f4d4fe0fefa02453ca316b819f637640214aa",
    "F/count-bar": "d3c75436aa0a8e9be82406d7348cb12ba2eeb4a4aed69235643ed5275395ad39",
    "F/two-ops": "4dee40d473c7c339798b04dd2f1e31f0eba3e75841ce4df370c6ea8e72fcdf67",
}


@pytest.mark.parametrize("case", sorted(_CONTROLS))
def test_a_control_renders_its_pinned_bytes(case):
    import hashlib

    svg = _quiet(lambda: _CONTROLS[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _CONTROL_DIGESTS[case]


# ---------------------------------------------------------------------------
# An operand draws in a composition what it draws alone
# ---------------------------------------------------------------------------
#
# The parts are whole operands: a host with attached layers, or a
# composition carrying a binding written on it. A binding such an operand's
# layer inherits stays inherited inside a later ``+``: a text layer draws no
# inherited color, a literal paint keeps its stroke, and a line's own width
# wins over an inherited size, exactly as when the operand renders alone.


def _host_label() -> fm.Chart:
    """A colored bar host with an attached text label, which draws no inherited color."""
    return mk("bar", x="g", y="y", color="g").layer(layer("text", text="y"))


def _host_grey_line() -> fm.Chart:
    """A colored point host with an attached grey line, whose literal stroke wins."""
    return P(color="c").layer(fm.Layer(mark="line", mark_kwargs={"stroke": "#999999"}))


def _host_width_line() -> fm.Chart:
    """A sized point host with an attached line owning its stroke width."""
    return P(size="z").layer(layer("line", stroke_width="r"))


def _composed_label() -> fm.Chart:
    """A composition whose color is written on it, reaching its bar and its text."""
    return (mk("bar", x="g", y="y") + mk("text", x="g", y="y", text="y")).encode(color="g")


def _lowered_label() -> fm.Chart:
    """A host with an aggregated color, beside an attached text that inherits no color."""
    return bar_mean(color=mean(fm.Color)).layer(layer("text", text=fm.Text("y", aggregate="mean")))


def _rule() -> fm.Chart:
    return mk("rule", y="r")


def _rule_d2() -> fm.Chart:
    return mk("rule", df2, y="w")


def _titled_host_label() -> fm.Chart:
    """A host whose y carries a caller's title, with an attached text label of the same column."""
    return mk("bar", x="g", y=fm.Y("y", title="Yield"), color="g").layer(layer("text", text="y"))


def _titled_host_grey_line() -> fm.Chart:
    """A host whose color carries a caller's title, with an attached grey line binding no field."""
    return mk("bar", x="g", y="y", color=fm.Color("g", title="Group")).layer(
        fm.Layer(mark="line", mark_kwargs={"stroke": "#999999"})
    )


_OPERANDS: dict[str, Callable[[], fm.Chart]] = {
    "host-label": _host_label,
    "titled-host-label": _titled_host_label,
    "titled-host-grey-line": _titled_host_grey_line,
    "host-grey-line": _host_grey_line,
    "host-width-line": _host_width_line,
    "composed-label": _composed_label,
    "lowered-label": _lowered_label,
}


def _operand_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for name, operand in _OPERANDS.items():
        cases |= _cases(f"{name}/+s", plus_doors(operand, _rule))
        cases |= _cases(f"{name}/+d", plus_doors(operand, _rule_d2))
    return cases


_OPERAND = _operand_cases()


@pytest.mark.parametrize("case", sorted(_OPERAND))
def test_an_operand_draws_inside_a_composition_what_it_draws_alone(case):
    """A host's labels stay dark, its grey line whole, its line's width its own.

    Rejects carrying an operand's inherited binding onto its layers as their
    own: the labels would take the palette, the grey line would split by
    the host's color, and the own-width line would refuse beside an
    inherited size.
    """
    check_rendering(_OPERAND[case])


def _x_tick_labels(svg: str) -> list[str]:
    """The bottom x axis's tick labels: the lowest row of centered chrome-colored text."""
    ticks = [
        e
        for e in _chrome_texts(svg)
        if e.get("fill") == _CHROME_TEXT
        and e.get("text-anchor") == "middle"
        and "transform" not in e.attrib
    ]
    bottom = max(float(e.get("y")) for e in ticks)
    return ["".join(e.itertext()).strip() for e in ticks if float(e.get("y")) == bottom]


@pytest.mark.parametrize("name", sorted(_OPERANDS))
def test_an_axis_layer_zero_does_not_draw_lists_only_what_its_drawers_draw(name):
    """``rule(df2, y="w") + operand``: the x axis is the operand's alone, with no stray category.

    The rule draws no x, so the operand's layers establish the x domain
    from the rows they draw. Rejects reading the field from layer 0's
    frame: df2's ``g`` added ``c`` (or the padding added ``null``).
    """
    operand = _OPERANDS[name]
    alone = _x_tick_labels(_quiet(lambda: operand().to_svg()))
    assert _x_tick_labels(_quiet(lambda: (_rule_d2() + operand()).to_svg())) == alone


@pytest.mark.parametrize(
    "build",
    [
        lambda: mk("rule", df2, y="w") + mk("bar", x="g", y="y"),
        lambda: _filtered(mk("rule", df2, y="w")) + mk("bar", x="g", y="y"),
        lambda: fm.LayerChart(mk("rule", df2, y="w"), mk("bar", x="g", y="y"))._build_merged(),
        lambda: fm.LayerChart(
            mk("rule", df2, y="w"), mk("bar", x="g", y="y"), resolve={"y": "independent"}
        ),
    ],
    ids=["+", "+-left-chain", "LCm", "LCi"],
)
def test_a_layer_zero_drawing_no_x_leaves_the_bars_categories_alone(build):
    """``rule(df2, y="w") + bar(df, x="g")`` lists a, b: the categories the bars draw."""
    assert _x_tick_labels(_quiet(lambda: build().to_svg())) == ["a", "b"]


def _tooltip_batches(chart: Any) -> Counter:
    """The multiset of per-batch tooltip field-name sets of *chart*'s interactive scene."""
    return Counter(frozenset(names) for names in interactive_tooltip_fields(chart))


@pytest.mark.parametrize("case", sorted(c for c in _OPERAND if not c.endswith("/LCs")))
def test_an_operand_lists_inside_a_composition_the_tooltips_it_lists_alone(case):
    """Interactively, too: each layer's auto tooltip fields are its fields alone.

    A layer binding a field lists its own fields; a layer binding none lists
    everything it draws. Rejects reading ownership off the wire by presence:
    a right operand's text would list the x the axis fill copied, and a
    grey line binding no field would drop its inherited color.
    """
    built = _OPERAND[case]
    expected = sum((_tooltip_batches(_quiet(part)) for part in built.parts), Counter())
    assert _tooltip_batches(_quiet(built.build)) == expected


@pytest.mark.parametrize("name", sorted(_OPERANDS))
def test_the_static_overlay_and_plus_draw_an_operand_alike(name):
    """Route parity: LCs and ``+`` draw the same marks for the same operands."""
    operand = _OPERANDS[name]
    plus = census(_quiet(lambda: (operand() + _rule()).to_svg()))
    overlay = census(_quiet(lambda: fm.LayerChart(operand(), _rule()).to_svg()))
    assert plus == overlay


# The first six rows of df: a different frame with df's column names.
_DF_HEAD = df.head(6)

# Each LD cell: the chart, its host alone, and the same layer attached to a
# host over the layer's own frame (``alone_host``), less that host's marks.
_LD: dict[
    str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart], Callable[[], fm.Chart], Any]
] = {
    "text-color-by-name": (
        lambda: mk("bar", x="g", y="y", color="g").layer(
            fm.Layer(data=df2, mark="text", encoding={"text": "y"})
        ),
        lambda: mk("bar", x="g", y="y", color="g"),
        lambda: mk("bar", df2, x="g", y="y", color="g"),
        layer("text", text="y"),
    ),
    "titled-text-color-by-name": (
        lambda: mk(
            "bar", x="g", y=fm.Y("y", title="Yield"), color=fm.Color("g", title="Group")
        ).layer(fm.Layer(data=df2, mark="text", encoding={"text": "y"})),
        lambda: mk("bar", x="g", y=fm.Y("y", title="Yield"), color=fm.Color("g", title="Group")),
        lambda: mk(
            "bar", df2, x="g", y=fm.Y("y", title="Yield"), color=fm.Color("g", title="Group")
        ),
        layer("text", text="y"),
    ),
    "grey-line-color-by-name": (
        lambda: P(color="g").layer(
            fm.Layer(data=df2, mark="line", mark_kwargs={"stroke": "#999999"})
        ),
        lambda: P(color="g"),
        lambda: P(df2, color="g"),
        fm.Layer(mark="line", mark_kwargs={"stroke": "#999999"}),
    ),
    # A slice of df keeps the size scale's union domain the host's own.
    "own-width-size-by-name": (
        lambda: P(size="z").layer(
            fm.Layer(data=_DF_HEAD, mark="line", encoding={"stroke_width": "r"})
        ),
        lambda: P(size="z"),
        lambda: P(_DF_HEAD, size="z"),
        layer("line", stroke_width="r"),
    ),
    # Over the host's own frame, so the color scale's union is the host's.
    "text-lowered-color": (
        lambda: bar_mean(color=mean(fm.Color)).layer(
            fm.Layer(data=df, mark="text", encoding={"text": fm.Text("y", aggregate="mean")})
        ),
        lambda: bar_mean(color=mean(fm.Color)),
        lambda: bar_mean(color=mean(fm.Color)),
        layer("text", text=fm.Text("y", aggregate="mean")),
    ),
}


@pytest.mark.parametrize("case", sorted(_LD))
def test_LD_a_data_layer_draws_its_inherited_bindings_as_inherited(case):
    """LD: a host binding read by name, or lowered, draws on the data layer as on the L door.

    The data layer's marks equal the same layer attached to the host over
    its own frame. Rejects carrying the inherited or lowered binding as the
    layer's own: the labels would take the palette, the grey line would
    split, the own-width line would refuse.
    """
    build, host, alone_host, attached = _LD[case]
    chart = _quiet(build)
    layer_alone = census(_quiet(lambda: alone_host().layer(attached).to_svg())) - census(
        _quiet(alone_host).to_svg()
    )
    assert census(_quiet(chart.to_svg)) == census(_quiet(host).to_svg()) + layer_alone
    assert "<html" in _quiet(lambda: chart.interactive().to_html()).lower()


@pytest.mark.parametrize("case", sorted(_LD))
def test_LD_a_data_layer_lists_the_tooltips_it_lists_on_the_L_door(case):
    """LD, interactive: the data layer's tooltip fields equal the same layer attached over its frame.

    Rejects counting an inherited color or the x/y fill as the layer's
    own (the text would list ``g`` and ``y``), dropping a no-own layer's
    inherited fields (the grey line would list only ``v`` and ``y``), and
    showing a renamed column under its internal name or a sibling channel's
    caller-written title.
    """
    build, host, alone_host, attached = _LD[case]
    attached_alone = _tooltip_batches(_quiet(lambda: alone_host().layer(attached)))
    expected = _tooltip_batches(_quiet(host)) + (
        attached_alone - _tooltip_batches(_quiet(alone_host))
    )
    assert _tooltip_batches(_quiet(build)) == expected


def _tooltip_names(chart: Any) -> list[list[str]]:
    """Per mark batch, the display names of its first mark's tooltip, in order."""
    return [
        [field["name"] for field in (batch.get("tooltips") or [{"fields": []}])[0]["fields"]]
        for panel in _scene(chart)["panels"]
        for batch in panel["marks"]
    ]


_TITLED_BAR = {"x": "g", "y": "y", "color": fm.Color("g", title="Group")}


@pytest.mark.parametrize(
    "door",
    ["L", "+", "+rev"],
)
def test_a_sibling_never_retitles_a_layers_tooltip(door):
    """A layer's tooltip names are the flat chart's, whatever sibling joins it.

    The bar's ``x`` lists ``g``, as flat; the color's own title titles the
    color channel alone. Rejects lending a caller's title to another
    channel's entry for the same column: the bar would list ``Group``.
    """
    flat = _tooltip_names(mk("bar", **_TITLED_BAR))[0]
    build = {
        "L": lambda: mk("bar", **_TITLED_BAR).layer(layer("rule", y="r")),
        "+": lambda: mk("bar", **_TITLED_BAR) + mk("rule", y="r"),
        "+rev": lambda: mk("rule", y="r") + mk("bar", **_TITLED_BAR),
    }[door]
    names = _tooltip_names(_quiet(build))
    assert (names[1] if door == "+rev" else names[0]) == flat


# A frame whose column only looks like a merge's rename.
_RENAME_SHAPED = pl.DataFrame(
    {"v": [1.0, 2.0, 3.0], "w__rhs_0001": [4.0, 6.0, 5.0], "r": [1.0, 2.0, 3.0]}
)


def _rename_shaped() -> fm.Chart:
    return mk("point", _RENAME_SHAPED, x="v", y="w__rhs_0001")


@pytest.mark.parametrize("door", ["L", "+", "+rev"])
def test_a_column_named_like_a_rename_keeps_its_name(door):
    """A tooltip restores a column's name only where a merge renamed it, never by its shape.

    Rejects reading the rename off the column's name: ``w__rhs_0001`` would
    show as ``w`` beside a sibling while the flat chart shows it whole.
    """
    flat = _tooltip_names(_rename_shaped())[0]
    assert flat == ["v", "w__rhs_0001"]
    rule = lambda: mk("rule", _RENAME_SHAPED, y="r")  # noqa: E731
    build = {
        "L": lambda: _rename_shaped().layer(layer("rule", y="r")),
        "+": lambda: _rename_shaped() + rule(),
        "+rev": lambda: rule() + _rename_shaped(),
    }[door]
    names = _tooltip_names(_quiet(build))
    assert (names[1] if door == "+rev" else names[0]) == flat


def test_a_no_own_layer_under_a_field_selection_keeps_its_fields_beside_the_selections():
    """GH #58 union over the effective list: the grey line lists g, v and y, not the selection's g alone."""
    selection = fm.selection_point(fields=["g"], name="pick")
    chart = (
        P(color="g")
        .layer(fm.Layer(mark="line", mark_kwargs={"stroke": "#999999"}))
        .add_selection(selection)
    )
    assert interactive_tooltip_fields(chart)[-1] == {"g", "v", "y"}


# Each layer's `inherited_channels` on the wire: exactly what it carries
# from a parent, never what it binds itself.
_PROVENANCE: dict[str, tuple[Callable[[], Any], list[list[str]]]] = {
    "operand-color-stays-own": (lambda: P(color="c") + line(), [[], []]),
    "composition-color-stays-chart-level": (
        lambda: (P() + line()).encode(color="c"),
        [[], []],
    ),
    # A text inherits no color: the host's or composition's color reaches
    # its statistic as a grouping only, never its wire.
    "host-label-left": (lambda: _host_label() + _rule(), [[], [], []]),
    "host-label-right": (lambda: _rule() + _host_label(), [[], [], []]),
    "composed-label-in-a-later-plus": (
        lambda: _composed_label() + _rule(),
        [["color"], [], []],
    ),
    # The data layer reads its own rows, so the host color names its own `g` from the chart level.
    "LD-color-by-name": (_LD["text-color-by-name"][0], [[], []]),
    "L-lowered-color": (_lowered_label, [[], []]),
    "own-color-beside-a-host-color": (
        lambda: P(color="c").layer(layer("line", color="g")) + _rule(),
        [[], [], []],
    ),
}


@pytest.mark.parametrize("case", sorted(_PROVENANCE))
def test_the_wire_lists_exactly_the_channels_a_layer_inherits(case):
    """On the wire, ``inherited_channels`` names a carried parent binding and no own one."""
    build, expected = _PROVENANCE[case]
    layers = _wire(_quiet(build))["layers"]
    assert [layer_.get("inherited_channels", []) for layer_ in layers] == expected


# ---------------------------------------------------------------------------
# Regression net: the remaining doors of the consistent cells, at their pinned counts
# ---------------------------------------------------------------------------

_NET_DOORS: dict[str, Case] = {
    "layer_aggregate/LCs": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g")), (bar_mean, lambda: P(df2, x="g"))
    ),
    "layer_aggregate/LCi": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g", y="w"), resolve={"y": "independent"}),
        (bar_mean, lambda: P(df2, x="g", y="w")),
    ),
    "layer_aggregate/LCx": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g")),
        (bar_mean, lambda: P(df2, x="g")),
        static=False,
    ),
    "composite_layer/LD": Case(
        lambda: _hist().layer(fm.Layer(data=df2, mark="tick", encoding={"x": "v"})),
        (_hist, lambda: mk("tick", df2, x="v")),
        marks={"rect": 4, "line": 6},
    ),
    "composite_layer/+d": Case(
        lambda: _hist() + mk("tick", df2, x="v"),
        (_hist, lambda: mk("tick", df2, x="v")),
        marks={"rect": 4, "line": 6},
    ),
    "composite_layer/+d-rev": Case(
        lambda: mk("tick", df2, x="v") + _hist(),
        (lambda: mk("tick", df2, x="v"), _hist),
        marks={"line": 6, "rect": 4},
    ),
    "composite_layer/SY": Case(
        lambda: _hist() + SY("r"), marks={"rect": 4, "polyline": 1}, chart_level=None
    ),
    "composite_layer/LCs": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v")),
        (_hist, lambda: mk("tick", df2, x="v")),
    ),
    "composite_layer/LCm": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v"))._build_merged(),
        (_hist, lambda: mk("tick", df2, x="v")),
    ),
    "composite_layer/LCi": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v", y="w"), resolve={"y": "independent"}),
        marks={"rect": 4, "line": 6},
    ),
    "composite_layer/LCx": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v")),
        (_hist, lambda: mk("tick", df2, x="v")),
        static=False,
    ),
    "composite_layer/Fc": Case(
        lambda: _hist().layer(layer("tick")).facet("c"),
        (lambda: _hist().facet("c"), lambda: mk("tick", x="v").facet("c")),
        faceted=True,
        chart_level=None,
    ),
    "span_no_statistic/+d-right-binds-none": Case(
        lambda: mk("rule", x="v", y="y", y2="z") + fm.Chart(df2).mark_rule(),
        (lambda: mk("rule", x="v", y="y", y2="z"), lambda: mk("rule", df2, x="v", y="y", y2="z")),
        marks={"line": 14},
    ),
    "facet_grain/F": Case(lambda: P().facet("c"), marks={"circle": 8}),
    "facet_grain/LD": Case(
        lambda: P().facet("g").layer(fm.Layer(data=df2, mark="tick")),
        marks={"circle": 8, "line": 6},
    ),
    "facet_grain/+d": Case(
        lambda: (P() + line(df2)).facet("g"),
        (lambda: P().facet("g"), lambda: line(df2).facet("g")),
        marks={"circle": 8, "polyline": 3},
    ),
    "facet_grain/+d-rev": Case(
        lambda: (line(df2) + P()).facet("g"),
        marks={"circle": 8, "polyline": 3},
    ),
    "span_composite/SY": Case(
        lambda: mk("errorbar", x="g", y="y") + SY("r"),
        marks={"line": 6, "polyline": 1},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_NET_DOORS))
def test_regression_net_doors(case):
    check_rendering(_NET_DOORS[case])


def test_a_layer_zero_binding_no_axis_is_refused_beside_an_aggregate():
    """``+rev`` with the right binding none: layer 0 owns the axes, so binding none it is refused."""
    with pytest.raises(ValueError, match=r"^layer 0: mark_point needs both positional channels"):
        (fm.Chart(df).mark_point() + bar_mean()).to_svg()


@pytest.mark.parametrize("channel", ["size", "opacity", "fill_opacity", "stroke_opacity"])
def test_a_later_layers_string_column_first_drawer_refuses_as_the_flat_chart_does(channel):
    """A later layer first drawing a numeric-only channel from a string column refuses alike."""
    with pytest.raises(ValueError) as flat:
        P(**{channel: "g"}).to_svg()
    with pytest.raises(ValueError) as later:
        (P() + P(y="r", **{channel: "g"})).to_svg()
    assert (
        str(later.value) == str(flat.value) == f"{channel}: column 'g' has unsupported dtype: Utf8"
    )


def _label_fills(svg: str) -> set[str]:
    """The fills of the text marks a chart draws inside its plot areas."""
    return {
        dict(signature[1]).get("fill")
        for signature in census(svg).elements()
        if signature[0] == "text"
    }


@pytest.mark.parametrize(
    "build",
    [_lowered_label, _LD["text-lowered-color"][0], _host_label],
    ids=["L-lowered", "LD-lowered", "L-raw"],
)
def test_a_text_layer_draws_no_inherited_color_however_it_was_lowered(build):
    """A text layer inheriting its host's color, raw or aggregated, draws its labels in the
    theme's text color, as a text layer under an uncolored host does.

    Rejects carrying the statistic-lowered color onto the text layer as its
    own: the labels would take the color scale.
    """
    uncolored = bar_mean().layer(layer("text", text=fm.Text("y", aggregate="mean")))
    assert _label_fills(_quiet(build).to_svg()) == _label_fills(uncolored.to_svg())


# ---------------------------------------------------------------------------
# A right operand's attached layer keeps its host's written axis
# ---------------------------------------------------------------------------

_RUG_HOSTS: dict[str, Callable[[], fm.Chart]] = {
    "histogram": lambda: fm.Chart(df).mark_histogram().encode(x="r"),
    "density": lambda: fm.Chart(df).mark_density().encode(x="r"),
    "boxplot": lambda: fm.Chart(df).mark_boxplot().encode(x="g", y="r"),
}


def _rugged(host: str) -> fm.Chart:
    return _RUG_HOSTS[host]().layer(layer("tick"))


def _tick_axes(chart: Any) -> dict[str, str | None]:
    """The x/y fields the attached tick (the last tick layer) draws, completed from the chart level."""
    wire = _wire(chart)
    tick = [layer_ for layer_ in wire["layers"] if layer_["mark"] == "tick"][-1]
    completed = {
        **{ch: v for ch, v in wire["encoding"].items() if "data_source" not in tick},
        **tick["encoding"],
    }
    return {
        ch: re.sub(r"__rhs_\d+$", "", completed[ch]["field"]) if ch in completed else None
        for ch in ("x", "y")
    }


# Layer 0 per host: its x reads like the host's (a boxplot's x is categorical).
_RUG_PARTNER_X = {"histogram": "v", "density": "v", "boxplot": "g"}

_RUG_DOORS: dict[str, Callable[[fm.Chart, str], Any]] = {
    "+s": lambda operand, x: P(x=x) + operand,
    "+d": lambda operand, x: P(df2, x=x) + operand,
    "+rev": lambda operand, x: operand + P(x=x),
    "+drev": lambda operand, x: operand + P(df2, x=x),
    "LCm": lambda operand, x: fm.LayerChart(P(x=x), operand)._build_merged(),
    "LCx": lambda operand, x: fm.LayerChart(P(x=x), operand),
}
# The doors whose layer 0 is the rugged operand, so its axis fallback is its own.
_RUG_OPERAND_FIRST = frozenset({"+rev", "+drev"})


def _line_xs_static(svg: str) -> list[float]:
    """The x1 of every line drawn in the plot areas: tick and rule marks."""
    root = _svg_root(svg)
    return [
        float(e.get("x1"))
        for group in _clip_groups(root)
        for e in group.iter()
        if _tag(e) == "line"
    ]


def _attached_tick_xs_interactive(chart: Any) -> list[float]:
    """The x of every node of the attached tick (the last tick batch) in the interactive scene."""
    batches = [
        batch
        for panel in _scene(chart)["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "tick"
    ]
    return [float(node["x1"]) for node in batches[-1]["nodes"]]


def _drawn_statically(xs: list[float], static: list[float]) -> bool:
    """Whether every x in *xs* is a distinct static line's x (SVG rounds to 3 decimals)."""
    pool = list(static)
    for x in xs:
        match = next((i for i, s in enumerate(pool) if abs(s - x) < 0.01), None)
        if match is None:
            return False
        pool.pop(match)
    return True


@pytest.mark.parametrize("door", sorted(_RUG_DOORS))
@pytest.mark.parametrize("host", sorted(_RUG_HOSTS))
def test_a_right_operands_attached_layer_draws_its_hosts_written_axis(host, door):
    """The tick attached to a right operand draws at its host's ``x``, as alone.

    A channel the host never wrote takes layer 0's (the axis fallback).
    Rejects filling the right operand's attached layer from the
    composition's written bindings: the tick would draw at layer 0's ``x``.
    The boxplot, whose tick reads its host's rows at the tail, is the control.
    """
    chart = _quiet(lambda: _RUG_DOORS[door](_rugged(host), _RUG_PARTNER_X[host]))
    alone = _tick_axes(_rugged(host))
    if door in _RUG_OPERAND_FIRST:
        expected = alone
    else:
        fallback = {"x": _RUG_PARTNER_X[host], "y": "y"}
        expected = {ch: alone[ch] if alone[ch] is not None else fallback[ch] for ch in ("x", "y")}
    merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    assert _tick_axes(merged) == expected
    # Interactive: the attached ticks land where the same door draws them
    # statically (the static overlay for LCx); a tick drawn at layer 0's x
    # lands where no static line is.
    ticks = _attached_tick_xs_interactive(chart)
    assert ticks and _drawn_statically(ticks, _line_xs_static(_quiet(chart.to_svg)))
    assert "<html" in _quiet(lambda: chart.interactive().to_html()).lower()


# ---------------------------------------------------------------------------
# Figure functions composed with ``+`` render byte-identical output
# ---------------------------------------------------------------------------

_FIG_RNG = np.random.default_rng(0)
_FIG_X = _FIG_RNG.uniform(0, 10, 40)
_FIG_DF = pl.DataFrame(
    {
        "x": _FIG_X,
        "y": 2 * _FIG_X + _FIG_RNG.normal(0, 2, 40),
        "g": ["a", "b"] * 20,
        "h": ["p", "p", "q", "q"] * 10,
        "s": _FIG_RNG.uniform(1, 5, 40),
    }
)
_FIG_MAT = pl.DataFrame(np.arange(12, dtype=float).reshape(3, 4), schema=["w", "x", "y", "z"])

_FIGURES: dict[str, Callable[[], Any]] = {
    "lmplot/plain": lambda: fm.lmplot(_FIG_DF, x="x", y="y"),
    "lmplot/hue": lambda: fm.lmplot(_FIG_DF, x="x", y="y", hue="g"),
    "lmplot/hue-size": lambda: fm.lmplot(_FIG_DF, x="x", y="y", hue="g", size="s"),
    "lmplot/loess": lambda: fm.lmplot(_FIG_DF, x="x", y="y", method="loess"),
    "lmplot/order2": lambda: fm.lmplot(_FIG_DF, x="x", y="y", order=2),
    "lmplot/x_bins": lambda: fm.lmplot(_FIG_DF, x="x", y="y", x_bins=4),
    "lmplot/scatter-false": lambda: fm.lmplot(_FIG_DF, x="x", y="y", scatter=False),
    "lmplot/robust-hue": lambda: fm.lmplot(_FIG_DF, x="x", y="y", method="robust", hue="g"),
    "lmplot/x_bins-hue": lambda: fm.lmplot(_FIG_DF, x="x", y="y", x_bins=4, hue="g"),
    "regplot/hue": lambda: fm.regplot(_FIG_DF, x="x", y="y", hue="g"),
    "lmplot/col-hue": lambda: fm.lmplot(_FIG_DF, x="x", y="y", col="h", hue="g"),
    "heatmap/annot": lambda: fm.heatmap(_FIG_MAT),
    "heatmap/annot-id": lambda: fm.heatmap(
        pl.DataFrame({"r": ["a", "b", "c"], "c1": [1.0, 2.0, 3.0], "c2": [4.0, 5.0, 6.0]})
    ),
}

# The SVG each figure renders.
_FIGURE_DIGESTS = {
    "lmplot/plain": "518c81c4c53bf1c9fadb0bc97d7adb390a5c97221c96391088f4c269286a8a75",
    "lmplot/hue": "f3ac385b78b66370f3426db7f3ad73f3d7d957de5c19fafbeadb2e6f3d050a55",
    "lmplot/hue-size": "63dd4f59a57794f3befcc121b50dbfcfc195b3829ca50b535a47b112c327129a",
    "lmplot/loess": "2fad5879a05738cd86da4128b4386309143f506ea73b0ffc6cca186e10f7107c",
    "lmplot/order2": "048830ead8c0029e013785a496e99abfd1cb89eed2084449c98f4ad498c8b961",
    "lmplot/x_bins": "d9ad14d881455d715013c30a08c88de1977cba1bb56f2aaef6519110da44e359",
    "lmplot/scatter-false": "e931ecef99739b21d5e19843ab325d867a1bbaabfbf596b1b1b4a4197134c183",
    "lmplot/robust-hue": "e4221df2adb38c4023b8846e1117a424d7aa0738cd0ebbd8bfacad4ce6a49b20",
    "lmplot/x_bins-hue": "9aee24538af3407953e91bac6949e91a21300421dcfcdc7cf2523aaaadbe979a",
    "regplot/hue": "f3ac385b78b66370f3426db7f3ad73f3d7d957de5c19fafbeadb2e6f3d050a55",
    "lmplot/col-hue": "4760e91c0648f68bbe436de48d991032fd31d992aebb5d04b720b374d8c1c104",
    "heatmap/annot": "23573f7dde2ca27786ff0e9168ebadf113ccaff2e8aa217eecea8b7cf3eb59a3",
    "heatmap/annot-id": "9c55288da7d73a05c6df4aea8752411e48f49b9215e6e7329a5d1d3eeadfb9e7",
}


# The interactive scene each hue figure draws.
_FIGURE_SCENE_DIGESTS = {
    "lmplot/hue": "de4baef90f117d8f893cd369c14229890eef23897bf8d9277e5017b80bd169cb",
    "lmplot/hue-size": "75235c3490d8f87598cfbc19a2cb3835c4bec5c72940cf473f29b05ac40206ef",
    "lmplot/robust-hue": "9828e396f1cfd7e3e85a42b5352ad38ab06546550e2edf45386fe278e1b3fc40",
    "lmplot/x_bins-hue": "03446a41b512f2d3eb720bcc9fcbedf2053981cf782f13e21de79a9aaebdd46b",
    "regplot/hue": "de4baef90f117d8f893cd369c14229890eef23897bf8d9277e5017b80bd169cb",
    "lmplot/col-hue": "b19963e854966e2b994caf12ffb1dcf215656327697fe8ef2d02c445eddc25be",
}


@pytest.mark.parametrize("case", sorted(_FIGURE_SCENE_DIGESTS))
def test_a_hue_figures_interactive_scene_renders_its_pinned_bytes(case):
    import hashlib

    scene = _quiet(lambda: _FIGURES[case]().interactive().scene_json)
    assert hashlib.sha256(scene.encode()).hexdigest() == _FIGURE_SCENE_DIGESTS[case]


@pytest.mark.parametrize("case", sorted(_FIGURES))
def test_a_figure_composed_with_plus_renders_its_pinned_bytes(case):
    """A figure's fit binds the hue it shares with the scatter itself, so ``+`` draws it unchanged.

    Rejects composing without the fit's own hue: each ``hue=`` fit would
    draw one pooled, uncolored line.
    """
    import hashlib

    svg = _quiet(lambda: _FIGURES[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _FIGURE_DIGESTS[case]


# ---------------------------------------------------------------------------
# One statistic plan per layer over a positional bin
# ---------------------------------------------------------------------------


def _row_bins(data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    """``x=bin(v)`` beside a raw ``y``: no aggregate, so each row is placed at its bin."""
    return mk("point", data, **{"x": fm.X("v", bin=True), "y": "y", **encoding})


def _two_means(mark: str = "bar", **encoding: Any) -> fm.Chart:
    """``x=bin(v)`` beside two distinct aggregates: one mark per bin carrying both."""
    return mk(
        mark,
        **{"x": fm.X("v", bin=True), "y": "mean(y)", "color": mean(fm.Color), **encoding},
    )


_ROW_BIN: dict[str, Case] = {
    "F": Case(_row_bins, marks={"circle": 8}),
    "F-y": Case(lambda: mk("point", x="v", y=fm.Y("y", bin=True)), marks={"circle": 8}),
    "F-color": Case(lambda: _row_bins(color="c"), marks={"circle": 8}, legends={"c": ("p", "q")}),
    "L-inherits-the-bin": Case(
        lambda: _row_bins().layer(layer("tick")),
        (_row_bins, lambda: mk("tick", x=fm.X("v", bin=True), y="y")),
        marks={"circle": 8, "line": 8},
        chart_level=None,
    ),
    "L-own-bin": Case(
        lambda: P().layer(layer("point", x=fm.X("v", bin=True))),
        (P, _row_bins),
        marks={"circle": 16},
        chart_level=None,
    ),
    # Every row of both operands: the oracle alone would pass a count bin
    # grouped by the raw `y` on both sides.
    **{
        door: replace(case, marks={"circle": 16})
        for door, case in _cases("+s", plus_doors(_row_bins, P)).items()
    },
    "Fc": Case(lambda: _row_bins().facet("c"), marks={"circle": 8}),
}

_BIN_TWO_AGGREGATES: dict[str, Case] = {
    "F": Case(_two_means, marks={"rect": 4}, legends=None),
    "L": Case(
        lambda: _two_means().layer(layer("point")),
        (_two_means, lambda: _two_means("point")),
        chart_level=None,
    ),
    "+s": Case(lambda: _two_means() + P(), (_two_means, P)),
    "+s-rev": Case(lambda: P() + _two_means(), (P, _two_means)),
}


def _binned_count(data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    return _binned(data, **encoding)


_COUNTING_BIN: dict[str, Case] = {
    "F": Case(_binned_count, marks={"rect": 4}),
    "F-grouped-by-color": Case(
        lambda: _binned_count(color="c"), marks={"rect": 8}, legends={"c": ("p", "q")}
    ),
    "L-inherits": Case(
        lambda: _binned_count().layer(layer("point")),
        (_binned_count, lambda: mk("point", x=fm.X("v", bin=True), y="count()")),
        chart_level=None,
    ),
    "L-label": Case(
        lambda: _binned_count().layer(layer("text", text="count")),
        (_binned_count, lambda: mk("text", x=fm.X("v", bin=True), y="count()", text="count")),
        chart_level=None,
    ),
    "+s": Case(lambda: _binned_count() + P(), (_binned_count, P)),
    "+d": Case(lambda: _binned_count() + P(df2), (_binned_count, lambda: P(df2))),
    "+r": Case(lambda: P() + _binned_count(), (P, _binned_count)),
    **_cases("LC", plus_doors(_binned_count, P)),
    "Fc": Case(lambda: _binned_count().facet("c"), marks={"rect": 6}),
}

_COUNTING_BIN_LABEL: dict[str, Case] = {
    "F-tooltip": Case(lambda: _binned_count(tooltip=mean(fm.Tooltip)), marks={"rect": 4}),
    "L-text": Case(
        lambda: _binned_count().layer(layer("text", text=mean(fm.Text))),
        (
            _binned_count,
            lambda: mk("text", x=fm.X("v", bin=True), y="count()", text=mean(fm.Text)),
        ),
        chart_level=None,
    ),
}

_BIN_CELLS = {
    **_cases("row_bin", _ROW_BIN),
    **_cases("counting_bin", _COUNTING_BIN),
    **_cases("counting_bin_label", _COUNTING_BIN_LABEL),
    **_cases("bin_two_aggregates", _BIN_TWO_AGGREGATES),
}


@pytest.mark.parametrize("case", sorted(_BIN_CELLS))
def test_a_binned_layer_computes_its_own_statistic_on_every_door(case):
    """Each layer bins at its own grain: per row without an aggregate, per bin with them.

    Rejects a count bin grouped by the raw fields (a raw ``y`` collapsed into
    counts) and the refusal of a second aggregating channel: each layer draws
    what it draws alone, on every door.
    """
    check_rendering(_BIN_CELLS[case])
    check_wire_seam(_BIN_CELLS[case])


def _svg_and_titles(chart: Any) -> tuple[str, list[str]]:
    svg = _quiet(chart.to_svg)
    return svg, axis_titles(svg)


@pytest.mark.parametrize(
    ("encoded", "transformed"),
    [
        pytest.param(
            _row_bins,
            lambda: (
                fm.Chart(df)
                .transform(fm.Bin("v", per_row=True))
                .mark_point()
                .encode(x=fm.X("bin_start", title="v"), x2=fm.X2("bin_end", type="Q"), y="y")
            ),
            id="x",
        ),
        pytest.param(
            lambda: mk("point", x="v", y=fm.Y("y", bin=True), color="c"),
            lambda: (
                fm.Chart(df)
                .transform(fm.Bin("y", per_row=True))
                .mark_point()
                .encode(
                    x="v", y=fm.Y("bin_start", title="y"), y2=fm.Y2("bin_end", type="Q"), color="c"
                )
            ),
            id="y-colored",
        ),
    ],
)
def test_a_positional_bin_without_an_aggregate_places_each_row_at_its_bin(encoded, transformed):
    """Every row is kept and drawn at its bin's start, with ``x2`` at the bin's end.

    The encoding route renders the bytes the explicit per-row ``fm.Bin``
    renders, so the edges are the bin transform's own. Rejects an implicit
    count per bin, a count grouped by the raw fields and edges computed
    outside the bin transform.
    """
    svg, titles = _svg_and_titles(encoded())
    assert svg == _quiet(transformed().to_svg)
    assert tag_counts(svg)["circle"] == df.height
    assert "bin_start" not in titles and "bin_end" not in titles


def test_a_row_binned_layer_keeps_its_raw_rows_unchanged():
    """The per-row bin moves no row on its unbinned axis: the points sit at the raw ``y``."""
    cy = re.compile(r'<circle [^>]*cy="([-\d.]+)"')
    assert sorted(cy.findall(_quiet(_row_bins().to_svg))) == sorted(cy.findall(_quiet(P().to_svg)))


def test_a_pre_bound_span_channel_is_kept():
    encoding = _wire(mk("bar", x=fm.X("v", bin=True), x2="r", y="y"))["encoding"]
    assert encoding["x"]["field"] == "bin_start"
    assert encoding["x2"]["field"] == "r"


def _bin_steps(chart: Any) -> list[dict]:
    """Every Bin step on the wire, including a layer's named or branched one."""
    steps = []
    for step in _wire(chart).get("transforms") or []:
        steps += [step] if step["type"] == "bin" else step.get("transforms", [])
    return [step for step in steps if step["type"] == "bin"]


def test_two_aggregates_beside_a_bin_draw_what_the_explicit_bin_draws():
    """``bar(x=bin(v), y=mean(y), color=mean(z))`` is one Bin computing both, per bin.

    Rejects the refusal of a second aggregating channel: the encoding route
    renders the bytes ``fm.Bin(aggregates=...)`` renders, titled by the
    source field and the aggregates' output names.
    """
    transformed = (
        fm.Chart(df)
        .transform(
            fm.Bin(
                "v",
                aggregates=[
                    fm.AggregateOp("y", "mean", "mean_y"),
                    fm.AggregateOp("z", "mean", "mean_z"),
                ],
            )
        )
        .mark_bar()
        .encode(
            x=fm.X("bin_start", title="v"),
            x2=fm.X2("bin_end", type="Q"),
            y="mean_y",
            color="mean_z",
        )
    )
    svg, titles = _svg_and_titles(_two_means())
    assert svg == _quiet(transformed.to_svg)
    assert sorted(titles) == ["mean_y", "v"]
    assert list(legend_set(svg)) == ["mean_z"]


#: Two bins over [0, 8): v < 4 holds rows 0-3, v >= 4 rows 4-7.
_HALVES = {"bin_count": 2, "extent": (0.0, 8.0), "nice": False}


def test_each_bin_carries_both_of_its_own_means():
    """Each bin's label reads its own mean ``z`` beside its mean ``y`` and ``r``."""
    chart = mk(
        "text",
        x=fm.X("v", bin=_HALVES),
        y="mean(y)",
        text=mean(fm.Text),
        tooltip=mean(fm.Tooltip, "r"),
    )
    halves = [df.filter(pl.col("v") < 4.0), df.filter(pl.col("v") >= 4.0)]
    texts = [
        "".join(e.itertext()).strip()
        for e in _svg_root(_quiet(chart.to_svg)).iter()
        if _tag(e) == "text" and e.get("fill") != _CHROME_TEXT and e.get("font-weight") is None
    ]
    assert texts[-2:] == [f"{half['z'].mean():g}" for half in halves]
    (step,) = _bin_steps(chart)
    assert step["aggregates"] == [
        {"field": "y", "fn": "mean", "as": "mean_y"},
        {"field": "z", "fn": "mean", "as": "mean_z"},
        {"field": "r", "fn": "mean", "as": "mean_r"},
    ]


@pytest.mark.parametrize(
    ("build", "aggregates"),
    [
        pytest.param(
            lambda: _two_means(tooltip=mean(fm.Tooltip, "r")),
            [("y", "mean", "mean_y"), ("z", "mean", "mean_z"), ("r", "mean", "mean_r")],
            id="F-tooltip-third",
        ),
        pytest.param(
            lambda: _two_means(tooltip=mean(fm.Tooltip, "y")),
            [("y", "mean", "mean_y"), ("z", "mean", "mean_z")],
            id="F-same-op-twice",
        ),
        pytest.param(
            lambda: _two_means().layer(layer("point", tooltip=mean(fm.Tooltip, "r"))),
            [("y", "mean", "mean_y"), ("z", "mean", "mean_z")],
            id="L-host",
        ),
        pytest.param(
            lambda: P() + _two_means(tooltip=mean(fm.Tooltip, "r")),
            [("y", "mean", "mean_y"), ("z", "mean", "mean_z"), ("r", "mean", "mean_r")],
            id="+s",
        ),
    ],
)
def test_a_bin_computes_one_op_per_distinct_field_and_op(build, aggregates):
    """Every distinct (field, op) beside the bin is one ``aggregates`` entry; a repeat is one op."""
    steps = _bin_steps(_quiet(build))
    assert [
        (entry["field"], entry["fn"], entry["as"]) for entry in steps[0]["aggregates"]
    ] == aggregates
    assert all("agg_field" not in step and "agg" not in step for step in steps)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: mk("bar", x=fm.X("v", bin=True), y="mean(y)"), id="F"),
        pytest.param(
            lambda: mk("bar", x=fm.X("v", bin=True), y="mean(y)", tooltip=mean(fm.Tooltip, "y")),
            id="F-same-op-twice",
        ),
        pytest.param(
            lambda: P() + mk("bar", x=fm.X("v", bin=True), y="mean(y)"),
            id="+s",
        ),
    ],
)
def test_one_aggregate_beside_a_bin_keeps_the_agg_field_wire(build):
    """One distinct aggregate folds through ``agg_field``/``agg`` and draws ``agg_value``."""
    (step,) = _bin_steps(_quiet(build))
    assert (step["agg_field"], step["agg"]) == ("y", "mean")
    assert "aggregates" not in step and "per_row" not in step


def test_an_aggregate_beside_a_bin_is_titled_by_its_output_name():
    """On the positional axis, the aggregate's axis reads ``mean_y``, the bin's ``v``."""
    _, titles = _svg_and_titles(mk("bar", x=fm.X("v", bin=True), y="mean(y)"))
    assert sorted(titles) == ["mean_y", "v"]


def test_a_grouped_bin_counts_each_group_into_the_pooled_edges():
    """A color beside a counting bin groups it, over one set of edges (``shared_extent``)."""
    transformed = (
        fm.Chart(df)
        .transform(fm.Bin("v", groupby=["c"], shared_extent=True))
        .mark_bar()
        .encode(x=fm.X("bin_start", title="v"), x2=fm.X2("bin_end", type="Q"), y="count", color="c")
    )
    assert _quiet(_binned_count(color="c").to_svg) == _quiet(transformed.to_svg)


# The SVG bytes each bin chart the statistic plan leaves unchanged renders: a
# counting bin, a grouped counting bin and a title-less value-channel fold.
_BIN_CONTROLS: dict[str, Callable[[], Any]] = {
    "F/count": _binned_count,
    "F/count-grouped": lambda: _binned_count(color="c"),
    "F/count-faceted": lambda: _binned_count().facet("c"),
    "F/tooltip-fold": lambda: _binned_count(tooltip=mean(fm.Tooltip)),
    "L/count-point": lambda: _binned_count().layer(layer("point")),
    "+s/count": lambda: _binned_count() + P(),
}

_BIN_CONTROL_DIGESTS = {
    "F/count": "b4913a77287da34ca2337f02819e0440b9eaa47957020fd20585a48a5a1250fb",
    "F/count-grouped": "523e831ec8872260fa2420632fd3a0dbd5d0bf4fb350da0a3af9be04722fbd54",
    "F/count-faceted": "40d4b34dc2cddc98a409ce0384ad1ae26291acdc0649a479d2c2422bf9c629fb",
    "F/tooltip-fold": "cf8d948f33f7343eb922c8ca7b775152ed8bdb4dc2c3302b0868bd7e04a92027",
    "L/count-point": "4e9f129c631c217484d9ecd0c0bbdd67c499495ff8340ab8e87b7beb4cd6a5b4",
    "+s/count": "c38f6416ccde1da922f83fd900f08a044b104a05017b34cc0801752162d06edc",
}


@pytest.mark.parametrize("case", sorted(_BIN_CONTROLS))
def test_a_counting_bin_renders_its_pinned_bytes(case):
    import hashlib

    svg = _quiet(lambda: _BIN_CONTROLS[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _BIN_CONTROL_DIGESTS[case]


# ---------------------------------------------------------------------------
# A Bin's own statistic parameters, on both spellings and every door
# ---------------------------------------------------------------------------

#: Each statistic parameter a Bin can carry itself, with the output column a
#: channel draws to read it.
_BIN_STATISTICS: dict[str, tuple[Callable[[], dict], str, str]] = {
    "groupby": (lambda: {"groupby": ["c"]}, "count", "groupby"),
    "cumulative": (lambda: {"cumulative": True}, "count", "cumulative"),
    "agg_field/agg": (lambda: {"agg_field": "z", "agg": "mean"}, "agg_value", "agg_field/agg"),
    "aggregates": (
        lambda: {"aggregates": [fm.AggregateOp("z", "mean", "mean_z")]},
        "mean_z",
        "aggregates",
    ),
}
_SPELLINGS = {"pre-built": "bin=Bin(...)", "dict": "bin={...}"}


def _declared_bin(param: str, spelling: str) -> Any:
    kwargs = _BIN_STATISTICS[param][0]()
    return fm.Bin("v", **kwargs) if spelling == "pre-built" else kwargs


def _bin_of(param: str, spelling: str, mark: str, **encoding: Any) -> Callable[[], fm.Chart]:
    return lambda: mk(mark, **{"x": fm.X("v", bin=_declared_bin(param, spelling)), **encoding})


def _declared_doors(
    build: Callable[[], fm.Chart], partner: Callable[[], fm.Chart] = P
) -> dict[str, Callable[[], Any]]:
    """Every door a binned chart is written on: flat, attached, composed, overlaid, faceted."""
    return {
        "F": build,
        "L": lambda: build().layer(layer("point")),
        "+s": lambda: build() + partner(),
        "+rev": lambda: partner() + build(),
        "LCs": lambda: fm.LayerChart(build(), partner()),
        "LCm": lambda: fm.LayerChart(build(), partner())._build_merged(),
        "LCi": lambda: fm.LayerChart(build(), partner(), resolve={"y": "independent"}),
        "LCx": lambda: fm.LayerChart(build(), partner()).interactive(),
        "Fc": lambda: build().facet("c"),
    }


def _render(chart: Any) -> str:
    return chart.to_svg() if hasattr(chart, "to_svg") else chart.to_html()


def _undrawn_statistic(param: str, spelling: str) -> str:
    # The columns this Bin computes: `agg_value` only under agg_field/agg,
    # each `aggregates` entry's name under aggregates.
    outputs = sorted(
        ["count", "density"]
        + (["agg_value"] if param == "agg_field/agg" else [])
        + (["mean_z"] if param == "aggregates" else [])
    )
    written = _SPELLINGS[spelling]
    return (
        f"x=X('v', {written}) declares {param}, but no channel draws the Bin's statistic: "
        f"'y' draws raw rows, so each row would be placed at its bin and {param} dropped. "
        f"Bind a channel to one of the Bin's output columns "
        f"({', '.join(repr(name) for name in outputs)}), or remove {param} from {written} "
        "to place each row at its bin."
    )


_UNDRAWN = {
    f"{param}/{spelling}/{door}": (param, spelling, build)
    for param in _BIN_STATISTICS
    for spelling in _SPELLINGS
    for door, build in _declared_doors(_bin_of(param, spelling, "point", y="y")).items()
}


@pytest.mark.parametrize("case", sorted(_UNDRAWN))
def test_a_bin_statistic_no_channel_draws_is_refused_naming_the_parameter(case):
    """Beside raw rows a Bin's own statistic would be dropped, so it is refused on every door.

    The message names the user's spelling, the parameter, the channel drawing
    raw rows and the columns to bind. Rejects letting the per-row placement
    reach Rust's refusal of a parameter the user never wrote, and rejects
    grouping the raw ``y`` into a statistic nothing draws.
    """
    param, spelling, build = _UNDRAWN[case]
    with pytest.raises(ValueError) as exc:
        _quiet(lambda: _render(build()))
    assert str(exc.value) == _undrawn_statistic(param, spelling)


def test_three_raw_axes_join_with_the_oxford_comma():
    """Three channels drawing raw rows beside an undrawn Bin statistic join with
    the Oxford comma too, not a bare ``and`` chain."""
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("v", bin=fm.Bin("v", groupby=["c"])),
            y="y",
            size="z",
            opacity="r",
        )
    )
    with pytest.raises(ValueError) as exc:
        _quiet(lambda: _render(chart))
    assert str(exc.value) == (
        "x=X('v', bin=Bin(...)) declares groupby, but no channel draws the Bin's "
        "statistic: 'y', 'size', and 'opacity' draw raw rows, so each row would be "
        "placed at its bin and groupby dropped. Bind a channel to one of the Bin's "
        "output columns ('count', 'density'), or remove groupby from bin=Bin(...) to "
        "place each row at its bin."
    )


def test_a_bin_with_three_statistic_parameters_joins_them_with_the_oxford_comma():
    """A Bin declaring groupby, cumulative and aggregates together joins all three with
    the Oxford comma, both in the declared-parameters clause and in its removal
    instruction, the same prose-list rule as every other 3+-item refusal."""
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X(
                "v",
                bin=fm.Bin(
                    "v",
                    groupby=["c"],
                    cumulative=True,
                    aggregates=[fm.AggregateOp("z", "mean", "mean_z")],
                ),
            ),
            y="y",
        )
    )
    with pytest.raises(ValueError) as exc:
        _quiet(lambda: _render(chart))
    assert str(exc.value) == (
        "x=X('v', bin=Bin(...)) declares groupby, cumulative, and aggregates, but no "
        "channel draws the Bin's statistic: 'y' draws raw rows, so each row would be "
        "placed at its bin and groupby, cumulative, and aggregates dropped. Bind a "
        "channel to one of the Bin's output columns ('count', 'density', 'mean_z'), or "
        "remove groupby, cumulative, and aggregates from bin=Bin(...) to place each row "
        "at its bin."
    )


#: Rects the flat chart draws: (output drawn, output plus a raw color).
_DECLARED_RECTS = {
    "groupby": (6, 32),
    "cumulative": (4, 28),
    "agg_field/agg": (4, 8),
    "aggregates": (4, 8),
}


def _drawn_cases() -> dict[str, Case]:
    out: dict[str, Case] = {}
    for param, (_, column, _) in _BIN_STATISTICS.items():
        for shape, extra in (("output", {}), ("output+color", {"color": "y"})):
            for spelling in _SPELLINGS:
                # Titled, since a drawn `agg_value` would otherwise title its axis.
                y = fm.Y(column, title="per bin")
                build = _bin_of(param, spelling, "bar", y=y, **extra)
                rects = _DECLARED_RECTS[param][bool(extra)]
                doors = {
                    "F": Case(build, marks={"rect": rects}),
                    "L": Case(
                        lambda build=build: build().layer(layer("point")),
                        (build, _bin_of(param, spelling, "point", y=y, **extra)),
                        chart_level=None,
                    ),
                    **plus_doors(build, P),
                    "Fc": Case(lambda build=build: build().facet("c")),
                }
                out.update(_cases(f"{param}/{shape}/{spelling}", doors))
    return out


_DECLARED_DRAWN = _drawn_cases()


@pytest.mark.parametrize("case", sorted(_DECLARED_DRAWN))
def test_a_bin_statistic_a_channel_draws_renders_on_every_door(case):
    """A channel drawing the Bin's declared output makes the Bin aggregate, on every door.

    An ``aggregates`` entry's output name is a declared column like
    ``count``, never a group key, and a raw color groups the Bin after its
    own groupby. Rejects reading only the registry's counting columns.
    """
    check_rendering(_DECLARED_DRAWN[case])
    check_wire_seam(_DECLARED_DRAWN[case])


@pytest.mark.parametrize("param", sorted(_BIN_STATISTICS))
@pytest.mark.parametrize("color", [False, True], ids=["output", "output+color"])
def test_a_pre_built_bin_draws_what_the_same_dict_draws(param, color):
    """Both spellings of one Bin are read alike: the same bytes, the same groupby."""
    column = _BIN_STATISTICS[param][1]
    extra = {"color": "y"} if color else {}
    pre, spelled = (_bin_of(param, s, "bar", y=column, **extra)() for s in _SPELLINGS)
    assert _quiet(pre.to_svg) == _quiet(spelled.to_svg)
    expected = (["c"] if param == "groupby" else []) + (["y"] if color else [])
    for chart in (pre, spelled):
        (step,) = _bin_steps(chart)
        assert step.get("groupby", []) == expected
        assert step["shared_extent"] is color


_OWN_AND_CHANNEL = {
    f"{param}/{door}": (param, build)
    for param in ("agg_field/agg", "aggregates")
    for door, build in _declared_doors(
        _bin_of(param, "dict", "bar", y=fm.Y("y", aggregate="sum"))
    ).items()
}


@pytest.mark.parametrize("case", sorted(_OWN_AND_CHANNEL))
def test_a_bins_own_aggregate_beside_a_channel_aggregate_is_refused(case):
    """The per-bin statistic is declared once, on the channels or on the Bin.

    Rejects replacing the Bin's own aggregate with the channel's silently.
    """
    param, build = _OWN_AND_CHANNEL[case]
    with pytest.raises(ValueError) as exc:
        _quiet(lambda: _render(build()))
    assert str(exc.value) == (
        f"x=X('v', bin={{...}}) declares its own {param} and channel 'y' carries "
        f"aggregate='sum': declare the per-bin statistic in one place. Drop {param} from "
        "bin={...} to compute the channel's aggregate per bin, or drop aggregate= and bind "
        "'y' to the Bin's output column."
    )


@pytest.mark.parametrize(
    ("bin_", "y", "groupby"),
    [
        pytest.param({"agg_field": "z", "agg": "mean"}, "count()", [], id="own-agg-beside-count"),
        pytest.param({"groupby": ["c"]}, "mean(y)", ["c"], id="own-groupby-beside-mean"),
        pytest.param({"cumulative": True}, "mean(y)", [], id="own-cumulative-beside-mean"),
    ],
)
def test_a_bins_own_parameter_beside_a_channel_count_or_aggregate_is_kept(bin_, y, groupby):
    """A count() reads the Bin's count beside its own aggregate; own groupby stays beside a mean."""
    chart = mk("bar", x=fm.X("v", bin=bin_), y=y)
    (step,) = _bin_steps(chart)
    assert step.get("groupby", []) == groupby
    assert all(step.get(k) == v for k, v in bin_.items() if k != "groupby")
    assert tag_counts(_quiet(chart.to_svg))["rect"] > 0


_UNREAD_AGG_VALUE = {
    f"{param}/{spelling}/{door}": (param, spelling, build)
    for param in ("groupby", "cumulative", "aggregates")
    for spelling in _SPELLINGS
    for door, build in _declared_doors(_bin_of(param, spelling, "bar", y="agg_value")).items()
    if door in ("F", "L")
}


@pytest.mark.parametrize("case", sorted(_UNREAD_AGG_VALUE))
def test_agg_value_beside_a_bin_computing_none_is_refused_as_an_undrawn_statistic(case):
    """A Bin without agg_field/agg computes no ``agg_value``, so ``y="agg_value"`` reads raw rows.

    The refusal lists only the columns this Bin computes. Rejects reading a
    Bin's columns from its kind's table, which lists ``agg_value`` for every
    Bin and sends the user to a column the Bin never emits.
    """
    param, spelling, build = _UNREAD_AGG_VALUE[case]
    with pytest.raises(ValueError) as exc:
        _quiet(lambda: _render(build()))
    assert str(exc.value) == _undrawn_statistic(param, spelling)


def _named_bin(spelling: str, name: str = "foo") -> Any:
    return fm.Bin("v", name=name) if spelling == "pre-built" else {"name": name}


_NAMED_ROUTES: dict[str, dict[str, Any]] = {
    "count": {"y": "count"},
    "count+color": {"y": "count", "color": "c"},
    "raw-y": {"y": "y"},
}

_NAMED = {
    f"{spelling}/{route}/{door}": (spelling, build)
    for spelling in _SPELLINGS
    for route, encoding in _NAMED_ROUTES.items()
    for door, build in _declared_doors(
        lambda spelling=spelling, encoding=encoding: mk(
            "bar", x=fm.X("v", bin=_named_bin(spelling)), **encoding
        )
    ).items()
}
# An empty name is a name too: refused alike, on every route.
_NAMED |= {
    f"{spelling}/empty-name/{door}": (spelling, build)
    for spelling in _SPELLINGS
    for door, build in _declared_doors(
        lambda spelling=spelling: mk("bar", x=fm.X("v", bin=_named_bin(spelling, "")), y="count")
    ).items()
}


def _named_refusal(written: str, spelling: str, name: str = "foo") -> str:
    return (
        f"{written} names its Bin {name!r}, but an encoding bin is computed for its own layer "
        "and published to no other layer, so name= would be ignored. "
        f"Remove name= from {spelling}; to publish the binned rows under a name another "
        f"layer reads, compute them with Chart.transform(fm.Bin('v', name={name!r}))."
    )


@pytest.mark.parametrize("case", sorted(_NAMED))
def test_a_named_encoding_bin_is_refused_on_every_route(case):
    """An encoding bin belongs to its layer, so a ``name=`` publishing its rows is refused.

    Rejects dropping the name silently, or refusing only when an unrelated
    channel happens to send the Bin through the chain.
    """
    spelling, build = _NAMED[case]
    with pytest.raises(ValueError) as exc:
        _quiet(lambda: _render(build()))
    written = _SPELLINGS[spelling]
    name = "" if "/empty-name/" in case else "foo"
    assert str(exc.value) == _named_refusal(f"x=X('v', {written})", written, name)


@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_a_named_color_bin_is_refused(spelling):
    """A binned color publishes no rows, so its Bin's name is refused when the channel is built."""
    with pytest.raises(ValueError) as exc:
        fm.Color("v", bin=_named_bin(spelling))
    written = _SPELLINGS[spelling]
    assert str(exc.value) == _named_refusal(f"color=Color('v', {written})", written)


# ---------------------------------------------------------------------------
# bin= on the color family is a binned color scale
# ---------------------------------------------------------------------------
#
# ``Color/Fill/Stroke(field, bin=...)`` keeps the rows and draws the field
# through a ``bin-ordinal`` scale whose edges Rust cuts with the Bin
# transform's rule. A statistic grouped by the color groups by its bins, and
# only a statistic lowered from an encoding or a mark does: it sits after its
# chain's statistics marker. Rejects the row-replacing ``Bin``: 32 circles
# from 8 rows, titled ``bin_start``.

_COLOR_CLASSES = {"color": fm.Color, "fill": fm.Fill, "stroke": fm.Stroke}
_STATISTICS_MARKER = {"type": "identity", "statistics": True}


def _binned_z(cls: type = fm.Color, data: pl.DataFrame = df, **encoding: Any) -> fm.Chart:
    """``P`` colored by ``z``'s bins through *cls*."""
    channel = cls.__name__.lower()
    return P(data, **{channel: cls("z", bin=True)}, **encoding)


def _binned_color_doors(cls: type) -> dict[str, Case]:
    def binned(data: pl.DataFrame = df) -> fm.Chart:
        return _binned_z(cls, data)

    color = lambda: {cls.__name__.lower(): cls("z", bin=True)}  # noqa: E731
    return {
        "F": Case(binned, marks={"circle": 8}),
        "L": Case(
            lambda: P().layer(layer("point", y="r", **color())),
            (P, lambda: P(y="r", **color())),
            chart_level=None,
        ),
        "LD": Case(
            lambda: P().layer(fm.Layer(data=df2, mark="point", encoding=color())),
            (P, lambda: binned(df2)),
            chart_level=None,
        ),
        "+s": Case(lambda: binned() + line(), (binned, line)),
        "+d": Case(lambda: binned() + P(df2), (binned, lambda: P(df2))),
        **plus_doors(binned, line),
        "Fc": Case(lambda: binned().facet("c"), (lambda: binned().facet("c"),), faceted=True),
    }


_BINNED_COLOR_CASES = {
    f"binned_color/{name}/{door}": case
    for name, cls in _COLOR_CLASSES.items()
    for door, case in _binned_color_doors(cls).items()
}


def _wire_layers_and_steps(chart: Any) -> list[dict]:
    """Every transform on *chart*'s wire, each branch's inner steps included at any depth."""
    return _steps_at_any_depth(chart)


@pytest.mark.parametrize("case", sorted(_BINNED_COLOR_CASES))
def test_a_binned_color_keeps_its_rows_and_draws_a_legend_of_bins_titled_by_its_field(
    case,
):
    """One mark per row, a legend of bin ranges titled ``z``, and no ``Bin`` step.

    Each layer draws what it draws alone (the oracle census) on every door,
    and the legend's title is the field on every door, never ``bin_start``.
    """
    c = _BINNED_COLOR_CASES[case]
    check_rendering(c)
    chart = _quiet(c.build)
    assert not [s for s in _wire_layers_and_steps(chart) if s["type"] == "bin"]
    if c.static:
        svg = _quiet(chart.to_svg)
        assert list(legend_set(svg)) == ["z"]
        assert _legend_title_count(svg, "z") == 1


@pytest.mark.parametrize("cls", sorted(_COLOR_CLASSES.values(), key=lambda c: c.__name__))
def test_a_secondary_y_inheriting_a_binned_color_keeps_the_hosts_rows_and_legend(cls):
    """SY: the host's 8 points stay 8 under the binned color its secondary layer inherits."""
    chart = _quiet(lambda: _binned_z(cls) + SY("r"))
    svg = _quiet(chart.to_svg)
    assert tag_counts(svg)["circle"] == 8
    assert legend_set(svg) == legend_set(_quiet(_binned_z(cls).to_svg))
    assert legend_set(svg) == {"z": _bin_range_labels(df["z"])}
    assert not [s for s in _wire_layers_and_steps(chart) if s["type"] == "bin"]


def _bin_range_labels(values: pl.Series) -> tuple[str, ...]:
    """The bin-range labels ``Color("z", bin=True)`` draws over *values* on one flat chart.

    A flat chart's binned color cuts the ``Bin`` transform's own edges over its
    rows, so this is the edge set every door must draw over the same values.
    """
    frame = pl.DataFrame({"v": range(len(values)), "y": range(len(values)), "z": values})
    chart = fm.Chart(frame).mark_point().encode(x="v", y="y", color=fm.Color("z", bin=True))
    return legend_set(_quiet(chart.to_svg))["z"]


# Inherited binned color (brief: an attached layer inheriting a binned color
# shares the host's scale): one legend over both layers' rows, no Bin step.
_INHERIT_ROWS = df.select("v", "r", "z").with_columns((pl.col("z") + 12.5).alias("z"))


def _inherit_door(door: str, color: Any) -> Callable[[], fm.Chart]:
    def build() -> fm.Chart:
        host = P(color=color)
        if door == "L":
            return host.layer(layer("point", y="r"))
        return host.layer(fm.Layer(data=_INHERIT_ROWS, mark="point", encoding={"y": "r"}))

    return build


@pytest.mark.parametrize("door", ["L", "LD"])
def test_an_attached_layer_inheriting_a_binned_color_shares_the_hosts_scale(door):
    """Host ``Color("z", bin=True)`` inherited by an attached point layer: one legend of bins.

    The legend's edges are ``fm.Bin``'s over both layers' ``z``, each layer
    draws one mark per row, and no Bin step is on the wire. Rejects a layer
    that drops the inherited bin rule and colors continuously, and edges cut
    from the host's rows alone.
    """
    chart = _quiet(_inherit_door(door, fm.Color("z", bin=True)))
    svg = _quiet(chart.to_svg)
    other = df["z"] if door == "L" else _INHERIT_ROWS["z"]
    assert tag_counts(svg)["circle"] == df.height + other.len()
    # One source counts once: the L layer reads the host's own rows.
    union = df["z"] if door == "L" else pl.concat([df["z"], other])
    assert legend_set(svg) == {"z": _bin_range_labels(union)}
    # The attached layer draws the host's binding whole, bin rule included: a
    # layer's own binding replaces an inherited one entirely.
    wire = _wire(chart)
    drawn = wire["layers"][1]["encoding"].get("color") or wire["encoding"]["color"]
    assert drawn["scale"]["type"] == "bin-ordinal" and "bin" in drawn["scale"]
    assert not [s for s in _wire_layers_and_steps(chart) if s["type"] == "bin"]


@pytest.mark.parametrize("door", ["L", "LD"])
def test_an_attached_layer_under_an_unbinned_host_color_draws_its_continuous_legend(door):
    """Control: the same chart with an unbinned ``z`` draws a continuous colorbar, not bins."""
    binned = legend_set(_quiet(_quiet(_inherit_door(door, fm.Color("z", bin=True))).to_svg))
    plain = legend_set(_quiet(_quiet(_inherit_door(door, fm.Color("z"))).to_svg))
    assert plain != binned


#: Each ``bin=`` spelling and the ``bin`` rule its scale carries on the wire.
_BIN_RULES: dict[str, tuple[Callable[[], Any], dict]] = {
    "True": (lambda: True, {"nice": True}),
    "Bin-count": (lambda: fm.Bin("z", bin_count=4), {"bin_count": 4, "nice": True}),
    "dict-width": (lambda: {"bin_width": 5.0}, {"bin_width": 5.0, "nice": True}),
    "dict-extent": (
        lambda: {"extent": (0.0, 60.0), "bin_count": 3, "nice": False},
        {"bin_count": 3, "extent": [0.0, 60.0], "nice": False},
    ),
}


@pytest.mark.parametrize("name", sorted(_COLOR_CLASSES))
@pytest.mark.parametrize("rule", sorted(_BIN_RULES))
def test_a_bins_parameters_map_to_the_binned_scales_bin_rule(rule, name):
    """``bin_count``, ``bin_width``, ``extent`` and ``nice`` reach the scale's ``bin`` rule."""
    value, expected = _BIN_RULES[rule]
    chart = P(**{name: _COLOR_CLASSES[name]("z", bin=value())})
    color = _wire(chart)["encoding"]["color"]
    assert color == {
        "field": "z",
        "scale": {"type": "bin-ordinal", "bin": expected},
        "bin_channel": _COLOR_CLASSES[name].__name__,
    }


def test_the_bin_rule_sets_the_legends_edges():
    """``bin={"extent": (0, 60), "bin_count": 3, "nice": False}`` draws edges 0, 20, 40, 60."""
    rule = {"extent": (0.0, 60.0), "bin_count": 3, "nice": False}
    svg = _quiet(P(color=fm.Color("z", bin=rule)).to_svg)
    assert legend_set(svg) == {"z": ("0", "20", "40", "60")}


#: Each Bin parameter a binned color refuses: it computes no statistic,
#: groups no edges, and publishes no rows.
_COLOR_BIN_REFUSED: dict[str, dict] = {
    "groupby": {"groupby": ["c"]},
    "cumulative": {"cumulative": True},
    "agg_field/agg": {"agg_field": "y", "agg": "mean"},
    "aggregates": {"aggregates": [fm.AggregateOp("y", "mean", "mean_y")]},
    "per_row": {"per_row": True},
    "shared_extent": {"shared_extent": True},
}


def _color_bin_value(param: str, spelling: str) -> Any:
    kwargs = _COLOR_BIN_REFUSED[param]
    return fm.Bin("z", **kwargs) if spelling == "pre-built" else dict(kwargs)


def _color_bin_refusal(cls: type, param: str, spelling: str) -> str:
    written = f"{cls._channel_name}={cls.__name__}('z', {_SPELLINGS[spelling]})"
    return (
        f"{written} declares {param}, but bin= on {cls.__name__} builds a binned color "
        "scale: each row keeps its 'z' and is colored by its bin, and no statistic is "
        f"computed, so {param} would be ignored. Remove {param} from {_SPELLINGS[spelling]}; "
        "to compute it, use .transform(fm.Bin('z', ...)) and bind its output column."
    )


_COLOR_BIN_REFUSAL_CASES = {
    f"binned_color/{name}/{param}/{spelling}/{door}": (name, param, spelling, door)
    for name in _COLOR_CLASSES
    for param in _COLOR_BIN_REFUSED
    for spelling in _SPELLINGS
    for door in _vocabulary_doors(name, lambda: None)
}


@pytest.mark.parametrize("case", sorted(_COLOR_BIN_REFUSAL_CASES))
def test_a_binned_colors_bin_declaring_a_statistic_or_grouping_is_refused_on_every_door(case):
    """A Bin carrying a statistic, a groupby, ``per_row`` or ``shared_extent`` refuses, named.

    Rejects ignoring the parameter, and the row-replacing Bin that computed it
    and replaced the chart's rows.
    """
    name, param, spelling, door = _COLOR_BIN_REFUSAL_CASES[case]
    cls = _COLOR_CLASSES[name]
    build = _vocabulary_doors(name, lambda: cls("z", bin=_color_bin_value(param, spelling)))[door]
    with pytest.raises(ValueError) as refused:
        _quiet(build)
    assert str(refused.value) == _color_bin_refusal(cls, param, spelling)


def test_a_binned_colors_bin_declaring_three_parameters_joins_them_with_the_oxford_comma():
    """Three declared parameters on a binned color's Bin join with the Oxford
    comma in both the declared-parameters clause and the removal
    instruction, the same prose-list rule as the positional Bin refusal."""

    def build():
        return (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="v",
                y="y",
                color=fm.Color(
                    "z",
                    bin=fm.Bin(
                        "z",
                        groupby=["c"],
                        cumulative=True,
                        aggregates=[fm.AggregateOp("z", "mean", "mean_z")],
                    ),
                ),
            )
        )

    with pytest.raises(ValueError) as refused:
        _quiet(build)
    assert str(refused.value) == (
        "color=Color('z', bin=Bin(...)) declares groupby, cumulative, and aggregates, "
        "but bin= on Color builds a binned color scale: each row keeps its 'z' and is "
        "colored by its bin, and no statistic is computed, so groupby, cumulative, and "
        "aggregates would be ignored. Remove groupby, cumulative, and aggregates from "
        "bin=Bin(...); to compute it, use .transform(fm.Bin('z', ...)) and bind its "
        "output column."
    )


@pytest.mark.parametrize("door", sorted(_vocabulary_doors("color", lambda: None)))
@pytest.mark.parametrize("name", sorted(_COLOR_CLASSES))
def test_bin_with_scale_is_refused_with_the_spec_string_on_every_door(name, door):
    cls = _COLOR_CLASSES[name]
    build = _vocabulary_doors(
        name, lambda: cls("z", bin=True, scale=fm.BinOrdinalScale(bins=[0.0, 20.0, 50.0]))
    )[door]
    with pytest.raises(ValueError) as refused:
        _quiet(build)
    assert str(refused.value) == (
        f"{cls.__name__}(bin=...) cannot be combined with scale=: bin= builds the channel's "
        "binned color scale. Pass explicit edges with scale=fm.BinOrdinalScale(bins=[...]) "
        "and drop bin=, or drop scale=."
    )


@pytest.mark.parametrize("name", sorted(_COLOR_CLASSES))
def test_bin_with_aggregate_on_one_color_channel_stays_refused(name):
    cls = _COLOR_CLASSES[name]
    with pytest.raises(ValueError) as refused:
        cls("z", bin=True, aggregate="mean")
    assert str(refused.value).startswith(
        f"encoding channel {cls._channel_name!r} carries both bin= and aggregate="
    )


@pytest.mark.parametrize("name", sorted(_COLOR_CLASSES))
def test_a_bin_that_is_no_bin_or_bins_another_field_is_refused(name):
    cls = _COLOR_CLASSES[name]
    with pytest.raises(ValueError) as not_a_bin:
        cls("z", bin=3)
    assert str(not_a_bin.value) == (
        f"{cls.__name__}(bin=3) is not a bin: pass bin=True, a dict of Bin parameters "
        "(bin_count, bin_width, extent, nice) or fm.Bin(...)."
    )
    with pytest.raises(ValueError) as other_field:
        cls("z", bin=fm.Bin("v"))
    assert str(other_field.value) == (
        f"{cls._channel_name}={cls.__name__}('z', bin=Bin(...)) bins 'v', but {cls.__name__} "
        "colors by 'z': a binned color bins the field it colors by. Build the Bin over 'z', "
        "or color by 'v'."
    )


#: ``df2`` with a string ``z``, the frame a right operand or data layer reads.
_STRING_Z = df2.with_columns(pl.col("g").alias("z"))

_NON_NUMERIC_DOORS: dict[str, Callable[[str, Any], fm.Chart]] = {
    "F": lambda name, color: P(**{name: color("g")}),
    "L": lambda name, color: P().layer(layer("point", y="r", **{name: color("g")})),
    "LD": lambda name, color: P().layer(
        fm.Layer(data=_STRING_Z, mark="point", encoding={name: color("z")})
    ),
    "+d": lambda name, color: P() + P(_STRING_Z, **{name: color("z")}),
    "transform": lambda name, color: (
        fm.Chart(df)
        .transform(fm.transform_calculate("s", "datum.g + 'x'"))
        .mark_point()
        .encode(x="v", y="y", **{name: color("s")})
    ),
}


@pytest.mark.parametrize("door", sorted(_NON_NUMERIC_DOORS))
@pytest.mark.parametrize("name", sorted(_COLOR_CLASSES))
def test_a_binned_color_over_a_non_numeric_column_is_refused_naming_its_class(name, door):
    """A bin cuts numbers: a string column is refused naming the class written, on every door.

    One voice on every door, including rows a transform, a right operand or a
    data layer produce, which Python cannot see. Rejects the wire's
    ``color(bin=...)``, which names the channel the family draws as rather
    than the class the caller wrote.
    """
    cls = _COLOR_CLASSES[name]
    chart = _NON_NUMERIC_DOORS[door](name, lambda field: cls(field, bin=True))
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg)
    assert re.search(
        rf"{cls.__name__}\(bin=\.\.\.\) bins numbers, but column '[gzs]' is Utf8\. "
        r"Bind a numeric column, or drop bin=$",
        str(refused.value),
    ), str(refused.value)
    assert "color(bin=" not in str(refused.value)


def _non_numeric_refusal(cls: type, column: str) -> str:
    return (
        f"{cls.__name__}(bin=...) bins numbers, but column {column!r} is Utf8. "
        "Bind a numeric column, or drop bin="
    )


#: Each layered shape whose later layer alone bins a string column, with the
#: class written and the column: refused whatever color scale another layer
#: establishes.
_LATER_LAYER_NON_NUMERIC: dict[str, tuple[Callable[[], Any], type, str]] = {
    "unbinned-categorical+binned": (
        lambda: P(fill="g") + P(fill=fm.Fill("g", bin=True)),
        fm.Fill,
        "g",
    ),
    "binned-numeric+binned": (
        lambda: P(fill=fm.Fill("z", bin=True)) + P(fill=fm.Fill("g", bin=True)),
        fm.Fill,
        "g",
    ),
    "host-color+layer-stroke": (
        lambda: P(color="z").layer(
            fm.Layer(mark="point", encoding={"stroke": fm.Stroke("g", bin=True)})
        ),
        fm.Stroke,
        "g",
    ),
    "+d-right-operand": (
        lambda: P(color=fm.Color("z", bin=True)) + P(_STRING_Z, color=fm.Color("z", bin=True)),
        fm.Color,
        "z",
    ),
    "+d-right-operand-interactive": (
        lambda: (
            P(color=fm.Color("z", bin=True)) + P(_STRING_Z, color=fm.Color("z", bin=True))
        ).interactive(),
        fm.Color,
        "z",
    ),
}


def _render_any(chart: Any) -> str:
    return chart.to_svg() if hasattr(chart, "to_svg") else chart.to_html()


@pytest.mark.parametrize("case", sorted(_LATER_LAYER_NON_NUMERIC))
def test_a_later_layers_binned_color_over_a_string_column_is_refused(case):
    """A binned color over text is refused on the layer that wrote it, whoever establishes the scale.

    Rejects judging the binding only when it establishes the color scale: an
    unbinned layer 0 left the later binned layer unchecked (it rendered
    silently), and a binned numeric layer 0 excluded it with a warning.
    """
    build, cls, column = _LATER_LAYER_NON_NUMERIC[case]
    with pytest.raises(ValueError) as refused:
        _quiet(lambda: _render_any(build()))
    assert str(refused.value) == _non_numeric_refusal(cls, column)


#: ``df`` with a string ``n``: the rows an own-data layer reads under a host
#: binning a numeric ``n``.
_GATE_ROWS = pl.DataFrame({"v": [1.0, 2.0], "y": [4.0, 3.0], "n": ["a", "b"]})
_GATE_HOST_ROWS = df.with_columns(pl.col("z").alias("n"))


def _literal_paint(cls: type) -> Callable[[], Any]:
    """Two points painted by a literal, composed under ``cls('g', bin=True)``."""
    name = cls.__name__.lower()
    paint = "stroke" if cls is fm.Stroke else "fill"

    def build() -> Any:
        a = fm.Chart(df).mark_point(**{paint: "black"}).encode(x="v", y="y")
        b = fm.Chart(df).mark_point(**{paint: "red"}).encode(x="v", y="y")
        return (a + b).encode(**{name: cls("g", bin=True)})

    return build


def _json_literal_fill() -> Any:
    """The JSON door: a chart-level binned Fill over text, a layer painting its own fill."""
    from ferrum import _core

    binned = {
        "field": "g",
        "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": False}},
        "bin_channel": "Fill",
    }
    xy = {"x": {"field": "v"}, "y": {"field": "y"}}
    spec = _core.ChartSpec.from_json(
        json.dumps(
            {
                "data": {"kind": "named", "name": "default"},
                "mark": "point",
                "encoding": {**xy, "color": binned},
                "layers": [{"mark": "point", "encoding": xy, "mark_style": {"fill": "black"}}],
            }
        )
    )
    return _core.render_svg(spec, df, viewport=(300.0, 200.0))


#: Each shape where the color scale reads a binned binding the layer itself
#: does not draw (its own literal paint clears it), with the class and column.
_SCALE_READ_NON_NUMERIC: dict[str, tuple[Callable[[], Any], type, str]] = {
    **{
        f"+literal-paint/{cls.__name__}": (lambda cls=cls: _literal_paint(cls)().to_svg(), cls, "g")
        for cls in (fm.Color, fm.Fill, fm.Stroke)
    },
    "+literal-paint/interactive": (
        lambda: _literal_paint(fm.Fill)().interactive().to_html(),
        fm.Fill,
        "g",
    ),
    "LD-literal-fill": (
        lambda: (
            fm.Chart(_GATE_HOST_ROWS)
            .mark_point()
            .encode(x="v", y="y", fill=fm.Fill("n", bin=True))
            .layer(fm.Layer(data=_GATE_ROWS, mark="point", mark_kwargs={"fill": "red"}))
            .to_svg()
        ),
        fm.Fill,
        "n",
    ),
    "LD-same-field-own-color": (
        lambda: (
            fm.Chart(_GATE_HOST_ROWS)
            .mark_point()
            .encode(x="v", y="y", fill=fm.Fill("n", bin=True))
            .layer(fm.Layer(data=_GATE_ROWS, mark="point", encoding={"color": "n"}))
            .to_svg()
        ),
        fm.Fill,
        "n",
    ),
    "json-literal-fill": (_json_literal_fill, fm.Fill, "g"),
}


@pytest.mark.parametrize("case", sorted(_SCALE_READ_NON_NUMERIC))
def test_a_binned_color_the_scale_reads_over_text_is_refused_naming_its_class(case):
    """A binned binding the color scale reads over text is refused, though no layer draws it.

    A layer painting its own literal fill draws no inherited color, but the
    scale still reads its column. Rejects a refusal naming the wrong channel
    with no ``bin=`` remedy (two literal-paint layers), and a silent exclusion
    of the binned rule (an own-data layer with a literal fill). The same-field
    own color, which the layer completes binned, stays refused.
    """
    build, cls, column = _SCALE_READ_NON_NUMERIC[case]
    with pytest.raises(ValueError) as refused:
        _quiet(build)
    assert str(refused.value) == _non_numeric_refusal(cls, column)


def test_a_later_layers_unbinned_string_color_under_a_binned_layer_is_excluded_once():
    """Control: a later layer whose own color is unbinned is excluded from the binned scale, warned once."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = (P(color=fm.Color("z", bin=True)) + P(color="g")).to_svg()
    excluded = [str(w.message) for w in caught if "stays out of that scale" in str(w.message)]
    assert len(excluded) == 1, excluded
    assert list(legend_set(svg)) == ["z"]


def test_an_encoding_spec_carries_bin_channel_only_beside_a_binned_color_scale():
    """``bin_channel`` names a binned color's class: without a binned scale it is refused."""
    with pytest.raises(ValueError, match=r"drop bin_channel$"):
        fm.EncodingSpec("x", bin_channel="Fill")
    spec = fm.EncodingSpec("z", scale={"type": "bin-ordinal", "bin": {}}, bin_channel="Fill")
    assert spec.bin_channel == "Fill"


#: Each door a binned color is written on, with the override applied to the chart.
_BINNED_OVERRIDE_DOORS: dict[str, Callable[[str, Any], Any]] = {
    "F": lambda name, ch: P(**{name: ch}),
    "+": lambda name, ch: (P() + line()).encode(**{name: ch}),
    "L": lambda name, ch: P().layer(fm.Layer(mark="point", encoding={"y": "r", name: ch})),
    "LD": lambda name, ch: P().layer(
        fm.Layer(data=df2, mark="point", encoding={"y": "y", name: ch})
    ),
}

#: Each refused leaf (wire key suffix) and a value; ``type`` values other than
#: ``bin-ordinal`` strip the binned scale.
_REFUSED_BINNED_LEAVES: dict[str, Any] = {
    "type-linear": ("type", "linear"),
    "bins": ("bins", [0.0, 25.0, 50.0]),
    "bin": ("bin", {"bin_count": 2}),
    "domain": ("domain", [0.0, 50.0]),
}

#: (class written, override prefix): the same-named leaf, and an alias
#: source's binned Fill reached through the color leaf.
_BINNED_OVERRIDE_CLASSES = {
    "color": ("color", fm.Color, "color"),
    "fill": ("fill", fm.Fill, "fill"),
    "stroke": ("stroke", fm.Stroke, "stroke"),
    "fill-via-color-leaf": ("fill", fm.Fill, "color"),
}

_BINNED_OVERRIDE_CASES = {
    f"{who}/{leaf}/{door}/{kind}": (who, leaf, door, kind)
    for who in _BINNED_OVERRIDE_CLASSES
    for leaf in [*_REFUSED_BINNED_LEAVES, *(["type-quantize"] if who == "color" else [])]
    for door in _BINNED_OVERRIDE_DOORS
    for kind in ("static", "interactive")
}


def _binned_override_chart(who: str, door: str, **leaves: Any) -> Any:
    name, cls, _ = _BINNED_OVERRIDE_CLASSES[who]
    chart = _BINNED_OVERRIDE_DOORS[door](name, cls("z", bin=True))
    return chart.override(**leaves) if leaves else chart


@pytest.mark.parametrize("case", sorted(_BINNED_OVERRIDE_CASES))
def test_an_override_leaf_replacing_a_binned_colors_scale_is_refused(case):
    """A scale override that replaces what ``bin=`` builds is refused as ``scale=`` is.

    Named in the channel door's voice: the class written and the override
    path written, never the wire's ``bin_channel``. Rejects the leaf
    silently replacing the bin rule or edges (``bins``, ``bin``), a refusal
    steering to ``bin``/``bins`` (``domain``), and a type override reaching
    the wire's ``bin_channel`` refusal.
    """
    who, leaf, door, kind = _BINNED_OVERRIDE_CASES[case]
    _, cls, prefix = _BINNED_OVERRIDE_CLASSES[who]
    key, value = ("type", "quantize") if leaf == "type-quantize" else _REFUSED_BINNED_LEAVES[leaf]
    path = f"{prefix}_scale_{key}"
    chart = _binned_override_chart(who, door, **{path: value})
    with pytest.raises(fm.FerrumOverrideError) as refused:
        _quiet(lambda: chart.to_svg() if kind == "static" else chart.interactive().to_html())
    assert str(refused.value) == (
        f"{cls.__name__}(bin=...) cannot be combined with Chart.override({path}=...): bin= "
        "builds the channel's binned color scale. Pass explicit edges with "
        "scale=fm.BinOrdinalScale(bins=[...]) and drop bin=, or drop "
        f"Chart.override({path}=...)."
    )
    assert "bin_channel" not in str(refused.value)


@pytest.mark.parametrize("door", sorted(_BINNED_OVERRIDE_DOORS))
@pytest.mark.parametrize("who", sorted(_BINNED_OVERRIDE_CLASSES))
def test_a_binned_colors_scheme_override_is_honored_and_its_own_type_is_a_no_op(who, door):
    """``<channel>_scale_scheme`` recolors the bins; ``<channel>_scale_type='bin-ordinal'`` changes nothing."""
    _, _, prefix = _BINNED_OVERRIDE_CLASSES[who]
    bare = _quiet(_binned_override_chart(who, door).to_svg)
    schemed = _quiet(
        _binned_override_chart(who, door, **{f"{prefix}_scale_scheme": "viridis"}).to_svg
    )
    same = _quiet(
        _binned_override_chart(who, door, **{f"{prefix}_scale_type": "bin-ordinal"}).to_svg
    )
    assert schemed != bare
    assert same == bare


def test_an_unbinned_colors_type_override_is_unchanged_by_the_binned_rule():
    """Control: on an unbinned color, ``color_scale_type`` still reaches the scale."""
    chart = P(color="z").override(color_scale_type="quantize")
    assert _wire(chart)["encoding"]["color"]["scale"]["type"] == "quantize"
    assert "bin_channel" not in _wire(chart)["encoding"]["color"]


@pytest.mark.parametrize("name", ["fill", "stroke"])
def test_a_fill_or_stroke_scale_bin_override_reaches_the_color_it_draws_as(name):
    """``.override(fill_scale_bin=...)`` bins the color ``Fill`` draws as, as ``color_scale_bin=`` does.

    Rejects dropping a ``fill_``/``stroke_`` scale override because the wire
    carries the binding under ``color``.
    """
    rule = {"extent": [0.0, 60.0], "bin_count": 3, "nice": False}
    chart = P(**{name: _COLOR_CLASSES[name]("z")}).override(
        **{f"{name}_scale_type": "bin-ordinal", f"{name}_scale_bin": rule}
    )
    assert _wire(chart)["encoding"]["color"]["scale"] == {"type": "bin-ordinal", "bin": rule}
    assert legend_set(_quiet(chart.to_svg)) == {"z": ("0", "20", "40", "60")}


#: Override cells: the chart (flat ``P`` or an attached layer binding the
#: colors), the override, and whether the leaf reaches the drawn color.
_OVERRIDE_SHAPES: dict[str, tuple[dict, dict, bool]] = {
    "fill-domain": ({"fill": "z"}, {"fill_scale_domain": [0, 100]}, True),
    "stroke-domain": ({"stroke": "z"}, {"stroke_scale_domain": [0, 100]}, True),
    "color+fill-leaf": ({"color": "z"}, {"fill_scale_domain": [0, 100]}, False),
    "color+dropped-stroke": (
        {"color": "z", "stroke": "y"},
        {"stroke_scale_domain": [0, 100]},
        False,
    ),
    "fill+stroke+stroke-leaf": (
        {"fill": "z", "stroke": "y"},
        {"stroke_scale_domain": [0, 100]},
        False,
    ),
    "categorical+fill-leaf": ({"color": "g"}, {"fill_scale_domain": ["b", "a"]}, False),
}


def _override_door(door: str, bindings: dict) -> fm.Chart:
    if door == "F":
        return P(**bindings)
    return P().layer(fm.Layer(mark="point", encoding={"y": "r", **bindings}))


@pytest.mark.parametrize("door", ["F", "L"])
@pytest.mark.parametrize("shape", sorted(_OVERRIDE_SHAPES))
def test_a_fill_or_stroke_scale_override_reaches_only_the_color_it_draws_as(shape, door):
    """A ``fill_``/``stroke_`` scale leaf reaches color only when that channel draws as color.

    Fill or stroke draws as color when it wins the alias (color unbound;
    fill before stroke). Where it wins, its leaf scales the color it draws;
    where it lost, it draws nothing and its leaf leaves the chart's bytes
    as they are without it. Rejects dropping a winner's leaf, and merging a
    loser's leaf into another channel's scale (which rescaled ``color='z'``
    and refused ``color='g'``).
    """
    bindings, leaves, reaches = _OVERRIDE_SHAPES[shape]
    bare = _quiet(_override_door(door, bindings).to_svg)
    overridden = _quiet(_override_door(door, bindings).override(**leaves).to_svg)
    if reaches:
        assert overridden != bare
        assert legend_set(overridden)["z"][0] == "0"
    else:
        assert overridden == bare


@pytest.mark.parametrize("door", ["F", "L"])
def test_a_color_leaf_wins_over_a_fill_leaf_on_the_same_key(door):
    chart = _override_door(door, {"fill": "z"})
    both = chart.override(fill_scale_domain=[0, 100], color_scale_domain=[0, 60])
    color_only = chart.override(color_scale_domain=[0, 60])
    assert _quiet(both.to_svg) == _quiet(color_only.to_svg)


#: Each ``type_`` spelling a binned color refuses, and the code it reads as.
_REFUSED_BIN_TYPES = {"N": "N", "nominal": "N", "T": "T", "temporal": "T"}


@pytest.mark.parametrize("spelling", sorted(_REFUSED_BIN_TYPES))
@pytest.mark.parametrize("name", sorted(_COLOR_CLASSES))
def test_a_binned_color_typed_nominal_or_temporal_is_refused(name, spelling):
    """A binned color is ordered by its bins: a nominal or temporal type contradicts it."""
    cls = _COLOR_CLASSES[name]
    with pytest.raises(ValueError) as refused:
        cls("z", bin=True, type_=spelling)
    assert str(refused.value) == (
        f"{cls.__name__}(bin=...) cannot be combined with type_="
        f"{_REFUSED_BIN_TYPES[spelling]!r}: a binned color is ordered by its bins, so it "
        "cannot be nominal or temporal. Drop type_= or use quantitative or ordinal."
    )
    with pytest.raises(ValueError):
        cls(f"z:{_REFUSED_BIN_TYPES[spelling]}", bin=True)


@pytest.mark.parametrize("sort", ["descending", ["b", "a"], "-y"])
@pytest.mark.parametrize(
    "name", sorted(n for n, c in _COLOR_CLASSES.items() if "sort" in c._honored_kwargs)
)
def test_a_binned_color_sorted_other_than_ascending_is_refused(name, sort):
    """A binned color lists its bins in ascending order; no ``sort=`` reaches another order."""
    cls = _COLOR_CLASSES[name]
    with pytest.raises(ValueError) as refused:
        cls("z", bin=True, sort=sort)
    assert str(refused.value) == (
        f"{cls.__name__}(bin=...) cannot be combined with sort={sort!r}: a binned color "
        "lists its bins in ascending order; sort= cannot reorder them. Drop sort=."
    )


def _binned_with(cls: type, **kwargs: Any) -> fm.Chart:
    return P(**{cls.__name__.lower(): cls("z", bin=True, **kwargs)})


#: Each honored kwarg a binned color keeps (the rest are refused above), and
#: what it changes on the drawn chart.
_BINNED_HONORED: dict[str, tuple[dict, Callable[[str], Any]]] = {
    "title": ({"title": "Zed"}, lambda svg: list(legend_set(svg))),
    "legend": ({"legend": None}, lambda svg: legend_set(svg)),
    "scheme": (
        {"scheme": "viridis"},
        lambda svg: sorted(set(re.findall(r'fill="(#[0-9a-f]{6})"', svg))),
    ),
    "type-Q": ({"type_": "Q"}, lambda svg: legend_set(svg)),
    "type-O": ({"type_": "O"}, lambda svg: legend_set(svg)),
    "sort-ascending": ({"sort": "ascending"}, lambda svg: legend_set(svg)),
}

#: What each honored kwarg reads as on the drawn chart, against the bare binned color.
_BINNED_HONORED_EXPECTED: dict[str, Callable[[Any, Any], bool]] = {
    "title": lambda got, bare: got == ["Zed"],
    "legend": lambda got, bare: got == {},
    "scheme": lambda got, bare: got != bare,
    "type-Q": lambda got, bare: got == bare,
    "type-O": lambda got, bare: got == bare,
    "sort-ascending": lambda got, bare: got == bare,
}


@pytest.mark.parametrize(
    ("name", "kwarg"),
    [
        (name, kwarg)
        for name, cls in sorted(_COLOR_CLASSES.items())
        for kwarg, (kwargs, _) in sorted(_BINNED_HONORED.items())
        if set(kwargs) - {"type_"} <= cls._honored_kwargs
    ],
)
def test_each_kwarg_a_binned_color_honors_reaches_its_drawn_chart(name, kwarg):
    """Every honored kwarg on a binned color is honored (or refused above), none silently dropped."""
    cls = _COLOR_CLASSES[name]
    kwargs, read = _BINNED_HONORED[kwarg]
    got = read(_quiet(_binned_with(cls, **kwargs).to_svg))
    bare = read(_quiet(_binned_with(cls).to_svg))
    assert _BINNED_HONORED_EXPECTED[kwarg](got, bare), (got, bare)


def test_a_binned_color_condition_reaches_the_wire():
    """``Color(bin=..., condition=...)``: the conditional reaches the wire beside the binned scale."""
    sel = fm.selection_point(fields=["g"], name="pick")
    chart = P(color=fm.Color("z", bin=True, condition=sel.when("#ff0000").otherwise("#cccccc")))
    wire = _wire(chart.add_params(sel))
    assert wire["encoding"]["color"]["scale"]["type"] == "bin-ordinal"
    assert [c["channel"] for c in wire.get("conditionals", [])] == ["color"]


# The statistics marker sits where the user's steps end, on every + door: a
# step the caller wrote never follows a marker in its chain, so it runs as
# written beside a binned operand.

_USER_STEP_TYPES = frozenset({"data_aggregate", "filter", "calculate", "composed"})


def _caller_steps_after_a_marker(steps: list[dict]) -> list[str]:
    """The caller-written steps that follow a statistics marker in their chain, at any depth."""
    out: list[str] = []
    marked = False
    for step in steps:
        if step == _STATISTICS_MARKER:
            marked = True
        elif marked and step["type"] in _USER_STEP_TYPES:
            out.append(step["type"])
        if step["type"] == "branch":
            out += _caller_steps_after_a_marker(step["transforms"])
    return out


def _user_mean() -> dict:
    return fm.transform_aggregate({"field": "y", "fn": "mean", "as": "my"}, groupby=["g", "v"])


def _user_copy() -> dict:
    """``my`` as a copy of ``y``: a step every operand of a same-frame composition can take."""
    return fm.transform_calculate("my", "datum.y")


def _user_bars(color: Any, data: pl.DataFrame = _GROUP_ROWS) -> fm.Chart:
    """Dodged bars of a caller's own aggregate grouped by raw ``v``: 16 bars."""
    rows = fm.Chart(data).transform(_user_mean())
    return rows.mark_bar(position=fm.Dodge()).encode(x="g", y="my", color=color)


def _box(color: Any) -> fm.Chart:
    return fm.Chart(_GROUP_ROWS).mark_boxplot().encode(x="g", y="y", color=color)


def _user_points(color: Any) -> fm.Chart:
    return fm.Chart(_GROUP_ROWS).mark_point().encode(x="g", y="my", color=color)


def _binned_hist(color: Any) -> fm.Chart:
    return fm.Chart(_GROUP_ROWS).mark_histogram().encode(x="y", color=color)


def _plain_points(color: Any) -> fm.Chart:
    return fm.Chart(_GROUP_ROWS).mark_point().encode(x="g", y="y", color=color)


def _hist_points(color: Any) -> fm.Chart:
    """Points on the histogram's continuous x."""
    return fm.Chart(_GROUP_ROWS).mark_point().encode(x="y", y="v", color=color)


#: ``_GROUP_ROWS`` as a different frame: a right operand over it reads its own rows.
_GROUP_ROWS_OTHER = _GROUP_ROWS.with_columns(pl.col("y").alias("y_copy"))


#: Each + door: the composition, and the parts whose standalone marks it draws.
_MARKER_DOORS: dict[str, tuple[Callable[[Any], Any], tuple[Callable[[Any], Any], ...]]] = {
    "L+B": (lambda c: _user_bars(c) + _box(c), (_user_bars, _box)),
    "B+L": (lambda c: _box(c) + _user_bars(c), (_box, _user_bars)),
    "(B+P).transform": (
        lambda c: (_box(c) + _user_points(c)).transform(_user_copy()),
        (
            lambda c: _box(c).transform(_user_copy()),
            lambda c: _user_points(c).transform(_user_copy()),
        ),
    ),
    "(Hist+P).transform": (
        lambda c: (_binned_hist(c) + _hist_points(c)).transform(fm.transform_filter("true")),
        (_binned_hist, _hist_points),
    ),
    "B+B": (lambda c: _box(c) + _box(c), (_box, _box)),
    "B.layer(LD)": (
        lambda c: _box(c).layer(fm.Layer(data=_GROUP_ROWS, mark="point", encoding={"y": "y"})),
        (_box, _plain_points),
    ),
    "+d": (lambda c: _box(c) + _user_bars(c, _GROUP_ROWS_OTHER), (_box, _user_bars)),
}


@pytest.mark.parametrize("door", sorted(_MARKER_DOORS))
def test_a_caller_step_never_follows_a_statistics_marker_on_any_plus_door(door):
    """No caller-written step follows a marker in its chain, so it runs as written.

    Each part draws inside the composition what it draws alone: the caller's
    own aggregate grouped by raw ``v`` keeps its 16 bars (or its points)
    beside a binned boxplot or histogram, in either order and after a
    ``.transform()`` on the composition, and the composition's own chain
    holds no marker. Rejects merging a binned operand's marked chain into
    the composition's chain, which put the caller's step after the marker
    and re-binned it (16 bars to 8).
    """
    compose, parts = _MARKER_DOORS[door]
    color = _binned_color()
    chart = _quiet(lambda: compose(color))
    steps = _wire(chart).get("transforms") or []
    assert _caller_steps_after_a_marker(steps) == []
    assert _STATISTICS_MARKER not in steps
    expected = Counter()
    for part in parts:
        expected.update(tag_counts(_quiet(_quiet(lambda part=part: part(color)).to_svg)))
    assert Counter(tag_counts(_quiet(chart.to_svg))) == expected


#: The SVG bytes each door renders under the unbinned color ``g``, unchanged by
#: the binned color's marker placement.
_MARKER_DOOR_CONTROL_DIGESTS = {
    "(B+P).transform": "d4a6869d9078d676d1d88c5efbfb58de1417f8909c63549bbd52f9501263d60f",
    "(Hist+P).transform": "1917be7c83632268e1a7619697c824483e7d5bd53aec5586f8787fe926cf3374",
    "+d": "ec4cbacb6fdc20927df4686cd4e87fbf525d59df87885f3708d3986559e3a0cf",
    "B+B": "9f75420b81af1aa458d60b85bf743cb46a4a121377e3540fbbaef047405ce4cd",
    "B+L": "ec4cbacb6fdc20927df4686cd4e87fbf525d59df87885f3708d3986559e3a0cf",
    "B.layer(LD)": "d4a6869d9078d676d1d88c5efbfb58de1417f8909c63549bbd52f9501263d60f",
    "L+B": "f53747f38c1fad6e474264df163ec2194962d55300b9c9aa3f5d3752e54c931f",
}


@pytest.mark.parametrize("door", sorted(_MARKER_DOORS))
def test_a_plus_door_with_no_binned_color_renders_its_pinned_bytes(door):
    """Control: unbinned, each door renders the bytes it rendered before."""
    import hashlib

    compose, _ = _MARKER_DOORS[door]
    chart = _quiet(lambda: compose("g"))
    svg = _quiet(chart.to_svg)
    assert hashlib.sha256(svg.encode()).hexdigest() == _MARKER_DOOR_CONTROL_DIGESTS[door]


# Statistics under a binned color group by its bins, on every door. The
# oracle is the same chart grouped by an explicit bin column (``v_bin``, the
# floor of ``v`` under the unit bins): the binned chart draws the marks it
# draws, on the same door.

#: ``_GROUP_ROWS`` plus a response a smooth can fit per bin.
_BIN_GROUP_ROWS = _GROUP_ROWS.with_columns((pl.col("y") * 0.5 + pl.col("v") ** 2).alias("w"))
_BIN_GROUP_ROWS_BINNED = _BIN_GROUP_ROWS.with_columns(pl.col("v").floor().alias("v_bin"))

_BIN_GROUPED_STATISTICS: dict[str, Callable[[fm.Chart, Any], fm.Chart]] = {
    "aggregate": lambda base, c: base.mark_bar().encode(x="g", y="mean(y)", color=c),
    "aggregate-stacked": lambda base, c: base.mark_bar(position=fm.Stack()).encode(
        x="g", y="mean(y)", color=c
    ),
    "aggregate-dodged": lambda base, c: base.mark_bar(position=fm.Dodge()).encode(
        x="g", y="mean(y)", color=c
    ),
    "bin+count": lambda base, c: base.mark_bar().encode(
        x=fm.X("y", bin={"bin_width": 4.0, "extent": (0.0, 16.0), "nice": False}),
        y="count()",
        color=c,
    ),
    "histogram": lambda base, c: base.mark_histogram(bin_width=4.0).encode(x="y", color=c),
    "density": lambda base, c: base.mark_density().encode(x="y", color=c),
    "boxplot": lambda base, c: base.mark_boxplot().encode(x="g", y="y", color=c),
    "violin": lambda base, c: base.mark_violin().encode(x="g", y="y", color=c),
}

#: Marks each statistic draws per (x, bin), pinned on the flat door.
_BIN_GROUPED_MARKS = {
    "aggregate": {"rect": 8},
    "aggregate-stacked": {"rect": 8},
    "aggregate-dodged": {"rect": 8},
    "bin+count": {"rect": 16},
    "histogram": {"rect": 12},
    "density": {"path": 4},
    "boxplot": {"line": 32, "rect": 8},
    "violin": {"path": 8, "line": 32, "rect": 8},
}


def _bin_grouped(statistic: str) -> Callable[[], fm.Chart]:
    return lambda: _BIN_GROUPED_STATISTICS[statistic](fm.Chart(_BIN_GROUP_ROWS), _binned_color())


def _bin_column_grouped(statistic: str) -> Callable[[], fm.Chart]:
    return lambda: _BIN_GROUPED_STATISTICS[statistic](
        fm.Chart(_BIN_GROUP_ROWS_BINNED), fm.Color("v_bin", type_="N")
    )


def _bin_partner() -> fm.Chart:
    return mk("rule", _BIN_GROUP_ROWS, y="y")


def _drawn(build: Callable[[], Any], door: str) -> Any:
    chart = _quiet(build)
    if door == "LCx":
        return Counter(kind for kind, _ in interactive_census(chart).elements())
    return tag_counts(_quiet(chart.to_svg))


_BIN_GROUPED_CASES = {
    f"{statistic}/{door}": (statistic, door)
    for statistic in _BIN_GROUPED_STATISTICS
    for door in _declared_doors(P)
}


@pytest.mark.parametrize("case", sorted(_BIN_GROUPED_CASES))
def test_a_statistic_under_a_binned_color_groups_by_its_bins_on_every_door(case):
    """Each statistic grouped by a binned color draws what grouping by the bin column draws.

    Rejects grouping by the raw values (16 bars, or two dodged to a slot,
    and one histogram, curve, box or violin per distinct value). A door
    that refuses the statistic's shape refuses both spellings alike.
    """
    statistic, door = _BIN_GROUPED_CASES[case]
    binned = _declared_doors(_bin_grouped(statistic), _bin_partner)[door]
    oracle = _declared_doors(_bin_column_grouped(statistic), _bin_partner)[door]
    if door == "LCx":
        binned = lambda: fm.LayerChart(_bin_grouped(statistic)(), _bin_partner())  # noqa: E731
        oracle = lambda: fm.LayerChart(_bin_column_grouped(statistic)(), _bin_partner())  # noqa: E731
    try:
        expected = _drawn(oracle, door)
    except ValueError as refused:
        with pytest.raises(ValueError) as also:
            _drawn(binned, door)
        assert str(also.value) == str(refused)
        return
    assert _drawn(binned, door) == expected
    if door == "F":
        assert expected == _BIN_GROUPED_MARKS[statistic]


@pytest.mark.parametrize("statistic", sorted(_BIN_GROUPED_STATISTICS))
def test_a_statistic_under_a_binned_color_sits_after_its_chains_statistics_marker(statistic):
    """The wire marks where the lowered statistics start, and the user's rows stay before it."""
    base = fm.Chart(_BIN_GROUP_ROWS).transform(fm.transform_filter("datum.y > 0"))
    steps = _wire(_BIN_GROUPED_STATISTICS[statistic](base, _binned_color()))["transforms"]
    marker = steps.index(_STATISTICS_MARKER)
    assert [s["type"] for s in steps[:marker]] == ["filter"]
    assert steps[marker + 1 :]
    assert steps.count(_STATISTICS_MARKER) == 1


def test_a_chart_drawing_no_binned_color_marks_its_statistics_all_the_same():
    """The marker leads every chain's lowered statistics, binned color or not."""
    for statistic in _BIN_GROUPED_STATISTICS:
        steps = _wire(_bin_column_grouped(statistic)())["transforms"]
        assert steps.count(_STATISTICS_MARKER) == 1
        assert steps[steps.index(_STATISTICS_MARKER) + 1 :]


#: Compositions whose operands carry caller steps, statistics, or both.
_OPERAND_CHAIN_DOORS: dict[str, Callable[[], fm.Chart]] = {
    "partner+aggregate": lambda: _bin_partner() + _bin_grouped("aggregate")(),
    "user-bars+aggregate": lambda: _user_bars(_binned_color()) + _bin_grouped("aggregate")(),
    "histogram+chained-points": lambda: (
        _binned_hist(_binned_color()) + _chained_points(_binned_color())
    ),
    "box+user-points": lambda: _box("g") + _user_points("g").transform(_user_copy()),
    # A Chart.layer statistic door: the box's own BoxStats lowering, and its
    # attached layer's own Aggregate lowering, both inside one operand.
    "box.layer(mean)+points": lambda: (
        _box("c").layer(fm.Layer(mark="point", encoding={"x": "g", "y": "mean(y)"}))
        + _user_points("c")
    ),
    # A figure-statistic door: displot's own KDE lowering beside its rug
    # overlay's, both inside one operand, with no `.layer()` involved.
    "displot-kde-rug+rule": lambda: (
        fm.displot(_GROUP_ROWS, x="v", kind="kde", rug=True)
        + fm.Chart(_GROUP_ROWS).mark_rule().encode(x="v")
    ),
}


def _each_lowering_opens_its_own_marker(chain: list[dict]) -> None:
    """Every lowering of *chain* (and of every branch it nests) is non-empty and led by its own marker."""
    markers = [i for i, step in enumerate(chain) if step == _STATISTICS_MARKER]
    for start, end in zip(markers, [*markers[1:], len(chain)]):
        lowering = chain[start + 1 : end]
        assert lowering and all(step["type"] not in _USER_STEP_TYPES for step in lowering)
    for step in chain:
        if step["type"] == "branch":
            _each_lowering_opens_its_own_marker(step["transforms"])


@pytest.mark.parametrize("door", sorted(_OPERAND_CHAIN_DOORS))
def test_each_operand_branch_leads_every_lowering_with_its_own_marker(door):
    """The top chain holds one branch per operand; in each, every lowering is led by its own marker.

    No caller-written step follows a chain's first marker, and every
    lowering after it -- an operand's mark statistics, a figure's own,
    each layer's encoding statistic -- opens behind a marker of its own,
    never sharing one with a lowering beside it, at any nesting depth.
    Rejects lowering a layer's statistic outside its operand's chain, as a
    sibling of the operand branches, and rejects two lowerings sharing one
    marker.
    """
    steps = _wire(_quiet(_OPERAND_CHAIN_DOORS[door]))["transforms"]
    assert [step["type"] for step in steps] == ["branch"] * len(steps)
    for branch in steps:
        chain = branch["transforms"]
        assert _caller_steps_after_a_marker(chain) == []
        _each_lowering_opens_its_own_marker(chain)


def test_a_non_finite_binned_value_keeps_its_own_group():
    """NaN, +inf and -inf fall in no bin, so each stays its own group beside the bins.

    Rejects dropping their rows: per ``g``, 4 bins plus 3 non-finite groups.
    """
    rows = _GROUP_ROWS.with_columns(
        pl.Series("v", [1.1, float("nan"), 2.2, float("inf"), 3.1, float("-inf"), 4.4, 4.9] * 2)
    )
    chart = (
        fm.Chart(rows)
        .mark_bar(position=fm.Dodge())
        .encode(x="g", y="mean(y)", color=_binned_color())
    )
    keys = rows.select(
        "g",
        pl.when(pl.col("v").is_finite())
        .then(pl.col("v").floor().cast(pl.Utf8))
        .otherwise(pl.col("v").cast(pl.Utf8))
        .alias("key"),
    )
    assert tag_counts(_quiet(chart.to_svg))["rect"] == keys.n_unique() == 14


def test_a_user_transform_grouped_by_the_binned_field_runs_as_written():
    """A caller's own ``.transform()`` aggregate grouped by ``v`` keeps every raw ``v`` group."""
    chart = (
        fm.Chart(_GROUP_ROWS)
        .transform(
            fm.transform_aggregate({"field": "y", "fn": "mean", "as": "my"}, groupby=["g", "v"])
        )
        .mark_bar(position=fm.Dodge())
        .encode(x="g", y="my", color=_binned_color())
    )
    assert tag_counts(_quiet(chart.to_svg))["rect"] == _GROUP_ROWS.select("g", "v").n_unique()


# Areas group by bin: one band per bin, stacked or not, on every door.

_AREA_BIN = {"bin_count": 2, "nice": False}
_AREA_ROWS_BINNED = df.with_columns(
    pl.when(pl.col("z") < 27.5).then(pl.lit("low")).otherwise(pl.lit("high")).alias("z_bin")
)


def _binned_area(data: pl.DataFrame, color: Any, position: Any) -> fm.Chart:
    return fm.Chart(data).mark_area(position=position).encode(x="v", y="y", color=color)


_AREA_CASES = {
    f"{stacked}/{door}": (stacked, door)
    for stacked in ("stacked", "unstacked")
    for door in _declared_doors(P)
}


@pytest.mark.parametrize("case", sorted(_AREA_CASES))
def test_an_area_under_a_binned_color_draws_one_band_per_bin_on_every_door(case):
    """An area under ``Color("z", bin=...)`` draws one band per bin, stacked or not.

    The oracle is the same area colored by an explicit two-bin column.
    Rejects one band per distinct value.
    """
    stacked, door = _AREA_CASES[case]
    position = fm.Stack() if stacked == "stacked" else None
    binned = lambda: _binned_area(df, fm.Color("z", bin=_AREA_BIN), position)  # noqa: E731
    oracle = lambda: _binned_area(_AREA_ROWS_BINNED, fm.Color("z_bin", type_="N"), position)  # noqa: E731
    built, expected = _declared_doors(binned, P)[door], _declared_doors(oracle, P)[door]
    if door == "LCx":
        built = lambda: fm.LayerChart(binned(), P())  # noqa: E731
        expected = lambda: fm.LayerChart(oracle(), P())  # noqa: E731
    assert _drawn(built, door) == _drawn(expected, door)
    if door == "F":
        assert _drawn(built, door) == {"path": 2}


# Concat and overlay panels share one binned edge set and legend by default.


def _union_rows() -> pl.DataFrame:
    return pl.concat([df.select("v", "y", "z"), df2.select("v", "y", "z")])


def _binned_panel(data: pl.DataFrame = df, **rule: Any) -> fm.Chart:
    return P(data, color=fm.Color("z", bin=rule or True))


_SHARED_BINNED: dict[str, Callable[[], Any]] = {
    "hconcat": lambda: fm.hconcat(_binned_panel(), _binned_panel(df2)),
    "vconcat": lambda: fm.vconcat(_binned_panel(), _binned_panel(df2)),
    "LCs": lambda: fm.LayerChart(_binned_panel(), _binned_panel(df2)),
}


@pytest.mark.parametrize("route", sorted(_SHARED_BINNED))
def test_panels_binning_one_field_alike_share_one_edge_set_and_legend(route):
    """Two panels binning ``z`` alike draw one legend cut from both panels' rows.

    The oracle is one chart over both frames' rows. Rejects a legend per
    panel, each over its own edges.
    """
    svg = _quiet(_SHARED_BINNED[route]().to_svg)
    union = _quiet(_binned_panel(_union_rows()).to_svg)
    assert legend_set(svg) == legend_set(union)
    assert _legend_title_count(svg, "z") == 1


def test_a_concat_leaves_its_color_resolve_unset_for_the_default_to_apply():
    from ferrum.composition import _lower_any

    tree = _lower_any(
        fm.hconcat(_binned_panel(), _binned_panel(df2)),
        is_root=True,
        auto_tooltips=False,
        payloads=[],
        leaf_inputs=[],
        leaf_nodes=[],
    )
    assert "color" not in json.dumps(tree.get("resolve") or {})


def test_an_independent_color_resolve_keeps_a_legend_per_panel():
    chart = fm.hconcat(
        _binned_panel(), _binned_panel(df2), resolve=fm.Resolve(scale={"color": "independent"})
    )
    assert _legend_title_count(_quiet(chart.to_svg), "z") == 2


def test_panels_binning_with_different_rules_keep_a_legend_each_by_default():
    chart = fm.hconcat(_binned_panel(bin_count=4), _binned_panel(df2, bin_count=2))
    assert _legend_title_count(_quiet(chart.to_svg), "z") == 2


def test_a_shared_color_over_differing_bin_rules_is_refused_as_a_typed_error():
    chart = fm.hconcat(
        _binned_panel(bin_count=4),
        _binned_panel(df2, bin_count=2),
        resolve=fm.Resolve(scale={"color": "shared"}),
    )
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg)
    assert str(refused.value) == (
        "composite scale resolution failed: failed to resolve shared color domain for field "
        "'z': a shared color needs one color scale on every panel, but one panel colors by "
        "'z' binned with bin={\"bin_count\":4,\"nice\":true} and another by 'z' binned with "
        'bin={"bin_count":2,"nice":true}. Make the panels\' colors match, or resolve color '
        "independently with Resolve(scale={'color': 'independent'})"
    )


def test_a_nested_concat_shares_the_inner_binned_panels_legend():
    chart = fm.vconcat(fm.hconcat(_binned_panel(), _binned_panel(df2)), P(color="c"))
    legends = legend_set(_quiet(chart.to_svg))
    assert legends["z"] == legend_set(_quiet(_binned_panel(_union_rows()).to_svg))["z"]
    assert legends["c"] == ("p", "q")


# A line-drawn mark under a binned color.

_LINE_UNDER_BINNED: dict[str, Callable[[], fm.Chart]] = {
    "line": lambda: (
        fm.Chart(df).mark_line().encode(x="v", y="y", color=fm.Color("z", bin=_AREA_BIN))
    ),
    "line-mean": lambda: (
        fm.Chart(df).mark_line().encode(x="v", y="mean(y)", color=fm.Color("z", bin=_AREA_BIN))
    ),
    "ribbon": lambda: (
        fm.Chart(df.with_columns((pl.col("y") + 1).alias("y_hi")))
        .mark_ribbon()
        .encode(x="v", y="y", y2="y_hi", color=fm.Color("z", bin=_AREA_BIN))
    ),
}


@pytest.mark.parametrize("mark", sorted(_LINE_UNDER_BINNED))
def test_a_line_under_a_binned_color_draws_one_series_per_bin(mark):
    """Two bins of ``z``: two series, each in its own bin's color, and a legend of bin ranges."""
    svg = _quiet(_LINE_UNDER_BINNED[mark]().to_svg)
    series = [e for e in census(svg).elements() if e[0] in ("polyline", "path")]
    assert len(series) == 2
    assert list(legend_set(svg)) == ["z"]


# A filter or calculate naming a column its rows lack is refused, and a panel
# draws while any layer has rows.


def _missing_column(step: str, column: str, available: pl.DataFrame) -> str:
    """The render's refusal of an expression step naming *column*, which rows of *available* lack."""
    return (
        f"transform failed: ValueError: {step}: column '{column}' not found; "
        f"available: {json.dumps(available.columns)}"
    )


#: The composite render's envelope around its first leaf's refusal.
_FIRST_LEAF = "failed to render composite leaf leaf #0: "


def _filter_count() -> Any:
    return fm.transform_filter("datum.count > 1")


def _calculate_count() -> Any:
    return fm.transform_calculate("doubled", "datum.count * 2")


#: Each door a step naming a column its rows lack is written on, the rows it
#: reads, and the envelope the door's render wraps the refusal in.
_MISSING_COLUMN_DOORS: dict[str, tuple[Callable[[Any], Any], pl.DataFrame, str]] = {
    "F": (lambda step: P().transform(step), df, ""),
    "L": (lambda step: P().transform(step).layer(layer("point", y="r")), df, ""),
    "LD": (
        lambda step: (
            P()
            .transform(step)
            .layer(fm.Layer(data=df2, mark="point", encoding={"x": "v", "y": "y"}))
        ),
        df,
        "",
    ),
    "+": (lambda step: (P() + P(y="r")).transform(step), df, ""),
    "+l": (lambda step: P().transform(step) + P(y="r"), df, ""),
    "+d": (lambda step: (P() + P(df2)).transform(step), df, ""),
    "LCs": (lambda step: fm.LayerChart(P().transform(step), P(y="r")), df, _FIRST_LEAF),
    "LCm": (lambda step: fm.LayerChart(P().transform(step), P(y="r"))._build_merged(), df, ""),
    "Fc": (lambda step: P().transform(step).facet("c"), df, ""),
    "hconcat": (lambda step: fm.hconcat(P().transform(step), P(y="r")), df, _FIRST_LEAF),
}

_MISSING_COLUMN_STEPS: dict[str, tuple[Callable[[], Any], str]] = {
    "filter": (_filter_count, "data_filter"),
    "calculate": (_calculate_count, "data_calculate"),
}


@pytest.mark.parametrize("step", sorted(_MISSING_COLUMN_STEPS))
@pytest.mark.parametrize("door", sorted(_MISSING_COLUMN_DOORS))
def test_a_step_naming_a_column_its_rows_lack_is_refused_on_every_door(door, step):
    """``filter('datum.count > 1')`` over rows without ``count`` names the column and the rows' columns.

    Rejects reading the missing column as null on every row, which emptied
    the rows (a filter) or wrote an all-null column (a calculate) and
    rendered a blank chart with no warning.
    """
    build, rows, envelope = _MISSING_COLUMN_DOORS[door]
    make_step, name = _MISSING_COLUMN_STEPS[step]
    chart = _quiet(lambda: build(make_step()))
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg)
    assert str(refused.value) == envelope + _missing_column(name, "count", rows)


#: ``df`` with a column every row of which is null.
_ALL_NULL_ROWS = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("n"))


@pytest.mark.parametrize(
    "step",
    [
        fm.transform_calculate("doubled", "datum.n * 2"),
        fm.transform_calculate("one", "1 + 0"),
        fm.transform_filter("1 < 2"),
    ],
    ids=["calculate-all-null", "calculate-literal", "filter-literal"],
)
def test_an_all_null_column_or_a_step_naming_no_column_runs_as_written(step):
    """A present all-null column is read as written and a literal step names nothing: every point draws."""
    chart = P(_ALL_NULL_ROWS).transform(step)
    assert census(_quiet(chart.to_svg)) == census(_quiet(P(_ALL_NULL_ROWS).to_svg))


def _emptied() -> fm.Chart:
    return P().transform(fm.transform_filter("datum.y > 100"))


#: Doors whose chart-level rows are empty while a layer reads rows of its own.
_EMPTY_CHART_ROWS_DOORS: dict[str, Case] = {
    "+s": Case(lambda: _emptied() + line(), (line,)),
    "LD": Case(
        lambda: _emptied().layer(fm.Layer(data=df2, mark="line", encoding={"x": "v", "y": "y"})),
        (lambda: line(df2),),
    ),
    "Fc": Case(
        lambda: (_emptied() + line()).facet("c"), (lambda: line().facet("c"),), faceted=True
    ),
}


@pytest.mark.parametrize("door", sorted(_EMPTY_CHART_ROWS_DOORS))
def test_a_panel_draws_while_any_layer_has_rows(door):
    """A filter emptying the left operand's rows leaves the other layer drawing its own.

    Rejects skipping the panel on the chart-level rows alone, which blanked
    the chart with no warning.
    """
    check_rendering(_EMPTY_CHART_ROWS_DOORS[door])


# A step written on a + composition reaches the left operand and every
# operand over the composition's frame, however the merge routed it, each over
# its own rows, and never an operand over a different frame. It runs where
# each reached operand's own steps end, before its
# statistics, when those rows carry the columns it reads, else on the first
# rows after the statistics that carry them; a chain whose rows never carry
# them is left as written, and a step no reached chain can read is refused.


def _scene_records(chart: Any) -> dict[str, Counter]:
    """Per interactive mark kind, the multiset of its marks' tooltip records over every panel."""
    out: dict[str, Counter] = {}
    for panel in _scene(chart)["panels"]:
        for batch in panel["marks"]:
            records = out.setdefault(batch["kind"], Counter())
            for tooltip in batch.get("tooltips") or []:
                records[tuple((f["name"], f["value"]) for f in tooltip["fields"])] += 1
    return out


def _counted_above_one(records: Counter) -> Counter:
    """The bar records whose ``count`` is above 1: the histogram's bars as the filter defines."""
    return Counter({r: n for r, n in records.items() if float(dict(r)["count"]) > 1})


def _hist_points_other(color: Any) -> fm.Chart:
    """``_hist_points`` over a different frame holding the same rows."""
    return fm.Chart(_GROUP_ROWS_OTHER).mark_point().encode(x="y", y="v", color=color)


def _chained_points(color: Any) -> fm.Chart:
    """``_hist_points`` with a chain of its own that keeps every row."""
    return _hist_points(color).transform(fm.transform_filter("datum.y > 0"))


def _bare_hist_points(color: Any) -> fm.Chart:
    """``_hist_points`` uncolored: a ``Layer(data=)`` attached to a composition, which colors nothing."""
    return fm.Chart(_GROUP_ROWS).mark_point().encode(x="y", y="v")


#: The composition steps: a column every operand carries, two source columns
#: the histogram's statistic drops, and the statistic's own output column.
_REACH_STEPS: dict[str, Callable[[], Any]] = {
    "v": lambda: fm.transform_filter("datum.v > 3"),
    "y": lambda: fm.transform_filter("datum.y > 8"),
    "g": lambda: fm.transform_filter("datum.g != 'b'"),
    "count": lambda: fm.transform_filter("datum.count > 1"),
}


@dataclass(frozen=True)
class _Reach:
    """One door: the composition without its step, and each part with whether the step reaches it.

    ``binned`` colors every part by ``v``'s unit bins (the histogram runs as
    its own branch behind a statistics marker), else by ``c``.
    """

    compose: Callable[[Any], Any]
    parts: tuple[tuple[Callable[[Any], Any], bool], ...]
    binned: bool = True
    facet: str | None = None


_REACH_DOORS: dict[str, _Reach] = {
    "+s": _Reach(
        lambda c: _binned_hist(c) + _hist_points(c),
        ((_binned_hist, True), (_hist_points, True)),
        binned=False,
    ),
    "+s/rev": _Reach(
        lambda c: _hist_points(c) + _binned_hist(c),
        ((_hist_points, True), (_binned_hist, True)),
        binned=False,
    ),
    "+s/chain-left": _Reach(
        lambda c: _chained_points(c) + _binned_hist(c),
        ((_chained_points, True), (_binned_hist, True)),
        binned=False,
    ),
    "+s/chain-right": _Reach(
        lambda c: _binned_hist(c) + _chained_points(c),
        ((_binned_hist, True), (_chained_points, True)),
        binned=False,
    ),
    "+s/marker-left": _Reach(
        lambda c: _binned_hist(c) + _hist_points(c),
        ((_binned_hist, True), (_hist_points, True)),
    ),
    "+s/marker-right": _Reach(
        lambda c: _hist_points(c) + _binned_hist(c),
        ((_hist_points, True), (_binned_hist, True)),
    ),
    "+d": _Reach(
        lambda c: _binned_hist(c) + _hist_points_other(c),
        ((_binned_hist, True), (_hist_points_other, False)),
    ),
    "+d/rev": _Reach(
        lambda c: _hist_points_other(c) + _binned_hist(c),
        ((_hist_points_other, True), (_binned_hist, False)),
    ),
    "(+s)+s": _Reach(
        lambda c: (_binned_hist(c) + _hist_points(c)) + _hist_points(c),
        ((_binned_hist, True), (_hist_points, True), (_hist_points, True)),
    ),
    "(+s)+d": _Reach(
        lambda c: (_binned_hist(c) + _hist_points(c)) + _hist_points_other(c),
        ((_binned_hist, True), (_hist_points, True), (_hist_points_other, False)),
    ),
    "s+(+s)": _Reach(
        lambda c: _hist_points(c) + (_binned_hist(c) + _hist_points(c)),
        ((_hist_points, True), (_binned_hist, True), (_hist_points, True)),
    ),
    "d+(+s)": _Reach(
        lambda c: _hist_points_other(c) + (_binned_hist(c) + _hist_points(c)),
        ((_hist_points_other, True), (_binned_hist, False), (_hist_points, False)),
    ),
    "Fc": _Reach(
        lambda c: (_binned_hist(c) + _hist_points(c)).facet("c"),
        ((_binned_hist, True), (_hist_points, True)),
        facet="c",
    ),
    "LD": _Reach(
        lambda c: _binned_hist(c).layer(
            fm.Layer(data=_GROUP_ROWS_OTHER, mark="point", encoding={"x": "y", "y": "v"})
        ),
        ((_binned_hist, True), (_hist_points_other, False)),
    ),
    "LD/same": _Reach(
        lambda c: _binned_hist(c).layer(
            fm.Layer(data=_GROUP_ROWS, mark="point", encoding={"x": "y", "y": "v"})
        ),
        ((_binned_hist, True), (_hist_points, True)),
    ),
    "(+d)+s": _Reach(
        lambda c: (_binned_hist(c) + _hist_points_other(c)) + _hist_points(c),
        ((_binned_hist, True), (_hist_points_other, False), (_hist_points, True)),
    ),
    "s+(+d)": _Reach(
        lambda c: _hist_points(c) + (_binned_hist(c) + _hist_points_other(c)),
        ((_hist_points, True), (_binned_hist, True), (_hist_points_other, False)),
    ),
    "(+d).layer(LD/same)": _Reach(
        lambda c: (_binned_hist(c) + _hist_points_other(c)).layer(
            fm.Layer(data=_GROUP_ROWS, mark="point", encoding={"x": "y", "y": "v"})
        ),
        ((_binned_hist, True), (_hist_points_other, False), (_bare_hist_points, True)),
    ),
    "(+d)+d": _Reach(
        lambda c: (_binned_hist(c) + _hist_points_other(c)) + _hist_points_other(c),
        ((_binned_hist, True), (_hist_points_other, False), (_hist_points_other, False)),
    ),
}


def _reach_color(door: _Reach) -> Any:
    return _binned_color() if door.binned else "c"


def _part(build: Callable[[Any], Any], color: Any, facet: str | None) -> fm.Chart:
    part = build(color)
    return part.facet(facet) if facet is not None else part


def _expected_reach_records(door: _Reach, step: str) -> dict[str, Counter]:
    """The marks each part draws alone: filtered as the step defines where it reaches, else as written."""
    color = _reach_color(door)
    out: dict[str, Counter] = {}
    for build, reached in door.parts:
        alone = _quiet(lambda build=build: _part(build, color, door.facet))
        if reached and step != "count":
            alone = alone.transform(_REACH_STEPS[step]())
        records = _quiet(lambda alone=alone: _scene_records(alone))
        if reached and step == "count" and "bar" in records:
            records["bar"] = _counted_above_one(records["bar"])
        for kind, counted in records.items():
            out[kind] = out.get(kind, Counter()) + counted
    return out


def _count_unreadable(door: _Reach) -> bool:
    """Whether no reached part draws the histogram, so no reached chain carries ``count``."""
    return not any(reached for build, reached in door.parts if build is _binned_hist)


_REACH_CASES = [
    (door, step)
    for door in sorted(_REACH_DOORS)
    for step in sorted(_REACH_STEPS)
    if not (step == "count" and _count_unreadable(_REACH_DOORS[door]))
]


@pytest.mark.parametrize(("door", "step"), _REACH_CASES)
def test_a_compositions_step_reaches_every_operand_over_its_frame(door, step):
    """Each reached part draws what it draws alone filtered by the step; every other part, as written.

    ``v``, ``y`` and ``g`` are on every operand's source rows, so the step
    runs before the histogram bins (its bars equal ``Hist().transform(f)``
    alone) and filters the points; ``count`` exists only after the
    histogram's statistic, so the bars above count 1 draw and the points
    draw unfiltered. Rejects placing the step by who reads a chain, after
    the left's statistics only: that refused ``y`` and ``g``, which the
    statistic drops, and never reached a right operand on the same frame.
    """
    spec = _REACH_DOORS[door]
    chart = _quiet(lambda: spec.compose(_reach_color(spec)).transform(_REACH_STEPS[step]()))
    expected = _expected_reach_records(spec, step)
    assert _quiet(lambda: _scene_records(chart)) == expected
    svg = _quiet(chart.to_svg)
    drawn = Counter(signature[0] for signature in census(svg).elements())
    assert drawn["circle"] == sum(expected.get("point", Counter()).values())


#: The doors whose reached operands never carry ``count``, and the left operand's columns.
_COUNT_REFUSED = {
    door: _GROUP_ROWS_OTHER for door, spec in _REACH_DOORS.items() if _count_unreadable(spec)
}


@pytest.mark.parametrize("door", sorted(_COUNT_REFUSED))
def test_a_compositions_step_no_reached_operand_can_read_is_refused(door):
    """``count`` names no column any reached operand's rows carry: refused over the left's rows.

    Rejects leaving the step unrun in silence, which drew every operand as
    if no step had been written.
    """
    spec = _REACH_DOORS[door]
    chart = _quiet(lambda: spec.compose(_reach_color(spec)).transform(_REACH_STEPS["count"]()))
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg)
    assert str(refused.value) == _missing_column("data_filter", "count", _COUNT_REFUSED[door])


def _normalized_wire(chart: Any) -> str:
    """The chart's spec JSON with its identity-derived step names made stable."""
    return re.sub(r"_(auto|ident)_[0-9a-f]+", r"_\1_", _quiet(lambda: chart.to_spec().to_json()))


#: The spec JSON each door sends with no step written on it, pinned beside the
#: census of what it draws: the marks of its parts, each drawn alone.
_REACH_DOOR_WIRE_DIGESTS: dict[str, str] = {
    "(+d)+d": "e4474f86bcbe15fba40e11f5ac06d03cb66308f761cbba171c5d114efee0cff5",
    "(+d)+s": "ba0da136f3252f2a77501c3dd20af170885ea65959edcfa750192e70da116767",
    "(+d).layer(LD/same)": "8e7e2e1a3d5ef36e56e7aa665f3b88e0cf72ebdec109f84048995d7f3b3cf8a4",
    "(+s)+d": "0225068f2923f769483b0c8b8b64d757dd7a70ea72902db06c5a1e54cbf5c7bb",
    "(+s)+s": "4189676ce4fe7b669d685cbf6adba88cc2ec0c80f69ee662b543dcfc968f8cc9",
    "+d": "b1f785a31e05fc7b58ed0331be5004e3369fea16882d126c54290f41717ba06f",
    "+d/rev": "4428721b969a9ac999eee8123137b51e1ef574ef675ab4613301c0f95612e957",
    "+s": "b6f5a689880d5031173f3c75e3f7e6e47d80680f7abf72112b346f891d6bda9f",
    "+s/chain-left": "0d991f50b6d7da65eb18e402b374734b26018b08ce513279bebbf016f71add9a",
    "+s/chain-right": "cc3ec7da9ab48f3f8b7ecc2bdae41ff32db160a3000d3e4f4b3854eb6b066216",
    "+s/marker-left": "8da077d0d75ec55c02c4ec69814bca84d9d468a33400b19e8bc60dcd1b368193",
    "+s/marker-right": "ec64be3200fd6262c047ca5d4e234c52ccbfa63a18eb70773499388db1b86f54",
    "+s/rev": "b5c2966684b4a18a87ae5458057b43b2b772890ebb96ee63de7fa8b883813051",
    "Fc": "6f2024105a629d37a4abe509fafad73d64f3aed3aac063a9c1885b9f165a0a5f",
    "LD": "329d76774e6813038be78fa11c7c3f607718990c836e6ba0d3a981f4597c057b",
    "LD/same": "7c6c4ea1a1ed9cb6d18493f15e57dc5d28a7484e4e8b4e51b0b873c63c196f8e",
    "d+(+s)": "ece143348ab29131cb449ae0fc66195406a5604607f94a15fd24cefea19dca42",
    "s+(+d)": "ceda6e94d9bef400d46a359ffc8da0be1f89370757da23fcddda9c1b93ca7b0e",
    "s+(+s)": "fc2c01eae4b42e08942abc49d55f381ff776b315b2c7499ca1fcabf0e3ff7c19",
}


@pytest.mark.parametrize("door", sorted(_REACH_DOORS))
def test_a_composition_without_a_step_draws_its_parts_and_pins_its_wire(door):
    """No composed step reaches the wire of a composition no step was written on."""
    import hashlib

    spec = _REACH_DOORS[door]
    color = _reach_color(spec)
    chart = _quiet(lambda: spec.compose(color))
    expected: dict[str, Counter] = {}
    for build, _ in spec.parts:
        part = _quiet(lambda build=build: _part(build, color, spec.facet))
        for kind, counted in _quiet(lambda part=part: _scene_records(part)).items():
            expected[kind] = expected.get(kind, Counter()) + counted
    assert _quiet(lambda: _scene_records(chart)) == expected
    wire = _normalized_wire(chart)
    assert '"composed"' not in wire
    assert hashlib.sha256(wire.encode()).hexdigest() == _REACH_DOOR_WIRE_DIGESTS[door]


def test_a_secondary_y_over_a_flat_chart_keeps_its_step_before_the_statistic():
    """Control: a flat host's own step runs before its histogram bins, with a secondary axis or not."""
    color = _binned_color()
    step = _REACH_STEPS["y"]
    chart = _quiet(lambda: (_binned_hist(color) + SY("v")).transform(step()))
    alone = _quiet(lambda: _binned_hist(color).transform(step()))
    assert (
        _quiet(lambda: _scene_records(chart))["bar"] == _quiet(lambda: _scene_records(alone))["bar"]
    )


def test_steps_on_a_composition_run_in_the_order_written():
    """A calculate then a filter reading its output: the filter runs after the calculate on every operand."""
    color = _binned_color()
    chart = _quiet(
        lambda: (
            (_binned_hist(color) + _hist_points(color))
            .transform(fm.transform_calculate("half", "datum.y / 2"))
            .transform(fm.transform_filter("datum.half > 4"))
        )
    )
    expected = _expected_reach_records(_REACH_DOORS["+s/marker-left"], "y")
    assert _quiet(lambda: _scene_records(chart)) == expected


# A composition's step runs at most once over the rows of each operand it
# reaches, whichever order the operands come in: a step that reads its own
# output draws the same values on every door.


def _by_value(records: Counter) -> Counter:
    """*records* with every numeric value read as a number, so a computed column compares by value."""

    def value(text: str) -> Any:
        try:
            return float(text)
        except ValueError:
            return text

    return Counter({tuple((k, value(v)) for k, v in r): n for r, n in records.items()})


def _shifted(records: Counter, column: str, by: float) -> Counter:
    """*records* with *column* raised by *by*: what a step adding *by* to it once draws."""
    return Counter(
        {tuple((k, v + by if k == column else v) for k, v in r): n for r, n in records.items()}
    )


def _records_by_value(chart: Any) -> dict[str, Counter]:
    return {kind: _by_value(r) for kind, r in _quiet(lambda: _scene_records(chart)).items()}


#: The doors where the composition's step reaches the histogram.
_ONCE_DOORS = sorted(door for door, spec in _REACH_DOORS.items() if not _count_unreadable(spec))


@pytest.mark.parametrize("door", _ONCE_DOORS)
def test_a_step_reading_its_own_output_runs_once_on_every_reached_operand(door):
    """``calculate('count', 'datum.count + 100')`` raises each reached histogram bar's count by 100, once.

    The points never carry ``count``, so they draw as written. Rejects
    running a deferred copy over every later output of the chain, which ran
    it again on rows that a copy had already passed (counts raised by 200
    when the points came first).
    """
    spec = _REACH_DOORS[door]
    color = _reach_color(spec)
    step = fm.transform_calculate("count", "datum.count + 100")
    chart = _quiet(lambda: spec.compose(color).transform(step))
    expected: dict[str, Counter] = {}
    for build, reached in spec.parts:
        part = _quiet(lambda build=build: _part(build, color, spec.facet))
        for kind, counted in _records_by_value(part).items():
            if reached and build is _binned_hist:
                counted = _shifted(counted, "count", 100.0)
            expected[kind] = expected.get(kind, Counter()) + counted
    assert _records_by_value(chart) == expected


# A composition's step reading an encoding statistic's output column runs on
# that statistic's rows on every door: over the same frame, beside an operand
# over another frame, nested, and before a layer over another frame is
# attached.


def _count_bars(color: Any) -> fm.Chart:
    """One bar per (g, color) group, drawing its row count as ``count_all``."""
    return fm.Chart(_GROUP_ROWS).mark_bar().encode(x="g", y="count()", color=color)


def _group_points_other(color: Any) -> fm.Chart:
    """``_plain_points`` over a different frame holding the same rows."""
    return fm.Chart(_GROUP_ROWS_OTHER).mark_point().encode(x="g", y="y", color=color)


def _bare_group_points_other(color: Any) -> fm.Chart:
    """``_group_points_other`` with no color of its own: the records a ``Layer(data=)`` lists."""
    return fm.Chart(_GROUP_ROWS_OTHER).mark_point().encode(x="g", y="y")


def _other_layer() -> fm.Layer:
    return fm.Layer(data=_GROUP_ROWS_OTHER, mark="point", encoding={"x": "g", "y": "y"})


@dataclass(frozen=True)
class _Counted:
    """One door: the composition with the step written on it, and each part with whether it is reached."""

    compose: Callable[[Any, Any], Any]
    parts: tuple[tuple[Callable[[Any], Any], bool], ...]


_COUNTED_DOORS: dict[str, _Counted] = {
    "+s": _Counted(
        lambda c, step: (_count_bars(c) + _plain_points(c)).transform(step),
        ((_count_bars, True), (_plain_points, True)),
    ),
    "+s/rev": _Counted(
        lambda c, step: (_plain_points(c) + _count_bars(c)).transform(step),
        ((_plain_points, True), (_count_bars, True)),
    ),
    "+d": _Counted(
        lambda c, step: (_count_bars(c) + _group_points_other(c)).transform(step),
        ((_count_bars, True), (_group_points_other, False)),
    ),
    "(+s)+d": _Counted(
        lambda c, step: ((_count_bars(c) + _plain_points(c)) + _group_points_other(c)).transform(
            step
        ),
        ((_count_bars, True), (_plain_points, True), (_group_points_other, False)),
    ),
    "(+d)+s": _Counted(
        lambda c, step: ((_count_bars(c) + _group_points_other(c)) + _plain_points(c)).transform(
            step
        ),
        ((_count_bars, True), (_group_points_other, False), (_plain_points, True)),
    ),
    "LD": _Counted(
        lambda c, step: _count_bars(c).layer(_other_layer()).transform(step),
        ((_count_bars, True), (_bare_group_points_other, False)),
    ),
    "+s then LD": _Counted(
        lambda c, step: (_count_bars(c) + _plain_points(c)).transform(step).layer(_other_layer()),
        ((_count_bars, True), (_plain_points, True), (_bare_group_points_other, False)),
    ),
    "+s then +d": _Counted(
        lambda c, step: (
            (_count_bars(c) + _plain_points(c)).transform(step) + _group_points_other(c)
        ),
        ((_count_bars, True), (_plain_points, True), (_group_points_other, False)),
    ),
}


#: Each step: what it does to the bars' records, which carry ``count_all``.
_COUNTED_STEPS: dict[str, tuple[Callable[[], Any], Callable[[Counter], Counter]]] = {
    "filter": (
        lambda: fm.transform_filter("datum.count_all > 1 && datum.g == 'a'"),
        lambda records: Counter({r: n for r, n in records.items() if dict(r)["g"] == "a"}),
    ),
    "calculate": (
        lambda: fm.transform_calculate("count_all", "datum.count_all + 100"),
        lambda records: _shifted(records, "count_all", 100.0),
    ),
}


@pytest.mark.parametrize("step", sorted(_COUNTED_STEPS))
@pytest.mark.parametrize("door", sorted(_COUNTED_DOORS))
def test_a_step_reading_an_encoding_statistics_output_runs_on_its_rows_on_every_door(door, step):
    """The bars draw as the step defines over their counts; every other part draws as written.

    Rejects lowering the bars' count outside their operand's chain, where a
    step deferred inside that chain never reached it once a layer or
    operand over another frame joined the composition (refused, ``column
    'count_all' not found``).
    """
    spec = _COUNTED_DOORS[door]
    make_step, effect = _COUNTED_STEPS[step]
    chart = _quiet(lambda: spec.compose("c", make_step()))
    expected: dict[str, Counter] = {}
    for build, reached in spec.parts:
        for kind, counted in _records_by_value(_quiet(lambda build=build: build("c"))).items():
            if reached and build is _count_bars:
                counted = effect(counted)
            expected[kind] = expected.get(kind, Counter()) + counted
    assert _records_by_value(chart) == expected


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_F_a_flat_charts_step_reading_its_statistics_output_is_refused_with_the_remedy(output):
    """A flat chart's ``.transform()`` runs before its encoding statistics, so a
    step reading the count they compute is refused naming the step, the
    statistic and the remedy. Rejects the step's own missing-column text
    (``column 'count_all' not found``), which says nothing of why.
    """
    chart = mk("bar", x="g", y="count()").transform(fm.transform_filter("datum.count_all > 1"))
    render = chart.to_svg if output == "static" else lambda: chart.interactive().to_html()
    message = (
        "transform failed: ValueError: data_filter: column 'count_all' is computed by the "
        "chart's statistics (stat_aggregate), which run after the chart's .transform() steps: "
        ".transform() reads the chart's rows before its statistics. Compute 'count_all' in "
        ".transform() before this step (e.g. with fm.transform_aggregate) and bind the "
        "encoding to it"
    )
    with pytest.raises(ValueError) as refused:
        _quiet(render)
    assert str(refused.value) == message


def _screen_axis_titles(chart: fm.Chart, output: str) -> dict[str, str]:
    """Each drawn axis title by the screen axis it sits on: a rotated title is the y axis's."""
    if output == "static":
        return {
            ("y" if "rotate(" in (e.get("transform") or "") else "x"): "".join(e.itertext()).strip()
            for e in _chrome_texts(chart.to_svg())
            if e.get("font-weight") == "600" and e.get("font-size") == "13"
        }
    scene = json.loads(chart.interactive().scene_json)
    titles: dict[str, str] = {}

    def walk(nodes: list) -> None:
        for node in nodes:
            if node.get("type") == "text" and node["style"]["font_weight"] != "normal":
                titles["y" if node["style"]["angle"] else "x"] = node["content"]
            walk(node.get("children", []))

    for panel in scene["panels"]:
        walk(panel["axes"])
    return titles


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_horizontal_count_catplot_titles_the_category_and_count_axes_by_their_channels(output):
    """``catplot(kind='count', orient='h')`` titles the category axis ``g`` and the count axis ``n``.

    The count channel's title moves with the channel under the flip, as its
    scale and ticks do. Rejects keeping the explicit title on its screen
    axis, which titled both axes ``n``.
    """
    chart = _quiet(lambda: fm.catplot(df, y="g", kind="count", orient="h"))
    assert _screen_axis_titles(chart, output) == {"x": "n", "y": "g"}


_SY_ROWS = pl.DataFrame({"g": ["a", "b", "c"], "v": [1.0, 2.0, 3.0], "r": [10.0, 20.0, 30.0]})


def _sy_line(y: str = "v") -> fm.Chart:
    return fm.Chart(_SY_ROWS).mark_line().encode(x="g", y=y)


#: Every door a layer with an independent y scale reaches a non-Cartesian
#: coord through: ``SecondaryY`` sugar with ``.coord()`` after and before
#: ``+``, an independent-y ``LayerChart`` member, a composite leaf, and the
#: polar coord the same rule covers.
_SECONDARY_Y_COORD_DOORS: dict[str, tuple[Callable[[], Any], str]] = {
    "SY/coord-after": (lambda: (_sy_line() + SY("r")).coord(fm.CoordFlip()), "fm.CoordFlip"),
    "SY/coord-before": (lambda: _sy_line().coord(fm.CoordFlip()) + SY("r"), "fm.CoordFlip"),
    "LCx/member-flip": (
        lambda: fm.LayerChart(
            _sy_line().coord(fm.CoordFlip()), _sy_line("r"), resolve={"y": "independent"}
        ),
        "fm.CoordFlip",
    ),
    "hconcat-leaf": (
        lambda: fm.hconcat(_sy_line(), (_sy_line() + SY("r")).coord(fm.CoordFlip())),
        "fm.CoordFlip",
    ),
    "SY/polar": (lambda: (_sy_line() + SY("r")).coord(fm.CoordPolar()), "fm.CoordPolar"),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_SECONDARY_Y_COORD_DOORS))
def test_a_secondary_y_under_a_flip_or_polar_coord_is_refused_with_the_remedy(door, output):
    """A right y axis has no place once the coord moves y off the vertical axis.

    Refused naming the coord and the remedy on every door and output.
    Rejects rendering, which drew the secondary axis under the other
    channel's title under a flip and dropped it under a polar coord.
    """
    build, coord = _SECONDARY_Y_COORD_DOORS[door]
    message = (
        f"SecondaryY (a layer with an independent y scale) is not supported under {coord}: its right y "
        "axis has no place once the coord moves y off the vertical axis; draw the series on a "
        "Cartesian panel (put the categories on x and drop the coord), or draw it on the shared y "
        "axis"
    )
    chart = build()
    render = chart.to_svg if output == "static" else lambda: chart.interactive().scene_json
    with pytest.raises(ValueError, match=re.escape(message) + "$"):
        _quiet(render)


@pytest.mark.parametrize(
    "build",
    [lambda: _sy_line() + SY("r"), lambda: _sy_line().coord(fm.CoordFlip())],
    ids=["SY-without-coord", "flip-without-SY"],
)
def test_a_secondary_y_or_a_flip_alone_still_renders(build):
    """The refusal needs both parts: either alone draws, statically and interactively."""
    chart = build()
    assert _quiet(chart.to_svg).startswith("<svg")
    assert _quiet(lambda: chart.interactive().scene_json)


# Every composed copy of a step sits in exactly one branch per distinct
# reached (frame, chain), and nowhere else.


def _composed_sites(steps: list[dict], branch: str | None = None) -> list[tuple[str | None, str]]:
    """``(top-level branch, group)`` of each composed step in *steps*, at any depth."""
    out: list[tuple[str | None, str]] = []
    for step in steps:
        if step["type"] == "composed":
            out.append((branch, step["group"]))
        elif step["type"] == "branch":
            out += _composed_sites(step["transforms"], step["name"] if branch is None else branch)
    return out


@pytest.mark.parametrize("door", sorted(_REACH_DOORS))
def test_each_composed_copy_sits_once_in_each_reached_operands_branch_and_nowhere_else(door):
    """One copy of the step per distinct reached (frame, chain) branch, and nowhere else.

    Every operand lowers to exactly one branch, and operands over one frame
    whose chains are equal read the same branch. The branches that draw a
    reached part hold the copy: every layer a reached part contributes reads
    inside one of them. Rejects copies
    placed by route (a top-level copy, a copy in a branch an operand over
    another frame reads, two copies in one operand's chain).
    """
    spec = _REACH_DOORS[door]
    chart = _quiet(
        lambda: spec.compose(_reach_color(spec)).transform(fm.transform_filter("datum.v > 3"))
    )
    wire = _wire(chart)
    steps = wire["transforms"]
    sites = _composed_sites(steps)
    assert len({group for _, group in sites}) == 1
    assert all(branch is not None for branch, _ in sites)
    assert len(sites) == len({branch for branch, _ in sites})
    branches = operand_branches(wire)
    assert len(steps) == len(branches)
    layer_branches = [
        max(
            (
                b
                for b in branches
                if layer["data_source"] in (b,) or layer["data_source"].startswith(f"{b}_")
            ),
            key=len,
        )
        for layer in wire["layers"]
    ]
    reached: list[bool] = []
    for build, is_reached in spec.parts:
        part_layers = _wire(_quiet(lambda build=build: _part(build, "c", spec.facet))).get("layers")
        reached += [is_reached] * (len(part_layers) if part_layers else 1)
    assert len(reached) == len(layer_branches)
    holding = {branch for branch, _ in sites}
    for branch, is_reached in zip(layer_branches, reached, strict=True):
        assert (branch in holding) == is_reached, branch


# A step written on a composition that names its output publishes it inside
# each operand it reaches, under that operand's own scoped key, and leaves
# every operand's rows as they were.


def _named(step: dict, name: str) -> dict:
    return {**step, "name": name}


def test_a_named_composition_step_publishes_once_in_each_reached_operand():
    """One named copy per reached operand's branch, none in another frame's; every mark draws as written."""
    color = _binned_color()
    spec = _REACH_DOORS["(+s)+d"]
    step = _named(fm.transform_calculate("half", "datum.y / 2"), "halved")
    chart = _quiet(lambda: spec.compose(color).transform(step))
    branches = _wire(chart)["transforms"]
    named = [
        [t["step"].get("name") for t in branch["transforms"] if t["type"] == "composed"]
        for branch in branches
    ]
    assert named == [["halved"], ["halved"], []]
    assert _quiet(lambda: _scene_records(chart)) == _quiet(
        lambda: _scene_records(spec.compose(color))
    )


def test_a_named_composition_step_no_reached_operand_can_read_is_refused():
    """A named ``filter('datum.count > 1')`` over operands that never carry ``count`` is refused."""
    spec = _REACH_DOORS["d+(+s)"]
    step = _named(fm.transform_filter("datum.count > 1"), "kept")
    chart = _quiet(lambda: spec.compose(_reach_color(spec)).transform(step))
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg)
    assert str(refused.value) == _missing_column("data_filter", "count", _GROUP_ROWS_OTHER)


# A box and a violin over one frame each read their own statistic's outputs:
# the composition renders both.

_BOX_PARTS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "box+violin": (
        lambda: fm.Chart(_GROUP_ROWS).mark_boxplot().encode(x="g", y="y"),
        lambda: fm.Chart(_GROUP_ROWS).mark_violin().encode(x="g", y="y"),
    ),
    "violin+box": (
        lambda: fm.Chart(_GROUP_ROWS).mark_violin().encode(x="g", y="y"),
        lambda: fm.Chart(_GROUP_ROWS).mark_boxplot().encode(x="g", y="y"),
    ),
    "qq(y)+qq(v)": (
        lambda: fm.Chart(_GROUP_ROWS).mark_qq().encode(x="y"),
        lambda: fm.Chart(_GROUP_ROWS).mark_qq().encode(x="v"),
    ),
}


@pytest.mark.parametrize("door", sorted(_BOX_PARTS))
def test_two_composite_marks_publishing_one_output_name_each_draw_their_own(door):
    """Each composite draws its own statistic's marks. Rejects refusing the composition as a name collision."""
    a, b = _BOX_PARTS[door]
    check_rendering(Case(lambda: a() + b(), (a, b)))


# A flat chart's own steps run before its encoding statistic, whichever of
# .encode() and .transform() is written first.

_V_ABOVE_TWO = df.filter(pl.col("v") > 2)

#: Each flat chart with an encoding statistic, built over a frame.
_FLAT_STATISTICS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "count": lambda data: mk("bar", data, x="g", y="count()"),
    "mean": lambda data: mk("bar", data, x="g", y="mean(y)"),
    "binned-count": lambda data: mk("bar", data, x=fm.X("y", bin=True), y="count()"),
    "colored-count": lambda data: mk("bar", data, x="g", y="count()", color="c"),
}


@pytest.mark.parametrize("order", ["transform-after-encode", "transform-before-encode"])
@pytest.mark.parametrize("statistic", sorted(_FLAT_STATISTICS))
def test_a_flat_charts_step_runs_before_its_encoding_statistic(statistic, order):
    """``filter('datum.v > 2')`` filters the rows the statistic computes over, in either order.

    The oracle is the statistic over the filtered frame. Rejects running
    the step on the statistic's output, which refused ``v`` (gone after
    the aggregate) or filtered the counted bars.
    """
    build = _FLAT_STATISTICS[statistic]
    step = fm.transform_filter("datum.v > 2")
    if order == "transform-after-encode":
        chart = build(df).transform(step)
    else:
        written = mk("bar", df).transform(step)
        chart = build(df)
        chart = written.encode(**{ch: v for ch, v in chart._encoding.items()})
    oracle = build(_V_ABOVE_TWO)
    assert _records_by_value(chart) == _records_by_value(oracle)
    assert census(_quiet(chart.to_svg)) == census(_quiet(oracle.to_svg))


# A chart with no rows to draw after its transforms draws its panel frame and
# axes over an explicit domain, else the unit fallback, and warns; a faceted
# one draws no facet panel and warns the same way.

_NO_ROWS_WARNING = "the chart has no rows to draw after its transforms; its panel is drawn empty"
_NO_FACET_ROWS_WARNING = (
    "the chart has no rows to draw after its transforms; no facet panel is drawn"
)


def _nothing_above(field: str = "y") -> Any:
    return fm.transform_filter(f"datum.{field} > 100")


#: Each door whose every layer has no rows; ``panels`` counts the plot clip
#: groups it draws: one per empty chart, and the hconcat's drawing leaf too.
_NO_ROWS_DOORS: dict[str, tuple[Callable[[], Any], int, int]] = {
    "F": (lambda: P().transform(_nothing_above()), 1, 0),
    "+": (lambda: (P() + P(y="r")).transform(_nothing_above()), 1, 0),
    "LD": (
        lambda: (
            P()
            .transform(_nothing_above())
            .layer(
                fm.Layer(
                    data=df2.filter(pl.col("y") > 100), mark="line", encoding={"x": "v", "y": "y"}
                )
            )
        ),
        1,
        0,
    ),
    "hconcat": (lambda: fm.hconcat(P().transform(_nothing_above()), P(y="r")), 2, 8),
}


def _warned(render: Callable[[], Any]) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render()
    return out, [str(w.message) for w in caught]


@pytest.mark.parametrize("door", sorted(_NO_ROWS_DOORS))
def test_a_chart_with_no_rows_draws_its_frame_and_axes_and_warns(door):
    """Static and interactive: a plot area and axis text, no mark from the empty chart, one warning.

    Rejects skipping the panel, which rendered a blank image with no
    warning.
    """
    build, panels, marks = _NO_ROWS_DOORS[door]
    chart = _quiet(build)
    svg, caught = _warned(chart.to_svg)
    assert caught.count(_NO_ROWS_WARNING) == 1
    assert len(_clip_groups(_svg_root(svg))) == panels
    assert sum(census(svg).values()) == marks
    assert any(re.fullmatch(r"-?[\d.,]+", text) for text in chrome_text(svg))
    scene, caught = _warned(lambda: _scene(chart))
    assert caught.count(_NO_ROWS_WARNING) == 1
    assert len(scene["panels"]) == panels
    assert sum(len(b["nodes"]) for p in scene["panels"] for b in p["marks"]) == marks


def test_an_empty_chart_draws_its_explicit_domain():
    """An explicit x domain [0, 10] draws its ticks over no rows."""
    chart = mk("point", x=fm.X("v", scale=fm.LinearScale(domain=[0, 10])), y="y").transform(
        _nothing_above()
    )
    svg, caught = _warned(chart.to_svg)
    assert caught.count(_NO_ROWS_WARNING) == 1
    texts = chrome_text(svg)
    assert "0" in texts and "10" in texts


def test_a_faceted_chart_with_no_rows_draws_no_facet_panel_and_warns():
    chart = P().transform(_nothing_above()).facet("c")
    svg, caught = _warned(chart.to_svg)
    assert caught.count(_NO_FACET_ROWS_WARNING) == 1
    assert _clip_groups(_svg_root(svg)) == []
    scene, caught = _warned(lambda: _scene(chart))
    assert caught.count(_NO_FACET_ROWS_WARNING) == 1
    assert scene["panels"] == []


def test_a_layer_charts_empty_member_warns_nothing_when_a_sibling_draws():
    """A static ``LayerChart`` with one empty member and one that draws warns nothing.

    The static overlay renders each member through its own leaf and judges
    the group of leaves together: a sibling that draws suppresses an empty
    leaf's own warning, the same as the merged flat chart (used by
    ``Chart.layer``/``+`` and every interactive render) already judges every
    layer as one chart. Rejects warning once per empty member when every
    member is empty (the group still warns, but once, like one chart does),
    and rejects an ``hconcat`` sibling (a separate chart, not a layer)
    suppressing the same warning.
    """
    empty = P().transform(_nothing_above())
    full = P()
    one_draws = fm.LayerChart(empty, full)
    svg, caught = _warned(one_draws.to_svg)
    assert caught == []
    assert sum(census(svg).values()) > 0
    _scene_out, caught_interactive = _warned(lambda: _scene(one_draws))
    assert caught_interactive == []

    all_empty = fm.LayerChart(empty, P().transform(_nothing_above()))
    _svg, caught_all = _warned(all_empty.to_svg)
    assert caught_all.count(_NO_ROWS_WARNING) == 1
    _scene_all, caught_all_interactive = _warned(lambda: _scene(all_empty))
    assert caught_all_interactive.count(_NO_ROWS_WARNING) == 1

    all_empty_three = fm.LayerChart(
        empty, P().transform(_nothing_above()), P(df2).transform(_nothing_above())
    )
    _svg3, caught_all_three = _warned(all_empty_three.to_svg)
    assert caught_all_three.count(_NO_ROWS_WARNING) == 1
    _scene3, caught_all_three_interactive = _warned(lambda: _scene(all_empty_three))
    assert caught_all_three_interactive.count(_NO_ROWS_WARNING) == 1

    hc = fm.hconcat(empty, full)
    _hc_svg, caught_hc = _warned(hc.to_svg)
    assert caught_hc.count(_NO_ROWS_WARNING) == 1

    hc_all_empty = fm.hconcat(all_empty, P())
    _hc2_svg, caught_hc2 = _warned(hc_all_empty.to_svg)
    assert caught_hc2.count(_NO_ROWS_WARNING) == 1


# Zero-row data (a source with no rows at all, not a filter that emptied one)
# draws its frame and axes and warns once on every route, the same way
# filter-emptied data already does.

#: Zero-row twin of ``df``, carrying the schema but no rows.
_Z = df.head(0)

#: Each door over zero-row data, building its chart.
_ZERO_ROW_DOORS: dict[str, Callable[[], Any]] = {
    "F": lambda: P(_Z),
    "+": lambda: P(_Z) + P(_Z, y="r"),
    "L": lambda: P(_Z).layer(fm.Layer(mark="line", encoding={"x": "v", "y": "y"})),
    "LCs": lambda: fm.LayerChart(P(_Z), P(_Z, y="y")),
    "LCm": lambda: fm.LayerChart(P(_Z), P(_Z, y="y"), resolve={"y": "independent"}),
}


@pytest.mark.parametrize("door", sorted(_ZERO_ROW_DOORS))
def test_zero_row_data_draws_its_frame_and_warns_once_on_every_route(door):
    """Zero-row data draws its frame and axes and warns once, on every layering door.

    Static and interactive agree. Rejects a blank placeholder comment with
    no warning, and rejects a typed all-empty refusal that leaves a
    ``LayerChart`` unrendered.
    """
    chart = _ZERO_ROW_DOORS[door]()
    svg, caught = _warned(chart.to_svg)
    assert caught.count(_NO_ROWS_WARNING) == 1
    assert len(_clip_groups(_svg_root(svg))) == (2 if door == "LCs" else 1)
    assert sum(census(svg).values()) == 0
    assert any(re.fullmatch(r"-?[\d.,]+", text) for text in chrome_text(svg))
    scene, caught = _warned(lambda: _scene(chart))
    assert caught.count(_NO_ROWS_WARNING) == 1
    assert len(scene["panels"]) == 1
    assert sum(len(b["nodes"]) for p in scene["panels"] for b in p["marks"]) == 0


@pytest.mark.parametrize("door", sorted(_ZERO_ROW_DOORS))
def test_zero_row_data_renders_as_its_filter_emptied_twin(door):
    """A zero-row source and a filter-emptied one render byte-identically, on every door.

    Both reach the same empty-batch render path, so a zero-row chart's SVG
    equals the same door built over ``df`` with a filter that drops every
    row, and their interactive scenes' panels are equal.
    """
    zero_chart = _ZERO_ROW_DOORS[door]()
    if door == "F":
        filtered_chart = P().transform(_nothing_above())
    elif door == "+":
        filtered_chart = (P() + P(y="r")).transform(_nothing_above())
    elif door == "L":
        filtered_chart = (
            P()
            .transform(_nothing_above())
            .layer(fm.Layer(mark="line", encoding={"x": "v", "y": "y"}))
        )
    elif door == "LCs":
        filtered_chart = fm.LayerChart(
            P().transform(_nothing_above()), P(y="y").transform(_nothing_above())
        )
    else:
        filtered_chart = fm.LayerChart(
            P().transform(_nothing_above()),
            P(y="y").transform(_nothing_above()),
            resolve={"y": "independent"},
        )
    zero_svg, _ = _warned(zero_chart.to_svg)
    filtered_svg, _ = _warned(filtered_chart.to_svg)
    assert zero_svg == filtered_svg
    zero_scene, _ = _warned(lambda: _scene(zero_chart))
    filtered_scene, _ = _warned(lambda: _scene(filtered_chart))
    assert zero_scene["panels"] == filtered_scene["panels"]


def test_zero_row_null_typed_columns_draw_the_frame():
    """A Null-dtype zero-row frame (no column ever typed) draws its frame too."""
    null_df = pl.DataFrame({"v": [], "y": []})
    chart = fm.Chart(null_df).mark_point().encode(x="v", y="y")
    for build in (chart, fm.LayerChart(chart, chart)):
        svg, caught = _warned(build.to_svg)
        assert caught.count(_NO_ROWS_WARNING) == 1
        assert sum(census(svg).values()) == 0
        assert any(re.fullmatch(r"-?[\d.,]+", text) for text in chrome_text(svg))
        scene, caught = _warned(lambda: _scene(build))
        assert caught.count(_NO_ROWS_WARNING) == 1


def test_a_faceted_chart_over_zero_rows_warns_as_when_filtered():
    """A ``.facet()`` over genuinely zero-row data warns, as a filter-emptied facet does."""
    chart = P(_Z).facet("g")
    svg, caught = _warned(chart.to_svg)
    assert caught.count(_NO_FACET_ROWS_WARNING) == 1
    assert _clip_groups(_svg_root(svg)) == []
    scene, caught = _warned(lambda: _scene(chart))
    assert caught.count(_NO_FACET_ROWS_WARNING) == 1
    assert scene["panels"] == []


def test_a_layer_charts_mixed_empty_members_warn_once():
    """One zero-row member beside a filter-emptied one still warns exactly once, either route."""
    mixed = fm.LayerChart(P(_Z), P().transform(_nothing_above()))
    svg, caught = _warned(mixed.to_svg)
    assert caught.count(_NO_ROWS_WARNING) == 1
    _scene_out, caught_interactive = _warned(lambda: _scene(mixed))
    assert caught_interactive.count(_NO_ROWS_WARNING) == 1

    one_draws = fm.LayerChart(P(_Z), P())
    svg2, caught2 = _warned(one_draws.to_svg)
    assert caught2 == []
    assert census(svg2) == census(_quiet(P().to_svg))


def test_a_grid_of_zero_row_charts_still_refuses():
    """A grid composition of zero-row children keeps refusing with its exact message.

    Zero-row data draws a frame on every non-grid door, but a grid cell has
    no data of its own to size a frame from (the sized-hole lowering
    requires at least one child with rows), so the all-empty grid refusal is
    unchanged.
    """
    with pytest.raises(
        ValueError, match=r"ConcatChart: every child's data is empty; nothing to render"
    ):
        fm.concat(P(_Z), P(_Z), columns=2).to_svg()
    with pytest.raises(
        ValueError, match=r"HConcatChart: every child's data is empty; nothing to render"
    ):
        fm.hconcat(P(_Z), P(_Z)).to_svg()
    with pytest.raises(
        ValueError, match=r"ConcatChart: every child's data is empty; nothing to render"
    ):
        fm.concat(P(_Z), P(_Z), columns=2).interactive()
    with pytest.raises(
        ValueError, match=r"HConcatChart: every child's data is empty; nothing to render"
    ):
        fm.hconcat(P(_Z), P(_Z)).interactive()


# Facet panels come from the rows the layers draw, in the facet partition's
# order whichever operand comes first, and a value no layer draws has no
# panel.

#: ``df``'s ``c`` with a value only the second half of the rows hold, so the
#: partition order is p, q, s.
_FACET_ROWS = df.with_columns(
    pl.Series("c", ["p", "p", "q", "q", "s", "s", "q", "p"]),
)


def _facet_titles(svg: str) -> list[str]:
    """The facet strip titles, in panel order: the chrome text naming a value of ``c``."""
    return [t for t in chrome_text(svg) if t in {"p", "q", "s"}]


def _only_s() -> fm.Chart:
    return P(_FACET_ROWS).transform(fm.transform_filter("datum.c == 's'"))


@pytest.mark.parametrize(
    "build",
    [
        lambda: (_only_s() + line(_FACET_ROWS)).facet("c"),
        lambda: (line(_FACET_ROWS) + _only_s()).facet("c"),
    ],
    ids=["filtered-left", "filtered-right"],
)
def test_facet_panels_follow_the_partition_order_in_either_operand_order(build):
    """Rejects ordering panels by the chart rows first and the layers after, which put ``s`` first."""
    assert _facet_titles(_quiet(build().to_svg)) == ["p", "q", "s"]


def test_a_facet_value_no_layer_draws_has_no_panel():
    """Both operands drop ``s``: no ``s`` panel, as on the flat chart.

    The binned histogram runs as its own branch, so the chart-level rows
    are the unread source, which still holds ``s``.
    """
    drop = fm.transform_filter("datum.c != 's'")
    hist = fm.Chart(_FACET_ROWS).mark_histogram().encode(x="y", color=fm.Color("z", bin=True))
    chart = (hist.transform(drop) + P(_FACET_ROWS).transform(drop)).facet("c")
    flat = P(_FACET_ROWS).transform(drop).facet("c")
    assert _facet_titles(_quiet(chart.to_svg)) == _facet_titles(_quiet(flat.to_svg)) == ["p", "q"]


#: A frame whose only facet value, ``s``, no other frame holds, and which has
#: no ``y`` column for a facet sort by ``sum(y)`` to read.
_S_ONLY = pl.DataFrame({"v3": [1.0, 2.0], "w3": [3.0, 4.0], "c": ["s", "s"]})
_SUM_Y_DESCENDING = {"field": "y", "op": "sum", "order": "descending"}

#: Each door a facet is declared on, after the composition.
_FACET_DECLARATIONS: dict[str, Callable[[fm.Chart], fm.Chart]] = {
    "facet()": lambda chart: chart.facet("c", sort=_SUM_Y_DESCENDING),
    "encode(facet=)": lambda chart: chart.encode(facet=fm.Facet("c", sort=_SUM_Y_DESCENDING)),
}


@pytest.mark.parametrize("door", sorted(_FACET_DECLARATIONS))
def test_a_partial_facet_sort_says_which_values_it_left_unsorted(door):
    """``s``'s rows lack ``y``: the other panels sort, ``s`` sorts as a value with no rows, and one warning names it.

    Rejects the warning that said every panel fell back to insertion order
    though the others were sorted, and (``encode(facet=)``) dropping the
    other frame's rows as null facet values, which drew no ``s`` panel.
    """
    chart = _FACET_DECLARATIONS[door](P() + mk("point", _S_ONLY, x="v3", y="w3"))
    svg, caught = _warned(chart.to_svg)
    assert caught == [
        'facet sort left the values ["s"] unsorted: their rows in the layer that draws '
        "them carry no 'y' value, so they sort as values with no rows; the other panels "
        "are sorted"
    ]
    sums = df.group_by("c").agg(pl.col("y").sum()).sort("y", descending=True)["c"].to_list()
    assert _facet_titles(svg) == [*sums, "s"]


#: Per ``g`` panel, ``f``'s row count falls as its DISTINCT-value count
#: rises: a=6 rows/1 distinct, b=2 rows/2 distinct, c=4 rows/3 distinct.
#: Counting observations (the pre-statistic rows) orders a, c, b; counting
#: ``f``'s drawn (post-aggregation, one row per distinct value) rows would
#: instead order c, b, a -- the opposite order, not a tie, so misreading
#: the grain is a real discriminating difference, not an accident of ties.
_FACET_COUNT_SORT_ROWS = pl.DataFrame(
    {
        "g": ["a"] * 6 + ["b"] * 2 + ["c"] * 4,
        "y": [3.0] * 6 + [8.0] * 2 + [5.0] * 4,
        "f": ["m"] * 6 + ["m", "n"] + ["m", "n", "o", "p"],
    }
)


def test_a_facet_count_sort_counts_observations_not_the_drawn_rows():
    """``facet(sort={'field': ..., 'op': 'count'})`` counts raw rows, even when the field is also drawn.

    ``f`` is both the chart's own ``x`` (so the drawn, post-aggregation
    rows carry a column named ``f`` too) and the sort's field: a sort that
    read the drawn rows whenever the field happens to be present there
    would count distinct ``f`` values per panel instead of observations,
    giving the opposite panel order. Rejects that misreading.
    """
    chart = (
        fm.Chart(_FACET_COUNT_SORT_ROWS)
        .mark_bar()
        .encode(x="f", y="mean(y)")
        .facet("g", sort={"field": "f", "op": "count", "order": "descending"})
    )
    svg, caught = _warned(chart.to_svg)
    assert caught == []
    assert [t for t in chrome_text(svg) if t in {"a", "b", "c"}] == ["a", "c", "b"]


# A step written on a composition that drops a column an operand's statistic
# reads is refused naming the step, the column and the remedy.


def test_a_composition_step_dropping_a_column_a_statistic_reads_is_refused():
    """``(box + points).transform(aggregate)`` drops ``y`` before the box's statistic reads it.

    Rejects the statistic's own missing-column text, which named no step the
    caller wrote.
    """
    chart = _quiet(lambda: (_box("g") + _user_points("g")).transform(_user_mean()))
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg)
    assert str(refused.value) == (
        "transform failed: ValueError: stat_box_stats: column 'y' was dropped by the step "
        "written on the composition's .transform() (data_aggregate), which runs on this "
        "operand's rows before its statistics; keep 'y' in that step's output, or write the "
        "step on the operands it should change instead of on the composition"
    )


# A figure's statistics run behind its chain's statistics marker, after every
# step the caller writes on the figure: a `.transform(filter(...))` on a
# figure filters the rows its statistic reads, as on a hand-built chart.

#: Figure rows: a group, two numeric series and a wide-matrix row id.
_FIGURE_ROWS = pl.DataFrame(
    {
        "g": ["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 20.3],
        "row": ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"],
    }
)
_FIGURE_FILTER = "datum.x > 3"
_FIGURE_ROWS_KEPT = _FIGURE_ROWS.filter(pl.col("x") > 3)

#: Each figure whose statistic a figure helper lowers, built over a frame.
_STATISTIC_FIGURES: dict[str, Callable[[pl.DataFrame], Any]] = {
    "catplot-count": lambda d: fm.catplot(d, x="g", kind="count"),
    "catplot-point": lambda d: fm.catplot(d, x="g", y="y", kind="point", seed=0),
    "displot-ecdf": lambda d: fm.displot(d, x="y", kind="ecdf"),
    "displot-hist-kde": lambda d: fm.displot(d, x="y", kde=True),
    "displot-hist-rug": lambda d: fm.displot(d, x="y", rug=True),
    "displot-rug-kde": lambda d: fm.displot(d, x="y", kind="rug", kde=True),
    "residplot": lambda d: fm.residplot(d, x="x", y="y"),
    "residplot-lowess": lambda d: fm.residplot(d, x="x", y="y", lowess=True),
    "lmplot": lambda d: fm.lmplot(d, x="x", y="y"),
    "heatmap": lambda d: fm.heatmap(d.select("row", "x", "y"), annot=False),
    "heatmap-annot": lambda d: fm.heatmap(d.select("row", "x", "y")),
}


@pytest.mark.parametrize("figure", sorted(_STATISTIC_FIGURES))
def test_a_figures_transform_filters_the_rows_its_statistic_reads(figure):
    """The figure filtered by ``x > 3`` draws what the figure over the kept rows draws.

    Rejects running the caller's step after the figure's statistic, which
    refused ``x`` (gone after the statistic) or filtered its output.
    """
    build = _STATISTIC_FIGURES[figure]
    chart = _quiet(lambda: build(_FIGURE_ROWS).transform(fm.transform_filter(_FIGURE_FILTER)))
    oracle = _quiet(lambda: build(_FIGURE_ROWS_KEPT))
    assert _records_by_value(chart) == _records_by_value(oracle)
    assert census(_quiet(chart.to_svg)) == census(_quiet(oracle.to_svg))


# Two operand frames are one frame when they hold the same schema and equal
# values, NaN equal to NaN: a clone, a table built twice and a LazyFrame
# collected again by a nested `+` are reached as the frame they copy.

_NAN_COLUMNS = {
    "x": [1.0, 4.0, 6.0, 7.0, 8.0],
    "y": [2.0, 3.0, 1.0, 5.0, 4.0],
    "w": [1.0, float("nan"), 2.0, 3.0, 4.0],
}
_NAN_ROWS = pl.DataFrame(_NAN_COLUMNS)
_OTHER_NAN_ROWS = pl.DataFrame({"x": [2.0, 9.0], "y": [1.0, 2.0], "q": [0.5, 0.5]})


def _same_frames(spelling: str) -> tuple[Any, Any]:
    """The frame as two operands receive it: one object, a clone, rebuilt, or one LazyFrame."""
    import pyarrow as pa

    if spelling == "same-object":
        return _NAN_ROWS, _NAN_ROWS
    if spelling == "clone":
        return _NAN_ROWS, _NAN_ROWS.clone()
    if spelling == "pyarrow-twice":
        return pa.table(_NAN_COLUMNS), pa.table(_NAN_COLUMNS)
    lazy = _NAN_ROWS.lazy()
    return lazy, lazy


def _nan_points(data: Any) -> fm.Chart:
    return fm.Chart(data).mark_point().encode(x="x", y="y")


def _nan_ticks(data: Any) -> fm.Chart:
    return fm.Chart(data).mark_tick().encode(x="x")


#: Each door: the composition of the frame's two receipts, and the operands over it.
_NAN_DOORS: dict[str, Callable[[Any, Any], Any]] = {
    "(+d)+s": lambda a, b: (_nan_points(a) + _nan_points(_OTHER_NAN_ROWS)) + _nan_ticks(b),
    "s+(+d)": lambda a, b: _nan_points(a) + (_nan_ticks(b) + _nan_points(_OTHER_NAN_ROWS)),
    "(+d).layer(LD/same)": lambda a, b: (_nan_points(a) + _nan_points(_OTHER_NAN_ROWS)).layer(
        fm.Layer(data=b, mark="tick", encoding={"x": "x"})
    ),
}


def _marks_by_kind(chart: Any) -> Counter:
    """The interactive scene's marks per kind, over every panel."""
    out: Counter = Counter()
    for panel in _scene(chart)["panels"]:
        for batch in panel["marks"]:
            out[batch["kind"]] += len(batch["nodes"])
    return out


@pytest.mark.parametrize("spelling", ["clone", "lazy", "pyarrow-twice", "same-object"])
@pytest.mark.parametrize("door", sorted(_NAN_DOORS))
def test_a_frame_holding_nan_is_one_frame_however_its_operands_receive_it(door, spelling):
    """``filter('datum.x > 5')`` keeps 3 ticks and 3 of the frame's points; the other frame's 2 points stay.

    Rejects Arrow table equality, which reads NaN unequal to itself, so a
    clone or a re-collected LazyFrame became a second frame the
    composition's step never reached (5 ticks).
    """
    a, b = _same_frames(spelling)
    chart = _quiet(lambda: _NAN_DOORS[door](a, b).transform(fm.transform_filter("datum.x > 5")))
    assert _quiet(lambda: _marks_by_kind(chart)) == Counter({"tick": 3, "point": 5})


@pytest.mark.parametrize("door", sorted(_NAN_DOORS))
def test_the_same_values_under_another_dtype_are_another_frame(door):
    """Control: ``x`` as integers is another frame, so the step does not reach its ticks (5)."""
    ints = _NAN_ROWS.with_columns(pl.col("x").cast(pl.Int64))
    chart = _quiet(
        lambda: _NAN_DOORS[door](_NAN_ROWS, ints).transform(fm.transform_filter("datum.x > 5"))
    )
    assert _quiet(lambda: _marks_by_kind(chart)) == Counter({"tick": 5, "point": 5})


#: The SVG a clone of a frame holding no NaN draws on each door, unchanged by
#: the NaN-aware frame identity.
_NO_NAN_CLONE_DIGESTS: dict[str, str] = {
    "(+d)+s": "1c979f6ddd1ece48579516c1aa2eafe96df90aafc6126dd2167ba7807922dd26",
    "(+d).layer(LD/same)": "1c979f6ddd1ece48579516c1aa2eafe96df90aafc6126dd2167ba7807922dd26",
    "s+(+d)": "82a38113738e96b1cd83d7019648b32dd1b92195401b7be3c4631c97afbc52b8",
}


@pytest.mark.parametrize("door", sorted(_NAN_DOORS))
def test_a_clone_of_a_frame_without_nan_draws_its_pinned_bytes(door):
    import hashlib

    plain = _NAN_ROWS.with_columns(pl.col("w").fill_nan(0.0))
    chart = _quiet(
        lambda: _NAN_DOORS[door](plain, plain.clone()).transform(fm.transform_filter("datum.x > 5"))
    )
    svg = _quiet(chart.to_svg)
    assert hashlib.sha256(svg.encode()).hexdigest() == _NO_NAN_CLONE_DIGESTS[door]


# On a + composition, a figure's statistics join the left operand's chain.


def _left_histogram() -> fm.Chart:
    """A left operand whose mark lowers a statistic of its own: the histogram's unnamed Bin."""
    return fm.Chart(df).mark_histogram().encode(x="v")


#: Where the step sits among the left operand's statistics, by reading.
_FIGURE_STATISTIC_ORDER = {
    "rows": ["figure_bins", None],
    "tail": [None, "figure_bins"],
}


@pytest.mark.parametrize("reading", sorted(_FIGURE_STATISTIC_ORDER))
def test_a_figure_statistic_on_a_pending_mark_takes_its_place_among_the_marks_statistics(reading):
    """The pending histogram resolves first: two lowerings, the step placed by *reading*, the resolved chain.

    The figure's own step opens its own lowering beside the histogram's
    Bin lowering, each behind its own marker. Rejects recording the step
    before the mark resolves, which put it ahead of the histogram's bins
    whatever the reading.
    """
    step = fm.Bin(field="v", name="figure_bins")
    steps = _wire(_quiet(lambda: _left_histogram()._with_statistics(step, reading=reading)))[
        "transforms"
    ]
    resolved = _quiet(
        lambda: _left_histogram()._resolve_pending()._with_statistics(step, reading=reading)
    )
    assert steps.count(_STATISTICS_MARKER) == 2
    assert [t.get("name") for t in steps if t["type"] == "bin"] == _FIGURE_STATISTIC_ORDER[reading]
    assert steps == _wire(resolved)["transforms"]


@pytest.mark.parametrize("reading", sorted(_FIGURE_STATISTIC_ORDER))
def test_a_figure_statistic_on_a_composition_with_a_pending_mark_takes_its_place_among_its_statistics(
    reading,
):
    """``(point + rule).mark_histogram()``: the pending mark resolves first, into the left operand.

    The left branch carries two lowerings, each behind its own marker: the
    histogram's Bin, and the step placed by *reading* among them, the
    chain the resolved composition gives (``_attach_host``). Rejects
    placing the step before the mark resolves, which put it ahead of the
    bins whatever the reading.
    """
    from ferrum.chart import _attach_host

    step = fm.Bin(field="v", name="figure_bins")

    def pending() -> fm.Chart:
        return (P() + mk("rule", x="v")).mark_histogram().encode(x="v")

    chart = _quiet(lambda: pending()._with_statistics(step, reading=reading))
    resolved = _quiet(lambda: _attach_host(pending())._with_statistics(step, reading=reading))
    left = _wire(chart)["transforms"][0]["transforms"]
    assert left.count(_STATISTICS_MARKER) == 2
    assert [t.get("name") for t in left if t["type"] == "bin"] == _FIGURE_STATISTIC_ORDER[reading]
    assert _wire(chart)["transforms"] == _wire(resolved)["transforms"]


def test_a_figure_statistic_on_a_mark_without_its_encoding_is_refused_as_layer_refuses_it():
    """A pending histogram with no ``.encode(x=...)`` refuses at the call, with ``.layer(...)``'s text."""
    step = fm.Bin(field="v", name="figure_bins")
    with pytest.raises(ValueError) as layered:
        fm.Chart(df).mark_histogram().layer(fm.Layer(mark="rule"))
    with pytest.raises(ValueError) as refused:
        fm.Chart(df).mark_histogram()._with_statistics(step)
    assert str(refused.value) == str(layered.value)


@pytest.mark.parametrize("reading", sorted(_FIGURE_STATISTIC_ORDER))
def test_a_figure_statistic_on_a_composition_joins_the_left_operands_branch(reading):
    """The step joins the left branch behind its marker, before the histogram's bins under
    ``"rows"`` and after them under ``"tail"``, as on the flat chart; the right branch has none.
    """
    step = fm.Bin(field="v", name="figure_bins")
    composed = _quiet(lambda: (_left_histogram() + P(df2))._with_statistics(step, reading=reading))
    flat = _quiet(
        lambda: _left_histogram()._resolve_pending()._with_statistics(step, reading=reading)
    )
    left, right = _wire(composed)["transforms"]
    chain = left["transforms"]
    marker = chain.index(_STATISTICS_MARKER)
    bins = [t.get("name") for t in chain[marker + 1 :] if t["type"] == "bin"]
    assert bins == _FIGURE_STATISTIC_ORDER[reading]
    assert chain == _wire(flat)["transforms"]
    assert all(t["type"] != "bin" for t in right["transforms"])


# A selection a composition filters by binds once per panel, however many
# operand branches run its filter.

_BRUSH = fm.selection_interval(name="brush", encodings=["x"])
_OTHER_BRUSH = fm.selection_interval(name="other", encodings=["y"])


def _by(selection: Any) -> Any:
    return fm.transform_filter(selection)


#: Each door, and the (param, panel) filter bindings its interactive scene carries.
_FILTER_BINDING_DOORS: dict[str, tuple[Callable[[], Any], list[tuple[str, int]]]] = {
    "composed-into-two-operands": (
        lambda: (
            (P().transform(fm.transform_filter("datum.v > 0")) + P(y="r"))
            .add_selection(_BRUSH)
            .transform(_by(_BRUSH))
        ),
        [("brush", 0)],
    ),
    "composed-nested": (
        lambda: (
            ((P().transform(fm.transform_filter("datum.v > 0")) + P(y="r")) + P(y="z"))
            .add_selection(_BRUSH)
            .transform(_by(_BRUSH))
        ),
        [("brush", 0)],
    ),
    "each-operand-by-the-same-brush": (
        lambda: (
            P().transform(_by(_BRUSH)).add_selection(_BRUSH)
            + P(y="r").transform(fm.transform_filter("datum.v > 0"), _by(_BRUSH))
        ),
        [("brush", 0)],
    ),
    "composed-faceted": (
        lambda: (
            (P().transform(fm.transform_filter("datum.v > 0")) + P(y="r"))
            .add_selection(_BRUSH)
            .transform(_by(_BRUSH))
            .facet("c")
        ),
        [("brush", 0), ("brush", 1)],
    ),
    "two-brushes": (
        lambda: (
            P().transform(_by(_BRUSH)).add_selection(_BRUSH)
            + P(y="r").transform(_by(_OTHER_BRUSH)).add_selection(_OTHER_BRUSH)
        ),
        [("brush", 0), ("other", 0)],
    ),
}


@pytest.mark.parametrize("door", sorted(_FILTER_BINDING_DOORS))
def test_each_param_binds_one_filter_per_panel(door):
    """One filter binding per param per panel, in first-appearance order.

    Rejects one binding per operand branch running the filter, which bound
    a brush copied into two operands (or written on each) twice per panel.
    """
    build, expected = _FILTER_BINDING_DOORS[door]
    scene = _quiet(lambda: _scene(build()))
    bindings = scene["interaction"].get("param_bindings") or []
    filters = [(b["param"], b["panel"]) for b in bindings if b["role"] == "filter"]
    assert filters == expected


# ---------------------------------------------------------------------------
# A composite statistic groups by the grouping channels its layer draws or
# inherits from its own parent
# ---------------------------------------------------------------------------


def _smooth(data: pl.DataFrame = df, *, ci: float | None = None, **encoding: Any) -> fm.Chart:
    """A linear fit, so a four-row group fits (a loess needs more rows than a group holds)."""
    chart = fm.Chart(data).mark_smooth(method="lm", ci=ci)
    return chart.encode(**{"x": "v", "y": "y", **encoding})


#: Each composite mark over ``df``, by the statistic step its desugar builds.
_COMPOSITES: dict[str, tuple[Callable[..., fm.Chart], str]] = {
    "histogram": (lambda **e: mk("histogram", x="v", **e), "bin"),
    "density": (lambda **e: mk("density", x="v", **e), "kde"),
    "boxplot": (lambda **e: mk("boxplot", x="g", y="y", **e), "box_stats"),
    "violin": (lambda **e: mk("violin", x="g", y="y", **e), "violin"),
    "smooth": (lambda **e: _smooth(**e), "smooth"),
    "smooth-ci": (lambda **e: _smooth(ci=0.95, **e), "smooth"),
    "boxen": (lambda **e: mk("boxen", x="g", y="z", **e), "letter_value"),
    "qq": (lambda **e: mk("qq", x="z", **e), "qq"),
    "contour": (lambda **e: mk("contour", x="v", y="y", **e), "kde2_d"),
}


def _statistic_groupby(chart: Any, step_type: str) -> list[list[str] | None]:
    """The ``groupby`` of every *step_type* step on *chart*'s wire, in written order."""
    return [s.get("groupby") for s in _steps_at_any_depth(chart) if s.get("type") == step_type]


def _dropped_warnings(build: Callable[[], Any]) -> tuple[Any, list[str]]:
    """*build*'s static render and every warning that a layer dropped a binding its rows lack."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
    return svg, [str(w.message) for w in caught if "was dropped" in str(w.message)]


# Per composite and grouping spelling: the groupby each statistic step carries.
_GROUPED: dict[str, tuple[str, str, list[str]]] = {
    **{
        f"{kind}/{channel}": (kind, channel, expected)
        for kind, expected in (
            ("histogram", ["c"]),
            ("density", ["c"]),
            ("smooth", ["c"]),
            ("smooth-ci", ["c"]),
            ("qq", ["c"]),
            ("boxen", ["c"]),
            ("boxplot", ["g", "c"]),
            ("violin", ["g", "c"]),
        )
        for channel in ("color", "fill", "stroke", "detail", "shape", "stroke_dash")
    },
    "contour/color": ("contour", "color", ["c"]),
    "contour/shape": ("contour", "shape", ["c"]),
}


@pytest.mark.parametrize("case", sorted(_GROUPED))
def test_a_composite_statistic_groups_by_each_grouping_channel(case):
    """Color under each spelling, detail, shape and stroke_dash each split the statistic.

    Every step the composite builds groups by the field, after the box and
    violin's category, so no sub-layer drops a binding its rows lack.
    Rejects reading only ``color`` (``mark_smooth(color=)`` was refused and
    ``mark_boxplot(shape=)`` drew one box per category, its outliers
    shaped by rows the box never split), and rejects an alias spelling
    grouping nothing (``fill=`` on a histogram was refused).
    """
    kind, channel, expected = _GROUPED[case]
    build, step = _COMPOSITES[kind]
    chart = build(**{channel: "c"})
    groupbys = _statistic_groupby(chart, step)
    assert groupbys and all(groupby == expected for groupby in groupbys)
    _svg, dropped = _dropped_warnings(lambda: chart)
    assert dropped == []


def _hand_built_fit(**encoding: Any) -> fm.Chart:
    """The linear fit built by hand: ``Smooth(groupby=["c"])`` drawn as a line."""
    fit = fm.Smooth("v", "y", method="lm", ci=None, groupby=["c"])
    return fm.Chart(df).transform(fit).mark_line().encode(x="x", y="y", **encoding)


def _hand_built_counts(**encoding: Any) -> fm.Chart:
    """The histogram built by hand: ``Bin(groupby=["c"])`` drawn as bars."""
    counts = fm.Bin("v", groupby=["c"])
    chart = fm.Chart(df).transform(counts).mark_bar()
    return chart.encode(x="bin_start", x2="bin_end", y="count", **encoding)


_HAND_BUILT: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "smooth/color": (lambda: _smooth(color="c"), lambda: _hand_built_fit(color="c")),
    "smooth/detail": (lambda: _smooth(detail="c"), lambda: _hand_built_fit(detail="c")),
    "smooth/stroke_dash": (
        lambda: _smooth(stroke_dash="c"),
        lambda: _hand_built_fit(stroke_dash="c"),
    ),
    "histogram/color": (
        lambda: mk("histogram", x="v", color="c"),
        lambda: _hand_built_counts(color="c"),
    ),
    "histogram/shape": (
        lambda: mk("histogram", x="v", shape="c"),
        lambda: _hand_built_counts(shape="c"),
    ),
}


@pytest.mark.parametrize("case", sorted(_HAND_BUILT))
def test_a_grouped_composite_draws_what_the_hand_built_groupby_transform_draws(case):
    """The composite equals the same transform written with ``groupby=``, mark for mark.

    Rejects one fit through the pooled rows (one polyline where the
    hand-built groupby draws two).
    """
    composite, hand_built = _HAND_BUILT[case]
    assert census(_quiet(composite().to_svg)) == census(_quiet(hand_built().to_svg))
    assert interactive_census(_quiet(composite)) == interactive_census(_quiet(hand_built))


# Per door: a grouped composite beside a rug, and the parts it draws alone.
_GROUPED_COMPOSITE_DOORS: dict[str, Case] = {
    "grouped_composite/F-smooth": Case(
        lambda: _smooth(color="c"), marks={"polyline": 2}, legends={"c": ("p", "q")}
    ),
    "grouped_composite/F-smooth-ci": Case(
        lambda: _smooth(ci=0.95, color="c"),
        marks={"path": 2, "polyline": 2},
        legends={"c": ("p", "q")},
    ),
    "grouped_composite/F-histogram": Case(
        lambda: mk("histogram", x="v", color="g"), marks={"rect": 6}, legends={"g": ("a", "b")}
    ),
    "grouped_composite/F-density": Case(
        lambda: mk("density", x="v", color="g"), marks={"path": 2}, legends={"g": ("a", "b")}
    ),
    "grouped_composite/F-boxplot": Case(
        lambda: mk("boxplot", x="g", y="y", color="c"),
        marks={"line": 16, "rect": 4},
        legends={"c": ("p", "q")},
    ),
    "grouped_composite/L-smooth-tick": Case(
        lambda: _smooth(color="c").layer(layer("tick")),
        (lambda: _smooth(color="c"), lambda: mk("tick", x="v", y="y", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "grouped_composite/L-smooth-ci-tick": Case(
        lambda: _smooth(ci=0.95, color="c").layer(layer("tick")),
        (lambda: _smooth(ci=0.95, color="c"), lambda: mk("tick", x="v", y="y", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "grouped_composite/SY-smooth": Case(
        lambda: _smooth(color="c") + SY("r"),
        marks={"polyline": 4},
        chart_level=None,
    ),
    "grouped_composite/Fc-smooth": Case(
        lambda: _smooth(color="c").facet("g"),
        marks={"polyline": 4},
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    **_cases(
        "grouped_composite/smooth-rug",
        plus_doors(lambda: _smooth(color="c"), lambda: mk("tick", x="v")),
    ),
    **_cases(
        "grouped_composite/boxplot-shape-point",
        {
            door: case
            for door, case in plus_doors(
                lambda: mk("boxplot", x="g", y="y", shape="c"), lambda: P(df2, x="g")
            ).items()
            # A multi-layer composite is no independent-y member.
            if door != "LCi"
        },
    ),
}


@pytest.mark.parametrize("case", sorted(_GROUPED_COMPOSITE_DOORS))
def test_a_grouped_composite_keeps_its_groups_on_every_door(case):
    """A composite's grouping holds beside a sibling, attached or composed, static and interactive.

    Rejects ``S(color=c).layer(tick)`` fitting one ungrouped curve (it was
    refused), and a sibling's door changing the composite's groups.
    """
    check_rendering(_GROUPED_COMPOSITE_DOORS[case])


def _panel_tag_counts(panel: Counter) -> dict[str, int]:
    """One panel's census as tag counts."""
    return dict(Counter(signature[0] for signature in panel.elements()))


def test_a_faceted_composite_groups_by_its_grouping_then_its_facet_fields():
    """The groupby is the grouping fields, then the facet's, within each panel."""
    chart = mk("density", x="v", color="c").facet("g")
    assert _statistic_groupby(chart, "kde") == [["c", "g"]]
    assert [_panel_tag_counts(panel) for panel in panel_census(_quiet(chart.to_svg))] == [
        {"path": 2},
        {"path": 2},
    ]


def test_a_smooth_under_a_binned_color_fits_one_curve_per_bin():
    """Each bin of ``Color("v", bin=...)`` gets its own fit, and the legend lists the bins.

    Rejects grouping by the raw ``v`` (eight one-row groups, no fit) and
    refusing the fit's color (the fit's rows carried no ``v``).
    """
    binned = fm.Color("v", bin={"bin_width": 4.0, "extent": (0.0, 8.0), "nice": False})
    svg, dropped = _dropped_warnings(lambda: _smooth(color=binned))
    assert dropped == []
    assert tag_counts(svg) == {"polyline": 2}
    assert legend_set(svg) == {"v": ("0", "4", "8")}


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_the_explicit_groupby_wins_over_the_charts_grouping(output):
    """A caller's own ``groupby=`` is the statistic's grouping.

    Its counts then lack the ``c`` the bars draw, so the chart is refused
    rather than drawn uncolored.
    """
    chart = fm.Chart(df).mark_histogram(groupby=["g"]).encode(x="v", color="c")
    assert _statistic_groupby(chart, "bin") == [["g"]]
    with pytest.raises(ValueError, match="unknown column 'c'"):
        _quiet(chart.to_svg if output == "static" else lambda: chart.interactive().scene_json)


#: A frame whose ``c`` has one one-row group and a null key.
_SPARSE = df.with_columns(cs=pl.Series(["p", "p", "p", "p", "p", "p", "q", None]))


def test_a_group_too_small_for_its_fit_follows_the_transforms_per_group_contract():
    """A one-row group cannot be fit: Smooth degrades that group and warns, as with groupby=."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grouped = _smooth(_SPARSE, color="cs").to_svg()
    with warnings.catch_warnings(record=True) as caught_by_hand:
        warnings.simplefilter("always")
        by_hand = (
            fm.Chart(_SPARSE)
            .mark_smooth(method="lm", groupby="cs")
            .encode(x="v", y="y", color="cs")
        ).to_svg()
    assert grouped == by_hand
    assert sorted(str(w.message) for w in caught) == sorted(str(w.message) for w in caught_by_hand)
    assert any("could not fit group" in str(w.message) for w in caught)


#: Per composite: the chart grouped through its encoding, and the same mark with ``groupby=``.
_NULL_KEY_TWINS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    "histogram": (
        lambda: mk("histogram", _SPARSE, x="v", color="cs"),
        lambda: fm.Chart(_SPARSE).mark_histogram(groupby=["cs"]).encode(x="v", color="cs"),
    ),
    "density": (
        lambda: mk("density", _SPARSE, x="v", color="cs"),
        lambda: fm.Chart(_SPARSE).mark_density(groupby=["cs"]).encode(x="v", color="cs"),
    ),
    "smooth": (
        lambda: _smooth(_SPARSE, color="cs"),
        lambda: (
            fm.Chart(_SPARSE)
            .mark_smooth(method="lm", groupby=["cs"])
            .encode(x="v", y="y", color="cs")
        ),
    ),
}


@pytest.mark.parametrize("kind", sorted(_NULL_KEY_TWINS))
def test_a_null_group_key_follows_the_transforms_null_key_policy(kind):
    """A null ``cs`` row groups exactly as the transform's own ``groupby=`` groups it."""
    grouped, by_hand = _NULL_KEY_TWINS[kind]
    assert _quiet(lambda: grouped().to_svg()) == _quiet(lambda: by_hand().to_svg())


# Each composite whose statistic now carries the hue, with its legend: the
# letter values, quantiles, isolines and residuals group by it, so every
# layer draws it and no layer drops it.
_RESTORED_HUE: dict[str, tuple[Callable[[], Any], dict[str, tuple[str, ...]]]] = {
    "mark_boxen-color": (
        lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z", color="c"),
        {"c": ("p", "q")},
    ),
    "mark_boxen-color-is-x": (
        lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z", color="g"),
        {"g": ("a", "b")},
    ),
    "catplot-boxen-hue": (
        lambda: fm.catplot(_CAT_DATA, x="cat", y="val", kind="boxen", hue="grp"),
        {"grp": ("X", "Y")},
    ),
    "mark_qq-color": (
        lambda: fm.Chart(df).mark_qq().encode(x="z", color="c"),
        {"c": ("p", "q")},
    ),
    "mark_contour-color": (
        lambda: fm.Chart(df).mark_contour().encode(x="v", y="y", color="g"),
        {"g": ("a", "b")},
    ),
    "residplot-encode-color": (
        lambda: fm.residplot(df, x="v", y="y", encode={"color": "c"}),
        {"c": ("p", "q")},
    ),
}


@pytest.mark.parametrize("case", sorted(_RESTORED_HUE))
def test_a_composite_statistic_carries_its_hue_so_every_layer_draws_it(case):
    """The hue groups the statistic, so its rows carry it: a legend, and no layer drops it.

    Rejects computing the statistic over the pooled rows, which dropped the
    hue from every layer with a warning and drew no legend.
    """
    build, legends = _RESTORED_HUE[case]
    svg, dropped = _dropped_warnings(build)
    assert dropped == []
    assert legend_set(svg) == legends


def test_qq_draws_one_reference_line_per_group():
    """Each group's quantiles get their own reference line, fit to that group's rows."""
    svg = _quiet(lambda: fm.Chart(df).mark_qq().encode(x="z", color="c").to_svg())
    assert tag_counts(svg) == {"circle": 8, "line": 2}


def test_a_contour_grouped_through_color_draws_the_isolines_groupby_draws():
    """``encode(color=g)`` splits the surfaces as ``groupby="g"`` does."""
    by_color = _quiet(lambda: fm.Chart(df).mark_contour().encode(x="v", y="y", color="g").to_svg())
    by_groupby = _quiet(
        lambda: fm.Chart(df).mark_contour(groupby="g").encode(x="v", y="y", color="g").to_svg()
    )
    assert census(by_color) == census(by_groupby)


def test_a_boxen_hue_draws_bands_per_category_and_hue():
    """Each (category, hue) group draws its own median tick and bands."""
    hued = _quiet(lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z", color="c").to_svg())
    plain = _quiet(lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z").to_svg())
    assert tag_counts(hued)["line"] == 2 * tag_counts(plain)["line"]


def test_residplot_refits_each_encoded_group_and_drops_the_single_corner_metrics():
    """An encoded hue refits each group's residuals, so the one-fit corner metrics are omitted.

    Rejects refusing the hue (the residuals carried no ``c``) and keeping
    the pooled fit's R² beside per-group residuals.
    """
    chart = fm.residplot(df, x="v", y="y", encode={"color": "c"})
    assert _statistic_groupby(chart, "smooth") == [["c"]]
    svg = _quiet(chart.to_svg)
    assert not any("R²" in text for text in chrome_text(svg))
    keyword = fm.residplot(df, x="v", y="y", shape="c")
    assert _statistic_groupby(keyword, "smooth") == [["c"]]


#: Every layered residplot door that reaches the point layer with
#: ``encode=``'s color bound: the zero rule drawn or not, the lowess
#: overlay, ``show_metrics=False``, and the post-hoc rebuild. The flat
#: path (no overlay requested at all) is excluded on purpose: nothing
#: makes it layered, so its own encoding -- not a layer's -- already
#: carries the binding, as the oracle cell below pins.
_RESID_POINT_TOOLTIP_DOORS: dict[str, Callable[[], fm.Chart]] = {
    "zero-line-off": lambda: _resid(zero_line=False, encode={"color": "c"}),
    "zero-line-on": lambda: _resid(encode={"color": "c"}),
    "lowess": lambda: _resid(lowess=True, encode={"color": "c"}),
    "show-metrics-false": lambda: _resid(show_metrics=False, encode={"color": "c"}),
    "post-hoc": lambda: _resid(zero_line=False).encode(color="c"),
}

#: Each door's static SVG bytes: the point layer's own ``encode=`` binding
#: reaches its interactive tooltip and leaves the static SVG at these bytes.
_RESID_POINT_TOOLTIP_DOOR_DIGESTS = {
    "zero-line-off": "c281b5e43f488f1c179086c2c1a2f66b36dbee1ee5f7f84ac8c63d08d4c5a250",
    "zero-line-on": "fe6951afdd0e59f9bb3c77da1802e865421d34eb2a83aca4c5b1a014f5779cdb",
    "lowess": "51b7a6bb6e72d60bc541d1f7128bac7ccc51de062ac50e36609eec0d806a822c",
    "show-metrics-false": "fe6951afdd0e59f9bb3c77da1802e865421d34eb2a83aca4c5b1a014f5779cdb",
    "post-hoc": "c281b5e43f488f1c179086c2c1a2f66b36dbee1ee5f7f84ac8c63d08d4c5a250",
}


@pytest.mark.parametrize("door", sorted(_RESID_POINT_TOOLTIP_DOORS))
def test_a_layered_residplots_points_keep_their_encoded_fields_in_the_tooltip(door):
    """The point layer binds every ``encode=`` binding itself, so its tooltip keeps it.

    Rejects reading the chart-level ``encode=`` as the layer's inherited
    parent: the per-layer tooltip rule lists a layer's own fields only
    when it has any, and the point layer already has its own (x, y
    residual), so an inherited ``color`` was shadowed and dropped. The
    static SVG renders at each door's pinned bytes; the binding reaches
    only the interactive tooltip.
    """
    import hashlib

    build = _RESID_POINT_TOOLTIP_DOORS[door]
    tooltips = interactive_tooltip_fields(_quiet(build))
    points = next(names for names in tooltips if "y residual" in names)
    assert "c" in points
    svg = _quiet(lambda: build().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _RESID_POINT_TOOLTIP_DOOR_DIGESTS[door]


def test_a_flat_residplots_own_encoding_already_carries_its_color():
    """Nothing requests an overlay, so the chart stays flat and ``encode=``
    lands on the chart's own encoding, which every consumer already reads.
    """
    build = lambda: fm.residplot(  # noqa: E731
        df, x="v", y="y", zero_line=False, show_metrics=False, encode={"color": "c"}
    )
    chart = build()
    assert chart._layers is None or len(chart._layers) == 0
    (points,) = interactive_tooltip_fields(_quiet(build))
    assert "c" in points


_AGGREGATED_COMPOSITE_DIGESTS = {
    "boxplot": "32a65d6338a647f449f41c04188848dad9fa033cf297a4a588f8b327708f276d",
    "histogram": "7fc8b97c5018d3cbf9b16af629f0f8c109eee84c4b8485e430e2b3fc92413413",
}


@pytest.mark.parametrize("kind", sorted(_AGGREGATED_COMPOSITE_DIGESTS))
def test_an_aggregate_on_a_composites_color_renders_as_it_did(kind):
    """``Color("z", aggregate="mean")`` on a histogram or box renders the bytes it rendered."""
    import hashlib

    build, _step = _COMPOSITES[kind]
    svg = _quiet(lambda: build(color=fm.Color("z", aggregate="mean")).to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _AGGREGATED_COMPOSITE_DIGESTS[kind]


def test_an_aggregate_on_a_fits_color_is_refused_as_it_was():
    """An aggregated color joins no grouping, so the fit's rows lack ``z`` and the mean refuses."""
    with pytest.raises(ValueError) as refused:
        _quiet(lambda: _smooth(color=fm.Color("z", aggregate="mean")).to_svg())
    assert (
        str(refused.value) == "transform failed: ValueError: stat_aggregate: column 'z' not found"
    )


# Composites the change does not reach: no grouping channel, or a grouping
# field the composite already groups by. Each renders the bytes it rendered.
_COMPOSITE_CONTROLS: dict[str, Callable[[], Any]] = {
    **{f"{kind}/F": build for kind, (build, _step) in _COMPOSITES.items()},
    **{
        f"{kind}/L": (lambda build=build: build().layer(layer("tick")))
        for kind, (build, _step) in _COMPOSITES.items()
    },
    **{
        f"{kind}/+rug": (lambda build=build: build() + mk("tick", x="v"))
        for kind, (build, _step) in _COMPOSITES.items()
    },
    **{
        f"{kind}/LCs": (lambda build=build: fm.LayerChart(build(), mk("tick", df2, x="v")))
        for kind, (build, _step) in _COMPOSITES.items()
    },
    **{
        f"{kind}/Fc": (lambda build=build: build().facet("c"))
        for kind, (build, _step) in _COMPOSITES.items()
    },
    "boxplot/color-is-x": lambda: mk("boxplot", x="g", y="y", color="g"),
    "violin/color-is-x": lambda: mk("violin", x="g", y="y", color="g"),
    "histogram/color": lambda: mk("histogram", x="v", color="g"),
    "density/color": lambda: mk("density", x="v", color="g"),
    "boxplot/color": lambda: mk("boxplot", x="g", y="y", color="c"),
    "violin/color": lambda: mk("violin", x="g", y="y", color="c"),
    "catplot-box-hue": lambda: fm.catplot(_CAT_DATA, x="cat", y="val", kind="box", hue="grp"),
    "displot-kde-hue": lambda: fm.displot(_CAT_DATA, x="val", hue="grp", kind="kde"),
    "contour/groupby": lambda: fm.Chart(df).mark_contour(groupby="g").encode(x="v", y="y"),
}

_COMPOSITE_CONTROL_DIGESTS: dict[str, str] = {
    "boxen/+rug": "2fb0b3a0a22181932eff7caa76c3779ae7be122062fd9f487455d26c7ffa5d13",
    "boxen/F": "07f1b8d46631dea387c8f79f4956694dc0fafe773c1b08f5561f4b717593c83c",
    "boxen/Fc": "e4f02dd48c6214c2b84e738e47bf24fa64e14e9bc34b22865f177207ae7eed32",
    "boxen/L": "1459de388c91d3d097e8a751692993e712168009c6698d1e14787595cd055994",
    "boxen/LCs": "5f6d9c2d2ee91023d1dce04492d21e0db2f911bf5b397434fc944be8c102767b",
    "boxplot/+rug": "78a08ff86bb5f4670aaa6e5768cede923acfa41b717d19086881716184d1dddd",
    "boxplot/F": "7312af034cc9add736a4483e31d96e1a158faa5d874630356241a11d2231306a",
    "boxplot/Fc": "f30bb77e41b89b86f03022728d21cea222d97a0c169e3b9341ca90595c07d94c",
    "boxplot/L": "e3e38c4096a9a14b8935a121958ec9521eb157f8cac2506ffd84160eb7334aa2",
    "boxplot/LCs": "f9b28b75c741ce0ab4e636ea100dbf496958426b3ccf974e8f154eb4fd44d698",
    "boxplot/color": "24a085e7c34374a48ffbba6402c52a237dcb8c579bc6e395b037ea4fd84b824b",
    "boxplot/color-is-x": "5ab35cff26faaa3860b2041053378ad81517df94327fef5e9d218d02c8bd91fd",
    "catplot-box-hue": "22aa5ebe2a0c4bf976e41f9b5ee8152bc449fff205a88348605c6396172c26be",
    "contour/+rug": "25a0b78861dc918b042be8fabfcaca155923d52375d216437b9d4cf60adf2cc2",
    "contour/F": "abf76917ca5eccf1d7f1995b92485a75f466f263a8053d03c999f212fdf20e46",
    "contour/Fc": "b10ad83eea90ba6639dab125268e93d2af528976e5585ffb7eaee86f8a49cc0d",
    "contour/L": "25a0b78861dc918b042be8fabfcaca155923d52375d216437b9d4cf60adf2cc2",
    "contour/LCs": "cec1b4594b81260baa85ae417cb33fa01d203f005b17e1e0a12d86dbabe5ed3f",
    "contour/groupby": "d3ae503b8103f09448afe3cf190db052407db636f11db1bc55072f8e18de71dd",
    "density/+rug": "b5184eaea8968799a3045653673ce7fea885ce0cca5bbe2351e1427be7ec7a9d",
    "density/F": "f163b168ec0074ad577648fc389f74af78c9f0cbf7042efa12cd17a2c074e3f0",
    "density/Fc": "1513a44ff3f57caa13babd5551882582bcb5163ef5c22e00b63db5f6669e6729",
    "density/L": "b5184eaea8968799a3045653673ce7fea885ce0cca5bbe2351e1427be7ec7a9d",
    "density/LCs": "fff6862672557ccb68a0dcc6abeb86b4a907bc0fe61ffb18aecb99ca9e5286a7",
    "density/color": "329a478d8290e1b7eb527be5385e8482dbf5f37deeba2be4b69710df6b73d484",
    "displot-kde-hue": "c1c50349dacf306bb635f01daeb024e8f57cf0319b81e74898330f51c9fac621",
    "histogram/+rug": "e6e5f94814f808e639d9966314ecb7e4f3e7d4f1f9818e510e35c3f0e932fc5d",
    "histogram/F": "b4913a77287da34ca2337f02819e0440b9eaa47957020fd20585a48a5a1250fb",
    "histogram/Fc": "40d4b34dc2cddc98a409ce0384ad1ae26291acdc0649a479d2c2422bf9c629fb",
    "histogram/L": "e6e5f94814f808e639d9966314ecb7e4f3e7d4f1f9818e510e35c3f0e932fc5d",
    "histogram/LCs": "f1bc1e8b83e132619ffa45c01c57d238f996b3cd508b69f0ef4337d2ff97a19b",
    "histogram/color": "2685ec30c7e134cd4746044195c7e374deceec8edbb48e23ed78aa02e7be17fa",
    "qq/+rug": "1ae5e25be02988956bf2068156568085bb09f4cae604461df1de3851cbb6ad2f",
    "qq/F": "07689d9d5965f22dbf35554e8b4bfdcd6fbafeccdef82ff21c8d496ada94f5a8",
    "qq/Fc": "cf1bccec74c736f5eb8c1878a50525864c0fd46f63ac4520806752fec4daf274",
    "qq/L": "72ffa2b60c3bc8673fa41de49f4485e27f885e067d204de36d986df06bc3edc0",
    "qq/LCs": "8895eac136b78004ebdbed47ef2a2e9f5b874d88a9f896d52061f57701639fe7",
    "smooth-ci/+rug": "c64ef7f8f4e6c9529b6ea56e0b9b54074d142f21bd4b70dafa98a46d57aa6666",
    "smooth-ci/F": "f148133044e97653d3478ce8c0ffba7bda12b78e874e4508eb7d98bd5628dc70",
    "smooth-ci/Fc": "12fab76ef2ae00efd922f69e15c123d68cb88bb69bdd5cd8bd368336e718504d",
    "smooth-ci/L": "c64ef7f8f4e6c9529b6ea56e0b9b54074d142f21bd4b70dafa98a46d57aa6666",
    "smooth-ci/LCs": "eba64ed71edabaf65990107e7f1a68d4bd8008079d56aeb93850711819680723",
    "smooth/+rug": "b89b8d826f9da69488e7b673c04edea8fda12c06a40e74800d681ef6fe9ab2d5",
    "smooth/F": "a4806ad17d72e760e348eca5b8fa5e06f2398a3a29a460b3a7a0ce82009876f8",
    "smooth/Fc": "cb31ffa5de55181779de48fdc03fb87f98d36c1b4259177d0f5db41c0e07b1b3",
    "smooth/L": "771b46feb4551e72fd6448c06491030d1679d9f263751bde3c2819e555653020",
    "smooth/LCs": "393b49ef4b68dc1c7fc8507538a573f9d0be03332d508a08c450553eb2cd4049",
    "violin/+rug": "dccb5762ff381d06af4f819108a8e04619f1d9d4582b04647198b28e287dd3d5",
    "violin/F": "576cdaaf65f1672d641462f86ac8e286e62afd090c05ea8ab4a50a803b5a7211",
    "violin/Fc": "d14c61945ee0df934f4e7d59e59ee7364248d5b8e4d1b06715ca9c0fa4c58862",
    "violin/L": "78a9f7dd6a5d29b0cfff561abc63a856110ae85fc32a9fa31ee678cb36bf622b",
    "violin/LCs": "c35e52c9cedf460ad0dc5774ea1ce106c743f45f2f13769bc15d269f307b3185",
    "violin/color": "fc6b619486bb01ad4c2dc7cfc02f6c1ca1e12a0b38b54347b408d740601d1085",
    "violin/color-is-x": "22ac7ceabbdafc4aae3f63a95d077b31381ff16c5e0e0ff312a632494fa4f216",
}


@pytest.mark.parametrize("case", sorted(_COMPOSITE_CONTROLS))
def test_a_composite_the_grouping_does_not_reach_renders_as_it_did(case):
    """Byte-identical controls: no grouping channel, or one the composite already groups by."""
    import hashlib

    svg = _quiet(lambda: _COMPOSITE_CONTROLS[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _COMPOSITE_CONTROL_DIGESTS[case]


# ---------------------------------------------------------------------------
# Attaching a label never changes its host's statistic, and a + operand's
# detail groups every layer of the operand at the operand's grain
# ---------------------------------------------------------------------------


def _stacked_mean_bars(**encoding: Any) -> fm.Chart:
    return _binned_mark("bar", fm.Stack(), color="c", **encoding)


def _labelled(**encoding: Any) -> fm.Chart:
    return _stacked_mean_bars(**encoding).layer(layer("text", text=_mean_text()))


#: Every door that attaches a mean label to a stacked mean bar.
_LABELLED_DOORS: dict[str, Callable[..., Any]] = {
    "L": lambda **e: _labelled(**e),
    "LD": lambda **e: _stacked_mean_bars(**e).layer(
        fm.Layer(
            data=_GROUP_ROWS, mark="text", encoding={"x": "g", "y": "mean(y)", "text": _mean_text()}
        )
    ),
    "+L": lambda **e: _labelled(**e) + _group_rule(),
    "+e": lambda **e: (
        _binned_mark("bar", fm.Stack(), **e) + _binned_mark("text", fm.Stack(), text=_mean_text())
    ).encode(color="c"),
    "LCs": lambda **e: fm.LayerChart(_labelled(**e), _group_rule()),
    "LCm": lambda **e: fm.LayerChart(_labelled(**e), _group_rule())._build_merged(),
}


def _bar_segments(chart: Any, kind: str) -> int:
    from tests._svg_rects import data_bar_rects, mark_layer

    if kind == "static":
        return len(data_bar_rects(mark_layer(_quiet(chart.to_svg))))
    marks = _quiet(lambda: _scene(chart))["panels"][0]["marks"]
    return sum(len(batch["nodes"]) for batch in marks if batch["kind"] == "bar")


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_LABELLED_DOORS))
@pytest.mark.parametrize("detail", [False, True], ids=["color", "color+detail"])
def test_attaching_a_label_never_changes_its_hosts_statistic(door, kind, detail):
    """The stacked mean bars draw one segment per group, alone and with a label attached.

    Four ``(g, c)`` segments, eight under a ``detail="f"``, on every door
    and both outputs. Rejects the label's statistic taking the bars' rows
    (three segments), and a ``+`` operand's detail grouping only the marks
    that draw detail (the bars grouped by ``(g, c)`` beside the flat
    chart's ``(g, c, f)``).
    """
    encoding = {"detail": "f"} if detail else {}
    alone = _bar_segments(_quiet(lambda: _stacked_mean_bars(**encoding)), kind)
    assert alone == _GROUP_ROWS.select(["g", "c", "f"] if detail else ["g", "c"]).n_unique()
    chart = _quiet(lambda: _LABELLED_DOORS[door](**encoding))
    if kind == "interactive" and isinstance(chart, fm.LayerChart):
        chart = chart._build_merged()
    assert _bar_segments(chart, kind) == alone


# ---------------------------------------------------------------------------
# A grouped boxen or qq under a facet: the null-key drop is counted over the frame
# ---------------------------------------------------------------------------

#: Panel ``F2``'s every ``h`` is null, so a statistic grouped by ``h`` keeps none of its rows.
_NULL_PANEL = pl.DataFrame(
    {
        "cat": ["a", "b"] * 10,
        "f": ["F1"] * 10 + ["F2"] * 10,
        "h": ["X", "Y"] * 5 + [None] * 10,
        "val": [float(i % 7) for i in range(20)],
    }
)

_NULL_PANEL_CHARTS: dict[str, tuple[Callable[[], fm.Chart], str]] = {
    "boxen": (
        lambda: fm.Chart(_NULL_PANEL).mark_boxen().encode(x="cat", y="val", color="h").facet("f"),
        "LetterValue",
    ),
    "qq": (
        lambda: fm.Chart(_NULL_PANEL).mark_qq().encode(x="val", color="h").facet("f"),
        "QQ",
    ),
}


@pytest.mark.parametrize("kind", sorted(_NULL_PANEL_CHARTS))
def test_a_faceted_hue_drops_its_null_key_rows_once_counted_over_the_whole_frame(kind):
    """The hue reaches the statistic under a facet, and its null-key drop warns once for every panel."""
    build, label = _NULL_PANEL_CHARTS[kind]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build().to_svg()
    assert [str(w.message) for w in caught if "null 'h'" in str(w.message)] == [
        f"{label}: 10 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]


def _drop_warnings(build: Callable[[], Any], output: Callable[[Any], Any]) -> tuple[Any, list[str]]:
    """*output* of *build*'s chart (or the exception it raised) and every null-key drop warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = output(build())
        except ValueError as refused:
            result = refused
    return result, [str(w.message) for w in caught if "belong to no group" in str(w.message)]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("kind", sorted(_NULL_PANEL_CHARTS))
def test_a_faceted_panel_the_null_key_drop_emptied_draws_empty_beside_the_drop_warning(
    kind, output
):
    """Panel ``F2``, whose every hue is null, draws its frame with no marks, and the drop warns.

    Rejects the panel vanishing (panels built from the rows drawn after the
    statistic), which drew ``F1`` alone.
    """
    build, label = _NULL_PANEL_CHARTS[kind]
    render = {"static": lambda chart: chart.to_svg(), "interactive": _scene}[output]
    drawn, dropped = _drop_warnings(build, render)
    assert dropped == [
        f"{label}: 10 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]
    if output == "static":
        panels = panel_census(drawn)
        assert len(panels) == 2
        assert panels[0] and not panels[1], [dict(panel) for panel in panels]
        assert "F2" in [text for text in chrome_text(drawn)]
    else:
        assert [
            sum(len(batch["nodes"]) for batch in panel["marks"]) > 0 for panel in drawn["panels"]
        ] == [
            True,
            False,
        ]


@pytest.mark.parametrize("facet", [True, False], ids=["faceted", "unfaceted"])
@pytest.mark.parametrize("kind", sorted(_NULL_PANEL_CHARTS))
def test_a_null_key_drop_that_empties_every_panel_refuses_beside_the_drop_warning(kind, facet):
    """Every hue null: the chart refuses with the statistic's empty-input refusal, and the drop warns.

    Faceted and unfaceted refuse alike. Rejects the faceted chart rendering
    with no panel (a ``NoRowsToDraw`` warning), and a refusal that hides the
    drop that caused it.
    """
    build, label = _NULL_PANEL_CHARTS[kind]
    refusal = {"boxen": "stat_letter_value", "qq": "stat_qq"}[kind]
    all_null = _NULL_PANEL.with_columns(pl.lit(None, dtype=pl.Utf8).alias("h"))
    chart = {
        "boxen": lambda: fm.Chart(all_null).mark_boxen().encode(x="cat", y="val", color="h"),
        "qq": lambda: fm.Chart(all_null).mark_qq().encode(x="val", color="h"),
    }[kind]
    refused, dropped = _drop_warnings(
        lambda: chart().facet("f") if facet else chart(), lambda c: c.to_svg()
    )
    assert isinstance(refused, ValueError)
    assert str(refused) == f"transform failed: ValueError: {refusal}: empty input after filter"
    assert dropped == [
        f"{label}: 20 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]


def test_two_grouped_steps_of_one_kind_count_their_null_key_drops_apart_under_a_facet():
    """Two named, caller-written ``QQ`` steps over the SAME rows and key report apart.

    ``q1`` and ``q2`` both read the chart's rows unchanged (named steps
    never advance the chain), but neither opens a lowering: a plain
    ``mark_point()`` lowers no composite statistic, so the chain carries no
    statistics marker at all, and a report with no lowering is never
    collapsed with another's. Each of the two user-written steps reports
    its own drop, faceted as unfaceted.
    """
    nulls = _NULL_PANEL.with_columns(pl.Series("h", ["X", "Y", None, "X"] * 5))

    def chart(facet: bool) -> fm.Chart:
        steps = (fm.QQ("val", groupby=["h"], name="q1"), fm.QQ("val", groupby=["h"], name="q2"))
        built = fm.Chart(nulls).transform(*steps).mark_point().encode(x="val", y="val")
        return built.facet("f") if facet else built

    each = "QQ: 5 rows have a null 'h' and belong to no group, so they were dropped; fill or filter the nulls in 'h'"
    for facet in (True, False):
        _svg, dropped = _drop_warnings(lambda facet=facet: chart(facet), lambda c: c.to_svg())
        assert dropped == [each, each], facet


# ---------------------------------------------------------------------------
# Each lowering reports its own null-key drop apart from every other
# lowering's, on every door that stacks one beside another
# ---------------------------------------------------------------------------

#: 30 rows, 10 with a null ``h``: the fixture the lowering cases group by.
_LOWERING_ROWS = pl.DataFrame(
    {
        "g": ["a", "b"] * 15,
        "h": ["X", "Y"] * 10 + [None] * 10,
        "v": [float(i % 9) for i in range(30)],
        "y": [float((i * 3) % 11) for i in range(30)],
    }
)
#: The same rows, twice, under two facet panels: the probe frame for a faceted lowering.
_LOWERING_ROWS_FACETED = pl.concat(
    [_LOWERING_ROWS.with_columns(pl.lit(panel).alias("f")) for panel in ("F1", "F2")]
)


def _h_box(data: pl.DataFrame = _LOWERING_ROWS) -> fm.Chart:
    return fm.Chart(data).mark_boxplot().encode(x="g", y="y", color="h")


def _h_violin(data: pl.DataFrame = _LOWERING_ROWS) -> fm.Chart:
    return fm.Chart(data).mark_violin().encode(x="g", y="y", color="h")


def _h_bar_mean(data: pl.DataFrame = _LOWERING_ROWS) -> fm.Chart:
    return fm.Chart(data).mark_bar().encode(x="g", y="mean(y)", color="h")


def _h_line_mean(data: pl.DataFrame = _LOWERING_ROWS) -> fm.Chart:
    return fm.Chart(data).mark_line().encode(x="g", y="mean(y)", color="h")


def _drop_labels(build: Callable[[], Any], output: str) -> list[str]:
    """The transform name leading each null-key drop warning *build*'s chart raises, in order."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    return [
        str(w.message).split(":", 1)[0] for w in caught if "belong to no group" in str(w.message)
    ]


#: Per case: the chart, and the lowering(s) its own null-key drop reports, in order.
_LOWERING_REPORT_CASES: dict[str, tuple[Callable[[], fm.Chart], list[str]]] = {
    "boxplot": (_h_box, ["BoxStats"]),
    "default-violin": (_h_violin, ["Violin"]),
    "boxplot/facet": (lambda: _h_box(_LOWERING_ROWS_FACETED).facet("f"), ["BoxStats"]),
    "default-violin/facet": (lambda: _h_violin(_LOWERING_ROWS_FACETED).facet("f"), ["Violin"]),
    "errorband": (
        lambda: fm.Chart(_LOWERING_ROWS).mark_errorband().encode(x="g", y="y", color="h"),
        ["ErrorExtent"],
    ),
    "boxen": (
        lambda: fm.Chart(_LOWERING_ROWS).mark_boxen().encode(x="g", y="y", color="h"),
        ["LetterValue"],
    ),
    "boxplot+plain-layer": (
        lambda: _h_box().layer(fm.Layer(mark="tick", encoding={"x": "g", "y": "y"})),
        ["BoxStats"],
    ),
    "boxplot+violin": (lambda: _h_box() + _h_violin(), ["BoxStats", "Violin"]),
    "layer-chart(boxplot,violin)": (
        lambda: fm.LayerChart(_h_box(), _h_violin()),
        ["BoxStats", "Violin"],
    ),
    "histogram+density": (
        lambda: (
            fm.Chart(_LOWERING_ROWS).mark_histogram().encode(x="v", color="h")
            + fm.Chart(_LOWERING_ROWS).mark_density().encode(x="v", color="h")
        ),
        ["Bin", "Kde"],
    ),
    "smooth(lm)+smooth(loess)": (
        lambda: (
            fm.Chart(_LOWERING_ROWS).mark_smooth(method="lm").encode(x="v", y="y", color="h")
            + fm.Chart(_LOWERING_ROWS).mark_smooth(method="loess").encode(x="v", y="y", color="h")
        ),
        ["Smooth", "Smooth"],
    ),
    "layer-two-means": (
        lambda: (
            fm.Chart(_LOWERING_ROWS)
            .mark_point()
            .encode(x="g", y="v", color="h")
            .layer(
                fm.Layer(mark="point", encoding={"y": "mean(y)"}),
                fm.Layer(mark="point", encoding={"y": "mean(y)"}),
            )
        ),
        ["Aggregate", "Aggregate"],
    ),
    "boxplot+layer-mean": (
        lambda: _h_box().layer(fm.Layer(mark="point", encoding={"x": "g", "y": "mean(y)"})),
        ["BoxStats", "Aggregate"],
    ),
    "bar-mean+line-mean": (lambda: _h_bar_mean() + _h_line_mean(), ["Aggregate", "Aggregate"]),
    "layer-chart(bar-mean,line-mean)": (
        lambda: fm.LayerChart(_h_bar_mean(), _h_line_mean()),
        ["Aggregate", "Aggregate"],
    ),
    "bar-mean+secondary-y": (
        lambda: _h_bar_mean() + fm.SecondaryY("v", mark="line"),
        ["Aggregate"],
    ),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_LOWERING_REPORT_CASES))
def test_each_lowering_reports_its_own_null_key_drop_apart_from_every_others(case, output):
    """Each lowering a chart runs reports its own null-key drop, named by its own transform.

    Rejects collapsing two lowerings that read the same rows under the same
    groupby into one report (a ``+`` operand's statistic swallowing the
    other's, a ``Chart.layer`` statistic sharing the host mark's report):
    two lowerings report twice, whichever door stacks them, and a single
    lowering (a plain boxplot, a plain layer added beside it, a mark beside
    a ``SecondaryY`` layer drawing no statistic of its own) still reports
    once.
    """
    build, expected = _LOWERING_REPORT_CASES[case]
    assert _drop_labels(build, output) == expected


@pytest.mark.parametrize(
    "plus_case, layerchart_case",
    [
        ("boxplot+violin", "layer-chart(boxplot,violin)"),
        ("bar-mean+line-mean", "layer-chart(bar-mean,line-mean)"),
    ],
)
def test_a_plus_operand_and_its_layerchart_equivalent_report_the_same_lowerings(
    plus_case, layerchart_case
):
    """``a + b`` and ``LayerChart(a, b)`` report the identical lowerings, static and interactive.

    Rejects a ``+`` composition and its ``LayerChart`` equivalent
    disagreeing on how many null-key drops one chart reports, as they did
    when every operand of a shared branch reported behind one outer
    marker.
    """
    plus_build, _ = _LOWERING_REPORT_CASES[plus_case]
    layerchart_build, _ = _LOWERING_REPORT_CASES[layerchart_case]
    for output in ("static", "interactive"):
        assert _drop_labels(plus_build, output) == _drop_labels(layerchart_build, output)


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_faceted_plus_operand_still_reports_each_lowering_apart(output):
    """``(B + V).facet('f')`` reports both lowerings, summed per panel, not collapsed into one."""
    faceted = lambda: (  # noqa: E731
        _h_box(_LOWERING_ROWS_FACETED) + _h_violin(_LOWERING_ROWS_FACETED)
    ).facet("f")
    assert _drop_labels(faceted, output) == ["BoxStats", "Violin"]


# ---------------------------------------------------------------------------
# A series mark whose statistic groups by a channel it does not draw draws one
# series per group, as a detail channel does
# ---------------------------------------------------------------------------

#: Per case: the chart, and the marks it draws over ``df``'s two ``c`` groups.
_SERIES_SPLIT: dict[str, tuple[Callable[[], fm.Chart], dict[str, int]]] = {
    "density/shape": (lambda: mk("density", x="v", shape="c"), {"path": 2}),
    "density/stroke_dash": (lambda: mk("density", x="v", stroke_dash="c"), {"path": 2}),
    "density/detail": (lambda: mk("density", x="v", detail="c"), {"path": 2}),
    "smooth/shape": (lambda: _smooth(shape="c"), {"polyline": 2}),
    "smooth/detail": (lambda: _smooth(detail="c"), {"polyline": 2}),
    "smooth-ci/detail": (lambda: _smooth(ci=0.95, detail="c"), {"path": 2, "polyline": 2}),
    "smooth-ci/shape": (lambda: _smooth(ci=0.95, shape="c"), {"path": 2, "polyline": 2}),
}


@pytest.mark.parametrize("case", sorted(_SERIES_SPLIT))
def test_a_series_grouped_by_a_channel_it_does_not_draw_draws_one_series_per_group(case):
    """One area, fit line or band per group, static and interactive, with no ignored-channel warning.

    Rejects one series zigzagging through every group's rows (one path or
    polyline), and the warning that the grouping channel was ignored when
    it split the statistic and the series.
    """
    build, marks = _SERIES_SPLIT[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
        scene_marks = Counter(
            batch["kind"]
            for panel in _scene(build())["panels"]
            for batch in panel["marks"]
            for _node in batch["nodes"]
        )
    assert tag_counts(svg) == marks
    assert sum(scene_marks.values()) == sum(marks.values()), scene_marks
    assert not [str(w.message) for w in caught if "has no effect" in str(w.message)]


def _user_fit(groupby: list[str] | None = None, **encoding: Any) -> fm.Chart:
    """A linear fit the user wrote as a step, drawn as a line."""
    fit = fm.Smooth("v", "y", method="lm", ci=None, groupby=groupby)
    return fm.Chart(df).transform(fit).mark_line().encode(x="x", y="y", **encoding)


#: Per case: a line over a step the user wrote, and the fit lines it draws
#: over ``df``'s two ``c`` groups.
_USER_STEP_SPLIT: dict[str, tuple[Callable[[], fm.Chart], int]] = {
    "grouped": (lambda: _user_fit(["c"]), 2),
    "grouped/shape": (lambda: _user_fit(["c"], shape="c"), 2),
    "grouped/stroke_dash": (lambda: _user_fit(["c"], stroke_dash="c"), 2),
    "grouped/detail": (lambda: _user_fit(["c"], detail="c"), 2),
    "ungrouped": (lambda: _user_fit(), 1),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_USER_STEP_SPLIT))
def test_a_line_follows_the_groupby_of_a_step_the_user_wrote(case, output):
    """One fit line per group of the user's ``Smooth(groupby=["c"])``, with no channel asking.

    A ``shape`` or ``stroke_dash``/``detail`` on the same field splits once
    (two lines, not four); an ungrouped fit draws one. Rejects the one line
    zigzagging through every group's fitted rows.
    """
    build, lines = _USER_STEP_SPLIT[case]
    if output == "static":
        drawn = tag_counts(_quiet(lambda: build().to_svg())).get("polyline", 0)
    else:
        drawn = sum(_scene_line_counts(_quiet(lambda: _scene(build()))))
    assert drawn == lines


#: 4 rows per (g, c) cell (16 rows total), so faceting by ``g`` still leaves
#: each panel with 4 rows per ``c`` group -- enough for ``smooth-ci``'s
#: confidence band to compute on every panel, not just on the whole chart.
_OWN_GROUPBY_ROWS = pl.DataFrame(
    {
        "g": ["a"] * 8 + ["b"] * 8,
        "c": (["p"] * 4 + ["q"] * 4) * 2,
        "v": [1.0, 2.0, 3.0, 4.0] * 4,
        "y": [
            3.0,
            4.2,
            5.1,
            6.3,
            2.0,
            1.3,
            4.5,
            3.8,
            5.0,
            6.5,
            4.2,
            7.1,
            1.0,
            3.3,
            2.8,
            4.6,
        ],
    }
)


def _own_groupby_smooth(ci: float | None = None) -> fm.Chart:
    """``mark_smooth``'s own ``groupby=``, with no grouping channel bound."""
    return (
        fm.Chart(_OWN_GROUPBY_ROWS)
        .mark_smooth(method="lm", ci=ci, groupby=["c"])
        .encode(x="v", y="y")
    )


def _own_groupby_density() -> fm.Chart:
    return fm.Chart(_OWN_GROUPBY_ROWS).mark_density(groupby=["c"]).encode(x="v")


#: Per case: a composite built from its own ``groupby=`` with no grouping
#: channel bound, its static tag counts, and its interactive per-kind node
#: counts, PER PANEL (``_OWN_GROUPBY_ROWS``'s two ``c`` groups, four rows
#: each; faceting by ``g`` still leaves each panel with both ``c`` groups
#: at four rows each, so every door's per-panel counts are the case's own,
#: unchanged by faceting).
_OWN_GROUPBY_NO_CHANNEL: dict[
    str, tuple[Callable[[], fm.Chart], dict[str, int], dict[str, int]]
] = {
    "smooth": (_own_groupby_smooth, {"polyline": 2}, {"line": 2}),
    "smooth-ci": (
        lambda: _own_groupby_smooth(ci=0.95),
        {"path": 2, "polyline": 2},
        {"ribbon": 2, "line": 2},
    ),
    "density": (_own_groupby_density, {"path": 2}, {"area": 2}),
    "smooth/color-control": (
        lambda: _own_groupby_smooth().encode(color="c"),
        {"polyline": 2},
        {"line": 2},
    ),
}


def _static_panel_tag_counts(svg: str) -> list[dict[str, int]]:
    """Every panel's tag counts, reusing the one static per-panel reader (``_panel_tag_counts``)."""
    return [_panel_tag_counts(counts) for counts in panel_census(svg)]


def _interactive_panel_kind_counts(chart: Any) -> list[dict[str, int]]:
    """Per panel, the interactive mark-batch kind->node-count map.

    A batch with no nodes (an empty kind the scene still lists) is left
    out, matching ``_static_panel_tag_counts``'s static side, which has no
    equivalent "absent tag" entry to omit.
    """
    out = []
    for panel in _quiet(lambda: _scene(chart))["panels"]:
        counts: Counter = Counter()
        for batch in panel["marks"]:
            counts[batch["kind"]] += len(batch["nodes"])
        out.append({kind: n for kind, n in counts.items() if n > 0})
    return out


#: Each door, wrapping a case's build function. ``+``/``LCs``/``LCm`` layer a
#: plain ``P()`` beside the composite (8 circles, df's whole height), so
#: those three panels' expected marks gain ``{"circle": 8}``; ``LCs`` renders
#: through the static overlay tree while ``LCm`` forces the merged flat path
#: statically too (``_build_merged()``), exercising both static mechanisms
#: beside the always-merged interactive route. ``Fc`` facets by ``g`` (df's
#: other grouping column, independent of ``c``), giving two panels each
#: still holding both ``c`` groups.
_OWN_GROUPBY_DOORS: dict[str, Callable[[Callable[[], fm.Chart]], fm.Chart]] = {
    "F": lambda build: build(),
    "+": lambda build: build() + P(),
    "LCs": lambda build: fm.LayerChart(build(), P()),
    "LCm": lambda build: fm.LayerChart(build(), P())._build_merged(),
    "Fc": lambda build: build().facet("g"),
}


@pytest.mark.parametrize("door", sorted(_OWN_GROUPBY_DOORS))
@pytest.mark.parametrize("case", sorted(_OWN_GROUPBY_NO_CHANNEL))
def test_a_composite_statistics_own_groupby_splits_its_line_ribbon_or_area_with_no_channel(
    case, door
):
    """``mark_smooth``/``mark_density``'s own ``groupby=`` splits what it draws, even unbound.

    The fit or density is already computed per group, and the series split
    reads that same groupby behind the chart's own statistics marker, so
    one line or area is drawn per group rather than joining every group's
    rows into one, on every layering and facet door, static and
    interactive. ``color="c"`` splits the same way through the channel
    (the control case). Rejects reading only a lowered Aggregate's groupby
    and never a declared ``Bin``'s (a declared ``Bin`` groupby splits too,
    pinned by ``test_a_line_splits_by_the_groupby_its_bin_declares``), and
    rejects doubling the split when a channel also covers the field.
    """
    build, marks, kinds = _OWN_GROUPBY_NO_CHANNEL[case]
    chart = _OWN_GROUPBY_DOORS[door](build)
    expected_marks = {**marks, "circle": 8} if door in ("+", "LCs", "LCm") else dict(marks)
    expected_kinds = {**kinds, "point": 8} if door in ("+", "LCs", "LCm") else dict(kinds)
    svg = _quiet(chart.to_svg)

    if door == "Fc":
        # Two independent facet panels, each still holding both `c` groups
        # at four rows each (P() is not layered on this door).
        assert _static_panel_tag_counts(svg) == [expected_marks, expected_marks]
        assert _interactive_panel_kind_counts(chart) == [expected_kinds, expected_kinds]
        return

    # F/+/LCm render (or merge) to one panel; LCs's static overlay keeps the
    # composite and P()'s layer as separate clip groups over the same rect
    # (two panel entries), so the total across panels is what one chart
    # draws -- interactive always merges LCs to one panel already.
    total_static: Counter = Counter()
    for panel_counts in _static_panel_tag_counts(svg):
        total_static.update(panel_counts)
    assert dict(total_static) == expected_marks

    total_interactive: Counter = Counter()
    for panel_counts in _interactive_panel_kind_counts(chart):
        total_interactive.update(panel_counts)
    assert dict(total_interactive) == expected_kinds


_ROC_Y_PRED = np.random.default_rng(1).dirichlet([1.0, 1.0, 1.0], size=90)
_ROC_Y_TRUE = [0, 1, 2] * 30


def test_a_diagnostic_composites_lines_follow_their_grouping_channels():
    """ROC/errorband-plus-line/``lmplot`` split their lines by the channel or grouping they declare.

    These composites' own groupby is a grouping channel (class, hue) or the
    chart's planner grouping, not an unbound own-``groupby=`` like
    ``mark_smooth``/``mark_density`` above, so they are a different
    provenance path through the same series split. Probed byte-identical
    across every demo diagnostic and composite variant; this guard pins the
    counts directly so a future provenance change cannot silently join them.
    """
    roc = fm.roc_chart(y_true=_ROC_Y_TRUE, y_pred=_ROC_Y_PRED)
    roc_lines = sum(_scene_line_counts(_quiet(lambda: _scene(roc))))
    assert roc_lines == 4  # 3 classes + the chance line

    roc_averaged = fm.roc_chart(y_true=_ROC_Y_TRUE, y_pred=_ROC_Y_PRED, per_class=False)
    roc_averaged_lines = sum(_scene_line_counts(_quiet(lambda: _scene(roc_averaged))))
    assert roc_averaged_lines == 2  # averaged curve + the chance line

    # h alternates every row (not every-other-pair), so each `g` value holds
    # both h groups -- a fixture where h correlated 1:1 with g (e.g.
    # `["x", "y"] * 4`, matching g's own `["a", "b"] * 4`) leaves each x-bin
    # with only one h value and the composite draws nothing at all.
    band_rows = df.with_columns(h=pl.Series(["x", "x", "y", "y"] * 2))
    errorband_plus_line = mk("errorband", band_rows, x="g", y="y", color="h") + mk(
        "line", band_rows, x="g", y="mean(y)", color="h"
    )
    scene = _quiet(lambda: _scene(errorband_plus_line))
    got: Counter = Counter()
    for panel in scene["panels"]:
        for batch in panel["marks"]:
            got[batch["kind"]] += len(batch["nodes"])
    assert dict(got) == {"ribbon": 2, "line": 2}

    # Likewise, hue and col must vary independently so every col panel
    # holds both hue values.
    d = df.with_columns(h=pl.Series(["x", "y"] * 4), col=pl.Series(["m", "m", "n", "n"] * 2))
    lmplot_chart = fm.lmplot(d, x="v", y="y", hue="h", col="col")
    lmplot_lines = sum(_scene_line_counts(_quiet(lambda: _scene(lmplot_chart))))
    assert lmplot_lines == 4  # 2 hue values x 2 col panels, one fit line each


#: Each door, building a line over a binned x whose Bin declares its own
#: groupby.
_BIN_GROUPBY_DOORS: dict[str, Callable[[Any], fm.Chart]] = {
    "F": lambda bin_: fm.Chart(df).mark_line().encode(x=fm.X("v", bin=bin_), y="count()"),
    "+": lambda bin_: fm.Chart(df).mark_line().encode(x=fm.X("v", bin=bin_), y="count()") + P(),
    "L": lambda bin_: (
        fm.Chart(df)
        .mark_line()
        .encode(x=fm.X("v", bin=bin_), y="count()")
        .layer(fm.Layer(mark="point", encoding={"x": "v", "y": "y"}))
    ),
    "LCs": lambda bin_: fm.LayerChart(
        fm.Chart(df).mark_line().encode(x=fm.X("v", bin=bin_), y="count()"), P()
    ),
    "LCm": lambda bin_: fm.LayerChart(
        fm.Chart(df).mark_line().encode(x=fm.X("v", bin=bin_), y="count()"), P()
    )._build_merged(),
    "Fc": lambda bin_: (
        fm.Chart(df).mark_line().encode(x=fm.X("v", bin=bin_), y="count()").facet("g")
    ),
}


@pytest.mark.parametrize("spelling", ["object", "dict"])
@pytest.mark.parametrize("door", sorted(_BIN_GROUPBY_DOORS))
def test_a_line_splits_by_the_groupby_its_bin_declares(door, spelling):
    """A line over ``X(bin=Bin(groupby=[...]))`` splits by the Bin's own declared groupby.

    Covers both spellings (``bin=fm.Bin('v', groupby=['c'])`` and
    ``bin={'groupby': ['c']}``), on every layering door. The lowered Bin's
    wire carries ``written_groupby`` naming the caller's own field. Rejects
    the planner-kind rule, which read provenance from the step's TYPE
    (Bin/Aggregate) rather than from who wrote the groupby, and so never
    split a declared Bin groupby at all.
    """
    bin_ = fm.Bin("v", groupby=["c"]) if spelling == "object" else {"groupby": ["c"]}
    build = _BIN_GROUPBY_DOORS[door]
    chart = build(bin_)
    svg = _quiet(chart.to_svg)
    expected = 4 if door == "Fc" else 2
    assert tag_counts(svg).get("polyline", 0) == expected

    line_nodes = sum(
        len(batch["nodes"])
        for panel in _quiet(lambda: _scene(chart))["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "line"
    )
    assert line_nodes == expected


def test_a_declared_bin_groupbys_wire_carries_written_groupby():
    """The lowered ``Bin``'s wire names the caller's declared groupby, under either spelling."""
    for bin_ in (fm.Bin("v", groupby=["c"]), {"groupby": ["c"]}):
        chart = fm.Chart(df).mark_line().encode(x=fm.X("v", bin=bin_), y="count()")
        wire = json.loads(chart.to_spec().to_json())
        bin_steps = [step for step in wire["transforms"] if step.get("type") == "bin"]
        assert any(step.get("written_groupby") == ["c"] for step in bin_steps)


def test_a_bin_with_no_declared_groupby_is_unaffected_by_written_groupby():
    """A planner-grouped Bin (``bin=True`` plus a grouping channel) carries no ``written_groupby``.

    The control: the planner's own addition to the Bin's groupby (from
    ``color``) must not be marked as caller-written, so the wire carries no
    ``written_groupby`` key at all.
    """
    chart = fm.Chart(df).mark_line().encode(x=fm.X("v", bin=True), y="count()", color="c")
    svg = _quiet(chart.to_svg)
    assert tag_counts(svg) == {"polyline": 2}
    wire = json.loads(chart.to_spec().to_json())
    bin_steps = [step for step in wire["transforms"] if step.get("type") == "bin"]
    assert all("written_groupby" not in step for step in bin_steps)


def test_a_shape_beside_a_user_grouped_step_is_still_reported_ignored():
    """The line splits by the step's group, and the shape it does not draw still warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = _user_fit(["c"], shape="c").to_svg()
    assert tag_counts(svg) == {"polyline": 2}
    assert [str(w.message) for w in caught if "has no effect" in str(w.message)] == [
        "mark_line: shape 'c' has no effect: no layer's mark draws shape (this line mark "
        "does not), so it gets no scale and no legend; it was ignored"
    ]


def test_a_density_grouped_by_shape_draws_what_its_detail_draws():
    """The shape-grouped density draws the detail-grouped density's areas, mark for mark."""
    by_shape = _quiet(lambda: mk("density", x="v", shape="c").to_svg())
    by_detail = _quiet(lambda: mk("density", x="v", detail="c").to_svg())
    assert census(by_shape) == census(by_detail)


# A sibling's grouping channel never splits another layer's series: each
# layer splits by its own bindings over its parent's.
_SERIES_HOSTS: dict[str, Callable[[], fm.Chart]] = {
    "line": lambda: agg("line", x="v"),
    "area": lambda: agg("area", x="v"),
    "smooth-ci": lambda: _smooth(ci=0.95),
}
_GROUPING_SIBLINGS: dict[str, Callable[[], fm.Chart]] = {
    "shape": lambda: P(shape="c"),
    "stroke_dash": lambda: mk("line", x="v", y="y", stroke_dash="c"),
    "detail": lambda: mk("line", x="v", y="y", detail="c"),
}
_SIBLING_SERIES: dict[str, Case] = {
    f"{host}+{channel}/{door}": case
    for host, build_host in _SERIES_HOSTS.items()
    for channel, build_sibling in _GROUPING_SIBLINGS.items()
    for door, case in {
        **plus_doors(build_host, build_sibling, lcs=False),
        "L": Case(
            lambda build_host=build_host, channel=channel: build_host().layer(
                layer("point", x="v", y="y", **{channel: "c"})
                if channel == "shape"
                else fm.Layer(mark="line", encoding={"x": "v", "y": "y", channel: "c"})
            ),
            (build_host, build_sibling),
        ),
    }.items()
    if door not in ("LCi",)
}


@pytest.mark.parametrize("case", sorted(_SIBLING_SERIES))
def test_a_siblings_grouping_channel_never_splits_another_layers_series(case):
    """Each layer draws what it draws alone, static and interactive, on every door.

    Rejects a sibling's shape, stroke_dash or detail splitting the host's
    line, area or smooth band and line.
    """
    check_rendering(_SIBLING_SERIES[case])


def test_a_smooth_ci_under_detail_carries_detail_on_its_band_and_line():
    """The band draws detail as the line does, so both carry it on the wire and split per group."""
    chart = _smooth(ci=0.95, detail="c")
    styles = {layer["mark"]: layer.get("mark_style", {}) for layer in _wire(chart)["layers"]}
    assert styles["ribbon"].get("detail") == "c" and styles["line"].get("detail") == "c"
    assert tag_counts(_quiet(chart.to_svg)) == {"path": 2, "polyline": 2}


# A grouped box, violin, density or histogram under a facet: a panel whose
# every hue is null draws empty, and the drop is counted over the whole
# frame. ``mark_boxplot`` lowers two named steps (``BoxStats``, ``Outliers``)
# over the same rows and key, so they report once, naming the first.
_GROUPED_FACET_PANELS: dict[str, tuple[Callable[[], fm.Chart], list[str]]] = {
    "boxplot": (
        lambda: fm.Chart(_NULL_PANEL).mark_boxplot().encode(x="cat", y="val", color="h").facet("f"),
        ["BoxStats"],
    ),
    "violin": (
        lambda: (
            fm.Chart(_NULL_PANEL)
            .mark_violin(inner="none")
            .encode(x="cat", y="val", color="h")
            .facet("f")
        ),
        ["Violin"],
    ),
    "density": (
        lambda: fm.Chart(_NULL_PANEL).mark_density().encode(x="val", color="h").facet("f"),
        ["Kde"],
    ),
    "histogram": (
        lambda: fm.Chart(_NULL_PANEL).mark_histogram().encode(x="val", color="h").facet("f"),
        ["Bin"],
    ),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("kind", sorted(_GROUPED_FACET_PANELS))
def test_a_grouped_statistics_facet_panel_the_null_key_drop_emptied_draws_empty(kind, output):
    """``F2``, whose every hue is null, draws its frame with no marks; each step counts its drop once.

    Rejects the panel vanishing and a per-panel all-null warning naming the
    facet field ``f``.
    """
    build, steps = _GROUPED_FACET_PANELS[kind]
    render = {"static": lambda chart: chart.to_svg(), "interactive": _scene}[output]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = render(build())
    said = [str(w.message) for w in caught]
    assert [m for m in said if "null" in m] == [
        f"{step}: 10 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
        for step in steps
    ]
    if output == "static":
        panels = panel_census(drawn)
        assert len(panels) == 2
        assert panels[0] and not panels[1], [dict(panel) for panel in panels]
    else:
        drawn_panels = [
            sum(len(batch["nodes"]) for batch in panel["marks"]) > 0 for panel in drawn["panels"]
        ]
        assert drawn_panels == [True, False]


#: 30 rows, two real ``h`` groups of 10 and 10 rows whose ``h`` is null.
_PARTLY_NULL_HUE = pl.DataFrame(
    {
        "x": list(range(30)),
        "y": [float(i % 5) for i in range(30)],
        "b": [float(i % 2) for i in range(30)],
        "h": ["p"] * 10 + ["q"] * 10 + [None] * 10,
    }
)


def test_mark_smooth_with_a_partly_null_hue_warns_once_and_still_draws_both_groups():
    """``mark_smooth(color='h')`` drops the null-``h`` rows from the per-group fit and warns once.

    Rejects the whole-batch-fallback policy staying silent on a partial
    drop, the way it only should for `Impute`'s fill (below).
    """
    chart = fm.Chart(_PARTLY_NULL_HUE).mark_smooth(method="lm").encode(x="x", y="y", color="h")
    drawn, dropped = _drop_warnings(lambda: chart, lambda c: c.to_svg())
    assert dropped == [
        "Smooth: 10 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]
    assert tag_counts(drawn).get("polyline", 0) == 2


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("method", ["lm", "loess", "robust", "logistic", "glm"])
def test_lmplot_with_a_partly_null_hue_warns_once_and_still_draws_both_groups(method, output):
    """``lmplot(method=..., hue='h')`` drops the null-``h`` rows from each group's fit and warns once.

    Every method routes through one of four transforms: ``lm``/``loess``
    both lower ``Smooth``, as ``residplot`` does.
    """
    transform = {
        "lm": "Smooth",
        "loess": "Smooth",
        "robust": "Robust",
        "logistic": "Logistic",
        "glm": "Glm",
    }[method]
    y = "b" if method == "logistic" else "y"
    chart = fm.lmplot(
        _PARTLY_NULL_HUE, x="x", y=y, hue="h", method=method, scatter=False, show_metrics=False
    )
    render = {"static": lambda c: c.to_svg(), "interactive": _scene}[output]
    drawn, dropped = _drop_warnings(lambda: chart, render)
    assert dropped == [
        f"{transform}: 10 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]
    if output == "static":
        assert tag_counts(drawn).get("polyline", 0) == 2
    else:
        assert _scene_line_counts(drawn) == [2]


#: 10 rows, every ``h`` null: the pooled-fit fixture.
_ALL_NULL_HUE = pl.DataFrame(
    {
        "x": list(range(10)),
        "y": [float(i % 5) for i in range(10)],
        "b": [float(i % 2) for i in range(10)],
        "h": pl.Series([None] * 10, dtype=pl.Utf8),
    }
)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("method", ["lm", "loess"])
def test_mark_smooth_with_an_all_null_hue_renders_one_pooled_fit(method, output):
    """``mark_smooth(color='h')``, every ``h`` null, draws one pooled fit and warns the all-null text.

    The pooled fit's output carries ``h`` all null at its original dtype,
    so color reads it without refusing.
    """
    chart = fm.Chart(_ALL_NULL_HUE).mark_smooth(method=method).encode(x="x", y="y", color="h")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = chart.to_svg() if output == "static" else _scene(chart)
    said = [str(w.message) for w in caught]
    assert said == [
        "Smooth: every row has a null in its groupby key (h), so the data could not be "
        "split into groups; the whole batch was used as one ungrouped computation instead"
    ]
    if output == "static":
        assert tag_counts(drawn).get("polyline", 0) == 1
    else:
        assert _scene_line_counts(drawn) == [1]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("method", ["lm", "robust", "logistic", "glm"])
def test_lmplot_with_an_all_null_hue_warns_the_all_null_text_alone_and_draws_one_fit(
    method, output
):
    """``lmplot(hue='h')``, every ``h`` null, warns the all-null text alone and draws one fit.

    Rejects the doubled voice a column-gate "color h was dropped" warning
    beside it would add: the pooled output carries ``h``, so no layer
    reads a column its rows lack.
    """
    transform = {"lm": "Smooth", "robust": "Robust", "logistic": "Logistic", "glm": "Glm"}[method]
    y = "b" if method == "logistic" else "y"
    chart = fm.lmplot(
        _ALL_NULL_HUE, x="x", y=y, hue="h", method=method, scatter=False, show_metrics=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = chart.to_svg() if output == "static" else _scene(chart)
    said = [str(w.message) for w in caught]
    assert said == [
        f"{transform}: every row has a null in its groupby key (h), so the data could not "
        "be split into groups; the whole batch was used as one ungrouped computation instead"
    ]
    if output == "static":
        assert tag_counts(drawn).get("polyline", 0) == 1
    else:
        assert _scene_line_counts(drawn) == [1]


#: ``_ALL_NULL_HUE``, twice, under two facet panels: every ``h`` null in every panel.
_ALL_NULL_HUE_FACETED = pl.concat(
    [_ALL_NULL_HUE.with_columns(pl.lit(panel).alias("f")) for panel in ("F1", "F2")]
)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("method", ["lm", "loess"])
def test_mark_smooth_faceted_with_an_all_null_hue_pools_one_fit_per_panel(method, output):
    """``mark_smooth(color='h').facet('f')``, every ``h`` null in every panel, pools a fit per panel.

    Each panel draws one default-colored pooled fit, and the chart raises
    only the ``GroupKeyAllNull`` warning. Rejects the pooled fallback
    nulling the facet column along with ``h``: that emptied every panel
    (``RenderWarning::NoRowsToDraw``) instead of drawing the per-panel
    pooled fit the facet column should still carry.
    """
    chart = (
        fm.Chart(_ALL_NULL_HUE_FACETED).mark_smooth(method=method).encode(x="x", y="y", color="h")
    ).facet("f")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = chart.to_svg() if output == "static" else _scene(chart)
    said = [str(w.message) for w in caught]
    assert said == [
        "Smooth: every row has a null in its groupby key (h), so the data could not be "
        "split into groups; the whole batch was used as one ungrouped computation instead"
    ]
    if output == "static":
        panels = panel_census(drawn)
        assert len(panels) == 2
        assert panels[0] and panels[1], [dict(panel) for panel in panels]
        assert tag_counts(drawn).get("polyline", 0) == 2
    else:
        assert _scene_line_counts(drawn) == [1, 1]


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_lmplot_faceted_with_an_all_null_hue_pools_one_fit_per_panel(output):
    """``lmplot(hue='h', col='f')``, every ``h`` null in every panel, pools a fit per panel.

    Each panel draws one default-colored pooled fit under ``method="logistic"``,
    and the chart raises only the ``GroupKeyAllNull`` warning (no doubled
    "color was dropped" column-gate voice beside it).
    """
    chart = fm.lmplot(
        _ALL_NULL_HUE_FACETED,
        x="x",
        y="b",
        hue="h",
        col="f",
        method="logistic",
        scatter=False,
        show_metrics=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = chart.to_svg() if output == "static" else _scene(chart)
    said = [str(w.message) for w in caught]
    assert said == [
        "Logistic: every row has a null in its groupby key (h), so the data could not be "
        "split into groups; the whole batch was used as one ungrouped computation instead"
    ]
    if output == "static":
        panels = panel_census(drawn)
        assert len(panels) == 2
        assert panels[0] and panels[1], [dict(panel) for panel in panels]
    else:
        assert _scene_line_counts(drawn) == [1, 1]


#: 12 rows, ``h`` all null, ``g`` a genuine two-value column (a x6, b x6): the
#: pooled key must null ``g`` too, since it is not constant across the batch.
_ALL_NULL_HUE_VARYING_DETAIL = pl.DataFrame(
    {
        "x": list(range(12)),
        "y": [float(i % 5) for i in range(12)],
        "g": ["a"] * 6 + ["b"] * 6,
        "h": pl.Series([None] * 12, dtype=pl.Utf8),
    }
)


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_pooled_fit_nulls_a_groupby_column_that_varies_not_just_the_all_null_one(output):
    """``mark_smooth(color='g', detail='h')``, ``h`` all null, draws one pooled fit with no 'a' legend.

    ``g`` (``a``/``b``) joins ``h`` in the statistic's groupby (color plus
    detail), so the pooled fit's key must null ``g`` too, since it is not
    the SAME value on every row the pooled fit covers. Rejects keeping
    ``g``'s first-seen value (``"a"``): that mislabels a fit pooled over
    both groups as belonging to one of them, drawing a one-entry 'a'
    legend for a fit that is not actually all ``a``.
    """
    chart = (
        fm.Chart(_ALL_NULL_HUE_VARYING_DETAIL)
        .mark_smooth(method="lm")
        .encode(x="x", y="y", color="g", detail="h")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if output == "static":
            drawn = chart.to_svg()
            assert legend_set(drawn) == {}
            assert tag_counts(drawn).get("polyline", 0) == 1
        else:
            drawn = _scene(chart)
            assert drawn["legend"] == []
            assert _scene_line_counts(drawn) == [1]
    said = [str(w.message) for w in caught]
    assert said == [
        "Smooth: every row has a null in its groupby key (g, h), so the data could not be "
        "split into groups; the whole batch was used as one ungrouped computation instead"
    ]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("entry", ["lmplot", "residplot"])
def test_a_figures_encode_door_warns_the_one_count_and_draws_two_fits(entry, output):
    """``lmplot(encode={'color': 'h'})`` and ``residplot(encode={'color': 'h'})`` warn once, draw two fits."""
    build = {
        "lmplot": lambda: fm.lmplot(
            _PARTLY_NULL_HUE, x="x", y="y", encode={"color": "h"}, scatter=False, show_metrics=False
        ),
        "residplot": lambda: fm.residplot(
            _PARTLY_NULL_HUE,
            x="x",
            y="y",
            encode={"color": "h"},
            lowess=True,
            show_metrics=False,
        ),
    }[entry]
    render = {"static": lambda c: c.to_svg(), "interactive": _scene}[output]
    drawn, dropped = _drop_warnings(build, render)
    assert dropped == [
        "Smooth: 10 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]
    if output == "static":
        assert tag_counts(drawn).get("polyline", 0) == 2
    else:
        assert _scene_line_counts(drawn) == [2]


#: 16 rows over two facet panels, each with 2 rows of a null ``h`` beside two real groups.
_PARTLY_NULL_HUE_FACETED = pl.DataFrame(
    {
        "x": list(range(16)),
        "y": [float(i % 5) for i in range(16)],
        "h": ["p", "p", "q", "q", None, None, "p", "q"] * 2,
        "f": ["F1"] * 8 + ["F2"] * 8,
    }
)


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_mark_smooth_facet_with_a_partly_null_hue_in_every_panel_warns_the_summed_count_once(
    output,
):
    """``mark_smooth(color='h').facet('f')``, both panels partly null, warns the summed count once.

    The control, the same frame with no null ``h``, warns about nothing.
    """
    render = {"static": lambda c: c.to_svg(), "interactive": _scene}[output]
    chart = (
        fm.Chart(_PARTLY_NULL_HUE_FACETED).mark_smooth(method="lm").encode(x="x", y="y", color="h")
    ).facet("f")
    _drawn, dropped = _drop_warnings(lambda: chart, render)
    assert dropped == [
        "Smooth: 4 rows have a null 'h' and belong to no group, so they were dropped; "
        "fill or filter the nulls in 'h'"
    ]
    control = fm.Chart(_PARTLY_NULL_HUE_FACETED.with_columns(pl.col("h").fill_null("p"))).facet("f")
    control = control.mark_smooth(method="lm").encode(x="x", y="y", color="h")
    _drawn, no_drop = _drop_warnings(lambda: control, render)
    assert no_drop == []


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_lmplot_faceted_with_a_hue_null_in_one_panel_names_that_panel_and_fits_the_other(output):
    """``lmplot(hue='h', col='f')``, ``h`` null only in panel F2: F2's text, F1's per-group fits.

    F1 draws one fit line per real group; F2, pooled, draws one.
    """
    data = pl.DataFrame(
        {
            "x": list(range(20)),
            "y": [float(i % 5) for i in range(20)],
            "h": ["p"] * 5 + ["q"] * 5 + [None] * 10,
            "f": ["F1"] * 10 + ["F2"] * 10,
        }
    )
    build = lambda: fm.lmplot(  # noqa: E731
        data, x="x", y="y", hue="h", col="f", scatter=False, show_metrics=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        drawn = chart.to_svg() if output == "static" else _scene(chart)
    said = [str(w.message) for w in caught if "groupby key" in str(w.message)]
    assert said == [
        "Smooth: every row in facet panel f=F2 has a null in its groupby key (h), so that "
        "panel's rows could not be split into groups; in that panel, the whole batch was "
        "used as one ungrouped computation instead"
    ]
    if output == "static":
        panels = panel_census(drawn)
        assert len(panels) == 2
        assert tag_counts(drawn).get("polyline", 0) == 3
    else:
        assert _scene_line_counts(drawn) == [2, 1]


#: The frame ``test_a_whole_batch_fallback_pooled_in_one_facet_panel_names_it`` facets:
#: F1's hue is partly null (both real groups kept), F2's is entirely null (pooled).
_FACET_PARTIAL_ALL_NULL = pl.DataFrame(
    {
        "x": list(range(8)),
        "y": [float(i) for i in range(8)],
        "h": ["p", "p", "q", "q", None, None, None, None],
        "f": ["F1"] * 4 + ["F2"] * 4,
    }
)

#: Per method: the static SVG and the interactive scene sha256.
_FACET_PARTIAL_ALL_NULL_DIGESTS = {
    "lm": (
        "f4c14ebbdcd7a7046f7f7d4c9532b4560141c5aa246bea2d9232d46f7de7f365",
        "f6c7378bf774c5340dd8d9d638f44c443fd67bffd8217fbd70e40b80cf873963",
    ),
    "loess": (
        "40acd92c197104fe9e9681505cb4c339b8347d9a1c3a48e4d626299a5ec5382c",
        "f47792a3184d842d6bab8946f82794f060793353bb50c21b0c1078dd21b55cd2",
    ),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("method", ["lm", "loess"])
def test_a_whole_batch_fallback_pooled_in_one_facet_panel_names_it(method, output):
    """``mark_smooth(color='h').facet('f')`` pooled only in the all-null panel names it.

    Rejects the whole-frame text, which would falsely claim every row (not
    just the panel's) had a null ``h``.
    """
    import hashlib

    chart = (
        fm.Chart(_FACET_PARTIAL_ALL_NULL)
        .mark_smooth(method=method)
        .encode(x="x", y="y", color="h")
        .facet("f")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = chart.to_svg() if output == "static" else chart.interactive().scene_json
    said = [str(w.message) for w in caught if "groupby key" in str(w.message)]
    assert said == [
        "Smooth: every row in facet panel f=F2 has a null in its groupby key (h), so that "
        "panel's rows could not be split into groups; in that panel, the whole batch was "
        "used as one ungrouped computation instead"
    ]
    svg_digest, scene_digest = _FACET_PARTIAL_ALL_NULL_DIGESTS[method]
    expected = svg_digest if output == "static" else scene_digest
    assert hashlib.sha256(drawn.encode()).hexdigest() == expected


def test_grouped_impute_with_a_partly_null_key_pushes_no_drop_count():
    """``transform_impute(groupby=['h'])`` fills a null-keyed row from the whole column; nothing dropped."""
    data = pl.DataFrame({"h": ["p", "p", "q", None], "v": [1.0, None, 3.0, 4.0]})
    chart = (
        fm.Chart(data)
        .transform(fm.transform_impute("v", method="mean", groupby=["h"]))
        .mark_point()
        .encode(x="v", y="v")
    )
    _drawn, dropped = _drop_warnings(lambda: chart, lambda c: c.to_svg())
    assert dropped == []


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_grouped_impute_faceted_with_every_key_null_in_one_panel_names_it_and_pushes_no_count(
    output,
):
    """``transform_impute(groupby=['h']).facet('f')``, every ``h`` null in F2: F2's text, no drop count."""
    data = pl.DataFrame(
        {
            "h": ["p", "q", "p", "q", None, None, None, None],
            "v": [1.0, None, 3.0, 4.0, 5.0, None, 7.0, 8.0],
            "f": ["F1"] * 4 + ["F2"] * 4,
        }
    )
    chart = (
        fm.Chart(data)
        .transform(fm.transform_impute("v", method="mean", groupby=["h"]))
        .mark_point()
        .encode(x="v", y="v")
        .facet("f")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg() if output == "static" else _scene(chart)
    said = [str(w.message) for w in caught]
    assert [m for m in said if "groupby key" in m] == [
        "Impute: every row in facet panel f=F2 has a null in its groupby key (h), so that "
        "panel's rows could not be split into groups; in that panel, the whole batch was "
        "used as one ungrouped computation instead"
    ]
    assert [m for m in said if "belong to no group" in m] == []


#: A frame with a binary outcome ``b`` and a count ``n`` over two ``c`` groups.
_FITS = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        "b": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "n": [1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 9.0, 7.0, 6.0, 4.0, 3.0, 1.0],
        "c": ["p"] * 6 + ["q"] * 6,
    }
)

_GROUPED_FITS: dict[str, Callable[..., Any]] = {
    "Logistic": lambda **k: fm.Logistic("x", "b", n_grid=20, **k),
    "Glm": lambda **k: fm.Glm("x", "n", family="poisson", n_grid=20, **k),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("fit", sorted(_GROUPED_FITS))
def test_a_logistic_or_glm_fit_with_groupby_fits_each_group(fit, output):
    """``groupby=["c"]`` fits each group on its own rows and the line draws one curve per group.

    Rejects one pooled fit over both groups (no ``groupby`` on the transform).
    """
    grouped = fm.Chart(_FITS).transform(_GROUPED_FITS[fit](groupby=["c"])).mark_line()
    grouped = grouped.encode(x="x", y="y")
    if output == "static":
        assert tag_counts(_quiet(grouped.to_svg)) == {"polyline": 2}
    else:
        assert sum(_scene_line_counts(_quiet(lambda: _scene(grouped)))) == 2
    spec = json.loads(grouped.to_spec().to_json())
    fitted = [t for t in spec["transforms"] if t["type"] == fit.lower()]
    assert fitted and fitted[0]["groupby"] == ["c"]


#: ``_FITS`` with a third group ``r`` of one row, which no fit can use.
_FITS_SPARSE = pl.concat([_FITS, pl.DataFrame({"x": [2.0], "b": [1.0], "n": [3.0], "c": ["r"]})])


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("fit", sorted(_GROUPED_FITS))
def test_a_logistic_or_glm_group_that_cannot_be_fit_draws_none_and_warns(fit, output):
    """Group ``r`` (one row) draws no fit and a warning names it; ``p`` and ``q`` still draw.

    Rejects the one bad group refusing the whole chart, as Smooth and
    Robust degrade a group instead.
    """
    chart = fm.Chart(_FITS_SPARSE).transform(_GROUPED_FITS[fit](groupby=["c"])).mark_line()
    chart = chart.encode(x="x", y="y")
    render = {"static": lambda c: c.to_svg(), "interactive": _scene}[output]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = render(chart)
    assert [str(w.message) for w in caught if "could not fit" in str(w.message)] == [
        f"{fit}: could not fit group c=r (fewer than 2 usable (non-null/non-NaN) x/y points); "
        "it degraded to an empty/flat output instead of failing the whole chart"
    ]
    if output == "static":
        assert tag_counts(drawn) == {"polyline": 2}
    else:
        lines = [b for panel in drawn["panels"] for b in panel["marks"] if b["kind"] == "line"]
        assert sum(len(batch["nodes"]) for batch in lines) == 2


#: ``_FITS`` with a third group ``s`` whose binary outcome ``b`` is perfectly
#: separated by ``x``.
_FITS_SEPARATED = pl.concat(
    [
        _FITS,
        pl.DataFrame(
            {
                "x": [-100.0, -90.0, -80.0, 80.0, 90.0, 100.0],
                "b": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "n": [1.0] * 6,
                "c": ["s"] * 6,
            }
        ),
    ]
)


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_perfectly_separated_logistic_group_draws_none_and_warns(output):
    """Group ``s`` separates perfectly, so it draws no fit and a warning names it; ``p`` and ``q`` draw.

    Rejects the one separated group refusing the whole chart.
    """
    chart = fm.Chart(_FITS_SEPARATED).transform(fm.Logistic("x", "b", n_grid=20, groupby=["c"]))
    chart = chart.mark_line().encode(x="x", y="y")
    render = {"static": lambda c: c.to_svg(), "interactive": _scene}[output]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawn = render(chart)
    assert [str(w.message) for w in caught if "could not fit" in str(w.message)] == [
        "Logistic: could not fit group c=s (perfect separation detected (model classifies every "
        "observation with probability > 0.999)); it degraded to an empty/flat output instead of "
        "failing the whole chart"
    ]
    if output == "static":
        assert tag_counts(drawn) == {"polyline": 2}
    else:
        lines = [b for panel in drawn["panels"] for b in panel["marks"] if b["kind"] == "line"]
        assert sum(len(batch["nodes"]) for batch in lines) == 2


@pytest.mark.parametrize("fit", sorted(_GROUPED_FITS))
def test_an_ungrouped_logistic_or_glm_fit_that_cannot_be_fit_still_refuses(fit):
    """With no groupby the one fit is the whole chart, so its bad rows refuse it."""
    one_row = _FITS_SPARSE.filter(pl.col("c") == "r")
    chart = fm.Chart(one_row).transform(_GROUPED_FITS[fit]()).mark_line().encode(x="x", y="y")
    with pytest.raises(ValueError, match="need at least 2 non-null observations"):
        _quiet(chart.to_svg)


# ---------------------------------------------------------------------------
# A composition's own encoding groups its composite operands' statistics:
# built when the spec is lowered, after the post-merge encoding is known
# ---------------------------------------------------------------------------


def _fit() -> fm.Chart:
    return _smooth()


def _rug() -> fm.Chart:
    return mk("tick", x="v")


#: Per door: a composition whose own ``color`` reaches a fit operand, and the
#: parts it draws alone (each operand under the composition's binding).
_POST_MERGE_DOORS: dict[str, Case] = {
    "+e": Case(
        lambda: (_fit() + P()).encode(color="c"),
        (lambda: _smooth(color="c"), lambda: P(color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "+e-rev": Case(
        lambda: (P() + _fit()).encode(color="c"),
        (lambda: P(color="c"), lambda: _smooth(color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "+e-ci": Case(
        lambda: (_smooth(ci=0.95) + P()).encode(color="c"),
        (lambda: _smooth(ci=0.95, color="c"), lambda: P(color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "nested-outer": Case(
        lambda: ((_fit() + P()) + _rug()).encode(color="c"),
        (lambda: _smooth(color="c"), lambda: P(color="c"), lambda: mk("tick", x="v", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "nested-inner": Case(
        lambda: (_fit() + P()).encode(color="c") + _rug(),
        (lambda: _smooth(color="c"), lambda: P(color="c"), _rug),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "attached-then-+e": Case(
        lambda: (_fit().layer(layer("tick")) + P()).encode(color="c"),
        (
            lambda: _smooth(color="c"),
            lambda: mk("tick", x="v", y="y", color="c"),
            lambda: P(color="c"),
        ),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "Fc": Case(
        lambda: (_fit() + P()).encode(color="c").facet("g"),
        (lambda: _smooth(color="c").facet("g"), lambda: P(color="c").facet("g")),
        faceted=True,
        chart_level=None,
    ),
    "histogram-+e": Case(
        lambda: (mk("histogram", x="v") + _rug()).encode(color="c"),
        (lambda: mk("histogram", x="v", color="c"), lambda: mk("tick", x="v", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "boxplot-+e": Case(
        lambda: (mk("boxplot", x="g", y="y") + P(x="g")).encode(color="c"),
        (lambda: mk("boxplot", x="g", y="y", color="c"), lambda: P(x="g", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "qq-+e": Case(
        lambda: (mk("qq", x="z") + P(x="z")).encode(color="c"),
        (lambda: mk("qq", x="z", color="c"), lambda: P(x="z", color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "contour-+e": Case(
        lambda: (mk("contour", x="v", y="y") + P()).encode(color="g"),
        (lambda: mk("contour", x="v", y="y", color="g"), lambda: P(color="g")),
        legends={"g": ("a", "b")},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("door", sorted(_POST_MERGE_DOORS))
def test_a_compositions_own_color_groups_its_composite_operands_statistic(door):
    """``(fit + points).encode(color="c")`` fits per ``c`` as its points color, static and interactive.

    Each operand draws what it draws alone under the composition's binding,
    on ``+`` in both orders, nested ``+`` (the outer and the inner
    composition's binding), a fit attached to before the merge, and faceted.
    Rejects building the fit when the mark resolved at the merge, before the
    composition's encoding existed, which fitted one curve through every
    row and dropped its color with a warning.
    """
    check_rendering(_POST_MERGE_DOORS[door])


#: Per composition: the groupby its fit's statistic carries.
_POST_MERGE_GROUPBY: dict[str, tuple[Callable[[], Any], str, list[str] | None]] = {
    "the-compositions-color": (lambda: (_fit() + P()).encode(color="c"), "smooth", ["c"]),
    "an-alias-on-the-composition": (lambda: (_fit() + P()).encode(fill="c"), "smooth", ["c"]),
    "the-operands-own-color-wins": (
        lambda: (_smooth(color="g") + P()).encode(color="c"),
        "smooth",
        ["g"],
    ),
    "the-operands-own-fill-shadows-the-compositions-color": (
        lambda: (_smooth(fill="g") + P()).encode(color="c"),
        "smooth",
        ["g"],
    ),
    "a-siblings-color-never-reaches-the-fit": (lambda: _fit() + P(color="c"), "smooth", None),
    "the-inner-compositions-color-wins-over-the-outer": (
        lambda: ((_fit() + P()).encode(color="c") + _rug()).encode(color="g"),
        "smooth",
        ["c"],
    ),
    "outer-detail-joins-inner-color": (
        lambda: ((_fit() + P()).encode(color="c") + _rug()).encode(detail="g"),
        "smooth",
        ["c", "g"],
    ),
    "faceted": (
        lambda: (_fit() + P()).encode(color="c").facet("g"),
        "smooth",
        ["c", "g"],
    ),
    "boxen": (
        lambda: (mk("boxen", x="g", y="z") + P(x="g", y="z")).encode(color="c"),
        "letter_value",
        ["c"],
    ),
    "contour": (lambda: (mk("contour", x="v", y="y") + P()).encode(color="g"), "kde2_d", ["g"]),
}


@pytest.mark.parametrize("case", sorted(_POST_MERGE_GROUPBY))
def test_a_compositions_grouping_reaches_a_fit_only_where_the_fit_binds_none(case):
    """The composition's grouping channels join the fit's groupby, the nearest binding winning.

    The operand's own binding, under any spelling, wins over the
    composition's; an inner composition's wins over an outer one's on the
    same channel, and the two join on different channels. Rejects a
    sibling's binding reaching the fit (sibling isolation) and the
    composition's binding overriding the operand's own.
    """
    build, step, expected = _POST_MERGE_GROUPBY[case]
    assert _statistic_groupby(_quiet(build), step) == [expected]


def test_a_composite_operands_layers_redraw_as_the_grouped_mark_draws_them():
    """A filled density grouped by its composition draws isoline segments, as a grouped one does.

    Rejects swapping only the statistic under layers built for the
    ungrouped density (filled polygons over grouped surfaces).
    """
    chart = _quiet(lambda: (mk("density", x="v", y="y") + P()).encode(color="g"))
    marks = [layer_["mark"] for layer_ in _wire(chart)["layers"]]
    assert marks == ["segment", "point"]
    alone = _quiet(lambda: mk("density", x="v", y="y", color="g").to_svg())
    assert census(_quiet(chart.to_svg)) == census(alone) + census(_quiet(P(color="g").to_svg))


def test_the_composition_is_rebuilt_without_changing_the_callers_chart():
    """Lowering the composition leaves the chart the caller holds as it was."""
    chart = (_fit() + P()).encode(color="c")
    operands = chart._operands
    layers = list(chart._layers)
    _quiet(chart.to_svg)
    assert chart._operands is operands
    assert chart._layers == layers


# ---------------------------------------------------------------------------
# A host's grouping bound after its layers were attached groups its statistic
# as the same binding written before them does, on its own and as a + operand
# ---------------------------------------------------------------------------


def _attached_fit() -> fm.Chart:
    return _smooth().layer(layer("tick"))


#: Per spelling: the host bound after ``.layer(...)``, and the same binding
#: written before it.
_POST_ATTACH: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
    **{
        f"smooth/{channel}": (
            lambda channel=channel: _attached_fit().encode(**{channel: "c"}),
            lambda channel=channel: _smooth(**{channel: "c"}).layer(layer("tick")),
        )
        for channel in ("color", "fill", "detail", "shape")
    },
    "density/color": (
        lambda: mk("density", x="v").layer(layer("rule")).encode(color="c"),
        lambda: mk("density", x="v", color="c").layer(layer("rule")),
    ),
}


@pytest.mark.parametrize("case", sorted(_POST_ATTACH))
def test_a_hosts_grouping_bound_after_attach_draws_what_it_draws_bound_before(case):
    """``S().layer(tick).encode(color="c")`` fits per ``c``, static and interactive, as the pre-attach spelling.

    Rejects building the fit when the layers were attached, before the
    host's later binding existed: it was refused (``unknown column 'c'``,
    with a remedy naming ``+``).
    """
    post, pre = _POST_ATTACH[case]
    assert _quiet(lambda: post().to_svg()) == _quiet(lambda: pre().to_svg())
    assert _quiet(lambda: post().interactive().scene_json) == _quiet(
        lambda: pre().interactive().scene_json
    )
    kind = "kde" if case.startswith("density") else "smooth"
    assert _statistic_groupby(_quiet(post), kind) == _statistic_groupby(_quiet(pre), kind)


#: Per spelling and door: a host bound after the attach composed with points,
#: and its parts.
_POST_ATTACH_DOORS: dict[str, Case] = {
    f"{channel}/{door}": Case(
        build,
        parts,
        legends={"c": ("p", "q")} if channel in ("color", "fill") else None,
        chart_level=None,
    )
    for channel in ("color", "fill", "detail", "shape")
    for door, build, parts in (
        (
            "+left",
            lambda channel=channel: _attached_fit().encode(**{channel: "c"}) + P(),
            (
                lambda channel=channel: _smooth(**{channel: "c"}),
                lambda channel=channel: mk("tick", x="v", y="y", **{channel: "c"}),
                P,
            ),
        ),
        (
            "+right",
            lambda channel=channel: P() + _attached_fit().encode(**{channel: "c"}),
            (
                P,
                lambda channel=channel: _smooth(**{channel: "c"}),
                lambda channel=channel: mk("tick", x="v", y="y", **{channel: "c"}),
            ),
        ),
    )
}


@pytest.mark.parametrize("door", sorted(_POST_ATTACH_DOORS))
def test_a_hosts_grouping_bound_after_attach_is_its_own_in_either_operand_order(door):
    """``P() + S().layer(tick).encode(color="c")`` fits per ``c`` like ``S().encode(color="c").layer(tick)``.

    Rejects the composition reading the host's record as ungrouped: one
    fit, its color dropped with a warning, so the order of ``.layer`` and
    ``.encode`` decided the chart.
    """
    check_rendering(_POST_ATTACH_DOORS[door])
    assert _statistic_groupby(_quiet(_POST_ATTACH_DOORS[door].build), "smooth") == [["c"]]


def test_a_regrouped_host_leaves_the_callers_chart_as_it_was():
    """Rendering the regrouped host never rewrites the chart the caller holds."""
    chart = _attached_fit().encode(color="c")
    transforms, layers, source = (
        list(chart._transforms),
        list(chart._layers),
        chart._composite_source,
    )
    _quiet(chart.to_svg)
    assert chart._transforms == transforms
    assert chart._layers == layers
    assert chart._composite_source is source


# ---------------------------------------------------------------------------
# A figure's statistics regroup under a grouping bound after the call, as a
# composite mark's do; show_metrics warns only when the caller asked for it,
# one voice whatever door bound the grouping
# ---------------------------------------------------------------------------


def _resid(**kwargs: Any) -> fm.Chart:
    return fm.residplot(df, x="v", y="y", **kwargs)


#: Per door: a figure under a later grouping, and the parts it draws alone.
_FIGURE_REGROUP_DOORS: dict[str, Case] = {
    "residplot+e": Case(
        lambda: (_resid() + P()).encode(color="c"),
        (lambda: _resid(encode={"color": "c"}), lambda: P(color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "residplot-lowess+e": Case(
        lambda: (_resid(lowess=True) + P()).encode(color="c"),
        (lambda: _resid(lowess=True, encode={"color": "c"}), lambda: P(color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "residplot.encode": Case(
        lambda: _resid().encode(color="c"),
        (lambda: _resid(encode={"color": "c"}),),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
    "lmplot+e": Case(
        lambda: (fm.lmplot(df, x="v", y="y") + P()).encode(color="c"),
        (lambda: fm.lmplot(df, x="v", y="y", encode={"color": "c"}), lambda: P(color="c")),
        legends={"c": ("p", "q")},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("door", sorted(_FIGURE_REGROUP_DOORS))
def test_a_figure_regroups_under_a_grouping_bound_after_the_call(door):
    """``(residplot(...) + P).encode(color="c")`` refits per ``c``, as ``residplot(encode=...)`` does.

    Each figure under the later binding draws what the same figure called
    with that ``encode=`` draws, static and interactive. Left at its
    default ``show_metrics``, no door warns about the single-fit corner it
    never asked for: it is dropped the same way everywhere.
    """
    case = _FIGURE_REGROUP_DOORS[door]
    check_rendering(case)
    _svg, dropped = _dropped_warnings(case.build)
    assert dropped == []
    assert _statistic_groupby(_quiet(case.build), "smooth")[0] == ["c"]


#: show_metrics=True, per entry: a maker of its chart, taking show_metrics=
#: and encode= as the figure's own constructor keywords, and the voice its
#: warning speaks as (every figure speaks as itself, regplot included).
_METRICS_WARN_ENTRIES: dict[str, tuple[Callable[..., fm.Chart], str]] = {
    "residplot": (
        lambda show_metrics=None, encode=None: _resid(show_metrics=show_metrics, encode=encode),
        "residplot",
    ),
    "residplot-lowess": (
        lambda show_metrics=None, encode=None: _resid(
            lowess=True, show_metrics=show_metrics, encode=encode
        ),
        "residplot",
    ),
    "lmplot-lm": (
        lambda show_metrics=None, encode=None: fm.lmplot(
            df, x="v", y="y", method="lm", show_metrics=show_metrics, encode=encode
        ),
        "lmplot",
    ),
    "lmplot-loess": (
        lambda show_metrics=None, encode=None: fm.lmplot(
            df, x="v", y="y", method="loess", show_metrics=show_metrics, encode=encode
        ),
        "lmplot",
    ),
    "lmplot-robust": (
        lambda show_metrics=None, encode=None: fm.lmplot(
            df, x="v", y="y", method="robust", show_metrics=show_metrics, encode=encode
        ),
        "lmplot",
    ),
    "regplot": (
        lambda show_metrics=None, encode=None: fm.regplot(
            df, x="v", y="y", show_metrics=show_metrics, encode=encode
        ),
        "regplot",
    ),
    "lmplot-facet": (
        lambda show_metrics=None, encode=None: fm.lmplot(
            df, x="v", y="y", col="g", show_metrics=show_metrics, encode=encode
        ),
        "lmplot",
    ),
}

#: Per entry: the doors it is built under. Composition doors (the rebuild
#: mechanism, generic over every entry) are exercised for every entry --
#: every entry here routes through Smooth, residplot and regplot included;
#: every entry gets the direct and post-hoc single-operand doors, where
#: the warning's per-method gate applies. The faceted entry's `+right` door
#: also drops its own facet with the generic "primary layer wins"
#: composition warning, pinned on its own below rather than skipped.
_METRICS_WARN_DOORS: dict[str, tuple[str, ...]] = {
    "residplot": ("encode=", ".encode", "+left", "+right", "nested"),
    "lmplot-lm": ("encode=", ".encode", "+left", "+right", "nested"),
    "residplot-lowess": ("encode=", ".encode", "+left", "+right", "nested"),
    "lmplot-loess": ("encode=", ".encode", "+left", "+right", "nested"),
    "lmplot-robust": ("encode=", ".encode", "+left", "+right", "nested"),
    "regplot": ("encode=", ".encode", "+left", "+right", "nested"),
    "lmplot-facet": ("encode=", ".encode", "+left", "+right", "nested"),
}


def _metrics_warn_build(entry: str, door: str, show_metrics: bool | None) -> Callable[[], fm.Chart]:
    make, _voice = _METRICS_WARN_ENTRIES[entry]
    if door == "encode=":
        return lambda: make(show_metrics=show_metrics, encode={"color": "c"})
    if door == ".encode":
        return lambda: make(show_metrics=show_metrics).encode(color="c")
    if door == "+left":
        return lambda: (make(show_metrics=show_metrics) + P()).encode(color="c")
    if door == "+right":
        return lambda: (P() + make(show_metrics=show_metrics)).encode(color="c")
    return lambda: ((make(show_metrics=show_metrics) + P()) + _rug()).encode(color="c")


_METRICS_WARN_CASES: list[tuple[str, str]] = [
    (entry, door) for entry, doors in _METRICS_WARN_DOORS.items() for door in doors
]
_METRICS_WARN_IDS = [f"{entry}/{door}" for entry, door in _METRICS_WARN_CASES]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", _METRICS_WARN_CASES, ids=_METRICS_WARN_IDS)
def test_a_figures_default_metrics_never_warns_under_a_grouping(case, output):
    """Left unset, ``show_metrics`` drops a single-fit corner silently on every door.

    Rejects the rebuild door warning about a layer the caller never asked
    for while the direct ``encode=`` door, which groups the identical
    statistic, stays silent -- one chart, one voice, whatever door bound
    the grouping.
    """
    entry, door = case
    build = _metrics_warn_build(entry, door, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    messages = [
        str(w.message)
        for w in caught
        if "corner is not drawn" in str(w.message) or "layer was dropped" in str(w.message)
    ]
    assert messages == []


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", _METRICS_WARN_CASES, ids=_METRICS_WARN_IDS)
def test_a_figures_explicit_metrics_warns_once_on_every_door_the_same_way(case, output):
    """``show_metrics=True`` with no single fit warns once, the same voice whatever door bound it.

    Rejects warning only on the post-hoc doors while the direct ``encode=``
    door, which groups the identical statistic, stays silent (the voice
    the removed ``_redrawn_layers`` warning had). The rendered chart is the
    one the default renders: only the warning differs.
    """
    entry, door = case
    _make, voice = _METRICS_WARN_ENTRIES[entry]
    default_build = _metrics_warn_build(entry, door, None)
    true_build = _metrics_warn_build(entry, door, True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = true_build()
        rendered = chart.to_svg() if output == "static" else chart.interactive().scene_json
    messages = [str(w.message) for w in caught if "corner is not drawn" in str(w.message)]
    assert messages == [
        f"{voice}: show_metrics=True has no single fit to describe: the fit is computed "
        "per group of 'c', so the R²/RMSE/MAE corner is not drawn"
    ]
    if output == "static":
        assert rendered == _quiet(lambda: default_build().to_svg())
    else:
        assert rendered == _quiet(lambda: default_build().interactive().scene_json)


def test_a_figures_explicit_metrics_warning_reads_grammatically_for_three_grouping_fields():
    """Three grouping fields join with the Oxford comma, not a bare ``and`` chain.

    ``lmplot``'s composite grouping (color, shape, detail all bound) can
    hold more than one field; the warning's field list goes through the
    same prose-list rule as every other 3+-item refusal.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.lmplot(
            df, x="v", y="y", show_metrics=True, encode={"color": "c", "shape": "g", "detail": "k"}
        ).to_svg()
    messages = [str(w.message) for w in caught if "corner is not drawn" in str(w.message)]
    assert messages == [
        "lmplot: show_metrics=True has no single fit to describe: the fit is computed "
        "per group of 'c', 'g', and 'k', so the R²/RMSE/MAE corner is not drawn"
    ]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("show_metrics", [None, True])
def test_lmplot_facet_plus_right_drops_its_facet_with_a_primary_layer_wins_warning(
    show_metrics, output
):
    """``P() + lmplot(col='g')`` drops the facet operand's own facet/theme; the warning is pinned.

    The `+right` door makes ``lmplot``'s faceted chart the secondary
    operand, which the generic composition rule always drops in favor of
    the primary layer's theme/facet, whatever ``show_metrics`` is. Rejects
    silently skipping this door's own warning in the generic metrics
    tables above, which filter it out by design (it names neither "corner
    is not drawn" nor "layer was dropped").
    """
    build = _metrics_warn_build("lmplot-facet", "+right", show_metrics)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    primary_layer_wins = [str(w.message) for w in caught if "primary layer wins" in str(w.message)]
    assert primary_layer_wins == [
        "Layered chart `+`: secondary layer's theme/facet is ignored; primary layer wins."
    ]


#: Charts a later grouping never reaches, and the SVG and interactive scene
#: digests they render.
_REGROUP_CONTROLS: dict[str, tuple[Callable[[], Any], tuple[str, str]]] = {
    "residplot+P": (
        lambda: _resid() + P(),
        (
            "fd49e8c120af904fad68873e0c1d3eb12d2bde65dcfdea252367c879f4c941d4",
            "4da8ed8948135722a0f089a12e5da3a1f7b678fed02660affd81c07049c3463d",
        ),
    ),
    "lmplot+P": (
        lambda: fm.lmplot(df, x="v", y="y") + P(),
        (
            "7947575162b432e08b6c869276319844f89c056c6f928f5cf3d5addc5a0f1144",
            "1d829d523228086c1f8e271e0f954f6e8df701743ccf794fb6b848cb168be497",
        ),
    ),
    "smooth-tick-pre-attach": (
        lambda: _smooth(color="c").layer(layer("tick")),
        (
            "29f975a0714ba420631e7eb407d53fefd9bf2897aacc0b8b7f71f70074e6d9a4",
            "c26bdbb2ce0872f8e52e3f744054d630ac4788c6740c5afff51f62ef604e9f78",
        ),
    ),
}


@pytest.mark.parametrize("case", sorted(_REGROUP_CONTROLS))
def test_a_chart_no_later_grouping_reaches_renders_as_it_did(case):
    """A figure under a composition binding no grouping channel, and a host bound before the attach."""
    import hashlib

    build, (svg_digest, scene_digest) = _REGROUP_CONTROLS[case]
    assert hashlib.sha256(_quiet(lambda: build().to_svg()).encode()).hexdigest() == svg_digest
    scene = _quiet(lambda: build().interactive().scene_json)
    assert hashlib.sha256(scene.encode()).hexdigest() == scene_digest


def test_a_regrouped_figure_leaves_the_callers_chart_as_it_was():
    """Lowering a composition holding a figure never rewrites the figure the caller holds."""
    figure = _resid()
    chart = (figure + P()).encode(color="c")
    transforms, layers = list(figure._transforms), list(figure._layers)
    operands = chart._operands
    _quiet(chart.to_svg)
    assert figure._transforms == transforms and figure._layers == layers
    assert chart._operands is operands


# ---------------------------------------------------------------------------
# A detail no mark draws is reported ignored only when no statistic grouped by it
# ---------------------------------------------------------------------------

_IGNORED_DETAIL = "encode(detail=...) is accepted but ignored by mark_"


def _detail_warnings(build: Callable[[], Any], output: str) -> list[str]:
    from ferrum._warn import reset_warnings

    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    return [str(w.message) for w in caught if _IGNORED_DETAIL in str(w.message)]


#: Charts whose detail groups a statistic, though no mark draws detail.
_DETAIL_GROUPS: dict[str, Callable[[], Any]] = {
    "histogram": lambda: mk("histogram", x="v", detail="c"),
    "boxplot": lambda: mk("boxplot", x="g", y="y", detail="c"),
    "violin": lambda: mk("violin", x="g", y="y", detail="c"),
    "qq": lambda: mk("qq", x="z", detail="c"),
    "density-2d": lambda: mk("density", x="v", y="y", detail="c"),
    "residplot": lambda: _resid(encode={"detail": "c"}),
    "errorbar": lambda: mk("errorbar", x="g", y="y", detail="c"),
    "bar-mean": lambda: bar_mean(detail="c"),
    "+text-label": lambda: (
        bar_mean(color="c", detail="z") + mk("text", x="g", y="mean(y)", text="mean(y)")
    ),
}

#: Charts whose detail does nothing: no mark draws it and no statistic groups by it.
_DETAIL_IGNORED: dict[str, tuple[Callable[[], Any], str]] = {
    "point": (lambda: P(detail="c"), "point"),
    "bar-raw": (lambda: mk("bar", x="g", y="y", detail="c"), "bar"),
    "+point-operand": (lambda: P(detail="c") + P(y="z"), "point"),
    "user-step": (
        lambda: (
            fm.Chart(df)
            .transform(fm.Aggregate([fm.AggregateOp("y", "mean", "m")], groupby=["g", "c"]))
            .mark_bar()
            .encode(x="g", y="m", detail="c")
        ),
        "bar",
    ),
    "point-aggregate": (
        lambda: mk("point", x="c", y="m", detail="c").transform(
            fm.Aggregate([fm.AggregateOp("y", "mean", "m")], groupby=["c"])
        ),
        "point",
    ),
    "point-window": (
        lambda: P(detail="c").transform(
            fm.transform_window({"op": "row_number", "as": "rk"}, groupby=["c"])
        ),
        "point",
    ),
    "point-join-aggregate": (
        lambda: P(detail="c").transform(
            fm.transform_join_aggregate({"field": "y", "fn": "mean", "as": "ym"}, groupby=["c"])
        ),
        "point",
    ),
    "point-plus-aggregate-branch": (
        lambda: (
            mk("point", x="c", y="m", detail="c").transform(
                fm.Aggregate([fm.AggregateOp("y", "mean", "m")], groupby=["c"])
            )
            + P(y="z")
        ),
        "point",
    ),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_DETAIL_GROUPS))
def test_a_detail_a_statistic_groups_by_is_never_reported_ignored(case, output):
    """Rejects the ignored-detail warning on a statistic split per detail value.

    It fired at aliasing, before any statistic was lowered, so every
    composite, figure, encoding aggregate and label grouped by detail was
    told its detail did nothing.
    """
    assert _detail_warnings(_DETAIL_GROUPS[case], output) == []


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_DETAIL_IGNORED))
def test_a_detail_nothing_draws_or_groups_by_is_reported_ignored_once(case, output):
    """The control: a detail no mark draws and no statistic groups by still warns, once.

    Includes a caller-written ``Aggregate``/``Window``/``JoinAggregate`` step
    that groups by the detail field for its own reason, before any mark
    resolved a statistic: the field is reported ignored anyway, on the flat
    chart and inside a ``+`` operand's own branch prefix. Rejects reading
    any step's ``groupby`` on the wire, not only a lowered statistic's.
    """
    build, mark = _DETAIL_IGNORED[case]
    (message,) = _detail_warnings(build, output)
    assert message.startswith(f"{_IGNORED_DETAIL}{mark};")


# ---------------------------------------------------------------------------
# A boxen palette is refused under a color the bands draw, on every spelling and door
# ---------------------------------------------------------------------------

_BOXEN_DF = pl.DataFrame(
    {"group": ["a"] * 200 + ["b"] * 200, "val": list(range(200)) + list(range(100, 300))}
).with_columns(c=pl.Series(["p", "q"] * 200))
_PALETTE = ["#111111", "#222222"]
_PALETTE_REFUSAL = (
    "mark_boxen(palette=...) cannot be combined with a color the bands draw (color=, fill= or "
    "stroke=, on the chart or on a composition it is part of): the color overrides every "
    "band's fill, so the palette would have no visible effect. Drop the color binding or drop "
    "palette="
)


def _boxen(**encoding: Any) -> fm.Chart:
    chart = fm.Chart(_BOXEN_DF).mark_boxen(palette=_PALETTE)
    return chart.encode(**{"x": "group", "y": "val", **encoding})


def _boxen_points(**encoding: Any) -> fm.Chart:
    return fm.Chart(_BOXEN_DF).mark_point().encode(**{"x": "group", "y": "val", **encoding})


_PALETTE_CONFLICTS: dict[str, Callable[[], Any]] = {
    "color": lambda: _boxen(color="c"),
    "fill": lambda: _boxen(fill="c"),
    "stroke": lambda: _boxen(stroke="c"),
    "+e": lambda: (_boxen() + _boxen_points()).encode(color="c"),
    "+e-fill": lambda: (_boxen_points() + _boxen()).encode(fill="c"),
    "nested": lambda: ((_boxen() + _boxen_points()) + _boxen_points()).encode(color="c"),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("case", sorted(_PALETTE_CONFLICTS))
def test_a_boxen_palette_under_a_color_the_bands_draw_is_refused(case, output):
    """Every spelling and door of a color the bands group by and draw refuses the palette.

    Rejects reading only the chart's literal ``color`` key: ``fill=``,
    ``stroke=`` and a composition's color drew the bands in the group
    colors and silently discarded the palette.
    """
    chart = _PALETTE_CONFLICTS[case]()
    with pytest.raises(ValueError) as refused:
        _quiet(chart.to_svg if output == "static" else lambda: chart.interactive().scene_json)
    assert str(refused.value) == _PALETTE_REFUSAL


_PALETTE_CONTROLS: dict[str, Callable[[], Any]] = {
    "no-color": lambda: (
        fm.Chart(_BOXEN_DF.drop("c")).mark_boxen(palette=_PALETTE).encode(x="group", y="val")
    ),
    "color_field": lambda: (
        fm.Chart(_BOXEN_DF.drop("c"))
        .mark_boxen(palette=_PALETTE, color_field="group")
        .encode(x="group", y="val")
    ),
}
_PALETTE_CONTROL_DIGESTS = (
    "913791d61cd343decbb2d1ed4b6e28c3d6addbe9146d3f1a48a71c12a817bf1a",
    "0d475e208cc304ed3c961d67ca592cce613562cde74b7fd27c0dcfc170e1d17a",
)


@pytest.mark.parametrize("case", sorted(_PALETTE_CONTROLS))
def test_a_boxen_palette_without_a_drawn_color_renders_as_it_did(case):
    import hashlib

    chart = _PALETTE_CONTROLS[case]()
    svg = _quiet(chart.to_svg)
    scene = _quiet(lambda: chart.interactive().scene_json)
    assert (
        hashlib.sha256(svg.encode()).hexdigest(),
        hashlib.sha256(scene.encode()).hexdigest(),
    ) == _PALETTE_CONTROL_DIGESTS


# ---------------------------------------------------------------------------
# Error bars and error bands group like the box: by every grouping channel
# ---------------------------------------------------------------------------

#: Per spelling: the errorbar or errorband, and the groupby its ErrorExtent carries.
_ERROR_GROUPED: dict[str, tuple[Callable[[], fm.Chart], list[str]]] = {
    **{
        f"errorbar/{channel}": (
            lambda channel=channel: mk("errorbar", x="g", y="y", **{channel: "c"}),
            ["g", "c"],
        )
        for channel in ("color", "fill", "stroke", "detail", "shape", "stroke_dash")
    },
    **{
        f"errorband/{channel}": (
            lambda channel=channel: mk("errorband", x="v", y="y", **{channel: "c"}),
            ["v", "c"],
        )
        for channel in ("color", "fill", "stroke", "detail", "shape", "stroke_dash")
    },
}


@pytest.mark.parametrize("case", sorted(_ERROR_GROUPED))
def test_an_error_extent_groups_by_each_grouping_channel(case):
    """Rejects ``ErrorExtent`` grouping by the hue only, so detail, shape and stroke_dash pooled."""
    build, expected = _ERROR_GROUPED[case]
    assert _statistic_groupby(_quiet(build), "error_extent") == [expected]
    _svg, dropped = _dropped_warnings(build)
    assert dropped == []


#: Four rows per (t, c): an error band needs several values at each x.
_BAND_DF = pl.DataFrame(
    {
        "t": [1.0, 2.0, 3.0] * 8,
        "v": [float((7 * i) % 11) for i in range(24)],
        "c": (["p"] * 3 + ["q"] * 3) * 4,
    }
)


def _band(**encoding: Any) -> fm.Chart:
    return mk("errorband", _BAND_DF, **{"x": "t", "y": "v", **encoding})


#: Per case: the chart, and the chart drawing the same marks through color.
_ERROR_DRAWN: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart], str, int]] = {
    "errorbar/detail": (
        lambda: mk("errorbar", x="g", y="y", detail="c"),
        lambda: mk("errorbar", x="g", y="y", color="c"),
        "line",
        12,
    ),
    "errorbar/shape": (
        lambda: mk("errorbar", x="g", y="y", shape="c"),
        lambda: mk("errorbar", x="g", y="y", color="c"),
        "line",
        12,
    ),
    "errorband/detail": (lambda: _band(detail="c"), lambda: _band(color="c"), "path", 2),
    "errorband/shape": (lambda: _band(shape="c"), lambda: _band(color="c"), "path", 2),
}


@pytest.mark.parametrize("case", sorted(_ERROR_DRAWN))
def test_an_error_extent_grouped_without_color_draws_one_bar_or_band_per_group(case):
    """One bar per (category, group) and one band per group, static and interactive.

    Rejects one bar or band through every group's rows.
    """
    build, colored, tag, count = _ERROR_DRAWN[case]
    assert tag_counts(_quiet(lambda: build().to_svg())).get(tag) == count
    assert tag_counts(_quiet(lambda: colored().to_svg())).get(tag) == count
    drawn = sum(_quiet(lambda: interactive_census(build())).values())
    assert drawn == sum(_quiet(lambda: interactive_census(colored())).values())


#: Error bars and bands no grouping reaches, and the SVG and scene digests
#: they render.
_ERROR_CONTROLS: dict[str, tuple[Callable[[], fm.Chart], tuple[str, str]]] = {
    "errorbar": (
        lambda: mk("errorbar", x="g", y="y"),
        (
            "c99631559dec413767c30efe68868c1fdd39a7c4cc9839b585255637a73a297f",
            "6b6954e6482c6d0c06e31ecfe897f7643f59947caf575f01167b5369fbb2fd95",
        ),
    ),
    "errorbar/detail-equal-to-x": (
        lambda: mk("errorbar", x="g", y="y", detail="g"),
        (
            "c99631559dec413767c30efe68868c1fdd39a7c4cc9839b585255637a73a297f",
            "6b6954e6482c6d0c06e31ecfe897f7643f59947caf575f01167b5369fbb2fd95",
        ),
    ),
    "errorbar/color": (
        lambda: mk("errorbar", x="g", y="y", color="c"),
        (
            "b61ca59f90e91c9b4fa52475b7a14f952922d977307be545f82f4e9c3f61548a",
            "dffa9265804efc4eedbc9740c2241cc2b50171f9696b85a324f29cb2328e76be",
        ),
    ),
    "errorband": (
        lambda: mk("errorband", x="v", y="y"),
        (
            "3a6f6f9f0abd135801fd7abea060c9d3be24013f00f8ba004b20fe1bf505053d",
            "53209cdbea9de0af9363149deb1a37bd4941e9e53d81213c34af230557d41363",
        ),
    ),
}


@pytest.mark.parametrize("case", sorted(_ERROR_CONTROLS))
def test_an_error_extent_no_new_grouping_reaches_renders_as_it_did(case):
    """No grouping channel, an existing hue, and a grouping field equal to x (deduplicated)."""
    import hashlib

    build, (svg_digest, scene_digest) = _ERROR_CONTROLS[case]
    assert hashlib.sha256(_quiet(lambda: build().to_svg()).encode()).hexdigest() == svg_digest
    scene = _quiet(lambda: build().interactive().scene_json)
    assert hashlib.sha256(scene.encode()).hexdigest() == scene_digest


# ---------------------------------------------------------------------------
# lmplot fits every method per group of its hue and every grouping channel
# ---------------------------------------------------------------------------

_LM_RNG = np.random.default_rng(0)
_LM_X = np.linspace(-3.0, 3.0, 120)
_LM_DF = pl.DataFrame(
    {
        "x": _LM_X,
        "y": 2 * _LM_X + _LM_RNG.normal(0, 1, 120),
        "b": (_LM_RNG.uniform(size=120) < 1 / (1 + np.exp(-2 * _LM_X))).astype(float),
        "g": ["a", "b"] * 60,
    }
)
_LM_METHODS = ("lm", "loess", "logistic", "glm", "robust")
_LM_STEP = {"lm": "smooth", "loess": "smooth", "logistic": "logistic", "glm": "glm"}
_LM_SPELLINGS: dict[str, dict[str, Any]] = {
    "hue": {"hue": "g"},
    "color": {"color": "g"},
    "fill": {"fill": "g"},
    "encode-color": {"encode": {"color": "g"}},
    "encode-detail": {"encode": {"detail": "g"}},
}


def _lmplot(method: str, **kwargs: Any) -> fm.Chart:
    y = "b" if method == "logistic" else "y"
    return fm.lmplot(_LM_DF, x="x", y=y, method=method, **kwargs)


@pytest.mark.parametrize("spelling", sorted(_LM_SPELLINGS))
@pytest.mark.parametrize("method", _LM_METHODS)
def test_lmplot_fits_each_method_per_group_of_every_grouping_spelling(method, spelling):
    """One fit line and band per group, static and interactive, in the group's color when colored.

    Rejects the pooled fit over every group (one line, a neutral stroke)
    that ``logistic``, ``glm`` and ``robust`` drew under ``hue=``, and that
    every method drew under ``color=``, ``fill=`` or an ``encode=`` grouping.
    """
    chart = _lmplot(method, **_LM_SPELLINGS[spelling])
    step = _LM_STEP.get(method, method)
    assert _statistic_groupby(chart, step) == [["g"]]
    svg = _quiet(chart.to_svg)
    assert tag_counts(svg).get("polyline") == 2
    assert tag_counts(svg).get("path") == 2
    batches = Counter(
        batch["kind"]
        for panel in _quiet(lambda: _scene(chart))["panels"]
        for batch in panel["marks"]
        for _node in batch["nodes"]
    )
    assert (batches["line"], batches["ribbon"]) == (2, 2)
    if spelling != "encode-detail":
        strokes = set(re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg))
        fills = set(re.findall(r'<circle[^>]*fill="([^"]+)"', svg))
        assert len(strokes) == 2 and strokes <= fills


#: Ungrouped dedicated fits, and the SVG and scene digests they render on the
#: tree before lmplot passed a grouping to every method.
_LM_CONTROLS: dict[str, tuple[str, str]] = {
    "logistic": (
        "9d01c587dfe98a80e44bbf2d4ce91e7bebfae318fe20abb53a3b7b816244bec3",
        "b67775cdcf8daad0354e56d374b0e97b1aacf0fec4cb7fb04286497d4fbbabf6",
    ),
    "glm": (
        "7684275e841b008430028dfcdf05f35e2b044179f5aefeecc95a439535f3a8fa",
        "fcef1074e4d8fa85f5296edcb618d977455eed4fbdcb527f68c0c1284d0943ac",
    ),
    "robust": (
        "137e9e41ef5e22b0ba446e2f4cbd1f78ade7c9ca499da90f24536752747a428d",
        "19d8fbaa69ad453943ff792d4c8b75c75fd96e02bc2556bb9c371e8259630acb",
    ),
}


@pytest.mark.parametrize("method", sorted(_LM_CONTROLS))
def test_an_ungrouped_lmplot_renders_as_it_did(method):
    import hashlib

    chart = _lmplot(method)
    svg_digest, scene_digest = _LM_CONTROLS[method]
    assert hashlib.sha256(_quiet(chart.to_svg).encode()).hexdigest() == svg_digest
    scene = _quiet(lambda: chart.interactive().scene_json)
    assert hashlib.sha256(scene.encode()).hexdigest() == scene_digest


# ---------------------------------------------------------------------------
# An explicit groupby= wins over the chart's grouping: what the user then gets
# ---------------------------------------------------------------------------


def test_an_explicit_groupby_grouping_by_the_drawn_color_draws_and_warns_the_undrawn_shape():
    """``groupby=["c"]`` with ``color="c", shape="c"``: two histograms, and shape reported ignored.

    The counts group by the explicit ``["c"]`` only, the bars draw what
    the same chart without shape draws, and shape, which no mark draws and
    the explicit groupby took no part from, warns that it did nothing.
    """
    grouped = fm.Chart(df).mark_histogram(groupby=["c"]).encode(x="v", color="c", shape="c")
    plain = fm.Chart(df).mark_histogram(groupby=["c"]).encode(x="v", color="c")
    assert _statistic_groupby(grouped, "bin") == [["c"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = grouped.to_svg()
    assert census(svg) == census(_quiet(plain.to_svg))
    assert _quiet(lambda: interactive_census(grouped)) == _quiet(lambda: interactive_census(plain))
    assert [str(w.message) for w in caught if "shape 'c'" in str(w.message)] == [
        "mark_histogram: shape 'c' has no effect: no layer's mark draws shape (this bar mark "
        "does not), so it gets no scale and no legend; it was ignored"
    ]
