"""The layer encoding model's rule table, cell by cell, on every door (spec §4.6, §6.1-§6.3).

A layer's effective encoding is its own bindings merged over exactly one
parent's, channel by channel, and never over a sibling's: a host for a layer
attached with ``Chart.layer`` or ``SecondaryY``, what the caller wrote on the
composition itself for a ``+`` operand's layer, and nothing for a
``fm.LayerChart`` member. The one cross-sibling flow is the axis: a layer that
binds no x/y takes layer 0's written x/y.

Every case is keyed by its spec §4.6 cell id and door code (§6.1):

- F flat chart, L ``Chart.layer(fm.Layer(...))``, LD ``fm.Layer(data=...)``;
- ``+`` / ``+rev`` the composition and its operands reversed (``+s`` over one
  frame, ``+d`` over two, named in the case), ``+e`` a binding written on the
  composition, SY ``+ fm.SecondaryY(...)``;
- LCs / LCm / LCi / LCx ``fm.LayerChart`` static, ``_build_merged()``,
  ``resolve={"y": "independent"}`` and ``.interactive()``;
- Fc faceted.

A rendering case asserts the oracle census (§6.2): each layer draws what it
draws alone, so the chart's marks equal the sum of its parts' marks rendered
standalone (geometry left out, every style attribute kept, since the panel's
shared axes move positions), its legend set equals the parts' (where the
case names one), and no axis or legend text is an internal column. It also
runs under I: the interactive scene's marks equal the parts' interactive
marks, and ``.interactive().to_html()`` renders. A refusal asserts the exact
string. The wire seam (§6.3) is read off every case: each layer's wire
encoding completed by the chart-level encoding names the fields its part
draws alone, and the chart-level encoding carries no binding written only on
one operand or layer.

Each test names the rejected alternative it discriminates against (spec §10):
D1-B (a grain-preserving leak still recolors or regroups a sibling), D1-C
(the left operand's encoding is the composition's, and a later binding is
promoted to the chart).
"""

from __future__ import annotations

import json
import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm

SY = fm.SecondaryY

# Spec §4.6 fixtures: duplicate (g, y) rows on purpose (rows 0 and 2 share (a, 3.0)).
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
# Census and oracle (spec §6.2)
# ---------------------------------------------------------------------------

_MARK_TAGS = ("circle", "rect", "path", "line", "polyline", "polygon", "text")
_GEOMETRY_ATTRS = frozenset(
    ("cx", "cy", "x", "y", "x1", "x2", "y1", "y2", "width", "height", "points", "d", "transform")
)
_FORBIDDEN_TITLES = ("bin_start", "bin_end", "agg_value")
# The theme's axis tick-label color (`tests/_svg_rects.THEME_CHROME_STROKES`).
_CHROME_TEXT = "#6b7280"


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _clip_groups(root: ET.Element) -> list[ET.Element]:
    """Each outermost plot-area clip group, in document order."""
    out: list[ET.Element] = []

    def walk(element: ET.Element) -> None:
        if _tag(element) == "g" and element.get("clip-path", "").startswith("url(#ferrum-clip-"):
            out.append(element)
            return
        for child in element:
            walk(child)

    walk(root)
    return out


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


def _chrome_texts(svg: str) -> list[ET.Element]:
    root = _svg_root(svg)
    inside = {id(e) for group in _clip_groups(root) for e in group.iter()}
    return [e for e in root.iter() if _tag(e) == "text" and id(e) not in inside]


def legend_set(svg: str) -> dict[str, tuple[str, ...]]:
    """Each legend's title and entries, read off the chrome: a bold start-anchored title, then its entries."""
    legends: dict[str, list[str]] = {}
    title: str | None = None
    title_x = 0.0
    for e in _chrome_texts(svg):
        text = "".join(e.itertext()).strip()
        if e.get("text-anchor") != "start" or "transform" in e.attrib:
            continue
        if e.get("font-weight") == "600" and e.get("font-size") != "13":
            title, title_x = text, float(e.get("x", "0"))
            legends.setdefault(title, [])
        elif title is not None and float(e.get("x", "0")) >= title_x:
            legends[title].append(text)
    return {t: tuple(entries) for t, entries in legends.items()}


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
    )
)


def _scene(chart: Any) -> dict:
    return json.loads(chart.interactive().scene_json)


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


def check_wire_seam(case: Case) -> None:
    """Spec §6.3: each layer's completed wire encoding draws what its part draws alone."""
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
# V1: a value channel's binding never crosses siblings
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


def _v1_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for channel, (value, _mark) in _VALUE_CHANNELS.items():
        binder = _binder(channel)
        legends = legend_set(_quiet(binder).to_svg())
        cases |= _cases(f"V1/{channel}/binder-then-line", plus_doors(binder, line, legends=legends))
        bound_line = lambda channel=channel, value=value: line(**{channel: value})  # noqa: E731
        bound_legends = legend_set(_quiet(bound_line).to_svg())
        cases |= _cases(
            f"V1/{channel}/point-then-bound-line",
            plus_doors(P, bound_line, legends=bound_legends),
        )
        cases[f"V1/{channel}/L-own"] = Case(
            lambda channel=channel, value=value: P().layer(layer("line", **{channel: value})),
            (P, bound_line),
            legends=bound_legends,
            chart_level=None,
        )
        cases[f"V1/{channel}/L-inherited"] = Case(
            lambda channel=channel, value=value: P(**{channel: value}).layer(layer("line")),
            (
                lambda channel=channel, value=value: P(**{channel: value}),
                lambda channel=channel, value=value: line(**{channel: value}),
            ),
            chart_level=None,
        )
    return cases


_V1 = _v1_cases()


@pytest.mark.parametrize("case", sorted(_V1))
def test_V1_a_value_binding_never_crosses_siblings(case):
    """V1: ``P(color="c") + line`` draws one polyline; the legend lists the binder's values only.

    Rejects D1-C: the left's color would split the line, and a right or
    attached binding promoted to the chart would recolor the points.
    """
    check_rendering(_V1[case])


def test_V1_the_left_operands_color_leaves_the_line_whole():
    """V1 named repro: one polyline on every route (rejects D1-C: 2 polylines)."""
    for door, built in plus_doors(lambda: P(color="c"), line).items():
        chart = _quiet(built.build)
        if built.static:
            assert tag_counts(_quiet(chart.to_svg))["polyline"] == 1, door
        kinds = [kind for kind, _ in interactive_census(chart).elements()]
        assert kinds.count("line") == 1, door


@pytest.mark.parametrize("order", ["bound-first", "bound-later"])
def test_V1_a_tooltip_reaches_only_its_own_layer_in_interactive_output(order):
    """V1 (I): the line's tooltip field never becomes the points' tooltip.

    Rejects tooltip promotion: the points would report ``z``.
    """
    bound = line(tooltip="z")
    chart = bound + P() if order == "bound-first" else P() + bound
    tooltips = interactive_tooltip_fields(chart)
    points = tooltips[1] if order == "bound-first" else tooltips[0]
    assert "z" not in points
    assert points == interactive_tooltip_fields(P())[0]


# ---------------------------------------------------------------------------
# V1 by name: a composition's binding names each layer's own column
# ---------------------------------------------------------------------------

_ABC = {"g": ("a", "b", "c")}

_V1_BY_NAME: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_V1_BY_NAME))
def test_V1_by_name_a_compositions_binding_colors_each_layer_by_its_own_column(case):
    """V1: ``(P(df) + line(df2)).encode(color="g")`` colors both by their own ``g``; one legend a, b, c.

    Rejects reading the left's column on the right's rows (the line would
    draw a null group).
    """
    check_rendering(_V1_BY_NAME[case])


# ---------------------------------------------------------------------------
# V2, V3, V4, V5, V6: statistics keep their grain beside any sibling
# ---------------------------------------------------------------------------

_V2: dict[str, Case] = {
    "F": Case(lambda: agg("point", color=mean(fm.Color)), marks={"circle": 2}),
    "L-inherits-the-hosts-aggregate": Case(
        lambda: agg("point", color=mean(fm.Color)).layer(layer("tick")),
        (lambda: agg("point", color=mean(fm.Color)), lambda: agg("tick", color=mean(fm.Color))),
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_V2))
def test_V2_a_value_aggregate_is_computed_per_layer(case):
    check_rendering(_V2[case])


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


def _v3_cases() -> dict[str, Case]:
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


_V3 = _v3_cases()


@pytest.mark.parametrize("case", sorted(_V3))
def test_V3_a_siblings_binding_never_changes_a_layers_grain(case):
    """V3: host 2 bars + 2 points; the tick keeps 2; the bar mean keeps 2; Fc 4 ticks.

    Rejects D1-B: a grain-preserving leak would still recolor the sibling,
    and D1-C: the sibling would group by the leaked field (7 bars).
    """
    check_rendering(_V3[case])


def test_V3_the_tick_beside_a_colored_mean_draws_two_ticks():
    """V3 named repro: ``point(g, mean(y), color=mean(z)) + tick(g, mean(y))`` keeps 2 ticks."""
    for channel, cls in _AGG_CLASSES.items():
        svg = _quiet(lambda: (agg("point", **{channel: mean(cls)}) + agg("tick")).to_svg())
        assert tag_counts(svg).get("line", 0) == 2, channel


def _v4_cases() -> dict[str, Case]:
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


_V4 = _v4_cases()


@pytest.mark.parametrize("case", sorted(_V4))
def test_V4_a_different_frame_sibling_never_empties_or_regroups_a_layer(case):
    """V4: left 2 ticks, right 3 points; no build error; ``bar_mean`` keeps 2 bars.

    Rejects D1-C: the left would group by the right's renamed column and
    draw nothing, and the opacity family's renamed promotion copy would
    refuse its stamped title.
    """
    check_rendering(_V4[case])


_V5 = _cases(
    "P(x=g)+point(mean(y),color=mean(z))",
    plus_doors(
        lambda: P(x="g"),
        lambda: agg("point", color=mean(fm.Color)),
        legends={
            "mean_z": tuple(legend_set(agg("point", color=mean(fm.Color)).to_svg())["mean_z"])
        },
    ),
)


@pytest.mark.parametrize("case", sorted(_V5))
def test_V5_a_right_aggregate_leaves_the_left_raw_and_uncolored(case):
    """V5: the left draws 8 raw uncolored points; one color legend from the right, titled ``mean_z``.

    Rejects D1-C: the left's raw z would draw through the promoted scale.
    """
    check_rendering(_V5[case])


_V6: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_V6))
def test_V6_an_alias_shadows_the_parents_channel(case):
    """V6: a layer's own ``fill``/``stroke`` is its color, so it groups by ``g`` alone (2 points).

    Rejects an alias-blind merge: the layer would also group by the
    parent's color field (7 points, or a vanished layer over two frames).
    """
    check_rendering(_V6[case])


# ---------------------------------------------------------------------------
# V9, V10, V11, V12
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


_V9: dict[str, Case] = {
    "SY(mean_r)": Case(lambda: _mean_r_by_v() + SY("mean_r"), marks={"circle": 8, "polyline": 1}),
    "L(text=my)": Case(
        lambda: _my_column().layer(layer("text", text="my")),
        marks={"circle": 8, "text": 8},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_V9))
def test_V9_a_user_transforms_output_binds_on_every_door(case):
    check_rendering(_V9[case])


def test_V9_the_attached_text_layer_draws_the_calculated_column():
    labels = set(chrome_text(_my_column().layer(layer("text", text="my")).to_svg()))
    assert set(df["k"]) <= labels


_V10: dict[str, Case] = {
    **{
        k: v
        for k, v in _cases(
            "color-union",
            plus_doors(lambda: P(color="g"), lambda: line(df2, color="g"), legends=_ABC, lcs=False),
        ).items()
        if not k.endswith("LCx")
    },
}


@pytest.mark.parametrize("case", sorted(_V10))
def test_V10_one_scale_and_one_legend_over_every_drawing_layer(case):
    """V10: a ``g`` color on both layers over two frames draws one legend a, b, c.

    Rejects first-layer domains: the legend would list a, b only.
    """
    case = _V10[case]
    chart = _quiet(case.build)
    assert legend_set(_quiet(chart.to_svg)) == _ABC


@pytest.mark.parametrize(
    "build",
    [lambda: P(size="z") + P(df2, size="z"), lambda: P(df2, size="z") + P(size="z")],
    ids=["size", "size-rev"],
)
def test_V10_a_size_on_both_layers_draws_one_legend_over_the_union(build):
    """V10: one size legend titled ``z`` whose entries span both frames' z (5..45).

    Rejects a first-layer domain: the legend would list either frame's
    entries alone.
    """
    legends = legend_set(_quiet(build).to_svg())
    assert list(legends) == ["z"]
    entries = legends["z"]
    assert entries != legend_set(P(size="z").to_svg())["z"]
    assert entries != legend_set(P(df2, size="z").to_svg())["z"]
    assert float(entries[0]) <= df["z"].min() and float(entries[-1]) >= 0.8 * df["z"].max()


def test_V10_an_opacity_on_both_layers_draws_one_legend():
    titles = list(legend_set(_quiet(lambda: P(opacity="z") + P(df2, opacity="z")).to_svg()))
    assert titles == ["z"]


def test_V10_a_type_conflict_excludes_the_later_layer_with_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = (P(color="z") + line(color="g")).to_svg()
    messages = [str(w.message) for w in caught]
    assert any("stays out of that scale" in m for m in messages), messages
    assert list(legend_set(svg)) == ["z"]


# ---------------------------------------------------------------------------
# V1, V10: the opacity family and stroke_width draw one legend on every door
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
def test_V10_an_opacity_or_width_binding_draws_one_legend_alike_on_every_door(case):
    """V1/V10: every door draws the flat chart's one legend, static and interactive.

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
def test_V10_an_explicit_domain_and_range_reach_the_wire_and_set_the_legend(channel):
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
def test_V10_two_layers_draw_one_legend_spanning_both_extents(channel, order):
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
def test_V10_a_string_layer_is_excluded_and_one_legend_stays(channel):
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


_MALFORMED_DOOR_SCALES: dict[str, tuple[dict, str]] = {
    "log": ({"type": "log"}, "type 'log' is not supported"),
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


_V11: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_V11))
def test_V11_an_aggregated_legend_is_titled_by_its_output_column(case):
    """V11: the color legend is titled ``mean_z`` on every door, same frame or not.

    Rejects the source-field title (``z``) a different-frame promotion copy drew.
    """
    case = _V11[case]
    check_rendering(case)
    assert list(legend_set(_quiet(_quiet(case.build).to_svg))) == ["mean_z"]


_V12: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_V12))
def test_V12_a_string_in_layer_encoding_draws_with_a_legend(case):
    """V12: ``encoding={"color": "c"}`` draws colored with a legend, as ``Color("c")`` does.

    Rejects a scale built only from the chart-level encoding: the string
    spelling drew no color.
    """
    check_rendering(_V12[case])


# ---------------------------------------------------------------------------
# G1, G3, G4, G6: grouping channels
# ---------------------------------------------------------------------------

_G1: dict[str, Case] = {
    **_cases("detail", plus_doors(lambda: line(detail="c"), lambda: line(y="r"))),
    **_cases("stroke_dash", plus_doors(lambda: line(stroke_dash="c"), lambda: line(y="r"))),
    **_cases("shape", plus_doors(lambda: P(shape="c"), lambda: P(y="r"))),
}


@pytest.mark.parametrize("case", sorted(_G1))
def test_G1_a_grouping_binding_never_splits_a_sibling(case):
    """G1: ``line(detail=c) + line`` = ``line + line(detail=c)`` = 3 polylines.

    Rejects D1-C: the left's detail would split the right line into 2.
    """
    check_rendering(_G1[case])


def test_G1_three_polylines_in_both_orders():
    for chart in (line(detail="c") + line(y="r"), line(y="r") + line(detail="c")):
        assert tag_counts(chart.to_svg())["polyline"] == 3


_G3: dict[str, Case] = {
    # The shape scale's second glyph is a square `<rect>`.
    "F": Case(lambda: agg("point", shape="c"), marks={"circle": 2, "rect": 2}),
    "L-inherits-the-hosts-shape": Case(
        lambda: agg("point", shape="c").layer(layer("tick")),
        (lambda: agg("point", shape="c"), lambda: agg("tick", shape="c")),
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_G3))
def test_G3_a_raw_grouping_channel_groups_an_aggregating_layer(case):
    """G3: flat 4 groups; an attached layer inheriting the host's ``shape`` groups by it too."""
    check_rendering(_G3[case])


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


def _g4_cases() -> dict[str, Case]:
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


_G4 = _g4_cases()


@pytest.mark.parametrize("case", sorted(_G4))
def test_G4_an_aggregating_sibling_keeps_its_grain_beside_a_raw_grouping_binding(case):
    """G4: the tick keeps 2 marks beside a raw detail/shape/stroke_dash/key/href/text/tooltip.

    Rejects D1-C: the tick would group by the sibling's raw field.
    """
    check_rendering(_G4[case])


_G6: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_G6))
def test_G6_an_attached_layers_own_grouping_channel_draws_with_a_legend(case):
    check_rendering(_G6[case])


# ---------------------------------------------------------------------------
# Parent rules (spec §4.1)
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
    """§4.1: a layer inherits its one parent's bindings whole, and nothing from a sibling.

    Rejects D1-C: ``(a + b).encode(detail="g") + c`` would split ``c``, and
    a LayerChart member would inherit an earlier member's composition color.
    """
    check_rendering(_PARENT[case])


# ---------------------------------------------------------------------------
# Axis fallback (spec §4.6 X1): the one cross-sibling flow
# ---------------------------------------------------------------------------


def _line_points() -> fm.Chart:
    """``mark_line(point=True)``: a line and a point layer, each binding no x/y of its own."""
    return fm.Chart(df).mark_line(point=True).encode(x="v", y="r")


_X1: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_X1))
def test_X1_a_layer_binding_no_axis_takes_layer_zeros_written_axis(case):
    check_rendering(_X1[case])


@pytest.mark.parametrize("order", ["right", "left"])
def test_X1_a_layered_operands_layers_draw_that_operands_written_axis(order):
    """X1: ``mark_line(point=True).encode(y="r")``'s two layers draw ``r`` beside ``P``, either order.

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
def test_X1_layer_zeros_written_axis_titles_the_axes(build, titles):
    assert axis_titles(build().to_svg()) == titles


def test_X1_a_drawn_fallback_column_the_layers_rows_lack_is_refused():
    """X1 +d reversed: the points take layer 0's ``x="w"``, which df lacks, so the chart refuses.

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
    # X2: an encoding aggregate, each layer over its own rows by its own groupby.
    "X2/F": Case(bar_mean, marks={"rect": 2}),
    "X2/L-own-sum": Case(
        lambda: bar_mean().layer(layer("point", y=fm.Y("y", aggregate="sum"))),
        (bar_mean, lambda: agg("point", y="sum(y)")),
        chart_level=None,
    ),
    "X2/L-raw-y": Case(
        lambda: bar_mean().layer(layer("point", y="y")),
        (bar_mean, lambda: P(x="g")),
        chart_level=None,
    ),
    "X2/L-finer-color": Case(
        lambda: bar_mean().layer(layer("point", color="c")),
        (bar_mean, lambda: agg("point", color="c")),
        chart_level=None,
    ),
    "X2/+s-right-binds-none": Case(
        lambda: bar_mean() + fm.Chart(df).mark_point(), (bar_mean, lambda: agg("point"))
    ),
    "X2/+d-right-binds-none": Case(
        lambda: bar_mean() + fm.Chart(df2).mark_point(), (bar_mean, lambda: agg("point", df2))
    ),
    "X2/+r": Case(lambda: P(x="g") + bar_mean(), (lambda: P(x="g"), bar_mean)),
    "X2/+r-diff": Case(lambda: P(df2, x="g") + bar_mean(), (lambda: P(df2, x="g"), bar_mean)),
    "X2/LCm": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g"))._build_merged(),
        (bar_mean, lambda: P(df2, x="g")),
    ),
    "X2/Fc": Case(lambda: bar_mean().facet("c"), marks={"rect": 4}),
    # X9: composite statistics per layer; attached layers read the host's rows.
    "X9/H-layer-rug": Case(
        lambda: _hist().layer(layer("tick")),
        (_hist, lambda: mk("tick", x="v")),
        marks={"rect": 4, "line": 8},
        chart_level=None,
    ),
    "X9/H+rug": Case(lambda: _hist() + mk("tick", x="v"), marks={"rect": 4, "line": 8}),
    "X9/rug+H": Case(lambda: mk("tick", x="v") + _hist(), marks={"line": 8, "rect": 4}),
    # X10: a user transform; attached layers read the host chain's output.
    # `datum.v > 3` keeps 4 of df's rows.
    "X10/L": Case(
        lambda: P().transform(fm.transform_filter("datum.v > 3")).layer(layer("tick")),
        marks={"circle": 4, "line": 4},
        chart_level=None,
    ),
    "X10/LD": Case(
        lambda: (
            P().transform(fm.transform_filter("datum.v > 3")).layer(fm.Layer(data=df2, mark="tick"))
        ),
        marks={"circle": 4, "line": 6},
    ),
    "X10/+e": Case(
        lambda: (P() + line(df2)).transform(fm.transform_filter("datum.v > 3")),
        marks={"circle": 4, "polyline": 1},
    ),
    # S1: a span with no statistic, read per layer.
    "S1/F": Case(lambda: mk("bar", x="g", y="y", y2="r"), marks={"rect": 8}),
    "S1/L-inherits-y2": Case(
        lambda: mk("bar", x="v", y="y", y2="r").layer(layer("tick")),
        marks={"rect": 8, "line": 8},
        chart_level=None,
    ),
    # S3: a binned channel gets its span; S4: a user transform's span.
    "S3/F": Case(_binned, marks={"rect": 4}),
    "S3/L-inherits-the-bin": Case(
        lambda: _binned().layer(layer("point")),
        (_binned, lambda: mk("point", x=fm.X("v", bin=True), y="count()")),
        chart_level=None,
    ),
    "S4/F": Case(
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
    # F1: facets partition the source; every statistic groups by them.
    "F1/L": Case(
        lambda: bar_mean().layer(layer("text", text=fm.Text("y", aggregate="mean"))).facet("c"),
        marks={"rect": 4, "text": 4},
        chart_level=None,
    ),
    "F1/SY": Case(
        lambda: (bar_mean() + SY("r")).facet("c"),
        marks={"rect": 4, "polyline": 2},
        chart_level=None,
    ),
    # P1, P3: polar.
    "P1/F": Case(lambda: mk("arc", theta="z", color="g").coord(fm.CoordPolar()), marks={"path": 8}),
    "P3/F": Case(
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
    # P6: the errorbar composite, per layer.
    "P6/+point": Case(
        lambda: mk("errorbar", x="g", y="y") + agg("point"),
        (lambda: mk("errorbar", x="g", y="y"), lambda: agg("point")),
    ),
    "P6/L-point": Case(
        lambda: mk("errorbar", x="g", y="y").layer(layer("point")), chart_level=None
    ),
}


@pytest.mark.parametrize("case", sorted(_NET))
def test_regression_net(case):
    check_rendering(_NET[case])


def test_X3_naming_another_layers_statistic_is_refused_with_the_statistic_remedy():
    with pytest.raises(ValueError, match=r"statistic"):
        bar_mean().layer(layer("text", text="mean_y")).to_svg()
    with pytest.raises(ValueError, match=r"statistic the base chart's encoding computes"):
        bar_mean() + SY("mean_y")
    with pytest.raises(ValueError, match=r"statistic"):
        _hist().layer(layer("tick", y="count")).to_svg()
    with pytest.raises(ValueError, match=r"statistic the base chart's mark_histogram computes"):
        _hist() + SY("count")


def test_F2_a_facet_on_a_right_operand_is_warned_and_ignored():
    with pytest.warns(UserWarning, match="secondary layer's theme/facet is ignored"):
        chart = P() + P(df2, y="w").facet("g")
    assert len(_clip_groups(_svg_root(chart.to_svg()))) == 1


def test_F4_an_aggregate_on_a_facet_channel_is_refused():
    with pytest.raises(ValueError) as exc:
        fm.Facet("c", aggregate="mean")
    assert str(exc.value).startswith("Facet(aggregate='mean') is not supported")


def test_P5_an_error_channel_on_encode_is_warned_and_omitted():
    with pytest.warns(UserWarning):
        svg = P(y_error="r").to_svg()
    assert tag_counts(svg) == {"circle": 8}


# ---------------------------------------------------------------------------
# A1: a different-frame operand reads only its own rows (spec §4.5, §5 A1)
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


_F3: dict[str, Case] = {
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


@pytest.mark.parametrize("case", sorted(_F3))
def test_F3_a_data_layer_inheriting_an_aggregate_draws_its_own_rows_means(case):
    """F3: the data layer averages its own rows by ``g``: 4 bars and 3 points faceted, nothing warned.

    Rejects reading the combined rows: the faceted data layer grouped by
    the renamed ``g`` copy, null on every row of the host's frame, and each
    panel warned that every row had a null in its groupby key.
    """
    check_rendering(_F3[case])
    assert _null_group_warnings(_F3[case].build) == []


def test_F3_the_faceted_data_layer_draws_its_means_in_its_own_panel():
    """F3 faceted: panel ``p`` draws the host's 2 bars and the data layer's 3 points; ``q`` the bars."""
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


def test_F3_a_data_layer_whose_frame_lacks_the_facet_field_repeats_in_every_panel():
    """F3 over the spec's df2, which has no ``c``: each panel draws 2 bars and 3 points, nothing warned.

    The data layer's ``mean(y)`` by ``g`` runs on its own 6 rows in every
    panel, as ggplot2 draws a layer whose data lacks the faceting variable
    (ruling 2026-10-17). Rejects the silent drop: the 3 points were drawn in
    no panel.
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
_M1_PAIRS: dict[str, tuple[Callable[[], fm.Chart], Callable[[], fm.Chart]]] = {
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


@pytest.mark.parametrize("case", sorted(_M1_PAIRS))
def test_M1_a_data_layer_draws_what_the_plus_it_lowers_to_draws(case):
    """M1 control: ``Layer(data=df2)`` over a different frame renders the bytes of its ``+``."""
    layered, plus = _M1_PAIRS[case]
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


_OVERLAP = _overlap_cases()


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


@pytest.mark.parametrize("case", sorted(_OVERLAP))
def test_A1_an_operand_sharing_column_names_reads_only_its_own_rows(case):
    """A1: each operand of ``P(df) + line(df2)`` draws its own rows alone, in both orders.

    No layer binding, groupby, legend, axis title or tooltip names a
    ``__rhs_`` column. Rejects reading the combined rows under renamed
    fields: the right layer bound ``v__rhs_0001``, and a faceted or
    aggregating right layer grouped by a column null on the left's rows.
    """
    built = _OVERLAP[case]
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
def test_V4_a_right_binding_on_a_shared_column_name_is_titled_by_its_field(case):
    """V4: ``P(df) + P(df2, size="z")`` draws one legend titled ``z``, with no title on the wire.

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
def test_V4_an_explicit_tooltip_on_a_shared_column_name_shows_its_field(order):
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

# The SVG bytes each spec §7 control renders; routing every different-frame
# operand through its own rows leaves them unchanged.
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
# X11: a SecondaryY on a composition reads the left operand's frame (spec §4.5)
# ---------------------------------------------------------------------------

_SECONDARY_W_REFUSAL = (
    "SecondaryY: field 'w' is not a column of the base chart's data (SecondaryY draws "
    "from the same table as the base chart); available columns: "
    "['v', 'y', 'z', 'r', 'g', 'c', 'k']"
)


@pytest.mark.parametrize("left", [P, bar_mean], ids=["raw", "aggregating"])
def test_X11_a_secondary_y_naming_a_right_only_column_is_refused_at_construction(left):
    """X11: ``(P(df) + P(df2, y="w")) + SY("w")`` raises the ``SecondaryY`` column message.

    Rejects reading the composition's combined frame, which holds the right
    operand's ``w``: the secondary layer drew nothing, silently.
    """
    composition = _quiet(lambda: left() + P(df2, y="w"))
    with pytest.raises(ValueError) as refused:
        composition + SY("w")
    assert str(refused.value) == _SECONDARY_W_REFUSAL


_W_ABSENT = "layer 2: encoding 'y' references unknown column 'w' (not present in that layer's data)"


def test_X11_a_secondary_y_on_a_left_chain_lacking_the_column_is_refused_at_render():
    """X11, left with a chain: the secondary layer's rows are the left's, which have no ``w``.

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


_X11: dict[str, tuple[Case, list[int]]] = {
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


@pytest.mark.parametrize("case", sorted(_X11))
def test_X11_a_secondary_y_on_a_composition_draws_the_left_operands_rows(case):
    """X11: the secondary line draws one vertex per row of the left operand's frame."""
    built, vertices = _X11[case]
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
# V10 on the static overlay: one legend per channel, as the merged chart draws
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
def test_V10_the_static_overlay_draws_one_legend_over_both_members(channel):
    """V10 LCs: two members binding one channel over df and df2 draw one legend, the merged chart's.

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


def interactive_legend_set(chart: Any) -> dict[str, tuple[str, ...]]:
    """Each interactive legend's title and its entry labels, in order."""
    legends: dict[str, list[str]] = {}
    title: str | None = None
    for node in _scene(chart)["legend"]:
        if node["type"] != "text":
            continue
        if (node.get("style") or {}).get("font_weight") == {"custom": "600"}:
            title = node["content"]
            legends[title] = []
        elif title is not None:
            legends[title].append(node["content"])
    return {t: tuple(entries) for t, entries in legends.items()}


# ---------------------------------------------------------------------------
# A layer takes an inherited positional channel only where its mark draws it
# ---------------------------------------------------------------------------
#
# Ruling 2026-10-17: a layer takes an inherited x, y, x2 or y2 only on a
# channel its mark draws (Rust's one per-mark table,
# ``ferrum._core.mark_inherited_positional``), and a drawn channel whose
# column the layer's rows lack is refused.


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
    assert [s.get("groupby") for s in _wire(chart)["transforms"] if s["type"] == "aggregate"] == [
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
    """F2: ``P + P(df2).facet("g")`` warns that the right operand's facet is ignored."""
    with pytest.warns(UserWarning, match="secondary layer's theme/facet is ignored"):
        (P() + P(df2).facet("g")).to_svg()


# ---------------------------------------------------------------------------
# Current behavior pinned for a later task (not the target)
# ---------------------------------------------------------------------------
#
# These pin what the built engine draws today for three cases the model
# wants refused or drawn differently. They state the present output plainly
# so a change to it is seen; the target belongs to a later task.


@pytest.mark.parametrize(
    ("build", "marks", "legends"),
    [
        (lambda: P(x="w") + P(df2), {"circle": 6}, {}),
        (lambda: P(color="w") + P(df2), {"circle": 14}, {"w": ("1", "2.25", "3.5", "4.75", "6")}),
        (
            lambda: fm.LayerChart(P(color="w"), P(df2))._build_merged(),
            {"circle": 14},
            {"w": ("1", "2.25", "3.5", "4.75", "6")},
        ),
    ],
    ids=["layer0-x-other-frame", "color-left", "LCm-color"],
)
def test_today_a_binding_to_a_column_only_the_other_frame_has_still_draws(build, marks, legends):
    """Today: layer 0 bound to a column only the other operand's frame holds is not refused.

    Layer 0's ``x="w"`` draws none of the left's points (6 circles are df2's),
    and its color on ``w`` draws a legend read from the other frame's rows.
    The model refuses them; layer 0 is exempt from the column gate today and
    the engine reads the field outside its own batch. A later layer's
    binding is refused (cell A7 below).
    """
    svg = _quiet(lambda: build().to_svg())
    assert tag_counts(svg) == marks
    assert legend_set(svg) == legends


# ---------------------------------------------------------------------------
# A7: the column gate covers every channel a layer's mark draws
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


_A7_REFUSALS: dict[str, tuple[Callable[[], Any], str]] = {
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
@pytest.mark.parametrize("case", sorted(_A7_REFUSALS))
def test_A7_a_drawn_binding_its_layers_rows_lack_is_refused(case, kind):
    """A7: a later layer's drawn binding to a column its own rows lack names the layer and channel.

    The color cells reject reading the other frame's column (``P(df2) +
    P(color="w")`` drew 14 circles under a ``w`` legend); the host cells
    reject drawing the line uncolored under the host's legend; a host
    conditional is refused on its own channel, not on the selection's
    tooltip, because the selection field never reaches a layer whose rows
    lack it.
    """
    build, message = _A7_REFUSALS[case]
    with pytest.raises(ValueError) as refused:
        chart = build()
        chart.to_svg() if kind == "static" else chart.interactive().to_html()
    assert str(refused.value) == message


# The SVG each rendering cell drew before the gate covered every channel:
# the gate refuses none of them.
_A7_RENDER_DIGESTS = {
    "selection-only": "9d910acf614a7f9c5c5c000fc2d6d525a6a95d475dd1060ac1328bd928beb313",
    "value-conditional": "9d910acf614a7f9c5c5c000fc2d6d525a6a95d475dd1060ac1328bd928beb313",
    "selection-plus": "9d910acf614a7f9c5c5c000fc2d6d525a6a95d475dd1060ac1328bd928beb313",
    "undrawn-host-shape": "bbc7316b39bcbcb990493cf76ec22e5c99c53ec7f41c8decf05df5b6faeec9e8",
}

_A7_RENDERS: dict[str, Callable[[], Any]] = {
    "selection-only": lambda: _over_df2(P().add_selection(_PICK_C)),
    "value-conditional": lambda: _over_df2(
        P(color=_PICK_C.when(fm.value("#d62728")).otherwise(fm.value("#cccccc"))).add_selection(
            _PICK_C
        )
    ),
    "selection-plus": lambda: P().add_selection(_PICK_C) + line(df2),
    "undrawn-host-shape": lambda: _over_df2(P(shape="c")),
}


@pytest.mark.parametrize("case", sorted(_A7_RENDERS))
def test_A7_a_layer_whose_rows_lack_only_an_unread_field_renders(case):
    """A7: a selection field, or an undrawn binding, its rows lack never refuses a layer.

    A field selection's ``c`` joins the host's tooltip but not the df2
    line's, whose rows lack it, and a line never reads the host's shape. Each
    renders static and interactive, the SVG unchanged, and the line's
    interactive tooltip lists only its own drawn fields. Rejects carrying
    the selection field (or the undrawn shape's column) into the line's
    tooltip, which the gate refuses.
    """
    import hashlib

    svg = _A7_RENDERS[case]().to_svg()
    assert hashlib.sha256(svg.encode()).hexdigest() == _A7_RENDER_DIGESTS[case]
    host, data_line = interactive_tooltip_fields(_A7_RENDERS[case]())
    assert data_line == {"v", "y"}
    assert "c" in host or case == "undrawn-host-shape"


def test_A7_a_selection_field_is_never_written_into_a_tooltip_binding():
    """The spec names no selection field in any tooltip; the scene lists ``c`` on the host only.

    A field selection's fields are match keys the render's tooltip reader
    appends where the rows hold them, so the column gate never judges them.
    """
    chart = _A7_RENDERS["selection-plus"]()
    wire = _wire(chart)
    assert "tooltip_fields" not in wire["encoding"]
    assert all("tooltip_fields" not in layer["encoding"] for layer in wire["layers"])
    host, data_line = interactive_tooltip_fields(chart)
    assert "c" in host and "c" not in data_line


def test_A7_a_selection_field_reaches_the_frame_of_a_chain_of_merges_that_carries_it():
    """In ``(P + P(df2)) + P(df3-like)``, the selection's ``c`` stays off df2's layer only."""
    with_c = df2.with_columns(c=pl.Series(["p", "q"] * 3))
    chart = (P().add_selection(_PICK_C) + P(df2)) + P(with_c)
    assert [names for names in interactive_tooltip_fields(chart)] == [
        {"c", "v", "y"},
        {"v", "y"},
        {"c", "v", "y"},
    ]
    chart.to_svg()


def test_A7_a_layer_keeps_the_selection_fields_its_rows_carry():
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
def test_A7_a_layers_own_undrawn_binding_stays_out_of_its_auto_tooltip(build, line_fields):
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
    return _W_CAT.with_columns(c=pl.Series(["p", "q", "q", "p"] * 15))


# Per family: the chart, and the mark kinds of the batches whose rows carry ``c``.
_SELECTION_FAMILIES: dict[str, tuple[Callable[[], fm.Chart], list[str]]] = {
    "boxen": (lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z"), []),
    "boxplot": (lambda: fm.Chart(df).mark_boxplot().encode(x="g", y="z"), []),
    "violin": (lambda: fm.Chart(df).mark_violin().encode(x="g", y="z"), []),
    "mark_qq": (lambda: fm.Chart(df).mark_qq().encode(x="z"), []),
    "mark_qq-color": (lambda: fm.Chart(df).mark_qq().encode(x="z", color="c"), []),
    "catplot-point": (lambda: fm.catplot(_sel_cat(), x="cat", y="val", kind="point"), []),
    "catplot-boxen-hue": (
        lambda: fm.catplot(_sel_cat(), x="cat", y="val", kind="boxen", hue="grp"),
        [],
    ),
    "lmplot-color": (
        lambda: fm.lmplot(_sel_cat(), x="x", y="y", encode={"color": "c"}),
        ["point"],
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


_A7_C7_BUILDS: dict[str, Callable[[], Any]] = {
    "+": lambda: bar_mean(color=mean(fm.Color)) + mk("tick", df2, color="g"),
    "L": lambda: bar_mean(color=mean(fm.Color)).layer(
        fm.Layer(data=df2, mark="tick", encoding={"color": "g"})
    ),
}


@pytest.mark.parametrize("kind", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_A7_C7_BUILDS))
def test_A7_a_tick_left_out_of_a_numeric_color_scale_draws_with_one_warning(door, kind):
    """A string-colored tick beside a mean-colored bar stays out of the scale, as a point does.

    Rejects the tick's color reader refusing the column's dtype
    (``column 'g' has unsupported dtype: Utf8``).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = _A7_C7_BUILDS[door]()
        chart.to_svg() if kind == "static" else chart.interactive().scene_json
    excluded = [str(w.message) for w in caught if "stays out of that scale" in str(w.message)]
    assert excluded == [
        "layer 1: its color scale differs from the one layer 0 establishes the color scale "
        "with, so it stays out of that scale; its marks draw through that scale and may fall "
        "outside its legend"
    ]


# ---------------------------------------------------------------------------
# A7, user ruling W (2026-10-17): a statistic sub-layer drops an inherited
# channel its statistic does not carry, with a warning
# ---------------------------------------------------------------------------

_W_RNG = np.random.default_rng(0)
_W_CAT = pl.DataFrame(
    {
        "cat": ["a", "b", "c"] * 20,
        "grp": ["X"] * 30 + ["Y"] * 30,
        "val": _W_RNG.normal(0, 1, 60),
        "x": _W_RNG.normal(0, 1, 60),
        "y": _W_RNG.normal(0, 1, 60),
    }
)


def _dropped(entry: str, mark: str, channel: str, field: str) -> str:
    """The warning, voiced by the call that built the layer (its internal index is not named)."""
    return (
        f"{entry}: {channel} '{field}' was dropped: the statistic this {mark} "
        f"mark draws does not carry '{field}', so it draws without {channel}"
    )


_W_CASES: dict[str, tuple[Callable[[], Any], list[str], str]] = {
    "mark_boxen-color": (
        lambda: fm.Chart(df).mark_boxen().encode(x="g", y="z", color="c"),
        [_dropped("mark_boxen", "point", "color", "c")],
        "03f2fae70d706aafac90797b57e800c7d5cf026ab1b7bf16a9d81de2fa5b76c0",
    ),
    "catplot-boxen-hue": (
        lambda: fm.catplot(_W_CAT, x="cat", y="val", kind="boxen", hue="grp"),
        [_dropped("catplot", "point", "color", "grp")],
        "f102bc411159b027438aee275985b550c5ae47eb7797d8d85e8ad045db17a1d1",
    ),
    "mark_qq-color": (
        lambda: fm.Chart(df).mark_qq().encode(x="z", color="c"),
        [_dropped("mark_qq", "rule", "color", "c")],
        "cb1a6d003edd2bedc566a85d310eb266430b00e0143ea0d5681cede1af393827",
    ),
    "lmplot-encode-color": (
        lambda: fm.lmplot(_W_CAT, x="x", y="y", encode={"color": "cat"}),
        [
            _dropped("lmplot", "ribbon", "color", "cat"),
            _dropped("lmplot", "line", "color", "cat"),
        ],
        "e77ac1f870b76a33fc8230b63c43fb4d7c69444b4b2197636496a7abd6faa46e",
    ),
    "catplot-point-tooltip": (
        lambda: fm.catplot(_W_CAT, x="cat", y="val", kind="point", tooltip="grp"),
        [_dropped("catplot", "point", "tooltip", "grp")],
        "d6002ccc9dd6882e00677cb808ef3fe957791e384381c9fab34961d2698a2dd6",
    ),
}


@pytest.mark.parametrize("case", sorted(_W_CASES))
def test_A7_W_a_statistic_layer_draws_without_an_inherited_channel_it_dropped(case):
    """Ruling W: the statistic sub-layer warns and draws the SVG it drew before the gate.

    Rejects refusing a call whose column is in the user's data (the
    statistic's output, not the user, dropped it), and rejects dropping it
    silently.
    """
    import hashlib

    build, expected, digest = _W_CASES[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
    assert [str(w.message) for w in caught if "was dropped" in str(w.message)] == expected
    assert hashlib.sha256(svg.encode()).hexdigest() == digest


def test_A7_W_a_raw_rows_layer_still_refuses_an_inherited_column_its_rows_lack():
    """The control: a ``Layer(data=)`` line reads the user's rows, so the host's color refuses."""
    with pytest.raises(ValueError) as refused:
        _over_df2(P(color="c")).to_svg()
    assert str(refused.value) == _absent("color", "c")


def test_today_a_secondary_y_on_a_left_chain_names_the_statistic_remedy():
    """Today: X11 with a left chain refuses with the statistic remedy, not the plain message.

    The right operand's own rows hold ``w``, and the render gate's remedy
    search finds it there.
    """
    with pytest.raises(ValueError) as refused:
        ((_filtered(P()) + P(df2, y="w")) + SY("w")).to_svg()
    assert str(refused.value) == (
        "layer 2: encoding 'y' references unknown column 'w' (not present in that layer's data)"
        " — 'w' is a statistic another layer's mark computes for itself; compute it in the "
        "chart's .transform(...) (e.g. fm.Bin, fm.transform_aggregate) so every layer reads "
        "it, or draw it from its own chart with "
        'fm.LayerChart(base, other, resolve={"y": "independent"})'
    )


def test_today_the_static_overlay_takes_its_x_axis_from_a_first_member_drawing_no_x():
    """Today: ``LayerChart(rule(df2, y="w"), bar(df, x="g"))`` static draws a 0..1 numeric x axis.

    The overlay's axis chrome comes from its first member, which draws no x;
    ``+``, LCm and LCi draw the bars' a, b.
    """
    svg = _quiet(lambda: fm.LayerChart(mk("rule", df2, y="w"), mk("bar", x="g", y="y")).to_svg())
    assert _x_tick_labels(svg) == [
        "0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1",
    ]
    assert axis_titles(svg) == ["w"]


# ---------------------------------------------------------------------------
# The wire seam (spec §6.3), read off every matrix case
# ---------------------------------------------------------------------------

_MATRIX: dict[str, Case] = {
    **{f"V1/{k}": v for k, v in _V1.items()},
    **{f"V1-by-name/{k}": v for k, v in _V1_BY_NAME.items()},
    **{f"V2/{k}": v for k, v in _V2.items()},
    **{f"V3/{k}": v for k, v in _V3.items()},
    **{f"V4/{k}": v for k, v in _V4.items()},
    **{f"V5/{k}": v for k, v in _V5.items()},
    **{f"V6/{k}": v for k, v in _V6.items()},
    **{f"V12/{k}": v for k, v in _V12.items()},
    **{f"G1/{k}": v for k, v in _G1.items()},
    **{f"G3/{k}": v for k, v in _G3.items()},
    **{f"G4/{k}": v for k, v in _G4.items()},
    **{f"G6/{k}": v for k, v in _G6.items()},
    **{f"parent/{k}": v for k, v in _PARENT.items()},
    **{f"X1/{k}": v for k, v in _X1.items()},
    **{f"net/{k}": v for k, v in _NET.items()},
    **{f"F3/{k}": v for k, v in _F3.items()},
    **{f"A1/{k}": v for k, v in _OVERLAP.items()},
    **{f"V4-titled/{k}": v for k, v in _RENAMED_TITLES.items()},
    **{f"X11/{k}": v for k, (v, _vertices) in _X11.items()},
}


@pytest.mark.parametrize("case", sorted(_MATRIX))
def test_each_layers_completed_wire_encoding_draws_what_it_draws_alone(case):
    """§6.3: wire layer encoding over the chart-level encoding = the layer's effective encoding.

    Rejects D1-C on the wire: the chart-level encoding would carry an
    operand's own binding (or a promoted copy), completing a sibling with it.
    """
    check_wire_seam(_MATRIX[case])


# Every matrix case reached through ``fm.LayerChart``: the prefix its four routes share.
_LAYER_CHART_CASES = sorted(key[: -len("LCs")] for key in _MATRIX if key.endswith("/LCs"))


@pytest.mark.parametrize("prefix", _LAYER_CHART_CASES)
def test_the_static_overlay_draws_the_legends_every_layer_chart_route_draws(prefix):
    """Route parity (spec §7): LCs draws LCm's and LCi's legend set, and LCx shows it interactively.

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
    with pytest.raises(TypeError):
        P().encode(color=None)
    chart = (P() + line(y="r")).encode()
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

    The left operand's color stays on its own layer (spec §4.6 V1), so the label reads
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
# Byte-identical controls (spec §7): charts the model does not reach
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
}

# The SVG each control renders on the tree before the one-parent model.
_CONTROL_DIGESTS = {
    "F/point-color": "a4a1d3b7746458ddbe4983e33cee3886e90b208e4f2e5be9c8c12482da5d5dff",
    "F/bar-mean-color": "3d61496ff5e8aba078879d090bafeb1a3ab31cfd72eeb86cb39ec6c617c73aa0",
    "F/point-size-opacity": "0f9930eab400f34f690bbbb5d8086dfc2284512b0ef1dda6b586ff4aa37d38fa",
    "L/host-color-line": "99a33743ce01c54b26f67dbfaa45dc58f34b9e2e9514e28cdf236bccdf267f78",
    "L/host-bar-mean-label": "358f05dc6992c4be27f63b1e4b37ad17849bc09d4f4293cb0561867de0d7dc5c",
    "+/both-color": "99a33743ce01c54b26f67dbfaa45dc58f34b9e2e9514e28cdf236bccdf267f78",
    "+/neither": "ab618fcb0f56aa5dc88e8f28c6c11214824166ef199be81771863ad707e4f386",
    "+/both-size": "356a7ea8ac422ec1379f30507987f35540a7f3933ec32ba8140799c0c2a6496e",
    "LCs/neither": "7d0f2c91ecb949cb1b20bb61543b06c1f7223390b381e767977fb5b53293bc24",
}


@pytest.mark.parametrize("case", sorted(_CONTROLS))
def test_a_control_renders_its_pinned_bytes(case):
    import hashlib

    svg = _quiet(lambda: _CONTROLS[case]().to_svg())
    assert hashlib.sha256(svg.encode()).hexdigest() == _CONTROL_DIGESTS[case]


# ---------------------------------------------------------------------------
# An operand draws in a composition what it draws alone (spec §7 isolation)
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
    """A host whose aggregated color its attached text inherits and lowers."""
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
    """§7 isolation: a host's labels stay dark, its grey line whole, its line's width its own.

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
    """§7 isolation, interactive: each layer's auto tooltip fields are its fields alone.

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
    "host-label-left": (lambda: _host_label() + _rule(), [[], ["color"], []]),
    "host-label-right": (lambda: _rule() + _host_label(), [[], [], ["color"]]),
    "composed-label-in-a-later-plus": (
        lambda: _composed_label() + _rule(),
        [["color"], ["color"], []],
    ),
    # The data layer reads its own rows, so the host color names its own `g` from the chart level.
    "LD-color-by-name": (_LD["text-color-by-name"][0], [[], []]),
    "L-lowered-color": (_lowered_label, [[], ["color"]]),
    "own-color-beside-a-host-color": (
        lambda: P(color="c").layer(layer("line", color="g")) + _rule(),
        [[], [], []],
    ),
}


@pytest.mark.parametrize("case", sorted(_PROVENANCE))
def test_the_wire_lists_exactly_the_channels_a_layer_inherits(case):
    """§6.3 wire seam: ``inherited_channels`` names a carried parent binding and no own one."""
    build, expected = _PROVENANCE[case]
    layers = _wire(_quiet(build))["layers"]
    assert [layer_.get("inherited_channels", []) for layer_ in layers] == expected


# ---------------------------------------------------------------------------
# Regression net: the remaining doors of the consistent cells, at their pinned counts
# ---------------------------------------------------------------------------

_NET_DOORS: dict[str, Case] = {
    "X2/LCs": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g")), (bar_mean, lambda: P(df2, x="g"))
    ),
    "X2/LCi": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g", y="w"), resolve={"y": "independent"}),
        (bar_mean, lambda: P(df2, x="g", y="w")),
    ),
    "X2/LCx": Case(
        lambda: fm.LayerChart(bar_mean(), P(df2, x="g")),
        (bar_mean, lambda: P(df2, x="g")),
        static=False,
    ),
    "X9/LD": Case(
        lambda: _hist().layer(fm.Layer(data=df2, mark="tick", encoding={"x": "v"})),
        (_hist, lambda: mk("tick", df2, x="v")),
        marks={"rect": 4, "line": 6},
    ),
    "X9/+d": Case(
        lambda: _hist() + mk("tick", df2, x="v"),
        (_hist, lambda: mk("tick", df2, x="v")),
        marks={"rect": 4, "line": 6},
    ),
    "X9/+d-rev": Case(
        lambda: mk("tick", df2, x="v") + _hist(),
        (lambda: mk("tick", df2, x="v"), _hist),
        marks={"line": 6, "rect": 4},
    ),
    "X9/SY": Case(lambda: _hist() + SY("r"), marks={"rect": 4, "polyline": 1}, chart_level=None),
    "X9/LCs": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v")),
        (_hist, lambda: mk("tick", df2, x="v")),
    ),
    "X9/LCm": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v"))._build_merged(),
        (_hist, lambda: mk("tick", df2, x="v")),
    ),
    "X9/LCi": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v", y="w"), resolve={"y": "independent"}),
        marks={"rect": 4, "line": 6},
    ),
    "X9/LCx": Case(
        lambda: fm.LayerChart(_hist(), mk("tick", df2, x="v")),
        (_hist, lambda: mk("tick", df2, x="v")),
        static=False,
    ),
    "X9/Fc": Case(
        lambda: _hist().layer(layer("tick")).facet("c"),
        (lambda: _hist().facet("c"), lambda: mk("tick", x="v").facet("c")),
        faceted=True,
        chart_level=None,
    ),
    "S1/+d-right-binds-none": Case(
        lambda: mk("rule", x="v", y="y", y2="z") + fm.Chart(df2).mark_rule(),
        (lambda: mk("rule", x="v", y="y", y2="z"), lambda: mk("rule", df2, x="v", y="y", y2="z")),
        marks={"line": 14},
    ),
    "F1/F": Case(lambda: P().facet("c"), marks={"circle": 8}),
    "F1/LD": Case(
        lambda: P().facet("g").layer(fm.Layer(data=df2, mark="tick")),
        marks={"circle": 8, "line": 6},
    ),
    "F1/+d": Case(
        lambda: (P() + line(df2)).facet("g"),
        (lambda: P().facet("g"), lambda: line(df2).facet("g")),
        marks={"circle": 8, "polyline": 3},
    ),
    "F1/+d-rev": Case(
        lambda: (line(df2) + P()).facet("g"),
        marks={"circle": 8, "polyline": 3},
    ),
    "P6/SY": Case(
        lambda: mk("errorbar", x="g", y="y") + SY("r"),
        marks={"line": 6, "polyline": 1},
        chart_level=None,
    ),
}


@pytest.mark.parametrize("case", sorted(_NET_DOORS))
def test_regression_net_doors(case):
    check_rendering(_NET_DOORS[case])


def test_X2_a_layer_zero_binding_no_axis_is_refused_beside_an_aggregate():
    """X2 +rev, right binds none reversed: layer 0 owns the axes, so binding none it is refused."""
    with pytest.raises(ValueError, match=r"^layer 0: mark_point needs both positional channels"):
        (fm.Chart(df).mark_point() + bar_mean()).to_svg()


@pytest.mark.parametrize("channel", ["size", "opacity", "fill_opacity", "stroke_opacity"])
def test_a_later_layers_string_column_first_drawer_refuses_as_the_flat_chart_does(channel):
    """V10: a later layer first drawing a numeric-only channel from a string column refuses alike."""
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
# A right operand's attached layer keeps its host's written axis (spec §4.1)
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
def test_X1_a_right_operands_attached_layer_draws_its_hosts_written_axis(host, door):
    """X1 / §4.1: the tick attached to a right operand draws at its host's ``x``, as alone.

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
# Figure functions composed with ``+`` draw as before
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

# The SVG each figure renders on the tree before its scatter and fit (or
# cells and labels) composed through ``+``.
_FIGURE_DIGESTS = {
    "lmplot/plain": "518c81c4c53bf1c9fadb0bc97d7adb390a5c97221c96391088f4c269286a8a75",
    "lmplot/hue": "f3ac385b78b66370f3426db7f3ad73f3d7d957de5c19fafbeadb2e6f3d050a55",
    "lmplot/hue-size": "63dd4f59a57794f3befcc121b50dbfcfc195b3829ca50b535a47b112c327129a",
    "lmplot/loess": "2fad5879a05738cd86da4128b4386309143f506ea73b0ffc6cca186e10f7107c",
    "lmplot/order2": "048830ead8c0029e013785a496e99abfd1cb89eed2084449c98f4ad498c8b961",
    "lmplot/x_bins": "d9ad14d881455d715013c30a08c88de1977cba1bb56f2aaef6519110da44e359",
    "lmplot/scatter-false": "e931ecef99739b21d5e19843ab325d867a1bbaabfbf596b1b1b4a4197134c183",
    "lmplot/robust-hue": "28b56d57025483f9f5b7654e08131ed2b22ca33fbd8590708de092e518c559ea",
    "lmplot/x_bins-hue": "9aee24538af3407953e91bac6949e91a21300421dcfcdc7cf2523aaaadbe979a",
    "regplot/hue": "f3ac385b78b66370f3426db7f3ad73f3d7d957de5c19fafbeadb2e6f3d050a55",
    "lmplot/col-hue": "4760e91c0648f68bbe436de48d991032fd31d992aebb5d04b720b374d8c1c104",
    "heatmap/annot": "f085f1007f5f446e1d40a4a0baa2c8151afa02272e91e0d4a4808f8b564c14ca",
    "heatmap/annot-id": "0ab65abfe968531c7a803ece1df1f48d4733b91f57c48143e5dd72b2cc489e86",
}


# The interactive scene each hue figure draws on the tree before its fit
# bound its hue itself: the fit's tooltips list the fields they listed then.
_FIGURE_SCENE_DIGESTS = {
    "lmplot/hue": "de4baef90f117d8f893cd369c14229890eef23897bf8d9277e5017b80bd169cb",
    "lmplot/hue-size": "75235c3490d8f87598cfbc19a2cb3835c4bec5c72940cf473f29b05ac40206ef",
    "lmplot/robust-hue": "8f49dd45c5ca48189db721f6da06b62135f1d647b7a7a1ad4f9cb7d4f9be2b73",
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
