"""`#119` matrix: a layer's style is its own.

Today `Chart.__add__` clones the LHS (`chart.py:2115`), so layer 0's own mark
kwargs also sit on the *composed chart*. `Chart.to_spec` (`chart.py:3195`,
`if mk: kw["mark_style"] = mk`) emits that hoisted value as a chart-level
`mark_style` whenever it is non-empty, layered or not. Rust's layered
fallback (`LayerPrepared::from_chart_and_layer`,
`render/prepare/mod.rs:206-209`) then applies it to any SIBLING layer that
set no kwargs of its own, stripping only `fill`/`stroke`. Every other
`MarkKwargsSpec` field cascades wholesale into a layer that never asked for
it -- a `mark_point(opacity=0.3, stroke_width=5) + mark_line()` draws the
line translucent and 5px wide; `mark_function(f, clip=False) + mark_point()`
lets the points escape the panel clip; a `LayerChart` renders differently on
`.interactive()` than on `to_svg()`.

The fix is a single producer-side gate: `Chart.to_spec` emits the chart-level
`mark_style` only for a FLAT chart (`resolved._layers is None`). Every
layered producer (`+`, `LayerChart`, independent-y, `SecondaryY`, and the two
desugar-prior-layer shapes -- `.mark_smooth()`'s single append and
`.mark_function()`'s multi-layer append) routes through this one `to_spec`
call, so gating it there closes the leak everywhere at once. The Rust half
(`crates/ferrum-core/src/spec/mark_style.rs`'s `MarkKwargsSpec::without_paint`
and its consumer in `render/prepare/mod.rs`) deletes the now-dead per-layer
fallback outright, so the built extension this module runs against no longer
has ANY chart-level-to-layer fallback at all -- see "Revert-provability" below
for what that means for this module's RED proof. This module is the
RED-first proof, per the wave's binding "matrix first" rule.

**Population, derived, not hand-listed.** `_rust_non_paint_fields()` reads
`MarkKwargsSpec`'s fields from the Rust-published
`ferrum._core.mark_kwargs_fields()` (serde's own field list, in declaration
order) and drops `fill`/`stroke` (paint,
already covered by the batch-A `without_paint` strip and out of `#119`'s
scope). `_primitive_reachable_fields()` intersects that set with the kwargs
some primitive `mark_*()` method honors (`ferrum.marks.base.honored_kwargs`,
read off Rust's style table) -- this is "every non-paint field reachable
from a mark kwarg" per the brief. It excludes the fields only ferrum's own
code writes (`clip`, `detail`, `value_axis`), which every `mark_*()` refuses,
and `zero`, which `mark_bar` consumes into the chart's zero-anchor rather
than its style. `_UNPROBED` names, with the reason, the reachable fields a
one-row probe cannot draw. `test_derived_population_matches_probe_table`
fails loudly if a future `MarkKwargsSpec` field is reachable through a mark
kwarg but has no row here.

**Path coverage is the full population everywhere a mark-compatibility fact
does not narrow it.** All seven §4.8 producers are covered, each of the four
that has a genuine interactive form split into its own static/interactive
cell (twelve (path, output-kind) entries in `PATHS`/`PATH_FIELDS`): `+`
static/interactive, `LayerChart` static/interactive, independent-y
static/interactive, `SecondaryY` static/interactive, and the two
desugar-prior-layer shapes (`.mark_smooth()`, `.mark_function()`
multi-layer), each static/interactive. `+`, `LayerChart` and independent-y
place no constraint on the reader mark, so all 24 fields apply.
`SecondaryY`'s `mark=` is validated against `ferrum._layer._FLAT_MARK_METHODS`
(`point, line, bar, area, rule, text, tick, rect`), so its field population
(`SECONDARY_Y_FIELDS`) is the same set intersected with that whitelist --
derived from the SAME constant `SecondaryY`'s own guard reads, not
hand-picked -- excluding only the `segment`/`label`-reader fields (`arrow`,
`arrow_size`, `leader_line`). Both desugar-prior-layer shapes always append a
`line`-kind layer, so their population (`LINE_READER_FIELDS`) is the six
fields a line actually renders differently by (`opacity`, `stroke_width`,
`stroke_dash`, `stroke_cap`, `stroke_join`, `interpolate`) -- not a
representative sample of the 24, but the field population's exact
intersection with what a line-kind reader can discriminate at all; a
non-line-reader field has no cell on these two paths because there is no
render fact for it to prove there.

**Geometry-sensitive fields on margin-reserving paths.** `dx`, `dy` and
`band_size` read an absolute pixel measurement. `independent-y` and
`SecondaryY` narrow the plot area to reserve room for the second y-axis, so
even a correctly un-leaked position differs from a bare flat chart's on
those two paths alone -- not because anything cascaded, but because the
panel geometry itself differs. `_GEOMETRY_SENSITIVE_FIELDS` +
`_MARGIN_RESERVING_PATHS` name exactly this intersection; those cells'
baseline is rendered through the SAME path with LHS ALSO kwarg-less
(identical panel geometry, guaranteed no leak source) instead of a bare
flat render. Every other cell keeps the simpler flat-alone baseline.

**Discriminator design.** Each field probe is either "value" kind (extract
one attribute from the RHS's own rendered element -- e.g. a `<polyline>`'s
`stroke-width`, a scene node's `style.width` -- and compare the combined
chart's value against the RHS's OWN flat-alone value) or "count" kind (an
extra element/node the field's presence toggles -- e.g. an area's border
path, a segment's arrowhead -- compared against
`solo(LHS) + solo(RHS-flat)`, since a leaking chart draws that element
TWICE, once per layer, while a correctly-scoped one draws it only where a
layer actually asked for it). Every value-kind field carries a vacuity
control, in BOTH output kinds (`test_vacuity_control_rhs_with_its_own_kwarg_differs`,
parametrized over `svg`/`scene`): the RHS rendered with its own (different)
value for that kwarg must differ from flat, in the SAME extractor
(`svg_value` or `scene_value`) the matrix cell used, so an interactive
cell's `scene_value` cannot pass by accident of a constant discriminator.
Count-kind fields get the same, over `svg_count`/`scene_count`.

`test_lhs_retains_its_own_mark_style` closes a separate risk: an over-broad
gate that also stripped layer 0's own per-layer `mark_style` (not just the
chart-level hoist this task actually fixes) could still pass every
render-equality cell above for a field the RHS mark happens to ignore. It
reads the wire directly (`layers[0].mark_style`), once per field via `+`
rather than once per cell, because that dict is built by the SAME
`_Layer`/`_build_layers_list` mechanism on every producer this module
exercises and `#119`'s fix never touches it.

**Revert-provability.** The built `ferrum._core` extension this module runs
against already has the Rust half (`without_paint` deleted): Rust no longer
reads chart-level `mark_style` for ANY layered spec, leaked or not. File-copy
reverting `chart.py` to its pre-`#119` state therefore reproduces RED on
exactly the six wire-level "no top-level `mark_style`" pins (the Python-only
fact the revert changes) and nothing else -- every render-based cell in this
module stays green on both sides of that revert, because the fallback they
would have exercised is gone from the binary, not because the Python gate
stopped mattering. This is expected and is recorded in the task report, not
silently absorbed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import mark_kwargs_fields
from ferrum._layer import _FLAT_MARK_METHODS, drawn_style_fields
from ferrum._scene import _render_scene
from ferrum.marks.base import honored_kwargs

from _svg_rects import THEME_BACKGROUND, data_bar_rects


# ---------------------------------------------------------------------------
# Derived population (never hand-listed -- see module docstring).
# ---------------------------------------------------------------------------


def _rust_non_paint_fields() -> list[str]:
    """Every `MarkKwargsSpec` field (Rust-published, declaration order), minus paint."""
    fields = mark_kwargs_fields()
    assert fields, "ferrum._core.mark_kwargs_fields() published no fields"
    return [f for f in fields if f not in ("fill", "stroke")]


def _primitive_reachable_fields() -> set[str]:
    """Non-paint fields some primitive `mark_*()` method puts on its chart's style.

    Read from Rust's style table through `honored_kwargs`, over every
    `Chart.mark_<name>` method whose name is a primitive mark, so a field
    a mark starts or stops drawing changes this population automatically.
    `zero` is consumed by `mark_bar` into the chart's zero-anchor
    (`Chart._set_mark`), never its style.
    """
    method_marks = [
        name[len("mark_") :]
        for name in dir(fm.Chart)
        if name.startswith("mark_") and drawn_style_fields(name[len("mark_") :]) is not None
    ]
    honored = set().union(*(honored_kwargs(m) for m in method_marks))
    return {f for f in _rust_non_paint_fields() if f in honored} - {"zero"}


#: Reachable fields no one-row probe here can draw, each with the reason.
_UNPROBED = {
    "cmap": "an image draws it only over raster data, which only mark_raster produces",
    "width": "an image draws it only over a url encoding, which needs fetchable images",
    "height": "an image draws it only over a url encoding, which needs fetchable images",
    "radius": "a text draws it only when its stacked theta places it on a CoordPolar wedge, "
    "which the independent-y and SecondaryY paths (Cartesian only) cannot carry; its "
    "scoping on + and LayerChart is pinned in test_layer_encoding_model.py",
}

PRIMITIVE_REACHABLE_FIELDS = _primitive_reachable_fields() - set(_UNPROBED)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------

#: One row is enough to discriminate every attribute-value field below, and
#: keeps "the RHS's own element" unambiguous (exactly one glyph per layer).
ROW = pl.DataFrame(
    {
        "x": [1.0],
        "y": [2.0],
        "y2": [5.0],
        "x2": [3.0],
        "yseg": [4.0],
        "cat": ["a"],
        "label": ["ZZPROBE"],
    }
)
#: `interpolate` needs a real multi-point curve to show a step shape; `y2`
#: is here for the `SecondaryY` path, which needs a field on the base data.
CURVE = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 1.0], "y2": [3.0, 5.0, 2.0]})
#: Long enough that `limit=` visibly truncates it. `y2` is here for the same
#: `SecondaryY` reason `CURVE` carries it.
LONG_LABEL = pl.DataFrame({"x": [1.0], "y": [2.0], "y2": [5.0], "label": ["ZZPROBELONGTEXT"]})

_BLUE = "#2563eb"


def _svg(chart: Any) -> str:
    return chart.to_svg()


def _scene(chart: Any) -> dict:
    if hasattr(chart, "_render_interactive"):
        scene_json, _packed = chart._render_interactive()
    else:
        scene_json, _packed = _render_scene(chart)
    return json.loads(scene_json)


def _mark_chart(df: pl.DataFrame, mark: str, kwargs: dict, encoding: dict):
    chart = getattr(fm.Chart(df), f"mark_{mark}")(**kwargs)
    return chart.encode(**encoding)


def _clip_group_content(svg: str) -> str:
    """Every top-level plot clip-path group's content, depth-aware.

    A naive non-greedy `(.*?)` regex stops at the first nested `</g>` (e.g.
    the `stroke_cap`/`stroke_join` wrapper), truncating early -- this walks
    `<g ...>`/`</g>` tokens to find the REAL matching close.
    """
    out = []
    for start_m in re.finditer(r'<g clip-path="url\(#ferrum-clip-\d+\)">', svg):
        i = start_m.end()
        depth = 1
        for tok_m in re.finditer(r"<g[ >]|</g>", svg[i:]):
            depth += 1 if tok_m.group(0).startswith("<g") else -1
            if depth == 0:
                out.append(svg[i : i + tok_m.start()])
                break
    return "".join(out)


def _svg_tag_attrs(svg: str, tag: str) -> list[dict[str, str]]:
    return [
        dict(re.findall(r'([\w-]+)="([^"]+)"', m)) for m in re.findall(rf"<{tag}([^/]*)/>", svg)
    ]


def _last_data_glyph(svg: str) -> tuple[str, dict[str, str]]:
    """Last (tag, attrs) among data-drawn point glyphs (`circle`/`rect`/`path`), in document order.

    `path` covers the non-quadrilateral constant shapes (diamond, triangle,
    cross): `mark_point`'s only non-circle/non-square shapes render as a
    filled path, not a `<polygon>`.
    """
    candidates: list[tuple[int, str, dict[str, str]]] = []
    for tag in ("circle", "rect", "path"):
        for m in re.finditer(rf"<{tag}([^/]*)/>", svg):
            attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
            if tag == "rect":
                fill = attrs.get("fill")
                if not fill or fill == THEME_BACKGROUND or fill.startswith("url("):
                    continue
            candidates.append((m.start(), tag, attrs))
    candidates.sort(key=lambda c: c[0])
    _, tag, attrs = candidates[-1]
    return tag, attrs


def _last_curve_element(svg: str) -> str:
    """Tag name of the last data curve element (`polyline` or `path`), by document position.

    A leaked `interpolate` turns a `mark_line`'s output from `<polyline>`
    into a stepped `<path>` (Rust's step/step-before/step-after renderer
    uses path commands, not a point list) -- the tag itself is the
    discriminator, robust to which tag the (possibly-leaking) render used.
    """
    candidates = []
    for tag in ("polyline", "path"):
        candidates.extend((m.start(), tag) for m in re.finditer(rf"<{tag}[^>]*/>", svg))
    candidates.sort()
    return candidates[-1][1]


def _last_tick_line(svg: str) -> dict[str, str]:
    ticks = [a for a in _svg_tag_attrs(svg, "line") if a.get("stroke") == _BLUE]
    return ticks[-1]


def _last_text_by_prefix(svg: str, prefix: str) -> tuple[dict[str, str], str]:
    matches = re.findall(rf"<text([^>]*)>({re.escape(prefix)}[^<]*)</text>", svg)
    attrs_str, content = matches[-1]
    return dict(re.findall(r'([\w-]+)="([^"]+)"', attrs_str)), content


_CAP_JOIN_WRAP_RE = re.compile(
    r'<g(?: stroke-linecap="(?P<cap>[^"]*)")?(?: stroke-linejoin="(?P<join>[^"]*)")?>'
    r"<polyline[^>]*/></g>"
    r"|(?P<bare><polyline[^>]*/>)"
)


def _line_cap_join_pairs(svg: str) -> list[tuple[str | None, str | None]]:
    """`(stroke_cap, stroke_join)` per polyline in document order.

    A polyline wrapped in `<g stroke-linecap="X" stroke-linejoin="Y">` reports
    `(X, Y)` -- either attribute (and its wrapping `<g>` entirely) is omitted
    when unset, so a line with only `stroke_cap=` set wraps in
    `<g stroke-linecap="X">` alone. A bare polyline (the renderer omits the
    wrapper entirely at butt/miter defaults) reports `(None, None)`.
    """
    content = _clip_group_content(svg)
    return [
        (None, None) if m.group("bare") else (m.group("cap"), m.group("join"))
        for m in _CAP_JOIN_WRAP_RE.finditer(content)
    ]


def _path_count(svg: str) -> int:
    return len(re.findall(r"<path[^>]*/>", svg))


def _blue_line_count(svg: str) -> int:
    return len(re.findall(r'<line[^>]*stroke="rgba\(37,99,235', svg))


def _reader_batch(scene: dict, kind: str) -> dict:
    batches = [m for m in scene["panels"][0]["marks"] if m["kind"] == kind]
    return batches[-1]


def _batch_node_total(scene: dict, kind: str) -> int:
    return sum(len(m["nodes"]) for m in scene["panels"][0]["marks"] if m["kind"] == kind)


# ---------------------------------------------------------------------------
# FieldProbe table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldProbe:
    field: str
    lhs_mark: str
    lhs_kwargs: dict
    rhs_mark: str
    reader_kind: str
    encoding: dict
    df: pl.DataFrame
    kind: str  # "value" | "count"
    svg_value: Callable[[str], Any] | None = None
    scene_value: Callable[[dict], Any] | None = None
    own_kwargs: dict | None = None
    svg_count: Callable[[str], int] | None = None
    scene_count: Callable[[dict], int] | None = None
    line_reader: bool = False


def _rhs(probe: FieldProbe, kwargs: dict | None = None):
    return _mark_chart(probe.df, probe.rhs_mark, kwargs or {}, probe.encoding)


def _lhs(probe: FieldProbe):
    return _mark_chart(probe.df, probe.lhs_mark, probe.lhs_kwargs, probe.encoding)


_XY = {"x": "x", "y": "y"}

PROBES: dict[str, FieldProbe] = {
    "size": FieldProbe(
        "size",
        "line",
        {"size": 30},
        "point",
        "point",
        _XY,
        ROW,
        "value",
        svg_value=lambda svg: float(_last_data_glyph(svg)[1]["r"]),
        scene_value=lambda s: _reader_batch(s, "point")["nodes"][0]["r"],
        own_kwargs={"size": 60},
    ),
    "opacity": FieldProbe(
        "opacity",
        "point",
        {"opacity": 0.3},
        "line",
        "line",
        _XY,
        CURVE,
        "value",
        svg_value=lambda svg: _svg_tag_attrs(svg, "polyline")[-1]["stroke"],
        scene_value=lambda s: _reader_batch(s, "line")["nodes"][0]["style"]["color"]["a"],
        own_kwargs={"opacity": 0.6},
        line_reader=True,
    ),
    "corner_radius": FieldProbe(
        "corner_radius",
        "bar",
        {"corner_radius": 6},
        "bar",
        "bar",
        _XY,
        ROW,
        "value",
        svg_value=lambda svg: data_bar_rects(svg)[-1].get("rx"),
        scene_value=lambda s: _reader_batch(s, "bar")["nodes"][0]["corner_radius"],
        own_kwargs={"corner_radius": 12},
    ),
    "stroke_width": FieldProbe(
        "stroke_width",
        "line",
        {"stroke_width": 5},
        "line",
        "line",
        _XY,
        CURVE,
        "value",
        svg_value=lambda svg: _svg_tag_attrs(svg, "polyline")[-1]["stroke-width"],
        scene_value=lambda s: _reader_batch(s, "line")["nodes"][0]["style"]["width"],
        own_kwargs={"stroke_width": 8},
        line_reader=True,
    ),
    "stroke_dash": FieldProbe(
        "stroke_dash",
        "line",
        {"stroke_dash": [4, 2]},
        "line",
        "line",
        _XY,
        CURVE,
        "value",
        svg_value=lambda svg: _svg_tag_attrs(svg, "polyline")[-1].get("stroke-dasharray"),
        scene_value=lambda s: _reader_batch(s, "line")["nodes"][0]["style"]["dash"],
        own_kwargs={"stroke_dash": [2, 2]},
        line_reader=True,
    ),
    "font_size": FieldProbe(
        "font_size",
        "text",
        {"font_size": 30},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0]["font-size"],
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["style"]["font_size"],
        own_kwargs={"font_size": 22},
    ),
    "font_weight": FieldProbe(
        "font_weight",
        "text",
        {"font_weight": "bold"},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0].get("font-weight"),
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["style"]["font_weight"],
        own_kwargs={"font_weight": "bold"},
    ),
    "align": FieldProbe(
        "align",
        "text",
        {"align": "left"},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0]["text-anchor"],
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["style"]["anchor"],
        own_kwargs={"align": "left"},
    ),
    "baseline": FieldProbe(
        "baseline",
        "text",
        {"baseline": "top"},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0].get("dominant-baseline"),
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["style"]["baseline"],
        own_kwargs={"baseline": "top"},
    ),
    "dx": FieldProbe(
        "dx",
        "text",
        {"dx": 20},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0]["x"],
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["x"],
        own_kwargs={"dx": 15},
    ),
    "dy": FieldProbe(
        "dy",
        "text",
        {"dy": 20},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0]["y"],
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["y"],
        own_kwargs={"dy": 15},
    ),
    "angle": FieldProbe(
        "angle",
        "text",
        {"angle": 45},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZPROBE")[0].get("transform"),
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["style"]["angle"],
        own_kwargs={"angle": 30},
    ),
    "interpolate": FieldProbe(
        "interpolate",
        "area",
        {"interpolate": "step"},
        "line",
        "line",
        _XY,
        CURVE,
        "value",
        svg_value=lambda svg: _last_curve_element(svg),
        scene_value=lambda s: _reader_batch(s, "line")["nodes"][0]["type"],
        own_kwargs={"interpolate": "step-after"},
        line_reader=True,
    ),
    "stroke_cap": FieldProbe(
        "stroke_cap",
        "line",
        {"stroke_cap": "round", "stroke_join": "round"},
        "line",
        "line",
        _XY,
        CURVE,
        "value",
        svg_value=lambda svg: _line_cap_join_pairs(svg)[-1][0],
        scene_value=lambda s: _reader_batch(s, "line")["nodes"][0]["style"]["stroke_cap"],
        own_kwargs={"stroke_cap": "square"},
        line_reader=True,
    ),
    "stroke_join": FieldProbe(
        "stroke_join",
        "line",
        {"stroke_cap": "round", "stroke_join": "round"},
        "line",
        "line",
        _XY,
        CURVE,
        "value",
        svg_value=lambda svg: _line_cap_join_pairs(svg)[-1][1],
        scene_value=lambda s: _reader_batch(s, "line")["nodes"][0]["style"]["stroke_join"],
        own_kwargs={"stroke_join": "bevel"},
        line_reader=True,
    ),
    "filled": FieldProbe(
        "filled",
        "point",
        {"filled": False},
        "point",
        "point",
        _XY,
        ROW,
        "value",
        svg_value=lambda svg: _last_data_glyph(svg)[1].get("fill"),
        scene_value=lambda s: _reader_batch(s, "point")["nodes"][0]["style"]["fill"],
        own_kwargs={"filled": False},
    ),
    "shape": FieldProbe(
        "shape",
        "point",
        {"shape": "square"},
        "point",
        "point",
        _XY,
        ROW,
        "value",
        svg_value=lambda svg: _last_data_glyph(svg)[0],
        scene_value=lambda s: _reader_batch(s, "point")["nodes"][0]["type"],
        own_kwargs={"shape": "diamond"},
    ),
    "limit": FieldProbe(
        "limit",
        "text",
        {"limit": 3},
        "text",
        "text",
        {"x": "x", "y": "y", "text": "label"},
        LONG_LABEL,
        "value",
        svg_value=lambda svg: _last_text_by_prefix(svg, "ZZ")[1],
        scene_value=lambda s: _reader_batch(s, "text")["nodes"][0]["content"],
        own_kwargs={"limit": 5},
    ),
    "band_size": FieldProbe(
        "band_size",
        "tick",
        {"band_size": 0.9},
        "tick",
        "tick",
        {"x": "cat", "y": "y"},
        ROW,
        "value",
        svg_value=lambda svg: round(
            float(_last_tick_line(svg)["x2"]) - float(_last_tick_line(svg)["x1"]), 3
        ),
        scene_value=lambda s: round(
            abs(
                _reader_batch(s, "tick")["nodes"][0]["x2"]
                - _reader_batch(s, "tick")["nodes"][0]["x1"]
            ),
            3,
        ),
        own_kwargs={"band_size": 0.3},
    ),
    "line": FieldProbe(
        "line",
        "area",
        {"line": True},
        "area",
        "area",
        _XY,
        CURVE,
        "count",
        svg_count=_path_count,
        scene_count=lambda s: _batch_node_total(s, "area"),
    ),
    "borders": FieldProbe(
        "borders",
        "area",
        {"borders": True},
        "area",
        "area",
        _XY,
        CURVE,
        "count",
        svg_count=_path_count,
        scene_count=lambda s: _batch_node_total(s, "area"),
    ),
    "arrow": FieldProbe(
        "arrow",
        "segment",
        {"arrow": True},
        "segment",
        "segment",
        {"x": "x", "y": "y", "x2": "x2", "y2": "yseg"},
        ROW,
        "count",
        svg_count=_path_count,
        scene_count=lambda s: _batch_node_total(s, "segment"),
    ),
    "arrow_size": FieldProbe(
        "arrow_size",
        "segment",
        {"arrow": True, "arrow_size": 20},
        "segment",
        "segment",
        {"x": "x", "y": "y", "x2": "x2", "y2": "yseg"},
        ROW,
        "count",
        svg_count=_path_count,
        scene_count=lambda s: _batch_node_total(s, "segment"),
    ),
    "leader_line": FieldProbe(
        "leader_line",
        "label",
        {"leader_line": True},
        "label",
        "label",
        {"x": "x", "y": "y", "text": "label"},
        ROW,
        "count",
        svg_count=_blue_line_count,
        scene_count=lambda s: _batch_node_total(s, "label"),
    ),
    "blend": FieldProbe(
        "blend",
        "point",
        {"blend": "additive"},
        "point",
        "point",
        _XY,
        ROW,
        "count",
        svg_count=lambda svg: svg.count("mix-blend-mode:screen"),
        scene_count=lambda s: sum(
            1 for m in s["panels"][0]["marks"] if m.get("blend") == "additive"
        ),
    ),
}


def test_derived_population_matches_probe_table():
    """`_primitive_reachable_fields()` and `PROBES` must name the same fields.

    Catches a future `MarkKwargsSpec` field that becomes reachable through a
    mark kwarg but gets no probe row -- the completeness guarantee behind
    "derived, never hand-listed".
    """
    assert set(PROBES) == PRIMITIVE_REACHABLE_FIELDS


LINE_READER_FIELDS = sorted(f for f, p in PROBES.items() if p.line_reader)
#: `SecondaryY` validates `mark=` against this fixed primitive whitelist
#: (`ferrum._layer._FLAT_MARK_METHODS`); a field whose reader is `segment` or
#: `label` cannot go through `SecondaryY` at all, at any output kind -- this
#: is derived from the same population `SecondaryY`'s own guard reads, not
#: hand-picked.
SECONDARY_Y_FIELDS = sorted(f for f, p in PROBES.items() if p.reader_kind in _FLAT_MARK_METHODS)
#: `independent-y` (`LayerChart(resolve={"y": "independent"})`) places no
#: constraint on the reader mark -- every derived field applies.
INDEPENDENT_Y_FIELDS = sorted(PROBES)


# ---------------------------------------------------------------------------
# Composition-path builders. Each takes the FieldProbe itself (not a bare
# (lhs, rhs) pair) because three of the seven producers -- SecondaryY, and
# the two desugar-prior-layer shapes -- do not accept an arbitrary RHS
# chart; they build their own appended layer from the probe's own fields.
# ---------------------------------------------------------------------------


def _plus_static(probe: FieldProbe) -> str:
    return _svg(_lhs(probe) + _rhs(probe))


def _plus_interactive(probe: FieldProbe) -> dict:
    return _scene(_lhs(probe) + _rhs(probe))


def _layerchart_static(probe: FieldProbe) -> str:
    return _svg(fm.LayerChart(_lhs(probe), _rhs(probe)))


def _layerchart_interactive(probe: FieldProbe) -> dict:
    return _scene(fm.LayerChart(_lhs(probe), _rhs(probe)))


def _independent_y_static(probe: FieldProbe) -> str:
    return _svg(fm.LayerChart(_lhs(probe), _rhs(probe), resolve={"y": "independent"}))


def _independent_y_interactive(probe: FieldProbe) -> dict:
    return _scene(fm.LayerChart(_lhs(probe), _rhs(probe), resolve={"y": "independent"}))


def _secondary_y_static(probe: FieldProbe) -> str:
    return _svg(_lhs(probe) + fm.SecondaryY(field="y2", mark=probe.rhs_mark))


def _secondary_y_interactive(probe: FieldProbe) -> dict:
    return _scene(_lhs(probe) + fm.SecondaryY(field="y2", mark=probe.rhs_mark))


def _desugar_smooth_chart(probe: FieldProbe):
    return _mark_chart(probe.df, probe.lhs_mark, probe.lhs_kwargs, probe.encoding).mark_smooth()


def _desugar_smooth_static(probe: FieldProbe) -> str:
    return _svg(_desugar_smooth_chart(probe))


def _desugar_smooth_interactive(probe: FieldProbe) -> dict:
    return _scene(_desugar_smooth_chart(probe))


def _desugar_function_chart(probe: FieldProbe):
    """The `mark_function(...)` MULTI-LAYER producer (decision doc,
    `_chart_methods_statistical.py:1474-1481`): calling `.mark_function()` on
    a chart that already has `_layers` set appends a third, kwarg-less
    `line`-kind layer via `self + fn_chart` -- a second, independent hoist
    opportunity distinct from both `+` (which this reuses to build the
    2-layer base) and `.mark_smooth()`'s single-append. `self` here is
    `_lhs(probe) + _rhs(probe)`, so the field's own reader mark occupies
    layer 1 and the function's line occupies layer 2 -- both kwarg-less,
    both read by the SAME (last-in-document-order) discriminator this
    module already uses for every other line-reader field.
    """
    base = _lhs(probe) + _rhs(probe)
    return base.mark_function(lambda v: v)


def _desugar_function_static(probe: FieldProbe) -> str:
    return _svg(_desugar_function_chart(probe))


def _desugar_function_interactive(probe: FieldProbe) -> dict:
    return _scene(_desugar_function_chart(probe))


PATHS: dict[str, Callable[[FieldProbe], Any]] = {
    "+static": _plus_static,
    "+interactive": _plus_interactive,
    "layerchart_static": _layerchart_static,
    "layerchart_interactive": _layerchart_interactive,
    "independent_y_static": _independent_y_static,
    "independent_y_interactive": _independent_y_interactive,
    "secondary_y_static": _secondary_y_static,
    "secondary_y_interactive": _secondary_y_interactive,
    "desugar_smooth_static": _desugar_smooth_static,
    "desugar_smooth_interactive": _desugar_smooth_interactive,
    "desugar_function_static": _desugar_function_static,
    "desugar_function_interactive": _desugar_function_interactive,
}

#: Which fields apply to each path -- every path but the four arbitrary-pair
#: ones (`+`, `LayerChart`, both output kinds) is constrained by what that
#: producer can actually build (see `SECONDARY_Y_FIELDS`/`LINE_READER_FIELDS`
#: above); every constraint is a real mark-compatibility fact, derived from
#: the same population `SecondaryY`/the desugar readers are built from, not
#: a hand-picked subset.
PATH_FIELDS: dict[str, list[str]] = {
    "+static": sorted(PROBES),
    "+interactive": sorted(PROBES),
    "layerchart_static": sorted(PROBES),
    "layerchart_interactive": sorted(PROBES),
    "independent_y_static": INDEPENDENT_Y_FIELDS,
    "independent_y_interactive": INDEPENDENT_Y_FIELDS,
    "secondary_y_static": SECONDARY_Y_FIELDS,
    "secondary_y_interactive": SECONDARY_Y_FIELDS,
    "desugar_smooth_static": LINE_READER_FIELDS,
    "desugar_smooth_interactive": LINE_READER_FIELDS,
    "desugar_function_static": LINE_READER_FIELDS,
    "desugar_function_interactive": LINE_READER_FIELDS,
}


def _render_solo(probe: FieldProbe, path_name: str, chart) -> Any:
    """Render *chart* alone in whichever output kind *path_name* uses (svg for a
    static path, scene dict for an interactive one) -- the solo-render half of a
    matrix cell, shared by both the "value" and "count" cell shapes below."""
    return _scene(chart) if path_name.endswith("interactive") else _svg(chart)


def _extract(probe: FieldProbe, path_name: str, rendered: Any) -> Any:
    is_scene = path_name.endswith("interactive")
    if probe.kind == "value":
        return probe.scene_value(rendered) if is_scene else probe.svg_value(rendered)
    return probe.scene_count(rendered) if is_scene else probe.svg_count(rendered)


# ---------------------------------------------------------------------------
# Matrix: every derived field x every path that can build it (§4.8's seven
# producers, each of the four arbitrary-pair ones split into its two output
# kinds -- twelve (path, output-kind) cells total, see PATH_FIELDS above for
# which fields reach which cell and why).
# ---------------------------------------------------------------------------

_CELLS = [(field, path) for path in PATHS for field in PATH_FIELDS[path]]

#: `dx`/`dy`/`band_size` read an absolute pixel measurement. On the two
#: margin-reserving paths -- `independent-y` and `SecondaryY` narrow the
#: plot area to make room for the second y-axis -- that measurement differs
#: from a bare flat chart's even with no leak at all, because the panel
#: geometry itself differs, not because anything cascaded. For these
#: (field, path) cells the baseline is rendered through the SAME path with
#: LHS ALSO kwarg-less (identical panel geometry, no leak source) instead of
#: a bare flat render; every other (field, path) cell keeps the simpler
#: flat-alone baseline, which is valid because `+`, `LayerChart` and the two
#: desugar paths reserve no such margin.
_GEOMETRY_SENSITIVE_FIELDS = {"dx", "dy", "band_size"}
_MARGIN_RESERVING_PATHS = {
    "independent_y_static",
    "independent_y_interactive",
    "secondary_y_static",
    "secondary_y_interactive",
}


@pytest.mark.parametrize("field,path_name", _CELLS, ids=[f"{f}-{p}" for f, p in _CELLS])
def test_kwarg_less_sibling_matches_its_flat_self(field, path_name):
    probe = PROBES[field]
    combined = PATHS[path_name](probe)
    if probe.kind == "value":
        if field in _GEOMETRY_SENSITIVE_FIELDS and path_name in _MARGIN_RESERVING_PATHS:
            baseline_chart = PATHS[path_name](replace(probe, lhs_kwargs={}))
            flat_rendered = _extract(probe, path_name, baseline_chart)
        else:
            flat_rendered = _extract(probe, path_name, _render_solo(probe, path_name, _rhs(probe)))
        assert _extract(probe, path_name, combined) == flat_rendered
    else:
        solo_lhs = _extract(probe, path_name, _render_solo(probe, path_name, _lhs(probe)))
        solo_rhs = _extract(probe, path_name, _render_solo(probe, path_name, _rhs(probe)))
        assert _extract(probe, path_name, combined) == solo_lhs + solo_rhs


@pytest.mark.parametrize(
    "field,output_kind",
    [
        (f, k)
        for f in sorted(f for f, p in PROBES.items() if p.kind == "value")
        for k in ("svg", "scene")
    ],
)
def test_vacuity_control_rhs_with_its_own_kwarg_differs(field, output_kind):
    """The discriminator is not trivially constant, in EITHER output kind: RHS
    WITH the kwarg differs from flat RHS. Covers both `svg_value` (the four
    static cells' extractor) and `scene_value` (the interactive cells' own,
    separate extractor -- e.g. `interpolate` reads a polyline/path SVG tag on
    one side and a scene node's `type` key on the other) so neither can pass
    an interactive cell by accident of a constant discriminator.
    """
    probe = PROBES[field]
    render = _scene if output_kind == "scene" else _svg
    value = probe.scene_value if output_kind == "scene" else probe.svg_value
    flat = value(render(_rhs(probe)))
    own = value(render(_rhs(probe, probe.own_kwargs)))
    assert own != flat, (
        f"{field}/{output_kind}: value did not change between flat and own-kwarg RHS -- vacuous discriminator"
    )


@pytest.mark.parametrize(
    "field,output_kind",
    [
        (f, k)
        for f in sorted(f for f, p in PROBES.items() if p.kind == "count")
        for k in ("svg", "scene")
    ],
)
def test_vacuity_control_solo_lhs_differs_from_solo_rhs(field, output_kind):
    """Count-kind vacuity, in both output kinds: LHS-alone's own count differs
    from RHS-alone's (the field visibly draws something the RHS's default
    does not)."""
    probe = PROBES[field]
    render = _scene if output_kind == "scene" else _svg
    count = probe.scene_count if output_kind == "scene" else probe.svg_count
    lhs_count = count(render(_lhs(probe)))
    rhs_count = count(render(_rhs(probe)))
    assert lhs_count != rhs_count


# ---------------------------------------------------------------------------
# The wire-level pin: a layered `to_spec()` carries no top-level `mark_style`.
# ---------------------------------------------------------------------------


def _spec_dict(chart) -> dict:
    """The wire-level `ChartSpec` as a dict. `ChartSpec` (the pyo3 class) exposes no
    readable `.mark_style` attribute despite the `.pyi` stub -- `to_json()` is the
    only way to observe what actually serialized."""
    return json.loads(chart.to_spec().to_json())


@pytest.mark.parametrize("field", sorted(PROBES))
def test_lhs_retains_its_own_mark_style(field):
    """An over-broad gate is not what closed `#119`: this checks the wire
    directly (`layers[0].mark_style`), the same evidence shape quality
    review used to probe it (`layers[0].mark_style={'size': 100, 'opacity':
    0.3}`). A gate that stripped layer 0's own per-layer `mark_style` --
    not just the chart-level hoist -- would still pass every render-equality
    cell above for a field whose RHS mark happens to ignore that field
    (e.g. `filled` on a `bar` RHS), because "RHS renders like its flat self"
    says nothing about layer 0. Checked once per field via `+` rather than
    once per (field, path) cell: `layers[0].mark_style` is built by
    `_Layer`/`_build_layers_list` from the SAME per-layer kwargs dict on
    every producer this module exercises -- `#119`'s fix never touches that
    path, only the chart-level emission beside it -- so this is one
    mechanism, not twelve independent ones.
    """
    probe = PROBES[field]
    spec = _spec_dict(_lhs(probe) + _rhs(probe))
    assert spec["layers"][0]["mark_style"] == probe.lhs_kwargs


def test_plus_layered_spec_has_no_top_level_mark_style():
    chart = fm.Chart(CURVE).mark_point(opacity=0.3).encode(**_XY) + fm.Chart(
        CURVE
    ).mark_line().encode(**_XY)
    spec = _spec_dict(chart)
    assert "mark_style" not in spec
    assert "layers" in spec


def test_layerchart_merged_spec_has_no_top_level_mark_style():
    a = fm.Chart(CURVE).mark_point(opacity=0.3).encode(**_XY)
    b = fm.Chart(CURVE).mark_line().encode(**_XY)
    spec = _spec_dict(fm.LayerChart(a, b)._build_merged())
    assert "mark_style" not in spec


def test_independent_y_merged_spec_has_no_top_level_mark_style():
    a = fm.Chart(CURVE).mark_point(opacity=0.3).encode(**_XY)
    b = fm.Chart(CURVE).mark_line().encode(**_XY)
    spec = _spec_dict(fm.LayerChart(a, b, resolve={"y": "independent"})._build_merged())
    assert "mark_style" not in spec


def test_secondary_y_spec_has_no_top_level_mark_style():
    base = fm.Chart(CURVE).mark_point(opacity=0.3).encode(**_XY)
    chart = base + fm.SecondaryY(field="y2", mark="line")
    spec = _spec_dict(chart)
    assert "mark_style" not in spec


def test_desugar_smooth_spec_has_no_top_level_mark_style():
    chart = fm.Chart(CURVE).mark_point(opacity=0.3).encode(**_XY).mark_smooth()
    spec = _spec_dict(chart)
    assert "mark_style" not in spec


def test_desugar_function_multi_layer_spec_has_no_top_level_mark_style():
    base = fm.Chart(CURVE).mark_point(opacity=0.3).encode(**_XY) + fm.Chart(
        CURVE
    ).mark_line().encode(**_XY)
    chart = base.mark_function(lambda v: v)
    spec = _spec_dict(chart)
    assert "mark_style" not in spec
    assert len(spec["layers"]) == 3


def test_flat_chart_spec_still_carries_mark_style():
    """Negative control: a genuinely flat (single-mark) chart is unaffected by the gate."""
    spec = _spec_dict(fm.Chart(ROW).mark_point(opacity=0.3).encode(**_XY))
    assert spec["mark_style"] == {"opacity": 0.3}


# ---------------------------------------------------------------------------
# The six §4.8 rows, named.
# ---------------------------------------------------------------------------


def test_row1_line_stroke_and_width_are_defaults_not_leaked():
    chart = fm.Chart(CURVE).mark_point(opacity=0.3, stroke_width=5).encode(**_XY) + fm.Chart(
        CURVE
    ).mark_line().encode(**_XY)
    line_attrs = _svg_tag_attrs(_svg(chart), "polyline")[-1]
    assert line_attrs["stroke"] == _BLUE
    assert line_attrs["stroke-width"] == "1.5"


def test_row2_clip_false_does_not_escape_sibling_points():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 1.0, 5.0]})
    chart = fm.Chart(df).mark_function(lambda v: v, clip=False) + fm.Chart(df).mark_point().encode(
        x="x", y="y"
    )
    svg = _svg(chart)
    content = _clip_group_content(svg)
    clipped_circles = len(re.findall(r"<circle", content))
    total_circles = len(re.findall(r"<circle", svg))
    assert total_circles == 4
    assert clipped_circles == 4, (
        f"expected 4 of 4 points inside the panel clip, got {clipped_circles}"
    )
    # Positive half (S2 finding #4): layer 0's own clip=False is retained, not
    # just the sibling's clip left alone -- the function's polyline still
    # renders OUTSIDE the clip group, exactly as a flat clip=False chart does.
    clipped_polylines = len(re.findall(r"<polyline", content))
    total_polylines = len(re.findall(r"<polyline", svg))
    assert total_polylines == 1
    assert clipped_polylines == 0, (
        "expected the function's own polyline to still escape the panel clip"
    )


def test_row3_detail_is_an_encoding_like_color():
    """`detail` is an encoding, not layer style, and stays with the operand that
    wrote it exactly as `color` does: the left line splits into three groups
    and the right line, which binds neither, draws one."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": ["p", "p", "q", "q", "r", "r"],
        }
    )

    def count(channel: str) -> int:
        chart = fm.Chart(df).mark_line().encode(x="x", y="y", **{channel: "g"}) + fm.Chart(
            df
        ).mark_line().encode(x="x", y="y")
        return len(re.findall(r"<polyline[^>]*/>", _svg(chart)))

    assert count("detail") == 4
    assert count("color") == 4


def _area_closing_axis(path_d: str) -> str:
    """Whether an area path's final two `L` points close along a fixed x (x baseline) or fixed y (y baseline)."""
    points = [tuple(float(v) for v in p.split()) for p in re.findall(r"L([\d.]+ [\d.]+)", path_d)][
        -2:
    ]
    (x1, y1), (x2, y2) = points
    if abs(x1 - x2) < 1e-6:
        return "x_baseline"
    assert abs(y1 - y2) < 1e-6
    return "y_baseline"


def test_row4_value_axis_does_not_leak_into_siblings_own_baseline():
    df = pl.DataFrame({"value": [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]})
    chart = fm.Chart(df).mark_density(orientation="horizontal").encode(y="value") + fm.Chart(
        df
    ).mark_area().encode(x="value", y="value")
    paths = re.findall(r"<path[^>]*/>", _svg(chart))
    assert _area_closing_axis(paths[-1]) == "y_baseline"


def test_row5_desugar_smooth_fit_line_is_default_not_leaked():
    chart = fm.Chart(CURVE).mark_point(opacity=0.3, stroke_dash=[4, 2]).encode(**_XY).mark_smooth()
    svg = _svg(chart)
    fit_attrs = _svg_tag_attrs(svg, "polyline")[-1]
    assert fit_attrs["stroke"] == _BLUE
    assert "stroke-dasharray" not in fit_attrs


def test_row6_layerchart_static_equals_interactive():
    a = fm.Chart(CURVE).mark_point(opacity=0.3, stroke_width=5).encode(**_XY)
    b = fm.Chart(CURVE).mark_line().encode(**_XY)
    layer = fm.LayerChart(a, b)
    static_attrs = _svg_tag_attrs(_svg(layer), "polyline")[-1]
    scene = _scene(layer)
    interactive_style = _reader_batch(scene, "line")["nodes"][0]["style"]

    assert static_attrs["stroke"] == _BLUE
    assert static_attrs["stroke-width"] == "1.5"
    assert interactive_style["color"]["a"] == 255
    assert interactive_style["width"] == 1.5


# ---------------------------------------------------------------------------
# The wire half: a layered `ChartSpec` carrying a chart-level `mark_style`
# (a shape the Python producer never emits) is refused, not silently ignored.
# ---------------------------------------------------------------------------

_LAYERED_MARK_STYLE_REFUSAL = (
    "ChartSpec.mark_style is set on a layered chart (2 layers), where it has no "
    "effect: a layered chart is styled per layer. Move the style onto each "
    "layer's own mark_style."
)


def _layered_spec_json(*, chart_mark_style: dict | None) -> str:
    chart = fm.Chart(CURVE).mark_point().encode(**_XY) + fm.Chart(CURVE).mark_line().encode(**_XY)
    spec = _spec_dict(chart)
    if chart_mark_style is not None:
        spec["mark_style"] = chart_mark_style
    return json.dumps(spec)


def _render_from_json(spec_json: str, output_kind: str) -> None:
    from ferrum import ChartSpec
    from ferrum._core import render_interactive, render_svg

    spec = ChartSpec.from_json(spec_json)
    renderer = render_svg if output_kind == "static" else render_interactive
    renderer(spec, CURVE.to_arrow(), viewport=(400.0, 300.0))


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
def test_layered_spec_with_chart_level_mark_style_is_refused(output_kind):
    with pytest.raises(ValueError) as exc:
        _render_from_json(_layered_spec_json(chart_mark_style={"size": 100}), output_kind)
    assert str(exc.value) == _LAYERED_MARK_STYLE_REFUSAL


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
def test_layered_spec_without_chart_level_mark_style_round_trips(output_kind):
    """Vacuity control: the same layered spec without the chart-level style renders."""
    _render_from_json(_layered_spec_json(chart_mark_style=None), output_kind)
