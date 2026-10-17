"""Cross-surface matrix: every appearance-channel kwarg works or is refused (#123).

Matrix-first per the 6A lesson (binding, see `.claude/output/decisions/
2026-09-23-wave6b-dedup-decision.md` §"Standing constraints"): this file is
written and run red *before* the `#123` source fix, then the fix turns cells
green. The channel population is derived from ``ferrum.encoding.appearance``
(every ``ChannelBase`` subclass that module defines — 11 classes), never
hand-listed; `test_channel_population_is_the_documented_eleven` is the vacuity
control pinning that derivation actually finds something and cannot pass
trivially on an empty or wrong set.

Each cell asserts one of three things, per spec §4.9's table
(`.claude/output/specs/2026-09-23-wave6b-dedup-design.md`):

- **works** — the kwarg changes the render, proved by a *concrete rendered
  attribute*: a literal hex color, a legend-title/entry-label text node, a
  glyph tag (or, for Shape, the exact glyph a specific category gets), a
  `stroke-dasharray` value per category, or a numeric attribute (radius,
  alpha) read off the SVG. A bare "the two SVG strings differ" is never
  the whole assertion (cycle-2 remediation: several cells only asserted
  that in cycle 1, which cannot tell a correct render from a different-but-
  still-wrong one — spec-review + quality-review finding).
- **refused** — construction raises ``ValueError`` whose message names both
  the public entry and the value received.
- **n/a** — the kwarg is not part of this channel's contract; today's
  generic "accepted but not yet honored" ``UserWarning`` still fires and
  nothing raises.

Cycle 2 (spec-review + quality-review remediation, 2026-09-23): the Rust
half (`render/scale_resolve/auxiliary.rs`) landed between cycle 1 and this
pass, so the four cells that were RUST-PENDING in cycle 1 (recorded by name,
not xfailed, per dispatch instruction) are now exercised for real and
strengthened to assert the concrete per-category glyph/pattern mapping
rather than a bare inequality.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import polars as pl
import pytest

import ferrum as fm
import ferrum.encoding.appearance as _appearance_module
from ferrum._core import aux_legend_channels  # type: ignore[attr-defined]
from ferrum._warn import reset_warnings
from ferrum.color import palette
from ferrum.encoding.base import ChannelBase

# ---------------------------------------------------------------------------
# Derived population
# ---------------------------------------------------------------------------


def _appearance_channel_classes() -> list[type]:
    """The appearance-channel population, derived by module identity.

    Walks every ``ChannelBase`` subclass (recursively) and keeps the ones
    whose *defining* module is ``ferrum.encoding.appearance`` -- exactly the
    11 appearance channels (Color, Fill, Stroke, Size, Shape, StrokeDash,
    Opacity, FillOpacity, StrokeOpacity, StrokeWidth, Angle). A new
    appearance channel lands in this matrix automatically; nothing here
    hand-lists a channel name.
    """
    seen: set[type] = set()
    stack: list[type] = [ChannelBase]
    discovered: list[type] = []
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                discovered.append(sub)
                stack.append(sub)
    return sorted(
        (c for c in discovered if c.__module__ == _appearance_module.__name__),
        key=lambda c: c.__name__,
    )


_APPEARANCE_CLASSES = _appearance_channel_classes()
_BY_NAME: dict[str, type] = {c._channel_name: c for c in _APPEARANCE_CLASSES}


def test_channel_population_is_the_documented_eleven() -> None:
    """Vacuity control: the derived population is exactly the 11 documented channels.

    Guards every other test in this file against a harness bug (e.g. an
    import-path typo) that would make ``_APPEARANCE_CLASSES`` empty or
    wrong, which would make every "cell" below vacuously pass by never
    running.
    """
    assert {c.__name__ for c in _APPEARANCE_CLASSES} == {
        "Color",
        "Fill",
        "Stroke",
        "Size",
        "Shape",
        "StrokeDash",
        "Opacity",
        "FillOpacity",
        "StrokeOpacity",
        "StrokeWidth",
        "Angle",
    }
    assert len(_APPEARANCE_CLASSES) == 11


# ---------------------------------------------------------------------------
# Fixture data + render helpers
# ---------------------------------------------------------------------------

_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        "g": ["a", "b", "c", "a", "b", "c"],
        "n": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)

# A 2-row, one-row-per-category fixture: document order equals category order
# (a then b), which the per-category glyph/dasharray assertions below rely on.
_DF2 = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "g": ["a", "b"]})

# Which fixture field a channel binds to for this matrix: the five
# color/shape/dash channels get the categorical field "g", everything else
# gets the numeric field "n". A hand-authored fixture-selection table (not
# the derived channel population itself), mirroring the shape
# `discriminate.py`/`refine.py` research probes used.
_CATEGORICAL_CHANNELS = frozenset({"color", "fill", "stroke", "shape", "stroke_dash"})

# Channels whose primary visible paint is the stroke, not the fill --
# rendered with mark_point(filled=False) so a style change on the stroke
# reaches visible pixels/attributes.
_STROKE_PRIMARY_CHANNELS = frozenset({"stroke", "stroke_width", "stroke_opacity", "stroke_dash"})

#: The theme's default single-series mark color -- every probe chart in this
#: file encodes exactly one appearance channel with no explicit color, so
#: every data-mark glyph paints (or outlines, for a stroke-primary channel)
#: in this one color. Used to isolate data-mark glyphs from axis/grid/legend
#: chrome in the raw SVG (verified by probe: chrome never uses this exact
#: hex value once ``legend=None`` removes the legend swatches).
_MARK_COLOR = "#2563eb"


def _field_for(channel_name: str) -> str:
    return "g" if channel_name in _CATEGORICAL_CHANNELS else "n"


def _render(
    channel_cls: type, *, df: pl.DataFrame = _DF, field: str | None = None, **kwargs: Any
) -> str:
    """Render a probe chart with *channel_cls* built with **kwargs, warnings silenced.

    *field* overrides the usual categorical-vs-numeric field selection (used
    by the `type=` render tests, which deliberately bind a normally-categorical
    channel to the numeric field "n" to exercise the quantitative-vs-nominal
    scale-resolution split).
    """
    reset_warnings()
    name = channel_cls._channel_name
    if field is None:
        field = _field_for(name)
    filled = name not in _STROKE_PRIMARY_CHANNELS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chart = (
            fm.Chart(df)
            .mark_point(filled=filled)
            .encode(x="x:Q", y="y:Q", **{name: channel_cls(field, **kwargs)})
        )
        return chart.to_svg()


def _circle_fill_colors(svg: str) -> list[str]:
    return re.findall(r'<circle[^>]*\sfill="([^"]+)"', svg)


def _circle_stroke_colors(svg: str) -> list[str]:
    return re.findall(r'<circle[^>]*\sstroke="([^"]+)"', svg)


def _circle_radii(svg: str) -> list[float]:
    return [float(v) for v in re.findall(r'<circle[^>]*\sr="([0-9.]+)"', svg)]


def _circle_fill_alphas(svg: str) -> list[float]:
    """Every ``rgba(r,g,b,a)`` alpha channel used as a mark's fill, in document order."""
    return [float(v) for v in re.findall(r"rgba\(\d+,\s*\d+,\s*\d+,\s*([0-9.]+)\)", svg)]


def _attr_values(svg: str, attr: str) -> list[float]:
    """Every numeric value of *attr* (e.g. ``fill-opacity``, ``stroke-opacity``), in order."""
    return [float(v) for v in re.findall(rf'{re.escape(attr)}="([0-9.]+)"', svg)]


def _legend_title_present(svg: str, field: str) -> bool:
    """Whether *field*'s bare legend-title text node (``>field<``) appears exactly once."""
    return svg.count(f">{field}<") == 1


def _legend_category_labels(svg: str) -> list[str]:
    """The single-letter category legend-entry labels (``>a<``, ``>b<``, ``>c<``), in order."""
    return re.findall(r">([abc])</text>", svg)


def _mark_glyph_kinds(svg: str, color: str = _MARK_COLOR) -> list[str]:
    """The shape-glyph kind of every data mark painted in *color*, in document order.

    Identifies each glyph by its SVG tag plus, for a ``<path>``, its vertex
    count -- ``ShapeKind::Diamond`` emits ``MoveTo`` + 3×``LineTo`` (4
    vertices) and ``ShapeKind::TriangleUp``/``TriangleDown`` emit ``MoveTo``
    + 2×``LineTo`` (3 vertices), matching `render/marks/point.rs`'s exact
    ``PathCmd`` sequence per shape read directly from source, not assumed.
    A ``ShapeKind::Cross`` glyph paints two consecutive ``<line>`` elements
    at the same center, which collapse into one ``"cross"`` entry here.
    """
    kinds: list[str] = []
    pattern = (
        rf'<circle[^>]*fill="{re.escape(color)}"[^>]*/>'
        rf'|<rect[^>]*fill="{re.escape(color)}"[^>]*/>'
        rf'|<path d="([^"]+)"[^>]*fill="{re.escape(color)}"[^>]*/>'
        rf'|<line[^>]*stroke="{re.escape(color)}"[^>]*/>'
    )
    for m in re.finditer(pattern, svg):
        text = m.group(0)
        if text.startswith("<circle"):
            kinds.append("circle")
        elif text.startswith("<rect"):
            kinds.append("square")
        elif text.startswith("<path"):
            vertices = m.group(1).count("L") + 1
            kinds.append({4: "diamond", 3: "triangle"}.get(vertices, f"path-{vertices}v"))
        elif text.startswith("<line"):
            kinds.append("line")
    # Collapse a consecutive <line>,<line> pair (one ShapeKind::Cross glyph)
    # into a single "cross" entry.
    collapsed: list[str] = []
    i = 0
    while i < len(kinds):
        if kinds[i] == "line" and i + 1 < len(kinds) and kinds[i + 1] == "line":
            collapsed.append("cross")
            i += 2
        else:
            collapsed.append(kinds[i])
            i += 1
    return collapsed


def _dasharrays(svg: str, color: str = _MARK_COLOR) -> list[str | None]:
    """The ``stroke-dasharray`` value (or ``None`` if absent/solid) for each
    ``<circle>`` stroked in *color*, in document order."""
    out: list[str | None] = []
    for m in re.finditer(rf'<circle[^>]*stroke="{re.escape(color)}"[^>]*/>', svg):
        dash = re.search(r'stroke-dasharray="([^"]+)"', m.group(0))
        out.append(dash.group(1) if dash else None)
    return out


def test_vacuity_control_render_helper_detects_a_real_difference() -> None:
    """Sanity control for the whole matrix: two different scheme values MUST render differently.

    If `_render` silently ignored its kwargs or always produced the same
    fixture SVG, every "changes the render" assertion below would pass
    vacuously. This proves the harness itself can tell two charts apart,
    using a kwarg (`scheme=`) this task does not touch.
    """
    a = _render(fm.Color, scheme="dark2")
    b = _render(fm.Color, scheme="set1")
    assert a != b


def test_vacuity_control_mark_glyph_kinds_reads_the_default_palette_assignment() -> None:
    """Sanity control for `_mark_glyph_kinds`: an unscaled Shape channel assigns
    `SHAPE_PALETTE` entries in first-appearance category order -- "a" (first)
    gets `ShapeKind::Circle`, "b" (second) gets `ShapeKind::Square` -- proving
    the helper reads real per-glyph identity, not a constant."""
    svg = _render(fm.Shape, df=_DF2, legend=None)
    assert _mark_glyph_kinds(svg) == ["circle", "square"]


# ---------------------------------------------------------------------------
# `type=` -- unaffected by #123; every channel honors it (accepts it
# silently and normalizes long spellings). Whether declaring it also moves
# the *render* is architecture-dependent, verified from Rust source rather
# than assumed:
#
# - Color, Fill, Stroke (`render/scale_resolve/color.rs:115`,
#   `infer_spec_type(c_enc, ...)`) and StrokeDash
#   (`render/scale_resolve/auxiliary.rs:477`, the same call) each consult
#   the declared/inferred type to choose a continuous-quantitative scale vs
#   a categorical one -- on a numeric field, declaring `type="N"` genuinely
#   flips the render from a color gradient / numeric palette index to a
#   discrete qualitative-palette assignment / `StrokeDashScale`.
# - Shape (`build_shape_scale`), Size (`build_size_scale`) and the opacity
#   family (`build_opacity_channel_scale`) never call `infer_spec_type` or
#   read `.type_` at all (grep of `auxiliary.rs`: the only `infer_spec_type`
#   call in the file is StrokeDash's, above) -- each always resolves its
#   scale the same way regardless of the declared type; StrokeWidth's
#   linear scale (`build_stroke_width_scale`) never reads it either, and
#   Angle has no scale at all (a per-row constant). For these seven, `type=` is architecturally
#   inert at render time; the wire-normalization check is the only
#   discriminating assertion that exists for them, not a shortcut around a
#   render check the brief demands but this cell happens not to need.
# ---------------------------------------------------------------------------

_TYPE_RENDER_AFFECTS_CHANNELS = frozenset({"color", "fill", "stroke", "stroke_dash"})
_TYPE_WIRE_ONLY_CHANNELS = frozenset(_BY_NAME) - _TYPE_RENDER_AFFECTS_CHANNELS

#: The default qualitative palette's first swatch -- present in a categorical
#: (nominal/ordinal) render of a numeric field, absent from the same field's
#: default continuous (quantitative) render, for Color/Fill (`fill=`) and
#: Stroke (`stroke=`). Used as the concrete discriminating attribute below.
_QUALITATIVE_MARKER_COLOR = "#2563eb"

_TYPE_RENDER_ATTR_BY_CHANNEL = {
    "color": "fill",
    "fill": "fill",
    "stroke": "stroke",
}


@pytest.mark.parametrize(
    "channel_cls",
    sorted(
        (_BY_NAME[n] for n in _TYPE_RENDER_AFFECTS_CHANNELS - {"stroke_dash"}),
        key=lambda c: c.__name__,
    ),
    ids=lambda c: c.__name__,
)
def test_type_changes_the_render_for_color_family(channel_cls: type) -> None:
    """Declaring `type="N"` on a numeric field flips Color/Fill/Stroke from a
    continuous gradient to the discrete qualitative palette -- concretely,
    the qualitative palette's first swatch appears under `type="N"` and is
    absent from the default (quantitative) render (the vacuity control)."""
    attr = _TYPE_RENDER_ATTR_BY_CHANNEL[channel_cls._channel_name]
    quantitative = _render(channel_cls, field="n", legend=None)
    nominal = _render(channel_cls, field="n", legend=None, type="N")
    quantitative_colors = (
        _circle_fill_colors(quantitative) if attr == "fill" else _circle_stroke_colors(quantitative)
    )
    nominal_colors = (
        _circle_fill_colors(nominal) if attr == "fill" else _circle_stroke_colors(nominal)
    )
    assert _QUALITATIVE_MARKER_COLOR not in quantitative_colors, (
        "vacuity control: the default (quantitative) render must NOT already "
        "carry the qualitative marker color"
    )
    assert _QUALITATIVE_MARKER_COLOR in nominal_colors, (
        f"{channel_cls.__name__}(type='N') must switch to the qualitative palette"
    )


def test_type_changes_the_render_for_stroke_dash() -> None:
    """Declaring `type="N"` on a numeric StrokeDash field flips it from the raw
    numeric dash-palette-index contract to `StrokeDashScale`'s categorical
    resolution -- concretely, the rendered dasharray sequence changes to the
    exact known values, not merely to *some* different sequence.

    `_DF`'s "n" column is `[1, 2, 3, 4, 5, 6]`. Under the default
    (quantitative) resolution, each row reads its raw value as a
    `resolve_stroke_dash` palette index -- values 4, 5 and 6 all resolve to
    the same pattern as 1, 2 and 3 (the crate's own index-cycling rule, not
    re-derived here). Under `type="N"` the 6 numeric values resolve instead
    as 6 distinct categories through `StrokeDashScale`, cycling the 4-slot
    palette (solid, "6,3", "2,3", "6,3,2,3") in first-appearance order --
    the 4th and 5th categories land back on the palette's first two slots
    (solid, "6,3"). Both sequences pinned by direct probe against the live
    renderer, not hand-derived.
    """
    quantitative = _dasharrays(_render(fm.StrokeDash, field="n", legend=None))
    nominal = _dasharrays(_render(fm.StrokeDash, field="n", legend=None, type="N"))
    assert quantitative == ["6,3", "2,3", "6,3,2,3", "6,3,2,3", "6,3,2,3", "6,3,2,3"]
    assert nominal == [None, "6,3", "2,3", "6,3,2,3", None, "6,3"]


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _TYPE_WIRE_ONLY_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_type_reaches_the_wire_normalized_for_type_blind_channels(channel_cls: type) -> None:
    """Shape/Size/Opacity-family/StrokeWidth/Angle: `type=` is honored (accepted,
    normalized, serialized) but architecturally never consulted by these
    channels' Rust scale builders -- see this section's header comment for
    the source citations. The wire is the only real discriminator here."""
    reset_warnings()
    field = _field_for(channel_cls._channel_name)
    ch = channel_cls(field, type="ordinal")
    assert ch.to_encoding_spec_dict()["type_"] == "O"


# ---------------------------------------------------------------------------
# `legend=None` / `legend=False` -- silent on "none"-surface channels,
# disables the rendered legend everywhere else (every appearance channel but
# Angle has `_legend_surface` "own", and none has "cascade").
#
# The opacity family and StrokeWidth draw a legend of their own once a field
# is bound to them, so they sit with the legend owners: `legend=None`
# suppresses it, `title=` and a legend dict label it.
# ---------------------------------------------------------------------------

_NO_LEGEND_CHANNELS = frozenset({"angle"})
_AUX_LEGEND_CHANNELS = frozenset(aux_legend_channels())
_OWN_NON_AUX_LEGEND_CHANNELS = frozenset({"color", "fill", "stroke"})


@pytest.mark.parametrize("channel_cls", _APPEARANCE_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("value", [None, False], ids=["None", "False"])
def test_legend_none_and_false(channel_cls: type, value: Any) -> None:
    name = channel_cls._channel_name
    field = _field_for(name)
    with_legend = _render(channel_cls)
    without_legend = _render(channel_cls, legend=value)
    if name in _NO_LEGEND_CHANNELS:
        assert with_legend == without_legend, (
            f"{name}(legend={value!r}) must stay silent (no legend exists)"
        )
    else:
        assert _legend_title_present(with_legend, field)
        assert not _legend_title_present(without_legend, field), (
            f"{name}(legend={value!r}) must disable the channel's own legend title"
        )


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _OWN_NON_AUX_LEGEND_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    "legend_value_factory",
    [lambda: {"title": "CustomLegendTitle"}, lambda: fm.Legend(title="CustomLegendTitle")],
    ids=["dict", "Legend"],
)
def test_legend_dict_or_object_works_for_own_non_aux_channels(
    channel_cls: type, legend_value_factory
) -> None:
    """Color/Fill/Stroke: a legend dict or ``Legend(...)`` sets the legend title text (works)."""
    svg = _render(channel_cls, legend=legend_value_factory())
    assert ">CustomLegendTitle<" in svg


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _AUX_LEGEND_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    "legend_value_factory",
    [lambda: {"orient": "bottom"}, lambda: fm.Legend(orient="bottom")],
    ids=["dict", "Legend"],
)
def test_legend_orient_refused_for_aux_channels(channel_cls: type, legend_value_factory) -> None:
    """Every aux-legend channel: a legend value with orient != 'none' is refused (#123, Q10)."""
    name = channel_cls.__name__
    with pytest.raises(ValueError, match=rf"{name}\(legend=\.\.\.\): orient='bottom'"):
        channel_cls(_field_for(channel_cls._channel_name), legend=legend_value_factory())


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _AUX_LEGEND_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    "legend_value_factory",
    [lambda: {"orient": "none"}, lambda: fm.Legend(orient="none")],
    ids=["dict", "Legend"],
)
def test_legend_orient_none_still_disables_aux_legend(
    channel_cls: type, legend_value_factory
) -> None:
    """Every aux-legend channel: orient='none' still disables its legend (works)."""
    field = _field_for(channel_cls._channel_name)
    with_legend = _render(channel_cls)
    without_legend = _render(channel_cls, legend=legend_value_factory())
    assert _legend_title_present(with_legend, field)
    assert not _legend_title_present(without_legend, field)


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _AUX_LEGEND_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    "legend_value_factory",
    [lambda: {"title": "T"}, lambda: fm.Legend(title="T")],
    ids=["dict", "Legend"],
)
def test_legend_dict_or_object_orient_free_works_for_aux_channels(
    channel_cls: type, legend_value_factory
) -> None:
    """Every aux-legend channel: an orient-free legend dict/``Legend(...)`` still
    configures the legend (works) -- spec §4.9's cell for the aux channels
    with a benign (non-orient) legend value, previously untested (cycle-1
    spec-review finding: only the orient-refusal and orient='none' paths
    were covered, so a regression that refused every aux legend dict would
    have passed the matrix)."""
    svg = _render(channel_cls, legend=legend_value_factory())
    assert ">T<" in svg


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _NO_LEGEND_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    "legend_value_factory",
    [lambda: {"title": "T"}, lambda: fm.Legend(title="T"), lambda: {"orient": "none"}],
    ids=["dict", "Legend", "dict_orient_none"],
)
def test_legend_dict_or_object_refused_for_no_legend_channels(
    channel_cls: type, legend_value_factory
) -> None:
    """Angle: any legend value refuses."""
    name = channel_cls.__name__
    with pytest.raises(ValueError, match=rf"{name}\(legend=.*\) is not supported"):
        channel_cls(_field_for(channel_cls._channel_name), legend=legend_value_factory())


# ---------------------------------------------------------------------------
# `title=` -- works on a legend owner, refused on every no-legend channel
# ---------------------------------------------------------------------------

_TITLE_REFUSED_CHANNELS = frozenset({"angle"})
_TITLE_WORKS_CHANNELS = frozenset(_BY_NAME) - _TITLE_REFUSED_CHANNELS


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _TITLE_WORKS_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_title_works(channel_cls: type) -> None:
    svg = _render(channel_cls, title="MyProbeTitle")
    assert ">MyProbeTitle<" in svg


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _TITLE_REFUSED_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_title_refused(channel_cls: type) -> None:
    name = channel_cls.__name__
    with pytest.raises(ValueError, match=rf"{name}\(title='MyProbeTitle'\) is not honored"):
        channel_cls(_field_for(channel_cls._channel_name), title="MyProbeTitle")


# ---------------------------------------------------------------------------
# `sort=` -- works only for Color/Shape/StrokeDash; n/a everywhere else
# ---------------------------------------------------------------------------

_SORT_WORKS_CHANNELS = frozenset({"color", "shape", "stroke_dash"})
_SORT_NA_CHANNELS = frozenset(_BY_NAME) - _SORT_WORKS_CHANNELS


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _SORT_WORKS_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_sort_works(channel_cls: type) -> None:
    """`sort=` reorders the legend's category-entry labels to the requested order."""
    svg = _render(channel_cls, sort=["c", "a", "b"], legend={"title": "Cats"})
    assert _legend_category_labels(svg) == ["c", "a", "b"]


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _SORT_NA_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_sort_na_warns_unchanged(channel_cls: type) -> None:
    reset_warnings()
    field = _field_for(channel_cls._channel_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        channel_cls(field, sort="descending")
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)


# ---------------------------------------------------------------------------
# `scheme=` -- works only for Color/Fill/Stroke; n/a everywhere else
# ---------------------------------------------------------------------------

_SCHEME_WORKS_CHANNELS = frozenset({"color", "fill", "stroke"})
_SCHEME_NA_CHANNELS = frozenset(_BY_NAME) - _SCHEME_WORKS_CHANNELS


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _SCHEME_WORKS_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_scheme_works(channel_cls: type) -> None:
    """`scheme="dark2"` paints marks with dark2's own palette colors, read from
    `ferrum.color.palette` rather than a color this test invented."""
    dark2 = palette("dark2", 3)
    svg = _render(channel_cls, scheme="dark2")
    if channel_cls._channel_name == "stroke":
        colors = _circle_stroke_colors(svg)
    else:
        colors = _circle_fill_colors(svg)
    assert set(dark2).issubset(set(colors))


@pytest.mark.parametrize(
    "channel_cls",
    sorted((_BY_NAME[n] for n in _SCHEME_NA_CHANNELS), key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_scheme_na_warns_unchanged(channel_cls: type) -> None:
    reset_warnings()
    field = _field_for(channel_cls._channel_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        channel_cls(field, scheme="dark2")
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)


# ---------------------------------------------------------------------------
# `scale=` -- the widest cell: works (Color/Fill/Stroke/Size/Opacity family/
# StrokeWidth), n/a (Angle), and the two bold #123 cells (Shape, StrokeDash).
# ---------------------------------------------------------------------------


def test_scale_works_for_color_family() -> None:
    for cls in (fm.Color, fm.Fill):
        svg = _render(
            cls,
            scale=fm.OrdinalScale(domain=["a", "b", "c"], range=["#ff0000", "#00ff00", "#0000ff"]),
        )
        assert "#ff0000" in svg and "#00ff00" in svg and "#0000ff" in svg

    svg = _render(
        fm.Stroke,
        scale=fm.OrdinalScale(domain=["a", "b", "c"], range=["#ff0000", "#00ff00", "#0000ff"]),
    )
    colors = _circle_stroke_colors(svg)
    assert "#ff0000" in colors and "#00ff00" in colors and "#0000ff" in colors


def test_scale_works_for_size() -> None:
    base = _render(fm.Size)
    out = _render(fm.Size, scale=fm.LinearScale(range=[10.0, 900.0]))
    assert max(_circle_radii(out)) > max(_circle_radii(base)) * 2


_OPACITY_ATTR_BY_CHANNEL = {
    "opacity": lambda svg: _circle_fill_alphas(svg),
    "fill_opacity": lambda svg: _attr_values(svg, "fill-opacity"),
    "stroke_opacity": lambda svg: _attr_values(svg, "stroke-opacity"),
}


@pytest.mark.parametrize(
    "channel_cls", [fm.Opacity, fm.FillOpacity, fm.StrokeOpacity], ids=lambda c: c.__name__
)
def test_scale_works_for_opacity_family(channel_cls: type) -> None:
    """An explicit `scale=` domain override produces the exact known opacity values.

    `_DF`'s "n" column is `[1, 2, 3, 4, 5, 6]`; against
    `LinearScale(domain=[0, 20], range=[0, 1])` each row's opacity is exactly
    `n / 20`. Asserting the known values (not just "differs from the
    no-scale baseline") catches a wrong-but-different mapping, e.g. an
    inverted or mis-scaled domain that would also produce a changed, but
    incorrect, render (quality-review finding). `opacity` bakes its value
    into the mark's `rgba(...)` fill alpha, which rounds during hex/rgba
    conversion, so it is compared with a loose tolerance; `fill_opacity`/
    `stroke_opacity` write the raw attribute and match exactly. The legend
    is suppressed so only the marks' alphas are read: its swatches carry
    alphas of their own.
    """
    extract = _OPACITY_ATTR_BY_CHANNEL[channel_cls._channel_name]
    out_values = extract(
        _render(
            channel_cls,
            scale=fm.LinearScale(domain=[0.0, 20.0], range=[0.0, 1.0]),
            legend=None,
        )
    )
    expected = [n / 20.0 for n in _DF["n"].to_list()]
    assert out_values == pytest.approx(expected, rel=0.05)


def test_scale_works_for_stroke_width() -> None:
    """An explicit `scale=` maps each row's value onto its pixel width.

    `_DF`'s "n" column is `[1, ..., 6]`; against
    `LinearScale(domain=[0, 12], range=[0, 6])` each point's outline is
    exactly `n / 2` pixels wide. The legend is suppressed so only the marks'
    widths are read.
    """
    svg = _render(
        fm.StrokeWidth, scale=fm.LinearScale(domain=[0.0, 12.0], range=[0.0, 6.0]), legend=None
    )
    widths = [
        float(w)
        for w in re.findall(
            rf'<circle[^>]*stroke="{re.escape(_MARK_COLOR)}"[^>]*stroke-width="([0-9.]+)"', svg
        )
    ]
    assert widths == pytest.approx([n / 2.0 for n in _DF["n"].to_list()])


def test_a_scale_whose_domain_equals_its_range_draws_literal_pixel_widths() -> None:
    """A column of pixel widths draws as those widths through an identity scale, with no legend."""
    svg = _render(
        fm.StrokeWidth, scale=fm.LinearScale(domain=[1.0, 6.0], range=[1.0, 6.0]), legend=None
    )
    widths = re.findall(r'<circle[^>]*stroke-width="([0-9.]+)"', svg)
    assert [float(w) for w in widths] == _DF["n"].to_list()
    assert not _legend_title_present(svg, "n")


# The five channels whose scale is the numeric auxiliary linear scale.
_NUMERIC_AUX_CLASSES = (fm.Size, fm.Opacity, fm.FillOpacity, fm.StrokeOpacity, fm.StrokeWidth)

_GATE_REMEDY = "pass fm.LinearScale(domain=[...], range=[...]) or drop scale="

# Each malformed scale, in the class spelling where one exists and the dict
# spelling, with the reason the gate gives for it.
_MALFORMED_AUX_SCALES: dict[str, tuple[Any, str]] = {
    "log": (fm.LogScale(), f"type 'log' is not supported; {_GATE_REMEDY}"),
    "log-dict": ({"type": "log"}, f"type 'log' is not supported; {_GATE_REMEDY}"),
    "sqrt": (fm.SqrtScale(), f"type 'sqrt' is not supported; {_GATE_REMEDY}"),
    "pow": (fm.PowScale(), f"type 'pow' is not supported; {_GATE_REMEDY}"),
    "symlog": (fm.SymlogScale(), f"type 'symlog' is not supported; {_GATE_REMEDY}"),
    "padding": (
        fm.LinearScale(padding=0.1),
        f"padding is not supported (it insets an axis); {_GATE_REMEDY}",
    ),
    "padding-dict": (
        {"type": "linear", "padding": 0.1},
        f"padding is not supported (it insets an axis); {_GATE_REMEDY}",
    ),
    "scheme-dict": (
        {"type": "linear", "scheme": "blues"},
        f"scheme is not supported (it names a color scheme); {_GATE_REMEDY}",
    ),
    "domain-3": ({"domain": [1.0, 2.0, 3.0]}, "domain must have length 2; got 3"),
    "range-1": ({"range": [1.0]}, "range must have length 2; got 1"),
}


@pytest.mark.parametrize("scale_id", sorted(_MALFORMED_AUX_SCALES))
@pytest.mark.parametrize("channel_cls", _NUMERIC_AUX_CLASSES, ids=lambda c: c.__name__)
def test_a_numeric_aux_channel_refuses_a_scale_it_cannot_draw(
    channel_cls: type, scale_id: str
) -> None:
    """Size, the opacity family and StrokeWidth refuse one set of scales at construction.

    The text names the channel and the reason. Rejects a scale dropped or
    drawn linearly without a word, and three outcomes (refuse, warn,
    silent) for one mistake across the five.
    """
    scale, reason = _MALFORMED_AUX_SCALES[scale_id]
    with pytest.raises(ValueError) as exc:
        channel_cls("n", scale=scale)
    assert str(exc.value) == f"{channel_cls._channel_name} scale: {reason}"


@pytest.mark.parametrize("channel_cls", _NUMERIC_AUX_CLASSES, ids=lambda c: c.__name__)
def test_a_numeric_aux_channel_accepts_every_linear_field_it_draws(channel_cls: type) -> None:
    """Domain, range, reverse, nice, zero and clamp construct without a word."""
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        channel_cls(
            "n", scale=fm.LinearScale(domain=[0.0, 10.0], range=[0.2, 0.8], reverse=True, nice=True)
        )
        channel_cls("n", scale={"domain": [1.0, 5.0], "zero": True, "clamp": True})


def test_scale_na_for_angle() -> None:
    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.Angle("n", scale=fm.LinearScale())
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)


def test_shape_scale_domain_reorder_selects_the_requested_glyph_assignment() -> None:
    """Shape(scale=OrdinalScale(domain=[...])) reassigns which glyph each category gets.

    2-row, one-row-per-category fixture: unscaled, category "a" (first
    appearance) gets `SHAPE_PALETTE[0]` (circle) and "b" gets
    `SHAPE_PALETTE[1]` (square, a <rect>). Reordering the domain to
    ["b", "a"] must swap that assignment exactly -- not merely produce
    *some* different SVG.
    """
    unscaled = _mark_glyph_kinds(_render(fm.Shape, df=_DF2, legend=None))
    assert unscaled == ["circle", "square"]

    reordered = _mark_glyph_kinds(
        _render(fm.Shape, df=_DF2, legend=None, scale=fm.OrdinalScale(domain=["b", "a"]))
    )
    assert reordered == ["square", "circle"], (
        f"domain=['b','a'] must swap a<->b's glyphs; got {reordered}"
    )


def test_shape_scale_range_selects_the_requested_glyphs() -> None:
    """Shape(scale=OrdinalScale(range=[...])) draws the exact requested point shapes,
    one per category, in the requested order -- not merely *a* different render."""
    df3 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})
    svg = _render(
        fm.Shape,
        df=df3,
        legend=None,
        scale=fm.OrdinalScale(domain=["a", "b", "c"], range=["square", "cross", "diamond"]),
    )
    assert _mark_glyph_kinds(svg) == ["square", "cross", "diamond"]

    # A different range, same categories: the mapping tracks the range list,
    # not a fixed shape-per-category coincidence.
    svg2 = _render(
        fm.Shape,
        df=df3,
        legend=None,
        scale=fm.OrdinalScale(domain=["a", "b", "c"], range=["diamond", "square", "cross"]),
    )
    assert _mark_glyph_kinds(svg2) == ["diamond", "square", "cross"]


def test_shape_scale_invalid_range_entry_refused_eagerly() -> None:
    """Shape(scale={'range': ['bogus']}) refuses at construction naming Shape(scale=...)
    and the value received.

    Python-fixable (no Rust dependency): `Shape._validate` checks the range
    eagerly, via `ferrum._validate.validate_choice`, against
    `ferrum.marks.base._VALID_POINT_SHAPES`.
    """
    with pytest.raises(
        ValueError, match=r"Shape\(scale=\.\.\.\): range entry must be one of.*'bogus'"
    ):
        fm.Shape("g", scale={"type": "ordinal", "range": ["bogus"]})


def test_shape_scale_range_entry_unhashable_refuses_valueerror_not_typeerror() -> None:
    """A non-str range entry (e.g. a nested list) refuses ValueError, not a raw
    TypeError leaking from an unhashable `in` check (quality-review finding)."""
    with pytest.raises(ValueError, match=r"Shape\(scale=\.\.\.\): range entry must be one of"):
        fm.Shape("g", scale={"type": "ordinal", "range": [["nested"]]})


def test_stroke_dash_scale_domain_reorder_selects_the_requested_pattern_assignment() -> None:
    """StrokeDash(scale=OrdinalScale(domain=[...])) reassigns dash patterns.

    2-row, one-row-per-category fixture: unscaled, "a" gets the palette's
    solid entry (no `stroke-dasharray` attribute) and "b" gets the first
    dash pattern (`"6,3"`). Reordering the domain to ["b", "a"] must swap
    that assignment exactly.
    """
    unscaled = _dasharrays(_render(fm.StrokeDash, df=_DF2, legend=None))
    assert unscaled == [None, "6,3"]

    reordered = _dasharrays(
        _render(fm.StrokeDash, df=_DF2, legend=None, scale=fm.OrdinalScale(domain=["b", "a"]))
    )
    assert reordered == ["6,3", None], (
        f"domain=['b','a'] must swap a<->b's patterns; got {reordered}"
    )


def test_stroke_dash_scale_range_refused() -> None:
    """StrokeDash(scale={'range': [...]}) refuses at construction naming the fixed
    palette AND the value received.

    Python-fixable (no Rust dependency): `StrokeDash._validate` refuses any
    explicit `range=` key outright (#123, user ruling Q1b).
    """
    with pytest.raises(
        ValueError,
        match=r"StrokeDash\(scale=\.\.\.\): range=\[\[1, 1\], \[4, 4\]\] is not supported",
    ):
        fm.StrokeDash("g", scale={"type": "ordinal", "range": [[1, 1], [4, 4]]})


# ---------------------------------------------------------------------------
# FacetRow/FacetCol(columns=) -- flips from warn to refuse (#123, Q2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel_cls", [fm.FacetRow, fm.FacetCol], ids=lambda c: c.__name__)
def test_facet_row_col_columns_refused(channel_cls: type) -> None:
    name = channel_cls.__name__
    with pytest.raises(ValueError, match=rf"{name}\(columns=2\) is grid-mode-inapplicable"):
        channel_cls("f", columns=2)


# ---------------------------------------------------------------------------
# pairplot(markers=) -- the user-visible #123 consequence (Q1)
# ---------------------------------------------------------------------------


def test_pairplot_hue_color_mapping_is_x_blue_y_red_z_gold() -> None:
    """Vacuity control for the pairplot glyph test below: pins the hue-category
    to color mapping it depends on, using a plain color-encoded chart with no
    `markers=` involved -- so the mapping is independent of the shape scale
    under test."""
    df = pl.DataFrame({"a": [1.0, 2, 3], "b": [2.0, 1, 4], "h": ["x", "y", "z"]})
    svg = fm.Chart(df).mark_point().encode(x="a:Q", y="b:Q", color="h:N").to_svg()
    fills = re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]{6})"', svg)
    assert fills[:3] == ["#2563eb", "#dc2626", "#d4a017"]


def test_pairplot_markers_draw_the_requested_glyphs() -> None:
    """pairplot(markers=[...]) draws the exact requested glyph for each hue category.

    Checked per category (by that category's own, marker-independent color,
    pinned by `test_pairplot_hue_color_mapping_is_x_blue_y_red_z_gold` above)
    rather than by aggregate SVG-diff or glyph-tag counts: a tag-count
    comparison alone cannot tell "drew the right glyph" from "drew a
    different, also-different glyph" -- probed and confirmed both marker
    lists below draw 9 `<rect>` + 3 `<circle>` with only the `<path>` count
    differing (10 vs 14), so a wrong-but-different glyph assignment would
    also have passed a tag-count-only check (spec-review finding, cycle 2).
    """
    df = pl.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "b": [2.0, 1, 4, 3, 6, 5], "h": list("xyzxyz")})
    svg_square_cross_diamond = fm.pairplot(
        df, hue="h", markers=["square", "cross", "diamond"]
    ).to_svg()
    svg_triangle_diamond_square = fm.pairplot(
        df, hue="h", markers=["triangle-up", "diamond", "square"]
    ).to_svg()

    # x -> #2563eb: "square" (markers[0]) in the first list, "triangle" (markers[0]) in the second.
    assert "square" in _mark_glyph_kinds(svg_square_cross_diamond, "#2563eb")
    assert "triangle" in _mark_glyph_kinds(svg_triangle_diamond_square, "#2563eb")
    assert "triangle" not in _mark_glyph_kinds(svg_square_cross_diamond, "#2563eb")
    assert "square" not in _mark_glyph_kinds(svg_triangle_diamond_square, "#2563eb")

    # y -> #dc2626: "cross" (markers[1]) in the first list, "diamond" (markers[1]) in the second.
    assert "cross" in _mark_glyph_kinds(svg_square_cross_diamond, "#dc2626")
    assert "diamond" in _mark_glyph_kinds(svg_triangle_diamond_square, "#dc2626")
    assert "diamond" not in _mark_glyph_kinds(svg_square_cross_diamond, "#dc2626")
    assert "cross" not in _mark_glyph_kinds(svg_triangle_diamond_square, "#dc2626")

    # z -> #d4a017: "diamond" (markers[2]) in the first list, "square" (markers[2]) in the second.
    assert "diamond" in _mark_glyph_kinds(svg_square_cross_diamond, "#d4a017")
    assert "square" in _mark_glyph_kinds(svg_triangle_diamond_square, "#d4a017")
    assert "square" not in _mark_glyph_kinds(svg_square_cross_diamond, "#d4a017")
    assert "diamond" not in _mark_glyph_kinds(svg_triangle_diamond_square, "#d4a017")
