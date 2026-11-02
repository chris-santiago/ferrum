"""A declared line's value channel is anchored at zero the way a bar's is.

``rise_from_zero`` is a producer-only ``mark_line`` style flag -- set only by
``displot(kind='ecdf')`` -- held off every public door by
``ferrum.marks.base._INTERNAL_STYLE_DOORS``, exactly as ``sort_vertices`` is.
No public call accepts it, so every declared chart here is built the way the
ecdf desugar itself sets an internal style field (``plots/distribution.py``'s
``chart._mark_kwargs = merge_style(chart._mark_kwargs, {"sort_vertices":
True})``): by patching ``_mark_kwargs``/``mark_kwargs`` directly with
:func:`ferrum._layer.merge_style`.

The one producer of the anchor, ``ferrum._spec_build._mark_zero_anchor``,
anchors a bar or a declared line on every door that reads an authored
encoding: a flat chart, a faceted chart, an independent-y layer (a
``SecondaryY`` slot or a ``LayerChart(resolve={"y": "independent"})``
member), and a composite leaf built from either. A layer on shared axes is
anchored by Rust's own layered rule instead (``numeric_domain_union_over``);
that route appears here only in the CoordFlip agreement tests, which check
that the two producers anchor the same drawn axis.

The anchor always lands on the AUTHORED channel that holds the value (the
declared ``value_axis`` style kwarg when present, else ``y``): a
``CoordFlip`` moves where the authored channels draw, not which one holds
the value, so the flip is never read while deciding which channel to anchor
-- the same split :func:`ferrum._spec_build._bar_value_on_x` keeps for a
bar.
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm
from ferrum._layer import _Layer, merge_style
from ferrum.encoding.positional import Y
from ferrum.structural import SecondaryY

from _snapshots import assert_svg_eq
from _svg_extents import value_axis_ladders
from _svg_rects import step_after_paths

_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
_FACETED_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "g": ["a", "a", "a", "b", "b", "b"],
    }
)
_COMPANION_DF = pl.DataFrame({"x": [1.0, 2.0], "y": [5.0, 6.0]})


def _declare(chart: fm.Chart, **extra: object) -> fm.Chart:
    """Patch *chart*'s own mark kwargs to declare ``rise_from_zero=True``.

    The one production mechanism (:func:`merge_style`) that sets an
    internal-only ``mark_style`` field, mirroring the ecdf desugar's own
    ``sort_vertices`` declaration (see module docstring) -- no public door
    accepts ``rise_from_zero`` directly.
    """
    chart._mark_kwargs = merge_style(chart._mark_kwargs, {"rise_from_zero": True, **extra})
    return chart


def _declare_false(chart: fm.Chart) -> fm.Chart:
    """Patch *chart*'s own mark kwargs to declare ``rise_from_zero=False``.

    Through the same :func:`merge_style` mechanism as :func:`_declare`: the
    ``Some(false)``/unset sibling route the Cases line names, which a
    producer keyed on the flag's mere presence (rather than its truth)
    would still anchor.
    """
    chart._mark_kwargs = merge_style(chart._mark_kwargs, {"rise_from_zero": False})
    return chart


def _y_scale(chart: fm.Chart) -> dict | None:
    spec = json.loads(chart.to_spec().to_json())
    return spec["encoding"]["y"].get("scale")


def _secondary_layer_y_scale(chart: fm.Chart) -> dict | None:
    spec = json.loads(chart.to_spec().to_json())
    (secondary,) = [layer for layer in spec["layers"] if layer["mark"] == "line"]
    return secondary["encoding"]["y"].get("scale")


def _riser_starts_at_zero(svg: str, zero_pixels: set, *, axis: str) -> None:
    """Every declared step-after line's riser (its traced path's first
    corner) starts at one of *zero_pixels*."""
    paths = step_after_paths(svg)
    assert paths, "no declared step-after line drawn"
    coord = 0 if axis == "x" else 1
    for _stroke, corners in paths:
        assert corners, "a declared line draws no corners"
        first = corners[0][coord]
        assert any(abs(first - zp) < 1e-6 for zp in zero_pixels), (
            f"riser does not start at the {axis} axis's 0 pixel: {first} not in {zero_pixels}"
        )


def _assert_riser_from_zero(svg: str, *, axis: str = "y") -> None:
    """A declared line's own *axis* rises from 0.

    Reads every drawn *axis* tick ladder and keeps the ones with a 0 tick --
    the declared line's own; an unrelated companion axis (a composite's
    other leaf, an independent-y layer's primary axis) draws none, so it is
    excluded rather than asserted against. At least one ladder must have a
    0 tick, and every drawn riser must start there.
    """
    ladders = value_axis_ladders(svg, axis=axis)
    assert ladders, f"no {axis} tick ladder drawn"
    zero_pixels = {ladder[0.0] for _, ladder in ladders if 0.0 in ladder}
    assert zero_pixels, f"no {axis} axis draws a 0 tick: {[sorted(lad) for _, lad in ladders]}"
    _riser_starts_at_zero(svg, zero_pixels, axis=axis)


def _assert_every_panel_rises_from_zero(svg: str, *, axis: str = "y") -> None:
    """Every panel of a faceted declared line rises from 0: every drawn
    *axis* ladder has a 0 tick (not merely one of them), and every panel's
    own riser starts there."""
    ladders = value_axis_ladders(svg, axis=axis)
    assert len(ladders) >= 2, (
        f"expected more than one {axis} ladder (a faceted chart); got {ladders}"
    )
    for _, ladder in ladders:
        assert 0.0 in ladder, f"the {axis} axis draws no 0 tick: {sorted(ladder)}"
    zero_pixels = {ladder[0.0] for _, ladder in ladders}
    paths = step_after_paths(svg)
    assert len(paths) >= 2, f"expected one riser per panel; got {len(paths)} declared lines"
    _riser_starts_at_zero(svg, zero_pixels, axis=axis)


def _assert_no_zero_tick(svg: str, *, axis: str) -> None:
    """No drawn *axis* tick ladder carries a 0 tick -- the unanchored axis
    under a one-sided anchor (the fixture data is always 1..3 or higher, so
    a 0 tick here can only be the anchor, never the data's own extent)."""
    ladders = value_axis_ladders(svg, axis=axis)
    assert ladders, f"no {axis} tick ladder drawn"
    assert all(0.0 not in ladder for _, ladder in ladders), (
        f"the {axis} axis should carry no 0 tick: {[sorted(lad) for _, lad in ladders]}"
    )


# ── door: a flat chart ───────────────────────────────────────────────────────


def _flat(*, declared: bool, y=None) -> fm.Chart:
    chart = fm.Chart(_DF).mark_line(interpolate="step-after").encode(x="x", y=y or "y")
    return _declare(chart) if declared else chart


def _flat_horizontal(*, declared: bool) -> fm.Chart:
    """The value channel is x: x holds the count-like value, y advances."""
    chart = fm.Chart(_DF).mark_line(interpolate="step-after").encode(x="y", y="x")
    return _declare(chart, value_axis="x") if declared else chart


def _flat_flipped(*, declared: bool, value_axis: str | None = None) -> fm.Chart:
    chart = (
        fm.Chart(_DF).mark_line(interpolate="step-after").encode(x="x", y="y").coord(fm.CoordFlip())
    )
    if not declared:
        return chart
    extra = {"value_axis": value_axis} if value_axis else {}
    return _declare(chart, **extra)


def test_flat_chart_rises_from_zero() -> None:
    _assert_riser_from_zero(_flat(declared=True).to_svg(), axis="y")


def test_flat_chart_explicit_domain_wins() -> None:
    y = fm.Y("y", scale={"domain": [0.5, 5.0]})
    scale = _y_scale(_flat(declared=True, y=y))
    assert scale is not None
    assert scale.get("domain") == [0.5, 5.0]
    assert "zero" not in scale or scale.get("zero") is not True
    assert "systemInjected" not in scale


def test_flat_chart_explicit_zero_false_equivalent_wins() -> None:
    """``fm.LinearScale`` takes no ``zero=`` constructor kwarg, so the
    caller's equivalent of ``zero=False`` on a channel's own scale is a raw
    scale dict carrying ``"zero": False`` -- the same raw-dict door
    ``scale=`` already accepts for ``domain=``/every other leaf
    (``ferrum.encoding._scale._scale_to_dict``).
    """
    y = fm.Y("y", scale={"zero": False})
    scale = _y_scale(_flat(declared=True, y=y))
    assert scale is not None
    assert scale.get("zero") is False
    assert "systemInjected" not in scale


def test_flat_horizontal_line_anchors_x() -> None:
    _assert_riser_from_zero(_flat_horizontal(declared=True).to_svg(), axis="x")


def test_flat_undeclared_line_draws_no_scale() -> None:
    assert _y_scale(_flat(declared=False)) is None


def test_flat_rise_from_zero_false_is_byte_identical_to_unset() -> None:
    declared_false = _declare_false(_flat(declared=False)).to_svg()
    reference = _flat(declared=False).to_svg()
    assert_svg_eq(
        declared_false,
        reference,
        name="flat rise_from_zero=False vs unset",
        regen_hint="rise_from_zero=False must render identically to unset; "
        "check _zero_anchor_target's mark_kwargs.get('rise_from_zero') is True gate",
    )


def test_flat_chart_under_coord_flip_rises_from_zero_on_drawn_x() -> None:
    """The value_axis is undeclared, so the authored value channel is y;
    CoordFlip draws y on screen x, so the anchor and the riser land there,
    and screen y (the authored observation channel) carries no 0 tick."""
    svg = _flat_flipped(declared=True).to_svg()
    _assert_riser_from_zero(svg, axis="x")
    _assert_no_zero_tick(svg, axis="y")


# ── door: a faceted chart ────────────────────────────────────────────────────


def _faceted(*, declared: bool) -> fm.Chart:
    chart = (
        fm.Chart(_FACETED_DF)
        .mark_line(interpolate="step-after")
        .encode(x="x", y="y")
        .facet(col="g")
    )
    return _declare(chart) if declared else chart


def test_faceted_chart_rises_from_zero() -> None:
    _assert_every_panel_rises_from_zero(_faceted(declared=True).to_svg(), axis="y")


def test_faceted_chart_explicit_domain_wins() -> None:
    y = fm.Y("y", scale={"domain": [0.5, 10.0]})
    chart = (
        fm.Chart(_FACETED_DF).mark_line(interpolate="step-after").encode(x="x", y=y).facet(col="g")
    )
    scale = _y_scale(_declare(chart))
    assert scale is not None
    assert scale.get("domain") == [0.5, 10.0]
    assert "zero" not in scale or scale.get("zero") is not True
    assert "systemInjected" not in scale


def test_faceted_chart_explicit_zero_false_equivalent_wins() -> None:
    y = fm.Y("y", scale={"zero": False})
    chart = (
        fm.Chart(_FACETED_DF).mark_line(interpolate="step-after").encode(x="x", y=y).facet(col="g")
    )
    scale = _y_scale(_declare(chart))
    assert scale is not None
    assert scale.get("zero") is False
    assert "systemInjected" not in scale


def test_faceted_horizontal_line_anchors_x() -> None:
    """A row facet, not column: `value_axis_ladders` keys an x ladder by its
    tick row's shared y position, which a COLUMN facet's side-by-side panels
    collide on (`panel_x_ladder`'s docstring); a ROW facet's panels each
    keep their own x tick row."""
    chart = (
        fm.Chart(_FACETED_DF)
        .mark_line(interpolate="step-after")
        .encode(x="y", y="x")
        .facet(row="g")
    )
    svg = _declare(chart, value_axis="x").to_svg()
    _assert_every_panel_rises_from_zero(svg, axis="x")


def test_faceted_undeclared_line_draws_no_scale() -> None:
    assert _y_scale(_faceted(declared=False)) is None


def test_faceted_rise_from_zero_false_is_byte_identical_to_unset() -> None:
    declared_false = _declare_false(_faceted(declared=False)).to_svg()
    reference = _faceted(declared=False).to_svg()
    assert_svg_eq(
        declared_false,
        reference,
        name="faceted rise_from_zero=False vs unset",
        regen_hint="rise_from_zero=False must render identically to unset; "
        "check _zero_anchor_target's mark_kwargs.get('rise_from_zero') is True gate",
    )


def test_faceted_chart_under_coord_flip_rises_from_zero_on_drawn_x() -> None:
    """A row facet, not column, for the same tick-row collision reason as
    `test_faceted_horizontal_line_anchors_x`."""
    chart = (
        fm.Chart(_FACETED_DF)
        .mark_line(interpolate="step-after")
        .encode(x="x", y="y")
        .facet(row="g")
        .coord(fm.CoordFlip())
    )
    svg = _declare(chart).to_svg()
    _assert_every_panel_rises_from_zero(svg, axis="x")
    _assert_no_zero_tick(svg, axis="y")


# ── door: an independent-y layer ─────────────────────────────────────────────


def _independent_y(*, declared: bool, y=None, step_after: bool = True) -> fm.Chart:
    base = fm.Chart(_DF).mark_point().encode(x="x", y="x")
    style: dict = {"interpolate": "step-after"} if step_after else {}
    if declared:
        style["rise_from_zero"] = True
    secondary = _Layer(
        mark="line",
        encoding={"y": y or Y("y")},
        mark_kwargs=merge_style({}, style) if style else None,
        independent_y=True,
        positional_source="written",
        chart_data=True,
    )
    return base.layer(secondary)


def test_independent_y_rises_from_zero() -> None:
    _assert_riser_from_zero(_independent_y(declared=True).to_svg(), axis="y")


def test_independent_y_explicit_domain_wins() -> None:
    y = Y("y", scale={"domain": [0.5, 5.0]})
    scale = _secondary_layer_y_scale(_independent_y(declared=True, y=y))
    assert scale is not None
    assert scale.get("domain") == [0.5, 5.0]
    assert "zero" not in scale or scale.get("zero") is not True
    assert "systemInjected" not in scale


def test_independent_y_explicit_zero_false_equivalent_wins() -> None:
    y = Y("y", scale={"zero": False})
    scale = _secondary_layer_y_scale(_independent_y(declared=True, y=y))
    assert scale is not None
    assert scale.get("zero") is False
    assert "systemInjected" not in scale


def test_independent_y_undeclared_layer_draws_no_scale() -> None:
    """An undeclared secondary line's own y channel carries no injected
    scale at all, whatever its other mark kwargs: the producer only ever
    anchors a line whose own mark kwargs carry `rise_from_zero=True`."""
    assert _secondary_layer_y_scale(_independent_y(declared=False)) is None


def test_independent_y_undeclared_layer_is_byte_identical() -> None:
    """With no mark-kwargs override at all (what the public `SecondaryY`
    door can build, since it has no `interpolate=` knob), the internal
    `_Layer` builder used above -- undeclared -- matches it byte-for-byte."""
    declared_off = _independent_y(declared=False, step_after=False).to_svg()
    base = fm.Chart(_DF).mark_point().encode(x="x", y="x")
    reference = (base + SecondaryY(mark="line", field="y")).to_svg()
    assert_svg_eq(
        declared_off,
        reference,
        name="independent-y undeclared _Layer vs SecondaryY",
        regen_hint="the internal _Layer builder must match SecondaryY's public door "
        "when neither declares rise_from_zero; check _independent_y's style dict",
    )


def test_independent_y_under_coord_flip_anchors_drawn_x_in_the_spec() -> None:
    """Render refuses `SecondaryY` under a flipped chart coord (its right y
    axis has no place once the coord moves y off the vertical axis), so this
    is spec-level only: the secondary layer's own authored y channel (its
    declared value) still carries the anchor, and its x carries none."""
    base = fm.Chart(_DF).mark_point().encode(x="x", y="x").coord(fm.CoordFlip())
    chart = base.layer(
        _Layer(
            mark="line",
            encoding={"y": Y("y")},
            mark_kwargs=merge_style({}, {"interpolate": "step-after", "rise_from_zero": True}),
            independent_y=True,
            positional_source="written",
            chart_data=True,
        )
    )
    y_scale = _secondary_layer_y_scale(chart)
    assert y_scale is not None
    assert y_scale.get("zero") is True
    assert y_scale.get("systemInjected") is True
    spec = json.loads(chart.to_spec().to_json())
    (secondary,) = [layer for layer in spec["layers"] if layer["mark"] == "line"]
    assert secondary["encoding"]["x"].get("scale") is None


# ── door: a composite (concat) leaf ──────────────────────────────────────────


def _companion() -> fm.Chart:
    return fm.Chart(_COMPANION_DF).mark_point().encode(x="x", y="y")


def test_composite_leaf_rises_from_zero() -> None:
    svg = fm.hconcat(_flat(declared=True), _companion()).to_svg()
    _assert_riser_from_zero(svg, axis="y")


def test_composite_leaf_explicit_domain_wins() -> None:
    y = fm.Y("y", scale={"domain": [0.5, 5.0]})
    leaf, _ = fm.hconcat(_flat(declared=True, y=y), _companion()).charts
    scale = _y_scale(leaf)
    assert scale is not None
    assert scale.get("domain") == [0.5, 5.0]
    assert "zero" not in scale or scale.get("zero") is not True
    assert "systemInjected" not in scale


def test_composite_leaf_explicit_zero_false_equivalent_wins() -> None:
    y = fm.Y("y", scale={"zero": False})
    leaf, _ = fm.hconcat(_flat(declared=True, y=y), _companion()).charts
    scale = _y_scale(leaf)
    assert scale is not None
    assert scale.get("zero") is False
    assert "systemInjected" not in scale


def test_composite_leaf_horizontal_line_anchors_x() -> None:
    svg = fm.hconcat(_flat_horizontal(declared=True), _companion()).to_svg()
    _assert_riser_from_zero(svg, axis="x")


def test_composite_leaf_undeclared_line_draws_no_scale() -> None:
    leaf, _ = fm.hconcat(_flat(declared=False), _companion()).charts
    assert _y_scale(leaf) is None


def test_composite_leaf_rise_from_zero_false_is_byte_identical_to_unset() -> None:
    declared_false_leaf = _declare_false(_flat(declared=False))
    declared_false = fm.hconcat(declared_false_leaf, _companion()).to_svg()
    reference = fm.hconcat(_flat(declared=False), _companion()).to_svg()
    assert_svg_eq(
        declared_false,
        reference,
        name="composite leaf rise_from_zero=False vs unset",
        regen_hint="rise_from_zero=False must render identically to unset; "
        "check _zero_anchor_target's mark_kwargs.get('rise_from_zero') is True gate",
    )


def test_composite_leaf_under_coord_flip_rises_from_zero_on_drawn_x() -> None:
    svg = fm.hconcat(_flat_flipped(declared=True), _companion()).to_svg()
    _assert_riser_from_zero(svg, axis="x")
    _assert_no_zero_tick(svg, axis="y")


# ── CoordFlip: the flat producer and Rust's layered rule agree ──────────────


def _layered_twin(line_chart: fm.Chart) -> fm.Chart:
    """The same declared line, layered with a point on shared axes -- Rust's
    own layered rule (`numeric_domain_union_over`) anchors this route,
    never `_mark_zero_anchor`."""
    point = fm.Chart(_DF).mark_point().encode(x="x", y="y")
    return line_chart + point


def test_flat_and_layered_producers_agree_under_one_flip_undeclared_orientation() -> None:
    """A declared line with no `value_axis` (an undeclared orientation) has
    its value on authored y; CoordFlip draws y on screen x. The flat chart
    (Python's producer) and its `+` twin (Rust's layered rule) must anchor
    the same drawn axis and start the riser there."""
    flat_svg = _flat_flipped(declared=True).to_svg()
    layered_svg = _layered_twin(_flat(declared=True)).coord(fm.CoordFlip()).to_svg()
    for svg in (flat_svg, layered_svg):
        _assert_riser_from_zero(svg, axis="x")
        _assert_no_zero_tick(svg, axis="y")


def test_flat_and_layered_producers_agree_under_one_flip_declared_value_axis_x() -> None:
    """A declared `value_axis='x'` line's value is its authored x; CoordFlip
    draws x on screen y. The flat chart and its `+` twin must agree, same as
    the undeclared-orientation pair above."""
    flat_svg = _flat_flipped(declared=True, value_axis="x").to_svg()
    line = _declare(_flat(declared=False), value_axis="x")
    layered_svg = _layered_twin(line).coord(fm.CoordFlip()).to_svg()
    for svg in (flat_svg, layered_svg):
        _assert_riser_from_zero(svg, axis="y")
        _assert_no_zero_tick(svg, axis="x")
