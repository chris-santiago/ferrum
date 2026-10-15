"""`Facet(sort=)` / malformed-sort-object matrix (#176(11) + N3, spec §6.4).

Four matrices:

**Panel-order matrix** (``test_facet_sort_panel_order``): facet door {wrap
``Facet``, grid ``FacetRow``, grid ``FacetCol``} x sort form {none,
``"ascending"``, ``"descending"``, explicit list, ``{field, op}`` object} ->
panel order read from the rendered strip text. The grid doors hold the
UNTESTED dimension to a single constant value (``"other"`` is always
``"z"``), the same isolation trick ``test_facet_merge_matrix.py``'s
``_COL_STRIP_DF``/``_ROW_STRIP_DF`` fixtures use, so the strip-text list for
the tested dimension has exactly one entry per category with no repeats.
Every sort form's expected order is independently computed from
``_SORT_DF`` (never by calling into the renderer), and the five expected
orders are pairwise distinct -- the matrix's vacuity control: a cell that
renders the wrong order fails, not passes vacuously. Rendering the
requested order is the Rust ordinal-domain sort resolution reused at the
facet partition (``FacetSpec.sort``/``row_sort``); ``none`` needs no sort
resolution to render in data order, and is the only cell that does not
exercise it.

**Degraded-but-well-formed matrix**
(``test_facet_degraded_sort_warns_like_x_not_silently``): a facet `sort=`
that is well-formed enough to reach the ordinal-domain sort resolution but
cannot be applied (an unsupported type, an unrecognized keyword string, or
an object naming a column absent from the data) warns exactly once with the
same reason `X(sort=...)` produces for the identical value, wrapped in
`RenderWarning::FacetSortSpecIgnored`'s own sentence -- ``"facet sort spec
could not be applied ({reason}); panels fall back to insertion order"`` --
rather than rendering silently in data order.

**Wire-door matrix** (``test_wire_malformed_sort_refuses_at_render`` +
its vacuity control ``test_wire_well_formed_sort_renders``, plus
``test_wire_facet_row_sort_without_row_refuses`` for the row-dimension-less
malformation): a hand-built spec (``ChartSpec.from_json``) is the one route
that bypasses every construction-time gate (Python channel or
`EncodingSpec`). Four doors name every place a sort value sits on the wire
-- a chart-level channel, a layer's own channel, a facet's `sort`, a
facet's `row_sort` -- crossed with both output kinds (static/interactive).
A malformed sort reaching any of these refuses with
`RenderError::SortSpecMalformed`'s Display -- ``"{location}: sort={value}
is malformed ({reason})"`` -- naming the location (`"x"`,
`"layers[1].x"`, `"facet.sort"`, `"facet.row_sort"`) and the value; the
same four doors with a well-formed sort (or none) render without raising.
`FacetSpec.row_sort` set while `row` is absent is a second, distinct
malformation sharing the same variant, with its own reason: ``"the facet
has no row dimension (row=) for row_sort to order"``. A separate pair of
tests (``test_row_sort_drops_on_grid_to_wrap_redeclaration`` +
``test_row_sort_grid_contradiction_still_refuses_not_a_stale_wrap``) pins
that this malformation is unreachable through any legitimate Python merge
sequence: a grid's `row_sort` drops when a redeclaration lowers it to wrap,
exactly as `row_title` already does.

**Malformed-sort-object matrix** (``test_malformed_sort_object_refuses`` +
``test_encoding_spec_malformed_sort_refuses``): malformed door x malformed
form {no ``field``, non-string ``field``, unknown ``op``, unknown
``order``, ``op`` present as an explicit ``None``, ``order`` present as an
explicit ``None``} -> exact ``ValueError``. A key that is *absent* means
"use the default" (well-formed, not in this population); a key that is
*present* with an explicit ``None`` is a malformed value, refused the same
as any other value outside the accepted vocabulary. Doors are derived,
never hand-listed: every channel class in ``ferrum.encoding.__all__`` whose
``_honored_kwargs`` contains ``"sort"`` (X, Y, Color, Shape, StrokeDash,
Facet, FacetRow, FacetCol), plus ``EncodingSpec``, tested separately since
its message text is Rust-owned.

**Controls**: a valid ``X(sort=...)`` form on an ordinal ``x`` renders its
own, independently-computed category order
(``test_valid_x_sort_forms_render_the_expected_order``), and its SVG and
wire both hash byte-identical against a capture from a BOTH-LANGUAGES
build of commit 892ea373 (the tip before this task) --
``src/ferrum/encoding/base.py`` and ``crates/ferrum-core/src/render/scale_resolve/domain.rs``
both reverted, since this task changed the render-time object-form parse
(``apply_sort_to_domain``/``parse_sort_field_object``) as well as the
Python construction-time gate, and only reverting the latter cannot
isolate this task's full effect. A companion check
(``test_valid_x_sort_forms_produce_distinct_svgs``) pins that the five
forms render at least four distinct SVGs among themselves, so the fixture
cannot silently regress to an axis sort has no effect on. A facet with no
``sort=`` declared omits both ``sort`` and ``row_sort`` from the wire,
byte-identical to before this task. Constructing
``Facet``/``FacetRow``/``FacetCol(sort=...)`` never emits the generic "not
yet honored" warning.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding.base import ChannelBase

from _svg_extents import assert_bars_start_on_the_zero_tick

# ---------------------------------------------------------------------------
# Panel-order matrix
# ---------------------------------------------------------------------------

# Insertion order of `cat`'s distinct values is banana, apple, cherry --
# deliberately not alphabetical, so "none" (data order) is distinguishable
# from "ascending". `v` is chosen so the aggregate-object sort form's order
# (by mean(v) per category) differs from every other form's order too:
# mean(v) is apple=4, cherry=5, banana=5.5 -> ["apple", "cherry", "banana"].
_SORT_DF = pl.DataFrame(
    {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
        "cat": ["banana", "apple", "cherry", "banana", "apple", "cherry"],
        "other": ["z", "z", "z", "z", "z", "z"],
        "v": [10.0, 5.0, 1.0, 1.0, 3.0, 9.0],
    }
)

_CAT_VALUES = frozenset({"apple", "banana", "cherry"})


def _base() -> "fm.Chart":
    return fm.Chart(_SORT_DF).mark_point().encode(x="x", y="y")


def _strip_order(svg: str, *, axis: str) -> list[str]:
    """Distinct `cat` strip-text values, in first-seen render order, read
    from the column strip (``axis="column"``, no ``rotate(90`` glyph) or the
    row strip (``axis="row"``, identified by ``rotate(90``) -- mirrors
    ``test_facet_merge_matrix.py``'s ``_axis_strip_texts`` helper."""
    import re

    out: list[str] = []
    for attrs, text in re.findall(r"<text([^>]*)>([^<]*)</text>", svg):
        if 'font-size="12"' not in attrs or "font-weight" in attrs:
            continue  # not a strip label
        is_row = "rotate(90" in attrs
        if (axis == "row") != is_row:
            continue
        if text in _CAT_VALUES and text not in out:
            out.append(text)
    return out


DOORS: dict[str, tuple[str, Callable[[Any], "fm.Chart"]]] = {
    "wrap_facet": ("column", lambda sort: _base().encode(facet=fm.Facet("cat", sort=sort))),
    "grid_facet_row": (
        "row",
        lambda sort: _base().encode(
            facet_row=fm.FacetRow("cat", sort=sort), facet_col=fm.FacetCol("other")
        ),
    ),
    "grid_facet_col": (
        "column",
        lambda sort: _base().encode(
            facet_col=fm.FacetCol("cat", sort=sort), facet_row=fm.FacetRow("other")
        ),
    ),
    # The `Chart.facet()` door, one row per declaration shape it routes.
    "method_field": ("column", lambda sort: _base().facet(field="cat", sort=sort)),
    "method_col": ("column", lambda sort: _base().facet(col="cat", sort=sort)),
    # A row-alone facet lowers to a wrap, whose one strip is the column strip.
    "method_row": ("column", lambda sort: _base().facet(row="cat", sort=sort)),
    "method_grid_row": (
        "row",
        lambda sort: _base().facet(row="cat", col="other", row_sort=sort),
    ),
    "method_grid_col": (
        "column",
        lambda sort: _base().facet(row="other", col="cat", sort=sort),
    ),
}

SORT_FORMS: dict[str, tuple[Any, list[str]]] = {
    "none": (None, ["banana", "apple", "cherry"]),
    "ascending": ("ascending", ["apple", "banana", "cherry"]),
    "descending": ("descending", ["cherry", "banana", "apple"]),
    "explicit_list": (["banana", "cherry", "apple"], ["banana", "cherry", "apple"]),
    "sort_object": ({"field": "v", "op": "mean"}, ["apple", "cherry", "banana"]),
}

# The five expected orders are pairwise distinct -- the vacuity control: a
# cell rendering data order (or any other wrong order) is caught by every
# non-"none" form, not just one.
assert len({tuple(order) for _, order in SORT_FORMS.values()}) == len(SORT_FORMS)

PANEL_ORDER_CELLS = [(door, form) for door in DOORS for form in SORT_FORMS]


@pytest.mark.parametrize(
    ("door_name", "form_name"),
    PANEL_ORDER_CELLS,
    ids=[f"{d}-{f}" for d, f in PANEL_ORDER_CELLS],
)
def test_facet_sort_panel_order(door_name: str, form_name: str) -> None:
    axis, build = DOORS[door_name]
    sort_value, expected = SORT_FORMS[form_name]
    svg = build(sort_value).to_svg()
    assert _strip_order(svg, axis=axis) == expected


# ---------------------------------------------------------------------------
# Degraded-but-well-formed facet sort: warns like X, never silently inert.
# ---------------------------------------------------------------------------

# Each value is well-formed enough to reach `apply_sort_to_domain`/
# `apply_sort_field_object` (never malformed-shape-refused at construction)
# but cannot be applied: an unsupported JSON type, an unrecognized keyword
# string, or a sort-field object naming a column that is not in the data.
# Reasons are read live off `X(sort=...)`, which shares the identical
# warning-generation code (`apply_sort_to_domain` is the one ordinal-domain
# sort resolution both channels call -- there is no facet-local sorter); the
# facet partition's own read of `ctx.category_field` succeeds for all three
# (the category column `"cat"` exists), so only the value-field/spec-shape
# clause of the reason varies, matching X's text exactly.
DEGRADED_SORT_FORMS: dict[str, Any] = {
    "unsupported_type": 5,
    "unrecognized_string": "bogus",
    "unresolvable_field": {"field": "nope", "op": "sum"},
}

_DEGRADED_REASONS: dict[str, str] = {
    "unsupported_type": "unsupported sort spec 5",
    "unrecognized_string": 'unrecognized sort spec "bogus"',
    "unresolvable_field": (
        "cannot read sort value field \"nope\": unknown column 'nope' referenced by an encoding"
    ),
}


def _facet_sort_ignored_message(reason: str) -> str:
    """`RenderWarning::FacetSortSpecIgnored`'s Display, a distinct variant
    from the positional axis's `SortSpecIgnored` (so the panel-loop's
    provisional-warning dedup, which only re-derives axis sort warnings, can
    exclude it): `"facet sort spec could not be applied ({reason}); panels
    fall back to insertion order"`."""
    return f"facet sort spec could not be applied ({reason}); panels fall back to insertion order"


@pytest.mark.parametrize("form_name", sorted(DEGRADED_SORT_FORMS), ids=sorted(DEGRADED_SORT_FORMS))
def test_facet_degraded_sort_warns_like_x_not_silently(form_name: str) -> None:
    """A degraded-but-well-formed `Facet(sort=...)` warns exactly once, with
    the same reason `X(sort=...)` produces for the identical value (wrapped
    in `FacetSortSpecIgnored`'s own sentence, not `SortSpecIgnored`'s),
    instead of rendering silently in data order."""
    import warnings

    value = DEGRADED_SORT_FORMS[form_name]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _base().encode(facet=fm.Facet("cat", sort=value)).to_svg()
    messages = [str(ww.message) for ww in w if "could not be applied" in str(ww.message)]
    assert messages == [_facet_sort_ignored_message(_DEGRADED_REASONS[form_name])]


# ---------------------------------------------------------------------------
# Wire-door refusal: a hand-built spec carrying a malformed sort object.
# ---------------------------------------------------------------------------

# Every route above constructs through a Python channel or `EncodingSpec`,
# which validate. `ChartSpec.from_json` is the one bypass: a raw wire dict
# can carry a malformed sort object no construction-time gate ever saw. Four
# doors name every location a sort value can sit on the wire.
_WIRE_MALFORMED_SORT = {"op": "sum"}  # well-formed shape minus a "field" -- malformed
_WIRE_MALFORMED_REASON = 'sort object missing string "field"'

# `RenderError::SortSpecMalformed`'s Display:
# `"{location}: sort={value} is malformed ({reason})"`. `value` is the JSON
# value's own compact serde Display (matching `json.dumps(..., separators=
# (",", ":"))`, confirmed live against the rebuilt extension -- not guessed).
_WIRE_DOOR_LOCATIONS: dict[str, str] = {
    "chart_channel": "x",
    "layer_channel": "layers[1].x",
    "facet_sort": "facet.sort",
    "facet_row_sort": "facet.row_sort",
}


def _wire_refusal_message(location: str, value: dict, reason: str) -> str:
    rendered = json.dumps(value, separators=(",", ":"))
    return f"{location}: sort={rendered} is malformed ({reason})"


def _render_from_wire(wire: dict, *, output_kind: str) -> None:
    from ferrum._core import ChartSpec, render_interactive, render_svg

    spec = ChartSpec.from_json(json.dumps(wire))
    renderer = render_svg if output_kind == "static" else render_interactive
    renderer(spec, _SORT_DF.to_arrow(), viewport=(400.0, 300.0))


def _chart_channel_wire() -> dict:
    wire = json.loads(_base().to_json())
    wire["encoding"]["x"]["sort"] = dict(_WIRE_MALFORMED_SORT)
    return wire


def _layer_channel_wire() -> dict:
    chart = fm.Chart(_SORT_DF).mark_point().encode(x="x", y="y") + fm.Chart(
        _SORT_DF
    ).mark_line().encode(x="x", y="y")
    wire = json.loads(chart.to_json())
    wire["layers"][1]["encoding"]["x"]["sort"] = dict(_WIRE_MALFORMED_SORT)
    return wire


def _facet_sort_wire() -> dict:
    wire = json.loads(_base().encode(facet=fm.Facet("cat")).to_json())
    wire["facet"]["sort"] = dict(_WIRE_MALFORMED_SORT)
    return wire


def _facet_row_sort_wire() -> dict:
    wire = json.loads(
        _base().encode(facet_col=fm.FacetCol("cat"), facet_row=fm.FacetRow("other")).to_json()
    )
    wire["facet"]["row_sort"] = dict(_WIRE_MALFORMED_SORT)
    return wire


WIRE_DOOR_BUILDERS: dict[str, Callable[[], dict]] = {
    "chart_channel": _chart_channel_wire,
    "layer_channel": _layer_channel_wire,
    "facet_sort": _facet_sort_wire,
    "facet_row_sort": _facet_row_sort_wire,
}


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
@pytest.mark.parametrize("door_name", sorted(WIRE_DOOR_BUILDERS))
def test_wire_malformed_sort_refuses_at_render(door_name: str, output_kind: str) -> None:
    """A hand-built spec (`ChartSpec.from_json`) carrying a malformed sort
    object refuses at render with `RenderError::SortSpecMalformed`, naming
    where the sort sits (chart channel, a layer's own channel, or a facet
    dimension) and the value -- the one route no construction-time gate can
    see."""
    wire = WIRE_DOOR_BUILDERS[door_name]()
    expected = _wire_refusal_message(
        _WIRE_DOOR_LOCATIONS[door_name], _WIRE_MALFORMED_SORT, _WIRE_MALFORMED_REASON
    )
    with pytest.raises(ValueError) as exc_info:
        _render_from_wire(wire, output_kind=output_kind)
    assert str(exc_info.value) == expected


# ---------------------------------------------------------------------------
# A grid-to-wrap redeclaration drops `row_sort`, exactly as it drops
# `row_title` -- so a legitimate merge sequence (spec §6.3) can never leave a
# stale `row_sort` for `test_wire_facet_row_sort_without_row_refuses` above
# to refuse on a chart the caller never wrote anything malformed into.
# ---------------------------------------------------------------------------

# Two ways to build the SAME grid-with-row_sort prior state, mirroring
# `test_facet_merge_matrix.py`'s `.facet()`-built vs `encode()`-built prior
# convention: a plain `.facet(row=, col=)` call plus a `row_sort`-setting
# `encode()` follow-up, and a wrap-then-promote sequence entirely through
# `encode()` (`FacetCol` alone, promoted by `FacetRow(sort=...)`).
_ROW_SORT_GRID_PRIORS: dict[str, Callable[[], "fm.Chart"]] = {
    "facet_built": lambda: (
        _base()
        .facet(row="other", col="cat")
        .encode(facet_row=fm.FacetRow("other", sort="descending"))
    ),
    "encode_built": lambda: (
        _base()
        .encode(facet_col=fm.FacetCol("cat"))
        .encode(facet_row=fm.FacetRow("other", sort="descending"))
    ),
}


@pytest.mark.parametrize(
    "prior_name", sorted(_ROW_SORT_GRID_PRIORS), ids=sorted(_ROW_SORT_GRID_PRIORS)
)
def test_row_sort_drops_on_grid_to_wrap_redeclaration(prior_name: str) -> None:
    prior = _ROW_SORT_GRID_PRIORS[prior_name]()
    assert prior._facet.row_sort == "descending"  # the prior really carries it

    redeclared = prior.facet(field="cat")
    assert redeclared._facet.row_sort is None
    d = json.loads(redeclared.to_json())
    assert "row_sort" not in d["facet"]
    redeclared.to_svg()  # renders; no facet.row_sort RenderError is reachable


@pytest.mark.parametrize(
    "prior_name", sorted(_ROW_SORT_GRID_PRIORS), ids=sorted(_ROW_SORT_GRID_PRIORS)
)
def test_row_sort_grid_contradiction_still_refuses_not_a_stale_wrap(prior_name: str) -> None:
    """The other wrap-only door, `encode(facet=Facet(...))`, still raises
    the grid contradiction (spec §6.3) rather than silently redeclaring --
    unaffected by a set `row_sort`, since the refusal fires before any
    field is touched."""
    prior = _ROW_SORT_GRID_PRIORS[prior_name]()
    with pytest.raises(ValueError, match="already a grid facet"):
        prior.encode(facet=fm.Facet("cat"))


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
def test_wire_facet_row_sort_without_row_refuses(output_kind: str) -> None:
    """`FacetSpec.row_sort` set while `row` is absent (wrap mode) is a
    second, distinct malformation sharing the same `SortSpecMalformed`
    variant: a well-formed sort value with nowhere to apply, not a bad
    shape. Only reachable through a hand-built spec -- no Python door
    exposes a wrap-mode facet's `row_sort` at all (`FacetRow(sort=...)`
    used alone routes its value into `sort`, never `row_sort`; see
    `merged_with`'s wrap-mode routing)."""
    wire = json.loads(_base().encode(facet=fm.Facet("cat")).to_json())
    wire["facet"]["row_sort"] = "ascending"
    expected = _wire_refusal_message(
        "facet.row_sort",
        "ascending",
        "the facet has no row dimension (row=) for row_sort to order",
    )
    with pytest.raises(ValueError) as exc_info:
        _render_from_wire(wire, output_kind=output_kind)
    assert str(exc_info.value) == expected


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
@pytest.mark.parametrize("door_name", sorted(WIRE_DOOR_BUILDERS))
def test_wire_well_formed_sort_renders(door_name: str, output_kind: str) -> None:
    """Vacuity control: the same four wire doors with a well-formed sort
    (or none) render without raising."""
    wire = WIRE_DOOR_BUILDERS[door_name]()
    if door_name in ("facet_sort", "facet_row_sort"):
        wire["facet"][door_name.removeprefix("facet_")] = "ascending"
    else:
        target = wire["encoding"] if door_name == "chart_channel" else wire["layers"][1]["encoding"]
        target["x"]["sort"] = "ascending"
    _render_from_wire(wire, output_kind=output_kind)


# ---------------------------------------------------------------------------
# No-warning control: `sort` is in `FACET`'s honored-kwarg set, so
# `Facet`/`FacetRow`/`FacetCol(sort=...)` never falls into the generic "not
# yet honored; planned for a future Phase" warn_once path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel_cls", [fm.Facet, fm.FacetRow, fm.FacetCol], ids=["Facet", "FacetRow", "FacetCol"]
)
def test_facet_sort_never_warns_not_yet_honored(channel_cls) -> None:
    import warnings

    from ferrum._warn import reset_warnings

    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        channel_cls("cat", sort="ascending")
    messages = [str(ww.message) for ww in w]
    assert not messages, (
        f"expected no warnings constructing {channel_cls.__name__}(sort=...): {messages}"
    )


def test_facet_method_sort_wire_matches_the_channel_door() -> None:
    """`Chart.facet(sort=/row_sort=)` writes the same wire as the channel
    doors it parallels, per declaration shape."""
    pairs = [
        (
            _base().facet(field="cat", sort="descending"),
            _base().encode(facet=fm.Facet("cat", sort="descending")),
        ),
        (
            _base().facet(col="cat", sort="descending"),
            _base().encode(facet_col=fm.FacetCol("cat", sort="descending")),
        ),
        (
            _base().facet(row="cat", sort="descending"),
            _base().encode(facet_row=fm.FacetRow("cat", sort="descending")),
        ),
        (
            _base().facet(row="other", col="cat", sort="ascending", row_sort="descending"),
            _base().encode(
                facet_col=fm.FacetCol("cat", sort="ascending"),
                facet_row=fm.FacetRow("other", sort="descending"),
            ),
        ),
    ]
    for method, channel in pairs:
        assert json.loads(method.to_json())["facet"] == json.loads(channel.to_json())["facet"]


@pytest.mark.parametrize(
    ("declared_as", "kwargs"),
    [
        ("field=", {"field": "cat"}),
        ("col= alone", {"col": "cat"}),
        ("row= alone", {"row": "cat"}),
    ],
    ids=["field", "col", "row"],
)
def test_facet_method_row_sort_on_a_single_dimension_refuses(
    declared_as: str, kwargs: dict
) -> None:
    """A single-dimension declaration has one panel order, which `sort=`
    sets; `row_sort=` there refuses exactly as `row_title=` does."""
    with pytest.raises(ValueError) as exc_info:
        _base().facet(**kwargs, row_sort="descending")
    assert str(exc_info.value) == (
        f"facet(row_sort=...) has no effect on a facet declared via {declared_as}: a "
        "single-dimension declaration has only one panel order, which `sort=` already "
        "sets; `row_sort=` names a second, distinct row-strip order that only a genuine "
        "two-dimension facet (`row=` and `col=` together) has. Pass the order via "
        "`sort=` instead, or use both `row=` and `col=` for a grid facet."
    )


def test_facet_no_sort_wire_omits_sort_keys() -> None:
    """Vacuity control: a facet with no `sort=` declared carries neither
    `sort` nor `row_sort` on the wire, byte-identical to before this task."""
    d = json.loads(_base().encode(facet=fm.Facet("cat")).to_json())
    assert "sort" not in d["facet"]
    assert "row_sort" not in d["facet"]


def test_facet_sort_wire_wrap_is_additive() -> None:
    d = json.loads(_base().encode(facet=fm.Facet("cat", sort="descending")).to_json())
    assert d["facet"]["sort"] == "descending"
    assert "row_sort" not in d["facet"]


def test_facet_sort_wire_grid_is_additive_per_dimension() -> None:
    d = json.loads(
        _base()
        .encode(
            facet_col=fm.FacetCol("cat", sort="ascending"),
            facet_row=fm.FacetRow("other", sort="descending"),
        )
        .to_json()
    )
    assert d["facet"]["sort"] == "ascending"
    assert d["facet"]["row_sort"] == "descending"


# ---------------------------------------------------------------------------
# Malformed-sort-object matrix
# ---------------------------------------------------------------------------

MALFORMED_FORMS: dict[str, dict] = {
    "no_field": {"op": "sum"},
    "non_string_field": {"field": 123, "op": "sum"},
    "unknown_op": {"field": "v", "op": "not_a_real_op"},
    "unknown_order": {"field": "v", "order": "not_a_real_order"},
    # A key that is PRESENT with an explicit None is a malformed value, not
    # an absent (defaulted) key -- distinct from omitting "op"/"order"
    # entirely, which is well-formed (op defaults to sum, order to
    # ascending) and is not in this population at all.
    "none_op": {"field": "v", "op": None},
    "none_order": {"field": "v", "order": None},
}

# Which key each malformed form's error names, for the malformed forms whose
# refusal routes through the op/order choice gate (every form except the
# two "field" shapes).
_CHOICE_PARAM: dict[str, str] = {
    "unknown_op": "op",
    "none_op": "op",
    "unknown_order": "order",
    "none_order": "order",
}


def _sort_honoring_channel_classes() -> list[type]:
    """Every channel class whose honored set contains `sort`, derived from
    `ferrum.encoding.__all__` + `_honored_kwargs` -- never hand-listed."""
    classes = []
    for name in fm.encoding.__all__:
        obj = getattr(fm.encoding, name)
        if isinstance(obj, type) and issubclass(obj, ChannelBase) and "sort" in obj._honored_kwargs:
            classes.append(obj)
    return classes


CHANNEL_CLASSES = _sort_honoring_channel_classes()
# The three facet channels must be part of the derived population now that
# `sort` has joined `FACET` -- if they were not, the population derivation
# itself would be silently wrong.
assert {"Facet", "FacetRow", "FacetCol"} <= {cls.__name__ for cls in CHANNEL_CLASSES}

CHANNEL_MALFORMED_CELLS = [
    (cls.__name__, form) for cls in CHANNEL_CLASSES for form in MALFORMED_FORMS
]


def _no_field_message(channel: str, value: dict) -> str:
    """Independent mirror of `encoding.base._SORT_FIELD_MESSAGE` -- the test
    oracle, not an import of the implementation's own template."""
    return (
        f'{channel}(sort={value!r}) is not a valid sort object: "field" must be a '
        f"string naming a column; got {value.get('field')!r}"
    )


def _choice_message(channel: str, param: str, value: Any, choices: list[str]) -> str:
    """Independent mirror of `validate_choice`'s message template."""
    return (
        f"{channel}(sort=...): {param} must be one of "
        f"{sorted(str(c) for c in choices)}; got {value!r}"
    )


@pytest.mark.parametrize(
    ("class_name", "form_name"),
    CHANNEL_MALFORMED_CELLS,
    ids=[f"{c}-{f}" for c, f in CHANNEL_MALFORMED_CELLS],
)
def test_malformed_sort_object_refuses(class_name: str, form_name: str) -> None:
    cls = next(c for c in CHANNEL_CLASSES if c.__name__ == class_name)
    value = MALFORMED_FORMS[form_name]
    with pytest.raises(ValueError) as exc_info:
        cls("v", sort=value)
    message = str(exc_info.value)
    if form_name in ("no_field", "non_string_field"):
        assert message == _no_field_message(class_name, value)
        return
    from ferrum._core import sort_accepted_vocabulary

    vocab = sort_accepted_vocabulary()
    param = _CHOICE_PARAM[form_name]
    assert message == _choice_message(class_name, param, value[param], vocab[param])


_METHOD_MALFORMED_CELLS = [
    (kwarg, form) for kwarg in ("sort", "row_sort") for form in MALFORMED_FORMS
]


@pytest.mark.parametrize(
    ("kwarg", "form_name"),
    _METHOD_MALFORMED_CELLS,
    ids=[f"{k}-{f}" for k, f in _METHOD_MALFORMED_CELLS],
)
def test_facet_method_malformed_sort_object_refuses(kwarg: str, form_name: str) -> None:
    """`Chart.facet(sort=/row_sort=)` refuses a malformed sort-field object at
    the call, in its own voice, with the channel doors' message shape."""
    value = MALFORMED_FORMS[form_name]
    with pytest.raises(ValueError) as exc_info:
        _base().facet(row="other", col="cat", **{kwarg: value})
    message = str(exc_info.value)
    if form_name in ("no_field", "non_string_field"):
        assert message == (
            f'Chart.facet({kwarg}={value!r}) is not a valid sort object: "field" must be a '
            f"string naming a column; got {value.get('field')!r}"
        )
        return
    from ferrum._core import sort_accepted_vocabulary

    param = _CHOICE_PARAM[form_name]
    choices = sort_accepted_vocabulary()[param]
    assert message == (
        f"Chart.facet({kwarg}=...): {param} must be one of "
        f"{sorted(str(c) for c in choices)}; got {value[param]!r}"
    )


# Independent mirror of the Rust reason text `EncodingSpec::new` raises for
# each of the four malformed shapes (`crates/ferrum-core/src/spec/encoding/mod.rs`,
# wrapping `render::scale_resolve::domain::parse_sort_field_object`'s
# `Result<_, String>`), confirmed against the live PyO3 constructor rather
# than mirrored blind.
_ENCODING_SPEC_REASONS: dict[str, Callable[[dict], str]] = {
    "no_field": lambda v: 'sort object missing string "field"',
    "non_string_field": lambda v: f'sort "field" must be a string, got {v["field"]}',
    "unknown_op": lambda v: f'unsupported sort op "{v["op"]}"',
    "unknown_order": lambda v: f'unsupported sort order "{v["order"]}"',
    # `op` and `order` share the same shape: a present, non-string value
    # (including an explicit `null`) hits a dedicated "must be a string"
    # branch before the string-vocabulary parser ever runs; a present
    # string outside the vocabulary hits "unsupported sort {op,order}
    # {value:?}". Confirmed live against the rebuilt extension, not
    # guessed.
    "none_op": lambda v: 'sort "op" must be a string, got null',
    "none_order": lambda v: 'sort "order" must be a string, got null',
}


@pytest.mark.parametrize("form_name", sorted(MALFORMED_FORMS), ids=sorted(MALFORMED_FORMS))
def test_encoding_spec_malformed_sort_refuses(form_name: str) -> None:
    """`EncodingSpec(field="a", sort={...})` refuses the same four malformed
    forms, naming `EncodingSpec` and the value: `EncodingSpec(sort=...):
    {reason}; received {value}` (`spec/encoding/mod.rs`)."""
    from ferrum._core import EncodingSpec

    value = MALFORMED_FORMS[form_name]
    received = json.dumps(value, separators=(",", ":"))
    expected = (
        f"EncodingSpec(sort=...): {_ENCODING_SPEC_REASONS[form_name](value)}; received {received}"
    )
    with pytest.raises(ValueError) as exc_info:
        EncodingSpec("a", sort=value)
    assert str(exc_info.value) == expected


# ---------------------------------------------------------------------------
# Valid-sort-form byte-identity control (X, the positional-channel
# precedent). `x` is ordinal (a string category column), not quantitative:
# sort has no effect on a quantitative axis's rendered order, so a
# quantitative fixture would pin five identical SVG hashes and could not
# catch a regression in the object-form render path
# (`apply_sort_to_domain`/`parse_sort_field_object`,
# `crates/ferrum-core/src/render/scale_resolve/domain.rs`) at all. Hashes
# are pinned from a BOTH-LANGUAGES revert of commit 892ea373 (the tip
# before this task) -- a Python-only revert cannot isolate this task's
# effect, since `domain.rs`'s render-time object-form parse changed by
# +186/-30 in the same task and is exactly what resolves these five forms.
# ---------------------------------------------------------------------------

_VALID_SORT_X_DF = pl.DataFrame(
    {
        "cat": ["banana", "apple", "cherry"],
        "y": [30.0, 10.0, 20.0],
    }
)

_VALID_SORT_FORMS: dict[str, Any] = {
    "ascending": "ascending",
    "descending": "descending",
    "explicit_list": ["cherry", "apple", "banana"],
    "shorthand": "-y",
    "sort_object": {"field": "y", "op": "mean", "order": "ascending"},
}

# Each form's expected category order, computed independently of the
# renderer (alphabetical for ascending/descending, the array verbatim for
# explicit_list, and directly from `y` for the two aggregate forms -- one
# row per category, so sum == mean == the row's own value). All five are
# pairwise distinct, the discriminating property this control exists to
# have: a fixture that regressed to a non-sorting axis would collapse these
# to one shared order and one shared SVG hash.
_VALID_SORT_EXPECTED_ORDER: dict[str, list[str]] = {
    "ascending": ["apple", "banana", "cherry"],
    "descending": ["cherry", "banana", "apple"],
    "explicit_list": ["cherry", "apple", "banana"],
    "shorthand": ["banana", "cherry", "apple"],  # "-y": descending by sum(y)
    "sort_object": ["apple", "cherry", "banana"],  # mean(y) ascending
}
assert len({tuple(o) for o in _VALID_SORT_EXPECTED_ORDER.values()}) == len(
    _VALID_SORT_EXPECTED_ORDER
)


def _x_cat_order(svg: str, cat_set: set[str]) -> list[str]:
    """Distinct x-axis category tick labels, in first-seen (left-to-right)
    document order -- mirrors `test_flexibility_campaign.py`'s helper of
    the same name."""
    import re

    seen: list[str] = []
    for t in re.findall(r"<text[^>]*>([^<]+)</text>", svg):
        if t in cat_set and t not in seen:
            seen.append(t)
    return seen


def _build_valid_sort_chart(value: Any) -> "fm.Chart":
    return fm.Chart(_VALID_SORT_X_DF).mark_bar().encode(x=fm.X("cat", type="N", sort=value), y="y")


# (svg_sha256[:16], wire_sha256[:16]), captured against a real build of
# commit 892ea373 with BOTH src/ferrum and crates/ferrum-core at that
# commit (a `git archive` checkout plus an extension compiled from it, not
# a file-copy revert of the current tree) -- byte-identical to the current
# tree's own hashes for every form, proving this task's Rust and Python
# changes together leave every valid form's render untouched. The SVG halves
# are re-measured with the bars grown from the 0 tick; the wire halves are
# the original measurement, so the sort itself still reaches the wire
# unchanged. Each SVG half encodes two changes: the sort forms leaving the
# render untouched, and the bars grown from the 0 tick, which
# `test_valid_x_sort_forms_start_on_the_zero_tick` asserts directly.
_VALID_SORT_HASHES: dict[str, tuple[str, str]] = {
    "ascending": ("2b97ce24c178b6b6", "7da2b45fd52a0a06"),
    "descending": ("ae23dc6a5bb62204", "24c72b8a3dffbddf"),
    "explicit_list": ("ef348b6ceece096c", "fb086988c2686cb8"),
    "shorthand": ("5ff8dc0cf99835ca", "7d5d947e3b082fdd"),
    "sort_object": ("f272e77125c5ef75", "048087a0413c1c1f"),
}


@pytest.mark.parametrize("form_name", sorted(_VALID_SORT_FORMS), ids=sorted(_VALID_SORT_FORMS))
def test_valid_x_sort_forms_render_the_expected_order(form_name: str) -> None:
    """Independent-oracle control: each valid form renders its own,
    independently-computed category order -- not merely that the SVG was
    produced, and not merely that the wire echoes the input verbatim."""
    value = _VALID_SORT_FORMS[form_name]
    svg = _build_valid_sort_chart(value).to_svg()
    assert _x_cat_order(svg, {"apple", "banana", "cherry"}) == _VALID_SORT_EXPECTED_ORDER[form_name]


@pytest.mark.parametrize("form_name", sorted(_VALID_SORT_FORMS), ids=sorted(_VALID_SORT_FORMS))
def test_valid_x_sort_forms_start_on_the_zero_tick(form_name: str) -> None:
    """The geometry the SVG digests pin: every bar grows from 0."""
    svg = _build_valid_sort_chart(_VALID_SORT_FORMS[form_name]).to_svg()
    assert_bars_start_on_the_zero_tick(svg)


@pytest.mark.parametrize("form_name", sorted(_VALID_SORT_FORMS), ids=sorted(_VALID_SORT_FORMS))
def test_valid_x_sort_forms_byte_identical(form_name: str) -> None:
    import hashlib

    value = _VALID_SORT_FORMS[form_name]
    chart = _build_valid_sort_chart(value)
    svg = chart.to_svg()
    wire = chart.to_json()
    svg_hash = hashlib.sha256(svg.encode()).hexdigest()[:16]
    wire_hash = hashlib.sha256(wire.encode()).hexdigest()[:16]
    assert (svg_hash, wire_hash) == _VALID_SORT_HASHES[form_name]
    # The wire hash pin above is the byte-identity proof; this restates what
    # it proves in a readable form for a failing assertion's message.
    assert json.loads(wire)["encoding"]["x"]["sort"] == value


def test_valid_x_sort_forms_produce_distinct_svgs() -> None:
    """Discriminating check on the fixture itself, independent of the
    pinned-hash control above: the five forms must not collapse onto fewer
    than four distinct rendered SVGs, or the fixture has silently regressed
    to an axis sort has no effect on (the exact quantitative-`x` failure
    mode this control replaces)."""
    svgs = {
        form: _build_valid_sort_chart(value).to_svg() for form, value in _VALID_SORT_FORMS.items()
    }
    distinct = len(set(svgs.values()))
    assert distinct >= 4, (
        f"expected at least 4 distinct SVGs among {len(svgs)} forms, got {distinct}"
    )
