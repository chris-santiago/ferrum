"""Facet-state merge matrix: `_Facet.merged_with` is the one facet-state writer.

Two matrices, generated from the population `_Facet.merged_with` was written
to cover (#176(10a)(10b) + N5, spec §4.6 (10b) and §6.3's orientation-aware
promotion and `.facet(field=)` redeclaration door):

**State-merge matrix** (``test_facet_merge_cell``): prior facet state
({none, wrap, grid, row-only, col-only}, each built through BOTH doors that
can produce it -- ``.facet()`` and the ``encode()`` facet channels -- so a
prior state left sitting in ``_encoding`` by an earlier ``encode()`` call is
exercised, not only a prior built through ``.facet()``, which never touches
``_encoding`` at all) x door (``encode(facet=...)``, ``encode(facet_col=...)``,
``encode(facet_row=...)``, and ``.facet(...)`` at each of its four arities)
x carried field (title, row_title, resolve, ``wrap_orient``, ncols/nrows, the
grid's other dimension). Every door in the cross product introduces a field
the prior state does not already use (``"k"`` for a generic/column
declaration, ``"m"`` for a row declaration), so a carried value can never be
confused with a freshly declared one. Every cell's expected outcome comes
from ``_expected``, an independent mirror of the mode-transition and carry
rule stated in spec §6.3 -- not a call into ``_Facet.merged_with`` -- so the
oracle and the implementation are two separately written expressions of the
same rule. The ``"none"`` prior row is the vacuity control, checked against a
wire dict pinned from the pre-T7 code by file-copy revert (not two post-fix
doors compared against each other).

**Empty-title matrix** (``test_empty_title_renders_bare_strip``, #176(10a)):
an empty or whitespace-only title, from every door that can set one, on
every strip kind (column, wrap, row) that door can drive, renders the exact
strip text list ``['p', 'q']`` -- the bare facet values, in order, with no
colon prefix and no duplication.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, NamedTuple, Optional, Union

import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# State-merge matrix
# ---------------------------------------------------------------------------

_STATE_DF = pl.DataFrame(
    {
        "x": [1, 2],
        "y": [3, 4],
        "g": ["a", "b"],
        "h": ["c", "d"],
        "k": ["e", "f"],
        "m": ["i", "j"],
    }
)


def _base_chart() -> "fm.Chart":
    return fm.Chart(_STATE_DF).mark_point().encode(x="x", y="y")


class PriorSpec(NamedTuple):
    build: Callable[[], "fm.Chart"]
    mode: Optional[str]  # None | "wrap" | "grid"
    field: Optional[str]  # the wrap field, when mode == "wrap"
    row: Optional[str]
    col: Optional[str]
    wrap_orient: Optional[str]  # None | "col" | "row"; only meaningful for wrap
    title: Optional[str]
    row_title: Optional[str]
    ncols: Optional[int]
    nrows: Optional[int]
    resolve: Optional[dict]


# Each non-"none" state is built two ways -- through `.facet()`, which never
# touches `_encoding`, and through the `encode()` facet channels, which
# leave a channel object sitting in `_encoding` after the call returns. Both
# variants of the same conceptual prior produce identical `_Facet` field
# values (verified live); the "_enc" variants exist specifically so a DOOR
# that is itself an `encode()` call exercises the stale-`_encoding` shape
# (#176(10b) finding 2) that a `.facet()`-built prior can never reach.
PRIORS: dict[str, PriorSpec] = {
    "none": PriorSpec(_base_chart, None, None, None, None, None, None, None, None, None, None),
    "wrap": PriorSpec(
        lambda: _base_chart().facet(field="g", title="WT", ncols=4).share_scale(x="independent"),
        "wrap",
        "g",
        None,
        None,
        None,
        "WT",
        None,
        4,
        None,
        {"x": "independent"},
    ),
    "wrap_enc": PriorSpec(
        lambda: (
            _base_chart()
            .encode(facet=fm.Facet("g", title="WT", columns=4))
            .share_scale(x="independent")
        ),
        "wrap",
        "g",
        None,
        None,
        None,
        "WT",
        None,
        4,
        None,
        {"x": "independent"},
    ),
    "grid": PriorSpec(
        lambda: (
            _base_chart()
            .facet(row="h", col="g", title="GCT", row_title="GRT", ncols=3, nrows=2)
            .share_scale(y="independent")
        ),
        "grid",
        None,
        "h",
        "g",
        None,
        "GCT",
        "GRT",
        3,
        2,
        {"y": "independent"},
    ),
    "grid_enc": PriorSpec(
        lambda: (
            _base_chart()
            .facet(row="h", col="g", ncols=3, nrows=2)
            .share_scale(y="independent")
            .encode(
                facet_col=fm.FacetCol("g", title="GCT"),
                facet_row=fm.FacetRow("h", title="GRT"),
            )
        ),
        "grid",
        None,
        "h",
        "g",
        None,
        "GCT",
        "GRT",
        3,
        2,
        {"y": "independent"},
    ),
    "row_only": PriorSpec(
        lambda: _base_chart().facet(row="h", title="RT").share_scale(x="independent"),
        "wrap",
        "h",
        None,
        None,
        "row",
        "RT",
        None,
        None,
        None,
        {"x": "independent"},
    ),
    "row_only_enc": PriorSpec(
        lambda: (
            _base_chart()
            .encode(facet_row=fm.FacetRow("h", title="RT"))
            .share_scale(x="independent")
        ),
        "wrap",
        "h",
        None,
        None,
        "row",
        "RT",
        None,
        None,
        None,
        {"x": "independent"},
    ),
    "col_only": PriorSpec(
        lambda: _base_chart().facet(col="g", title="CT", ncols=5).share_scale(y="independent"),
        "wrap",
        "g",
        None,
        None,
        "col",
        "CT",
        None,
        5,
        None,
        {"y": "independent"},
    ),
    "col_only_enc": PriorSpec(
        lambda: (
            _base_chart()
            .facet(col="g", ncols=5)
            .share_scale(y="independent")
            .encode(facet_col=fm.FacetCol("g", title="CT"))
        ),
        "wrap",
        "g",
        None,
        None,
        "col",
        "CT",
        None,
        5,
        None,
        {"y": "independent"},
    ),
}


class DoorSpec(NamedTuple):
    kind: str  # "generic" | "facet_field" | "col" | "row" | "grid"
    apply: Callable[["fm.Chart"], "fm.Chart"]
    declare: str  # the contradiction message's leading entry (kind == "generic" only)
    new_field: Optional[str]  # single-dimension doors ("generic"/"facet_field"/"col"/"row")
    new_row: Optional[str]  # "grid" door only
    new_col: Optional[str]  # "grid" door only


DOORS: dict[str, DoorSpec] = {
    "encode_facet": DoorSpec(
        "generic",
        lambda c: c.encode(facet=fm.Facet("k")),
        "encode(facet=Facet('k'))",
        "k",
        None,
        None,
    ),
    "encode_facet_col": DoorSpec(
        "col", lambda c: c.encode(facet_col=fm.FacetCol("k")), "", "k", None, None
    ),
    "encode_facet_row": DoorSpec(
        "row", lambda c: c.encode(facet_row=fm.FacetRow("m")), "", "m", None, None
    ),
    "facet_field": DoorSpec(
        "facet_field", lambda c: c.facet(field="k"), "Chart.facet(field='k')", "k", None, None
    ),
    "facet_col": DoorSpec("col", lambda c: c.facet(col="k"), "", "k", None, None),
    "facet_row": DoorSpec("row", lambda c: c.facet(row="m"), "", "m", None, None),
    "facet_row_col": DoorSpec("grid", lambda c: c.facet(row="m", col="k"), "", None, "m", "k"),
}


class Expected(NamedTuple):
    mode: str
    field: Optional[str]
    row: Optional[str]
    col: Optional[str]
    wrap_orient: Optional[str]
    title: Optional[str]
    row_title: Optional[str]
    ncols: Optional[int]
    nrows: Optional[int]
    resolve: Optional[dict]


def _expected(prior: PriorSpec, door: DoorSpec) -> Union[Expected, str]:
    """Independent mirror of spec §6.3's merge rule -- the test oracle, not
    a call into `_Facet.merged_with`. Returns `"CONTRADICTION"` for the
    cells the rule refuses.

    Mode transitions are orientation-aware: onto a single-dimension prior, a
    `"col"`/`"row"` door of the SAME orientation as the prior replaces the
    field and stays wrap; a door of the OTHER orientation (which includes a
    generic, orientation-less prior) promotes to a grid, keeping the prior's
    field on its own axis. `"generic"` (`encode(facet=...)`) onto a grid
    refuses; `"facet_field"` (`.facet(field=...)`) onto a grid redeclares --
    replaces the grid with a wrap, dropping row_title (no home in wrap
    mode).
    """
    ncols, nrows, resolve = prior.ncols, prior.nrows, prior.resolve

    if door.kind == "grid":
        return Expected(
            "grid",
            None,
            door.new_row,
            door.new_col,
            None,
            prior.title,
            prior.row_title,
            ncols,
            nrows,
            resolve,
        )

    if door.kind in ("generic", "facet_field"):
        # A wrap-only declaration states no orientation of its own -- `Facet`
        # and `.facet(field=...)` have no orientation parameter -- so
        # `wrap_orient` is unmentioned and survives from `prior` (`None` for
        # a grid prior, since a grid's own `wrap_orient` is always `None`).
        carried_orient = prior.wrap_orient if prior.mode == "wrap" else None
        if prior.mode == "grid":
            if door.kind == "generic":
                return "CONTRADICTION"
            return Expected(
                "wrap",
                door.new_field,
                None,
                None,
                carried_orient,
                prior.title,
                None,
                ncols,
                nrows,
                resolve,
            )
        return Expected(
            "wrap",
            door.new_field,
            None,
            None,
            carried_orient,
            prior.title,
            None,
            ncols,
            nrows,
            resolve,
        )

    if door.kind == "col":
        if prior.mode == "grid":
            return Expected(
                "grid",
                None,
                prior.row,
                door.new_field,
                None,
                prior.title,
                prior.row_title,
                ncols,
                nrows,
                resolve,
            )
        if prior.mode == "wrap" and prior.wrap_orient == "col":
            return Expected(
                "wrap",
                door.new_field,
                None,
                None,
                "col",
                prior.title,
                None,
                ncols,
                nrows,
                resolve,
            )
        if prior.mode == "wrap":
            # Promotion moves the prior's field to the row axis, and its
            # title -- which labeled that field's own single strip -- moves
            # with it, landing in `row_title`, not `title`. `title` (the new
            # col strip) holds only what THIS door states; no main-matrix
            # col door restates a title in these "carry" cells, so it is
            # `None` here, not `prior.title`.
            return Expected(
                "grid",
                None,
                prior.field,
                door.new_field,
                None,
                None,
                prior.title,
                ncols,
                nrows,
                resolve,
            )
        return Expected(
            "wrap", door.new_field, None, None, "col", None, None, ncols, nrows, resolve
        )

    # door.kind == "row"
    if prior.mode == "grid":
        return Expected(
            "grid",
            None,
            door.new_field,
            prior.col,
            None,
            prior.title,
            prior.row_title,
            ncols,
            nrows,
            resolve,
        )
    if prior.mode == "wrap" and prior.wrap_orient == "row":
        return Expected(
            "wrap",
            door.new_field,
            None,
            None,
            "row",
            prior.title,
            None,
            ncols,
            nrows,
            resolve,
        )
    if prior.mode == "wrap":
        return Expected(
            "grid",
            None,
            door.new_field,
            prior.field,
            None,
            prior.title,
            prior.row_title,
            ncols,
            nrows,
            resolve,
        )
    return Expected("wrap", door.new_field, None, None, "row", None, None, ncols, nrows, resolve)


CELLS = [(p, d) for p in PRIORS for d in DOORS]


@pytest.mark.parametrize(("prior_name", "door_name"), CELLS, ids=[f"{p}-{d}" for p, d in CELLS])
def test_facet_merge_cell(prior_name: str, door_name: str) -> None:
    prior = PRIORS[prior_name]
    door = DOORS[door_name]
    chart = prior.build()
    expected = _expected(prior, door)

    if expected == "CONTRADICTION":
        with pytest.raises(ValueError) as exc_info:
            door.apply(chart)
        assert str(exc_info.value) == (
            f"{door.declare}: the chart is already a grid facet (row={prior.row!r}, "
            f"col={prior.col!r}); a wrap facet cannot replace it. Use facet_row=/"
            "facet_col= to change a dimension, or call .facet(...) to redeclare the facet."
        )
        return

    result = door.apply(chart)
    facet = result._facet
    assert facet is not None
    # FacetMode/WrapOrient are str-backed (`FacetMode.WRAP == "wrap"`), so
    # compare directly rather than through `str()`, which gives
    # "FacetMode.WRAP".
    assert facet.mode_kind == expected.mode
    assert facet.field == expected.field
    assert facet.row == expected.row
    assert facet.col == expected.col
    assert facet.wrap_orient == expected.wrap_orient
    assert facet.title == expected.title
    assert facet.row_title == expected.row_title
    assert facet.ncols == expected.ncols
    assert facet.nrows == expected.nrows
    assert facet.resolve == expected.resolve


# Pinned from the pre-T7 code (commit 9a78f029) by file-copy revert: neither
# door carries anything when there is no prior facet, so this is the exact
# wire dict the ORIGINAL (unfixed) `.facet()`/`encode(facet=...)` doors
# produced for a single, unmerged call -- the true baseline the "no prior:
# byte-identical to today" rule promises, not a post-fix door compared
# against another post-fix door.
_PRE_T7_FACET_FIELD_DICT = {"field": "k", "mode": {"kind": "wrap", "ncols": 1}}
_PRE_T7_ENCODE_FACET_DICT = {"field": "k", "mode": {"kind": "wrap", "ncols": 1}}


def test_facet_merge_no_prior_wire_matches_pinned_pre_fix_baseline() -> None:
    """Vacuity control: a facet declaration with nothing to merge against
    produces the exact wire dict the pre-T7 code produced for the same
    single, unmerged call."""
    base = _base_chart()
    facet_field = json.loads(base.facet(field="k").to_json())["facet"]
    encode_facet = json.loads(base.encode(facet=fm.Facet("k")).to_json())["facet"]
    assert facet_field == _PRE_T7_FACET_FIELD_DICT
    assert encode_facet == _PRE_T7_ENCODE_FACET_DICT


# ---------------------------------------------------------------------------
# Retitle-same-field: a same-orientation redeclaration of the SAME field is
# a retitle, not a diagonal self-grid (#176(10b) finding 1's motivating bug).
# ---------------------------------------------------------------------------


def test_facet_merge_retitle_same_field_row_facet_door() -> None:
    c = _base_chart().facet(row="h").facet(row="h", title="New")
    f = c._facet
    assert f.mode_kind == "wrap"
    assert f.field == "h"
    assert f.row is None and f.col is None
    assert f.wrap_orient == "row"
    assert f.title == "New"


def test_facet_merge_retitle_same_field_row_encode_door() -> None:
    c = (
        _base_chart()
        .encode(facet_row=fm.FacetRow("h"))
        .encode(facet_row=fm.FacetRow("h", title="New"))
    )
    f = c._facet
    assert f.mode_kind == "wrap"
    assert f.field == "h"
    assert f.wrap_orient == "row"
    assert f.title == "New"


def test_facet_merge_retitle_same_field_col_facet_door() -> None:
    c = _base_chart().facet(col="g").facet(col="g", title="New")
    f = c._facet
    assert f.mode_kind == "wrap"
    assert f.field == "g"
    assert f.wrap_orient == "col"
    assert f.title == "New"


def test_facet_merge_retitle_same_field_col_encode_door() -> None:
    c = (
        _base_chart()
        .encode(facet_col=fm.FacetCol("g"))
        .encode(facet_col=fm.FacetCol("g", title="New"))
    )
    f = c._facet
    assert f.mode_kind == "wrap"
    assert f.field == "g"
    assert f.wrap_orient == "col"
    assert f.title == "New"


# ---------------------------------------------------------------------------
# Promotion rendering: a promoted field's title moves with it, on the
# rendered strip -- not just in `_facet`'s fields. `_STATE_DF`'s two rows
# make a promoted 2x2 grid diagonal-sparse (each row/col category appears in
# only one panel, and an empty panel's strip does not render), which would
# make a render check pass or fail by which diagonal happens to be
# populated rather than by the fix -- so promotion rendering uses its own
# DataFrame, a full cross product of every field pair a promotion cell can
# produce (`g`-`k`, `h`-`k`, `g`-`m`).
# ---------------------------------------------------------------------------

_PROMO_DF = pl.DataFrame(
    {
        "x": [1, 2, 3, 4],
        "y": [1, 2, 3, 4],
        "g": ["a", "a", "b", "b"],
        "h": ["c", "c", "d", "d"],
        "k": ["e", "f", "e", "f"],
        "m": ["i", "j", "i", "j"],
    }
)


def _promo_chart() -> "fm.Chart":
    return fm.Chart(_PROMO_DF).mark_point().encode(x="x", y="y")


def _axis_strip_texts(svg: str) -> tuple[list[str], list[str]]:
    """(column-strip texts, row-strip texts), each de-duplicated in
    first-seen order. A column strip renders once per row of panels and a
    row strip once per column of panels (real ferrum behavior, not a bug),
    so the same distinct text repeats; `rotate(90` on the row-strip glyph is
    what distinguishes the two axes (see the wrap/grid strip fixtures
    above)."""
    col_texts: list[str] = []
    row_texts: list[str] = []
    for attrs, text in re.findall(r"<text([^>]*)>([^<]*)</text>", svg):
        if 'font-size="12"' not in attrs or "font-weight" in attrs:
            continue  # not a strip label (tick labels are 11px, axis titles 13px+weight)
        target = row_texts if "rotate(90" in attrs else col_texts
        if text not in target:
            target.append(text)
    return col_texts, row_texts


class PromotionSpec(NamedTuple):
    facet_door: Callable[["fm.Chart"], "fm.Chart"]
    encode_door: Callable[["fm.Chart"], "fm.Chart"]
    expected_col_texts: list[str]
    expected_row_texts: list[str]


PROMOTIONS: dict[str, PromotionSpec] = {
    # row-oriented wrap ('h', titled) + a col declaration ('k', untitled):
    # promotes to row='h', col='k'. The reviewers' exact probe (finding 1 /
    # spec issue 1): the prior's title must land on the ROW strip, not the
    # (untitled) column strip it used to mislabel.
    "row_only_promoted_by_col": PromotionSpec(
        lambda c: c.facet(row="h", title="RT").facet(col="k"),
        lambda c: c.encode(facet_row=fm.FacetRow("h", title="RT")).encode(
            facet_col=fm.FacetCol("k")
        ),
        ["e", "f"],
        ["RT: c", "RT: d"],
    ),
    # generic (orientation-less) wrap ('g', titled) + a col declaration
    # ('k', untitled): promotes to row='g', col='k'. Spec issue 1's third
    # probe (`.facet(field='g', title='GT').facet(col='k')`).
    "generic_promoted_by_col": PromotionSpec(
        lambda c: c.facet(field="g", title="GT").facet(col="k"),
        lambda c: c.encode(facet=fm.Facet("g", title="GT")).encode(facet_col=fm.FacetCol("k")),
        ["e", "f"],
        ["GT: a", "GT: b"],
    ),
    # generic (orientation-less) wrap ('g', titled) + a row declaration
    # ('m', untitled): promotes to row='m', col='g'. The mirror direction --
    # unaffected by the bug, since the prior's field lands on the COL axis,
    # where its title already belonged in `title`.
    "generic_promoted_by_row": PromotionSpec(
        lambda c: c.facet(field="g", title="GT").facet(row="m"),
        lambda c: c.encode(facet=fm.Facet("g", title="GT")).encode(facet_row=fm.FacetRow("m")),
        ["GT: a", "GT: b"],
        ["i", "j"],
    ),
    # column-oriented wrap ('g', titled) + a row declaration ('m',
    # untitled): promotes to row='m', col='g'. Also unaffected by the bug,
    # for the same reason as the previous case.
    "col_only_promoted_by_row": PromotionSpec(
        lambda c: c.facet(col="g", title="CT").facet(row="m"),
        lambda c: c.encode(facet_col=fm.FacetCol("g", title="CT")).encode(
            facet_row=fm.FacetRow("m")
        ),
        ["CT: a", "CT: b"],
        ["i", "j"],
    ),
}

PROMOTION_CELLS = [(name, sub) for name in PROMOTIONS for sub in ("facet_door", "encode_door")]


@pytest.mark.parametrize(
    ("promo_name", "sub_door"),
    PROMOTION_CELLS,
    ids=[f"{n}-{d}" for n, d in PROMOTION_CELLS],
)
def test_promotion_title_lands_on_the_correct_rendered_strip(
    promo_name: str, sub_door: str
) -> None:
    spec = PROMOTIONS[promo_name]
    door = getattr(spec, sub_door)
    svg = door(_promo_chart()).to_svg()
    col_texts, row_texts = _axis_strip_texts(svg)
    assert col_texts == spec.expected_col_texts, (
        f"column strip: expected {spec.expected_col_texts}, got {col_texts}"
    )
    assert row_texts == spec.expected_row_texts, (
        f"row strip: expected {spec.expected_row_texts}, got {row_texts}"
    )


# ---------------------------------------------------------------------------
# share_scale() through merged_with (finding 7): the resolve-only door
# ---------------------------------------------------------------------------


def test_share_scale_carries_every_other_field_through_merged_with() -> None:
    c = (
        _base_chart()
        .facet(row="h", col="g", title="GCT", row_title="GRT", ncols=3, nrows=2)
        .share_scale(y="independent")
    )
    f = c._facet
    assert f.mode_kind == "grid"
    assert f.row == "h" and f.col == "g"
    assert f.title == "GCT" and f.row_title == "GRT"
    assert f.ncols == 3 and f.nrows == 2
    assert f.resolve == {"y": "independent"}

    c2 = c.share_scale(x="independent")
    f2 = c2._facet
    assert f2.resolve == {"y": "independent", "x": "independent"}
    # everything else survives untouched
    assert f2.mode_kind == "grid"
    assert f2.row == "h" and f2.col == "g"
    assert f2.title == "GCT" and f2.row_title == "GRT"
    assert f2.ncols == 3 and f2.nrows == 2


# ---------------------------------------------------------------------------
# Empty-title matrix (#176(10a))
# ---------------------------------------------------------------------------

_WRAP_TITLE_DF = pl.DataFrame({"x": [1, 2], "y": [1, 2], "cat": ["p", "q"]})
# A column strip renders once per ROW of panels and a row strip once per
# COLUMN of panels, so a genuine 2x2 grid would render each strip's "p"/"q"
# labels twice over (once per repeated row/column). These two fixtures each
# hold the OTHER dimension to a single value, isolating the strip under test
# to exactly one label per category -- still a real grid (both `row=` and
# `col=` given), just degenerate 1xN/Nx1, so the untitled dimension's bare
# strip text ("z") never collides with the `p`/`q` values under test.
_COL_STRIP_DF = pl.DataFrame(
    {"x": [1, 2], "y": [1, 2], "col_cat": ["p", "q"], "row_cat": ["z", "z"]}
)
_ROW_STRIP_DF = pl.DataFrame(
    {"x": [1, 2], "y": [1, 2], "col_cat": ["z", "z"], "row_cat": ["p", "q"]}
)


def _wrap_title_base() -> "fm.Chart":
    return fm.Chart(_WRAP_TITLE_DF).mark_point().encode(x="x", y="y")


def _col_strip_base() -> "fm.Chart":
    return fm.Chart(_COL_STRIP_DF).mark_point().encode(x="x", y="y")


def _row_strip_base() -> "fm.Chart":
    return fm.Chart(_ROW_STRIP_DF).mark_point().encode(x="x", y="y")


class StripSpec(NamedTuple):
    strip_kind: str  # "column" | "wrap" | "row"
    build: Callable[[Any], "fm.Chart"]


STRIP_CELLS: dict[str, StripSpec] = {
    "facet_title_wrap": StripSpec(
        "wrap", lambda title: _wrap_title_base().facet(field="cat", title=title)
    ),
    "facet_title_column": StripSpec(
        "column",
        lambda title: _col_strip_base().facet(row="row_cat", col="col_cat", title=title),
    ),
    "facet_row_title_row": StripSpec(
        "row",
        lambda title: _row_strip_base().facet(row="row_cat", col="col_cat", row_title=title),
    ),
    "channel_facet_wrap": StripSpec(
        "wrap", lambda title: _wrap_title_base().encode(facet=fm.Facet("cat", title=title))
    ),
    "channel_facet_col_wrap": StripSpec(
        "wrap",
        lambda title: _wrap_title_base().encode(facet_col=fm.FacetCol("cat", title=title)),
    ),
    "channel_facet_col_grid": StripSpec(
        "column",
        lambda title: _col_strip_base().encode(
            facet_col=fm.FacetCol("col_cat", title=title), facet_row=fm.FacetRow("row_cat")
        ),
    ),
    "channel_facet_row_wrap": StripSpec(
        "wrap",
        lambda title: _wrap_title_base().encode(facet_row=fm.FacetRow("cat", title=title)),
    ),
    "channel_facet_row_grid": StripSpec(
        "row",
        lambda title: _row_strip_base().encode(
            facet_col=fm.FacetCol("col_cat"), facet_row=fm.FacetRow("row_cat", title=title)
        ),
    ),
}

EMPTY_TITLES = ["", "   "]

TITLE_CELLS = [(door, title) for door in STRIP_CELLS for title in EMPTY_TITLES]


def _strip_value_texts(svg: str) -> list[str]:
    """Text nodes whose whole content is the bare facet value `p`/`q`, or a
    `"<title>: <value>"` strip label for one of them -- distinguishing strip
    labels from axis ticks/titles, which never render `p`/`q` verbatim in
    this fixture (its data and axis domains are all numeric)."""
    return re.findall(r"<text[^>]*>((?:[^<]*: )?[pq])</text>", svg)


@pytest.mark.parametrize(
    ("door_name", "title"),
    TITLE_CELLS,
    ids=[f"{d}-{'empty' if t == '' else 'whitespace'}" for d, t in TITLE_CELLS],
)
def test_empty_title_renders_bare_strip(door_name: str, title: str) -> None:
    spec = STRIP_CELLS[door_name]
    svg = spec.build(title).to_svg()
    assert _strip_value_texts(svg) == ["p", "q"], (
        f"expected the exact strip-text list ['p', 'q'] for an empty/whitespace title "
        f"({spec.strip_kind} strip), got {_strip_value_texts(svg)}"
    )
