"""Facet value type and serialization helpers.

Extracted from ``chart.py`` (cohesion finding C2/C10). The public ``Chart``
surface is unchanged: ``Chart.facet()`` constructs a :class:`_Facet` via these
constructors and ``Chart._build_facet_dict`` / ``Chart._infer_facet_cardinality``
delegate to the near-pure free functions :func:`build_facet_dict` and
:func:`infer_facet_cardinality`.

C10 type-safety: ``mode_kind`` and ``wrap_orient`` are ``str``-backed enums and
field-validity is enforced at construction time, so a ``_Facet`` can no longer
hold a mode/field combination that is invalid by convention. The ``str`` base on
the enums keeps every existing string comparison (``f.mode_kind == "wrap"``) and
the serialized JSON byte-identical with the legacy stringly-typed form.

These functions take primitives (a ``_Facet`` and a DataFrame) and return plain
results — they never import ``Chart``, so ``chart.py`` imports this module one
directionally with no cycle.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from ferrum._coerce import to_arrow_table


class FacetDeclaration(str, Enum):
    """Which dimension a facet door declares in one call to :meth:`_Facet.merged_with`.

    ``GENERIC`` is the ``encode(facet=...)`` door's wrap-only declaration
    (refuses onto an existing grid); ``FACET_FIELD`` is
    ``.facet(field=...)``'s wrap-only declaration (redeclares: replaces an
    existing grid with a wrap instead of refusing); ``COL``/``ROW`` declare
    one grid dimension (``FacetCol``/``FacetRow`` / ``.facet(col=...)``/
    ``.facet(row=...)`` alone); ``GRID`` declares both dimensions at once
    (``.facet(row=..., col=...)``); ``RESOLVE`` is ``Chart.share_scale()``'s
    declaration, the only field it changes is ``resolve``.
    """

    GENERIC = "generic"
    FACET_FIELD = "facet_field"
    COL = "col"
    ROW = "row"
    GRID = "grid"
    RESOLVE = "resolve"


class FacetMode(str, Enum):
    """Tagged-union discriminator for :class:`_Facet`.

    ``str``-backed so ``FacetMode.WRAP == "wrap"`` and JSON serialization stay
    byte-identical with the legacy string discriminator.
    """

    WRAP = "wrap"
    GRID = "grid"


class WrapOrient(str, Enum):
    """Original orientation of a wrap-mode facet (drives default ``ncols``).

    ``str``-backed for the same byte-stability reason as :class:`FacetMode`.
    """

    COL = "col"
    ROW = "row"


def _coerce_mode(value: Any) -> FacetMode:
    if isinstance(value, FacetMode):
        return value
    try:
        return FacetMode(value)
    except ValueError as exc:
        raise ValueError(f"_Facet.mode_kind must be 'wrap' or 'grid'; got {value!r}") from exc


def _coerce_orient(value: Any) -> Optional[WrapOrient]:
    if value is None or isinstance(value, WrapOrient):
        return value
    try:
        return WrapOrient(value)
    except ValueError as exc:
        raise ValueError(
            f"_Facet.wrap_orient must be 'col', 'row', or None; got {value!r}"
        ) from exc


@dataclass(frozen=True)
class _Facet:
    """Internal facet spec — frozen dataclass replacing the legacy dict shape.

    ``mode_kind`` is the tagged-union discriminator: ``FacetMode.WRAP`` uses
    ``field``; ``FacetMode.GRID`` uses ``row`` and ``col``. Other fields are
    sizing hints honored by :func:`build_facet_dict` when serialising to the
    Rust ``FacetSpec``.

    ``resolve`` carries per-channel scale-resolution modes set via
    ``Chart.share_scale()``.  Keys are channel names (``"x"``, ``"y"``);
    values are ``"shared"`` or ``"independent"``.  Absent keys default to
    ``"shared"`` in the Rust layer.

    ``wrap_orient`` records the original orientation of a wrap facet so
    serialization can infer a side-by-side ``ncols`` for column facets:
    ``WrapOrient.COL`` (``facet(col=X)`` alone → one horizontal row),
    ``WrapOrient.ROW`` (``facet(row=X)`` alone → vertical stack), or ``None``
    (generic ``facet(field=X)`` wrap, which stays a vertical stack unless
    ``ncols`` is given).  Unused for ``FacetMode.GRID``.

    C10: ``mode_kind``/``wrap_orient`` are coerced to enums and field-validity is
    validated at construction, so an invalid mode/field combination raises rather
    than silently mis-serializing. ``mode_kind`` and ``wrap_orient`` still compare
    equal to their string forms (the enums subclass ``str``), so existing call
    sites and the serialized JSON are unaffected.

    #238: ``ncols``/``nrows`` arrive already validated as positive ints
    (:func:`ferrum._validate.coerce_positive_int`) by the public door that
    wrote them, each refusing in its own voice: ``Chart.facet(ncols=/nrows=)``
    and ``Facet(columns=)`` (at channel construction). A carried value from an
    earlier ``_Facet`` passed the same check when it was first written.
    """

    mode_kind: FacetMode
    field: Optional[str] = None
    row: Optional[str] = None
    col: Optional[str] = None
    ncols: Optional[int] = None
    nrows: Optional[int] = None
    resolve: Optional[dict] = None
    wrap_orient: Optional[WrapOrient] = None
    # F-L01-04: title for whichever dimension the column/wrap strip renders
    # (Rust FacetSpec.title — mirrors FacetSpec::column_strip_title's routing:
    # in wrap mode, including a row-only facet lowered to wrap, this is the
    # ONLY title that has any effect). row_title only has an effect once the
    # facet is genuinely grid mode (both row and col set) — see
    # FacetSpec::row_strip_title. Callers that synthesize a wrap-mode `_Facet`
    # from a `FacetRow(title=...)` channel must route that title into `title`,
    # not `row_title`, or it silently has no effect on the rendered strip.
    title: Optional[str] = None
    row_title: Optional[str] = None
    # #176(11) + N3 (spec §6.4): panel order for whichever dimension the
    # column/wrap strip renders -- mirrors `title`'s routing exactly (the
    # ONLY sort that has any effect in wrap mode, including a row-only facet
    # lowered to wrap). `row_sort` only has an effect once the facet is
    # genuinely grid mode -- mirrors `row_title`. Either value is the raw
    # sort spec (a direction string, an explicit list, or a sort-field
    # object), already validated at channel construction
    # (`encoding.base._validate_sort_object`); this module does not
    # re-validate it.
    sort: Optional[Any] = None
    row_sort: Optional[Any] = None

    def __post_init__(self) -> None:
        # Coerce string inputs to enums (frozen → use object.__setattr__) so the
        # existing _Facet(mode_kind="wrap", ...) call sites keep working while the
        # stored value is a validated enum.
        mode = _coerce_mode(self.mode_kind)
        object.__setattr__(self, "mode_kind", mode)
        object.__setattr__(self, "wrap_orient", _coerce_orient(self.wrap_orient))

        # Field-validity by mode (C10): wrap uses `field`, grid uses row/col.
        if mode is FacetMode.WRAP:
            if self.field is None:
                raise ValueError("_Facet wrap mode requires `field`")
        else:  # grid
            if self.row is None and self.col is None:
                raise ValueError("_Facet grid mode requires `row` and/or `col`")
            if self.wrap_orient is not None:
                raise ValueError("_Facet grid mode does not use `wrap_orient`")

    @staticmethod
    def merged_with(
        existing: Optional["_Facet"],
        *,
        declare: str,
        kind: FacetDeclaration,
        field: Optional[str] = None,
        row: Optional[str] = None,
        col: Optional[str] = None,
        title: Optional[str] = None,
        row_title: Optional[str] = None,
        ncols: Optional[int] = None,
        nrows: Optional[int] = None,
        resolve: Optional[dict] = None,
        sort: Optional[Any] = None,
        row_sort: Optional[Any] = None,
    ) -> "_Facet":
        """The one writer of facet state (#176(10b) + N5, spec §6.3).

        *existing* is the chart's prior facet state (``None`` on the first
        facet declaration of a chart). *kind* names which dimension this
        call declares (a :class:`FacetDeclaration`): ``"generic"`` (the ``encode(facet=...)`` door's
        wrap-only declaration -- refuses onto an existing grid), ``"facet_field"``
        (the ``.facet(field=...)`` door's wrap-only declaration -- the
        redeclaration door the ``"generic"`` refusal names, so it replaces
        an existing grid with a wrap instead of refusing), ``"col"``
        (``FacetCol`` / ``.facet(col=...)`` alone), ``"row"`` (``FacetRow``
        / ``.facet(row=...)`` alone), ``"grid"`` (``.facet(row=..., col=...)``
        together, which always fully redeclares both dimensions
        unambiguously), or ``"resolve"`` (``Chart.share_scale()``, which
        changes only ``resolve`` and requires *existing*). A field this call
        does not state survives from *existing*; a field it does state wins
        (last writer). Carrying is by wire field, not by dimension: any
        field the declaration handlers do not explicitly list -- including
        ``resolve`` -- rides through :func:`dataclasses.replace`
        automatically, so a handler that does not yet know about a field
        cannot silently drop it.

        Mode transitions are orientation-aware. Onto an existing grid,
        ``"col"``/``"row"`` replace only that dimension. Onto an existing
        single-dimension facet, the wrap's own orientation decides: a
        ``"col"``/``"row"`` declaration of the SAME orientation as the
        existing wrap replaces the field and stays wrap (``FacetRow('g')``
        then ``FacetRow('h')`` is a row facet on ``'h'``; retitling the same
        field keeps that field); a declaration of the OTHER orientation
        (including a generic, orientation-less prior) promotes to a grid
        that keeps the prior field on its own axis (``FacetRow('h')`` onto a
        column wrap on ``'g'`` is ``row='h', col='g'``; ``FacetCol('k')``
        onto a row facet on ``'g'`` is ``row='g', col='k'``). A ``"generic"``
        declaration onto an existing grid is a contradiction -- it cannot
        state which dimension to keep -- and raises ``ValueError`` naming
        *declare* and the prior grid state; a ``"facet_field"`` declaration
        onto an existing grid is the redeclaration that refusal names, and
        replaces the grid with a wrap on the stated field (unmentioned
        fields survive; ``row_title`` has no home in wrap mode and is
        dropped, not carried).

        *sort*/*row_sort* (#176(11) + N3, spec §6.4) route exactly like
        *title*/*row_title*: *sort* is "this call's own value for whichever
        dimension its own kwarg represents" (the ``col`` dimension for a
        ``"col"`` door, the row-or-lowered-to-wrap dimension for a ``"row"``
        door), and each declaration handler decides -- exactly as it already does
        for title -- whether that value lands in the built ``_Facet``'s
        ``sort`` or ``row_sort`` slot, moving with a promoted field the same
        way its title does.
        """
        stated = _Stated(
            field=field,
            row=row,
            col=col,
            title=title,
            row_title=row_title,
            ncols=ncols,
            nrows=nrows,
            resolve=resolve,
            sort=sort,
            row_sort=row_sort,
        )
        handler = _DECLARATION_HANDLERS[FacetDeclaration(kind)]
        return handler(existing, _prior_of(existing), stated, declare)


class _Prior(Enum):
    """The shape of a chart's facet state before a declaration merges onto it."""

    NONE = "none"
    GRID = "grid"
    WRAP_COL = "wrap_col"
    WRAP_ROW = "wrap_row"
    #: A wrap with no orientation of its own (``Facet``/``.facet(field=...)``).
    WRAP_GENERIC = "wrap_generic"


def _prior_of(existing: Optional[_Facet]) -> _Prior:
    if existing is None:
        return _Prior.NONE
    if existing.mode_kind is FacetMode.GRID:
        return _Prior.GRID
    if existing.wrap_orient is WrapOrient.COL:
        return _Prior.WRAP_COL
    if existing.wrap_orient is WrapOrient.ROW:
        return _Prior.WRAP_ROW
    return _Prior.WRAP_GENERIC


@dataclass(frozen=True)
class _Stated:
    """What one facet declaration states; ``None`` is "not stated"."""

    field: Optional[str]
    row: Optional[str]
    col: Optional[str]
    title: Optional[str]
    row_title: Optional[str]
    ncols: Optional[int]
    nrows: Optional[int]
    resolve: Optional[dict]
    sort: Optional[Any]
    row_sort: Optional[Any]


def _carry(existing: Optional[_Facet], attr: str, value: Any) -> Any:
    """*value* when stated, else *existing*'s value for *attr* (``None`` without one)."""
    if value is not None:
        return value
    return getattr(existing, attr) if existing is not None else None


def _build(existing: Optional[_Facet], **fields: Any) -> _Facet:
    """A fresh ``_Facet`` of *fields*, or *existing* with *fields* replaced.

    ``dataclasses.replace`` copies every field the caller does not list
    (currently just ``resolve``) from *existing*, so a merge that does not
    know about a field cannot silently drop it.
    """
    if existing is None:
        return _Facet(**fields)
    return dataclasses.replace(existing, **fields)


def _wrap(
    existing: Optional[_Facet],
    stated: _Stated,
    field: Optional[str],
    *,
    orient: Optional[WrapOrient],
    title: Optional[str],
    sort: Optional[Any],
) -> _Facet:
    return _build(
        existing,
        mode_kind=FacetMode.WRAP,
        field=field,
        row=None,
        col=None,
        wrap_orient=orient,
        title=title,
        row_title=None,
        sort=sort,
        row_sort=None,
        ncols=_carry(existing, "ncols", stated.ncols),
        nrows=_carry(existing, "nrows", stated.nrows),
    )


def _grid(
    existing: Optional[_Facet],
    stated: _Stated,
    *,
    row: Optional[str],
    col: Optional[str],
    title: Optional[str],
    row_title: Optional[str],
    sort: Optional[Any],
    row_sort: Optional[Any],
) -> _Facet:
    return _build(
        existing,
        mode_kind=FacetMode.GRID,
        field=None,
        row=row,
        col=col,
        wrap_orient=None,
        title=title,
        row_title=row_title,
        sort=sort,
        row_sort=row_sort,
        ncols=_carry(existing, "ncols", stated.ncols),
        nrows=_carry(existing, "nrows", stated.nrows),
    )


def _grid_carrying_labels(
    existing: Optional[_Facet], stated: _Stated, *, row: Optional[str], col: Optional[str]
) -> _Facet:
    """A grid on *row* x *col* whose titles and sorts are stated-or-carried per slot."""
    return _grid(
        existing,
        stated,
        row=row,
        col=col,
        title=_carry(existing, "title", stated.title),
        row_title=_carry(existing, "row_title", stated.row_title),
        sort=_carry(existing, "sort", stated.sort),
        row_sort=_carry(existing, "row_sort", stated.row_sort),
    )


def _declare_resolve(
    existing: Optional[_Facet], prior: _Prior, stated: _Stated, declare: str
) -> _Facet:
    return _build(existing, resolve=stated.resolve)


def _declare_grid(
    existing: Optional[_Facet], prior: _Prior, stated: _Stated, declare: str
) -> _Facet:
    return _grid_carrying_labels(existing, stated, row=stated.row, col=stated.col)


def _declare_facet_field(
    existing: Optional[_Facet], prior: _Prior, stated: _Stated, declare: str
) -> _Facet:
    # Onto a grid this is the redeclaration the generic door's refusal
    # names: the grid is replaced by a wrap, dropping row_title/row_sort (no
    # home in wrap mode) rather than carrying them. A wrap-only declaration
    # states no orientation of its own, so `wrap_orient` carries from
    # `existing` (always `None` on a grid): `.facet(col='g').encode(
    # facet=Facet('k'))` keeps the side-by-side `ncols` a bare `None` would
    # collapse to 1.
    return _wrap(
        existing,
        stated,
        stated.field,
        orient=_carry(existing, "wrap_orient", None),
        title=_carry(existing, "title", stated.title),
        sort=_carry(existing, "sort", stated.sort),
    )


def _declare_generic(
    existing: Optional[_Facet], prior: _Prior, stated: _Stated, declare: str
) -> _Facet:
    if prior is _Prior.GRID:
        raise ValueError(
            f"{declare}: the chart is already a grid facet (row={existing.row!r}, "
            f"col={existing.col!r}); a wrap facet cannot replace it. Use "
            "facet_row=/facet_col= to change a dimension, or call .facet(...) "
            "to redeclare the facet."
        )
    return _declare_facet_field(existing, prior, stated, declare)


def _declare_col(
    existing: Optional[_Facet], prior: _Prior, stated: _Stated, declare: str
) -> _Facet:
    if prior is _Prior.GRID:
        return _grid_carrying_labels(existing, stated, row=existing.row, col=stated.col)
    if prior in (_Prior.WRAP_ROW, _Prior.WRAP_GENERIC):
        # Other orientation: promote, keeping the prior field on its own
        # (row) axis -- and its title/sort with it. The prior wrap's
        # title/sort labeled its own single strip; now that field is the row,
        # so they move to `row_title`/`row_sort`. `title`/`sort` (the new col
        # strip) hold only what this call states about its col field.
        return _grid(
            existing,
            stated,
            row=existing.field,
            col=stated.col,
            title=stated.title,
            row_title=stated.row_title if stated.row_title is not None else existing.title,
            sort=stated.sort,
            row_sort=stated.row_sort if stated.row_sort is not None else existing.sort,
        )
    # No prior, or a column wrap: replace the field, stay a column wrap.
    return _wrap(
        existing,
        stated,
        stated.col,
        orient=WrapOrient.COL,
        title=_carry(existing, "title", stated.title),
        sort=_carry(existing, "sort", stated.sort),
    )


def _declare_row(
    existing: Optional[_Facet], prior: _Prior, stated: _Stated, declare: str
) -> _Facet:
    if prior is _Prior.GRID:
        return _grid_carrying_labels(existing, stated, row=stated.row, col=existing.col)
    if prior in (_Prior.WRAP_COL, _Prior.WRAP_GENERIC):
        # Other orientation: promote, keeping the prior field on its own
        # (col) axis.
        return _grid_carrying_labels(existing, stated, row=stated.row, col=existing.field)
    # No prior, or a row wrap: replace the field, stay a row wrap. The row
    # door's own title/sort arrive as `row_title`/`row_sort`, but a row-alone
    # wrap reads the wire's `title`/`sort` keys, so they land there.
    return _wrap(
        existing,
        stated,
        stated.row,
        orient=WrapOrient.ROW,
        title=_carry(existing, "title", stated.row_title),
        sort=_carry(existing, "sort", stated.row_sort),
    )


_DeclarationHandler = Callable[[Optional[_Facet], _Prior, _Stated, str], _Facet]

_DECLARATION_HANDLERS: dict[FacetDeclaration, _DeclarationHandler] = {
    FacetDeclaration.GENERIC: _declare_generic,
    FacetDeclaration.FACET_FIELD: _declare_facet_field,
    FacetDeclaration.COL: _declare_col,
    FacetDeclaration.ROW: _declare_row,
    FacetDeclaration.GRID: _declare_grid,
    FacetDeclaration.RESOLVE: _declare_resolve,
}


def infer_facet_cardinality(data: Any, col_name: Optional[str]) -> int:
    """Return the number of distinct values in *col_name* within *data*.

    Falls back to 1 when *col_name* is ``None``, *data* is absent, or the column
    is not found — preserving the existing behaviour for cases where inference is
    not possible.
    """
    if col_name is None or data is None:
        return 1
    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            if col_name in data.columns:
                return data[col_name].n_unique()
            return 1
        # PyArrow fallback
        import pyarrow as pa

        tbl = data if isinstance(data, pa.Table) else to_arrow_table(data)
        if col_name in tbl.schema.names:
            col = tbl.column(col_name)
            return len(col.unique())
        return 1
    except Exception:
        return 1


def facet_repeat(facet_dict: dict, frames: tuple) -> list[dict]:
    """The ``repeat`` entries of a facet wire: each operand frame lacking a facet field.

    *frames* holds a ``+`` composition's frame selectors, one per frame
    (``Chart._frame_selectors``: each frame's row marker and own columns). A
    frame whose columns lack the facet's field or row field draws its rows
    in every panel along that field, its statistics running on its own
    rows, as ggplot2 draws a layer whose data lacks the faceting variable. A
    frame holding the field partitions normally. Empty for a chart over one
    frame.
    """
    fields = [facet_dict["field"]] + ([facet_dict["row"]] if "row" in facet_dict else [])
    repeat = []
    for frame in frames:
        lacking = [field for field in fields if field not in frame.columns]
        if lacking:
            repeat.append({"marker": frame.marker, "fields": lacking})
    return repeat


def build_facet_dict(facet: _Facet, data: Any) -> dict:
    """Convert a :class:`_Facet` to the JSON dict Rust's ``FacetSpec`` expects.

    The ``"resolve"`` key is emitted only when at least one channel is set to
    ``"independent"``.  Absent keys default to ``"shared"`` in Rust, so omitting
    the key entirely when all channels are shared keeps the output byte-stable
    with existing goldens.

    *data* is consulted only to infer ``nrows``/``ncols`` cardinality when the
    user did not supply them.
    """
    f = facet
    # Build the resolve sub-dict only when there is at least one non-default
    # (independent) resolution; omit entirely otherwise.
    resolve_dict: dict | None = None
    if f.resolve:
        non_default = {ch: mode for ch, mode in f.resolve.items() if mode == "independent"}
        if non_default:
            resolve_dict = dict(f.resolve)

    if f.mode_kind == FacetMode.WRAP:
        # ncols selection (KG-6 / issue #24): an explicit ncols always wins.
        # A column facet (`facet(col=X)` with no ncols) lays its panels in a
        # single horizontal row, so ncols = n_distinct(field).  Row facets and
        # the generic `facet(field=X)` wrap stay a vertical stack (ncols = 1).
        if f.ncols is not None:
            ncols = f.ncols
        elif f.wrap_orient == WrapOrient.COL:
            ncols = infer_facet_cardinality(data, f.field)  # side-by-side row
        else:  # "row" or generic wrap
            ncols = 1
        d: dict = {"field": f.field, "mode": {"kind": "wrap", "ncols": int(ncols)}}
    else:
        # grid: col is the primary (column) field; row is the secondary (row) field.
        field = f.col if f.col is not None else (f.field or "")
        # Infer nrows/ncols from the data when the user did not supply them.
        # This is the common case: facet(row="a", col="b") without explicit
        # nrows/ncols.  Without inference the geometry defaults to 1×1, dropping
        # all but one panel.
        nrows = f.nrows if f.nrows is not None else infer_facet_cardinality(data, f.row)
        ncols = f.ncols if f.ncols is not None else infer_facet_cardinality(data, f.col)
        d = {
            "field": field,
            "mode": {"kind": "grid", "nrows": int(nrows), "ncols": int(ncols)},
        }
        if f.row is not None:
            d["row"] = f.row

    if resolve_dict is not None:
        d["resolve"] = resolve_dict
    # F-L01-04: title (column/wrap-strip) and row_title (grid row-strip) are
    # both top-level FacetSpec keys, independent of mode.  Emitted only when
    # set, so no-title facets stay byte-identical with pre-fix goldens.
    if f.title is not None:
        d["title"] = f.title
    if f.row_title is not None:
        d["row_title"] = f.row_title
    # #176(11) + N3 (spec §6.4): sort (column/wrap-strip panel order) and
    # row_sort (grid row-strip panel order) mirror title/row_title exactly --
    # both top-level FacetSpec keys, independent of mode, emitted only when
    # set, so a facet with no sort declared stays byte-identical.
    if f.sort is not None:
        d["sort"] = f.sort
    if f.row_sort is not None:
        d["row_sort"] = f.row_sort
    return d
