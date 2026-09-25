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
from typing import Any, Literal, Optional

from ferrum._coerce import to_arrow_table

#: Which dimension a facet door declares in one call to
#: :meth:`_Facet.merged_with`. ``"generic"`` is the ``encode(facet=...)``
#: door's wrap-only declaration (raises onto an existing grid); ``"facet_field"``
#: is ``.facet(field=...)``'s wrap-only declaration (redeclares -- replaces an
#: existing grid with a wrap instead of raising); ``"col"``/``"row"`` declare
#: one grid dimension (``FacetCol``/``FacetRow`` / ``.facet(col=...)``/
#: ``.facet(row=...)`` alone); ``"grid"`` declares both dimensions at once
#: (``.facet(row=..., col=...)``); ``"resolve"`` is ``Chart.share_scale()``'s
#: declaration -- the only field it changes is ``resolve``.
FacetDeclarationKind = Literal["generic", "facet_field", "col", "row", "grid", "resolve"]


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
    (:func:`ferrum._validate.validate_positive_int`) by the public door that
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
        kind: FacetDeclarationKind,
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
        call declares: ``"generic"`` (the ``encode(facet=...)`` door's
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
        field this method's branches do not explicitly list -- including
        ``resolve`` -- rides through :func:`dataclasses.replace`
        automatically, so a branch that does not yet know about a field
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
        door), and each branch below decides -- exactly as it already does
        for title -- whether that value lands in the built ``_Facet``'s
        ``sort`` or ``row_sort`` slot, moving with a promoted field the same
        way its title does.
        """

        def carry(attr: str, value):
            if value is not None:
                return value
            return getattr(existing, attr) if existing is not None else None

        def build(**structural) -> "_Facet":
            # `existing is None`: nothing to carry, construct fresh.
            # Otherwise `dataclasses.replace` copies every field this call
            # doesn't list here (currently just `resolve`) from `existing`
            # automatically.
            if existing is None:
                return _Facet(**structural)
            return dataclasses.replace(existing, **structural)

        def wrap_call(wrap_field: Optional[str], *, wrap_orient, wrap_title, wrap_sort) -> "_Facet":
            return build(
                mode_kind="wrap",
                field=wrap_field,
                row=None,
                col=None,
                wrap_orient=wrap_orient,
                title=wrap_title,
                row_title=None,
                sort=wrap_sort,
                row_sort=None,
                ncols=ncols_v,
                nrows=nrows_v,
            )

        def grid_call(
            *, grid_row, grid_col, grid_title, grid_row_title, grid_sort, grid_row_sort
        ) -> "_Facet":
            return build(
                mode_kind="grid",
                field=None,
                row=grid_row,
                col=grid_col,
                wrap_orient=None,
                title=grid_title,
                row_title=grid_row_title,
                sort=grid_sort,
                row_sort=grid_row_sort,
                ncols=ncols_v,
                nrows=nrows_v,
            )

        if kind == "resolve":
            return build(resolve=resolve)

        ncols_v = carry("ncols", ncols)
        nrows_v = carry("nrows", nrows)
        on_grid = existing is not None and existing.mode_kind == FacetMode.GRID
        on_wrap = existing is not None and existing.mode_kind == FacetMode.WRAP

        if kind == "grid":
            return grid_call(
                grid_row=row,
                grid_col=col,
                grid_title=carry("title", title),
                grid_row_title=carry("row_title", row_title),
                grid_sort=carry("sort", sort),
                grid_row_sort=carry("row_sort", row_sort),
            )

        if kind in ("generic", "facet_field"):
            if on_grid:
                if kind == "generic":
                    raise ValueError(
                        f"{declare}: the chart is already a grid facet (row={existing.row!r}, "
                        f"col={existing.col!r}); a wrap facet cannot replace it. Use "
                        "facet_row=/facet_col= to change a dimension, or call .facet(...) "
                        "to redeclare the facet."
                    )
                # "facet_field": `.facet(field=...)` is the redeclaration door
                # the refusal above names -- falls through to the shared
                # wrap-building return below, dropping row_title/row_sort (no
                # home in wrap mode) rather than carrying them.
            return wrap_call(
                field,
                # A wrap-only declaration states no orientation of its own
                # (`Facet`/`.facet(field=...)` have no orientation
                # parameter), so `wrap_orient` is a field this call does not
                # mention and carries from `existing` per the field rule
                # (spec §6.3) -- `.facet(col='g').encode(facet=Facet('k'))`
                # keeps the side-by-side `ncols` a bare `wrap_orient=None`
                # would silently collapse to 1. `existing.wrap_orient` is
                # always `None` already when `existing` is a grid (the
                # redeclare case), so this needs no on_grid special case.
                wrap_orient=carry("wrap_orient", None),
                wrap_title=carry("title", title),
                wrap_sort=carry("sort", sort),
            )

        if kind == "col":
            if on_grid:
                return grid_call(
                    grid_row=existing.row,
                    grid_col=col,
                    grid_title=carry("title", title),
                    grid_row_title=carry("row_title", row_title),
                    grid_sort=carry("sort", sort),
                    grid_row_sort=carry("row_sort", row_sort),
                )
            if on_wrap and existing.wrap_orient == WrapOrient.COL:
                # same orientation: replace the field, stay wrap.
                return wrap_call(
                    col,
                    wrap_orient="col",
                    wrap_title=carry("title", title),
                    wrap_sort=carry("sort", sort),
                )
            if on_wrap:
                # other orientation: promote, keeping the prior field on its
                # own (row) axis -- and its title/sort with it. The prior
                # wrap's title/sort labeled its own single strip; now that
                # field is the row, so they move to `row_title`/`row_sort`,
                # not `title`/`sort`. `title`/`sort` (the new col strip) hold
                # only what THIS call states about the col field it is
                # declaring; nothing carries into them from the prior wrap,
                # whose title/sort belong to a different axis now.
                return grid_call(
                    grid_row=existing.field,
                    grid_col=col,
                    grid_title=title,
                    grid_row_title=row_title if row_title is not None else existing.title,
                    grid_sort=sort,
                    grid_row_sort=row_sort if row_sort is not None else existing.sort,
                )
            return wrap_call(
                col,
                wrap_orient="col",
                wrap_title=carry("title", title),
                wrap_sort=carry("sort", sort),
            )

        # kind == "row"
        if on_grid:
            return grid_call(
                grid_row=row,
                grid_col=existing.col,
                grid_title=carry("title", title),
                grid_row_title=carry("row_title", row_title),
                grid_sort=carry("sort", sort),
                grid_row_sort=carry("row_sort", row_sort),
            )
        if on_wrap and existing.wrap_orient == WrapOrient.ROW:
            # same orientation: replace the field, stay wrap. The row door's
            # own title/sort arrive via `row_title`/`row_sort` (F-L01-04: a
            # row-alone wrap reads the wire's `title` key, not `row_title`),
            # so those -- not `title`/`sort` -- are this call's stated
            # values here.
            wrap_title = row_title if row_title is not None else existing.title
            wrap_sort = row_sort if row_sort is not None else existing.sort
            return wrap_call(row, wrap_orient="row", wrap_title=wrap_title, wrap_sort=wrap_sort)
        if on_wrap:
            # other orientation: promote, keeping the prior field on its own
            # (col) axis.
            return grid_call(
                grid_row=row,
                grid_col=existing.field,
                grid_title=carry("title", title),
                grid_row_title=carry("row_title", row_title),
                grid_sort=carry("sort", sort),
                grid_row_sort=carry("row_sort", row_sort),
            )
        return wrap_call(row, wrap_orient="row", wrap_title=row_title, wrap_sort=row_sort)


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
