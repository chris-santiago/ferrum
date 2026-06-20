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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ferrum._coerce import to_arrow_table


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
    """

    mode_kind: FacetMode
    field: Optional[str] = None
    row: Optional[str] = None
    col: Optional[str] = None
    ncols: Optional[int] = None
    nrows: Optional[int] = None
    resolve: Optional[dict] = None
    wrap_orient: Optional[WrapOrient] = None

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
        if resolve_dict is not None:
            d["resolve"] = resolve_dict
        return d
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
    return d
