"""Facet encoding channels (Facet, FacetRow, FacetCol).

Each channel references the ``FACET`` role from `_honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum._validate import validate_positive_int
from ferrum.encoding._honored import FACET, FACET_WRAP
from ferrum.encoding.base import _FACET_COLUMNS_MESSAGE, ChannelBase


class Facet(ChannelBase):
    """Facet channel — splits a chart into a grid by levels of a field.

    Wraps the chart into a faceted layout where each panel shows data for
    one level of the facet field.  Pass this channel object to
    ``Chart.encode(facet=...)``.

    Parameters
    ----------
    field : str
        Column name to facet by.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type.  Inferred from the column dtype when omitted; ``"N"`` or
        ``"O"`` are the most common choices.
    title : str, optional
        Facet strip title.  When set, the strip text renders
        ``"{title}: {value}"``; when omitted, the strip shows the bare
        value (not the field name).
    columns : int, optional
        Maximum number of panels per row (wrap-count).  Writes the same
        sink as ``Chart.facet(ncols=N)``; when both are used, whichever is
        called last wins.  ``None`` lets the renderer choose.
    sort : optional
        Panel order — the same sort vocabulary the positional channels
        accept: ``"ascending"``/``"descending"``, an explicit list of
        values, or a sort-field object (``{"field": ..., "op": ...,
        "order": ...}``) ordering by another field's aggregate. A sort
        object without a string ``"field"``, or with an unrecognized
        ``"op"``/``"order"``, raises ``ValueError`` at construction naming
        ``Facet(sort=...)`` and the value.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet=fm.Facet("species")).mark_point()
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet=fm.Facet("species", columns=3)).mark_point()
    >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().facet("species", ncols=3)
    """

    _channel_name = "facet"
    _honored_kwargs = FACET_WRAP

    def _validate(self) -> None:
        super()._validate()
        validate_positive_int("Facet(columns=...)", self._kwargs.get("columns"))


class FacetRow(ChannelBase):
    """Facet-row channel — splits a chart into rows by levels of a field.

    Pass this channel object to ``Chart.encode(facet_row=...)`` to create a
    row-faceted layout where each row shows data for one level of the field.

    Parameters
    ----------
    field : str
        Column name to facet rows by.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type.  Inferred from the column dtype when omitted; ``"N"`` or
        ``"O"`` are the most common choices.
    title : str, optional
        Row-strip title.  When set, the strip text renders
        ``"{title}: {value}"``; when omitted, the strip shows the bare
        value (not the field name). NOTE: ``FacetRow`` used alone (no
        ``facet_col``) lowers to a wrap-mode facet where the wrap/column
        strip is the one rendered, so ``title`` here drives THAT strip; in
        grid mode (used together with ``facet_col``), ``title`` drives the
        row-header strip.
    sort : optional
        Row order — same vocabulary and construction-time validation as
        ``Facet(sort=...)``; ordered by the same strip ``title`` follows
        (the wrap/column strip alone, or the row-header strip in grid mode).

    Notes
    -----
    ``columns`` is **not** honored (2026-09-23, #123): row/column faceting
    has no wrap-count to apply it to, so ``FacetRow(field, columns=N)``
    raises ``ValueError`` at construction naming ``Facet(columns=...)`` for
    wrap-mode panel-per-row counts instead.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet_row=fm.FacetRow("year")).mark_point()
    """

    _channel_name = "facet_row"

    _honored_kwargs = FACET
    _refused_kwargs = {"columns": _FACET_COLUMNS_MESSAGE}


class FacetCol(ChannelBase):
    """Facet-column channel — splits a chart into columns by levels of a field.

    Pass this channel object to ``Chart.encode(facet_col=...)`` to create a
    column-faceted layout where each column shows data for one level of the
    field.

    Parameters
    ----------
    field : str
        Column name to facet columns by.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type.  Inferred from the column dtype when omitted; ``"N"`` or
        ``"O"`` are the most common choices.
    title : str, optional
        Column/wrap-strip title.  When set, the strip text renders
        ``"{title}: {value}"``; when omitted, the strip shows the bare
        value (not the field name).
    sort : optional
        Column order — same vocabulary and construction-time validation as
        ``Facet(sort=...)``; ordered on the column/wrap strip.

    Notes
    -----
    ``columns`` is **not** honored (2026-09-23, #123): row/column faceting
    has no wrap-count to apply it to, so ``FacetCol(field, columns=N)``
    raises ``ValueError`` at construction naming ``Facet(columns=...)`` for
    wrap-mode panel-per-row counts instead.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet_col=fm.FacetCol("species")).mark_point()
    """

    _channel_name = "facet_col"

    _honored_kwargs = FACET
    _refused_kwargs = {"columns": _FACET_COLUMNS_MESSAGE}
