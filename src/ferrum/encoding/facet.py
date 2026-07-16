"""Facet encoding channels (Facet, FacetRow, FacetCol).

Each channel references the ``FACET`` role from `_honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum.encoding._honored import FACET
from ferrum.encoding.base import ChannelBase


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
        Facet panel title override.  When omitted the field name is used.

    Notes
    -----
    ``columns`` (number of facets per row) is accepted as a kwarg but is
    reserved for future use (no-op today) — it triggers a one-time
    deprecation warning.  If you need column-wrap control today, use
    ``Chart.facet(ncols=N)`` directly.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet=fm.Facet("species")).mark_point()
    >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().facet("species", ncols=3)
    """

    _channel_name = "facet"
    _honored_kwargs = FACET


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
        Row-facet title override.  When omitted the field name is used.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet_row=fm.FacetRow("year")).mark_point()
    """

    _channel_name = "facet_row"

    _honored_kwargs = FACET


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
        Column-facet title override.  When omitted the field name is used.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", facet_col=fm.FacetCol("species")).mark_point()
    """

    _channel_name = "facet_col"

    _honored_kwargs = FACET
