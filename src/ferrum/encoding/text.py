"""Text and tooltip encoding channels (Text, Detail, Tooltip, Href, ...)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


class Text(ChannelBase):
    """Text channel — maps a field to text-mark content.

    Renders each data point's field value as a text label.  Primarily used
    with the ``mark_text`` mark.

    Parameters
    ----------
    field : str
        Column name whose values are rendered as text.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    format : str, optional
        Number or date format string (e.g. ``".2f"`` for two decimal places,
        ``"%b %Y"`` for abbreviated month and year).
    formatType : str, optional
        Format type hint; ``"number"`` or ``"time"``.  Used in combination
        with ``format``.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", text=fm.Text("label"))
    >>> fm.Chart(df).encode(x="x", y="y", text=fm.Text("value", format=".1f"))
    """

    _channel_name = "text"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "format", "formatType"])


class Detail(ChannelBase):
    """Detail channel — adds a field to the encoding without a visual variable.

    Groups marks by the levels of the field without mapping those levels to
    any visual property (color, size, shape, etc.).  Useful for drawing one
    line per group in a line chart.

    Parameters
    ----------
    field : str
        Column name to group by.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="year", y="value", detail=fm.Detail("series"))
    """

    _channel_name = "detail"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Tooltip(ChannelBase):
    """Tooltip channel — specifies which fields appear in the hover tooltip.

    Accepts one or more field names (as strings or ``TooltipField`` helpers)
    that are shown when the viewer hovers over a mark in an interactive
    renderer.

    Parameters
    ----------
    *fields : str or TooltipField
        One or more column names or ``TooltipField(...)`` instances to
        include in the tooltip.  Passing a single string is equivalent to
        ``Tooltip(TooltipField(field))``.

    Notes
    -----
    ``type`` kwarg is accepted but is reserved for future use (no-op today)
    — it triggers a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg",
    ...     tooltip=fm.Tooltip("hp", "mpg"))
    >>> fm.Chart(df).encode(x="hp", y="mpg",
    ...     tooltip=fm.Tooltip("hp", fm.TooltipField("cyl", title="Cylinders")))
    """

    _channel_name = "tooltip"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])

    def __init__(self, *fields, **kwargs):
        # Tooltip(*fields) is a special case: takes a list of fields, not just one
        if len(fields) == 1:
            super().__init__(fields[0], **kwargs)
            self._field_list = [fields[0]]
        else:
            super().__init__(None, **kwargs)
            self._field_list = list(fields)


class TooltipField(ChannelBase):
    """Helper for an individual tooltip field with optional title and format.

    Used inside ``Tooltip(*fields)`` to customise how a single column is
    displayed in the hover tooltip.  Not used as a top-level encoding channel.

    Parameters
    ----------
    field : str
        Column name.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    title : str, optional
        Custom tooltip label for this field.
    format : str, optional
        Number or date format string (e.g. ``".1f"``, ``"%Y-%m-%d"``).
    formatType : str, optional
        Format type hint; ``"number"`` or ``"time"``.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Tooltip("hp", fm.TooltipField("mpg", title="MPG", format=".1f"))
    """

    _channel_name = "tooltip_field"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "title", "format", "formatType"])


class Href(ChannelBase):
    """URL-link channel — maps a field to a clickable URL.

    When the chart is rendered in an interactive renderer, marks become
    clickable hyperlinks pointing to the URL stored in ``field``.

    Parameters
    ----------
    field : str
        Column name containing the URL string for each mark.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.  Interactive renderers
    only; SVG export does not embed hyperlinks.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", href=fm.Href("url"))
    """

    _channel_name = "href"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Description(ChannelBase):
    """Accessibility description channel — maps a field to per-mark alt text.

    The description text is used by screen readers and other accessibility
    tools to describe each individual mark.

    Parameters
    ----------
    field : str
        Column name whose values are used as the accessibility description.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", description=fm.Description("alt_text"))
    """

    _channel_name = "description"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Key(ChannelBase):
    """Key channel — maps a field to a unique key per mark.

    Provides a stable identity for each mark when joining across data
    updates (e.g. animated transitions or streaming data).

    Parameters
    ----------
    field : str
        Column name whose values uniquely identify each mark.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", key=fm.Key("id"))
    """

    _channel_name = "key"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
