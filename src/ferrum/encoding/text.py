"""Text and tooltip encoding channels (Text, Detail, Tooltip, Href, ...).

Each channel references one named role from `_honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum.encoding._honored import BARE, TEXT_FORMATTED, TEXT_FORMATTED_TITLED
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
    format_type : str, optional
        Format type hint; ``"number"`` or ``"time"``. Used in combination
        with ``format``. ``formatType`` is accepted as a Vega-compat alias
        and normalizes to the same wire key.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", text=fm.Text("label"))
    >>> fm.Chart(df).encode(x="x", y="y", text=fm.Text("value", format=".1f"))
    """

    _channel_name = "text"

    _honored_kwargs = TEXT_FORMATTED


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="year", y="value", detail=fm.Detail("series"))
    """

    _channel_name = "detail"

    _honored_kwargs = BARE


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg",
    ...     tooltip=fm.Tooltip("hp", "mpg"))
    >>> fm.Chart(df).encode(x="hp", y="mpg",
    ...     tooltip=fm.Tooltip("hp", fm.TooltipField("cyl", title="Cylinders")))
    """

    _channel_name = "tooltip"

    _honored_kwargs = BARE

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
    format_type : str, optional
        Format type hint; ``"number"`` or ``"time"``. ``formatType`` is
        accepted as a Vega-compat alias and normalizes to the same wire key.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg",
    ...     tooltip=fm.Tooltip("hp", fm.TooltipField("mpg", title="MPG", format=".1f")))
    """

    _channel_name = "tooltip_field"

    _honored_kwargs = TEXT_FORMATTED_TITLED


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
    Interactive renderers only; SVG export does not embed hyperlinks.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", href=fm.Href("url"))
    """

    _channel_name = "href"

    _honored_kwargs = BARE


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", description=fm.Description("alt_text"))
    """

    _channel_name = "description"

    _honored_kwargs = BARE


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", key=fm.Key("id"))
    """

    _channel_name = "key"

    _honored_kwargs = BARE


class Url(ChannelBase):
    """Image URL channel — maps a field to a base64 data URL for ``mark_image`` tiles.

    Each row provides a ``data:image/...;base64,<payload>`` URL that is
    placed as an image tile at the position given by the ``x`` and ``y``
    encodings.  Used exclusively with [mark_image][ferrum.Chart.mark_image].

    Parameters
    ----------
    field : str
        Column name containing the base64 data URL for each tile.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type.  Inferred from the column dtype when omitted (typically
        ``"N"`` for string columns).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).mark_image().encode(x="x:Q", y="y:Q", url=fm.Url("data_url"))
    """

    _channel_name = "url"

    _honored_kwargs = BARE
