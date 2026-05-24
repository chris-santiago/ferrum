"""Named format preset registry for axis and legend labels.

Presets resolve to d3-format strings (for numeric) or strftime-style strings
(for time).  The special sentinel ``"__ordinal__"`` is resolved on the Rust
side into ordinal suffix formatting (1st, 2nd, 3rd, …).
"""

from __future__ import annotations


NUMERIC_PRESETS: dict[str, str] = {
    "integer": ",.0f",
    "decimal": ",.2f",
    "decimal1": ",.1f",
    "percent": ".1%",
    "percent_int": ".0%",
    "si": ".2s",
    "currency": "$,.0f",
    "currency_cents": "$,.2f",
    "compact": ".2~s",
    "scientific": ".2e",
    "ordinal": "__ordinal__",  # sentinel — resolved in Rust
}

TIME_PRESETS: dict[str, str] = {
    "date_short": "%b %-d",
    "date_long": "%B %-d, %Y",
    "date_iso": "%Y-%m-%d",
    "month": "%b",
    "month_year": "%b %Y",
    "year": "%Y",
    "time": "%H:%M",
    "time_12h": "%-I:%M %p",
    "datetime": "%b %-d, %H:%M",
}

ALL_PRESETS: dict[str, str] = {**NUMERIC_PRESETS, **TIME_PRESETS}


def resolve_format(name: str) -> str:
    """Resolve a named preset to its d3-format or strftime string.

    Parameters
    ----------
    name : str
        A key from ``ALL_PRESETS``.

    Returns
    -------
    str
        The corresponding format string.

    Raises
    ------
    ValueError
        If *name* is not a recognized preset key.

    Examples
    --------
    >>> resolve_format("percent")
    '.1%'
    >>> resolve_format("date_iso")
    '%Y-%m-%d'
    """
    try:
        return ALL_PRESETS[name]
    except KeyError:
        known = sorted(ALL_PRESETS)
        raise ValueError(f"Unknown format preset {name!r}. Known presets: {known}") from None


def is_time_preset(name: str) -> bool:
    """Return True if *name* is a time format preset.

    Parameters
    ----------
    name : str
        A preset name (does not need to be a valid preset; returns False for
        unknown names).

    Returns
    -------
    bool

    Examples
    --------
    >>> is_time_preset("date_iso")
    True
    >>> is_time_preset("percent")
    False
    """
    return name in TIME_PRESETS
