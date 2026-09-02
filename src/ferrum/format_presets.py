"""Named format preset registry for axis and legend labels.

Presets resolve to d3-format strings (for numeric) or strftime-style strings
(for time).  The special sentinel ``"__ordinal__"`` is resolved on the Rust
side into ordinal suffix formatting (1st, 2nd, 3rd, …).

``resolve_format_or_raw`` is the single permissive entry point every
emission surface (chart-level ``AxisConfig``, per-channel
``fm.Axis(label_format=)``, per-channel ``fm.Legend(format=)``, encoding
``format=``, and the raw-dict normalize paths) routes through so a preset
name never reaches the Rust d3-format parser unresolved (NF-B1,
2026-09-02). Unlike ``resolve_format``, it never raises: an unrecognized
name is an honest raw d3-format/strftime spec supplied by the caller.
"""

from __future__ import annotations

from typing import Any

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


def resolve_format_or_raw(value: Any) -> tuple[Any, str | None]:
    """Resolve *value* as a named preset, or pass it through as a raw format spec.

    The permissive counterpart to :func:`resolve_format`. Every format-bearing
    surface (``AxisConfig.label_format``, ``fm.Axis(label_format=)``,
    ``fm.Legend(format=)``, encoding ``format=``, and the raw-dict axis/legend
    normalize paths) calls this instead of ``resolve_format`` so a caller can
    supply either a named preset or an already-valid raw d3-format/strftime
    string interchangeably — an unrecognized name is never treated as an
    error, since it may be a legitimate raw spec (NF-B1).

    Parameters
    ----------
    value : Any
        Ordinarily either a key from :data:`ALL_PRESETS`, or a raw
        d3-format / strftime string supplied directly by the caller. Callers
        on all four permissive surfaces pass through whatever the user
        supplied without pre-validating its type, so a non-``str`` value
        (e.g. an unhashable list) can reach here too; see Returns.

    Returns
    -------
    tuple[Any, str | None]
        ``(resolved_spec, format_type)``. For a recognized preset,
        *resolved_spec* is the preset's d3-format/strftime string (or the
        ``"__ordinal__"`` sentinel) and *format_type* is ``"number"`` or
        ``"time"``. For an unrecognized name, *value* passes through
        unchanged as *resolved_spec* and *format_type* is ``None`` (the
        consuming surface's own default, or an explicitly-set format type,
        applies). For a non-``str`` *value*, it likewise passes through
        unchanged with *format_type* ``None`` — this function never raises;
        the caller's own typed refusal (dataclass type checking, Rust wire
        deserialization) is what surfaces the error for a bad type.

    Examples
    --------
    >>> resolve_format_or_raw("percent")
    ('.1%', 'number')
    >>> resolve_format_or_raw("date_iso")
    ('%Y-%m-%d', 'time')
    >>> resolve_format_or_raw(",.2f")
    (',.2f', None)
    """
    if not isinstance(value, str):
        # Not a preset name and not a raw spec this function can resolve;
        # pass through unchanged so the caller's own typed refusal (dataclass
        # type checking, Rust wire deserialization, etc.) still fires instead
        # of a bare TypeError from the `in NUMERIC_PRESETS` membership test.
        return value, None
    if value in NUMERIC_PRESETS:
        return NUMERIC_PRESETS[value], "number"
    if value in TIME_PRESETS:
        return TIME_PRESETS[value], "time"
    return value, None


def resolve_format_field(
    raw_value: Any, explicit_format_type: str | None
) -> tuple[Any, str | None]:
    """Resolve a (format, format_type) field pair for one ``to_dict()`` call.

    Shared by :meth:`ferrum.axis.Axis.to_dict` and
    :meth:`ferrum.legend.Legend.to_dict` (and the raw-dict axis/legend
    normalize paths), which each carry a format field (``label_format`` /
    ``format``) alongside a sibling format-type field (``label_format_type``
    / ``format_type``) that the caller may set explicitly. An explicitly-set
    *explicit_format_type* always wins over the type a preset resolves to
    (explicit-format-wins); when the caller left it unset, the preset's own
    type (if any) fills it.

    Parameters
    ----------
    raw_value : Any, optional
        The format field's raw value (ordinarily a preset name or raw spec
        string), or ``None`` if unset. The raw-dict normalize paths forward
        whatever the caller put in the dict without pre-validating its type;
        see :func:`resolve_format_or_raw` for the non-``str`` passthrough
        contract.
    explicit_format_type : str, optional
        The sibling format-type field's value as set by the caller, or
        ``None`` if unset.

    Returns
    -------
    tuple[Any, str | None]
        ``(resolved_spec, resolved_format_type)``. Both are ``None`` when
        *raw_value* is ``None`` (nothing to resolve); *resolved_format_type*
        is ``explicit_format_type`` when given, else the preset-derived type.
    """
    if raw_value is None:
        return None, explicit_format_type
    spec, derived_type = resolve_format_or_raw(raw_value)
    resolved_type = explicit_format_type if explicit_format_type is not None else derived_type
    return spec, resolved_type


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
