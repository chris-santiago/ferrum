"""Coordinate wrapper types for annotation positioning.

Coordinates in annotation primitives can be expressed in these ways:

- ``float`` / ``int`` — data-space coordinates (the default)
- ``datetime.date`` / ``datetime.datetime`` — temporal data-space coordinate,
  converted to epoch-milliseconds (UTC) to match ferrum's internal temporal
  representation.  Naive ``datetime`` objects are treated as UTC.
- ISO-8601 date or datetime string — parsed and converted to epoch-milliseconds.
- Non-ISO-8601 string — treated as an ordinal category label, stored as
  ``OrdinalCategoryCoord`` and resolved against the axis ordinal domain at
  render time.
- ``PixelCoord`` — absolute pixel offset from the plot origin
- ``NormCoord`` — normalized [0, 1] fraction of the plot area
- ``OrdinalCategoryCoord`` — a category label resolved to a band center at
  render time via the chart's ordinal domain

Use the [px][ferrum.px] and [norm][ferrum.norm] factory functions as readable shorthands.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any, TypeAlias, Union


@dataclass(frozen=True)
class PixelCoord:
    """An absolute pixel coordinate relative to the plot origin.

    Parameters
    ----------
    value : float
        Pixel offset from the plot's top-left corner.
    """

    value: float


@dataclass(frozen=True)
class NormCoord:
    """A normalized coordinate in [0, 1] relative to the plot area.

    Parameters
    ----------
    value : float
        0.0 is the left/bottom edge, 1.0 is the right/top edge.
    """

    value: float


@dataclass(frozen=True)
class OrdinalCategoryCoord:
    """A category label coordinate that resolves to its band center at render time.

    Use this (or rely on automatic coercion from a non-ISO-8601 string) when
    annotating a chart whose axis is an ordinal (categorical) scale.  The
    category string is matched against the axis ordinal domain and the
    annotation is placed at the band center for that category.

    Parameters
    ----------
    value : str
        Category label as it appears in the data column.
    """

    value: str


# CoordValue accepts numbers, temporal Python types, ISO strings, coordinate
# wrappers, or ordinal category labels.  Temporal ISO strings are converted to
# epoch-ms at serialization time; ordinal category labels are resolved to norm
# coordinates against the chart's domain at render time.
CoordValue: TypeAlias = Union[
    float,
    int,
    _dt.date,
    _dt.datetime,
    str,
    PixelCoord,
    NormCoord,
    OrdinalCategoryCoord,
]

# Unix epoch as a date — used by temporal_coord_to_epoch_ms.
_EPOCH_DATE = _dt.date(1970, 1, 1)
_MS_PER_DAY = 86_400_000


def _is_iso8601_string(s: str) -> bool:
    """Return True if *s* can be parsed as an ISO-8601 date or datetime string.

    Tests ``YYYY-MM-DD`` (date) and ``YYYY-MM-DDTHH:...`` (datetime) forms.
    Does not validate full ISO-8601 generality — just the subset that
    ``temporal_coord_to_epoch_ms`` accepts.

    Parameters
    ----------
    s : str
        Candidate string.

    Returns
    -------
    bool
    """
    try:
        _dt.date.fromisoformat(s)
        return True
    except ValueError:
        pass
    try:
        _dt.datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


def temporal_coord_to_epoch_ms(value: _dt.date | _dt.datetime | str) -> float:
    """Convert a temporal coordinate value to epoch-milliseconds (UTC).

    This is the canonical conversion used by ferrum's annotation layer to
    align date/datetime coordinates with the epoch-ms representation that the
    Rust renderer uses for temporal axes (same units as ``_coerce.py`` and the
    D3 time scale).

    Parameters
    ----------
    value : datetime.date, datetime.datetime, or str
        - ``datetime.date`` — midnight UTC on that calendar date.
        - ``datetime.datetime`` — naive datetimes are treated as UTC (consistent
          with ``_coerce.py``'s handling of polars ``Date``/``Datetime`` columns).
          Aware datetimes are converted to UTC before computing the epoch offset.
        - ``str`` — parsed as an ISO-8601 date (``YYYY-MM-DD``) or datetime
          (``YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]``).  Date-only strings are
          treated as midnight UTC.

    Returns
    -------
    float
        Milliseconds since the Unix epoch (1970-01-01T00:00:00 UTC).

    Raises
    ------
    ValueError
        If a string cannot be parsed as an ISO-8601 date or datetime.

    Examples
    --------
    >>> from datetime import date, datetime
    >>> temporal_coord_to_epoch_ms(date(2020, 6, 1))
    1590969600000.0
    >>> temporal_coord_to_epoch_ms("2020-06-01")
    1590969600000.0
    """
    if isinstance(value, str):
        # Try date first (YYYY-MM-DD), then datetime.
        try:
            value = _dt.date.fromisoformat(value)
        except ValueError:
            try:
                value = _dt.datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse annotation coordinate {value!r} as an ISO-8601 date or "
                    f"datetime. Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'."
                ) from exc

    # datetime must be checked before date because datetime is a subclass of date.
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            # Treat naive datetime as UTC — consistent with _coerce.py which casts
            # polars Date/Datetime columns without timezone info to timestamp[ms].
            seconds = (value - _dt.datetime(1970, 1, 1)).total_seconds()
        else:
            epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
            seconds = (value - epoch).total_seconds()
        return seconds * 1000.0

    # Plain date: midnight UTC.
    days = (value - _EPOCH_DATE).days
    return float(days * _MS_PER_DAY)


def px(value: float) -> PixelCoord:
    """Construct a pixel-space coordinate.

    Parameters
    ----------
    value : float
        Pixel offset.

    Returns
    -------
    PixelCoord

    Examples
    --------
    >>> px(50)
    PixelCoord(value=50)
    """
    return PixelCoord(value)


def norm(value: float) -> NormCoord:
    """Construct a normalized [0, 1] coordinate.

    Parameters
    ----------
    value : float
        Normalized fraction (0.0–1.0).

    Returns
    -------
    NormCoord

    Examples
    --------
    >>> norm(0.5)
    NormCoord(value=0.5)
    """
    return NormCoord(value)


# ---------------------------------------------------------------------------
# Coordinate coercion and serialization
#
# These functions are the single source of truth for turning a raw annotation
# coordinate (numeric, temporal, ISO string, ordinal label, or wrapper) into
# either a stored ``CoordValue`` (for primitive dataclass fields) or a
# renderer-serializable form (for ``to_dict()``).  They were previously split
# across ``annotations.py`` (``_coerce_coord``/``_coerce_coord_to_numeric``)
# and ``annotation/primitives.py`` (``_coord``/``_sanitize_coord``), which
# duplicated the raw-input classification.  The classification now lives once
# in :func:`_classify_raw_coord`; both the coerce path and the serialize path
# consume it.
# ---------------------------------------------------------------------------


def _classify_raw_coord(v: CoordValue) -> CoordValue:
    """Resolve a raw annotation coordinate to a canonical stored form.

    This is the shared classification both ``_coerce_coord`` (which stores the
    result in a primitive) and ``_coord`` (which serializes it) build on:

    - ``PixelCoord`` / ``NormCoord`` / ``OrdinalCategoryCoord`` pass through
      unchanged — the caller already expressed precise intent.
    - ``datetime.date`` / ``datetime.datetime`` → epoch-milliseconds float.
    - String that IS an ISO-8601 date/datetime → epoch-milliseconds float.
    - String that is NOT ISO-8601 → ``OrdinalCategoryCoord`` (resolved against
      the chart's ordinal domain at render time).
    - Numeric (float, int) → float (data-space value).

    The returned value is always a type that :func:`_coord` knows how to
    serialize.
    """
    if isinstance(v, (PixelCoord, NormCoord, OrdinalCategoryCoord)):
        return v
    # datetime must be checked before date (datetime is a subclass of date).
    if isinstance(v, (_dt.datetime, _dt.date)):
        return temporal_coord_to_epoch_ms(v)
    if isinstance(v, str):
        if _is_iso8601_string(v):
            return temporal_coord_to_epoch_ms(v)
        return OrdinalCategoryCoord(v)
    return float(v)


def _coerce_coord(v: CoordValue) -> CoordValue:
    """Normalize an annotation positional coordinate for storage in a primitive.

    Rules (see :func:`_classify_raw_coord`):
    - ``PixelCoord`` / ``NormCoord`` / ``OrdinalCategoryCoord`` pass through.
    - ``datetime.date`` / ``datetime.datetime`` → epoch-milliseconds float.
    - ISO-8601 string → epoch-milliseconds float.
    - Non-ISO-8601 string → ``OrdinalCategoryCoord``.
    - Numeric → float.

    The returned value is always a type that :func:`_coord` knows how to
    serialize.
    """
    return _classify_raw_coord(v)


def _coerce_coord_to_numeric(v: CoordValue) -> float:
    """Convert a coordinate to a plain float for use in a polars DataFrame column.

    ``PixelCoord`` and ``NormCoord`` do not have a meaningful data-space value,
    so they are mapped to 0.0 as a placeholder (the primitive carries the real
    coord; the DataFrame column for the mark_rule is only used as a shape hint
    and is not used for rendering when the primitive is present).

    ``OrdinalCategoryCoord`` is mapped to 0.0 similarly — its true pixel
    position is resolved at render time from the ordinal domain.
    """
    if isinstance(v, (PixelCoord, NormCoord, OrdinalCategoryCoord)):
        return 0.0
    return float(v)  # already a float after _coerce_coord


def _sanitize_coord(v: Any) -> Any:
    """Replace NaN/Inf with 0.0 so JSON serialization doesn't crash."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v


def _coord(v: CoordValue) -> Any:
    """Normalize a CoordValue to a renderer-serializable form.

    Temporal values (``date``, ``datetime``, ISO strings) are converted to
    epoch-milliseconds (UTC) so they align with the Rust renderer's temporal
    scale — the same units that ``_coerce.py`` produces for temporal data
    columns.  Numeric values and coordinate wrappers pass through unchanged.
    ``OrdinalCategoryCoord`` is serialized as ``{"category": value}``; this
    dict is resolved to a ``{"norm": ...}`` entry by ``_resolve_chart_config``
    before the annotation list is sent to the Rust renderer.
    """
    if isinstance(v, PixelCoord):
        return {"px": _sanitize_coord(v.value)}
    if isinstance(v, NormCoord):
        return {"norm": _sanitize_coord(v.value)}
    if isinstance(v, OrdinalCategoryCoord):
        return {"category": v.value}
    # datetime must be checked before date (datetime is a subclass of date).
    if isinstance(v, (_dt.datetime, _dt.date)):
        return temporal_coord_to_epoch_ms(v)
    if isinstance(v, str):
        # Plain strings reaching _coord() should already have been coerced to
        # OrdinalCategoryCoord or epoch-ms by _coerce_coord().  Strings that
        # arrive here are treated as ISO-8601 (legacy path / direct primitive
        # construction); non-ISO strings raise so the caller is aware.
        return temporal_coord_to_epoch_ms(v)
    return _sanitize_coord(v)  # plain numeric — data-space
