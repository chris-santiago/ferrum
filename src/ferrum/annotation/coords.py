"""Coordinate wrapper types for annotation positioning.

Coordinates in annotation primitives can be expressed in three ways:

- ``float`` / ``int`` — data-space coordinates (the default)
- ``datetime.date`` / ``datetime.datetime`` — temporal data-space coordinate,
  converted to epoch-milliseconds (UTC) to match ferrum's internal temporal
  representation.  Naive ``datetime`` objects are treated as UTC.
- ISO-8601 date or datetime string — parsed and converted to epoch-milliseconds.
- ``PixelCoord`` — absolute pixel offset from the plot origin
- ``NormCoord`` — normalized [0, 1] fraction of the plot area

Use the :func:`px` and :func:`norm` factory functions as a readable shorthand.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TypeAlias, Union


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


# CoordValue accepts numbers, temporal Python types, ISO strings, or coordinate
# wrappers.  Temporal values are converted to epoch-ms at serialization time
# by _coord() in primitives.py so they align with ferrum's internal scale.
CoordValue: TypeAlias = Union[
    float,
    int,
    _dt.date,
    _dt.datetime,
    str,
    PixelCoord,
    NormCoord,
]

# Unix epoch as a date — used by temporal_coord_to_epoch_ms.
_EPOCH_DATE = _dt.date(1970, 1, 1)
_MS_PER_DAY = 86_400_000


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
