"""Coordinate wrapper types for annotation positioning.

This module also owns :func:`temporal_coord_to_epoch_ms`, the ONE Python-side
temporal→epoch-ms conversion. It is not annotation-only: ``BreakAxis.gap``
(``ferrum.structural``) and the encoding layer's TimeScale-domain coercion
(``ferrum.encoding._scale``) both route through it, so the set of temporal
types ferrum accepts is decided here once rather than per consumer.

Coordinates in annotation primitives can be expressed in these ways:

- ``float`` / ``int`` — data-space coordinates (the default)
- ``datetime.date`` / ``datetime.datetime`` — temporal data-space coordinate,
  converted to epoch-milliseconds (UTC) to match ferrum's internal temporal
  representation.  Naive ``datetime`` objects are treated as UTC.
  ``pandas.Timestamp`` rides this branch (it is a ``datetime.datetime``
  subclass).
- ``numpy.datetime64`` — the same, via an exact microsecond conversion that
  refuses an out-of-range value rather than letting a fixed-width cast wrap.
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
from typing import TYPE_CHECKING, Any, TypeAlias, Union

if TYPE_CHECKING:  # numpy is imported lazily at runtime (see _is_datetime64)
    import numpy as np


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
        0.0 is the left/top edge, 1.0 is the right/bottom edge. The y axis
        follows SVG's downward-increasing convention (top = 0.0, bottom =
        1.0); the x axis is unaffected (left = 0.0, right = 1.0).
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
    "np.datetime64",
    str,
    PixelCoord,
    NormCoord,
    OrdinalCategoryCoord,
]

# Unix epoch as a date — used by temporal_coord_to_epoch_ms.
_EPOCH_DATE = _dt.date(1970, 1, 1)
_MS_PER_DAY = 86_400_000

#: Refusal text for a not-a-time input (``pandas.NaT``,
#: ``numpy.datetime64('NaT')``). One message for both spellings: they are the
#: same mistake, and a consumer that prefixes its own label (``BreakAxis.gap[0]:
#: ...``) should not have to know which one it caught.
_NOT_A_TIME = "is NaT (not-a-time), which is not a valid temporal value"


class _TemporalParseError(ValueError):
    """A string temporal value that is not an ISO-8601 date or datetime.

    A distinct type — not a bare ``ValueError`` — because every consumer of
    :func:`temporal_coord_to_epoch_ms` re-words THIS failure in its own
    vocabulary (a bad date in a break gap is a break-gap mistake; in a scale
    domain, a scale-domain mistake) while passing the seam's other refusals
    (NaT, out-of-range ``datetime64``) through as-is. Catching plain
    ``ValueError`` would re-word all three as "cannot parse", which is false
    for two of them.
    """


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


# Exact microseconds-per-tick ratio (numerator, denominator) for every unit
# a ``numpy.datetime64`` dtype can carry: Y M W D h m s ms us ns ps fs as.
# ``W`` through ``as`` are genuine fixed-length durations, so their ratios
# are mathematically exact -- e.g. one ``ns`` tick is exactly 1/1000 of a
# microsecond, regardless of which instant it falls on. ``Y`` and ``M`` are
# NOT fixed-length (a Gregorian year is 365 or 366 days; a calendar month is
# 28-31 days -- probed directly against ``np.datetime64``'s own calendar
# arithmetic), so no rational ratio is exact for them. Their entries here
# are instead a *proven* upper bound (no calendar year exceeds 366 days, no
# calendar month exceeds 31 days), used only to gate the overflow refusal in
# :func:`_epoch_us_for_datetime64` below -- never to compute an actual epoch
# value, which for ``Y``/``M`` can only come from numpy's own calendar-aware
# cast (see that function's docstring for why that is still safe to call).
_DATETIME64_US_PER_TICK: dict[str, tuple[int, int]] = {
    "Y": (366 * 86400 * 10**6, 1),
    "M": (31 * 86400 * 10**6, 1),
    "W": (7 * 86400 * 10**6, 1),
    "D": (86400 * 10**6, 1),
    "h": (3600 * 10**6, 1),
    "m": (60 * 10**6, 1),
    "s": (10**6, 1),
    "ms": (1000, 1),
    "us": (1, 1),
    "ns": (1, 1000),
    "ps": (1, 10**6),
    "fs": (1, 10**9),
    "as": (1, 10**12),
}

# Units whose ratio above is mathematically exact (every unit but the two
# calendar ones), so their epoch-microsecond value can be computed directly
# from the ratio with integer floor division -- no numpy round-trip.
_DATETIME64_EXACT_UNITS = frozenset(_DATETIME64_US_PER_TICK) - {"Y", "M"}

# The largest microsecond magnitude a signed 64-bit epoch offset can hold.
# The negative side has one extra unit of headroom (``-2**63`` is itself the
# ``NaT`` sentinel, already excluded by the ``np.isnat`` check upstream), so
# this symmetric, slightly-conservative bound never accepts past the true
# boundary. Pinned at its exact threshold, both signs, by
# ``tests/test_temporal_coord_seam.py``.
_DATETIME64_MAX_ABS_US = 2**63 - 1


def _is_datetime64(value: Any) -> bool:
    """Return True when *value* is a ``numpy.datetime64`` scalar.

    numpy is imported inside the call rather than at module scope, matching
    this package's other numpy touch points (``ferrum._coerce``,
    ``ferrum.transforms``) — importing ``ferrum`` must not pull numpy in.
    """
    import numpy as np

    return isinstance(value, np.datetime64)


def _epoch_us_for_datetime64(value: np.datetime64) -> int:
    """Return a non-``NaT`` ``np.datetime64``'s epoch offset in exact
    microseconds, refusing it first if that offset cannot fit.

    Every unit ``np.datetime64`` supports -- ``Y M W D h m s ms us ns ps fs
    as`` -- passes through the identical arbitrary-precision proof before
    any fixed-width cast can run, regardless of the dtype's ``count``:
    ``raw`` is the value's raw, unscaled tick count (a reinterpret via
    ``int64``, never scaled, so it cannot itself wrap), ``unit``/``count``
    are read from the dtype's metadata (``np.datetime_data``, a pure
    metadata read), and the two are combined with the unit's
    microseconds-per-tick ratio using Python's arbitrary-precision integers,
    which cannot overflow at any magnitude. A zero-count dtype
    (``datetime64[0us]`` etc., which denotes no duration) is refused before
    any arithmetic, so ``count >= 1`` holds throughout the proof. There is
    no branch that skips this proof for any unit or any ``count`` -- a fine
    unit (``us``/``ns``/
    ``ps``/``fs``/``as``) whose ``count`` multiplies the tick back up past
    microsecond resolution (e.g. ``datetime64[1000us]``) goes through
    exactly the same arithmetic as a coarse one (#230-6 cycle-5; a
    cycle-4 comment asserted fine units "cannot grow past int64 range and
    need no check", which is false whenever ``count`` > 1).

    For every unit except ``Y``/``M``, the ratio is exact, so the proof's
    own arithmetic *is* the epoch-microsecond value -- no numpy cast runs
    at all for those units, so there is nothing for a second, independently
    fixed-width cast to disagree with. ``Y``/``M`` are not fixed-length
    durations (see :data:`_DATETIME64_US_PER_TICK`'s docstring), so no
    ratio can give their true calendar value; once this proof accepts a
    ``Y``/``M`` bound using its conservative upper-bound ratio, the bound's
    true magnitude is guaranteed no larger than the already-accepted
    estimate, so numpy's own calendar-aware cast to ``datetime64[us]`` --
    the only correct source for a real calendar date -- is now provably
    safe to call and cannot itself wrap (#230-6 cycle-3's failure mode,
    where a second numpy cast could wrap right along with the value being
    checked, is structurally excluded: the safety proof never depends on a
    numpy cast).

    Raises ``ValueError`` naming the value; callers that own a label
    (``BreakAxis.gap[0]``, ...) prefix it themselves.
    """
    import numpy as np

    raw = int(value.astype("int64"))  # native-unit tick count, unscaled
    unit, count = np.datetime_data(value.dtype)
    if count < 1:
        # `datetime64[0us]` (and every other zero-count unit) is a legal,
        # constructible numpy dtype whose ticks denote NO duration: the
        # multiplicative proof below would degenerate to `0 > bound` and
        # silently accept a fabricated 1970 epoch instant, and numpy's own
        # casts raise SystemError on it. Refuse before any arithmetic —
        # negative counts are rejected by numpy at dtype-parse time, so
        # `count >= 1` holds everywhere past this line (#230-6 cycle-6).
        raise ValueError(
            f"{value!r} has a zero-count datetime64 unit "
            f"('{count}{unit}'), which denotes no duration and cannot be "
            "interpreted as an instant"
        )
    ratio = _DATETIME64_US_PER_TICK.get(unit)
    if ratio is None:
        # Every unit np.datetime64 can carry is tabled above; a non-NaT
        # value with any other unit metadata is not constructible, so this
        # is unreachable in practice. Refuse rather than silently accept
        # under REFUSE MALFORMED, in case a future numpy adds a new unit.
        raise ValueError(f"{value!r} has an unrecognized datetime64 unit {unit!r}")
    numerator, denominator = ratio
    if abs(raw) * count * numerator > _DATETIME64_MAX_ABS_US * denominator:
        raise ValueError(
            f"{value!r} is out of range to represent as a microsecond-precision epoch offset"
        )
    if unit in _DATETIME64_EXACT_UNITS:
        return (raw * count * numerator) // denominator
    return int(value.astype("datetime64[us]").astype("int64"))


def temporal_coord_to_epoch_ms(value: _dt.date | _dt.datetime | str | np.datetime64) -> float:
    """Convert a temporal value to epoch-milliseconds (UTC).

    This is ferrum's canonical Python-side temporal conversion: the annotation
    layer, ``BreakAxis.gap`` (``ferrum.structural``) and the encoding layer's
    TimeScale-domain coercion (``ferrum.encoding._scale``) all reach the
    epoch-ms representation the Rust renderer uses for temporal axes (same
    units as ``_coerce.py`` and the D3 time scale) through this one function,
    so all three accept the same types and refuse the same inputs.

    Parameters
    ----------
    value : datetime.date, datetime.datetime, numpy.datetime64, or str
        - ``datetime.date`` — midnight UTC on that calendar date.
        - ``datetime.datetime`` — naive datetimes are treated as UTC (consistent
          with ``_coerce.py``'s handling of polars ``Date``/``Datetime`` columns).
          Aware datetimes are converted to UTC before computing the epoch offset.
          ``pandas.Timestamp`` is a subclass and rides this branch.
        - ``numpy.datetime64`` — converted at microsecond resolution, the same
          resolution ``datetime``/``pd.Timestamp`` top out at, so the same
          instant spelled either way yields the identical float. A value finer
          than microseconds (``[ns]`` and below) loses that extra precision,
          exactly as it would converting through ``pd.Timestamp``.
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
        If a string cannot be parsed as an ISO-8601 date or datetime (raised
        as :class:`_TemporalParseError`, a ``ValueError`` subclass consumers
        catch to re-word in their own vocabulary); if the value is
        not-a-time (``pandas.NaT``, ``numpy.datetime64('NaT')``) — REFUSE
        MALFORMED: a NaN epoch offset is not a coordinate, and letting one
        through produced a leaked JSON-parse diagnostic two layers
        downstream; or if a ``numpy.datetime64`` is out of range to represent
        as a microsecond-precision epoch offset.
    TypeError
        If the value is not a temporal type at all (a bare number, a list, an
        arbitrary object). A number is not a coordinate that needs converting
        — callers that accept epoch-ms floats handle them before calling here.

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
                raise _TemporalParseError(
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
        epoch_ms = seconds * 1000.0
        if math.isnan(epoch_ms):
            # `pandas.NaT` is a `datetime.datetime` subclass whose epoch
            # subtraction yields NaN instead of raising, so it reaches here
            # looking like an ordinary datetime.
            raise ValueError(f"{value!r} {_NOT_A_TIME}")
        return epoch_ms

    if isinstance(value, _dt.date):
        # Plain date: midnight UTC.
        days = (value - _EPOCH_DATE).days
        return float(days * _MS_PER_DAY)

    if _is_datetime64(value):
        import numpy as np

        if np.isnat(value):
            raise ValueError(f"{value!r} {_NOT_A_TIME}")
        return _epoch_us_for_datetime64(value) / 1000.0

    raise TypeError(
        "temporal value must be a datetime.date, datetime.datetime, "
        "numpy.datetime64, or an ISO-8601 date/datetime string; got "
        f"{type(value).__name__}"
    )


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
    - ``datetime.date`` / ``datetime.datetime`` / ``numpy.datetime64`` →
      epoch-milliseconds float.
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
    if _is_datetime64(v):
        return temporal_coord_to_epoch_ms(v)
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

    Temporal values (``date``, ``datetime``, ``numpy.datetime64``, ISO
    strings) are converted to epoch-milliseconds (UTC) so they align with the
    Rust renderer's temporal scale — the same units that ``_coerce.py``
    produces for temporal data columns.  Numeric values and coordinate
    wrappers pass through unchanged.
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
    if _is_datetime64(v):
        # Primitives constructed directly (not through _coerce_coord) can
        # still carry a raw datetime64; it used to reach json.dumps as-is and
        # die with "Object of type datetime64 is not JSON serializable".
        return temporal_coord_to_epoch_ms(v)
    return _sanitize_coord(v)  # plain numeric — data-space
