"""Structural feature dataclasses for advanced chart layout options."""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any

from ferrum._validate import (
    validate_choice,
    validate_finite_ordered_pair,
    validate_fraction_value,
    validate_pixel_value,
)

_VALID_BREAK_STYLES = frozenset({"slash", "zigzag", "wave", "gap"})
_VALID_CONNECT_STYLES = frozenset({"bracket", "lines", "none"})
_INSET_BOUNDS_NAMES = ("left", "top", "right", "bottom")


@dataclass(frozen=True)
class SecondaryY:
    """A secondary y-axis encoding overlaid on a chart.

    ``chart + SecondaryY(...)`` desugars to an appended layer on *chart*: mark
    ``mark``, ``y`` encoding on ``field`` (carrying ``axis``/``scale``), ``x``
    inherited from the base chart, color literal ``color``, opacity
    ``opacity`` — flagged as an independent-y layer (GH #52). The base
    chart's own layer(s) are unchanged, so ``layered_chart + SecondaryY(...)``
    keeps the base layers sharing the left axis while only the appended
    layer gets its own right axis; adding multiple ``SecondaryY`` instances
    stacks multiple right axes outward. The base chart must carry an ``x``
    encoding for the secondary layer to inherit — adding ``SecondaryY`` to
    a chart with no ``x`` raises ``ValueError``. ``field`` is read from the
    base chart's own table (the desugar performs no data merge), so a
    ``field`` that is not a column of the base data also raises
    ``ValueError`` at ``+`` time. ``mark`` must name a primitive mark
    (``point``, ``line``, ``bar``, ``area``, ``rule``, ``text``, ``tick``,
    ``rect``) -- a composite mark name (e.g. ``"boxplot"``) would otherwise
    bypass the ``mark_*()`` desugar pipeline and reach the renderer as an
    unknown primitive, so it raises ``ValueError`` at ``+`` time; use
    ``LayerChart(chart, other_chart, resolve={"y": "independent"})`` for a
    composite overlay on a secondary axis instead.

    This re-bases the feature onto ferrum's per-layer independent-y
    subsystem: unlike the original overlay-only renderer, the secondary
    series now reserves its own right-side margin band (so the plot area
    narrows to make room for it, rather than the axis overdrawing the plot),
    gets a real axis (ticks, labels, per-encoding ``Axis(...)`` config), and
    is fully interactive (tooltips, zoom/pan, hit-testing) like any other
    layer.

    Parameters
    ----------
    field : str
        Data field to encode on the secondary y axis.
    mark : str, default "line"
        Mark type for the secondary series.
    axis : Axis, optional
        Per-axis configuration for the secondary y axis.
    color : str, optional
        Color for the secondary mark.
    opacity : float, optional
        Opacity for the secondary mark.
    scale : Scale, optional
        Scale configuration for the secondary y axis.
    """

    field: str
    mark: str = "line"
    axis: Any = None  # Axis instance or None
    color: str | None = None
    opacity: float | None = None
    scale: Any = None  # Scale instance or None


@dataclass(frozen=True)
class BreakAxis:
    """An axis break that omits a region of the scale to skip outlier values.

    Parameters
    ----------
    axis : str
        Which axis to break: ``"x"`` or ``"y"``.
    gap : tuple or list
        A single ``(start, end)`` break region or a list of ``(start, end)``
        tuples for multiple breaks. Each bound is either numeric
        (``int``/``float``, data-space) or temporal (``str`` ISO-8601 date/
        datetime, ``datetime.date``, ``datetime.datetime``,
        ``numpy.datetime64``, or ``pandas.Timestamp``) — temporal bounds are
        coerced to epoch-milliseconds UTC at construction time via the same
        conversion the encoding layer applies to temporal scale domains and
        annotation coordinates (#230-6). A naive (timezone-unaware) string or
        ``datetime`` bound is *treated as* UTC; a timezone-aware ``datetime``
        or ``pandas.Timestamp`` bound is *converted* to UTC before its epoch
        offset is computed — the same naive/aware split
        ``ferrum.annotation.coords.temporal_coord_to_epoch_ms`` documents,
        since that is the function this coercion reuses. Both bounds of one
        gap must be the same kind (numeric or temporal); a pair mixing the two raises
        ``ValueError``. Once coerced, each ``(start, end)`` pair must be
        finite (no NaN/±inf) and correctly ordered (``start < end``); a
        malformed, non-finite, reversed, or unparseable-string gap raises
        ``ValueError`` at construction time (#92, #230-6) instead of sailing
        through to a blank or nonsense render. A temporal gap on a
        non-temporal axis is out-of-domain and warns ``BreakAxisIneffective``
        at render time rather than refusing at construction.
    break_size : float, default 12
        Visual size of the break indicator in pixels.
    break_style : str, default "slash"
        Break indicator style: ``"slash"``, ``"zigzag"``, ``"wave"``, or ``"gap"``.
    """

    axis: str
    gap: tuple | list
    break_size: float = 12
    break_style: str = "slash"

    def __post_init__(self) -> None:
        validate_choice("BreakAxis.axis", "axis", self.axis, ("x", "y"))
        validate_choice(
            "BreakAxis.break_style", "break_style", self.break_style, _VALID_BREAK_STYLES
        )
        coerced_gap = _coerce_and_validate_break_gaps(self.gap)
        # BreakAxis is frozen; temporal bounds must be replaced with their
        # coerced epoch-ms floats so _render.py's serialization path (which
        # reads self.gap directly) never has to re-run this coercion or ship
        # a raw date string to the Rust wire, which only accepts [f64; 2].
        object.__setattr__(self, "gap", coerced_gap)


def _normalize_break_gaps(gap: Any) -> list[Any]:
    """Classify *gap* as a single ``(start, end)`` pair or a list of pairs.

    A bare ``(start, end)`` tuple is one break region; anything else is
    treated as a list of break regions. This is the single owner of that
    classification heuristic — ``ferrum._render.Chart._serialize_structural``
    calls this instead of hand-rolling its own copy, so the construction-time
    coercion/validation in :func:`_coerce_and_validate_break_gaps` and the
    wire-serialization path that actually reaches Rust always agree on what
    shape *gap* has.

    Elements are returned as-is (not coerced to ``list``); a caller that needs
    JSON-safe output does that conversion itself.
    """
    if isinstance(gap, tuple) and len(gap) == 2 and not isinstance(gap[0], (list, tuple)):
        return [gap]
    return list(gap)


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
# boundary.
_DATETIME64_MAX_ABS_US = 2**63 - 1


def _epoch_us_for_datetime64(label: str, value: Any) -> int:
    """Return a non-``NaT`` ``np.datetime64`` bound's epoch offset in exact
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
            f"{label}: {value!r} has a zero-count datetime64 unit "
            f"('{count}{unit}'), which denotes no duration and cannot be "
            "interpreted as an instant"
        )
    ratio = _DATETIME64_US_PER_TICK.get(unit)
    if ratio is None:
        # Every unit np.datetime64 can carry is tabled above; a non-NaT
        # value with any other unit metadata is not constructible, so this
        # is unreachable in practice. Refuse rather than silently accept
        # under REFUSE MALFORMED, in case a future numpy adds a new unit.
        raise ValueError(f"{label}: {value!r} has an unrecognized datetime64 unit {unit!r}")
    numerator, denominator = ratio
    if abs(raw) * count * numerator > _DATETIME64_MAX_ABS_US * denominator:
        raise ValueError(
            f"{label}: {value!r} is out of range to represent as a "
            "microsecond-precision epoch offset"
        )
    if unit in _DATETIME64_EXACT_UNITS:
        return (raw * count * numerator) // denominator
    return int(value.astype("datetime64[us]").astype("int64"))


def _coerce_temporal_gap_bound(label: str, value: Any) -> float:
    """Coerce one temporal ``BreakAxis.gap`` bound to epoch-milliseconds (UTC).

    Reuses :func:`ferrum.annotation.coords.temporal_coord_to_epoch_ms` — the
    same UTC-by-contract conversion the encoding layer's TimeScale-domain
    coercion (``ferrum.encoding._scale._convert_temporal_domain_elements``)
    and annotation coordinates already use (SPEC-04) — for
    ``str``/``datetime.date``/``datetime.datetime`` bounds; ``pd.Timestamp``
    is a ``datetime.datetime`` subclass and is covered by that same branch.
    That reused function does not itself refuse ``pd.NaT`` (also a
    ``datetime.datetime`` subclass): subtracting it from the epoch produces a
    NaN second count rather than raising, so the NaN result is caught here
    and refused with the same message as a ``np.datetime64`` ``NaT`` bound,
    rather than sailing through to a generic "not finite" diagnosis two
    layers downstream. ``np.datetime64`` has no analog in the reused
    function (it is not a ``datetime.datetime`` subclass), so it is
    converted directly via numpy's own microsecond cast — microsecond, not
    millisecond, because ``datetime.datetime``/``pd.Timestamp`` (the
    ``temporal_coord_to_epoch_ms`` branch below) top out at microsecond
    resolution too, via ``total_seconds() * 1000.0``. Casting at the same
    resolution as that sibling branch means a ``datetime64`` bound and a
    ``pd.Timestamp``/``datetime`` bound for the *same instant* coerce to the
    identical fractional-millisecond float, rather than one truncating a
    sub-millisecond digit the other preserves. A ``datetime64`` bound finer
    than microseconds (``[ns]`` and below) loses that extra precision on the
    cast, same as it would converting through ``pd.Timestamp`` or Python's
    own ``datetime``; only a genuine overflow of the microsecond-resolution
    epoch offset (e.g. a value with unit ``"Y"`` many billions of years out,
    or a fine unit whose dtype ``count`` multiplies it back up past
    microsecond resolution) is refused as out of range. That overflow is
    detected *before* any fixed-width cast runs, via
    :func:`_epoch_us_for_datetime64`'s arbitrary-precision Python-integer
    proof over the value's raw, unscaled tick count and its unit/count
    metadata — uniformly for every ``np.datetime64`` unit, never by
    comparing against a second numpy cast. A numpy reference cast is
    fixed-width and can itself silently wrap for an extreme enough input, so
    two independently-wrapped casts can land in coincidental agreement and
    fool a wrap-detector built from them; Python ints cannot wrap, so that
    failure mode is unrepresentable here. The proof is exact for every unit
    except the two calendar units (``Y``, ``M``, which are not fixed-length
    durations): for those it is a proven conservative upper bound that can
    refuse a band of large-but-representable values a few hundred years
    before the true boundary, but never accept one past it (see
    :func:`_epoch_us_for_datetime64`'s docstring).

    Raises
    ------
    ValueError
        If a ``np.datetime64`` or ``pd.Timestamp`` bound is ``NaT``, a
        ``np.datetime64`` bound overflows the ``int64`` range a
        microsecond-resolution epoch offset can represent, or a string bound
        cannot be parsed as an ISO-8601 date or datetime — naming the
        offending value in every case (#230-6).
    """
    import numpy as np

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError(f"{label}: got NaT (not-a-time), which is not a valid bound")
        us_int = _epoch_us_for_datetime64(label, value)
        return us_int / 1000.0

    from ferrum.annotation.coords import temporal_coord_to_epoch_ms

    try:
        epoch_ms = temporal_coord_to_epoch_ms(value)
    except ValueError as exc:
        raise ValueError(
            f"{label}: cannot parse {value!r} as an ISO-8601 date or datetime "
            "(use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')"
        ) from exc
    if math.isnan(epoch_ms):
        raise ValueError(f"{label}: got NaT (not-a-time), which is not a valid bound")
    return epoch_ms


def _classify_and_coerce_gap_bound(label: str, value: Any) -> tuple[str, Any]:
    """Classify one ``BreakAxis.gap`` bound and coerce it if temporal.

    Returns ``(kind, coerced_value)``:

    - ``"numeric"``: ``int``/``float`` (``bool`` excluded), unchanged.
    - ``"temporal"``: ``str``/``datetime.date``/``datetime.datetime``/
      ``numpy.datetime64`` (``pandas.Timestamp`` included via the
      ``datetime.datetime`` subclass check), coerced to epoch-ms UTC via
      :func:`_coerce_temporal_gap_bound`.
    - ``"other"``: anything else (``bool``, ``None``, a list, a dict, ...),
      passed through unchanged so
      :func:`ferrum._validate.validate_finite_ordered_pair` raises its own
      numeric-type message downstream — matching pre-#230-6 behavior for
      non-string malformed bounds.

    A ``str``/``date``/``datetime``/``datetime64`` bound is coerced
    immediately rather than deferred, so an unparseable string names itself
    in the raised ``ValueError`` instead of being reported as "not numeric":
    strings are meaningful ``BreakAxis.gap`` input now (temporal), unlike the
    ordinal-category vocabulary annotation coordinates use for non-ISO
    strings, so a bad one must be diagnosed as a bad date, not a bad number.
    """
    if isinstance(value, bool):
        return "other", value
    if isinstance(value, (int, float)):
        return "numeric", value
    if isinstance(value, (_dt.date, _dt.datetime, str)):
        return "temporal", _coerce_temporal_gap_bound(label, value)
    import numpy as np

    if isinstance(value, np.datetime64):
        return "temporal", _coerce_temporal_gap_bound(label, value)
    return "other", value


def _coerce_and_validate_break_gaps(gap: Any) -> tuple | list:
    """Coerce and validate ``BreakAxis.gap``, returning its stored form.

    *gap* is normalized to a list of ``(start, end)`` pairs via
    :func:`_normalize_break_gaps` (matching the shape ``_render.py`` will
    later emit to Rust; classification runs on the *raw* pair shapes, before
    any bound coercion, so it agrees with the pre-#230-6 behavior exactly).
    Each pair must have exactly two elements. Each bound is then classified
    and coerced via :func:`_classify_and_coerce_gap_bound`: a pair mixing a
    numeric bound with a temporal bound raises ``ValueError`` (#230-6) before
    the coerced pair is checked for finiteness and ordering
    (:func:`ferrum._validate.validate_finite_ordered_pair`). An empty ``gap``
    (no break regions at all) is refused as malformed too, as is a ``gap``
    that is not iterable (e.g. a bare number).

    Returns the coerced gap in the same single-pair-vs-list-of-pairs shape as
    the input (bare ``(start, end)`` stays bare; a list of pairs stays a
    list), for ``BreakAxis.__post_init__`` to store back onto the frozen
    dataclass so temporal bounds never reach wire serialization unconverted.
    """
    if not isinstance(gap, (tuple, list)):
        raise ValueError(
            f"BreakAxis.gap must be a (start, end) tuple or a list of such "
            f"tuples, not {type(gap).__name__}; got {gap!r}"
        )
    pairs = _normalize_break_gaps(gap)
    if not pairs:
        raise ValueError(f"BreakAxis.gap must contain at least one (start, end) pair; got {gap!r}")
    coerced_pairs = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"BreakAxis.gap[{i}] must be a (start, end) pair of length 2; got {pair!r}"
            )
        raw_start, raw_end = pair
        label = f"BreakAxis.gap[{i}]"
        start_kind, start = _classify_and_coerce_gap_bound(label, raw_start)
        end_kind, end = _classify_and_coerce_gap_bound(label, raw_end)
        if {start_kind, end_kind} == {"numeric", "temporal"}:
            numeric_val, temporal_val = (
                (raw_start, raw_end) if start_kind == "numeric" else (raw_end, raw_start)
            )
            raise ValueError(
                f"{label}: mixes a numeric bound ({numeric_val!r}) with a temporal "
                f"bound ({temporal_val!r}); both bounds of one gap must be the same kind"
            )
        validate_finite_ordered_pair(label, start, end)
        coerced_pairs.append((start, end))

    # Derive the return shape from `pairs` itself rather than re-running the
    # single-pair-vs-list classification: `_normalize_break_gaps` is that
    # heuristic's single owner (see its docstring), so `pairs[0] is gap`
    # holds exactly when the single-pair branch wrapped `gap` in a new list
    # — a bare (start, end) tuple stays bare, anything else stays a list.
    if pairs[0] is gap:
        return coerced_pairs[0]
    return coerced_pairs


def _inset_bound_kind_and_value(entry: Any) -> tuple[str, float]:
    """Classify one ``Inset.bounds`` entry as ``("norm", value)`` or ``("px", value)``.

    Mirrors the wire contract emitted by ``ferrum._render._inset_coord``:
    [NormCoord][ferrum.NormCoord] and bare numbers are normalized-fraction
    bounds (a bare number is *never* data-space, per the 2026-05-24
    inset-panels design); [PixelCoord][ferrum.PixelCoord] is a pixel-offset
    bound. See the batch-H design spec §4 H6 / §6.
    """
    from ferrum.annotation.coords import NormCoord, PixelCoord

    if isinstance(entry, NormCoord):
        return "norm", entry.value
    if isinstance(entry, PixelCoord):
        return "px", entry.value
    return "norm", float(entry)


def _validate_inset_bounds(bounds: tuple) -> None:
    """Raise ``ValueError`` for an out-of-range, negative-px, or empty inset box.

    Each of the four ``bounds`` entries (``left``, ``top``, ``right``,
    ``bottom``) is classified via :func:`_inset_bound_kind_and_value`, then
    checked against its own kind's contract: a normalized entry must be a
    finite fraction in ``[0, 1]`` (:func:`ferrum._validate.validate_fraction_value`);
    a pixel entry must be a finite, non-negative pixel value
    (:func:`ferrum._validate.validate_pixel_value`). The box is additionally
    refused as empty when a coordinate pair is of the *same* kind and has
    zero or negative extent (``left >= right`` or ``top >= bottom``) — a
    mixed-kind pair (one normalized, one pixel) is not directly comparable
    on a shared scale, so it is not checked for emptiness (spec §4 H6:
    "compared within like kinds").
    """
    if len(bounds) != 4:
        raise ValueError(
            "Inset.bounds must have exactly 4 entries (left, top, right, bottom); "
            f"got {len(bounds)}"
        )
    kinds: list[str] = []
    values: list[float] = []
    for i, entry in enumerate(bounds):
        kind, value = _inset_bound_kind_and_value(entry)
        label = f"Inset.bounds[{i}] ({_INSET_BOUNDS_NAMES[i]})"
        if kind == "norm":
            validate_fraction_value(label, value)
        else:
            validate_pixel_value(label, value)
        kinds.append(kind)
        values.append(value)

    left, top, right, bottom = values
    if kinds[0] == kinds[2] and left >= right:
        raise ValueError(
            f"Inset.bounds: empty box — left ({left!r}) must be less than "
            f"right ({right!r}); got bounds={bounds!r}"
        )
    if kinds[1] == kinds[3] and top >= bottom:
        raise ValueError(
            f"Inset.bounds: empty box — top ({top!r}) must be less than "
            f"bottom ({bottom!r}); got bounds={bounds!r}"
        )


@dataclass(frozen=True)
class Inset:
    """An inset chart embedded within a parent chart's plot area.

    Parameters
    ----------
    chart : Chart
        The chart to embed as an inset.
    bounds : tuple
        ``(left, top, right, bottom)`` boundary coordinates for the inset
        within the parent plot area.  Each coordinate may be a bare
        ``float`` (a normalized ``[0, 1]`` fraction of the plot area — the
        same as [NormCoord][ferrum.NormCoord]), [NormCoord][ferrum.NormCoord]
        itself, or [PixelCoord][ferrum.PixelCoord] (an absolute pixel offset
        from the plot area's origin, resolved in Rust against the computed
        plot rect at layout time). Mixed kinds within one ``bounds`` tuple
        are allowed; each entry resolves independently. A normalized entry
        outside ``[0, 1]``, a negative pixel entry, or an empty box (``left
        >= right`` or ``top >= bottom``, compared only within like kinds)
        raises ``ValueError`` at construction time.

        .. note::
           **Dated note (2026-09-11, batch H, F-L08-08):** earlier revisions
           of this docstring described a bare ``float`` as a *data-space*
           coordinate and stated that [PixelCoord][ferrum.PixelCoord] was
           unsupported for inset bounds. Neither was ever true of the
           design this feature shipped from (the 2026-05-24 inset-panels
           design fixed bounds as normalized-or-pixel, never data-space) —
           the docstring claim was same-commit authorship drift, and
           passing ``fm.px(...)`` previously raised ``TypeError`` as a
           stopgap rather than being resolved. Both are corrected here:
           bare floats and [NormCoord][ferrum.NormCoord] are normalized
           fractions, and [PixelCoord][ferrum.PixelCoord] now resolves
           correctly.
    border : bool, default True
        Draw a border around the inset.
    border_color : str, default "#999"
        Border color.
    border_dash : list[float], optional
        Border dash pattern.
    background : str or None, default "#fff"
        Inset background color; ``None`` for transparent.
    shadow : bool, default False
        Apply a drop shadow to the inset.
    connect_to : tuple, optional
        Data coordinates ``(x, y)`` of the source region in the parent chart
        that this inset zooms into.  Draws a connector from the parent region
        to the inset when provided.
    connect_style : str, default "lines"
        Connector style: ``"bracket"``, ``"lines"``, or ``"none"``.
    """

    chart: Any  # Chart instance
    bounds: tuple
    border: bool = True
    border_color: str = "#999"
    border_dash: list[float] | None = None
    background: str | None = "#fff"
    shadow: bool = False
    connect_to: tuple | None = None
    connect_style: str = "lines"

    def __post_init__(self) -> None:
        validate_choice(
            "Inset.connect_style", "connect_style", self.connect_style, _VALID_CONNECT_STYLES
        )
        _validate_inset_bounds(self.bounds)
