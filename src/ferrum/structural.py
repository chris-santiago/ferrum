"""Structural feature dataclasses for advanced chart layout options."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from ferrum._validate import (
    is_real_number,
    validate_choice,
    validate_finite_ordered_pair,
    coerce_fraction_value,
    coerce_pixel_value,
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
        A single ``(start, end)`` break region or a sequence of
        ``(start, end)`` pairs for multiple breaks. Each bound is either
        numeric (``int``/``float`` or a numpy scalar — ``np.int64``,
        ``np.float32``, ... — data-space) or temporal (``str`` ISO-8601 date/
        datetime, ``datetime.date``, ``datetime.datetime``,
        ``numpy.datetime64``, or ``pandas.Timestamp``) — temporal bounds are
        coerced to epoch-milliseconds UTC at construction time by
        ``ferrum.annotation.coords.temporal_coord_to_epoch_ms``, literally
        the same function the encoding layer applies to temporal scale
        domains and annotation coordinates, so all three accept the same
        types (#230-6). A naive (timezone-unaware) string or ``datetime``
        bound is *treated as* UTC; a timezone-aware ``datetime`` or
        ``pandas.Timestamp`` bound is *converted* to UTC before its epoch
        offset is computed — that function's documented naive/aware split.
        Both bounds of one gap must be the same kind (numeric or temporal); a
        pair mixing the two raises ``ValueError``. Once coerced, each
        ``(start, end)`` pair must be finite (no NaN/±inf) and correctly
        ordered (``start < end``); a malformed, non-finite, reversed,
        not-a-time (``pd.NaT``/``np.datetime64('NaT')``), out-of-range
        ``datetime64``, or unparseable-string gap raises ``ValueError`` at
        construction time (#92, #230-6) instead of sailing through to a blank
        or nonsense render. A temporal gap on a non-temporal axis is
        out-of-domain and warns ``BreakAxisIneffective`` at render time
        rather than refusing at construction.

        The stored ``gap`` keeps the container type it was given (a tuple of
        pairs stays a tuple, so a tuple-spelled ``BreakAxis`` stays hashable)
        while each pair is canonicalized to a 2-tuple of coerced floats.
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


def _coerce_temporal_gap_bound(label: str, value: Any) -> float:
    """Coerce one temporal ``BreakAxis.gap`` bound to epoch-milliseconds (UTC).

    Delegates the whole conversion to
    :func:`ferrum.annotation.coords.temporal_coord_to_epoch_ms`, the one
    Python-side temporal seam — the same function the encoding layer's
    TimeScale-domain coercion (``ferrum.encoding._scale._convert_temporal_domain_elements``)
    and annotation coordinates go through (SPEC-04). Every accepted spelling
    (``str`` ISO-8601, ``datetime.date``, ``datetime.datetime``,
    ``pandas.Timestamp`` via the ``datetime`` subclass, ``numpy.datetime64``)
    and every refusal (not-a-time, out-of-range ``datetime64``, unparseable
    string) is decided there, so a gap bound and a scale domain element of
    the same type cannot drift apart. This function only supplies the
    ``BreakAxis.gap[i]`` label, which the seam has no business knowing.

    The seam's refusals are surfaced in two shapes: an unparseable string
    (its own ``_TemporalParseError``) is re-worded in this module's
    vocabulary, because "a bad date in a break gap" is a break-gap mistake
    and must read as one; every other refusal is the seam's own sentence
    with the bound's label prefixed.

    Raises
    ------
    ValueError
        If the bound is ``pd.NaT``/``np.datetime64('NaT')``, a
        ``np.datetime64`` out of range to represent as a
        microsecond-precision epoch offset, or a string that cannot be
        parsed as an ISO-8601 date or datetime — naming the offending value
        in every case (#230-6).
    """
    from ferrum.annotation.coords import _TemporalParseError, temporal_coord_to_epoch_ms

    try:
        return temporal_coord_to_epoch_ms(value)
    except _TemporalParseError as exc:
        raise ValueError(
            f"{label}: cannot parse {value!r} as an ISO-8601 date or datetime "
            "(use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _classify_and_coerce_gap_bound(label: str, value: Any) -> tuple[str, Any]:
    """Classify one ``BreakAxis.gap`` bound and coerce it if temporal.

    Returns ``(kind, coerced_value)``:

    - ``"numeric"``: a real number (:func:`ferrum._validate.is_real_number`:
      builtins, numpy scalars such as the ``np.int64`` a
      ``DataFrame[col].max()`` hands back, ``Decimal``), coerced to ``float``.
      ``bool`` is excluded: a boolean gap bound is a mistake, not a 0/1
      break region.
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
    if is_real_number(value):
        return "numeric", float(value)
    if isinstance(value, (_dt.date, _dt.datetime, str)):
        return "temporal", _coerce_temporal_gap_bound(label, value)

    # `_is_datetime64` is the seam's own predicate, so "which numpy values are
    # temporal" is answered in the one place that converts them.
    from ferrum.annotation.coords import _is_datetime64

    if _is_datetime64(value):
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

    Returns the coerced gap in the same shape AND the same container type as
    the input, for ``BreakAxis.__post_init__`` to store back onto the frozen
    dataclass so temporal bounds never reach wire serialization unconverted:
    a bare ``(start, end)`` stays bare, a tuple of pairs stays a tuple, a list
    of pairs stays a list. The container type is load-bearing, not cosmetic —
    ``BreakAxis`` is a frozen dataclass, so rebuilding a tuple-spelled ``gap``
    as a list silently made the instance UNHASHABLE (design review F3;
    ``hash(BreakAxis(axis="y", gap=((1, 5), (6, 9))))`` raised ``TypeError``
    where it had always worked), and made hashability depend on which of two
    equivalent spellings the caller happened to choose. Each *pair* is
    canonicalized to a 2-tuple regardless of how it was spelled — that is the
    shape every reader downstream expects, and a list pair was never hashable
    in any spelling.
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
    # — a bare (start, end) tuple stays bare, anything else is a sequence of
    # pairs and keeps the caller's own container type (F3).
    if pairs[0] is gap:
        return coerced_pairs[0]
    if isinstance(gap, tuple):
        return tuple(coerced_pairs)
    return coerced_pairs


def _inset_bound_kind_and_value(bound: Any) -> tuple[str, Any]:
    """Classify one ``Inset.bounds`` entry as ``("norm", value)`` or ``("px", value)``.

    The one classifier for both the construction-time validator
    (:func:`_validate_inset_bounds`) and ``ferrum._render``'s wire adapter
    (`#239`): [NormCoord][ferrum.NormCoord] and bare numbers are
    normalized-fraction bounds (a bare number is *never* data-space, per the
    2026-05-24 inset-panels design); [PixelCoord][ferrum.PixelCoord] is a
    pixel-offset bound. See the batch-H design spec §4 H6 / §6.

    This does not validate — it classifies. A value that is a real number
    (:func:`ferrum._validate.is_real_number`, so never ``bool`` and never a
    numeric string) becomes a ``float``; anything else is returned as-is, for
    the validator to refuse by its own type name. There is no duck-typed
    ``float()`` attempt.
    """
    from ferrum.annotation.coords import NormCoord, PixelCoord

    if isinstance(bound, NormCoord):
        kind, value = "norm", bound.value
    elif isinstance(bound, PixelCoord):
        kind, value = "px", bound.value
    else:
        kind, value = "norm", bound
    return kind, float(value) if is_real_number(value) else value


def _validate_inset_bounds(bounds: tuple) -> None:
    """Raise ``ValueError`` for an out-of-range, negative-px, or empty inset box.

    Each of the four ``bounds`` entries (``left``, ``top``, ``right``,
    ``bottom``) is classified via :func:`_inset_bound_kind_and_value`, then
    checked against its own kind's contract: a normalized entry must be a
    finite fraction in ``[0, 1]`` (:func:`ferrum._validate.coerce_fraction_value`);
    a pixel entry must be a finite, non-negative pixel value
    (:func:`ferrum._validate.coerce_pixel_value`). Both are called with
    ``required=True`` — a bound entry has no optional-unset case the way an
    optional padding/opacity field does, so a ``None`` entry (bare, or
    ``PixelCoord(None)``/``NormCoord(None)``'s unwrapped ``.value``) is
    refused by the validator matching its own kind, not silently passed
    through as "unset" (`#239`). The box is additionally refused as empty
    when a coordinate pair is of the *same* kind and has zero or negative
    extent (``left >= right`` or ``top >= bottom``) — a mixed-kind pair (one
    normalized, one pixel) is not directly comparable on a shared scale, so
    it is not checked for emptiness (spec §4 H6: "compared within like
    kinds").
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
            coerce_fraction_value(label, value, required=True)
        else:
            coerce_pixel_value(label, value, required=True)
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
