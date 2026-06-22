"""Position-adjustment value classes: Identity, Dodge, Jitter, Stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---- Mark stack capability (two distinct questions; see CHART-08) -------------
#
# These two frozensets both answer "can this mark stack?", but they answer
# *different* questions and are intentionally NOT the same set:
#
#   _STACKABLE_MARKS  — which marks' Rust renderers *consume* the
#       ``__stack_y_base__`` column emitted by ``apply_stack``.  Drives
#       ``chart.py::_strip_unstackable``: a ``stack=`` declared on any other
#       mark is silently dropped by the renderer, so we warn + strip it.  Only
#       bar/area draw base→top from ``__stack_y_base__``.
#
#   _STACK_ELIGIBLE   — which marks accept a ``position=fm.Stack()`` adjustment.
#       Drives ``validate_position_eligibility``.  Broader than the consumers
#       above because annotation-style marks (text/point/rule/tick) ride on a
#       Stack to *label* segments (Schwabish SB-followup, ``anchor="mid"``),
#       and the composite histogram/density desugar to bar/area underneath.
#
# A maintainer changing which marks stack must consider BOTH sets and decide
# which question the change answers.  They overlap on bar/area but are not one
# concept split across files.

# Marks whose Rust renderers consume the ``__stack_y_base__`` column produced
# by ``apply_stack`` (see ``crates/ferrum-core/src/render/marks/bar.rs`` and
# ``area.rs``).  Every other mark type silently drops stacked data, so when
# ``stack=`` is set on a non-stackable mark the chart layer warns and strips it
# before forwarding to Rust (``chart.py::_strip_unstackable``).
_STACKABLE_MARKS = frozenset(["bar", "area"])


# ---- Eligibility matrix ------------------------------------------------------

_DODGE_ELIGIBLE = frozenset(
    [
        "bar",
        "point",
        "box",
        "boxplot",
        "boxen",
        "swarm",
        "violin",
        "errorbar",
        "errorband",
        "ribbon",
        # Composite marks that desugar to bar/area underneath:
        "histogram",
        "density",
    ]
)
_JITTER_ELIGIBLE = frozenset(["point", "swarm", "tick"])
# Marks that accept a ``position=fm.Stack()`` adjustment (see CHART-08 note
# above for how this differs from ``_STACKABLE_MARKS``).
_STACK_ELIGIBLE = frozenset(
    [
        # Rect-style marks: y maps to segment TOP (renderer draws base→top).
        "bar",
        "area",
        "ribbon",
        # Annotation-style marks (Schwabish SB-followup 2026-05-12): y maps
        # to segment MIDPOINT so a same-data overlay lands at the visual
        # centre of each stacked-bar segment (e.g. per-segment count text
        # on class_prediction_error_chart).
        "text",
        "point",
        "rule",
        "tick",
        # Composite marks that desugar to bar/area underneath:
        "histogram",
        "density",
    ]
)


# ---- Valid-value sets --------------------------------------------------------

_VALID_JITTER_AXES = {"x", "y", "both"}
_VALID_STACK_ANCHORS = {"top", "mid"}

# Canonical set of real stack-offset strategies.  Single source of truth for
# every stack-offset validator (Stack, transform_stack, and the encoding
# ``stack=`` path, which layers bool/falsy-string normalization on top).
STACK_OFFSETS = frozenset({"zero", "normalize", "center"})


def _validate_stack_offset(value: str, *, where: str) -> None:
    """Raise ``ValueError`` if *value* is not a real stack offset.

    Single canonical validator for the three stack-offset entry points
    (``Stack``, ``transform_stack``, and the encoding ``stack=`` path). The
    error text is identical across all three callers apart from the *where*
    prefix that names the failing site.

    Parameters
    ----------
    value : str
        The candidate offset (already known to be one of the real-offset
        spellings or an unrecognized string — bool/falsy normalization happens
        before this call on the encoding path).
    where : str
        Caller label used as the message prefix (e.g. ``"Stack"``,
        ``"transform_stack"``, or a channel name like ``"Y"``).

    Raises
    ------
    ValueError
        When *value* is not one of :data:`STACK_OFFSETS`.
    """
    if value not in STACK_OFFSETS:
        raise ValueError(
            f"{where}: stack offset {value!r} is not valid; expected one of {sorted(STACK_OFFSETS)}"
        )


# ---- Value classes -----------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """Explicit no-op position adjustment.

    Distinct from ``position=None`` (which means "no adjustment declared"):
    ``Identity`` is part of the spec and round-trips through JSON. Use it
    when composing layered charts that need to opt out of an inherited stack
    or dodge on a per-layer basis.

    Eligible marks: all.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="grp", y="val").mark_bar(position=fm.Identity())
    """

    def to_spec_dict(self) -> dict:
        """Return the spec dict ``{"type": "identity"}``."""
        return {"type": "identity"}


@dataclass(frozen=True)
class Dodge:
    """Side-by-side placement of marks grouped by a channel.

    Eligible marks: ``bar``, ``point``, ``box``, ``boxplot``, ``boxen``,
    ``swarm``, ``violin``, ``errorbar``, ``errorband``, ``ribbon``,
    ``histogram``, ``density``.

    Parameters
    ----------
    by : str, optional
        Channel name to group by. Defaults to the color/fill channel when
        omitted.
    padding : float, default 0.05
        Gap between dodged groups as a fraction of band width. Must be in
        ``[0, 1)``.

    Raises
    ------
    ValueError
        If ``padding`` is outside ``[0, 1)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="cat", y="val", color="grp").mark_bar(
    ...     position=fm.Dodge()
    ... )
    """

    by: Optional[str] = None
    padding: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.padding < 1.0):
            raise ValueError(f"Dodge: padding must be in [0, 1); got {self.padding}")

    def to_spec_dict(self) -> dict:
        """Return the serialized spec dict for this position adjustment."""
        d: dict = {"type": "dodge", "padding": self.padding}
        if self.by is not None:
            d["by"] = self.by
        return d


@dataclass(frozen=True)
class Jitter:
    """Random per-row noise applied to x, y, or both axes.

    Uses a ChaCha8 RNG seeded from ``seed``, making SVG output
    byte-deterministic for a given dataset and seed.  When ``seed=None`` the
    Rust renderer derives a per-row seed from the row's data values via
    xxh3 — output remains deterministic across runs for fixed inputs.

    Eligible marks: ``point``, ``swarm``, ``tick``.

    Parameters
    ----------
    axis : {"x", "y", "both"}, default "x"
        Which axis or axes to jitter.
    width : float, default 0.4
        Maximum absolute displacement in scaled units. Must be ``> 0``.
    seed : int or None, default None
        RNG seed.  ``None`` means per-row data-derived seed (still
        deterministic).

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``"x"``, ``"y"``, ``"both"``.
    ValueError
        If ``width`` is ``<= 0``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="grp", y="value").mark_point(
    ...     position=fm.Jitter(width=0.3, seed=42)
    ... )
    """

    axis: str = "x"
    width: float = 0.4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.axis not in _VALID_JITTER_AXES:
            raise ValueError(f"Jitter: axis must be 'x'|'y'|'both'; got '{self.axis}'")
        if self.width <= 0.0:
            raise ValueError(f"Jitter: width must be > 0; got {self.width}")

    def to_spec_dict(self) -> dict:
        """Return the serialized spec dict for this position adjustment."""
        d: dict = {"type": "jitter", "axis": self.axis, "width": self.width}
        if self.seed is not None:
            d["seed"] = self.seed
        return d


@dataclass(frozen=True)
class Stack:
    """Vertical accumulation of marks grouped by a channel.

    Eligible marks: rect-style (``bar``, ``area``, ``ribbon``,
    ``histogram``, ``density``) and annotation-style (``text``,
    ``point``, ``rule``, ``tick``). The latter sit on top of a
    stacked layer to label segments.

    Parameters
    ----------
    by : str, optional
        Channel name whose distinct values define the stack groups.
        Defaults to the color/fill channel when omitted.
    offset : {"zero", "normalize", "center"}, default "zero"
        Stack baseline strategy:

        - ``"zero"`` — standard cumulative stack from y = 0.
        - ``"normalize"`` — 100 % stack; each x-bin scales to a total of 1.
        - ``"center"`` — streamgraph; symmetric around y = 0.
    anchor : {"top", "mid"}, default "top"
        Where each row's y output lands within its segment. ``"top"``
        (default) returns the segment top so rect-style marks (bar,
        area, ribbon) draw from ``__stack_y_base__`` to ``y``.
        ``"mid"`` returns the segment midpoint so an annotation mark
        (``mark_text``, ``mark_point``, …) sits at the visual centre
        of each stacked segment — used by composite marks like
        ``mark_class_prediction_error(show_counts=True)`` to land
        per-segment count labels without duplicating cumsum logic
        in Python.

    Raises
    ------
    ValueError
        If ``offset`` is not one of ``"zero"``, ``"normalize"``,
        ``"center"`` or ``anchor`` is not one of ``"top"``, ``"mid"``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="year", y="count", color="category").mark_bar(
    ...     position=fm.Stack(offset="normalize")
    ... )
    """

    by: Optional[str] = None
    offset: str = "zero"
    anchor: str = "top"

    def __post_init__(self) -> None:
        _validate_stack_offset(self.offset, where="Stack")
        if self.anchor not in _VALID_STACK_ANCHORS:
            raise ValueError(f"Stack: anchor must be 'top'|'mid'; got '{self.anchor}'")

    def to_spec_dict(self) -> dict:
        """Return the serialized spec dict for this position adjustment."""
        d: dict = {"type": "stack", "offset": self.offset, "anchor": self.anchor}
        if self.by is not None:
            d["by"] = self.by
        return d


# ---- Eligibility validator ---------------------------------------------------


def validate_position_eligibility(mark_name: str, position) -> None:
    """Raise ``TypeError`` if ``mark_name`` does not accept ``position``.

    Called by ``Chart.mark_<name>(position=...)`` at construction time.
    Identity is accepted by every mark; other adjustments are constrained
    per the eligibility matrix in this module.
    """
    if position is None:
        return
    if isinstance(position, Identity):
        return
    if isinstance(position, Dodge):
        eligible = _DODGE_ELIGIBLE
        kind = "Dodge"
    elif isinstance(position, Jitter):
        eligible = _JITTER_ELIGIBLE
        kind = "Jitter"
    elif isinstance(position, Stack):
        eligible = _STACK_ELIGIBLE
        kind = "Stack"
    else:
        raise TypeError(f"unknown position adjustment: {type(position).__name__}")
    if mark_name not in eligible:
        raise TypeError(
            f"mark_{mark_name} does not accept {kind}; eligible marks: {sorted(eligible)}"
        )
