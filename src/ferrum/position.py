"""Position-adjustment value classes: Identity, Dodge, Jitter, Stack."""
from __future__ import annotations
from typing import Optional


# ---- Eligibility matrix ------------------------------------------------------

_DODGE_ELIGIBLE = frozenset([
    "bar", "point", "box", "boxplot", "boxen", "swarm", "violin",
    "errorbar", "errorband", "ribbon",
    # Composite marks that desugar to bar/area underneath:
    "histogram", "density",
])
_JITTER_ELIGIBLE = frozenset(["point", "swarm", "tick"])
_STACK_ELIGIBLE = frozenset([
    # Rect-style marks: y maps to segment TOP (renderer draws base→top).
    "bar", "area", "ribbon",
    # Annotation-style marks (Schwabish SB-followup 2026-05-12): y maps
    # to segment MIDPOINT so a same-data overlay lands at the visual
    # centre of each stacked-bar segment (e.g. per-segment count text
    # on class_prediction_error_chart).
    "text", "point", "rule", "tick",
    # Composite marks that desugar to bar/area underneath:
    "histogram", "density",
])


# ---- Valid-value sets --------------------------------------------------------

_VALID_JITTER_AXES = {"x", "y", "both"}
_VALID_STACK_OFFSETS = {"zero", "normalize", "center"}
_VALID_STACK_ANCHORS = {"top", "mid"}


# ---- Value classes -----------------------------------------------------------

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

    __slots__ = ()

    def to_spec_dict(self) -> dict:
        """Return the spec dict ``{"type": "identity"}``."""
        return {"type": "identity"}

    def __repr__(self) -> str:
        """Return ``Identity()``."""
        return "Identity()"

    def __eq__(self, other) -> bool:
        """Return True if *other* is also an ``Identity`` instance."""
        return isinstance(other, Identity)

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        return hash("Identity")


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

    __slots__ = ("by", "padding")

    def __init__(self, by: Optional[str] = None, *, padding: float = 0.05) -> None:
        if not (0.0 <= padding < 1.0):
            raise ValueError(f"Dodge: padding must be in [0, 1); got {padding}")
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "padding", padding)

    def to_spec_dict(self) -> dict:
        """Return the serialized spec dict for this position adjustment."""
        d: dict = {"type": "dodge", "padding": self.padding}
        if self.by is not None:
            d["by"] = self.by
        return d

    def __setattr__(self, name, value):
        """Raise AttributeError — Dodge is immutable."""
        raise AttributeError(f"Dodge is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        """Return a constructor-style string representation."""
        return f"Dodge(by={self.by!r}, padding={self.padding})"

    def __eq__(self, other) -> bool:
        """Return True if *other* is a ``Dodge`` with identical fields."""
        return (
            isinstance(other, Dodge)
            and self.by == other.by
            and self.padding == other.padding
        )

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        return hash(("Dodge", self.by, self.padding))


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

    __slots__ = ("axis", "width", "seed")

    def __init__(
        self,
        axis: str = "x",
        *,
        width: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        if axis not in _VALID_JITTER_AXES:
            raise ValueError(
                f"Jitter: axis must be 'x'|'y'|'both'; got '{axis}'"
            )
        if width <= 0.0:
            raise ValueError(f"Jitter: width must be > 0; got {width}")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "seed", seed)

    def to_spec_dict(self) -> dict:
        """Return the serialized spec dict for this position adjustment."""
        d: dict = {"type": "jitter", "axis": self.axis, "width": self.width}
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    def __setattr__(self, name, value):
        """Raise AttributeError — Jitter is immutable."""
        raise AttributeError(f"Jitter is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        """Return a constructor-style string representation."""
        return f"Jitter(axis={self.axis!r}, width={self.width}, seed={self.seed})"

    def __eq__(self, other) -> bool:
        """Return True if *other* is a ``Jitter`` with identical fields."""
        return (
            isinstance(other, Jitter)
            and self.axis == other.axis
            and self.width == other.width
            and self.seed == other.seed
        )

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        return hash(("Jitter", self.axis, self.width, self.seed))


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

    __slots__ = ("by", "offset", "anchor")

    def __init__(
        self,
        by: Optional[str] = None,
        *,
        offset: str = "zero",
        anchor: str = "top",
    ) -> None:
        if offset not in _VALID_STACK_OFFSETS:
            raise ValueError(
                f"Stack: offset must be 'zero'|'normalize'|'center'; got '{offset}'"
            )
        if anchor not in _VALID_STACK_ANCHORS:
            raise ValueError(
                f"Stack: anchor must be 'top'|'mid'; got '{anchor}'"
            )
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "anchor", anchor)

    def to_spec_dict(self) -> dict:
        """Return the serialized spec dict for this position adjustment."""
        d: dict = {"type": "stack", "offset": self.offset, "anchor": self.anchor}
        if self.by is not None:
            d["by"] = self.by
        return d

    def __setattr__(self, name, value):
        """Raise AttributeError — Stack is immutable."""
        raise AttributeError(f"Stack is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        """Return a constructor-style string representation."""
        return f"Stack(by={self.by!r}, offset={self.offset!r}, anchor={self.anchor!r})"

    def __eq__(self, other) -> bool:
        """Return True if *other* is a ``Stack`` with identical fields."""
        return (
            isinstance(other, Stack)
            and self.by == other.by
            and self.offset == other.offset
            and self.anchor == other.anchor
        )

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        return hash(("Stack", self.by, self.offset, self.anchor))


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
            f"mark_{mark_name} does not accept {kind}; "
            f"eligible marks: {sorted(eligible)}"
        )
