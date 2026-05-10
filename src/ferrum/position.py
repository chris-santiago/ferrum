"""Phase 9c — position adjustments (Identity, Dodge, Jitter, Stack).

These are immutable Python value classes. They serialize to a ``{"type":
"<kind>", ...}`` dict consumed by Rust ChartSpec.position / Layer.position.
Eligibility per-mark is enforced at chart-build time via
``validate_position_eligibility(mark, position)``.

The eligibility matrix mirrors the design spec §6.4:

    Identity: every mark.
    Dodge:    bar, point, box, boxplot, swarm, violin, errorbar, errorband, ribbon.
    Jitter:   point, swarm, tick.
    Stack:    bar, area, ribbon.
"""
from __future__ import annotations
from typing import Optional


# ---- Eligibility matrix ------------------------------------------------------

_DODGE_ELIGIBLE = frozenset([
    "bar", "point", "box", "boxplot", "swarm", "violin",
    "errorbar", "errorband", "ribbon",
])
_JITTER_ELIGIBLE = frozenset(["point", "swarm", "tick"])
_STACK_ELIGIBLE = frozenset(["bar", "area", "ribbon"])


# ---- Valid-value sets --------------------------------------------------------

_VALID_JITTER_AXES = {"x", "y", "both"}
_VALID_STACK_OFFSETS = {"zero", "normalize", "center"}


# ---- Value classes -----------------------------------------------------------

class Identity:
    """Explicit no-op position adjustment.

    Distinct from ``position=None`` (the default which means "no adjustment
    declared at all"): ``Identity`` is part of the spec and round-trips through
    JSON. Useful when constructing layered charts from sugar functions that
    want to be explicit about not stacking/dodging.
    """

    __slots__ = ()

    def to_spec_dict(self) -> dict:
        return {"type": "identity"}

    def __repr__(self) -> str:
        return "Identity()"

    def __eq__(self, other) -> bool:
        return isinstance(other, Identity)

    def __hash__(self) -> int:
        return hash("Identity")


class Dodge:
    """Side-by-side dodge across the ``by`` channel (defaults to color/fill).

    ``padding`` is the gap between dodged groups as a fraction of the band width.
    """

    __slots__ = ("by", "padding")

    def __init__(self, by: Optional[str] = None, *, padding: float = 0.05) -> None:
        if not (0.0 <= padding < 1.0):
            raise ValueError(f"Dodge: padding must be in [0, 1); got {padding}")
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "padding", padding)

    def to_spec_dict(self) -> dict:
        d: dict = {"type": "dodge", "padding": self.padding}
        if self.by is not None:
            d["by"] = self.by
        return d

    def __setattr__(self, name, value):
        raise AttributeError(f"Dodge is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"Dodge(by={self.by!r}, padding={self.padding})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Dodge)
            and self.by == other.by
            and self.padding == other.padding
        )

    def __hash__(self) -> int:
        return hash(("Dodge", self.by, self.padding))


class Jitter:
    """Random per-row noise on x and/or y; deterministic given a seed.

    With ``seed=None`` the Rust render pass derives a per-row seed via
    ``xxh3::hash64(f"{x}|{y}")`` so output is still byte-deterministic across
    runs for fixed inputs.
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
        d: dict = {"type": "jitter", "axis": self.axis, "width": self.width}
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    def __setattr__(self, name, value):
        raise AttributeError(f"Jitter is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"Jitter(axis={self.axis!r}, width={self.width}, seed={self.seed})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Jitter)
            and self.axis == other.axis
            and self.width == other.width
            and self.seed == other.seed
        )

    def __hash__(self) -> int:
        return hash(("Jitter", self.axis, self.width, self.seed))


class Stack:
    """Vertical accumulation grouped by ``by`` channel.

    ``offset="zero"`` (standard cumulative stack), ``"normalize"`` (100%
    stack — each row scaled to a per-x total of 1.0), or ``"center"``
    (streamgraph; symmetric around 0 per x-bin).
    """

    __slots__ = ("by", "offset")

    def __init__(self, by: Optional[str] = None, *, offset: str = "zero") -> None:
        if offset not in _VALID_STACK_OFFSETS:
            raise ValueError(
                f"Stack: offset must be 'zero'|'normalize'|'center'; got '{offset}'"
            )
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "offset", offset)

    def to_spec_dict(self) -> dict:
        d: dict = {"type": "stack", "offset": self.offset}
        if self.by is not None:
            d["by"] = self.by
        return d

    def __setattr__(self, name, value):
        raise AttributeError(f"Stack is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"Stack(by={self.by!r}, offset={self.offset!r})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Stack)
            and self.by == other.by
            and self.offset == other.offset
        )

    def __hash__(self) -> int:
        return hash(("Stack", self.by, self.offset))


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
