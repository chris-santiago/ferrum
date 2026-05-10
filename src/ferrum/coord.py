"""Coordinate system value classes for Chart.coord().

CoordFlip is fully implemented in Phase 8a.
All others raise NotImplementedError — planned for Phase 9+.
"""
from __future__ import annotations


class CoordFlip:
    """Flip x/y axes (horizontal bar chart, etc.)."""

    def __repr__(self) -> str:
        return "CoordFlip()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CoordFlip)

    def __hash__(self) -> int:
        return hash("CoordFlip")


class CoordCartesian:
    """Standard Cartesian coordinates — default. Planned Phase 9+."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordCartesian is planned for Phase 9+; use the default coordinate system."
        )


class CoordPolar:
    """Polar coordinates (pie / radial). Planned Phase 9+."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordPolar is planned for Phase 9+."
        )


class CoordGeo:
    """Geographic / map projection coordinates. Planned Phase 9+."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordGeo is planned for Phase 9+."
        )


class CoordFixed:
    """Fixed-ratio coordinates (aspect locked). Planned Phase 9+."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordFixed is planned for Phase 9+."
        )
