"""Coordinate-system declarations: CoordFlip (functional), others planned for Phase 9+."""
from __future__ import annotations


class CoordFlip:
    """Flip the x and y axes — e.g. for horizontal bar charts.

    Pass to ``Chart.coord(CoordFlip())``.  This is the only coordinate
    class currently implemented; all others raise ``NotImplementedError``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="value", y="category").mark_bar().coord(
    ...     fm.CoordFlip()
    ... )
    """

    def __repr__(self) -> str:
        """Return ``CoordFlip()``."""
        return "CoordFlip()"

    def __eq__(self, other: object) -> bool:
        """Return True if *other* is also a ``CoordFlip`` instance."""
        return isinstance(other, CoordFlip)

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        return hash("CoordFlip")


class CoordCartesian:
    """Standard Cartesian coordinates — planned for Phase 9+.

    Currently raises ``NotImplementedError`` on construction. The default
    coordinate system (no explicit ``Chart.coord()`` call) is already
    Cartesian; this class exists for explicit declaration.

    Raises
    ------
    NotImplementedError
        ``CoordCartesian`` is planned for Phase 9+; omit ``Chart.coord()``
        for Cartesian behavior today.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordCartesian is planned for Phase 9+; use the default coordinate system."
        )


class CoordPolar:
    """Polar coordinates for pie and radial charts — planned for Phase 9+.

    Raises
    ------
    NotImplementedError
        ``CoordPolar`` is planned for Phase 9+.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordPolar is planned for Phase 9+."
        )


class CoordGeo:
    """Geographic map-projection coordinates — planned for Phase 9+.

    Raises
    ------
    NotImplementedError
        ``CoordGeo`` is planned for Phase 9+.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordGeo is planned for Phase 9+."
        )


class CoordFixed:
    """Fixed aspect-ratio coordinates — planned for Phase 9+.

    Raises
    ------
    NotImplementedError
        ``CoordFixed`` is planned for Phase 9+.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "CoordFixed is planned for Phase 9+."
        )
