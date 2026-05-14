"""Coordinate-system declarations for ferrum charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CoordFlip:
    """Flip the x and y axes — e.g. for horizontal bar charts.

    Pass to ``Chart.coord(CoordFlip())``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="value", y="category").mark_bar().coord(
    ...     fm.CoordFlip()
    ... )
    """

    def _to_spec_dict(self) -> str:
        """Return the string token passed to ChartSpec coord param."""
        return "flip"

    def __repr__(self) -> str:
        """Return ``CoordFlip()``."""
        return "CoordFlip()"

    def __eq__(self, other: object) -> bool:
        """Return True if *other* is also a ``CoordFlip`` instance."""
        return isinstance(other, CoordFlip)

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        return hash("CoordFlip")


@dataclass(frozen=True)
class CoordCartesian:
    """Standard Cartesian coordinates with optional domain and clip overrides.

    Parameters
    ----------
    xlim : (float, float) | None
        Explicit x-axis domain ``(min, max)``.  ``None`` uses data extent.
    ylim : (float, float) | None
        Explicit y-axis domain ``(min, max)``.  ``None`` uses data extent.
    expand : bool
        Add padding around the data extent (default ``True``).
    clip : bool
        Clip marks to the plot area (default ``True``).

    Examples
    --------
    >>> fm.Chart(df).mark_point().encode(x="x", y="y").coord(
    ...     fm.CoordCartesian(xlim=(0, 100))
    ... )
    """

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    expand: bool = True
    clip: bool = True

    def _to_spec_dict(self) -> dict:
        """Return dict serialization for ChartSpec coord param."""
        d: dict = {"kind": "cartesian", "expand": self.expand, "clip": self.clip}
        if self.xlim is not None:
            d["x_domain"] = list(self.xlim)
        if self.ylim is not None:
            d["y_domain"] = list(self.ylim)
        return d


@dataclass(frozen=True)
class CoordFixed:
    """Cartesian coordinates with a fixed aspect ratio.

    Parameters
    ----------
    ratio : float
        Width-to-height ratio of one data unit.  ``1.0`` makes the plot
        square in data space.
    xlim : (float, float) | None
        Explicit x-axis domain.
    ylim : (float, float) | None
        Explicit y-axis domain.
    expand : bool
        Add padding around data extent (default ``True``).
    clip : bool
        Clip marks to plot area (default ``True``).

    Examples
    --------
    >>> fm.Chart(df).mark_point().encode(x="x", y="y").coord(
    ...     fm.CoordFixed(ratio=1.0)
    ... )
    """

    ratio: float = 1.0
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    expand: bool = True
    clip: bool = True

    def _to_spec_dict(self) -> dict:
        """Return dict serialization for ChartSpec coord param."""
        d: dict = {
            "kind": "fixed",
            "ratio": self.ratio,
            "expand": self.expand,
            "clip": self.clip,
        }
        if self.xlim is not None:
            d["x_domain"] = list(self.xlim)
        if self.ylim is not None:
            d["y_domain"] = list(self.ylim)
        return d


@dataclass(frozen=True)
class CoordPolar:
    """Polar coordinates for pie and radial charts.

    The ``theta`` parameter names which encoding channel (``"x"`` or ``"y"``)
    is interpreted as the angular variable.  The other channel is used as the
    radial variable (for scatter-in-polar mode); omit it for pie/donut charts.

    Parameters
    ----------
    theta : {"x", "y"}
        Which encoding channel maps to the angle.
    start : float
        Starting angle in radians (default ``0`` = 12 o'clock).
    direction : {1, -1}
        ``1`` for clockwise, ``-1`` for counter-clockwise.

    Examples
    --------
    >>> fm.Chart(df).mark_arc().encode(
    ...     x="category", color="category", size="value"
    ... ).coord(fm.CoordPolar(theta="x"))
    """

    theta: Literal["x", "y"] = "x"
    start: float = 0.0
    direction: Literal[1, -1] = 1

    def _to_spec_dict(self) -> dict:
        """Return dict serialization for ChartSpec coord param."""
        direction_str = "clockwise" if self.direction == 1 else "counter_clockwise"
        return {
            "kind": "polar",
            "theta": self.theta,
            "start_angle": self.start,
            "direction": direction_str,
            "inner_radius": 0.0,
        }


@dataclass(frozen=True)
class CoordGeo:
    """Geographic map-projection coordinates.

    Parameters
    ----------
    projection : str
        One of ``"mercator"``, ``"albers_usa"``, ``"equal_earth"``,
        ``"natural_earth"``, ``"orthographic"``, ``"equirectangular"``.

    Examples
    --------
    >>> fm.Chart(geojson_df).mark_geoshape().coord(
    ...     fm.CoordGeo(projection="equal_earth")
    ... )
    """

    projection: Literal[
        "mercator",
        "albers_usa",
        "equal_earth",
        "natural_earth",
        "orthographic",
        "equirectangular",
    ] = "mercator"

    def _to_spec_dict(self) -> dict:
        """Return dict serialization for ChartSpec coord param."""
        return {"kind": "geo", "projection": self.projection}
