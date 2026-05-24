"""Coordinate wrapper types for annotation positioning.

Coordinates in annotation primitives can be expressed in three ways:

- ``float`` — data-space coordinates (the default)
- ``PixelCoord`` — absolute pixel offset from the plot origin
- ``NormCoord`` — normalized [0, 1] fraction of the plot area

Use the :func:`px` and :func:`norm` factory functions as a readable shorthand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


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


CoordValue: TypeAlias = float | PixelCoord | NormCoord


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
