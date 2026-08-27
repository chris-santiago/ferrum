"""Structural feature dataclasses for advanced chart layout options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ferrum._validate import validate_choice

_VALID_BREAK_STYLES = frozenset({"slash", "zigzag", "wave", "gap"})
_VALID_CONNECT_STYLES = frozenset({"bracket", "lines", "none"})


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
        tuples for multiple breaks.
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


@dataclass(frozen=True)
class Inset:
    """An inset chart embedded within a parent chart's plot area.

    Parameters
    ----------
    chart : Chart
        The chart to embed as an inset.
    bounds : tuple
        ``(left, top, right, bottom)`` boundary coordinates for the inset
        within the parent plot area.  Each coordinate may be a ``float``
        (data-space), [PixelCoord][ferrum.PixelCoord], or
        [NormCoord][ferrum.NormCoord].
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
