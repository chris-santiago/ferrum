"""Annotation primitive dataclasses and factory functions.

Each factory function returns a frozen dataclass instance that can be
collected into an :class:`~ferrum.annotation.container.Annotate` container
and attached to a chart.

All coordinate arguments accept ``float`` (data-space), :class:`~ferrum.annotation.coords.PixelCoord`
(absolute pixels), or :class:`~ferrum.annotation.coords.NormCoord` (normalized [0, 1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ferrum.annotation.coords import CoordValue


# ---------------------------------------------------------------------------
# Dataclass types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationText:
    """A text label at a fixed position."""

    x: CoordValue
    y: CoordValue
    text: str
    font_size: float
    color: str
    anchor: str
    baseline: str
    angle: float
    dx: float
    dy: float
    z: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        return {
            "type": "text",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "text": self.text,
            "font_size": self.font_size,
            "color": self.color,
            "anchor": self.anchor,
            "baseline": self.baseline,
            "angle": self.angle,
            "dx": self.dx,
            "dy": self.dy,
            "z": self.z,
        }


@dataclass(frozen=True)
class AnnotationArrow:
    """An arrow between two data points."""

    x: CoordValue
    y: CoordValue
    x2: CoordValue
    y2: CoordValue
    stroke: str
    stroke_width: float
    head_size: float
    curve: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        return {
            "type": "arrow",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "stroke": self.stroke,
            "stroke_width": self.stroke_width,
            "head_size": self.head_size,
            "curve": self.curve,
        }


@dataclass(frozen=True)
class AnnotationRect:
    """A filled rectangle region."""

    x1: CoordValue
    y1: CoordValue
    x2: CoordValue
    y2: CoordValue
    fill: str
    opacity: float
    stroke: str | None
    corner_radius: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "rect",
            "x1": _coord(self.x1),
            "y1": _coord(self.y1),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "fill": self.fill,
            "opacity": self.opacity,
            "corner_radius": self.corner_radius,
        }
        if self.stroke is not None:
            d["stroke"] = self.stroke
        return d


@dataclass(frozen=True)
class AnnotationLine:
    """A line segment between two points."""

    x1: CoordValue
    y1: CoordValue
    x2: CoordValue
    y2: CoordValue
    stroke: str
    stroke_width: float
    dash: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "line",
            "x1": _coord(self.x1),
            "y1": _coord(self.y1),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "stroke": self.stroke,
            "stroke_width": self.stroke_width,
        }
        if self.dash is not None:
            d["dash"] = self.dash
        return d


@dataclass(frozen=True)
class AnnotationSpan:
    """A shaded span along one axis (like a horizontal or vertical band)."""

    axis: str
    start: CoordValue
    end: CoordValue
    fill: str
    opacity: float
    label: str | None
    label_position: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "span",
            "axis": self.axis,
            "start": _coord(self.start),
            "end": _coord(self.end),
            "fill": self.fill,
            "opacity": self.opacity,
            "label_position": self.label_position,
        }
        if self.label is not None:
            d["label"] = self.label
        return d


@dataclass(frozen=True)
class AnnotationBracket:
    """A bracket annotation with a label."""

    x1: CoordValue
    y1: CoordValue
    x2: CoordValue
    y2: CoordValue
    label: str
    direction: str
    stroke: str
    tip_length: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        return {
            "type": "bracket",
            "x1": _coord(self.x1),
            "y1": _coord(self.y1),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "label": self.label,
            "direction": self.direction,
            "stroke": self.stroke,
            "tip_length": self.tip_length,
        }


@dataclass(frozen=True)
class AnnotationCallout:
    """A callout bubble with optional connecting arrow."""

    x: CoordValue
    y: CoordValue
    text: str
    text_x: CoordValue | None
    text_y: CoordValue | None
    arrow: str
    padding: float
    background: str
    border_color: str
    border_radius: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "callout",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "text": self.text,
            "arrow": self.arrow,
            "padding": self.padding,
            "background": self.background,
            "border_color": self.border_color,
            "border_radius": self.border_radius,
        }
        if self.text_x is not None:
            d["text_x"] = _coord(self.text_x)
        if self.text_y is not None:
            d["text_y"] = _coord(self.text_y)
        return d


@dataclass(frozen=True)
class AnnotationImage:
    """An image placed at a data or pixel coordinate."""

    x: CoordValue
    y: CoordValue
    src: str
    width: float
    height: float
    anchor: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        return {
            "type": "image",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "src": self.src,
            "width": self.width,
            "height": self.height,
            "anchor": self.anchor,
        }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _sanitize_coord(v: Any) -> Any:
    """Replace NaN/Inf with 0.0 so JSON serialization doesn't crash."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v


def _coord(v: CoordValue) -> Any:
    """Normalize a CoordValue to a renderer-serializable form."""
    from ferrum.annotation.coords import PixelCoord, NormCoord

    if isinstance(v, PixelCoord):
        return {"px": _sanitize_coord(v.value)}
    if isinstance(v, NormCoord):
        return {"norm": _sanitize_coord(v.value)}
    return _sanitize_coord(v)  # plain float — data-space


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def text(
    x: CoordValue,
    y: CoordValue,
    text: str,
    *,
    font_size: float = 12,
    color: str = "#333",
    anchor: str = "start",
    baseline: str = "middle",
    angle: float = 0,
    dx: float = 0,
    dy: float = 0,
    z: str = "above_marks",
) -> AnnotationText:
    """Create a text annotation.

    Parameters
    ----------
    x, y : CoordValue
        Position (data-space float, :func:`px`, or :func:`norm`).
    text : str
        Label text.
    font_size : float, default 12
        Font size in points.
    color : str, default "#333"
        Text color.
    anchor : str, default "start"
        Horizontal anchor: ``"start"``, ``"middle"``, or ``"end"``.
    baseline : str, default "middle"
        Vertical baseline: ``"top"``, ``"middle"``, or ``"bottom"``.
    angle : float, default 0
        Rotation angle in degrees.
    dx : float, default 0
        Horizontal pixel offset from the anchor point.
    dy : float, default 0
        Vertical pixel offset from the anchor point.
    z : str, default "above_marks"
        Z-layer: ``"above_marks"`` or ``"below_marks"``.

    Returns
    -------
    AnnotationText
    """
    return AnnotationText(
        x=x,
        y=y,
        text=text,
        font_size=font_size,
        color=color,
        anchor=anchor,
        baseline=baseline,
        angle=angle,
        dx=dx,
        dy=dy,
        z=z,
    )


def arrow(
    x: CoordValue,
    y: CoordValue,
    x2: CoordValue,
    y2: CoordValue,
    *,
    stroke: str = "#333",
    stroke_width: float = 1.5,
    head_size: float = 8,
    curve: str = "straight",
) -> AnnotationArrow:
    """Create an arrow annotation.

    Parameters
    ----------
    x, y : CoordValue
        Arrow tail position.
    x2, y2 : CoordValue
        Arrow head position.
    stroke : str, default "#333"
        Arrow stroke color.
    stroke_width : float, default 1.5
        Stroke width in pixels.
    head_size : float, default 8
        Arrowhead size in pixels.
    curve : str, default "straight"
        Path style: ``"straight"``, ``"arc"``, or ``"elbow"``.

    Returns
    -------
    AnnotationArrow
    """
    return AnnotationArrow(
        x=x,
        y=y,
        x2=x2,
        y2=y2,
        stroke=stroke,
        stroke_width=stroke_width,
        head_size=head_size,
        curve=curve,
    )


def rect(
    x1: CoordValue,
    y1: CoordValue,
    x2: CoordValue,
    y2: CoordValue,
    *,
    fill: str,
    opacity: float = 0.1,
    stroke: str | None = None,
    corner_radius: float = 0,
) -> AnnotationRect:
    """Create a filled rectangle annotation.

    Parameters
    ----------
    x1, y1 : CoordValue
        Top-left corner position.
    x2, y2 : CoordValue
        Bottom-right corner position.
    fill : str
        Fill color.
    opacity : float, default 0.1
        Fill opacity.
    stroke : str, optional
        Border color; no border when ``None``.
    corner_radius : float, default 0
        Corner rounding radius.

    Returns
    -------
    AnnotationRect
    """
    return AnnotationRect(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        fill=fill,
        opacity=opacity,
        stroke=stroke,
        corner_radius=corner_radius,
    )


def line(
    x1: CoordValue,
    y1: CoordValue,
    x2: CoordValue,
    y2: CoordValue,
    *,
    stroke: str = "#333",
    stroke_width: float = 1,
    dash: list[float] | None = None,
) -> AnnotationLine:
    """Create a line segment annotation.

    Parameters
    ----------
    x1, y1 : CoordValue
        Start position.
    x2, y2 : CoordValue
        End position.
    stroke : str, default "#333"
        Stroke color.
    stroke_width : float, default 1
        Stroke width in pixels.
    dash : list[float], optional
        SVG dash array, e.g. ``[4, 4]``.

    Returns
    -------
    AnnotationLine
    """
    return AnnotationLine(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        stroke=stroke,
        stroke_width=stroke_width,
        dash=dash,
    )


def span(
    axis: str,
    start: CoordValue,
    end: CoordValue,
    *,
    fill: str,
    opacity: float = 0.3,
    label: str | None = None,
    label_position: str = "top",
) -> AnnotationSpan:
    """Create a shaded axis span annotation.

    Parameters
    ----------
    axis : str
        Axis to span: ``"x"`` or ``"y"``.
    start : CoordValue
        Start of the span in data coordinates.
    end : CoordValue
        End of the span in data coordinates.
    fill : str
        Band fill color.
    opacity : float, default 0.3
        Fill opacity.
    label : str, optional
        Text label to display in the band.
    label_position : str, default "top"
        Where to place the label: ``"top"``, ``"middle"``, or ``"bottom"``.

    Returns
    -------
    AnnotationSpan
    """
    return AnnotationSpan(
        axis=axis,
        start=start,
        end=end,
        fill=fill,
        opacity=opacity,
        label=label,
        label_position=label_position,
    )


def bracket(
    x1: CoordValue,
    y1: CoordValue,
    x2: CoordValue,
    y2: CoordValue,
    *,
    label: str,
    direction: str = "above",
    stroke: str = "#333",
    tip_length: float = 6,
) -> AnnotationBracket:
    """Create a bracket annotation with a label.

    Parameters
    ----------
    x1, y1 : CoordValue
        First end of the bracket.
    x2, y2 : CoordValue
        Second end of the bracket.
    label : str
        Label text above/below the bracket.
    direction : str, default "above"
        Which side the bracket opens toward: ``"above"`` or ``"below"``.
    stroke : str, default "#333"
        Bracket stroke color.
    tip_length : float, default 6
        Length of the bracket end ticks in pixels.

    Returns
    -------
    AnnotationBracket
    """
    return AnnotationBracket(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        label=label,
        direction=direction,
        stroke=stroke,
        tip_length=tip_length,
    )


def callout(
    x: CoordValue,
    y: CoordValue,
    text: str,
    *,
    text_x: CoordValue | None = None,
    text_y: CoordValue | None = None,
    arrow: str = "curved",
    padding: float = 4,
    background: str = "#fff",
    border_color: str = "#ccc",
    border_radius: float = 3,
) -> AnnotationCallout:
    """Create a callout bubble annotation.

    Parameters
    ----------
    x, y : CoordValue
        Data point being annotated.
    text : str
        Callout text.
    text_x, text_y : CoordValue, optional
        Position of the text bubble; defaults to a smart offset from ``(x, y)``.
    arrow : str, default "curved"
        Connector style: ``"curved"``, ``"straight"``, or ``"none"``.
    padding : float, default 4
        Padding inside the bubble in pixels.
    background : str, default "#fff"
        Bubble background color.
    border_color : str, default "#ccc"
        Bubble border color.
    border_radius : float, default 3
        Bubble corner radius.

    Returns
    -------
    AnnotationCallout
    """
    return AnnotationCallout(
        x=x,
        y=y,
        text=text,
        text_x=text_x,
        text_y=text_y,
        arrow=arrow,
        padding=padding,
        background=background,
        border_color=border_color,
        border_radius=border_radius,
    )


def image(
    x: CoordValue,
    y: CoordValue,
    src: str,
    *,
    width: float = 50,
    height: float = 50,
    anchor: str = "center",
) -> AnnotationImage:
    """Create an image annotation.

    Parameters
    ----------
    x, y : CoordValue
        Image anchor position.
    src : str
        Image URL or base64 data URI.
    width : float, default 50
        Image width in pixels.
    height : float, default 50
        Image height in pixels.
    anchor : str, default "center"
        Anchor point on the image: ``"center"``, ``"top-left"``, etc.

    Returns
    -------
    AnnotationImage
    """
    return AnnotationImage(
        x=x,
        y=y,
        src=src,
        width=width,
        height=height,
        anchor=anchor,
    )
