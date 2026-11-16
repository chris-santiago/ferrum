"""Annotation primitive dataclasses and factory functions.

Each factory function returns a frozen dataclass instance that can be
collected into an [Annotate][ferrum.Annotate] container
and attached to a chart.

All coordinate arguments accept:

- ``float`` / ``int`` — data-space numeric coordinates
- ``datetime.date`` / ``datetime.datetime`` — temporal data-space coordinates
  (converted to epoch-milliseconds UTC at serialization time)
- ISO-8601 date/datetime strings — parsed and converted to epoch-milliseconds
- any other string — a category label, placed where the axis draws that
  category (the band centre; a span covers whole bands)
- [PixelCoord][ferrum.PixelCoord] — absolute pixel offset
- [NormCoord][ferrum.NormCoord] — normalized [0, 1] fraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ``_coord`` is the single source of truth for coordinate serialization; it
# lives in ``coords.py`` alongside the wrapper types and coercion functions.
# Re-imported (not re-defined) here so the ``to_dict()`` methods below and any
# caller importing ``ferrum.annotation.primitives._coord`` keep working.
from ferrum.annotation.coords import CoordValue, _coord  # noqa: F401
from ferrum._entry import _voiced
from ferrum._validate import (
    is_non_finite_number,
    is_none_color_sentinel,
    validate_choice,
    validate_str,
)


# ---------------------------------------------------------------------------
# Rust-vocabulary constants
# ---------------------------------------------------------------------------
#
# Finding P6 claim-check (2026-08-27): the validated set for a field consumed
# by Rust must be the union of every literal value Rust's own parser
# recognizes and renders distinctly -- including aliases -- not just the
# canonical token shown in this module's docstrings. Validating against a
# narrower, docstring-only vocabulary regresses previously-working input
# (e.g. ``anchor="left"``, reachable via ``annotate_text``'s
# ``_ANCHOR_TO_ALIGN`` alias pass-through) into a construction-time
# ``ValueError``. Each constant below is cross-checked against its Rust
# match arm; see the per-field comment for the exact source location.

# render/annotation/mod.rs::parse_anchor -- "start"|"left" -> Start,
# "end"|"right" -> End, anything else (including "middle") -> Middle.
_VALID_TEXT_ANCHORS = ("start", "left", "end", "right", "middle")

# render/annotation/mod.rs::parse_baseline -> render/draw.rs::parse_text_baseline
# -- four DISTINCT rendered variants, not three: Top ("top"/"hanging"/
# "text-before-edge"), Middle ("middle"/"central", also the default for any
# unrecognized string), Bottom ("bottom"/"text-after-edge"/"ideographic"),
# and Alphabetic ("alphabetic" -- a real fourth variant absent from the
# public docstring's {"top", "middle", "bottom"} vocabulary).
_VALID_TEXT_BASELINES = (
    "top",
    "hanging",
    "text-before-edge",
    "middle",
    "central",
    "bottom",
    "text-after-edge",
    "ideographic",
    "alphabetic",
)

# render/annotation/mod.rs::build_annotations -- exact-match "below_marks" routes
# to the pre-marks bucket; every other string (including "above_marks")
# routes to the post-marks bucket. No further Rust-recognized alias exists.
_VALID_ANNOTATION_Z = ("above_marks", "below_marks")

# render/annotation/mod.rs::emit_bracket -- "up"/"down"/"left"/"right" are the tip directions the
# renderer draws. "above" is the documented default and the synonym of "up"; "below" is the
# synonym of "down".
_VALID_BRACKET_DIRECTIONS = ("above", "below", "up", "down", "left", "right")


def _validate_paint(entry: str, field: str, value: Any) -> None:
    """Refuse *value* as the colour *field* of the public call *entry*.

    Only ``None`` and the paint-clear words ``"none"`` and ``"transparent"``
    skip the colour parser; these are the paint words the annotation renderer
    draws. The marks' literal-colour gate also lets the internal theme tokens
    (``"theme:font"``, ``"theme:label"``) through, but the renderer does not
    draw them on an annotation, so this gate refuses them. A non-string is
    refused as the marks refuse it, and any other string must parse as a
    colour. The refusal is an ``EntryRefusal`` voiced as *entry*, so an
    ``annotate_*`` helper re-voices it as itself.
    """
    from ferrum.marks.base import _validate_literal_color

    if value is None:
        return
    _voiced(entry, lambda voice: validate_str(voice, field, value))
    if is_none_color_sentinel(value):
        return
    _voiced(entry, lambda voice: _validate_literal_color(voice, field, value))


def _validate_coord(entry: str, field: str, axis: str, value: Any) -> None:
    """Refuse *value* as the coordinate *field* of the public call *entry* when it is a number
    with no position (NaN, an infinity, or an integer too large for a float).

    Every other coordinate (a finite number, a date, a category label, ``NormCoord``,
    ``PixelCoord``) passes: only a number can be non-finite. *axis* is the axis the coordinate
    names. The refusal is an ``EntryRefusal`` voiced as *entry*, so an ``annotate_*`` helper
    re-voices it as itself.
    """

    def check(voice: str) -> None:
        if is_non_finite_number(value):
            raise ValueError(
                f"{voice}: {field}={value} is not a finite number, so it has no position; "
                f"pass a value the {axis} axis holds, fm.NormCoord or fm.PixelCoord"
            )

    _voiced(entry, check)


def _validate_coords(entry: str, **fields: Any) -> None:
    """:func:`_validate_coord` over each ``field=value``, in the order given.

    A field's axis is the letter that names it: the first letter of ``x`` and ``y2``, the last
    word of ``text_y``.
    """
    for field, value in fields.items():
        axis = field[0] if field[0] in "xy" else field.split("_")[-1][0]
        _validate_coord(entry, field, axis, value)


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
    color: str | None
    anchor: str
    baseline: str
    angle: float
    dx: float
    dy: float
    z: str

    def __post_init__(self) -> None:
        validate_choice("AnnotationText.anchor", "anchor", self.anchor, _VALID_TEXT_ANCHORS)
        validate_choice("AnnotationText.baseline", "baseline", self.baseline, _VALID_TEXT_BASELINES)
        validate_choice("AnnotationText.z", "z", self.z, _VALID_ANNOTATION_Z)
        _validate_coords("annotation.text", x=self.x, y=self.y)
        _validate_paint("annotation.text", "color", self.color)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "text",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "text": self.text,
            "font_size": self.font_size,
            "anchor": self.anchor,
            "baseline": self.baseline,
            "angle": self.angle,
            "dx": self.dx,
            "dy": self.dy,
            "z": self.z,
        }
        if self.color is not None:
            d["color"] = self.color
        return d


@dataclass(frozen=True)
class AnnotationArrow:
    """An arrow between two data points."""

    x: CoordValue
    y: CoordValue
    x2: CoordValue
    y2: CoordValue
    stroke: str | None
    stroke_width: float
    head_size: float

    def __post_init__(self) -> None:
        _validate_coords("annotation.arrow", x=self.x, y=self.y, x2=self.x2, y2=self.y2)
        _validate_paint("annotation.arrow", "stroke", self.stroke)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "arrow",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "stroke_width": self.stroke_width,
            "head_size": self.head_size,
        }
        if self.stroke is not None:
            d["stroke"] = self.stroke
        return d


@dataclass(frozen=True)
class AnnotationRect:
    """A filled rectangle region."""

    x1: CoordValue
    y1: CoordValue
    x2: CoordValue
    y2: CoordValue
    fill: str | None
    opacity: float
    stroke: str | None
    corner_radius: float

    def __post_init__(self) -> None:
        _validate_coords("annotation.rect", x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
        _validate_paint("annotation.rect", "fill", self.fill)
        _validate_paint("annotation.rect", "stroke", self.stroke)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "rect",
            "x1": _coord(self.x1),
            "y1": _coord(self.y1),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "opacity": self.opacity,
            "corner_radius": self.corner_radius,
        }
        if self.fill is not None:
            d["fill"] = self.fill
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
    stroke: str | None
    stroke_width: float
    dash: list[float] | None
    #: The public helper that wrote this line's coordinates itself (not the caller), so a
    #: refusal at render time names that call. Unset for every line the caller positions.
    entry: str | None = None

    def __post_init__(self) -> None:
        _validate_coords("annotation.line", x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
        _validate_paint("annotation.line", "stroke", self.stroke)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "line",
            "x1": _coord(self.x1),
            "y1": _coord(self.y1),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "stroke_width": self.stroke_width,
        }
        if self.stroke is not None:
            d["stroke"] = self.stroke
        if self.dash is not None:
            d["dash"] = self.dash
        if self.entry is not None:
            d["entry"] = self.entry
        return d


@dataclass(frozen=True)
class AnnotationSpan:
    """A shaded span along one axis (like a horizontal or vertical band)."""

    axis: str
    start: CoordValue
    end: CoordValue
    fill: str | None
    opacity: float
    label: str | None
    label_position: str

    def __post_init__(self) -> None:
        validate_choice("AnnotationSpan.axis", "axis", self.axis, ("x", "y"))
        validate_choice(
            "AnnotationSpan.label_position",
            "label_position",
            self.label_position,
            ("top", "middle", "bottom"),
        )
        _validate_coord("annotation.span", "start", self.axis, self.start)
        _validate_coord("annotation.span", "end", self.axis, self.end)
        _validate_paint("annotation.span", "fill", self.fill)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "span",
            "axis": self.axis,
            "start": _coord(self.start),
            "end": _coord(self.end),
            "opacity": self.opacity,
            "label_position": self.label_position,
        }
        if self.fill is not None:
            d["fill"] = self.fill
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
    stroke: str | None
    tip_length: float

    def __post_init__(self) -> None:
        validate_choice(
            "AnnotationBracket.direction", "direction", self.direction, _VALID_BRACKET_DIRECTIONS
        )
        _validate_coords("annotation.bracket", x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
        _validate_paint("annotation.bracket", "stroke", self.stroke)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "bracket",
            "x1": _coord(self.x1),
            "y1": _coord(self.y1),
            "x2": _coord(self.x2),
            "y2": _coord(self.y2),
            "label": self.label,
            "direction": self.direction,
            "tip_length": self.tip_length,
        }
        if self.stroke is not None:
            d["stroke"] = self.stroke
        return d


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
    background: str | None
    border_color: str | None
    border_radius: float

    def __post_init__(self) -> None:
        validate_choice(
            "AnnotationCallout.arrow", "arrow", self.arrow, ("curved", "straight", "none")
        )
        _validate_coords("annotation.callout", x=self.x, y=self.y)
        if self.text_x is not None:
            _validate_coord("annotation.callout", "text_x", "x", self.text_x)
        if self.text_y is not None:
            _validate_coord("annotation.callout", "text_y", "y", self.text_y)
        _validate_paint("annotation.callout", "background", self.background)
        _validate_paint("annotation.callout", "border_color", self.border_color)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for renderer transport."""
        d: dict[str, Any] = {
            "type": "callout",
            "x": _coord(self.x),
            "y": _coord(self.y),
            "text": self.text,
            "arrow": self.arrow,
            "padding": self.padding,
            "border_radius": self.border_radius,
        }
        if self.background is not None:
            d["background"] = self.background
        if self.border_color is not None:
            d["border_color"] = self.border_color
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

    def __post_init__(self) -> None:
        _validate_coords("annotation.image", x=self.x, y=self.y)

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
# Factory functions
# ---------------------------------------------------------------------------


def text(
    x: CoordValue,
    y: CoordValue,
    text: str,
    *,
    font_size: float = 12,
    color: str | None = None,
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
        Position (data-space float, a category label, [px][ferrum.px], or
        [norm][ferrum.norm]).
    text : str
        Label text.
    font_size : float, default 12
        Font size in points.
    color : str, optional
        Text color. Defaults to the theme's text colour (``font_color``).
    anchor : str, default "start"
        Horizontal anchor: ``"start"``, ``"middle"``, or ``"end"`` (SVG
        vocabulary); ``"left"`` and ``"right"`` are accepted aliases for
        ``"start"`` and ``"end"`` respectively.
    baseline : str, default "middle"
        Vertical baseline: ``"top"``, ``"middle"``, ``"bottom"``, or
        ``"alphabetic"``. Each also accepts CSS ``dominant-baseline``
        aliases: ``"hanging"``/``"text-before-edge"`` for ``"top"``;
        ``"central"`` for ``"middle"``; ``"text-after-edge"``/
        ``"ideographic"`` for ``"bottom"``.
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
    stroke: str | None = None,
    stroke_width: float = 1.5,
    head_size: float = 8,
) -> AnnotationArrow:
    """Create an arrow annotation.

    Parameters
    ----------
    x, y : CoordValue
        Arrow tail position.
    x2, y2 : CoordValue
        Arrow head position.
    stroke : str, optional
        Arrow stroke color. Defaults to the theme's text colour (``font_color``).
    stroke_width : float, default 1.5
        Stroke width in pixels.
    head_size : float, default 8
        Arrowhead size in pixels.

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
    )


def rect(
    x1: CoordValue,
    y1: CoordValue,
    x2: CoordValue,
    y2: CoordValue,
    *,
    fill: str | None = None,
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
    fill : str, optional
        Fill color. Defaults to the theme's text colour (``font_color``).
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
    stroke: str | None = None,
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
    stroke : str, optional
        Stroke color. Defaults to the theme's text colour (``font_color``).
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
    fill: str | None = None,
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
        Start of the span in data coordinates. Both ends are categories or both
        are values; a span over categories shades whole bands, from the outer
        edge of the lower-placed category's band to the outer edge of the
        higher-placed one (``start == end`` shades that band).
    end : CoordValue
        End of the span, of the same kind as ``start``.
    fill : str, optional
        Band fill color. Defaults to the theme's text colour (``font_color``).
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
    stroke: str | None = None,
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
        Which side the tip ticks point toward: ``"up"``, ``"down"``,
        ``"left"``, or ``"right"``. ``"above"`` is a synonym for ``"up"``
        (the default); ``"below"`` is a synonym for ``"down"``. Under a
        caller's ``CoordFlip`` the direction exchanges with the coordinate
        pair, so a bracket's tips and label still point away from its
        baseline.
    stroke : str, optional
        Bracket stroke color. Defaults to the theme's text colour (``font_color``).
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
    background: str | None = None,
    border_color: str | None = None,
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
        Connector style: ``"curved"``, ``"straight"``, or ``"none"`` (no
        leader line). ``"curved"`` and ``"straight"`` currently render
        identically (a straight leader line) — a known renderer limitation
        (no curve is implemented), not a Python-side validation gap.
    padding : float, default 4
        Padding inside the bubble in pixels.
    background : str, optional
        Bubble background color. Defaults to the theme's background colour.
    border_color : str, optional
        Bubble border color; the leader line takes it too. Defaults to the
        theme's text colour (``font_color``), which the bubble's text always
        takes.
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
