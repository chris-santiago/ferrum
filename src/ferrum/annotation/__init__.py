"""Annotation package — primitives, coordinates, and the Annotate container.

Usage::

    import ferrum.annotation as ann
    from ferrum.annotation import Annotate

    annotations = Annotate([
        ann.text(1.0, 2.0, "label"),
        ann.span("x", 0, 1, fill="#eee", opacity=0.2),
        ann.arrow(0, 0, 1, 1),
    ])
"""

from __future__ import annotations

from ferrum.annotation.coords import (
    CoordValue,
    NormCoord,
    PixelCoord,
    norm,
    px,
    temporal_coord_to_epoch_ms,
)
from ferrum.annotation.container import Annotate
from ferrum.annotation.primitives import (
    AnnotationArrow,
    AnnotationBracket,
    AnnotationCallout,
    AnnotationImage,
    AnnotationLine,
    AnnotationRect,
    AnnotationSpan,
    AnnotationText,
    arrow,
    bracket,
    callout,
    image,
    line,
    rect,
    span,
    text,
)

__all__ = [
    # Coordinate wrappers
    "CoordValue",
    "PixelCoord",
    "NormCoord",
    "px",
    "norm",
    "temporal_coord_to_epoch_ms",
    # Container
    "Annotate",
    # Dataclass types
    "AnnotationText",
    "AnnotationArrow",
    "AnnotationRect",
    "AnnotationLine",
    "AnnotationSpan",
    "AnnotationBracket",
    "AnnotationCallout",
    "AnnotationImage",
    # Factory functions
    "text",
    "arrow",
    "rect",
    "line",
    "span",
    "bracket",
    "callout",
    "image",
]
