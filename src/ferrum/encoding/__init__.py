"""Encoding channels — declarative mappings from data fields to visual variables."""

from __future__ import annotations

import functools

from ferrum.encoding.positional import (
    X,
    Y,
    X2,
    Y2,
    XError,
    YError,
    XError2,
    YError2,
    Theta,
    Radius,
    Theta2,
    Radius2,
)
from ferrum.encoding.appearance import (
    Color,
    Fill,
    Stroke,
    Opacity,
    FillOpacity,
    StrokeOpacity,
    StrokeWidth,
    StrokeDash,
    Size,
    Shape,
    Angle,
)
from ferrum.encoding.text import (
    Text,
    Detail,
    Tooltip,
    TooltipField,
    Href,
    Description,
    Key,
    Url,
)
from ferrum.encoding.facet import Facet, FacetRow, FacetCol
from ferrum.encoding._aliases import apply_channel_aliases

__all__ = [
    "X",
    "Y",
    "X2",
    "Y2",
    "XError",
    "YError",
    "XError2",
    "YError2",
    "Theta",
    "Radius",
    "Theta2",
    "Radius2",
    "Color",
    "Fill",
    "Stroke",
    "Opacity",
    "FillOpacity",
    "StrokeOpacity",
    "StrokeWidth",
    "StrokeDash",
    "Size",
    "Shape",
    "Angle",
    "Text",
    "Detail",
    "Tooltip",
    "TooltipField",
    "Href",
    "Description",
    "Key",
    "Url",
    "Facet",
    "FacetRow",
    "FacetCol",
]


# ---------------------------------------------------------------------------
# Channel-lookup helpers (extracted from chart.py)
# ---------------------------------------------------------------------------


@functools.cache
def _channel_class_map() -> dict:
    """Build the channel-name -> channel-class mapping (cached; once-per-process)."""
    return {
        "x": X,
        "y": Y,
        "x2": X2,
        "y2": Y2,
        "x_error": XError,
        "y_error": YError,
        "x_error2": XError2,
        "y_error2": YError2,
        "theta": Theta,
        "radius": Radius,
        "theta2": Theta2,
        "radius2": Radius2,
        "color": Color,
        "fill": Fill,
        "stroke": Stroke,
        "opacity": Opacity,
        "fill_opacity": FillOpacity,
        "stroke_opacity": StrokeOpacity,
        "stroke_width": StrokeWidth,
        "stroke_dash": StrokeDash,
        "size": Size,
        "shape": Shape,
        "angle": Angle,
        "text": Text,
        "detail": Detail,
        "tooltip": Tooltip,
        "tooltip_field": TooltipField,
        "href": Href,
        "description": Description,
        "key": Key,
        "url": Url,
        "facet": Facet,
        "facet_row": FacetRow,
        "facet_col": FacetCol,
    }


def _channel_class_for(name: str):
    """Return the channel-class for a given parameter name."""
    return _channel_class_map().get(name)


# Re-exported from encoding/_aliases.py (its real home, next to the channel
# classes); the package init stays an export surface.  The leading-underscore
# name is preserved because chart.py imports it from here.
_apply_channel_aliases = apply_channel_aliases
