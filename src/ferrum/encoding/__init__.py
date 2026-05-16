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


def _apply_channel_aliases(enc: dict, mk: dict) -> tuple[dict, dict]:
    """Apply channel-alias rules, mapping convenience channels to their targets.

    Operates on shallow copies of the encoding and mark-kwargs dicts from
    ``to_spec()`` — does not mutate the chart's internal state.

    Alias rules (order matters — earlier aliases take priority):

    1. ``fill`` -> ``color`` when ``color`` is not already present.
    2. ``stroke`` -> ``color`` when ``color`` is not already present;
       when ``color`` IS present, the stroke encoding is silently dropped.
    3. ``detail`` -> ``mk["detail"]`` via ``setdefault`` (always, regardless
       of other channels).

    Note: ``fill_opacity`` is no longer aliased to ``opacity``. It is a
    first-class renderer-honored channel that emits a per-element SVG
    ``fill-opacity`` attribute, separate from ``opacity`` (which bakes
    into the fill RGBA alpha).

    Returns the (possibly-modified) ``(enc, mk)`` pair.
    """
    from ferrum.repeat import _RepeatPlaceholder

    # Fill -> color
    if "fill" in enc and "color" not in enc:
        enc["color"] = enc["fill"]

    # Stroke -> color (when color absent); silent drop otherwise.
    if "stroke" in enc:
        stroke_ch = enc["stroke"]
        if "color" not in enc:
            enc["color"] = stroke_ch
        elif stroke_ch.field is not None and not isinstance(stroke_ch.field, _RepeatPlaceholder):
            # Can't map to a scale -- inject as a mark_style grouping hint.
            # mark_style.stroke expects a hex color, not a field name, so
            # this is a best-effort: when the user maps a field to stroke
            # while color is already mapped, the stroke encoding is silently
            # stored but produces no visual effect.
            pass

    # Detail -> mark_style.detail
    if "detail" in enc:
        detail_ch = enc["detail"]
        if detail_ch.field is not None and not isinstance(detail_ch.field, _RepeatPlaceholder):
            mk.setdefault("detail", detail_ch.field)

    return enc, mk
