"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""
from __future__ import annotations

from typing import Any, ClassVar


_VALID_MARK_KWARGS = frozenset([
    "size", "stroke", "fill", "opacity", "corner_radius",
    "stroke_width", "stroke_dash", "font_size", "font_weight",
    "align", "baseline", "dx", "dy", "angle",
    # Mark-specific (validated per-mark):
    "interpolate", "stroke_cap", "stroke_join",            # line/area
    "orient",                                              # bar/tick
    "filled", "shape",                                      # point
    "limit",                                               # text
    "band_size",                                           # tick
    "line", "borders",                                     # area / errorband
    # Statistical mark kwargs (forwarded to transform):
    "method", "ci", "bandwidth", "degree", "n",            # smooth
    "kernel", "extent", "cumulative",                      # density
    "bin_count", "bin_width", "density", "right",          # histogram
    "multiple",                                            # density/histogram
])


class MarkBase:
    """Validate + store mark-level keyword arguments.

    Used by mark_*() builder functions in marks/__init__.py to validate kwargs
    before serializing them into ChartSpec.mark_style.
    """

    def __init__(self, mark_name: str, **kwargs: Any) -> None:
        self.mark_name = mark_name
        for k in kwargs:
            if k not in _VALID_MARK_KWARGS:
                raise TypeError(
                    f"mark_{mark_name}: unknown keyword argument {k!r}. "
                    f"Valid: {sorted(_VALID_MARK_KWARGS)}"
                )
        self._kwargs = dict(kwargs)

    def to_mark_kwargs_dict(self) -> dict:
        """Subset of kwargs that map to MarkKwargsSpec fields. Other kwargs
        (e.g. statistical mark kwargs like `bandwidth`) are returned in
        `to_transform_kwargs()` if applicable, not here."""
        out = {}
        for k in ("size", "stroke", "fill", "opacity", "corner_radius",
                  "stroke_width", "stroke_dash", "font_size", "font_weight",
                  "align", "baseline", "dx", "dy", "angle"):
            if k in self._kwargs:
                out[k] = self._kwargs[k]
        return out
