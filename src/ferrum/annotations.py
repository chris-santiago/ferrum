"""Lightweight annotation helpers — sugar over primitive marks."""
from __future__ import annotations

from typing import Optional

import polars as pl

from ferrum.chart import Chart


def annotate_hline(y: float, *, label: Optional[str] = None,
                   stroke: Optional[str] = None, stroke_dash=None) -> Chart:
    """Horizontal reference line at y. Returns a single-mark Chart."""
    df = pl.DataFrame({"_y": [y]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(y="_y")


def annotate_vline(x: float, *, label: Optional[str] = None,
                   stroke: Optional[str] = None, stroke_dash=None) -> Chart:
    """Vertical reference line at x."""
    df = pl.DataFrame({"_x": [x]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(x="_x")


def annotate_rect(x1: float, x2: float, y1: float, y2: float, *,
                  fill: Optional[str] = None, opacity: float = 0.1,
                  label: Optional[str] = None) -> Chart:
    """Shaded rectangle region between (x1, y1) and (x2, y2).

    Phase 8a note: x2/y2 are accepted-and-deferred channels; this annotation
    produces a degenerate rect at (x1, y1) until the renderer honors X2/Y2 (Phase 9).
    """
    df = pl.DataFrame({"_x1": [x1], "_x2": [x2], "_y1": [y1], "_y2": [y2]})
    kwargs: dict = {"opacity": opacity}
    if fill is not None:
        kwargs["fill"] = fill
    return Chart(df).mark_rect(**kwargs).encode(x="_x1", y="_y1")


def annotate_text(x: float, y: float, text: str, *, dx: float = 0, dy: float = 0,
                  align: str = "center", baseline: str = "middle",
                  font_size: Optional[float] = None, color: Optional[str] = None,
                  angle: Optional[float] = None) -> Chart:
    """Free text annotation at (x, y).

    Phase 8a note: Text channel is accepted-and-deferred; the actual text content
    goes via mark_kwargs once rendered (Phase 9 wires Text channel properly).
    """
    df = pl.DataFrame({"_x": [x], "_y": [y], "_text": [text]})
    kwargs: dict = {"dx": dx, "dy": dy, "align": align, "baseline": baseline}
    if font_size is not None:
        kwargs["font_size"] = font_size
    if color is not None:
        kwargs["fill"] = color
    if angle is not None:
        kwargs["angle"] = angle
    return Chart(df).mark_text(**kwargs).encode(x="_x", y="_y")
