"""Figure-chrome positioning resolution for composite and single-chart renders.

The Rust SVG compositors (``compose_svg_horizontal`` / ``compose_svg_vertical``
/ ``compose_svg_grid``) emit a figure-level title / subtitle / caption band
around the composed panels.  The horizontal position of that band is controlled
by three keyword arguments on the Rust side:

- ``left_inset``  — x of a ``start``-anchored band (left padding, in pixels)
- ``right_inset`` — distance from the right edge of an ``end``-anchored band
- ``anchor``      — ``"start"`` / ``"middle"`` / ``"end"`` text alignment

When any of these is omitted (``None``), the Rust emitter applies its own
default (inset ``16.0``, anchor ``"start"``).  This module resolves those values
from a chart's merged ``configure`` dict so that ``configure_padding(left=...)``
and ``configure_title(anchor=...)`` reach the chrome band.  Omitting a key whose
source value is unset lets the Rust default stand, which keeps a no-config
composite emitting its chrome at the default inset.

The two functions here are pure: they take plain data (a list of ``Configure``
objects, or an already-merged dict) and return plain data.  They do not import
``Chart``, touch the filesystem, or read environment.
"""

from __future__ import annotations

from typing import Any


def merge_configure_layers(layers: Any) -> dict:
    """Merge a list of ``Configure`` objects into one config dict.

    Uses the same deep-merge semantics as
    ``_RenderMixin._resolve_chart_config``: each layer contributes its
    ``.to_dict()``; nested dict values are shallow-merged so later layers win
    key-by-key, and non-dict values are replaced outright.  Later layers in the
    list override earlier ones.

    Returns an empty dict when *layers* is falsy (no configure layers set).
    """
    if not layers:
        return {}
    merged: dict = {}
    for cfg in layers:
        d = cfg.to_dict()
        for key, val in d.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val
    return merged


def chrome_kwargs(merged: dict) -> dict:
    """Resolve figure-chrome positioning kwargs from a merged config dict.

    Reads ``configure_padding(left=/right=)`` and ``configure_title(anchor=)``
    out of *merged* and returns a kwargs dict containing **only** the keys whose
    source value is actually set.  Any of ``left_inset`` / ``right_inset`` /
    ``anchor`` whose source is ``None`` or absent is omitted so that the Rust
    emitter's default applies.  A merged dict with no relevant keys yields an
    empty dict, leaving every chrome position at its Rust default.
    """
    padding = merged.get("padding") or {}
    title = merged.get("title") or {}

    kwargs: dict = {}
    left = padding.get("left")
    if left is not None:
        kwargs["left_inset"] = left
    right = padding.get("right")
    if right is not None:
        kwargs["right_inset"] = right
    anchor = title.get("anchor")
    if anchor is not None:
        kwargs["anchor"] = anchor
    return kwargs
