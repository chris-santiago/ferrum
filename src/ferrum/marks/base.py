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
    """Validate and store mark-level keyword arguments for primitive marks.

    Used by ``mark_*()`` builder functions to validate kwargs before
    serializing them into ``ChartSpec.mark_style``. Raises ``TypeError``
    on the first unrecognized key so typos surface immediately rather than
    silently disappearing into the renderer.

    Parameters
    ----------
    mark_name : str
        Name of the mark (e.g. ``"point"``, ``"bar"``). Used in error
        messages only.
    **kwargs : Any
        Style overrides forwarded to the renderer (e.g. ``size=100``,
        ``fill="red"``). Every key must appear in ``_VALID_MARK_KWARGS``.

    Raises
    ------
    TypeError
        If any keyword argument is not in the renderer's allowlist.

    Examples
    --------
    >>> from ferrum.marks.base import MarkBase
    >>> mb = MarkBase("point", size=80, fill="steelblue")
    >>> mb.to_mark_kwargs_dict()
    {'size': 80, 'fill': 'steelblue'}
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
        """Return the subset of stored kwargs that map to ``MarkKwargsSpec`` fields.

        Statistical mark kwargs (``bandwidth``, ``method``, ``ci``, etc.)
        are not included here — they are consumed directly by the desugar
        functions (``desugar_density``, ``desugar_smooth``, …) which build
        the transform objects before this dict is ever inspected.

        Returns
        -------
        dict
            Mapping of renderer-level style keys to their values.  Only
            keys present in the stored kwargs are included; absent keys do
            not appear (no ``None`` defaults).

        Examples
        --------
        >>> mb = MarkBase("bar", size=40, opacity=0.8)
        >>> mb.to_mark_kwargs_dict()
        {'size': 40, 'opacity': 0.8}
        """
        out = {}
        for k in ("size", "stroke", "fill", "opacity", "corner_radius",
                  "stroke_width", "stroke_dash", "font_size", "font_weight",
                  "align", "baseline", "dx", "dy", "angle"):
            if k in self._kwargs:
                out[k] = self._kwargs[k]
        return out
