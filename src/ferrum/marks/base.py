"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""

from __future__ import annotations

from typing import Any


# Aliases map user-friendly names to their canonical renderer-level keys.
# Canonical keys (fill, opacity, stroke_dash) continue to work unchanged.
_MARK_KWARG_ALIASES: dict[str, str] = {
    "color": "fill",
    "alpha": "opacity",
    "linetype": "stroke_dash",
    "line_type": "stroke_dash",
}

_LINETYPE_MAP: dict[str, list[float]] = {
    "solid": [],
    "dashed": [4.0, 2.0],
    "dotted": [1.0, 3.0],
    "dashdot": [4.0, 2.0, 1.0, 2.0],
    "longdash": [8.0, 4.0],
}

_VALID_MARK_KWARGS = frozenset(
    [
        "size",
        "stroke",
        "fill",
        "opacity",
        "corner_radius",
        "stroke_width",
        "stroke_dash",
        "font_size",
        "font_weight",
        "align",
        "baseline",
        "dx",
        "dy",
        "angle",
        # Mark-specific (validated per-mark):
        "interpolate",
        "stroke_cap",
        "stroke_join",  # line/area
        "orient",  # bar/tick
        "filled",
        "shape",  # point
        "limit",  # text
        "band_size",  # tick
        "line",
        "borders",  # area / errorband
        # Statistical mark kwargs (forwarded to transform):
        "method",
        "ci",
        "bandwidth",
        "degree",
        "n",  # smooth
        "kernel",
        "extent",
        "cumulative",  # density
        "bin_count",
        "bin_width",
        "density",
        "right",  # histogram
        "multiple",  # density/histogram
        "blend",  # layer blend mode ("normal", "additive")
        "leader_line",  # label: draw thin leader line from data point to label
    ]
)


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
        # Resolve aliases before validation so canonical keys always pass
        # and friendly aliases (color, alpha, linetype) are transparently
        # remapped.  Canonical keys (fill, opacity, stroke_dash) still work
        # unchanged — the alias dict only covers the friendly names.
        resolved: dict[str, Any] = {}
        for k, v in kwargs.items():
            canonical = _MARK_KWARG_ALIASES.get(k, k)
            if canonical == "stroke_dash" and k in ("linetype", "line_type") and isinstance(v, str):
                if v in _LINETYPE_MAP:
                    v = _LINETYPE_MAP[v]
                else:
                    v = [float(x) for x in v.split(",") if x.strip()]
            resolved[canonical] = v
        for k in resolved:
            if k not in _VALID_MARK_KWARGS:
                raise TypeError(
                    f"mark_{mark_name}: unknown keyword argument {k!r}. "
                    f"Valid: {sorted(_VALID_MARK_KWARGS)}"
                )
        self._kwargs = resolved

    @property
    def kwargs(self) -> dict[str, Any]:
        """Read-only view of the resolved (alias-expanded) kwargs dict."""
        return dict(self._kwargs)

    def to_mark_kwargs_dict(self) -> dict:
        """Return the subset of stored kwargs that map to ``MarkKwargsSpec`` fields.

        Statistical mark kwargs (``bandwidth``, ``method``, ``ci``, etc.)
        are not included here — they are consumed directly by the desugar
        functions (``desugar_density``, ``desugar_smooth``, …) which build
        the transform objects before this dict is ever inspected.

        ``orient`` is consumed Python-side (sets ``_coord = "flip"`` on the
        chart) and is never forwarded to the Rust renderer.

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
        for k in (
            # Core style fields (Phase 8a)
            "size",
            "stroke",
            "fill",
            "opacity",
            "corner_radius",
            "stroke_width",
            "stroke_dash",
            "font_size",
            "font_weight",
            "align",
            "baseline",
            "dx",
            "dy",
            "angle",
            # S1: interpolate (line/area)
            "interpolate",
            # S2: stroke_cap (line)
            "stroke_cap",
            # S3: stroke_join (line/area)
            "stroke_join",
            # S5: filled (point)
            "filled",
            # S6: shape (point, constant)
            "shape",
            # S7: limit (text)
            "limit",
            # S8: band_size (tick/rect)
            "band_size",
            # S9: line border on area
            "line",
            # S10: borders on area/errorband
            "borders",
            # S11: blend mode (layer blend)
            "blend",
            # S12: leader_line (label)
            "leader_line",
        ):
            if k in self._kwargs:
                out[k] = self._kwargs[k]
        # S4: orient="horizontal" → consumed Python-side; set coord flip flag.
        # The caller (_set_mark) reads this via orient_coord_flip().
        return out

    def orient_coord_flip(self) -> bool:
        """Return True if ``orient="horizontal"`` was passed, indicating coord flip.

        Used by ``Chart._set_mark`` to set ``_coord = "flip"`` without
        forwarding ``orient`` to the Rust renderer.

        Returns
        -------
        bool
            True when ``orient="horizontal"`` is in the stored kwargs.

        Examples
        --------
        >>> mb = MarkBase("bar", orient="horizontal")
        >>> mb.orient_coord_flip()
        True
        >>> MarkBase("bar").orient_coord_flip()
        False
        """
        return self._kwargs.get("orient") == "horizontal"
