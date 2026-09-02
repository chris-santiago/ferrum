"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""

from __future__ import annotations

from typing import Any

from ferrum._validate import is_none_color_sentinel, validate_choice

# Marks whose primary visual channel is stroke rather than fill.
# For these marks, the user-facing ``color=`` alias resolves to ``stroke``
# instead of ``fill``.  Fill-primary marks (bar, area, point, rect, arc,
# tick, …) keep the ``color`` → ``fill`` mapping.
_STROKE_PRIMARY_MARKS: frozenset[str] = frozenset(["line", "rule", "segment", "trail"])

# Aliases map user-friendly names to their canonical renderer-level keys.
# ``color`` is handled separately in ``MarkBase.__init__`` because the
# canonical target depends on whether the mark is stroke-primary or not.
# All other aliases are mark-type-independent.
_MARK_KWARG_ALIASES: dict[str, str] = {
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


def _is_paint_sentinel(value: str) -> bool:
    """Return ``True`` if *value* is a non-color paint sentinel, not a color.

    Mirrors ``resolve_paint_color`` in
    ``crates/ferrum-core/src/render/draw.rs`` exactly, so this Python gate
    accepts/rejects the identical set of strings as the Rust authority:

    - ``"theme:label"`` — exact string match (draw.rs matches it exactly,
      not trimmed, not case-folded). An internal theme-lookup token
      composite marks pass as ``stroke=``/``fill=`` (see
      ``marks/composite.py``).
    - ``"none"``/``"transparent"`` — trimmed and case-insensitive (draw.rs:
      ``trimmed.eq_ignore_ascii_case("none") || trimmed.eq_ignore_ascii_case("transparent")``),
      so ``"None"``, ``"NONE"``, ``" none "``, ``"Transparent"``,
      ``"TRANSPARENT"``, and ``" transparent "`` all count. Both spellings
      are an explicit paint-clear (spec §4.1; ``"transparent"`` joined
      ``"none"`` as a clearing spelling in the 2026-09-01 T8 quality-review
      supersession — refusing a real CSS Color 4 keyword with a message
      that promises CSS-name support recreated the same divergence class
      this batch remediates).

    All three sentinel spellings must be checked BEFORE
    ``ferrum.color.to_hex`` and never reach the parser — all raise inside
    ``to_hex`` by design, see
    ``tests/test_color_vocabulary.py::TestSentinelsAreNotColors``. The
    ``"none"``/``"transparent"`` arm is shared with
    ``ferrum._validate.is_none_color_sentinel`` (``selection.py`` composes
    from the same predicate for its own, differently-handled refusal).
    """
    return value == "theme:label" or is_none_color_sentinel(value)


# Canonical set of valid constant shape names for mark_point(shape=...).
# Must stay in sync with shape_from_str() in crates/ferrum-core/src/render/marks/point.rs.
_VALID_POINT_SHAPES: frozenset[str] = frozenset(
    [
        "circle",
        "square",
        "cross",
        "diamond",
        "triangle-up",
        "triangle_up",
        "triangle-down",
        "triangle_down",
        "|",
        "vline",
        "-",
        "hline",
    ]
)

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
        "zero",  # mark_bar: suppress the y-scale zero-anchor (zero=False)
    ]
)


def _validate_literal_color(mark_name: str, key: str, value: str) -> None:
    """Raise ``ValueError`` if *value* is not a color ``ferrum.color.to_hex`` accepts.

    ``to_hex`` is ferrum's single Rust color parser (``parse_color`` exposed
    via ``ferrum._core``); this function is a thin construction-time gate
    around it, not a second color vocabulary — the raised message wraps
    to_hex's own accepted-forms text unchanged. Callers must short-circuit
    the non-color sentinels (``"none"``, ``"transparent"``, ``"theme:label"``)
    before calling this function — each raises here by design (``"none"``/
    ``"transparent"`` with to_hex's sentinel-aware clearing-paint message,
    ``"theme:label"`` with the generic accepted-forms text), matching
    to_hex's behavior.

    The import is lazy (inside this function, not at module load time) to
    avoid a module-init-order dependency on ``ferrum.color``, but it is not
    guarded: ``ferrum.marks.base`` cannot be reached without
    ``ferrum._core`` already having loaded (``ferrum/__init__.py`` imports
    it unconditionally before anything else), so ``ferrum.color`` — whose
    only import is ``ferrum._core`` — is always importable by the time this
    runs. If that ever stops being true (e.g. a stale compiled extension
    missing ``parse_color_to_hex``), that is a real build/environment
    defect and must fail loudly here, not silently disable this batch's
    "bad colors fail at construction" guarantee.
    """
    from ferrum.color import to_hex

    try:
        to_hex(value)
    except ValueError as exc:
        raise ValueError(f"mark_{mark_name}: {key}={value!r} is not a valid color ({exc})") from exc


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
        #
        # ``color`` is mark-type-aware: stroke-primary marks (line, rule,
        # segment, trail) map it to ``stroke``; all other marks map to ``fill``.
        color_target = "stroke" if mark_name in _STROKE_PRIMARY_MARKS else "fill"
        resolved: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k == "color":
                canonical = color_target
            else:
                canonical = _MARK_KWARG_ALIASES.get(k, k)
            if canonical == "stroke_dash" and isinstance(v, str):
                # Named forms ("dashed", "dotted", ...) are only recognized
                # via the linetype/line_type aliases; the canonical
                # ``stroke_dash=`` kwarg takes the documented "a,b" comma
                # -split form directly (no named-form lookup for it).
                if k in ("linetype", "line_type") and v in _LINETYPE_MAP:
                    v = _LINETYPE_MAP[v]
                else:
                    try:
                        v = [float(x) for x in v.split(",") if x.strip()]
                    except ValueError as exc:
                        raise ValueError(
                            f"mark_{mark_name}: {k}={v!r} is not a valid stroke_dash "
                            "value. Expected a numeric list (e.g. [4.0, 2.0]), a "
                            'comma-separated numeric string (e.g. "4,2"), or — via '
                            f"linetype=/line_type= — a named linetype "
                            f"(one of {sorted(_LINETYPE_MAP)})."
                        ) from exc
            resolved[canonical] = v
        for k in resolved:
            if k not in _VALID_MARK_KWARGS:
                raise TypeError(
                    f"mark_{mark_name}: unknown keyword argument {k!r}. "
                    f"Valid: {sorted(_VALID_MARK_KWARGS)}"
                )
        # Validate shape= value for point marks. The constant shape is a fixed
        # string that must name one of the glyphs supported by the Rust renderer;
        # an unknown name would silently default to circle at render time.
        shape_val = resolved.get("shape")
        if shape_val is not None and isinstance(shape_val, str):
            validate_choice(f"mark_{mark_name}", "shape", shape_val, _VALID_POINT_SHAPES)
        # Validate literal fill/stroke color strings at construction time via
        # ferrum's single Rust color parser (ferrum.color.to_hex) so bad
        # colors fail immediately rather than silently at render time. The
        # ``color=`` alias is already folded into ``fill``/``stroke`` above,
        # so checking those two canonical keys covers all three user-facing
        # spellings. Paint sentinels (see ``_is_paint_sentinel``) are
        # short-circuited ahead of the parser — both raise in to_hex by
        # design and are not colors. The stored value stays the user's
        # original string; to_hex is consulted only for validation.
        for paint_key in ("fill", "stroke"):
            paint_val = resolved.get(paint_key)
            if isinstance(paint_val, str) and not _is_paint_sentinel(paint_val):
                _validate_literal_color(mark_name, paint_key, paint_val)
        self._kwargs = resolved

    @property
    def kwargs(self) -> dict[str, Any]:
        """Read-only view of the resolved (alias-expanded) kwargs dict."""
        return dict(self._kwargs)

    def to_mark_kwargs_dict(self) -> dict:
        """Return the subset of stored kwargs that map to ``MarkKwargsSpec`` fields.

        Statistical mark kwargs — the "forwarded to the transform" group in
        *Mark style kwargs* in the marks & encodings guide
        (``docs/site/guide/marks-encodings.md``), e.g.
        ``bandwidth``, ``method``, ``ci`` — are not included here; they are
        consumed directly by the desugar functions (``desugar_density``,
        ``desugar_smooth``, …) which build the transform objects before
        this dict is ever inspected.

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

    def zero_anchor(self) -> bool:
        """Return the effective value of the ``zero=`` parameter (default ``True``).

        Used by ``Chart._set_mark`` to store the zero-anchor preference on the
        chart so ``_build_encoding_specs`` can suppress the ``scale.zero=True``
        injection when the caller passes ``zero=False``.

        Returns
        -------
        bool
            ``False`` when ``zero=False`` was explicitly passed; ``True`` otherwise.

        Examples
        --------
        >>> MarkBase("bar", zero=False).zero_anchor()
        False
        >>> MarkBase("bar").zero_anchor()
        True
        """
        return bool(self._kwargs.get("zero", True))

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
