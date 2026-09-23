"""MarkBase — kwarg validation + storage for mark style overrides.

Phase 8a: only constant overrides are supported (e.g. mark_point(size=100)).
Encoding-driven overrides come through .encode(size=Size("col")).
"""

from __future__ import annotations

import math
from typing import Any

from ferrum._validate import is_none_color_sentinel, validate_choice, validate_color_value

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


def _stroke_dash_error(context: str, value: Any) -> str:
    """Build the refusal message for a *value* :func:`_normalize_stroke_dash` rejects."""
    return (
        f"{context}={value!r} is not a valid stroke_dash value. Expected a "
        "numeric list (e.g. [4.0, 2.0]), a comma-separated numeric string "
        f'(e.g. "4,2"), or a named linetype (one of {sorted(_LINETYPE_MAP)}).'
    )


def _check_dash_lengths(lengths: Any, *, context: str, original: Any) -> None:
    """Validate every dash length in *lengths* (#128 cycle-3): each must be
    a real number (not ``bool``, not a numeric-looking string), finite, and
    non-negative. Raises ``ValueError`` naming *context* and *original* --
    the value the caller actually passed, not the intermediate parsed form
    -- via :func:`_stroke_dash_error`, so a malformed length reads the same
    refusal as any other malformed ``stroke_dash=`` value.
    """
    for item in lengths:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(_stroke_dash_error(context, original))
        fv = float(item)
        if not math.isfinite(fv) or fv < 0:
            raise ValueError(_stroke_dash_error(context, original))


def _normalize_stroke_dash(value: Any, *, context: str) -> Any:
    """Normalize a user-supplied ``stroke_dash=`` value to its wire form (#124).

    Shared by :class:`MarkBase` (the mark-kwarg path, including the
    ``linetype=``/``line_type=`` aliases), ``annotate_hline``/
    ``annotate_vline`` (``ferrum/annotations.py``, which route
    ``stroke_dash=`` straight into the mark and the ``AnnotationLine``
    primitive with no ``linetype=`` alias of their own), ``fm.SelectionMark``
    (``ferrum/selection.py``, whose ``stroke_dash=`` used to pass through
    completely unvalidated), and the conditional-literal ``stroke_dash``
    handler (``ferrum/selection.py``'s ``_literal_stroke_dash``, #128). One
    normalization means the four surfaces cannot each accept a different
    form set — before this helper, named linetypes ("dashed", ...) worked
    only via the ``linetype=``/``line_type=`` aliases on ``MarkBase``, were
    unreachable by any spelling on the annotation constructors, and
    "wiggly" silently constructed a ``SelectionMark``. This reverses
    `f8148bdc` (batch A T9, 2026-09-02), which deliberately restricted the
    canonical ``stroke_dash=`` kwarg to the comma-split form; `#124`, filed
    the same day by that batch's final design review, ruled the restriction
    should not have shipped — canonical ``stroke_dash=`` is the superset,
    not a subset of ``linetype=``.

    Accepts, in order:

    - ``None`` — passes through unchanged (solid / mark default).
    - a named linetype string: one of ``sorted(_LINETYPE_MAP)`` (``"solid"``,
      ``"dashed"``, ``"dotted"``, ``"dashdot"``, ``"longdash"``).
    - a comma-separated numeric string, spaces around each number tolerated
      (e.g. ``"4,2"``, ``"4, 2"``); ``""`` is the empty-dash spelling and
      normalizes to ``[]`` (solid), matching an already-empty list/tuple.
    - a bare number, or a list/tuple of numbers — passed through unchanged
      (same container type, same element types: ``[4, 2]`` stays a list of
      ints, ``(4.0, 2.0)`` stays a tuple of floats). Named-linetype and
      comma-split resolution are string-only features, so these forms were
      never re-shaped by this helper.
    - ``bool`` (bare, or as a list/tuple element) is refused (#128
      cycle-3): it is an ``int`` subclass, but a caller writing
      ``stroke_dash=True`` never meant a length of ``1``.

    Every dash length -- however it was spelled (a comma-string element, a
    bare number, or a list/tuple element) -- must be finite and
    non-negative (#128 cycle-3, closing the gap where only the conditional
    literal's list/tuple form validated lengths: ``"-4,2"``, ``"nan,2"``,
    a bare ``-3`` or ``nan``, and ``[-4, 2]`` all now refuse identically on
    every surface this helper serves).

    Anything that is a ``str`` but neither a recognized linetype name nor a
    valid comma-split numeric string, or that has a negative/non-finite/
    non-numeric length by any spelling, raises ``ValueError`` naming
    *context* (the public entry + kwarg the caller invoked) and the
    offending value — junk refuses on every surface, unchanged from before
    this helper.

    Parameters
    ----------
    value:
        The raw ``stroke_dash=`` value as the caller received it.
    context:
        Pre-composed ``"<entry>: <kwarg>"`` label, without the value — this
        function appends ``={value!r}`` itself, matching
        :func:`ferrum._validate.validate_color_value`'s ``label`` convention.

    Raises
    ------
    ValueError
        If *value* is a ``str`` that is neither a recognized linetype name
        nor a valid comma-separated numeric string, or if any resolved
        length is ``bool``, non-numeric, negative, or non-finite.
    """
    if isinstance(value, str):
        if value in _LINETYPE_MAP:
            dash = list(_LINETYPE_MAP[value])
        else:
            try:
                dash = [float(x) for x in value.split(",") if x.strip()]
            except ValueError as exc:
                raise ValueError(_stroke_dash_error(context, value)) from exc
        _check_dash_lengths(dash, context=context, original=value)
        return dash
    if isinstance(value, bool):
        raise ValueError(_stroke_dash_error(context, value))
    if isinstance(value, (int, float)):
        _check_dash_lengths([value], context=context, original=value)
        return value
    if isinstance(value, (list, tuple)):
        _check_dash_lengths(value, context=context, original=value)
        return value
    return value


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

# Core style fields honored by every mark (Phase 8a: "covers all kwargs
# accepted by the 8 primitive mark_*() Python methods" —
# crates/ferrum-core/src/spec/mark_style.rs's own module doc). ``blend``
# (layer blend mode) joins this baseline: it is mark-agnostic, not owned by
# any one mark family.
_UNIVERSAL_MARK_KWARGS: frozenset[str] = frozenset(
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
        "blend",
    ]
)

# Per-primitive-mark additions, mirrored from MarkKwargsSpec's own
# section-header comments in crates/ferrum-core/src/spec/mark_style.rs (each
# field's doc comment names its owning mark(s)) plus ``orient``, which is
# Python-only (consumed as a coord-flip flag, never forwarded to Rust — see
# ``orient_coord_flip``) but still mark-specific (bar/tick).  Drift-pinned
# against the Rust section headers by
# ``tests/marks/test_mark_honored_kwargs.py``.
_PRIMITIVE_MARK_EXTRAS: dict[str, frozenset[str]] = {
    "point": frozenset({"filled", "shape"}),
    "line": frozenset({"interpolate", "stroke_cap", "stroke_join"}),
    "area": frozenset({"interpolate", "stroke_join", "line", "borders"}),
    "bar": frozenset({"orient", "zero"}),
    "rule": frozenset(),
    "text": frozenset({"limit"}),
    "tick": frozenset({"orient", "band_size"}),
    "rect": frozenset({"band_size"}),
    "segment": frozenset({"arrow", "arrow_size"}),
    "arc": frozenset(),
    "image": frozenset(),
    "geoshape": frozenset(),
    "label": frozenset({"leader_line"}),
}

#: Per-mark honored-kwarg vocabularies (#243), replacing the single flat
#: allowlist that every mark used to share regardless of whether it actually
#: honored a given field. Keyed by the 13 primitive mark names (the ones
#: ``MarkBase``/``Chart._set_mark`` validate directly); every composite,
#: statistical, and diagnostic mark name (``"boxplot"``, ``"density"``,
#: ``"roc"``, …) reached through :func:`mark_honors` falls back to
#: :data:`_UNIVERSAL_MARK_KWARGS` — as of this writing every one of those
#: marks' own ``**kwargs`` surface only ever exercises the universal set
#: (mark-specific behavior on composite marks is threaded through named
#: parameters instead, e.g. ``mark_ribbon(interpolate=...)``, never through
#: ``**mark_kwargs``), so the fallback is exact, not an approximation. A
#: future composite mark that wants to honor a mark-specific field must add
#: an explicit entry here rather than relying on the fallback.
MARK_HONORED_KWARGS: dict[str, frozenset[str]] = {
    mark: _UNIVERSAL_MARK_KWARGS | extras for mark, extras in _PRIMITIVE_MARK_EXTRAS.items()
}


def mark_honors(mark_name: str, kwarg: str) -> bool:
    """Return ``True`` if ``mark_name`` honors the (already alias-resolved) *kwarg*.

    Looks up :data:`MARK_HONORED_KWARGS`; a *mark_name* with no explicit
    entry (every non-primitive mark) falls back to
    :data:`_UNIVERSAL_MARK_KWARGS` — see :data:`MARK_HONORED_KWARGS`'s
    docstring for why that fallback is exact today.

    Examples
    --------
    >>> mark_honors("bar", "zero")
    True
    >>> mark_honors("bar", "arrow")
    False
    >>> mark_honors("boxplot", "opacity")
    True
    """
    return kwarg in MARK_HONORED_KWARGS.get(mark_name, _UNIVERSAL_MARK_KWARGS)


def _honored_set_for(mark_name: str) -> frozenset[str]:
    """The honored set :func:`mark_honors` actually consulted for *mark_name*."""
    return MARK_HONORED_KWARGS.get(mark_name, _UNIVERSAL_MARK_KWARGS)


def _mark_kwarg_error(mark_name: str, key: str) -> str:
    """Build the refusal message for one kwarg *key* not honored by *mark_name*.

    Two phrasings, both naming the honored set (decision: "naming the mark
    and its honored set"):

    - *key* names no style field on any mark at all (a genuine typo, e.g.
      ``mark_point(squiggly=True)``): "unknown keyword argument" -- the
      pre-#243 wording, kept so existing typo-detection tests/messages are
      unaffected.
    - *key* is a real field, just not one *mark_name* honors (e.g.
      ``mark_bar(arrow=True)`` -- ``arrow`` is real, but only
      ``mark_segment`` honors it): "is not honored by", the new #243
      per-mark distinction.
    """
    honored = sorted(_honored_set_for(mark_name))
    if key not in _VALID_MARK_KWARGS:
        return f"mark_{mark_name}: unknown keyword argument {key!r}. Honored: {honored}"
    return f"mark_{mark_name}: keyword argument {key!r} is not honored by mark_{mark_name}. Honored: {honored}"


def _mark_kwargs_error_plural(mark_name: str, keys: list[str]) -> str:
    """Plural counterpart of :func:`_mark_kwarg_error` for a batch of *keys*
    (used by ``validate_user_mark_kwargs``, which reports every offending
    key from one call rather than failing on the first)."""
    honored = sorted(_honored_set_for(mark_name))
    if all(key not in _VALID_MARK_KWARGS for key in keys):
        return f"mark_{mark_name}: unknown keyword argument(s) {keys!r}. Honored: {honored}"
    return (
        f"mark_{mark_name}: keyword argument(s) {keys!r} not honored by "
        f"mark_{mark_name}. Honored: {honored}"
    )


#: The full known mark-kwarg vocabulary — the union honored by at least one
#: primitive mark. Used by the ``Chart.override(mark_<leaf>=...)`` registry
#: (``_override_apply.py``) to decide whether a ``mark_`` override *path*
#: exists at all; that registry is built once at import time with no
#: per-chart mark context, so it validates against this union rather than a
#: specific mark's narrower honored set (see ``_override_apply.py``'s module
#: docstring for the tracked gap).  Kept under the historical name so the two
#: existing drift guards (``tests/test_override.py``,
#: ``tests/test_mark_style_kwargs_doc.py``) that import it directly keep
#: working unchanged.
_VALID_MARK_KWARGS: frozenset[str] = frozenset().union(*MARK_HONORED_KWARGS.values())


def _validate_literal_color(mark_name: str, key: str, value: str) -> None:
    """Raise ``ValueError`` if *value* is not a color ``ferrum.color.to_hex`` accepts.

    A thin, label-composing wrapper over
    :func:`ferrum._validate.validate_color_value` (close-remediation, batch
    F) — the shared gate ``configure._validate_config_color`` also composes
    over, collapsing what were two independently hand-copied ``try:
    to_hex(...) except ValueError: raise ValueError(...)`` bodies into one.
    Not a second color vocabulary — the raised message wraps to_hex's own
    accepted-forms text unchanged. Callers must short-circuit the non-color
    sentinels (``"none"``, ``"transparent"``, ``"theme:label"``) before
    calling this function — each raises here by design (``"none"``/
    ``"transparent"`` with to_hex's sentinel-aware clearing-paint message,
    ``"theme:label"`` with the generic accepted-forms text), matching
    to_hex's behavior. This mark-kwarg call site already restricts *value*
    to ``str`` before calling (see its own ``isinstance`` guard), so it
    leaves ``validate_color_value``'s ``require_str`` at its default
    ``False`` -- unlike ``configure.py``'s config-surface fields, which
    request it to refuse a structurally-valid-to-``to_hex`` non-``str``
    (e.g. an RGB(A) tuple) at construction.

    The ``ferrum.color`` import is lazy (inside this function, not at
    module load time) to avoid a module-init-order dependency: ``ferrum.marks.base``
    cannot be reached without ``ferrum._core`` already having loaded
    (``ferrum/__init__.py`` imports it unconditionally before anything
    else), so ``ferrum.color`` — whose only import is ``ferrum._core`` — is
    always importable by the time this runs.
    """
    from ferrum.color import to_hex

    validate_color_value(f"mark_{mark_name}: {key}", value, to_hex)


class MarkBase:
    """Validate and store mark-level keyword arguments for primitive marks.

    Used by ``mark_*()`` builder functions to validate kwargs before
    serializing them into ``ChartSpec.mark_style``. Raises ``TypeError``
    on the first keyword argument *mark_name* does not honor — whether
    because it names no known style field at all, or because it names a
    real field owned by a different mark family — so typos and
    doesn't-apply-here kwargs both surface immediately rather than
    silently disappearing into the renderer (#243).

    Parameters
    ----------
    mark_name : str
        Name of the mark (e.g. ``"point"``, ``"bar"``). Used in error
        messages only.
    **kwargs : Any
        Style overrides forwarded to the renderer (e.g. ``size=100``,
        ``fill="red"``). Every key must be honored by *mark_name* per
        :func:`mark_honors` (aliases resolve first).

    Raises
    ------
    TypeError
        If any keyword argument is not honored by *mark_name*.

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
            if canonical == "stroke_dash":
                # Named forms ("dashed", "dotted", ...), the comma-split
                # form, and list/tuple passthrough are all resolved by the
                # one shared helper (#124) -- reached identically whether
                # the user wrote the canonical ``stroke_dash=`` kwarg or
                # either alias, so a named string works either way.
                v = _normalize_stroke_dash(v, context=f"mark_{mark_name}: {k}")
            resolved[canonical] = v
        for k in resolved:
            if not mark_honors(mark_name, k):
                raise TypeError(_mark_kwarg_error(mark_name, k))
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
            # Batch H T1/T2: arrow / arrow_size (segment)
            "arrow",
            "arrow_size",
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
