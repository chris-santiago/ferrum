"""Channel-alias rules — map convenience channels onto their render targets.

The alias graph is expressed as DATA (the ``_ENCODING_ALIASES`` table) driven by
one generic loop, so the policy is visible in one place rather than spread across
inline branches. `apply_channel_aliases` operates on shallow copies of the
encoding and mark-kwargs dicts from ``Chart.to_spec`` — it does not mutate the
chart's internal state.

Ordering is load-bearing: rules are applied top-to-bottom, and an earlier alias
that already populated the target wins over a later one (the later one then sees
``color`` present and follows its conflict policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Marks whose Rust builder reads mark_style.group.detail for per-group
# splitting (crates/ferrum-core/src/render/draw.rs, area.rs, polygon.rs;
# channels.rs's build_color_detail_groups). Every other mark ignores the
# field entirely, so `detail` on those marks is warn_once'd after aliasing.
_DETAIL_CONSUMING_MARKS = frozenset(("line", "area", "polygon"))


@dataclass(frozen=True)
class _EncodingAlias:
    """One ``source -> target`` aliasing rule within the encoding dict.

    Parameters
    ----------
    source :
        The convenience channel name read from the encoding dict.
    target :
        The canonical channel name the source maps onto.
    on_conflict :
        What to do when *target* is already present in the encoding:

        - ``"keep_target"`` — silently leave the existing target, dropping the
          source's mapping (no warning).
        - ``"warn_drop"`` — drop the source's mapping and emit a one-time
          warning (only when the source carries a concrete field).
    """

    source: str
    target: str
    on_conflict: Literal["keep_target", "warn_drop"]


# Order matters: fill is resolved before stroke, so when both are present and
# color is absent, fill wins the color target and stroke then sees color present
# and follows its warn_drop policy.
_ENCODING_ALIASES: tuple[_EncodingAlias, ...] = (
    _EncodingAlias(source="fill", target="color", on_conflict="keep_target"),
    _EncodingAlias(source="stroke", target="color", on_conflict="warn_drop"),
)


def _channel_field(ch: object) -> str | None:
    """Return the concrete field name bound to one encoding value, or ``None``.

    Encoding values come in two legitimate shapes: a plain string (the
    layered path -- ``ferrum.layer.Layer``'s own docstring example is
    ``encoding={"x": "x", "y": "y"}`` -- a string source *is* its own field
    name) or a ``ChannelBase`` instance (the chart-level path, always).
    ``Repeat`` placeholders (``RepeatChart`` template channels, resolved
    later by ``.expand()``) and any other field-less channel are not
    concrete and return ``None``.

    The single field-extraction primitive shared by every consumer that
    needs to know "does this encoding value carry a real column name" --
    :func:`apply_channel_aliases`'s conflict-warn branch,
    :func:`alias_detail_channel`, and ``_spec_build``'s bucket-partition
    safety net (:func:`ferrum._spec_build._warn_unbucketed_channels`) -- so
    the string-vs-``ChannelBase`` handling cannot drift between them again
    (2026-08-27 P1 remediation, quality-review finding: it already had,
    once, between the chart-level and layered safety nets).
    """
    from ferrum.repeat import _RepeatPlaceholder

    if isinstance(ch, str):
        return ch
    field = getattr(ch, "field", None)
    if field is None or isinstance(field, _RepeatPlaceholder):
        return None
    return field


def resolve_color_alias(enc: dict) -> object | None:
    """Return the value ``color`` would resolve to under the alias rules, read-only.

    Side-effect-free by construction: does not mutate *enc*, does not warn,
    and does not consult ``detail`` at all. Reads ``_ENCODING_ALIASES``
    directly (the same table :func:`apply_channel_aliases` drives) rather
    than calling that function, because callers that only need to know
    "what would this encoding's color be" -- ``composition._promote_layer_color``
    resolving a layer's chart-level color-scale promotion at ``Chart + Chart``
    merge time, before ``Chart.to_spec()``'s later, warning-emitting alias
    pass ever runs -- must never trigger the ``stroke_dropped_by_color``
    warning as a side effect of asking the question (2026-08-27 P1
    remediation, quality-review finding: the prior fix instead built a
    ``detail``-stripped throwaway copy of the encoding to suppress
    :func:`alias_detail_channel`'s warning, a comment-enforced invariant
    that this read-only resolver makes structurally impossible to violate).

    Returns ``None`` when neither ``color``, ``fill``, nor ``stroke`` is
    present in *enc*.
    """
    if "color" in enc:
        return enc["color"]
    for rule in _ENCODING_ALIASES:
        if rule.target == "color" and rule.source in enc:
            return enc[rule.source]
    return None


def apply_channel_aliases(enc: dict, mk: dict, mark: str | None = None) -> tuple[dict, dict]:
    """Apply channel-alias rules, mapping convenience channels to their targets.

    Operates on shallow copies of the encoding and mark-kwargs dicts from
    ``to_spec()`` — does not mutate the chart's internal state.

    Encoding-dict aliases (in `_ENCODING_ALIASES`, applied in order):

    1. ``fill`` -> ``color`` when ``color`` is not already present.
    2. ``stroke`` -> ``color`` when ``color`` is not already present;
       when ``color`` IS present, the stroke encoding is dropped with a
       one-time warning.

    Separately, ``detail`` -> ``mk["detail"]`` via :func:`alias_detail_channel`
    (always, regardless of other channels — it targets the mark-style kwargs,
    not the encoding dict). *mark* is the resolved chart's final mark name,
    used to warn_once when the target mark's Rust builder does not consume
    ``mark_style.group.detail``.

    Encoding values may be either a ``ChannelBase`` instance (chart-level
    path, always) or a plain string (the layered path's legitimate shorthand
    -- see :func:`_channel_field`); both shapes are handled identically here.

    Note: ``fill_opacity`` is no longer aliased to ``opacity``. It is a
    first-class renderer-honored channel that emits a per-element SVG
    ``fill-opacity`` attribute, separate from ``opacity`` (which bakes into the
    fill RGBA alpha).

    Returns the (possibly-modified) ``(enc, mk)`` pair.
    """
    from ferrum._warn import warn_once

    for rule in _ENCODING_ALIASES:
        if rule.source not in enc:
            continue
        if rule.target not in enc:
            enc[rule.target] = enc[rule.source]
            continue
        # Target already present — apply the conflict policy.
        if rule.on_conflict == "warn_drop":
            # A plain-string source (the layered path's shorthand) is just as
            # concrete a field as a ChannelBase one and must still warn --
            # `_channel_field` handles both shapes (2026-08-27 P1
            # remediation, quality-review finding: `src_ch.field` used to be
            # dereferenced unconditionally here, which raised AttributeError
            # for a string `stroke`/`fill` on a layer, crashing a chart that
            # rendered before this module started routing layer encodings
            # through this function).
            if _channel_field(enc[rule.source]) is not None:
                # Can't map to a scale -- the source encoding produces no visual
                # effect when the target is already mapped.  Warn once so callers
                # know the channel was dropped.
                warn_once(
                    "encoding",
                    "stroke_dropped_by_color",
                    "encode(stroke=...) is ignored when color= is also encoded; "
                    "stroke is aliased to color only when color is absent.",
                )
        # "keep_target" drops the source silently; both policies leave the
        # existing target untouched.

    mk = alias_detail_channel(enc, mk, mark)
    return enc, mk


def alias_detail_channel(enc: dict, mk: dict, mark: str | None) -> dict:
    """Alias ``detail`` (if present in *enc*) into mark-style kwargs *mk*.

    Shared by the chart-level pass (:func:`apply_channel_aliases`) and the
    per-layer pass in ``SpecBuildMixin._build_layers_list`` so a layer's own
    ``detail`` encoding gets the identical ``mark_style`` routing and
    mark-conditional ``warn_once`` as the chart-level channel — before this
    (2026-08-27 P1 remediation) a layer's own ``detail`` was dropped
    entirely because only the chart-level alias pass ran.

    Warns once when *mark* is not one of the Rust builders that read
    ``mark_style.group.detail`` (``_DETAIL_CONSUMING_MARKS``); no-op when
    ``detail`` is absent or carries no concrete field -- string-valued
    (``Layer(encoding={"detail": "g"})``) and ``ChannelBase``-valued
    ``detail`` are both concrete fields, handled identically via
    :func:`_channel_field` (2026-08-27 P1 remediation, quality-review
    finding: ``getattr(detail_ch, "field", None)`` used to return ``None``
    for a string, silently dropping a per-layer string ``detail`` with no
    routing and no warning -- the exact defect class this whole batch
    exists to eliminate). Does not mutate *enc*; returns the
    (possibly-modified) *mk*.
    """
    from ferrum._warn import warn_once

    detail_ch = enc.get("detail")
    if detail_ch is None:
        return mk
    field = _channel_field(detail_ch)
    if field is None:
        return mk
    mk.setdefault("detail", field)
    if mark not in _DETAIL_CONSUMING_MARKS:
        warn_once(
            "encoding",
            "detail",
            message=(
                f"encode(detail=...) is accepted but ignored by mark_{mark or 'point'}; "
                "only mark_line, mark_area, and mark_polygon read the detail "
                "channel for per-group splitting."
            ),
        )
    return mk
