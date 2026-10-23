"""Channel-alias rules — map convenience channels onto their render targets.

The alias graph is expressed as DATA (the ``_ENCODING_ALIASES`` table, declared
in ``_channel_policy`` beside the other per-channel-name routing data and
re-exported here) driven by one generic loop, so the policy is visible in one
place rather than spread across inline branches. `apply_channel_aliases`
operates on shallow copies of the encoding and mark-kwargs dicts from
``Chart.to_spec`` — it does not mutate the chart's internal state.

Ordering is load-bearing: rules are applied top-to-bottom, and an earlier alias
that already populated the target wins over a later one (the later one then sees
``color`` present and follows its conflict policy).
"""

from __future__ import annotations

from collections.abc import Iterable

from ferrum._layer import layer_draws, marks_drawing

# The alias TABLE itself lives in the sibling routing module
# (``_channel_policy``, the zero-import leaf that owns every other
# per-channel-name disposition) and is re-exported here, where the loop that
# applies it lives, so existing importers of
# ``ferrum.encoding._aliases._ENCODING_ALIASES`` are unaffected. It moved
# 2026-09-11 (batch-H close re-review) so the conditional vocabulary could
# DERIVE its own alias map from this one table instead of hand-copying the
# source->target pairs: deriving in the other direction would have made the
# leaf module import this one, breaking the #103 leaf invariant that
# ``tests/test_finding_p1.py`` pins. Data sits with the other routing data;
# behavior stays here.
from ferrum.encoding._channel_policy import (  # noqa: F401  (re-exported)
    _ENCODING_ALIASES,
    _EncodingAlias,
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


def resolve_channel_alias(enc: dict, channel: str) -> object | None:
    """Return the value *channel* would resolve to under the alias rules, read-only.

    Side-effect-free by construction: does not mutate *enc*, does not warn,
    and does not consult ``detail`` at all. Reads ``_ENCODING_ALIASES``
    directly (the same table :func:`apply_channel_aliases` drives) rather
    than calling that function, because callers that only need to know
    "what would this encoding's *channel* be", before ``Chart.to_spec()``'s
    later, warning-emitting alias pass ever runs, must never trigger the
    ``stroke`` warning as a side effect of asking the question.

    Returns ``None`` when neither *channel* nor any alias source targeting
    it (``fill``/``stroke`` for ``color``) is present in *enc*.
    """
    source = alias_source(enc, channel)
    return enc[source] if source is not None else None


def alias_source(enc: dict, channel: str) -> str | None:
    """Return the name of the binding *channel* resolves to under the alias rules, read-only.

    *channel* itself when *enc* binds it, else the first alias source
    targeting it that *enc* binds, in ``_ENCODING_ALIASES`` order (``fill``
    before ``stroke`` for ``color``): the binding :func:`apply_channel_aliases`
    keeps, and :func:`resolve_channel_alias` returns. ``None`` when none is
    bound.
    """
    if channel in enc:
        return channel
    for rule in _ENCODING_ALIASES:
        if rule.target == channel and rule.source in enc:
            return rule.source
    return None


def resolved_channel(name: str) -> str:
    """Return the channel *name* draws as: an alias source's target, else *name* itself.

    ``fill`` and ``stroke`` draw as ``color`` (``_ENCODING_ALIASES``), so a
    layer's own ``fill`` and an inherited ``color`` bind one channel and the
    layer's own binding wins.
    """
    for rule in _ENCODING_ALIASES:
        if rule.source == name:
            return rule.target
    return name


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
                #
                # Keys on the real kwarg name ("stroke"), not a synthetic
                # "stroke_dropped_by_color" stand-in (#240, 2026-09-11 wave-1
                # hardening: that was a message baked into the key, working
                # around the old (channel, kwarg)-only identity). warn_once
                # now keys on the resolved message too, so the fixed literal
                # message below still dedups across repeated calls exactly
                # as before.
                warn_once(
                    "encoding",
                    "stroke",
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

    Only a mark whose row of Rust's style table draws ``detail``
    (:func:`ferrum._layer.layer_draws`) gets it written into *mk*; any other
    mark's style is left without the field the renderer would refuse.
    Whether such a detail did nothing is decided once the chart's
    statistics are lowered (:func:`warn_ignored_details`). No-op when
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
    detail_ch = enc.get("detail")
    if detail_ch is None:
        return mk
    field = _channel_field(detail_ch)
    if field is None:
        return mk
    if layer_draws(mark, "detail"):
        mk.setdefault("detail", field)
    return mk


def _prose_list(items: list[str]) -> str:
    """Join *items* as English prose: "", "a", "a and b", or "a, b, and c"."""
    if not items:
        return ""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _warn_detail_ignored(mark: str | None) -> None:
    from ferrum._warn import warn_once

    drawers = [f"mark_{m}" for m in marks_drawing("detail")]
    if not drawers:
        clause = "no mark reads"
    elif len(drawers) == 1:
        clause = f"only {drawers[0]} reads"
    else:
        clause = f"only {_prose_list(drawers)} read"
    warn_once(
        "encoding",
        "detail",
        message=(
            f"encode(detail=...) is accepted but ignored by mark_{mark or 'point'}; "
            f"{clause} the detail "
            "channel for per-group splitting."
        ),
    )


def layered_chart_detail(detail_ch: object, layer_marks: list) -> list[object | None]:
    """Return, per layer, the chart-level ``detail`` that layer inherits, or ``None``.

    A layered chart's chart-level encoding is the parent of every layer, which
    inherits each channel it does not bind (``color`` is inherited the same
    way, by Rust). ``detail`` is
    mark style on the wire, not an encoding Rust inherits, so Python hands it
    to each layer: the returned list, aligned with *layer_marks*, holds the
    channel for every layer whose mark draws ``detail`` (Rust's style table)
    and ``None`` for the rest. A
    layer's own ``detail`` still wins (``_build_layers_list`` only fills an
    absent one). Whether a detail no layer draws did nothing is decided once
    the chart's statistics are lowered (:func:`warn_ignored_details`).
    """
    if _channel_field(detail_ch) is None:
        return [None] * len(layer_marks)
    return [detail_ch if layer_draws(mark, "detail") else None for mark in layer_marks]


def warn_ignored_details(
    details: Iterable[tuple[str, str | None]], drawn: set[str], grouped: set[str]
) -> None:
    """Warn once per detail field that no mark draws and no statistic groups by.

    *details* holds each ``(field, mark)`` a chart binds on ``detail``: its
    own, each layer's own, and each layer's grouping-only member, with the
    mark that binds it. *drawn* is the fields a mark's ``mark_style.detail``
    names, and *grouped* every field a step of the chart's pipeline groups
    by. Detail is the grouping channel: a field a line splits by, or a
    statistic computed one per value of, did what it was bound for, so only
    a field that did neither is reported as ignored.
    """
    reported: set[str] = set()
    for field, mark in details:
        if field in drawn or field in grouped or field in reported:
            continue
        reported.add(field)
        _warn_detail_ignored(mark)
