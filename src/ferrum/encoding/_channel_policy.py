"""Channel-disposition policy — the five-bucket taxonomy over every encoding channel.

Every channel in :func:`ferrum.encoding._channel_class_map` falls into exactly
one of the five buckets below (test-enforced partition, see
``tests/test_finding_p1.py``). ``Chart.encode()`` is a total function over
that map: each channel either renders, aliases, or ``warn_once``'s — never a
silent drop. See ``ferrum-spec.md`` §3.2 (2026-08-27 dated note) for the
user-facing contract this partition implements.

Relocated 2026-08-27 (#103) from ``ferrum.chart`` to this near-leaf module
(the ``_honored.py`` sibling) so encoding-vocabulary knowledge lives in
``ferrum.encoding`` rather than the top-level ``Chart`` value class;
``chart.py``/``_spec_build.py`` are consumers, not the source of truth. This
is a pure relocation — bucket membership, tuple order, and every provenance
comment are unchanged.

The same per-channel-name axis also owns the conditional vocabulary
(``_CONDITIONAL_CHANNELS`` / :func:`conditional_wire_channel`): "can a
selection conditional target this channel name, and which wire channel does
it land on" is a routing question, not a kwarg-scope one.

It also owns the ``_ENCODING_ALIASES`` table as of 2026-09-11 (batch-H close
re-review), relocated from ``_aliases`` (which re-exports it and still owns
the loop that applies it) so the conditional vocabulary can *derive* its own
alias map from the one table instead of hand-copying the source→target
pairs. Deriving the other way would have made this module import
``_aliases`` and broken the leaf invariant above, which
``tests/test_finding_p1.py`` pins: routing DATA belongs here, alias
BEHAVIOR stays there.

Role: this module answers "what does ``Chart.encode()`` *do* with a given
channel *name*" (render it into its own ``EncodingSpec``, alias it to
another channel, warn-and-drop it, remap it under polar, route it
through faceting, or carry a conditional) — a per-channel ROUTING
disposition. See
``ferrum.encoding._honored`` for the per-channel-type KWARG scope (a
different "honored" question); the naming overlap between the two is
deliberate, not an accidental collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


# ALIAS table. Order matters: fill is resolved before stroke, so when both are
# present and color is absent, fill wins the color target and stroke then sees
# color present and follows its warn_drop policy.
#
# Declared here rather than in ``_aliases`` (which re-exports it, and owns the
# loop that applies it) as of 2026-09-11, batch-H close re-review: the
# conditional vocabulary below derives its own alias map from this table, and
# deriving in the other direction would have made this leaf module import
# ``_aliases``, breaking the #103 leaf invariant.
_ENCODING_ALIASES: tuple[_EncodingAlias, ...] = (
    _EncodingAlias(source="fill", target="color", on_conflict="keep_target"),
    _EncodingAlias(source="stroke", target="color", on_conflict="warn_drop"),
)

# RENDERER_HONORED: channels honored by the renderer at to_spec() time —
# each becomes its own EncodingSpec entry, on both the chart-level
# (_build_encoding_specs) and layered (_build_layers_list) paths.
#
# This is a tuple, not a frozenset: iteration order drives EncodingSpec
# build order and auto-tooltip field order (_spec_build._auto_tooltip_fields
# iterates it directly) — order is load-bearing, preserve it exactly.
_RENDERER_HONORED_CHANNELS = (
    "x",
    "y",
    "x2",
    "y2",
    "color",
    "size",
    "shape",
    "opacity",
    "text",
    "tooltip",
    "href",
    "description",
    "url",
    # Per-element stroke/angle channels wired to SVG attributes (Task 10).
    "stroke_opacity",
    "stroke_width",
    "stroke_dash",
    "angle",
    # Per-element fill-opacity SVG attribute (distinct from opacity which bakes
    # into RGBA alpha on the fill color).
    "fill_opacity",
    # Promoted from a silent drop 2026-08-27 (P1 remediation): the channel
    # gets its own EncodingSpec and reaches the scene graph on both paths
    # (ChartSpec(key=...) -> scene_build::extract_keys -> MarkBatch.keys, in
    # both static and interactive scene JSON) -- that observable reach is
    # why this is bucketed RENDERER_HONORED rather than WARN, even though
    # NO renderer currently consumes MarkBatch.keys: static SVG is
    # byte-identical with and without key=, and the WASM runtime never
    # reads it (verified 2026-08-27, quality-review finding -- an earlier
    # version of this comment overclaimed "fully wired" to mean visually
    # rendered). A visual/identity consumer (e.g. transition-matching on
    # data updates) is a separate, un-scoped feature -- see the
    # archaeology doc's `Key` row for the tracked gap.
    "key",
)
# ALIAS: channels that redirect to another channel or to mark-style kwargs
# rather than becoming their own EncodingSpec. No warning for fill/stroke
# (encoding/_aliases.py: fill/stroke -> color, with the existing
# stroke-dropped-by-color warning on conflict). `detail` aliases to
# mark_style.detail on every mark, but only mark_line/mark_area/mark_polygon's
# Rust builders read it (render/draw.rs) -- on any other mark
# `alias_detail_channel` warn_once's, on both the chart-level and per-layer
# paths (2026-08-27 P1 remediation).
_ALIAS_CHANNELS = frozenset(("fill", "stroke", "detail"))
# WARN: channels accepted, warn_once'd, and absent from the resulting spec
# (never reach a `kw[axis] = EncodingSpec(...)` assignment nor any Rust
# `Encoding` field) via the `_build_encoding_specs` safety net in
# `_spec_build.py`. x_error*/y_error*: no explicit-error-column feature
# exists for mark_errorbar (it computes its own extents from the aggregated
# data; logged P1 follow-up). tooltip_field: documented as "not used as a
# top-level encoding channel" (only valid inside Tooltip(*fields)).
_WARN_CHANNELS = frozenset(("x_error", "y_error", "x_error2", "y_error2", "tooltip_field"))
# POLAR: theta/radius/theta2/radius2 remap to x/y (untouched) when CoordPolar
# is set -- _resolve_polar_remapping pops them from `enc` before the safety
# net below ever sees them. Without CoordPolar they are never remapped, stay
# in `enc`, and fall through to the safety net's warn_once (dropped the
# unconditional whitelist 2026-08-27 -- previously silent regardless of
# coord).
_POLAR_CHANNELS = frozenset(("theta", "radius", "theta2", "radius2"))
# FACET: facet/facet_row/facet_col have a separate code path through
# resolved._facet (encode() synthesizes it) — no silent-drop, no warn.
_FACET_CHANNELS = frozenset(("facet", "facet_row", "facet_col"))

# CONDITIONAL: the channel names a selection-driven conditional
# (``sel.when(...).otherwise(...)``) can target. This is the Python-side
# source of truth the H9 ``condition=`` translation, the appearance
# channels' ``_honored_kwargs``, and ``ferrum-spec.md`` (§3.2's dated
# conditional-channel note, cross-referenced from §3.10) all cite, so
# the three stop being independent lists (2026-09-11, batch-H close
# remediation: they had already drifted — ``Fill``/``Stroke`` documented
# ``condition`` as honored while the stamp died in serde).
#
# Membership is EXACTLY the ``ChannelName`` variants that have an
# ``EncodingValue`` counterpart (``crates/ferrum-scene/src/selection.rs``: the
# ``EncodingValue::channel()`` mapping, which is what the WASM runtime
# actually dispatches on -- ``ferrum-wasm/src/conditional.rs``'s WASM-05
# note: "the runtime apply path no longer dispatches on [channel]; value is
# the source of truth"). No exceptions in either direction:
# ``tests/test_condition_translation.py::test_conditional_channels_mirror_rust_encoding_values``
# parses that Rust enum and asserts set equality, so the "mirrored from Rust"
# wording is checked rather than aspirational, and a wire channel with no
# value variant is a refusal (below) rather than a silently inert accept.
_CONDITIONAL_CHANNELS = frozenset(
    (
        "color",
        "size",
        "opacity",
        "stroke_width",
        "stroke_opacity",
        "stroke_dash",
        "fill_opacity",
        "angle",
    )
)

# Channel names a user could reasonably declare a conditional on that the
# wire CANNOT express, mapped to the reason the refusal states. One home for
# every such refusal -- an alias source (``stroke``) and a bare
# ``ChannelName`` variant with no value counterpart (``shape``, ``x``, ``y``)
# are the same defect from the caller's side, so they get one mechanism and
# one message shape rather than a set per kind.
#
# Every entry is a REFUSAL, never a silent accept: an accepted-but-inert
# conditional is the silent-drop shape this batch exists to remove, and an
# accepted-but-wrong one is worse. Each reason names the missing Rust value
# and, where one is planned, the follow-up that would let the channel join
# ``_CONDITIONAL_CHANNELS`` for real.
#
# The mirror test asserts these keys cover every ``ChannelName`` variant with
# no ``EncodingValue``, so the NEXT such gap is a test failure rather than a
# discovery (both ``shape`` and ``stroke`` were discoveries).
_CONDITIONAL_REFUSALS: dict[str, str] = {
    # 2026-09-11, batch-H close re-review. The only conditional value that
    # could carry a stroke colour is `EncodingValue::Color`, which the WASM
    # runtime applies as `*view.fill_color = color_to_linear(c, 1.0)`
    # (`apply_value_common`); `FillStrokeView` has no `stroke_color` member,
    # while `CircleInstance`/`RectInstance` each carry one the conditional
    # path never touches. `fill -> color` has no such problem (`Color` *is*
    # the fill), which is why fill stays an alias and stroke does not.
    "stroke": (
        "{name!r} is refused even though it aliases to 'color' for its base "
        "encoding: the only conditional colour value the renderer has repaints "
        "the mark's FILL, so a {name} conditional would change the wrong visual "
        "property. Follow-up: EncodingValue::StrokeColor."
    ),
    # 2026-09-11, adjudicated in the same re-review after the mirror test
    # surfaced it. `ChannelName::Shape` exists, so the spec serialized and
    # round-tripped -- but with no `EncodingValue::Shape` the conditional could
    # only carry a `Field` value, which `apply_value_common` returns `false`
    # for. Accepted, serialized, and a runtime no-op.
    "shape": (
        "{name!r} is a wire channel name but has no conditional value variant, so "
        "a {name} conditional would serialize and then apply as a no-op -- accepted "
        "and inert rather than working. Follow-up: EncodingValue::Shape."
    ),
    # x/y are in the wire's ChannelName enum for the OTHER consumer of that
    # enum -- a selection's `encodings=` projection list -- and have no
    # positional value variant. A numeric branch declared on `x` resolves to
    # an `opacity` value (`selection._resolve_encoding_value`) and would fade
    # marks instead of moving them. No follow-up is planned: conditionally
    # repositioning a mark is not a feature this pipeline has.
    "x": (
        "{name!r} is a selection-projection channel (a selection's encodings= list), "
        "not a conditional target: there is no positional conditional value, so a "
        "numeric branch would resolve as an opacity and fade marks instead of "
        "moving them."
    ),
}
_CONDITIONAL_REFUSALS["y"] = _CONDITIONAL_REFUSALS["x"]

# Convenience channels that carry a conditional onto their alias target,
# DERIVED from the one alias table (``_ENCODING_ALIASES`` above) rather than
# hand-copied from it, so a new alias rule cannot leave the two out of sync,
# minus any source the refusal registry rejects. The conditional stamp needs
# only the flat source -> target rename; that table's per-rule
# ``on_conflict`` policy governs the ENCODING dict and does not transfer
# (the conditional list has its own collision rule -- see
# ``chart._append_conditional``).
_CONDITIONAL_CHANNEL_ALIASES = {
    rule.source: rule.target
    for rule in _ENCODING_ALIASES
    if rule.target in _CONDITIONAL_CHANNELS and rule.source not in _CONDITIONAL_REFUSALS
}

# Human-readable supported set for the refusal message: the canonical names
# plus each alias annotated with the channel it lands on.
_CONDITIONAL_CHANNELS_DISPLAY = ", ".join(
    sorted(_CONDITIONAL_CHANNELS)
    + [f"{src} (-> {dst})" for src, dst in sorted(_CONDITIONAL_CHANNEL_ALIASES.items())]
)


def channel_honors_condition(name: str) -> bool:
    """Return whether encode key *name* honors a ``condition=`` kwarg.

    The single boolean the ``_honored_kwargs`` derivation
    (``ChannelBase.__init_subclass__``, ``ferrum.encoding.base``) is built
    on: *name* honors ``condition`` iff :func:`conditional_wire_channel`
    would resolve it rather than raise -- i.e. *name* is either a real
    conditional channel or an un-refused alias source. Exposed as its own
    predicate (rather than making every caller wrap
    ``conditional_wire_channel`` in a ``try/except``) so the honored-kwarg
    table can query the vocabulary without constructing and discarding an
    exception on every one of the ~30 channel classes at import time.
    """
    return name in _CONDITIONAL_CHANNELS or name in _CONDITIONAL_CHANNEL_ALIASES


def conditional_wire_channel(name: str) -> str:
    """Return the wire channel a conditional declared on encode key *name* targets.

    ``Chart.encode()`` calls this for both spellings of a conditional --
    ``encode(<channel>=<ConditionalSpec>)`` and
    ``encode(<channel>=<Channel>(field, condition=<ConditionalSpec>))`` --
    so one vocabulary governs both.

    Parameters
    ----------
    name :
        The ``encode()`` keyword the conditional was declared on, already
        snake_case (e.g. ``"opacity"``, ``"stroke_width"``, ``"fill"``).

    Returns
    -------
    str
        The channel name to stamp onto the ``ConditionalSpec``: *name*
        itself, or its alias target (``fill`` -> ``color``).

    Raises
    ------
    ValueError
        When *name* is not a channel a conditional can target. Raised at
        ``encode()`` time, naming the supported set -- the alternative is a
        raw serde ``unknown variant`` error thrown from deep inside
        ``to_spec()``, at render time, with no mention of what would work.
        A name in :data:`_CONDITIONAL_REFUSALS` also gets that entry's
        reason: those are the refusals a caller would otherwise read as
        arbitrary, because the channel exists and its sibling works.
    """
    aliased = _CONDITIONAL_CHANNEL_ALIASES.get(name)
    if aliased is not None:
        return aliased
    if name in _CONDITIONAL_CHANNELS:
        return name
    reason = _CONDITIONAL_REFUSALS.get(name)
    hint = f" {reason.format(name=name)}" if reason is not None else ""
    raise ValueError(
        f"condition= is not supported on the {name!r} encoding channel; a selection "
        f"conditional can target only: {_CONDITIONAL_CHANNELS_DISPLAY}. Declare it on "
        f"one of those channels instead (e.g. encode(color=<ConditionalSpec>)).{hint}"
    )


# The union of the three buckets that never reach `_spec_build`'s
# bucket-partition safety net: RENDERER_HONORED and ALIAS each have their
# own dedicated handling, and FACET is resolved entirely outside the
# encoding-warn path. WARN and POLAR-without-CoordPolar are deliberately
# NOT unioned in here -- their absence is what makes them fall through to
# the safety net's warn_once. Declared once, beside the buckets it is
# derived from, and consumed by the single enforcement point
# `ferrum._spec_build._warn_unbucketed_channels` (both the chart-level and
# per-layer safety nets call that one helper; the two used to hand-copy this
# union and had already drifted on string-channel handling before being
# unified — 2026-08-27 P1 remediation, quality-review finding).
_SPEC_KNOWN_CHANNELS = frozenset(_RENDERER_HONORED_CHANNELS) | _ALIAS_CHANNELS | _FACET_CHANNELS
