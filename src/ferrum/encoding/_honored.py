"""Honored-kwarg vocabulary — one source of truth for every channel's contract.

A channel's ``_honored_kwargs`` is the single authority for which keyword
arguments it honors: it drives both the ``warn_once`` guard in
`__init__` and the serializer in
`to_encoding_spec_dict` (which iterates this set). A kwarg is
serialized iff it is honored, so the guard and the serializer cannot drift.

The role constants below are *named* so the X-vs-X2 and Theta-vs-Theta2 splits
are self-documenting rather than hand-written ``frozenset([...])`` literals
scattered across the channel modules. They compose upward (each adds to the one
below it) so a reader can see at a glance what each role layers on.

Membership here was IDENTICAL to the per-class literals the channel modules
declared before this consolidation — that consolidation was a vocabulary
unification, not a contract change. **Dated note (2026-08-28, batch-A
appearance-resolution spec §4.3):** that invariant no longer holds for every
role. StrokeOpacity/StrokeDash moved from ``APPEARANCE_BASE`` to
``APPEARANCE_FULL`` as a deliberate contract change — see those constants'
docstrings below. **Further dated note (2026-09-01, batch-A T12 spec-review
addendum):** StrokeDash moved a second time, from ``APPEARANCE_FULL`` to
``APPEARANCE_SORT`` (mirroring Shape), so ``sort=`` reaches the wire instead
of being silently dropped while Rust's ``build_stroke_dash_scale`` already
reads it.

**``condition`` is no longer a role-set member at all (2026-09-11, batch-H
close cycle 3).** Between 2026-09-11's remediation and re-review, ``condition``
bounced between four channels moving onto ``APPEARANCE_BASE_CONDITION`` /
``APPEARANCE_CONDITION`` (StrokeWidth/Angle, FillOpacity/StrokeOpacity — the
wire genuinely honors their conditional) and two moving onto
``APPEARANCE_SCHEME_NO_CONDITION`` / ``APPEARANCE_SORT_NO_CONDITION``
(Stroke, Shape — the wire cannot express theirs). Six role sets carrying two
copies of the same yes/no answer the conditional vocabulary already owns
(``ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS`` /
``_CONDITIONAL_CHANNEL_ALIASES`` / ``_CONDITIONAL_REFUSALS``) is exactly the
hand-synced-copies shape this batch's alias-table work fixed elsewhere by
deriving instead of duplicating — and it is how Fill/Stroke and later
Shape/StrokeDash drifted from the vocabulary in the first place, one channel
at a time. ``ChannelBase.__init_subclass__`` (``ferrum.encoding.base``) now
derives the bit directly from ``_channel_policy.channel_honors_condition``,
keyed on each class's own ``_channel_name`` — every role constant below
declares its BASE contract only, with no ``condition`` member and no
``*_NO_CONDITION`` twin; the hook adds or removes the single bit after the
class body finishes executing. The two former ``_NO_CONDITION`` sets are
deleted: ``Shape`` and ``StrokeDash`` now share plain ``APPEARANCE_SORT``,
and ``Fill``/``Stroke`` now share plain ``APPEARANCE_SCHEME`` — the
derivation, not a second set, is what keeps Stroke/Shape without the bit and
Fill/StrokeDash with it. The two tables are now structurally unable to
disagree again.

**Legend bits are derived, not declared.** No role set below carries
``legend``, and none carries the ``title`` that labels a legend: both follow
the channel's ``_legend_surface`` (``ChannelBase.__init_subclass__`` via
``base._derive_honored``). A role set declares ``title`` only where it labels
something other than a legend (X/Y's axis title, a facet strip, a tooltip
field). The appearance roles compose from orthogonal atoms (``SCALE``,
``SORT``, ``SCHEME``), so Size (``own`` surface) and Angle (``none``)
differ in ``scale`` while each one's legend bits follow its surface.

The appearance-channel membership is pinned by
``tests/test_appearance_honored_kwargs.py``, which documents each intentional
change with a dated comment.

Role: this module answers "which *kwargs* does one channel *instance*
honor" (e.g. does ``X(...)`` accept ``sort=``; does ``Color(...)`` accept
``legend=``) — a per-channel-type KWARG scope. See
``ferrum.encoding._channel_policy`` for the per-channel-name ROUTING
disposition (a different "honored" question); the naming overlap between
the two is deliberate, not an accidental collision.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Positional roles
# ---------------------------------------------------------------------------

#: Primary positional channels (X, Y). The full set: data type, implicit
#: bin/aggregate transforms, scale + axis + legend customization, ordinal sort,
#: stacking, imputation, and tick-format. (= the old positional
#: ``_RENDERED_HONORED``.)
PRIMARY_POSITIONAL = frozenset(
    {
        "type",
        # bin/aggregate are consumed via to_implicit_transforms, not the
        # spec-dict loop; they are honored here so the warn guard stays silent.
        "bin",
        "aggregate",
        "scale",
        "title",
        # sort — honored by scale_resolve.rs ordinal domain builder.
        "sort",
        # axis dict — honored by prepare.rs AxisInput construction.
        "axis",
        # stack — honored by position.rs Stack strategy selection.
        "stack",
        # impute dict — honored by prepare.rs apply_impute.
        "impute",
        # format string + type — honored by prepare.rs apply_tick_format.
        # format_type is the canonical snake_case spelling; formatType is the
        # Vega-compat camelCase alias (D-FMT-1). Both serialize to the wire
        # key format_type via _emit_format_type in base.py.
        "format",
        "format_type",
        "formatType",
    }
)

#: Secondary-extent positional channels (X2, Y2, X/Y error pairs, Theta2,
#: Radius2). These carry no scale of their own — they reuse the primary
#: channel's scale — so only the data type is honored. The span ends (X2, Y2,
#: Theta2, Radius2) also compose ``AGGREGATE`` in their module: they draw the
#: value they are bound to. The error channels do not.
SECONDARY_EXTENT = frozenset({"type"})

#: Primary polar value channels (Theta, Radius): data type plus stacking
#: (wind-rose / coxcomb / pie accumulation). Both also compose ``AGGREGATE``
#: in their module.
POLAR_PRIMARY = frozenset({"type", "stack"})


# ---------------------------------------------------------------------------
# Appearance roles (compose upward; final sets match the prior _APPEARANCE_*)
#
# None of these carry ``condition`` — see the module docstring's 2026-09-11
# cycle-3 note. ``ChannelBase.__init_subclass__`` adds or removes that single
# bit per class, keyed on ``_channel_name`` against
# ``_channel_policy.channel_honors_condition``, so a channel's *documented*
# conditional support (StrokeWidth/Angle/FillOpacity/StrokeOpacity/Size/
# Opacity/StrokeDash/Color/Fill: yes; Shape/Stroke: no, the wire cannot
# express theirs) is never spelled out as a role-set member here.
# ---------------------------------------------------------------------------

#: Minimum appearance contract: the data type. (Angle, a per-row constant
#: with no scale resolution.)
APPEARANCE_BASE = frozenset({"type"})

#: Orthogonal capability atoms that compose onto ``APPEARANCE_BASE``.
SCALE = frozenset({"scale"})
SORT = frozenset({"sort"})
SCHEME = frozenset({"scheme"})

#: The ``aggregate=`` kwarg on a channel whose mark draws the aggregated
#: value (a span end, a polar angle or radius, a label's text, a color, a
#: size, an opacity, ...). Consumed through ``to_implicit_transforms``, never
#: serialized: the chart's (or each layer's) aggregate step computes the value
#: and the channel is re-pointed at its output column. Composed onto those
#: channels' own sets in their modules, so channels where an aggregate draws
#: nothing (Detail, Shape, the facet and error channels, ...) refuse it.
AGGREGATE = frozenset({"aggregate"})

#: The ``bin=`` kwarg on a color-family channel (Color, Fill, Stroke): the
#: channel draws its field through a binned color scale whose edges Rust cuts
#: from the rows it draws, so the rows are left unchanged
#: (``ChannelBase._color_bin_scale``).
BINNED_COLOR = frozenset({"bin"})

#: Adds an explicit scale override. (Size, Opacity, FillOpacity,
#: StrokeOpacity and StrokeWidth, which also draw a legend and so honor
#: ``legend``/``title`` through their ``own`` surface.)
APPEARANCE_FULL = APPEARANCE_BASE | SCALE

#: Adds an explicit domain-order sort. (StrokeDash and Shape both use this
#: set, as of 2026-09-01 / 2026-09-11 respectively — Shape mirrors
#: StrokeDash's sort role even though the derivation leaves Shape WITHOUT
#: ``condition``, since ``EncodingValue::Shape`` does not exist:
#: ``stroke_dash`` is a real conditional channel and gains the bit;
#: ``shape`` is refused by ``_channel_policy._CONDITIONAL_REFUSALS`` and does
#: not. One set, two derived outcomes — that split is the point of deriving
#: rather than hand-copying.)
APPEARANCE_SORT = APPEARANCE_FULL | SORT

#: Adds a named color scheme. (Fill and Stroke both use this set, same split
#: as ``APPEARANCE_SORT`` above: ``fill`` aliases to ``color`` and gains
#: ``condition``; ``stroke`` is refused — its only conditional colour value
#: (``EncodingValue::Color``) would repaint the mark's fill, not its stroke
#: — and does not.)
APPEARANCE_SCHEME = APPEARANCE_FULL | SCHEME

#: The Color channel: full appearance contract plus both sort and scheme.
#: (Gains ``condition`` via the derivation — ``color`` is the canonical
#: conditional channel.)
APPEARANCE_COLOR = APPEARANCE_FULL | SORT | SCHEME


# ---------------------------------------------------------------------------
# Text / grouping / facet roles
# ---------------------------------------------------------------------------

#: Bare channels that honor only the data type (Detail, Tooltip, Href,
#: Description, Key, Url). Distinct from SECONDARY_EXTENT in intent even though
#: both currently equal {"type"}.
BARE = frozenset({"type"})

#: Text-content channels with number/date formatting (Text).
#:
#: ``format_type`` is the canonical snake_case spelling (D-FMT-1); ``formatType``
#: is the accepted Vega-compat camelCase alias. Both are honored on text channels
#: and positional channels; both serialize to the wire key ``format_type`` via
#: ``_emit_format_type`` in ``base.py``.
TEXT_FORMATTED = frozenset({"type", "format", "format_type", "formatType"})

#: A formatted text field that also carries its own label (TooltipField).
TEXT_FORMATTED_TITLED = TEXT_FORMATTED | {"title"}

#: Facet channels (Facet, FacetRow, FacetCol): data type, panel title, and
#: panel order. ``sort`` accepts the same vocabulary the positional channels
#: accept (direction string, explicit list, sort-field object) and is
#: honored by the ordinal domain's sort resolution, reused at the facet
#: partition (#176(11) + N3, spec §6.4).
FACET = frozenset({"type", "title", "sort"})

#: Facet only (wrap mode via ``encode(facet=...)``): adds ``columns``, the
#: wrap-count sink also written by ``Chart.facet(ncols=N)`` (F-L06-02).
#: FacetRow/FacetCol keep the base ``FACET`` set -- ``columns`` is
#: grid-mode-inapplicable and stays a warn-and-drop there.
FACET_WRAP = FACET | {"columns"}
