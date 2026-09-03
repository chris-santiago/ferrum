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
reads it. The appearance-channel membership is pinned by
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
        # legend dict — honored by render::prepare::legend::per_channel_legend_specs
        # (the color > x > y cascade NF-B13 gave X/Y a real consumer for,
        # 2026-09-02 batch-B task 7; the never-existent legend_orient_override
        # citation this replaced predates that consumer).
        "legend",
    }
)

#: Secondary-extent positional channels (X2, Y2, X/Y error pairs, Theta2,
#: Radius2). These carry no scale of their own — they reuse the primary
#: channel's scale — so only the data type is honored.
SECONDARY_EXTENT = frozenset({"type"})

#: Primary polar value channels (Theta, Radius): data type plus stacking
#: (wind-rose / coxcomb / pie accumulation).
POLAR_PRIMARY = frozenset({"type", "stack"})


# ---------------------------------------------------------------------------
# Appearance roles (compose upward; final sets match the prior _APPEARANCE_*)
# ---------------------------------------------------------------------------

#: Minimum appearance contract — data type + legend suppression/customization.
#: (StrokeWidth, Angle — both documented as per-row constants with no scale
#: resolution; see ``spec/encoding.rs:746``.)
APPEARANCE_BASE = frozenset({"type", "legend"})

#: Adds an explicit scale override and legend title. (FillOpacity,
#: StrokeOpacity — batch-A §4.3 opens the wire gate so these two serialize
#: ``scale=``/``title=`` instead of warn-and-drop. StrokeDash started here
#: too on 2026-08-28 but moved up to ``APPEARANCE_SORT`` on 2026-09-01 — see
#: that constant's docstring.)
APPEARANCE_FULL = APPEARANCE_BASE | {"scale", "title"}

#: Adds a conditional encoding. (Size, Opacity.)
APPEARANCE_CONDITION = APPEARANCE_FULL | {"condition"}

#: Adds an explicit domain-order sort. (Shape; StrokeDash as of 2026-09-01,
#: batch-A T12 spec-review addendum — mirrors Shape since the Rust
#: ``build_stroke_dash_scale`` domain builder already reads ``sort``.)
APPEARANCE_SORT = APPEARANCE_CONDITION | {"sort"}

#: Adds a named color scheme. (Fill, Stroke.)
APPEARANCE_SCHEME = APPEARANCE_CONDITION | {"scheme"}

#: The Color channel: full appearance contract plus both sort and scheme.
APPEARANCE_COLOR = APPEARANCE_CONDITION | {"sort", "scheme"}


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

#: Facet channels (Facet, FacetRow, FacetCol): data type + panel title.
FACET = frozenset({"type", "title"})
