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

Membership here is IDENTICAL to the per-class literals the channel modules
declared before this consolidation — this is a vocabulary unification, not a
contract change. The appearance-channel membership is additionally pinned by
``tests/test_appearance_honored_kwargs.py``.
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
        # legend dict — honored by prepare.rs legend_orient_override / title.
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
#: (StrokeOpacity, StrokeWidth, StrokeDash, Angle.)
APPEARANCE_BASE = frozenset({"type", "legend"})

#: Adds an explicit scale override and legend title. (FillOpacity.)
APPEARANCE_FULL = APPEARANCE_BASE | {"scale", "title"}

#: Adds a conditional encoding. (Size, Opacity.)
APPEARANCE_CONDITION = APPEARANCE_FULL | {"condition"}

#: Adds an explicit domain-order sort. (Shape.)
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
