"""Channel-disposition policy — the five-bucket taxonomy over every encoding channel.

Every channel in :func:`ferrum.encoding._channel_class_map` falls into exactly
one of the five buckets below (test-enforced partition, see
``tests/test_finding_p1.py``). ``Chart.encode()`` is a total function over
that map: each channel either renders, aliases, or ``warn_once``'s — never a
silent drop. See ``ferrum-spec.md`` §3.2 (2026-08-27 dated note) for the
user-facing contract this partition implements.

Relocated 2026-08-27 (#103) from ``ferrum.chart`` to this zero-import leaf
(the ``_honored.py`` sibling) so encoding-vocabulary knowledge lives in
``ferrum.encoding`` rather than the top-level ``Chart`` value class;
``chart.py``/``_spec_build.py`` are consumers, not the source of truth. This
is a pure relocation — bucket membership, tuple order, and every provenance
comment are unchanged.

Role: this module answers "what does ``Chart.encode()`` *do* with a given
channel *name*" (render it into its own ``EncodingSpec``, alias it to
another channel, warn-and-drop it, remap it under polar, or route it
through faceting) — a per-channel ROUTING disposition. Its sibling
``ferrum.encoding._honored`` answers a different question, "which *kwargs*
does one channel *instance* honor" (e.g. does ``X(...)`` accept ``sort=``)
— a per-channel-type KWARG scope. Both modules use "honored" for their own
vocabulary; the overlap in name is deliberate, not an accidental collision.
"""

from __future__ import annotations

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
