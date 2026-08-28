"""Spec-building helpers for the :class:`~ferrum.chart.Chart` value class.

``SpecBuildMixin`` holds the cohesive cluster of spec-*building* helper methods
that ``Chart.to_spec`` orchestrates: layer-list / encoding-spec / facet-dict
assembly, the polar-channel remap, the pending-aggregate / pending-bin
resolvers, the reactive-parameter collection + validation, and the tooltip
injection helpers (cohesion finding CHART/MOD-03).

These stay ``self``-methods (the mixin is mixed into ``Chart``'s MRO) so every
``self._build_*()`` / ``chart._build_*()`` call resolves unchanged.
``Chart.to_spec`` itself stays in ``chart.py`` and calls into this mixin via
``self``.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ferrum._facet import (
    build_facet_dict as _build_facet_dict_fn,
    infer_facet_cardinality as _infer_facet_cardinality_fn,
)
from ferrum.encoding._channel_policy import _RENDERER_HONORED_CHANNELS, _SPEC_KNOWN_CHANNELS
from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin
from ferrum._layer_transforms import _infer_agg_groupby

# Channels excluded from auto-tooltip field derivation: tooltip/tooltip_fields
# are the explicit-tooltip escape hatch (handled by the caller before this
# runs), and detail/key/href/description/url each have their own dedicated
# purpose that isn't "show this raw value on hover".
_AUTO_TOOLTIP_SKIP = frozenset(("tooltip", "detail", "key", "href", "description", "url"))


def _auto_tooltip_fields(enc: dict) -> list[dict]:
    """Derive auto-tooltip ``{"field": ...}`` entries from one encoding dict.

    Shared by :meth:`SpecBuildMixin._inject_auto_tooltips`'s chart-level and
    per-layer injection (GH #52 Task 10f bug #2) so both derive fields the
    same way from whichever encoding dict they're given -- the chart-level
    ``kw["encoding"]`` or one layer's own ``kw["layers"][i]["encoding"]``.

    Parameters
    ----------
    enc : dict
        A parsed-JSON encoding dict (channel name -> ``{"field": ..., ...}``).

    Returns
    -------
    list of dict
        ``[{"field": name}, ...]`` in ``_RENDERER_HONORED_CHANNELS`` order,
        deduplicated by field name. When the channel's serialized dict
        carries an explicit ``title`` -- either user-set, or stamped by
        ``Chart.__add__``'s collision-rename to preserve the original
        column name (see ``chart._rename_encoding_fields``, GH #71) -- the
        entry also carries a ``"title"`` display key, mirroring the shape
        the explicit multi-field ``Tooltip(*fields)`` path already emits in
        :meth:`SpecBuildMixin._build_encoding_specs`. ``field`` always stays
        the (possibly renamed) column used for the actual value lookup;
        ``title`` is presentation-only.
    """
    auto_fields: list[dict] = []
    seen: set[str] = set()
    for ch_name in _RENDERER_HONORED_CHANNELS:
        if ch_name in _AUTO_TOOLTIP_SKIP:
            continue
        ch_dict = enc.get(ch_name)
        if ch_dict is None:
            continue
        field = ch_dict.get("field") if isinstance(ch_dict, dict) else None
        if field and isinstance(field, str) and field not in seen:
            entry: dict = {"field": field}
            title = ch_dict.get("title")
            if title:
                entry["title"] = title
            auto_fields.append(entry)
            seen.add(field)
    return auto_fields


def _warn_channel_dropped(ch_name: str) -> None:
    """``warn_once`` for a channel that is accepted but never reaches the spec.

    Shared by the chart-level (:meth:`SpecBuildMixin._build_encoding_specs`)
    and per-layer (:meth:`SpecBuildMixin._build_layers_list`) safety nets so
    both bucket enforcements emit the identical, accurate message. The
    channel never becomes a ``kw[axis] = EncodingSpec(...)`` assignment nor
    any Rust ``Encoding`` field -- earlier wording ("Stored on EncodingSpec
    for forward-compatibility ... planned for a future Phase") claimed
    otherwise and was corrected 2026-08-27 (P1 remediation, spec-review
    finding).
    """
    from ferrum._warn import warn_once

    warn_once(
        "encoding",
        ch_name,
        message=(
            f"Encoding channel {ch_name!r} is accepted but not rendered; "
            "the renderer ignores it and it is omitted from the spec."
        ),
    )


def _warn_unbucketed_channels(enc: dict) -> None:
    """Warn once on every channel in *enc* outside the RENDERER_HONORED/
    ALIAS/FACET buckets that still carries a concrete field.

    The single enforcement point for the bucket-partition safety net --
    shared by the chart-level (:meth:`SpecBuildMixin._build_encoding_specs`)
    and per-layer (:meth:`SpecBuildMixin._build_layers_list`) passes, so the
    channel-known-set check and the string-vs-``ChannelBase`` field
    extraction cannot drift between them again (2026-08-27 P1 remediation,
    quality-review finding: the two were hand-copied and had already
    drifted -- the layered copy handled plain-string encoding values, the
    chart-level copy did not, even though chart-level `enc` values are
    always ``ChannelBase`` in practice).

    This is what actually produces the ``warn_once`` for the WARN bucket
    (``x_error``/``y_error``/``x_error2``/``y_error2``/``tooltip_field``)
    and for POLAR channels left in *enc* because no ``CoordPolar`` coord
    remapped them away (chart-level: ``_resolve_polar_remapping`` pops
    them from `enc` before this runs when ``CoordPolar`` IS set, so this
    never sees them in that case; the layered path has no per-layer
    ``CoordPolar`` remap at all, so a layer's own polar channel warns here
    unconditionally, regardless of the chart's coord) -- see the bucket
    partition in ``chart.py``.

    *enc* values may be either a ``ChannelBase`` instance or a plain string
    (the layered path's legitimate shorthand).
    """
    from ferrum.encoding._aliases import _channel_field

    for ch_name, ch in enc.items():
        if ch_name in _SPEC_KNOWN_CHANNELS:
            continue
        if _channel_field(ch) is None:
            continue
        _warn_channel_dropped(ch_name)


def _selection_field_names(selections: list) -> set[str]:
    """Collect the union of field names tracked by field-based selections.

    Shared by :meth:`SpecBuildMixin._inject_selection_tooltips` and
    :meth:`SpecBuildMixin._inject_auto_tooltips`, both of which need "every
    field named in an active selection's ``fields`` list" to keep
    cross-panel linked-selection matching fields rather than only data
    index. Callers apply their own ordering (``_inject_selection_tooltips``
    merges with existing fields then sorts; ``_inject_auto_tooltips`` sorts
    directly) -- this helper only collects the unordered set.

    Parameters
    ----------
    selections : list
        Resolved selection objects (e.g. ``self._selections`` or
        ``resolved._selections``); entries may be ``None``.

    Returns
    -------
    set of str
        Field names from every selection whose ``params`` dict carries a
        non-empty ``"fields"`` list.
    """
    field_names: set[str] = set()
    for s in selections:
        if s is not None and hasattr(s, "params") and s.params.get("fields"):
            field_names.update(s.params["fields"])
    return field_names


class SpecBuildMixin:
    """Spec-building helper methods consumed by ``Chart.to_spec``.

    Mixed into ``Chart`` so the methods remain bound (``self`` is the chart) and
    every existing ``self._build_*`` / ``chart._build_*`` call site resolves
    without change.
    """

    def _resolve_polar_remapping(self, resolved, enc: dict) -> dict:
        """Remap theta/radius channel keys to x/y for CoordPolar charts.

        Rust's encoding layer only knows x/y; the spec-side coord conversion in
        scene_build.rs handles the polar→Cartesian pixel transformation.  When
        CoordPolar is set, theta (the angular variable) maps to whichever
        Cartesian axis the coord declares, and radius maps to the other.

        Parameters
        ----------
        resolved :
            The resolved ``Chart`` whose ``_coord`` and ``_mark`` are inspected.
        enc :
            Shallow copy of the encoding dict (already alias-remapped).  Modified
            in place and returned.

        Returns
        -------
        dict
            The remapped encoding dict.
        """
        from ferrum.coord import CoordPolar

        if not isinstance(resolved._coord, CoordPolar):
            return enc

        theta_ch = resolved._coord.theta  # "x" or "y"
        radius_ch = "y" if theta_ch == "x" else "x"
        # Second-extent channels mirror the primary axis assignment:
        # theta2 -> x2 when theta->x, else y2; radius2 -> y2 when radius->y, else x2.
        theta2_ch = f"{theta_ch}2"
        radius2_ch = f"{radius_ch}2"
        if "theta" in enc:
            enc[theta_ch] = enc.pop("theta")
        if "theta2" in enc:
            enc[theta2_ch] = enc.pop("theta2")
        if "radius" in enc:
            enc[radius_ch] = enc.pop("radius")
        if "radius2" in enc:
            enc[radius2_ch] = enc.pop("radius2")
        return enc

    def _build_encoding_specs(
        self, resolved, enc: dict, agg_field_remap: dict, override_encoding: dict | None = None
    ) -> dict:
        """Build ``EncodingSpec`` entries for each honored channel.

        Iterates over ``_RENDERER_HONORED_CHANNELS``, converts each present
        channel to an ``EncodingSpec``, handles multi-field tooltip serialization,
        applies the bar-chart y-zero-anchor injection, and warns on unrecognized
        channels.

        Parameters
        ----------
        resolved :
            The resolved ``Chart`` whose ``_mark`` is inspected for bar-chart
            zero-anchor injection.
        enc :
            Encoding dict after alias remapping and polar remapping.
        agg_field_remap :
            Map from original field name → output column name for any
            ``_PendingAggregate`` transforms already present.
        override_encoding :
            ``Chart.override`` encoding-scale payload (``{channel: {"scale": {...}}}``),
            or ``None``.  Applied last per channel so an override scale leaf beats the
            channel's own ``scale=`` setting (override wins the cascade, spec §7).

        Returns
        -------
        dict
            Partial ``kw`` dict containing only encoding-related keys:
            channel names mapped to ``EncodingSpec`` instances, plus
            ``"tooltip_fields"`` when applicable.
        """
        from ferrum import EncodingSpec
        from ferrum.chart import _apply_inferred_type, _strip_unstackable
        from ferrum.repeat import _RepeatPlaceholder

        # Bucket-partition safety net (single enforcement point, see
        # _warn_unbucketed_channels) -- warns on any channel outside the
        # RENDERER_HONORED/ALIAS/FACET buckets that still carries a concrete
        # field: the WARN bucket (x_error*, tooltip_field), and POLAR
        # channels left in `enc` because no CoordPolar coord remapped them
        # away.
        _warn_unbucketed_channels(enc)

        # Build full EncodingSpec instances per channel so honored kwargs
        # (scale, title) and deferred kwargs (axis, legend, sort, ...) flow to Rust.
        # Phase 7 + 8a's ChartSpec(...) accepts EncodingSpec instances or strings.
        kw: dict = {}
        for axis in _RENDERER_HONORED_CHANNELS:
            if axis not in enc:
                continue
            ch = enc[axis]
            if ch.field is None:
                # Multi-field Tooltip(*fields) — serialize as tooltip_fields JSON list.
                if axis == "tooltip" and hasattr(ch, "_field_list") and ch._field_list:
                    tf_list = []
                    for f in ch._field_list:
                        if isinstance(f, str):
                            tf_list.append({"field": f})
                        elif hasattr(f, "field") and f.field:
                            entry: dict = {"field": f.field}
                            d_f = f.to_encoding_spec_dict()
                            if d_f.get("format"):
                                entry["format"] = d_f["format"]
                            if d_f.get("title"):
                                entry["title"] = d_f["title"]
                            tf_list.append(entry)
                    if tf_list:
                        kw["tooltip_fields"] = json.dumps(tf_list)
                    continue
                # Aggregate shorthands like count() have field=None but the
                # transform emits an output column (e.g. "count_all").  If there
                # is a remap entry for "" (the sentinel for no-source-field), fall
                # through to the EncodingSpec-building code below so the encoding
                # points at the correct output column.
                if "" not in agg_field_remap:
                    continue
            # Phase 9: skip channels whose field is an unresolved Repeat
            # placeholder. RepeatChart.expand() materializes concrete charts
            # before render; the bare template's spec just omits placeholder
            # channels (they're not meaningful standalone).
            if isinstance(ch.field, _RepeatPlaceholder):
                continue
            d = ch.to_encoding_spec_dict()
            # Auto-infer temporal type from column dtype when no explicit type
            # annotation was given.  Explicit ":T" / ":Q" / type_= / type= always
            # win; only the unannotated case is changed here.
            d = _apply_inferred_type(d, d.get("field"), resolved._data)
            # Bar y-axis zero-anchor (gallery defaults A3): inject
            # scale.zero=True on the y-encoding so bar charts always
            # start at zero unless the caller explicitly sets domain or
            # zero on their Y() channel.  The injected scale must carry
            # `type` because Rust's ScaleSpec is a tagged enum.
            #
            # D3: suppress the injection when the caller passed zero=False OR
            # when y2 is bound (the bar extent is taken literally from [y, y2]
            # and force-anchoring to zero would distort the domain).
            # x2 only affects the horizontal bin width of histograms; it must
            # NOT suppress the y zero-anchor.
            _y2_bound = "y2" in enc
            _zero_anchor_wanted = getattr(resolved, "_mark_zero", True) and not _y2_bound
            # D4-C: only inject the linear+zero scale when the y encoding is
            # quantitative.  Ordinal/nominal y (e.g. string category on a
            # horizontal bar) requires a band scale — forcing linear here raises
            # "unsupported dtype: Utf8" in the Rust scale resolver.  When type_
            # is absent the encoding defaults to quantitative (Altair convention),
            # so we only suppress when an explicit categorical type is set.
            _y_enc_type = d.get("type_")
            _y_is_categorical = _y_enc_type in ("O", "N", "ordinal", "nominal")
            if (
                axis == "y"
                and resolved._mark == "bar"
                and _zero_anchor_wanted
                and not _y_is_categorical
            ):
                scale = d.get("scale") or {}
                if "domain" not in scale and "zero" not in scale:
                    d["scale"] = {"type": scale.get("type", "linear"), **scale, "zero": True}
            # D5b: strip ``stack=`` on marks that cannot honor stacking and emit
            # a one-time UserWarning.  Only ``bar`` and ``area`` consume the
            # ``__stack_y_base__`` column emitted by ``apply_stack``; every other
            # mark type silently drops marks when the stacking path executes.
            # Apply the guard only on the positional value channel (``y`` for
            # normal orientation, ``x`` when coord-flipped).
            _strip_unstackable(d, resolved._mark)
            # Chart.override(<channel>_scale_<leaf>=...) — merge the override scale
            # leaves into this channel's scale dict LAST so override wins over the
            # channel's own scale= setting (override wins the cascade, spec §7).
            # Route the merged dict back through _scale_to_dict so a typeless
            # override scale (e.g. just `domain`) gains the `type` discriminator
            # Rust's tagged-enum ScaleSpec deserialiser requires.
            if override_encoding is not None:
                scale_overrides = override_encoding.get(axis, {}).get("scale")
                if scale_overrides:
                    from ferrum.encoding._scale import _scale_to_dict

                    existing_scale = d.get("scale")
                    base_scale = existing_scale if isinstance(existing_scale, dict) else {}
                    d["scale"] = _scale_to_dict({**base_scale, **scale_overrides})
            # `field` is positional; rest are keyword-only on EncodingSpec.__new__.
            # The Python-visible param name is `type_` (Rust signature `type_: Option<&str>`).
            field = d.pop("field")
            # Remap aggregate fields: if this channel's field was aggregated by a
            # _PendingAggregate transform, point the encoding at the output column.
            # Use "" as the lookup key for count() (field=None) since _PendingAggregate
            # stores field="" for count-style aggregates with no source column.
            _remap_key = field if field is not None else ""
            field = agg_field_remap.get(_remap_key, field)
            kw[axis] = EncodingSpec(field, **d)
        return kw

    def _build_layers_list(self, layers: list | None = None) -> list:
        """Convert ``_layers`` to a list of JSON-serializable dicts for Rust.

        ``coerce_layers`` in Rust runs ``json.dumps()`` on each dict, so every
        value must be JSON-serializable (no PyO3 objects).

        Parameters
        ----------
        layers :
            Override layer list to serialize.  Defaults to ``self._layers``;
            ``to_spec`` passes the aggregate-resolved layers so the originating
            chart object is never mutated.
        """
        from ferrum.chart import _apply_inferred_type, _strip_unstackable
        from ferrum.encoding._aliases import apply_channel_aliases
        from ferrum._layer_transforms import _transforms_to_json_list

        out = []
        for layer in (layers if layers is not None else self._layers) or []:
            # Per-layer channel aliasing (mirrors Chart.to_spec's chart-level
            # pass exactly): fill/stroke -> color (with the same
            # stroke-dropped-by-color conflict warning) and detail ->
            # mark_style.detail (mark-conditional warn_once). Operates on a
            # shallow copy so the layer's own frozen `_Layer.encoding` is
            # never mutated. `layer_mk` starts from the layer's own
            # mark-style kwargs so an internal composite mark that already
            # set e.g. `mark_kwargs={"detail": ...}` directly is not
            # clobbered (`alias_detail_channel` uses `setdefault`).
            layer_enc = dict(layer.encoding)
            layer_mk = dict(layer.mark_kwargs) if layer.mark_kwargs else {}
            layer_enc, layer_mk = apply_channel_aliases(layer_enc, layer_mk, layer.mark)

            # Bucket-partition safety net (single enforcement point, see
            # _warn_unbucketed_channels; mirrors _build_encoding_specs's
            # chart-level call exactly). Catches per-layer x_error*/
            # tooltip_field, and theta/radius/theta2/radius2 -- the layered
            # path has no per-layer CoordPolar remap, so a layer's own polar
            # channel warns and never renders regardless of the chart's
            # coord.
            _warn_unbucketed_channels(layer_enc)

            encoding_dict: dict = {}
            for axis in _RENDERER_HONORED_CHANNELS:
                ch = layer_enc.get(axis)
                if ch is None:
                    continue
                if hasattr(ch, "to_encoding_spec_dict"):
                    # ChannelBase subclass — convert to a plain JSON-serializable dict.
                    d = ch.to_encoding_spec_dict()
                    field = d.get("field")
                    if not field:
                        continue
                    # Auto-infer temporal type from column dtype when no explicit type
                    # annotation was given (same logic as _build_encoding_specs).
                    d = _apply_inferred_type(d, field, self._data)
                    # Build a JSON-safe dict matching EncodingSpec's JSON shape.
                    # Note: to_encoding_spec_dict() emits the data-type under
                    # "type_" (Python convention to avoid shadowing the builtin),
                    # but the Rust serde wire format uses "type" (no underscore).
                    # Also normalize shorthand ("Q", "N", "O", "T") to the full
                    # lowercase form that serde deserializes; the PyO3 path does
                    # this via DataType::from_str, but serde only knows "quantitative"
                    # etc.
                    _TYPE_EXPAND = {
                        "Q": "quantitative",
                        "N": "nominal",
                        "O": "ordinal",
                        "T": "temporal",
                    }
                    enc_json_dict: dict = {"field": field}
                    if raw_type := d.get("type_"):
                        enc_json_dict["type"] = _TYPE_EXPAND.get(raw_type, raw_type)
                    # D5b (layered path): strip ``stack=`` on marks that cannot
                    # honor stacking and warn. Shares ``_strip_unstackable`` with
                    # the single-chart path in ``_build_encoding_specs``.
                    _strip_unstackable(d, layer.mark or "point")
                    for opt_key in (
                        "title",
                        "aggregate",
                        "scheme",
                        "format",
                        "format_type",
                        "scale",
                        "axis",
                        "legend",
                        "sort",
                        "stack",
                        "impute",
                    ):
                        if d.get(opt_key):
                            enc_json_dict[opt_key] = d[opt_key]
                    encoding_dict[axis] = enc_json_dict
                elif isinstance(ch, str):
                    encoding_dict[axis] = {"field": ch}
            layer_dict: dict = {
                "mark": layer.mark or "point",
                "encoding": encoding_dict,
            }
            # Wire format to Rust's coerce_layers preserves the legacy
            # ``mark_style`` key. ``layer_mk`` was already alias-aggregated
            # above (fill/stroke -> color, detail -> mark_style.detail).
            if layer_mk:
                layer_dict["mark_style"] = layer_mk
            # data_source: composite-mark layers may pull from a named transform
            # output instead of the final pipeline batch. Only emit when set.
            if layer.data_source is not None:
                layer_dict["data_source"] = layer.data_source
            # Serialize transforms: PyO3 objects need round-tripping through ChartSpec JSON.
            if layer.transforms:
                layer_dict["transforms"] = _transforms_to_json_list(layer.transforms)
            # Phase 9c — per-layer position adjustment. Serialize value classes
            # via ``to_spec_dict``.
            if layer.position is not None:
                layer_dict["position"] = (
                    layer.position.to_spec_dict()
                    if hasattr(layer.position, "to_spec_dict")
                    else layer.position
                )
            if layer.blend is not None:
                layer_dict["blend"] = layer.blend
            if layer.name is not None:
                layer_dict["name"] = layer.name
            # Secondary-y-axis wire flag (GH #52 Task 4). Only emitted when
            # True -- absent/false deserializes and renders byte-identically
            # to a pre-#52 spec (see Layer::independent_y in Rust spec/layer.rs).
            if layer.independent_y:
                layer_dict["independent_y"] = True
            out.append(layer_dict)
        return out

    def _build_facet_dict(self) -> dict:
        """Convert internal _facet to the JSON dict Rust's FacetSpec expects.

        Delegates to :func:`ferrum._facet.build_facet_dict`; the facet machinery
        lives in ``_facet.py`` (cohesion finding C2/C10).
        """
        return _build_facet_dict_fn(self._facet, self._data)

    def _infer_facet_cardinality(self, col_name: str | None) -> int:
        """Return the number of distinct values in *col_name* from self._data.

        Delegates to :func:`ferrum._facet.infer_facet_cardinality`.
        """
        return _infer_facet_cardinality_fn(self._data, col_name)

    def _resolve_pending_aggregates(self, resolved, effective_transforms: list) -> list:
        """Resolve ``_PendingAggregate`` sentinels to concrete ``Aggregate`` objects.

        Any Aggregate shorthand like ``encode(y="mean(val):Q")`` defers groupby
        assignment until all sibling encoding fields are visible.  This method
        infers groupby from non-aggregate encoding fields (mirrors Altair behaviour)
        and replaces each sentinel with a concrete ``Aggregate`` transform.

        Parameters
        ----------
        resolved :
            The resolved ``Chart`` whose ``_encoding`` and ``_facet`` are used
            to collect non-aggregate groupby fields.
        effective_transforms :
            The current list of transforms, which may contain
            ``_PendingAggregate`` sentinels.

        Returns
        -------
        list
            A new list with all ``_PendingAggregate`` sentinels replaced by
            concrete ``Aggregate`` instances.  If no sentinels are present,
            the input list is returned unchanged.
        """
        if not any(isinstance(t, _PendingAggregate) for t in effective_transforms):
            return effective_transforms

        from ferrum import Aggregate, AggregateOp

        # Collect fields from channels that carry no aggregate (shared with the
        # per-layer path via ``_infer_agg_groupby``).
        non_agg_fields = _infer_agg_groupby(resolved._encoding)
        # Include facet fields (row/col grouping dimensions).
        if resolved._facet is not None:
            for _ff in (resolved._facet.col, resolved._facet.row, resolved._facet.field):
                if _ff and _ff not in non_agg_fields:
                    non_agg_fields.append(_ff)
        # Replace sentinels with concrete Aggregate objects.
        resolved_transforms: list = []
        for t in effective_transforms:
            if isinstance(t, _PendingAggregate):
                resolved_transforms.append(
                    Aggregate(
                        [AggregateOp(t.field, t.agg, t.output_col)],
                        groupby=non_agg_fields,
                    )
                )
            else:
                resolved_transforms.append(t)
        return resolved_transforms

    def _resolve_pending_bins(self, effective_transforms: list) -> list:
        """Resolve ``_PendingBin`` sentinels to concrete unnamed ``Bin`` objects.

        Used by the single-chart path (``to_spec``).  Named transforms (used
        by the layered path via ``_resolve_layer_bins``) are NOT produced here;
        the single-chart path keeps the Bin unnamed so it chains through the
        pipeline as before — byte-stable with pre-FA4 behavior.

        Parameters
        ----------
        effective_transforms :
            The current list of transforms, which may contain ``_PendingBin``
            sentinels.

        Returns
        -------
        list
            A new list with all ``_PendingBin`` sentinels replaced by concrete
            unnamed ``Bin`` instances.  If no sentinels are present, the input
            list is returned unchanged.
        """
        if not any(isinstance(t, _PendingBin) for t in effective_transforms):
            return effective_transforms

        from ferrum import Bin

        resolved_transforms: list = []
        for t in effective_transforms:
            if isinstance(t, _PendingBin):
                if t.bin_obj is not None:
                    # Pre-built Bin instance — use it directly (single-chart
                    # path always produces unnamed transforms).  The instance
                    # already bakes in the correct field, bin_count, etc.
                    resolved_transforms.append(t.bin_obj)
                else:
                    bin_kwargs = dict(t.bin_kwargs)
                    bin_kwargs.pop("name", None)  # single-chart bins are always unnamed
                    resolved_transforms.append(Bin(t.field, **bin_kwargs))
            else:
                resolved_transforms.append(t)
        return resolved_transforms

    @staticmethod
    def _collect_params(resolved, enc: dict) -> list:
        """Collect the unified, de-duplicated reactive-parameter list (D6).

        Order: registered selections, explicit ``add_params`` variables, then
        any ``Parameter`` referenced as a scale domain -- on the chart-level
        encoding *and* on every layer's own encoding (GH #72: a layer-bound
        domain param, e.g. an independent-y layer's ``Y(..., scale={"domain":
        fm.param(...)})``, must reach the wire exactly like a chart-level one;
        before this fix ``_collect_params`` only scanned ``enc``, so the layer
        param never reached ``spec.params`` and Rust's substitution store was
        empty for it). Deduplicated by ``.name`` preserving first-seen order,
        regardless of which layer declares a given name.

        Raises
        ------
        ValueError
            If a name is registered as both a ``Selection`` and a
            ``VariableParameter`` (cross-kind collision).  This is always a
            user error — two distinct reactive-object kinds cannot share a name
            without producing a silently wrong spec.
        """
        from ferrum.chart import _check_param_collision
        from ferrum.parameter import Parameter, VariableParameter

        # All encodings to scan for scale-domain Parameters: the chart-level
        # encoding plus every layer's own encoding (layers may be absent).
        all_encodings = [enc]
        for layer in resolved._layers or []:
            if layer is not None and layer.encoding:
                all_encodings.append(layer.encoding)

        # Detect cross-kind collisions before building the ordered list.
        # Guard against None entries that should not exist but are technically
        # possible given that _selections / _params are bare untyped lists.
        selection_names = {sel.name for sel in resolved._selections if sel is not None}
        variable_names = {
            p.name for p in resolved._params if p is not None and isinstance(p, VariableParameter)
        }
        for name in sorted(selection_names & variable_names):
            _check_param_collision(name, is_selection=True, context="param collection")
        # Also check domain-referenced VariableParameters against selections.
        for encoding in all_encodings:
            for ch in encoding.values():
                scale = ch.option("scale") if isinstance(ch, ChannelBase) else None
                if isinstance(scale, dict):
                    domain = scale.get("domain")
                    if isinstance(domain, VariableParameter) and domain.name in selection_names:
                        _check_param_collision(
                            domain.name, is_selection=False, context="scale domain"
                        )

        ordered: list = []
        seen: set[str] = set()

        def _add(p) -> None:
            if isinstance(p, Parameter) and p.name not in seen:
                seen.add(p.name)
                ordered.append(p)

        for sel in resolved._selections:
            _add(sel)
        for p in resolved._params:
            _add(p)
        for encoding in all_encodings:
            for ch in encoding.values():
                scale = ch.option("scale") if isinstance(ch, ChannelBase) else None
                if isinstance(scale, dict):
                    domain = scale.get("domain")
                    if isinstance(domain, Parameter):
                        _add(domain)
        return ordered

    @staticmethod
    def _validate_params_finite(params_list: list) -> None:
        """Raise a legible ValueError if any VariableParameter carries a non-finite float.

        ``json.dumps`` emits the non-JSON tokens ``Infinity`` / ``NaN`` for
        Python's ``math.inf`` and ``math.nan``, which the Rust serde
        deserializer rejects with a cryptic column-offset error.  This guard
        runs before serialization so the user sees the offending parameter name
        and a clear explanation.

        Only ``VariableParameter`` values are checked; ``Selection`` objects do
        not carry scalar ``value`` fields in the same way.

        Raises
        ------
        ValueError
            If any parameter's ``value`` (or an element of it when it is a
            list) is ``math.inf``, ``-math.inf``, or ``math.nan``.
        """
        from ferrum.parameter import VariableParameter

        def _has_non_finite(v: Any) -> bool:
            if isinstance(v, float) and not math.isfinite(v):
                return True
            if isinstance(v, (list, tuple)):
                return any(_has_non_finite(item) for item in v)
            return False

        for p in params_list:
            if not isinstance(p, VariableParameter):
                continue
            if _has_non_finite(p.value):
                raise ValueError(
                    f"Parameter {p.name!r} has a non-finite value ({p.value!r}). "
                    f"Parameter values must be finite numbers. "
                    f"Use a finite bound instead of Inf or NaN."
                )

    @staticmethod
    def _inject_selection_tooltips(kw: dict, selections: list) -> None:
        """Merge selection-tracked fields into the spec-assembly tooltip keys.

        Called during ``to_spec()`` to ensure that every field named in an
        active selection's ``fields`` list is also present in the chart's
        tooltip, so cross-panel linked-selection can match marks by field
        value rather than only by data index.

        Operates on the ``kw`` dict being assembled before ``ChartSpec`` is
        constructed (``tooltip`` values are still Python channel objects;
        ``tooltip_fields`` is a JSON string when present).  Mutates *kw* in
        place; returns nothing.

        Parameters
        ----------
        kw : dict
            The spec-assembly keyword dict (in-progress ``ChartSpec`` kwargs).
        selections : list
            The resolved selections list (``resolved._selections``).
        """
        sel_fields = _selection_field_names(selections)
        if not sel_fields:
            return
        existing: set[str] = set()
        if "tooltip_fields" in kw:
            for entry in json.loads(kw["tooltip_fields"]):
                existing.add(entry.get("field", ""))
        elif "tooltip" in kw:
            existing.add(getattr(kw["tooltip"], "field", ""))
        missing = sel_fields - existing
        if not missing:
            return
        tf_list: list[dict] = []
        if "tooltip_fields" in kw:
            tf_list = json.loads(kw["tooltip_fields"])
        elif "tooltip" in kw:
            tf_list = [{"field": getattr(kw["tooltip"], "field", "")}]
            del kw["tooltip"]
        for f in sorted(missing):
            tf_list.append({"field": f})
        kw["tooltip_fields"] = json.dumps(tf_list)

    def _inject_auto_tooltips(self, kw: dict) -> dict:
        """Inject auto-generated tooltip fields into a serialised spec dict.

        Takes a plain-dict representation of a ``ChartSpec`` (as returned by
        ``json.loads(spec.to_json())``) and adds ``encoding.tooltip_fields``
        from the encoded channels when no explicit tooltip is already present.
        Returns the mutated dict.

        This is called by the interactive renderer after ``to_spec()``; static
        SVG/PNG renders skip it to avoid bloating the output with tooltip data.
        Explicit ``tooltip=`` or ``tooltip_fields=`` encodings always win.

        Layered charts also get PER-LAYER tooltip fields (GH #52 Task 10f bug
        #2): each ``kw["layers"][i]`` gets its own ``encoding.tooltip_fields``
        derived from that layer's own merged encoding (``kw["layers"][i]
        ["encoding"]``, already populated per layer by
        :meth:`_build_layers_list`), not the chart-level fields. Without this,
        every non-primary layer's tooltip reports the PRIMARY layer's fields
        (confirmed via headless WASM capture -- hovering a secondary-y-axis
        layer's mark showed the primary layer's ``x``/``revenue`` instead of
        its own field). An explicit chart-level ``tooltip``/``tooltip_fields``
        wins over the chart-level auto-injection (skipping it entirely).

        Whether it ALSO short-circuits the per-layer loop below depends on
        the *provenance* of the chart-level value, not merely on whether a
        layer happens to carry its own explicit tooltip (GH #78 fixed a
        false positive in the earlier structural proxy -- see below):

        - ``Chart.__add__`` promotes the *primary* layer's own explicit
          ``tooltip=`` onto the merged chart-level encoding (``new =
          lhs._clone()``), and ``_expand_layers`` gives that same layer's
          own ``encoding`` dict the identical key -- so the primary layer's
          own encoding ALSO carries the explicit tooltip. ``Chart.__add__``
          records this as ``self._tooltip_promoted = True``. In that case
          the chart-level value is just a view of the primary layer's, and
          other layers must still get their own auto-injected fields --
          otherwise Rust's chart-level tooltip fallback leaks the primary
          layer's fields onto every other layer's marks.
        - A ``tooltip=`` set directly on an already-merged chart (e.g.
          ``merged.encode(tooltip=...)``) only touches the chart-level
          ``_encoding`` -- no layer's own encoding carries it, and
          ``Chart.encode()`` resets ``_tooltip_promoted`` to ``False`` when
          it sees a ``tooltip=`` channel. That is a genuine chart-wide
          override, so it short-circuits every layer's auto-injection (a
          per-layer auto injection would otherwise beat the explicit
          tooltip in Rust's ``inherit_from`` merge). This case holds even
          when SOME layer independently carries its own explicit tooltip
          (e.g. from an earlier promotion) -- the earlier structural proxy
          (``any_layer_explicit``) treated that as evidence of promotion
          and incorrectly ran the per-layer loop for the *other*,
          tooltip-less layers, injecting spurious per-layer fields that beat
          the genuine chart-wide override in Rust's ``inherit_from`` merge.
          Those layers must instead fall back to the chart-level value, same
          as the all-implicit case.
        - ``_inject_selection_tooltips`` (called earlier in ``to_spec_dict``,
          before this method sees the wire) merges a field-based selection's
          ``fields`` into chart-level ``tooltip_fields`` whenever the chart
          carries an active field-based selection (GH #58) -- including when
          the chart-level tooltip is ALSO promoted (it rewrites the
          promoted single ``tooltip`` into a merged ``tooltip_fields`` list
          if the selection names a field the promoted tooltip didn't
          already cover). Neither case is a genuine chart-wide override --
          selection injection exists purely to make cross-layer selection
          matching work -- so the per-layer loop must still run whenever a
          field-based selection is active, and each layer's own auto fields
          are unioned with the selection's fields (selection fields appended
          after each layer's own, deduplicated by field name, mirroring the
          existing-first-then-selection-appended order
          ``_inject_selection_tooltips`` itself uses for the unlayered
          case). The selection's field set is re-derived directly from
          ``self._selections`` (not read off the wire) so the union applies
          uniformly regardless of whether the chart-level tooltip's OTHER
          provenance is promoted, selection-only, or absent.

        The short-circuit discriminator is therefore provenance, not
        structure: a chart-level explicit tooltip short-circuits the loop
        only when it is neither promoted (``self._tooltip_promoted``) nor
        selection-injected (chart-level ``tooltip``/``tooltip_fields``
        present in the wire but absent from ``self._encoding`` -- the only
        way that combination arises is ``_inject_selection_tooltips`` having
        added it with no other source, since ``tooltip_fields`` is never
        itself a settable channel and a promoted or directly-encoded
        tooltip always leaves ``self._encoding`` carrying ``tooltip``). A
        layer that itself carries an explicit ``tooltip``/``tooltip_fields``
        is always left untouched by the loop (its own explicit value always
        wins for that layer). Unlayered and single-layer charts emit the
        exact same wire as before this fix (no ``kw["layers"]`` key, or a
        layers list whose entries already carry no distinct-from-chart-level
        fields to add).

        Parameters
        ----------
        kw : dict
            Parsed JSON dict from ``json.loads(spec.to_json())``.  Modified
            in place and returned.

        Returns
        -------
        dict
            The same dict with ``encoding.tooltip_fields`` added at the chart
            level and, for layered charts, on each layer's own encoding.
        """
        enc = kw.get("encoding") or {}
        chart_level_explicit = "tooltip" in enc or "tooltip_fields" in enc
        if not chart_level_explicit:
            auto_fields = _auto_tooltip_fields(enc)
            if auto_fields:
                kw.setdefault("encoding", {})["tooltip_fields"] = auto_fields

        layers = kw.get("layers") or []
        # GH #58: a selection-injected chart-level tooltip_fields entry never
        # appears in self._encoding (only _inject_selection_tooltips's mutation
        # of the wire kw put it there); a promoted or directly-.encode()'d
        # tooltip always does. That distinguishes "the chart-level tooltip
        # exists SOLELY because of a selection" (short-circuit must not fire)
        # from the promoted/genuine cases, without threading extra state
        # through to_spec_dict.
        selection_injected = chart_level_explicit and "tooltip" not in self._encoding
        if chart_level_explicit and not self._tooltip_promoted and not selection_injected:
            # Genuine chart-wide override (set directly via
            # .encode(tooltip=...), not promoted from a layer and not
            # injected by a field-based selection) -- it wins for every
            # layer via Rust's chart-level tooltip_fields fallback
            # (Encoding::inherit_from).
            return kw

        # GH #58: the selection's own field set, re-derived directly from
        # self._selections (mirroring _inject_selection_tooltips's own
        # field-collection) rather than gated on `selection_injected`. A
        # PROMOTED chart-level tooltip can coexist with a field-based
        # selection -- _inject_selection_tooltips still merges the
        # selection's fields into the chart-level tooltip_fields in that
        # case (self._encoding keeps carrying the promoted "tooltip" key, so
        # `selection_injected` is False there) -- and every tooltip-less
        # layer must still pick up those selection fields, not just its own
        # auto fields.
        selection_field_names = _selection_field_names(self._selections)
        selection_fields = (
            [{"field": f} for f in sorted(selection_field_names)] if selection_field_names else None
        )

        for layer in layers:
            layer_enc = layer.get("encoding") or {}
            if "tooltip" in layer_enc or "tooltip_fields" in layer_enc:
                continue
            layer_auto_fields = _auto_tooltip_fields(layer_enc)
            if selection_fields:
                seen_fields = {f.get("field") for f in layer_auto_fields}
                layer_auto_fields = layer_auto_fields + [
                    dict(f) for f in selection_fields if f.get("field") not in seen_fields
                ]
            if layer_auto_fields:
                layer.setdefault("encoding", {})["tooltip_fields"] = layer_auto_fields
        return kw
