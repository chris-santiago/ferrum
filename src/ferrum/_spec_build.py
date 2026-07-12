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
        deduplicated by field name.
    """
    from ferrum.chart import _RENDERER_HONORED_CHANNELS

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
            auto_fields.append({"field": field})
            seen.add(field)
    return auto_fields


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
        # Arc marks need a dummy y (or x) so scale_resolve doesn't fail when
        # only one axis is encoded.  The arc builder ignores the dummy scale.
        if resolved._mark == "arc":
            if theta_ch == "x" and "y" not in enc and "x" in enc:
                enc["y"] = enc["x"]
            elif theta_ch == "y" and "x" not in enc and "y" in enc:
                enc["x"] = enc["y"]
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
        from ferrum._warn import warn_once
        from ferrum.chart import (
            _FACET_CHANNELS,
            _POLAR_CHANNELS,
            _RENDERER_HONORED_CHANNELS,
            _SILENT_CHANNELS,
            _apply_inferred_type,
            _strip_unstackable,
        )
        from ferrum.repeat import _RepeatPlaceholder

        # Safety net: warn on channels outside all known sets.
        _all_known = (
            frozenset(_RENDERER_HONORED_CHANNELS)
            | _SILENT_CHANNELS
            | _POLAR_CHANNELS
            | _FACET_CHANNELS
        )
        for ch_name, ch in enc.items():
            if ch_name in _all_known:
                continue
            field = getattr(ch, "field", None)
            if field is None or isinstance(field, _RepeatPlaceholder):
                continue
            warn_once(
                "encoding",
                ch_name,
                message=(
                    f"Encoding channel {ch_name!r} is accepted but not yet "
                    "rendered; the SVG will omit it (planned for a future Phase). "
                    "Stored on EncodingSpec for forward-compatibility."
                ),
            )

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
        from ferrum.chart import (
            _RENDERER_HONORED_CHANNELS,
            _apply_inferred_type,
            _strip_unstackable,
        )
        from ferrum._layer_transforms import _transforms_to_json_list

        out = []
        for layer in (layers if layers is not None else self._layers) or []:
            encoding_dict: dict = {}
            for axis in _RENDERER_HONORED_CHANNELS:
                ch = layer.encoding.get(axis)
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
            # ``mark_style`` key.
            if layer.mark_kwargs:
                layer_dict["mark_style"] = dict(layer.mark_kwargs)
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
        sel_fields: set[str] = set()
        for s in selections:
            if hasattr(s, "params") and s.params.get("fields"):
                sel_fields.update(s.params["fields"])
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
        where the chart-level value came from (GH #71 defect 3):

        - ``Chart.__add__`` promotes the *primary* layer's own explicit
          ``tooltip=`` onto the merged chart-level encoding (``new =
          lhs._clone()``), and ``_expand_layers`` gives that same layer's
          own ``encoding`` dict the identical key -- so the primary layer's
          own encoding ALSO carries the explicit tooltip. In that case the
          chart-level value is just a view of the primary layer's, and
          other layers must still get their own auto-injected fields --
          otherwise Rust's chart-level tooltip fallback leaks the primary
          layer's fields onto every other layer's marks.
        - A ``tooltip=`` set directly on an already-merged chart (e.g.
          ``merged.encode(tooltip=...)``) only touches the chart-level
          ``_encoding`` -- no layer's own encoding carries it. That is a
          genuine chart-wide override, so it short-circuits every layer's
          auto-injection (a per-layer auto injection would otherwise beat
          the explicit tooltip in Rust's ``inherit_from`` merge).

        The discriminator is therefore: does *any* layer's own encoding
        already carry an explicit ``tooltip``/``tooltip_fields``? If yes,
        the per-layer loop still runs for the remaining (tooltip-less)
        layers; if no layer has one of its own, the chart-level explicit
        value short-circuits the whole loop. A layer that itself carries an
        explicit ``tooltip``/``tooltip_fields`` is always left untouched by
        the loop (its own explicit value always wins for that layer).
        Unlayered and single-layer charts emit the exact same wire as
        before this fix (no ``kw["layers"]`` key, or a layers list whose
        entries already carry no distinct-from-chart-level fields to add).

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
        any_layer_explicit = any(
            "tooltip" in (layer.get("encoding") or {})
            or "tooltip_fields" in (layer.get("encoding") or {})
            for layer in layers
        )
        if chart_level_explicit and not any_layer_explicit:
            # The chart-level tooltip did not come from a promoted layer --
            # it is a genuine chart-wide override, so it wins for every layer.
            return kw

        for layer in layers:
            layer_enc = layer.get("encoding") or {}
            if "tooltip" in layer_enc or "tooltip_fields" in layer_enc:
                continue
            layer_auto_fields = _auto_tooltip_fields(layer_enc)
            if layer_auto_fields:
                layer.setdefault("encoding", {})["tooltip_fields"] = layer_auto_fields
        return kw
