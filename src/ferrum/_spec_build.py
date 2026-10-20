"""Spec-building helpers for the :class:`~ferrum.chart.Chart` value class.

``SpecBuildMixin`` holds the cohesive cluster of spec-*building* helper methods
that ``Chart.to_spec`` orchestrates: layer-list / encoding-spec / facet-dict
assembly, the polar-channel lowering, the pending-aggregate / pending-bin
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
from typing import Any, NamedTuple

import pyarrow as pa

from ferrum._core import (
    EncodingSpec as _EncodingSpec,
    column_reads_categorical as _core_column_reads_categorical,
    scale_accepted_keys as _scale_accepted_keys,
    validate_scale_dict as _validate_scale_dict,
)
from ferrum._facet import (
    build_facet_dict as _build_facet_dict_fn,
    facet_repeat as _facet_repeat_fn,
    infer_facet_cardinality as _infer_facet_cardinality_fn,
)
from ferrum._coerce import normalize_for_rust
from ferrum._shorthand import parse_shorthand
from ferrum.coord import _coord_wire
from ferrum.encoding._channel_policy import _RENDERER_HONORED_CHANNELS, _SPEC_KNOWN_CHANNELS
from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin
from ferrum._encode_lowering import (
    GROUPING_KEY,
    _polar_channel_targets,
    _statistic_plan,
    _theta_axis,
)
from ferrum._layer import layer_mark
from ferrum._layer_transforms import _POSITIONAL, _split_chain, _with_statistics_marker
from ferrum._override_apply import OverridePayload
from ferrum._override_consume import (
    merge_encoding_scale as _merge_encoding_scale,
    refuse_binned_color_scale_overrides as _refuse_binned_color_scale_overrides,
    refuse_malformed_scale_overrides as _refuse_malformed_scale_overrides,
)


class _ResolvedStatistic(NamedTuple):
    """``SpecBuildMixin._resolve_statistic``'s two coupled outputs.

    The transform list and the encoding are resolved together: which column
    each channel draws depends on the step the statistic computes.
    """

    transforms: list
    encoding: dict


def _override_channels(channel: str, source: str) -> list[str]:
    """The override channel names whose ``<name>_scale_*`` leaves reach *channel*'s scale.

    *source* is the binding that draws as *channel* (``alias_source``): when
    it is an alias (``fill`` or ``stroke`` drawing as ``color``), its leaves
    apply, then *channel*'s own, which win per leaf. A source the alias rules
    dropped draws nothing, so its leaves reach no scale.
    """
    return [source, channel] if source != channel else [channel]


def _color_source(channel: str, color_source: str | None) -> str | None:
    """The alias source drawing as *channel*: *color_source* on ``color``, else ``None``."""
    return color_source if channel == "color" else None


def _with_override_scale(
    d: dict, channel: str, override: OverridePayload | None, source: str | None = None
) -> dict:
    """Return channel spec dict *d* with ``Chart.override``'s scale leaves for *channel* merged in.

    The one place an override's ``<channel>_scale_*`` leaves meet a channel's
    own ``scale=`` (*source*: the alias source drawing as *channel*, whose
    leaves apply too, see :func:`_override_channels`), for the chart-level encoding and for every layer that
    declares the channel alike, so a layered chart renders the override exactly
    as the channel's own ``scale=`` would. Override leaves win (spec §7). The
    merged dict goes back through ``_scale_to_dict`` so a typeless override
    (just ``domain``) gains the ``type`` discriminator the wire requires.

    Four refusal layers run in order, named after the value's own origin
    (#157):

    - on a binned color (*d* carries ``bin_channel``, the class written),
      :func:`~ferrum._override_consume.refuse_binned_color_scale_overrides`
      refuses a leaf that would replace what ``bin=`` builds, naming the
      class and the override path;
    - :func:`~ferrum._override_consume.merge_encoding_scale` refuses the
      CHART's own keys first (its own docstring states each of its internal
      stages' exact voice: the tag check, the two shape-probe buckets, and
      the old-type padding-family bounds loop);
    - :func:`~ferrum._override_consume.refuse_malformed_scale_overrides`
      refuses the OVERRIDE's own values, naming the flat override path the
      caller actually wrote (``x_scale_padding``, not ``padding``);
    - the merged gate (``validate_scale_dict``) is the backstop for anything
      left, so a refused override still refuses in the gate's own voice on
      every path, including a layer-only channel whose JSON wire dict would
      otherwise refuse in serde's.

    *d* is never mutated.
    """
    if override is None:
        return d
    scale_overrides: dict = {}
    paths: dict = {}
    for name in _override_channels(channel, source or channel):
        scale_overrides.update(override.encoding.get(name, {}).get("scale") or {})
        paths.update(override.encoding_paths.get(name, {}))
    if not scale_overrides:
        return d
    if d.get("bin_channel"):
        _refuse_binned_color_scale_overrides(d["bin_channel"], scale_overrides, paths)
    from ferrum.encoding._scale import _scale_to_dict

    existing_scale = d.get("scale")
    base_scale = existing_scale if isinstance(existing_scale, dict) else {}
    # `_merge_encoding_scale` alone (no `_scale_to_dict` conversion yet) gives
    # the effective type for the per-key probe below. The probe must run
    # BEFORE `_scale_to_dict` converts the merged dict as a whole: that
    # conversion (temporal domain elements, a reactive Parameter) can itself
    # raise on a malformed override value -- e.g. an unparseable date in
    # `x_scale_domain=["garbage", ...]` -- and a raise there would reach the
    # caller unnamed, before the probe ever got a turn. The probe's own
    # per-key fragment runs the identical conversion (see
    # `refuse_malformed_scale_overrides`), so it catches that failure too.
    pre_merge = _merge_encoding_scale(base_scale, scale_overrides)
    # `pre_merge.get("type")`, not `.get("type", "linear")`: the typeless
    # default is `_scale_to_dict`'s one to own (`out.setdefault("type",
    # "linear")`). Passing `None` through to `refuse_malformed_scale_overrides`
    # lets its own per-key fragment omit the `type` key too, so the same
    # single call supplies the default for both the probe and the merge.
    _refuse_malformed_scale_overrides(pre_merge.get("type"), scale_overrides, paths)
    merged = _scale_to_dict(pre_merge)
    _validate_scale_dict(merged)
    return {**d, "scale": merged}


# Encoding ``type_`` values that make a channel categorical (band scale).
_CATEGORICAL_TYPES = ("O", "N", "ordinal", "nominal")


def _bar_value_on_x(enc: dict, data: Any) -> bool:
    """Whether a bar draws its value along its authored ``x`` channel.

    A bar's orientation follows its ``y``: a categorical ``y`` makes it a
    native horizontal bar whose value runs along ``x``; otherwise the value is
    ``y``. ``y`` is categorical when declared ordinal/nominal, or when it is
    undeclared and its column in *data* reads categorical to the render
    (``ferrum._core.column_reads_categorical``: a string, boolean or
    dictionary column; a date reads numeric). ``CoordFlip`` (and
    ``orient="horizontal"``, which desugars to it) moves where the authored
    channels draw, not which one holds the value, so it does not enter here.
    """
    y = enc.get("y")
    if y is None:
        return False
    field, declared = _encoding_field_and_type(y)
    if declared is not None:
        return declared in _CATEGORICAL_TYPES
    return _column_reads_categorical(data, field)


def _encoding_field_and_type(value: Any) -> tuple[Any, str | None]:
    """The ``(field, declared type)`` of one encoding value, in any spelling.

    A channel object (``fm.Y("v", type_="N")``) answers from its own spec
    dict; a string (a layer's ``encoding={"y": "v"}``) is read as encoding
    shorthand, so ``"v:N"`` declares ``N``. A string the shorthand grammar
    does not parse is a plain field name with no declared type.
    """
    if isinstance(value, str):
        try:
            field, declared, _aggregate = parse_shorthand(value)
        except ValueError:
            return value, None
        return field, declared
    return value.field, value.to_encoding_spec_dict().get("type_")


def _column_reads_categorical(data: Any, field: Any) -> bool:
    """Whether *field*'s column in *data* reads categorical to the render.

    *data* is the chart's frame (a ``polars.DataFrame``, a ``pyarrow.Table``
    or ``None``); a missing frame, a non-string field (an aggregate's
    ``None``, a repeat placeholder) or a column the frame lacks reads not
    categorical, as an absent column does in the render. The predicate reads
    the schema the render receives: an empty slice of *data* passed through
    ``normalize_for_rust``, which decodes dictionaries and casts null, date
    and duration columns.
    """
    if data is None or not isinstance(field, str):
        return False
    empty = data.slice(0, 0) if isinstance(data, pa.Table) else data.head(0).to_arrow()
    return _core_column_reads_categorical(normalize_for_rust(empty).schema, field)


def _bar_zero_anchor(
    d: dict, *, mark: Any, mark_zero: bool, value_on_x: bool, axis: str, enc: dict
) -> dict:
    """Return channel dict *d* with a bar's zero anchor injected, when it applies.

    The one producer of the bar zero anchor: ``scale.zero=True`` on a bar's
    value channel, so a bar starts at zero unless the caller set ``domain`` or
    ``zero`` on that channel. Used for a flat chart's encoding and for every
    independent-y bar layer (a ``SecondaryY`` slot, a
    ``LayerChart(resolve={"y": "independent"})`` member), so each gets the
    same answer for every caller-scale shape. *d* is never mutated.

    - The value channel is ``y``, or ``x`` for a native horizontal bar
      (*value_on_x*, from :func:`_bar_value_on_x`); any other *axis* is
      returned unchanged.
    - Suppressed when the bar opted out (*mark_zero* false, ``zero=False``) or
      the value channel's span partner (``y2`` for ``y``, ``x2`` for ``x``) is
      bound in *enc*: the bar's extent is then taken literally from
      ``[y, y2]``, and anchoring at zero would distort the domain. The other
      channel's partner only sets a histogram's bin width, so it does not
      suppress.
    - Only a quantitative value channel takes the linear+zero scale: an
      ordinal/nominal one needs a band scale, and forcing linear raises
      "unsupported dtype: Utf8" in the Rust scale resolver. An absent
      ``type_`` is quantitative (Altair convention), so only an explicit
      categorical type suppresses.
    - ``zero`` is a ``ScaleSpec::Linear``-only wire field and the raw-dict
      scale gate refuses it elsewhere, so it is added only to a scale with no
      ``domain`` or ``zero`` whose type accepts it, asked of the gate itself
      (:func:`_scale_type_accepts_zero`). The scale carries ``type`` because
      Rust's ``ScaleSpec`` is a tagged enum.
    - ``systemInjected`` marks the scale as ferrum's only when this function
      authored all of it, so Rust's "is there an explicit scale here" gates
      (``composite::leaf_positional_domain``'s sharing exclusion and
      ``positional::build_axis_scale``'s bypass) let a layered bar join the
      shared domain union. A caller's own ``scale=`` (or an
      ``override(<axis>_scale_*=...)`` leaf) that only gained ``zero`` here is
      not marked: its ``padding``/``nice``/``range``/``clamp`` are real intent,
      which only the explicit-scale path honors.
    """
    is_value_channel = axis == "x" if value_on_x else axis == "y"
    if (
        not is_value_channel
        or mark != "bar"
        or not mark_zero
        or f"{axis}2" in enc
        or d.get("type_") in _CATEGORICAL_TYPES
    ):
        return d
    scale = d.get("scale") or {}
    scale_type = scale.get("type")
    if "domain" in scale or "zero" in scale or not _scale_type_accepts_zero(scale_type):
        return d
    anchored = {"type": scale_type or "linear", **scale, "zero": True}
    if not scale:
        anchored["systemInjected"] = True
    return {**d, "scale": anchored}


def _scale_type_accepts_zero(scale_type: Any) -> bool:
    """Whether a scale ``"type"`` tag accepts the ``zero`` wire key.

    Asked of the gate's own published table
    (``ferrum._core.scale_accepted_keys``) rather than compared against a
    ``"linear"`` literal, so the bar zero-anchor tracks the Rust schema instead
    of mirroring one row of it.  ``None`` means the dict carries no ``type`` at
    all, which reaches the wire as ``linear`` (``_scale_to_dict``'s default);
    any non-``str`` or unknown tag answers ``False`` so the zero-anchor stays
    out of the way and the gate itself does the refusing.
    """
    if scale_type is None:
        return True
    if not isinstance(scale_type, str):
        return False
    try:
        return "zero" in _scale_accepted_keys(scale_type)
    except ValueError:
        return False


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
        carries the caller's ``title``, the entry also carries it as a
        ``"title"`` display key, mirroring the shape the explicit
        multi-field ``Tooltip(*fields)`` path already emits in
        :meth:`SpecBuildMixin._build_encoding_specs`. ``field`` is the
        column used for the value lookup; ``title`` is presentation-only.
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


def _inherited_wire_channels(layer, encoding_dict: dict) -> list[str]:
    """The wire channels of *layer*'s serialized *encoding_dict* the layer inherited.

    Each of ``layer.inherited_channels`` is named by the channel it draws as
    (``fill`` and ``stroke`` draw as ``color``), kept only when the layer's
    wire encoding carries it: ``detail`` serializes as mark style, a channel
    a sibling spelling of the layer's own shadows never reaches the wire as
    inherited, and a positional channel's provenance is
    ``inherited_positional``'s.
    """
    from ferrum.encoding._aliases import resolved_channel

    out: list[str] = []
    for channel in layer.inherited_channels:
        wire = resolved_channel(channel)
        # An alias source whose target the layer binds itself draws nothing.
        shadowed = (
            wire != channel and wire in layer.encoding and wire not in layer.inherited_channels
        )
        positional = wire in ("x", "y", "x2", "y2")
        if wire in encoding_dict and wire not in out and not shadowed and not positional:
            out.append(wire)
    return out


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
    remapped them away. :meth:`SpecBuildMixin._with_polar_channels_lowered`
    moves them onto x/y first wherever a ``CoordPolar`` applies (the chart's
    own, and each layer's effective coord), so this sees a polar channel only
    where no polar coord claims it, which is exactly when it would never
    render. See the bucket partition in ``chart.py``.

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


def _remap_polar_channels(coord, enc: dict) -> dict:
    """Move ``theta``/``radius`` channel keys of *enc* onto x/y for a ``CoordPolar`` *coord*.

    See :func:`_polar_channel_targets` for the mapping. Any other coord
    (including ``None``) returns *enc* untouched. *enc* is modified in place
    and returned (callers pass a copy).
    """
    for polar, cartesian in _polar_channel_targets(coord):
        if polar in enc:
            enc[cartesian] = enc.pop(polar)
    return enc


def _polar_lowered_layer(layer, coord):
    """*layer* with its polar channels moved onto x/y for its effective *coord*.

    A layer's effective coord is its own, else the chart's, the fallback
    Rust resolves through ``Layer::effective_coord``. The provenance of a
    moved channel moves with it: a polar binding the layer inherited becomes
    an inherited positional one, since on the wire it is an x/y binding.
    """
    from dataclasses import replace

    targets = dict(_polar_channel_targets(coord))
    if not any(channel in layer.encoding for channel in targets):
        return layer
    moved = tuple(targets[c] for c in layer.inherited_channels if c in targets)
    return replace(
        layer,
        encoding=_remap_polar_channels(coord, dict(layer.encoding)),
        inherited_channels=tuple(c for c in layer.inherited_channels if c not in targets),
        inherited_positional=layer.inherited_positional + moved,
    )


def _layer_drawn_encoding(layer: dict, encoding: dict) -> dict:
    """The channels of *encoding* the wire *layer*'s mark draws from the bound field itself.

    A non-positional channel is kept when the mark draws it (Rust's per-mark
    table, ``ferrum._core.mark_drawn_channels``). A positional channel is kept
    when the layer draws its own field there (Rust's drawn-column answer,
    ``ferrum._core.drawn_positional_columns``): a raster image draws its
    ``x``/``y`` through extent columns its rows hold instead of the bound
    fields, so neither is listed.
    """
    from ferrum._core import drawn_positional_columns, mark_drawn_channels

    mark = layer["mark"]
    drawn = set(mark_drawn_channels(mark))
    has_url = "url" in encoding or "url" in (layer.get("encoding") or {})
    kept = {}
    for channel, value in encoding.items():
        if channel in _POSITIONAL:
            field = value.get("field") if isinstance(value, dict) else None
            if field is None or drawn_positional_columns(mark, has_url, channel, field) == [field]:
                kept[channel] = value
        elif channel in drawn:
            kept[channel] = value
    return kept


class SpecBuildMixin:
    """Spec-building helper methods consumed by ``Chart.to_spec``.

    Mixed into ``Chart`` so the methods remain bound (``self`` is the chart) and
    every existing ``self._build_*`` / ``chart._build_*`` call site resolves
    without change.
    """

    def _with_polar_channels_lowered(self):
        """This chart with every polar binding moved onto the x/y channel it draws as.

        Runs once, before any statistic is lowered, on the chart's encoding
        and its written record (``_written_positional``, under its own coord)
        and on each layer's (under the layer's effective coord,
        :func:`_polar_lowered_layer`). An aggregate on ``theta`` or
        ``radius`` is then the aggregate on the channel the remap produces,
        grouped by the remapped channel set, and an inherited polar binding
        follows the positional inheritance rules: a layer reading a named
        output (its own statistic's) takes it from the chart's encoding as it
        takes x/y. Returns *self* when no coord is polar, otherwise a clone,
        so the caller's chart is never mutated.
        """
        layers = self._layers or []
        lowered_layers = [
            _polar_lowered_layer(layer, layer.coord if layer.coord is not None else self._coord)
            for layer in layers
        ]
        written = self._written_positional or {}
        moves_chart = any(
            polar in self._encoding or polar in written
            for polar, _ in _polar_channel_targets(self._coord)
        )
        if not moves_chart and all(a is b for a, b in zip(lowered_layers, layers)):
            return self
        new = self._clone()
        new._encoding = _remap_polar_channels(self._coord, dict(self._encoding))
        if self._written_positional is not None:
            new._written_positional = _remap_polar_channels(
                self._coord, dict(self._written_positional)
            )
        if self._layers is not None:
            new._layers = lowered_layers
        return new

    def _build_encoding_specs(
        self,
        resolved,
        enc: dict,
        override: OverridePayload | None = None,
        *,
        mark_zero: bool,
        color_source: str | None = None,
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
            Encoding dict after alias remapping, polar remapping and statistic
            lowering: an aggregated channel already names its output column.
        override :
            The resolved ``Chart.override`` payload, or ``None``. Its
            ``encoding``/``encoding_paths`` fields are applied FIRST per
            channel, before the bar zero-anchor injection (see the in-body
            comment below for why), so an override scale leaf beats the
            channel's own ``scale=`` setting (override wins the cascade,
            spec §7).
        mark_zero :
            Whether a bar chart's y-scale is anchored at zero: ``False`` once
            the chart's ``zero=False`` has been consumed, whichever door set it.

        Returns
        -------
        dict
            Partial ``kw`` dict containing only encoding-related keys:
            channel names mapped to ``EncodingSpec`` instances, plus
            ``"tooltip_fields"`` when applicable.
        """
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
        bar_value_on_x = _bar_value_on_x(enc, resolved._data)
        theta_axis = _theta_axis(resolved._coord)
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
            # Chart.override(<channel>_scale_<leaf>=...) wins over the channel's
            # own scale= (spec §7).
            #
            # Applied BEFORE the bar zero-anchor injection below (not after,
            # as an earlier revision had it) so the zero-anchor sees the
            # FINAL, fully-resolved effective scale type. Injecting zero
            # first and merging an ``override(y_scale_type=...)`` on top
            # used to leave a `zero: True` key stamped onto whatever type
            # the override switched to (e.g. `{"type": "log", "zero": True}`
            # from a plain-y `.override(y_scale_type="log")`) — the exact
            # gate-refused shape the zero-anchor fix below exists to avoid,
            # just reached through override composition instead of a direct
            # scale= dict.
            d = _with_override_scale(d, axis, override, _color_source(axis, color_source))
            d = _bar_zero_anchor(
                d,
                mark=resolved._mark,
                mark_zero=mark_zero,
                value_on_x=bar_value_on_x,
                axis=axis,
                enc=enc,
            )
            # D5b: strip ``stack=`` on marks that cannot honor stacking and emit
            # a one-time UserWarning (see ``_strip_unstackable``). Only a flat
            # chart's own mark draws this encoding: a layered chart's
            # chart-level encoding is its layers' parent, judged on each
            # layer that inherits it (``_layers_over_parent``).
            if not resolved._layers:
                _strip_unstackable(d, resolved._mark, polar_theta=axis == theta_axis)
            # `field` is positional; rest are keyword-only on EncodingSpec.__new__.
            # The Python-visible param name is `type_` (Rust signature `type_: Option<&str>`).
            field = d.pop("field")
            kw[axis] = _EncodingSpec(field, **d)
        return kw

    def _build_layers_list(
        self,
        layers: list | None = None,
        *,
        inherited_details: list[object | None] | None = None,
        override: OverridePayload | None = None,
    ) -> list:
        """Convert ``_layers`` to a list of JSON-serializable dicts for Rust.

        ``coerce_layers`` in Rust runs ``json.dumps()`` on each dict, so every
        value must be JSON-serializable (no PyO3 objects).

        Parameters
        ----------
        layers :
            Override layer list to serialize.  Defaults to ``self._layers``;
            ``to_spec`` passes the aggregate-resolved layers so the originating
            chart object is never mutated.
        inherited_details :
            Per layer, the chart-level ``detail`` channel that layer inherits
            when it has none of its own, or ``None`` (see
            :func:`ferrum.encoding._aliases.layered_chart_detail`, which owns
            the "which marks read ``detail``" rule).
        override :
            The resolved ``Chart.override`` payload, or ``None``. Merged
            into every channel a layer declares, exactly as
            ``_build_encoding_specs`` merges it into the chart-level encoding;
            a layer that does not declare the channel inherits the chart-level
            (already overridden) one. ``to_spec`` builds the chart-level
            encoding first, so a refused override refuses there, before any
            layer is serialized.
        """
        from ferrum.chart import _apply_inferred_type, _strip_unstackable
        from ferrum.encoding._aliases import alias_source, apply_channel_aliases

        out = []
        layer_list = (layers if layers is not None else self._layers) or []
        inherited = inherited_details or [None] * len(layer_list)
        for layer, inherited_detail in zip(layer_list, inherited, strict=True):
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
            # The grouping-only key is no channel: it skips the alias and
            # bucket passes and reaches the wire as the layer's `group`.
            grouping = layer_enc.pop(GROUPING_KEY, None)
            if inherited_detail is not None:
                layer_enc.setdefault("detail", inherited_detail)
            layer_mk = dict(layer.mark_kwargs) if layer.mark_kwargs else {}
            layer_color_source = alias_source(layer_enc, "color")
            layer_enc, layer_mk = apply_channel_aliases(layer_enc, layer_mk, layer.mark)

            # Bucket-partition safety net (single enforcement point, see
            # _warn_unbucketed_channels; mirrors _build_encoding_specs's
            # chart-level call exactly). Catches per-layer x_error*/
            # tooltip_field, and any theta/radius/theta2/radius2 that
            # `_with_polar_channels_lowered` did NOT claim -- i.e. a polar
            # channel on a layer whose effective coord is not polar, which is
            # exactly the case where it could never render.
            _warn_unbucketed_channels(layer_enc)

            encoding_dict: dict = {}
            # An independent-y layer (a `SecondaryY` slot, a
            # `LayerChart(resolve={"y": "independent"})` member) resolves its own
            # y scale, so a bar there takes the bar zero anchor from the same
            # producer a flat bar does; a layer on the shared axes is anchored by
            # Rust's layered rule instead.
            layer_theta_axis = _theta_axis(layer.coord if layer.coord is not None else self._coord)
            anchor_bar_value = layer.independent_y and layer_mark(layer.mark) == "bar"
            layer_value_on_x = anchor_bar_value and _bar_value_on_x(layer_enc, self._data)
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
                    d = _with_override_scale(
                        d, axis, override, _color_source(axis, layer_color_source)
                    )
                    if anchor_bar_value:
                        d = _bar_zero_anchor(
                            d,
                            mark="bar",
                            mark_zero=layer_mk.get("zero") is not False,
                            value_on_x=layer_value_on_x,
                            axis=axis,
                            enc=layer_enc,
                        )
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
                    _strip_unstackable(
                        d, layer_mark(layer.mark), polar_theta=axis == layer_theta_axis
                    )
                    # Presence, not truthiness: each key's own handler in
                    # ``_SPEC_DICT_HANDLERS`` already decides whether the key
                    # belongs in `d` (e.g. `_emit_axis`/`_emit_format` only add
                    # it for a real value). A truthy check here second-guesses
                    # that decision and silently drops legitimate falsy wire
                    # values -- notably ``title=""`` (`_emit_title`'s explicit
                    # "suppress the title" sentinel, ``None``-passed-by-caller
                    # forwarded as an empty string), which used to vanish here
                    # so a layer's title suppression silently did nothing
                    # (F-L09-08). The sentinel's job is now only what it says:
                    # draw no title. It is no longer also the way to drop a
                    # hidden axis's gutter -- since #182 layout reserves a title
                    # gutter only for an axis that draws one, which is why
                    # `plots/matrix.py`'s dendrogram `title=""` workaround is
                    # gone.
                    #
                    # This allowlist is still hand-curated against Rust's
                    # ``EncodingSpec`` (`crates/ferrum-core/src/spec/encoding.rs`),
                    # not derived from it, so it can drift from what the struct
                    # actually accepts. ``aggregate`` is dead weight here (no
                    # handler ever emits it into `d`, so it can never match
                    # `opt_key in d`); kept because removing it is a separate
                    # cleanup from this allowlist's job. ``condition`` was
                    # removed (#237, 2026-09-20): `d` (`ch.to_encoding_spec_dict()`)
                    # never carries a ``"condition"`` key any more -- the
                    # per-channel condition path has exactly one mechanism now,
                    # ``ChartSpec.conditionals``, populated upstream by
                    # `_strip_layer_conditionals` before `d` is built -- so
                    # forwarding it here would have been dead code even before
                    # the key existed to forward.
                    for opt_key in (
                        "title",
                        "aggregate",
                        "scheme",
                        "format",
                        "scale",
                        "axis",
                        "legend",
                        "sort",
                        "stack",
                        "impute",
                        "bin_channel",
                    ):
                        if opt_key in d:
                            enc_json_dict[opt_key] = d[opt_key]
                    # `format_type` needs its own line, not the loop above:
                    # `EncodingSpec.format_type` carries `#[serde(rename =
                    # "formatType")]` plus `deny_unknown_fields`
                    # (encoding.rs:792), so forwarding the Python-side
                    # snake_case key verbatim makes Rust reject the whole
                    # layer with "unknown field `format_type`" -- for ANY
                    # layered channel with a format that derives one (e.g.
                    # `X(field, format="percent")`, where `_emit_format`
                    # derives `format_type="number"`). `_build_encoding_specs`
                    # (the chart-level path) never hits this: it passes `**d`
                    # straight into the PyO3 `_EncodingSpec` constructor, which
                    # takes the snake_case Python name directly and handles
                    # the wire rename internally. Only this JSON-dict wire
                    # path needs the rename spelled out by hand.
                    if "format_type" in d:
                        enc_json_dict["formatType"] = d["format_type"]
                    encoding_dict[axis] = enc_json_dict
                elif isinstance(ch, str):
                    d = _with_override_scale(
                        {"field": ch}, axis, override, _color_source(axis, layer_color_source)
                    )
                    if anchor_bar_value:
                        # The shorthand's declared type decides the anchor, as
                        # a channel object's `type_` does; the wire dict keeps
                        # the string's own spelling.
                        _field, declared = _encoding_field_and_type(ch)
                        scale = _bar_zero_anchor(
                            {**d, "type_": declared},
                            mark="bar",
                            mark_zero=layer_mk.get("zero") is not False,
                            value_on_x=layer_value_on_x,
                            axis=axis,
                            enc=layer_enc,
                        ).get("scale")
                        if scale is not None:
                            d = {**d, "scale": scale}
                    encoding_dict[axis] = d
            # Rust's position grouping reads `group` (dodge and stack slots by
            # the field the parent draws as color, by its bins when it bins);
            # no other reader sees it.
            group = grouping.wire() if grouping is not None else None
            if group is not None:
                encoding_dict[GROUPING_KEY] = group
            layer_dict: dict = {
                "mark": layer_mark(layer.mark),
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
            # Phase 9c — per-layer position adjustment. Serialize value classes
            # via ``to_spec_dict``.
            if layer.position is not None:
                layer_dict["position"] = (
                    layer.position.to_spec_dict()
                    if hasattr(layer.position, "to_spec_dict")
                    else layer.position
                )
            if layer.name is not None:
                layer_dict["name"] = layer.name
            # Secondary-y-axis wire flag (GH #52 Task 4). Only emitted when
            # True -- absent/false deserializes and renders byte-identically
            # to a pre-#52 spec (see Layer::independent_y in Rust spec/layer.rs).
            if layer.independent_y:
                layer_dict["independent_y"] = True
            # Per-layer coordinate system (`#172`(1)). Only emitted when the
            # layer carries one of its own -- absent means "inherit the chart's",
            # which is every layer of every chart that does not mix coordinate
            # systems, so those specs serialize byte-identically to a pre-#172
            # wire (see `Layer::coord` in Rust `spec/layer.rs`). Serialized
            # through `coord_wire`, the one canonical-wire-form rule
            # (`ferrum.coord`), because this dict is deserialized by serde and
            # has no PyO3 constructor to widen `orient="horizontal"`'s bare
            # `"flip"` token the way the chart-level slot's does.
            if layer.coord is not None:
                layer_dict["coord"] = _coord_wire(layer.coord)
            # The public entry that emitted this layer, voiced by Rust's
            # per-layer render refusals; absent for a layer the caller
            # declared directly, so those specs are unchanged.
            if layer.entry is not None:
                layer_dict["entry"] = layer.entry
            # Positional channels copied from the operand chart; absent when
            # none was copied, so every other spec is unchanged on the wire.
            if layer.inherited_positional:
                layer_dict["inherited_positional"] = list(layer.inherited_positional)
            # Non-positional channels the layer carries from a parent, as the
            # wire channels they draw as; absent when it carries none, so
            # every other spec is unchanged on the wire.
            inherited_channels = _inherited_wire_channels(layer, encoding_dict)
            if inherited_channels:
                layer_dict["inherited_channels"] = inherited_channels
            # Absent unless the layer reads its chart's data through a named
            # output, so every other spec is unchanged on the wire.
            if layer.reads_chart_data:
                layer_dict["reads_chart_data"] = True
            out.append(layer_dict)
        return out

    def _build_facet_dict(self) -> dict:
        """Convert internal _facet to the JSON dict Rust's FacetSpec expects.

        Delegates to :func:`ferrum._facet.build_facet_dict`; the facet machinery
        lives in ``_facet.py`` (cohesion finding C2/C10). On a ``+``
        composition over different frames, each operand frame lacking a facet
        field is declared under ``repeat`` by its row marker, so Rust draws
        its rows in every panel along that field
        (:func:`ferrum._facet.facet_repeat`).
        """
        d = _build_facet_dict_fn(self._facet, self._data)
        repeat = _facet_repeat_fn(d, self._frame_selectors)
        if repeat:
            d["repeat"] = repeat
        return d

    def _infer_facet_cardinality(self, col_name: str | None) -> int:
        """Return the number of distinct values in *col_name* from self._data.

        Delegates to :func:`ferrum._facet.infer_facet_cardinality`.
        """
        return _infer_facet_cardinality_fn(self._data, col_name)

    @staticmethod
    def _resolve_statistic(resolved, effective_transforms: list, enc: dict) -> _ResolvedStatistic:
        """Run a single chart's encoding statistic: its one ``Aggregate`` or ``Bin`` (spec §6.4).

        The chart is one layer, so its statistic is the plan
        :func:`~ferrum._encode_lowering._statistic_plan` reads off *enc*,
        grouped like every layer's, never a sequence of steps: a second
        ``Aggregate`` would read the first one's output, where its source
        column is gone, and a ``Bin`` replaces the batch schema with its own
        columns. It runs after every step the caller wrote with
        ``.transform(...)``, whether written before or after ``encode()``,
        behind the chain's statistics marker and after any statistic the
        chart's mark lowered. The ``aggregate=``/``bin=`` sentinels
        ``encode()`` put on the transform list are dropped; the plan
        computes what the encoding asks for, and a sentinel a later
        ``encode()`` left behind asks for nothing.

        Parameters
        ----------
        resolved :
            The resolved ``Chart``, whose facet fields join the groupby.
        effective_transforms :
            The current transform list, which may hold the sentinels.
        enc :
            The chart's encoding dict, after alias and polar remapping. Never
            mutated; the encoding the chart draws comes back on the result.

        Returns
        -------
        _ResolvedStatistic
            Both inputs unchanged when the list holds no sentinel.
        """
        if not any(isinstance(t, (_PendingAggregate, _PendingBin)) for t in effective_transforms):
            return _ResolvedStatistic(effective_transforms, enc)
        plan = _statistic_plan(enc, resolved._facet)
        own, lowered = _split_chain(effective_transforms)
        if plan.computes:
            lowered = (*lowered, plan.step())
        return _ResolvedStatistic([*own, *_with_statistics_marker(list(lowered))], plan.encoding)

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

        A chart-level ``tooltip`` is the parent's: written on the chart a
        layer is attached to, or on a composition itself, never copied up
        from a layer (a ``+`` operand's own tooltip stays on its layers). It
        therefore reaches every layer that binds none, through Rust's
        chart-level fallback (``Encoding::inherit_from``), and short-circuits
        the per-layer loop below.

        Each layer the loop fills lists either its own bindings' fields,
        which stay its own, or, when its own drawn encoding names no field,
        the fields of everything it draws over the chart-level encoding
        (without x/y/x2/y2 for a layer reading a named output, as Rust's
        ``inherit_non_positional``). That second list mirrors the parent's
        bindings, so it is listed in the layer's ``inherited_channels`` as
        ``tooltip_fields``, and the column gate judges it as the inherited
        read it stands for. It may be empty: an empty list still binds the
        tooltip, so the chart-level auto list never reaches a layer that
        draws none of its fields. A layer's auto fields name only the
        channels its mark draws (:func:`_layer_drawn_encoding`).

        A field selection's ``fields`` (GH #58) are match keys, not
        bindings, and are never written here: Rust's one tooltip reader
        appends them after each layer's own entries, skipping a field the
        layer's rows lack, for both output kinds. A layer that itself
        carries an explicit ``tooltip``/``tooltip_fields`` is left untouched
        by the loop. An unlayered chart has no ``kw["layers"]`` key, so only
        the chart-level fields apply.

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
        if "tooltip" in enc or "tooltip_fields" in enc:
            # The parent's written tooltip -- it reaches every layer that
            # binds none via Rust's chart-level tooltip_fields fallback
            # (Encoding::inherit_from).
            return kw
        auto_fields = _auto_tooltip_fields(enc)
        if auto_fields:
            kw.setdefault("encoding", {})["tooltip_fields"] = auto_fields

        for layer in kw.get("layers") or []:
            layer_enc = layer.setdefault("encoding", {})
            if "tooltip" in layer_enc or "tooltip_fields" in layer_enc:
                continue
            # A layer lists its own fields: its wire encoding less what it
            # carries from a parent (``inherited_channels`` and
            # ``inherited_positional``, the wire twin of
            # ``_layer_transforms._own_channels``).
            carried = set(layer.get("inherited_channels") or ()) | set(
                layer.get("inherited_positional") or ()
            )
            own = {channel: value for channel, value in layer_enc.items() if channel not in carried}
            own_fields = _auto_tooltip_fields(_layer_drawn_encoding(layer, own))
            if own_fields:
                layer_enc["tooltip_fields"] = own_fields
                continue
            parent = {
                channel: value
                for channel, value in enc.items()
                if channel != "tooltip_fields"
                and ("data_source" not in layer or channel not in _POSITIONAL)
            }
            layer_enc["tooltip_fields"] = _auto_tooltip_fields(
                _layer_drawn_encoding(layer, {**parent, **layer_enc})
            )
            layer["inherited_channels"] = [*layer.get("inherited_channels", ()), "tooltip_fields"]
        return kw
