"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

if TYPE_CHECKING:
    from ferrum._interactive import InteractiveChart
    from ferrum.selection import Selection

from ferrum._coerce import to_arrow_table, to_polars
from ferrum._desugar import _resolve_pending_impl
from ferrum._layer import _POSITIONAL_WIRE, _Layer, _PendingMark, inherited_positional_taken
from ferrum._compose import (
    apply_post_build_positional,
    attach_to_rows,
    carried_by,
    chart_frames,
    chart_operands,
    composite_source,
    combined_frames,
    known_chart_data_columns,
    lower_operands,
    merged_frames,
    reached_by_composition,
    refuse_missing_statistic_inputs,
    regrouped,
)
from ferrum._encode_lowering import GROUPING_KEY
from ferrum._layer_transforms import (
    _Branch,
    _ComposedStep,
    _NamedTransform,
    _StatisticsMarker,
    _advances_chain,
    _composition_encoding,
    _layer_pending_aggregates,
    _layer_pending_bins,
    _layers_over_parent,
    _own_channels,
    _resolve_layer_statistics,
    _scoped_to_layers,
    _split_chain,
    _statistic_columns,
    _statistics_start,
    _transforms_to_json_list_named,
    _wire_statistic_grouped_fields,
    _with_statistics_marker,
)
from ferrum._configure_mixin import ConfigureMixin
from ferrum.marks._chart_methods_diagnostic import DiagnosticMarksMixin
from ferrum.marks._chart_methods_statistical import (
    StatisticalMarksMixin,
    _refuse_primitive_mark_on_layered_chart,
)
from ferrum._render import _RenderMixin
from ferrum._shorthand import parse_shorthand
from ferrum._spec_build import SpecBuildMixin, _encoding_field_and_type
from ferrum._spec_view import _SpecView
from ferrum.encoding._channel_policy import _FACET_CHANNELS, conditional_wire_channel
from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin, _validate_sort_object
from ferrum.marks.base import _FIGURE_ONLY_KINDS, MarkBase
from ferrum.position import stack_honored, warn_stack_ignored

# Re-exported from this module so existing ``from ferrum.chart import ...``
# importers keep resolving after the cohesion split (CHART-01/02): the canonical
# homes are now ``_desugar.py`` and ``_layer_transforms.py``.  Only the symbols
# with a remaining ``from ferrum.chart import`` consumer are re-exported here;
# the rest are imported directly from their new modules by their callers.
from ferrum._desugar import (  # noqa: F401  (back-compat re-export)
    _resolve_cat_axis,
    _split_style_kwargs,
)

from ferrum._layer import _FLAT_MARK_METHODS


def _strip_unstackable(d: dict, mark: str | None, *, polar_theta: bool) -> None:
    """Strip an unsupported ``stack=`` from the encoding dict of a mark that draws it.

    The mark honors it per :func:`ferrum.position.stack_honored`; on any
    other mark and channel the stacking path silently drops marks, so the
    ``stack=`` is popped (the mark renders unstacked) and warned once
    (:func:`ferrum.position.warn_stack_ignored`). A falsy ``stack``
    (``None``/``False``) means "do not stack" and is left untouched. Shared
    by the flat chart's encoding and each layer's own.
    """
    if d.get("stack") and not stack_honored(mark, polar_theta=polar_theta):
        warn_stack_ignored(d.pop("stack"), mark)


_logger = logging.getLogger(__name__)


from ferrum._entry import _voiced
from ferrum._validate import coerce_positive_int, coerce_viewport_value
from ferrum._facet import (
    FacetDeclaration,
    _Facet,
    build_facet_dict as _build_facet_dict_fn,
    infer_facet_cardinality as _infer_facet_cardinality_fn,
)
from ferrum.encoding import _channel_class_map, _channel_class_for, _apply_channel_aliases
from ferrum.encoding._aliases import alias_source
from ferrum.encoding._resolve import channel_class, channel_for_column, channel_for_value
from ferrum.encoding._aliases import layered_chart_detail
from ferrum.selection import ConditionalSpec


from ferrum.composition import (
    _bind_operand_conditionals,
    _carry_layer_coord,
    _carry_layer_position,
    _ChartOperatorMixin,
    _conditional_layer_scopes,
    _expand_layers,
    _validate_share_modes,
    _warn_on_layer_conflicts,
)


def _viewport_dimension(entry: str, value: Any, *, param: str) -> Any:
    """Return a ``width``/``height`` value to store, refusing a malformed one as *entry*.

    The rule is :func:`ferrum._validate.coerce_viewport_value` (``None`` and
    ``"container"`` unchanged, any other value a finite, positive number
    stored as a ``float``). The refusal is an
    :class:`~ferrum._entry.EntryRefusal` reading ``{entry}({param}=...) ...``,
    so a figure function or composite that sets the size on the caller's
    behalf re-voices it as itself.
    """
    return _voiced(entry, lambda voice: coerce_viewport_value(f"{voice}({param}=...)", value))


# The channels a written record keeps: the positional ones, and the polar
# ones a ``CoordPolar`` draws as x/y/x2/y2 once ``to_spec`` lowers them.
_WRITTEN_CHANNELS = _POSITIONAL_WIRE + ("theta", "theta2", "radius", "radius2")


def _written_bindings(chart: "Chart") -> dict:
    """Return the positional and polar bindings *chart*'s callers wrote, latest per channel.

    A chart ``chart + chart`` produced carries them in ``_written_positional``:
    its left operand's at the merge, then each ``.encode(...)`` on the
    composition. So does a host whose mark statistics moved its tail when
    layers were attached (:func:`_attach_host`): its pre-resolution bindings,
    then each later ``.encode(...)``. Any other chart's own pre-resolution
    encoding is its written bindings (a composite that rewrites x/y is still
    pending when ``+`` reads it). A channel absent from the result is one no
    caller wrote.
    """
    if chart._written_positional is not None:
        return dict(chart._written_positional)
    return {ch: chart._encoding[ch] for ch in _WRITTEN_CHANNELS if ch in chart._encoding}


def _attach_host(chart: "Chart") -> "Chart":
    """Return a resolved clone of *chart* that layers can be attached to.

    The chart's data is normalized before its pending mark resolves, so
    unsupported data is refused here rather than replaced by a layer's.
    When the statistics the resolution appends move the tail off the
    chart's own rows, the clone keeps the positional bindings the caller
    wrote, which the resolution may have rewritten, as the record an
    attached layer's x/y come from (:func:`_written_bindings`).
    """
    pre = chart._with_polars_data()
    resolved = pre._resolve_pending()
    new = resolved._clone()
    if resolved is pre:
        return new
    _, statistics = _split_chain(list(new._transforms))
    if any(_advances_chain(t) for t in statistics) and new._written_positional is None:
        new._written_positional = _written_bindings(pre)
    return new


def _warn_ignored_details(
    chart: "Chart",
    chart_detail: Any,
    layers: list | None,
    kw: dict,
    transforms: list,
) -> None:
    """Report each ``detail`` field of *chart* that no mark draws and no statistic groups by.

    Decided once the chart's statistics are lowered, after any composition
    regrouped its operands, so a detail that groups a composite, an
    encoding aggregate, a figure's statistic or a label's aggregate is
    never reported, whatever the call order. *chart_detail* is a layered
    chart's own detail, *layers* the lowered layers and *kw* the spec
    keywords built for them (``mark_style`` / ``layers``), and
    *transforms* the lowered pipeline.
    """
    from ferrum.encoding._aliases import _channel_field, warn_ignored_details

    if layers is None:
        details = [(_channel_field(chart._encoding.get("detail")), chart._mark)]
        drawn = {kw.get("mark_style", {}).get("detail")}
    else:
        details = [(_channel_field(chart_detail), layers[0].mark if layers else None)]
        for layer in layers:
            details.append((_channel_field(layer.encoding.get("detail")), layer.mark))
            grouping = layer.encoding.get(GROUPING_KEY)
            if grouping is not None:
                details += [
                    (_channel_field(binding), layer.mark)
                    for channel, binding in grouping.bindings
                    if channel == "detail"
                ]
        drawn = {(entry.get("mark_style") or {}).get("detail") for entry in kw.get("layers", [])}
    details = [(field, mark) for field, mark in details if isinstance(field, str)]
    if not details:
        return
    grouped = _wire_statistic_grouped_fields(_transforms_to_json_list_named(list(transforms)))
    warn_ignored_details(details, drawn, grouped)


def _fill_chart_positional(layers: list, chart: "Chart") -> list:
    """Copy *chart*'s positional channels onto each layer that takes them from it.

    A layer with ``positional_source`` set reads a named output, and Rust
    passes such a layer none of the chart's positional channels, so each of
    x, y, x2 and y2 its source binds, the layer does not, and the layer's mark
    draws (:func:`ferrum._layer.inherited_positional_taken`) is copied onto the
    layer and listed in its ``inherited_positional``: Rust treats a listed channel as inherited, not
    owned, exactly as it treats a chart channel a layer reading the chart's
    rows inherits. The source is the chart's encoding for ``"encoding"`` (an
    operand's tail reader, drawing as on its own chart) and the chart's
    written bindings for ``"written"`` (a right operand's layer, and a layer
    attached to the chart, see :func:`_written_bindings`). Every other layer
    is returned unchanged.
    """
    from dataclasses import replace as _dc_replace

    written = None
    out = []
    for layer in layers:
        if layer.positional_source is None:
            out.append(layer)
            continue
        if layer.positional_source == "encoding":
            source = chart._encoding
        else:
            if written is None:
                written = _written_bindings(chart)
            source = written
        offered = [ch for ch in _POSITIONAL_WIRE if ch in source and ch not in layer.encoding]
        taken = inherited_positional_taken(layer.mark, _own_channels(layer), offered)
        filled = {ch: copy.copy(source[ch]) for ch in offered if ch in taken}
        out.append(
            _dc_replace(
                layer,
                encoding={**layer.encoding, **filled},
                inherited_positional=layer.inherited_positional + tuple(filled),
            )
        )
    return out


def _reading_operand_tail(layers: list) -> list:
    """*layers*, each tail reader taking the positional channels it does not bind from its chart.

    A layer reading its operand's tail (no ``data_source``, no chart-data
    role) reads it through a named output once its operand is lowered, where
    Rust passes it none of the chart's positional channels; it takes them
    from its operand chart's encoding (``positional_source="encoding"``,
    :func:`_fill_chart_positional`). A layer already given a source keeps it.
    """
    return [
        replace(l, positional_source="encoding")
        if l.data_source is None and not l.chart_data and l.positional_source is None
        else l
        for l in layers
    ]


def _take_layers(new: "Chart") -> list:
    """Turn the attached host *new* into layers, leaving its encoding statistics to them.

    The host's encoding aggregates and bins leave its chain (each layer
    computes its own, see :func:`ferrum.composition._expand_layers`).
    Returns the host's layers; ``new._transforms`` is updated.
    """
    layers, chain = _expand_layers(new)
    new._transforms = chain
    return layers


def _desugar_secondary_y(chart: "Chart", feature: "SecondaryY") -> "Chart":
    """Desugar ``chart + SecondaryY(...)`` into an appended independent-y layer.

    The base chart's existing layer(s) are unchanged (a multi-layer base
    keeps its internal sharing), plus one appended layer: mark
    ``feature.mark``, ``y`` encoding on ``feature.field`` (with
    ``feature.axis``/``feature.scale`` attached), color literal
    ``feature.color``, opacity ``feature.opacity``, flagged
    ``independent_y=True`` so Rust resolves its y-scale independently and
    renders it as a stacked right axis (layer 0 is always the primary/left
    axis).

    The secondary layer is attached to the base chart, as a ``Chart.layer``
    layer is: it reads the base chart's data, its rows after its
    ``.transform(...)`` steps and never a composite mark's statistics, and
    it binds only ``y``. Its ``x`` is the one the caller wrote for the base
    chart, read when the spec is built, so an ``.encode(x=...)`` after this
    call reaches it (on a composition, the composition's written ``x``).

    Refusals, before any layer is built:

    - ``feature.mark`` must be a primitive mark (``_FLAT_MARK_METHODS``): a
      composite mark name would bypass the ``mark_*()`` desugar and reach
      Rust as an unknown primitive.
    - The base chart must have a written ``x``.
    - When the base chart's data is its source with only named steps
      applied, its columns are known here, so a field outside them is
      refused now: with the statistic message when the resolved base binds
      that field for its own layers (a histogram's ``count``), else with the
      plain one. Otherwise Rust's column gate refuses at render.
    """
    from ferrum.encoding.positional import Y as _Y

    if feature.mark not in _FLAT_MARK_METHODS:
        raise ValueError(
            f"SecondaryY: mark={feature.mark!r} is not a primitive mark; must be "
            f"one of {sorted(_FLAT_MARK_METHODS)}. A composite mark name (e.g. "
            "'boxplot', 'violin') cannot be used here -- for a composite overlay "
            "on a secondary axis, use "
            "LayerChart(chart, other_chart, resolve={'y': 'independent'}) instead."
        )

    new = _attach_host(chart)
    if _written_bindings(new).get("x") is None:
        raise ValueError(
            "SecondaryY: the base chart must have an x encoding to inherit "
            "-- call .encode(x=...) before adding SecondaryY(...)."
        )
    _refuse_secondary_field(new, feature.field)

    y_kwargs: dict = {}
    if feature.axis is not None:
        y_kwargs["axis"] = feature.axis
    if feature.scale is not None:
        y_kwargs["scale"] = feature.scale
    y_encoding = _Y(feature.field, **y_kwargs)

    mark_overrides: dict = {}
    if feature.color is not None:
        mark_overrides["color"] = feature.color
    if feature.opacity is not None:
        mark_overrides["opacity"] = feature.opacity
    mark_kwargs = (
        MarkBase(feature.mark, **mark_overrides).to_mark_kwargs_dict() if mark_overrides else None
    )

    secondary_layer = _Layer(
        mark=feature.mark,
        encoding={"y": y_encoding},
        mark_kwargs=mark_kwargs,
        independent_y=True,
        positional_source="written",
        chart_data=True,
    )
    new._layers = _take_layers(new) + [secondary_layer]
    return new


def _refuse_secondary_field(base: "Chart", field: str) -> None:
    """Refuse a ``SecondaryY`` field absent from the rows its secondary layer reads.

    The secondary layer reads the base chart's data, so the check reads
    ``known_chart_data_columns``: on a ``+`` composition, the left
    operand's columns, never the right operand's. Checked when Python knows
    those columns (the base's data is its source with only named steps applied; an aggregate or bin on
    the base's own encoding is computed per layer and leaves them alone). A
    field that is the output column of such an encoding statistic, or that a
    composite base binds on its resolved chart encoding or on any of its
    layers, is a statistic computed for the base's own layers.
    """
    columns = known_chart_data_columns(base)
    if columns is None or field in columns:
        return
    if field in _statistic_columns(base._encoding):
        raise ValueError(
            f"SecondaryY: field {field!r} is a statistic the base chart's encoding computes "
            "for its own layers, not a column of the base chart's data (SecondaryY draws "
            "the base chart's rows after its .transform(...) steps). Compute it with "
            ".transform(...) (e.g. fm.transform_aggregate), or draw it from its own chart "
            "with LayerChart(base, other, resolve={'y': 'independent'}). "
            f"Available columns: {columns}"
        )
    if base._composite_kind is not None and field in _bound_fields(base):
        figure_call = _FIGURE_ONLY_KINDS.get(base._composite_kind)
        builder = figure_call.name if figure_call is not None else f"mark_{base._composite_kind}"
        raise ValueError(
            f"SecondaryY: field {field!r} is a statistic the base chart's "
            f"{builder} computes for its own layers, not a column of "
            "the base chart's data (SecondaryY draws the base chart's rows after its "
            ".transform(...) steps). Compute it with .transform(...) (e.g. fm.Bin, "
            "fm.transform_aggregate), or draw it from its own chart with "
            "LayerChart(base, other, resolve={'y': 'independent'}). "
            f"Available columns: {columns}"
        )
    raise ValueError(
        f"SecondaryY: field {field!r} is not a column of the "
        "base chart's data (SecondaryY draws from the same table as "
        f"the base chart); available columns: {columns}"
    )


def _bound_fields(chart: "Chart") -> set:
    """Return the fields *chart*'s encoding and its own mark's layers bind, in any spelling.

    A layer attached to the chart (holding the chart-data role) is not one
    of the mark's layers, so its fields are not the mark's statistics.
    """
    encodings = [chart._encoding] + [
        layer.encoding for layer in chart._layers or [] if not layer.chart_data
    ]
    return {
        _encoding_field_and_type(value)[0]
        for encoding in encodings
        for value in encoding.values()
        if isinstance(value, (str, ChannelBase))
    }


def _with_polars_operands(lhs: "Chart", rhs: "Chart") -> "tuple[Chart, Chart]":
    """Normalize both ``+`` operands' data to polars, collecting shared data once.

    When both operands carry the same data object, the right operand reuses
    the left operand's converted frame, so a shared ``LazyFrame`` is collected
    once per ``+`` and the frame registry's identity fast path holds
    (:func:`ferrum._compose.merged_frames`).
    """
    lhs_in = lhs._with_polars_data()
    if rhs._data is lhs._data and lhs_in is not lhs:
        rhs_in = rhs._clone()
        rhs_in._data = lhs_in._data
        return lhs_in, rhs_in
    return lhs_in, rhs._with_polars_data()


def _annotations_from_primitive(primitive: object) -> list:
    """Expand a ``Chart._annotation_primitive`` value into ``Annotate`` containers.

    Most ``annotate_*`` helpers set a single primitive (``AnnotationLine``,
    ``AnnotationText``, etc.), which becomes one ``Annotate``. A labelled
    ``annotate_hline``/``annotate_vline`` sets a *list* of primitives (the
    reference line plus its label text) so that ``+`` overlay — which reads
    only ``_annotation_primitive`` and discards the helper chart's mark
    layers — carries both. Each list element becomes its own ``Annotate``
    entry in the destination chart's ``_annotations``.
    """
    from ferrum.annotation.container import Annotate

    if isinstance(primitive, list):
        return [Annotate(item) for item in primitive]
    return [Annotate(primitive)]


def _json_equal(first: object, second: object) -> bool:
    """Return True when *first* and *second* serialize to the same JSON text.

    The one definition of value identity "on the wire": a Python ``list``
    and a ``tuple`` with the same elements serialize to the same JSON array
    and count as equal, while a Python ``int`` and ``bool`` that compare
    equal under ``==`` (``1 == True``) serialize to different JSON
    (``1`` vs. ``true``) and count as different. Shared by
    :func:`_selection_wire_equal` (whole-declaration identity) and
    :func:`_selection_redefinition_message` (per-field diff), so both read
    the same notion of "differs".
    """
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def _selection_wire_equal(first: "Selection", second: "Selection") -> bool:
    """Return True when *first* and *second* serialize to the same wire shape.

    Compares through :meth:`Selection.to_param_spec_dict` via
    :func:`_json_equal`, rather than Python ``==`` on the dataclasses. A
    ``fields=['x']`` declaration and a ``fields=('x',)`` declaration are
    different Python values (list vs. tuple) but serialize to the same
    JSON array -- the only distinction the Rust composite renderer's
    wire-JSON ``SelectionRedefined`` check can see. Comparing the
    dataclasses directly is stricter than the wire and would refuse input
    the wire itself would merge.
    """
    return _json_equal(first.to_param_spec_dict(), second.to_param_spec_dict())


def _append_unique_by_name(seq: list, item: object, *, entry: str) -> None:
    """Append *item* to *seq* if no element with the same ``.name`` is present.

    Preserves insertion order (first-seen wins).  Mutates *seq* in place.
    *item* must have a ``.name`` attribute; *seq* may contain ``None`` entries
    which are skipped during the existence check. *entry* names the public
    call the caller was reached through (``"add_selection()"``,
    ``"add_params()"``, ``"encode()"``, ``"Chart.layer()"``,
    ``"Chart.conditional()"``, or ``"chart + chart"``) -- required, not
    defaulted, so the refusal below cannot silently inherit the wrong voice,
    the same discipline :func:`_append_conditional`'s *context* parameter
    already enforces for conditional collisions.

    Every current caller appends a ``Selection``.  A name collision is
    checked for identity of *definition on the wire* (:func:`_selection_wire_equal`),
    not identity of name and not Python ``==``: two ``Selection`` objects
    sharing a name are expected to be the same declaration -- the
    linked-views case, where one ``Selection`` is handed to
    ``add_selection()``/``add_params()``/``encode()`` on more than one
    chart, and the charts later merge via ``+`` (``Chart.__add__``). That
    case is accepted silently. A colliding ``Selection`` whose wire form
    actually differs raises ``ValueError`` naming the differing fields --
    the within-one-chart counterpart of the Rust composite renderer's
    ``CompositeRenderError::SelectionRedefined``, which refuses the same
    shape across charts.
    """
    for existing in seq:
        if existing is None or not hasattr(existing, "name"):
            continue
        if existing.name != item.name:
            continue
        if not _selection_wire_equal(existing, item):
            raise ValueError(_selection_redefinition_message(existing, item, entry=entry))
        return
    seq.append(item)


def _selection_redefinition_message(first: "Selection", second: "Selection", *, entry: str) -> str:
    """Format the refusal for two same-named selections with different definitions.

    Leads with *entry*, the public call that raised it, in the same
    ``"{context}: ..."`` voice :func:`_conditional_collision_message` already
    uses for the sibling conditional-collision refusal. Lists only the
    fields that actually differ, in the same "declares ..., declares ..."
    voice as the Rust composite renderer's ``SelectionRedefined`` message:
    the full declaration would bury the one or two fields that disagree. A
    differing ``kind`` (``"point"`` vs. ``"interval"``) is reported as
    ``type`` alone and nothing else, exactly as the Rust message does when
    the kinds differ -- once the kind itself disagrees, no other field is
    comparable.

    The diff comes from the same wire dicts
    (:meth:`Selection.to_param_spec_dict`) and the same :func:`_json_equal`
    that :func:`_selection_wire_equal` uses to decide the refusal fires at
    all.
    """
    name = first.name
    first_wire = first.to_param_spec_dict()
    second_wire = second.to_param_spec_dict()
    if first_wire["kind"] != second_wire["kind"]:
        first_decl = f"type={first_wire['kind']!r}"
        second_decl = f"type={second_wire['kind']!r}"
    else:
        first_select = first_wire["select"]
        second_select = second_wire["select"]
        ordered_keys: list[str] = []
        seen: set[str] = set()
        for key in (*first_select.keys(), *second_select.keys(), "bind"):
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)
        diffs = []
        for key in ordered_keys:
            if key == "bind":
                v1, v2 = first_wire.get("bind"), second_wire.get("bind")
            else:
                v1, v2 = first_select.get(key), second_select.get(key)
            if not _json_equal(v1, v2):
                diffs.append((key, v1, v2))
        first_decl = ", ".join(f"{key}={v1!r}" for key, v1, _ in diffs)
        second_decl = ", ".join(f"{key}={v2!r}" for key, _, v2 in diffs)
    return (
        f"{entry}: selection {name!r} is declared twice with different definitions: "
        f"first declares {first_decl}, second declares {second_decl}. "
        f"One name is one selection on this chart: pass the same Selection object "
        f"to every call that declares {name!r}, or give the second one its own name."
    )


def _wire_channel_of_conditional(spec: "ConditionalSpec") -> str:
    """Return the wire channel a ``ConditionalSpec`` will serialize under.

    A spec stamped by ``encode()`` or :meth:`Chart.conditional` carries its
    channel; an unstamped one infers it from its branches
    (``selection._infer_conditional_channel``) exactly as
    ``ConditionalSpec.to_spec_dict`` does, so the collision check sees the
    channel the wire will actually carry. Split out of
    :func:`_channel_claimed_by` so a caller that already knows
    *spec* is a ``ConditionalSpec`` (:meth:`Chart.conditional`) gets a
    non-``Optional`` result with no narrowing needed at the call site.
    """
    from ferrum.selection import _infer_conditional_channel

    if spec.channel is not None:
        return spec.channel
    return _infer_conditional_channel(spec.if_selected, spec.if_not)


def _channel_claimed_by(spec: object) -> Optional[str]:
    """Return the wire channel *spec* will serialize under, or ``None``.

    Returns ``None`` for anything that is not a ``ConditionalSpec`` --
    ``Chart.conditional()`` is annotated to accept any object, so the
    conditional list is not guaranteed homogeneous, and a foreign entry
    should not be treated as claiming a channel.
    """
    if not isinstance(spec, ConditionalSpec):
        return None
    return _wire_channel_of_conditional(spec)


def _conditional_on_wire(chart: "Chart", wire: str) -> Optional["ConditionalSpec"]:
    """Return the conditional already on *chart* for wire channel *wire*, if any.

    The single "is this channel taken, and by what" query shared by all
    three collision sites -- ``_append_conditional``'s cross-call check,
    ``Chart.conditional()``'s, and ``Chart.__add__``'s merge check. A caller
    checks ``is not None`` for "is it taken" and reuses the same result to
    name the earlier declaration in the collision message, so no site scans
    ``chart._conditionals`` twice.
    """
    return next((c for c in chart._conditionals if _channel_claimed_by(c) == wire), None)


def _conditional_declaration_label(spec: "ConditionalSpec", *, whose: str) -> str:
    """Name a conditional declaration that carries no parameter spelling of its own.

    A spec already on a chart, or one handed whole to ``Chart.conditional()``
    or merged by ``+``, does not record the entry point that declared it, so
    it is named by *whose* it is (``"the chart's existing"``, ``"this
    call's"``, ``"the left chart's"``, ``"the right chart's"``) and the
    selection driving it (every ``ConditionalSpec`` carries a
    ``selection_name``). The role keeps the two sides of a collision apart
    when both are driven by the same selection.
    """
    return f"{whose} conditional on selection {spec.selection_name!r}"


def _conditional_collision_message(
    *, context: str, wire: str, first: str, second: str, detail: str = ""
) -> str:
    """One message for every one-conditional-per-wire-channel refusal.

    Shared by the ``+``, ``Chart.conditional()``, ``encode()`` and
    ``Chart.layer(Layer(encoding=...))`` collision sites
    (:func:`_conditional_on_wire`'s callers). Names the public entry that
    raised (*context*), the wire channel, and both colliding declarations,
    so a chained merge like ``(a + b) + c`` names the two original
    declarations instead of just reporting that the channel is already
    taken.
    """
    return (
        f"{context}: {first} and {second} both declare a conditional on the "
        f"{wire!r} wire channel{detail}; a channel can carry only one "
        f"conditional. Drop one of the two declarations, or move one to a "
        f"different channel."
    )


def _append_conditional(
    target: "Chart",
    spec: "ConditionalSpec",
    *,
    source_key: str,
    claimed_here: dict[str, str],
    context: str,
    declare: Callable[[str], str],
) -> "ConditionalSpec":
    """Record one conditional on *target*, refusing a wire-channel collision.

    The single append site for every conditional-attaching entry point:
    ``encode()``'s two spellings (``encode(<channel>=cond)`` and
    ``<Channel>(field, condition=cond)``) and ``Chart.layer(Layer(encoding=...))``'s
    two mirror spellings (:func:`_strip_layer_conditionals`). Resolves the
    wire channel through the one vocabulary, refuses a second conditional on
    a channel that already has one, stamps the channel, and auto-registers
    the spec's source selection. *context* names the public entry that
    called this (``"encode()"`` or ``"Chart.layer()"``) and *declare* formats
    one entry's own key into that entry's declaration text (``"encode(color=...)"``
    or ``"Layer(encoding={'color': ...})"``) -- both required, not defaulted,
    so a caller cannot silently inherit the wrong voice. Returns the recorded
    spec, the object a declaring layer holds in ``_Layer.conditionals``.

    **Collision rule.** A wire channel carries at most one conditional. Two
    declarations that resolve to the same channel — ``encode(color=condA,
    fill=condB)``, since ``fill`` aliases to ``color`` — used to produce two
    wire conditionals with the same ``channel``, no warning, and no defined
    winner: whichever the runtime applied last silently discarded the other
    declaration. Refusing names the conflict at the call that creates it.
    *claimed_here* maps wire channel -> the key that claimed it earlier in
    the same call (``encode()`` or ``Chart.layer(*layers)``), so a same-call
    collision can name both spellings via *declare*; a collision against a
    conditional already on the chart names the channel and the earlier
    declaration's selection, since which entry point produced it is not
    recorded on the spec.

    ``Chart.__add__``'s layer merge enforces the identical rule via its own
    inline check rather than calling this function (it has no single
    *source_key*/*claimed_here* to resolve -- it is merging two already-built
    conditional lists, not appending one new declaration), and the answer is
    the same: the merged chart holds one conditional per wire channel,
    whichever layers each reaches, so two on one channel have no defined
    winner and refuse.
    """
    wire = conditional_wire_channel(source_key)
    alias_detail = (
        f" ({source_key} carries its conditional onto {wire})" if source_key != wire else ""
    )
    prior_key = claimed_here.get(wire)
    if prior_key is not None:
        raise ValueError(
            _conditional_collision_message(
                context=context,
                wire=wire,
                first=declare(prior_key),
                second=declare(source_key),
                detail=alias_detail,
            )
        )
    earlier = _conditional_on_wire(target, wire)
    if earlier is not None:
        raise ValueError(
            _conditional_collision_message(
                context=context,
                wire=wire,
                first=_conditional_declaration_label(earlier, whose="the chart's existing"),
                second=declare(source_key),
                detail=alias_detail,
            )
        )
    claimed_here[wire] = source_key
    recorded = replace(spec, channel=wire)
    target._conditionals.append(recorded)
    sel = spec.selection
    if sel is not None:
        _append_unique_by_name(target._selections, sel, entry=context)
    return recorded


def _refuse_alias_field_dropped_with_kept_conditional(
    encoding: dict,
    alias_conditional_sources: dict[str, str],
    *,
    declare: Callable[[str], str],
) -> None:
    """Refuse when an alias source's base field would be dropped while its
    OWN conditional survives and overrides the alias target.

    Batch-H close cycle 3 (H3-2): ``_append_conditional``'s collision rule
    refuses two conditionals landing on one wire channel, but not this
    half-and-half case -- ``encode(color=fm.Color("d"), fill=fm.Fill("c",
    condition=cond))`` used to build silently: ``fill``'s own field ``"c"``
    is dropped by ``_ENCODING_ALIASES``' ``keep_target`` policy (``color`` is
    already present), while ``fill``'s conditional -- extracted independently
    in ``encode()``'s ``ChannelBase`` branch -- survives and overrides
    ``color`` at selection time. The user's field mapping silently vanishes
    while the call *looks* like it did something (the conditional still
    fires). This is the surviving twin of the cycle-2 fix that refused two
    conditionals on one channel: same shape, one side is a base field instead
    of a second conditional.

    Parameters
    ----------
    encoding :
        The encoding dict to check (``Chart._encoding`` for the chart-level
        call site, or a single layer's own encoding dict for the
        ``Chart.layer()`` call site -- alias resolution is scoped per-layer,
        so the check is too).
    alias_conditional_sources :
        Wire channel -> the alias-source encode key whose OWN
        ``condition=`` resolved to that wire channel during this call (e.g.
        ``{"color": "fill"}``). Populated only for the ``ChannelBase``
        branch (a bare ``ConditionalSpec`` spelling carries no base field to
        collide).
    declare :
        Formats an encode key into its call-site spelling for the error
        message (``"encode({name}=...)"`` for the chart-level path,
        ``"Layer(encoding={{{name!r}: ...}})"`` for the layer path).

    Raises
    ------
    ValueError
        Naming both the target channel and the alias-source key, the same
        message shape as ``_append_conditional``'s same-call collision.
    """
    for wire, source_name in alias_conditional_sources.items():
        if wire not in encoding:
            continue
        source_channel = encoding.get(source_name)
        field = getattr(source_channel, "field", None)
        if field is None:
            # A fieldless alias source (e.g. bare fm.Fill(condition=cond),
            # matching the equivalent encode(fill=<ConditionalSpec>) bare
            # spelling) has no field for the alias to drop -- there is
            # nothing to collide with color's own field, so this is a fully
            # supported call, not the split this guard exists to refuse
            # (2026-09-11, batch-H close cycle 4, S3: the pre-cycle-4 code
            # raised here unconditionally, refusing a call that built fine
            # before cycle 3 and that the bare-ConditionalSpec spelling
            # still accepts).
            continue
        raise ValueError(
            f"{declare(wire)} and {declare(source_name)} both declare the {wire!r} wire "
            f"channel: {source_name}'s field ({field!r}) would be silently dropped by the "
            f"{source_name} -> {wire} alias since {wire} is already encoded, but "
            f"{source_name}'s condition= survives and overrides {wire} at selection time "
            f"-- that split (field dropped, conditional kept) is refused. Encode "
            f"{source_name} only when {wire} is absent, or move its condition= to "
            f"{wire} directly."
        )


def _layer_operand(layer: Any, *, host: "Chart") -> "Chart":
    """Build the chart that a public ``Layer`` with its own data composes with through ``+``.

    The operand draws the one layer over the layer's data. The layer's
    encoding is also the operand's own, scoped to that one layer, and the
    conditionals the layer declares belong to that layer. The layer
    is drawn in *host*'s coordinate system, so the operand takes the host's
    coord. It declares no facet: the composed chart keeps the host's, as a
    ``+`` keeps its left operand's, and partitions the layer's rows by it.
    """
    operand = Chart(layer.data)
    encoding, owned = _strip_layer_conditionals(operand, layer.encoding, claimed_here={})
    operand._encoding = dict(encoding)
    operand._layers = [
        _Layer(
            name=layer.name,
            mark=layer.mark,
            encoding=encoding,
            mark_kwargs=dict(layer.mark_kwargs) if layer.mark_kwargs else None,
            conditionals=owned,
        )
    ]
    operand._coord = host._coord
    return operand


def _strip_layer_conditionals(
    target: "Chart",
    encoding: dict,
    *,
    claimed_here: dict[str, str],
) -> tuple[dict, tuple]:
    """Lift a layer-declared channel's ``condition=`` into *target*'s chart-level conditionals.

    ``Chart.layer(Layer(encoding={...}))`` is a fourth public way to attach a
    conditional (H3-1, batch-H close cycle 3) alongside ``encode()``'s two
    spellings and ``Chart.conditional()`` -- but unlike those three, a
    layer's ``encoding`` dict never passes through ``Chart.encode()``, so a
    channel-carried ``condition=`` used to serialize straight into the dead
    per-layer ``EncodingSpec.condition`` blob (formerly via the Python-side
    ``_emit_condition``, deleted by ``#237``; nothing on the Rust side reads
    the field -- ``crates/ferrum-core/src/spec/encoding.rs`` stores it as an
    opaque, tolerated ``serde_json::Value`` for wire-compat only) instead of
    the chart's real ``conditionals`` pipeline. Accepted, serialized, and
    inert -- the exact silent-drop shape this batch's ``encode()`` work
    removed for every other entry point, one code path over.

    Mirrors **both** of ``encode()``'s conditional branches, not just the
    channel-carried one (2026-09-11, batch-H close cycle 4, S2: the bare
    ``ConditionalSpec`` spelling -- ``Layer(encoding={"color": cond})``,
    matching ``encode(color=cond)`` -- used to fall through this loop
    untouched, since ``isinstance(cond, ChannelBase)`` is False for it, and
    then vanish entirely at serialization: ``_build_layers_list`` only
    recognizes ``ChannelBase`` values, so the key was silently dropped with
    no warning, no conditional, nothing). Both branches resolve the wire
    channel through the one conditional vocabulary (refusing the same
    unsupported channels with the same typed error) and append to
    ``target._conditionals`` via :func:`_append_conditional` (so a
    layer-declared conditional collides with an ``encode()``-declared one on
    the same wire channel exactly as two ``encode()`` calls would); the
    channel-carried branch additionally rebuilds the channel without
    ``condition`` in its kwargs and is checked against the
    alias-vs-base-field collision :func:`_refuse_alias_field_dropped_with_kept_conditional`
    describes -- scoped to *this layer's own* encoding dict, matching
    ``_build_layers_list``'s per-layer alias resolution.

    Parameters
    ----------
    target :
        The chart the conditional is lifted onto (``Chart.layer()``'s
        in-progress ``new``).
    encoding :
        One ``Layer``'s own encoding dict (``PublicLayer.encoding``).
    claimed_here :
        Wire channel -> the encode key that claimed it earlier in this
        ``Chart.layer(*layers)`` call, shared across every layer in the
        call (mirrors ``encode()``'s per-call ``claimed_here``) -- so two
        layers attached in the same call declaring a conditional on the
        same wire channel collide exactly as two ``encode()`` kwargs would.

    Returns
    -------
    tuple of (dict, tuple)
        A new encoding dict with every ``condition=`` kwarg stripped
        (*encoding* itself is never mutated), and the conditionals this layer
        declared, which the caller hands to the layer's
        ``_Layer.conditionals`` so they belong to that layer alone.
    """

    def declare(n: str) -> str:
        return f"Layer(encoding={{{n!r}: ...}})"

    stripped: dict = {}
    owned: list = []
    alias_conditional_sources: dict[str, str] = {}
    for name, channel in encoding.items():
        if isinstance(channel, ConditionalSpec):
            # Layer(encoding={<channel>: <ConditionalSpec>}): the bare
            # spelling, matching encode(<channel>=<ConditionalSpec>) exactly
            # -- the wire channel is known from the encode key, and there is
            # no base field to preserve, so this entry does not join
            # `stripped` at all (mirrors encode()'s `continue`).
            owned.append(
                _append_conditional(
                    target,
                    channel,
                    source_key=name,
                    claimed_here=claimed_here,
                    context="Chart.layer()",
                    declare=declare,
                )
            )
            continue
        if isinstance(channel, ChannelBase):
            cond = channel._kwargs.get("condition")
            if cond is not None:
                if not isinstance(cond, ConditionalSpec):
                    raise TypeError(
                        f"Layer(encoding={{{name!r}: {channel.__class__.__name__}(condition=...)}}) "
                        f"must be a ConditionalSpec built via sel.when(...).otherwise(...) (or "
                        f"fm.when(sel).then(...).otherwise(...)); got {type(cond).__name__}."
                    )
                wire = conditional_wire_channel(name)
                owned.append(
                    _append_conditional(
                        target,
                        cond,
                        source_key=name,
                        claimed_here=claimed_here,
                        context="Chart.layer()",
                        declare=declare,
                    )
                )
                if wire != name:
                    alias_conditional_sources[wire] = name
                remaining_kwargs = {k: v for k, v in channel._kwargs.items() if k != "condition"}
                channel = channel.__class__(channel.field, **remaining_kwargs)
        stripped[name] = channel
    _refuse_alias_field_dropped_with_kept_conditional(
        stripped,
        alias_conditional_sources,
        declare=declare,
    )
    return stripped, tuple(owned)


def _refuse_cross_operand_reactive_names(charts: list) -> None:
    """Refuse a name one operand declares as a selection and an earlier one as a variable parameter.

    Each of *charts*, in order, is checked against the operands before it
    only: its selections against their variable parameters, its variable
    parameters against their selections, in ``+``'s words
    (:func:`_check_param_collision`, context ``layer merge``). A chart's
    own selection and parameter sharing a name are left to its spec build,
    which refuses them as a ``param collection`` collision on every door.
    Shared by ``Chart.__add__`` (the merged left side, then the right
    operand) and the static ``LayerChart`` overlay (its members in order).
    """
    from ferrum.parameter import VariableParameter

    selection_names: set[str] = set()
    variable_names: set[str] = set()
    for chart in charts:
        selections = {s.name for s in getattr(chart, "_selections", None) or [] if s is not None}
        variables = {
            p.name
            for p in getattr(chart, "_params", None) or []
            if isinstance(p, VariableParameter)
        }
        for name in sorted(selections & variable_names):
            _check_param_collision(name, is_selection=True, context="layer merge")
        for name in sorted(variables & selection_names):
            _check_param_collision(name, is_selection=False, context="layer merge")
        selection_names |= selections
        variable_names |= variables


def _check_param_collision(
    name: str,
    *,
    is_selection: bool,
    context: str = "layer merge",
) -> None:
    """Raise ``ValueError`` when a reactive-parameter name collides across kinds.

    A name registered as both a ``Selection`` and a ``VariableParameter``
    is always a user error: two reactive-object kinds cannot share a name
    without producing a silently wrong spec.

    Parameters
    ----------
    name:
        The colliding reactive-parameter name.
    is_selection:
        ``True`` when *name* is the Selection side (the other object is a
        VariableParameter); ``False`` when *name* is the VariableParameter
        side (the other object is a Selection).
    context:
        Short description of where the collision was detected, used in the
        error message for easier diagnosis.
    """
    if is_selection:
        detail = f"{name!r} is a Selection on one side and a VariableParameter on the other"
    else:
        detail = f"{name!r} is a VariableParameter on one side and a Selection on the other"
    raise ValueError(
        f"Reactive-parameter name collision ({context}): {detail}. "
        f"A name must resolve to a single reactive-object kind. Rename one of them."
    )


def _coalesce_facet_rhs_columns(chart: "Chart") -> "Chart":
    """Fill each facet field's nulls from the renamed copies a ``+`` composition made of it.

    A composition over several frames renames each frame's column whose name
    an earlier frame already holds (``chart._frame_selectors``), so a facet
    on that name would read null on that frame's rows in the combined batch.
    Each renamed copy of a facet field fills it where it is null. Any copy of
    a non-facet column is left alone.

    Returns the chart, its data replaced on this clone when a copy filled
    the facet column.
    """
    if chart._facet is None or chart._data is None or not chart._frame_selectors:
        return chart

    import polars as pl

    if not isinstance(chart._data, pl.DataFrame):
        return chart

    f = chart._facet
    facet_fields = [ff for ff in (f.field, f.col, f.row) if ff is not None]

    df = chart._data
    for facet_col in facet_fields:
        if facet_col not in df.columns:
            continue
        copies = [
            renamed
            for selector in chart._frame_selectors
            for original, renamed in selector.renames
            if original == facet_col
        ]
        if not copies:
            continue
        expr = pl.coalesce([pl.col(facet_col), *(pl.col(c) for c in copies)])
        df = df.with_columns(expr.alias(facet_col))
    chart._data = df
    return chart


def _parsed_facet_arg(name: str, value: Any, *, declare: str) -> Optional[ChannelBase]:
    """Parse one of :meth:`Chart.facet`'s ``field=``/``row=``/``col=`` arguments the column-keyword way.

    *value* is read through :func:`~ferrum.encoding._resolve.channel_for_column`:
    shorthand (``"g:N"``) parses as ``encode(facet=...)``/``encode(facet_col=...)``/
    ``encode(facet_row=...)`` parses it, a channel object works directly,
    and a column name the shorthand grammar cannot read (``"grp (k)"``)
    binds that exact column -- leniently, unlike ``encode()``'s own
    strict channels, which refuse such a string. A value that is none of
    str/channel/``Repeat`` raises the parse's own ``TypeError`` naming
    *declare*. ``None`` when *value* is ``None``. Returning the parsed
    CHANNEL, not a fixed tuple of the options it happens to carry, is
    what lets :meth:`Chart.facet` read every option ``encode(facet...)``
    honors (title, sort, a ``Facet``'s own ``columns=``) from one place,
    wherever its own keyword for that option is left unset -- the same
    precedence ``encode(facet_col=...)`` applies to a channel's own
    options.
    """
    if value is None:
        return None
    return channel_for_column(name, value, declare=declare)


def _infer_type_from_data(field: str | None, data: Any) -> str | None:
    """Return ``"T"`` when *field* is a temporal column in *data*, else ``None``.

    Checks polars Datetime/Date/Time/Duration dtypes first (fast path), then
    falls back to PyArrow timestamp/date32/date64 types.  Returns ``None``
    when *data* is ``None``, *field* is ``None``, the column is absent, or the
    dtype is not temporal.

    This is the sole dtype-to-type inference path used by ``_build_encoding_specs``
    and ``_build_layers_list``.  Explicit user-supplied type annotations always
    win over this inference at the call sites.
    """
    if field is None or data is None:
        return None

    # Polars fast path — covers the most common case without round-tripping
    # through Arrow.  Duration maps to epoch-ms integers via _coerce.py (cast
    # to Int64), so it is intentionally excluded here: it should keep the
    # default quantitative type, not get a temporal scale.
    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            if field not in data.columns:
                return None
            dtype = data[field].dtype
            if isinstance(dtype, (pl.Datetime, pl.Date, pl.Time)):
                return "T"
            return None
    except ImportError:
        pass

    # PyArrow fallback — covers PyArrow Table inputs and narwhals-coerced paths.
    try:
        import pyarrow as pa

        if isinstance(data, pa.Table):
            if field not in data.schema.names:
                return None
            field_type = data.schema.field(field).type
            if (
                pa.types.is_timestamp(field_type)
                or pa.types.is_date32(field_type)
                or pa.types.is_date64(field_type)
            ):
                return "T"
            if pa.types.is_time32(field_type) or pa.types.is_time64(field_type):
                return "T"
            return None
    except ImportError:
        pass

    return None


def _apply_inferred_type(d: dict, field: str | None, data: Any) -> dict:
    """Return *d* with ``"type_": "T"`` added when *field* is a temporal column.

    Applies temporal type inference only when ``"type_"`` is absent from *d*
    (i.e., the user did not supply an explicit type annotation).  When the
    inferred type is ``None`` (non-temporal or missing column), *d* is returned
    unchanged.

    This is the single apply-site for the infer-and-write pattern shared by
    ``_build_encoding_specs`` and ``_build_layers_list``.
    """
    if "type_" in d:
        return d
    inferred = _infer_type_from_data(field, data)
    if inferred is not None:
        return {**d, "type_": inferred}
    return d


class Chart(
    ConfigureMixin,
    StatisticalMarksMixin,
    DiagnosticMarksMixin,
    SpecBuildMixin,
    _RenderMixin,
    _ChartOperatorMixin,
):
    """Top-level chart value class.

    Every method returns a new ``Chart`` — the object is effectively immutable and
    safe to reuse across branches of a pipeline.  The fluent API follows the pattern::

        fm.Chart(df).mark_point().encode(x="sepal_length", y="sepal_width")

    Parameters
    ----------
    data : DataFrame-like or None, optional
        Input data.  Accepts Polars ``DataFrame``, pandas ``DataFrame``, PyArrow
        ``Table`` / ``RecordBatch``, dict-of-arrays, list-of-records, or a 2-D
        NumPy array.  Passed through ``ferrum._coerce.to_arrow_table`` at render
        time.  ``None`` is allowed when composing layered charts that share a
        parent's data.
    width : int or float or "container", optional
        Chart width in pixels, or ``"container"`` to fill the parent element.
        Default is ``600``. A number must be finite and positive; anything
        else raises ``ValueError`` naming ``Chart(width=...)``.
    height : int or float or "container", optional
        Chart height in pixels.  Default is ``400``. Same rule as ``width``.
    title : str, optional
        Title rendered above the plot area.
    description : str, optional
        Accessible description attached to the SVG root (``<title>`` element).

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> svg = chart.to_svg()
    """

    __slots__ = (
        "_data",
        "_mark",
        "_mark_kwargs",
        "_encoding",
        "_transforms",
        "_facet",
        "_coord",
        "_theme",
        "_layers",
        "_width",
        "_height",
        "_title",
        "_description",
        "_pending_stat_mark",  # _PendingMark when mark_* called before .encode()
        "_position",  # Phase 9c — Identity / Dodge / Jitter / Stack (or None)
        "_axis_x",
        "_axis_y",  # spec-level axis suppression (.axis(x=False) / .axis(y=False))
        "_composite_kind",
        "_entry",  # str or None -- the public entry a flat chart's one mark speaks as
        "_selections",
        "_conditionals",
        "_params",  # list[Parameter] — D6 reactive parameters (fm.param / selections)
        "_render_config",
        "_configure",  # list[Configure] — accumulated configure layers
        "_annotations",  # list[Annotate] — accumulated annotation layers
        "_structural",  # list — accumulated structural features (BreakAxis, Inset)
        "_overrides",  # dict — spec-path override kwargs
        "_annotation_primitive",  # optional annotation primitive for annotate_* helpers
        "_mark_zero",  # bool — False when mark_bar(zero=False) suppresses the y zero-anchor
        "_figure_caption",  # str or None — figure-level caption rendered below the SVG
        "_written_positional",  # dict or None — the positional and polar bindings the
        # callers wrote, latest per channel (see _written_bindings): on a chart `+`
        # produced, and on a host whose mark statistics moved its tail when layers
        # were attached (_attach_host); None on any other chart.
        "_operands",  # tuple of _Operand — a + composition's operands, left first
        # (ferrum._compose); () on a chart of one operand, read through
        # chart_operands.
        "_composite_source",  # _CompositeSource or None — the rebuild record of
        # statistics built before their grouping was known (a composite mark
        # that resolved early, a figure's), kept so a later .encode() or a
        # composition it becomes an operand of regroups them (ferrum._compose.regrouped).
        "_frames",  # tuple of frames — the composition's frame registry, each
        # operand frame once, the composition's own first; () on a chart of
        # one operand, read through chart_frames.
        "_frame_selectors",  # tuple of _OperandFrame — each registered frame's
        # marker, renames and columns in _data when the registry holds several
        # frames; () otherwise.
        "_post_build_positional",  # _PostBuildPositional or None — a regression
        # figure's (regplot/lmplot/residplot) contract for an x/y binding its
        # caller writes after the figure was built, applied by
        # Chart._resolve_pending (ferrum._compose.apply_post_build_positional);
        # None on every other chart, and reset to None on a `+` result as
        # _composite_source is.
    )

    def __init__(
        self,
        data: Any = None,
        *,
        width: Optional[Union[int, float, str]] = None,
        height: Optional[Union[int, float, str]] = None,
        title: "Optional[Union[str, 'Title']]" = None,
        description: Optional[str] = None,
    ) -> None:
        self._data = data
        self._mark = None
        self._mark_kwargs = {}
        self._encoding: dict[str, ChannelBase] = {}
        self._transforms: list = []
        self._facet = None
        self._coord = None
        self._theme = None
        self._layers: Optional[list] = None
        self._width = _viewport_dimension("Chart", width, param="width")
        self._height = _viewport_dimension("Chart", height, param="height")
        # Schwabish SB1 (2026-05-11): accept Title value class or plain str.
        from ferrum.title import Title as _TitleCls

        if title is None:
            self._title = None
        elif isinstance(title, _TitleCls):
            self._title = title
        else:
            self._title = _TitleCls(text=str(title))
        self._description = description
        self._pending_stat_mark: Optional[_PendingMark] = None
        self._position = None
        self._axis_x: Optional[bool] = None
        self._axis_y: Optional[bool] = None
        self._composite_kind: Optional[str] = None
        self._entry: Optional[str] = None
        self._selections: list = []
        self._conditionals: list = []
        self._params: list = []
        self._render_config = None
        self._configure: list = []
        self._annotations: list = []
        self._structural: list = []
        self._overrides: dict = {}
        self._annotation_primitive = None
        self._mark_zero: bool = True
        self._figure_caption: Optional[str] = None
        self._written_positional: Optional[dict[str, Any]] = None
        self._operands: tuple = ()
        self._composite_source = None
        self._frames: tuple = ()
        self._frame_selectors: tuple = ()
        self._post_build_positional = None

    def _clone(self) -> "Chart":
        new = object.__new__(Chart)
        for slot in self.__slots__:
            val = getattr(self, slot)
            if isinstance(val, (list, dict)):
                setattr(new, slot, copy.copy(val))
            else:
                setattr(new, slot, val)
        return new

    def _with_polars_data(self) -> "Chart":
        """Return a chart whose data is the canonical ``polars.DataFrame``.

        Every reader downstream of a render or spec entry (auto-raster gate,
        type inference, ordinal-domain lookup, label maps, desugars) answers
        from one frame type, and a ``LazyFrame`` is collected once here rather
        than once per reader. Unsupported data (a ``Path`` or ``str``) raises
        ``to_polars``'s ``TypeError: Unsupported data type`` here, before any
        desugar-time validation. Returns *self* when the data is ``None`` or
        already a ``polars.DataFrame``; otherwise a clone, so the caller's
        chart is never mutated.
        """
        import polars as pl

        if self._data is None or isinstance(self._data, pl.DataFrame):
            return self
        new = self._clone()
        new._data = to_polars(self._data)
        return new

    def _append_configure(self, config) -> "Chart":
        """Clone self, append *config* to ``_configure``, return new Chart."""
        new = self._clone()
        new._configure = new._configure + [config]
        return new

    def _resolve_pending(self) -> "Chart":
        """Resolve a pending statistical mark desugar once encoding is known.

        Calling a composite/diagnostic ``mark_*()`` before ``.encode()`` stores
        a ``_PendingMark`` sentinel. ``_resolve_pending`` is called at the
        start of every render/spec-build path to apply ``desugar_fn`` against
        the now-populated encoding dict.

        ``desugar_fn(x_field, y_field, **kwargs)`` returns a
        ``MarkDesugarResult``. When ``result.layers`` is set, the chart enters
        layered mode; otherwise it applies the single-mark ``result.mark``,
        ``result.remap``, and ``result.position``.

        The desugar body lives in :func:`ferrum._desugar._resolve_pending_impl`.
        On a ``+`` composition, whose data is its left operand's, the
        statistics the mark lowers join that operand's chain, after its own
        statistics, and the layers the mark adds read inside it.

        A chart of one operand keeps the record of its pending mark
        (``_composite_source``), so a later ``.encode()`` on it, or a
        composition it becomes an operand of, can build its statistics again
        under the grouping it binds. Every call ends by regrouping the
        statistics built before that grouping was known: a chart of one
        operand under its own grouping bindings, a composition's operands
        under the composition's (:func:`ferrum._compose.regrouped`).

        A regression figure's x/y binding carries its own, separate
        record (``_post_build_positional``): applied first
        (:func:`ferrum._compose.apply_post_build_positional`), before the
        pending mark above resolves, so a figure's statistic refusal and
        re-point run before any `_PENDING_PREP` step reads the (possibly
        rewritten) encoding.
        """
        self = apply_post_build_positional(self)
        resolved = _resolve_pending_impl(self)
        if resolved is not self and not self._operands:
            resolved._composite_source = composite_source(self, resolved)
            return resolved
        if resolved is not self:
            own, lowered = _split_chain(list(resolved._transforms))
            left = self._operands[0]
            resolved._operands = (
                replace(left, statistics=left.statistics + lowered),
                *self._operands[1:],
            )
            resolved._transforms = list(own)
        return regrouped(resolved)

    def _has_independent_y_layer(self) -> bool:
        """Return whether any of this chart's layers carry ``independent_y=True``.

        The plain-``Chart`` capability predicate that mirrors
        ``LayerChart._y_independent()`` for the *other* spelling of
        dual-axis (GH #52/#71): ``chart + SecondaryY(...)``
        (:func:`_desugar_secondary_y`) never becomes a ``LayerChart`` --
        it appends one flagged layer onto this chart's own ``_layers`` --
        so composition-level independent-y conflict detection
        (:func:`ferrum.composition._contains_independent_y_layer`) consults
        this predicate for a leaf ``Chart`` the same way it consults
        ``LayerChart._y_independent()`` for the layered spelling. A chart
        with no ``_layers`` (single-mark, never desugared) cannot carry the
        flag.
        """
        if not self._layers:
            return False
        return any(layer.independent_y for layer in self._layers)

    # ---- Marks (primitives) ----

    def _set_mark(self, name: str, **kwargs: Any) -> "Chart":
        # A layered chart has no mark of its own for this call to replace;
        # refused here, before `position=`/kwargs are read, in the same
        # place `_refuse_fitted_mark_after_pending_composite` runs first for
        # a fitted mark called after a pending composite.
        _refuse_primitive_mark_on_layered_chart(f"mark_{name}", self._layers)
        # Phase 9c -- pull `position=` out of kwargs and validate eligibility
        # before constructing the MarkBase (which would reject unknown kwargs).
        position = kwargs.pop("position", None)
        if position is not None:
            from ferrum.position import validate_position_eligibility

            validate_position_eligibility(name, position)
        m = MarkBase(name, **kwargs)
        new = self._clone()
        new._mark = name
        # D3: zero= is consumed below, not forwarded as this chart's style.
        new._mark_kwargs = {k: v for k, v in m.to_mark_kwargs_dict().items() if k != "zero"}
        new._position = position
        # The mark this call sets speaks as this call, not as whatever entry
        # built the mark it replaces.
        new._entry = None
        # A primitive mark call replaces whatever mark came before it,
        # pending composite included: a chart holds one _pending_stat_mark
        # slot, so a queued composite desugar is cleared here, rather than
        # left for _resolve_pending to run against a mark this call already
        # replaced.
        new._pending_stat_mark = None
        new._composite_source = None
        # A replaced mark's figure, if any, does not own this chart's x/y
        # going forward: its post-build record would otherwise refuse or
        # re-point a later `.encode()` in the replaced figure's voice, for
        # a mark this call already replaced (`_composite_source`/
        # `_composite_kind`'s own reason, just above and below).
        new._post_build_positional = None
        # _composite_kind is the other half of that same composite state
        # (_set_composite_mark sets both together); left stale it would
        # still read the replaced composite's name, e.g. _apply_auto_raster
        # (_render.py) reading "histogram" on a chart that is now a plain
        # mark_point() and so skipping its auto-raster eligibility check for
        # no reason, and _overrides._valid_layer_names validating a
        # layer_kwargs=/mark= door against the replaced composite's named
        # sub-layers instead of the primitive's own (none).
        new._composite_kind = None
        # S4: orient="horizontal" → coord flip (consumed Python-side).
        if m.orient_coord_flip():
            new._coord = "flip"
        # D3: zero=False suppresses the bar y-scale zero-anchor injection.
        new._mark_zero = m.zero_anchor()
        return new

    def _set_composite_mark(
        self,
        name: str,
        desugar_fn,
        kwargs: dict,
        *,
        placeholder: str,
        position=None,
        data_transform=None,
        prior_mark: str | None = None,
    ) -> "Chart":
        # Shared scaffold for composite/diagnostic mark_* methods. Validates the
        # position adjustment, clones, sets the placeholder mark (overridden by
        # layered mode at render time), optionally rewrites the polars data,
        # and stashes the desugar callable in the 3-tuple _pending_stat_mark
        # sentinel resolved by _resolve_pending once .encode() is known.
        if position is not None:
            from ferrum.position import validate_position_eligibility

            validate_position_eligibility(name, position)
        new = self._clone()
        new._mark = placeholder
        new._entry = None  # the mark this call sets speaks as this call
        if data_transform is not None and new._data is not None:
            # Coerce through the canonical to-polars entry point before
            # applying the row-filter closure, so `data_transform` runs for
            # every supported input type (pandas, pyarrow, dict, etc.), not
            # just polars-backed charts. The render path (`_render.py`)
            # coerces `_data` lazily via `to_arrow_table` regardless of its
            # concrete type, so holding a polars frame here downstream is
            # safe -- it is the same shape already produced by `.layer()`
            # and `+` concatenation (chart.py:1516, 1749).
            new._data = data_transform(to_polars(new._data))
        new._pending_stat_mark = _PendingMark(name, dict(kwargs), desugar_fn, prior_mark=prior_mark)
        new._composite_source = None
        # As in _set_mark: a composite mark call replaces what this chart's
        # x/y meant to whatever figure built it, so that figure's post-build
        # record does not apply to the mark this call is queuing.
        new._post_build_positional = None
        new._position = position
        new._composite_kind = name
        if prior_mark is None:
            # A composite mark call replaces the chart's mark as a primitive
            # call does (`_set_mark`, above): its style and `zero=` go with
            # it, so a chained `mark_point(size=30).mark_ribbon()` draws the
            # ribbon `mark_ribbon()` draws on its own, not one restyled by
            # the point it replaced. `mark_smooth()` with `prior_mark` set
            # keeps the prior primitive as its own layer, which the
            # dispatcher builds from this style (`_desugar.py`'s
            # `_build_prior_layer` call), so both stay.
            new._mark_kwargs = {}
            new._mark_zero = True
        return new

    def _native_size(self) -> tuple[float, float]:
        """Return the ``(width, height)`` this chart renders at on its own.

        Its declared size, or else the ``ferrum.config`` default the renderer
        falls back to; read at call time, so a ``config.set`` in force applies.
        """
        from ferrum import config

        return (
            self._width or float(config.get("width")),
            self._height or float(config.get("height")),
        )

    def _with_entry(self, entry: str) -> "Chart":
        # A clone whose every layer -- the ones it already has, the ones its
        # pending composite mark emits at resolution, or its one unlayered
        # mark -- names *entry*, the public figure function or visualizer
        # that built it, so a warning or refusal about any of them speaks as
        # that call (`_Layer.entry`; the chart-level `_entry` is the flat
        # chart's, see `to_spec`). Every door inside the figure that writes a
        # caller's style already refused it in this voice (`speak_as`), a
        # pending mark resolved later refuses in it too (`_resolve_pending`),
        # and Rust holds every layer's size to the one size rule at render,
        # wrapped in the layer's entry.
        new = self._clone()
        if new._layers:
            new._layers = [replace(layer, entry=entry) for layer in new._layers]
        if new._pending_stat_mark is not None:
            new._pending_stat_mark = replace(new._pending_stat_mark, entry=entry)
        if not new._layers and new._pending_stat_mark is None:
            new._entry = entry
        return new

    def _with_post_build_positional(self, record: Any) -> "Chart":
        # A clone carrying *record* (a `ferrum._compose._PostBuildPositional`),
        # a regression figure's contract for an x/y binding its caller
        # writes after the build -- read by `_resolve_pending`, the one
        # place every door reaches (render, `to_spec`, `+`, `.layer`).
        new = self._clone()
        new._post_build_positional = record
        return new

    def mark_point(self, **kwargs) -> "Chart":
        """Render data as points (scatter plot).

        Parameters
        ----------
        size : float, optional
            Point area in square pixels.  Default is ``36``.
        fill : str, optional
            Fill colour override (CSS colour string or hex).  Normally driven
            by the ``color`` encoding.
        stroke : str, optional
            Stroke colour for the point outline.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        filled : bool, optional
            Whether points are filled.  Default is ``True``.
        shape : str, optional
            Point shape: ``"circle"``, ``"square"``, ``"cross"``, ``"diamond"``,
            ``"triangle-up"``, ``"triangle-down"``, ``"|"`` / ``"vline"``,
            ``"-"`` / ``"hline"``.
        stroke_width : float, optional
            Stroke width in pixels.
        stroke_dash : str or list, optional
            Dash pattern, e.g. ``"4,2"`` or a named linetype (``"solid"``,
            ``"dashed"``, ``"dotted"``, ``"dashdot"``, ``"longdash"``).
            Applies to the point outline stroke.
        position : Position, optional
            Position adjustment — ``fm.Jitter()``, ``fm.Dodge()``, etc.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"point"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_point(size=60, opacity=0.7).encode(x="x", y="y")
        Chart(mark='point', encoding=['x', 'y'])
        """
        return self._set_mark("point", **kwargs)

    def mark_line(self, **kwargs) -> "Chart":
        """Render data as a connected line.

        Parameters
        ----------
        stroke : str, optional
            Stroke colour override.
        stroke_width : float, optional
            Line width in pixels.  Defaults to the theme's
            ``line_stroke_width`` (``1.5`` in the default theme).
        size : float, optional
            Line width as a dot area in square pixels: the line draws
            ``2 * sqrt(size / pi)`` pixels wide, the diameter of a point of
            that area.  Must be a positive number.  A ``size`` encoding
            outranks it and tapers the line per vertex.  ``size`` and
            ``stroke_width`` both set the line's width, so a line that binds
            both (each as a kwarg or an encoding) raises ``ValueError`` when
            rendered. A conditional ``size=fm.when(sel).then(a).otherwise(b)``
            on the line draws the base width in static output; the
            interactive view redraws each vertex at the width of its row's
            active branch as the selection changes. Every branch must be
            positive, and a conditional ``size`` plus ``stroke_width`` raises
            ``ValueError`` like any other ``size``. A conditional encoded on
            the chart as a whole (including ``mark_line(point=True)`` and a
            layered chart) applies to every layer, lines included, as a
            field ``size`` does; one encoded on another layer's own chart or
            ``Layer`` leaves this line's width alone.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        interpolate : str, optional
            Line interpolation, validated against the live
            ``ferrum._core.interpolate_accepted_values()`` vocabulary (spec
            §4.6). An unrecognized value is refused at declaration
            (construction time), not at render.
        stroke_cap : str, optional
            Line cap: ``"butt"``, ``"round"``, ``"square"``.
        stroke_join : str, optional
            Line join: ``"miter"``, ``"round"``, ``"bevel"``.
        position : Position, optional
            Position adjustment.
        point : bool, optional
            When ``True``, overlay a ``mark_point`` layer on top of the line.
            Mirrors the Altair ``mark_line(point=True)`` shorthand.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"line"``.  When ``point=True``
            the result is a multi-layer chart (line + points).

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_line(stroke_width=3, interpolate="monotone").encode(x="x", y="y")
        Chart(mark='line', encoding=['x', 'y'])

        A line that tapers with ``y``:

        >>> fm.Chart(df).mark_line().encode(x="x", y="y", size="y")
        Chart(mark='line', encoding=['x', 'y', 'size'])
        """
        point_overlay = kwargs.pop("point", False)
        line_chart = self._set_mark("line", **kwargs)
        if point_overlay:
            point_chart = self._set_mark("point")
            return line_chart + point_chart
        return line_chart

    def mark_circle(self, **kwargs) -> "Chart":
        """Render data as filled circles — shorthand for ``mark_point(shape="circle")``.

        Mirrors the Altair ``mark_circle()`` convenience method.  All keyword
        arguments are forwarded to [mark_point][ferrum.Chart.mark_point], including aliases such
        as ``color`` and ``alpha``.

        Parameters
        ----------
        **kwargs
            Any keyword accepted by [mark_point][ferrum.Chart.mark_point] (``size``, ``fill``,
            ``opacity``, ``color``, ``alpha``, etc.).

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"point"`` and ``shape="circle"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_circle(size=80).encode(x="x", y="y")
        Chart(mark='point', encoding=['x', 'y'])
        """
        return self.mark_point(shape="circle", **kwargs)

    def mark_square(self, **kwargs) -> "Chart":
        """Render data as filled squares — shorthand for ``mark_point(shape="square")``.

        Mirrors the Altair ``mark_square()`` convenience method.  All keyword
        arguments are forwarded to [mark_point][ferrum.Chart.mark_point], including aliases such
        as ``color`` and ``alpha``.

        Parameters
        ----------
        **kwargs
            Any keyword accepted by [mark_point][ferrum.Chart.mark_point] (``size``, ``fill``,
            ``opacity``, ``color``, ``alpha``, etc.).

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"point"`` and ``shape="square"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_square(size=80).encode(x="x", y="y")
        Chart(mark='point', encoding=['x', 'y'])
        """
        return self.mark_point(shape="square", **kwargs)

    def mark_bar(self, **kwargs) -> "Chart":
        """Render data as bars.

        Parameters
        ----------
        fill : str, optional
            Bar fill colour override.
        stroke : str, optional
            Bar stroke colour.
        stroke_width : float, optional
            Bar stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        corner_radius : float, optional
            Rounded corner radius in pixels.
        orient : str, optional
            ``"vertical"`` (default) or ``"horizontal"``.
        position : Position, optional
            Position adjustment — ``fm.Stack()``, ``fm.Dodge()``, etc.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"bar"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
        >>> fm.Chart(df).mark_bar(corner_radius=4).encode(x="cat", y="val")
        Chart(mark='bar', encoding=['x', 'y'])
        """
        return self._set_mark("bar", **kwargs)

    def mark_area(self, **kwargs) -> "Chart":
        """Render data as a filled area between the line and a baseline.

        Parameters
        ----------
        fill : str, optional
            Fill colour override.
        stroke : str, optional
            Border stroke colour.
        stroke_width : float, optional
            Border stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        interpolate : str, optional
            Area boundary interpolation, validated against the live
            ``ferrum._core.interpolate_accepted_values()`` vocabulary (spec
            §4.6). An unrecognized value is refused at declaration
            (construction time), not at render.
        line : bool, optional
            Whether to draw the top boundary as a line.  Default is ``False``.
        position : Position, optional
            Position adjustment — e.g. ``fm.Stack()``.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"area"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_area(opacity=0.5, line=True).encode(x="x", y="y")
        Chart(mark='area', encoding=['x', 'y'])
        """
        return self._set_mark("area", **kwargs)

    def mark_rule(self, **kwargs) -> "Chart":
        """Render a horizontal or vertical reference rule spanning the plot area.

        A rule spans the full width (when ``y`` is encoded) or full height
        (when ``x`` is encoded).  Encoding both ``x``/``x2`` or ``y``/``y2``
        draws finite line segments instead.

        Parameters
        ----------
        stroke : str, optional
            Rule colour override.
        stroke_width : float, optional
            Rule width in pixels.  Default is ``1``.
        stroke_dash : str or list, optional
            Dash pattern, e.g. ``"4,2"`` or a named linetype (``"solid"``,
            ``"dashed"``, ``"dotted"``, ``"dashdot"``, ``"longdash"``).
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"rule"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"y": [0.0]})
        >>> fm.Chart(df).mark_rule(stroke="red", stroke_width=2).encode(y="y")
        Chart(mark='rule', encoding=['y'])
        """
        return self._set_mark("rule", **kwargs)

    def mark_text(self, **kwargs) -> "Chart":
        """Render data values as text labels.

        Places a free-positioned text glyph at each (x, y) data coordinate.
        No collision avoidance is applied and no leader line is drawn; the
        glyph sits exactly at the encoded position plus any ``dx``/``dy``
        pixel offset.  Use this mark when you control placement explicitly
        (e.g. inline bar labels, callout annotations at known coordinates).

        See also [mark_label][ferrum.Chart.mark_label] for a point-anchored annotation with
        automatic collision avoidance and optional leader-line support.

        Requires a ``text`` encoding channel pointing at the column to display.

        Parameters
        ----------
        fill : str, optional
            Text colour override.
        font_size : float, optional
            Font size in points.  Default is ``11``.
        font_weight : str, optional
            CSS font-weight as a string, e.g. ``"bold"``, ``"400"``, ``"700"``.
            A number is refused: write ``"700"``, not ``700``.
        align : str, optional
            Horizontal alignment: ``"left"``, ``"center"``, ``"right"``.
        baseline : str, optional
            Vertical alignment: ``"top"``, ``"middle"``, ``"bottom"``,
            ``"alphabetic"``.
        dx : float, optional
            Horizontal pixel offset from the anchor point.
        dy : float, optional
            Vertical pixel offset from the anchor point.
        angle : float, optional
            Rotation in degrees.
        limit : float, optional
            Maximum rendered width in pixels; clips overflow.
        radius : float, optional
            Under ``CoordPolar``, the distance in pixels from the center at
            which a label placed by its ``theta`` wedge sits. Without it the
            label sits mid-ring, ``(inner_radius + outer_radius) / 2`` (half
            the outer radius on a pie); a ``radius`` encoding wins over it.
            Warned and ignored on a Cartesian text and beside a ``radius``
            encoding.
        position : Position, optional
            Position adjustment.

        Notes
        -----
        Under ``CoordPolar`` a text that binds ``theta`` is placed as the arc
        places its wedges: the theta values stack into one pie sweep (or one
        band per category on a nominal theta) and each label sits at its
        wedge's mid-angle, so ``arc + text`` over the same ``theta`` draws a
        labelled pie.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"text"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "label": ["A", "B"]})
        >>> fm.Chart(df).mark_text(dy=-8).encode(x="x", y="y", text="label")
        Chart(mark='text', encoding=['x', 'y', 'text'])
        """
        return self._set_mark("text", **kwargs)

    def mark_tick(self, **kwargs) -> "Chart":
        """Render data as short tick marks (rug / strip plot).

        Per-row ``color``/``fill``/``stroke_opacity``/``opacity`` encodings
        render on ticks, but precedence against a constant mark-kwarg is not
        uniform: a constant ``stroke=`` wins over an encoded ``color=``, but
        an encoded ``opacity=`` wins over a constant ``opacity=`` (the
        encoded value replaces it outright). A per-row *encoded*
        ``stroke_dash=`` is a documented, unclosed exception — it has no
        effect on ticks at all, unlike ``mark_rule``/``mark_line``.

        Parameters
        ----------
        stroke : str, optional
            Tick colour override.
        stroke_width : float, optional
            Tick stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        band_size : float, optional
            Full rendered tick length as a fraction of a reference extent,
            matching ``mark_rect``'s ``band_size`` convention. On an
            ordinal tick (an ordinal x or y channel present, e.g. a
            strip plot or a boxplot median tick) the reference is that
            channel's own category band width, default ``0.6`` — this
            holds whether or not the *other* axis is also declared. On a
            **rug** tick — a single quantitative axis encoded, no
            categorical channel — the reference is the panel's cross-axis
            extent (panel height for an x-rug, panel width for a y-rug); an
            unset ``band_size`` keeps the fixed tick length ferrum has
            always used for rugs (``band_size`` was silently ignored there
            before, not defaulting to ``0.6``). A tick whose x resolves to
            a **quantitative** scale while y is *also* separately declared
            (not a true rug, since a rug has no y at all — this is the
            boxplot-cap/errorbar-cap idiom) keeps a fixed length
            unconditionally and ignores ``band_size``, by design.
        orient : str, optional
            ``"vertical"`` (default ticks perpendicular to x) or
            ``"horizontal"``.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"tick"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.1, 2.3, 3.5, 2.2, 1.8]})
        >>> fm.Chart(df).mark_tick().encode(x="x")
        Chart(mark='tick', encoding=['x'])
        """
        return self._set_mark("tick", **kwargs)

    def mark_rect(self, **kwargs) -> "Chart":
        """Render data as rectangles (heatmap cells, Gantt bars).

        Requires ``x``/``x2`` or ``y``/``y2`` encoding pairs (or ordinal
        ``x``/``y`` for a heatmap cell grid).

        Parameters
        ----------
        fill : str, optional
            Rectangle fill colour override.
        stroke : str, optional
            Rectangle border stroke colour.
        stroke_width : float, optional
            Border stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        corner_radius : float, optional
            Rounded corner radius in pixels.
        band_size : float, optional
            Rectangle width as a fraction of the category band (0–1,
            default 0.6) when the rect sits on an ordinal axis (e.g. a
            boxplot body). The same full-length convention as
            ``mark_tick``'s ``band_size``.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"rect"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"row": ["A", "B"], "col": ["X", "Y"], "val": [1.0, 0.5]})
        >>> fm.Chart(df).mark_rect().encode(x="col", y="row", color="val")
        Chart(mark='rect', encoding=['x', 'y', 'color'])
        """
        return self._set_mark("rect", **kwargs)

    def mark_segment(self, *, position=None, **kwargs) -> "Chart":
        """Render data as line segments from ``(x, y)`` to ``(x2, y2)``.

        Distinct from ``mark_rule`` (axis-aligned only); segments may take any
        direction. Requires ``x``, ``y``, ``x2``, ``y2`` on the encoding.

        Parameters
        ----------
        stroke : str, optional
            Segment stroke colour override.
        stroke_width : float, optional
            Segment stroke width in pixels.
        stroke_dash : list of float, comma-separated str, or named linetype, optional
            Dash pattern (alternating on/off pixel lengths), e.g. ``"4,2"``
            or a named linetype (``"solid"``, ``"dashed"``, ``"dotted"``,
            ``"dashdot"``, ``"longdash"``).
        opacity : float, optional
            Stroke opacity in ``[0, 1]``.
        position : Position, optional
            Position adjustment.
        **kwargs
            Additional mark-style overrides forwarded to the segment layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for segment rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [0, 1], "y": [0, 1], "x2": [1, 2], "y2": [1, 0]})
        >>> fm.Chart(df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2")
        Chart(mark='segment', encoding=['x', 'y', 'x2', 'y2'])
        """
        return self._set_mark("segment", position=position, **kwargs)

    # Statistical marks (mark_density .. mark_function) → StatisticalMarksMixin
    # Diagnostic marks (mark_residuals .. mark_parallel_coordinates) → DiagnosticMarksMixin

    def mark_arc(self, **kwargs) -> "Chart":
        """Render data as arcs (pie or donut slices).

        Requires ``Chart.coord(fm.CoordPolar(theta="x"))`` to be set — any
        other coordinate system (the default ``CoordCartesian``, or an
        explicitly set ``CoordFlip``, ``CoordGeo``, or ``CoordFixed``) raises
        ``ValueError`` naming the required ``.coord()`` fix; there is no
        silent Cartesian fallback.  The theta-mapped encoding channel (``x``
        by default) determines each slice's angular sweep proportional to
        its value.

        Parameters
        ----------
        **kwargs
            Constant mark-style overrides: ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``opacity``, ``corner_radius``
            (per-corner tangent-circle fillets on the wedge — also applies to
            a polar ``mark_bar``);
            aliases ``color``→``fill`` and ``alpha``→``opacity``.  Unknown keys
            raise ``TypeError``.  See *Mark style kwargs* in the marks &
            encodings guide (``docs/site/guide/marks-encodings.md``) for the
            full accepted set.

        Examples
        --------
        >>> fm.Chart(df).mark_arc().encode(x="value", color="category").coord(
        ...     fm.CoordPolar(theta="x")
        ... )
        """
        return self._set_mark("arc", **kwargs)

    def mark_image(self, **kwargs) -> "Chart":
        """Render data as raster images.

        Each row in the dataset becomes one image tile.  Supply a base64-encoded
        PNG/JPEG via the ``url`` encoding channel.  Requires Cartesian coordinates;
        returns an empty scene for Polar or Geo coord systems.

        Parameters
        ----------
        **kwargs
            Constant mark-style overrides: ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``opacity``, ``corner_radius``;
            aliases ``color``→``fill`` and ``alpha``→``opacity``.  Unknown keys
            raise ``TypeError``.  See *Mark style kwargs* in the marks &
            encodings guide (``docs/site/guide/marks-encodings.md``) for the
            full accepted set.

        Examples
        --------
        >>> fm.Chart(df).mark_image().encode(x="x", y="y", url="data_url")
        """
        return self._set_mark("image", **kwargs)

    def mark_geoshape(self, **kwargs) -> "Chart":
        """Render geographic shapes from a GeoJSON FeatureCollection.

        Pass a GeoJSON FeatureCollection dict to ``Chart(data)`` — ferrum
        auto-detects the format and splits properties into encoding channels
        and geometry into a ``__geometry__`` column.  Set the projection via
        ``Chart.coord(fm.CoordGeo(projection="mercator"))``.  Rendering with a
        ``__geometry__`` column present under any non-Geo coordinate system
        (the default ``CoordCartesian``, or an explicitly set ``CoordFlip``,
        ``CoordPolar``, or ``CoordFixed``) raises ``ValueError`` naming the
        required fix; there is no silent fallback to a flat scatter or an
        empty panel.

        Parameters
        ----------
        **kwargs
            Constant mark-style overrides: ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``opacity``, ``corner_radius``;
            aliases ``color``→``fill`` and ``alpha``→``opacity``.  Unknown keys
            raise ``TypeError``.  See *Mark style kwargs* in the marks &
            encodings guide (``docs/site/guide/marks-encodings.md``) for the
            full accepted set.

        Examples
        --------
        >>> fm.Chart(geojson_data).mark_geoshape().coord(
        ...     fm.CoordGeo(projection="equal_earth")
        ... )
        """
        return self._set_mark("geoshape", **kwargs)

    def mark_label(self, **kwargs) -> "Chart":
        """Render point-anchored text annotations near data points.

        Each label is anchored to its (x, y) data point and positioned by a
        greedy collision-avoidance algorithm (not placed at an arbitrary
        coordinate like [mark_text][ferrum.Chart.mark_text]).  An optional leader line
        (``leader_line=True``) draws a thin connecting line from the data
        point to the placed label, useful when the label lands far from its
        source point.  Use this mark when you want the renderer to find
        non-overlapping placements automatically.

        See also [mark_text][ferrum.Chart.mark_text] for a free-positioned text glyph that
        places the label at the exact encoded coordinate without collision
        avoidance or a leader line.

        Each row in the dataset becomes one text label.  By default the
        renderer uses a greedy collision-avoidance algorithm: for each label
        in row order it tries a ranked list of candidate offsets (above, below,
        right, left, diagonals) and picks the first placement whose estimated
        bounding box overlaps no previously-placed label.  When every candidate
        overlaps something, the least-bad placement is chosen.

        When **both** ``dx`` and ``dy`` are supplied explicitly, collision
        avoidance is bypassed and those fixed offsets are applied to every
        label (manual positioning path).

        Parameters
        ----------
        dx : float, optional
            Fixed horizontal offset in pixels.  Must be combined with ``dy``
            to bypass collision avoidance.
        dy : float, optional
            Fixed vertical offset in pixels.  Must be combined with ``dx``
            to bypass collision avoidance.  Default when auto-placing is to
            prefer ``dy = -8`` (above the point).
        font_size : float, optional
            Label font size in points (default 11).
        leader_line : bool, optional
            When ``True``, a thin line is drawn from each data point to its
            placed label position.  Useful when labels are placed far from
            their source points.  Default ``False``.
        **kwargs
            Additional mark-style overrides, e.g. ``fill``, ``opacity``,
            ``font_weight``. See *Mark style kwargs* in the marks &
            encodings guide (``docs/site/guide/marks-encodings.md``) for
            the full accepted set.

        Examples
        --------
        >>> fm.Chart(df).mark_label().encode(x="x:Q", y="y:Q", text="label")
        >>> fm.Chart(df).mark_label(dx=5, dy=-12).encode(x="x:Q", y="y:Q", text="label")
        >>> fm.Chart(df).mark_label(leader_line=True).encode(x="x:Q", y="y:Q", text="label")
        """
        return self._set_mark("label", **kwargs)

    # ---- Encoding ----

    def encode(self, **channels: Any) -> "Chart":
        """Set or update encoding channels on this chart.

        Each keyword argument maps a channel name to a field shorthand string
        (e.g. ``"species:N"``) or an explicit channel object (e.g.
        ``fm.X("sepal_length", type="Q")``).

        Parameters
        ----------
        x : str or X
            Field mapped to the x position.  Shorthand format:
            ``"field"`` (type inferred), ``"field:Q"`` (quantitative),
            ``"field:N"`` (nominal), ``"field:O"`` (ordinal),
            ``"field:T"`` (temporal), or ``"agg(field):Q"`` (aggregation).
        y : str or Y
            Field mapped to the y position.
        x2 : str or X2, optional
            Secondary x position (band / segment end).
        y2 : str or Y2, optional
            Secondary y position.
        color : str or Color, optional
            Field or value driving mark colour.
        fill : str or Fill, optional
            Field or value driving mark fill colour.
        stroke : str or Stroke, optional
            Field or value driving mark stroke colour.
        size : str or Size, optional
            Field or value driving mark size.
        shape : str or Shape, optional
            Field or value driving mark shape.
        opacity : str or Opacity, optional
            Field or value driving mark opacity.
        text : str or Text, optional
            Field rendered as text labels (used with ``mark_text``).
        detail : str or Detail, optional
            Additional grouping field that does not map to any visual property.
        tooltip : str or Tooltip, optional
            Field shown on hover.
        **channels
            Any other valid channel names (``x_error``, ``y_error``, ``theta``,
            ``radius``, etc.).

        A ``None`` value binds nothing: the channel keeps what it had, so
        ``encode(color=None)`` leaves the chart as it was.

        Returns
        -------
        Chart
            New ``Chart`` with updated encoding channels.

        Raises
        ------
        ValueError
            If an unknown channel name is passed.
        TypeError
            If a value is not ``None``, a string, a channel instance, or a Repeat
            placeholder.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "b", "a"]})
        >>> fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="c:N")
        Chart(mark='point', encoding=['x', 'y', 'color'])
        """
        new = self._clone()

        def declare(n: str) -> str:
            return f"encode({n}=...)"

        # wire channel -> the encode key that claimed it in THIS call, so a
        # same-call collision can name both spellings (see _append_conditional).
        claimed_here: dict[str, str] = {}
        # wire channel -> the alias-source encode key whose OWN condition=
        # resolved to it in THIS call (e.g. {"color": "fill"}), checked
        # against the final _encoding once every channel in this call has
        # been processed (see _refuse_alias_field_dropped_with_kept_conditional).
        alias_conditional_sources: dict[str, str] = {}
        # Every key is checked before its value is read, so an unknown name is
        # refused even when it carries None. A None binds nothing: the
        # channel keeps what it had, as a None in Layer(encoding=) leaves the
        # channel to the layer's parent.
        for name in channels:
            channel_class(name)
        channels = {name: value for name, value in channels.items() if value is not None}
        for name, value in channels.items():
            cls = channel_class(name)

            if isinstance(value, ConditionalSpec):
                # encode(<channel>=cond): the wire channel is KNOWN from the
                # encode key (already snake_case, e.g. "opacity", "size",
                # "stroke_width"). Resolve it through the one conditional
                # vocabulary (fill lands on color; an unsupported channel
                # refuses here, typed, instead of in serde) so both the wire
                # channel and the value-kind resolution are correct, then
                # record the conditional and auto-register its source
                # selection. Conditionals live in _conditionals, not _encoding.
                _append_conditional(
                    new,
                    value,
                    source_key=name,
                    claimed_here=claimed_here,
                    context="encode()",
                    declare=declare,
                )
                continue

            channel = channel_for_value(name, value, declare=declare(name))
            cond = channel._kwargs.get("condition")
            if cond is not None:
                # ChannelClass(field=..., condition=<ConditionalSpec>): the
                # sibling of the encode(<channel>=cond) form above --
                # same translation (stamp the channel from the encode key,
                # append to _conditionals, auto-register the selection),
                # except the base field encoding also has to be preserved.
                # Rebuild the channel without `condition` in its kwargs so
                # it never reaches to_encoding_spec_dict -- the per-channel
                # EncodingSpec.condition blob it would have round-tripped
                # through had exactly one writer (_emit_condition, deleted
                # by #237) and no reader; the real conditional lives only
                # in _conditionals. The wire
                # channel resolves through the SAME vocabulary the
                # encode(<channel>=cond) form above uses, so the two
                # spellings cannot disagree about which channels accept a
                # conditional or where fill/stroke land.
                if not isinstance(cond, ConditionalSpec):
                    raise TypeError(
                        f"{cls.__name__}(condition=...) must be a ConditionalSpec built "
                        f"via sel.when(...).otherwise(...) (or fm.when(sel).then(...)"
                        f".otherwise(...)); got {type(cond).__name__}. Use one of the two "
                        f"working forms: encode({name}=<ConditionalSpec>) or "
                        f"{cls.__name__}(field=..., condition=<ConditionalSpec>)."
                    )
                wire = conditional_wire_channel(name)
                _append_conditional(
                    new,
                    cond,
                    source_key=name,
                    claimed_here=claimed_here,
                    context="encode()",
                    declare=declare,
                )
                if wire != name:
                    # name is an alias source (only "fill" -- the
                    # refusal registry already excludes "stroke"):
                    # checked below once this channel's own field lands
                    # in new._encoding[name].
                    alias_conditional_sources[wire] = name
                remaining_kwargs = {k: v for k, v in channel._kwargs.items() if k != "condition"}
                channel = channel.__class__(channel.field, **remaining_kwargs)

            new._encoding[name] = channel
            new._transforms.extend(channel.to_implicit_transforms())

        # A positional channel bound here on a chart that keeps a written
        # record (a composition, or a host with attached layers) is its
        # callers' latest written binding (see _written_bindings).
        bound = [name for name in channels if name in _WRITTEN_CHANNELS]
        if bound and new._written_positional is not None:
            new._written_positional = {
                **new._written_positional,
                **{ch: new._encoding[ch] for ch in bound},
            }

        # H3-2 (batch-H close cycle 3): refuse an alias source's base field
        # silently dropping while its OWN condition= survives and overrides
        # the target it collided with. Checked once, after every channel in
        # this call has landed in new._encoding, so it catches both
        # encode(color=..., fill=Fill(..., condition=...)) orderings.
        _refuse_alias_field_dropped_with_kept_conditional(
            new._encoding,
            alias_conditional_sources,
            declare=declare,
        )

        # Synthesize _facet whenever facet encoding channels are passed in this
        # call. This wires encode(facet_col=...) / encode(facet_row=...) /
        # encode(facet=...) to the same _facet path that .facet() uses.
        # _Facet.merged_with is the only facet-state writer (#176(10b) + N5,
        # spec §6.3): what a prior .facet()/encode() call set and this call
        # does not restate (title, row_title, resolve, ncols/nrows, the grid's
        # other dimension) survives; what this call states wins. The
        # declaration kind is read from `channels` (the keys THIS call
        # passed), never from `new._encoding`: `_encoding` persists across
        # chained `.encode()` calls, so a facet channel from an EARLIER call
        # would otherwise still be sitting there and get misread as part of
        # THIS call's declaration.
        if any(name in _FACET_CHANNELS for name in channels):
            facet_enc = new._encoding.get("facet") if "facet" in channels else None
            col_enc = new._encoding.get("facet_col") if "facet_col" in channels else None
            row_enc = new._encoding.get("facet_row") if "facet_row" in channels else None
            existing = new._facet
            if facet_enc is not None:
                # F-L06-02: Facet(field, columns=N) writes the same ncols
                # sink Chart.facet(ncols=N) honors. Precedence is last-writer-
                # wins: an explicit `columns=` on this channel overrides an
                # ncols carried over from an earlier .facet() call. Scoped to
                # Facet only -- FacetRow/FacetCol (grid mode, below) don't
                # honor `columns` (not in their `_honored_kwargs`).
                new._facet = _Facet.merged_with(
                    existing,
                    declare=f"encode(facet={facet_enc!r})",
                    kind=FacetDeclaration.GENERIC,
                    field=facet_enc.field,
                    title=facet_enc.option("title"),
                    ncols=facet_enc.option("columns"),
                    sort=facet_enc.option("sort"),
                )
            elif col_enc is not None and row_enc is not None:
                # Grid mode: col drives the column/wrap strip (title, sort),
                # row drives the row-header strip (row_title, row_sort) --
                # both are real, independent strips here.
                new._facet = _Facet.merged_with(
                    existing,
                    declare=f"encode(facet_col={col_enc!r}, facet_row={row_enc!r})",
                    kind=FacetDeclaration.GRID,
                    col=col_enc.field,
                    row=row_enc.field,
                    title=col_enc.option("title"),
                    row_title=row_enc.option("title"),
                    sort=col_enc.option("sort"),
                    row_sort=row_enc.option("sort"),
                )
            elif col_enc is not None:
                new._facet = _Facet.merged_with(
                    existing,
                    declare=f"encode(facet_col={col_enc!r})",
                    kind=FacetDeclaration.COL,
                    col=col_enc.field,
                    title=col_enc.option("title"),
                    sort=col_enc.option("sort"),
                )
            elif row_enc is not None:
                # CRITICAL: a row-only facet lowers to WRAP mode (row unset on
                # FacetSpec), where the column/wrap strip renders and reads
                # `title`, not `row_title` (FacetSpec::column_strip_title /
                # row_strip_title). Routing FacetRow(title=...)/(sort=...) to
                # `row_title`/`row_sort` here would silently have no effect
                # on the rendered strip -- merged_with's "row" kind routes
                # each correctly for whichever mode this declaration
                # resolves to.
                new._facet = _Facet.merged_with(
                    existing,
                    declare=f"encode(facet_row={row_enc!r})",
                    kind=FacetDeclaration.ROW,
                    row=row_enc.field,
                    row_title=row_enc.option("title"),
                    row_sort=row_enc.option("sort"),
                )
            # As `.facet()` does: another frame's renamed copy of a facet
            # column fills it, so that frame's rows carry their own values.
            new = _coalesce_facet_rhs_columns(new)

        return new

    def layer(self, *layers) -> "Chart":
        """Add one or more layer objects to this chart.

        Accepts both public ``Layer`` instances (user-facing API) and
        internal ``_Layer`` instances (used by ferrum internals). An
        internal ``_Layer`` is appended verbatim, with no conditional
        stripping -- every internal builder that constructs one is required
        to have already routed any ``condition=`` through ``encode()`` (or
        left none) on the chart-level encoding it derives ``_Layer.encoding``
        from, so it never carries a live ``condition=`` kwarg for this
        method to miss (see :func:`_strip_layer_conditionals`, which only
        the public-``Layer`` branch below calls).

        A ``Layer`` without ``data`` is attached to this chart and reads the
        chart's data: its source rows after the steps added with
        ``.transform(...)`` (including a stat transform such as ``fm.Swarm``
        or ``fm.Bin``, and an aggregate or bin on the chart's own encoding),
        never the statistics a composite mark computes for its own layers (a
        histogram's bins and counts, a density grid, a fitted curve). A
        column the chart's data does not hold is refused at render, and a
        statistic is refused with the remedy: compute it in the chart's
        ``.transform(...)`` so every layer reads it, or draw it from its own
        chart composed with ``+``. Each of ``x``, ``y``, ``x2`` and ``y2``
        the layer does not bind is the one the caller wrote for the chart,
        the latest per channel, so an ``.encode(x=...)`` after this call
        reaches the layer. A ``.transform(...)`` after this call joins the
        chart's steps, so the order of the two calls does not matter.

        A ``Layer(data=df, ...)`` composes with this chart exactly as
        ``chart + Chart(df)`` drawing that one layer does: it reads all of
        ``df``, never this chart's steps, as its own operand. It is attached
        here all the same, so this chart's bindings and conditionals reach
        it, as they reach every layer attached here, each field naming the
        same-named column of ``df``. Over this chart's own frame, a step
        written on the result reaches it, as on ``+``. On a chart
        with no data, the first ``Layer`` with ``data`` supplies the chart's
        data instead, and the layer is attached to it. Layers keep the order
        they are given in.

        Parameters
        ----------
        *layers : Layer or _Layer
            Layer objects to append. Public ``Layer`` instances may carry an
            independent ``data=`` DataFrame.

        A ``Layer(encoding={...})`` channel's ``condition=`` is honored: it is
        lifted into this chart's real conditional pipeline (the same
        ``_conditionals`` list ``encode(<channel>=cond)`` populates), gated
        by the same conditional vocabulary -- a ``stroke``/``shape``/``x``/
        ``y`` conditional inside a layer's encoding raises the same typed
        ``ValueError`` ``encode()`` raises, instead of silently reaching the
        Rust wire's dead per-channel ``condition`` blob (see
        :func:`_strip_layer_conditionals`). A conditional declared on a
        ``Layer``'s own encoding belongs to that layer alone; one declared
        on this chart reaches every layer attached to it, as this chart's
        field encodings do.

        Each attached layer draws its own bindings merged over this chart's,
        channel by channel: a channel the layer binds (``fill`` and
        ``stroke`` bind ``color``) is its own, and every other binding this
        chart carries reaches it whole, aggregate, bin, scale and title
        included, and is computed over the rows the layer reads. A layer's
        own binding never reaches this chart's marks or another layer's: a
        ``Layer`` colored by a field leaves the chart's own marks uncolored.
        The panel still draws one scale and one legend per channel, over
        every layer that draws it.

        Returns
        -------
        Chart
            This chart with the new layers appended.
        """
        from ferrum.layer import Layer as PublicLayer

        new = _attach_host(self)
        new._layers = _take_layers(new)
        # wire channel -> the encode key that claimed it in THIS layer() call,
        # shared across every layer being attached (mirrors encode()'s
        # per-call claimed_here) -- see _strip_layer_conditionals.
        claimed_here: dict[str, str] = {}
        for ly in layers:
            if isinstance(ly, _Layer):
                new._layers = new._layers + [ly]
            elif isinstance(ly, PublicLayer):
                if ly.data is not None and new._data is not None:
                    new = new._merge(_layer_operand(ly, host=new), host=True)
                    continue
                if ly.data is not None:
                    new._data = to_polars(ly.data)
                encoding, owned = _strip_layer_conditionals(
                    new, ly.encoding, claimed_here=claimed_here
                )
                new._layers = new._layers + [
                    _Layer(
                        name=ly.name,
                        mark=ly.mark,
                        encoding=encoding,
                        mark_kwargs=dict(ly.mark_kwargs) if ly.mark_kwargs else None,
                        conditionals=owned,
                        chart_data=True,
                    )
                ]
            else:
                raise TypeError(
                    f"layer() expects Layer or _Layer instances; got {type(ly).__name__}"
                )
        return new

    @property
    def layer_names(self) -> list[str]:
        """Named sub-layers after composite-mark desugar resolution.

        Returns ``[]`` for single-mark charts with no layers.  Accessing
        this property forces resolution of any pending ``_PendingMark``.
        """
        resolved = self._resolve_pending()
        if not resolved._layers:
            return []
        return [ly.name for ly in resolved._layers if ly.name is not None]

    def transform(self, *transforms) -> "Chart":
        """Append one or more data transforms to the chart's pipeline.

        Transforms are executed in order by the Rust engine before rendering.
        Multiple calls to ``transform()`` accumulate — each appends to the
        existing pipeline.

        The steps are the chart's own chain: on a single chart they run
        before its encoding statistic (an ``aggregate=`` or ``bin=`` on a
        channel) and before a composite mark's statistics, whether this call
        comes before or after ``.encode(...)``, ``.layer(...)`` or ``+
        fm.SecondaryY(...)``, and every layer attached to the chart reads
        their output. A step reading a column only such a statistic computes
        is refused: compute it with ``.transform(...)``
        (``fm.transform_aggregate``), or write the step on a ``+``
        composition, whose steps run on a statistic's output.

        On a composition (``a + b``) each step reaches the left operand and
        every operand over the composition's frame, however it was composed,
        and never an operand over a different frame (as Vega-Lite's layer
        transform reaches the children that inherit its data). Over each
        reached operand's rows the step runs after that operand's own steps
        and before its statistics when those rows carry every column the step
        reads, so ``filter('datum.y > 8')`` filters a histogram's rows before
        it bins; otherwise it runs on the first rows after the statistics
        that carry them, so ``filter('datum.count > 1')`` keeps the bars
        above a count of 1. It runs at most once over each operand's rows.
        An operand whose rows never carry them draws as written, and a step
        no reached operand's rows can read is refused, naming the column and
        the left operand's columns. A step that names its output publishes
        it inside each reached operand and leaves that operand's rows as
        they were.

        Parameters
        ----------
        *transforms
            One or more transform specs, as built by the ``fm.transform_*``
            builders (e.g. ``fm.transform_filter(...)``,
            ``fm.transform_aggregate(...)``, ``fm.transform_window(...)``).
            Accepts any object that serialises to a valid ``TransformSpec`` JSON
            shape.

        Returns
        -------
        Chart
            New ``Chart`` with the additional transforms appended.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").transform(
        ...     fm.transform_filter("datum.x > 1")
        ... )
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        if self._operands:
            # A composition: each step runs as one composed step that every
            # operand over the composition's frame runs over its own rows.
            copies = tuple(_ComposedStep(t) for t in transforms)
            new._operands = reached_by_composition(self._operands, copies)
            return new
        # The chart's own steps run before the statistics its mark lowered,
        # whether written before or after the mark resolved.
        at = _statistics_start(list(self._transforms))
        new._transforms = [*self._transforms[:at], *transforms, *self._transforms[at:]]
        return new

    def _with_statistics(self, *steps, reading: Literal["tail", "rows"] = "tail") -> "Chart":
        """Return a clone running *steps*, statistics a figure lowers for its own layers.

        They open their own lowering, behind their own statistics marker,
        as a composite mark's statistics do: with ``reading="rows"`` it sits
        right after every step the caller wrote, before any marker already
        in the chain, so a named step among *steps* still reads the
        caller's rows and acts on the figure as on a hand-built chart; with
        ``"tail"`` it is appended after every lowering already there, so
        *steps* read that tail's output. On a ``+`` composition, whose data
        is its left operand's, they join that operand's statistics the same
        way, as a later pending mark's do. A pending composite mark resolves
        first, as it does for ``.layer(...)`` and ``+``, so the steps take
        their place among its statistics by *reading* on every door.
        """
        if self._pending_stat_mark is not None:
            self = _attach_host(self)
        new = self._clone()
        if self._operands:
            left = self._operands[0]
            at = (
                _statistics_start(list(left.statistics))
                if reading == "rows"
                else len(left.statistics)
            )
            statistics = (
                *left.statistics[:at],
                *_with_statistics_marker(list(steps)),
                *left.statistics[at:],
            )
            new._operands = (replace(left, statistics=statistics), *self._operands[1:])
            return new
        transforms = list(self._transforms)
        at = _statistics_start(transforms) if reading == "rows" else len(transforms)
        transforms[at:at] = _with_statistics_marker(list(steps))
        new._transforms = transforms
        return new

    # ---- Composition operators ----

    def __add__(self, other: Any) -> "Chart":
        """Compose a chart with another chart, configuration, annotation, or structural feature.

        Dispatches on the type of *other*:

        - ``Chart`` — overlay as a multi-layer composite with shared scales
        - ``Configure`` — append chart-level configuration
        - ``Annotate`` or annotation primitive — append annotation layer
        - ``SecondaryY`` — desugar to an appended independent-y layer (GH #52):
          same mark/x/color/opacity/axis/scale semantics as before, but now
          renders through the real per-layer independent-y subsystem (band
          reservation, real axis layout, interactivity) instead of the
          legacy overlay-only ``secondary_axis`` renderer.
        - ``BreakAxis``, ``Inset`` — append structural feature

        When composing two ``Chart`` objects, data-merging uses diagonal
        concatenation with null-padding for non-overlapping columns.

        Parameters
        ----------
        other : Chart, Configure, Annotate, annotation primitive, SecondaryY, BreakAxis, or Inset
            The element to compose with this chart.

        Returns
        -------
        Chart

        Raises
        ------
        TypeError
            (via ``NotImplemented``) if ``other`` is not a recognized type.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
        >>> line = fm.Chart(df).mark_line().encode(x="x", y="y")
        >>> layered = scatter + line
        """
        from ferrum.configure import Configure
        from ferrum.annotation.container import Annotate
        from ferrum.annotation.primitives import (
            AnnotationArrow,
            AnnotationBracket,
            AnnotationCallout,
            AnnotationImage,
            AnnotationLine,
            AnnotationRect,
            AnnotationSpan,
            AnnotationText,
        )
        from ferrum.structural import SecondaryY, BreakAxis, Inset

        # Dispatch: Configure layer
        if isinstance(other, Configure):
            new = self._clone()
            new._configure = new._configure + [other]
            return new

        # Dispatch: Annotate container
        if isinstance(other, Annotate):
            new = self._clone()
            new._annotations = new._annotations + [other]
            return new

        # Dispatch: bare annotation primitive — wrap in Annotate
        _ANNOTATION_TYPES = (
            AnnotationText,
            AnnotationArrow,
            AnnotationRect,
            AnnotationLine,
            AnnotationSpan,
            AnnotationBracket,
            AnnotationCallout,
            AnnotationImage,
        )
        if isinstance(other, _ANNOTATION_TYPES):
            new = self._clone()
            new._annotations = new._annotations + [Annotate(other)]
            return new

        # Dispatch: SecondaryY — desugars to an appended independent-y layer
        # (GH #52), not a structural-spec entry. See _desugar_secondary_y.
        if isinstance(other, SecondaryY):
            return _desugar_secondary_y(self, other)

        # Dispatch: structural features
        if isinstance(other, (BreakAxis, Inset)):
            new = self._clone()
            new._structural = new._structural + [other]
            return new

        if not isinstance(other, Chart):
            return NotImplemented
        return self._merge(other, host=False)

    def _merge(self, other: "Chart", *, host: bool) -> "Chart":
        """Compose this chart with *other* as ``chart + other`` does.

        The result's operands are this chart's followed by *other*'s
        (:mod:`ferrum._compose`), each keeping its own frame, steps and
        statistics, and its layers those of both. A step written on the
        result later reaches every operand over its frame, this chart's.

        With ``host=True`` this chart is a host being extended
        (``Chart.layer``'s ``Layer(data=...)``), not a peer operand: its
        chart-wide conditionals are left unbound, so they reach every layer
        attached to it, before or after this merge, and the operand's lack of
        the host's theme or facet is not a conflict to warn about. With ``host=False``
        (``chart + chart``) each operand's conditionals belong to the layers
        that operand contributes.
        """
        lhs_in, rhs_in = _with_polars_operands(self, other)
        # Resolve pending statistical marks before snapshotting encoding dicts.
        lhs = lhs_in._resolve_pending()
        rhs = rhs_in._resolve_pending()
        new = lhs._clone()
        # The composition's statistics are its operands'.
        new._composite_source = None
        # The composition binding is the result's own, not a regression
        # figure's: a later binding on it is read as any other chart-level
        # binding is, not through this record.
        new._post_build_positional = None
        # Read before resolution, so a pending composite's written x/y are kept
        # and a composed left passes its own record on.
        new._written_positional = _written_bindings(lhs_in)
        lhs_layers, _ = _expand_layers(lhs)
        rhs_layers, _ = _expand_layers(rhs)
        # An operand's own encoding is the parent of the layers it contributes
        # and of no other, so its bindings move onto those layers and the
        # composition keeps only the left's positional, polar and facet
        # channels: the axis fallback and the panel's facet. What the caller
        # later writes on the composition itself is the parent of every
        # layer. A host being extended keeps its encoding, the parent of
        # every layer attached to it, this one included.
        rhs_layers = _scoped_to_layers(rhs_layers, rhs._encoding)
        if not host:
            lhs_layers = _scoped_to_layers(lhs_layers, lhs._encoding)
            new._encoding = _composition_encoding(lhs._encoding)
        # An operand's own position= adjusts the layers it contributes and no
        # sibling's, so it moves onto those layers and the composition keeps
        # none. A host being extended keeps its position, the parent of every
        # layer attached to it, as it keeps its encoding.
        rhs_layers = _carry_layer_position(rhs_layers, rhs._position)
        if not host:
            lhs_layers = _carry_layer_position(lhs_layers, lhs._position)
            new._position = None
        # Each operand's conditionals belong to the layers that operand
        # contributes, carried by identity through every rewrite below. A
        # host's own stay chart-wide.
        if not host:
            lhs_layers = _bind_operand_conditionals(lhs_layers, lhs._conditionals)
        rhs_layers = _bind_operand_conditionals(rhs_layers, rhs._conditionals)
        # The RHS's chart encoding stops applying once it is an operand, so
        # every RHS layer that takes positional channels from that chart gets
        # them now, from the RHS's own bindings: a layer attached to the RHS
        # takes the x/y/x2/y2 the RHS's callers wrote, as when the RHS's own
        # spec is built, and a layered RHS's tail readers take the x/y they
        # do not bind from the RHS's encoding, as on its own chart. The
        # composition's written bindings fill only what the RHS never wrote.
        rhs_layers = [
            replace(l, positional_source="written")
            if l.chart_data and l.positional_source is None
            else l
            for l in rhs_layers
        ]
        if rhs._layers is not None:
            rhs_layers = _reading_operand_tail(rhs_layers)
        rhs_layers = _fill_chart_positional(rhs_layers, rhs)
        # The RHS's own coordinate system travels with the layers it
        # contributes (#172(1)). `new` keeps the LHS's coord at chart level --
        # the panel's chrome, aspect ratio and scale domains are one thing --
        # so without this the RHS's declaration was simply dropped: a
        # `CoordFlip` rendered unflipped, silently.
        rhs_layers = _carry_layer_coord(rhs_layers, lhs._coord, rhs._coord)

        # When the LHS is an annotate_* helper chart, its annotation primitive
        # fully describes the visual element — the mark layers must be excluded
        # (same logic as the RHS path below). Use RHS layers only.
        if lhs._annotation_primitive is not None:
            if rhs._annotation_primitive is not None:
                from ferrum._layer import _Layer as _CarrierLayer

                carrier = _CarrierLayer(
                    mark="point",
                    encoding=dict(lhs._encoding),
                    mark_kwargs={"opacity": 0},
                )
                new._layers = [carrier]
                # Rebuild from the two primitive expansions ONLY -- do not
                # keep `new._annotations`'s carried-over value.  `new` was
                # cloned from `lhs`, and a labelled annotate_hline/vline's
                # `_annotations` already holds its (dormant) label Annotate
                # for standalone rendering; keeping it here would double it
                # up with the label re-supplied by
                # `_annotations_from_primitive(lhs._annotation_primitive)`.
                # Both operands are fresh annotate_* helpers whose entire
                # visual content lives in their primitives, so the clone's
                # `_annotations` carries nothing that isn't already covered.
                new._annotations = _annotations_from_primitive(
                    lhs._annotation_primitive
                ) + _annotations_from_primitive(rhs._annotation_primitive)
            else:
                new = rhs._clone()
                new._layers = rhs_layers
                if not host:
                    new._position = None
                new._annotations = new._annotations + _annotations_from_primitive(
                    lhs._annotation_primitive
                )
            new._annotation_primitive = None
            if not host:
                _warn_on_layer_conflicts(lhs, rhs)
            return new

        # When the RHS is an annotate_* helper chart, its annotation primitive
        # fully describes the visual element for both SVG and interactive
        # rendering.  The mark layers inside the annotate_* chart (mark_rule,
        # mark_rect, mark_line, mark_text) must be excluded — they would
        # produce a duplicate rendering alongside the annotation primitive.
        # The RHS's data and chain are not needed by any layer.
        if rhs._annotation_primitive is not None:
            new._layers = lhs_layers  # LHS layers only — no RHS mark layers
            new._annotations = new._annotations + _annotations_from_primitive(
                rhs._annotation_primitive
            )
            if not host:
                _warn_on_layer_conflicts(lhs, rhs)
            return new

        # Every operand's layers read its own chain's rows, where the
        # pipeline passes them none of the chart's positional channels, so
        # each tail reader takes those it does not bind from its operand's
        # chart: the RHS's own encoding now, then the composition's written
        # bindings; the LHS's the composition's encoding when the spec is
        # built, so a later `.encode(x=...)` on the composition reaches them.
        rhs_layers = _fill_chart_positional(_reading_operand_tail(rhs_layers), rhs)
        rhs_layers = [
            replace(l, positional_source="written") if l.positional_source is not None else l
            for l in rhs_layers
        ]
        lhs_layers = _reading_operand_tail(lhs_layers)
        # The operands: the LHS's, then the RHS's over the merged frame
        # registry, each RHS layer naming its operand's new place.
        # A composed operand's encoding, moved onto its layers above, stays
        # the parent of its operands' composite statistics.
        lhs_operands = chart_operands(lhs)
        if not host and lhs._operands:
            lhs_operands = carried_by(lhs_operands, lhs._encoding)
        rhs_operands = chart_operands(rhs)
        if rhs._operands:
            rhs_operands = carried_by(rhs_operands, rhs._encoding)
        frames, moved = merged_frames(chart_frames(lhs_in), chart_frames(rhs_in))
        new._operands = lhs_operands + tuple(
            replace(op, frame=moved[op.frame]) for op in rhs_operands
        )
        rhs_layers = [replace(l, operand=l.operand + len(lhs_operands)) for l in rhs_layers]
        new._frames = frames
        new._data, new._frame_selectors = combined_frames(frames)
        new._transforms = []
        new._layers = lhs_layers + rhs_layers
        # A facet the left operand already carries partitions the combined
        # rows too, so another frame's renamed copy of a facet column fills
        # it, as .facet() does when declared after the merge.
        _coalesce_facet_rhs_columns(new)
        # Merge RHS selections and conditionals into the layered chart
        # so interactive features from all layers are preserved. Each side's
        # conditionals stay owned by that side's layers (bound above).
        _refuse_cross_operand_reactive_names([new, rhs])
        if rhs._selections:
            for s in rhs._selections:
                if s is None:
                    continue
                _append_unique_by_name(new._selections, s, entry="chart + chart")
        if rhs._conditionals:
            # One conditional per wire channel, enforced across the merge
            # too: the merged chart's `conditionals` list holds at most one
            # per channel, whichever layers each would reach (a conditional
            # reaches the layers that declare it; one declared on a chart
            # or host as a whole reaches every layer), so two colliding on
            # one channel is exactly as undefined here as within one
            # `encode()` call. Checked against `new._conditionals` as it
            # accumulates, so a third+ chart in a chained `a + b + c` merge
            # collides against everything merged so far.
            for cond in rhs._conditionals:
                wire = _channel_claimed_by(cond)
                earlier = _conditional_on_wire(new, wire) if wire is not None else None
                if earlier is not None:
                    raise ValueError(
                        _conditional_collision_message(
                            context="chart + chart",
                            wire=wire,
                            first=_conditional_declaration_label(earlier, whose="the left chart's"),
                            second=_conditional_declaration_label(cond, whose="the right chart's"),
                            detail=(
                                " -- the merged chart holds one conditional per wire "
                                "channel, whichever layers each would reach"
                            ),
                        )
                    )
                new._conditionals.append(cond)
        if rhs._params:
            existing_param_names = {p.name for p in new._params if p is not None}
            for p in rhs._params:
                if p is None:
                    continue
                if p.name not in existing_param_names:
                    new._params.append(p)
        # Merge RHS configure/annotation/structural/override slots.
        if rhs._configure:
            new._configure = new._configure + rhs._configure
        if rhs._annotations:
            new._annotations = new._annotations + rhs._annotations
        if rhs._structural:
            new._structural = new._structural + rhs._structural
        if rhs._overrides:
            from ferrum._override_apply import merge_overrides

            new._overrides = merge_overrides(new._overrides, rhs._overrides)
        if not host:
            _warn_on_layer_conflicts(lhs, rhs)
        return new

    # __or__/__and__ (horizontal/vertical concatenation) are inherited from
    # _ChartOperatorMixin, shared with every composition view (#245(5)).

    # ---- Facet / coord / theme ----

    def facet(
        self,
        field: Any = None,
        *,
        row: Any = None,
        col: Any = None,
        ncols: Optional[int] = None,
        nrows: Optional[int] = None,
        title: Optional[str] = None,
        row_title: Optional[str] = None,
        sort: Any = None,
        row_sort: Any = None,
    ) -> "Chart":
        """Facet this chart into small multiples by a field.

        Two modes are supported:

        - **Wrap** — a single field is wrapped across columns (rows auto).
          Pass ``field=`` or equivalently ``col=`` alone.
        - **Grid** — two fields define row × column layout.  Pass both
          ``row=`` and ``col=``.

        Parameters
        ----------
        field : str or encoding, optional
            Column name used for wrap-mode faceting, read the
            column-keyword way (the same reading a figure's ``hue=``
            gets): shorthand (``"g:N"``) parses as ``encode(facet=...)``
            parses it, a non-shorthand string (``"Weight (kg)"``) is the
            column it names rather than raising ``encode()``'s parse
            error, and a ``fm.Facet(...)`` object works directly. A value
            that is none of str/``Facet``/``Repeat`` placeholder raises
            the parse's own ``TypeError``. Where this call passes no
            ``title=``/``sort=``, the channel's own are used, and a
            ``fm.Facet(..., columns=N)`` sets the wrap count (``ncols``)
            when this call passes no ``ncols=``. Mutually exclusive with
            using both ``row`` and ``col`` together. A prior
            facet declaration (from an earlier ``.facet()`` or
            ``encode(facet...)`` call) is merged, not discarded: a field
            this call does not restate (``title``, ``ncols``, ``nrows``,
            the ``share_scale()`` resolution) survives, and ``field=``
            replaces an existing **grid** facet with a wrap on this field
            (its ``row_title``, which has no home in wrap mode, is
            dropped).
        row : str or encoding, optional
            Column name for the row dimension (grid mode), read as
            ``encode(facet_row=...)`` reads its channel (shorthand or a
            ``fm.FacetRow(...)`` object; see ``field`` above for the
            title/sort fallback and the parse's own ``TypeError``).  When
            used alone, behaves as wrap mode on the row axis, UNLESS it
            merges onto a prior facet with a different, column-role
            dimension (a column wrap, or a grid), in which case it promotes
            to/updates a grid (row=this field, col=the prior's), keeping
            the prior's field on its own axis.
        col : str or encoding, optional
            Column name for the column dimension (wrap or grid mode), read
            as ``encode(facet_col=...)`` reads its channel (shorthand or a
            ``fm.FacetCol(...)`` object; see ``field`` above). Used alone,
            the same merge rule applies with the roles reversed.
        ncols : int or None, optional
            Maximum number of columns in wrap mode.  ``None`` lets the renderer
            choose.
        nrows : int or None, optional
            Maximum number of rows.  ``None`` lets the renderer choose.
        title : str, optional
            Title for the primary/wrap dimension's strip (the column strip in
            grid mode; the only strip in wrap mode, including a row-only
            facet). Strip text renders ``"{title}: {value}"`` when set, the
            bare value otherwise.
        row_title : str, optional
            Title for the row strip in **grid** mode only (``row=`` and
            ``col=`` both given) -- equivalent to
            ``encode(facet_row=fm.FacetRow(field, title=...))`` alongside a
            titled ``facet_col=``. A single-dimension declaration (``field=``,
            ``col=`` alone, or ``row=`` alone) has only one title, which
            ``title=`` already sets -- ``row_title=`` has no separate meaning
            there and raises, even when merging with a prior facet would
            produce a grid, since ``title=`` already covers that one
            dimension's title regardless of which mode the merge resolves
            to.
        sort : str, list, or dict, optional
            Panel order for the primary/wrap dimension's strip, routed exactly
            like ``title=``, on the same contract as ``Facet(sort=...)``: a
            direction (``"ascending"``/``"descending"``), an explicit list of
            values, or a sort-field object (``{"field": ..., "op": ...,
            "order": ...}``). A malformed sort-field object raises
            ``ValueError`` naming ``Chart.facet(sort=...)``.
        row_sort : str, list, or dict, optional
            Panel order for the row strip in **grid** mode only, routed
            exactly like ``row_title=`` (equivalent to
            ``encode(facet_row=fm.FacetRow(field, sort=...))``), with the same
            accepted forms as ``sort=``. A single-dimension declaration has
            only one panel order, which ``sort=`` already sets, so
            ``row_sort=`` raises there.

        Returns
        -------
        Chart
            New ``Chart`` with faceting configured.

        Raises
        ------
        ValueError
            If neither ``field``, ``row``, nor ``col`` is provided, if
            ``row_title=`` or ``row_sort=`` is given alongside a
            single-dimension declaration (``field=``, ``col=`` alone, or
            ``row=`` alone), or if ``sort=``/``row_sort=`` is a malformed
            sort-field object.
        TypeError
            If ``field``/``row``/``col`` is none of a string, the matching
            encoding object, or a ``Repeat`` placeholder.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1,2,3]*3, "y": [4,5,6]*3,
        ...                    "g": ["a","a","b","b","c","c","a","b","c"]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").facet(col="g", ncols=2)
        Chart(mark='point', encoding=['x', 'y'])
        """

        def _refuse_row_title_on_wrap(declared_as: str) -> None:
            if row_title is not None:
                raise ValueError(
                    f"facet(row_title=...) has no effect on a facet declared via "
                    f"{declared_as}: a single-dimension declaration has only one "
                    "title, which `title=` already sets; `row_title=` names a "
                    "second, distinct row-strip title that only a genuine "
                    "two-dimension facet (`row=` and `col=` together) has. Pass "
                    "the title via `title=` instead, or use both `row=` and "
                    "`col=` for a grid facet."
                )
            if row_sort is not None:
                raise ValueError(
                    f"facet(row_sort=...) has no effect on a facet declared via "
                    f"{declared_as}: a single-dimension declaration has only one "
                    "panel order, which `sort=` already sets; `row_sort=` names a "
                    "second, distinct row-strip order that only a genuine "
                    "two-dimension facet (`row=` and `col=` together) has. Pass "
                    "the order via `sort=` instead, or use both `row=` and "
                    "`col=` for a grid facet."
                )

        for kwarg, value in (("sort", sort), ("row_sort", row_sort)):
            if value is not None:
                _validate_sort_object("Chart.facet", value, kwarg=kwarg)
        ncols = coerce_positive_int("Chart.facet(ncols=...)", ncols)
        nrows = coerce_positive_int("Chart.facet(nrows=...)", nrows)
        new = self._clone()
        existing = self._facet

        field_ch = _parsed_facet_arg("facet", field, declare="Chart.facet(field=...)")
        row_ch = _parsed_facet_arg("facet_row", row, declare="Chart.facet(row=...)")
        col_ch = _parsed_facet_arg("facet_col", col, declare="Chart.facet(col=...)")

        if field_ch is not None:
            _refuse_row_title_on_wrap("field=")
            new._facet = _Facet.merged_with(
                existing,
                declare=f"Chart.facet(field={field_ch.field!r})",
                kind=FacetDeclaration.FACET_FIELD,
                field=field_ch.field,
                title=title if title is not None else field_ch.option("title"),
                sort=sort if sort is not None else field_ch.option("sort"),
                ncols=ncols if ncols is not None else field_ch.option("columns"),
                nrows=nrows,
            )
        elif row_ch is not None and col_ch is not None:
            new._facet = _Facet.merged_with(
                existing,
                declare=f"Chart.facet(row={row_ch.field!r}, col={col_ch.field!r})",
                kind=FacetDeclaration.GRID,
                row=row_ch.field,
                col=col_ch.field,
                title=title if title is not None else col_ch.option("title"),
                row_title=row_title if row_title is not None else row_ch.option("title"),
                sort=sort if sort is not None else col_ch.option("sort"),
                row_sort=row_sort if row_sort is not None else row_ch.option("sort"),
                ncols=ncols,
                nrows=nrows,
            )
        elif col_ch is not None:
            _refuse_row_title_on_wrap("col= alone")
            new._facet = _Facet.merged_with(
                existing,
                declare=f"Chart.facet(col={col_ch.field!r})",
                kind=FacetDeclaration.COL,
                col=col_ch.field,
                title=title if title is not None else col_ch.option("title"),
                sort=sort if sort is not None else col_ch.option("sort"),
                ncols=ncols,
                nrows=nrows,
            )
        elif row_ch is not None:
            _refuse_row_title_on_wrap("row= alone")
            new._facet = _Facet.merged_with(
                existing,
                declare=f"Chart.facet(row={row_ch.field!r})",
                kind=FacetDeclaration.ROW,
                row=row_ch.field,
                row_title=title if title is not None else row_ch.option("title"),
                row_sort=sort if sort is not None else row_ch.option("sort"),
                ncols=ncols,
                nrows=nrows,
            )
        else:
            raise ValueError("facet() requires either `field=`, or `row=`/`col=`")
        # A + merge renamed the right operand's copy of a shared column, so
        # its rows are null in a facet on that name; fill the facet field
        # from the copies the merges recorded so every layer's rows carry it.
        new = _coalesce_facet_rhs_columns(new)
        return new

    def share_scale(
        self,
        *,
        x: Optional[str] = None,
        y: Optional[str] = None,
    ) -> "Chart":
        """Set per-channel facet scale resolution for this faceted chart.

        Each panel can either lock to the union domain across all partitions
        (``"shared"``, the default) or compute its own domain from its
        partition's data (``"independent"``).

        Parameters
        ----------
        x : str or None, optional
            Scale resolution for the x channel.  ``"shared"`` or
            ``"independent"``.  ``None`` leaves the current setting unchanged.
        y : str or None, optional
            Scale resolution for the y channel.  ``"shared"`` or
            ``"independent"``.  ``None`` leaves the current setting unchanged.

        Returns
        -------
        Chart
            New ``Chart`` with the scale resolution updated.

        Raises
        ------
        ValueError
            If called before ``facet()``, or if a value is not ``"shared"``
            or ``"independent"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"g": ["a","a","b","b"], "x": [1,2,1,2],
        ...                    "y": [1.0,2.0,100.0,101.0]})
        >>> (fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        ...  .facet(col="g").share_scale(y="independent"))
        Chart(mark='point', encoding=['x', 'y'])
        """
        if self._facet is None:
            raise ValueError("share_scale() requires a faceted chart; call facet() first")
        to_validate = {ch: mode for ch, mode in (("x", x), ("y", y)) if mode is not None}
        _validate_share_modes(to_validate)
        # Build updated resolve dict from existing + new values.
        existing_resolve = self._facet.resolve or {}
        new_resolve = dict(existing_resolve)
        if x is not None:
            new_resolve["x"] = x
        if y is not None:
            new_resolve["y"] = y
        new = self._clone()
        # _Facet.merged_with is the only facet-state writer (spec §6.3):
        # share_scale() states only `resolve`, so every other field --
        # mode, field/row/col, title, row_title, ncols, nrows -- carries
        # through `merged_with`'s "resolve" kind unchanged.
        new._facet = _Facet.merged_with(
            self._facet,
            declare="Chart.share_scale(...)",
            kind=FacetDeclaration.RESOLVE,
            resolve=new_resolve if new_resolve else None,
        )
        return new

    def theme(self, theme: Any) -> "Chart":
        """Attach a ``Theme`` to this chart, overriding the process-level default.

        Per-chart theme always wins over ``ferrum.set_default_theme()``.
        Theme objects are immutable value classes — modifying the original
        ``Theme`` after calling ``.theme()`` has no effect on the chart.

        Parameters
        ----------
        theme : Theme
            A ``ferrum.Theme`` instance (e.g. the built-in ``fm.themes.dark``,
            or ``fm.Theme(background="white", font="serif")``).

        Returns
        -------
        Chart
            New ``Chart`` with the specified theme attached.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> dark = fm.themes.dark
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").theme(dark)
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        new._theme = theme
        return new

    def axis(
        self,
        *,
        x: Optional[bool] = None,
        y: Optional[bool] = None,
        show: Optional[bool] = None,
    ) -> "Chart":
        """Suppress (or restore) the chart's x/y axis decorations.

        Spec-level axis suppression — when ``x=False`` (or ``y=False``), the
        corresponding axis line, ticks, tick labels, and axis title are
        omitted at layout time. The plot area's pixel rect is unchanged
        (gutters reserved for axis decorations are preserved), so this
        method is intended for sub-charts whose axes are shared with a
        neighbouring chart in a compound view: clustermap dendrograms,
        JointChart marginals, RepeatChart off-diagonal panels.

        Parameters
        ----------
        x : bool, optional
            ``False`` hides the x axis; ``True`` shows it; ``None`` leaves
            the current setting (default visible).
        y : bool, optional
            ``False`` hides the y axis; ``True`` shows it; ``None`` leaves
            the current setting (default visible).
        show : bool, optional
            Shorthand for ``axis(x=show, y=show)``. Mutually exclusive with
            per-axis ``x``/``y`` arguments.

        Returns
        -------
        Chart
            New ``Chart`` with the requested axis-visibility settings.

        Raises
        ------
        ValueError
            If ``show`` is combined with ``x`` or ``y``.

        Examples
        --------
        Hide both axes (e.g. for a dendrogram panel):

        >>> chart.axis(show=False)

        Hide just the x axis (top marginal of a JointChart):

        >>> chart.axis(x=False)
        """
        if show is not None and (x is not None or y is not None):
            raise ValueError("Chart.axis: pass either show=… OR x=/y=, not both")
        new = self._clone()
        if show is not None:
            new._axis_x = show
            new._axis_y = show
        else:
            if x is not None:
                new._axis_x = x
            if y is not None:
                new._axis_y = y
        return new

    # ---- Declarative configuration surface (provided by ConfigureMixin) ----

    def override(self, **kwargs: Any) -> "Chart":
        """Store low-level spec-path overrides to be applied at render time.

        Overrides are merged into `_overrides` with later calls winning
        on key conflicts, except that a ``mark_<leaf>=None`` is unset and
        leaves a value an earlier call set. The ``override()`` call never
        mutates the receiver.

        Parameters
        ----------
        **kwargs
            Arbitrary spec-path key/value pairs forwarded verbatim to the
            renderer (e.g. ``x_axis_label_angle=-45``).

        Returns
        -------
        Chart
            New ``Chart`` with the overrides merged.

        Examples
        --------
        >>> chart.override(x_axis_label_angle=-45, width=600)
        """
        from ferrum._override_apply import merge_overrides

        new = self._clone()
        new._overrides = merge_overrides(new._overrides, kwargs)
        return new

    def coord(self, coord: Any) -> "Chart":
        """Set the coordinate system for this chart.

        Accepts ``CoordFlip`` (swap x/y axes), ``CoordCartesian`` (the
        default; also carries ``xlim``/``ylim``/``expand``/``clip``),
        ``CoordPolar`` (pie/donut/radar, required by ``mark_arc``),
        ``CoordGeo`` (map projections, required by ``mark_geoshape`` when
        the data carries geometry), and ``CoordFixed`` (fixed aspect ratio).

        Parameters
        ----------
        coord : CoordFlip | CoordCartesian | CoordPolar | CoordGeo | CoordFixed
            A coordinate-system object.  Pass ``fm.CoordFlip()`` to swap the
            horizontal and vertical axes.

        Returns
        -------
        Chart
            New ``Chart`` with the coordinate system set.

        Raises
        ------
        TypeError
            If ``coord`` is not a supported coordinate-system type.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
        >>> fm.Chart(df).mark_bar().encode(x="cat", y="val").coord(fm.CoordFlip())
        Chart(mark='bar', encoding=['x', 'y'])
        """
        from ferrum.coord import CoordCartesian, CoordFixed, CoordFlip, CoordGeo, CoordPolar

        new = self._clone()
        if isinstance(coord, (CoordFlip, CoordCartesian, CoordFixed, CoordPolar, CoordGeo)):
            new._coord = coord
        else:
            raise TypeError(
                f"unsupported coord: {type(coord).__name__}; "
                "expected CoordFlip, CoordCartesian, CoordFixed, CoordPolar, or CoordGeo"
            )
        return new

    def xlim(self, lo: float, hi: float) -> "Chart":
        """Set the x-axis domain to ``[lo, hi]`` (plotnine-style).

        Equivalent to ``.coord(fm.CoordCartesian(xlim=(lo, hi)))``.  When a
        ``CoordCartesian`` is already set on this chart its ``ylim`` (and other
        parameters) are preserved; only ``xlim`` is updated.

        Parameters
        ----------
        lo : float
            Lower bound of the x-axis domain.
        hi : float
            Upper bound of the x-axis domain.

        Returns
        -------
        Chart
            New ``Chart`` with the x-axis domain constrained.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").xlim(0, 10)
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.coord import CoordCartesian
        import dataclasses

        existing = self._coord
        if isinstance(existing, CoordCartesian):
            new_coord = dataclasses.replace(existing, xlim=(lo, hi))
        else:
            new_coord = CoordCartesian(xlim=(lo, hi))
        return self.coord(new_coord)

    def ylim(self, lo: float, hi: float) -> "Chart":
        """Set the y-axis domain to ``[lo, hi]`` (plotnine-style).

        Equivalent to ``.coord(fm.CoordCartesian(ylim=(lo, hi)))``.  When a
        ``CoordCartesian`` is already set on this chart its ``xlim`` (and other
        parameters) are preserved; only ``ylim`` is updated.

        Parameters
        ----------
        lo : float
            Lower bound of the y-axis domain.
        hi : float
            Upper bound of the y-axis domain.

        Returns
        -------
        Chart
            New ``Chart`` with the y-axis domain constrained.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").ylim(0, 20)
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.coord import CoordCartesian
        import dataclasses

        existing = self._coord
        if isinstance(existing, CoordCartesian):
            new_coord = dataclasses.replace(existing, ylim=(lo, hi))
        else:
            new_coord = CoordCartesian(ylim=(lo, hi))
        return self.coord(new_coord)

    # ---- Properties ----

    def properties(
        self,
        *,
        width: Optional[Union[int, float, str]] = None,
        height: Optional[Union[int, float, str]] = None,
        title=None,
        subtitle=None,
        caption=None,
        description=None,
        render_config=None,
    ) -> "Chart":
        """Set chart-level display properties.

        Only the keyword arguments that are explicitly provided are updated;
        unset properties inherit from the existing chart.

        Parameters
        ----------
        width : int or float or "container" or None, optional
            Chart width in pixels, or ``"container"`` to fill the parent. A
            number must be finite and positive; it is stored as a ``float``.
        height : int or float or "container" or None, optional
            Chart height in pixels, under the same rule as ``width``.
        title : str or None, optional
            Chart title rendered above the plot area.
        subtitle : str or None, optional
            Chart subtitle rendered below the title.  On a faceted chart this
            renders inside the facet grid via the ``Title`` spec.  On a plain
            single-panel chart it behaves identically to
            ``.labs(subtitle=...)``.
        caption : str or None, optional
            Figure-level caption rendered below the chart.  Applied as a
            post-render chrome band so that it appears beneath the plot area
            regardless of chart type.
        description : str or None, optional
            Accessible description attached to the SVG root.
        render_config : RenderConfig or None, optional
            Rendering policy configuration. Controls auto-raster threshold
            and behavior. For one-off overrides prefer the ``raster=``
            keyword on ``.show()`` / ``.save()`` / ``.to_svg()`` instead.

        Returns
        -------
        Chart
            New ``Chart`` with the specified properties updated.

        Raises
        ------
        ValueError
            When ``width`` or ``height`` is neither ``"container"`` nor a
            finite, positive number (a ``bool`` is refused), naming
            ``Chart.properties(width=...)``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").properties(
        ...     width=400, height=300, title="My Chart"
        ... )
        Chart(mark='point', encoding=['x', 'y'])
        """
        import dataclasses

        from ferrum.title import Title as _TitleCls

        new = self._clone()
        if width is not None:
            new._width = _viewport_dimension("Chart.properties", width, param="width")
        if height is not None:
            new._height = _viewport_dimension("Chart.properties", height, param="height")
        if title is not None:
            # Schwabish SB1: accept Title value class or plain str.
            new._title = title if isinstance(title, _TitleCls) else _TitleCls(text=str(title))
        if subtitle is not None:
            # Route subtitle into the Title value class, preserving any
            # existing title text (same logic as labs(subtitle=...)).
            existing = new._title
            if isinstance(existing, _TitleCls):
                new._title = dataclasses.replace(existing, subtitle=subtitle)
            else:
                new._title = _TitleCls(text="", subtitle=subtitle)
        if caption is not None:
            new._figure_caption = caption
        if description is not None:
            new._description = description
        if render_config is not None:
            new._render_config = render_config
        return new

    def labs(self, **kwargs) -> "Chart":
        """Set human-readable labels for axes and title (plotnine-style).

        A convenience wrapper around ``properties()`` and per-channel ``title``
        kwargs.  Only the keys you provide are updated; everything else is
        inherited from the existing chart.

        Parameters
        ----------
        title : str, optional
            Chart title — delegates to ``.properties(title=...)``.
        subtitle : str, optional
            Chart subtitle — delegates to ``.properties(subtitle=...)``.
        x : str, optional
            Horizontal-axis title.  Wraps the existing ``x`` encoding channel
            in a new ``X(field, title=x, **original_kwargs)`` value, or creates
            ``X(None, title=x)`` if no ``x`` encoding is present.
        y : str, optional
            Vertical-axis title.  Analogous to ``x`` above for the ``y``
            channel.

        Returns
        -------
        Chart
            New ``Chart`` with the specified labels applied.

        Raises
        ------
        ValueError
            If any key in ``kwargs`` is not one of ``title``, ``subtitle``,
            ``x``, or ``y``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").labs(
        ...     title="My Chart", x="Horizontal", y="Vertical"
        ... )
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.encoding.positional import X, Y

        remaining = dict(kwargs)
        c = self._clone()

        if "title" in remaining:
            c = c.properties(title=remaining.pop("title"))
        if "subtitle" in remaining:
            c = c.properties(subtitle=remaining.pop("subtitle"))

        for axis_name, cls in (("x", X), ("y", Y)):
            if axis_name not in remaining:
                continue
            label = remaining.pop(axis_name)
            existing = c._encoding.get(axis_name)
            if existing is not None and hasattr(existing, "field"):
                # Reconstruct the channel preserving all kwargs except title.
                base_kwargs = {k: v for k, v in existing._kwargs.items() if k != "title"}
                c._encoding[axis_name] = cls(existing.field, title=label, **base_kwargs)
            else:
                # No existing typed channel — create a title-only placeholder.
                # The field is set to None so shorthand in _encoding is kept;
                # if the existing entry is a plain str we preserve it and wrap.
                if isinstance(existing, str):
                    parsed_field, parsed_type, _ = parse_shorthand(existing)
                    base_kwargs: dict = {}
                    if parsed_type is not None:
                        base_kwargs["type"] = parsed_type
                    c._encoding[axis_name] = cls(
                        parsed_field or existing, title=label, **base_kwargs
                    )
                else:
                    c._encoding[axis_name] = cls(None, title=label)

        if remaining:
            raise ValueError(f"labs(): unknown label keys: {sorted(remaining.keys())}")
        return c

    # ---- Spec output ----

    def to_spec(self, *, _override_payload=None):
        """Build the Rust ``ChartSpec`` for this chart.

        Resolves any pending statistical-mark desugar, converts Python encoding
        channel objects to ``EncodingSpec`` instances, and constructs the
        ``ChartSpec`` PyO3 object that the Rust renderer consumes.

        When the chart carries ``Chart.override`` kwargs, the spec-targeted pieces
        of the override payload (encoding scales, mark style, coord) are applied
        last so override wins the cascade (spec §7).  ``_render_inputs`` builds the
        payload once per render and threads it via ``_override_payload``; a direct
        ``.to_spec()`` call builds it on demand from ``self._overrides`` so the
        standalone spec also reflects overrides.  The chart-config, property, and
        deprecation pieces are applied by the render path, not here.

        Returns
        -------
        ChartSpec
            The fully-resolved ``ferrum._core.ChartSpec`` for this chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> spec = fm.Chart(df).mark_point().encode(x="x", y="y").to_spec()
        >>> spec.mark
        'point'
        """
        from ferrum import ChartSpec

        # Resolve any pending statistical mark desugar (mark called before encode).
        # Every polar binding draws as x/y from here on, so each statistic
        # below is lowered over the channels the chart actually draws.
        resolved = self._with_polars_data()._resolve_pending()._with_polar_channels_lowered()

        # Chart.override spec-piece payload (encoding scales / mark style / coord).
        # Built on demand for standalone .to_spec() callers; the render path passes
        # the once-per-render payload via _override_payload.  None when no override.
        if _override_payload is None and self._overrides:
            from ferrum._override_apply import build_payload

            _override_payload = build_payload(self._overrides)

        # --- Channel aliasing (operates on a shallow copy to avoid mutating self) ---
        enc = dict(resolved._encoding)  # shallow copy — safe for alias remapping
        chart_detail = None
        if resolved._layers is None:
            mk = dict(resolved._mark_kwargs) if resolved._mark_kwargs else {}
        else:
            # `#119`: a layered chart's style is its own (spec §4.8). Its
            # `_mark_kwargs` is layer 0's, copied up by the clone that built
            # it, so it never reaches the wire. A chart-level `detail` is an
            # encoding every layer binding none inherits, like `color`; it is
            # mark style on the wire, so it is handed to the layers
            # (`_build_layers_list`) rather than routed into a chart-level
            # `mark_style`.
            mk = {}
            chart_detail = enc.pop("detail", None)
        color_source = alias_source(enc, "color")
        enc, mk = _apply_channel_aliases(enc, mk, resolved._mark)

        # --- The chart's own encoding statistic (single-chart) ---
        # Runs BEFORE the encoding specs are built: the statistic's one step
        # (an Aggregate, or a Bin that replaces the batch schema with its own
        # columns or appends its bin columns per row) decides which column
        # each channel draws.
        # A layered chart's encoding statistics are per layer (resolved below,
        # each inside the chain of the rows its layer reads), never a
        # chart-level step.
        effective_transforms = list(resolved._transforms) if resolved._transforms else []
        if resolved._layers is None:
            effective_transforms, enc = self._resolve_statistic(resolved, effective_transforms, enc)

        # Mark style: apply mark_* override last so it beats the chart's mark style.
        # A layered chart (multiple marks) has no single primary mark — apply_mark_style
        # raises FerrumOverrideError on a mark_* override there (spec §11 Q4).
        if _override_payload is not None and _override_payload.mark_style:
            from ferrum._override_consume import apply_mark_style

            mk = apply_mark_style(
                mk, _override_payload, mark=None if resolved._layers is not None else resolved._mark
            )
        # A single bar chart's `zero` is consumed here, whichever door wrote it
        # (a figure's `mark=`, a composite's catch-all, `.override(mark_zero=)`),
        # exactly as `mark_bar(zero=)` is: it decides the y zero-anchor below
        # and never reaches the wire, where only a layered bar reads it. Every
        # door stored it as a bool, or ``None`` for unset.
        mark_zero = resolved._mark_zero
        if resolved._mark == "bar":
            zero = mk.pop("zero", None)
            if zero is not None:
                mark_zero = zero

        # --- Build EncodingSpec entries for each honored channel ---
        kw: dict = {"mark": resolved._mark or "point", "data": "default"}
        kw.update(
            self._build_encoding_specs(
                resolved, enc, _override_payload, mark_zero=mark_zero, color_source=color_source
            )
        )

        # --- Lower the operands (layered charts) ---
        # Each layer that aggregates or bins gets its own named Aggregate or
        # Bin whose output the layer reads via ``data_source`` (per-layer
        # transforms are never executed standalone by the renderer), so two
        # layers compute their statistics independently, each in its
        # operand's chain (``ferrum._compose.lower_operands``). ``resolved``
        # may be ``self`` (no pending desugar), so the resolved layers are
        # threaded through ``_build_layers_list`` rather than mutated onto
        # the chart.
        serialized_layers = resolved._layers
        if resolved._layers:
            # A layer attached to this chart reads its operand's rows; one that
            # reads them through a named output takes each x/y/x2/y2 it does
            # not bind from the bindings the caller wrote for this chart.
            layers = attach_to_rows(resolved._layers, resolved)
            # Each layer draws its effective encoding over this chart's, and
            # computes the statistics it binds or inherits whole, grouped by
            # that encoding, inside the chain of the operand it belongs to.
            layers = _fill_chart_positional(
                _layers_over_parent(layers, resolved._encoding, resolved._coord), resolved
            )
            refuse_missing_statistic_inputs(layers, resolved)
            layers, layer_statistics = _resolve_layer_statistics(layers, resolved)
            serialized_layers, effective_transforms = lower_operands(
                resolved, layers, layer_statistics
            )

        # --- Transform serialization ---
        if effective_transforms:
            # If any transform is a _NamedTransform, a _Branch or a plain dict
            # (Phase 12 data transforms), serialize everything via JSON so Rust
            # receives the full pipeline through the serde path.
            has_named = any(
                isinstance(t, (_NamedTransform, _Branch, _StatisticsMarker, _ComposedStep))
                for t in effective_transforms
            )
            has_dict = any(isinstance(t, dict) for t in effective_transforms)
            if has_named or has_dict:
                xform_json = _transforms_to_json_list_named(effective_transforms)
                kw["transforms_json"] = json.dumps(xform_json)
            else:
                kw["transforms"] = effective_transforms

        # --- Remaining chart-level properties ---
        if resolved._facet is not None:
            kw["facet"] = resolved._build_facet_dict()
        # Coord: serialize the chart's coord, then apply any coord_* override last
        # (override wins; reconstructs the coord dataclass with the override leaves).
        if _override_payload is not None and _override_payload.coord:
            from ferrum._override_consume import apply_coord

            coord_spec = apply_coord(resolved._coord, _override_payload)
            if coord_spec is not None:
                kw["coord"] = coord_spec
        elif resolved._coord is not None:
            c = resolved._coord
            # Back-compat: orient_coord_flip sets _coord = "flip" (a string).
            # New coord objects expose to_spec_dict(); CoordFlip returns "flip".
            kw["coord"] = c.to_spec_dict() if hasattr(c, "to_spec_dict") else c
        if mk:
            kw["mark_style"] = mk
        if resolved._layers is not None:
            kw["layers"] = resolved._build_layers_list(
                serialized_layers,
                inherited_details=layered_chart_detail(
                    chart_detail, [layer.mark for layer in serialized_layers]
                ),
                override=_override_payload,
            )
        _warn_ignored_details(resolved, chart_detail, serialized_layers, kw, effective_transforms)
        if resolved._position is not None:
            kw["position"] = resolved._position.to_spec_dict()
        if resolved._title is not None:
            # Schwabish SB1: Title.to_spec_dict() emits the JSON shape that
            # Rust's ChartSpec accepts (subtitle, anchor, offset, font sizes).
            kw["title"] = resolved._title.to_spec_dict()
        # --- Per-channel axis=None suppression (D8) ---
        # X("a:Q", axis=None) / Y("b:Q", axis=None) routes into the same
        # axis_x / axis_y suppression machinery as Chart.axis(x=False).
        # Precedence: Chart.axis() (resolved._axis_x/_axis_y, set below) wins
        # over per-channel axis=None when both are present.
        _ch_x = enc.get("x")
        if (
            isinstance(_ch_x, ChannelBase)
            and "axis" in _ch_x._kwargs
            and _ch_x.option("axis") is None
            and resolved._axis_x is None
        ):
            kw["axis_x"] = False
        _ch_y = enc.get("y")
        if (
            isinstance(_ch_y, ChannelBase)
            and "axis" in _ch_y._kwargs
            and _ch_y.option("axis") is None
            and resolved._axis_y is None
        ):
            kw["axis_y"] = False
        # Chart-level axis() always wins — overwrite whatever per-channel set.
        if resolved._axis_x is not None:
            kw["axis_x"] = resolved._axis_x
        if resolved._axis_y is not None:
            kw["axis_y"] = resolved._axis_y
        if resolved._description:
            kw["chart_description"] = resolved._description
        # The chart-level entry names a flat chart's one mark. A layered
        # chart's entries ride on its layers (`_expand_layers` moved the
        # flat chart's onto the layer it built), so a chart-level one there
        # would only be a stale fallback for layers another operand brought.
        if resolved._entry is not None and not resolved._layers:
            kw["entry"] = resolved._entry

        # --- Reactive-parameter (D6) params section ---
        # Unified declaration: registered selections (auto-promoted),
        # explicit fm.param() variables, and any Parameter referenced as a
        # scale domain. Deduped by name (first-seen wins); omitted entirely
        # when empty so param-free specs stay byte-identical to before.
        # Collected here (ahead of conditional serialization below) because
        # the same name-keyed registry resolves a Parameter.ref() at a
        # conditional value site (F-L13-01) -- the chart's registered
        # params must be available before conditionals are serialized.
        params_list = self._collect_params(resolved, enc)
        param_registry = {p.name: p for p in params_list}

        # --- Conditional / selection injection ---
        if resolved._selections:
            kw["selections"] = json.dumps([s.to_spec_dict() for s in resolved._selections])
        if resolved._conditionals:
            conditionals = resolved._conditionals
            scopes = (
                _conditional_layer_scopes(conditionals, resolved._layers)
                if resolved._layers
                else [None] * len(conditionals)
            )
            wire_conditionals = []
            for c, scope in zip(conditionals, scopes):
                wire = c.to_spec_dict(params=param_registry)
                # The layers this conditional belongs to: it restyles only
                # the marks of the layers listed, on every channel.
                # Absent on an unlayered chart, meaning its one layer.
                if scope is not None:
                    wire["layers"] = scope
                wire_conditionals.append(wire)
            kw["conditionals"] = json.dumps(wire_conditionals)

        if params_list:
            self._validate_params_finite(params_list)
            kw["params"] = json.dumps([p.to_param_spec_dict() for p in params_list])

        return ChartSpec(**kw)

    def _build_spec(self):
        """Build the chart spec for callers that want typed Python access to
        layers (composite-mark tests, future internal renderer wiring).

        For single-layer charts this is a thin alias for ``to_spec``. For
        layered charts it returns a Python-side ``_SpecView`` that wraps the
        underlying ``ChartSpec`` and exposes ``.layers`` as a list of
        ``types.SimpleNamespace`` items with ``.mark``, ``.encoding``,
        ``.mark_kwargs``, and ``.data_source`` attributes — matching the
        Layer-instance contract from spec §12.1 without requiring a parallel
        ``PyLayer`` Rust class. JSON / serialization remains the underlying
        ``ChartSpec`` (delegated via ``__getattr__``).
        """
        resolved = self._with_polars_data()._resolve_pending()
        spec = resolved.to_spec()
        if resolved._layers is None:
            return spec
        return _SpecView(spec, resolved._layers)

    def to_json(self, *, indent=None) -> str:
        """Serialise the chart specification to a JSON string.

        .. note::
            Unlike [to_svg][ferrum.Chart.to_svg], [to_png][ferrum.Chart.to_png], and [to_html][ferrum.Chart.to_html] — which
            return *rendered* output — ``to_json`` returns the chart
            **specification** (the declaration you built), not the rendered
            scene graph.  ``save(path)`` with a ``.json`` extension writes the
            rendered scene graph instead, which is a different artifact.  For
            the specification as a Python value, see [to_dict][ferrum.Chart.to_dict].

        Calls ``to_spec()`` to build the ``ChartSpec`` and then serialises it
        via the Rust ``serde_json`` encoder.  When ``indent`` is given the
        compact JSON is reformatted via ``json.loads`` / ``json.dumps`` on the
        Python side (the Rust encoder always produces compact output).

        Parameters
        ----------
        indent : int or None, optional
            Number of spaces to use for pretty-printing.  ``None`` (default)
            returns compact single-line JSON.

        Returns
        -------
        str
            JSON representation of the chart specification.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1], "y": [2]})
        >>> spec_json = fm.Chart(df).mark_point().encode(x="x", y="y").to_json()
        >>> '"mark"' in spec_json
        True
        """
        spec = self.to_spec()
        compact = spec.to_json()
        if indent is None:
            return compact
        return json.dumps(json.loads(compact), indent=indent)

    def to_dict(self) -> dict:
        """Serialise the chart specification to a plain Python dict (Altair-style).

        Equivalent to ``json.loads(self.to_json())``.  Useful for introspection,
        testing, and interoperability with tools that consume chart-spec dicts.

        Returns
        -------
        dict
            The chart specification as a nested Python dictionary.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1], "y": [2]})
        >>> d = fm.Chart(df).mark_point().encode(x="x", y="y").to_dict()
        >>> d["mark"]
        'point'
        """
        return json.loads(self.to_json())

    # pipe() is inherited from _ChartOperatorMixin, shared with every
    # composition view (#245(5)).

    # ---- Selections / interactivity (spec §3.10) ----

    def add_selection(self, *selections) -> "Chart":
        """Attach interactive selection(s) to this chart.

        Per ``ferrum-spec.md §3.10``, the SVG/PNG renderer silently
        ignores selections — they are intended for the WASM renderer (Phase 11).
        This method accepts any number of selection objects and returns a new
        ``Chart`` unchanged so that user code building selection-aware charts
        remains forward-compatible without raising under SVG/PNG rendering.

        Parameters
        ----------
        *selections
            Any number of selection objects (currently ignored).

        Returns
        -------
        Chart
            New ``Chart`` (clone), with the selections recorded but not
            rendered.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y").add_selection()
        """
        from ferrum.selection import Selection

        for sel in selections:
            if not isinstance(sel, Selection):
                raise TypeError(
                    f"add_selection() expects Selection instances (from "
                    f"fm.selection_point(), fm.selection_interval(), etc.); "
                    f"got {type(sel).__name__!r}. Did you mean add_params()?"
                )
        new = self._clone()
        for sel in selections:
            _append_unique_by_name(new._selections, sel, entry="add_selection()")
        return new

    def add_params(self, *params) -> "Chart":
        """Attach reactive variable parameter(s) to this chart.

        Records one or more [Parameter][ferrum.Parameter] objects (built
        via ``fm.param()``) so they are emitted into the spec's ``params``
        section.  Parameters drive reactive scale domains (``scale={"domain":
        param}``) and crossfilters (``transform_filter(param)``) in the WASM
        runtime; at static render time their initial ``value`` is used.

        Selections referenced as scale domains or registered via
        ``add_selection`` are auto-promoted into the params section at
        serialization, so they do not need to be passed here.  Duplicate names
        are de-duplicated (first-seen wins) when the spec is built.

        Parameters
        ----------
        *params : Parameter
            Any number of parameter objects (typically ``fm.param(...)``).

        Returns
        -------
        Chart
            New ``Chart`` (clone) with the parameters recorded.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> k = fm.param("k", value=3)
        >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y").add_params(k)
        """
        from ferrum.parameter import Parameter
        from ferrum.selection import Selection

        for p in params:
            if not isinstance(p, Parameter):
                raise TypeError(
                    f"add_params() expects Parameter instances (from fm.param() "
                    f"or selection constructors); got {type(p).__name__!r}."
                )
        new = self._clone()
        new._params.extend(params)
        # Selections passed via add_params must also land in _selections so that
        # the WASM runtime (toggle_legend, crossfilter) can find the SelectionSpec.
        # Dedup by name to avoid double-registration when the same Selection is
        # later also passed to add_selection().
        for p in params:
            if isinstance(p, Selection):
                _append_unique_by_name(new._selections, p, entry="add_params()")
        return new

    def interactive(self, *, toolbar: bool = True) -> "InteractiveChart":
        """Return an interactive rendering of this chart.

        Wraps the chart in an ``InteractiveChart`` widget backed by the WASM
        renderer, enabling selections, pan/zoom, and conditional encodings in
        Jupyter and HTML exports.

        Parameters
        ----------
        toolbar : bool, default True
            Whether to show the interactive toolbar (zoom/pan controls, export
            button). Set to ``False`` to render without the toolbar.

        Returns
        -------
        InteractiveChart
            An interactive widget/container backed by the WASM renderer.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y").interactive()
        """
        from ferrum._interactive import InteractiveChart

        return InteractiveChart(self, toolbar=toolbar)

    def conditional(self, spec: Any) -> "Chart":
        """Apply a conditional encoding to this chart.

        Convenience sugar: ``chart.conditional(sel.when(Color("x")).otherwise(value("#ccc")))``
        is equivalent to::

            chart.add_selection(sel).encode(color=sel.when(Color("x")).otherwise(value("#ccc")))

        Parameters
        ----------
        spec : ConditionalSpec
            A ``ConditionalSpec`` produced by ``sel.when(...).otherwise(...)``.

        Returns
        -------
        Chart
            New ``Chart`` with the conditional recorded.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> from ferrum.selection import selection_point, value
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "z": ["a", "b"]})
        >>> sel = selection_point(fields=["z"])
        >>> chart = (
        ...     fm.Chart(df)
        ...     .mark_point()
        ...     .encode(x="x", y="y")
        ...     .conditional(sel.when(fm.Color("z")).otherwise(value("#ccc")))
        ... )
        """
        new = self._clone()
        # A spec built by sel.when(...) carries channel=None (the channel is
        # inferred from the branch value at serialization). A spec that DOES
        # name a channel goes through the same vocabulary encode() uses, so
        # this entry point cannot smuggle an unsupported channel past the
        # typed refusal and into serde.
        if isinstance(spec, ConditionalSpec):
            # Resolve (and vocabulary-check) the channel once, here, and stamp
            # it, so serialization uses the channel this declaration was
            # checked under: a stamped spec's own, or the one its branches name.
            wire = conditional_wire_channel(_wire_channel_of_conditional(spec))
            new_spec = replace(spec, channel=wire)
            # Same one-conditional-per-wire-channel rule encode() enforces.
            earlier = _conditional_on_wire(new, wire)
            if earlier is not None:
                raise ValueError(
                    _conditional_collision_message(
                        context="Chart.conditional()",
                        wire=wire,
                        first=_conditional_declaration_label(earlier, whose="the chart's existing"),
                        second=_conditional_declaration_label(new_spec, whose="this call's"),
                    )
                )
            spec = new_spec
        new._conditionals.append(spec)
        if hasattr(spec, "selection_name"):
            # Ensure the selection is also registered so scene_build can wire it.
            # Auto-register the carried selection when the spec knows it,
            # so the explicit path benefits like encode(<channel>=cond) does.
            carried = getattr(spec, "selection", None)
            if carried is not None and carried.name == spec.selection_name:
                _append_unique_by_name(new._selections, carried, entry="Chart.conditional()")
            else:
                existing = {s.name for s in new._selections if hasattr(s, "name")}
                if spec.selection_name not in existing:
                    raise ValueError(
                        f"Chart.conditional(): no selection named {spec.selection_name!r} "
                        f"is attached to this chart. Call .add_selection(sel) first, or use "
                        f"chart.add_selection(sel).encode(...) with the conditional encoding."
                    )
        return new

    def __repr__(self) -> str:
        """Return a concise string representation of the chart.

        Returns
        -------
        str
            A string of the form ``Chart(mark='point', encoding=['x', 'y'])``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> repr(fm.Chart(df).mark_point().encode(x="x", y="y"))
        "Chart(mark='point', encoding=['x', 'y'])"
        """
        return f"Chart(mark={self._mark!r}, encoding={list(self._encoding.keys())})"
