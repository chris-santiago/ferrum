"""Composite-mark desugar for the :class:`~ferrum.chart.Chart` value class.

This module owns the composite-mark expansion helpers and the body of
``Chart._resolve_pending`` (cohesion finding CHART-02 / CHART-07).  A
composite/diagnostic ``mark_*()`` called before ``.encode()`` stores a
``_PendingMark`` sentinel; ``_resolve_pending_impl`` is invoked at the start of
every render / spec-build path to apply ``desugar_fn`` against the now-populated
encoding dict.

``Chart._resolve_pending`` is a one-line delegator to
``_resolve_pending_impl(self)`` so external callers
(``chart._resolve_pending()``) keep working unchanged.

The historical per-``kind`` ``if`` chain is replaced by a dispatch table
(``_PENDING_PREP``) keyed by mark kind.  ``_resolve_pending_impl`` is a uniform
orchestrator: resolve x/y fields and sorts → split style kwargs → run the
kind's declared kwarg-injection prep steps → call ``desugar_fn`` → run the
kind's declared post-desugar Float64-cast steps → route layered-vs-single
result.  The dispatch table produces the *exact same* transform-kwarg
injections, sort keys, color/groupby inference, name forcing, and dtype casts,
in the same order, as the old chain — the change is structural, not behavioural.

These are free functions; ``chart.py`` imports them, so this module must never
import ``Chart`` at module level (that would create an import cycle).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from ferrum._layer import _PRIMITIVE_MARKS, _Layer
from ferrum.encoding.base import ChannelBase
from ferrum.encoding._scale import _scale_to_dict
from ferrum.marks.base import MarkBase
from ferrum.marks.statistical import _build_prior_layer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ferrum.chart import Chart


def _resolve_sort_for_composite(sort: Any, x_field: str | None, y_field: str | None) -> Any:
    """Resolve a shorthand sort string to an explicit dict for composite marks.

    Composite desugars expand a single chart-level encoding into many layers
    with different y columns (e.g., ``lower_whisker``, ``q1``), so the Rust
    renderer would misidentify the sort value field when a shorthand like
    ``"-y"`` is forwarded verbatim.  This function converts shorthands to an
    explicit ``{"field": ..., "op": "sum", "order": ...}`` dict using the
    original user field names, so the sort resolves correctly regardless of
    which layer's y column the renderer inspects first.

    Non-shorthand sort values (explicit arrays, explicit dicts) are returned
    unchanged.
    """
    if not isinstance(sort, str):
        return sort
    _SHORTHANDS = {
        "-y": (y_field, "sum", "descending"),
        "y": (y_field, "sum", "ascending"),
        "-x": (x_field, "sum", "descending"),
        "x": (x_field, "sum", "ascending"),
    }
    if sort in _SHORTHANDS:
        field_name, op, order = _SHORTHANDS[sort]
        if field_name is None:
            return sort  # cannot resolve without a field name; leave as-is
        return {"field": field_name, "op": op, "order": order}
    return sort


def _resolve_cat_axis(
    transform_kwargs: dict,
    x_field: str | None,
    y_field: str | None,
    x_sort: Any,
    y_sort: Any,
) -> tuple[str | None, Any, str | None]:
    """Return ``(cat_field, cat_sort, val_field)`` for a composite mark.

    Composite marks that build a categorical positional axis declare their
    orientation with one of two historically divergent conventions:

    - ``horizontal: bool`` (boxplot, boxen) — ``True`` puts the categorical
      axis on **y** and the value axis on **x**.
    - ``orient: "vertical" | "horizontal"`` (swarm) — ``"horizontal"`` puts the
      value axis on **x**, hence the categorical axis on **y**.

    Both reduce to a single question: *is the categorical axis y?*  This helper
    normalizes them into one boolean and returns the categorical field, the
    sort that applies to that categorical axis, and the value field.

    Precedence: the two keys are set by disjoint marks today (no mark passes
    both), so they never conflict in practice.  Should both ever be present,
    ``horizontal=True`` and ``orient="horizontal"`` agree (cat axis is y); a
    truthy ``horizontal`` therefore wins, and ``orient`` is consulted only when
    ``horizontal`` is falsy.  This preserves each mark's existing result.
    """
    horizontal = bool(transform_kwargs.get("horizontal", False))
    orient = transform_kwargs.get("orient", "vertical")
    cat_is_y = horizontal or orient == "horizontal"
    if cat_is_y:
        return y_field, y_sort, x_field
    return x_field, x_sort, y_field


def _split_style_kwargs(desugar_fn, user_kwargs: dict) -> tuple[dict, dict]:
    """Partition a composite mark's user kwargs into transform vs. style keys.

    A user kwarg whose name is a declared parameter of *desugar_fn* (positional-
    or-keyword / keyword-only) is a transform kwarg; every other key is a
    constant mark-style override (e.g. ``opacity``, ``stroke_width``).  This is
    per-mark and resolves name collisions automatically: ``fill`` is a real
    ``desugar_density`` parameter, so it stays on the transform side, while
    ``opacity`` matches no parameter and routes to style.

    ``inspect.signature`` follows ``__wrapped__``, so the thin ``_resolve_*``
    wrappers (which use ``**kwargs``) still expose the underlying desugar
    signature.  ``x_field``/``y_field`` are passed positionally and
    ``chart_encoding`` is injected separately, so they never appear in
    *user_kwargs*.

    Returns
    -------
    tuple[dict, dict]
        ``(transform_kwargs, style_kwargs)``.
    """
    params = inspect.signature(desugar_fn).parameters
    transform_names = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    transform_kwargs: dict = {}
    style_kwargs: dict = {}
    for k, v in user_kwargs.items():
        if k in transform_names:
            transform_kwargs[k] = v
        else:
            style_kwargs[k] = v
    return transform_kwargs, style_kwargs


def _apply_style_to_layer(layer: _Layer, style_dict: dict) -> _Layer:
    """Merge constant *style_dict* into *layer*'s ``mark_kwargs``.

    User style values override any desugar-set per-layer defaults.  When
    *style_dict* is empty the layer is returned unchanged so output stays
    byte-identical to pre-feature renders.
    """
    if not style_dict:
        return layer
    merged = {**(layer.mark_kwargs or {}), **style_dict}
    return replace(layer, mark_kwargs=merged)


def _apply_remap(encoding: dict, remap: dict, orig_encoding: dict | None = None) -> None:
    """Apply a desugar remap to an encoding dict, preserving axis titles.

    Modifies *encoding* in place.  When *orig_encoding* is provided and
    a remap changes the field name (e.g. ``"tip"`` -> ``"bin_start"``),
    the original field name is injected as the ``title`` kwarg so axes
    keep a human-readable label.

    Parameters
    ----------
    encoding : dict
        Encoding dict to modify (e.g. ``chart._encoding``).
    remap : dict
        Mapping from channel name to new field name (``result.remap``).
    orig_encoding : dict or None, optional
        The chart's encoding dict *before* the remap, used to recover
        original field names for title preservation.  ``None`` disables
        title preservation.
    """
    from ferrum.encoding import X, X2, Y, Y2

    def _orig_field(channel: str) -> str | None:
        if orig_encoding is None:
            return None
        enc = orig_encoding.get(channel)
        if enc is None:
            return None
        return enc.field if isinstance(enc, ChannelBase) else enc

    if "x" in remap:
        x_field = _orig_field("x")
        title = x_field if (x_field and remap["x"] != x_field) else None
        encoding["x"] = X(remap["x"], type="Q", title=title) if title else X(remap["x"], type="Q")
    if "y" in remap:
        y_field = _orig_field("y")
        title = y_field if (y_field and remap["y"] != y_field) else None
        encoding["y"] = Y(remap["y"], type="Q", title=title) if title else Y(remap["y"], type="Q")
    if "x2" in remap:
        encoding["x2"] = X2(remap["x2"], type="Q")
    if "y2" in remap:
        encoding["y2"] = Y2(remap["y2"], type="Q")


def _explicit_positional_scale(chart_channel: Any) -> Any:
    """Return the raw ``scale=`` value from a chart-level positional channel.

    ``chart_channel`` is a chart-level ``_encoding["x"]`` / ``_encoding["y"]``
    value. After ``Chart.encode()`` this is always a ``ChannelBase`` instance
    (bare strings passed to ``encode()`` are converted immediately), so a
    non-``ChannelBase`` value never carries a scale.
    """
    if isinstance(chart_channel, ChannelBase):
        return chart_channel.option("scale")
    return None


def _scale_domain(scale: Any) -> Any:
    """Return the ``domain`` carried by a raw scale value, or ``None``.

    *scale* is either a plain dict (e.g. ``{"type": "log"}``, as composite
    desugars author today) or a ferrum ``*Scale`` pyclass instance (e.g.
    ``fm.LinearScale(domain=(0, 1))``, as users pass via ``scale=``).
    """
    if scale is None:
        return None
    if isinstance(scale, dict):
        return scale.get("domain")
    return getattr(scale, "domain", None)


def _merge_positional_channel_scale(layer_value: Any, chart_scale: Any, axis: str) -> Any:
    """Return a replacement value for one layer channel with *chart_scale*
    propagated in, or ``None`` when *layer_value* should be left untouched.

    Implements the per-channel half of the scale-through-desugar propagation
    rule (design spec §6, GH #45 prerequisite):

    - no scale on the layer channel -> attach the chart-level scale wholesale;
    - scale without a domain (e.g. validation_curve's ``{"type": "log"}``) ->
      merge in the chart-level domain only, preserving the layer's own
      ``type``/``range``/other keys;
    - scale with a domain already (e.g. SHAP ``x_scale_domain``) -> leave
      untouched, the mark-computed domain wins.
    """
    if isinstance(layer_value, str):
        from ferrum.encoding import _channel_class_for

        channel_cls = _channel_class_for(axis)
        return channel_cls(layer_value, scale=chart_scale)

    if not isinstance(layer_value, ChannelBase):
        return None

    layer_scale = layer_value.option("scale")
    if layer_scale is None:
        new_kwargs = dict(layer_value._kwargs)
        new_kwargs["scale"] = chart_scale
        return type(layer_value)(layer_value.field, **new_kwargs)

    if _scale_domain(layer_scale) is not None:
        return None  # mark-computed domain wins

    chart_domain = _scale_domain(chart_scale)
    if chart_domain is None:
        return None  # nothing to merge

    merged_scale = {**_scale_to_dict(layer_scale), "domain": chart_domain}
    new_kwargs = dict(layer_value._kwargs)
    new_kwargs["scale"] = merged_scale
    return type(layer_value)(layer_value.field, **new_kwargs)


def _propagate_positional_scale(chart_encoding: dict, layer_encoding: dict) -> dict:
    """Merge chart-level explicit x/y scales into *layer_encoding*.

    Composite-mark desugars rebuild each layer's encoding from scratch,
    carrying only field names/titles (see module docstring), which silently
    drops any explicit ``scale=`` the user set on the chart-level positional
    channel (GH #45 prerequisite; design spec §6). x2/y2 companions share
    their axis's scale through the layer's primary x/y channel and never
    carry their own scale (``SECONDARY_EXTENT`` honors no ``scale`` kwarg,
    and the Rust axis-scale resolver only ever consults the primary channel's
    ``EncodingSpec.scale``), so only ``x``/``y`` are propagation targets.

    Returns *layer_encoding* unchanged (same object) when no channel
    qualifies, so callers can skip rebuilding the owning ``_Layer``.
    """
    updates: dict = {}
    for axis in ("x", "y"):
        if axis not in layer_encoding:
            continue
        chart_scale = _explicit_positional_scale(chart_encoding.get(axis))
        if chart_scale is None:
            continue
        merged = _merge_positional_channel_scale(layer_encoding[axis], chart_scale, axis)
        if merged is not None:
            updates[axis] = merged
    if not updates:
        return layer_encoding
    return {**layer_encoding, **updates}


def _propagate_layer_scale(chart_encoding: dict, layer: _Layer) -> _Layer:
    """Return *layer* with the chart-level x/y scale propagated onto its encoding."""
    new_encoding = _propagate_positional_scale(chart_encoding, layer.encoding)
    if new_encoding is layer.encoding:
        return layer
    return replace(layer, encoding=new_encoding)


def cast_columns_to_float64(data, fields, *, int_only: bool = True) -> Any:
    """Return *data* with columns in *fields* cast to Float64.

    Several Rust composite transforms (swarm, box_stats, letter_value, violin)
    require their value column to be ``Float64``.  This helper performs the cast
    in one place where the encoding (value field name) and the polars data are
    both available.  *data* is returned unchanged when polars is unavailable,
    *data* is not a polars DataFrame, or no listed field qualifies for casting.

    Parameters
    ----------
    data:
        The polars DataFrame to cast.
    fields:
        Iterable of column names to consider for casting.
    int_only:
        When ``True`` (default), cast only integer-dtype columns
        (``Int8``/``Int16``/…/``UInt64``).  This is the predicate used by
        boxplot, boxen, and violin.
        When ``False``, cast any column whose dtype is not already ``Float64``
        — including ``Float32`` and numeric-string columns.  This is the
        predicate required by the swarm transform, which the Rust layer rejects
        unless the value column is exactly ``Float64``.
    """
    try:
        import polars as pl
    except ImportError:
        return data
    if data is None or not isinstance(data, pl.DataFrame):
        return data
    _INT_DTYPES = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
    try:
        if int_only:
            casts = [
                pl.col(fld).cast(pl.Float64)
                for fld in fields
                if fld and fld in data.columns and data[fld].dtype in _INT_DTYPES
            ]
        else:
            casts = [
                pl.col(fld).cast(pl.Float64)
                for fld in fields
                if fld and fld in data.columns and data[fld].dtype != pl.Float64
            ]
        if casts:
            return data.with_columns(casts)
    except (TypeError, AttributeError, ValueError):
        return data
    return data


@dataclass
class _ResolveContext:
    """Mutable working state threaded through the per-kind preprocessing steps.

    A single context object lets the dispatch-table prep functions read and
    write the same fields the old per-``kind`` ``if`` chain mutated inline
    (``transform_kwargs``, ``x_sort``/``y_sort``, and the post-desugar cast
    field list), so the orchestration stays a flat, readable sequence.
    """

    chart: "Chart"
    kind: str
    x_field: str | None
    y_field: str | None
    transform_kwargs: dict
    x_sort: Any
    y_sort: Any
    # Fields to Float64-cast on the cloned chart's data after desugar runs.
    cast_fields: list[str] = field(default_factory=list)
    # When False, cast any non-Float64 column (swarm semantics).
    # When True (default), cast only integer-dtype columns (box/violin semantics).
    cast_int_only: bool = True

    def _color_field(self) -> str | None:
        color_enc = self.chart._encoding.get("color")
        if color_enc is None:
            return None
        return color_enc.field if isinstance(color_enc, ChannelBase) else color_enc


# --- Per-kind preprocessing steps (mutate the _ResolveContext in place) ------
#
# Each step mirrors one block of the original `_resolve_pending` body exactly.
# The dispatch table below assigns each kind its ordered list of steps so the
# net effect (injected kwargs, sort nulling, cast fields) is byte-identical.


def _prep_ribbon_y2(ctx: _ResolveContext) -> None:
    """Ribbon needs y2 from the encoding — inject as a transform kwarg."""
    y2_enc = ctx.chart._encoding.get("y2")
    if y2_enc is not None:
        y2_field = y2_enc.field if isinstance(y2_enc, ChannelBase) else y2_enc
        ctx.transform_kwargs = {**ctx.transform_kwargs, "y2_field": y2_field}


def _prep_boxen_agg_sort(ctx: _ResolveContext) -> None:
    """Route an aggregate categorical sort into the LetterValue transform.

    The LetterValue transform renames the groupby column to "group" and drops
    the original value column from its per-depth output batches.  A field-based
    aggregate sort dict therefore cannot resolve through the standard layer-sort
    path — the value column is absent from the layer batch.  Instead, hand the
    aggregate op/order to the transform via its ``sort=`` kwarg and drop the
    categorical layer sort for the aggregate case.  Non-aggregate forms still
    resolve via the standard layer-sort path on the renamed ``group`` column.
    """
    cat_is_y = bool(ctx.transform_kwargs.get("horizontal", False))
    cat_sort = ctx.y_sort if cat_is_y else ctx.x_sort
    if isinstance(cat_sort, dict) and "field" in cat_sort:
        ctx.transform_kwargs = {
            **ctx.transform_kwargs,
            "boxen_agg_sort": {
                "op": cat_sort.get("op", "sum"),
                "order": cat_sort.get("order", "ascending"),
            },
        }
        # Drop the categorical layer sort for the aggregate case; the
        # transform's row order drives the domain.
        if cat_is_y:
            ctx.y_sort = None
        else:
            ctx.x_sort = None


def _prep_boxen_palette_color_conflict(ctx: _ResolveContext) -> None:
    """Reject ``mark_boxen(palette=...)`` combined with a chart-level
    ``.encode(color=...)`` channel.

    ``palette`` colors each depth band via a per-layer ``fill=`` style
    kwarg (design spec §4.4, #91). A chart-level color *encoding* always
    overrides a layer's ``fill=`` -- the same precedence a plain ``fill=``
    mark kwarg has under any encoded channel, true for every mark, not
    boxen-specific. Under ``palette=None`` that precedence is harmless (the
    ramp still shows through as varying opacity); under ``palette=...``
    every band would render the *same* encoded color at the palette path's
    fixed opacity=1.0, silently discarding the palette and collapsing the
    letter-value nesting into one flat block per group. This runs at
    ``_resolve_pending_impl`` time, after ``.encode()`` is fully known
    regardless of call order (``mark_boxen()`` before or after
    ``.encode(color=...)``), so it catches both.

    ``color_field=`` (boxen's own per-group grouping kwarg, resolved
    entirely inside the transform) is unaffected -- this only inspects the
    chart-level ``color`` *encoding* channel, a different mechanism, and
    ``color_field=`` grants no hue: it only changes which column the
    ``LetterValue`` transform groups by, so every band still renders in
    one theme color at ramp opacities (or the flat palette fill). boxen
    has no hue channel at all today (violin-hue is the tracked frontier
    work, out of scope here) -- the error below must not imply one exists.
    """
    if ctx.transform_kwargs.get("palette") is None:
        return
    if ctx._color_field() is not None:
        raise ValueError(
            "mark_boxen(palette=...) cannot be combined with a chart-level "
            "color encoding (.encode(color=...)): the color channel "
            "overrides every band's fill, so the palette would have no "
            "visible effect. boxen has no hue channel -- drop the color "
            "encoding, or drop palette=."
        )


def _prep_cat_axis_sorts(ctx: _ResolveContext) -> None:
    """Inject x_sort and y_sort for composites with a categorical positional axis.

    Lets the desugar function forward sort to the categorical channel encoding,
    enabling e.g. X("c:N", sort="-y") on mark_boxplot/violin/etc.

    errorbar always treats x as the categorical grouping axis (see
    desugar_errorbar), so it only consumes x_sort; y_sort is never read there
    and is deliberately not injected (R4: dead-param removal).
    """
    if ctx.x_sort is not None:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "x_sort": ctx.x_sort}
    if ctx.y_sort is not None and ctx.kind != "errorbar":
        ctx.transform_kwargs = {**ctx.transform_kwargs, "y_sort": ctx.y_sort}


def _prep_color_field(ctx: _ResolveContext) -> None:
    """Thread the color (hue) field into the desugar so each (x, hue) group gets
    its own split summary, and every emitted layer carries a consistent color
    encoding.

    Respect an explicit caller-supplied color_field (non-falsy overrides the
    encoding-inferred one); a None/empty value from the mark call's default is
    treated as "not set" so the encoding can still inject it.
    """
    if ctx.transform_kwargs.get("color_field"):
        return
    color_field = ctx._color_field()
    if color_field:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "color_field": color_field}


def _prep_groupby_from_color(ctx: _ResolveContext) -> None:
    """density/histogram: auto-set groupby from color encoding when not explicit."""
    if "groupby" in ctx.transform_kwargs:
        return
    color_field = ctx._color_field()
    if color_field:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "groupby": color_field}


def _prep_hex_field_from_color(ctx: _ResolveContext) -> None:
    """hex: infer aggregate field from color encoding when not explicit."""
    if not (
        ctx.transform_kwargs.get("aggregate", "count") != "count"
        and ctx.transform_kwargs.get("field") is None
    ):
        return
    color_field = ctx._color_field()
    if color_field:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "field": color_field}


def _prep_smooth_force_name(ctx: _ResolveContext) -> None:
    """When mark_smooth was called on a chart that already had a primitive mark,
    force the Smooth transform to be named so __final__ stays the raw data for
    the scatter layer.
    """
    prior_mark = ctx.chart._pending_stat_mark.prior_mark
    if prior_mark is not None and "name" not in ctx.transform_kwargs:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "name": "smooth"}


def _cast_swarm_value(ctx: _ResolveContext) -> None:
    """swarm: the Rust swarm transform requires the value column to be Float64.

    Detect the value field via the categorical-axis convention, where both the
    encoding (value field name) and the data are available.

    The predicate is "any non-Float64 dtype" (not just integers), matching the
    original chart.py swarm cast block.  Float32 and numeric-string columns must
    also be cast so the Rust layer's hard Float64 requirement is satisfied.
    """
    _, _, value_field = _resolve_cat_axis(
        ctx.transform_kwargs, ctx.x_field, ctx.y_field, ctx.x_sort, ctx.y_sort
    )
    if value_field:
        ctx.cast_fields = [value_field]
        ctx.cast_int_only = False


def _cast_xy_value(ctx: _ResolveContext) -> None:
    """boxplot/boxen/violin: the Rust box_stats / letter_value / violin transforms
    require the value column to be Float64.

    catplot orient="h" encodes x=numeric, y=categorical but does not pass
    horizontal=True to mark_boxplot — CoordFlip handles the visual flip at render
    time.  Instead of relying on the horizontal kwarg, detect the value field by
    dtype: cast whichever of x_field / y_field holds an integer column (string /
    categorical columns are skipped by ``cast_columns_to_float64``).
    """
    ctx.cast_fields = [f for f in (ctx.x_field, ctx.y_field) if f]


# Each kind's ordered list of (pre-desugar) prep steps and (post-desugar) cast
# steps.  Order within each list matches the original control flow exactly.
@dataclass(frozen=True)
class _KindPrep:
    pre: tuple[Callable[[_ResolveContext], None], ...] = ()
    post_cast: tuple[Callable[[_ResolveContext], None], ...] = ()


_PENDING_PREP: dict[str, _KindPrep] = {
    "ribbon": _KindPrep(pre=(_prep_ribbon_y2,)),
    "boxen": _KindPrep(
        pre=(_prep_boxen_agg_sort, _prep_cat_axis_sorts, _prep_boxen_palette_color_conflict),
        post_cast=(_cast_xy_value,),
    ),
    "boxplot": _KindPrep(
        pre=(_prep_cat_axis_sorts, _prep_color_field),
        post_cast=(_cast_xy_value,),
    ),
    "errorbar": _KindPrep(pre=(_prep_cat_axis_sorts, _prep_color_field)),
    "violin": _KindPrep(
        pre=(_prep_cat_axis_sorts, _prep_color_field),
        post_cast=(_cast_xy_value,),
    ),
    "swarm": _KindPrep(pre=(_prep_cat_axis_sorts,), post_cast=(_cast_swarm_value,)),
    "errorband": _KindPrep(pre=(_prep_color_field,)),
    "density": _KindPrep(pre=(_prep_groupby_from_color,)),
    "histogram": _KindPrep(pre=(_prep_groupby_from_color,)),
    "hex": _KindPrep(pre=(_prep_hex_field_from_color,)),
    "smooth": _KindPrep(pre=(_prep_smooth_force_name,)),
}


def _resolve_pending_impl(chart: "Chart") -> "Chart":
    """Resolve a pending statistical mark desugar once encoding is known.

    Body of ``Chart._resolve_pending`` — extracted here as a free function so
    the chart module keeps only the fluent surface and ``to_spec``
    orchestration.  ``Chart._resolve_pending(self)`` delegates to this.

    Calling a composite/diagnostic ``mark_*()`` before ``.encode()`` stores a
    ``_PendingMark`` sentinel.  This is called at the start of every render /
    spec-build path to apply ``desugar_fn`` against the now-populated encoding
    dict.

    ``desugar_fn(x_field, y_field, **kwargs)`` returns a ``MarkDesugarResult``.
    When ``result.layers`` is set, the chart enters layered mode; otherwise it
    applies the single-mark ``result.mark``, ``result.remap``, and
    ``result.position``.
    """
    if chart._pending_stat_mark is None:
        return chart
    kind = chart._pending_stat_mark.kind
    kwargs = chart._pending_stat_mark.kwargs
    desugar_fn = chart._pending_stat_mark.desugar_fn
    x_enc = chart._encoding.get("x")
    y_enc = chart._encoding.get("y")
    x_field = (
        (x_enc.field if isinstance(x_enc, ChannelBase) else x_enc) if x_enc is not None else None
    )
    y_field = (
        (y_enc.field if isinstance(y_enc, ChannelBase) else y_enc) if y_enc is not None else None
    )
    # If encodings still contain unresolved repeat placeholders, defer
    # desugaring until the RepeatChart substitutes concrete field names.
    from ferrum.repeat import _RepeatPlaceholder

    if isinstance(x_field, _RepeatPlaceholder) or isinstance(y_field, _RepeatPlaceholder):
        return chart
    # Capture sort from the original channel objects before they are reduced to
    # bare field strings.  Composite desugars that build a categorical positional
    # axis (boxplot, boxen, errorbar, violin, swarm) receive these so they can
    # wrap the `cat` field in X(cat, sort=...) / Y(cat, sort=...) and preserve
    # the user's sort intent through the desugar expansion.
    #
    # Shorthand sorts ("-y", "y", "-x", "x") are resolved to explicit dicts here
    # using the original field names.  Composite desugars replace the y-channel
    # field with computed columns (e.g. lower_whisker, q1), so the Rust renderer
    # would otherwise sort by the wrong column.
    _raw_x_sort = x_enc.option("sort") if isinstance(x_enc, ChannelBase) else None
    _raw_y_sort = y_enc.option("sort") if isinstance(y_enc, ChannelBase) else None
    x_sort = (
        _resolve_sort_for_composite(_raw_x_sort, x_field, y_field)
        if _raw_x_sort is not None
        else None
    )
    y_sort = (
        _resolve_sort_for_composite(_raw_y_sort, x_field, y_field)
        if _raw_y_sort is not None
        else None
    )
    # Partition the user kwargs by the desugar signature: declared params are
    # transform kwargs; everything else is a constant mark-style override.
    # ``style_dict`` is validated/normalised through MarkBase so aliases
    # (color→fill, alpha→opacity) resolve and unknown keys raise.
    transform_kwargs, style_kwargs = _split_style_kwargs(desugar_fn, kwargs)
    style_dict = MarkBase(kind, **style_kwargs).to_mark_kwargs_dict()

    # Run the kind's declared per-kind kwarg-injection prep steps (no-op for
    # primitive/uncatalogued kinds), preserving the original ordering exactly.
    ctx = _ResolveContext(
        chart=chart,
        kind=kind,
        x_field=x_field,
        y_field=y_field,
        transform_kwargs=transform_kwargs,
        x_sort=x_sort,
        y_sort=y_sort,
    )
    prep = _PENDING_PREP.get(kind)
    if prep is not None:
        for step in prep.pre:
            step(ctx)
    transform_kwargs = ctx.transform_kwargs

    result = desugar_fn(x_field, y_field, **transform_kwargs)
    new = chart._clone()
    new._pending_stat_mark = None

    # Run the kind's declared post-desugar Float64-cast steps against the cloned
    # chart's data, where both the encoding (value field name) and the data are
    # available.
    if prep is not None and prep.post_cast:
        for step in prep.post_cast:
            step(ctx)
        if ctx.cast_fields:
            new._data = cast_columns_to_float64(
                new._data, ctx.cast_fields, int_only=ctx.cast_int_only
            )

    # Build a scatter layer from the prior mark if present.
    _prior_mark = chart._pending_stat_mark.prior_mark
    _prior_layer = None
    if _prior_mark is not None and _prior_mark in _PRIMITIVE_MARKS:
        _prior_layer = _build_prior_layer(
            _prior_mark,
            chart._encoding,
            chart._mark_kwargs,
            chart._position,
        )

    # Layered mode: result.layers is set.
    if result.layers is not None:
        new._transforms = list(chart._transforms) + list(result.transforms)
        # Apply constant style to every emitted layer (user value wins over any
        # desugar-set per-layer default).  The prior primitive layer is left
        # untouched.
        emitted = [_apply_style_to_layer(lyr, style_dict) for lyr in result.layers]
        # Scale-through-desugar propagation (design spec §6, GH #45
        # prerequisite): reattach any chart-level explicit x/y scale the
        # desugar's fresh per-layer encodings would otherwise drop.
        emitted = [_propagate_layer_scale(chart._encoding, lyr) for lyr in emitted]
        all_layers = emitted
        if _prior_layer is not None:
            all_layers = [_prior_layer] + emitted
        new._layers = all_layers
        new._mark = None  # signals layered mode in to_spec
        return new

    # Single-mark mode.
    mark = result.mark
    transforms = result.transforms
    remap = result.remap
    if _prior_layer is not None:
        smooth_enc = dict(chart._encoding)
        if remap:
            _apply_remap(smooth_enc, remap, orig_encoding=chart._encoding)
        smooth_enc = _propagate_positional_scale(chart._encoding, smooth_enc)
        # Style applies to the composite's emitted smooth layer only; the prior
        # scatter layer keeps its own mark_kwargs.
        smooth_layer = _Layer(
            mark=mark,
            encoding=smooth_enc,
            mark_kwargs=(dict(style_dict) if style_dict else None),
            data_source="smooth",
        )
        new._transforms = list(chart._transforms) + list(transforms)
        new._layers = [_prior_layer, smooth_layer]
        new._mark = None
        return new
    new._mark = mark
    new._transforms = list(chart._transforms) + list(transforms)
    # Only override the cloned default when the user passed style kwargs, so
    # output stays byte-identical when none are given.
    if style_dict:
        new._mark_kwargs = style_dict
    if result.position is not None:
        new._position = result.position
    if remap:
        _apply_remap(new._encoding, remap, orig_encoding=chart._encoding)
    new._encoding = _propagate_positional_scale(chart._encoding, new._encoding)
    return new
