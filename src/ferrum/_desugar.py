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
kind's declared kwarg-injection prep steps → call ``desugar_fn`` → route
layered-vs-single result.  The dispatch table produces the *exact same*
transform-kwarg injections, sort keys, color/groupby inference, and name
forcing, in the same order, as the old chain — the change is structural, not
behavioural.

Composite marks used to carry a post-desugar Float64-cast step
(``cast_columns_to_float64``) here, because the Rust transforms it fed
(``box_stats``/``letter_value``/``violin``/``swarm``) once hard-rejected any
value column that was not exactly ``Float64``.  The Rust dtype-policy sweep
(batch D, M1) moved that widening into the transform engine itself
(``numeric_util::coerce_to_float64``): every integer/unsigned/float dtype now
coerces engine-side, and anything else raises a typed refusal naming the
transform, column, and dtype.  The Python-side mirror is gone — it was
redundant for the int-only box/boxen/violin predicate and actively wrong for
swarm's any-non-Float64 predicate, which silently coerced numeric-*string*
columns instead of letting the engine refuse them by type.

These are free functions; ``chart.py`` imports them, so this module must never
import ``Chart`` at module level (that would create an import cycle).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Union

from ferrum._layer import (
    _FLAT_MARK_METHODS,
    _Layer,
    MarkDesugarResult,
    apply_catch_all_style,
    catch_all_over,
    titled_encoding,
)
from ferrum.encoding.base import ChannelBase
from ferrum.encoding._scale import _scale_to_dict
from ferrum.marks._desugar_helpers import DISC_FLAG_DEFAULTS
from ferrum._entry import EntryRefusal, voiced_as
from ferrum._warn import user_stacklevel
from ferrum.marks.base import MarkBase, refuse_style_no_layer_draws, resolve_layer_style
from ferrum.marks.statistical import _build_prior_layer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ferrum.chart import Chart


class _UnsetType:
    """Sentinel marking a composite-mark bool kwarg as "caller never passed it".

    ``mark_discrimination_threshold``'s ``threshold_line``/``optimum_label``
    default to a concrete bool the same way every other flag on this module's
    marks does, which makes an explicit ``optimum_label=True`` from the
    caller indistinguishable from the parameter's own default once it
    reaches :func:`_prep_discrimination_threshold_sentinels` (#189) -- and
    that distinction is exactly what decides whether a ``metrics=`` subset
    missing the f1-optimum row should refuse (explicit request, self-
    contradicting) or silently downgrade (default request, batch-F T3's
    crash-shape fix).

    Why a third sentinel rather than one of the two that already exist
    (corrected 2026-09-17; the earlier rationale named a dependency
    inversion that is real for only one of them):

    - ``ferrum.plots._helpers._UNSET`` genuinely cannot be reused here.
      ``plots`` sits *above* ``marks``/``_desugar`` in the import graph
      (``plots/_helpers.py`` imports from ``ferrum.marks``), so importing it
      from this module would invert that dependency.
    - ``ferrum._title_sentinel._UNSET`` imports nothing from ``ferrum`` at
      all, so reusing it would invert nothing. It is not reused because its
      instance is the subject of two *named, enforced contracts* scoped to
      the ``Axis``/``Legend`` value objects -- the three-way ``title``
      contract (``serialize_title``) and the two-way omit-vs-explicit
      contract (``is_unspecified``) -- and every ``is`` test on it in that
      module would silently start answering for this module's mark kwargs
      too. Sharing an identity means sharing those contracts; the flags here
      obey neither.

    The instance is a **singleton**, matching ``_title_sentinel``'s own
    ``__new__``, and preserves its identity through ``copy.deepcopy`` and
    pickling. That is load-bearing, not tidiness: every consumer tests
    ``value is _UNSET``, so a copy that produced a second instance would
    read as an *explicit* ``True``-ish value, and #189's disposition would
    flip from "silently downgrade" to "refuse" on a chart that had merely
    been round-tripped. ``Chart._clone()``'s shallow ``copy.copy`` never
    could break it, but ``deepcopy`` and ``pickle`` both did before
    (measured 2026-09-17: ``copy.deepcopy(_UNSET) is _UNSET`` was ``False``).
    """

    _instance: "_UnsetType | None" = None

    def __new__(cls) -> "_UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_UNSET"

    def __deepcopy__(self, memo: dict) -> "_UnsetType":
        """The singleton is its own deep copy -- identity is the value."""
        return self

    def __reduce__(self) -> tuple:
        """Unpickle through ``__new__``, which hands back the one instance."""
        return (_UnsetType, ())


#: Sentinel default for ``mark_discrimination_threshold(threshold_line=, optimum_label=)``.
_UNSET = _UnsetType()

#: Type alias for a bool kwarg that distinguishes "omitted" (``_UNSET``) from
#: an explicit ``True``/``False``.
BoolOrUnset = Union[bool, _UnsetType]


def resolve_disc_flag(flag: str, value: BoolOrUnset) -> bool:
    """Resolve one ``mark_discrimination_threshold`` flag to a concrete bool.

    ``_UNSET`` (the caller never passed the kwarg) resolves to ``flag``'s
    entry in :data:`ferrum.marks._desugar_helpers.DISC_FLAG_DEFAULTS` -- the
    one owner of those two values, homed there because the desugar's own
    signature reads them too and cannot import this module (see that
    constant's note). Anything else is the caller's own value. The two
    consumers that must agree -- this module's
    :func:`_resolve_or_refuse_disc_flag` and the mark method's
    ``*_effective`` resolution -- both call this, so neither can spell a
    default the other does not.

    An unknown ``flag`` is an internal programming error (both call sites
    pass string literals), and it is refused by name rather than as a bare
    ``KeyError`` on a dict the caller cannot see -- the same voice
    ``_SourceCache``'s malformed-key ``TypeError`` and
    ``_curve_metric``'s unreachable branch speak in.
    """
    if flag not in DISC_FLAG_DEFAULTS:
        raise KeyError(
            f"resolve_disc_flag: {flag!r} is not a discrimination-threshold flag; "
            f"DISC_FLAG_DEFAULTS (ferrum.marks._desugar_helpers) owns "
            f"{sorted(DISC_FLAG_DEFAULTS)}. Add the flag there before resolving it."
        )
    return DISC_FLAG_DEFAULTS[flag] if value is _UNSET else bool(value)


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
    *user_kwargs*. A ``_``-prefixed parameter is injected by a prep step
    from private chart state (e.g. ``desugar_hex``'s ``_color``), never
    a user kwarg: a user key of that name is a style key, which ``MarkBase``
    refuses as unknown.

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
        and not name.startswith("_")
    }
    transform_kwargs: dict = {}
    style_kwargs: dict = {}
    for k, v in user_kwargs.items():
        if k in transform_names:
            transform_kwargs[k] = v
        else:
            style_kwargs[k] = v
    return transform_kwargs, style_kwargs


def validate_layer_kwargs(kind: str, layer_kwargs: Any, *, where: str | None = None) -> dict:
    """Check a ``layer_kwargs=`` door against *kind*'s own layer vocabulary.

    ``layer_kwargs={"line": {"stroke": "red"}}`` aims mark kwargs at ONE
    emitted sub-layer of a composite mark, where the mark's bare
    ``**mark_kwargs`` reach every layer it emits (a bare ``stroke=`` on a
    ``ci``-bearing ``mark_smooth`` repaints the confidence band too). It is
    the fluent-path twin of the figure-level ``mark={"line": {...}}``
    override, and it is validated against the same vocabulary:
    ``_overrides.LAYER_NAME_CATALOG[kind]``, the registry every composite
    mark already populates with the sub-layers it can draw. This check runs
    at the call, before the desugar, so it judges names only; once the
    desugar has run, the dispatcher holds each aimed dict to its layer's
    mark and refuses one aimed at a catalogued layer this call did not
    draw (a ``ribbon`` under ``ci=None``), as ``_apply_mark_overrides``
    does -- one vocabulary, two doors, not two vocabularies.

    An unknown name is REFUSED rather than silently ignored, in the voice of
    the entry the caller invoked (*where*, defaulting to ``mark_<kind>``):
    a typo'd layer name otherwise renders a plausible chart with the styling
    nowhere in it.

    Returns the per-layer mapping with empty per-layer dicts dropped, so a
    caller forwarding an absent option (``lmplot``'s ``line_kws=None`` ->
    ``{}``) leaves every layer untouched and its render byte-identical.

    Raises
    ------
    EntryRefusal
        A ``ValueError`` voiced as *where*, re-voiceable as the figure
        function or visualizer that wrote the dict, if a key is not a
        sub-layer *kind* can draw, or *layer_kwargs*, or
        any of its values, is not a dict -- mirroring
        ``_apply_mark_overrides``' refusal of a non-dict override value.
    """
    if layer_kwargs is None:
        return {}
    entry = where or f"mark_{kind}"
    if not isinstance(layer_kwargs, dict):
        got = type(layer_kwargs).__name__
        raise EntryRefusal(
            lambda voice: (
                f"{voice}(layer_kwargs=) must be a dict mapping a sub-layer name to that "
                f"layer's mark kwargs, e.g. layer_kwargs={{'line': {{'stroke': 'red'}}}}; "
                f"got {got}."
            ),
            entry,
        )
    from ferrum._overrides import LAYER_NAME_CATALOG

    valid = LAYER_NAME_CATALOG.get(kind, frozenset())
    aimed: dict = {}
    for name, style in layer_kwargs.items():
        if name not in valid:
            raise EntryRefusal(_unknown_layer_kwargs_message(name, valid), entry)
        if not isinstance(style, dict):
            got = type(style).__name__
            raise EntryRefusal(
                lambda voice: (
                    f"{voice}(layer_kwargs={{{name!r}: ...}}) must be a dict of mark "
                    f"kwargs for that layer; got {got}."
                ),
                entry,
            )
        if style:
            aimed[name] = dict(style)
    return aimed


def _unknown_layer_kwargs_message(name: Any, valid: frozenset[str]) -> Callable[[str], str]:
    """The refusal text, for any voice, of ``layer_kwargs=`` aimed at *name*, no sub-layer."""

    def message_for(voice: str) -> str:
        drawn = (
            f"{voice} draws {sorted(valid)}"
            if valid
            else f"{voice} draws no named sub-layers, so it has nothing to aim at"
        )
        return (
            f"{voice}(layer_kwargs=): unknown sub-layer {name!r}; {drawn}. "
            "Pass style that should reach every layer as a plain mark kwarg "
            f"instead, e.g. {voice}(opacity=0.5)."
        )

    return message_for


def _aim_layer_kwargs(layer: _Layer, layer_kwargs: dict, entry: str) -> _Layer:
    """Apply the caller's per-layer style to *layer*, recording the keys as theirs.

    The style is held to the layer's own mark (:func:`resolve_layer_style`,
    voiced as *entry*).

    Recorded on ``caller_style_keys`` (not just merged into ``mark_kwargs``)
    because the chart-level catch-all is spread over the same layers moments
    later: an instruction aimed at one layer is more specific than one aimed
    at the mark as a whole, and ``apply_catch_all_style`` reads that
    distinction off this field.  Layers this call did not name are returned
    unchanged.
    """
    aimed = layer_kwargs.get(layer.name) if layer.name is not None else None
    if not aimed:
        return layer
    aimed = resolve_layer_style(entry, layer.mark, aimed)
    return replace(
        layer,
        mark_kwargs={**(layer.mark_kwargs or {}), **aimed},
        caller_style_keys=layer.caller_style_keys | frozenset(aimed),
    )


def _refuse_undrawn_layer_kwargs(layer_kwargs: dict, drawn: set, entry: str) -> None:
    """Refuse per-layer style aimed at a catalogued layer this call did not draw."""
    undrawn = sorted(name for name in layer_kwargs if name not in drawn)
    if undrawn:
        raise EntryRefusal(
            lambda voice: (
                f"{voice}(layer_kwargs=): this call draws no {undrawn!r} sub-layer, so "
                f"nothing would take the style; it draws {sorted(drawn)}"
            ),
            entry,
        )


def _aim_single_mark_kwargs(
    result: MarkDesugarResult, layer_kwargs: dict, entry: str
) -> tuple[dict | None, frozenset]:
    """The single-mark half of :func:`_aim_layer_kwargs`.

    A single-mark result has no ``_Layer`` to carry a name, so the layer it
    corresponds to is the one it declares in ``layer_name`` (``None`` when
    the mark is not a catalogued sub-layer of anything -- then no per-layer
    instruction can reach it, and the caller's ``**mark_kwargs`` are its only
    door).  Returns the pair the single-mark merge consumes, unchanged when
    nothing was aimed here.
    """
    aimed = layer_kwargs.get(result.layer_name) if result.layer_name is not None else None
    if not aimed:
        return result.mark_kwargs, result.caller_style_keys
    aimed = resolve_layer_style(entry, result.mark, aimed)
    return (
        {**(result.mark_kwargs or {}), **aimed},
        result.caller_style_keys | frozenset(aimed),
    )


def _merge_desugar_mark_kwargs(
    desugar_kwargs: dict | None,
    style_dict: dict,
    caller_style_keys: frozenset,
) -> dict | None:
    """Merge a single-mark result's ``mark_kwargs`` with the caller's catch-all.

    Precedence is ``_layer.catch_all_over``'s, the same rule
    ``_layer.apply_catch_all_style`` applies in layered mode, so the two
    shapes of a desugar result cannot answer the question differently.
    Returns ``None`` when neither side has anything to set, so output stays
    byte-identical to before either source existed.

    *caller_style_keys* is required, with no default:
    ``apply_catch_all_style`` reads its layer's copy unconditionally, and a
    default here would let a future call site omit the record and silently
    get the pre-#199 catch-all-always-wins ordering instead of an error.
    """
    if not desugar_kwargs and not style_dict:
        return None
    return catch_all_over(desugar_kwargs, style_dict, caller_style_keys)


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
        return layer_value._with_options(scale=chart_scale)

    if _scale_domain(layer_scale) is not None:
        return None  # mark-computed domain wins

    chart_domain = _scale_domain(chart_scale)
    if chart_domain is None:
        return None  # nothing to merge

    merged_scale = {**_scale_to_dict(layer_scale), "domain": chart_domain}
    return layer_value._with_options(scale=merged_scale)


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


@dataclass
class _ResolveContext:
    """Mutable working state threaded through the per-kind preprocessing steps.

    A single context object lets the dispatch-table prep functions read and
    write the same fields the old per-``kind`` ``if`` chain mutated inline
    (``transform_kwargs``, ``x_sort``/``y_sort``, ``color_sort``), so the
    orchestration stays a flat, readable sequence.
    """

    chart: "Chart"
    kind: str
    x_field: str | None
    y_field: str | None
    transform_kwargs: dict
    x_sort: Any
    y_sort: Any
    color_sort: Any
    #: The scale type the caller DECLARED on the chart-level x/y channel
    #: (normalized single-letter code), or ``None`` when they declared none.
    #: Never an inferred type -- see :func:`_prep_cat_axis_types`.
    x_type: str | None
    y_type: str | None

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


def _prep_boxen_ramp_depth(ctx: _ResolveContext) -> None:
    """Inject the real per-chart materialized letter-value depth so the
    default opacity ramp normalizes against what THIS chart actually
    renders, not the mark's full registry-derived depth cap (quality-review
    REVISE, 2026-09-06 cycle 2): raising that cap (6->12, F-L03-04b) must
    not flatten the ramp's visual contrast on a chart whose real rendered
    depth hasn't changed at all.

    This is the one place a composite desugar's prep step reads the bound
    chart data (``ctx.chart._data``) -- desugar functions otherwise never
    touch it (see ``composite._boxen_band_color_index``'s docstring on why
    real per-group depth is normally Rust-only). The computation itself
    stays in ``composite.py`` (:func:`_boxen_materialized_depth`), scoped to
    a lightweight arithmetic mirror of the depth-selection formula, never
    the quantile math; a `palette=` chart skips this entirely since it
    never uses the opacity ramp.

    ``group_field`` MUST mirror ``desugar_boxen``'s own ``group =
    color_field if color_field else cat`` exactly (spec-review cycle-2
    finding: the fallback used to also consult ``ctx._color_field()`` --
    the chart-level ``.encode(color=...)`` channel -- which is not a
    grouping boxen ever uses. ``desugar_boxen``'s own docstring is explicit
    that ``palette`` "does not interact with ``color_field`` (which only
    changes the ``LetterValue`` groupby, not color", i.e. the *only*
    grouping input is the ``color_field=`` mark kwarg, never the chart-level
    color encoding -- boxen's ``_PENDING_PREP`` tuple deliberately omits
    ``_prep_color_field`` for exactly this reason. Estimating depth from the
    wrong grouping reproduces the cycle-1 symptom (a ramp normalized
    against a depth the chart doesn't render) through a new mechanism, on
    any chart that merely adds a cosmetic ``.encode(color=...)`` without
    ``palette=``.

    The chart's facet field(s), when present (``ctx.chart._facet``), are
    folded into :func:`_boxen_materialized_depth`'s groupby (batch-E close
    remediation, 2026-09-07 -- see that function's ``extra_groupby_fields``
    docstring): a faceted chart's real per-panel depth can be materially
    shallower than the depth this desugar would estimate by pooling a
    group's rows across every panel, which otherwise flattens the ramp
    contrast on every panel toward whichever one pools deepest. This is a
    disclosed byte-stability exception: a faceted ``mark_boxen()`` chart's
    rendered SVG bytes may change from before this fix.

    **Scope of the facet fix, stated exactly (cycle-2 design review S3):**
    this only works when ``ctx.chart._facet`` is already set at the moment
    this prep step runs, i.e. when ``.facet()`` was called BEFORE the
    pending mark was resolved. ``_resolve_pending_impl`` runs eagerly from
    ``Chart.__add__`` and ``Chart.layer()`` (``chart.py`` -- both call
    ``self._resolve_pending()`` before doing their own work), so
    ``mark_boxen() + mark_point()`` followed by ``.facet(col=...)`` resolves
    the boxen desugar (and this prep step) BEFORE ``.facet()`` ever runs --
    ``ctx.chart._facet`` is still ``None`` at that point, so the fix does
    not apply and the chart still pools across panels exactly as before
    this batch. ``mark_boxen().facet(col=...)`` (and the equivalent
    ``encode(facet_col=...)`` / ``catplot(kind="boxen", col=...)``
    spellings, which resolve the pending mark no earlier) ARE fixed, since
    ``.facet()`` runs first there and ``ctx.chart._facet`` is populated by
    the time this prep step reads it. A lifecycle re-plumb so the layered
    path also resolves facet-aware is a filed follow-up, not attempted
    here -- see ``tests/marks/test_boxen_ramp_prep.py``'s
    ``test_layered_boxen_facet_still_pools_across_panels_pending_lifecycle_fix``
    for the pinned pooled value.
    """
    if ctx.transform_kwargs.get("palette") is not None:
        return
    horizontal = bool(ctx.transform_kwargs.get("horizontal", False))
    value_field = ctx.x_field if horizontal else ctx.y_field
    cat_field = ctx.y_field if horizontal else ctx.x_field
    group_field = ctx.transform_kwargs.get("color_field") or cat_field
    if value_field is None or group_field is None:
        return

    from ferrum.marks.composite import _boxen_materialized_depth

    # Read k_depth/k_proportion as passed, with no re-declared default and no
    # presence guard (design review cycle-2 S2): `Chart.mark_boxen()` always
    # forwards both explicitly (its own signature carries the real defaults,
    # "tukey"/0.007), so by the time a boxen reaches this prep step both keys
    # are always present -- a THIRD copy of those literals here would only
    # ever drift from `desugar_boxen`'s own defaults if either default
    # changed. A future caller that reaches this prep step without either
    # key is an impossible-today case that should raise loud (`KeyError`),
    # not silently degrade to the full-cap fallback ramp -- this project's
    # no-silent-fallback constraint declines the "safer to not raise" trade
    # a membership guard would make here.
    #
    # Facet-scoped groupby (2026-09-07, disclosed byte-stability rider) --
    # ONLY when `ctx.chart._facet` is already populated at this point (see
    # this function's docstring "Scope of the facet fix" section: `+`/
    # `.layer()` resolve the pending mark before `.facet()` can ever run, so
    # this reads `None` and the fix is inert on that lifecycle -- pinned by
    # `test_layered_boxen_facet_still_pools_across_panels_pending_lifecycle_fix`).
    # When it IS populated, folding the facet field(s) into
    # `_boxen_materialized_depth`'s groupby corrects the estimate: a faceted
    # chart's LetterValue transform runs on each facet panel's own row
    # subset, but this desugar runs once for the whole chart, so pooling a
    # group's rows ACROSS panels (the pre-fix behavior) can only ever
    # estimate a depth >= what any single panel really materializes, which
    # flattens every panel's ramp contrast toward the panel with the
    # deepest pooled group -- see `_boxen_materialized_depth`'s
    # `extra_groupby_fields` docstring and `tests/marks/test_boxen_ramp_prep.py`'s
    # faceted case for the measured effect. `_Facet.field`/`.row`/`.col`
    # mirrors `_spec_build.py`'s own `(resolved._facet.col, resolved._facet.row,
    # resolved._facet.field)` sweep: wrap mode sets only `field`, grid mode
    # sets only `row`/`col` (`_Facet.__post_init__` enforces this), so no
    # mode branch is needed.
    facet = ctx.chart._facet
    facet_fields = (facet.field, facet.row, facet.col) if facet is not None else ()

    depth = _boxen_materialized_depth(
        ctx.chart._data,
        value_field=value_field,
        group_field=group_field,
        k_depth=ctx.transform_kwargs["k_depth"],
        k_proportion=ctx.transform_kwargs["k_proportion"],
        extra_groupby_fields=tuple(f for f in facet_fields if f),
    )
    if depth is not None:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "materialized_depth": depth}


#: Kinds whose categorical positional axis is ALWAYS x: their desugars take no
#: ``y_sort``/``y_type`` parameter at all (``desugar_errorbar`` groups by
#: ``x_field`` unconditionally; ``desugar_errorband`` aggregates by it), so
#: injecting one would be a ``TypeError`` rather than a silently dead kwarg.
#: Spelled once here because two prep steps below key the same carve-out on it.
_CAT_AXIS_ALWAYS_X = frozenset({"errorbar", "errorband"})


def _prep_cat_axis_sorts(ctx: _ResolveContext) -> None:
    """Inject x_sort and y_sort for composites with a categorical positional axis.

    Lets the desugar function forward sort to the categorical channel encoding,
    enabling e.g. X("c:N", sort="-y") on mark_boxplot/violin/etc.

    errorbar always treats x as the categorical grouping axis (see
    desugar_errorbar), so it only consumes x_sort; y_sort is never read there
    and is deliberately not injected (R4: dead-param removal) -- that carve-out
    is :data:`_CAT_AXIS_ALWAYS_X`, shared with :func:`_prep_cat_axis_types`.
    """
    if ctx.x_sort is not None:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "x_sort": ctx.x_sort}
    if ctx.y_sort is not None and ctx.kind not in _CAT_AXIS_ALWAYS_X:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "y_sort": ctx.y_sort}


def _prep_cat_axis_types(ctx: _ResolveContext) -> None:
    """Inject x_type and y_type -- the scale type the CALLER declared (#171).

    ``_resolve_pending_impl`` reduces the chart-level positional channels to
    bare field-name strings before running a composite desugar, which used to
    discard everything else the caller stated on them. ``sort=`` was already
    rescued (:func:`_prep_cat_axis_sorts`); the declared *type* was not, so
    ``.encode(x=fm.X("cyl", type_="N"))`` -- and its ``"cyl:N"`` shorthand --
    was inert on precisely the marks whose band geometry depends on resolving
    ordinal. The desugars re-bind the type through
    :func:`~ferrum.marks._desugar_helpers.typed_positional_channel`, which also
    owns the "default the category channel to Nominal" half of the rule.

    Only the declared type is threaded here, never an inferred one: a kind that
    receives ``None`` is the caller having stated nothing, which is what lets
    the desugar apply its own default (and what keeps
    ``catplot(native_scale=True)``'s deliberate ``type_="Q"`` authoritative
    rather than overruled).

    :data:`_CAT_AXIS_ALWAYS_X` kinds take ``x_type`` only, for the same reason
    they take ``x_sort`` only.
    """
    if ctx.x_type is not None:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "x_type": ctx.x_type}
    if ctx.y_type is not None and ctx.kind not in _CAT_AXIS_ALWAYS_X:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "y_type": ctx.y_type}


def _prep_color_field(ctx: _ResolveContext) -> None:
    """Thread the color (hue) field into the desugar so each (x, hue) group gets
    its own split summary, and every emitted layer carries a consistent color
    encoding.

    Respect an explicit caller-supplied color_field (non-falsy overrides the
    encoding-inferred one); a None/empty value from the mark call's default is
    treated as "not set" so the encoding can still inject it.

    Also threads ``color_sort`` -- the chart's ``.encode(color=Color(...,
    sort=[...]))`` level order (F-L09-04) -- alongside the field, so a
    ``hue_order=`` built by a figure function (or a caller's own explicit
    ``Color(sort=...)``) reaches the composite desugar's color channel
    instead of being flattened away by :meth:`_ResolveContext._color_field`.
    Two guards apply, independently of the ``color_field`` early-return above:

    - **Type guard.** Only a literal level list/tuple is forwarded. A color
      sort is documented as a literal hue-level order, not a shorthand
      (``"-y"``) or an aggregate-sort dict -- those forms cannot resolve
      against a composite desugar's renamed output columns (e.g.
      ``lower_whisker``), so forwarding them would turn a pre-fix silent
      no-op into a noisy, misleading ``UserWarning`` naming an internal
      column the caller never wrote. Out-of-contract shapes stay exactly as
      inert as they were before this field existed.
    - **Override guard.** An explicit ``color_sort=`` mark kwarg (reachable
      via ``mark_boxplot(color_sort=...)`` etc., since it is a declared
      desugar parameter) is never clobbered by the encoding-derived value --
      this matches ``color_field``'s own override precedence exactly, not
      just in spirit: whichever of the two is caller-supplied wins over
      whatever ``.encode(color=...)`` would otherwise inject.
    """
    if ctx.transform_kwargs.get("color_field"):
        return
    color_field = ctx._color_field()
    if color_field:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "color_field": color_field}
        if isinstance(ctx.color_sort, (list, tuple)) and not ctx.transform_kwargs.get("color_sort"):
            ctx.transform_kwargs = {**ctx.transform_kwargs, "color_sort": ctx.color_sort}


def _prep_groupby_from_color(ctx: _ResolveContext) -> None:
    """density/histogram: auto-set groupby from color encoding when not explicit."""
    if "groupby" in ctx.transform_kwargs:
        return
    color_field = ctx._color_field()
    if color_field:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "groupby": color_field}


def _prep_hex_field_from_color(ctx: _ResolveContext) -> None:
    """hex: a chart-level color names the aggregated column, or is refused.

    The cells are always colored by Hex's per-cell aggregate. A color naming
    the column an aggregate reads supplies ``field`` when none was given. That
    color, or one naming Hex's output column itself, styles the cell color
    binding with its channel properties (title, scheme, legend), its scheme
    becoming the cells' ``cmap`` (refused beside a different ``cmap=``). Any
    other color would be drawn nowhere, so it is refused. Every refusal is
    voiced as the public entry the caller invoked.
    """
    from ferrum._registry import aggregate_needs_field, registry_output_columns

    color_field = ctx._color_field()
    if not color_field:
        return
    kwargs = ctx.transform_kwargs
    aggregate = kwargs.get("aggregate", "count")
    reads_column = aggregate_needs_field("hex", aggregate)
    if reads_column and kwargs.get("field") is None:
        kwargs = {**kwargs, "field": color_field}
    names_aggregate = reads_column and color_field == kwargs.get("field")
    if not names_aggregate and color_field != registry_output_columns("hex", ["value"])["value"]:
        of_field = f" of field={kwargs.get('field')!r}" if reads_column else ""
        entry = ctx.chart._pending_stat_mark.entry or "mark_hex"
        raise ValueError(
            f"{entry}: color={color_field!r} is drawn nowhere; the cells are colored by "
            f"aggregate={aggregate!r}{of_field}. Bind color= to the aggregated column, "
            "or drop it."
        )
    # The polygon cells sample their own `cmap`, so a scheme on the caller's
    # color reaches them through it and the legend agrees with the cells.
    color = ctx.chart._encoding["color"]
    scheme = color.option("scheme") if isinstance(color, ChannelBase) else None
    if scheme is not None:
        cmap = kwargs.get("cmap")
        if cmap is not None and cmap != scheme:
            entry = ctx.chart._pending_stat_mark.entry or "mark_hex"
            raise ValueError(
                f"{entry}: cmap={cmap!r} and color=fm.Color(..., scheme={scheme!r}) both "
                "set the cell colormap; pass only one."
            )
        kwargs = {**kwargs, "cmap": scheme}
    ctx.transform_kwargs = {**kwargs, "_color": color}


def _resolve_or_refuse_disc_flag(
    ctx: _ResolveContext,
    *,
    flag: str,
    sentinel_column: str,
    has_columns: bool,
    columns: Any,
) -> None:
    """Resolve one ``discrimination_threshold`` bool flag to a concrete value.

    Shared by ``optimum_label`` and ``threshold_line`` (#189): both are
    gated by the same "did ``_disc_threshold_prep`` inject the matching
    sentinel column" question, and both need the same explicit-vs-default
    disposition once the answer is no -- an explicit request (the caller
    passed a real bool to ``mark_discrimination_threshold``, distinguishable
    from the default via the ``_UNSET`` sentinel) is a self-contradicting
    ask (asking to label/rule an F1 optimum that ``metrics=`` filtered away)
    and refuses typed; the parameter's own default silently downgrades to
    ``False``, preserving batch-F T3's crash-shape fix for callers who never
    touched the flag.

    The default itself comes from
    :data:`ferrum.marks._desugar_helpers.DISC_FLAG_DEFAULTS` via
    :func:`resolve_disc_flag`, the same call the mark method makes when it
    decides whether to inject ``flag``'s sentinel columns -- so the
    "is there data for this layer?" and "was this layer asked for?"
    questions cannot be answered from two different defaults.
    """
    raw = ctx.transform_kwargs.get(flag, _UNSET)
    explicit = raw is not _UNSET
    resolved = resolve_disc_flag(flag, raw)
    if resolved and (not has_columns or sentinel_column not in columns):
        if explicit:
            from ferrum.marks._desugar_helpers import _normalize_names

            metrics = ctx.transform_kwargs.get("metrics")
            # `metrics=` genuinely excluding "f1" and `metrics=` including
            # "f1" but the underlying data having no "f1" row are two
            # different failures with two different fixes -- checking which
            # one actually happened (via the same loose case/spacing match
            # `_disc_threshold_prep` uses, since `metrics=` may be relabeled
            # for display before it gets here) keeps the message truthful in
            # both shapes instead of always blaming `metrics=`.
            metrics_include_f1 = bool(metrics) and "f1" in _normalize_names(metrics)
            if metrics_include_f1:
                raise ValueError(
                    f"mark_discrimination_threshold({flag}=True) has no "
                    f"F1-optimum row to draw from: metrics={metrics!r} "
                    "includes 'f1', but the data's 'metric' column has no "
                    f"'f1' row, so the sentinel column {sentinel_column!r} "
                    f"was never injected. Add an 'f1' row to the data, or "
                    f"omit {flag}= to fall back to the default (silently "
                    "unlabeled) behavior."
                )
            raise ValueError(
                f"mark_discrimination_threshold({flag}=True) has no F1-optimum "
                f"row to draw from: metrics={metrics!r} excludes 'f1', so the "
                f"sentinel column {sentinel_column!r} was never injected. "
                f"Include 'f1' in metrics=, or omit {flag}= to fall back to "
                "the default behavior (the layer is silently omitted)."
            )
        resolved = False
    ctx.transform_kwargs = {**ctx.transform_kwargs, flag: resolved}


def _prep_discrimination_threshold_sentinels(ctx: _ResolveContext) -> None:
    """Resolve the optimum-label / threshold-line flags against their sentinel columns.

    ``Chart.mark_discrimination_threshold``'s ``data_transform``
    (``_disc_threshold_prep`` in ``marks/_chart_mixins/_classification.py``)
    injects BOTH ``_threshold_best`` (for ``threshold_line``) and
    ``_optimum_x``/``_optimum_y``/``_optimum_text`` (for ``optimum_label``)
    from the SAME F1-optimum-row lookup, gated by the SAME
    ``f1_rows.height == 0`` early return -- with no f1 row there is no
    optimum to draw a rule at or label. ``threshold_line``/``optimum_label``
    themselves default independently of ``metrics=``, so a ``metrics=``
    subset that drops f1 used to reach
    :func:`ferrum.marks.diagnostic.desugar_discrimination_threshold` with
    one or both flags ``True`` and no sentinel columns to back them -- the
    emitted ``mark_rule``/``mark_text`` layers referenced columns that did
    not exist. This is the enforcement point for that shared contract: it
    threads the transform's actual answer ("were the sentinels injected?")
    back into the desugar kwargs for EACH sentinel family independently, so
    neither layer is ever emitted for data that cannot back it, rather than
    letting the desugar and the transform disagree.

    #189: a caller-supplied ``True`` (distinguished from the mark's own
    default via the ``_UNSET`` sentinel -- see :func:`_resolve_or_refuse_disc_flag`)
    now raises instead of vanishing; the un-passed default still downgrades
    silently. Either way, by the time this function returns,
    ``transform_kwargs[flag]`` is always a concrete ``bool`` -- the desugar
    itself never sees the sentinel.
    """
    data = ctx.chart._data
    columns = getattr(data, "columns", None) if data is not None else None
    has_columns = columns is not None

    _resolve_or_refuse_disc_flag(
        ctx,
        flag="optimum_label",
        sentinel_column="_optimum_x",
        has_columns=has_columns,
        columns=columns,
    )
    _resolve_or_refuse_disc_flag(
        ctx,
        flag="threshold_line",
        sentinel_column="_threshold_best",
        has_columns=has_columns,
        columns=columns,
    )


def _prep_smooth_force_name(ctx: _ResolveContext) -> None:
    """When mark_smooth was called on a chart that already had a primitive mark,
    force the Smooth transform to be named so __final__ stays the raw data for
    the scatter layer.
    """
    prior_mark = ctx.chart._pending_stat_mark.prior_mark
    if prior_mark is not None and "name" not in ctx.transform_kwargs:
        ctx.transform_kwargs = {**ctx.transform_kwargs, "name": "smooth"}


# Each kind's ordered list of pre-desugar prep steps.  Order within each list
# matches the original control flow exactly.  (A post-desugar Float64-cast
# step used to live here for boxen/boxplot/violin/swarm — see the module
# docstring for why it was removed.)
@dataclass(frozen=True)
class _KindPrep:
    pre: tuple[Callable[[_ResolveContext], None], ...] = ()


_PENDING_PREP: dict[str, _KindPrep] = {
    "ribbon": _KindPrep(pre=(_prep_ribbon_y2,)),
    "boxen": _KindPrep(
        pre=(
            _prep_boxen_agg_sort,
            _prep_cat_axis_sorts,
            _prep_cat_axis_types,
            _prep_boxen_palette_color_conflict,
            _prep_boxen_ramp_depth,
        ),
    ),
    "boxplot": _KindPrep(pre=(_prep_cat_axis_sorts, _prep_cat_axis_types, _prep_color_field)),
    "errorbar": _KindPrep(pre=(_prep_cat_axis_sorts, _prep_cat_axis_types, _prep_color_field)),
    "violin": _KindPrep(pre=(_prep_cat_axis_sorts, _prep_cat_axis_types, _prep_color_field)),
    "swarm": _KindPrep(pre=(_prep_cat_axis_sorts, _prep_cat_axis_types)),
    "errorband": _KindPrep(pre=(_prep_cat_axis_types, _prep_color_field)),
    "density": _KindPrep(pre=(_prep_groupby_from_color,)),
    "histogram": _KindPrep(pre=(_prep_groupby_from_color,)),
    "hex": _KindPrep(pre=(_prep_hex_field_from_color,)),
    "smooth": _KindPrep(pre=(_prep_smooth_force_name,)),
    "discrimination_threshold": _KindPrep(pre=(_prep_discrimination_threshold_sentinels,)),
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

    Every style refusal the resolution raises speaks as the public entry
    the caller invoked: the figure function or visualizer stamped on the
    pending mark, which may resolve after that call has returned, or
    ``mark_<kind>`` for a mark method called directly.
    """
    if chart._pending_stat_mark is None:
        return chart
    entry = chart._pending_stat_mark.entry or f"mark_{chart._pending_stat_mark.kind}"
    with voiced_as(entry):
        return _resolve_pending_as(chart, entry)


def _resolve_pending_as(chart: "Chart", entry: str) -> "Chart":
    """:func:`_resolve_pending_impl`'s resolution of a pending mark, voiced as *entry*."""
    kind = chart._pending_stat_mark.kind
    # Copied because `layer_kwargs` is popped out below and the pending mark
    # belongs to `chart`, which a caller may resolve more than once.
    kwargs = dict(chart._pending_stat_mark.kwargs)
    desugar_fn = chart._pending_stat_mark.desugar_fn
    # The per-layer style door, taken out before the transform/style split:
    # it addresses LAYERS, not the mark, so it is neither a transform kwarg
    # nor a `MarkBase` style key (which would reject it as unknown). Every
    # composite mark routed through this dispatcher gets the same door and
    # the same refusal, keyed on the same `LAYER_NAME_CATALOG` vocabulary.
    layer_kwargs = validate_layer_kwargs(kind, kwargs.pop("layer_kwargs", None), where=entry)
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
    # A color sort is a literal hue-level order (e.g. `hue_order=`), not a
    # shorthand/aggregate sort spec like x_sort/y_sort -- it is forwarded
    # verbatim, never through `_resolve_sort_for_composite` (F-L09-04).
    color_enc = chart._encoding.get("color")
    color_sort = color_enc.option("sort") if isinstance(color_enc, ChannelBase) else None
    # The declared scale type, captured for the same reason and from the same
    # objects as the sorts above: the bare field names below carry neither, and
    # a composite mark's band geometry needs the categorical channel to resolve
    # ordinal regardless of the column's Arrow dtype (#171). `option("type")`
    # returns the normalized single-letter code for every spelling the caller
    # could have used (`type_="nominal"`, `type_="N"`, `"cyl:N"`), or None when
    # they declared nothing at all.
    x_type = x_enc.option("type") if isinstance(x_enc, ChannelBase) else None
    y_type = y_enc.option("type") if isinstance(y_enc, ChannelBase) else None
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
        color_sort=color_sort,
        x_type=x_type,
        y_type=y_type,
    )
    prep = _PENDING_PREP.get(kind)
    if prep is not None:
        for step in prep.pre:
            step(ctx)
    transform_kwargs = ctx.transform_kwargs

    result = desugar_fn(x_field, y_field, **transform_kwargs)
    # Every layer this desugar emits (or, when it lowers to one unlayered
    # mark, the chart itself) names the public entry the caller invoked, and
    # so does every warning and refusal raised about it here: what the
    # desugar degraded, and a catch-all style field no emitted mark draws
    # (size= included), so a figure function forwarding style to a composite
    # that draws none of it refuses in its own voice.
    for note in result.degraded:
        warnings.warn(f"{entry}: {note}", stacklevel=user_stacklevel())
    if style_dict:
        refuse_style_no_layer_draws(
            entry,
            style_dict,
            [lyr.mark for lyr in result.layers] if result.layers is not None else [result.mark],
        )
    new = chart._clone()
    new._pending_stat_mark = None

    # Keep the prior primitive mark as layer(s). On an already-layered chart
    # those are its existing layers, each with its own style: the chart's
    # `_mark`/`_mark_kwargs` there are only layer 0's, copied up by the clone
    # that built it (spec §4.8).
    _prior_mark = chart._pending_stat_mark.prior_mark
    _prior_layers: list = []
    if _prior_mark is not None and _prior_mark in _FLAT_MARK_METHODS:
        if chart._layers:
            _prior_layers = list(chart._layers)
        else:
            # A figure function's voice covers the primitive it drew before
            # this composite too; a plain `mark_point().mark_smooth()` chain's
            # point keeps its own.
            _prior_layers = [
                replace(
                    _build_prior_layer(
                        _prior_mark,
                        chart._encoding,
                        chart._mark_kwargs,
                        chart._position,
                    ),
                    entry=chart._pending_stat_mark.entry,
                )
            ]

    # Layered mode: result.layers is set.
    if result.layers is not None:
        new._transforms = list(chart._transforms) + list(result.transforms)
        # Aim the caller's per-layer `layer_kwargs=` at the layers they named,
        # THEN apply the chart-level catch-all to every emitted layer, on
        # `catch_all_over`'s terms: it overrides a desugar-set default, but
        # not a key the layer records in `caller_style_keys` -- which is what
        # `_aim_layer_kwargs` just recorded for the keys aimed here.  The
        # prior primitive layer is left untouched.
        _refuse_undrawn_layer_kwargs(
            layer_kwargs, {lyr.name for lyr in result.layers if lyr.name is not None}, entry
        )
        emitted = [
            replace(
                apply_catch_all_style(_aim_layer_kwargs(lyr, layer_kwargs, entry), style_dict),
                entry=entry,
            )
            for lyr in result.layers
        ]
        # Scale-through-desugar propagation (design spec §6, GH #45
        # prerequisite): reattach any chart-level explicit x/y scale the
        # desugar's fresh per-layer encodings would otherwise drop.
        emitted = [_propagate_layer_scale(chart._encoding, lyr) for lyr in emitted]
        titles = chart._pending_stat_mark.legend_titles
        if titles:
            emitted = [
                replace(lyr, encoding=titled_encoding(lyr.encoding, titles)) for lyr in emitted
            ]
        new._layers = _prior_layers + emitted
        new._mark = None  # signals layered mode in to_spec
        return new

    # Single-mark mode.
    mark = result.mark
    transforms = result.transforms
    remap = result.remap
    # The single-mark twin of the per-layer aiming above: a result that
    # declares WHICH catalogued layer its one mark is can be styled through
    # the same `layer_kwargs=` door, and the keys aimed there are recorded as
    # the caller's so the chart-level catch-all below cannot override them.
    _refuse_undrawn_layer_kwargs(
        layer_kwargs, {result.layer_name} if result.layer_name is not None else set(), entry
    )
    single_mark_kwargs, single_caller_keys = _aim_single_mark_kwargs(result, layer_kwargs, entry)
    if _prior_layers:
        if result.data is not None:
            # No shipped desugar produces this combination today (the only
            # `.data`-producing desugar, `desugar_function`, never runs with a
            # prior_mark -- `Chart.mark_function` calls it directly, bypassing
            # this dispatcher entirely). Guarded rather than silently dropped:
            # a `_Layer` has no per-layer raw-data slot, so there is no
            # correct place to attach a synthetic replacement table here
            # without silently discarding either it or the prior layer's own
            # data -- adding that slot is required before a future desugar
            # can combine the two.
            raise ValueError(
                f"{kind!r} desugar returned both `data` (a synthetic "
                "replacement table) and ran alongside a prior primitive-mark "
                "layer; MarkDesugarResult.data has no defined meaning in "
                "that combination (a _Layer has no per-layer raw-data slot)."
            )
        smooth_enc = dict(chart._encoding)
        if remap:
            _apply_remap(smooth_enc, remap, orig_encoding=chart._encoding)
        smooth_enc = _propagate_positional_scale(chart._encoding, smooth_enc)
        smooth_enc = titled_encoding(smooth_enc, chart._pending_stat_mark.legend_titles)
        # Style applies to the composite's emitted smooth layer only; the prior
        # scatter layer keeps its own mark_kwargs. Which side wins on a shared
        # key is `catch_all_over`'s call, from the caller-named keys.
        smooth_layer = _Layer(
            mark=mark,
            encoding=smooth_enc,
            mark_kwargs=_merge_desugar_mark_kwargs(
                single_mark_kwargs, style_dict, single_caller_keys
            ),
            data_source="smooth",
            entry=entry,
        )
        new._transforms = list(chart._transforms) + list(transforms)
        new._layers = _prior_layers + [smooth_layer]
        new._mark = None
        return new
    new._mark = mark
    new._entry = entry
    new._transforms = list(chart._transforms) + list(transforms)
    # Merge the result's resolved mark_kwargs with the caller's catch-all
    # style kwargs -- on `catch_all_over`'s terms, from the caller-named
    # keys, not a fixed winner -- then
    # merge THAT onto `new._mark_kwargs` -- which `_clone()` already
    # inherited from `chart._mark_kwargs` -- rather than replacing it
    # outright. Mirrors `Chart.mark_function`'s own direct-path merge
    # (`{**new._mark_kwargs, **line_mark_kwargs}`,
    # `marks/_chart_methods_statistical.py`): a chart that already carries
    # its own mark kwargs (e.g. `.mark_line(stroke_width=7)` called before a
    # dispatcher-routed desugar) must keep them, not lose them to whatever
    # this desugar/caller combination computed (design review cycle-2 S2).
    # Only touches `new._mark_kwargs` when there is something new to fold
    # in, so output stays byte-identical when neither source has anything.
    merged_mark_kwargs = _merge_desugar_mark_kwargs(
        single_mark_kwargs, style_dict, single_caller_keys
    )
    if merged_mark_kwargs:
        new._mark_kwargs = {**new._mark_kwargs, **merged_mark_kwargs}
    if result.data is not None:
        new._data = result.data
    if result.position is not None:
        new._position = result.position
    if remap:
        _apply_remap(new._encoding, remap, orig_encoding=chart._encoding)
    new._encoding = _propagate_positional_scale(chart._encoding, new._encoding)
    new._encoding = titled_encoding(new._encoding, chart._pending_stat_mark.legend_titles)
    return new
