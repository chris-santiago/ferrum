"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional, Union

from ferrum._coerce import to_arrow_table
from ferrum._layer import _Layer, _PendingMark
from ferrum._configure_mixin import ConfigureMixin
from ferrum._marks_diagnostic import DiagnosticMarksMixin
from ferrum._marks_statistical import StatisticalMarksMixin
from ferrum._render import _RenderMixin
from ferrum._shorthand import parse_shorthand
from ferrum._spec_view import _SpecView
from ferrum.encoding.base import ChannelBase, _PendingAggregate
from ferrum.marks.base import MarkBase
from ferrum.marks.statistical import _build_prior_layer


_PRIMITIVE_MARKS = frozenset(["point", "line", "bar", "area", "rule", "text", "tick", "rect"])

# Marks whose Rust renderers consume the ``__stack_y_base__`` column produced
# by ``apply_stack`` (see ``crates/ferrum-core/src/render/marks/bar.rs`` and
# ``area.rs``).  Every other mark type silently drops stacked data, so when
# ``stack=`` is set on a non-stackable mark we warn and strip it before
# forwarding to the Rust layer.
_STACKABLE_MARKS = frozenset(["bar", "area"])


def _strip_unstackable(d: dict, mark: str | None) -> None:
    """Strip an unsupported ``stack=`` from a non-stackable mark's encoding dict.

    Only ``bar``/``area`` consume the ``__stack_y_base__`` column emitted by
    ``apply_stack``; every other mark silently drops its marks when the stacking
    path runs. When ``stack=`` is requested on such a mark we pop it (so the mark
    renders unstacked) and emit a one-time ``UserWarning`` naming the dropped
    field. A falsy ``stack`` (``None``/``False``) means "do not stack" and is a
    no-op, so it is left untouched. Shared by the single-chart and layered paths.
    """
    if d.get("stack") and mark not in _STACKABLE_MARKS:
        from ferrum._warn import warn_once

        stack_val = d.pop("stack")
        mark_label = f"mark_{mark}" if mark else "this mark"
        warn_once(
            "encoding",
            f"stack_on_{mark}",
            message=(
                f"stack={stack_val!r} is not supported by {mark_label} and was ignored. "
                "Stacking is only honored by mark_bar and mark_area."
            ),
        )


# Channels honored by the renderer at to_spec() time. Other channels in
# resolved._encoding (Stroke, Fill, Tooltip, etc.) are stored on the spec
# but ignored at render time; ferrum-spec.md §3.2 promises a one-time
# UserWarning per channel when this happens.
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
)
# Channels that are silently accepted but produce no visual encoding in the
# current static SVG renderer.  They are handled by special-case logic in
# to_spec() (alias to another channel, inject into mark_style, or simply
# stored for future interactive rendering).  No warning emitted.
_SILENT_CHANNELS = frozenset(
    (
        "fill",  # alias → color encoding
        "stroke",  # alias → color encoding or mark_style.stroke
        # fill_opacity is intentionally NOT listed here: it is wired through Rust
        # mark renderers as a per-element SVG fill-opacity attribute.
        "detail",  # injected into mark_style.detail
        "key",  # stored for future interactive/animated rendering
        "x_error",  # used through composite mark desugar (mark_errorbar)
        "y_error",  # used through composite mark desugar (mark_errorbar)
        "x_error2",  # used through composite mark desugar (mark_errorbar)
        "y_error2",  # used through composite mark desugar (mark_errorbar)
    )
)
# Polar channels raise NotImplementedError when a chart is actually rendered
# with them, rather than emitting a misleading "not yet rendered" warning.
_POLAR_CHANNELS = frozenset(("theta", "radius", "theta2", "radius2"))
# Facet channels have a separate code path through resolved._facet — no
# silent-drop, no warn.
_FACET_CHANNELS = frozenset(("facet", "facet_row", "facet_col"))

_logger = logging.getLogger(__name__)


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
        field, op, order = _SHORTHANDS[sort]
        if field is None:
            return sort  # cannot resolve without a field name; leave as-is
        return {"field": field, "op": op, "order": order}
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


@dataclass(frozen=True)
class _Facet:
    """Internal facet spec — frozen dataclass replacing the legacy dict shape.

    ``mode_kind`` is the tagged-union discriminator: ``"wrap"`` uses ``field``;
    ``"grid"`` uses ``row`` and ``col``. Other fields are sizing hints honored
    by ``_build_facet_dict()`` when serialising to the Rust ``FacetSpec``.

    ``resolve`` carries per-channel scale-resolution modes set via
    ``Chart.share_scale()``.  Keys are channel names (``"x"``, ``"y"``);
    values are ``"shared"`` or ``"independent"``.  Absent keys default to
    ``"shared"`` in the Rust layer.
    """

    mode_kind: str
    field: Optional[str] = None
    row: Optional[str] = None
    col: Optional[str] = None
    ncols: Optional[int] = None
    nrows: Optional[int] = None
    resolve: Optional[dict] = None


from ferrum.encoding import _channel_class_map, _channel_class_for, _apply_channel_aliases


class _NamedTransform:
    """Pairs a PyO3 transform object with an explicit name for chart-level serialization.

    When a single-mark chart's transforms are promoted to chart level during
    ``+`` composition, we need them to be *named* in the Rust pipeline so that
    ``FINAL_OUTPUT_KEY`` remains the original input batch (named transforms do
    not advance the unnamed chain).  The corresponding ``_Layer`` sets
    ``data_source`` to the same name so it reads the correct output.
    """

    __slots__ = ("transform", "name")

    def __init__(self, transform: object, name: str) -> None:
        self.transform = transform
        self.name = name

    def _inner_eq(self, other: object) -> bool:
        inner = other.transform if isinstance(other, _NamedTransform) else other
        return bool(self.transform == inner)


from ferrum.composition import (
    _expand_layers,
    _merge_top_transforms,
    _promote_layer_color,
    _warn_on_layer_conflicts,
)


def _rename_encoding_fields(encoding: dict, renames: dict[str, str]) -> dict:
    """Return a copy of *encoding* with field names replaced per *renames*."""
    from ferrum.encoding.base import ChannelBase

    out = {}
    for ch, val in encoding.items():
        if isinstance(val, ChannelBase):
            if val.field in renames:
                import copy

                val = copy.copy(val)
                val.field = renames[val.field]
        elif isinstance(val, str):
            bare, _, suffix = val.partition(":")
            if bare in renames:
                val = renames[bare] + (":" + suffix if suffix else "")
        out[ch] = val
    return out


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


def _to_polars(data):
    """Convert arbitrary chart data to a polars DataFrame.

    Used by ``__add__`` to null-pad merge when two charts have
    different data.
    """
    import polars as pl

    if isinstance(data, pl.DataFrame):
        return data
    return pl.from_arrow(to_arrow_table(data))


def _coalesce_facet_rhs_columns(chart: "Chart") -> "Chart":
    """Coalesce renamed RHS copies of facet fields back into the primary column.

    When ``Chart.__add__`` merges two DataFrames with overlapping column names,
    it renames the RHS columns to ``"{col}__rhs_{hex}"``.  If one of those
    columns is a facet field, the RHS layer rows end up with ``null`` in the
    facet column and are dropped by the facet partitioner.

    This function detects such renamed copies and coalesces them so the facet
    column is populated for all layers' rows.  Only columns that are facet
    fields are coalesced; other renamed columns are left alone so genuinely
    different columns (with different semantics) are not silently merged.

    Returns the (possibly mutated) chart.  The chart's data is updated in-place
    on the clone.
    """
    if chart._facet is None or chart._data is None:
        return chart

    import polars as pl

    if not isinstance(chart._data, pl.DataFrame):
        return chart

    f = chart._facet
    # Collect facet fields that are used for partitioning.
    facet_fields = [ff for ff in (f.field, f.col, f.row) if ff is not None]

    df = chart._data
    modified = False
    for facet_col in facet_fields:
        if facet_col not in df.columns:
            continue
        # Find any "__rhs_" renamed copies of this facet column.
        rhs_prefix = f"{facet_col}__rhs_"
        rhs_copies = [c for c in df.columns if c.startswith(rhs_prefix)]
        if not rhs_copies:
            continue
        # Coalesce: fill nulls in the primary facet column from RHS copies.
        # The primary column has the LHS values; RHS rows have null there.
        expr = pl.col(facet_col)
        for rhs_col in rhs_copies:
            expr = pl.coalesce([expr, pl.col(rhs_col)])
        df = df.with_columns(expr.alias(facet_col))
        modified = True

    if modified:
        chart._data = df

    return chart


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


class Chart(ConfigureMixin, StatisticalMarksMixin, DiagnosticMarksMixin, _RenderMixin):
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
    width : int or "container", optional
        Chart width in pixels, or ``"container"`` to fill the parent element.
        Default is ``600``.
    height : int or "container", optional
        Chart height in pixels.  Default is ``400``.
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
    >>> svg = chart.show_svg()
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
        "_selections",
        "_conditionals",
        "_render_config",
        "_configure",  # list[Configure] — accumulated configure layers
        "_annotations",  # list[Annotate] — accumulated annotation layers
        "_structural",  # list — accumulated structural features (SecondaryY, BreakAxis, Inset)
        "_overrides",  # dict — spec-path override kwargs
        "_annotation_primitive",  # optional annotation primitive for annotate_* helpers
        "_mark_zero",  # bool — False when mark_bar(zero=False) suppresses the y zero-anchor
        "_figure_caption",  # str or None — figure-level caption rendered below the SVG
    )

    def __init__(
        self,
        data: Any = None,
        *,
        width: Optional[Union[int, str]] = None,
        height: Optional[Union[int, str]] = None,
        title: "Optional[Union[str, 'Title']]" = None,
        description: Optional[str] = None,
    ) -> None:
        self._data = data
        self._mark = None
        self._mark_kwargs = {}
        self._encoding: dict = {}
        self._transforms: list = []
        self._facet = None
        self._coord = None
        self._theme = None
        self._layers: Optional[list] = None
        self._width = width
        self._height = height
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
        self._selections: list = []
        self._conditionals: list = []
        self._render_config = None
        self._configure: list = []
        self._annotations: list = []
        self._structural: list = []
        self._overrides: dict = {}
        self._annotation_primitive = None
        self._mark_zero: bool = True
        self._figure_caption: Optional[str] = None

    def _clone(self) -> "Chart":
        new = object.__new__(Chart)
        for slot in self.__slots__:
            val = getattr(self, slot)
            if isinstance(val, (list, dict)):
                setattr(new, slot, copy.copy(val))
            else:
                setattr(new, slot, val)
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
        """
        if self._pending_stat_mark is None:
            return self
        kind = self._pending_stat_mark.kind
        kwargs = self._pending_stat_mark.kwargs
        desugar_fn = self._pending_stat_mark.desugar_fn
        x_enc = self._encoding.get("x")
        y_enc = self._encoding.get("y")
        x_field = (
            (x_enc.field if isinstance(x_enc, ChannelBase) else x_enc)
            if x_enc is not None
            else None
        )
        y_field = (
            (y_enc.field if isinstance(y_enc, ChannelBase) else y_enc)
            if y_enc is not None
            else None
        )
        # If encodings still contain unresolved repeat placeholders, defer
        # desugaring until the RepeatChart substitutes concrete field names.
        from ferrum.repeat import _RepeatPlaceholder

        if isinstance(x_field, _RepeatPlaceholder) or isinstance(y_field, _RepeatPlaceholder):
            return self
        # Capture sort from the original channel objects before they are reduced
        # to bare field strings.  Composite desugars that build a categorical
        # positional axis (boxplot, boxen, errorbar, violin, swarm) receive these
        # so they can wrap the `cat` field in X(cat, sort=...) / Y(cat, sort=...)
        # and preserve the user's sort intent through the desugar expansion.
        #
        # Shorthand sorts ("-y", "y", "-x", "x") are resolved to explicit dicts
        # here using the original field names.  Composite desugars replace the
        # y-channel field with computed columns (e.g. lower_whisker, q1), so the
        # Rust renderer would otherwise sort by the wrong column.
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
        # Partition the user kwargs by the desugar signature: declared params
        # are transform kwargs; everything else is a constant mark-style
        # override.  ``style_dict`` is validated/normalised through MarkBase so
        # aliases (color→fill, alpha→opacity) resolve and unknown keys raise.
        transform_kwargs, style_kwargs = _split_style_kwargs(desugar_fn, kwargs)
        style_dict = MarkBase(kind, **style_kwargs).to_mark_kwargs_dict()
        # Ribbon needs y2 from the encoding — inject as a transform kwarg.
        if kind == "ribbon":
            y2_enc = self._encoding.get("y2")
            if y2_enc is not None:
                y2_field = y2_enc.field if isinstance(y2_enc, ChannelBase) else y2_enc
                transform_kwargs = {**transform_kwargs, "y2_field": y2_field}
        # Boxen: the LetterValue transform renames the groupby column to "group"
        # and drops the original value column from its per-depth output batches.
        # A field-based aggregate sort dict (e.g. {"field": "val", "op": "sum",
        # "order": "descending"}) therefore cannot resolve through the standard
        # layer-sort path — the value column is absent from the layer batch.
        # Instead, hand the aggregate op/order to the LetterValue transform via
        # its ``sort=`` kwarg: it reorders its emitted group rows so the
        # categorical axis domain falls out in first-appearance order, with no
        # explicit categorical layer sort needed.  Non-aggregate forms (explicit
        # list, "ascending"/"descending", None) still resolve via the standard
        # layer-sort path on the renamed ``group`` column, so they pass through
        # unchanged.
        if kind == "boxen":
            cat_is_y = bool(transform_kwargs.get("horizontal", False))
            cat_sort = y_sort if cat_is_y else x_sort
            if isinstance(cat_sort, dict) and "field" in cat_sort:
                transform_kwargs = {
                    **transform_kwargs,
                    "boxen_agg_sort": {
                        "op": cat_sort.get("op", "sum"),
                        "order": cat_sort.get("order", "ascending"),
                    },
                }
                # Drop the categorical layer sort for the aggregate case; the
                # transform's row order drives the domain.
                if cat_is_y:
                    y_sort = None
                else:
                    x_sort = None
        # Composites with a categorical positional axis: inject x_sort and y_sort
        # so the desugar function can forward sort to the categorical channel
        # encoding, enabling e.g. X("c:N", sort="-y") on mark_boxplot/violin/etc.
        #
        # errorbar always treats x as the categorical grouping axis (see
        # desugar_errorbar), so it only consumes x_sort; y_sort is never read
        # there and is deliberately not injected (R4: dead-param removal).
        if kind in ("boxplot", "boxen", "errorbar", "violin", "swarm"):
            if x_sort is not None:
                transform_kwargs = {**transform_kwargs, "x_sort": x_sort}
            if y_sort is not None and kind != "errorbar":
                transform_kwargs = {**transform_kwargs, "y_sort": y_sort}
        # density/histogram: auto-set groupby from color encoding when not explicit.
        if kind in ("density", "histogram") and "groupby" not in transform_kwargs:
            color_enc = self._encoding.get("color")
            if color_enc is not None:
                color_field = color_enc.field if isinstance(color_enc, ChannelBase) else color_enc
                if color_field:
                    transform_kwargs = {**transform_kwargs, "groupby": color_field}
        # hex: infer aggregate field from color encoding when not explicit.
        if (
            kind == "hex"
            and transform_kwargs.get("aggregate", "count") != "count"
            and transform_kwargs.get("field") is None
        ):
            color_enc = self._encoding.get("color")
            if color_enc is not None:
                color_field = color_enc.field if isinstance(color_enc, ChannelBase) else color_enc
                if color_field:
                    transform_kwargs = {**transform_kwargs, "field": color_field}
        # When mark_smooth was called on a chart that already had a primitive
        # mark (e.g. chart.mark_point().mark_smooth().encode(...)), preserve
        # the existing mark as a scatter layer. Force the Smooth transform
        # to be named so __final__ stays as the raw data for the scatter layer.
        _prior_mark = self._pending_stat_mark.prior_mark
        if _prior_mark is not None and kind == "smooth" and "name" not in transform_kwargs:
            transform_kwargs = {**transform_kwargs, "name": "smooth"}
        result = desugar_fn(x_field, y_field, **transform_kwargs)
        new = self._clone()
        new._pending_stat_mark = None

        # swarm: Rust swarm transform requires the value column to be Float64.
        # Cast it here where we have both encoding (value field name) and data.
        if kind == "swarm":
            try:
                import polars as pl

                _, _, value_field = _resolve_cat_axis(
                    transform_kwargs, x_field, y_field, x_sort, y_sort
                )
                if (
                    value_field
                    and new._data is not None
                    and isinstance(new._data, pl.DataFrame)
                    and value_field in new._data.columns
                    and new._data[value_field].dtype != pl.Float64
                ):
                    new._data = new._data.with_columns(pl.col(value_field).cast(pl.Float64))
            except (ImportError, TypeError, AttributeError, ValueError):
                pass

        # boxplot / boxen / violin: Rust box_stats / letter_value / violin transforms
        # require the value column to be Float64.  Cast integer columns here, where
        # we have both the encoding (value field name) and the data frame.
        #
        # Note: catplot orient="h" encodes x=numeric, y=categorical but does not
        # pass horizontal=True to mark_boxplot — CoordFlip handles the visual flip
        # at render time.  Instead of relying on the horizontal kwarg, we detect
        # the value field by dtype: cast whichever of x_field / y_field holds an
        # integer column (skipping string/categorical columns which cannot be cast).
        if kind in ("boxplot", "boxen", "violin"):
            try:
                import polars as pl

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
                if new._data is not None and isinstance(new._data, pl.DataFrame):
                    casts = []
                    for fld in (x_field, y_field):
                        if fld and fld in new._data.columns and new._data[fld].dtype in _INT_DTYPES:
                            casts.append(pl.col(fld).cast(pl.Float64))
                    if casts:
                        new._data = new._data.with_columns(casts)
            except (ImportError, TypeError, AttributeError, ValueError):
                pass

        # Build a scatter layer from the prior mark if present.
        _prior_layer = None
        if _prior_mark is not None and _prior_mark in _PRIMITIVE_MARKS:
            _prior_layer = _build_prior_layer(
                _prior_mark,
                self._encoding,
                self._mark_kwargs,
                self._position,
            )

        # Layered mode: result.layers is set.
        if result.layers is not None:
            new._transforms = list(self._transforms) + list(result.transforms)
            # Apply constant style to every emitted layer (user value wins over
            # any desugar-set per-layer default).  The prior primitive layer is
            # left untouched.
            emitted = [_apply_style_to_layer(lyr, style_dict) for lyr in result.layers]
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
            from ferrum._layer import _Layer as _SmoothLyr

            smooth_enc = dict(self._encoding)
            if remap:
                _apply_remap(smooth_enc, remap, orig_encoding=self._encoding)
            # Style applies to the composite's emitted smooth layer only; the
            # prior scatter layer keeps its own mark_kwargs.
            smooth_layer = _SmoothLyr(
                mark=mark,
                encoding=smooth_enc,
                mark_kwargs=(dict(style_dict) if style_dict else None),
                data_source="smooth",
            )
            new._transforms = list(self._transforms) + list(transforms)
            new._layers = [_prior_layer, smooth_layer]
            new._mark = None
            return new
        new._mark = mark
        new._transforms = list(self._transforms) + list(transforms)
        # Only override the cloned default when the user passed style kwargs, so
        # output stays byte-identical when none are given.
        if style_dict:
            new._mark_kwargs = style_dict
        if result.position is not None:
            new._position = result.position
        if remap:
            _apply_remap(new._encoding, remap, orig_encoding=self._encoding)
        return new

    # ---- Marks (primitives) ----

    def _set_mark(self, name: str, **kwargs: Any) -> "Chart":
        # Phase 9c -- pull `position=` out of kwargs and validate eligibility
        # before constructing the MarkBase (which would reject unknown kwargs).
        position = kwargs.pop("position", None)
        if position is not None:
            from ferrum.position import validate_position_eligibility

            validate_position_eligibility(name, position)
        m = MarkBase(name, **kwargs)
        new = self._clone()
        new._mark = name
        new._mark_kwargs = m.to_mark_kwargs_dict()
        new._position = position
        # S4: orient="horizontal" → coord flip (consumed Python-side).
        if m.orient_coord_flip():
            new._coord = "flip"
        # D3: zero=False suppresses the bar y-scale zero-anchor injection.
        # Consumed Python-side; not forwarded to the Rust renderer.
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
        if data_transform is not None and new._data is not None:
            try:
                import polars as pl

                if isinstance(new._data, pl.DataFrame):
                    new._data = data_transform(new._data)
            except ImportError:
                pass
        new._pending_stat_mark = _PendingMark(name, dict(kwargs), desugar_fn, prior_mark=prior_mark)
        new._position = position
        new._composite_kind = name
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
            Line width in pixels.  Default is ``2``.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        interpolate : str, optional
            Line interpolation: ``"linear"``, ``"monotone"``, ``"step"``,
            ``"step-before"``, ``"step-after"``, ``"basis"``, ``"cardinal"``.
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
        arguments are forwarded to :meth:`mark_point`, including aliases such
        as ``color`` and ``alpha``.

        Parameters
        ----------
        **kwargs
            Any keyword accepted by :meth:`mark_point` (``size``, ``fill``,
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
        arguments are forwarded to :meth:`mark_point`, including aliases such
        as ``color`` and ``alpha``.

        Parameters
        ----------
        **kwargs
            Any keyword accepted by :meth:`mark_point` (``size``, ``fill``,
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
            Area boundary interpolation: ``"linear"``, ``"monotone"``,
            ``"step"``, ``"basis"``, ``"cardinal"``.
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
            Dash pattern, e.g. ``"4,2"``.
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

        Requires a ``text`` encoding channel pointing at the column to display.

        Parameters
        ----------
        fill : str, optional
            Text colour override.
        font_size : float, optional
            Font size in points.  Default is ``11``.
        font_weight : str or int, optional
            CSS font-weight, e.g. ``"bold"``, ``400``, ``700``.
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
        position : Position, optional
            Position adjustment.

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

        Parameters
        ----------
        stroke : str, optional
            Tick colour override.
        stroke_width : float, optional
            Tick stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        band_size : float, optional
            Tick length as a fraction of band width (0–1, default 0.3).
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
        stroke_dash : list of float, optional
            Dash pattern (alternating on/off pixel lengths).
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

        Requires ``Chart.coord(fm.CoordPolar(theta="x"))`` to be set.
        The theta-mapped encoding channel (``x`` by default) determines
        each slice's angular sweep proportional to its value.

        Parameters
        ----------
        **kwargs
            Mark style overrides: ``color``, ``opacity``, ``stroke_width``, etc.

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
            Mark style overrides: ``width``, ``height``, ``opacity``, etc.

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
        ``Chart.coord(fm.CoordGeo(projection="mercator"))``.

        Parameters
        ----------
        **kwargs
            Mark style overrides: ``color``, ``opacity``, ``stroke_width``, etc.

        Examples
        --------
        >>> fm.Chart(geojson_data).mark_geoshape().coord(
        ...     fm.CoordGeo(projection="equal_earth")
        ... )
        """
        return self._set_mark("geoshape", **kwargs)

    def mark_label(self, **kwargs) -> "Chart":
        """Render positioned text labels near data points with collision avoidance.

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
            Additional mark style overrides (``fill``, ``opacity``,
            ``font_weight``, etc.).

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

        Returns
        -------
        Chart
            New ``Chart`` with updated encoding channels.

        Raises
        ------
        ValueError
            If an unknown channel name is passed.
        TypeError
            If a value is not a string, channel instance, or Repeat placeholder.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "b", "a"]})
        >>> fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="c:N")
        Chart(mark='point', encoding=['x', 'y', 'color'])
        """
        new = self._clone()
        for name, value in channels.items():
            cls = _channel_class_for(name)
            if cls is None:
                raise ValueError(f"unknown encoding channel: {name!r}")

            if isinstance(value, ChannelBase):
                channel = value
            elif isinstance(value, str):
                field, type_, agg = parse_shorthand(value)
                kw = {}
                if type_:
                    kw["type"] = type_
                if agg:
                    kw["aggregate"] = agg
                channel = cls(field, **kw)
            else:
                # Phase 9: accept Repeat sentinels (Repeat.column / .row / .layer).
                from ferrum.repeat import _RepeatPlaceholder

                if isinstance(value, _RepeatPlaceholder):
                    channel = cls(value)
                else:
                    raise TypeError(
                        f"encode({name}=...) expects str, {cls.__name__} instance, "
                        f"or Repeat placeholder; got {type(value).__name__}"
                    )

            new._encoding[name] = channel
            new._transforms.extend(channel.to_implicit_transforms())

        # Synthesize _facet whenever facet encoding channels are passed in this
        # call. This wires encode(facet_col=...) / encode(facet_row=...) /
        # encode(facet=...) to the same _facet path that .facet() uses.
        if any(name in _FACET_CHANNELS for name in channels):
            facet_enc = new._encoding.get("facet")
            col_enc = new._encoding.get("facet_col")
            row_enc = new._encoding.get("facet_row")
            # Preserve ncols/nrows from any existing _facet (e.g. a prior .facet() call).
            existing = new._facet
            ncols = existing.ncols if existing is not None else None
            nrows = existing.nrows if existing is not None else None
            if facet_enc is not None:
                new._facet = _Facet(
                    mode_kind="wrap", field=facet_enc.field, ncols=ncols, nrows=nrows
                )
            elif col_enc is not None and row_enc is not None:
                new._facet = _Facet(
                    mode_kind="grid", col=col_enc.field, row=row_enc.field, ncols=ncols, nrows=nrows
                )
            elif col_enc is not None:
                new._facet = _Facet(mode_kind="wrap", field=col_enc.field, ncols=ncols, nrows=nrows)
            elif row_enc is not None:
                new._facet = _Facet(mode_kind="wrap", field=row_enc.field, ncols=ncols, nrows=nrows)

        return new

    def layer(self, *layers) -> "Chart":
        """Add one or more layer objects to this chart.

        Accepts both public ``Layer`` instances (user-facing API) and
        internal ``_Layer`` instances (used by ferrum internals).

        When a ``Layer(data=df, ...)`` has its own ``data`` attribute, that
        data is merged with the chart's existing data via diagonal concatenation
        (same strategy as the ``+`` operator). Each layer's encoding references
        only its own columns; null-padded rows in the merged batch are invisible
        to mark renderers that skip null values.

        Parameters
        ----------
        *layers : Layer or _Layer
            Layer objects to append. Public ``Layer`` instances may carry an
            independent ``data=`` DataFrame.

        Returns
        -------
        Chart
            This chart with the new layers appended.
        """
        from ferrum.layer import Layer as PublicLayer

        resolved = self._resolve_pending()
        new = resolved._clone()
        existing, _ = _expand_layers(new)
        converted = []
        for ly in layers:
            if isinstance(ly, _Layer):
                converted.append(ly)
            elif isinstance(ly, PublicLayer):
                if ly.data is not None:
                    # Merge the layer's data with the chart's data via diagonal
                    # concatenation (mirrors the __add__ strategy for independent data).
                    # When the chart has no data yet (data=None), the layer's data
                    # becomes the chart's data.
                    import polars as pl

                    layer_df = _to_polars(ly.data)
                    if new._data is None:
                        new._data = layer_df
                    else:
                        try:
                            chart_df = _to_polars(new._data)
                            new._data = pl.concat([chart_df, layer_df], how="diagonal")
                        except (TypeError, ValueError):
                            new._data = layer_df
                converted.append(
                    _Layer(
                        name=ly.name,
                        mark=ly.mark,
                        encoding=dict(ly.encoding),
                        transforms=list(ly.transforms),
                        mark_kwargs=dict(ly.mark_kwargs) if ly.mark_kwargs else None,
                    )
                )
            else:
                raise TypeError(
                    f"layer() expects Layer or _Layer instances; got {type(ly).__name__}"
                )
        new._layers = existing + converted
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

        Parameters
        ----------
        *transforms
            One or more transform objects (e.g. ``fm.Filter(...)``,
            ``fm.Aggregate(...)``, ``fm.Sort(...)``, ``fm.Window(...)``).
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
        ...     fm.Filter("datum.x > 1")
        ... )
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        new._transforms = list(self._transforms) + list(transforms)
        return new

    # ---- Composition operators ----

    def __add__(self, other: Any) -> "Chart":
        """Compose a chart with another chart, configuration, annotation, or structural feature.

        Dispatches on the type of *other*:

        - ``Chart`` — overlay as a multi-layer composite with shared scales
        - ``Configure`` — append chart-level configuration
        - ``Annotate`` or annotation primitive — append annotation layer
        - ``SecondaryY``, ``BreakAxis``, ``Inset`` — append structural feature

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

        # Dispatch: structural features
        if isinstance(other, (SecondaryY, BreakAxis, Inset)):
            new = self._clone()
            new._structural = new._structural + [other]
            return new

        if not isinstance(other, Chart):
            return NotImplemented
        # Resolve pending statistical marks before snapshotting encoding dicts.
        lhs = self._resolve_pending()
        rhs = other._resolve_pending()
        new = lhs._clone()
        lhs_layers, _ = _expand_layers(lhs)
        rhs_layers, rhs_top_xforms = _expand_layers(rhs)

        # When the LHS is an annotate_* helper chart, its annotation primitive
        # fully describes the visual element — the mark layers must be excluded
        # (same logic as the RHS path below). Use RHS layers only.
        if lhs._annotation_primitive is not None:
            if rhs._annotation_primitive is not None:
                from ferrum._layer import _Layer as _CarrierLayer

                carrier = _CarrierLayer(
                    mark="point",
                    encoding=dict(lhs._encoding),
                    mark_kwargs={"opacity": 0, "size": 0},
                )
                new._layers = [carrier]
                new._annotations = new._annotations + [
                    Annotate(lhs._annotation_primitive),
                    Annotate(rhs._annotation_primitive),
                ]
            else:
                new = rhs._clone()
                new._layers = rhs_layers
                new._annotations = new._annotations + [Annotate(lhs._annotation_primitive)]
            new._annotation_primitive = None
            _warn_on_layer_conflicts(lhs, rhs)
            return new

        # When the RHS is an annotate_* helper chart, its annotation primitive
        # fully describes the visual element for both SVG and interactive
        # rendering.  The mark layers inside the annotate_* chart (mark_rule,
        # mark_rect, mark_line, mark_text) must be excluded — they would
        # produce a duplicate rendering alongside the annotation primitive.
        # Data merging and transform routing for the RHS are also skipped since
        # the annotation data is not needed by any layer.
        if rhs._annotation_primitive is not None:
            new._layers = lhs_layers  # LHS layers only — no RHS mark layers
            new._annotations = new._annotations + [Annotate(rhs._annotation_primitive)]
            _warn_on_layer_conflicts(lhs, rhs)
            return new

        # Data merging: when data differs, decide whether to diagonal-concat
        # or route the RHS through a named Identity transform.
        if not self._shares_data_with(other):
            import polars as pl

            lhs_df = _to_polars(self._data)
            rhs_df = _to_polars(other._data)
            overlap = set(lhs_df.columns) & set(rhs_df.columns)

            if overlap:
                # Column names collide — rename RHS columns so diagonal
                # concat produces disjoint null-padded columns, AND route
                # RHS layers through a named Identity transform with
                # data_source so inherit_non_positional prevents the Rust
                # renderer from injecting chart-level positional channels.
                from ferrum._core import PyIdentity as _RustIdentity
                from dataclasses import replace as _dc_replace

                suffix = f"__rhs_{id(other) & 0xFFFFFFFF:08x}"
                col_renames = {c: f"{c}{suffix}" for c in overlap}
                rhs_df = rhs_df.rename(col_renames)

                auto_name = f"_ident_{id(other) & 0xFFFFFFFF:08x}"
                identity_xform = _NamedTransform(_RustIdentity(auto_name), auto_name)
                rhs_layers = [
                    _dc_replace(
                        l,
                        encoding=_rename_encoding_fields(l.encoding, col_renames),
                        data_source=auto_name,
                    )
                    for l in rhs_layers
                ]
                new._data = pl.concat([lhs_df, rhs_df], how="diagonal")
                _merge_top_transforms(new, [identity_xform])
            else:
                new._data = pl.concat([lhs_df, rhs_df], how="diagonal")

        # Named-transform routing for RHS transforms (smooth, etc.):
        # wrap as _NamedTransform so FINAL_OUTPUT_KEY stays the LHS batch.
        if not new._transforms and rhs_top_xforms:
            auto_name = f"_auto_{id(rhs_top_xforms[-1]) & 0xFFFFFFFF:08x}"
            named_xforms = [_NamedTransform(t, auto_name) for t in rhs_top_xforms]
            from dataclasses import replace as _dc_replace

            rhs_layers = [_dc_replace(l, data_source=auto_name) for l in rhs_layers]
            _merge_top_transforms(new, named_xforms)
        else:
            _merge_top_transforms(new, rhs_top_xforms)

        new._layers = lhs_layers + rhs_layers
        # D2: when the LHS has no color encoding, promote the first layer's
        # color encoding to chart level so the Rust renderer can build the
        # correct color scale.  Without this, build_color_scale sees
        # spec.encoding.color = None and returns no color scale, causing every
        # layer with a color encoding to fall back to the theme default color.
        _promote_layer_color(new)
        # Merge RHS selections and conditionals into the layered chart
        # so interactive features from all layers are preserved.
        if rhs._selections:
            existing_names = {s.name for s in new._selections}
            for s in rhs._selections:
                if s.name not in existing_names:
                    new._selections.append(s)
        if rhs._conditionals:
            new._conditionals.extend(rhs._conditionals)
        # Merge RHS configure/annotation/structural/override slots.
        if rhs._configure:
            new._configure = new._configure + rhs._configure
        if rhs._annotations:
            new._annotations = new._annotations + rhs._annotations
        if rhs._structural:
            new._structural = new._structural + rhs._structural
        if rhs._overrides:
            new._overrides = {**new._overrides, **rhs._overrides}
        _warn_on_layer_conflicts(lhs, rhs)
        return new

    def _shares_data_with(self, other: "Chart") -> bool:
        # Identity first (fast path), then Arrow value equality. Used by __add__
        # to decide overlay-vs-concat without forcing a coerce when the python
        # objects are the same DataFrame.
        if self._data is other._data:
            return True
        try:
            return to_arrow_table(self._data).equals(to_arrow_table(other._data))
        except Exception:
            return False

    def __or__(self, other: "Chart") -> "HConcatChart":
        """Place two charts side-by-side (horizontal concatenation).

        Produces an ``HConcatChart`` that renders both charts at the same height
        in adjacent columns.  Use ``|`` to build multi-panel layouts.

        Parameters
        ----------
        other : Chart
            The chart to place to the right.

        Returns
        -------
        HConcatChart
            Horizontally concatenated composite chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> left = fm.Chart(df).mark_point().encode(x="x", y="y")
        >>> right = fm.Chart(df).mark_bar().encode(x="x", y="y")
        >>> left | right
        HConcatChart(n=2)
        """
        from ferrum.composition import HConcatChart

        return HConcatChart([self, other])

    def __and__(self, other: "Chart") -> "VConcatChart":
        """Stack two charts vertically (vertical concatenation).

        Produces a ``VConcatChart`` that renders both charts stacked on top of
        each other in the same column.  Use ``&`` to build vertically-stacked
        panel layouts.

        Parameters
        ----------
        other : Chart
            The chart to place below.

        Returns
        -------
        VConcatChart
            Vertically concatenated composite chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> top = fm.Chart(df).mark_point().encode(x="x", y="y")
        >>> bottom = fm.Chart(df).mark_bar().encode(x="x", y="y")
        >>> top & bottom
        VConcatChart(n=2)
        """
        from ferrum.composition import VConcatChart

        return VConcatChart([self, other])

    # ---- Facet / coord / theme ----

    def facet(
        self,
        field: Optional[str] = None,
        *,
        row: Optional[str] = None,
        col: Optional[str] = None,
        ncols: Optional[int] = None,
        nrows: Optional[int] = None,
    ) -> "Chart":
        """Facet this chart into small multiples by a field.

        Two modes are supported:

        - **Wrap** — a single field is wrapped across columns (rows auto).
          Pass ``field=`` or equivalently ``col=`` alone.
        - **Grid** — two fields define row × column layout.  Pass both
          ``row=`` and ``col=``.

        Parameters
        ----------
        field : str, optional
            Column name used for wrap-mode faceting.  Mutually exclusive with
            using both ``row`` and ``col`` together.
        row : str, optional
            Column name for the row dimension (grid mode).  When used alone,
            behaves as wrap mode on the row axis.
        col : str, optional
            Column name for the column dimension (wrap or grid mode).
        ncols : int or None, optional
            Maximum number of columns in wrap mode.  ``None`` lets the renderer
            choose.
        nrows : int or None, optional
            Maximum number of rows.  ``None`` lets the renderer choose.

        Returns
        -------
        Chart
            New ``Chart`` with faceting configured.

        Raises
        ------
        ValueError
            If neither ``field``, ``row``, nor ``col`` is provided.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1,2,3]*3, "y": [4,5,6]*3,
        ...                    "g": ["a","a","b","b","c","c","a","b","c"]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").facet(col="g", ncols=2)
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        if field is not None:
            new._facet = _Facet(
                mode_kind="wrap",
                field=field,
                ncols=ncols,
                nrows=nrows,
            )
        elif row is not None and col is not None:
            new._facet = _Facet(
                mode_kind="grid",
                row=row,
                col=col,
                nrows=nrows,
                ncols=ncols,
            )
        elif col is not None:
            new._facet = _Facet(
                mode_kind="wrap",
                field=col,
                ncols=ncols,
                nrows=nrows,
            )
        elif row is not None:
            new._facet = _Facet(
                mode_kind="wrap",
                field=row,
                nrows=nrows,
                ncols=ncols,
            )
        else:
            raise ValueError("facet() requires either `field=`, or `row=`/`col=`")
        # 2c fix: coalesce any renamed RHS copies of the facet field(s) back
        # into the primary column.  When Chart.__add__ merges two DataFrames
        # with overlapping column names, it renames the RHS columns to
        # "{col}__rhs_{hex}".  If one of those columns is the facet field,
        # the RHS layer's rows end up with null in the facet column and are
        # dropped by the facet partitioner.  We detect and coalesce here so
        # all layers' rows carry the facet field value.
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
        _valid = ("shared", "independent")
        if x is not None and x not in _valid:
            raise ValueError(f"share_scale: x={x!r}; expected 'shared' or 'independent'")
        if y is not None and y not in _valid:
            raise ValueError(f"share_scale: y={y!r}; expected 'shared' or 'independent'")
        # Build updated resolve dict from existing + new values.
        existing_resolve = self._facet.resolve or {}
        new_resolve = dict(existing_resolve)
        if x is not None:
            new_resolve["x"] = x
        if y is not None:
            new_resolve["y"] = y
        new = self._clone()
        new._facet = replace(self._facet, resolve=new_resolve if new_resolve else None)
        return new

    def theme(self, theme: Any) -> "Chart":
        """Attach a ``Theme`` to this chart, overriding the process-level default.

        Per-chart theme always wins over ``ferrum.set_default_theme()``.
        Theme objects are immutable value classes — modifying the original
        ``Theme`` after calling ``.theme()`` has no effect on the chart.

        Parameters
        ----------
        theme : Theme
            A ``ferrum.Theme`` instance (e.g. ``fm.themes.dark()``,
            ``fm.Theme(background="white", font="serif")``).

        Returns
        -------
        Chart
            New ``Chart`` with the specified theme attached.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> dark = fm.themes.dark()
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

        Overrides are merged into :attr:`_overrides` with later calls winning
        on key conflicts. The ``override()`` call never mutates the receiver.

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
        new = self._clone()
        new._overrides = {**new._overrides, **kwargs}
        return new

    def coord(self, coord: Any) -> "Chart":
        """Set the coordinate system for this chart.

        Currently only ``CoordFlip`` is supported (swaps x and y axes).

        Parameters
        ----------
        coord : CoordFlip
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

    @staticmethod
    def _transforms_to_json_list(transforms: list) -> list:
        """Serialize a list of Python transform objects to JSON-safe dicts.

        ``coerce_layers`` in Rust calls ``json.dumps()`` on each layer dict, so
        PyO3 transform objects must be converted to plain dicts first.  We do
        this by round-tripping through ``ChartSpec.to_json()``.
        """
        if not transforms:
            return []
        from ferrum import ChartSpec

        # Build a minimal spec with the transforms; extract the "transforms" array.
        dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=transforms)
        parsed = json.loads(dummy.to_json())
        return parsed.get("transforms", [])

    @staticmethod
    def _transforms_to_json_list_named(transforms: list) -> list:
        """Like ``_transforms_to_json_list`` but handles ``_NamedTransform`` wrappers
        and plain dicts (Phase 12 data transforms).

        For each ``_NamedTransform``, serializes the inner transform and injects
        the ``name`` field.  Plain dicts are passed through as-is (they already
        match the Rust ``TransformSpec`` serde shape).  PyO3 transform objects
        are serialized via the ChartSpec round-trip.
        Mixing named and unnamed in one list is valid — unnamed transforms chain,
        named transforms fan-out without advancing ``FINAL_OUTPUT_KEY``.
        """
        if not transforms:
            return []
        from ferrum import ChartSpec

        # Separate dicts (already JSON-ready) from PyO3 objects (need round-trip).
        # Build the output list preserving order.
        result: list = []
        # Collect runs of PyO3 objects to batch-serialize through ChartSpec.
        pyo3_batch: list = []  # (original_index, transform_item) pairs
        for i, t in enumerate(transforms):
            inner = t.transform if isinstance(t, _NamedTransform) else t
            if isinstance(inner, dict):
                # Flush any pending PyO3 batch first.
                if pyo3_batch:
                    pyo3_objs = [item for _, item in pyo3_batch]
                    dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=pyo3_objs)
                    serialized = json.loads(dummy.to_json()).get("transforms", [])
                    for j, (orig_idx, orig_t) in enumerate(pyo3_batch):
                        entry = serialized[j] if j < len(serialized) else {}
                        if isinstance(transforms[orig_idx], _NamedTransform):
                            entry["name"] = transforms[orig_idx].name
                        result.append(entry)
                    pyo3_batch = []
                # Dict transform — pass through, inject name if wrapped.
                entry = dict(inner)
                if isinstance(t, _NamedTransform):
                    entry["name"] = t.name
                result.append(entry)
            else:
                pyo3_batch.append((i, inner))
        # Flush remaining PyO3 batch.
        if pyo3_batch:
            pyo3_objs = [item for _, item in pyo3_batch]
            dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=pyo3_objs)
            serialized = json.loads(dummy.to_json()).get("transforms", [])
            for j, (orig_idx, orig_t) in enumerate(pyo3_batch):
                entry = serialized[j] if j < len(serialized) else {}
                if isinstance(transforms[orig_idx], _NamedTransform):
                    entry["name"] = transforms[orig_idx].name
                result.append(entry)
        return result

    def _build_layers_list(self) -> list:
        """Convert internal _layers to a list of JSON-serializable dicts for Rust.

        ``coerce_layers`` in Rust runs ``json.dumps()`` on each dict, so every
        value must be JSON-serializable (no PyO3 objects).
        """
        out = []
        for layer in self._layers or []:
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
                layer_dict["transforms"] = Chart._transforms_to_json_list(layer.transforms)
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
            out.append(layer_dict)
        return out

    def _build_facet_dict(self) -> dict:
        """Convert internal _facet to the JSON dict Rust's FacetSpec expects.

        The ``"resolve"`` key is emitted only when at least one channel is set
        to ``"independent"``.  Absent keys default to ``"shared"`` in Rust, so
        omitting the key entirely when all channels are shared keeps the output
        byte-stable with existing goldens.
        """
        f = self._facet
        # Build the resolve sub-dict only when there is at least one
        # non-default (independent) resolution; omit entirely otherwise.
        resolve_dict: dict | None = None
        if f.resolve:
            non_default = {ch: mode for ch, mode in f.resolve.items() if mode == "independent"}
            if non_default:
                resolve_dict = dict(f.resolve)

        if f.mode_kind == "wrap":
            ncols = f.ncols or 1  # u32 required; default 1
            d: dict = {"field": f.field, "mode": {"kind": "wrap", "ncols": int(ncols)}}
            if resolve_dict is not None:
                d["resolve"] = resolve_dict
            return d
        # grid: col is the primary (column) field; row is the secondary (row) field.
        field = f.col if f.col is not None else (f.field or "")
        # Infer nrows/ncols from the data when the user did not supply them.
        # This is the common case: facet(row="a", col="b") without explicit
        # nrows/ncols.  Without inference the geometry defaults to 1×1, dropping
        # all but one panel.
        nrows = f.nrows if f.nrows is not None else self._infer_facet_cardinality(f.row)
        ncols = f.ncols if f.ncols is not None else self._infer_facet_cardinality(f.col)
        d = {
            "field": field,
            "mode": {"kind": "grid", "nrows": int(nrows), "ncols": int(ncols)},
        }
        if f.row is not None:
            d["row"] = f.row
        if resolve_dict is not None:
            d["resolve"] = resolve_dict
        return d

    def _infer_facet_cardinality(self, col_name: Optional[str]) -> int:
        """Return the number of distinct values in *col_name* from self._data.

        Falls back to 1 when *col_name* is None, the data is absent, or the
        column is not found — so the existing behaviour is preserved for cases
        where inference is not possible.
        """
        if col_name is None or self._data is None:
            return 1
        try:
            import polars as pl

            if isinstance(self._data, pl.DataFrame):
                if col_name in self._data.columns:
                    return self._data[col_name].n_unique()
                return 1
            # PyArrow fallback
            import pyarrow as pa

            tbl = self._data if isinstance(self._data, pa.Table) else to_arrow_table(self._data)
            if col_name in tbl.schema.names:
                col = tbl.column(col_name)
                return len(col.unique())
            return 1
        except Exception:
            return 1

    # ---- Properties ----

    def properties(
        self,
        *,
        width=None,
        height=None,
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
        width : int or "container" or None, optional
            Chart width in pixels, or ``"container"`` to fill the parent.
        height : int or "container" or None, optional
            Chart height in pixels.
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
            keyword on ``.show()`` / ``.save()`` / ``.show_svg()`` instead.

        Returns
        -------
        Chart
            New ``Chart`` with the specified properties updated.

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
            new._width = width
        if height is not None:
            new._height = height
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
        from ferrum.title import Title as _TitleCls

        remaining = dict(kwargs)
        c = self._clone()

        if "title" in remaining:
            c = c.properties(title=remaining.pop("title"))
        if "subtitle" in remaining:
            subtitle_text = remaining.pop("subtitle")
            # Merge subtitle into the existing Title object (preserving other
            # Title fields such as anchor, font, etc.) or create a new one.
            existing_title = c._title
            if isinstance(existing_title, _TitleCls):
                import dataclasses

                c._title = dataclasses.replace(existing_title, subtitle=subtitle_text)
            else:
                # No title set yet — create a Title with an empty main text.
                c._title = _TitleCls(text="", subtitle=subtitle_text)

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
                    from ferrum._shorthand import parse_shorthand

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

    def _build_encoding_specs(self, resolved, enc: dict, agg_field_remap: dict) -> dict:
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

        Returns
        -------
        dict
            Partial ``kw`` dict containing only encoding-related keys:
            channel names mapped to ``EncodingSpec`` instances, plus
            ``"tooltip_fields"`` when applicable.
        """
        from ferrum import EncodingSpec
        from ferrum._warn import warn_once
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
        from ferrum.repeat import _RepeatPlaceholder as _RPH

        # Collect fields from channels that carry no aggregate.
        non_agg_fields: list[str] = []
        for _ch in resolved._encoding.values():
            if not isinstance(_ch, ChannelBase):
                continue
            if _ch._kwargs.get("aggregate"):
                continue
            f = _ch.field
            if f is None or isinstance(f, _RPH):
                continue
            if f not in non_agg_fields:
                non_agg_fields.append(f)
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

    def to_spec(self):
        """Build the Rust ``ChartSpec`` for this chart.

        Resolves any pending statistical-mark desugar, converts Python encoding
        channel objects to ``EncodingSpec`` instances, and constructs the
        ``ChartSpec`` PyO3 object that the Rust renderer consumes.

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
        resolved = self._resolve_pending()

        # --- Channel aliasing (operates on a shallow copy to avoid mutating self) ---
        enc = dict(resolved._encoding)  # shallow copy — safe for alias remapping
        mk = dict(resolved._mark_kwargs) if resolved._mark_kwargs else {}
        enc, mk = _apply_channel_aliases(enc, mk)

        # --- Aggregate field remap: map original field → output_col for encoding ---
        # _PendingAggregate sentinels carry the original field name but Aggregate
        # transforms emit the output under a new column name (e.g. "mean_val").
        # Build a remap dict now so the EncodingSpec loop below uses the correct
        # post-aggregation field name. Key: original_field, Value: output_col.
        agg_field_remap: dict[str, str] = {}
        if resolved._transforms:
            for _t in resolved._transforms:
                # Use "is not None" so count() (field="") is included.
                if isinstance(_t, _PendingAggregate) and _t.field is not None:
                    agg_field_remap[_t.field] = _t.output_col

        # --- CoordPolar: remap theta/radius → x/y so Rust sees Cartesian channels ---
        enc = self._resolve_polar_remapping(resolved, enc)

        # --- Build EncodingSpec entries for each honored channel ---
        kw: dict = {"mark": resolved._mark or "point", "data": "default"}
        kw.update(self._build_encoding_specs(resolved, enc, agg_field_remap))

        # --- Resolve _PendingAggregate sentinels to concrete Aggregate objects ---
        effective_transforms = list(resolved._transforms) if resolved._transforms else []
        effective_transforms = self._resolve_pending_aggregates(resolved, effective_transforms)

        # --- Transform serialization ---
        if effective_transforms:
            # If any transform is a _NamedTransform or a plain dict (Phase 12
            # data transforms), serialize everything via JSON so Rust receives
            # the full pipeline through the serde path.
            has_named = any(isinstance(t, _NamedTransform) for t in effective_transforms)
            has_dict = any(isinstance(t, dict) for t in effective_transforms)
            if has_named or has_dict:
                xform_json = Chart._transforms_to_json_list_named(effective_transforms)
                kw["transforms_json"] = json.dumps(xform_json)
            else:
                kw["transforms"] = effective_transforms

        # --- Remaining chart-level properties ---
        if resolved._facet is not None:
            kw["facet"] = resolved._build_facet_dict()
        if resolved._coord is not None:
            c = resolved._coord
            # Back-compat: orient_coord_flip sets _coord = "flip" (a string).
            # New coord objects expose to_spec_dict(); CoordFlip returns "flip".
            kw["coord"] = c.to_spec_dict() if hasattr(c, "to_spec_dict") else c
        if mk:
            kw["mark_style"] = mk
        if resolved._layers is not None:
            kw["layers"] = resolved._build_layers_list()
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

        # --- Conditional / selection injection ---
        if resolved._selections:
            kw["selections"] = json.dumps([s.to_spec_dict() for s in resolved._selections])
            # Auto-inject selection fields into tooltip so cross-panel linked
            # selection can match marks by field values (not just data indices).
            sel_fields = set()
            for s in resolved._selections:
                if hasattr(s, "params") and s.params.get("fields"):
                    sel_fields.update(s.params["fields"])
            if sel_fields:
                existing = set()
                if "tooltip_fields" in kw:
                    for entry in json.loads(kw["tooltip_fields"]):
                        existing.add(entry.get("field", ""))
                elif "tooltip" in kw:
                    existing.add(getattr(kw["tooltip"], "field", ""))
                missing = sel_fields - existing
                if missing:
                    tf_list = []
                    if "tooltip_fields" in kw:
                        tf_list = json.loads(kw["tooltip_fields"])
                    elif "tooltip" in kw:
                        tf_list = [{"field": getattr(kw["tooltip"], "field", "")}]
                        del kw["tooltip"]
                    for f in sorted(missing):
                        tf_list.append({"field": f})
                    kw["tooltip_fields"] = json.dumps(tf_list)
        if resolved._conditionals:
            kw["conditionals"] = json.dumps([c.to_spec_dict() for c in resolved._conditionals])

        return ChartSpec(**kw)

    def _inject_auto_tooltips(self, kw: dict) -> dict:
        """Inject auto-generated tooltip fields into a serialised spec dict.

        Takes a plain-dict representation of a ``ChartSpec`` (as returned by
        ``json.loads(spec.to_json())``) and adds ``encoding.tooltip_fields``
        from the encoded channels when no explicit tooltip is already present.
        Returns the mutated dict.

        This is called by the interactive renderer after ``to_spec()``; static
        SVG/PNG renders skip it to avoid bloating the output with tooltip data.
        Explicit ``tooltip=`` or ``tooltip_fields=`` encodings always win.

        Parameters
        ----------
        kw : dict
            Parsed JSON dict from ``json.loads(spec.to_json())``.  Modified
            in place and returned.

        Returns
        -------
        dict
            The same dict with ``encoding.tooltip_fields`` added when
            applicable.
        """
        enc = kw.get("encoding") or {}
        if "tooltip" in enc or "tooltip_fields" in enc:
            return kw
        _TOOLTIP_SKIP = frozenset(("tooltip", "detail", "key", "href", "description", "url"))
        auto_fields: list[dict] = []
        seen_auto: set[str] = set()
        for _ch_name in _RENDERER_HONORED_CHANNELS:
            if _ch_name in _TOOLTIP_SKIP:
                continue
            ch_dict = enc.get(_ch_name)
            if ch_dict is None:
                continue
            field = ch_dict.get("field") if isinstance(ch_dict, dict) else None
            if field and isinstance(field, str) and field not in seen_auto:
                auto_fields.append({"field": field})
                seen_auto.add(field)
        if auto_fields:
            kw.setdefault("encoding", {})["tooltip_fields"] = auto_fields
        return kw

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
        resolved = self._resolve_pending()
        spec = resolved.to_spec()
        if resolved._layers is None:
            return spec
        return _SpecView(spec, resolved._layers)

    def to_json(self, *, indent=None) -> str:
        """Serialise the chart specification to a JSON string.

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

    # ---- Selections / interactivity (spec §3.10) ----

    def add_selection(self, *selections) -> "Chart":
        """Attach interactive selection(s) to this chart.

        Per ``ferrum-spec.md §3.10`` (L736), the SVG/PNG renderer silently
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
        new = self._clone()
        new._selections.extend(selections)
        return new

    def interactive(self, *, toolbar: bool = True) -> "Chart":
        """Mark this chart as interactive.

        Per ``ferrum-spec.md §3.10`` (L736), interactive features (selections,
        pan/zoom, conditional encodings) are silently ignored under SVG/PNG.
        Returns a new ``Chart`` so chained construction patterns work today
        and will gain real interactivity once the Phase 11 WASM renderer ships.

        Parameters
        ----------
        toolbar : bool, default True
            Whether to show the interactive toolbar (zoom/pan controls, export
            button). Set to ``False`` to render without the toolbar.

        Returns
        -------
        Chart
            New ``Chart`` (clone).

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
        new._conditionals.append(spec)
        if hasattr(spec, "selection_name"):
            # Ensure the selection is also registered so scene_build can wire it.
            existing = {s.name: s for s in new._selections if hasattr(s, "name")}
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
