"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ferrum._interactive import InteractiveChart

from ferrum._coerce import to_arrow_table
from ferrum._desugar import _resolve_pending_impl
from ferrum._layer import _Layer, _PendingMark
from ferrum._layer_transforms import (
    _NamedTransform,
    _resolve_layer_aggregates,
    _resolve_layer_bins,
    _transforms_to_json_list_named,
)
from ferrum._configure_mixin import ConfigureMixin
from ferrum.marks._chart_methods_diagnostic import DiagnosticMarksMixin
from ferrum.marks._chart_methods_statistical import StatisticalMarksMixin
from ferrum._render import _RenderMixin
from ferrum._shorthand import parse_shorthand
from ferrum._spec_build import SpecBuildMixin
from ferrum._spec_view import _SpecView
from ferrum.encoding.base import ChannelBase, _PendingAggregate
from ferrum.marks.base import MarkBase
from ferrum.position import _STACKABLE_MARKS

# Re-exported from this module so existing ``from ferrum.chart import ...``
# importers keep resolving after the cohesion split (CHART-01/02): the canonical
# homes are now ``_desugar.py`` and ``_layer_transforms.py``.  Only the symbols
# with a remaining ``from ferrum.chart import`` consumer are re-exported here;
# the rest are imported directly from their new modules by their callers.
from ferrum._desugar import (  # noqa: F401  (back-compat re-export)
    _resolve_cat_axis,
    _split_style_kwargs,
)

# Single-sourced in ``_layer`` (the desugar leaf); re-exported here so any
# existing ``from ferrum.chart import _PRIMITIVE_MARKS`` keeps resolving.
from ferrum._layer import _PRIMITIVE_MARKS  # noqa: F401  (back-compat re-export)


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


from ferrum._facet import (
    _Facet,
    build_facet_dict as _build_facet_dict_fn,
    infer_facet_cardinality as _infer_facet_cardinality_fn,
)
from ferrum.encoding import _channel_class_map, _channel_class_for, _apply_channel_aliases
from ferrum.selection import ConditionalSpec


from ferrum.composition import (
    _expand_layers,
    _merge_top_transforms,
    _promote_layer_color,
    _validate_share_modes,
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


def _column_names_for_validation(data) -> "Optional[list]":
    """Return *data*'s column names for a cheap boundary check, or ``None``.

    ``Chart._data`` is stored raw (polars/pandas frame, pyarrow
    ``Table``/``RecordBatch``, dict-of-arrays, list-of-records, ndarray), so
    name extraction must be type-aware: pyarrow's ``.columns`` yields column
    OBJECTS, not names. Returns ``None`` for inputs without a cheap name
    listing — callers should then skip validation rather than materialize
    an Arrow table just to check a name.
    """
    if data is None:
        return None
    try:
        import pyarrow as pa

        if isinstance(data, pa.Table):
            return list(data.column_names)
        if isinstance(data, pa.RecordBatch):
            return list(data.schema.names)
    except ImportError:
        pass
    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return list(data.columns)
        if isinstance(data, pl.LazyFrame):
            return list(data.collect_schema().names())
    except ImportError:
        pass
    columns = getattr(data, "columns", None)
    if columns is not None and all(isinstance(c, str) for c in columns):
        return list(columns)
    return None


def _desugar_secondary_y(chart: "Chart", feature: "SecondaryY") -> "Chart":
    """Desugar ``chart + SecondaryY(...)`` into an appended independent-y layer (GH #52).

    Per the secondary-y-axis design spec §4: the base chart's existing
    layer(s) are unchanged (a multi-layer base keeps its internal sharing),
    plus one appended layer — mark ``feature.mark``, ``y`` encoding on
    ``feature.field`` (with ``feature.axis``/``feature.scale`` attached),
    ``x`` inherited from the base chart, color literal ``feature.color``,
    opacity ``feature.opacity`` — flagged ``independent_y=True`` so Rust
    resolves its y-scale independently and renders it as a stacked right
    axis (slot contract: layer 0 is always the primary/left axis).

    Mirrors the same "expand-then-append" pattern ``Chart.__add__`` uses for
    ``Chart + Chart``: pending marks are resolved, the base is expanded into
    its layer list (a single-mark chart becomes a one-element list), and the
    chart-level transforms are replaced with ``_expand_layers``'s filtered
    top-level list so an encoding-implicit aggregate does not double-run
    once the chart carries ``_layers``.
    """
    from ferrum.encoding.positional import Y as _Y

    resolved = chart._resolve_pending()
    new = resolved._clone()
    x_encoding = new._encoding.get("x")
    if x_encoding is None:
        raise ValueError(
            "SecondaryY: the base chart must have an x encoding to inherit "
            "-- call .encode(x=...) before adding SecondaryY(...)."
        )
    # SecondaryY reads its field from the base chart's own table (no data
    # merge happens in this desugar), so validate it here with the same
    # boundary-error courtesy as the x-encoding check above instead of
    # letting a typo surface as a downstream Rust column error. Column
    # names are extracted type-aware (pyarrow's `.columns` is column
    # OBJECTS, not names — see the isinstance dispatch used for temporal
    # inference above); inputs without a cheap name listing (dict-of-arrays,
    # list-of-records, ndarray) skip the check and keep render-time errors.
    column_names = _column_names_for_validation(new._data)
    if column_names is not None and feature.field not in column_names:
        raise ValueError(
            f"SecondaryY: field {feature.field!r} is not a column of the "
            "base chart's data (SecondaryY draws from the same table as "
            f"the base chart); available columns: {list(column_names)}"
        )

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
        encoding={"x": x_encoding, "y": y_encoding},
        mark_kwargs=mark_kwargs,
        independent_y=True,
    )
    existing_layers, top_xforms = _expand_layers(new)
    new._transforms = top_xforms
    new._layers = existing_layers + [secondary_layer]
    return new


def _append_unique_by_name(seq: list, item: object) -> None:
    """Append *item* to *seq* if no element with the same ``.name`` is present.

    Preserves insertion order (first-seen wins).  Mutates *seq* in place.
    *item* must have a ``.name`` attribute; *seq* may contain ``None`` entries
    which are skipped during the existence check.
    """
    existing = {el.name for el in seq if el is not None and hasattr(el, "name")}
    if item.name not in existing:
        seq.append(item)


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


class Chart(
    ConfigureMixin,
    StatisticalMarksMixin,
    DiagnosticMarksMixin,
    SpecBuildMixin,
    _RenderMixin,
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
        self._params: list = []
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

        The desugar body lives in :func:`ferrum._desugar._resolve_pending_impl`
        (cohesion split CHART-02/07); this method is a thin delegator so external
        callers (``chart._resolve_pending()``) keep working unchanged.
        """
        return _resolve_pending_impl(self)

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

        Places a free-positioned text glyph at each (x, y) data coordinate.
        No collision avoidance is applied and no leader line is drawn; the
        glyph sits exactly at the encoded position plus any ``dx``/``dy``
        pixel offset.  Use this mark when you control placement explicitly
        (e.g. inline bar labels, callout annotations at known coordinates).

        See also :meth:`mark_label` for a point-anchored annotation with
        automatic collision avoidance and optional leader-line support.

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
        """Render point-anchored text annotations near data points.

        Each label is anchored to its (x, y) data point and positioned by a
        greedy collision-avoidance algorithm (not placed at an arbitrary
        coordinate like :meth:`mark_text`).  An optional leader line
        (``leader_line=True``) draws a thin connecting line from the data
        point to the placed label, useful when the label lands far from its
        source point.  Use this mark when you want the renderer to find
        non-overlapping placements automatically.

        See also :meth:`mark_text` for a free-positioned text glyph that
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

            if isinstance(value, ConditionalSpec):
                # encode(<channel>=cond): the wire channel is KNOWN from the
                # encode key (already snake_case, e.g. "opacity", "size",
                # "stroke_width"). Stamp it so both the wire channel and the
                # value-kind resolution are correct, then record the conditional
                # and auto-register its source selection. Conditionals live in
                # _conditionals, not _encoding.
                new._conditionals.append(replace(value, channel=name))
                sel = value.selection
                if sel is not None:
                    _append_unique_by_name(new._selections, sel)
                continue

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
                new._facet = _Facet(
                    mode_kind="wrap",
                    field=col_enc.field,
                    ncols=ncols,
                    nrows=nrows,
                    wrap_orient="col",
                )
            elif row_enc is not None:
                new._facet = _Facet(
                    mode_kind="wrap",
                    field=row_enc.field,
                    ncols=ncols,
                    nrows=nrows,
                    wrap_orient="row",
                )

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
        # Resolve pending statistical marks before snapshotting encoding dicts.
        lhs = self._resolve_pending()
        rhs = other._resolve_pending()
        new = lhs._clone()
        lhs_layers, lhs_top_xforms = _expand_layers(lhs)
        rhs_layers, rhs_top_xforms = _expand_layers(rhs)
        # Adopt the LHS's *filtered* top-level transforms (``_expand_layers``
        # strips encoding-implicit ``_PendingAggregate`` sentinels — each layer
        # aggregates its own data via ``_resolve_layer_aggregates``).  The clone
        # above copied ``lhs._transforms`` verbatim, which would re-introduce a
        # spurious chart-level aggregate over the merged batch.
        new._transforms = lhs_top_xforms

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
            from ferrum.parameter import VariableParameter

            existing_variable_names = {
                p.name for p in new._params if p is not None and isinstance(p, VariableParameter)
            }
            for s in rhs._selections:
                if s is None:
                    continue
                if s.name in existing_variable_names:
                    _check_param_collision(s.name, is_selection=True, context="layer merge")
                _append_unique_by_name(new._selections, s)
        if rhs._conditionals:
            new._conditionals.extend(rhs._conditionals)
        if rhs._params:
            from ferrum.parameter import VariableParameter

            existing_selection_names = {s.name for s in new._selections if s is not None}
            existing_param_names = {p.name for p in new._params if p is not None}
            for p in rhs._params:
                if p is None:
                    continue
                if isinstance(p, VariableParameter) and p.name in existing_selection_names:
                    _check_param_collision(p.name, is_selection=False, context="layer merge")
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
                wrap_orient=None,
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
                wrap_orient="col",
            )
        elif row is not None:
            new._facet = _Facet(
                mode_kind="wrap",
                field=row,
                nrows=nrows,
                ncols=ncols,
                wrap_orient="row",
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
            keyword on ``.show()`` / ``.save()`` / ``.to_svg()`` instead.

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
        resolved = self._resolve_pending()

        # Chart.override spec-piece payload (encoding scales / mark style / coord).
        # Built on demand for standalone .to_spec() callers; the render path passes
        # the once-per-render payload via _override_payload.  None when no override.
        if _override_payload is None and self._overrides:
            from ferrum._override_apply import build_payload

            _override_payload = build_payload(self._overrides)

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
        _override_encoding = _override_payload.encoding if _override_payload is not None else None
        kw.update(self._build_encoding_specs(resolved, enc, agg_field_remap, _override_encoding))

        # --- Resolve _PendingAggregate sentinels to concrete Aggregate objects ---
        effective_transforms = list(resolved._transforms) if resolved._transforms else []
        effective_transforms = self._resolve_pending_aggregates(resolved, effective_transforms)

        # --- Resolve _PendingBin sentinels to concrete unnamed Bin objects (single-chart) ---
        # For layered charts the bin sentinels are resolved per-layer below, so
        # this call is a no-op on that path (sentinels already stripped from
        # top-level transforms by _expand_layers).
        effective_transforms = self._resolve_pending_bins(effective_transforms)

        # --- Resolve per-layer encoding aggregates (layered charts) ---
        # Each aggregating layer becomes a NAMED chart-level Aggregate transform
        # whose output the layer reads via ``data_source`` (per-layer transforms
        # are never executed standalone by the renderer).  Two aggregating layers
        # therefore aggregate independently.  No-op when no layer aggregates.
        # ``resolved`` may be ``self`` (no pending desugar), so the resolved
        # layers are threaded through ``_build_layers_list`` rather than mutated
        # onto the chart.
        serialized_layers = resolved._layers
        if resolved._layers:
            serialized_layers, layer_agg_transforms = _resolve_layer_aggregates(resolved._layers)
            if layer_agg_transforms:
                effective_transforms = effective_transforms + layer_agg_transforms

        # --- Resolve per-layer encoding bins (layered charts) ---
        # Each binning layer becomes a NAMED chart-level Bin transform whose
        # output the layer reads via ``data_source`` (fan-out: does not advance
        # the unnamed chain, so the shared batch is not corrupted for other
        # layers or for named aggregate transforms).  Run after aggregate
        # resolution so the layer list processes the aggregate-resolved layers.
        if resolved._layers:
            serialized_layers, layer_bin_transforms = _resolve_layer_bins(serialized_layers)
            if layer_bin_transforms:
                effective_transforms = effective_transforms + layer_bin_transforms

        # --- Transform serialization ---
        if effective_transforms:
            # If any transform is a _NamedTransform or a plain dict (Phase 12
            # data transforms), serialize everything via JSON so Rust receives
            # the full pipeline through the serde path.
            has_named = any(isinstance(t, _NamedTransform) for t in effective_transforms)
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
        # Mark style: apply mark_* override last so it beats the chart's mark style.
        # A layered chart (multiple marks) has no single primary mark — apply_mark_style
        # raises FerrumOverrideError on a mark_* override there (spec §11 Q4).
        if _override_payload is not None and _override_payload.mark_style:
            from ferrum._override_consume import apply_mark_style

            mk = apply_mark_style(mk, _override_payload, is_multi_mark=resolved._layers is not None)
        if mk:
            kw["mark_style"] = mk
        if resolved._layers is not None:
            kw["layers"] = resolved._build_layers_list(serialized_layers)
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
            self._inject_selection_tooltips(kw, resolved._selections)
        if resolved._conditionals:
            kw["conditionals"] = json.dumps([c.to_spec_dict() for c in resolved._conditionals])

        # --- Reactive-parameter (D6) params section ---
        # Unified declaration: registered selections (auto-promoted),
        # explicit fm.param() variables, and any Parameter referenced as a
        # scale domain. Deduped by name (first-seen wins); omitted entirely
        # when empty so param-free specs stay byte-identical to before.
        params_list = self._collect_params(resolved, enc)
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
        resolved = self._resolve_pending()
        spec = resolved.to_spec()
        if resolved._layers is None:
            return spec
        return _SpecView(spec, resolved._layers)

    def to_json(self, *, indent=None) -> str:
        """Serialise the chart specification to a JSON string.

        .. note::
            Unlike :meth:`to_svg`, :meth:`to_png`, and :meth:`to_html` — which
            return *rendered* output — ``to_json`` returns the chart
            **specification** (the declaration you built), not the rendered
            scene graph.  ``save(path)`` with a ``.json`` extension writes the
            rendered scene graph instead, which is a different artifact.  For
            the specification as a Python value, see :meth:`to_dict`.

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
            _append_unique_by_name(new._selections, sel)
        return new

    def add_params(self, *params) -> "Chart":
        """Attach reactive variable parameter(s) to this chart.

        Records one or more :class:`~ferrum.parameter.Parameter` objects (built
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
                _append_unique_by_name(new._selections, p)
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
        new._conditionals.append(spec)
        if hasattr(spec, "selection_name"):
            # Ensure the selection is also registered so scene_build can wire it.
            # Auto-register the carried selection when the spec knows it,
            # so the explicit path benefits like encode(<channel>=cond) does.
            carried = getattr(spec, "selection", None)
            if carried is not None and carried.name == spec.selection_name:
                _append_unique_by_name(new._selections, carried)
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
