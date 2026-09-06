"""Heavy-stat-mark desugar helpers (Phase 8b Sub-batch F).

Each desugar_<name> returns a ``MarkDesugarResult`` — either in layered mode
(``layers`` set) or single-mark mode (``mark``/``transforms``/``remap``).
"""

from __future__ import annotations
from typing import Any, Optional

from ferrum import (
    BoxStats,
    Contour,
    Hex,
    Kde2D,
    QQ,
    Raster,
    Swarm,
    Violin,
)
from ferrum._core import transform_registry
from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum._registry import registry_output_columns, registry_secondary_columns
from ferrum._validate import validate_choice
from ferrum.marks._desugar_helpers import (
    nominal_color_channel,
    resolve_color_groupby,
)


def desugar_contour(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    thresholds: int = 6,
    smooth: bool = True,
    fill: bool = True,
    cmap: str | None = None,
    groupby: str | None = None,
) -> MarkDesugarResult:
    """Bivariate-density contour mark desugar.

    Converts ``chart.mark_contour(...)`` into a ``Kde2D`` → ``Contour``
    transform chain plus a polygon or segment layer.  Also used by
    ``desugar_density`` when both x and y encodings are present.

    Data contract
    -------------
    Input: DataFrame with numeric columns ``x_field`` and ``y_field``.

    ``Kde2D`` (unnamed — advances the chain) estimates a 2D kernel-density
    surface on a 128×128 grid.  When ``groupby`` is set, one surface is
    emitted per group and a trailing group column (any groupby-supported
    dtype — string, integer, float, or boolean — preserved at its
    original dtype) is appended.

    ``Contour`` (named ``"contour"``) traces iso-density curves on that
    surface and produces:
    ``[level_id (UInt32), contour_x (Float64), contour_y (Float64)]``
    plus the trailing group column when the input was grouped.

    Layers emitted (no groupby)
    ---------------------------
    1. ``polygon`` — ``x="contour_x"``, ``y="contour_y"``,
       ``color="level_value"``, ``mark_kwargs={"cmap": cmap, "detail": "level_id"}``.

    Layers emitted (with groupby)
    -----------------------------
    1. ``segment`` — ``x="contour_x"``, ``y="contour_y"``,
       ``x2="contour_x2"``, ``y2="contour_y2"``, ``color=groupby``.
       Isoline segments are used for grouped contours because the isoband
       ``level_id`` encoding is not globally unique across groups; polygon
       grouping by ``level_id`` would merge polygons from different groups
       into incorrect shapes.  Segment marks are per-row and color each
       group's contour lines categorically — each group is visually distinct.

    Parameters
    ----------
    x_field : str or None
        Numeric x field. Required.
    y_field : str or None
        Numeric y field. Required.
    bandwidth : str or float, default "scott"
        KDE bandwidth rule or numeric value.
    thresholds : int, default 6
        Number of iso-density contour levels.
    smooth : bool, default True
        Whether to smooth the KDE grid before contouring.
    fill : bool, default False
        Whether the contour polygons are filled (passed to ``Contour``).
        Ignored when ``groupby`` is set — grouped contours always use
        isoline (segment) mode to avoid cross-group polygon merging.
    cmap : str or None, default None
        Colormap name applied to the polygon layer.  ``None`` defers to the
        theme's sequential scheme.  Ignored when ``groupby`` is set.
    groupby : str or None, default None
        Group-key column (any groupby-supported dtype — string, integer,
        float, or boolean — preserved at its original dtype). When set,
        ``Kde2D`` computes one surface per group and ``Contour``
        propagates the group column.  The layer
        uses segment marks and colors by the group field so each group's
        contour lines are a distinct categorical color.

    Returns
    -------
    MarkDesugarResult

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_contour("x", "y")
    >>> result.layers[0].mark
    'polygon'
    >>> result_grouped = desugar_contour("x", "y", groupby="g")
    >>> result_grouped.layers[0].mark
    'segment'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_contour() requires .encode(x=..., y=...)")
    # Kde2D is UNNAMED so it advances the chain (current → Kde2D output);
    # Contour then runs on the chained Kde2D output. Contour is named so the
    # downstream layer can route through data_source="contour".
    kde_kwargs: dict = dict(x=x_field, y=y_field, bandwidth=bandwidth, n=128)
    if groupby is not None:
        kde_kwargs["groupby"] = groupby
    transforms = [
        Kde2D(**kde_kwargs),
        Contour(
            thresholds=thresholds,
            fill=False if groupby is not None else fill,
            smooth=smooth,
            name="contour",
        ),
    ]
    from ferrum.encoding import X, Y

    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of any of these columns raises here instead of
    # silently reading a stale name.  All six are declared `contour` outputs;
    # each branch below reads a subset (isoline branches use the segment
    # endpoints, the isoband branch uses level_id/level_value).
    _contour_cols = registry_output_columns(
        "contour",
        ["contour_x", "contour_y", "contour_x2", "contour_y2", "level_id", "level_value"],
    )

    if groupby is not None:
        # Grouped contour: use isoline (segment) mode so each row is an
        # independent line segment — no polygon grouping needed, no cross-group
        # level_id collision.  Color by the group column for categorical hue —
        # typed Nominal, because `segment` is one of the marks whose inert
        # continuous-color handling is silent: an Int64 `groupby` bound as a
        # bare string drew both groups' isolines in one colour with no warning.
        # The Contour transform outputs isoline columns:
        #   level_id, level_value, contour_x, contour_y, contour_x2, contour_y2, <groupby>
        layers = [
            _Layer(
                name="segment",
                mark="segment",
                encoding={
                    "x": X(_contour_cols["contour_x"], title=x_field),
                    "y": Y(_contour_cols["contour_y"], title=y_field),
                    "x2": _contour_cols["contour_x2"],
                    "y2": _contour_cols["contour_y2"],
                    "color": nominal_color_channel(groupby),
                },
                mark_kwargs=None,
                data_source="contour",
            )
        ]
    elif fill:
        # Isoband mode: Contour emits polygon vertex rows (x, y per vertex).
        # Use polygon mark grouped by level_id; color by level_value so each
        # density band gets a distinct fill from the sequential colormap.
        mk = {"detail": _contour_cols["level_id"]}
        if cmap is not None:
            mk["cmap"] = cmap
        layers = [
            _Layer(
                name="polygon",
                mark="polygon",
                encoding={
                    "x": X(_contour_cols["contour_x"], title=x_field),
                    "y": Y(_contour_cols["contour_y"], title=y_field),
                    "color": _contour_cols["level_value"],
                },
                mark_kwargs=mk,
                data_source="contour",
            )
        ]
    else:
        # Isoline mode: Contour emits one row per segment with
        # (contour_x, contour_y, contour_x2, contour_y2).
        # Use segment mark -- each row draws one line from (x,y) to (x2,y2).
        mk = {}
        if cmap is not None:
            mk["cmap"] = cmap
        layers = [
            _Layer(
                name="segment",
                mark="segment",
                encoding={
                    "x": X(_contour_cols["contour_x"], title=x_field),
                    "y": Y(_contour_cols["contour_y"], title=y_field),
                    "x2": _contour_cols["contour_x2"],
                    "y2": _contour_cols["contour_y2"],
                },
                mark_kwargs=mk if mk else None,
                data_source="contour",
            )
        ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names(
    "contour",
    frozenset(
        {
            "polygon",
            "segment",
        }
    ),
)


def desugar_violin(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    inner: Optional[str] = "box",
    x_sort: Any = None,
    y_sort: Any = None,
    color_field: str | None = None,
    shared_extent: bool = False,
    horizontal: bool = False,
) -> MarkDesugarResult:
    """Violin-plot composite mark desugar.

    Converts ``chart.mark_violin(...)`` into a ``Violin`` transform plus a
    polygon layer for the violin body, with an optional inner mark overlay.

    Data contract
    -------------
    Input: DataFrame with categorical column ``x_field`` and numeric column
    ``y_field``.

    ``Violin`` (named ``"violin"``) estimates a per-group KDE mirrored into
    a polygon shape and produces:
    ``[group_id (UInt32), violin_x (Float64), violin_y (Float64)]``

    Layers emitted
    --------------
    1. ``polygon`` — ``x=x_field``, ``y="violin_y"``,
       ``mark_kwargs={"detail": "group_id", "fill_opacity": 0.5}``.
    2. Inner marks — vary by ``inner`` value (see below).

    Inner variants
    --------------
    * ``inner=None``         — polygon only.
    * ``inner="point"``      — raw points overlaid at the original
      ``(x_field, y_field)`` positions.
    * ``inner="quartile"``   — ``BoxStats`` transform (named ``"quart"``)
      adds three ``rule`` layers at ``q1``, ``median``, ``q3``.
    * ``inner="box"`` (default) — full ``desugar_boxplot`` overlay with
      ``whisker_mult=1.5``, ``outliers=False``, ``size=0.1``.

    Parameters
    ----------
    x_field : str or None
        Categorical (grouping) field. Required.
    y_field : str or None
        Numeric value field. Required.
    bandwidth : str or float, default "scott"
        KDE bandwidth rule or numeric value.
    inner : {"box", "quartile", "point", None}, default "box"
        Style of inner mark drawn inside the violin body.
    x_sort, y_sort : optional
        Sort order applied to the categorical positional axis, injected by
        the composite-mark expansion from ``sort=`` on the encoding.
    color_field : str or None, default None
        Hue field, injected by the composite-mark expansion from a ``color=``
        encoding.  When set, the ``Violin`` (and inner ``BoxStats``) groupby
        becomes ``[x_field, color_field]`` so the KDE is computed per (x, hue)
        group, the violin body gains a ``color`` encoding, and the per-hue
        violins are overlaid (distinct fills, ``fill_opacity=0.5``) within each
        x-category band.  When ``None`` the output is byte-identical to a
        single-group violin.
    shared_extent : bool, default False
        When ``True``, all groups within the violin share the same KDE
        evaluation range (the cross-group global min/max), making the value
        axis directly comparable across groups.  When ``False`` (default),
        each group's KDE is evaluated on its own per-group data range.
        Mirrors ``mark_density(multiple="stack"/"fill")`` behavior.
    horizontal : bool, default False
        When ``True``, swap axes so the categorical grouping is on ``y`` and
        the value distribution is on ``x``.  Mirrors the ``horizontal``
        parameter on ``desugar_boxplot``.  The body encoding is swapped
        (``y=cat``, ``x=violin_y``) and the inner box/quartile/point layers
        are similarly swapped.

        Horizontal rendering is fully supported: the ``Violin`` transform
        emits ``__pos_y_offset__`` (scaled by panel height) instead of
        ``__pos_x_offset__`` when ``horizontal=True``, so the KDE width
        expands along the category band on the ``y`` axis.

    Returns
    -------
    MarkDesugarResult
        Layered mode (``.layers`` set).

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``, or if ``inner``
        is not one of the valid options.

    Examples
    --------
    >>> result = desugar_violin("species", "petal_length", inner=None)
    >>> result.layers[0].mark
    'polygon'
    >>> len(result.layers)
    1
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_violin() requires .encode(x=..., y=...)")
    validate_choice("mark_violin", "inner", inner, ("box", "quartile", "point", None))
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either column raises here instead of silently
    # reading a stale name.
    _violin_cols = registry_output_columns("violin", ["violin_y", "group_id"])

    from ferrum.encoding import X, Y

    # Resolve cat/val axes.  When horizontal=True the categorical grouping moves
    # to y and the numeric values move to x, mirroring the boxplot pattern.
    if horizontal:
        cat_field = y_field  # categorical grouping on y
        val_field = x_field  # numeric values on x
        cat_sort = y_sort
    else:
        cat_field = x_field  # categorical grouping on x (default)
        val_field = y_field  # numeric values on y
        cat_sort = x_sort

    # Wrap the categorical encoding in X/Y with sort when a sort is present.
    if horizontal:
        cat_enc = Y(cat_field, sort=cat_sort) if cat_sort is not None else cat_field
    else:
        cat_enc = X(cat_field, sort=cat_sort) if cat_sort is not None else cat_field

    # When a hue (color) field is present, the KDE — and every inner summary —
    # must split per (cat, hue) group rather than pooling across hues.  The
    # Violin and BoxStats transforms propagate all groupby columns to their
    # output, so `color_field` is available as an output column on the violin
    # batch and can drive a per-hue fill on the body polygon.
    # `detail="group_id"` keeps each (cat, hue) group drawing as a distinct
    # closed polygon, so the per-hue violins overlay within each category band.
    # Add the hue field to the groupby only when it is a distinct column; when
    # color encodes the same field as cat the KDE is already split per category
    # and the body just colors by the surviving cat column.
    groupby, split_hue = resolve_color_groupby(cat_field, color_field, [cat_field])

    # Body encoding: for vertical violin x=cat, y=violin_y (value grid points).
    # For horizontal violin y=cat, x=violin_y (value grid points on x-axis).
    # violin_x (the mirrored KDE density offset) is handled by the position
    # offset column emitted by the Rust Violin transform (__pos_x_offset__ for
    # vertical; __pos_y_offset__ for horizontal — both are implemented; the
    # Violin transform emits __pos_y_offset__ from panel height).
    if horizontal:
        body_encoding: dict = {"y": cat_enc, "x": X(_violin_cols["violin_y"], title=val_field)}
    else:
        body_encoding = {"x": cat_enc, "y": Y(_violin_cols["violin_y"], title=val_field)}
    # Attach the per-hue body fill only when the hue is a distinct column from
    # the categorical axis (split_hue). When color encodes the same field as
    # cat, the color encoding is redundant with the axis, so it is suppressed to
    # match the errorbar/errorband siblings.
    if split_hue:
        body_encoding["color"] = nominal_color_channel(color_field)

    transforms = [
        Violin(
            field=val_field,
            groupby=groupby,
            bandwidth=bandwidth,
            shared_extent=shared_extent,
            horizontal=horizontal,
            name="violin",
        )
    ]
    violin_layer = _Layer(
        name="body",
        mark="polygon",
        encoding=body_encoding,
        mark_kwargs={"detail": _violin_cols["group_id"], "fill_opacity": 0.5},
        data_source="violin",
    )
    if inner is None:
        return MarkDesugarResult(transforms=transforms, layers=[violin_layer])
    if inner == "point":
        # Raw points on the original data (not the violin batch).
        if horizontal:
            point_encoding: dict = {"y": cat_enc, "x": val_field}
        else:
            point_encoding = {"x": cat_enc, "y": val_field}
        if color_field is not None:
            point_encoding["color"] = nominal_color_channel(color_field)
        # raw points read from the original (unsplit) data, so coloring by the
        # hue column is always valid regardless of split_hue. Typed Nominal
        # rather than taking the `point` carve-out: this layer shares a chart
        # (and therefore one color scale) with the violin body layer above,
        # which is Nominal -- leaving this one to infer would ask that single
        # scale to be Continuous and Nominal at once for a numeric hue.
        return MarkDesugarResult(
            transforms=transforms,
            layers=[
                violin_layer,
                _Layer(name="point", mark="point", encoding=point_encoding),
            ],
        )
    if inner == "quartile":
        transforms.append(BoxStats(field=val_field, groupby=groupby, name="quart"))
        layers = [violin_layer]
        # Derived from the registry (spec §4.6 / M6a) rather than hardcoded,
        # so a Rust-side rename of any of these columns raises here instead
        # of silently reading a stale name.
        _quart_cols = registry_output_columns("box_stats", ["q1", "median", "q3"])
        for role, col in _quart_cols.items():
            mk = {} if role == "median" else {"stroke_dash": [2, 2]}
            if horizontal:
                quart_encoding: dict = {"y": cat_enc, "x": col}
            else:
                quart_encoding = {"x": cat_enc, "y": col}
            # The q1/median/q3 rules read the split "quart" BoxStats output
            # (grouped by the same split_hue-gated `groupby` as the body), so
            # attach the per-hue color only when the hue is a distinct column
            # from the categorical axis (split_hue). When color encodes the same
            # field as cat, the color is redundant with the axis and is
            # suppressed, matching the body polygon and the errorbar/errorband
            # siblings. (Contrast with the point inner above, which reads the
            # unsplit original data and so always colors by the hue column.)
            if split_hue:
                quart_encoding["color"] = nominal_color_channel(color_field)
            layers.append(
                _Layer(
                    name=col,
                    mark="rule",
                    encoding=quart_encoding,
                    mark_kwargs=mk if mk else None,
                    data_source="quart",
                )
            )
        return MarkDesugarResult(transforms=transforms, layers=layers)
    # inner == "box": delegate to desugar_boxplot with the swapped axes so the
    # inner box also respects the horizontal orientation.
    from ferrum.marks.composite import desugar_boxplot

    box_result = desugar_boxplot(
        x_field,
        y_field,
        whisker_mult=1.5,
        outliers=False,
        size=0.1,
        horizontal=horizontal,
        x_sort=x_sort,
        y_sort=y_sort,
        color_field=color_field if split_hue else None,
    )
    return MarkDesugarResult(
        transforms=[*transforms, *box_result.transforms],
        layers=[violin_layer, *box_result.layers],
    )


register_layer_names(
    "violin",
    frozenset(
        {
            "body",
            "point",
            "q1",
            "median",
            "q3",
            "whisker",
            "lower_cap",
            "upper_cap",
            "box",
            "outlier",
        }
    ),
)


#: ``QQ``'s spec ``name=``, so the reference layer's ``data_source`` can be
#: composed from it rather than repeated as a second literal.
_QQ_TRANSFORM_NAME = "qq_main"

#: ``QQ``'s secondary reference-line batch's key under DEFAULT construction,
#: as ``transform_registry("qq")["secondary_outputs"]`` declares it -- and
#: what ``registry_secondary_columns``'s ``key`` argument validates against
#: below. A caller-supplied ``name=`` (here, ``_QQ_TRANSFORM_NAME``) prefixes
#: it as ``"{name}_qq_line"`` (R3 close-remediation: `qq` now prefixes like
#: ``letter_value``/``linkage`` already did), so the reference layer's real
#: ``data_source`` is the composed ``f"{_QQ_TRANSFORM_NAME}_{_QQ_LINE_KEY}"``,
#: not this bare key.
_QQ_LINE_KEY = "qq_line"


def desugar_qq(
    field: str,
    *,
    distribution: str = "normal",
    dequantize: bool = False,
    line: bool = True,
) -> MarkDesugarResult:
    """Q-Q (quantile-quantile) plot mark desugar.

    Converts ``chart.mark_qq(...)`` into a ``QQ`` transform plus a point
    layer and an optional reference-line rule layer.

    Data contract
    -------------
    Input: DataFrame with numeric column ``field``.

    ``QQ`` (named ``"qq_main"``) computes theoretical vs. sample quantiles
    and produces:
    ``[theoretical (Float64), sample (Float64)]``

    When ``line=True``, ``QQ`` also emits a secondary named output, published
    under ``"qq_main_qq_line"`` (the spec's ``name="qq_main"`` prefixes the
    registry's default ``"qq_line"`` key, like every other named secondary
    output), a single-row batch with columns:
    ``[qq_line_x_start, qq_line_x_end, qq_line_y_start, qq_line_y_end]``

    Layers emitted
    --------------
    1. ``point`` — ``x="theoretical"``, ``y="sample"`` (from ``"qq_main"``).
    2. ``rule``  — ``x="qq_line_x_start"``, ``y="qq_line_y_start"``,
       ``x2="qq_line_x_end"``, ``y2="qq_line_y_end"``
       (from ``"qq_main_qq_line"``).  Only when ``line=True``.

    Parameters
    ----------
    field : str
        Numeric column to compute quantiles from.
    distribution : str, default "normal"
        Theoretical distribution to compare against, validated against the
        live ``transform_registry("qq")["methods"]`` vocabulary (spec §4.6)
        -- currently ``"normal"``, ``"uniform"``, ``"exponential"``,
        ``"lognormal"``.
    dequantize : bool, default False
        Whether to jitter duplicate quantile values slightly.
    line : bool, default True
        Whether to add the Q-Q reference line.

    Returns
    -------
    MarkDesugarResult
        Layered mode (``.layers`` set).

    Raises
    ------
    ValueError
        If ``distribution`` is not one of the valid options.

    Examples
    --------
    >>> result = desugar_qq("residuals")
    >>> [layer.mark for layer in result.layers]
    ['point', 'rule']
    """
    validate_choice("mark_qq", "distribution", distribution, transform_registry("qq")["methods"])
    transforms = [
        QQ(
            field=field,
            distribution=distribution,
            dequantize=dequantize,
            emit_line=line,
            name=_QQ_TRANSFORM_NAME,
        )
    ]
    from ferrum.encoding import X, Y

    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either column raises here instead of silently
    # reading a stale name.
    _qq_cols = registry_output_columns("qq", ["theoretical", "sample"])
    layers = [
        _Layer(
            name="point",
            mark="point",
            encoding={
                "x": X(_qq_cols["theoretical"], title="Theoretical Quantiles"),
                "y": Y(_qq_cols["sample"], title="Sample Quantiles"),
            },
            data_source=_QQ_TRANSFORM_NAME,
        )
    ]
    if line:
        # The reference line rides QQ's SECONDARY named batch, declared in the
        # registry since 2026-09-06 -- so its four columns check the same way
        # the primary two above do. `_QQ_LINE_KEY` validates the batch's
        # DEFAULT-construction key; the real `data_source` is that key
        # prefixed by the spec's `name=`, same as `letter_value`/`linkage`
        # (R3 close-remediation: `qq` used to publish this batch unprefixed,
        # the one kind of three that didn't follow the rule).
        _qq_line_cols = registry_secondary_columns(
            "qq",
            _QQ_LINE_KEY,
            ["qq_line_x_start", "qq_line_y_start", "qq_line_x_end", "qq_line_y_end"],
        )
        layers.append(
            _Layer(
                name="reference",
                mark="rule",
                encoding={
                    "x": _qq_line_cols["qq_line_x_start"],
                    "y": _qq_line_cols["qq_line_y_start"],
                    "x2": _qq_line_cols["qq_line_x_end"],
                    "y2": _qq_line_cols["qq_line_y_end"],
                },
                data_source=f"{_QQ_TRANSFORM_NAME}_{_QQ_LINE_KEY}",
            )
        )
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names(
    "qq",
    frozenset(
        {
            "point",
            "reference",
        }
    ),
)


def desugar_raster(
    x_field: str | None,
    y_field: str | None,
    *,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str | None = None,
    resolution: Any = "screen",
    blend: str = "alpha",
    min_count: Optional[int] = None,
    log_scale: bool = False,
) -> MarkDesugarResult:
    """Datashader-style raster aggregation mark desugar.

    Converts ``chart.mark_raster(...)`` into a ``Raster`` transform plus an
    image layer.  Suitable for rendering very large point clouds (millions of
    rows) as a pixel grid without overplotting.

    Data contract
    -------------
    Input: DataFrame with numeric columns ``x_field`` and ``y_field``.

    ``Raster`` (named ``"raster"``) bins data onto a pixel grid and produces
    a single-row summary batch consumed by the image renderer:
    ``[x_min, x_max, y_min, y_max (Float64), width, height (UInt32),
    pixel_data (Binary)]``

    Layers emitted
    --------------
    1. ``image`` — ``x=x_field``, ``y=y_field``,
       ``mark_kwargs={"cmap": cmap}``.

    Parameters
    ----------
    x_field : str or None
        Numeric x field. Required.
    y_field : str or None
        Numeric y field. Required.
    aggregate : {"count", "density", "mean", "sum", "any"}, default "count"
        Aggregation function applied in each pixel cell.  ``"mean"`` and
        ``"sum"`` require ``field``; ``"count"``, ``"density"``, and
        ``"any"`` need none.
    field : str or None, default None
        Column to aggregate for ``mean`` or ``sum``; required when
        ``aggregate`` is ``"mean"`` or ``"sum"``, ignored otherwise.
    cmap : str or None, default None
        Colormap applied to the image layer.  ``None`` defers to the theme's
        sequential scheme.
    resolution : int, tuple of (int, int), or "screen", default "screen"
        Pixel grid resolution.  An int sets a square grid; a ``(nx, ny)``
        tuple sets width and height independently.  ``"screen"`` genuinely
        scales with a pre-layout estimate of the chart's dimensions (not an
        exact match for the rendered plot area — see ``mark_raster``'s
        docstring).
    blend : {"alpha", "additive"}, default "alpha"
        Pixel-level blending mode.  ``"additive"`` renders with
        ``mix-blend-mode:screen`` in the SVG output.
    min_count : int or None, default None
        Pixel cells with fewer than ``min_count`` points render as
        transparent.
    log_scale : bool, default False
        Whether to apply log-scale mapping to cell counts before colormap.

    Returns
    -------
    MarkDesugarResult
        Layered mode with a single image layer (``.layers`` set).

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``, or if
        ``aggregate`` is ``"mean"`` or ``"sum"`` and ``field`` is ``None``.

    Examples
    --------
    >>> result = desugar_raster("x", "y")
    >>> result.layers[0].mark
    'image'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_raster() requires .encode(x=..., y=...)")
    if aggregate in ("mean", "sum") and field is None:
        raise ValueError(f"mark_raster aggregate={aggregate!r} requires field=...")

    # Default to log-scale for count/density aggregates to improve contrast
    # when most cells are empty and a few have high counts.
    effective_log_scale = log_scale if log_scale else (aggregate in ("count", "density"))
    transforms = [
        Raster(
            x=x_field,
            y=y_field,
            aggregate=aggregate,
            field=field,
            resolution=resolution,
            min_count=min_count,
            log_scale=effective_log_scale,
            name="raster",
        )
    ]
    mk: dict = {}
    if cmap is not None:
        mk["cmap"] = cmap
    layers = [
        _Layer(
            name="image",
            mark="image",
            encoding={"x": x_field, "y": y_field},
            mark_kwargs=mk if mk else None,
            data_source="raster",
            blend="additive" if blend == "additive" else None,
        )
    ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names(
    "raster",
    frozenset(
        {
            "image",
        }
    ),
)


def desugar_hex(
    x_field: str | None,
    y_field: str | None,
    *,
    bin_size: Optional[float] = None,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str | None = None,
    stroke: Optional[str] = None,
    stroke_width: float = 0,
) -> MarkDesugarResult:
    """Hexagonal-bin mark desugar.

    Converts ``chart.mark_hex(...)`` into a ``Hex`` transform plus a polygon
    layer.  Each hexagon represents a spatial bin; color encodes the
    aggregated value.

    Data contract
    -------------
    Input: DataFrame with numeric columns ``x_field`` and ``y_field``.

    ``Hex`` (named ``"hex"``) bins data into a hexagonal grid and produces:
    ``[hex_x (Float64), hex_y (Float64), hex_id (Int64),
    <aggregate column>]``

    Layers emitted
    --------------
    1. ``polygon`` — ``x="hex_x"``, ``y="hex_y"``,
       ``mark_kwargs={"cmap": cmap, "detail": "hex_id"}``.

    Parameters
    ----------
    x_field : str or None
        Numeric x field. Required.
    y_field : str or None
        Numeric y field. Required.
    bin_size : float or None, default None
        Hexagon bin radius in data units.  ``None`` auto-selects.
    aggregate : str, default "count"
        Aggregation function applied per hex cell, validated against the
        live ``transform_registry("hex")["methods"]`` vocabulary (spec
        §4.6) -- currently ``"count"``, ``"mean"``, ``"sum"``, ``"min"``,
        ``"max"``, ``"median"``, ``"std"``, or ``"var"``.
    field : str or None, default None
        Column to aggregate for ``mean`` or ``sum``; required when
        ``aggregate`` is not ``"count"``.
    cmap : str or None, default None
        Colormap name applied to the polygon layer.  ``None`` defers to the
        theme's sequential scheme.
    stroke : str or None, default None
        Border color for hex cells.  Passed directly to the polygon layer's
        ``stroke`` mark kwarg, which the Rust polygon renderer reads via
        ``resolve_mark_style``.  A stroke color with ``stroke_width`` left at
        its ``0`` default renders **no visible border** — there is no call-time
        auto-bump of width from a stroke color (literal semantics, consistent
        with all other polygon-family marks).
    stroke_width : float, default 0
        Border width for hex cells in pixels.  Only produces a visible border
        when non-zero.  Passed directly to the polygon layer's
        ``stroke_width`` mark kwarg.

    Returns
    -------
    MarkDesugarResult
        Layered mode with a single polygon layer (``.layers`` set).

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``; if ``aggregate`` is
        not one of the eight valid functions; or if ``aggregate`` is not
        ``"count"`` and ``field`` is ``None``.

    Examples
    --------
    >>> result = desugar_hex("x", "y")
    >>> result.layers[0].mark
    'polygon'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_hex() requires .encode(x=..., y=...)")
    validate_choice("mark_hex", "aggregate", aggregate, transform_registry("hex")["methods"])
    if aggregate != "count" and field is None:
        raise ValueError(f"mark_hex aggregate={aggregate!r} requires field=...")
    from ferrum.encoding import X, Y

    transforms = [
        Hex(x=x_field, y=y_field, bin_size=bin_size, aggregate=aggregate, field=field, name="hex")
    ]
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of any of these columns raises here instead of
    # silently reading a stale name. "value" is Hex's aggregate output column
    # name regardless of which `aggregate=` was requested.
    _hex_cols = registry_output_columns("hex", ["hex_x", "hex_y", "hex_id", "value"])
    mk: dict = {"detail": _hex_cols["hex_id"], "opacity": 1.0}
    if cmap is not None:
        mk["cmap"] = cmap
    if stroke is not None:
        mk["stroke"] = stroke
    if stroke_width != 0:
        mk["stroke_width"] = stroke_width
    layers = [
        _Layer(
            name="polygon",
            mark="polygon",
            encoding={
                "x": X(_hex_cols["hex_x"], title=x_field),
                "y": Y(_hex_cols["hex_y"], title=y_field),
                "color": _hex_cols["value"],
            },
            mark_kwargs=mk,
            data_source="hex",
        )
    ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names(
    "hex",
    frozenset(
        {
            "polygon",
        }
    ),
)


def desugar_swarm(
    x_field: str | None,
    y_field: str | None,
    *,
    size: int = 4,
    orient: str = "vertical",
    spacing: float = 1.0,
    side: str = "both",
    dodge: Optional[str] = None,
    x_sort: Any = None,
    y_sort: Any = None,
) -> MarkDesugarResult:
    """Beeswarm plot mark desugar.

    Converts ``chart.mark_swarm(...)`` into a ``Swarm`` transform plus a
    point layer.  Points are displaced along the cross-axis to avoid overlap
    while preserving their position on the value axis.

    Data contract
    -------------
    Input: DataFrame with categorical column (``x_field`` when vertical, or
    ``y_field`` when horizontal) and numeric value column (the other).

    ``Swarm`` (named ``"swarm"``) computes non-overlapping positions and
    emits the original columns plus:
    ``[swarm_x (Float64, nullable), swarm_y (Float64, nullable),
    __pos_x_offset__ (Float64)]`` for vertical, or
    ``__pos_y_offset__ (Float64)`` for horizontal.

    For ``orient="vertical"``, the renderer uses the original ``cat``/``val``
    columns for axis labeling, and ``__pos_x_offset__`` shifts each point
    within the category band.  For ``orient="horizontal"``, the layer encodes
    the original ``val``/``cat`` fields and the renderer applies
    ``__pos_y_offset__`` to spread points along the y axis.

    Layers emitted
    --------------
    1. ``point`` — vertical: ``x=cat``, ``y=val``; horizontal:
       ``x=val``, ``y=cat``.  Both read from ``data_source="swarm"``.

    Parameters
    ----------
    x_field : str or None
        Field name on the x axis (category when vertical). Required.
    y_field : str or None
        Field name on the y axis (value when vertical). Required.
    size : int, default 4
        Point diameter in pixels (the transform's placement radius is
        ``size / 2``).  The rendered pixel *spread* along the cross-axis
        (not this diameter) responds to a pre-layout estimate of the
        chart's *value-axis* panel dimension (height when vertical, width
        when ``orient="horizontal"``), inversely: a longer value axis
        needs less displacement to avoid collisions and produces a smaller
        spread at the same ``size``.  The cross-axis dimension has no
        effect.
    orient : {"vertical", "horizontal"}, default "vertical"
        Swarm orientation.  ``"vertical"`` means the value axis is y.
    spacing : float, default 1.0
        Minimum spacing multiplier between adjacent points.  Subject to the
        same value-axis-dependent spread as ``size`` above.
    side : {"both", "left", "right"}, default "both"
        Which side(s) of the category axis to spread points onto.
    dodge : str or None, default None
        Column name to sub-group by within each category band.  Each
        dodge group is offset side-by-side so groups do not overlap.
    x_sort, y_sort : optional
        Sort order applied to the categorical positional axis, injected by
        the composite-mark expansion from ``sort=`` on the encoding.

    Returns
    -------
    MarkDesugarResult
        Layered mode with a single point layer (``.layers`` set).

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_swarm("species", "petal_length")
    >>> result.layers[0].mark
    'point'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_swarm() requires .encode(x=..., y=...)")
    cat = x_field if orient == "vertical" else y_field
    val = y_field if orient == "vertical" else x_field
    # Resolve the categorical axis sort.  When vertical, x is categorical
    # (x_sort applies); when horizontal, y is categorical (y_sort applies).
    cat_sort = x_sort if orient == "vertical" else y_sort
    swarm_kwargs: dict = dict(
        category=cat,
        value=val,
        point_size=float(size),
        spacing=spacing,
        side=side,
        orient=orient,
        name="swarm",
    )
    if dodge is not None:
        swarm_kwargs["dodge"] = str(dodge)
    transforms = [Swarm(**swarm_kwargs)]
    from ferrum.encoding import X, Y

    if orient == "vertical":
        # Encode the chart's original category & value fields so the ordinal x
        # axis renders properly with the category labels. The Swarm transform
        # emits `__pos_x_offset__` (pixel offset on the cross axis) which the
        # renderer's standard position-offset path applies on top of the category
        # band center — same mechanism Dodge uses (Phase 9c).
        x_enc_val = X(cat, sort=cat_sort) if cat_sort is not None else cat
        layers = [
            _Layer(
                name="point",
                mark="point",
                encoding={"x": x_enc_val, "y": val},
                data_source="swarm",
            )
        ]
    else:
        # Horizontal swarm: value axis = x, category axis = y.
        # The Rust transform receives orient="horizontal" and emits `__pos_y_offset__`
        # (pixel offset on the y cross-axis) instead of `__pos_x_offset__`. Encoding
        # the original field names lets the ordinal y axis render category labels
        # correctly, and the renderer picks up `__pos_y_offset__` automatically via
        # `render/position.rs::read_position_offsets`.
        y_enc_val = Y(cat, sort=cat_sort) if cat_sort is not None else cat
        layers = [
            _Layer(
                name="point",
                mark="point",
                encoding={"x": val, "y": y_enc_val},
                data_source="swarm",
            )
        ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names(
    "swarm",
    frozenset(
        {
            "point",
        }
    ),
)


def desugar_function(
    fn,
    parent_chart_x_data=None,
    *,
    domain: Optional[tuple] = None,
    n: int = 200,
    clip: bool = True,
) -> MarkDesugarResult:
    """Arbitrary-function line mark desugar — the only synthetic-data desugar.

    Materializes a new Arrow table by evaluating ``fn`` over ``n`` evenly
    spaced x-values in ``domain``, then returns a single-mark result whose
    ``.data`` field carries the synthetic table.  No transforms are emitted.

    Data contract
    -------------
    Input: no input DataFrame required.  If ``domain`` is ``None``, the
    x-range is inferred from ``parent_chart_x_data`` (a numpy array of the
    parent chart's x-column values).

    Output (synthetic table): columns ``x`` (Float64), ``y`` (Float64)
    with ``n`` rows.

    Layers emitted
    --------------
    1. ``line`` — ``x="x"``, ``y="y"`` via the encoding remap
       ``{"x": "x", "y": "y"}``.  Returned in single-mark mode (``.mark``,
       ``.remap``, ``.data``) rather than the layered form.

    Parameters
    ----------
    fn : callable
        A function accepting a numpy array of shape ``(n,)`` and returning
        a numpy array of the same shape.
    parent_chart_x_data : numpy.ndarray or None, default None
        x-column values from the parent chart; used to infer ``domain``
        when ``domain`` is not supplied.
    domain : tuple[float, float] or None, default None
        Explicit ``(x_min, x_max)`` range to evaluate over.  Required if
        there is no parent chart with x data.
    n : int, default 200
        Number of evaluation points.
    clip : bool, default True
        Reserved for future use (no-op today).

    Returns
    -------
    MarkDesugarResult
        Single-mark mode with ``.mark="line"``, ``.remap={"x": "x", "y": "y"}``,
        and ``.data`` set to the synthetic ``pyarrow.Table``.

    Raises
    ------
    ValueError
        If ``domain`` is ``None`` and ``parent_chart_x_data`` is empty or
        ``None``, or if ``fn`` does not return a numpy array of shape
        ``(n,)``.

    Examples
    --------
    >>> import numpy as np
    >>> result = desugar_function(np.sin, domain=(0, 6.28))
    >>> result.mark
    'line'
    >>> result.data.num_rows
    200
    """
    if not clip:
        raise ValueError("mark_function(clip=False) is not supported; clipping is always enabled")
    import numpy as np
    import pyarrow as pa

    if domain is not None:
        d = domain
    elif parent_chart_x_data is not None and len(parent_chart_x_data) > 0:
        d = (float(np.nanmin(parent_chart_x_data)), float(np.nanmax(parent_chart_x_data)))
    else:
        raise ValueError(
            "mark_function requires explicit domain when chart has no other data layers"
        )

    xs = np.linspace(d[0], d[1], n)
    ys = fn(xs)
    if not isinstance(ys, np.ndarray) or ys.shape != (n,):
        raise ValueError(
            f"mark_function callable must return numpy array of shape ({n},); got shape {getattr(ys, 'shape', None)}"
        )

    synthetic = pa.Table.from_pydict({"x": xs, "y": ys})
    return MarkDesugarResult(mark="line", remap={"x": "x", "y": "y"}, data=synthetic)
