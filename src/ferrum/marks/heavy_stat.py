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
from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names


def desugar_contour(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    thresholds: int = 6,
    smooth: bool = True,
    fill: bool = True,
    cmap: str | None = None,
) -> tuple:
    """Bivariate-density contour mark desugar.

    Converts ``chart.mark_contour(...)`` into a ``Kde2D`` → ``Contour``
    transform chain plus a polygon layer.  Also used by
    ``desugar_density`` when both x and y encodings are present.

    Data contract
    -------------
    Input: DataFrame with numeric columns ``x_field`` and ``y_field``.

    ``Kde2D`` (unnamed — advances the chain) estimates a 2D kernel-density
    surface on a 128×128 grid.

    ``Contour`` (named ``"contour"``) traces iso-density curves on that
    surface and produces:
    ``[level_id (UInt32), contour_x (Float64), contour_y (Float64)]``

    Layers emitted
    --------------
    1. ``polygon`` — ``x="contour_x"``, ``y="contour_y"``,
       ``mark_kwargs={"cmap": cmap, "detail": "level_id"}``.

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
    cmap : str or None, default None
        Colormap name applied to the polygon layer.  ``None`` defers to the
        theme's sequential scheme.

    Returns
    -------
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_contour("x", "y")
    >>> result[0]
    '__layered__'
    >>> result[4][0]["mark"]
    'polygon'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_contour() requires .encode(x=..., y=...)")
    # Kde2D is UNNAMED so it advances the chain (current → Kde2D output);
    # Contour then runs on the chained Kde2D output. Contour is named so the
    # downstream layer can route through data_source="contour".
    transforms = [
        Kde2D(x=x_field, y=y_field, bandwidth=bandwidth, n=128),
        Contour(thresholds=thresholds, fill=fill, smooth=smooth, name="contour"),
    ]
    if fill:
        # Isoband mode: Contour emits polygon vertex rows (x, y per vertex).
        # Use polygon mark grouped by level_id; color by level_value so each
        # density band gets a distinct fill from the sequential colormap.
        mk: dict = {"detail": "level_id"}
        if cmap is not None:
            mk["cmap"] = cmap
        layers = [
            _Layer(
                name="polygon",
                mark="polygon",
                encoding={"x": "contour_x", "y": "contour_y", "color": "level_value"},
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
                    "x": "contour_x",
                    "y": "contour_y",
                    "x2": "contour_x2",
                    "y2": "contour_y2",
                },
                mark_kwargs=mk if mk else None,
                data_source="contour",
            )
        ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names("contour", frozenset({
    "polygon", "segment",
}))


def desugar_violin(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    inner: Optional[str] = "box",
) -> tuple:
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
      ``extent=1.5``, ``outliers=False``, ``size=0.1``.

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

    Returns
    -------
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``, or if ``inner``
        is not one of the valid options.

    Examples
    --------
    >>> result = desugar_violin("species", "petal_length", inner=None)
    >>> result[4][0]["mark"]
    'polygon'
    >>> len(result[4])
    1
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_violin() requires .encode(x=..., y=...)")
    if inner not in ("box", "quartile", "point", None):
        raise ValueError(
            f"mark_violin inner must be one of 'box', 'quartile', 'point', or None; got {inner!r}"
        )

    from ferrum.encoding import Y

    transforms = [Violin(field=y_field, groupby=[x_field], bandwidth=bandwidth, name="violin")]
    violin_layer = _Layer(
        name="body",
        mark="polygon",
        encoding={"x": x_field, "y": Y("violin_y", title=y_field)},
        mark_kwargs={"detail": "group_id", "fill_opacity": 0.5},
        data_source="violin",
    )
    if inner is None:
        return MarkDesugarResult(transforms=transforms, layers=[violin_layer])
    if inner == "point":
        return MarkDesugarResult(
            transforms=transforms,
            layers=[violin_layer, _Layer(name="point", mark="point", encoding={"x": x_field, "y": y_field})],
        )
    if inner == "quartile":
        transforms.append(BoxStats(field=y_field, groupby=[x_field], name="quart"))
        layers = [violin_layer]
        for col in ("q1", "median", "q3"):
            mk = {} if col == "median" else {"stroke_dash": [2, 2]}
            layers.append(
                _Layer(
                    name=col,
                    mark="rule",
                    encoding={"x": x_field, "y": col},
                    mark_kwargs=mk if mk else None,
                    data_source="quart",
                )
            )
        return MarkDesugarResult(transforms=transforms, layers=layers)
    # inner == "box"
    from ferrum.marks.composite import desugar_boxplot

    box_result = desugar_boxplot(
        x_field, y_field, extent=1.5, outliers=False, size=0.1
    )
    return MarkDesugarResult(
        transforms=[*transforms, *box_result.transforms],
        layers=[violin_layer, *box_result.layers],
    )


register_layer_names("violin", frozenset({
    "body", "point", "q1", "median", "q3",
    "whisker", "lower_cap", "upper_cap", "box", "outlier",
}))


def desugar_qq(
    field: str,
    *,
    distribution: str = "normal",
    dequantize: bool = False,
    line: bool = True,
) -> tuple:
    """Q-Q (quantile-quantile) plot mark desugar.

    Converts ``chart.mark_qq(...)`` into a ``QQ`` transform plus a point
    layer and an optional reference-line rule layer.

    Data contract
    -------------
    Input: DataFrame with numeric column ``field``.

    ``QQ`` (named ``"qq_main"``) computes theoretical vs. sample quantiles
    and produces:
    ``[theoretical (Float64), sample (Float64)]``

    When ``line=True``, ``QQ`` also emits a secondary named output
    ``"qq_line"`` (a single-row batch) with columns:
    ``[qq_line_x_start, qq_line_x_end, qq_line_y_start, qq_line_y_end]``

    Layers emitted
    --------------
    1. ``point`` — ``x="theoretical"``, ``y="sample"`` (from ``"qq_main"``).
    2. ``rule``  — ``x="qq_line_x_start"``, ``y="qq_line_y_start"``,
       ``x2="qq_line_x_end"``, ``y2="qq_line_y_end"``
       (from ``"qq_line"``).  Only when ``line=True``.

    Parameters
    ----------
    field : str
        Numeric column to compute quantiles from.
    distribution : {"normal", "uniform", "exponential"}, default "normal"
        Theoretical distribution to compare against.
    dequantize : bool, default False
        Whether to jitter duplicate quantile values slightly.
    line : bool, default True
        Whether to add the Q-Q reference line.

    Returns
    -------
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``.

    Raises
    ------
    ValueError
        If ``distribution`` is not one of the valid options.

    Examples
    --------
    >>> result = desugar_qq("residuals")
    >>> [l.mark for l in result[4]]
    ['point', 'rule']
    """
    if distribution not in ("normal", "uniform", "exponential"):
        raise ValueError(
            f"mark_qq distribution must be 'normal', 'uniform', or 'exponential'; got {distribution!r}"
        )
    transforms = [
        QQ(
            field=field,
            distribution=distribution,
            dequantize=dequantize,
            emit_line=line,
            name="qq_main",
        )
    ]
    from ferrum.encoding import X, Y

    layers = [
        _Layer(
            name="point",
            mark="point",
            encoding={
                "x": X("theoretical", title="Theoretical Quantiles"),
                "y": Y("sample", title="Sample Quantiles"),
            },
            data_source="qq_main",
        )
    ]
    if line:
        layers.append(
            _Layer(
                name="reference",
                mark="rule",
                encoding={
                    "x": "qq_line_x_start",
                    "y": "qq_line_y_start",
                    "x2": "qq_line_x_end",
                    "y2": "qq_line_y_end",
                },
                data_source="qq_line",
            )
        )
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names("qq", frozenset({
    "point", "reference",
}))


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
) -> tuple:
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
    aggregate : {"count", "mean", "sum"}, default "count"
        Aggregation function applied in each pixel cell.
    field : str or None, default None
        Column to aggregate for ``mean`` or ``sum``; required when
        ``aggregate`` is not ``"count"``.
    cmap : str or None, default None
        Colormap applied to the image layer.  ``None`` defers to the theme's
        sequential scheme.
    resolution : int or "screen", default "screen"
        Pixel grid resolution.  ``"screen"`` infers from chart dimensions.
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
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``, or if
        ``aggregate`` is ``"mean"`` or ``"sum"`` and ``field`` is ``None``.

    Examples
    --------
    >>> result = desugar_raster("x", "y")
    >>> result[4][0].mark
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


register_layer_names("raster", frozenset({
    "image",
}))


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
) -> tuple:
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
    aggregate : {"count", "mean", "sum"}, default "count"
        Aggregation function applied per hex cell.
    field : str or None, default None
        Column to aggregate for ``mean`` or ``sum``; required when
        ``aggregate`` is not ``"count"``.
    cmap : str or None, default None
        Colormap name applied to the polygon layer.  ``None`` defers to the
        theme's sequential scheme.
    stroke : str or None, default None
        Reserved for future use (no-op today).
    stroke_width : float, default 0
        Reserved for future use (no-op today).

    Returns
    -------
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``, or if
        ``aggregate`` is ``"mean"`` or ``"sum"`` and ``field`` is ``None``.

    Notes
    -----
    Aggregate values other than ``"count"``, ``"mean"``, and ``"sum"`` emit
    a ``warn_once`` and fall back to ``"count"``.

    Examples
    --------
    >>> result = desugar_hex("x", "y")
    >>> result[4][0].mark
    'polygon'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_hex() requires .encode(x=..., y=...)")
    if stroke is not None:
        raise ValueError(f"mark_hex(stroke={stroke!r}) is not supported")
    if stroke_width != 0:
        raise ValueError(f"mark_hex(stroke_width={stroke_width!r}) is not supported")
    _VALID_AGGREGATES = ("count", "mean", "sum", "min", "max", "median", "std", "var")
    if aggregate not in _VALID_AGGREGATES:
        raise ValueError(
            f"mark_hex aggregate={aggregate!r} must be one of {_VALID_AGGREGATES}"
        )
    if aggregate != "count" and field is None:
        raise ValueError(f"mark_hex aggregate={aggregate!r} requires field=...")
    transforms = [
        Hex(x=x_field, y=y_field, bin_size=bin_size, aggregate=aggregate, field=field, name="hex")
    ]
    mk: dict = {"detail": "hex_id", "opacity": 1.0}
    if cmap is not None:
        mk["cmap"] = cmap
    layers = [
        _Layer(
            name="polygon",
            mark="polygon",
            encoding={"x": "hex_x", "y": "hex_y", "color": "value"},
            mark_kwargs=mk,
            data_source="hex",
        )
    ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names("hex", frozenset({
    "polygon",
}))


def desugar_swarm(
    x_field: str | None,
    y_field: str | None,
    *,
    size: int = 4,
    orient: str = "vertical",
    spacing: float = 1.0,
    side: str = "both",
    dodge: Optional[str] = None,
) -> tuple:
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
        Point radius in pixels.
    orient : {"vertical", "horizontal"}, default "vertical"
        Swarm orientation.  ``"vertical"`` means the value axis is y.
    spacing : float, default 1.0
        Minimum spacing multiplier between adjacent points.
    side : {"both", "left", "right"}, default "both"
        Which side(s) of the category axis to spread points onto.
    dodge : str or None, default None
        Column name to sub-group by within each category band.  Each
        dodge group is offset side-by-side so groups do not overlap.

    Returns
    -------
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_swarm("species", "petal_length")
    >>> result[4][0].mark
    'point'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_swarm() requires .encode(x=..., y=...)")
    cat = x_field if orient == "vertical" else y_field
    val = y_field if orient == "vertical" else x_field
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
    if orient == "vertical":
        # Encode the chart's original category & value fields so the ordinal x
        # axis renders properly with the category labels. The Swarm transform
        # emits `__pos_x_offset__` (pixel offset on the cross axis) which the
        # renderer's standard position-offset path applies on top of the category
        # band center — same mechanism Dodge uses (Phase 9c).
        layers = [
            _Layer(
                name="point",
                mark="point",
                encoding={"x": cat, "y": val},
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
        layers = [
            _Layer(
                name="point",
                mark="point",
                encoding={"x": val, "y": cat},
                data_source="swarm",
            )
        ]
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names("swarm", frozenset({
    "point",
}))


def desugar_function(
    fn,
    parent_chart_x_data=None,
    *,
    domain: Optional[tuple] = None,
    n: int = 200,
    clip: bool = True,
) -> tuple:
    """Arbitrary-function line mark desugar — the only synthetic-data desugar.

    Materializes a new Arrow table by evaluating ``fn`` over ``n`` evenly
    spaced x-values in ``domain``, then returns a 4-tuple whose fourth
    element is the synthetic table.  No transforms are emitted.

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
       ``{"x": "x", "y": "y"}``.  Returned as the legacy 4-tuple
       ``(mark, transforms, remap, synthetic)`` rather than the 5-tuple
       layered form.

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
    tuple
        4-tuple ``("line", [], {"x": "x", "y": "y"}, synthetic_table)``
        where ``synthetic_table`` is a ``pyarrow.Table``.

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
    >>> result[0]
    'line'
    >>> result[3].num_rows
    200
    """
    if not clip:
        raise ValueError(
            "mark_function(clip=False) is not supported; clipping is always enabled"
        )
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
