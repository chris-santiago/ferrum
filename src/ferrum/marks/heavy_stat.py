"""Heavy-stat-mark desugar helpers (Phase 8b Sub-batch F).

Each desugar_<name> returns the unified 4-tuple
    (mark, transforms, encoding_remap, synthetic_data)
or 5-tuple for layered:
    ("__layered__", transforms, None, None, layers)
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
from ferrum._layer import _Layer


def desugar_contour(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    thresholds: int = 6,
    smooth: bool = True,
    fill: bool = False,
    cmap: str = "viridis",
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
    cmap : str, default "viridis"
        Colormap name applied to the polygon layer.

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
    # downstream polygon layer can route through data_source="contour".
    transforms = [
        Kde2D(x=x_field, y=y_field, bandwidth=bandwidth, n=128),
        Contour(thresholds=thresholds, fill=fill, smooth=smooth, name="contour"),
    ]
    layers = [
        _Layer(
            mark="polygon",
            encoding={"x": "contour_x", "y": "contour_y"},
            mark_kwargs={"cmap": cmap, "detail": "level_id"},
            data_source="contour",
        )
    ]
    return ("__layered__", transforms, None, None, layers)


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

    transforms = [Violin(field=y_field, groupby=[x_field], bandwidth=bandwidth, name="violin")]
    violin_layer = _Layer(
        mark="polygon",
        encoding={"x": x_field, "y": "violin_y"},
        mark_kwargs={"detail": "group_id", "fill_opacity": 0.5},
        data_source="violin",
    )
    if inner is None:
        return ("__layered__", transforms, None, None, [violin_layer])
    if inner == "point":
        return (
            "__layered__",
            transforms,
            None,
            None,
            [violin_layer, _Layer(mark="point", encoding={"x": x_field, "y": y_field})],
        )
    if inner == "quartile":
        transforms.append(BoxStats(field=y_field, groupby=[x_field], name="quart"))
        layers = [violin_layer]
        for col in ("q1", "median", "q3"):
            mk = {} if col == "median" else {"stroke_dash": [2, 2]}
            layers.append(
                _Layer(
                    mark="rule",
                    encoding={"x": x_field, "y": col},
                    mark_kwargs=mk if mk else None,
                    data_source="quart",
                )
            )
        return ("__layered__", transforms, None, None, layers)
    # inner == "box"
    from ferrum.marks.composite import desugar_boxplot

    _, box_t, _, _, box_layers = desugar_boxplot(
        x_field, y_field, extent=1.5, outliers=False, size=0.1
    )
    return ("__layered__", [*transforms, *box_t], None, None, [violin_layer, *box_layers])


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
    layers = [
        _Layer(
            mark="point",
            encoding={"x": "theoretical", "y": "sample"},
            data_source="qq_main",
        )
    ]
    if line:
        layers.append(
            _Layer(
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
    return ("__layered__", transforms, None, None, layers)


def desugar_raster(
    x_field: str | None,
    y_field: str | None,
    *,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str = "viridis",
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
    cmap : str, default "viridis"
        Colormap applied to the image layer.
    resolution : int or "screen", default "screen"
        Pixel grid resolution.  ``"screen"`` infers from chart dimensions.
    blend : {"alpha", "additive"}, default "alpha"
        Pixel-level blending mode.  ``"additive"`` emits a ``warn_once``
        and falls back to ``"alpha"`` (deferred to Phase 11).
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
    if blend == "additive":
        from ferrum._warn import warn_once

        warn_once(
            "mark_raster",
            "blend_additive",
            "mark_raster blend='additive' deferred to Phase 11; using alpha blending",
        )

    transforms = [
        Raster(
            x=x_field,
            y=y_field,
            aggregate=aggregate,
            field=field,
            resolution=resolution,
            min_count=min_count,
            log_scale=log_scale,
            name="raster",
        )
    ]
    layers = [
        _Layer(
            mark="image",
            encoding={"x": x_field, "y": y_field},
            mark_kwargs={"cmap": cmap},
            data_source="raster",
        )
    ]
    return ("__layered__", transforms, None, None, layers)


def desugar_hex(
    x_field: str | None,
    y_field: str | None,
    *,
    bin_size: Optional[float] = None,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str = "viridis",
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
    cmap : str, default "viridis"
        Colormap name applied to the polygon layer.
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
    if aggregate in ("mean", "sum") and field is None:
        raise ValueError(f"mark_hex aggregate={aggregate!r} requires field=...")
    if aggregate not in ("count", "mean", "sum"):
        from ferrum._warn import warn_once

        warn_once(
            "mark_hex",
            "aggregate_unsupported",
            f"mark_hex aggregate={aggregate!r} deferred; falling back to 'count'",
        )
        aggregate = "count"
    transforms = [
        Hex(x=x_field, y=y_field, bin_size=bin_size, aggregate=aggregate, field=field, name="hex")
    ]
    layers = [
        _Layer(
            mark="polygon",
            encoding={"x": "hex_x", "y": "hex_y"},
            mark_kwargs={"cmap": cmap, "detail": "hex_id"},
            data_source="hex",
        )
    ]
    return ("__layered__", transforms, None, None, layers)


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
    __pos_x_offset__ (Float64)]``

    For ``orient="vertical"``, the renderer uses the original ``cat``/``val``
    columns for axis labeling, and ``__pos_x_offset__`` shifts each point
    within the category band.  For ``orient="horizontal"``, the layer encodes
    ``x="swarm_x"``, ``y="swarm_y"`` directly.

    Layers emitted
    --------------
    1. ``point`` — vertical: ``x=cat``, ``y=val``; horizontal:
       ``x="swarm_x"``, ``y="swarm_y"``.  Both read from ``data_source="swarm"``.

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
        Reserved for future use.  Emits a ``warn_once`` and renders as a
        single-group swarm (no-op today).

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
    if dodge is not None:
        from ferrum._warn import warn_once

        warn_once(
            "mark_swarm",
            "dodge",
            "mark_swarm dodge= is not yet supported; rendering single-group swarm",
        )
    cat = x_field if orient == "vertical" else y_field
    val = y_field if orient == "vertical" else x_field
    transforms = [
        Swarm(
            category=cat,
            value=val,
            point_size=float(size),
            spacing=spacing,
            side=side,
            name="swarm",
        )
    ]
    if orient == "vertical":
        # Encode the chart's original category & value fields so the ordinal x
        # axis renders properly with the category labels. The Swarm transform
        # also emits a `__pos_x_offset__` column (pixel offset on the cross axis)
        # which the renderer's standard position-offset path applies on top of
        # the category band center — same mechanism Dodge uses (Phase 9c).
        layers = [
            _Layer(
                mark="point",
                encoding={"x": cat, "y": val},
                data_source="swarm",
            )
        ]
    else:
        # TODO(phase-10g+): horizontal-orient swarm still uses the legacy
        # value-axis-data-unit encoding. Fixing it requires either threading
        # orient through the Rust transform so it emits __pos_y_offset__ instead,
        # or adding a column-rename step in the Python pipeline. Lightly tested
        # path; smoke render still produces an SVG.
        layers = [
            _Layer(
                mark="point",
                encoding={"x": "swarm_x", "y": "swarm_y"},
                data_source="swarm",
            )
        ]
    return ("__layered__", transforms, None, None, layers)


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
    return ("line", [], {"x": "x", "y": "y"}, synthetic)
