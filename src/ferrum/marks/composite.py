"""Composite-mark desugar helpers (Phase 8b).

Each desugar_<name> returns a 5-tuple:
    ("__layered__", transforms: list, _ignored: None, _ignored: None, layers: list)

Where `layers` is a list of ``ferrum._layer._Layer`` instances.
"""

from __future__ import annotations
from typing import Any, Optional

from ferrum import BoxStats, ErrorExtent, LetterValue, Outliers
from ferrum._layer import _Layer
from ferrum._overrides import register_layer_names
from ferrum.encoding import X, Y


def desugar_boxplot(
    x_field: str | None,
    y_field: str | None,
    *,
    extent: float | str = 1.5,
    outliers: bool = True,
    size: Optional[float] = None,
    color_field: Optional[str] = None,
    horizontal: bool = False,
) -> tuple:
    """Box-plot composite mark desugar.

    Converts ``chart.mark_boxplot(...)`` into a ``BoxStats`` transform plus
    five (or six) primitive layers: a whisker rule, lower and upper whisker
    caps, an IQR rect, a median tick, and an optional outlier point layer.

    Data contract
    -------------
    Input: any DataFrame with categorical column ``x_field`` and numeric
    column ``y_field`` (or the reverse when ``horizontal=True``).

    Output — ``BoxStats`` named ``"box"`` produces:
    ``[<groupby cols>, q1, median, q3, lower_whisker, upper_whisker]``

    When ``outliers=True``, an ``Outliers`` transform named ``"outliers"``
    produces rows of the original ``val`` column for observations outside
    the whiskers: ``[<groupby cols>, <val>]``.

    Layers emitted
    --------------
    1. ``rule``   — ``y=lower_whisker``, ``y2=upper_whisker`` (whiskers).
    2. ``tick``   — ``y=lower_whisker``, ``band_size=0.3`` (lower cap).
    3. ``tick``   — ``y=upper_whisker``, ``band_size=0.3`` (upper cap).
    4. ``rect``   — ``y=q1``, ``y2=q3``, axis title set to original field
       name (IQR box, ``width=size``).
    5. ``tick``   — ``y=median``, dark stroke (median line).
    6. ``point``  — ``y=<val>``, ``filled=False`` from the ``"outliers"``
       output (when ``outliers=True``).

    Parameters
    ----------
    x_field : str or None
        Categorical (grouping) field name. Required.
    y_field : str or None
        Numeric (value) field name. Required.
    extent : float or "min-max", default 1.5
        IQR multiplier for whisker length, or ``"min-max"`` to extend
        whiskers to the data minimum/maximum.
    outliers : bool, default True
        Whether to overlay an outlier point layer.
    size : float or None, default None
        Box half-width as a fraction of the band (default ``0.6``).
    color_field : str or None, default None
        Optional column to use for color encoding (also added to the
        ``groupby`` list for the transforms).
    horizontal : bool, default False
        If ``True``, flip axes so that categories are on the y-axis.

    Returns
    -------
    tuple
        5-tuple ``("__layered__", transforms, None, None, layers)``
        consumed by ``Chart._resolve_pending``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_boxplot("species", "sepal_length")
    >>> result[0]
    '__layered__'
    >>> len(result[4])  # 6 layers: whisker, lower cap, upper cap, box, median, outlier
    6
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_boxplot() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    groupby = [cat] + ([color_field] if color_field else [])

    transforms = [
        BoxStats(field=val, groupby=groupby, whisker_extent=_extent_to_box(extent), name="box")
    ]
    if outliers:
        transforms.append(
            Outliers(field=val, groupby=groupby, extent=_extent_to_iqr_k(extent), name="outliers")
        )

    band = size or 0.6

    def enc(y_col, y2_col=None, *, title=None):
        if horizontal:
            d: dict = {"x": X(y_col, title=title) if title else y_col, "y": cat}
            if y2_col:
                d["x2"] = y2_col
        else:
            d = {"x": cat, "y": Y(y_col, title=title) if title else y_col}
            if y2_col:
                d["y2"] = y2_col
        return d

    layers = [
        _Layer(name="whisker", mark="rule", encoding=enc("lower_whisker", "upper_whisker", title=val), data_source="box"),
        _Layer(name="lower_cap", mark="tick", encoding=enc("lower_whisker"), mark_kwargs={"band_size": 0.3}, data_source="box"),
        _Layer(name="upper_cap", mark="tick", encoding=enc("upper_whisker"), mark_kwargs={"band_size": 0.3}, data_source="box"),
        _Layer(
            name="box", mark="rect", encoding=enc("q1", "q3", title=val), mark_kwargs={"band_size": band}, data_source="box"
        ),
        _Layer(
            name="median",
            mark="tick",
            encoding=enc("median"),
            mark_kwargs={"band_size": band, "stroke": "#222222", "stroke_width": 2},
            data_source="box",
        ),
    ]
    if outliers:
        layers.append(_Layer(name="outlier", mark="point", encoding=enc(val), mark_kwargs={"filled": False}, data_source="outliers"))

    return ("__layered__", transforms, None, None, layers)


register_layer_names("boxplot", frozenset({
    "whisker", "lower_cap", "upper_cap", "box", "median", "outlier",
}))


def desugar_errorbar(
    x_field: str | None,
    y_field: str | None,
    *,
    extent: str = "ci",
    ticks: bool = True,
) -> tuple:
    """Error-bar composite mark desugar.

    Converts ``chart.mark_errorbar(...)`` into an ``ErrorExtent`` transform
    plus a ranged-rule layer (and optional endpoint tick layers).

    Data contract
    -------------
    Input: DataFrame with categorical column ``x_field`` and numeric column
    ``y_field`` (one row per observation; the transform aggregates).

    Output — ``ErrorExtent`` named ``"err"`` produces:
    ``[<x_field>, lower, upper]``

    Layers emitted
    --------------
    1. ``rule``  — ``y=lower``, ``y2=upper`` (vertical error span).
    2. ``tick``  — ``y=lower`` (bottom cap, ``band_size=0.3``).  When
       ``ticks=True`` only.
    3. ``tick``  — ``y=upper`` (top cap, ``band_size=0.3``).  When
       ``ticks=True`` only.

    Parameters
    ----------
    x_field : str or None
        Categorical (grouping) field. Required.
    y_field : str or None
        Numeric value field. Required.
    extent : str, default "ci"
        Aggregation method passed to ``ErrorExtent``.  One of ``"ci"``
        (95 % bootstrap CI), ``"stderr"``, ``"stdev"``, or ``"iqr"``.
    ticks : bool, default True
        Whether to add endpoint tick marks at the top and bottom of each
        error bar.

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
    >>> result = desugar_errorbar("day", "tip")
    >>> result[0]
    '__layered__'
    >>> [l.mark for l in result[4]]
    ['rule', 'tick', 'tick']
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_errorbar() requires .encode(x=..., y=...)")
    transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
    layers = [
        _Layer(
            name="rule",
            mark="rule",
            encoding={"x": x_field, "y": "lower", "y2": "upper"},
            data_source="err",
        ),
    ]
    if ticks:
        layers.extend(
            [
                _Layer(
                    name="lower_cap",
                    mark="tick",
                    encoding={"x": x_field, "y": "lower"},
                    mark_kwargs={"band_size": 0.3},
                    data_source="err",
                ),
                _Layer(
                    name="upper_cap",
                    mark="tick",
                    encoding={"x": x_field, "y": "upper"},
                    mark_kwargs={"band_size": 0.3},
                    data_source="err",
                ),
            ]
        )
    return ("__layered__", transforms, None, None, layers)


register_layer_names("errorbar", frozenset({
    "rule", "lower_cap", "upper_cap",
}))


def desugar_errorband(
    x_field: str | None,
    y_field: str | None,
    *,
    extent: str = "ci",
    borders: bool = False,
) -> tuple:
    """Error-band (shaded CI ribbon) composite mark desugar.

    Converts ``chart.mark_errorband(...)`` into an ``ErrorExtent`` transform
    plus a translucent ribbon layer (and optional border line layers).

    Data contract
    -------------
    Input: DataFrame with continuous ``x_field`` and numeric ``y_field``
    (one row per observation; the transform aggregates by x).

    Output — ``ErrorExtent`` named ``"err"`` produces:
    ``[<x_field>, lower, upper]``

    Layers emitted
    --------------
    1. ``ribbon`` — ``y=lower``, ``y2=upper``, ``opacity=0.2, stroke="none"`` (shaded band).
    2. ``line``   — ``y=lower`` (bottom border).  When ``borders=True`` only.
    3. ``line``   — ``y=upper`` (top border).  When ``borders=True`` only.

    Parameters
    ----------
    x_field : str or None
        Continuous x field. Required.
    y_field : str or None
        Numeric value field. Required.
    extent : str, default "ci"
        Aggregation method passed to ``ErrorExtent``.  One of ``"ci"``,
        ``"stderr"``, ``"stdev"``, or ``"iqr"``.
    borders : bool, default False
        Whether to draw solid border lines at the upper and lower edges of
        the ribbon.

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
    >>> result = desugar_errorband("x", "y")
    >>> result[0]
    '__layered__'
    >>> result[4][0].mark
    'ribbon'
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_errorband() requires .encode(x=..., y=...)")
    transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
    layers = [
        _Layer(
            name="ribbon",
            mark="ribbon",
            encoding={"x": x_field, "y": Y("lower", title=y_field), "y2": "upper"},
            mark_kwargs={"opacity": 0.2, "stroke": "none"},
            data_source="err",
        ),
    ]
    if borders:
        layers.extend(
            [
                _Layer(name="lower_border", mark="line", encoding={"x": x_field, "y": "lower"}, data_source="err"),
                _Layer(name="upper_border", mark="line", encoding={"x": x_field, "y": "upper"}, data_source="err"),
            ]
        )
    return ("__layered__", transforms, None, None, layers)


register_layer_names("errorband", frozenset({
    "ribbon", "lower_border", "upper_border",
}))


def desugar_ribbon(
    x_field: str | None,
    y_field: str | None,
    *,
    y2_field: str | None = None,
    opacity: float = 0.2,
    interpolate: str = "linear",
) -> tuple:
    """Primitive ribbon (shaded band) mark desugar — no transform.

    Emits a single ribbon layer directly.  Unlike ``desugar_errorband``,
    no aggregation transform is applied; the data must already carry the
    lower and upper bound columns.

    ``y2_field`` is resolved from the chart's encoding state and passed in
    by ``Chart._resolve_pending`` (the special case for ``kind == "ribbon"``)
    because ribbon is always called after ``.encode(x=..., y=..., y2=...)``.

    Data contract
    -------------
    Input: DataFrame pre-computed with columns ``x_field``, ``y_field``
    (lower bound), and ``y2_field`` (upper bound).  No transforms emitted.

    Layers emitted
    --------------
    1. ``ribbon`` — ``x=x_field``, ``y=y_field``, ``y2=y2_field``,
       ``opacity=opacity``, ``stroke="none"``.

    Parameters
    ----------
    x_field : str or None
        Continuous x field. Required.
    y_field : str or None
        Lower-bound y field. Required.
    y2_field : str or None, default None
        Upper-bound y field.  Required (must be supplied by the caller
        from the chart's encoding state).
    opacity : float, default 0.2
        Ribbon fill opacity.
    interpolate : str, default "linear"
        Reserved for future use (no-op today — the renderer uses linear
        interpolation unconditionally).

    Returns
    -------
    tuple
        5-tuple ``("__layered__", [], None, None, layers)``.

    Raises
    ------
    ValueError
        If ``x_field``, ``y_field``, or ``y2_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_ribbon("x", "lower", y2_field="upper")
    >>> result[4][0].mark
    'ribbon'
    """
    if interpolate != "linear":
        raise ValueError(
            f"mark_ribbon(interpolate={interpolate!r}) is not supported; "
            "only 'linear' interpolation is available"
        )
    if x_field is None or y_field is None:
        raise ValueError("mark_ribbon() requires .encode(x=..., y=...)")
    if y2_field is None:
        raise ValueError(
            "mark_ribbon() requires y2 in encoding (e.g. .encode(x=, y=lower, y2=upper))"
        )
    layers = [
        _Layer(
            name="ribbon",
            mark="ribbon",
            encoding={"x": x_field, "y": y_field, "y2": y2_field},
            mark_kwargs={"opacity": opacity, "stroke": "none"},
        ),
    ]
    return ("__layered__", [], None, None, layers)


register_layer_names("ribbon", frozenset({
    "ribbon",
}))


def desugar_boxen(
    x_field: str | None,
    y_field: str | None,
    *,
    k_depth: str = "tukey",
    k_proportion: float = 0.007,
    outlier_threshold: float = 1.5,
    palette=None,
    horizontal: bool = False,
    color_field: str | None = None,
) -> tuple:
    """Letter-value (boxen) composite mark.

    Desugars into a `LetterValue` transform plus N nested rect bands (one per
    depth, opacity ramping outer→inner), a median rule, and a point layer for
    outliers. Each rect layer reads from a dedicated `lv_depth_K` named output
    (no overlapping rows).

    LetterValue secondary-output schemas:
        ``lv_depth_K``:  ``[group, lower, upper, level]``
        ``lv_outliers``: ``[group, value, is_outlier]``

    Layer encodings therefore use ``"group"`` / ``"lower"`` / ``"upper"`` /
    ``"value"`` rather than the original chart-level column names — but the
    chart-level x/y encoding still references the user's columns, so axis
    scales resolve naturally (LetterValue copies the groupby values verbatim
    into ``group``, and quantile outputs lie within the original value range).
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_boxen() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    group = color_field if color_field else cat

    transforms = [
        LetterValue(
            value=val,
            group=group,
            k_depth=k_depth,
            k_proportion=k_proportion,
            outlier_threshold=outlier_threshold,
            name="lv",
        ),
    ]

    # Per-depth named outputs (lv_depth_1 … lv_depth_6) let each rect layer
    # read its own slice — no overlap. K_MAX = 6 visible bands; for data with
    # fewer effective depths, unused outputs are zero-row batches and render
    # nothing.
    K_MAX = 6
    layers: list = []
    for k in range(1, K_MAX + 1):
        opacity = 0.85 - (0.55 * (k - 1) / max(K_MAX - 1, 1))
        enc = (
            {"x": "lower", "x2": "upper", "y": "group"}
            if horizontal
            else {"x": "group", "y": "lower", "y2": "upper"}
        )
        layers.append(
            _Layer(
                name=f"depth_{k}",
                mark="rect",
                encoding=enc,
                mark_kwargs={"opacity": opacity},
                data_source=f"lv_depth_{k}",
            )
        )

    # Median rule: at depth=1, ``lower == upper == median``.
    layers.append(
        _Layer(
            name="median",
            mark="rule",
            encoding=({"x": "lower", "y": "group"} if horizontal else {"x": "group", "y": "lower"}),
            data_source="lv_depth_1",
        )
    )

    # Outliers: point layer reading from the dedicated outliers output. Schema:
    # [group, value, is_outlier].
    layers.append(
        _Layer(
            name="outlier",
            mark="point",
            encoding=({"x": "value", "y": "group"} if horizontal else {"x": "group", "y": "value"}),
            data_source="lv_outliers",
        )
    )

    return ("__layered__", transforms, None, None, layers)


register_layer_names("boxen", frozenset({
    "depth_1", "depth_2", "depth_3", "depth_4", "depth_5", "depth_6",
    "median", "outlier",
}))


def _extent_to_box(extent):
    return "min-max" if extent == "min-max" else float(extent)


def _extent_to_iqr_k(extent):
    return 1.5 if extent == "min-max" else float(extent)
