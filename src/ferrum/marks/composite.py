"""Composite-mark desugar helpers (Phase 8b).

Each desugar_<name> returns a ``MarkDesugarResult`` with ``layers`` set to a
list of ``ferrum._layer._Layer`` instances (layered mode).

Vocabulary note
---------------
The keyword ``extent`` is reserved for the data-domain sense (a ``[min, max]``
pair bounding a scale or region).  The mark kwargs that were previously named
``extent`` have been renamed:

* ``mark_errorbar`` / ``mark_errorband``: ``extent`` → **``method``**
  (the aggregation method forwarded to ``ErrorExtent``).
* ``mark_boxplot``: ``extent`` → **``whisker_mult``**
  (the IQR multiplier for whisker length).

The old ``extent=`` spelling is accepted as a deprecated alias for one release;
supplying both the canonical name and ``extent=`` in the same call raises a
``TypeError``.
"""

from __future__ import annotations
import warnings
from typing import Any, Optional

from ferrum import BoxStats, ErrorExtent, LetterValue, Outliers
from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum.encoding import X, Y
from ferrum.marks._desugar_helpers import resolve_color_groupby
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)

# Sentinel used to detect "caller did not supply this parameter" in the
# _resolve_deprecated_extent helper.  Identity comparison against this
# object is always reliable, unlike value comparison against floats or
# runtime-built strings (which may not be interned).
_MISSING = object()

# Number of nested letter-value bands ``desugar_boxen`` builds.  This single
# constant drives both the ``range`` loop that creates the ``depth_*`` layers
# and the ``register_layer_names("boxen", ...)`` call that declares their names,
# so the two cannot drift apart.
_BOXEN_K_MAX = 6


def _resolve_deprecated_extent(
    canonical_name: str,
    canonical_value: Any,
    mark_kwargs: dict,
    *,
    real_default: Any,
) -> Any:
    """Resolve a renamed ``extent=`` alias to its canonical kwarg.

    Parameters
    ----------
    canonical_name : str
        The new canonical kwarg name (``"method"`` or ``"whisker_mult"``).
    canonical_value : Any
        The value supplied under the canonical name.  Pass ``_MISSING`` when
        the caller did not supply the canonical kwarg (i.e. its parameter
        default is ``_MISSING``).
    mark_kwargs : dict
        The ``**mark_kwargs`` remainder dict, consumed in-place: if
        ``"extent"`` is found, it is popped and returned as the resolved
        value.
    real_default : Any
        The true default for the canonical parameter (e.g. ``1.5`` or
        ``"ci"``).  Returned when neither the canonical kwarg nor the alias
        were supplied.

    Returns
    -------
    Any
        The resolved value: canonical kwarg wins; alias accepted when the
        canonical kwarg is absent; both supplied raises ``TypeError``;
        neither supplied returns ``real_default``.

    Raises
    ------
    TypeError
        When both the canonical kwarg and ``extent=`` are supplied.
    """
    alias_value = mark_kwargs.pop("extent", _MISSING)
    canonical_supplied = canonical_value is not _MISSING
    alias_supplied = alias_value is not _MISSING

    if not alias_supplied:
        # No alias; return canonical if given, else the real default.
        return canonical_value if canonical_supplied else real_default
    # Alias was supplied.
    if canonical_supplied:
        raise TypeError(
            f"Cannot supply both '{canonical_name}=' and the deprecated 'extent=' alias; "
            f"use '{canonical_name}=' only."
        )
    warnings.warn(
        f"The 'extent=' keyword argument is deprecated for this mark. "
        f"Use '{canonical_name}=' instead. "
        "'extent' is reserved for data-domain [min, max] bounds.",
        DeprecationWarning,
        stacklevel=4,
    )
    return alias_value


def desugar_boxplot(
    x_field: str | None,
    y_field: str | None,
    *,
    whisker_mult: float | str = _MISSING,  # type: ignore[assignment]
    outliers: bool = True,
    size: Optional[float] = None,
    color_field: Optional[str] = None,
    horizontal: bool = False,
    x_sort: Any = None,
    y_sort: Any = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    2. ``tick``   — ``y=lower_whisker``, ``band_size=band`` (lower cap, tracks
       the box's width).
    3. ``tick``   — ``y=upper_whisker``, ``band_size=band`` (upper cap, tracks
       the box's width).
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
    whisker_mult : float or "min-max", default 1.5
        IQR multiplier for whisker length, or ``"min-max"`` to extend
        whiskers to the data minimum/maximum.  Forwarded as
        ``BoxStats.whisker_extent`` / ``Outliers.extent`` (transform field
        names are unchanged).

        .. deprecated::
            The old spelling ``extent=`` is accepted as an alias for one
            release.  Use ``whisker_mult=`` instead.  ``extent`` is reserved
            for data-domain ``[min, max]`` bounds.
    outliers : bool, default True
        Whether to overlay an outlier point layer.
    size : float or None, default None
        Box width as a fraction of the band (default ``0.6``). The lower
        and upper whisker caps and the median tick all share this width.
    color_field : str or None, default None
        Optional column to use for color encoding (also added to the
        ``groupby`` list for the transforms).
    horizontal : bool, default False
        If ``True``, flip axes so that categories are on the y-axis.
    x_sort, y_sort : optional
        Sort order applied to the categorical positional axis, injected by
        the composite-mark expansion from ``sort=`` on the encoding.

    Returns
    -------
    MarkDesugarResult
        Layered mode (``.layers`` set) consumed by ``Chart._resolve_pending``.

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.
    TypeError
        If both ``whisker_mult=`` and the deprecated ``extent=`` alias are
        supplied in the same call.

    Examples
    --------
    >>> result = desugar_boxplot("species", "sepal_length")
    >>> len(result.layers)  # whisker, lower cap, upper cap, box, median, outlier
    6
    >>> [layer.mark for layer in result.layers]
    ['rule', 'tick', 'tick', 'rect', 'tick', 'point']
    """
    whisker_mult = _resolve_deprecated_extent(
        "whisker_mult", whisker_mult, mark_kwargs, real_default=1.5
    )
    user_kw = _validate("boxplot", mark_kwargs)
    if x_field is None or y_field is None:
        raise ValueError("mark_boxplot() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    # Only split by hue when color_field is a distinct column from the
    # categorical axis. When color encodes the same field as cat (e.g.
    # color="cat:N"), adding it to groupby would create a duplicate entry
    # that the Rust BoxStats transform rejects.
    groupby, split_hue = resolve_color_groupby(cat, color_field, [cat])

    transforms = [
        BoxStats(
            field=val, groupby=groupby, whisker_extent=_extent_to_box(whisker_mult), name="box"
        )
    ]
    if outliers:
        transforms.append(
            Outliers(
                field=val,
                groupby=groupby,
                extent=_extent_to_iqr_k(whisker_mult),
                name="outliers",
            )
        )

    band = size or 0.6
    # Resolve the sort for the categorical axis.  When horizontal, y is the
    # categorical channel and y_sort applies; otherwise x is categorical and
    # x_sort applies.
    cat_sort = y_sort if horizontal else x_sort

    def enc(y_col, y2_col=None, *, title=None):
        if horizontal:
            d: dict = {"x": X(y_col, title=title) if title else y_col}
            d["y"] = Y(cat, sort=cat_sort) if cat_sort is not None else cat
            if y2_col:
                d["x2"] = y2_col
        else:
            d = {"x": X(cat, sort=cat_sort) if cat_sort is not None else cat}
            d["y"] = Y(y_col, title=title) if title else y_col
            if y2_col:
                d["y2"] = y2_col
        # Attach the per-layer color encoding only when the hue is a distinct
        # column from the categorical axis (split_hue). When color encodes the
        # same field as cat, the color encoding is redundant with the axis, so
        # it is suppressed to match the errorbar/errorband siblings.
        if split_hue:
            d["color"] = color_field
        return d

    layers = [
        _Layer(
            name="whisker",
            mark="rule",
            encoding=enc("lower_whisker", "upper_whisker", title=val),
            mark_kwargs={"stroke": "theme:label", "stroke_dash": []},
            data_source="box",
        ),
        _Layer(
            name="lower_cap",
            mark="tick",
            encoding=enc("lower_whisker"),
            mark_kwargs={"band_size": band, "stroke": "theme:label"},
            data_source="box",
        ),
        _Layer(
            name="upper_cap",
            mark="tick",
            encoding=enc("upper_whisker"),
            mark_kwargs={"band_size": band, "stroke": "theme:label"},
            data_source="box",
        ),
        _Layer(
            name="box",
            mark="rect",
            encoding=enc("q1", "q3", title=val),
            mark_kwargs={"band_size": band},
            data_source="box",
        ),
        _Layer(
            name="median",
            mark="tick",
            encoding=enc("median"),
            # tick's band_size is a full-width factor (same convention as
            # rect's), so this literally spans the box's width.
            mark_kwargs={"band_size": band, "stroke": "theme:label", "stroke_width": 2},
            data_source="box",
        ),
    ]
    if outliers:
        layers.append(
            _Layer(
                name="outlier",
                mark="point",
                encoding=enc(val),
                mark_kwargs={"filled": False},
                data_source="outliers",
            )
        )

    return MarkDesugarResult(transforms=transforms, layers=_apply(layers, user_kw))


register_layer_names(
    "boxplot",
    frozenset(
        {
            "whisker",
            "lower_cap",
            "upper_cap",
            "box",
            "median",
            "outlier",
        }
    ),
)


def desugar_errorbar(
    x_field: str | None,
    y_field: str | None,
    *,
    method: str = _MISSING,  # type: ignore[assignment]
    ticks: bool = True,
    color_field: Optional[str] = None,
    x_sort: Any = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    2. ``tick``  — ``y=lower`` (bottom cap, ``band_size=0.6``).  When
       ``ticks=True`` only.
    3. ``tick``  — ``y=upper`` (top cap, ``band_size=0.6``).  When
       ``ticks=True`` only.

    Parameters
    ----------
    x_field : str or None
        Categorical (grouping) field. Required.
    y_field : str or None
        Numeric value field. Required.
    method : str, default "ci"
        Aggregation method passed to ``ErrorExtent``.  One of ``"ci"``
        (95 % bootstrap CI), ``"stderr"``, ``"stdev"``, or ``"iqr"``.
        Forwarded verbatim as ``ErrorExtent.method`` (wire field unchanged).

        .. deprecated::
            The old spelling ``extent=`` is accepted as an alias for one
            release.  Use ``method=`` instead.  ``extent`` is reserved for
            data-domain ``[min, max]`` bounds.
    ticks : bool, default True
        Whether to add endpoint tick marks at the top and bottom of each
        error bar.
    color_field : str or None, default None
        Optional column to use for color encoding.  When set (and distinct from
        ``x_field``), it is added to the ``ErrorExtent`` groupby so the error
        extent is computed per (x, color) group, and every emitted layer carries
        a ``color`` encoding so each group is visually distinct.  When ``None``,
        behavior is byte-stable with the previous (no-hue) implementation.
    x_sort : optional
        Sort order applied to the categorical positional axis, injected by
        the composite-mark expansion from ``sort=`` on the encoding.

    Returns
    -------
    MarkDesugarResult
        Layered mode (``.layers`` set).

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.
    TypeError
        If both ``method=`` and the deprecated ``extent=`` alias are
        supplied in the same call.

    Examples
    --------
    >>> result = desugar_errorbar("day", "tip")
    >>> [layer.mark for layer in result.layers]
    ['rule', 'tick', 'tick']
    """
    method = _resolve_deprecated_extent("method", method, mark_kwargs, real_default="ci")
    user_kw = _validate("errorbar", mark_kwargs)
    if x_field is None or y_field is None:
        raise ValueError("mark_errorbar() requires .encode(x=..., y=...)")
    # Errorbar always uses x_field as the categorical grouping axis.
    # Include color_field in the groupby when it is a distinct column so the
    # error extent is computed per (x, hue) group, not pooled across hues.
    groupby, split_hue = resolve_color_groupby(x_field, color_field, [x_field])
    x_enc_val = X(x_field, sort=x_sort) if x_sort is not None else x_field

    def _enc(y_col: str, y2_col: str | None = None) -> dict:
        d: dict = {"x": x_enc_val, "y": y_col}
        if y2_col is not None:
            d["y2"] = y2_col
        if split_hue:
            d["color"] = color_field
        return d

    transforms = [ErrorExtent(field=y_field, groupby=groupby, method=method, name="err")]
    layers = [
        _Layer(
            name="rule",
            mark="rule",
            encoding=_enc("lower", "upper"),
            mark_kwargs={"stroke": "theme:label", "stroke_dash": []},
            data_source="err",
        ),
    ]
    if ticks:
        layers.extend(
            [
                _Layer(
                    name="lower_cap",
                    mark="tick",
                    # Literal 0.6 is the deliberate full-length default
                    # (GH #85), not an un-migrated half-width 0.3 x 2:
                    # errorbar exposes no size= knob for caps to track,
                    # unlike desugar_boxplot's band-derived caps.
                    encoding=_enc("lower"),
                    mark_kwargs={"band_size": 0.6, "stroke": "theme:label"},
                    data_source="err",
                ),
                _Layer(
                    name="upper_cap",
                    mark="tick",
                    encoding=_enc("upper"),
                    mark_kwargs={"band_size": 0.6, "stroke": "theme:label"},
                    data_source="err",
                ),
            ]
        )
    return MarkDesugarResult(transforms=transforms, layers=_apply(layers, user_kw))


register_layer_names(
    "errorbar",
    frozenset(
        {
            "rule",
            "lower_cap",
            "upper_cap",
        }
    ),
)


def desugar_errorband(
    x_field: str | None,
    y_field: str | None,
    *,
    method: str = _MISSING,  # type: ignore[assignment]
    borders: bool = False,
    color_field: Optional[str] = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    method : str, default "ci"
        Aggregation method passed to ``ErrorExtent``.  One of ``"ci"``,
        ``"stderr"``, ``"stdev"``, or ``"iqr"``.  Forwarded verbatim as
        ``ErrorExtent.method`` (wire field unchanged).

        .. deprecated::
            The old spelling ``extent=`` is accepted as an alias for one
            release.  Use ``method=`` instead.  ``extent`` is reserved for
            data-domain ``[min, max]`` bounds.
    borders : bool, default False
        Whether to draw solid border lines at the upper and lower edges of
        the ribbon.
    color_field : str or None, default None
        Optional column to use for color encoding.  When set (and distinct from
        ``x_field``), it is added to the ``ErrorExtent`` groupby so the error
        extent is computed per (x, color) group, and every emitted layer carries
        a ``color`` encoding so each group's ribbon is visually distinct.  When
        ``None``, behavior is byte-stable with the previous (no-hue) implementation.

    Returns
    -------
    MarkDesugarResult
        Layered mode (``.layers`` set).

    Raises
    ------
    ValueError
        If either ``x_field`` or ``y_field`` is ``None``.
    TypeError
        If both ``method=`` and the deprecated ``extent=`` alias are
        supplied in the same call.

    Examples
    --------
    >>> result = desugar_errorband("x", "y")
    >>> result.layers[0].mark
    'ribbon'
    """
    method = _resolve_deprecated_extent("method", method, mark_kwargs, real_default="ci")
    user_kw = _validate("errorband", mark_kwargs)
    if x_field is None or y_field is None:
        raise ValueError("mark_errorband() requires .encode(x=..., y=...)")
    # Include color_field in the groupby when it is a distinct column so the
    # error extent is computed per (x, hue) group, not pooled across hues.
    groupby, split_hue = resolve_color_groupby(x_field, color_field, [x_field])

    def _enc_ribbon() -> dict:
        d: dict = {"x": x_field, "y": Y("lower", title=y_field), "y2": "upper"}
        if split_hue:
            d["color"] = color_field
        return d

    def _enc_border(y_col: str) -> dict:
        d: dict = {"x": x_field, "y": y_col}
        if split_hue:
            d["color"] = color_field
        return d

    transforms = [ErrorExtent(field=y_field, groupby=groupby, method=method, name="err")]
    layers = [
        _Layer(
            name="ribbon",
            mark="ribbon",
            encoding=_enc_ribbon(),
            mark_kwargs={"opacity": 0.2, "stroke": "none"},
            data_source="err",
        ),
    ]
    if borders:
        layers.extend(
            [
                _Layer(
                    name="lower_border",
                    mark="line",
                    encoding=_enc_border("lower"),
                    data_source="err",
                ),
                _Layer(
                    name="upper_border",
                    mark="line",
                    encoding=_enc_border("upper"),
                    data_source="err",
                ),
            ]
        )
    return MarkDesugarResult(transforms=transforms, layers=_apply(layers, user_kw))


register_layer_names(
    "errorband",
    frozenset(
        {
            "ribbon",
            "lower_border",
            "upper_border",
        }
    ),
)


def desugar_ribbon(
    x_field: str | None,
    y_field: str | None,
    *,
    y2_field: str | None = None,
    opacity: float = 0.2,
    interpolate: str = "linear",
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    MarkDesugarResult
        Layered mode with a single ribbon layer (``.layers`` set, no
        transforms).

    Raises
    ------
    ValueError
        If ``x_field``, ``y_field``, or ``y2_field`` is ``None``.

    Examples
    --------
    >>> result = desugar_ribbon("x", "lower", y2_field="upper")
    >>> result.layers[0].mark
    'ribbon'
    """
    user_kw = _validate("ribbon", mark_kwargs)
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
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names(
    "ribbon",
    frozenset(
        {
            "ribbon",
        }
    ),
)


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
    x_sort: Any = None,
    y_sort: Any = None,
    boxen_agg_sort: dict | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    user_kw = _validate("boxen", mark_kwargs)
    if x_field is None or y_field is None:
        raise ValueError("mark_boxen() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    group = color_field if color_field else cat
    # Resolve the sort for the categorical axis.  When horizontal, y is the
    # categorical channel; otherwise x is categorical.
    cat_sort = y_sort if horizontal else x_sort
    # Aggregate-form sort (op/order) is resolved inside the LetterValue transform,
    # which emits its group rows in aggregate order.  The categorical axis domain
    # then falls out in first-appearance order, so no explicit layer sort is set.
    # Non-aggregate sorts (explicit list / "ascending" / "descending") still apply
    # to the renamed "group" encoding in the layers.
    if boxen_agg_sort is not None:
        cat_sort = None
    # The LetterValue transform renames the groupby column to "group" in its
    # output, so sort must be applied to the "group" encoding in the layers.
    group_enc_x = X("group", sort=cat_sort) if cat_sort is not None else "group"
    group_enc_y = Y("group", sort=cat_sort) if cat_sort is not None else "group"

    transforms = [
        LetterValue(
            value=val,
            group=group,
            k_depth=k_depth,
            k_proportion=k_proportion,
            outlier_threshold=outlier_threshold,
            name="lv",
            sort=boxen_agg_sort,
        ),
    ]

    # Per-depth named outputs (lv_depth_1 … lv_depth_{_BOXEN_K_MAX}) let each
    # rect layer read its own slice (no overlap). ``_BOXEN_K_MAX`` visible
    # bands; for data with fewer effective depths, unused outputs are zero-row
    # batches and render nothing.  The same constant derives the registry's
    # ``depth_*`` names below, so the loop bound and the registered set cannot
    # drift apart.
    layers: list = []
    for k in range(1, _BOXEN_K_MAX + 1):
        opacity = 0.85 - (0.55 * (k - 1) / max(_BOXEN_K_MAX - 1, 1))
        enc = (
            {"x": "lower", "x2": "upper", "y": group_enc_y}
            if horizontal
            else {"x": group_enc_x, "y": "lower", "y2": "upper"}
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
            encoding=(
                {"x": "lower", "y": group_enc_y} if horizontal else {"x": group_enc_x, "y": "lower"}
            ),
            mark_kwargs={"stroke": "theme:label", "stroke_dash": []},
            data_source="lv_depth_1",
        )
    )

    # Outliers: point layer reading from the dedicated outliers output. Schema:
    # [group, value, is_outlier].
    layers.append(
        _Layer(
            name="outlier",
            mark="point",
            encoding=(
                {"x": "value", "y": group_enc_y} if horizontal else {"x": group_enc_x, "y": "value"}
            ),
            data_source="lv_outliers",
        )
    )

    return MarkDesugarResult(transforms=transforms, layers=_apply(layers, user_kw))


register_layer_names(
    "boxen",
    frozenset({f"depth_{k}" for k in range(1, _BOXEN_K_MAX + 1)}) | {"median", "outlier"},
)


def _extent_to_box(extent):
    return "min-max" if extent == "min-max" else float(extent)


def _extent_to_iqr_k(extent):
    return 1.5 if extent == "min-max" else float(extent)
