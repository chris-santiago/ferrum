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
import math
import warnings
from typing import Any, Optional, Sequence

from ferrum import BoxStats, ErrorExtent, LetterValue, Outliers
from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum._registry import registry_output_columns, registry_secondary_columns
from ferrum.color import palette as _named_palette_colors
from ferrum.encoding import X, Y
from ferrum.marks._desugar_helpers import (
    nominal_color_channel,
    resolve_color_groupby,
)
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)

# Sentinel used to detect "caller did not supply this parameter" in the
# _resolve_deprecated_extent helper.  Identity comparison against this
# object is always reliable, unlike value comparison against floats or
# runtime-built strings (which may not be interned).
_MISSING = object()

#: The ``name=`` ``desugar_boxen`` gives its ``LetterValue`` transform.
#: ``letter_value`` prefixes every secondary batch key with it
#: (``"lv_depth_3"``, ``"lv_outliers"``), so this constant is what turns a
#: registry-declared key into the wire name a layer's ``data_source`` uses.
_LV_TRANSFORM_NAME = "lv"

#: ``letter_value``'s declared secondary key for the outlier rows.
_LV_OUTLIERS_KEY = "outliers"

#: ``letter_value``'s declared secondary key for the innermost depth band.
#: Spelled as a literal (rather than ``_lv_depth_key(1)``) because it is the
#: key ``desugar_boxen`` hands ``registry_secondary_columns``, and the static
#: sweep in ``tests/test_registry_output_column_consumers.py`` reads that
#: argument from the source. ``tests/test_boxen_secondary_outputs.py`` pins it
#: equal to ``_lv_depth_key(1)``.
_LV_DEPTH_1_KEY = "depth_1"


def _lv_depth_key(k: int) -> str:
    """``letter_value``'s declared secondary key for depth band *k*.

    Mirrors ``letter_value::depth_key`` on the Rust side. Every ``k`` this
    mark builds a layer for is pinned against the registry's declared keys by
    ``tests/test_boxen_secondary_outputs.py``, so a Rust-side change to the
    key shape fails there rather than yielding a layer silently bound to a
    batch nothing publishes.
    """
    return f"depth_{k}"


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
       output (when ``outliers=True`` and ``whisker_mult != "min-max"``).

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
        Whether to overlay an outlier point layer.  Forced off when
        ``whisker_mult="min-max"`` regardless of this flag: whiskers already
        reach the data minimum/maximum, so nothing can lie beyond them.
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
    # whisker_mult="min-max" extends the whiskers to the data extremes, so by
    # definition no observation lies beyond them -- reuse the outliers=False
    # suppression path below (both the Outliers transform and the outlier
    # layer) rather than adding a second mechanism (F-L03-10).
    outliers = outliers and whisker_mult != "min-max"
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
                extent=float(whisker_mult),
                name="outliers",
            )
        )

    band = size or 0.6
    # Resolve the sort for the categorical axis.  When horizontal, y is the
    # categorical channel and y_sort applies; otherwise x is categorical and
    # x_sort applies.
    cat_sort = y_sort if horizontal else x_sort
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of any of these columns raises here instead of
    # silently reading a stale name.
    _box_cols = registry_output_columns(
        "box_stats", ["q1", "median", "q3", "lower_whisker", "upper_whisker"]
    )

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
            d["color"] = nominal_color_channel(color_field)
        return d

    layers = [
        _Layer(
            name="whisker",
            mark="rule",
            encoding=enc(_box_cols["lower_whisker"], _box_cols["upper_whisker"], title=val),
            mark_kwargs={"stroke": "theme:label", "stroke_dash": []},
            data_source="box",
        ),
        _Layer(
            name="lower_cap",
            mark="tick",
            encoding=enc(_box_cols["lower_whisker"]),
            mark_kwargs={"band_size": band, "stroke": "theme:label"},
            data_source="box",
        ),
        _Layer(
            name="upper_cap",
            mark="tick",
            encoding=enc(_box_cols["upper_whisker"]),
            mark_kwargs={"band_size": band, "stroke": "theme:label"},
            data_source="box",
        ),
        _Layer(
            name="box",
            mark="rect",
            encoding=enc(_box_cols["q1"], _box_cols["q3"], title=val),
            mark_kwargs={"band_size": band},
            data_source="box",
        ),
        _Layer(
            name="median",
            mark="tick",
            encoding=enc(_box_cols["median"]),
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
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either column raises here instead of silently
    # reading a stale name.
    _err_cols = registry_output_columns("error_extent", ["lower", "upper"])

    def _enc(y_col: str, y2_col: str | None = None) -> dict:
        d: dict = {"x": x_enc_val, "y": y_col}
        if y2_col is not None:
            d["y2"] = y2_col
        if split_hue:
            d["color"] = nominal_color_channel(color_field)
        return d

    transforms = [ErrorExtent(field=y_field, groupby=groupby, method=method, name="err")]
    layers = [
        _Layer(
            name="rule",
            mark="rule",
            encoding=_enc(_err_cols["lower"], _err_cols["upper"]),
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
                    encoding=_enc(_err_cols["lower"]),
                    mark_kwargs={"band_size": 0.6, "stroke": "theme:label"},
                    data_source="err",
                ),
                _Layer(
                    name="upper_cap",
                    mark="tick",
                    encoding=_enc(_err_cols["upper"]),
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
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either column raises here instead of silently
    # reading a stale name.
    _err_cols = registry_output_columns("error_extent", ["lower", "upper"])

    def _enc_ribbon() -> dict:
        d: dict = {
            "x": x_field,
            "y": Y(_err_cols["lower"], title=y_field),
            "y2": _err_cols["upper"],
        }
        if split_hue:
            d["color"] = nominal_color_channel(color_field)
        return d

    def _enc_border(y_col: str) -> dict:
        d: dict = {"x": x_field, "y": y_col}
        if split_hue:
            d["color"] = nominal_color_channel(color_field)
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
                    encoding=_enc_border(_err_cols["lower"]),
                    data_source="err",
                ),
                _Layer(
                    name="upper_border",
                    mark="line",
                    encoding=_enc_border(_err_cols["upper"]),
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
       ``opacity=opacity``, ``stroke="none"``, ``interpolate=interpolate``.

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
        Boundary interpolation: ``"linear"``, ``"step"``, ``"step-before"``,
        ``"step-after"``, ``"monotone"``, ``"basis"``, ``"cardinal"``.
        Forwarded to the emitted layer's mark style; an unrecognized value
        is refused by the renderer.

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
            mark_kwargs={"opacity": opacity, "stroke": "none", "interpolate": interpolate},
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


def _boxen_depth_count() -> int:
    """Number of nested letter-value bands ``desugar_boxen`` builds.

    Derived by counting ``letter_value``'s registry-declared ``depth_*``
    secondary-output keys (Rust's ``K_MAX_NAMED``) rather than a
    hand-copied literal, so a Rust-side change to that cap is picked up
    here automatically instead of drifting into a stale Python constant
    (F-L03-04b). This single derivation drives both the ``range()`` loop
    that creates ``desugar_boxen``'s ``depth_*`` layers and the
    ``register_layer_names("boxen", ...)`` call that declares their names,
    so the two cannot drift from each other either.
    """
    from ferrum._core import transform_registry

    keys = [key for key, _cols in transform_registry("letter_value")["secondary_outputs"]]
    return sum(1 for key in keys if key.startswith("depth_"))


def _k_for_depth(k_depth: str, k_proportion: float, n: int) -> int:
    """Thin arithmetic mirror of ``letter_value::k_for_depth`` (Rust),
    restricted to the piece that decides HOW MANY bands a group of size *n*
    materializes -- never the quantile values themselves, which stay
    exclusively Rust-computed everywhere else in this module. See
    :func:`_boxen_materialized_depth`'s docstring for why this mirror
    exists at all and what happens if it ever drifts from the real
    formula. Unrecognized ``k_depth`` values fall back to the full
    configured cap (the caller only uses this for a styling estimate; the
    real transform is what actually validates ``k_depth`` and raises).
    """
    depth_count = _boxen_depth_count()
    if n <= 0:
        return 1
    if k_depth == "tukey":
        k = math.floor(math.log2(n)) - 3
    elif k_depth == "proportion":
        target = max(math.ceil(k_proportion * n), 1)
        k = 1
        while (1 << (k - 1)) < target:
            k += 1
            if k > depth_count:
                break
    elif k_depth == "trustworthy":
        k = math.floor(math.log2(n / 3.32))
    elif k_depth == "full":
        k = math.floor(math.log2(n))
    else:
        return depth_count
    return max(min(k, depth_count), 1)


def _boxen_materialized_depth(
    data: Any, *, value_field: str, group_field: str, k_depth: str, k_proportion: float
) -> Optional[int]:
    """The deepest letter-value depth ANY group in *data* will actually
    materialize, guarded to a minimum of ``2`` (the base band always
    anchors the ramp, matching ``_boxen_band_color_index``'s ``max(k, 2)``
    convention elsewhere in this file) -- or ``None`` when it can't be
    determined (absent or unreadable data), in which case the caller falls
    back to the full configured depth cap, exactly the pre-2026-09-06-
    cycle-2 behavior (never a crash; this is a styling estimate, not a
    correctness-critical value).

    Why this exists: raising the registry-derived depth cap 6->12
    (F-L03-04b) also raised the default opacity ramp's normalization
    range, which flattened the ramp's visual contrast on every chart whose
    REAL rendered depth had not changed at all (quality-review REVISE,
    2026-09-06 cycle 2) -- the ramp must track what a chart actually
    renders, not the mark's full configured ceiling. Desugar functions in
    this module otherwise never touch the bound data (by design -- real
    per-group depth is normally a Rust-side-only quantity, see
    ``_boxen_band_color_index``'s docstring), so this is a narrow,
    deliberate exception scoped to one styling computation: it counts real
    per-group row counts here and reuses :func:`_k_for_depth`'s lightweight
    arithmetic (not the quantile math) to estimate the deepest band any
    group reaches. A drift between this estimate and the real Rust
    computation degrades to a slightly-off ramp contrast, never a wrong
    rendered value -- the actual bands, still built for every
    ``k`` in ``1..depth_count`` regardless of this estimate, read their
    real data from Rust exactly as before.
    """
    if data is None:
        return None
    import polars as pl

    try:
        from ferrum._coerce import to_polars

        frame = to_polars(data)
    except (TypeError, ValueError, KeyError, ImportError, pl.exceptions.PolarsError):
        # Best-effort styling estimate only: an unsupported/unreadable data
        # shape falls back to the full configured cap rather than crashing
        # a rendering-only computation. Deliberately NOT a bare `except
        # Exception` (quality-review S2, cycle 3): a broader catch here
        # would silently reinstate the exact cap-normalization regression
        # this function exists to prevent, with no signal, for ANY
        # unexpected error rather than just the coercion failures this is
        # actually meant to absorb.
        return None
    if value_field not in frame.columns or group_field not in frame.columns:
        return None

    # Mirror `letter_value::extract_values` exactly (quality-review S1,
    # cycle 3): Rust drops a row when its VALUE is null or NaN, but keeps a
    # null GROUP as its own group (BTreeMap keyed by `Option<String>`) --
    # `drop_nulls()` over both columns previously did the opposite on both
    # counts (dropped null-group rows, kept NaN values), which measurably
    # overcounted a group's real size whenever its value column carried
    # NaN (400 rows with 200 NaN estimated depth 5, Rust materializes 4).
    value_col = pl.col(value_field)
    valid = value_col.is_not_null()
    if frame.schema[value_field] in (pl.Float32, pl.Float64):
        valid = valid & value_col.is_not_nan()
    try:
        sizes = (
            frame.select([group_field, value_field])
            .filter(valid)
            .group_by(group_field)
            .len()["len"]
            .to_list()
        )
    except (TypeError, ValueError, KeyError, pl.exceptions.PolarsError):
        return None
    if not sizes:
        return None
    return max(max(_k_for_depth(k_depth, k_proportion, n) for n in sizes), 2)


# Number of color slots ``palette=`` can ever actually paint. ``k=1`` (the
# median band) is always zero-pixel by construction (see
# ``_boxen_band_color_index``) and borrows ``k=2``'s slot rather than
# consuming one of its own, so only bands ``k=2..depth_count`` are
# independently colorable -- one fewer than the derived depth count.
# Requesting exactly this many colors (not the full depth count) means a
# full-depth dataset shows every generated color, including a named
# continuous palette's endpoint (``viridis`` reaches ``#fde725``), instead
# of a structurally-dead last slot.
_BOXEN_VISIBLE_BANDS = _boxen_depth_count() - 1


def _resolve_boxen_palette(palette: str | Sequence[str] | None) -> list[str] | None:
    """Expand ``palette=`` into exactly ``_BOXEN_VISIBLE_BANDS`` colors, in
    **inside-out** order: ``colors[0]`` is spent on the innermost
    *visible* band (``k=2``), ``colors[1]`` the next band out, and so on --
    see ``_boxen_band_color_index`` for how a raw depth ``k`` maps into
    this list (``k=1``, the always-zero-pixel median band, never gets a
    slot of its own; it borrows ``k=2``'s color).

    ``None`` means "no palette": the caller keeps the opacity-ramp shading
    that is the ``palette=None`` behavior. A named palette (``str``) is
    expanded via ``ferrum.color.palette``, which raises the same
    ``ValueError`` shape ``scheme=`` raises for an unrecognized name, and
    already cycles categorical palettes shorter than ``n`` colors. An
    explicit color sequence is cycled to length manually here, since
    ``ferrum.color.palette`` only cycles *named* palettes. A value that is
    neither a ``str`` nor iterable (e.g. ``palette=5``) raises the same
    named ``ValueError`` shape as the empty-sequence case below, rather
    than leaking a bare ``TypeError`` from ``list(palette)``.
    """
    if palette is None:
        return None
    if isinstance(palette, str):
        return _named_palette_colors(palette, n=_BOXEN_VISIBLE_BANDS)
    try:
        colors = list(palette)
    except TypeError as exc:
        raise ValueError(
            f"mark_boxen(palette=...): expected a palette name (str), a "
            f"sequence of color strings, or None, got {palette!r}"
        ) from exc
    if not colors:
        raise ValueError("mark_boxen(palette=...): color sequence must not be empty")
    return [colors[i % len(colors)] for i in range(_BOXEN_VISIBLE_BANDS)]


def _boxen_band_color_index(k: int) -> int:
    """Map a raw ``LetterValue`` depth ``k`` to a 0-based index into the
    inside-out color list ``_resolve_boxen_palette`` returns.

    ``k=1`` is *always* a zero-pixel band by construction, not merely for
    small samples: ``lv_depth_1``'s rows satisfy ``lower == upper ==
    median`` (it is the median tick's own data source, see the comment
    above the median-tick layer below), so it never has a visible extent
    for any dataset. ``k=2`` is the **base band**: the first real interval
    beyond the median, guaranteed to render whenever ``LetterValue``
    produces *any* non-degenerate depth at all (real depth count is never
    less than 2 once it exceeds 1). Anchoring ``colors[0]`` at ``k=2``
    (spec §4.4, re-amended after a second quality-review pass) rather than
    at the widest *configured* band (the deepest depth the mark builds a
    layer for) is deliberate:
    "outermost rendered band gets colors[0]" is not decidable at desugar
    time (no data access here -- real per-group depth is a Rust-side
    quantity, computed later, and can differ between groups in the same
    chart), so anchoring at the widest *configured* slot made ``colors[0]``
    render only when a dataset happened to reach full depth -- dead for
    every typical dataset under the mark's default ``k_depth="tukey"``
    (measured: only colors 4-5 of a 6-color palette ever appeared at
    n=200). Anchoring at the guaranteed-present base band instead makes
    ``colors[0]`` visible for *any* dataset with at least one real depth
    level, and later colors consume outward (``k=3`` -> ``colors[1]``,
    ``k=4`` -> ``colors[2]``, ...) as richer data materializes more bands
    -- a band that never materializes for a given dataset simply leaves
    its color, and every color past it, unused, the same tail-truncation
    behavior as a palette longer than the number of real categories
    anywhere else in the codebase. ``k=1`` borrows index 0 (``k=2``'s
    slot) via ``visible_k = max(k, 2)``, so it never independently
    consumes a color either.
    """
    visible_k = max(k, 2)
    return visible_k - 2


def desugar_boxen(
    x_field: str | None,
    y_field: str | None,
    *,
    k_depth: str = "tukey",
    k_proportion: float = 0.007,
    outlier_threshold: float = 1.5,
    palette: str | Sequence[str] | None = None,
    horizontal: bool = False,
    color_field: str | None = None,
    x_sort: Any = None,
    y_sort: Any = None,
    boxen_agg_sort: dict | None = None,
    materialized_depth: int | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """Letter-value (boxen) composite mark.

    Desugars into a `LetterValue` transform plus N nested rect bands (one per
    depth, opacity ramping outer→inner), a band-confined median tick, and a point layer for
    outliers. Each rect layer reads from a dedicated `lv_depth_K` named output
    (no overlapping rows).

    ``materialized_depth`` is internal plumbing, not a public ``mark_boxen()``
    parameter: ``_desugar._prep_boxen_ramp_depth`` computes it from the bound
    data (via :func:`_boxen_materialized_depth`) and injects it before this
    function runs, the same idiom ``boxen_agg_sort`` already uses. ``None``
    (no data available, or the ``palette=`` path, which never reads it) falls
    back to the full configured depth cap.

    LetterValue secondary-output schemas:
        ``lv_depth_K``:  ``[group, lower, upper, level]``
        ``lv_outliers``: ``[group, value, is_outlier]``

    Layer encodings therefore use ``"group"`` / ``"lower"`` / ``"upper"`` /
    ``"value"`` rather than the original chart-level column names — but the
    chart-level x/y encoding still references the user's columns, so axis
    scales resolve naturally (LetterValue copies the groupby values verbatim
    into ``group``, and quantile outputs lie within the original value range).

    Those two schemas are read through ``registry_secondary_columns``, not
    ``registry_output_columns`` (spec §4.6 / M6a): they are ``letter_value``'s
    SECONDARY batches, and its *primary* schema happens to also list
    ``"group"``/``"lower"``/``"upper"``, so checking them against the primary
    would pass today and stay silently green if a secondary batch's own
    columns were renamed independently — a lookup against the wrong schema is
    worse than no lookup. The registry declares the secondary batches since
    2026-09-06, so the right schema is now available to check against.

    ``palette`` colors the depth bands directly, at opacity 1.0, replacing
    the opacity ramp. The mapping anchors on the **base band** (spec §4.4,
    re-amended after a second quality-review pass): ``colors[0]`` lands on
    ``k=2``, the innermost real interval beyond the median -- guaranteed
    to render whenever ``LetterValue`` produces any non-degenerate depth
    at all, for any dataset -- and later colors consume outward
    (``k=3``, ``k=4``, ...) as richer data materializes more bands.
    ``k=1``, the median-tick band, is always a zero-pixel degenerate row
    (``lower == upper == median``) and never consumes a color of its own;
    it borrows ``k=2``'s. See ``_boxen_band_color_index`` for the exact
    mapping and why the anchor is inside-out rather than the intuitive
    "outermost band gets colors[0]" (that framing is not decidable at
    desugar time -- no data access here -- and was measured to leave
    ``colors[0]`` unused on every typical dataset under the default
    ``k_depth``). Solid fills make occlusion structural, so bands paint
    **widest-first**
    (largest present ``k`` painted first/under, ``k=1`` painted last/on
    top) -- the only way the nesting stays visible once the alpha-blended
    ramp is gone; the ``palette=None`` branch below keeps its original
    ascending-``k`` layer order untouched (and stays byte-identical). This
    is depth-band coloring, not a seaborn-style hue mapping -- boxen has
    **no hue channel today** (hue-vs-palette support is the tracked
    violin-hue frontier work, out of scope here), and ``palette`` does not
    interact with ``color_field`` (which only changes the ``LetterValue``
    groupby, not color -- every band still renders in one theme color at
    ramp opacities, or the flat palette fill, regardless of
    ``color_field``). A chart-level ``.encode(color=...)`` channel *does*
    conflict with ``palette``: the renderer's color-encoding resolution
    always overrides a layer's ``fill=`` kwarg (the same precedence a
    plain ``fill=`` has under any encoded channel), so combining the two
    would silently drop every band's palette color while still forcing
    opacity to 1.0 -- ``ferrum._desugar._prep_boxen_palette_color_conflict``
    raises a ``ValueError`` for that combination instead of rendering a
    chart where ``palette`` had no visible effect (drop the color
    encoding, or drop ``palette=``; boxen has no hue channel to fall back
    to). ``palette=None`` (the default) keeps the opacity-ramp shading
    applied to the ordinary mark-color resolution (an explicit ``fill=``
    override, else the chart's ``color`` encoding through the theme's
    categorical palette, else the theme's default ``mark_color``). An
    explicit user ``fill=`` or ``opacity=`` mark kwarg still wins over
    either shading, per the usual ``apply_user_mark_kwargs`` precedence.
    """
    user_kw = _validate("boxen", mark_kwargs)
    palette_colors = _resolve_boxen_palette(palette)
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

    # `LetterValue`'s SECONDARY batches (spec §4.6 / M6a), declared in the
    # registry since 2026-09-06 and read through the secondary lookup rather
    # than the primary one -- see this function's docstring for why the
    # primary schema is the wrong thing to check these against. Every
    # `depth_{k}` batch carries the same schema, so one lookup covers every
    # band layer this mark builds; that the registry declares a batch for
    # each k this mark builds is pinned by
    # `tests/test_boxen_secondary_outputs.py`.
    depth_cols = registry_secondary_columns(
        "letter_value", _LV_DEPTH_1_KEY, ["group", "lower", "upper"]
    )
    outlier_cols = registry_secondary_columns("letter_value", _LV_OUTLIERS_KEY, ["group", "value"])
    depth_count = _boxen_depth_count()

    # The LetterValue transform renames the groupby column to "group" and the
    # value axis to "lower"/"upper"/"value" in its output, so every layer's
    # positional encoding must be retitled with the user's own field names
    # (F-L03-12) or the rendered axes show LetterValue's internal column
    # names instead. Unlike desugar_boxplot's enc() helper (composite.py
    # ~L280), whose categorical field is never renamed, boxen renames BOTH
    # channels identically across every layer, so both are titled once here
    # rather than per call site.
    group_col = depth_cols["group"]
    group_enc_x = (
        X(group_col, title=group, sort=cat_sort)
        if cat_sort is not None
        else X(group_col, title=group)
    )
    group_enc_y = (
        Y(group_col, title=group, sort=cat_sort)
        if cat_sort is not None
        else Y(group_col, title=group)
    )

    def enc(value_col: str, value2_col: str | None = None) -> dict:
        """This layer's positional encoding: the categorical channel from
        ``group_enc_x``/``group_enc_y`` (already titled with ``group``) plus
        the value channel titled with ``val``, the user's original field."""
        if horizontal:
            d: dict = {"x": X(value_col, title=val), "y": group_enc_y}
            if value2_col:
                d["x2"] = value2_col
        else:
            d = {"x": group_enc_x, "y": Y(value_col, title=val)}
            if value2_col:
                d["y2"] = value2_col
        return d

    transforms = [
        LetterValue(
            value=val,
            group=group,
            k_depth=k_depth,
            k_proportion=k_proportion,
            outlier_threshold=outlier_threshold,
            name=_LV_TRANSFORM_NAME,
            sort=boxen_agg_sort,
        ),
    ]

    # Per-depth named outputs (lv_depth_1 … lv_depth_{depth_count}) let each
    # rect layer read its own slice (no overlap). `depth_count` (derived from
    # the registry above) visible bands; for data with fewer effective
    # depths, unused outputs are zero-row batches and render nothing.  The
    # same derived count drives the registry's `depth_*` names below, so the
    # loop bound and the registered set cannot drift apart.
    #
    # Paint order (spec §4.4, amended 2026-08-27 after quality review):
    # k=1 is the innermost band; under a palette, every band renders at
    # opacity=1.0 (no alpha blending to reveal what's underneath), so the
    # only way nesting stays visible is z-order -- widest band first/under,
    # narrowest (k=1) last/on top. Iterate descending only in the palette
    # branch; the palette=None branch keeps its original ascending order
    # (and stays byte-identical) since the opacity ramp already makes the
    # painted-last widest band read as nested via alpha blending.
    depth_ks = (
        range(depth_count, 0, -1) if palette_colors is not None else range(1, depth_count + 1)
    )
    # Opacity ramp normalization (quality-review REVISE, 2026-09-06 cycle 2):
    # normalized against `ramp_depth` -- the depth THIS chart actually
    # materializes (`materialized_depth`, injected by
    # `_desugar._prep_boxen_ramp_depth` from the bound data), falling back to
    # the full configured `depth_count` only when that couldn't be determined
    # -- never against `depth_count` itself. Raising the registry-derived cap
    # (6->12, F-L03-04b) must not flatten a chart's ramp contrast when its
    # real rendered depth hasn't changed; normalizing by the configured cap
    # unconditionally did exactly that (measured: adjacent-band opacity step
    # shrank from ~0.11 to ~0.05 on an unchanged n=200 chart). Depths beyond
    # `ramp_depth` (empty bands in the ordinary case; a possible underestimate
    # in the pathological case) are clamped rather than left to go negative.
    ramp_depth = materialized_depth if materialized_depth is not None else depth_count
    ramp_denominator = max(ramp_depth - 1, 1)
    layers: list = []
    for k in depth_ks:
        if palette_colors is not None:
            depth_kwargs = {"fill": palette_colors[_boxen_band_color_index(k)], "opacity": 1.0}
        else:
            opacity = 0.85 - (0.55 * (k - 1) / ramp_denominator)
            depth_kwargs = {"opacity": min(max(opacity, 0.15), 0.85)}
        layers.append(
            _Layer(
                name=f"depth_{k}",
                mark="rect",
                encoding=enc(depth_cols["lower"], depth_cols["upper"]),
                mark_kwargs=depth_kwargs,
                data_source=f"{_LV_TRANSFORM_NAME}_{_lv_depth_key(k)}",
            )
        )

    # Median tick: at depth=1, ``lower == upper == median``. A tick (not a
    # rule) confined to the deepest rendered band's own width rule so the
    # median mark spans only its category's band instead of the full panel
    # width (F-L03-02). No explicit `band_size` here: `tick.rs`'s ordinal
    # "boxplot median" branch already resolves an unset `band_size` to 0.6
    # (`unwrap_or(0.6)`), the exact same default `rect.rs` resolves for
    # every depth band above (neither ever overrides it) -- restating the
    # literal here would let a future change to that shared default move
    # the bands while leaving the median tick pinned at the old value.
    # Declares BOTH positional channels (category + value) -- a
    # single-channel tick normalizes to a rug (`Mark::Tick` own-channel
    # normalization), which this is not.
    layers.append(
        _Layer(
            name="median",
            mark="tick",
            encoding=enc(depth_cols["lower"]),
            mark_kwargs={"stroke": "theme:label", "stroke_dash": []},
            data_source=f"{_LV_TRANSFORM_NAME}_{_lv_depth_key(1)}",
        )
    )

    # Outliers: point layer reading from the dedicated outliers output.
    layers.append(
        _Layer(
            name="outlier",
            mark="point",
            encoding=enc(outlier_cols["value"]),
            data_source=f"{_LV_TRANSFORM_NAME}_{_LV_OUTLIERS_KEY}",
        )
    )

    return MarkDesugarResult(transforms=transforms, layers=_apply(layers, user_kw))


register_layer_names(
    "boxen",
    frozenset({f"depth_{k}" for k in range(1, _boxen_depth_count() + 1)}) | {"median", "outlier"},
)


def _extent_to_box(extent):
    return "min-max" if extent == "min-max" else float(extent)
