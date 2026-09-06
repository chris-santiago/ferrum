"""Statistical mark desugaring — convert mark_density/histogram/smooth kwargs
into MarkDesugarResult objects consumed by Chart._resolve_pending."""

from __future__ import annotations

from typing import Any

from ferrum import Bin, Kde, Smooth
from ferrum._core import transform_registry
from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum._registry import registry_output_columns
from ferrum._validate import validate_choice
from ferrum.marks._fit_layers import build_fit_layers

_ORIENTATIONS = ("horizontal", "vertical")
_MULTIPLE_MODES = ("dodge", "fill", "layer", "stack")


def desugar_density(
    field: str,
    *,
    chart_encoding: Any = None,
    bandwidth: Any = "scott",
    bw_adjust: float = 1.0,
    kernel: str = "gaussian",
    n: int = 512,
    extent: Any = None,
    cumulative: bool = False,
    multiple: str = "layer",
    fill: bool = True,
    groupby: Any = None,
    orientation: str = "vertical",
    # Bivariate-only kwargs (forwarded to desugar_contour when both x and y
    # encoded). Ignored on the 1D path.
    thresholds: int = 6,
    smooth: bool = True,
    cmap: str | None = None,
) -> MarkDesugarResult:
    """Kernel-density-estimate area/line mark desugar.

    Routes to either a 1D or bivariate 2D KDE path based on the chart's
    encoding state.

    **1D path** (only x encoded): single-mark ``MarkDesugarResult`` with
    ``mark="area"`` (when ``fill=True``) or ``mark="line"`` (when
    ``fill=False``), a single ``Kde`` transform, and the encoding remap
    ``{"x": "value", "y": "density"}``.

    **2D/bivariate path** (both x AND y encoded): routes through
    ``desugar_contour(fill=True)`` and returns its layered
    ``MarkDesugarResult`` (``.layers`` set).

    Data contract (1D path)
    -----------------------
    Input: DataFrame with numeric column ``field``.

    ``Kde`` (unnamed) produces: ``[value (Float64), density (Float64)]``

    Layers emitted (1D)
    -------------------
    Single-mark mode, not a layer list.  The chart's x encoding remaps to
    ``"value"`` and y to ``"density"``.

    Parameters
    ----------
    field : str
        Numeric column to estimate density from.
    chart_encoding : dict or None, default None
        The chart's current encoding dict.  If both ``"x"`` and ``"y"``
        are bound, routes through ``desugar_contour``.
    bandwidth : str or float, default "scott"
        KDE bandwidth rule or numeric value.
    bw_adjust : float, default 1.0
        Multiplier applied to the resolved bandwidth after rule evaluation.
        Works with all bandwidth rules (``"scott"``, ``"silverman"``, numeric).
    kernel : str, default "gaussian"
        Kernel function, validated against the live
        ``transform_registry("kde")["methods"]`` vocabulary (the Rust
        dispatch's ``kde::KERNELS``, queried at call time — spec §4.6) and
        forwarded to the ``Kde`` transform.
    n : int, default 512
        Number of evaluation points.
    extent : tuple[float, float] or None, default None
        Explicit ``[min, max]`` range for the KDE.
    cumulative : bool, default False
        Whether to emit the cumulative density rather than the PDF.
    multiple : {"layer", "stack", "fill", "dodge"}, default "layer"
        How to render multiple density curves.  ``"stack"`` and ``"fill"``
        stack areas; ``"dodge"`` is not yet implemented.
    fill : bool, default True
        If ``True`` (default), emit ``mark_area``; otherwise ``mark_line``.
    groupby : str, optional
        Optional grouping column forwarded to the ``Kde`` transform.  When
        set, the density is computed per group and the output retains the
        group column so a downstream ``color=`` encoding can map to it.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Axis along which the density curve is drawn.  ``"vertical"`` (the
        default) puts the kernel value on the x-axis and the density on the
        y-axis.  ``"horizontal"`` swaps them — used for JointChart's right
        marginal where the data dimension shares the y-axis with the centre
        scatter.  Output column ordering does not change; only the
        ``encoding_remap`` flips ``x`` ↔ ``y``.
    thresholds : int, default 6
        Passed through to ``desugar_contour`` on the bivariate path.
    smooth : bool, default True
        Passed through to ``desugar_contour`` on the bivariate path.
    cmap : str or None, default None
        Passed through to ``desugar_contour`` on the bivariate path.  ``None``
        defers to the theme's sequential scheme.

    Returns
    -------
    MarkDesugarResult
        Single-mark mode on the 1D path: ``.mark`` is ``"area"`` or ``"line"``,
        ``.remap`` is ``{"x": "value", "y": "density"}``, and ``.position`` is a
        ``Stack`` adjuster for ``multiple="stack"`` or ``"fill"``.  Layered mode
        on the 2D bivariate path (``.layers`` set, via ``desugar_contour``).

    Raises
    ------
    ValueError
        If ``multiple`` is not one of ``"layer"``, ``"stack"``, ``"fill"``,
        ``"dodge"`` (dodge raises because it is not yet implemented).

    Examples
    --------
    >>> result = desugar_density("value")
    >>> result.mark
    'area'
    >>> result.remap
    {'x': 'value', 'y': 'density'}
    """
    # Bivariate routing: when the chart has both x and y bound, emit a 2D KDE
    # contour fill instead of a 1D KDE area.
    if chart_encoding is not None:
        x_enc = chart_encoding.get("x")
        y_enc = chart_encoding.get("y")
        if x_enc is not None and y_enc is not None:
            from ferrum.encoding.base import ChannelBase
            from ferrum.marks.heavy_stat import desugar_contour

            x_field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
            y_field = y_enc.field if isinstance(y_enc, ChannelBase) else y_enc
            return desugar_contour(
                x_field,
                y_field,
                fill=True,
                bandwidth=bandwidth,
                thresholds=thresholds,
                smooth=smooth,
                cmap=cmap,
                groupby=groupby if isinstance(groupby, str) else None,
            )

    if kernel is not None:
        validate_choice("mark_density", "kernel", kernel, transform_registry("kde")["methods"])
    validate_choice("mark_density", "multiple", multiple, _MULTIPLE_MODES)
    validate_choice("mark_density", "orientation", orientation, _ORIENTATIONS)
    kde_kwargs: dict = dict(
        bandwidth=bandwidth,
        bw_adjust=float(bw_adjust),
        n=n,
        extent=extent,
        cumulative=cumulative,
        # shared_extent=True forces a global x-grid across groups (required for stack/fill).
        shared_extent=(multiple in ("stack", "fill")),
        kernel=kernel or "gaussian",
    )
    if groupby is not None:
        kde_kwargs["groupby"] = groupby
    transforms = [Kde(field, **kde_kwargs)]
    # Kde produces columns ("value", "density") — remap both x and y.
    # With groupby, output also contains the group column for color encoding.
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either column raises here instead of silently
    # reading a stale name.
    _kde_cols = registry_output_columns("kde", ["value", "density"])
    if orientation == "horizontal":
        encoding_remap = {"y": _kde_cols["value"], "x": _kde_cols["density"]}
    else:
        encoding_remap = {"x": _kde_cols["value"], "y": _kde_cols["density"]}
    mark = "area" if fill else "line"

    # multiple="stack" and "fill" use the Stack position adjuster.
    # multiple="dodge": Rust KDE normalize_mode handles per-group normalization.
    #
    # GH #77: orientation="horizontal" remaps the value column onto
    # encoding.x directly (`encoding_remap` above) without setting
    # CoordFlip, so the Stack must say so explicitly — otherwise Rust's
    # `apply_stack` falls back to `coord_flipped` (False here) and
    # cumulates the wrong column.
    stack_value_axis = "x" if orientation == "horizontal" else None
    if multiple == "stack":
        from ferrum.position import Stack

        position = Stack(offset="zero", value_axis=stack_value_axis)
    elif multiple == "fill":
        from ferrum.position import Stack

        position = Stack(offset="normalize", value_axis=stack_value_axis)
    elif multiple == "dodge":
        # For density plots, "dodge" produces overlapping curves (same as "layer").
        # Histogram dodge (side-by-side bars) is handled separately in mark_histogram.
        position = None
    else:
        position = None

    return MarkDesugarResult(
        mark=mark, transforms=transforms, remap=encoding_remap, position=position
    )


def desugar_histogram(
    field: str,
    *,
    bin_count: Any = None,
    bin_width: Any = None,
    extent: Any = None,
    nice: bool = True,
    density: bool = False,
    cumulative: bool = False,
    right: bool = False,
    multiple: str = "layer",
    groupby: Any = None,
    orientation: str = "vertical",
) -> MarkDesugarResult:
    """Histogram mark desugar.

    Converts ``chart.mark_histogram(...)`` into a ``Bin`` transform plus a
    bar layer that reads the binned output columns.

    Data contract
    -------------
    Input: DataFrame with numeric column ``field``.

    ``Bin`` (unnamed) produces:
    ``[bin_start (Float64), bin_end (Float64), count (Int64),
    density (Float64)]``

    Encoding remap
    --------------
    ``x → bin_start``, ``x2 → bin_end``,
    ``y → "density"`` (when ``density=True``) or ``"count"`` (default).

    Parameters
    ----------
    field : str
        Numeric column to bin.
    bin_count : int or None, default None
        Desired number of bins.  ``None`` uses Sturges' rule.
    bin_width : float or None, default None
        Explicit bin width in data units; overrides ``bin_count``.
    extent : tuple[float, float] or None, default None
        Explicit ``[min, max]`` range for the binning.
    nice : bool, default True
        Whether to extend the bin range to round numbers.
    density : bool, default False
        If ``True``, encode y as ``"density"`` (area = 1) instead of
        ``"count"``.
    cumulative : bool, default False
        Whether to accumulate counts/density across bins.
    right : str, default False
        Reserved for future use (no-op today — bin intervals are always
        left-closed, right-open ``[lo, hi)``).
    multiple : str, default "layer"
        Reserved for future use (no-op today — only ``"layer"`` overlap
        is supported; stacking deferred).
    groupby : list or None, default None
        Optional grouping columns forwarded to the ``Bin`` transform.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Axis along which bars are drawn.  ``"vertical"`` (the default) puts
        bin edges on the x-axis and count/density on the y-axis.
        ``"horizontal"`` swaps them — used for JointChart's right marginal
        where the binned dimension shares the y-axis with the centre
        scatter.  Output column ordering does not change; only the
        ``encoding_remap`` keys flip ``x`` ↔ ``y`` (so the chart needs to
        bind the binned field via ``encode(y=...)`` rather than
        ``encode(x=...)``).

    Returns
    -------
    MarkDesugarResult
        Single-mark mode with ``.mark="bar"`` and ``.remap`` mapping the
        positional channels onto the binned output columns.  ``.position`` is
        a ``Stack`` or ``Dodge`` adjuster when ``multiple`` requests one.

    Examples
    --------
    >>> result = desugar_histogram("tip")
    >>> result.mark
    'bar'
    >>> result.remap
    {'x': 'bin_start', 'x2': 'bin_end', 'y': 'count'}
    """
    if right:
        raise ValueError(
            "mark_histogram(right=True) is not supported; "
            "bins are always left-closed, right-open [lo, hi)"
        )
    validate_choice("mark_histogram", "multiple", multiple, _MULTIPLE_MODES)
    validate_choice("mark_histogram", "orientation", orientation, _ORIENTATIONS)
    bin_kwargs: dict = dict(
        bin_count=bin_count,
        bin_width=bin_width,
        extent=extent,
        nice=nice,
        cumulative=cumulative,
        # shared_extent=True forces a global bin-edge range across all groups so
        # that multiple="stack" / "fill" always produces aligned bars that
        # apply_stack can match on x-key pairs.
        shared_extent=(multiple in ("stack", "fill")),
    )
    if groupby is not None:
        bin_kwargs["groupby"] = groupby
    transforms = [Bin(field, **bin_kwargs)]
    # Phase 5 Bin produces columns (bin_start, bin_end, count, density).
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of any of them raises here instead of silently
    # remapping a channel onto a column the transform no longer emits.
    _bin_cols = registry_output_columns("bin", ["bin_start", "bin_end", "count", "density"])
    count_column = _bin_cols["density"] if density else _bin_cols["count"]
    if orientation == "horizontal":
        # Horizontal bars: bin extents on y-axis, count on x-axis.
        encoding_remap = {
            "y": _bin_cols["bin_start"],
            "y2": _bin_cols["bin_end"],
            "x": count_column,
        }
    else:
        encoding_remap = {
            "x": _bin_cols["bin_start"],
            "x2": _bin_cols["bin_end"],
            "y": count_column,
        }

    # multiple= → position adjustment on the count/density axis.
    #
    # GH #77: orientation="horizontal" remaps the count/density column onto
    # encoding.x directly (`encoding_remap` above) without setting
    # CoordFlip, so the Stack must say so explicitly — otherwise Rust's
    # `apply_stack` falls back to `coord_flipped` (False here) and
    # cumulates the wrong column.
    stack_value_axis = "x" if orientation == "horizontal" else None
    position = None
    if multiple == "stack":
        from ferrum.position import Stack

        position = Stack(offset="zero", value_axis=stack_value_axis)
    elif multiple == "fill":
        from ferrum.position import Stack

        position = Stack(offset="normalize", value_axis=stack_value_axis)
    elif multiple == "dodge":
        from ferrum.position import Dodge

        position = Dodge()

    return MarkDesugarResult(
        mark="bar", transforms=transforms, remap=encoding_remap, position=position
    )


def desugar_smooth(
    x_field: str,
    y_field: str,
    *,
    method: str = "loess",
    ci: float | None = None,
    bandwidth: float | None = None,
    degree: int | None = None,
    k: int | None = None,
    n: int = 200,
    seed: int = 0,
    x_bins: Any = None,
    x_estimator: Any = None,
    show_metrics: bool = False,
    groupby: Any = None,
    name: str | None = None,
    x_range: list[float] | None = None,
) -> MarkDesugarResult:
    """Smoothed-regression line (LOESS/etc.) mark desugar.

    Converts ``chart.mark_smooth(...)`` into a ``Smooth`` transform plus
    either a single line layer (no CI) or a ribbon + line layer pair (with CI).

    Data contract
    -------------
    Input: DataFrame with numeric columns ``x_field`` and ``y_field``.

    Without CI — ``Smooth`` (unnamed) produces: ``[x (Float64), y (Float64)]``

    With CI — ``Smooth`` (named ``"smooth"``) produces:
    ``[x (Float64), y (Float64), ci_lower (Float64), ci_upper (Float64)]``

    Layers emitted
    --------------
    *No CI*: single-mark mode with ``.mark="line"`` and remap
    ``{"x": "x", "y": "y"}`` (no ``.layers``).

    *With CI*:
    1. ``ribbon`` — ``y="ci_lower"``, ``y2="ci_upper"``, ``opacity=0.3``
       (CI band).
    2. ``line``   — ``y="y"`` (mean/fitted line).

    Parameters
    ----------
    x_field : str
        Numeric predictor column.
    y_field : str
        Numeric response column.
    method : str, default "loess"
        Smoothing method (e.g. ``"loess"``, ``"linear"``, ``"quadratic"``,
        ``"glm"``, ``"gam"``).  Validated in Python against the live
        ``transform_registry("smooth")["methods"]`` vocabulary (spec §4.6)
        before the ``Smooth`` transform is constructed, so an unknown
        spelling raises here, naming it, rather than at the Rust boundary.
    ci : float or None, default None
        Confidence interval level (e.g. ``0.95``).  ``None`` disables the
        CI band and returns a single-mark (line) result.
    bandwidth : float or None, default None
        LOESS span, the fraction of points in each local neighbourhood.
        ``None`` (the default) forwards nothing, so ``Smooth``'s own default
        (``0.75``) applies. Forwarded only when explicitly set, exactly as
        ``degree`` is: ``bandwidth`` is the LOESS knob and no other method
        reads it, so an explicit value on a non-loess method raises rather
        than silently doing nothing (``method="gam"`` takes ``k=``,
        ``"lm"``/``"linear"`` take ``degree=``).
    degree : int or None, default None
        Polynomial degree passed through to the ``Smooth`` transform.
        Acceptance depends on ``method``:

        - ``"lm"``/``"linear"``: any ``degree >= 1``. Omitting it or
          passing ``1`` keeps a plain linear fit; ``>= 2`` promotes to a
          real polynomial fit (``degree=2`` is byte-equal to
          ``method="quadratic"``). ``degree=0`` raises. A degree that
          leaves the design rank-deficient for the data at hand (fewer
          usable points than ``degree + 1`` after dropping null/NaN rows
          and any ``x_bins`` aggregation) also raises, but only for a
          whole-frame fit; inside one group of a ``groupby`` fit, or one
          panel of a faceted chart, that slice's output degrades to NaN
          instead.
        - ``"quadratic"``/``"cubic"``: accepted only when it matches the
          method's fixed degree (``2``/``3`` respectively) as a no-op;
          any other value raises, naming ``method="lm", degree=<d>`` as
          the way to get that degree.
        - ``"loess"``: ``1`` or ``2`` (Rust defaults to ``2`` when
          omitted); any other value raises.
        - ``"log"``/``"sqrt"``/``"glm"``/``"gam"``: no degree concept;
          passing any value raises. ``"gam"`` fits a penalized cubic
          B-spline whose degree is fixed at 3 and whose smoothness is set
          by a GCV-chosen penalty, so there is nothing a ``degree`` could
          mean for it.
    k : int or None, default None
        ``method="gam"``'s spline basis dimension -- the number of B-spline
        basis functions the penalized fit may spend, i.e. the *ceiling* on
        how wiggly the curve can get before the GCV-chosen penalty pulls it
        back. ``None`` (the default) forwards nothing, so ``Smooth``'s own
        default of ``10`` applies. The effective dimension is
        ``clamp(k, 4, distinct_x)``: a ``k`` above the number of distinct x
        values is capped there (no basis can outrun its data), and ``k < 4``
        raises, since no smaller cubic basis exists. Raise it when the
        default under-smooths a signal with many turning points. Raises for
        any method other than ``"gam"`` -- the other methods have their own
        knobs (``degree=`` for ``"lm"``/``"linear"``, ``bandwidth=`` for
        ``"loess"``) and would otherwise ignore it silently.
    n : int, default 200
        Number of evaluation points in the output.
    seed : int, default 0
        RNG seed for bootstrap CI (``ci`` path only).  Pinned to
        ``ChaCha8Rng`` for byte-deterministic SVG goldens.
    x_bins : int or None, default None
        Optional number of x bins for aggregated scatter smoothing.
    x_estimator : str or None, default None
        Aggregation function per x bin (e.g. ``"mean"``).
    show_metrics : bool, default False
        When ``True``, emit a ``"metrics"`` layer annotating the fit with
        regression statistics (e.g. R²).
    groupby : str or None, default None
        Optional grouping column forwarded to the ``Smooth`` transform.
        When set, the smoothing is computed per group and the output
        retains the group column so a downstream ``color=`` encoding
        can map to it.
    name : str or None, default None
        Optional name for the ``Smooth`` transform output, used to wire
        downstream layers to its emitted columns.
    x_range : [float, float] or None, default None
        Explicit evaluation domain ``[x_min, x_max]`` forwarded to the
        ``Smooth`` transform.  When set, the fit line is evaluated over
        this range rather than the observed data range — used by
        ``lmplot(truncate=False)`` to extend the fit beyond the data.

    Returns
    -------
    MarkDesugarResult
        Single-mark mode (``.mark="line"``, remap ``{"x": "x", "y": "y"}``)
        when ``ci=None`` and ``show_metrics=False``.  Layered mode (``.layers``
        set: a ribbon CI band then the fitted line) when ``ci`` is set.

    Examples
    --------
    >>> result = desugar_smooth("x", "y")
    >>> result.mark
    'line'
    >>> result_ci = desugar_smooth("x", "y", ci=0.95)
    >>> [layer.mark for layer in result_ci.layers]
    ['ribbon', 'line']
    """
    validate_choice("mark_smooth", "method", method, transform_registry("smooth")["methods"])
    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of any of these columns raises here instead of
    # silently reading a stale name. Only the single-mark path below reads
    # this lookup; the layered path gets its channels from
    # `build_fit_layers`, which performs the same lookup for the whole
    # fit family.
    _smooth_cols = registry_output_columns("smooth", ["x", "y"])

    if ci is None and not show_metrics:
        # 8a-compatible single-line path: keep the single-mark result so the
        # 6 SVG goldens stay byte-identical. Only thread x_bins/x_estimator/
        # degree/bandwidth/k when explicitly set; otherwise omit so Rust's own
        # per-method default applies (batch-D T9: forwarding an
        # always-2 `degree` here would silently promote every bare
        # `method="lm"` call to a quadratic fit once "lm"+degree>=2
        # routes to a polynomial). `bandwidth` joined that rule in R2:
        # forwarding it unconditionally meant every method got an explicit
        # LOESS span it does not read, which only worked because `PySmooth`
        # tolerates an explicit value equal to its own default. Omitting it
        # makes that tolerance matter to direct `Smooth(...)` callers alone.
        smooth_kwargs: dict = dict(method=method, ci=None, n=n)
        if bandwidth is not None:
            smooth_kwargs["bandwidth"] = bandwidth
        if degree is not None:
            smooth_kwargs["degree"] = degree
        if k is not None:
            smooth_kwargs["k"] = k
        if x_bins is not None:
            smooth_kwargs["x_bins"] = x_bins
        if x_estimator is not None:
            smooth_kwargs["x_estimator"] = x_estimator
        if groupby is not None:
            smooth_kwargs["groupby"] = groupby
        if name is not None:
            smooth_kwargs["name"] = name
        if x_range is not None:
            smooth_kwargs["x_range"] = x_range
        transforms = [Smooth(x_field, y_field, **smooth_kwargs)]
        encoding_remap = {"x": _smooth_cols["x"], "y": _smooth_cols["y"]}
        return MarkDesugarResult(mark="line", transforms=transforms, remap=encoding_remap)

    # Layered path: ci band and/or metrics-corner overlay. `bandwidth`,
    # `degree` and `k` are forwarded only when explicit — see the no-CI path
    # above for why.
    smooth_kwargs = dict(method=method, ci=ci, n=n, seed=seed, name="smooth")
    if bandwidth is not None:
        smooth_kwargs["bandwidth"] = bandwidth
    if degree is not None:
        smooth_kwargs["degree"] = degree
    if k is not None:
        smooth_kwargs["k"] = k
    if x_bins is not None:
        smooth_kwargs["x_bins"] = x_bins
    if x_estimator is not None:
        smooth_kwargs["x_estimator"] = x_estimator
    if groupby is not None:
        smooth_kwargs["groupby"] = groupby
    if x_range is not None:
        smooth_kwargs["x_range"] = x_range
    if show_metrics:
        # Schwabish SB-followup (2026-05-12): emit R²/RMSE/MAE columns
        # on the fit-grid output so a same-source mark_text layer reads
        # them. Computed in Rust against the raw input — no Python-side
        # OLS duplication.
        smooth_kwargs["inject_metrics"] = True
    transforms = [Smooth(x_field, y_field, **smooth_kwargs)]
    layers = build_fit_layers("smooth", data_source="smooth", ci=ci, metrics=show_metrics)
    return MarkDesugarResult(transforms=transforms, layers=layers)


register_layer_names(
    "smooth",
    frozenset(
        {
            "ribbon",
            "line",
            "metrics",
        }
    ),
)


# ---- Resolve adapters ---------------------------------------------------
# These convert the mark-specific deferred-resolution semantics (orientation
# field-channel choice, bivariate routing, post-desugar remap rewriting) into
# the generic ``desugar_fn(x_field, y_field, **kwargs)`` protocol that
# ``Chart._resolve_pending`` consumes uniformly.


def _resolve_density(x_field, y_field, **kwargs):
    """Resolve a deferred ``mark_density()`` call once the encoding is known."""
    orientation = kwargs.get("orientation", "vertical")
    field = y_field if orientation == "horizontal" else x_field
    if field is None:
        ch = "y" if orientation == "horizontal" else "x"
        raise ValueError(f"mark_density() requires .encode({ch}=...) to specify the density field")
    # Reconstruct a chart_encoding hint so density's bivariate path can route
    # through desugar_contour(fill=True) when both x AND y are encoded.
    chart_encoding: dict = {}
    if x_field is not None:
        chart_encoding["x"] = x_field
    if y_field is not None:
        chart_encoding["y"] = y_field
    return desugar_density(field, chart_encoding=chart_encoding, **kwargs)


def _resolve_histogram(x_field, y_field, **kwargs):
    """Resolve a deferred ``mark_histogram()`` call once the encoding is known.

    ``desugar_histogram`` already emits the correctly-oriented remap
    (vertical: ``{x, x2, y}``; horizontal: ``{x, y, y2}``); pass through
    unchanged.
    """
    orientation = kwargs.get("orientation", "vertical")
    field = y_field if orientation == "horizontal" else x_field
    if field is None:
        ch = "y" if orientation == "horizontal" else "x"
        raise ValueError(
            f"mark_histogram() requires .encode({ch}=...) to specify the histogram field"
        )
    return desugar_histogram(field, **kwargs)


def _resolve_smooth(x_field, y_field, **kwargs):
    """Resolve a deferred ``mark_smooth()`` call once the encoding is known."""
    if x_field is None or y_field is None:
        raise ValueError("mark_smooth() requires .encode(x=..., y=...)")
    return desugar_smooth(x_field, y_field, **kwargs)


# Expose the wrapped desugar signature so ``inspect.signature(..)`` (which
# follows ``__wrapped__``) can partition user kwargs into transform vs. style
# keys in ``Chart._resolve_pending``.  The thin ``_resolve_*`` wrappers use
# ``**kwargs`` and would otherwise hide the real transform parameter names.
_resolve_density.__wrapped__ = desugar_density
_resolve_histogram.__wrapped__ = desugar_histogram
_resolve_smooth.__wrapped__ = desugar_smooth


def _build_prior_layer(mark, encoding, mark_kwargs, position):
    """Build a ``_Layer`` preserving a prior primitive mark as a scatter layer.

    Used by ``Chart._resolve_pending`` when ``mark_smooth()`` is called on a
    chart that already has a primitive mark set (e.g.
    ``chart.mark_point().mark_smooth()``).
    """
    return _Layer(
        mark=mark,
        encoding=dict(encoding),
        mark_kwargs=dict(mark_kwargs) if mark_kwargs else None,
        position=position,
    )
