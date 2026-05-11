"""Statistical mark desugaring — convert mark_density/histogram/smooth kwargs
into (mark, transforms, encoding_remap) tuples consumed by Chart."""
from __future__ import annotations

from typing import Any

from ferrum import Bin, Kde, Smooth


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
    # Bivariate-only kwargs (forwarded to desugar_contour when both x and y
    # encoded). Ignored on the 1D path.
    thresholds: int = 6,
    smooth: bool = True,
    cmap: str = "viridis",
) -> tuple:
    """Kernel-density-estimate area/line mark desugar.

    Routes to either a 1D or bivariate 2D KDE path based on the chart's
    encoding state.

    **1D path** (only x encoded): returns the legacy 3-tuple
    ``(mark, transforms, remap)`` with ``mark="area"`` (when ``fill=True``)
    or ``mark="line"`` (when ``fill=False``), a single ``Kde`` transform, and
    the encoding remap ``{"x": "value", "y": "density"}``.

    **2D/bivariate path** (both x AND y encoded): routes through
    ``desugar_contour(fill=True)`` and returns the 5-tuple
    ``("__layered__", transforms, None, None, layers)``.

    Data contract (1D path)
    -----------------------
    Input: DataFrame with numeric column ``field``.

    ``Kde`` (unnamed) produces: ``[value (Float64), density (Float64)]``

    Layers emitted (1D)
    -------------------
    Returns legacy 3-tuple, not a layer list.  The chart's x encoding
    remaps to ``"value"`` and y to ``"density"``.

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
        Multiplier applied to a numeric ``bandwidth``.  Raises
        ``NotImplementedError`` when combined with a string bandwidth rule
        (e.g. ``"scott"``).
    kernel : str, default "gaussian"
        Reserved for future use (no-op today — the ``Kde`` transform uses
        Gaussian exclusively).
    n : int, default 512
        Number of evaluation points.
    extent : tuple[float, float] or None, default None
        Explicit ``[min, max]`` range for the KDE.
    cumulative : bool, default False
        Whether to emit the cumulative density rather than the PDF.
    multiple : str, default "layer"
        Reserved for future use (only ``"layer"`` is supported today;
        other values raise ``NotImplementedError``).
    fill : bool, default True
        If ``True`` (default), emit ``mark_area``; otherwise ``mark_line``.
    thresholds : int, default 6
        Passed through to ``desugar_contour`` on the bivariate path.
    smooth : bool, default True
        Passed through to ``desugar_contour`` on the bivariate path.
    cmap : str, default "viridis"
        Passed through to ``desugar_contour`` on the bivariate path.

    Returns
    -------
    tuple
        3-tuple ``(mark, transforms, remap)`` on the 1D path, or 5-tuple
        ``("__layered__", transforms, None, None, layers)`` on the 2D path.

    Raises
    ------
    NotImplementedError
        If ``multiple != "layer"`` or if ``bw_adjust`` is combined with a
        string bandwidth rule.
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
                x_field, y_field, fill=True,
                bandwidth=bandwidth, thresholds=thresholds,
                smooth=smooth, cmap=cmap,
            )

    del kernel  # informational; underlying Kde uses gaussian only
    if multiple != "layer":
        # `multiple` parameter from spec §3.3 deferred (no stack support yet).
        raise NotImplementedError(
            f"mark_density(multiple={multiple!r}) lands in Phase 11; "
            "only 'layer' is supported today."
        )
    # Resolve bw_adjust on the numeric path; raise on the "scott" path.
    if bw_adjust != 1.0:
        if isinstance(bandwidth, (int, float)):
            bandwidth = float(bandwidth) * float(bw_adjust)
        else:
            raise NotImplementedError(
                "mark_density(bw_adjust=...) with a string bandwidth rule "
                "('scott', etc.) requires resolving the rule on the data "
                "first; pass a numeric bandwidth (and bw_adjust multiplies "
                "it), or land bw_adjust support inside the Rust Kde."
            )

    transforms = [Kde(field, bandwidth=bandwidth, n=n, extent=extent, cumulative=cumulative)]
    # Phase 5 Kde produces columns ("value", "density") — remap both x and y.
    encoding_remap = {"x": "value", "y": "density"}
    mark = "area" if fill else "line"
    return (mark, transforms, encoding_remap)


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
) -> tuple[str, list, dict]:
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

    Returns
    -------
    tuple
        3-tuple ``("bar", transforms, encoding_remap)``.

    Examples
    --------
    >>> result = desugar_histogram("tip")
    >>> result[0]
    'bar'
    >>> result[2]
    {'x': 'bin_start', 'x2': 'bin_end', 'y': 'count'}
    """
    del right, multiple  # forwarded to renderer in a later phase
    bin_kwargs: dict = dict(bin_count=bin_count, bin_width=bin_width, extent=extent,
                            nice=nice, cumulative=cumulative)
    if groupby is not None:
        bin_kwargs["groupby"] = groupby
    transforms = [Bin(field, **bin_kwargs)]
    # Phase 5 Bin produces columns (bin_start, bin_end, count, density)
    y_column = "density" if density else "count"
    encoding_remap = {"x": "bin_start", "x2": "bin_end", "y": y_column}
    return ("bar", transforms, encoding_remap)


def desugar_smooth(
    x_field: str,
    y_field: str,
    *,
    method: str = "loess",
    ci: float | None = None,
    bandwidth: float = 0.75,
    degree: int = 2,
    n: int = 200,
    seed: int = 0,
    x_bins: Any = None,
    x_estimator: Any = None,
) -> tuple:
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
    *No CI*: returns the legacy 3-tuple ``("line", transforms, remap)``
    with remap ``{"x": "x", "y": "y"}``.

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
        Smoothing method (e.g. ``"loess"``, ``"linear"``, ``"quadratic"``).
    ci : float or None, default None
        Confidence interval level (e.g. ``0.95``).  ``None`` disables the
        CI band and returns a single-line legacy 3-tuple.
    bandwidth : float, default 0.75
        Smoothing bandwidth fraction (LOESS).
    degree : int, default 2
        Polynomial degree for the smoother.
    n : int, default 200
        Number of evaluation points in the output.
    seed : int, default 0
        RNG seed for bootstrap CI (``ci`` path only).  Pinned to
        ``ChaCha8Rng`` for byte-deterministic SVG goldens.
    x_bins : int or None, default None
        Optional number of x bins for aggregated scatter smoothing.
    x_estimator : str or None, default None
        Aggregation function per x bin (e.g. ``"mean"``).

    Returns
    -------
    tuple
        3-tuple ``("line", transforms, remap)`` when ``ci=None``, or
        5-tuple ``("__layered__", transforms, None, None, layers)``
        when ``ci`` is set.

    Examples
    --------
    >>> result = desugar_smooth("x", "y")
    >>> result[0]
    'line'
    >>> result_ci = desugar_smooth("x", "y", ci=0.95)
    >>> result_ci[0]
    '__layered__'
    """

    if ci is None:
        # 8a-compatible single-line path: keep the legacy 3-tuple shape so the
        # 6 SVG goldens stay byte-identical. Only thread x_bins/x_estimator when
        # explicitly set; otherwise omit (so existing goldens stay identical).
        smooth_kwargs: dict = dict(method=method, ci=None,
                                    bandwidth=bandwidth, degree=degree, n=n)
        if x_bins is not None:
            smooth_kwargs["x_bins"] = x_bins
        if x_estimator is not None:
            smooth_kwargs["x_estimator"] = x_estimator
        transforms = [Smooth(x_field, y_field, **smooth_kwargs)]
        encoding_remap = {"x": "x", "y": "y"}
        return ("line", transforms, encoding_remap)

    # CI band path (NEW in 8b — replaces former warn-once deferral).
    smooth_kwargs = dict(method=method, ci=ci, bandwidth=bandwidth,
                          degree=degree, n=n, seed=seed, name="smooth")
    if x_bins is not None:
        smooth_kwargs["x_bins"] = x_bins
    if x_estimator is not None:
        smooth_kwargs["x_estimator"] = x_estimator
    transforms = [Smooth(x_field, y_field, **smooth_kwargs)]
    layers = [
        {"mark": "ribbon",
         "encoding": {"x": "x", "y": "ci_lower", "y2": "ci_upper"},
         "mark_kwargs": {"opacity": 0.3},
         "data_source": "smooth"},
        {"mark": "line",
         "encoding": {"x": "x", "y": "y"},
         "data_source": "smooth"},
    ]
    return ("__layered__", transforms, None, None, layers)
