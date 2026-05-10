"""Statistical mark desugaring — convert mark_density/histogram/smooth kwargs
into (mark, transforms, encoding_remap) tuples consumed by Chart."""
from __future__ import annotations

from typing import Any

from ferrum import Bin, Kde, Smooth


def desugar_density(field: str, *, chart_encoding: Any = None, **kwargs: Any) -> tuple:
    """mark_density desugar.

    1D path (only x encoded): ``mark_area`` + ``Kde(field)`` + remap
    ``x → value``, ``y → density``. Returns the legacy 3-tuple
    ``(mark, transforms, remap)``.

    Bivariate path (both x AND y encoded — Phase 8b): routes through
    ``desugar_contour(fill=True)`` to emit a filled-contour layer over a
    2D KDE. Returns the 5-tuple
    ``("__layered__", transforms, None, None, layers)``.
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
            # Forward only kwargs that desugar_contour understands; the 1D
            # KDE-only kwargs (n, extent, cumulative, kernel, multiple) are
            # silently dropped in the bivariate branch.
            contour_kwargs = {}
            if "bandwidth" in kwargs:
                contour_kwargs["bandwidth"] = kwargs["bandwidth"]
            if "thresholds" in kwargs:
                contour_kwargs["thresholds"] = kwargs["thresholds"]
            if "smooth" in kwargs:
                contour_kwargs["smooth"] = kwargs["smooth"]
            if "cmap" in kwargs:
                contour_kwargs["cmap"] = kwargs["cmap"]
            return desugar_contour(x_field, y_field, fill=True, **contour_kwargs)

    bandwidth = kwargs.pop("bandwidth", "scott")
    kernel = kwargs.pop("kernel", "gaussian")
    n = kwargs.pop("n", 512)
    extent = kwargs.pop("extent", None)
    cumulative = kwargs.pop("cumulative", False)
    # `multiple` parameter from spec §3.3 deferred (no stack support yet)
    if kwargs.pop("multiple", "layer") != "layer":
        # warn-once at Chart layer; here we just drop it
        pass

    transforms = [Kde(field, bandwidth=bandwidth, n=n, extent=extent, cumulative=cumulative)]
    # Phase 5 Kde produces columns ("value", "density") — remap both x and y.
    encoding_remap = {"x": "value", "y": "density"}
    return ("area", transforms, encoding_remap)


def desugar_histogram(field: str, **kwargs: Any) -> tuple[str, list, dict]:
    """mark_histogram → mark_bar + Bin(field, ...) + count or density on y."""
    bin_count = kwargs.pop("bin_count", None)
    bin_width = kwargs.pop("bin_width", None)
    extent = kwargs.pop("extent", None)
    nice = kwargs.pop("nice", True)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    right = kwargs.pop("right", False)
    multiple = kwargs.pop("multiple", "layer")

    transforms = [Bin(field, bin_count=bin_count, bin_width=bin_width, extent=extent,
                      nice=nice, cumulative=cumulative)]
    # Phase 5 Bin produces columns (bin_start, bin_end, count, density)
    y_column = "density" if density else "count"
    encoding_remap = {"x": "bin_start", "x2": "bin_end", "y": y_column}
    return ("bar", transforms, encoding_remap)


def desugar_smooth(x_field: str, y_field: str, **kwargs: Any) -> tuple:
    """mark_smooth → mark_line + Smooth(x, y, ...).

    With ``ci=None`` (the default): single ``line`` mark layer, returns the
    legacy 3-tuple ``(mark, transforms, remap)``.

    With ``ci`` set (e.g. ``0.95``) — Phase 8b: layered output emitting a
    ribbon (CI band, semi-transparent) below a line, both bound to the same
    named ``Smooth`` transform output. Returns the 5-tuple
    ``("__layered__", transforms, None, None, layers)``.
    """
    method = kwargs.pop("method", "loess")
    ci = kwargs.pop("ci", None)
    bandwidth = kwargs.pop("bandwidth", 0.75)
    degree = kwargs.pop("degree", 2)
    n = kwargs.pop("n", 200)
    seed = kwargs.pop("seed", 0)
    x_bins = kwargs.pop("x_bins", None)
    x_estimator = kwargs.pop("x_estimator", None)

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
