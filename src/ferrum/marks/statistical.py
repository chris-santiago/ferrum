"""Statistical mark desugaring — convert mark_density/histogram/smooth kwargs
into (mark, transforms, encoding_remap) tuples consumed by Chart."""
from __future__ import annotations

from typing import Any

from ferrum import Bin, Kde, Smooth


def desugar_density(field: str, **kwargs: Any) -> tuple[str, list, dict]:
    """mark_density → mark_area + Kde(field, ...) + remap y → density column."""
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
    # Phase 5 Kde produces columns (field, "density"); encoding_remap tells Chart
    # to treat the density column as y when wiring the area mark
    encoding_remap = {"y": "density"}
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

    transforms = [Bin(field, bin_count=bin_count, bin_width=bin_width, extent=extent, nice=nice)]
    # Phase 5 Bin produces columns (bin_start, bin_end, count, density)
    y_column = "density" if density else "count"
    encoding_remap = {"x": "bin_start", "x2": "bin_end", "y": y_column}
    return ("bar", transforms, encoding_remap)


def desugar_smooth(x_field: str, y_field: str, **kwargs: Any) -> tuple[str, list, dict]:
    """mark_smooth → mark_line + Smooth(x, y, ...). Phase 8a does NOT render the CI band."""
    method = kwargs.pop("method", "loess")
    ci = kwargs.pop("ci", None)
    bandwidth = kwargs.pop("bandwidth", 0.75)
    degree = kwargs.pop("degree", 2)
    n = kwargs.pop("n", 200)

    if ci is not None:
        # warn-once: CI band requires Phase 8b ribbon mark
        from ferrum._warn import warn_once
        warn_once("mark_smooth", "ci",
                  "mark_smooth(ci=...) requires the ribbon mark; deferred to Phase 8b. "
                  "Smooth curve rendered without CI band.")

    transforms = [Smooth(x_field, y_field, method=method, ci=None, bandwidth=bandwidth,
                         degree=degree, n=n)]
    # Phase 5 Smooth produces (x, y) columns named after as_ tuple; default ("x", "y")
    encoding_remap = {"x": "x", "y": "y"}
    return ("line", transforms, encoding_remap)
