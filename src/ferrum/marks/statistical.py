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
    """mark_density desugar.

    1D path (only x encoded): ``mark_area`` (when ``fill=True``, the default)
    or ``mark_line`` (when ``fill=False``) over ``Kde(field)`` with remap
    ``x → value``, ``y → density``. Returns the legacy 3-tuple
    ``(mark, transforms, remap)``.

    Bivariate path (both x AND y encoded — Phase 8b): routes through
    ``desugar_contour(fill=True)`` to emit a filled-contour layer over a
    2D KDE. Returns the 5-tuple
    ``("__layered__", transforms, None, None, layers)``.

    Every kwarg is explicitly named so Python rejects unknown kwargs at
    the call boundary — no silent-drop fallthrough. ``bw_adjust`` scales
    a numeric ``bandwidth`` multiplicatively; combining it with the
    string ``"scott"`` rule requires resolving Scott's factor against
    the data which the Kde transform does in Rust, so the no-op default
    1.0 passes through and any non-default value raises (the path-of-
    correctness fix lives in the Rust transform).
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
    """mark_histogram → mark_bar + Bin(field, ...) + count or density on y.

    Every kwarg is explicitly named so Python rejects unknown kwargs at
    the call boundary — no silent-drop fallthrough.
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
    """mark_smooth → mark_line + Smooth(x, y, ...).

    With ``ci=None`` (the default): single ``line`` mark layer, returns the
    legacy 3-tuple ``(mark, transforms, remap)``.

    With ``ci`` set (e.g. ``0.95``) — Phase 8b: layered output emitting a
    ribbon (CI band, semi-transparent) below a line, both bound to the same
    named ``Smooth`` transform output. Returns the 5-tuple
    ``("__layered__", transforms, None, None, layers)``.

    Every kwarg is explicitly named so Python rejects unknown kwargs at
    the call boundary — no silent-drop fallthrough.
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
