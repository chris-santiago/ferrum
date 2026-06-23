"""Statistical mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` — they call ``self._set_composite_mark``,
``self._clone``, ``self._mark``, ``self._encoding``, ``self._data``, ``self._layers``,
``self._transforms``, ``self._position``, etc.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import Any

from ferrum._layer import _PRIMITIVE_MARKS
from ferrum.encoding.base import ChannelBase
from ferrum.marks.composite import _MISSING
from ferrum.marks.statistical import (
    _resolve_density,
    _resolve_histogram,
    _resolve_smooth,
)


def _normalize_orient(orient: str | None, horizontal: bool) -> bool:
    """Return the effective ``horizontal`` flag from the canonical ``orient`` kwarg.

    ``orient`` is the canonical spelling across the mark family.  ``horizontal``
    is a legacy bool alias accepted by boxplot/boxen/violin.  When ``orient`` is
    not ``None``, it wins; otherwise the explicit ``horizontal`` value is used.

    Valid values for ``orient``: ``"vertical"`` (default) or ``"horizontal"``.
    Any other string raises ``ValueError`` immediately so the error surfaces at
    the call site rather than silently being absorbed as a style kwarg.
    """
    if orient is not None:
        if orient not in ("vertical", "horizontal"):
            raise ValueError(f"orient must be 'vertical' or 'horizontal'; got {orient!r}")
        return orient == "horizontal"
    return bool(horizontal)


def _normalize_orient_to_orientation(kwargs: dict) -> dict:
    """Normalize ``orient=`` to ``orientation=`` for density/histogram desugars.

    ``desugar_density`` and ``desugar_histogram`` use ``orientation`` as their
    internal parameter name (historical spelling).  This helper pops ``orient``
    from a copy of *kwargs* and injects ``orientation`` so the desugar receives
    its expected param name.  ``orient`` wins over ``orientation`` when both are
    present.

    Returns a new dict; the caller's original dict is not mutated.
    """
    if "orient" not in kwargs:
        return kwargs
    orient_val = kwargs["orient"]
    if orient_val not in ("vertical", "horizontal"):
        raise ValueError(f"orient must be 'vertical' or 'horizontal'; got {orient_val!r}")
    new_kwargs = {k: v for k, v in kwargs.items() if k != "orient"}
    new_kwargs["orientation"] = orient_val
    return new_kwargs


class StatisticalMarksMixin:
    """Mixin providing statistical mark methods for Chart."""

    # ---- Marks (statistical) ----

    def mark_density(self, *, position=None, **kwargs) -> "Chart":
        """Render a kernel-density estimate (KDE) curve or filled contour.

        When only ``x`` is encoded, draws a 1-D KDE area curve.  When both
        ``x`` and ``y`` are encoded, renders a bivariate filled-contour density
        (routes through ``desugar_contour(fill=True)``).  Can be called before
        or after ``.encode()``.

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            KDE bandwidth.  ``"scott"`` and ``"silverman"`` use the
            corresponding rule-of-thumb estimates.  Default is ``"scott"``.
        kernel : str, optional
            Kernel type: ``"gaussian"``, ``"tophat"``, ``"epanechnikov"``,
            ``"exponential"``, ``"linear"``, ``"cosine"``.  Default is
            ``"gaussian"``.
        extent : list of float or None, optional
            ``[min, max]`` evaluation range.  Defaults to data extent.
        cumulative : bool, optional
            Produce a CDF instead of a PDF.  Default is ``False``.
        n : int, optional
            Number of evaluation points.  Default is ``200``.
        multiple : str, optional
            How to handle multiple densities when a ``color`` encoding is
            present: ``"layer"`` (default), ``"stack"``, ``"fill"``.
        orient : {"vertical", "horizontal"}, optional
            Canonical orientation alias for ``orientation``.  ``"horizontal"``
            swaps the value and density axes.  Takes precedence over
            ``orientation`` when both are given.
        orientation : {"vertical", "horizontal"}, optional
            Legacy spelling.  ``orient`` is preferred.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the rendered area/line layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for density rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"val": [1.0, 2.1, 1.8, 3.0, 2.5]})
        >>> fm.Chart(df).mark_density().encode(x="val")
        Chart(mark='area', encoding=['x'])
        """
        # Normalize orient= → orientation= so the desugar's existing param name
        # is used.  orient= wins over orientation= when both are present.
        kwargs = _normalize_orient_to_orientation(kwargs)
        return self._set_composite_mark(
            "density",
            _resolve_density,
            kwargs,
            placeholder="area",
            position=position,
        )

    def mark_histogram(self, *, position=None, **kwargs) -> "Chart":
        """Render data as a histogram.

        Bins the ``x``-encoded column and encodes bin extents as ``x``/``x2``
        with counts (or densities) on ``y``.  Can be called before or after
        ``.encode(x=...)``.

        Parameters
        ----------
        bin_count : int, optional
            Target number of bins.  Ignored when ``bin_width`` is set.
            Default is chosen automatically from Sturges' rule.
        bin_width : float, optional
            Exact bin width in data units.  Overrides ``bin_count``.
        density : bool, optional
            Normalise counts to a probability density.  Default is ``False``.
        right : bool, optional
            Whether bins are closed on the right.  Default is ``False``.
            ``right=True`` is not currently supported and raises
            ``ValueError``; bins are always left-closed, right-open
            ``[lo, hi)``.
        cumulative : bool, optional
            Render a cumulative histogram.  Default is ``False``.
        multiple : str, optional
            How to handle grouped histograms when a ``color`` encoding is
            present: ``"layer"`` (default), ``"stack"``, ``"dodge"``.
        orient : {"vertical", "horizontal"}, optional
            Canonical orientation alias for ``orientation``.  ``"horizontal"``
            swaps the bin-edge and count axes.  Takes precedence over
            ``orientation`` when both are given.
        orientation : {"vertical", "horizontal"}, optional
            Legacy spelling.  ``orient`` is preferred.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the rendered bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for histogram rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"val": [1.0, 1.2, 2.3, 3.1, 2.8, 1.5]})
        >>> fm.Chart(df).mark_histogram(bin_count=5).encode(x="val")
        Chart(mark='bar', encoding=['x', 'x2', 'y'])
        """
        # Normalize orient= → orientation= so the desugar's existing param name
        # is used.  orient= wins over orientation= when both are present.
        kwargs = _normalize_orient_to_orientation(kwargs)
        return self._set_composite_mark(
            "histogram",
            _resolve_histogram,
            kwargs,
            placeholder="bar",
            position=position,
        )

    def mark_smooth(self, *, position=None, **kwargs) -> "Chart":
        """Render a smoothed regression line with optional confidence interval band.

        Fits a smooth curve through ``(x, y)`` data using the method specified by
        ``method``.  When ``ci`` is set, emits a layered ribbon (CI band) + line
        chart.  Can be called before or after ``.encode(x=..., y=...)``.

        Parameters
        ----------
        method : str, optional
            Smoothing method.  Default is ``"loess"``.

            - ``"lm"`` or ``"linear"`` -- linear regression (OLS, degree-1 polynomial)
            - ``"loess"`` -- local regression (LOWESS)
            - ``"quadratic"`` -- polynomial regression, degree 2
            - ``"cubic"`` -- polynomial regression, degree 3
            - ``"log"`` -- linear fit on log-transformed x
            - ``"sqrt"`` -- linear fit on sqrt-transformed x
        degree : int, optional
            Polynomial degree for ``"linear"``/``"quadratic"``/``"cubic"``.
        bandwidth : float, optional
            LOESS bandwidth fraction in ``(0, 1]``.  Default is ``0.75``.
        n : int, optional
            Number of evaluation points along the x range.  Default is ``200``.
        ci : float or None, optional
            Confidence level for the interval band, e.g. ``0.95``.  When set,
            produces a layered ribbon + line chart.  Default is ``None``.
        groupby : str, optional
            Group-key column (Utf8).  When set, the smooth is computed
            independently per group and the group column is preserved in
            the output so downstream ``color=`` encoding maps to it.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to all emitted layers (the regression
            line, plus the CI ribbon when ``ci`` is set).

        Returns
        -------
        Chart
            New ``Chart`` configured for smooth rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.1, 1.9, 3.1, 4.0]})
        >>> fm.Chart(df).mark_smooth(method="linear", ci=0.95).encode(x="x", y="y")
        Chart(mark='line', encoding=['x', 'y'])
        """
        _pm = self._mark if self._mark in _PRIMITIVE_MARKS else None
        return self._set_composite_mark(
            "smooth",
            _resolve_smooth,
            kwargs,
            placeholder="line",
            position=position,
            prior_mark=_pm,
        )

    # ---- Marks (deferred) ----

    def mark_boxplot(
        self,
        *,
        whisker_mult: float | str = _MISSING,  # type: ignore[assignment]
        size=None,
        width=None,
        outliers=True,
        color_field=None,
        horizontal=False,
        orient=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Tukey boxplot via composite mark desugaring.

        Desugars to a multi-layer chart: box (IQR rect), upper and lower
        whisker rules, median rule, and (optionally) outlier points.  Requires
        a categorical ``x`` encoding and a continuous ``y`` (or the reverse
        when ``horizontal=True`` or ``orient="horizontal"``).

        Parameters
        ----------
        whisker_mult : float or "min-max", optional
            Whisker reach as a multiple of IQR, or ``"min-max"`` to extend
            whiskers to the data minimum/maximum.  Default is ``1.5``.

            .. deprecated::
                The old spelling ``extent=`` is accepted as an alias for one
                release.  Use ``whisker_mult=`` instead.  ``extent`` is
                reserved for data-domain ``[min, max]`` bounds.
        size : float or None, optional
            Box band-width as a fraction of the category band (default ``0.6``).
            Alias: ``width``.
        width : float or None, optional
            Alias for ``size``.  Controls box band-width as a fraction of the
            category band.  When both ``size`` and ``width`` are provided,
            ``size`` takes precedence.
        outliers : bool, optional
            Whether to draw outlier points beyond the whiskers.  Default is
            ``True``.
        color_field : str or None, optional
            Column name to drive per-group fill colour on the box layer.
        horizontal : bool, optional
            Swap x and y axes so boxes run horizontally.  Default is ``False``.
            Legacy alias; prefer ``orient="horizontal"``.
        orient : {"vertical", "horizontal"} or None, optional
            Canonical orientation control.  ``"horizontal"`` is equivalent to
            ``horizontal=True``.  When both are given, ``orient`` wins if it is
            not ``None``.
        position : Position, optional
            Position adjustment -- e.g. ``fm.Dodge()`` for side-by-side boxes.
        **mark_kwargs
            Additional mark-style overrides forwarded to each constituent layer.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the boxplot.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"species": ["a"]*5 + ["b"]*5, "val": [1,2,3,2,1,4,5,4,5,6]})
        >>> fm.Chart(df).mark_boxplot().encode(x="species", y="val")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.composite import desugar_boxplot, _resolve_deprecated_extent

        # width is an alias for size; size takes precedence when both are given.
        effective_size = size if size is not None else width
        # Normalize orient/horizontal: orient= is canonical; horizontal= is a legacy
        # alias. When orient is given explicitly, it overrides horizontal.
        effective_horizontal = _normalize_orient(orient, horizontal)
        # Resolve the deprecated extent= alias at the public API level using the
        # shared resolver from composite.py.  Sentinel-based detection ensures that
        # whisker_mult=1.5 (the default value, passed explicitly) is correctly
        # treated as "canonically supplied" and triggers TypeError when extent= is
        # also given.  mark_kwargs is mutated in-place (extent popped) by the helper.
        whisker_mult = _resolve_deprecated_extent(
            "whisker_mult", whisker_mult, mark_kwargs, real_default=1.5
        )

        return self._set_composite_mark(
            "boxplot",
            desugar_boxplot,
            {
                "whisker_mult": whisker_mult,
                "size": effective_size,
                "outliers": outliers,
                "color_field": color_field,
                "horizontal": effective_horizontal,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_boxen(
        self,
        *,
        k_depth: str = "tukey",
        k_proportion: float = 0.007,
        outlier_threshold: float = 1.5,
        palette=None,
        horizontal: bool = False,
        orient=None,
        color_field=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a letter-value (boxen) plot via composite mark desugaring.

        Produces nested rectangular bands for each letter-value depth, a median
        rule, and an outlier-point layer via the ``LetterValue`` transform.
        Requires a categorical ``x`` encoding and a continuous ``y`` (or the
        reverse when ``horizontal=True`` or ``orient="horizontal"``).

        Parameters
        ----------
        k_depth : {"proportion", "trustworthy", "full"}, default "proportion"
            Rule for choosing the number of letter-value levels.
        k_proportion : float, optional
            Proportion parameter used when ``k_depth="proportion"``.  Default
            is ``0.007``.
        outlier_threshold : float, optional
            IQR multiple beyond which points are considered outliers.  Default
            is ``1.5``.
        palette : list of str or None, optional
            Colour palette applied to successive depth bands.  ``None`` uses
            the active theme's categorical palette.
        horizontal : bool, optional
            Swap axes so bands run horizontally.  Default is ``False``.
            Legacy alias; prefer ``orient="horizontal"``.
        orient : {"vertical", "horizontal"} or None, optional
            Canonical orientation control.  ``"horizontal"`` is equivalent to
            ``horizontal=True``.  When both are given, ``orient`` wins if it is
            not ``None``.
        color_field : str or None, optional
            Column name to drive per-group colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Additional mark-style overrides forwarded to constituent layers.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the boxen plot.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*20 + ["b"]*20,
        ...                    "val": list(range(20)) + list(range(10, 30))})
        >>> fm.Chart(df).mark_boxen().encode(x="group", y="val")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.composite import desugar_boxen

        effective_horizontal = _normalize_orient(orient, horizontal)

        return self._set_composite_mark(
            "boxen",
            desugar_boxen,
            {
                "k_depth": k_depth,
                "k_proportion": k_proportion,
                "outlier_threshold": outlier_threshold,
                "palette": palette,
                "horizontal": effective_horizontal,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_errorbar(
        self, *, method=_MISSING, ticks=True, position=None, **mark_kwargs
    ) -> "Chart":
        """Render error bars via the ``ErrorExtent`` transform.

        Computes extent values (CI, SD, SEM, or IQR) per group defined by the
        ``x`` encoding, then draws a vertical rule spanning the extent with
        optional tick caps.

        Parameters
        ----------
        method : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Aggregation method: confidence interval (``"ci"``), standard error
            (``"stderr"``), standard deviation (``"stdev"``), or
            interquartile range (``"iqr"``).  Forwarded verbatim as
            ``ErrorExtent.method`` (wire field unchanged).

            .. deprecated::
                The old spelling ``extent=`` is accepted as an alias for one
                release.  Use ``method=`` instead.  ``extent`` is reserved for
                data-domain ``[min, max]`` bounds.
        ticks : bool, optional
            Whether to draw horizontal tick caps at the extent endpoints.
            Default is ``True``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to constituent rule/tick layers.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the error bars.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*10 + ["b"]*10,
        ...                    "val": list(range(10)) + list(range(5, 15))})
        >>> fm.Chart(df).mark_errorbar(method="stdev").encode(x="group", y="val")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.composite import desugar_errorbar, _resolve_deprecated_extent

        # Resolve the deprecated extent= alias at the public API level using the
        # shared resolver from composite.py.  Using the _MISSING sentinel for the
        # method default means method="ci" (the canonical default) passed explicitly
        # is correctly detected as "canonically supplied" and triggers TypeError
        # when extent= is also given.  mark_kwargs is mutated in-place by the helper.
        method = _resolve_deprecated_extent("method", method, mark_kwargs, real_default="ci")

        return self._set_composite_mark(
            "errorbar",
            desugar_errorbar,
            {"method": method, "ticks": ticks, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_errorband(
        self, *, method=_MISSING, borders=False, position=None, **mark_kwargs
    ) -> "Chart":
        """Render an error band (ribbon) via the ``ErrorExtent`` transform.

        Similar to ``mark_errorbar`` but renders the extent as a filled ribbon
        rather than whisker rules.  Optionally draws border lines at the upper
        and lower extent edges.

        Parameters
        ----------
        method : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Aggregation method: confidence interval (``"ci"``), standard error
            (``"stderr"``), standard deviation (``"stdev"``), or
            interquartile range (``"iqr"``).  Forwarded verbatim as
            ``ErrorExtent.method`` (wire field unchanged).

            .. deprecated::
                The old spelling ``extent=`` is accepted as an alias for one
                release.  Use ``method=`` instead.  ``extent`` is reserved for
                data-domain ``[min, max]`` bounds.
        borders : bool, optional
            Whether to draw line borders at the band edges.  Default is ``False``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides (e.g. ``opacity``) forwarded to the ribbon layer.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the error band.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": list(range(10)), "y": [float(i) for i in range(10)]})
        >>> fm.Chart(df).mark_errorband(method="ci", borders=True).encode(x="x", y="y")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.composite import desugar_errorband, _resolve_deprecated_extent

        # Resolve the deprecated extent= alias at the public API level using the
        # shared resolver from composite.py.  Using the _MISSING sentinel for the
        # method default means method="ci" (the canonical default) passed explicitly
        # is correctly detected as "canonically supplied" and triggers TypeError
        # when extent= is also given.  mark_kwargs is mutated in-place by the helper.
        method = _resolve_deprecated_extent("method", method, mark_kwargs, real_default="ci")

        return self._set_composite_mark(
            "errorband",
            desugar_errorband,
            {"method": method, "borders": borders, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_ribbon(
        self, *, opacity=0.3, interpolate="linear", position=None, **mark_kwargs
    ) -> "Chart":
        """Render a ribbon (filled area between ``y`` and ``y2`` along ``x``).

        Requires both ``y`` and ``y2`` encoding channels.  Typically used to
        visualise confidence intervals alongside a ``mark_line`` in the same
        layered chart.

        Parameters
        ----------
        opacity : float, optional
            Fill opacity of the ribbon.  Default is ``0.3``.
        interpolate : str, optional
            Boundary interpolation: ``"linear"``, ``"monotone"``, ``"step"``,
            ``"basis"``, ``"cardinal"``.  Default is ``"linear"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Additional mark-style overrides (e.g. ``fill``, ``stroke``).

        Returns
        -------
        Chart
            New ``Chart`` configured for ribbon rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y_lo": [0.5, 1.5, 2.5], "y_hi": [1.5, 2.5, 3.5]})
        >>> fm.Chart(df).mark_ribbon().encode(x="x", y="y_lo", y2="y_hi")
        Chart(mark='ribbon', encoding=['x', 'y', 'y2'])
        """
        from ferrum.marks.composite import desugar_ribbon

        return self._set_composite_mark(
            "ribbon",
            desugar_ribbon,
            {"opacity": opacity, "interpolate": interpolate, **mark_kwargs},
            placeholder="ribbon",
            position=position,
        )

    def mark_contour(
        self,
        *,
        bandwidth="scott",
        thresholds=6,
        smooth=True,
        fill=False,
        scheme=None,
        cmap=None,
        groupby=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a bivariate contour plot via ``Kde2D`` + ``Contour`` transforms.

        Estimates a 2-D kernel density and draws iso-contour lines (or filled
        contour regions when ``fill=True``).  Requires both ``x`` and ``y``
        encodings.

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            Bandwidth for the 2-D KDE.  Default is ``"scott"``.
        thresholds : int or list of float, optional
            Number of evenly-spaced contour levels, or an explicit list of
            density thresholds.  Default is ``6``.
        smooth : bool, optional
            Whether to smooth contour paths via Gaussian filtering before
            contouring.  Default is ``True``.
        fill : bool, optional
            Render filled contour regions instead of contour lines.  Default is
            ``False``.  When ``groupby`` is set, this parameter is ignored and
            isoline (segment) mode is always used.
        scheme : str or None, optional
            Colour map name applied to contour levels.  ``None`` (default) defers
            to the theme's sequential scheme.  Ignored when ``groupby`` is set.
        cmap : str or None, optional
            Back-compat alias for ``scheme``.  Pass at most one of the two.
        groupby : str or None, optional
            Group column (Utf8). When set, one 2-D KDE surface is computed per
            group and contour lines for each group are colored by the group field.
            Use ``.encode(color=<groupby>)`` to match legend colors to contours.
            Grouped contours use isoline (segment) mode to avoid cross-group
            polygon merging issues.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to the polygon/path layers.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for contour rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.0, 1.5], "y": [4.0, 5.0, 4.5, 3.0, 4.0]})
        >>> fm.Chart(df).mark_contour(thresholds=4, fill=True).encode(x="x", y="y")
        Chart(mark='polygon', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_contour
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        return self._set_composite_mark(
            "contour",
            desugar_contour,
            {
                "bandwidth": bandwidth,
                "thresholds": thresholds,
                "smooth": smooth,
                "fill": fill,
                "cmap": resolve_cmap_alias(scheme=scheme, cmap=cmap, where="mark_contour"),
                "groupby": groupby,
                **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_violin(
        self,
        *,
        bandwidth="scott",
        inner="box",
        shared_extent=False,
        orient=None,
        horizontal=False,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render violin plots via the ``Violin`` transform.

        Estimates a mirrored KDE per group and overlays an optional inner
        summary (box, quartile lines, or individual points).  Requires a
        categorical ``x`` and continuous ``y`` encoding (or swapped when
        ``orient="horizontal"`` or ``horizontal=True``).

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            KDE bandwidth.  Default is ``"scott"``.
        inner : {"box", "quartile", "point", "none"}, default "box"
            Inner mark drawn on top of each violin:
            ``"box"`` -- IQR box + median rule,
            ``"quartile"`` -- three horizontal quartile rules,
            ``"point"`` -- individual data points (strip),
            ``"none"`` -- no inner mark.
        shared_extent : bool, default False
            When ``True``, all groups share the same KDE evaluation range
            (cross-group global min/max), making the value axis directly
            comparable across groups.  When ``False`` (default), each group
            evaluates its KDE on its own per-group data range.
        orient : {"vertical", "horizontal"}, default "vertical"
            Canonical orientation control.  ``"horizontal"`` swaps the axes so
            the categorical grouping is on ``y`` and the value distribution is
            on ``x``.  Equivalent to ``horizontal=True``.  When both are given,
            ``orient`` wins.
        horizontal : bool, default False
            Legacy alias for ``orient="horizontal"``.  Kept for backward
            compatibility.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to the violin polygon layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for violin rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*10 + ["b"]*10,
        ...                    "val": list(range(10)) + list(range(5, 15))})
        >>> fm.Chart(df).mark_violin(inner="quartile").encode(x="group", y="val")
        Chart(mark='polygon', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_violin

        effective_horizontal = _normalize_orient(orient, horizontal)

        return self._set_composite_mark(
            "violin",
            desugar_violin,
            {
                "bandwidth": bandwidth,
                "inner": inner,
                "shared_extent": shared_extent,
                "horizontal": effective_horizontal,
                **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_qq(
        self, *, distribution="normal", dequantize=False, line=True, position=None, **mark_kwargs
    ) -> "Chart":
        """Render a quantile-quantile plot.

        Computes theoretical vs. sample quantiles via the ``QQ`` transform.
        Reads the sample column from the ``x`` encoding; ``y`` is ignored.

        Parameters
        ----------
        distribution : {"normal", "uniform", "exponential", "lognormal"}, default "normal"
            Theoretical distribution to compare the sample against.
        dequantize : bool, optional
            Whether to apply rank-based dequantization before comparison.
            Default is ``False``.
        line : bool, optional
            Whether to overlay a 45-degree reference line.  Default is ``True``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the scatter layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for QQ rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"val": [1.2, 0.8, 1.5, -0.5, 0.3, 2.1, -1.0, 0.6]})
        >>> fm.Chart(df).mark_qq(distribution="normal").encode(x="val")
        Chart(mark='point', encoding=['x'])
        """
        from ferrum.marks.heavy_stat import desugar_qq

        def _resolve_qq(x_field, y_field, **kw):
            # QQ is single-column: use x_field as the sample field. y_field ignored.
            if x_field is None:
                raise ValueError("mark_qq() requires .encode(x=...) to specify the sample field")
            return desugar_qq(x_field, **kw)

        # Expose desugar_qq's signature for the transform/style kwarg split in
        # Chart._resolve_pending (inspect.signature follows __wrapped__).
        _resolve_qq.__wrapped__ = desugar_qq

        return self._set_composite_mark(
            "qq",
            _resolve_qq,
            {"distribution": distribution, "dequantize": dequantize, "line": line, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_raster(
        self,
        *,
        aggregate="count",
        field=None,
        scheme=None,
        cmap=None,
        resolution="screen",
        blend="alpha",
        min_count=None,
        log_scale=False,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a 2-D raster (pixel heatmap) via the ``Raster`` transform.

        Bins data into a pixel grid and colours each pixel by an aggregate
        statistic.  Most useful for very large datasets where a scatter plot
        would overplot.  Requires ``x`` and ``y`` encodings.

        Parameters
        ----------
        aggregate : str, optional
            Aggregate function applied to each pixel bin: ``"count"`` (default),
            ``"sum"``, ``"mean"``, ``"min"``, ``"max"``.  When ``aggregate`` is
            not ``"count"``, ``field`` must be provided.
        field : str or None, optional
            Column name to aggregate.  Required unless ``aggregate="count"``.
        scheme : str or None, optional
            Colour map name.  ``None`` (default) defers to the theme's sequential
            scheme.
        cmap : str or None, optional
            Back-compat alias for ``scheme``.  Pass at most one of the two.
        resolution : "screen" or int, optional
            Pixel grid resolution.  ``"screen"`` (default) matches the rendered
            chart dimensions; pass an integer to set an explicit grid width.
        blend : str, optional
            Alpha-compositing blend mode.  Default is ``"alpha"``.
        min_count : int or None, optional
            Minimum count threshold; pixels below this are rendered transparent.
            ``None`` (default) shows all pixels.
        log_scale : bool, optional
            Apply a log transform to pixel values before colour mapping.
            Default is ``False``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to the image layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for raster rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.5], "y": [4.0, 5.0, 4.5, 4.0]})
        >>> fm.Chart(df).mark_raster(cmap="plasma").encode(x="x", y="y")
        Chart(mark='image', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_raster
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        return self._set_composite_mark(
            "raster",
            desugar_raster,
            {
                "aggregate": aggregate,
                "field": field,
                "cmap": resolve_cmap_alias(scheme=scheme, cmap=cmap, where="mark_raster"),
                "resolution": resolution,
                "blend": blend,
                "min_count": min_count,
                "log_scale": log_scale,
                **mark_kwargs,
            },
            placeholder="image",
            position=position,
        )

    def mark_hex(
        self,
        *,
        bin_size=None,
        aggregate="count",
        field=None,
        scheme=None,
        cmap=None,
        stroke=None,
        stroke_width=0,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a hexagonal bin plot via the ``Hex`` transform.

        Bins data into a regular hexagonal grid and colours each cell by an
        aggregate statistic.  Requires ``x`` and ``y`` encodings.

        Parameters
        ----------
        bin_size : float or None, optional
            Hexagon radius in data units.  ``None`` (default) chooses a size
            automatically from the data range and chart dimensions.
        aggregate : str, optional
            Aggregate function: ``"count"`` (default), ``"sum"``, ``"mean"``,
            ``"min"``, ``"max"``.
        field : str or None, optional
            Column to aggregate.  Required unless ``aggregate="count"``.
        scheme : str or None, optional
            Colour map name.  ``None`` (default) defers to the theme's sequential
            scheme.
        cmap : str or None, optional
            Back-compat alias for ``scheme``.  Pass at most one of the two.
        stroke : str or None, optional
            Hex border colour.  ``None`` (default) means no border.
        stroke_width : float, optional
            Hex border width in pixels.  Default is ``0``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides forwarded to the polygon layer.
            Note: ``stroke`` and ``stroke_width`` are method parameters (cell borders);
            ``scheme`` (alias ``cmap``) is a transform parameter (fill colormap). Other
            style kwargs like ``opacity``, ``fill``, ``size`` override the polygon defaults.

        Returns
        -------
        Chart
            New ``Chart`` configured for hexagonal binning.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 1.5, 3.0, 2.5], "y": [4.0, 5.0, 4.5, 3.0, 4.0]})
        >>> fm.Chart(df).mark_hex(bin_size=0.5).encode(x="x", y="y")
        Chart(mark='polygon', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_hex
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        return self._set_composite_mark(
            "hex",
            desugar_hex,
            {
                "bin_size": bin_size,
                "aggregate": aggregate,
                "field": field,
                "cmap": resolve_cmap_alias(scheme=scheme, cmap=cmap, where="mark_hex"),
                "stroke": stroke,
                "stroke_width": stroke_width,
                **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_swarm(
        self,
        *,
        size=4,
        orient=None,
        horizontal=None,
        spacing=1.0,
        side="both",
        dodge=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a beeswarm (strip swarm) plot via the ``Swarm`` transform.

        Computes non-overlapping point positions along the categorical axis
        using a deterministic placement algorithm seeded for reproducibility.
        Requires a continuous ``y`` (or ``x``) and an optional categorical
        grouping axis.

        Parameters
        ----------
        size : float, optional
            Point diameter in pixels.  Default is ``4``.
        orient : {"vertical", "horizontal"} or None, optional
            Canonical orientation control.  ``"vertical"`` (default) spreads
            points along x; ``"horizontal"`` spreads along y.  When both
            ``orient`` and ``horizontal`` are given, ``orient`` wins.
        horizontal : bool or None, optional
            Legacy alias for ``orient``.  ``True`` is equivalent to
            ``orient="horizontal"``.  Ignored when ``orient`` is also given;
            prefer ``orient`` in new code.
        spacing : float, optional
            Minimum spacing between point centres as a fraction of ``size``.
            Default is ``1.0``.
        side : {"both", "left", "right"}, default "both"
            Side of the axis on which points can be placed.
        dodge : str or None, optional
            Column name to use as a secondary grouping variable for dodging.
            ``None`` (default) means no dodging.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to the point layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for beeswarm rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*8 + ["b"]*8, "val": list(range(16))})
        >>> fm.Chart(df).mark_swarm(size=6).encode(x="group", y="val")
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_swarm

        # Normalize: orient= is canonical; horizontal= is a legacy alias.
        # When orient is not None it wins unconditionally (consistent with
        # boxplot/boxen/violin which call _normalize_orient).  When orient is
        # None, fall back to horizontal (or default to "vertical").
        effective_horizontal = _normalize_orient(orient, horizontal or False)
        effective_orient = "horizontal" if effective_horizontal else "vertical"

        return self._set_composite_mark(
            "swarm",
            desugar_swarm,
            {
                "size": size,
                "orient": effective_orient,
                "spacing": spacing,
                "side": side,
                "dodge": dodge,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_function(
        self, fn, *, domain=None, n=200, clip=True, position=None, **mark_kwargs
    ) -> "Chart":
        """Render a mathematical function as a line.

        Evaluates ``fn(xs)`` on ``n`` evenly-spaced ``x`` values in the
        specified domain and renders the result as a line.  The input data on
        the chart is replaced with the synthetic dataset; use ``+`` composition
        to overlay a function on a scatter chart.

        Parameters
        ----------
        fn : callable
            A function accepting a 1-D NumPy array of x values and returning a
            1-D array of y values.
        domain : list of float or None, optional
            ``[x_min, x_max]`` evaluation range.  ``None`` (default) infers
            the range from the parent chart's ``x`` column if available,
            otherwise raises ``ValueError``.
        n : int, optional
            Number of evaluation points.  Default is ``200``.
        clip : bool, optional
            Whether to clip rendered line to the plot viewport.  Default is
            ``True``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to the line layer (e.g. ``stroke``,
            ``stroke_width``).

        Returns
        -------
        Chart
            New ``Chart`` whose data is the synthetic ``(x, y)`` table.

        Raises
        ------
        ValueError
            When ``domain`` is not provided and no parent ``x`` data can be
            inferred.

        Examples
        --------
        >>> import ferrum as fm
        >>> import numpy as np
        >>> import polars as pl
        >>> fm.Chart(None).mark_function(np.sin, domain=[0, 6.28], n=100)
        Chart(mark='line', encoding=['x', 'y'])
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility

            validate_position_eligibility("function", position)
        from ferrum.marks.heavy_stat import desugar_function

        # Infer parent x data for domain resolution.
        # For multi-layer charts, try the first layer's data; for single-chart,
        # use self._data.
        parent_x_data = None
        x_enc = self._encoding.get("x")
        data_source = self._data
        if data_source is None and self._layers:
            # Extract data from the first non-function layer.
            for existing_layer in self._layers:
                if hasattr(existing_layer, "_data") and existing_layer._data is not None:  # type: ignore[attr-defined]
                    data_source = existing_layer._data  # type: ignore[attr-defined]
                    break
        if x_enc is not None and data_source is not None:
            try:
                from ferrum._coerce import to_arrow_table

                x_field_name = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
                tbl = to_arrow_table(data_source)
                if x_field_name in tbl.column_names:
                    parent_x_data = tbl[x_field_name]
            except (ImportError, TypeError, ValueError, KeyError):
                pass
        elif x_enc is None and data_source is not None and domain is None:
            # Encoding not set yet (mark_function called before .encode()).
            # Infer domain from the first numeric column in the chart's data.
            try:
                import pyarrow as pa
                from ferrum._coerce import to_arrow_table

                tbl = to_arrow_table(data_source)
                for _col_name in tbl.column_names:
                    _col = tbl[_col_name]
                    if pa.types.is_floating(_col.type) or pa.types.is_integer(_col.type):
                        parent_x_data = _col
                        break
            except (ImportError, TypeError, ValueError, KeyError):
                pass

        result = desugar_function(
            fn,
            parent_chart_x_data=parent_x_data,
            domain=domain,
            n=n,
            clip=clip,
            **mark_kwargs,
        )
        mark = result.mark
        transforms = result.transforms
        remap = result.remap
        synthetic = result.data
        if self._layers is not None and self._layers:
            # Multi-layer: build a fresh single-chart with the function data and
            # compose via + so it becomes a proper layer alongside existing layers.
            fn_chart = self.__class__(synthetic)
            fn_chart._mark = mark
            if remap:
                from ferrum._desugar import _apply_remap

                _apply_remap(fn_chart._encoding, remap, orig_encoding=self._encoding)
            fn_chart._position = position
            return self + fn_chart

        new = self._clone()
        new._mark = mark
        new._data = synthetic
        new._transforms = list(self._transforms) + list(transforms)
        if remap:
            from ferrum._desugar import _apply_remap

            _apply_remap(new._encoding, remap, orig_encoding=self._encoding)
        new._position = position
        return new
