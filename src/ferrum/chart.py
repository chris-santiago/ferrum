"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from ferrum._coerce import to_arrow_table
from ferrum._shorthand import parse_shorthand
from ferrum.encoding.base import ChannelBase
from ferrum.marks.base import MarkBase
from ferrum.marks.deferred import deferred_mark_error, PHASE_8B_MARKS, PHASE_9_PLUS_MARKS
from ferrum.marks.statistical import desugar_density, desugar_histogram, desugar_smooth


_PRIMITIVE_MARKS = frozenset(["point", "line", "bar", "area", "rule", "text", "tick", "rect"])

_CHANNEL_CLASSES_BY_NAME: dict = {}


def _channel_class_for(name: str):
    """Return the channel-class for a given parameter name (lazy import to avoid cycles)."""
    if not _CHANNEL_CLASSES_BY_NAME:
        from ferrum.encoding import (
            X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
            Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
            StrokeWidth, StrokeDash, Size, Shape, Angle,
            Text, Detail, Tooltip, TooltipField, Href, Description, Key,
            Facet, FacetRow, FacetCol,
        )
        _CHANNEL_CLASSES_BY_NAME.update({
            "x": X, "y": Y, "x2": X2, "y2": Y2,
            "x_error": XError, "y_error": YError, "x_error2": XError2, "y_error2": YError2,
            "theta": Theta, "radius": Radius,
            "color": Color, "fill": Fill, "stroke": Stroke,
            "opacity": Opacity, "fill_opacity": FillOpacity, "stroke_opacity": StrokeOpacity,
            "stroke_width": StrokeWidth, "stroke_dash": StrokeDash,
            "size": Size, "shape": Shape, "angle": Angle,
            "text": Text, "detail": Detail, "tooltip": Tooltip, "tooltip_field": TooltipField,
            "href": Href, "description": Description, "key": Key,
            "facet": Facet, "facet_row": FacetRow, "facet_col": FacetCol,
        })
    return _CHANNEL_CLASSES_BY_NAME.get(name)


class Chart:
    """Top-level chart value class.

    Immutable — every method returns a new ``Chart``. Pass data once at
    construction; declare encodings, marks, and transforms with chained
    methods; render with ``show_svg`` / ``show_png`` or display in a
    Jupyter cell.

    Parameters
    ----------
    data : polars.DataFrame, pyarrow.Table, pandas.DataFrame, or any object \
            implementing ``__arrow_c_stream__``, optional
        Input data. Polars and pyarrow flow through Arrow C Data Interface
        with zero copies; other DataFrame types use ``narwhals``.
    width : int or str, optional
        Chart width in pixels (or CSS string). Defaults to 600 at render time.
    height : int or str, optional
        Chart height in pixels (or CSS string). Defaults to 400 at render time.
    title : str, optional
        Chart title rendered above the axes.
    description : str, optional
        Accessible description for screen readers; not rendered visually.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).encode(x="hp", y="mpg").mark_point()
    >>> svg = chart.show_svg()
    """

    __slots__ = (
        "_data", "_mark", "_mark_kwargs", "_encoding", "_transforms",
        "_facet", "_coord", "_theme", "_layers",
        "_width", "_height", "_title", "_description",
        "_pending_stat_mark",  # (kind, kwargs) when mark_* called before .encode()
        "_position",           # Phase 9c — Identity / Dodge / Jitter / Stack (or None)
    )

    def __init__(
        self,
        data: Any = None,
        *,
        width: Optional[Union[int, str]] = None,
        height: Optional[Union[int, str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        self._data = data
        self._mark = None
        self._mark_kwargs = {}
        self._encoding: dict = {}
        self._transforms: list = []
        self._facet = None
        self._coord = None
        self._theme = None
        self._layers: Optional[list] = None
        self._width = width
        self._height = height
        self._title = title
        self._description = description
        self._pending_stat_mark: Optional[tuple] = None  # (kind, kwargs)
        self._position = None

    def _clone(self) -> "Chart":
        new = object.__new__(Chart)
        new._data = self._data
        new._mark = self._mark
        new._mark_kwargs = dict(self._mark_kwargs)
        new._encoding = dict(self._encoding)
        new._transforms = list(self._transforms)
        new._facet = self._facet
        new._coord = self._coord
        new._theme = self._theme
        new._layers = None if self._layers is None else list(self._layers)
        new._width = self._width
        new._height = self._height
        new._title = self._title
        new._description = self._description
        new._pending_stat_mark = self._pending_stat_mark
        new._position = self._position
        return new

    def _resolve_pending(self) -> "Chart":
        """Resolve a pending statistical mark desugar once encoding is known.

        Calling ``mark_density/histogram/smooth`` before ``.encode()`` stores a
        ``_pending_stat_mark`` sentinel.  ``_resolve_pending`` is called at the
        start of every render/spec-build path to apply the desugar against the
        now-populated encoding dict.

        Two sentinel shapes are supported:
        - 2-tuple ``(kind, kwargs)`` — legacy form for density/histogram/smooth,
          dispatched by ``kind`` below.
        - 3-tuple ``(kind, kwargs, desugar_fn)`` — generic form used by composite
          marks (Phase 8b Tasks 23-33). ``desugar_fn(x_field, y_field, **kwargs)``
          may return a 5-tuple ``("__layered__", transforms, _, _, layers)`` to
          emit a multi-layer ChartSpec, or the legacy 3-tuple
          ``(mark, transforms, remap)`` for a single-mark desugar.
        """
        if self._pending_stat_mark is None:
            return self
        # 3-tuple form: generic desugar callable.
        if len(self._pending_stat_mark) == 3:
            kind, kwargs, desugar_fn = self._pending_stat_mark
            x_enc = self._encoding.get("x")
            y_enc = self._encoding.get("y")
            x_field = (
                (x_enc.field if isinstance(x_enc, ChannelBase) else x_enc)
                if x_enc is not None
                else None
            )
            y_field = (
                (y_enc.field if isinstance(y_enc, ChannelBase) else y_enc)
                if y_enc is not None
                else None
            )
            # Ribbon needs y2 from the encoding — inject as a kwarg.
            if kind == "ribbon":
                y2_enc = self._encoding.get("y2")
                if y2_enc is not None:
                    y2_field = (
                        y2_enc.field if isinstance(y2_enc, ChannelBase) else y2_enc
                    )
                    kwargs = {**kwargs, "y2_field": y2_field}
            result = desugar_fn(x_field, y_field, **kwargs)
            new = self._clone()
            new._pending_stat_mark = None
            if (
                isinstance(result, tuple)
                and len(result) >= 1
                and result[0] == "__layered__"
            ):
                # ("__layered__", transforms, _, _, layers)
                _, transforms, _ignored1, _ignored2, layers_list = result
                new._transforms = list(self._transforms) + list(transforms or [])
                new._layers = list(layers_list)
                new._mark = None  # signals layered mode in to_spec
                return new
            # Legacy single-mark 3-tuple: (mark, transforms, remap)
            mark, transforms, remap = result
            new._mark = mark
            new._transforms = list(self._transforms) + list(transforms or [])
            if remap:
                from ferrum.encoding import X, X2, Y, Y2  # noqa: F401
                if "x" in remap:
                    new._encoding["x"] = X(remap["x"], type="Q")
                if "y" in remap:
                    new._encoding["y"] = Y(remap["y"], type="Q")
                if "x2" in remap:
                    new._encoding["x2"] = X2(remap["x2"], type="Q")
            return new
        # 2-tuple legacy form.
        kind, kwargs = self._pending_stat_mark
        new = self._clone()
        new._pending_stat_mark = None
        if kind == "density":
            x_enc = new._encoding.get("x")
            if x_enc is None:
                raise ValueError("mark_density() requires .encode(x=...) to specify the density field")
            field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
            result = desugar_density(field, chart_encoding=new._encoding, **kwargs)
            if (
                isinstance(result, tuple)
                and len(result) >= 1
                and result[0] == "__layered__"
            ):
                # Bivariate density routed through desugar_contour(fill=True).
                _, transforms, _ignored1, _ignored2, layers_list = result
                new._transforms = list(new._transforms) + list(transforms or [])
                new._layers = list(layers_list)
                new._mark = None  # signals layered mode
            else:
                mark, transforms, remap = result
                new._mark = mark
                new._transforms = list(new._transforms) + transforms
                from ferrum.encoding import X, Y
                new._encoding["x"] = X(remap["x"], type="Q")
                new._encoding["y"] = Y(remap["y"], type="Q")
        elif kind == "histogram":
            x_enc = new._encoding.get("x")
            if x_enc is None:
                raise ValueError("mark_histogram() requires .encode(x=...) to specify the histogram field")
            field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
            mark, transforms, remap = desugar_histogram(field, **kwargs)
            new._mark = mark
            new._transforms = list(new._transforms) + transforms
            from ferrum.encoding import X, X2, Y
            new._encoding["x"] = X(remap["x"], type="Q")
            new._encoding["x2"] = X2(remap["x2"], type="Q")
            new._encoding["y"] = Y(remap["y"], type="Q")
        elif kind == "smooth":
            x_enc = new._encoding.get("x")
            y_enc = new._encoding.get("y")
            if x_enc is None or y_enc is None:
                raise ValueError("mark_smooth() requires .encode(x=..., y=...)")
            x_field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
            y_field = y_enc.field if isinstance(y_enc, ChannelBase) else y_enc
            result = desugar_smooth(x_field, y_field, **kwargs)
            if (
                isinstance(result, tuple)
                and len(result) >= 1
                and result[0] == "__layered__"
            ):
                # ("__layered__", transforms, _, _, layers) — 8b ci-band path
                _, transforms, _ignored1, _ignored2, layers_list = result
                new._transforms = list(new._transforms) + list(transforms or [])
                new._layers = list(layers_list)
                new._mark = None  # signals layered mode
            else:
                mark, transforms, remap = result
                new._mark = mark
                new._transforms = list(new._transforms) + transforms
                # Smooth's output schema uses literal "x"/"y" columns; apply the
                # remap so the encoding references the post-transform schema.
                if remap:
                    from ferrum.encoding import X, Y
                    if "x" in remap:
                        new._encoding["x"] = X(remap["x"], type="Q")
                    if "y" in remap:
                        new._encoding["y"] = Y(remap["y"], type="Q")
        return new

    # ---- Marks (primitives) ----

    def _set_mark(self, name: str, **kwargs: Any) -> "Chart":
        # Phase 9c — pull `position=` out of kwargs and validate eligibility
        # before constructing the MarkBase (which would reject unknown kwargs).
        position = kwargs.pop("position", None)
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility(name, position)
        m = MarkBase(name, **kwargs)
        new = self._clone()
        new._mark = name
        new._mark_kwargs = m.to_mark_kwargs_dict()
        new._position = position
        return new

    def mark_point(self, **kwargs) -> "Chart":
        """Render data as a scatter plot with point marks.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``size``, ``opacity``, ``position``).

        Returns
        -------
        Chart
            New chart with the point mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point()
        """
        return self._set_mark("point", **kwargs)

    def mark_line(self, **kwargs) -> "Chart":
        """Render data as connected line segments.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``stroke_width``, ``interpolate``, ``position``).

        Returns
        -------
        Chart
            New chart with the line mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="year", y="value").mark_line()
        """
        return self._set_mark("line", **kwargs)

    def mark_bar(self, **kwargs) -> "Chart":
        """Render data as rectangular bars.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``opacity``, ``position``).

        Returns
        -------
        Chart
            New chart with the bar mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="category", y="count").mark_bar()
        """
        return self._set_mark("bar", **kwargs)

    def mark_area(self, **kwargs) -> "Chart":
        """Render data as a filled area between y and the baseline.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``opacity``, ``interpolate``, ``position``).

        Returns
        -------
        Chart
            New chart with the area mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="year", y="value").mark_area()
        """
        return self._set_mark("area", **kwargs)

    def mark_rule(self, **kwargs) -> "Chart":
        """Render data as axis-aligned horizontal or vertical rule lines.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``stroke_width``, ``position``).

        Returns
        -------
        Chart
            New chart with the rule mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(y="threshold").mark_rule()
        """
        return self._set_mark("rule", **kwargs)

    def mark_text(self, **kwargs) -> "Chart":
        """Render data as text labels at each data point.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``size``, ``align``, ``position``).

        Returns
        -------
        Chart
            New chart with the text mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x", y="y", text="label").mark_text()
        """
        return self._set_mark("text", **kwargs)

    def mark_tick(self, **kwargs) -> "Chart":
        """Render data as short tick marks along an axis.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``size``, ``thickness``, ``position``).

        Returns
        -------
        Chart
            New chart with the tick mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="value", y="category").mark_tick()
        """
        return self._set_mark("tick", **kwargs)

    def mark_rect(self, **kwargs) -> "Chart":
        """Render data as filled rectangles; requires x, x2, y, y2 encodings.

        Parameters
        ----------
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``opacity``, ``position``).

        Returns
        -------
        Chart
            New chart with the rect mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x1", x2="x2", y="y1", y2="y2").mark_rect()
        """
        return self._set_mark("rect", **kwargs)

    # ---- Marks (statistical) ----

    def mark_density(self, *, position=None, **kwargs) -> "Chart":
        """Render a kernel density estimate (KDE) as a smoothed area curve.

        1D KDE when only ``x`` is encoded; bivariate filled-contour over a 2D
        KDE when both ``x`` and ``y`` are encoded. May be called before or
        after ``.encode()``.

        Parameters
        ----------
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to each mark.
        **kwargs
            Additional options forwarded to the underlying KDE transform
            (e.g. ``bandwidth``, ``n``).

        Returns
        -------
        Chart
            New chart with the density mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp").mark_density()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("density", position)
        x_enc = self._encoding.get("x")
        if x_enc is None:
            # Encoding not yet set — defer resolution to render time.
            new = self._clone()
            new._mark = "area"  # placeholder so _mark is not None
            new._pending_stat_mark = ("density", dict(kwargs))
            new._position = position
            return new
        field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
        result = desugar_density(field, chart_encoding=self._encoding, **kwargs)
        new = self._clone()
        if (
            isinstance(result, tuple)
            and len(result) >= 1
            and result[0] == "__layered__"
        ):
            # Bivariate density routed through desugar_contour(fill=True).
            _, transforms, _ignored1, _ignored2, layers_list = result
            new._transforms = list(self._transforms) + list(transforms or [])
            new._layers = list(layers_list)
            new._mark = None  # signals layered mode
        else:
            mark, transforms, remap = result
            new._mark = mark
            new._transforms = list(self._transforms) + transforms
            from ferrum.encoding import X, Y
            new._encoding["x"] = X(remap["x"], type="Q")
            new._encoding["y"] = Y(remap["y"], type="Q")
        new._position = position
        return new

    def mark_histogram(self, *, position=None, **kwargs) -> "Chart":
        """Render data as a histogram of binned counts or densities.

        May be called before or after ``.encode(x=...)``.

        Parameters
        ----------
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to each bar.
        **kwargs
            Additional options forwarded to the underlying ``Bin`` transform
            (e.g. ``bin_count``, ``bin_width``, ``density``, ``cumulative``).

        Returns
        -------
        Chart
            New chart with the histogram mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp").mark_histogram(bin_count=20)
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("histogram", position)
        x_enc = self._encoding.get("x")
        if x_enc is None:
            new = self._clone()
            new._mark = "bar"  # placeholder
            new._pending_stat_mark = ("histogram", dict(kwargs))
            new._position = position
            return new
        field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
        mark, transforms, remap = desugar_histogram(field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        from ferrum.encoding import X, X2, Y
        new._encoding["x"] = X(remap["x"], type="Q")
        new._encoding["x2"] = X2(remap["x2"], type="Q")
        new._encoding["y"] = Y(remap["y"], type="Q")
        new._position = position
        return new

    def mark_smooth(self, *, position=None, **kwargs) -> "Chart":
        """Render a smoothed regression line over x/y data.

        May be called before or after ``.encode(x=..., y=...)``. When ``ci``
        is set, emits a layered ribbon (confidence-interval band) plus line.

        Parameters
        ----------
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to each mark.
        **kwargs
            Options forwarded to the ``Smooth`` transform. Key options:
            ``method`` (``"loess"`` or ``"lm"``), ``ci`` (float or ``None``),
            ``bandwidth`` (float, LOESS only), ``seed`` (int).

        Returns
        -------
        Chart
            New chart with the smooth mark set; may be layered when ``ci`` is
            enabled.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_smooth(method="lm")
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("smooth", position)
        x_enc = self._encoding.get("x")
        y_enc = self._encoding.get("y")
        if x_enc is None or y_enc is None:
            new = self._clone()
            new._mark = "line"  # placeholder
            new._pending_stat_mark = ("smooth", dict(kwargs))
            new._position = position
            return new
        x_field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
        y_field = y_enc.field if isinstance(y_enc, ChannelBase) else y_enc
        result = desugar_smooth(x_field, y_field, **kwargs)
        new = self._clone()
        if (
            isinstance(result, tuple)
            and len(result) >= 1
            and result[0] == "__layered__"
        ):
            _, transforms, _ignored1, _ignored2, layers_list = result
            new._transforms = list(self._transforms) + list(transforms or [])
            new._layers = list(layers_list)
            new._mark = None  # signals layered mode
        else:
            mark, transforms, remap = result
            new._mark = mark
            new._transforms = list(self._transforms) + transforms
            # Smooth's output schema uses literal "x"/"y" columns; apply the
            # remap so the encoding references the post-transform schema.
            if remap:
                from ferrum.encoding import X, Y
                if "x" in remap:
                    new._encoding["x"] = X(remap["x"], type="Q")
                if "y" in remap:
                    new._encoding["y"] = Y(remap["y"], type="Q")
        new._position = position
        return new

    # ---- Marks (deferred) ----

    def mark_boxplot(
        self,
        *,
        extent=1.5,
        size=None,
        outliers=True,
        color_field=None,
        horizontal=False,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a composite boxplot (box + whiskers + median + optional outliers).

        Desugars to a multi-layer chart using the ``BoxStats`` transform.
        Requires ``x`` (categorical group) and ``y`` (numeric value) encodings.

        Parameters
        ----------
        extent : float, default 1.5
            IQR multiplier for whisker length. Points beyond the whiskers are
            drawn as outliers when ``outliers=True``.
        size : float, optional
            Box width in pixels. Defaults to an automatic value.
        outliers : bool, default True
            Whether to render outlier points beyond the whiskers.
        color_field : str, optional
            Column to map to box fill color.
        horizontal : bool, default False
            Rotate the plot so boxes are horizontal (requires ``y`` as the
            group field and ``x`` as the numeric field).
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the box layers.
        **mark_kwargs
            Additional mark-style overrides forwarded to each layer.

        Returns
        -------
        Chart
            New layered chart with the boxplot composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="species", y="petal_length").mark_boxplot()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("boxplot", position)
        from ferrum.marks.composite import desugar_boxplot
        new = self._clone()
        new._mark = "point"  # placeholder; layered mode overrides
        new._pending_stat_mark = (
            "boxplot",
            {
                "extent": extent,
                "size": size,
                "outliers": outliers,
                "color_field": color_field,
                "horizontal": horizontal,
                **mark_kwargs,
            },
            desugar_boxplot,
        )
        new._position = position
        return new

    def mark_boxen(
        self,
        *,
        k_depth: str = "proportion",
        k_proportion: float = 0.007,
        outlier_threshold: float = 1.5,
        palette=None,
        horizontal: bool = False,
        color_field=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a letter-value (boxen) plot.

        Composite mark that desugars to nested rect bands per letter-value
        depth, a median rule, and an outlier-point layer via the
        ``LetterValue`` transform.

        Parameters
        ----------
        k_depth : {"proportion", "tukey", "trustworthy", "full"}, default "proportion"
            Method for determining the number of letter-value levels to show.
        k_proportion : float, default 0.007
            Proportion of the sample used when ``k_depth="proportion"``.
        outlier_threshold : float, default 1.5
            IQR-based threshold beyond which points are treated as outliers.
        palette : list of str, optional
            Custom color palette for the nested rect bands (outermost to
            innermost).
        horizontal : bool, default False
            Rotate so the distribution spreads horizontally.
        color_field : str, optional
            Column to map to box fill color.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the box layers.
        **mark_kwargs
            Additional mark-style overrides forwarded to each layer.

        Returns
        -------
        Chart
            New layered chart with the letter-value composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="species", y="petal_length").mark_boxen()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("boxen", position)
        from ferrum.marks.composite import desugar_boxen
        new = self._clone()
        new._mark = "point"  # placeholder; layered mode overrides
        new._pending_stat_mark = (
            "boxen",
            {
                "k_depth": k_depth,
                "k_proportion": k_proportion,
                "outlier_threshold": outlier_threshold,
                "palette": palette,
                "horizontal": horizontal,
                "color_field": color_field,
                **mark_kwargs,
            },
            desugar_boxen,
        )
        new._position = position
        return new

    def mark_errorbar(self, *, extent="ci", ticks=True, position=None, **mark_kwargs) -> "Chart":
        """Render error bars showing a spread or confidence interval.

        Uses the ``ErrorExtent`` transform to compute interval bounds from the
        encoded ``y`` field.

        Parameters
        ----------
        extent : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Interval type. ``"ci"`` uses a 95 % bootstrap confidence interval;
            ``"stderr"`` ± one standard error; ``"stdev"`` ± one standard
            deviation; ``"iqr"`` inter-quartile range.
        ticks : bool, default True
            Whether to draw horizontal caps (ticks) at the interval endpoints.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to each errorbar.
        **mark_kwargs
            Additional mark-style overrides forwarded to the rule/tick layers.

        Returns
        -------
        Chart
            New layered chart with the errorbar composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="species", y="value").mark_errorbar()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("errorbar", position)
        from ferrum.marks.composite import desugar_errorbar
        new = self._clone()
        new._mark = "point"
        new._pending_stat_mark = (
            "errorbar",
            {"extent": extent, "ticks": ticks, **mark_kwargs},
            desugar_errorbar,
        )
        new._position = position
        return new

    def mark_errorband(self, *, extent="ci", borders=False, position=None, **mark_kwargs) -> "Chart":
        """Render a filled ribbon representing a confidence or spread interval.

        Uses the ``ErrorExtent`` transform to compute interval bounds, then
        renders the band as a ribbon mark.

        Parameters
        ----------
        extent : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Interval type. ``"ci"`` uses a 95 % bootstrap confidence interval;
            ``"stderr"`` ± one standard error; ``"stdev"`` ± one standard
            deviation; ``"iqr"`` inter-quartile range.
        borders : bool, default False
            Whether to draw boundary lines at the top and bottom edges of the
            band.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the band.
        **mark_kwargs
            Additional mark-style overrides forwarded to the ribbon layer.

        Returns
        -------
        Chart
            New layered chart with the errorband composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x", y="value").mark_errorband(extent="stdev")
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("errorband", position)
        from ferrum.marks.composite import desugar_errorband
        new = self._clone()
        new._mark = "point"
        new._pending_stat_mark = (
            "errorband",
            {"extent": extent, "borders": borders, **mark_kwargs},
            desugar_errorband,
        )
        new._position = position
        return new

    def mark_ribbon(self, *, opacity=0.3, interpolate="linear", position=None, **mark_kwargs) -> "Chart":
        """Render a filled ribbon between ``y`` and ``y2`` along ``x``.

        Requires both ``y`` and ``y2`` in the encoding. Useful for showing
        a pre-computed interval band (e.g. min/max or CI bounds stored as
        separate columns).

        Parameters
        ----------
        opacity : float, default 0.3
            Fill opacity of the ribbon area.
        interpolate : str, default "linear"
            Line interpolation method between data points (e.g.
            ``"linear"``, ``"monotone"``, ``"step"``).
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the ribbon.
        **mark_kwargs
            Additional mark-style overrides forwarded to the ribbon layer.

        Returns
        -------
        Chart
            New chart with the ribbon composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x", y="lower", y2="upper").mark_ribbon()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("ribbon", position)
        from ferrum.marks.composite import desugar_ribbon
        new = self._clone()
        new._mark = "ribbon"
        new._pending_stat_mark = (
            "ribbon",
            {"opacity": opacity, "interpolate": interpolate, **mark_kwargs},
            desugar_ribbon,
        )
        new._position = position
        return new

    def mark_contour(
        self,
        *,
        bandwidth="scott",
        thresholds=6,
        smooth=True,
        fill=False,
        cmap="viridis",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a contour plot of the bivariate density.

        Uses the ``Kde2D`` and ``Contour`` transforms. Requires both ``x`` and
        ``y`` encodings.

        Parameters
        ----------
        bandwidth : str or float, default "scott"
            KDE bandwidth selector. ``"scott"`` and ``"silverman"`` invoke
            automatic rules; a float sets the bandwidth directly.
        thresholds : int or list of float, default 6
            Number of contour levels, or explicit threshold values.
        smooth : bool, default True
            Whether to apply smoothing to the iso-lines.
        fill : bool, default False
            If ``True``, render filled contour bands instead of iso-lines.
        cmap : str, default "viridis"
            Color map name for coloring contour levels.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to each contour layer.
        **mark_kwargs
            Additional mark-style overrides forwarded to the polygon layers.

        Returns
        -------
        Chart
            New layered chart with the contour composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x", y="y").mark_contour(thresholds=8)
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("contour", position)
        from ferrum.marks.heavy_stat import desugar_contour
        new = self._clone()
        new._mark = "polygon"  # placeholder; layered mode overrides
        new._pending_stat_mark = (
            "contour",
            {
                "bandwidth": bandwidth, "thresholds": thresholds, "smooth": smooth,
                "fill": fill, "cmap": cmap, **mark_kwargs,
            },
            desugar_contour,
        )
        new._position = position
        return new

    def mark_violin(self, *, bandwidth="scott", inner="box", position=None, **mark_kwargs) -> "Chart":
        """Render a violin plot of the distribution per group.

        Uses the ``Violin`` transform to compute density shapes. Requires
        ``x`` (categorical group) and ``y`` (numeric value) encodings.

        Parameters
        ----------
        bandwidth : str or float, default "scott"
            KDE bandwidth selector. ``"scott"`` and ``"silverman"`` invoke
            automatic rules; a float sets the bandwidth directly.
        inner : {"box", "quartile", "point", None}, default "box"
            Inner overlay drawn inside the violin shape.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the violin layers.
        **mark_kwargs
            Additional mark-style overrides forwarded to the polygon layers.

        Returns
        -------
        Chart
            New layered chart with the violin composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="species", y="sepal_length").mark_violin()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("violin", position)
        from ferrum.marks.heavy_stat import desugar_violin
        new = self._clone()
        new._mark = "polygon"
        new._pending_stat_mark = (
            "violin",
            {"bandwidth": bandwidth, "inner": inner, **mark_kwargs},
            desugar_violin,
        )
        new._position = position
        return new

    def mark_qq(self, *, distribution="normal", dequantize=False, line=True, position=None, **mark_kwargs) -> "Chart":
        """Render a quantile-quantile (QQ) plot against a reference distribution.

        Reads the sample field from the ``x`` encoding. The ``y`` encoding is
        ignored — theoretical quantiles become the y axis.

        Parameters
        ----------
        distribution : {"normal", "uniform", "exponential"}, default "normal"
            Reference distribution to plot against.
        dequantize : bool, default False
            Whether to dequantize tied values with small random jitter before
            computing quantiles.
        line : bool, default True
            Whether to overlay the 45-degree reference line.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the point marks.
        **mark_kwargs
            Additional mark-style overrides forwarded to the point layers.

        Returns
        -------
        Chart
            New layered chart with the QQ composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="residual").mark_qq()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("qq", position)
        from ferrum.marks.heavy_stat import desugar_qq
        new = self._clone()
        new._mark = "point"

        def _resolve_qq(x_field, y_field, **kw):
            # QQ is single-column: use x_field as the sample field. y_field ignored.
            if x_field is None:
                raise ValueError("mark_qq() requires .encode(x=...) to specify the sample field")
            return desugar_qq(x_field, **kw)

        new._pending_stat_mark = (
            "qq",
            {"distribution": distribution, "dequantize": dequantize, "line": line, **mark_kwargs},
            _resolve_qq,
        )
        new._position = position
        return new

    def mark_raster(
        self,
        *,
        aggregate="count",
        field=None,
        cmap="viridis",
        resolution="screen",
        blend="alpha",
        min_count=None,
        log_scale=False,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a 2D density raster image over x/y data.

        Uses the ``Raster`` transform to bin data onto a pixel grid and
        aggregate within each cell. Suitable for large-N scatter data.

        Parameters
        ----------
        aggregate : str, default "count"
            Aggregation function applied to each pixel cell (e.g. ``"count"``,
            ``"mean"``, ``"sum"``).
        field : str, optional
            Column used for aggregation. Required when ``aggregate`` is not
            ``"count"``.
        cmap : str, default "viridis"
            Color map name for mapping aggregated values to colors.
        resolution : str or int, default "screen"
            Pixel grid resolution. ``"screen"`` infers from the chart viewport.
        blend : str, default "alpha"
            Blending mode used when compositing the raster.
        min_count : int, optional
            Minimum cell count below which cells are rendered as transparent.
        log_scale : bool, default False
            Whether to apply a log scale to aggregated values before color
            mapping.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the image mark.
        **mark_kwargs
            Additional mark-style overrides forwarded to the image layer.

        Returns
        -------
        Chart
            New chart with the raster composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x", y="y").mark_raster()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("raster", position)
        from ferrum.marks.heavy_stat import desugar_raster
        new = self._clone()
        new._mark = "image"
        new._pending_stat_mark = (
            "raster",
            {
                "aggregate": aggregate, "field": field, "cmap": cmap, "resolution": resolution,
                "blend": blend, "min_count": min_count, "log_scale": log_scale, **mark_kwargs,
            },
            desugar_raster,
        )
        new._position = position
        return new

    def mark_hex(
        self,
        *,
        bin_size=None,
        aggregate="count",
        field=None,
        cmap="viridis",
        stroke=None,
        stroke_width=0,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render data as hexagonally binned aggregates.

        Uses the ``Hex`` transform to bin x/y data onto a hexagonal grid and
        aggregate within each bin. Requires ``x`` and ``y`` encodings.

        Parameters
        ----------
        bin_size : int or float, optional
            Diameter of each hexagon in pixels. Inferred from data density
            when omitted.
        aggregate : str, default "count"
            Aggregation function for each hex bin (e.g. ``"count"``,
            ``"mean"``, ``"sum"``).
        field : str, optional
            Column used for aggregation. Required when ``aggregate`` is not
            ``"count"``.
        cmap : str, default "viridis"
            Color map name for mapping aggregated values to fill color.
        stroke : str, optional
            Hex border color. ``None`` disables the border.
        stroke_width : float, default 0
            Border stroke width in pixels.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the polygon marks.
        **mark_kwargs
            Additional mark-style overrides forwarded to the polygon layer.

        Returns
        -------
        Chart
            New chart with the hexagonal binning composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x", y="y").mark_hex()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("hex", position)
        from ferrum.marks.heavy_stat import desugar_hex
        new = self._clone()
        new._mark = "polygon"
        new._pending_stat_mark = (
            "hex",
            {
                "bin_size": bin_size, "aggregate": aggregate, "field": field, "cmap": cmap,
                "stroke": stroke, "stroke_width": stroke_width, **mark_kwargs,
            },
            desugar_hex,
        )
        new._position = position
        return new

    def mark_swarm(
        self,
        *,
        size=4,
        orient="vertical",
        spacing=1.0,
        side="both",
        dodge=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a beeswarm (jittered scatter) plot for categorical data.

        Uses the ``Swarm`` transform to compute non-overlapping point positions
        along the categorical axis. Requires ``x`` (categorical) and ``y``
        (numeric) encodings (or the reverse for horizontal orientation).

        Parameters
        ----------
        size : float, default 4
            Radius of each point in pixels; also controls the layout padding.
        orient : {"vertical", "horizontal"}, default "vertical"
            Direction of the swarm layout. ``"vertical"`` spreads points
            horizontally within each category.
        spacing : float, default 1.0
            Extra gap between adjacent points as a fraction of point diameter.
        side : {"both", "left", "right"}, default "both"
            Which side of the category axis to allow points to spread onto.
        dodge : str, optional
            Column used to sub-group points within each category level.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied after swarm layout.
        **mark_kwargs
            Additional mark-style overrides forwarded to the point layer.

        Returns
        -------
        Chart
            New chart with the beeswarm composite mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="species", y="petal_length").mark_swarm()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("swarm", position)
        from ferrum.marks.heavy_stat import desugar_swarm
        new = self._clone()
        new._mark = "point"
        new._pending_stat_mark = (
            "swarm",
            {
                "size": size, "orient": orient, "spacing": spacing, "side": side,
                "dodge": dodge, **mark_kwargs,
            },
            desugar_swarm,
        )
        new._position = position
        return new

    def mark_function(self, fn, *, domain=None, n=200, clip=True, position=None, **mark_kwargs) -> "Chart":
        """Render a mathematical function as a line over a numeric domain.

        Materializes a synthetic dataset by evaluating ``fn`` over ``n``
        evenly-spaced x values within ``domain``. The function data replaces
        any data passed to ``Chart()``.

        Parameters
        ----------
        fn : callable
            A function ``fn(x: numpy.ndarray) -> numpy.ndarray`` that maps x
            values to y values.
        domain : tuple of (float, float), optional
            The x range over which to evaluate ``fn``. Inferred from the
            parent chart's x data when omitted.
        n : int, default 200
            Number of sample points used to draw the function curve.
        clip : bool, default True
            Whether to clip the rendered line to the chart viewport.
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to the line mark.
        **mark_kwargs
            Additional mark-style overrides forwarded to the line layer.

        Returns
        -------
        Chart
            New chart backed by synthetic data with the function line mark.

        Examples
        --------
        >>> import ferrum as fm
        >>> import numpy as np
        >>> fm.Chart(df).encode(x="x").mark_function(np.sin)
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("function", position)
        if self._layers is not None and self._layers:
            raise NotImplementedError(
                "mark_function as a layer in a multi-layer Chart is deferred to Phase 9+; "
                "use a separate Chart composed via + instead"
            )
        from ferrum.marks.heavy_stat import desugar_function
        # Try to infer parent x data for domain inference
        parent_x_data = None
        x_enc = self._encoding.get("x")
        if x_enc is not None and self._data is not None:
            try:
                from ferrum._coerce import to_arrow_table
                x_field_name = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
                tbl = to_arrow_table(self._data)
                if x_field_name in tbl.column_names:
                    parent_x_data = tbl[x_field_name].to_numpy()
            except Exception:
                pass

        mark, transforms, remap, synthetic = desugar_function(
            fn, parent_chart_x_data=parent_x_data, domain=domain, n=n, clip=clip, **mark_kwargs,
        )
        new = self._clone()
        new._mark = mark
        new._data = synthetic
        new._transforms = list(self._transforms) + list(transforms)
        if remap:
            from ferrum.encoding import X, Y
            if "x" in remap:
                new._encoding["x"] = X(remap["x"], type="Q")
            if "y" in remap:
                new._encoding["y"] = Y(remap["y"], type="Q")
        new._position = position
        return new

    def mark_arc(self, **kwargs) -> "Chart":
        """Arc mark — deferred to a future phase.

        Parameters
        ----------
        **kwargs
            Accepted for API compatibility; raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always; arc marks are not yet implemented.
        """
        raise deferred_mark_error("arc")

    def mark_image(self, **kwargs) -> "Chart":
        """Image mark — deferred to a future phase.

        Parameters
        ----------
        **kwargs
            Accepted for API compatibility; raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always; image marks are not yet implemented.
        """
        raise deferred_mark_error("image")

    def mark_geoshape(self, **kwargs) -> "Chart":
        """Geographic shape mark — deferred to a future phase.

        Parameters
        ----------
        **kwargs
            Accepted for API compatibility; raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always; geoshape marks are not yet implemented.
        """
        raise deferred_mark_error("geoshape")
    def mark_segment(self, *, position=None, **kwargs) -> "Chart":
        """Render a diagonal line segment from (x, y) to (x2, y2).

        Distinct from ``mark_rule``, which is axis-aligned only; segments may
        take any direction. Requires ``x``, ``y``, ``x2``, and ``y2``
        encodings.

        Parameters
        ----------
        position : Identity, Dodge, Jitter, or Stack, optional
            Position adjustment applied to each segment.
        **kwargs
            Mark-style overrides forwarded to the renderer (e.g. ``color``,
            ``stroke_width``).

        Returns
        -------
        Chart
            New chart with the segment mark set.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="x1", y="y1", x2="x2", y2="y2").mark_segment()
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("segment", position)
        new = self._set_mark("segment", **kwargs)
        new._position = position
        return new
    def mark_label(self, **kwargs) -> "Chart":
        """Label mark — deferred to a future phase.

        Parameters
        ----------
        **kwargs
            Accepted for API compatibility; raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always; label marks are not yet implemented.
        """
        raise deferred_mark_error("label")

    # ---- Encoding ----

    def encode(self, **channels: Any) -> "Chart":
        """Declare data-to-visual encoding channels.

        Each keyword argument maps a channel name to a column name (string
        shorthand) or a typed channel object (e.g. ``X``, ``Color``).
        Shorthand strings accept ``"field"``, ``"field:Q"``, or
        ``"agg(field):Q"`` syntax.

        Parameters
        ----------
        **channels : str or ChannelBase
            Keyword arguments where each key is a channel name (``x``, ``y``,
            ``color``, ``size``, ``shape``, ``opacity``, ``x2``, ``y2``,
            ``text``, ``tooltip``, ``facet``, ``facet_row``, ``facet_col``,
            etc.) and each value is a column-name string or a typed channel
            instance.

        Returns
        -------
        Chart
            New chart with the declared encodings applied.

        Raises
        ------
        ValueError
            When an unknown channel name is passed.
        TypeError
            When a value is not a string, channel instance, or Repeat
            placeholder.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg", color="cyl:N").mark_point()
        """
        new = self._clone()
        for name, value in channels.items():
            cls = _channel_class_for(name)
            if cls is None:
                raise ValueError(f"unknown encoding channel: {name!r}")

            if isinstance(value, ChannelBase):
                channel = value
            elif isinstance(value, str):
                field, type_, agg = parse_shorthand(value)
                kw = {}
                if type_: kw["type"] = type_
                if agg: kw["aggregate"] = agg
                channel = cls(field, **kw)
            else:
                # Phase 9: accept Repeat sentinels (Repeat.column / .row / .layer).
                from ferrum.repeat import _RepeatPlaceholder
                if isinstance(value, _RepeatPlaceholder):
                    channel = cls(value)
                else:
                    raise TypeError(
                        f"encode({name}=...) expects str, {cls.__name__} instance, "
                        f"or Repeat placeholder; got {type(value).__name__}"
                    )

            new._encoding[name] = channel
            new._transforms.extend(channel.to_implicit_transforms())
        return new

    def transform(self, *transforms) -> "Chart":
        """Append one or more data transforms to the chart's pipeline.

        Transforms are applied in order before rendering. Multiple calls
        accumulate; earlier transforms feed into later ones.

        Parameters
        ----------
        *transforms : transform objects
            One or more Rust-backed transform instances (e.g. ``Aggregate``,
            ``Bin``, ``Smooth``, ``Kde``) to append to the pipeline.

        Returns
        -------
        Chart
            New chart with the additional transforms appended.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").transform(fm.Aggregate("mean", "mpg")).mark_bar()
        """
        new = self._clone()
        new._transforms = list(self._transforms) + list(transforms)
        return new

    # ---- Composition operators ----

    def __add__(self, other: "Chart") -> "Chart":
        """Overlay two charts into a multi-layer chart (``chart1 + chart2``).

        When both charts share the same data (identity or Arrow value
        equality), produces a layered ``Chart``. When the data differs,
        falls through to horizontal concatenation and emits a
        ``UserWarning``.

        Parameters
        ----------
        other : Chart
            The chart to overlay on top of this one.

        Returns
        -------
        Chart
            Layered chart when data matches; ``HConcatChart`` otherwise.

        Examples
        --------
        >>> import ferrum as fm
        >>> base = fm.Chart(df).encode(x="hp", y="mpg")
        >>> base.mark_point() + base.mark_smooth()
        """
        if not isinstance(other, Chart):
            return NotImplemented

        same_data = self._data is other._data
        if not same_data:
            try:
                a = to_arrow_table(self._data)
                b = to_arrow_table(other._data)
                same_data = a.equals(b)
            except Exception:
                same_data = False

        if not same_data:
            import warnings
            warnings.warn(
                "Layered charts with differing data render as horizontal concatenation. "
                "Use a shared DataFrame for true overlay.",
                UserWarning,
                stacklevel=2,
            )
            return self.__or__(other)

        # Same data — build a multi-layer chart.
        # Resolve pending statistical marks before snapshotting encoding dicts.
        lhs = self._resolve_pending()
        rhs = other._resolve_pending()
        new = lhs._clone()
        new._layers = [
            {
                "mark": lhs._mark,
                "encoding": dict(lhs._encoding),
                "transforms": list(lhs._transforms),
                "mark_style": dict(lhs._mark_kwargs),
                "position": lhs._position,
            },
            {
                "mark": rhs._mark,
                "encoding": dict(rhs._encoding),
                "transforms": list(rhs._transforms),
                "mark_style": dict(rhs._mark_kwargs),
                "position": rhs._position,
            },
        ]
        # Warn if secondary layer has conflicting theme/facet/coord
        if (
            (rhs._theme is not None and rhs._theme != lhs._theme)
            or rhs._facet != lhs._facet
            or rhs._coord != lhs._coord
        ):
            import warnings
            warnings.warn(
                "Layered chart `+`: secondary layer's theme/facet/coord is ignored; "
                "primary layer wins.",
                UserWarning,
                stacklevel=2,
            )
        return new

    def __or__(self, other: "Chart") -> "HConcatChart":
        """Concatenate two charts side-by-side (``chart1 | chart2``).

        Parameters
        ----------
        other : Chart
            Chart to place to the right of this one.

        Returns
        -------
        HConcatChart
            Horizontally concatenated chart pair.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).mark_point() | fm.Chart(df).mark_bar()
        """
        from ferrum.composition import HConcatChart
        return HConcatChart([self, other])

    def __and__(self, other: "Chart") -> "VConcatChart":
        """Stack two charts vertically (``chart1 & chart2``).

        Parameters
        ----------
        other : Chart
            Chart to place below this one.

        Returns
        -------
        VConcatChart
            Vertically concatenated chart pair.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).mark_point() & fm.Chart(df).mark_histogram()
        """
        from ferrum.composition import VConcatChart
        return VConcatChart([self, other])

    # ---- Facet / coord / theme ----

    def facet(
        self,
        field: Optional[str] = None,
        *,
        row: Optional[str] = None,
        col: Optional[str] = None,
        ncols: Optional[int] = None,
        nrows: Optional[int] = None,
    ) -> "Chart":
        """Split the chart into a small-multiples grid by a categorical column.

        Two layouts are supported. Wrap mode places panels left-to-right then
        wraps onto new rows. Grid mode places panels at explicit
        ``row`` × ``col`` positions.

        Parameters
        ----------
        field : str, optional
            Column used for wrap-mode faceting. Equivalent to passing
            ``col=field`` with no ``row``.
        row : str, optional
            Column for the row dimension in grid mode.
        col : str, optional
            Column for the column dimension in grid (or wrap) mode.
        ncols : int, optional
            Maximum number of columns per row in wrap mode.
        nrows : int, optional
            Maximum number of rows per column in grid mode.

        Returns
        -------
        Chart
            New chart with faceting configured.

        Raises
        ------
        ValueError
            When neither ``field``, ``row``, nor ``col`` is supplied.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().facet("cyl", ncols=2)
        """
        new = self._clone()
        if field is not None:
            new._facet = {
                "field": field,
                "mode_kind": "wrap",
                "ncols": ncols,
                "nrows": nrows,
            }
        elif row is not None and col is not None:
            new._facet = {
                "row": row,
                "col": col,
                "mode_kind": "grid",
                "nrows": nrows,
                "ncols": ncols,
            }
        elif col is not None:
            new._facet = {
                "field": col,
                "mode_kind": "wrap",
                "ncols": ncols,
                "nrows": nrows,
            }
        elif row is not None:
            new._facet = {
                "field": row,
                "mode_kind": "wrap",
                "nrows": nrows,
                "ncols": ncols,
            }
        else:
            raise ValueError("facet() requires either `field=`, or `row=`/`col=`")
        return new

    def theme(self, theme: Any) -> "Chart":
        """Attach a ``Theme`` to this chart, overriding the process default.

        Per-chart theme always wins over ``ferrum.set_default_theme()``.

        Parameters
        ----------
        theme : Theme
            Theme value to apply at render time.

        Returns
        -------
        Chart
            New chart with the theme attached.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().theme(fm.themes.dark)
        """
        new = self._clone()
        new._theme = theme
        return new

    def coord(self, coord: Any) -> "Chart":
        """Set the coordinate system for this chart.

        Currently only ``CoordFlip`` is supported.

        Parameters
        ----------
        coord : CoordFlip
            Coordinate system instance to apply.

        Returns
        -------
        Chart
            New chart with the coordinate system set.

        Raises
        ------
        TypeError
            When a coordinate type other than ``CoordFlip`` is passed.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="value", y="category").mark_bar().coord(fm.CoordFlip())
        """
        from ferrum.coord import CoordFlip
        new = self._clone()
        if isinstance(coord, CoordFlip):
            new._coord = "flip"
        else:
            raise TypeError(
                f"unsupported coord: {type(coord).__name__}; only CoordFlip supported in Phase 8a"
            )
        return new

    @staticmethod
    def _transforms_to_json_list(transforms: list) -> list:
        """Serialize a list of Python transform objects to JSON-safe dicts.

        ``coerce_layers`` in Rust calls ``json.dumps()`` on each layer dict, so
        PyO3 transform objects must be converted to plain dicts first.  We do
        this by round-tripping through ``ChartSpec.to_json()``.
        """
        if not transforms:
            return []
        import json as _json
        from ferrum import ChartSpec
        # Build a minimal spec with the transforms; extract the "transforms" array.
        dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=transforms)
        parsed = _json.loads(dummy.to_json())
        return parsed.get("transforms", [])

    def _build_layers_list(self) -> list:
        """Convert internal _layers to a list of JSON-serializable dicts for Rust.

        ``coerce_layers`` in Rust runs ``json.dumps()`` on each dict, so every
        value must be JSON-serializable (no PyO3 objects).
        """
        out = []
        for layer in (self._layers or []):
            encoding_dict: dict = {}
            for axis in ("x", "y", "x2", "y2", "color", "size", "shape", "opacity"):
                ch = layer.get("encoding", {}).get(axis)
                if ch is None:
                    continue
                if hasattr(ch, "to_encoding_spec_dict"):
                    # ChannelBase subclass — convert to a plain JSON-serializable dict.
                    d = ch.to_encoding_spec_dict()
                    field = d.get("field")
                    if not field:
                        continue
                    # Build a JSON-safe dict matching EncodingSpec's JSON shape.
                    enc_json_dict: dict = {"field": field}
                    if d.get("type"):
                        enc_json_dict["type_"] = d["type"]
                    for opt_key in ("title", "aggregate", "scheme"):
                        if d.get(opt_key):
                            enc_json_dict[opt_key] = d[opt_key]
                    encoding_dict[axis] = enc_json_dict
                elif isinstance(ch, str):
                    encoding_dict[axis] = {"field": ch}
            # Accept either legacy `mark_style` (8a layered overlays) or
            # `mark_kwargs` (composite-mark desugar, Phase 8b Tasks 23-33).
            mark_style = layer.get("mark_style") or layer.get("mark_kwargs") or {}
            layer_dict: dict = {"mark": layer.get("mark", "point"), "encoding": encoding_dict}
            if mark_style:
                layer_dict["mark_style"] = dict(mark_style)
            # data_source: composite-mark layers may pull from a named transform
            # output instead of the final pipeline batch. Only emit when set.
            data_source = layer.get("data_source")
            if data_source is not None:
                layer_dict["data_source"] = data_source
            # Serialize transforms: PyO3 objects need round-tripping through ChartSpec JSON.
            raw_transforms = layer.get("transforms") or []
            if raw_transforms:
                layer_dict["transforms"] = Chart._transforms_to_json_list(raw_transforms)
            # Phase 9c — per-layer position adjustment. Serialize value classes
            # via ``to_spec_dict``; allow already-dict payloads to pass through.
            position = layer.get("position")
            if position is not None:
                layer_dict["position"] = (
                    position.to_spec_dict()
                    if hasattr(position, "to_spec_dict")
                    else position
                )
            out.append(layer_dict)
        return out

    def _build_facet_dict(self) -> dict:
        """Convert internal _facet to the JSON dict Rust's FacetSpec expects."""
        f = self._facet
        mode_kind = f.get("mode_kind", "wrap")
        if mode_kind == "wrap":
            field = f["field"]
            ncols = f.get("ncols") or 1  # u32 required; default 1
            return {"field": field, "mode": {"kind": "wrap", "ncols": int(ncols)}}
        else:  # grid
            # Rust FacetSpec has a single `field`. Use col as primary, row for nrows.
            field = f.get("col", f.get("field", ""))
            nrows = f.get("nrows") or 1
            ncols = f.get("ncols") or 1
            return {
                "field": field,
                "mode": {"kind": "grid", "nrows": int(nrows), "ncols": int(ncols)},
            }

    # ---- Properties ----

    def properties(self, *, width=None, height=None, title=None, description=None) -> "Chart":
        """Set chart-level display properties.

        Parameters
        ----------
        width : int or str, optional
            Chart width in pixels or as a CSS string. Overrides the value set
            at construction.
        height : int or str, optional
            Chart height in pixels or as a CSS string. Overrides the value set
            at construction.
        title : str, optional
            Chart title rendered above the axes.
        description : str, optional
            Accessible description for screen readers; not rendered visually.

        Returns
        -------
        Chart
            New chart with the specified properties updated.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().properties(width=500, title="MPG vs HP")
        """
        new = self._clone()
        if width is not None: new._width = width
        if height is not None: new._height = height
        if title is not None: new._title = title
        if description is not None: new._description = description
        return new

    # ---- Spec output ----

    def to_spec(self) -> "ChartSpec":
        """Build and return the ``ChartSpec`` intermediate representation.

        Resolves any pending statistical mark desugar (when a mark method was
        called before ``.encode()``), then constructs the ``ChartSpec`` that
        the Rust renderer consumes.

        Returns
        -------
        ChartSpec
            Fully resolved chart specification ready for rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> spec = fm.Chart(df).encode(x="hp", y="mpg").mark_point().to_spec()
        """
        # Resolve any pending statistical mark desugar (mark called before encode).
        resolved = self._resolve_pending()
        from ferrum import ChartSpec, EncodingSpec
        # Build full EncodingSpec instances per channel so honored kwargs
        # (scale, title) and deferred kwargs (axis, legend, sort, ...) flow to Rust.
        # Phase 7 + 8a's ChartSpec(...) accepts EncodingSpec instances or strings.
        kw = {"mark": resolved._mark or "point", "data": "default"}
        from ferrum.repeat import _RepeatPlaceholder
        for axis in ("x", "y", "color", "size", "shape", "opacity"):
            if axis in resolved._encoding:
                ch = resolved._encoding[axis]
                if ch.field is None:
                    continue   # Tooltip(*fields) etc. with no single field
                # Phase 9: skip channels whose field is an unresolved Repeat
                # placeholder. RepeatChart.expand() materializes concrete charts
                # before render; the bare template's spec just omits placeholder
                # channels (they're not meaningful standalone).
                if isinstance(ch.field, _RepeatPlaceholder):
                    continue
                d = ch.to_encoding_spec_dict()
                # `field` is positional; rest are keyword-only on EncodingSpec.__new__.
                # The Python-visible param name is `type_` (Rust signature `type_: Option<&str>`).
                field = d.pop("field")
                kw[axis] = EncodingSpec(field, **d)
        if resolved._transforms:
            kw["transforms"] = list(resolved._transforms)
        if resolved._facet is not None:
            kw["facet"] = resolved._build_facet_dict()
        if resolved._coord is not None:
            kw["coord"] = resolved._coord
        if resolved._mark_kwargs:
            kw["mark_style"] = dict(resolved._mark_kwargs)
        if resolved._layers is not None:
            kw["layers"] = resolved._build_layers_list()
        if resolved._position is not None:
            kw["position"] = resolved._position.to_spec_dict()
        return ChartSpec(**kw)

    def _build_spec(self):
        """Build the chart spec with typed Python access to layers.

        For single-layer charts this is a thin alias for ``to_spec``. For
        layered charts it returns a Python-side ``_SpecView`` that wraps the
        underlying ``ChartSpec`` and exposes ``.layers`` as a list of
        ``types.SimpleNamespace`` items with ``.mark``, ``.encoding``,
        ``.mark_kwargs``, and ``.data_source`` attributes — matching the
        Layer-instance contract from spec §12.1 without requiring a parallel
        ``PyLayer`` Rust class. JSON / serialization remains the underlying
        ``ChartSpec`` (delegated via ``__getattr__``).
        """
        resolved = self._resolve_pending()
        spec = resolved.to_spec()
        if resolved._layers is None:
            return spec
        return _SpecView(spec, resolved._layers)

    def to_json(self, *, indent=None) -> str:
        """Serialize the chart specification to a JSON string.

        Returns
        -------
        str
            JSON-encoded ``ChartSpec``.

        Examples
        --------
        >>> import ferrum as fm
        >>> json_str = fm.Chart(df).encode(x="hp", y="mpg").mark_point().to_json()
        """
        spec = self.to_spec()
        return spec.to_json()

    def show_svg(self) -> str:
        """Render the chart to an SVG string.

        Returns
        -------
        str
            An SVG document string representing the chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> svg = fm.Chart(df).encode(x="hp", y="mpg").mark_point().show_svg()
        """
        # Stub — full impl in Task 32
        from ferrum._core import render_svg
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_svg(spec, data, viewport=viewport, theme=theme_dict)

    def show_png(self) -> bytes:
        """Render the chart to a PNG byte string.

        Returns
        -------
        bytes
            PNG-encoded image bytes.

        Examples
        --------
        >>> import ferrum as fm
        >>> png = fm.Chart(df).encode(x="hp", y="mpg").mark_point().show_png()
        """
        from ferrum._core import render_png
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_png(spec, data, viewport=viewport, theme=theme_dict)

    def save(self, path, *, format=None, **render_kwargs) -> None:
        """Save the chart to disk as SVG or PNG.

        The output format is inferred from the file extension when ``format``
        is not provided.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. Extension determines format when ``format``
            is ``None``.
        format : {"svg", "png"}, optional
            Explicit output format. Overrides extension-based inference.
        **render_kwargs
            Additional keyword arguments forwarded to the renderer.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().save("chart.svg")
        """
        from ferrum.display import save_chart
        save_chart(self, path, format=format, **render_kwargs)

    def show(self) -> None:
        """Display the chart inline in Jupyter or open in a browser fallback.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.Chart(df).encode(x="hp", y="mpg").mark_point().show()
        """
        from ferrum.display import show_chart
        show_chart(self)

    def _repr_svg_(self) -> str | None:
        """Jupyter SVG rich display hook."""
        try:
            return self.show_svg()
        except Exception:
            return None  # let Jupyter fall back to __repr__

    def _repr_html_(self) -> str | None:
        """Jupyter HTML rich display hook — wraps SVG in a <div>."""
        try:
            return f"<div>{self.show_svg()}</div>"
        except Exception:
            return None

    # Stubs for Phase 11
    def add_selection(self, *selections) -> "Chart":
        """Add interactive selection brushes — deferred to Phase 11.

        Parameters
        ----------
        *selections
            Accepted for API compatibility; raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always; selections require the interactive renderer (Phase 11).
        """
        raise NotImplementedError("selections require .interactive() — Phase 11")

    def interactive(self) -> "Chart":
        """Enable the interactive renderer — deferred to Phase 11.

        Raises
        ------
        NotImplementedError
            Always; the interactive renderer is not yet implemented.
        """
        raise NotImplementedError("interactive renderer — Phase 11")

    def __repr__(self) -> str:
        """Return a concise string representation of the chart.

        Returns
        -------
        str
            String showing the active mark and declared encoding channels.
        """
        return f"Chart(mark={self._mark!r}, encoding={list(self._encoding.keys())})"


class _SpecView:
    """Python-side typed view over a layered ``ChartSpec``.

    Exposes ``.layers`` as a list of ``types.SimpleNamespace`` items so callers
    can write ``spec.layers[0].mark.name`` and ``spec.layers[0].data_source``
    against the spec returned by ``Chart._build_spec()``. All other attribute
    access (``to_json``, ``mark``, ``encoding``, ``transforms``, etc.) and
    serialization fall through to the underlying ``ChartSpec`` instance.

    This is the Python-side typed view, not a parallel Rust type — Rust's
    ``coerce_layers`` already converts the same source dicts into ``Layer``
    structs internally during ``ChartSpec(...)`` construction.
    """

    __slots__ = ("_spec", "_layer_dicts", "_layers_cached")

    def __init__(self, spec, layer_dicts: list) -> None:
        self._spec = spec
        self._layer_dicts = layer_dicts
        self._layers_cached: Optional[list] = None

    @property
    def layers(self) -> list:
        if self._layers_cached is not None:
            return self._layers_cached
        from types import SimpleNamespace
        out = []
        for d in self._layer_dicts:
            mark_name = d.get("mark", "point")
            mark_obj = SimpleNamespace(name=mark_name)
            mark_kwargs = d.get("mark_kwargs") or d.get("mark_style")
            ns = SimpleNamespace(
                mark=mark_obj,
                encoding=d.get("encoding") or {},
                mark_kwargs=mark_kwargs if mark_kwargs else None,
                data_source=d.get("data_source"),
                transforms=list(d.get("transforms") or []),
            )
            out.append(ns)
        self._layers_cached = out
        return out

    def to_json(self, *args, **kwargs) -> str:
        return self._spec.to_json(*args, **kwargs)

    def __getattr__(self, name: str):
        # Called only if normal attribute lookup fails — delegates everything
        # else to the underlying ChartSpec.
        return getattr(self._spec, name)

    def __repr__(self) -> str:
        return f"_SpecView({self._spec!r}, layers={len(self._layer_dicts)})"
