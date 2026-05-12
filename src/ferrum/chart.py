"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from ferrum._coerce import to_arrow_table
from ferrum._shorthand import parse_shorthand
from ferrum._spec_view import _SpecView
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

    Every method returns a new ``Chart`` — the object is effectively immutable and
    safe to reuse across branches of a pipeline.  The fluent API follows the pattern::

        fm.Chart(df).mark_point().encode(x="sepal_length", y="sepal_width")

    Parameters
    ----------
    data : DataFrame-like or None, optional
        Input data.  Accepts Polars ``DataFrame``, pandas ``DataFrame``, PyArrow
        ``Table`` / ``RecordBatch``, dict-of-arrays, list-of-records, or a 2-D
        NumPy array.  Passed through ``ferrum._coerce.to_arrow_table`` at render
        time.  ``None`` is allowed when composing layered charts that share a
        parent's data.
    width : int or "container", optional
        Chart width in pixels, or ``"container"`` to fill the parent element.
        Default is ``600``.
    height : int or "container", optional
        Chart height in pixels.  Default is ``400``.
    title : str, optional
        Title rendered above the plot area.
    description : str, optional
        Accessible description attached to the SVG root (``<title>`` element).

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> svg = chart.show_svg()
    """

    __slots__ = (
        "_data", "_mark", "_mark_kwargs", "_encoding", "_transforms",
        "_facet", "_coord", "_theme", "_layers",
        "_width", "_height", "_title", "_description",
        "_pending_stat_mark",  # (kind, kwargs) when mark_* called before .encode()
        "_position",           # Phase 9c — Identity / Dodge / Jitter / Stack (or None)
        "_axis_x", "_axis_y",  # spec-level axis suppression (.axis(x=False) / .axis(y=False))
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
        self._axis_x: Optional[bool] = None
        self._axis_y: Optional[bool] = None

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
        new._axis_x = self._axis_x
        new._axis_y = self._axis_y
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
            orientation = kwargs.get("orientation", "vertical")
            field_channel = "y" if orientation == "horizontal" else "x"
            enc = new._encoding.get(field_channel)
            if enc is None:
                raise ValueError(
                    f"mark_density() requires .encode({field_channel}=...) "
                    f"to specify the density field"
                )
            field = enc.field if isinstance(enc, ChannelBase) else enc
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
            orientation = kwargs.get("orientation", "vertical")
            field_channel = "y" if orientation == "horizontal" else "x"
            enc = new._encoding.get(field_channel)
            if enc is None:
                raise ValueError(
                    f"mark_histogram() requires .encode({field_channel}=...) "
                    f"to specify the histogram field"
                )
            field = enc.field if isinstance(enc, ChannelBase) else enc
            mark, transforms, remap = desugar_histogram(field, **kwargs)
            new._mark = mark
            new._transforms = list(new._transforms) + transforms
            from ferrum.encoding import X, X2, Y, Y2
            if orientation == "horizontal":
                new._encoding["y"] = Y(remap["y"], type="Q")
                new._encoding["y2"] = Y2(remap["y2"], type="Q")
                new._encoding["x"] = X(remap["x"], type="Q")
            else:
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

    def _set_composite_mark(
        self,
        name: str,
        desugar_fn,
        kwargs: dict,
        *,
        placeholder: str,
        position=None,
        data_transform=None,
    ) -> "Chart":
        # Shared scaffold for composite/diagnostic mark_* methods. Validates the
        # position adjustment, clones, sets the placeholder mark (overridden by
        # layered mode at render time), optionally rewrites the polars data,
        # and stashes the desugar callable in the 3-tuple _pending_stat_mark
        # sentinel resolved by _resolve_pending once .encode() is known.
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility(name, position)
        new = self._clone()
        new._mark = placeholder
        if data_transform is not None and new._data is not None:
            try:
                import polars as pl
                if isinstance(new._data, pl.DataFrame):
                    new._data = data_transform(new._data)
            except ImportError:
                pass
        new._pending_stat_mark = (name, dict(kwargs), desugar_fn)
        new._position = position
        return new

    def mark_point(self, **kwargs) -> "Chart":
        """Render data as points (scatter plot).

        Parameters
        ----------
        size : float, optional
            Point area in square pixels.  Default is ``36``.
        fill : str, optional
            Fill colour override (CSS colour string or hex).  Normally driven
            by the ``color`` encoding.
        stroke : str, optional
            Stroke colour for the point outline.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        filled : bool, optional
            Whether points are filled.  Default is ``True``.
        shape : str, optional
            Point shape: ``"circle"``, ``"square"``, ``"cross"``, ``"diamond"``,
            ``"triangle-up"``, ``"triangle-down"``.
        stroke_width : float, optional
            Stroke width in pixels.
        position : Position, optional
            Position adjustment — ``fm.Jitter()``, ``fm.Dodge()``, etc.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"point"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_point(size=60, opacity=0.7).encode(x="x", y="y")
        Chart(mark='point', encoding=['x', 'y'])
        """
        return self._set_mark("point", **kwargs)

    def mark_line(self, **kwargs) -> "Chart":
        """Render data as a connected line.

        Parameters
        ----------
        stroke : str, optional
            Stroke colour override.
        stroke_width : float, optional
            Line width in pixels.  Default is ``2``.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        interpolate : str, optional
            Line interpolation: ``"linear"``, ``"monotone"``, ``"step"``,
            ``"step-before"``, ``"step-after"``, ``"basis"``, ``"cardinal"``.
        stroke_cap : str, optional
            Line cap: ``"butt"``, ``"round"``, ``"square"``.
        stroke_join : str, optional
            Line join: ``"miter"``, ``"round"``, ``"bevel"``.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"line"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_line(stroke_width=3, interpolate="monotone").encode(x="x", y="y")
        Chart(mark='line', encoding=['x', 'y'])
        """
        return self._set_mark("line", **kwargs)

    def mark_bar(self, **kwargs) -> "Chart":
        """Render data as bars.

        Parameters
        ----------
        fill : str, optional
            Bar fill colour override.
        stroke : str, optional
            Bar stroke colour.
        stroke_width : float, optional
            Bar stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        corner_radius : float, optional
            Rounded corner radius in pixels.
        orient : str, optional
            ``"vertical"`` (default) or ``"horizontal"``.
        position : Position, optional
            Position adjustment — ``fm.Stack()``, ``fm.Dodge()``, etc.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"bar"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
        >>> fm.Chart(df).mark_bar(corner_radius=4).encode(x="cat", y="val")
        Chart(mark='bar', encoding=['x', 'y'])
        """
        return self._set_mark("bar", **kwargs)

    def mark_area(self, **kwargs) -> "Chart":
        """Render data as a filled area between the line and a baseline.

        Parameters
        ----------
        fill : str, optional
            Fill colour override.
        stroke : str, optional
            Border stroke colour.
        stroke_width : float, optional
            Border stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        interpolate : str, optional
            Area boundary interpolation: ``"linear"``, ``"monotone"``,
            ``"step"``, ``"basis"``, ``"cardinal"``.
        line : bool, optional
            Whether to draw the top boundary as a line.  Default is ``False``.
        position : Position, optional
            Position adjustment — e.g. ``fm.Stack()``.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"area"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_area(opacity=0.5, line=True).encode(x="x", y="y")
        Chart(mark='area', encoding=['x', 'y'])
        """
        return self._set_mark("area", **kwargs)

    def mark_rule(self, **kwargs) -> "Chart":
        """Render a horizontal or vertical reference rule spanning the plot area.

        A rule spans the full width (when ``y`` is encoded) or full height
        (when ``x`` is encoded).  Encoding both ``x``/``x2`` or ``y``/``y2``
        draws finite line segments instead.

        Parameters
        ----------
        stroke : str, optional
            Rule colour override.
        stroke_width : float, optional
            Rule width in pixels.  Default is ``1``.
        stroke_dash : str or list, optional
            Dash pattern, e.g. ``"4,2"``.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"rule"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"y": [0.0]})
        >>> fm.Chart(df).mark_rule(stroke="red", stroke_width=2).encode(y="y")
        Chart(mark='rule', encoding=['y'])
        """
        return self._set_mark("rule", **kwargs)

    def mark_text(self, **kwargs) -> "Chart":
        """Render data values as text labels.

        Requires a ``text`` encoding channel pointing at the column to display.

        Parameters
        ----------
        fill : str, optional
            Text colour override.
        font_size : float, optional
            Font size in points.  Default is ``11``.
        font_weight : str or int, optional
            CSS font-weight, e.g. ``"bold"``, ``400``, ``700``.
        align : str, optional
            Horizontal alignment: ``"left"``, ``"center"``, ``"right"``.
        baseline : str, optional
            Vertical alignment: ``"top"``, ``"middle"``, ``"bottom"``,
            ``"alphabetic"``.
        dx : float, optional
            Horizontal pixel offset from the anchor point.
        dy : float, optional
            Vertical pixel offset from the anchor point.
        angle : float, optional
            Rotation in degrees.
        limit : float, optional
            Maximum rendered width in pixels; clips overflow.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"text"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "label": ["A", "B"]})
        >>> fm.Chart(df).mark_text(dy=-8).encode(x="x", y="y", text="label")
        Chart(mark='text', encoding=['x', 'y', 'text'])
        """
        return self._set_mark("text", **kwargs)

    def mark_tick(self, **kwargs) -> "Chart":
        """Render data as short tick marks (rug / strip plot).

        Parameters
        ----------
        stroke : str, optional
            Tick colour override.
        stroke_width : float, optional
            Tick stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        band_size : float, optional
            Tick length in pixels.
        orient : str, optional
            ``"vertical"`` (default ticks perpendicular to x) or
            ``"horizontal"``.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"tick"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.1, 2.3, 3.5, 2.2, 1.8]})
        >>> fm.Chart(df).mark_tick().encode(x="x")
        Chart(mark='tick', encoding=['x'])
        """
        return self._set_mark("tick", **kwargs)

    def mark_rect(self, **kwargs) -> "Chart":
        """Render data as rectangles (heatmap cells, Gantt bars).

        Requires ``x``/``x2`` or ``y``/``y2`` encoding pairs (or ordinal
        ``x``/``y`` for a heatmap cell grid).

        Parameters
        ----------
        fill : str, optional
            Rectangle fill colour override.
        stroke : str, optional
            Rectangle border stroke colour.
        stroke_width : float, optional
            Border stroke width in pixels.
        opacity : float, optional
            Overall opacity in ``[0, 1]``.
        corner_radius : float, optional
            Rounded corner radius in pixels.
        position : Position, optional
            Position adjustment.

        Returns
        -------
        Chart
            New ``Chart`` with mark set to ``"rect"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"row": ["A", "B"], "col": ["X", "Y"], "val": [1.0, 0.5]})
        >>> fm.Chart(df).mark_rect().encode(x="col", y="row", color="val")
        Chart(mark='rect', encoding=['x', 'y', 'color'])
        """
        return self._set_mark("rect", **kwargs)

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
        position : Position, optional
            Position adjustment.

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
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("density", position)
        orientation = kwargs.get("orientation", "vertical")
        # For horizontal orientation, the data field is bound to y rather
        # than x (JointChart's right-marginal pattern).
        field_channel = "y" if orientation == "horizontal" else "x"
        enc = self._encoding.get(field_channel)
        if enc is None:
            # Encoding not yet set — defer resolution to render time.
            new = self._clone()
            new._mark = "area"  # placeholder so _mark is not None
            new._pending_stat_mark = ("density", dict(kwargs))
            new._position = position
            return new
        field = enc.field if isinstance(enc, ChannelBase) else enc
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
            Whether bins are closed on the right.  Default is ``True``.
        cumulative : bool, optional
            Render a cumulative histogram.  Default is ``False``.
        multiple : str, optional
            How to handle grouped histograms when a ``color`` encoding is
            present: ``"layer"`` (default), ``"stack"``, ``"dodge"``.
        position : Position, optional
            Position adjustment.

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
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("histogram", position)
        orientation = kwargs.get("orientation", "vertical")
        # For horizontal orientation, the binned column is bound to y rather
        # than x (JointChart's right-marginal pattern).
        field_channel = "y" if orientation == "horizontal" else "x"
        enc = self._encoding.get(field_channel)
        if enc is None:
            new = self._clone()
            new._mark = "bar"  # placeholder
            new._pending_stat_mark = ("histogram", dict(kwargs))
            new._position = position
            return new
        field = enc.field if isinstance(enc, ChannelBase) else enc
        mark, transforms, remap = desugar_histogram(field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        from ferrum.encoding import X, X2, Y, Y2
        if orientation == "horizontal":
            new._encoding["y"] = Y(remap["y"], type="Q")
            new._encoding["y2"] = Y2(remap["y2"], type="Q")
            new._encoding["x"] = X(remap["x"], type="Q")
        else:
            new._encoding["x"] = X(remap["x"], type="Q")
            new._encoding["x2"] = X2(remap["x2"], type="Q")
            new._encoding["y"] = Y(remap["y"], type="Q")
        new._position = position
        return new

    def mark_smooth(self, *, position=None, **kwargs) -> "Chart":
        """Render a smoothed regression line with optional confidence interval band.

        Fits a smooth curve through ``(x, y)`` data using the method specified by
        ``method``.  When ``ci`` is set, emits a layered ribbon (CI band) + line
        chart.  Can be called before or after ``.encode(x=..., y=...)``.

        Parameters
        ----------
        method : str, optional
            Smoothing method: ``"loess"`` (default), ``"linear"``, ``"quadratic"``,
            ``"cubic"``, ``"log"``, ``"sqrt"``.
        degree : int, optional
            Polynomial degree for ``"linear"``/``"quadratic"``/``"cubic"``.
        bandwidth : float, optional
            LOESS bandwidth fraction in ``(0, 1]``.  Default is ``0.75``.
        n : int, optional
            Number of evaluation points along the x range.  Default is ``200``.
        ci : float or None, optional
            Confidence level for the interval band, e.g. ``0.95``.  When set,
            produces a layered ribbon + line chart.  Default is ``None``.
        position : Position, optional
            Position adjustment.

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
        """Render a Tukey boxplot via composite mark desugaring.

        Desugars to a multi-layer chart: box (IQR rect), upper and lower
        whisker rules, median rule, and (optionally) outlier points.  Requires
        a categorical ``x`` encoding and a continuous ``y`` (or the reverse
        when ``horizontal=True``).

        Parameters
        ----------
        extent : float, optional
            Whisker reach as a multiple of IQR.  Default is ``1.5``.
        size : float or None, optional
            Box width in pixels.  ``None`` (default) lets the renderer choose.
        outliers : bool, optional
            Whether to draw outlier points beyond the whiskers.  Default is
            ``True``.
        color_field : str or None, optional
            Column name to drive per-group fill colour on the box layer.
        horizontal : bool, optional
            Swap x and y axes so boxes run horizontally.  Default is ``False``.
        position : Position, optional
            Position adjustment — e.g. ``fm.Dodge()`` for side-by-side boxes.
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
        from ferrum.marks.composite import desugar_boxplot
        return self._set_composite_mark(
            "boxplot",
            desugar_boxplot,
            {
                "extent": extent,
                "size": size,
                "outliers": outliers,
                "color_field": color_field,
                "horizontal": horizontal,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

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
        """Render a letter-value (boxen) plot via composite mark desugaring.

        Produces nested rectangular bands for each letter-value depth, a median
        rule, and an outlier-point layer via the ``LetterValue`` transform.
        Requires a categorical ``x`` encoding and a continuous ``y`` (or the
        reverse when ``horizontal=True``).

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
        return self._set_composite_mark(
            "boxen",
            desugar_boxen,
            {
                "k_depth": k_depth,
                "k_proportion": k_proportion,
                "outlier_threshold": outlier_threshold,
                "palette": palette,
                "horizontal": horizontal,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_errorbar(self, *, extent="ci", ticks=True, position=None, **mark_kwargs) -> "Chart":
        """Render error bars via the ``ErrorExtent`` transform.

        Computes extent values (CI, SD, SEM, or IQR) per group defined by the
        ``x`` encoding, then draws a vertical rule spanning the extent with
        optional tick caps.

        Parameters
        ----------
        extent : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Extent measure: confidence interval (``"ci"``), standard error
            (``"stderr"``), standard deviation (``"stdev"``), or
            interquartile range (``"iqr"``).
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
        >>> fm.Chart(df).mark_errorbar(extent="stdev").encode(x="group", y="val")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.composite import desugar_errorbar
        return self._set_composite_mark(
            "errorbar",
            desugar_errorbar,
            {"extent": extent, "ticks": ticks, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_errorband(self, *, extent="ci", borders=False, position=None, **mark_kwargs) -> "Chart":
        """Render an error band (ribbon) via the ``ErrorExtent`` transform.

        Similar to ``mark_errorbar`` but renders the extent as a filled ribbon
        rather than whisker rules.  Optionally draws border lines at the upper
        and lower extent edges.

        Parameters
        ----------
        extent : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Extent measure: confidence interval (``"ci"``), standard error
            (``"stderr"``), standard deviation (``"stdev"``), or
            interquartile range (``"iqr"``).
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
        >>> fm.Chart(df).mark_errorband(extent="ci", borders=True).encode(x="x", y="y")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.composite import desugar_errorband
        return self._set_composite_mark(
            "errorband",
            desugar_errorband,
            {"extent": extent, "borders": borders, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_ribbon(self, *, opacity=0.3, interpolate="linear", position=None, **mark_kwargs) -> "Chart":
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
        cmap="viridis",
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
            ``False``.
        cmap : str, optional
            Colour map name applied to contour levels.  Default is ``"viridis"``.
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
        return self._set_composite_mark(
            "contour",
            desugar_contour,
            {
                "bandwidth": bandwidth, "thresholds": thresholds, "smooth": smooth,
                "fill": fill, "cmap": cmap, **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_violin(self, *, bandwidth="scott", inner="box", position=None, **mark_kwargs) -> "Chart":
        """Render violin plots via the ``Violin`` transform.

        Estimates a mirrored KDE per group and overlays an optional inner
        summary (box, quartile lines, or individual points).  Requires a
        categorical ``x`` and continuous ``y`` encoding (or swapped when
        ``horizontal`` is set).

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            KDE bandwidth.  Default is ``"scott"``.
        inner : {"box", "quartile", "point", "none"}, default "box"
            Inner mark drawn on top of each violin:
            ``"box"`` — IQR box + median rule,
            ``"quartile"`` — three horizontal quartile rules,
            ``"point"`` — individual data points (strip),
            ``"none"`` — no inner mark.
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
        return self._set_composite_mark(
            "violin",
            desugar_violin,
            {"bandwidth": bandwidth, "inner": inner, **mark_kwargs},
            placeholder="polygon",
            position=position,
        )

    def mark_qq(self, *, distribution="normal", dequantize=False, line=True, position=None, **mark_kwargs) -> "Chart":
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
        cmap="viridis",
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
        cmap : str, optional
            Colour map name.  Default is ``"viridis"``.
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
        return self._set_composite_mark(
            "raster",
            desugar_raster,
            {
                "aggregate": aggregate, "field": field, "cmap": cmap, "resolution": resolution,
                "blend": blend, "min_count": min_count, "log_scale": log_scale, **mark_kwargs,
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
        cmap="viridis",
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
        cmap : str, optional
            Colour map name.  Default is ``"viridis"``.
        stroke : str or None, optional
            Hex border colour.  ``None`` (default) means no border.
        stroke_width : float, optional
            Hex border width in pixels.  Default is ``0``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to the polygon layer.

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
        return self._set_composite_mark(
            "hex",
            desugar_hex,
            {
                "bin_size": bin_size, "aggregate": aggregate, "field": field, "cmap": cmap,
                "stroke": stroke, "stroke_width": stroke_width, **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

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
        """Render a beeswarm (strip swarm) plot via the ``Swarm`` transform.

        Computes non-overlapping point positions along the categorical axis
        using a deterministic placement algorithm seeded for reproducibility.
        Requires a continuous ``y`` (or ``x``) and an optional categorical
        grouping axis.

        Parameters
        ----------
        size : float, optional
            Point diameter in pixels.  Default is ``4``.
        orient : {"vertical", "horizontal"}, default "vertical"
            Orientation of the swarm axis.  ``"vertical"`` spreads points
            along x; ``"horizontal"`` spreads along y.
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
        return self._set_composite_mark(
            "swarm",
            desugar_swarm,
            {
                "size": size, "orient": orient, "spacing": spacing, "side": side,
                "dodge": dodge, **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_function(self, fn, *, domain=None, n=200, clip=True, position=None, **mark_kwargs) -> "Chart":
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
        NotImplementedError
            When called on a chart that already has layers (use ``+`` instead).
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

    # ---- Marks (diagnostic — Phase 10) ----

    def mark_residuals(
        self,
        *,
        kind: str = "studentized",
        reference_line: bool = True,
        cook_threshold: float | str | None = None,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a residuals diagnostic plot.

        Plots fitted values (``y_pred``) on x against residuals (raw or
        studentized) on y.  Data must carry the schema emitted by
        ``ModelSource.predictions()``: ``y_true``, ``y_pred``, ``residual``,
        ``studentized_residual``, and ``cooks_distance``.

        When ``reference_line=True`` a sentinel ``_ref_zero`` column is
        injected so the downstream ``mark_rule`` draws a single horizontal line
        at y=0.

        When ``cook_threshold`` is set, high-leverage points (Cook's D above
        the threshold) are highlighted as a second ``mark_point`` layer drawn
        in red with a black outline.

        Parameters
        ----------
        kind : {"studentized", "raw"}, default "studentized"
            Residual type to plot on the y axis.  ``"studentized"`` uses
            ``studentized_residual``; ``"raw"`` uses ``residual``.
        reference_line : bool, optional
            Whether to draw a horizontal reference line at y=0.  Default is
            ``True``.
        cook_threshold : float, "auto", or None, optional
            Cook's Distance threshold for outlier highlighting.  ``"auto"``
            uses the conventional ``4 / n`` rule.  ``None`` (default) disables
            outlier highlighting.
        color_field : str or None, optional
            Column name to drive per-group colour on the scatter layer.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the scatter layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for residuals rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.predictions()).mark_residuals(cook_threshold="auto")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_residuals
        from ferrum._diagnostics.charts import _inject_constant
        return self._set_composite_mark(
            "residuals",
            desugar_residuals,
            {
                "kind": kind,
                "reference_line": reference_line,
                "cook_threshold": cook_threshold,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=(
                (lambda df: _inject_constant(df, "_ref_zero", 0.0))
                if reference_line
                else None
            ),
        )

    def mark_prediction_error(
        self,
        *,
        identity_line: bool = True,
        ci: float | None = None,
        reference_band: bool = False,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render an actual-vs-predicted plot.

        Plots ``y_true`` on x against ``y_pred`` on y as scatter points.
        When ``identity_line=True`` the data is pre-sorted ascending by
        ``y_true`` so the downstream ``mark_line`` renders a monotonic y=x
        diagonal.  Data must carry ``y_true`` and ``y_pred`` columns (schema
        from ``ModelSource.predictions()``).

        Parameters
        ----------
        identity_line : bool, optional
            Whether to overlay a y=x identity reference line.  Default is
            ``True``.
        ci : float or None, optional
            Confidence level for a prediction-interval band (e.g. ``0.95``).
            ``None`` (default) omits the band.
        reference_band : bool, optional
            Whether to draw a shaded reference band around the identity line.
            Default is ``False``.
        color_field : str or None, optional
            Column name to drive per-group colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the scatter layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for prediction-error rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.predictions()).mark_prediction_error(ci=0.95)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_prediction_error
        from ferrum._diagnostics.charts import _sort_by
        return self._set_composite_mark(
            "prediction_error",
            desugar_prediction_error,
            {
                "identity_line": identity_line,
                "ci": ci,
                "reference_band": reference_band,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=(
                (lambda df: _sort_by(df, "y_true")) if identity_line else None
            ),
        )

    def mark_roc(
        self,
        *,
        average: str | None = None,
        reference_line: bool = True,
        annotate_auc: bool = False,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Receiver Operating Characteristic (ROC) curve.

        Plots false-positive rate (``fpr``) on x against true-positive rate
        (``tpr``) on y as a line per class.  Data must carry the schema emitted
        by ``ModelSource.roc_curve()``: ``fpr``, ``tpr``, ``threshold``,
        ``class``, ``auc``.  When ``reference_line=True`` the data is
        pre-sorted ascending by ``fpr`` before rendering.

        Parameters
        ----------
        average : str or None, optional
            When the data contains a macro/micro average row with
            ``class=average``, pass ``"macro"`` or ``"micro"`` to keep only
            that average line.  ``None`` (default) renders all classes.
        reference_line : bool, optional
            Whether to overlay a diagonal chance-level reference line
            (TPR=FPR).  Default is ``True``.
        annotate_auc : bool, optional
            Whether to annotate each curve with its AUC value.  Default is
            ``False``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for ROC rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.roc_curve()).mark_roc(annotate_auc=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_roc
        from ferrum._diagnostics.charts import _sort_by
        return self._set_composite_mark(
            "roc",
            desugar_roc,
            {
                "average": average,
                "reference_line": reference_line,
                "annotate_auc": annotate_auc,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=(
                (lambda df: _sort_by(df, "fpr")) if reference_line else None
            ),
        )

    def mark_pr(
        self,
        *,
        average: str | None = None,
        annotate_ap: bool = False,
        iso_lines: bool = False,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Precision-Recall (PR) curve.

        Plots recall on x against precision on y as a line per class.  Data
        must carry the schema emitted by ``ModelSource.pr_curve()``:
        ``precision``, ``recall``, ``threshold``, ``class``, ``ap``.

        Parameters
        ----------
        average : str or None, optional
            Filter to a specific average type row (e.g. ``"macro"``).
            ``None`` (default) renders all classes.
        annotate_ap : bool, optional
            Whether to annotate each curve with its average-precision (AP)
            value.  Default is ``False``.
        iso_lines : bool, optional
            Whether to draw iso-F1 reference lines in the background.
            Default is ``False``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for PR-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.pr_curve()).mark_pr(annotate_ap=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_pr
        return self._set_composite_mark(
            "pr",
            desugar_pr,
            {
                "average": average,
                "annotate_ap": annotate_ap,
                "iso_lines": iso_lines,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_calibration(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        reference_line: bool = True,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a calibration (reliability) curve.

        Plots mean predicted probability on x against fraction of positive
        outcomes (empirical probability) on y.  A perfectly calibrated model
        lies on the diagonal.  Data must carry the schema emitted by
        ``ModelSource.calibration_curve()``: ``mean_predicted``,
        ``fraction_positive``, ``count``.

        When ``reference_line=True`` the data is pre-sorted ascending by
        ``mean_predicted`` before rendering.

        Parameters
        ----------
        n_bins : int, optional
            Number of calibration bins.  Default is ``10``.
        strategy : {"uniform", "quantile"}, default "uniform"
            Binning strategy.  ``"uniform"`` uses equally-spaced bins;
            ``"quantile"`` uses equal-frequency bins.
        reference_line : bool, optional
            Whether to overlay a perfect-calibration diagonal reference line.
            Default is ``True``.
        color_field : str or None, optional
            Column name to drive per-group colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for calibration rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.calibration_curve()).mark_calibration(n_bins=15)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_calibration
        from ferrum._diagnostics.charts import _sort_by
        return self._set_composite_mark(
            "calibration",
            desugar_calibration,
            {
                "n_bins": n_bins,
                "strategy": strategy,
                "reference_line": reference_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=(
                (lambda df: _sort_by(df, "mean_predicted"))
                if reference_line
                else None
            ),
        )

    def mark_gain(
        self,
        *,
        reference_lines: bool = True,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a cumulative gain chart.

        Plots percent of population contacted on x against the cumulative gain
        (fraction of positive cases captured) on y.  Data must carry the schema
        emitted by ``ModelSource.cumulative_gain()``: ``percent_population``,
        ``gain``, ``class``.  The no-skill diagonal baseline is encoded as rows
        with ``class='baseline'``.

        Parameters
        ----------
        reference_lines : bool, optional
            Whether to draw the no-skill baseline diagonal.  Default is
            ``True``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for cumulative-gain rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.cumulative_gain()).mark_gain()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_gain
        return self._set_composite_mark(
            "gain",
            desugar_gain,
            {
                "reference_lines": reference_lines,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_lift(
        self,
        *,
        reference_line: bool = True,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a lift curve chart.

        Plots percent of population targeted on x against lift (ratio of
        positive-case density to baseline) on y.  Data must carry the schema
        emitted by ``ModelSource.lift_curve()``: ``percent_population``,
        ``lift``, ``class``.  The no-skill lift=1 baseline is encoded as rows
        with ``class='baseline'``.

        Parameters
        ----------
        reference_line : bool, optional
            Whether to draw the lift=1 baseline rule.  Default is ``True``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for lift-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.lift_curve()).mark_lift()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_lift
        return self._set_composite_mark(
            "lift",
            desugar_lift,
            {
                "reference_line": reference_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_discrimination_threshold(
        self,
        *,
        metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
        n_thresholds: int = 50,
        threshold_line: bool = False,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a discrimination-threshold sweep plot.

        Sweeps the decision threshold from 0 to 1 and plots multiple metrics
        (precision, recall, F1, queue rate) as lines against the threshold
        value.  Data must be in long form with columns ``threshold``, ``metric``,
        ``value`` — the figure builder handles unpivoting from
        ``ModelSource.discrimination_threshold()`` output.

        Parameters
        ----------
        metrics : tuple of str, optional
            Metric names to include.  Default is
            ``("precision", "recall", "f1", "queue_rate")``.
        n_thresholds : int, optional
            Number of evenly-spaced threshold steps to evaluate.  Default is
            ``50``.
        threshold_line : bool, optional
            Whether to draw a vertical rule at the optimal threshold.  Default
            is ``False``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for discrimination-threshold
            rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.discrimination_threshold()).mark_discrimination_threshold()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_discrimination_threshold
        return self._set_composite_mark(
            "discrimination_threshold",
            desugar_discrimination_threshold,
            {
                "metrics": metrics,
                "n_thresholds": n_thresholds,
                "threshold_line": threshold_line,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_confusion(
        self,
        *,
        normalize: str | None = None,
        annotate: bool = True,
        color_field: str = "value",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a confusion matrix as an annotated heatmap.

        Renders an ordinal heatmap with actual class on one axis and predicted
        class on the other.  Data must carry the long-form schema emitted by
        ``ModelSource.confusion_matrix()``: ``actual``, ``predicted``,
        ``value``, ``value_fmt``.  When ``annotate=True``, a second
        ``mark_text`` layer reads ``value_fmt`` for per-cell text labels.

        Parameters
        ----------
        normalize : {"true", "pred", "all"} or None, optional
            Normalise cell counts.  ``None`` (default) shows raw counts.
        annotate : bool, optional
            Whether to overlay per-cell count / percentage text.  Default is
            ``True``.
        color_field : str, optional
            Column name driving the heatmap colour scale.  Default is
            ``"value"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for confusion-matrix rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.confusion_matrix()).mark_confusion(normalize="true")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_confusion
        return self._set_composite_mark(
            "confusion",
            desugar_confusion,
            {
                "normalize": normalize,
                "annotate": annotate,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_class_prediction_error(
        self,
        *,
        normalize: bool = False,
        color_field: str = "actual",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a class prediction error bar chart.

        Renders one stacked bar per predicted class, with segments coloured by
        actual class.  This surfaces systematic mis-classification patterns not
        obvious in a confusion matrix.  Data must carry long-form columns
        ``(actual, predicted, value)`` — same shape as
        ``ModelSource.confusion_matrix(normalize=None)``.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalise each bar to 100%.  Default is ``False``.
        color_field : str, optional
            Column driving the segment colour.  Default is ``"actual"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for class-prediction-error
            rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.confusion_matrix()).mark_class_prediction_error(normalize=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_class_prediction_error
        return self._set_composite_mark(
            "class_prediction_error",
            desugar_class_prediction_error,
            {
                "normalize": normalize,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_importance(
        self,
        *,
        orient: str = "horizontal",
        error_bars: bool = True,
        top_k: int | None = None,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a feature-importance bar chart.

        Renders one bar per feature, sorted descending by importance.  When
        ``error_bars=True`` and the data carries ``imp_lower``/``imp_upper``
        columns a second errorbar layer is added.  Data must carry the schema
        emitted by ``ModelSource.importances()``: ``feature``, ``importance``,
        ``std``.  The chart builder computes the bound columns and truncates to
        ``top_k`` rows before calling this method.

        Parameters
        ----------
        orient : {"horizontal", "vertical"}, default "horizontal"
            Bar orientation.  ``"horizontal"`` places features on the y axis
            with importance on x (default); ``"vertical"`` swaps axes.
        error_bars : bool, optional
            Whether to draw error bars from ``imp_lower``/``imp_upper``.
            Default is ``True``.
        top_k : int or None, optional
            Limit results to the top-*k* features by importance.  Truncation
            is applied by the figure-level chart builder
            (``importance_chart``); when ``mark_importance`` is called
            directly this parameter is forwarded to the desugar function but
            the desugar layer treats it as informational — actual filtering
            must be done on the DataFrame before passing it to ``Chart``.
        color_field : str or None, optional
            Column name to drive per-feature colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for feature-importance rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.importances()).mark_importance(top_k=10)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_importance
        return self._set_composite_mark(
            "importance",
            desugar_importance,
            {
                "orient": orient,
                "error_bars": error_bars,
                "top_k": top_k,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_shap_beeswarm(
        self,
        *,
        max_display: int = 20,
        color_bar: bool = True,
        order: str = "abs_mean",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP beeswarm summary plot.

        Visualises the distribution of SHAP values across samples for each
        feature as a swarm of points, coloured by the feature's original value.
        Data must carry the long-form schema from ``ModelSource.shap_values()``
        pre-filtered to the top ``max_display`` features by the chart builder.

        Parameters
        ----------
        max_display : int, optional
            Maximum number of top features to show.  Default is ``20``.
        color_bar : bool, optional
            Whether to render a colour bar for the feature-value scale.
            Default is ``True``.
        order : {"abs_mean", "mean", "none"}, default "abs_mean"
            Feature ordering: by mean absolute SHAP (``"abs_mean"``), signed
            mean SHAP (``"mean"``), or original order (``"none"``).
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the point layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for SHAP beeswarm rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.shap_values()).mark_shap_beeswarm(max_display=15)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_shap_beeswarm
        return self._set_composite_mark(
            "shap_beeswarm",
            desugar_shap_beeswarm,
            {
                "max_display": max_display,
                "color_bar": color_bar,
                "order": order,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_shap_bar(
        self,
        *,
        max_display: int = 20,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP aggregated-bar feature importance chart.

        Shows mean absolute SHAP values per feature as a horizontal bar chart.
        Data must carry the long-form schema from ``ModelSource.shap_values()``
        pre-filtered to the top ``max_display`` features by the chart builder.

        Parameters
        ----------
        max_display : int, optional
            Maximum number of top features to show.  Default is ``20``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for SHAP bar rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.shap_values()).mark_shap_bar(max_display=10)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_shap_bar
        return self._set_composite_mark(
            "shap_bar",
            desugar_shap_bar,
            {"max_display": max_display, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_pdp(
        self,
        *,
        kind: str = "average",
        ice_alpha: float = 0.2,
        center: bool = False,
        color_field: str | None = "feature",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render partial-dependence plots (PDP / ICE).

        Visualises how the model's output varies as a function of one feature
        while marginalising over all others.  Data must carry the long-form
        schema from ``ModelSource.partial_dependence()``: ``feature``,
        ``feature_value``, ``pd_value``.  The chart builder pre-sorts ascending
        by ``feature_value``.

        Parameters
        ----------
        kind : {"average", "individual", "both"}, default "average"
            What to render.  ``"average"`` draws the mean PD line;
            ``"individual"`` draws ICE lines (one per sample);
            ``"both"`` overlays average on top of ICE lines.
        ice_alpha : float, optional
            Opacity of individual ICE lines when ``kind`` is ``"individual"``
            or ``"both"``.  Default is ``0.2``.
        center : bool, optional
            Whether to centre ICE lines at their first value (centred ICE).
            Default is ``False``.
        color_field : str or None, optional
            Column name driving per-feature colour.  Default is ``"feature"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for PDP / ICE rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.partial_dependence()).mark_pdp(kind="both", center=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_pdp
        return self._set_composite_mark(
            "pdp",
            desugar_pdp,
            {
                "kind": kind,
                "ice_alpha": ice_alpha,
                "center": center,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_shap_waterfall(
        self,
        *,
        sample_idx: int = -1,
        max_display: int = 20,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP waterfall chart for one sample.

        Shows how each feature pushes the model output from the base value
        toward the final prediction for a single observation.  Data must carry
        the long-form schema from ``ModelSource.shap_values()`` for the chosen
        ``sample_idx``.

        Parameters
        ----------
        sample_idx : int
            Row index of the sample to explain.  Must be provided explicitly;
            the default ``-1`` is a guard sentinel that raises ``ValueError``
            immediately so callers get a clear error at call time rather than
            at render time.
        max_display : int, optional
            Maximum number of features to show (smallest-magnitude features
            are collapsed into an ``"other"`` row).  Default is ``20``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for SHAP waterfall rendering.

        Raises
        ------
        ValueError
            If ``sample_idx`` is not provided (left at its default of ``-1``).

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.shap_values()).mark_shap_waterfall(sample_idx=0)
        Chart(mark='point', encoding=[])
        """
        if sample_idx == -1:
            raise ValueError(
                "mark_shap_waterfall requires sample_idx=<int>; "
                "pass an integer index (e.g. sample_idx=0) to select the sample to explain."
            )
        from ferrum.marks.diagnostic import desugar_shap_waterfall
        return self._set_composite_mark(
            "shap_waterfall",
            desugar_shap_waterfall,
            {
                "sample_idx": sample_idx,
                "max_display": max_display,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_learning_curve(
        self,
        *,
        ci_style: str = "band",
        color_field: str | None = "split",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a learning curve (train size vs. CV score).

        Plots training set size on x against CV score on y, with separate
        lines for training and validation splits.  Data must carry the schema
        emitted by ``ModelSource.learning_curve()``, pre-deduped per
        ``(train_size, split)`` by the chart builder.

        Parameters
        ----------
        ci_style : {"band", "bars", "none"}, default "band"
            How to display cross-validation variance.  ``"band"`` draws a
            shaded ribbon; ``"bars"`` draws error bars; ``"none"`` omits CI.
        color_field : str or None, optional
            Column name to drive per-split line colour.  Default is
            ``"split"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for learning-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_train, y_train)
        >>> fm.Chart(src.learning_curve()).mark_learning_curve(ci_style="band")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_learning_curve
        return self._set_composite_mark(
            "learning_curve",
            desugar_learning_curve,
            {"ci_style": ci_style, "color_field": color_field, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_validation_curve(
        self,
        *,
        log_scale: bool = False,
        ci_style: str = "band",
        color_field: str | None = "split",
        param_label: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a validation curve (hyperparameter vs. CV score).

        Plots a swept hyperparameter value on x against CV score on y, with
        separate lines for training and validation splits.  Data must carry the
        schema emitted by ``ModelSource.validation_curve()``, pre-deduped per
        ``(param_value, split)`` by the chart builder.

        Parameters
        ----------
        log_scale : bool, optional
            Whether to use a log scale on the x axis.  Useful for parameters
            like regularisation strength that span orders of magnitude.
            Default is ``False``.
        ci_style : {"band", "bars", "none"}, default "band"
            How to display cross-validation variance.
        color_field : str or None, optional
            Column name to drive per-split line colour.  Default is
            ``"split"``.
        param_label : str or None, optional
            Human-readable x-axis title for the hyperparameter being swept.
            The chart builder forwards the user's ``param`` argument here.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for validation-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_train, y_train)
        >>> fm.Chart(src.validation_curve(param="C")).mark_validation_curve(
        ...     log_scale=True, param_label="C"
        ... )
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_validation_curve
        return self._set_composite_mark(
            "validation_curve",
            desugar_validation_curve,
            {
                "log_scale": log_scale,
                "ci_style": ci_style,
                "color_field": color_field,
                "param_label": param_label,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_cv_scores(
        self,
        *,
        kind: str = "box",
        split: str = "both",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a per-fold cross-validation score summary.

        Shows the distribution of CV scores across folds as a box plot, strip
        plot, or bar chart.  Data must carry the schema emitted by
        ``ModelSource.cv_scores()``.  The chart builder pre-aggregates per
        split when ``kind="bar"`` and passes raw per-fold rows for ``"box"``
        or ``"strip"``.

        Parameters
        ----------
        kind : {"box", "strip", "bar"}, default "box"
            Summary plot type.
        split : {"train", "test", "both"}, default "both"
            Which CV split(s) to display.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to constituent layers.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for CV-score rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_train, y_train)
        >>> fm.Chart(src.cv_scores()).mark_cv_scores(kind="strip", split="test")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_cv_scores
        return self._set_composite_mark(
            "cv_scores",
            desugar_cv_scores,
            {"kind": kind, "split": split, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_alpha_selection(
        self,
        *,
        log_scale: bool = True,
        highlight_best: bool = True,
        ci_style: str = "band",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a regularisation-strength (alpha) selection curve.

        Sweeps the regularisation parameter ``alpha`` and plots CV score as a
        function of alpha, with variance bands.  When ``highlight_best=True``
        a vertical rule is drawn at the alpha that maximises ``mean_score``.
        Data must carry the schema emitted by ``ModelSource.alpha_selection()``:
        ``alpha``, ``mean_score``, ``score_lo``, ``score_hi``, ``split``.

        Parameters
        ----------
        log_scale : bool, optional
            Whether to use a log scale on the x axis.  Default is ``True``
            (regularisation parameters typically span orders of magnitude).
        highlight_best : bool, optional
            Whether to draw a vertical reference rule at the optimal alpha.
            Default is ``True``.
        ci_style : {"band", "bars", "none"}, default "band"
            How to display CV variance.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to constituent layers.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for alpha-selection rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(lasso, X_train, y_train)
        >>> fm.Chart(src.alpha_selection()).mark_alpha_selection(log_scale=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_alpha_selection
        from ferrum._diagnostics.charts import _inject_constant

        def _inject_best_alpha(df):
            import polars as pl
            if "alpha" not in df.columns or "mean_score" not in df.columns:
                return df
            agg = (
                df.group_by("alpha")
                .agg(pl.col("mean_score").first())
                .sort("mean_score", descending=True)
            )
            if agg.height == 0:
                return df
            return _inject_constant(df, "_best_alpha", float(agg["alpha"][0]))

        return self._set_composite_mark(
            "alpha_selection",
            desugar_alpha_selection,
            {
                "log_scale": log_scale,
                "highlight_best": highlight_best,
                "ci_style": ci_style,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_inject_best_alpha if highlight_best else None,
        )

    # ---- Marks (clustering / manifold — Phase 10f) ----

    def mark_silhouette(
        self,
        *,
        zero_line: bool = True,
        color_field: str | None = "cluster",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Rousseeuw silhouette plot.

        Displays one horizontal bar per sample whose width encodes its
        silhouette coefficient, coloured by cluster assignment.  Samples within
        each cluster are stacked vertically in descending coefficient order.
        Data must carry the schema emitted by ``ModelSource.silhouette()``:
        ``sample_id``, ``y_position``, ``cluster``, ``silhouette_value``.

        When ``zero_line=True`` a sentinel ``_ref_zero`` column is injected
        so the downstream ``mark_rule`` renders a single vertical rule at x=0.

        The method pre-computes ``_silhouette_x_lo``, ``_silhouette_x_hi``,
        ``_silhouette_y_lo``, and ``_silhouette_y_hi`` columns from the raw
        data so the renderer can draw rect marks directly.

        Parameters
        ----------
        zero_line : bool, optional
            Whether to draw a vertical reference rule at silhouette = 0.
            Default is ``True``.
        color_field : str or None, optional
            Column name driving per-cluster colour.  Default is ``"cluster"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for silhouette rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(kmeans, X)
        >>> fm.Chart(src.silhouette()).mark_silhouette()
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_silhouette
        from ferrum._diagnostics.charts import _inject_constant

        def _silhouette_prep(df):
            import polars as pl
            if (
                "y_position" not in df.columns
                or "silhouette_value" not in df.columns
            ):
                return df
            df = df.with_columns([
                pl.min_horizontal(pl.lit(0.0), pl.col("silhouette_value"))
                .alias("_silhouette_x_lo"),
                pl.max_horizontal(pl.lit(0.0), pl.col("silhouette_value"))
                .alias("_silhouette_x_hi"),
                (pl.col("y_position").cast(pl.Float64) - 0.5)
                .alias("_silhouette_y_lo"),
                (pl.col("y_position").cast(pl.Float64) + 0.5)
                .alias("_silhouette_y_hi"),
            ])
            if zero_line:
                df = _inject_constant(df, "_ref_zero", 0.0)
            return df

        return self._set_composite_mark(
            "silhouette",
            desugar_silhouette,
            {
                "zero_line": zero_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="rect",
            position=position,
            data_transform=_silhouette_prep,
        )

    def mark_pca_scree(
        self,
        *,
        cumulative_line: bool = True,
        threshold_line: float | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a PCA scree plot.

        Displays explained variance ratio per component as bars and optionally
        overlays the cumulative variance ratio as a line.  Data must carry the
        schema emitted by ``ModelSource.pca_variance()``: ``component``,
        ``explained_variance_ratio``, ``cumulative_variance_ratio``.

        When ``threshold_line`` is non-None a sentinel ``_threshold_line``
        column is injected so the downstream ``mark_rule`` draws a single
        horizontal reference line at the threshold value.

        Parameters
        ----------
        cumulative_line : bool, optional
            Whether to overlay a cumulative explained variance line.  Default
            is ``True``.
        threshold_line : float or None, optional
            Y-position of an optional horizontal threshold reference line
            (e.g. ``0.95`` for 95% explained variance).  ``None`` (default)
            omits the line.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for PCA scree rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(pca_model, X)
        >>> fm.Chart(src.pca_variance()).mark_pca_scree(threshold_line=0.95)
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import (
            desugar_pca_scree,
            desugar_pca_scree_with_threshold,
        )
        from ferrum._diagnostics.charts import _inject_constant

        def _pca_scree_prep(df):
            import polars as pl
            if (
                "component" not in df.columns
                or "explained_variance_ratio" not in df.columns
            ):
                return df
            df = df.with_columns([
                (pl.col("component").cast(pl.Float64) - 0.4)
                .alias("_pca_bar_x_lo"),
                (pl.col("component").cast(pl.Float64) + 0.4)
                .alias("_pca_bar_x_hi"),
                pl.lit(0.0).alias("_pca_bar_y_lo"),
                pl.col("explained_variance_ratio")
                .alias("_pca_bar_y_hi"),
            ])
            # X-axis anchor: same scale-extension trick as _y_axis_anchor — the
            # bar rects sit at component ± 0.4 but the line layer's x is the
            # bare component column. Without this, bars at the first/last
            # component get clipped by the narrower x domain.
            n_anchor = df.height
            x_lo = float(df["_pca_bar_x_lo"].min() or 0.0)
            x_hi = float(df["_pca_bar_x_hi"].max() or 0.0)
            x_anchor_vals = [x_lo, x_hi] + [None] * max(0, n_anchor - 2)
            df = df.with_columns(
                pl.Series(
                    "_x_axis_anchor", x_anchor_vals[:n_anchor],
                    dtype=pl.Float64,
                ),
            )
            # Scale-resolution anchor: render/prepare.rs:265 feeds layer-0's
            # y+y2 into the y-axis domain computation. Layer-0 here is the
            # cumulative line (y range ≈ [evr[0], sum(evr)]); the bar baseline
            # at 0 and any threshold rule (0.95 default) sit outside that
            # range. Stash both anchor values into a y2 column on layer-0 so
            # the scale union covers [0, threshold].
            anchor_hi = max(
                float(threshold_line) if threshold_line is not None else 0.0,
                float(df["cumulative_variance_ratio"].max() or 0.0),
            )
            n = df.height
            anchor_vals = [0.0, float(anchor_hi)] + [None] * max(0, n - 2)
            df = df.with_columns(
                pl.Series(
                    "_y_axis_anchor", anchor_vals[:n],
                    dtype=pl.Float64,
                ),
            )
            if threshold_line is not None:
                df = _inject_constant(
                    df, "_threshold_line", float(threshold_line),
                )
            return df

        fn = (
            desugar_pca_scree_with_threshold
            if threshold_line is not None
            else desugar_pca_scree
        )
        return self._set_composite_mark(
            "pca_scree",
            fn,
            {"cumulative_line": cumulative_line, **mark_kwargs},
            placeholder="rect",
            position=position,
            data_transform=_pca_scree_prep,
        )

    def mark_intercluster_distance(
        self,
        *,
        label_clusters: bool = True,
        color_field: str | None = "cluster",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a 2-D intercluster distance (MDS) plot.

        Embeds cluster centres into 2-D using MDS and visualises each centre
        as a point whose area encodes cluster size.  With ``label_clusters=True``
        a ``mark_text`` overlay labels each point by its cluster id.  Data must
        carry the schema emitted by ``ModelSource.intercluster_distance()``:
        ``cluster``, ``x``, ``y``, ``size``.

        Parameters
        ----------
        label_clusters : bool, optional
            Whether to overlay cluster-id text labels.  Default is ``True``.
        color_field : str or None, optional
            Column name driving per-cluster colour.  Default is ``"cluster"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the point layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for intercluster-distance
            rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(kmeans, X)
        >>> fm.Chart(src.intercluster_distance()).mark_intercluster_distance()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_intercluster_distance
        return self._set_composite_mark(
            "intercluster_distance",
            desugar_intercluster_distance,
            {
                "label_clusters": label_clusters,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_decision_boundary(
        self,
        *,
        proba: bool = False,
        color_field: str = "z",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a decision-boundary heatmap.

        Colours a pixel grid by the model's predicted class (``proba=False``)
        or class probability (``proba=True``) at each grid point.  Data must
        carry pre-computed cell bound columns ``x``, ``x2``, ``y``, ``y2``
        and a prediction column ``z``.  The chart builder helper
        ``_decision_boundary_chart_from_source`` produces these columns from a
        ``ModelSource``.

        Parameters
        ----------
        proba : bool, optional
            Whether to colour by predicted probability rather than class index.
            Default is ``False``.
        color_field : str, optional
            Column name for the colour encoding.  Default is ``"z"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for decision-boundary rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.decision_boundary()).mark_decision_boundary(proba=True)
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_decision_boundary
        return self._set_composite_mark(
            "decision_boundary",
            desugar_decision_boundary,
            {
                "proba": proba,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="rect",
            position=position,
        )

    def mark_rank1d(
        self,
        *,
        orient: str = "horizontal",
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a univariate feature-ranking bar chart.

        Ranks features by a univariate score (e.g. mutual information, ANOVA
        F-statistic) and displays them as a bar chart sorted by rank.  Data must
        carry the schema emitted by ``ModelSource.rank1d()``: ``feature``,
        ``score``, ``rank``.

        Parameters
        ----------
        orient : {"horizontal", "vertical"}, default "horizontal"
            Bar orientation.  ``"horizontal"`` places features on the y axis
            with score on x; ``"vertical"`` places features on x.
        color_field : str or None, optional
            Column name driving per-feature colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for rank-1D rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.rank1d()).mark_rank1d()
        Chart(mark='bar', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_rank1d
        return self._set_composite_mark(
            "rank1d",
            desugar_rank1d,
            {
                "orient": orient,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="bar",
            position=position,
        )

    def mark_rank2d(
        self,
        *,
        annot: bool = True,
        color_field: str = "correlation",
        text_field: str = "correlation_fmt",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a pairwise feature-ranking correlation heatmap.

        Displays pairwise feature correlation scores as a colour-coded matrix.
        When ``annot=True`` a text overlay renders each cell's value to 2 dp.
        Data must carry the schema emitted by ``ModelSource.rank2d()``:
        ``feature_x``, ``feature_y``, ``correlation``.  The chart builder
        appends a ``correlation_fmt`` (Utf8) column when ``annot=True``.

        Parameters
        ----------
        annot : bool, optional
            Whether to overlay per-cell correlation value text.  Default is
            ``True``.
        color_field : str, optional
            Column driving the heatmap colour scale.  Default is
            ``"correlation"``.
        text_field : str, optional
            Column read by the text layer.  Default is ``"correlation_fmt"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for rank-2D rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.rank2d()).mark_rank2d(annot=True)
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_rank2d
        return self._set_composite_mark(
            "rank2d",
            desugar_rank2d,
            {
                "annot": annot,
                "color_field": color_field,
                "text_field": text_field,
                **mark_kwargs,
            },
            placeholder="rect",
            position=position,
        )

    def mark_parallel_coordinates(
        self,
        *,
        alpha: float = 0.5,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a parallel coordinates plot.

        Draws one polyline per sample across normalised feature axes.  Data
        must carry the long-form schema produced by
        ``_parallel_coords_chart_from_dataframe``: ``feature`` (Utf8),
        ``value`` (Float64), ``sample_id`` (Utf8), and (optionally) a hue
        column passed via ``color_field``.  The line layer uses
        ``mark_style.detail = "sample_id"`` so each sample renders as its own
        polyline.

        Parameters
        ----------
        alpha : float, optional
            Line opacity.  Default is ``0.5``.
        color_field : str or None, optional
            Column name driving per-sample (or per-class) line colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for parallel-coordinates rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"feature": ["a","a","b","b"],
        ...                    "value": [0.5, 0.3, 0.8, 0.2],
        ...                    "sample_id": ["s0", "s1", "s0", "s1"]})
        >>> fm.Chart(df).mark_parallel_coordinates(color_field="sample_id")
        Chart(mark='line', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_parallel_coordinates
        return self._set_composite_mark(
            "parallel_coordinates",
            desugar_parallel_coordinates,
            {
                "alpha": alpha,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="line",
            position=position,
        )

    def mark_arc(self, **kwargs):
        """Render data as arcs (pie/donut slices).

        .. note::
            Not yet implemented.  Calling this method raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always — deferred to a future phase.
        """
        raise deferred_mark_error("arc")

    def mark_image(self, **kwargs):
        """Render data as raster images.

        .. note::
            Not yet implemented.  Calling this method raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always — deferred to a future phase.
        """
        raise deferred_mark_error("image")

    def mark_geoshape(self, **kwargs):
        """Render geographic shape data.

        .. note::
            Not yet implemented.  Calling this method raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always — deferred to a future phase.
        """
        raise deferred_mark_error("geoshape")
    def mark_segment(self, *, position=None, **kwargs) -> "Chart":
        """Diagonal line segment from (x, y) to (x2, y2).

        Distinct from ``mark_rule`` (axis-aligned only); segments may take any
        direction. Requires ``x``, ``y``, ``x2``, ``y2`` on the encoding.
        """
        return self._set_mark("segment", position=position, **kwargs)
    def mark_label(self, **kwargs):
        """Render smart text labels with automatic collision avoidance.

        .. note::
            Not yet implemented.  Calling this method raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always — deferred to a future phase.
        """
        raise deferred_mark_error("label")

    # ---- Encoding ----

    def encode(self, **channels: Any) -> "Chart":
        """Set or update encoding channels on this chart.

        Each keyword argument maps a channel name to a field shorthand string
        (e.g. ``"species:N"``) or an explicit channel object (e.g.
        ``fm.X("sepal_length", type="Q")``).

        Parameters
        ----------
        x : str or X
            Field mapped to the x position.  Shorthand format:
            ``"field"`` (type inferred), ``"field:Q"`` (quantitative),
            ``"field:N"`` (nominal), ``"field:O"`` (ordinal),
            ``"field:T"`` (temporal), or ``"agg(field):Q"`` (aggregation).
        y : str or Y
            Field mapped to the y position.
        x2 : str or X2, optional
            Secondary x position (band / segment end).
        y2 : str or Y2, optional
            Secondary y position.
        color : str or Color, optional
            Field or value driving mark colour.
        fill : str or Fill, optional
            Field or value driving mark fill colour.
        stroke : str or Stroke, optional
            Field or value driving mark stroke colour.
        size : str or Size, optional
            Field or value driving mark size.
        shape : str or Shape, optional
            Field or value driving mark shape.
        opacity : str or Opacity, optional
            Field or value driving mark opacity.
        text : str or Text, optional
            Field rendered as text labels (used with ``mark_text``).
        detail : str or Detail, optional
            Additional grouping field that does not map to any visual property.
        tooltip : str or Tooltip, optional
            Field shown on hover.
        **channels
            Any other valid channel names (``x_error``, ``y_error``, ``theta``,
            ``radius``, etc.).

        Returns
        -------
        Chart
            New ``Chart`` with updated encoding channels.

        Raises
        ------
        ValueError
            If an unknown channel name is passed.
        TypeError
            If a value is not a string, channel instance, or Repeat placeholder.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "b", "a"]})
        >>> fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="c:N")
        Chart(mark='point', encoding=['x', 'y', 'color'])
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

        Transforms are executed in order by the Rust engine before rendering.
        Multiple calls to ``transform()`` accumulate — each appends to the
        existing pipeline.

        Parameters
        ----------
        *transforms
            One or more transform objects (e.g. ``fm.Filter(...)``,
            ``fm.Aggregate(...)``, ``fm.Sort(...)``, ``fm.Window(...)``).
            Accepts any object that serialises to a valid ``TransformSpec`` JSON
            shape.

        Returns
        -------
        Chart
            New ``Chart`` with the additional transforms appended.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").transform(
        ...     fm.Filter("datum.x > 1")
        ... )
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        new._transforms = list(self._transforms) + list(transforms)
        return new

    # ---- Composition operators ----

    def __add__(self, other: "Chart") -> "Chart":
        """Overlay two charts as a multi-layer composite.

        When both sides share the same data (identity check or Arrow value
        equality), produces a single multi-layer ``Chart`` that renders all
        layers within one plot area with shared x/y scales.

        When data differs, falls back to horizontal concatenation (same
        behaviour as ``__or__``) and emits a ``UserWarning`` so the caller
        knows the result is side-by-side rather than overlaid.

        Parameters
        ----------
        other : Chart
            The chart to overlay on top of this one.

        Returns
        -------
        Chart or HConcatChart
            Multi-layer ``Chart`` when data matches; ``HConcatChart`` otherwise.

        Raises
        ------
        TypeError
            (via ``NotImplemented``) if ``other`` is not a ``Chart``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
        >>> line = fm.Chart(df).mark_line().encode(x="x", y="y")
        >>> layered = scatter + line
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
                "Combine the two DataFrames into one with null padding "
                "(see decision_boundary_chart for an example) to get a true overlay.",
                UserWarning,
                stacklevel=2,
            )
            return self.__or__(other)

        # Same data — build a multi-layer chart.
        # Resolve pending statistical marks before snapshotting encoding dicts.
        lhs = self._resolve_pending()
        rhs = other._resolve_pending()
        new = lhs._clone()

        def _expand(c):
            """Return (layer_dicts, top_level_transforms) for one side of `+`.

            Composite-mark charts arrive pre-layered (c._layers is not None,
            c._mark is None) — splat their layers as-is and carry their
            top-level transforms across. Plain single-mark charts wrap into a
            one-element list with the chart's own mark/encoding/etc.
            """
            if c._layers is not None:
                return list(c._layers), list(c._transforms or [])
            return [{
                "mark": c._mark,
                "encoding": dict(c._encoding),
                "transforms": list(c._transforms),
                "mark_style": dict(c._mark_kwargs),
                "position": c._position,
            }], []

        lhs_layers, _ = _expand(lhs)  # lhs top-level xforms already in `new` via _clone()
        rhs_layers, rhs_top_xforms = _expand(rhs)
        new._layers = lhs_layers + rhs_layers
        # Merge RHS top-level transforms (e.g. composite-mark expansion produced
        # them) into the combined chart's top-level pipeline, deduping by
        # identity to avoid re-running a transform shared across layers.
        existing_ids = {id(t) for t in (new._transforms or [])}
        for t in rhs_top_xforms:
            if id(t) not in existing_ids:
                new._transforms = list(new._transforms or []) + [t]
                existing_ids.add(id(t))
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
        """Place two charts side-by-side (horizontal concatenation).

        Produces an ``HConcatChart`` that renders both charts at the same height
        in adjacent columns.  Use ``|`` to build multi-panel layouts.

        Parameters
        ----------
        other : Chart
            The chart to place to the right.

        Returns
        -------
        HConcatChart
            Horizontally concatenated composite chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> left = fm.Chart(df).mark_point().encode(x="x", y="y")
        >>> right = fm.Chart(df).mark_bar().encode(x="x", y="y")
        >>> left | right
        HConcatChart(n=2)
        """
        from ferrum.composition import HConcatChart
        return HConcatChart([self, other])

    def __and__(self, other: "Chart") -> "VConcatChart":
        """Stack two charts vertically (vertical concatenation).

        Produces a ``VConcatChart`` that renders both charts stacked on top of
        each other in the same column.  Use ``&`` to build vertically-stacked
        panel layouts.

        Parameters
        ----------
        other : Chart
            The chart to place below.

        Returns
        -------
        VConcatChart
            Vertically concatenated composite chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> top = fm.Chart(df).mark_point().encode(x="x", y="y")
        >>> bottom = fm.Chart(df).mark_bar().encode(x="x", y="y")
        >>> top & bottom
        VConcatChart(n=2)
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
        """Facet this chart into small multiples by a field.

        Two modes are supported:

        - **Wrap** — a single field is wrapped across columns (rows auto).
          Pass ``field=`` or equivalently ``col=`` alone.
        - **Grid** — two fields define row × column layout.  Pass both
          ``row=`` and ``col=``.

        Parameters
        ----------
        field : str, optional
            Column name used for wrap-mode faceting.  Mutually exclusive with
            using both ``row`` and ``col`` together.
        row : str, optional
            Column name for the row dimension (grid mode).  When used alone,
            behaves as wrap mode on the row axis.
        col : str, optional
            Column name for the column dimension (wrap or grid mode).
        ncols : int or None, optional
            Maximum number of columns in wrap mode.  ``None`` lets the renderer
            choose.
        nrows : int or None, optional
            Maximum number of rows.  ``None`` lets the renderer choose.

        Returns
        -------
        Chart
            New ``Chart`` with faceting configured.

        Raises
        ------
        ValueError
            If neither ``field``, ``row``, nor ``col`` is provided.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1,2,3]*3, "y": [4,5,6]*3,
        ...                    "g": ["a","a","b","b","c","c","a","b","c"]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").facet(col="g", ncols=2)
        Chart(mark='point', encoding=['x', 'y'])
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
        """Attach a ``Theme`` to this chart, overriding the process-level default.

        Per-chart theme always wins over ``ferrum.set_default_theme()``.
        Theme objects are immutable value classes — modifying the original
        ``Theme`` after calling ``.theme()`` has no effect on the chart.

        Parameters
        ----------
        theme : Theme
            A ``ferrum.Theme`` instance (e.g. ``fm.themes.dark()``,
            ``fm.Theme(background="white", font="serif")``).

        Returns
        -------
        Chart
            New ``Chart`` with the specified theme attached.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> dark = fm.themes.dark()
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").theme(dark)
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        new._theme = theme
        return new

    def axis(
        self,
        *,
        x: Optional[bool] = None,
        y: Optional[bool] = None,
        show: Optional[bool] = None,
    ) -> "Chart":
        """Suppress (or restore) the chart's x/y axis decorations.

        Spec-level axis suppression — when ``x=False`` (or ``y=False``), the
        corresponding axis line, ticks, tick labels, and axis title are
        omitted at layout time. The plot area's pixel rect is unchanged
        (gutters reserved for axis decorations are preserved), so this
        method is intended for sub-charts whose axes are shared with a
        neighbouring chart in a compound view: clustermap dendrograms,
        JointChart marginals, RepeatChart off-diagonal panels.

        Parameters
        ----------
        x : bool, optional
            ``False`` hides the x axis; ``True`` shows it; ``None`` leaves
            the current setting (default visible).
        y : bool, optional
            ``False`` hides the y axis; ``True`` shows it; ``None`` leaves
            the current setting (default visible).
        show : bool, optional
            Shorthand for ``axis(x=show, y=show)``. Mutually exclusive with
            per-axis ``x``/``y`` arguments.

        Returns
        -------
        Chart
            New ``Chart`` with the requested axis-visibility settings.

        Raises
        ------
        ValueError
            If ``show`` is combined with ``x`` or ``y``.

        Examples
        --------
        Hide both axes (e.g. for a dendrogram panel):

        >>> chart.axis(show=False)

        Hide just the x axis (top marginal of a JointChart):

        >>> chart.axis(x=False)
        """
        if show is not None and (x is not None or y is not None):
            raise ValueError(
                "Chart.axis: pass either show=… OR x=/y=, not both"
            )
        new = self._clone()
        if show is not None:
            new._axis_x = show
            new._axis_y = show
        else:
            if x is not None:
                new._axis_x = x
            if y is not None:
                new._axis_y = y
        return new

    def coord(self, coord: Any) -> "Chart":
        """Set the coordinate system for this chart.

        Currently only ``CoordFlip`` is supported (swaps x and y axes).

        Parameters
        ----------
        coord : CoordFlip
            A coordinate-system object.  Pass ``fm.CoordFlip()`` to swap the
            horizontal and vertical axes.

        Returns
        -------
        Chart
            New ``Chart`` with the coordinate system set.

        Raises
        ------
        TypeError
            If ``coord`` is not a supported coordinate-system type.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
        >>> fm.Chart(df).mark_bar().encode(x="cat", y="val").coord(fm.CoordFlip())
        Chart(mark='bar', encoding=['x', 'y'])
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
            for axis in ("x", "y", "x2", "y2", "color", "size", "shape", "opacity", "text"):
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
                    for opt_key in ("title", "aggregate", "scheme", "format", "formatType", "scale"):
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

        Only the keyword arguments that are explicitly provided are updated;
        unset properties inherit from the existing chart.

        Parameters
        ----------
        width : int or "container" or None, optional
            Chart width in pixels, or ``"container"`` to fill the parent.
        height : int or "container" or None, optional
            Chart height in pixels.
        title : str or None, optional
            Chart title rendered above the plot area.
        description : str or None, optional
            Accessible description attached to the SVG root.

        Returns
        -------
        Chart
            New ``Chart`` with the specified properties updated.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").properties(
        ...     width=400, height=300, title="My Chart"
        ... )
        Chart(mark='point', encoding=['x', 'y'])
        """
        new = self._clone()
        if width is not None: new._width = width
        if height is not None: new._height = height
        if title is not None: new._title = title
        if description is not None: new._description = description
        return new

    # ---- Spec output ----

    def to_spec(self):
        """Build the Rust ``ChartSpec`` for this chart.

        Resolves any pending statistical-mark desugar, converts Python encoding
        channel objects to ``EncodingSpec`` instances, and constructs the
        ``ChartSpec`` PyO3 object that the Rust renderer consumes.

        Returns
        -------
        ChartSpec
            The fully-resolved ``ferrum._core.ChartSpec`` for this chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> spec = fm.Chart(df).mark_point().encode(x="x", y="y").to_spec()
        >>> spec.mark
        'point'
        """
        # Resolve any pending statistical mark desugar (mark called before encode).
        resolved = self._resolve_pending()
        from ferrum import ChartSpec, EncodingSpec
        # Build full EncodingSpec instances per channel so honored kwargs
        # (scale, title) and deferred kwargs (axis, legend, sort, ...) flow to Rust.
        # Phase 7 + 8a's ChartSpec(...) accepts EncodingSpec instances or strings.
        kw = {"mark": resolved._mark or "point", "data": "default"}
        from ferrum.repeat import _RepeatPlaceholder
        for axis in ("x", "y", "x2", "y2", "color", "size", "shape", "opacity", "text"):
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
        if resolved._title is not None:
            kw["title"] = resolved._title
        if resolved._axis_x is not None:
            kw["axis_x"] = resolved._axis_x
        if resolved._axis_y is not None:
            kw["axis_y"] = resolved._axis_y
        return ChartSpec(**kw)

    def _build_spec(self):
        """Build the chart spec for callers that want typed Python access to
        layers (composite-mark tests, future internal renderer wiring).

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
        """Serialise the chart specification to a JSON string.

        Calls ``to_spec()`` to build the ``ChartSpec`` and then serialises it
        via the Rust ``serde_json`` encoder.  When ``indent`` is given the
        compact JSON is reformatted via ``json.loads`` / ``json.dumps`` on the
        Python side (the Rust encoder always produces compact output).

        Parameters
        ----------
        indent : int or None, optional
            Number of spaces to use for pretty-printing.  ``None`` (default)
            returns compact single-line JSON.

        Returns
        -------
        str
            JSON representation of the chart specification.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1], "y": [2]})
        >>> spec_json = fm.Chart(df).mark_point().encode(x="x", y="y").to_json()
        >>> '"mark"' in spec_json
        True
        """
        import json as _json
        spec = self.to_spec()
        compact = spec.to_json()
        if indent is None:
            return compact
        return _json.dumps(_json.loads(compact), indent=indent)

    def show_svg(self) -> str:
        """Render the chart to an SVG string.

        Calls the Rust ``render_svg`` engine with the chart's spec, data, viewport,
        and theme.  The returned string is a complete ``<svg>…</svg>`` document.

        Returns
        -------
        str
            SVG markup for the chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        >>> svg.startswith("<svg")
        True
        """
        # Stub — full impl in Task 32
        from ferrum._core import render_svg
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_svg(spec, data, viewport=viewport, theme=theme_dict)

    def show_png(self) -> bytes:
        """Render the chart to PNG bytes.

        Calls the Rust ``render_png`` engine (SVG → PNG rasterisation via
        ``resvg``).  Returns raw PNG bytes that can be written to a file or
        displayed in a Jupyter notebook via ``IPython.display.Image``.

        Returns
        -------
        bytes
            PNG-encoded image data.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> png = fm.Chart(df).mark_point().encode(x="x", y="y").show_png()
        >>> png[:4] == b'\\x89PNG'
        True
        """
        from ferrum._core import render_png
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_png(spec, data, viewport=viewport, theme=theme_dict)

    def save(self, path, *, format=None, **render_kwargs) -> None:
        """Save the chart to a file on disk.

        Delegates to ``ferrum.display.save_chart``.  The file format is
        inferred from the file extension when ``format`` is not given.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path.  Extension determines the default format:
            ``.svg`` → SVG, ``.png`` → PNG.
        format : {"svg", "png"} or None, optional
            Explicit format override.  ``None`` (default) infers from ``path``.
        **render_kwargs
            Additional keyword arguments forwarded to the underlying renderer
            (e.g. ``width``, ``height`` viewport overrides).

        Returns
        -------
        None

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").save("/tmp/chart.svg")
        """
        from ferrum.display import save_chart
        save_chart(self, path, format=format, **render_kwargs)

    def show(self) -> None:
        """Display the chart inline or in a browser.

        In a Jupyter notebook the SVG is rendered inline via
        ``_repr_svg_``.  Outside of a notebook, the SVG is written to a
        temporary file and opened in the system browser via
        ``ferrum.display.show_chart``.

        Returns
        -------
        None

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").show()  # doctest: +SKIP
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
    def add_selection(self, *selections):
        """Add interactive selection(s) to this chart.

        .. note::
            Not yet implemented.  Interactive selections require the Phase 11
            renderer.  Calling this method raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always — deferred to Phase 11.
        """
        raise NotImplementedError("selections require .interactive() — Phase 11")

    def interactive(self):
        """Enable interactive rendering for this chart.

        .. note::
            Not yet implemented.  Calling this method raises
            ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always — deferred to Phase 11.
        """
        raise NotImplementedError("interactive renderer — Phase 11")

    def __repr__(self) -> str:
        """Return a concise string representation of the chart.

        Returns
        -------
        str
            A string of the form ``Chart(mark='point', encoding=['x', 'y'])``.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> repr(fm.Chart(df).mark_point().encode(x="x", y="y"))
        "Chart(mark='point', encoding=['x', 'y'])"
        """
        return f"Chart(mark={self._mark!r}, encoding={list(self._encoding.keys())})"
