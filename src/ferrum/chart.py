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
    """Top-level chart value class. Immutable — every method returns a new Chart."""

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

    def mark_point(self, **kwargs):  return self._set_mark("point", **kwargs)
    def mark_line(self, **kwargs):   return self._set_mark("line", **kwargs)
    def mark_bar(self, **kwargs):    return self._set_mark("bar", **kwargs)
    def mark_area(self, **kwargs):   return self._set_mark("area", **kwargs)
    def mark_rule(self, **kwargs):   return self._set_mark("rule", **kwargs)
    def mark_text(self, **kwargs):   return self._set_mark("text", **kwargs)
    def mark_tick(self, **kwargs):   return self._set_mark("tick", **kwargs)
    def mark_rect(self, **kwargs):   return self._set_mark("rect", **kwargs)

    # ---- Marks (statistical) ----

    def mark_density(self, *, position=None, **kwargs) -> "Chart":
        """Density plot. 1D KDE when only x is encoded; bivariate (filled
        contour over 2D KDE — Phase 8b) when both x and y are encoded.
        Can be called before or after ``.encode()``.
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
        """Histogram. Can be called before or after .encode(x=...)."""
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
        """Smooth/regression line. Can be called before or after .encode(x=..., y=...).

        With ``ci=`` set, emits a layered ribbon (CI band) + line (Phase 8b).
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
        """Composite boxplot. Desugars to box+whisker+median (+optional outlier) layers."""
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
        """Letter-value (boxen) plot. Composite mark — desugars to nested rect
        bands per letter-value depth, plus a median rule and an outlier-point
        layer, via the ``LetterValue`` transform.
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
        """Errorbar mark via ErrorExtent transform."""
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
        """Errorband mark (ribbon) via ErrorExtent transform."""
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
        """Ribbon mark — fills closed area between y and y2 along x. Requires y2 in encoding."""
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
        """Contour plot via Kde2D + Contour transforms."""
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
        """Violin plot via Violin transform; optional inner box/quartile/point overlay."""
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
        """QQ plot. Reads `field` from x encoding (single-column input)."""
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
        """2D raster (heatmap) via Raster transform."""
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
        """Hexagonal binning via Hex transform."""
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
        """Beeswarm plot via Swarm transform."""
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
        """Function plot. Materializes synthetic data via fn(xs)."""
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

    def mark_arc(self, **kwargs):           raise deferred_mark_error("arc")
    def mark_image(self, **kwargs):         raise deferred_mark_error("image")
    def mark_geoshape(self, **kwargs):      raise deferred_mark_error("geoshape")
    def mark_segment(self, *, position=None, **kwargs) -> "Chart":
        """Diagonal line segment from (x, y) to (x2, y2).

        Distinct from ``mark_rule`` (axis-aligned only); segments may take any
        direction. Requires ``x``, ``y``, ``x2``, ``y2`` on the encoding.
        """
        if position is not None:
            from ferrum.position import validate_position_eligibility
            validate_position_eligibility("segment", position)
        new = self._set_mark("segment", **kwargs)
        new._position = position
        return new
    def mark_label(self, **kwargs):         raise deferred_mark_error("label")

    # ---- Encoding ----

    def encode(self, **channels: Any) -> "Chart":
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
        new = self._clone()
        new._transforms = list(self._transforms) + list(transforms)
        return new

    # ---- Composition operators ----

    def __add__(self, other: "Chart") -> "Chart":
        """Overlay two charts.

        If both charts share the same data object (identity check) or equivalent
        data (value equality via Arrow), produces a multi-layer ``Chart``.
        If the data differs, falls through to an ``HConcatChart`` with a
        ``UserWarning``.
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
        """Horizontal concatenation: ``chart1 | chart2``."""
        from ferrum.composition import HConcatChart
        return HConcatChart([self, other])

    def __and__(self, other: "Chart") -> "VConcatChart":
        """Vertical concatenation: ``chart1 & chart2``."""
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
        """Set faceting on this chart.

        Single-field wrap mode: ``facet(field="col")`` or ``facet(col="col")``.
        Grid mode: ``facet(row="year", col="species")``.
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
        """Attach a Theme to this chart instance (overrides the process default)."""
        new = self._clone()
        new._theme = theme
        return new

    def coord(self, coord: Any) -> "Chart":
        """Set the coordinate system. Only CoordFlip is supported in Phase 8a."""
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
        new = self._clone()
        if width is not None: new._width = width
        if height is not None: new._height = height
        if title is not None: new._title = title
        if description is not None: new._description = description
        return new

    # ---- Spec output ----

    def to_spec(self):
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
        spec = self.to_spec()
        return spec.to_json()

    def show_svg(self) -> str:
        # Stub — full impl in Task 32
        from ferrum._core import render_svg
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_svg(spec, data, viewport=viewport, theme=theme_dict)

    def show_png(self) -> bytes:
        from ferrum._core import render_png
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_png(spec, data, viewport=viewport, theme=theme_dict)

    def save(self, path, *, format=None, **render_kwargs) -> None:
        """Save chart to disk (svg or png). Format inferred from extension."""
        from ferrum.display import save_chart
        save_chart(self, path, format=format, **render_kwargs)

    def show(self) -> None:
        """Display chart: Jupyter inline SVG or browser fallback."""
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
        raise NotImplementedError("selections require .interactive() — Phase 11")

    def interactive(self):
        raise NotImplementedError("interactive renderer — Phase 11")

    def __repr__(self) -> str:
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
