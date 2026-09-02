"""Rendering mixin — extracted from chart.py to reduce file size.

The ``_RenderMixin`` class provides rendering, display, and auto-raster
methods that ``Chart`` inherits via mixin.  All methods operate through
``self`` (which is a ``Chart`` instance at runtime) and access ``Chart``
slots like ``_render_config``, ``_mark``, ``_data``, ``_encoding``, etc.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from ferrum._coerce import normalize_for_rust, to_arrow_table
from ferrum._render_prepare import (
    _apply_label_maps,
    _collect_label_maps,
    _resolve_category_coords_in_annotations,
)
from ferrum.encoding.base import ChannelBase

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


def _sanitize_for_rust(tbl: "pyarrow.Table") -> "pyarrow.Table":
    """Compatibility shim — delegates to ``ferrum._coerce.normalize_for_rust``.

    .. deprecated::
        Import ``normalize_for_rust`` from ``ferrum._coerce`` directly.
        This shim exists only so that existing tests that imported
        ``_sanitize_for_rust`` from ``ferrum._render`` continue to work
        without modification.
    """
    return normalize_for_rust(tbl)


def _warn_large_chart(mark_count: int) -> None:
    """Emit a guidance warning when a chart has many marks but is ineligible for auto-raster."""
    warnings.warn(
        f"Chart has {mark_count:,} marks which may produce large output. "
        f"Use `.mark_raster()` for efficient rendering, or set "
        f"`raster=False` to suppress this warning.",
        UserWarning,
        stacklevel=5,
    )


class _RenderMixin:
    """Rendering, display, and auto-raster methods for ``Chart``.

    Mixed into ``Chart`` so that ``chart.py`` stays focused on
    declaration, encoding, mark, and composition logic.
    """

    # Mark types eligible for automatic raster substitution.
    _AUTO_RASTER_ELIGIBLE_MARKS = frozenset(["point", "bar", "rect", "tick", "rule", "segment"])

    def _apply_auto_raster(self) -> "Chart":
        """Return *self* unchanged, or a substituted chart with ``mark_raster``.

        Called between ``_resolve_pending()`` and ``to_spec()`` inside
        ``_render_inputs()``.  When the mark count exceeds
        ``RenderConfig.raster_threshold`` and the mark type is eligible,
        the chart is transparently replaced with a ``mark_raster`` equivalent
        so the SVG stays compact.

        The substitution **will not fire** when:

        - ``raster_threshold`` is ``None`` (auto-raster disabled).
        - The mark count is below the threshold.
        - The mark type is not a per-element type (line, area, hex, raster,
          image, etc. are excluded).
        - The chart was produced by a composite/statistical mark (histogram,
          density, hex, etc.) where the row count does not reflect the
          actual SVG element count.
        - The chart has an active ``color`` encoding (rasterising would
          silently discard categorical information).
        - Both ``x`` and ``y`` quantitative encodings are not present.

        When the chart is over-threshold but ineligible, a guidance warning
        is emitted suggesting ``mark_raster()`` or ``raster=False``.
        """
        from ferrum.render_config import _RASTER_AGGREGATES_NEEDING_FIELD, RenderConfig

        cfg = self._render_config or RenderConfig()

        # Disabled?
        if cfg.raster_threshold is None:
            return self

        threshold = cfg.raster_threshold
        behavior = cfg.raster_behavior

        # Count marks -- row count of the resolved data.
        if self._data is None:
            return self
        mark_count = len(self._data)
        if mark_count < threshold:
            return self

        # Check mark type eligibility.
        mark = self._mark
        if mark not in self._AUTO_RASTER_ELIGIBLE_MARKS:
            return self

        # Composite/statistical marks (histogram, density, hex, raster, smooth,
        # boxplot, etc.) produce aggregate output -- the raw row count does not
        # reflect the actual SVG element count. Skip auto-raster for these.
        if self._composite_kind is not None:
            return self

        # Check for active color encoding -- do NOT auto-raster.
        if "color" in self._encoding:
            _warn_large_chart(mark_count)
            return self

        # Check x and y are both present and quantitative.
        x_enc = self._encoding.get("x")
        y_enc = self._encoding.get("y")
        if x_enc is None or y_enc is None:
            _warn_large_chart(mark_count)
            return self

        # Determine if both are quantitative.
        def _is_quantitative(enc) -> bool:
            if isinstance(enc, ChannelBase):
                t = enc._kwargs.get("type")
                return t in ("Q", "quantitative")
            # Raw string shorthand -- check for :Q suffix.
            if isinstance(enc, str) and ":Q" in enc:
                return True
            return False

        if not _is_quantitative(x_enc) or not _is_quantitative(y_enc):
            _warn_large_chart(mark_count)
            return self

        # Error mode -- raise instead of substituting.
        if behavior == "error":
            raise ValueError(
                f"Auto-raster: {mark_count:,} marks exceed threshold "
                f"{threshold:,}. Pass raster=False to .show()/.save() to disable."
            )

        # Perform the substitution via mark_raster.
        x_field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
        y_field = y_enc.field if isinstance(y_enc, ChannelBase) else y_enc

        substituted = self._clone()
        # Apply mark_raster by going through the composite mark path.
        # This sets _pending_stat_mark which will be resolved by to_spec().
        from ferrum._layer import _PendingMark
        from ferrum.marks.heavy_stat import desugar_raster

        # Only forward raster_field to aggregates that actually consume a value
        # column ("mean"/"sum"). "count"/"density"/"any" need no field, so a
        # stray raster_field must stay genuinely inert for them -- forwarding
        # it unconditionally would resolve it in the Rust Raster transform
        # regardless of aggregate, surfacing a deferred-to-render
        # "column not found" error for an unused value, which is exactly the
        # deferred-failure shape RenderConfig.__post_init__ exists to remove.
        needs_field = cfg.raster_aggregate in _RASTER_AGGREGATES_NEEDING_FIELD
        substituted._pending_stat_mark = _PendingMark(
            "raster",
            {
                "aggregate": cfg.raster_aggregate,
                "field": cfg.raster_field if needs_field else None,
                "cmap": cfg.raster_scheme,
                "resolution": "screen",
                "blend": "alpha",
                "min_count": None,
                "log_scale": False,
            },
            desugar_raster,
            prior_mark=None,
        )
        substituted._mark = "image"  # placeholder for raster
        substituted._composite_kind = "raster"

        if behavior == "warn":
            warnings.warn(
                f"Auto-raster: substituted mark_raster for {mark} "
                f"({mark_count:,} marks > threshold {threshold:,}). "
                f"Pass raster=False to .show()/.save() to disable.",
                UserWarning,
                stacklevel=4,
            )

        return substituted

    def _with_raster_override(self, raster: bool | None) -> "Chart":
        """Return a clone with auto-raster forced on/off, or *self* if None."""
        if raster is None:
            return self
        from ferrum.render_config import RenderConfig
        import dataclasses

        base = self._render_config or RenderConfig()
        if raster is False:
            merged = dataclasses.replace(base, raster_threshold=None)
        else:
            merged = dataclasses.replace(base, raster_threshold=0)
        new = self._clone()
        new._render_config = merged
        return new

    def _resolve_chart_config(self) -> dict:
        """Merge _configure layers and annotations into a single dict for the Rust binding."""
        merged: dict = {}
        for cfg in self._configure:
            d = cfg.to_dict()
            for key, val in d.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
        # GH #74: configure_legend(orient="none") joins the SAME suppression
        # mechanism Color(legend=None) uses (encoding.<channel>.legend.disabled
        # on the Rust side) -- there is no LegendOrient::None variant. Derived
        # here, AFTER the full _configure layer merge above (not per-layer
        # inside LegendConfig.to_dict()), so a later, more-specific
        # configure_legend(...) layer -- e.g. a leaf's own call overriding a
        # composite-cascaded orient="none" injected by _inject_parent_config --
        # correctly wins and the suppression does not stick, matching the
        # existing "per-chart layers (appear later) override" cascade
        # convention documented on _inject_parent_config.
        legend_cfg = merged.get("legend")
        if isinstance(legend_cfg, dict) and legend_cfg.get("orient") == "none":
            # Shallow-copy before mutating: in the single-config-layer branch
            # above, `merged["legend"]` is assigned by reference from
            # `LegendConfig.to_dict()` (line ~211), so mutating in place would
            # silently corrupt a dict a caller may still hold a reference to.
            legend_cfg = dict(legend_cfg)
            legend_cfg["disabled"] = True
            merged["legend"] = legend_cfg
        if self._annotations:
            ann_list = []
            for annotate in self._annotations:
                ann_list.extend(annotate.to_dict_list())
            # Resolve any {"category": value} coordinate dicts to {"norm": frac}
            # so the Rust renderer receives only the Data/Pixel/Norm variants it
            # knows about.  This handles ordinal category string annotations.
            ann_list = _resolve_category_coords_in_annotations(ann_list, self)
            merged["annotations"] = ann_list
        if self._structural:
            merged["structural"] = [self._serialize_structural(feat) for feat in self._structural]
        return merged

    @staticmethod
    def _serialize_structural(feat) -> dict:
        """Convert a structural feature dataclass to its dict form for the Rust binding.

        ``SecondaryY`` is not handled here — it desugars to an appended
        independent-y layer at ``Chart.__add__`` time (GH #52,
        ``_desugar_secondary_y``) rather than accumulating in ``_structural``.
        """
        from ferrum.structural import BreakAxis, Inset

        if isinstance(feat, BreakAxis):
            gaps = feat.gap
            # Normalize single (start, end) tuple to a list of [start, end] pairs.
            if (
                isinstance(gaps, tuple)
                and len(gaps) == 2
                and not isinstance(gaps[0], (list, tuple))
            ):
                normalized_gaps = [list(gaps)]
            else:
                normalized_gaps = [list(g) for g in gaps]
            return {
                "type": "break_axis",
                "axis": feat.axis,
                "gaps": normalized_gaps,
                "break_size": feat.break_size,
                "break_style": feat.break_style,
            }
        elif isinstance(feat, Inset):
            from ferrum.annotation.coords import NormCoord, PixelCoord

            def _inset_coord(c: object) -> float:
                """Convert a bound coordinate to a normalized float for InsetSpec.

                NormCoord and plain floats are passed as-is (both already in [0,1]).
                PixelCoord is not supported for Inset bounds because the Rust side
                expects normalized coordinates and has no access to plot dimensions
                at serialization time.
                """
                if isinstance(c, NormCoord):
                    return c.value
                if isinstance(c, PixelCoord):
                    raise TypeError(
                        "Inset bounds do not support px() coordinates. "
                        "Use fm.norm(f) (normalized [0, 1]) or plain floats instead."
                    )
                return float(c)

            inset_svg = feat.chart.to_svg()
            d = {
                "type": "inset",
                "svg": inset_svg,
                "bounds": [_inset_coord(c) for c in feat.bounds],
                "border": feat.border,
                "border_color": feat.border_color,
                "background": feat.background,
                "shadow": feat.shadow,
                "connect_style": feat.connect_style,
            }
            if feat.border_dash is not None:
                d["border_dash"] = feat.border_dash
            if feat.connect_to is not None:
                d["connect_to"] = list(feat.connect_to)
            return d
        else:
            raise TypeError(f"Unknown structural feature type: {type(feat)}")

    def _render_inputs(self, *, _auto_tooltips: bool = False) -> tuple:
        import json

        from ferrum._core import ChartSpec

        resolved = self._resolve_pending()
        chart = resolved._apply_auto_raster()
        # Chart.override: build the payload once per render (guarded on _overrides so a
        # chart that never calls .override() renders byte-identically). The spec-piece
        # payload is threaded into to_spec(); chart_config / viewport / deprecations are
        # applied below, all LAST among presentation sources so override wins (spec §7).
        override_payload = None
        if chart._overrides:
            from ferrum._override_apply import build_payload

            override_payload = build_payload(chart._overrides)
        spec = chart.to_spec(_override_payload=override_payload)
        if _auto_tooltips:
            kw = json.loads(spec.to_json())
            kw = chart._inject_auto_tooltips(kw)
            spec = ChartSpec.from_json(json.dumps(kw))
        # F17: apply Axis(label_map=...) column-value remapping before data
        # reaches Rust so the scale domain uses the display labels.
        label_maps = _collect_label_maps(chart)
        raw_data = _apply_label_maps(chart._data, label_maps) if label_maps else chart._data
        data = normalize_for_rust(to_arrow_table(raw_data))
        from ferrum import config as _config

        viewport = (
            chart._width or float(_config.get("width")),
            chart._height or float(_config.get("height")),
        )
        from ferrum.themes._defaults import get_default_theme

        effective_theme = chart._theme or get_default_theme()
        theme_dict = effective_theme.to_spec_dict() if effective_theme else {}
        chart_config_dict = chart._resolve_chart_config()
        if override_payload is not None:
            from ferrum._override_consume import (
                apply_properties,
                emit_deprecations,
                merge_chart_config,
            )

            chart_config_dict = merge_chart_config(chart_config_dict, override_payload)
            viewport = apply_properties(viewport, override_payload)
            emit_deprecations(override_payload)
        return spec, data, viewport, theme_dict, chart_config_dict

    def to_svg(self, *, raster: bool | None = None) -> str:
        """Return the chart rendered as an SVG string.

        This **returns** the SVG markup; it does not display the chart.
        Use [show][ferrum.Chart.show] to display inline or in a browser, or
        [save][ferrum.Chart.save] to write to disk.

        Parameters
        ----------
        raster : bool or None, default None
            Override the auto-raster policy for this render only.
            ``False`` forces per-element SVG regardless of mark count.
            ``True`` forces raster aggregation.  ``None`` uses the chart's
            ``RenderConfig`` policy.

        Returns
        -------
        str
            SVG markup for the chart.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        >>> svg.startswith("<svg")
        True
        """
        from ferrum._core import render_svg

        chart = self._with_raster_override(raster)
        spec, data, viewport, theme_dict, chart_config_dict = chart._render_inputs()
        if data.num_rows == 0:
            w, h = viewport
            empty_svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
                f"<!-- empty dataset --></svg>"
            )
            figure_caption = getattr(chart, "_figure_caption", None)
            if figure_caption is not None:
                from ferrum._chrome import chrome_kwargs
                from ferrum._core import wrap_svg_with_chrome

                return wrap_svg_with_chrome(
                    empty_svg,
                    caption=figure_caption,
                    **chrome_kwargs(chart_config_dict),
                )
            return empty_svg
        svg = render_svg(
            spec,
            data,
            viewport=viewport,
            theme=theme_dict,
            chart_config=chart_config_dict or None,
        )
        # Post-wrap with a caption band when one has been set via
        # .properties(caption=...).  wrap_svg_with_chrome is a no-op wrapper
        # when no chrome is given, so this branch is only reached when a
        # caption is actually present.
        figure_caption = getattr(chart, "_figure_caption", None)
        if figure_caption is not None:
            from ferrum._chrome import chrome_kwargs
            from ferrum._core import wrap_svg_with_chrome

            svg = wrap_svg_with_chrome(
                svg,
                caption=figure_caption,
                **chrome_kwargs(chart_config_dict),
            )
        return svg

    def to_png(self, *, raster: bool | None = None, scale: float = 2.0) -> bytes:
        """Return the chart rendered as PNG bytes.

        This **returns** the PNG-encoded image data; it does not display the
        chart.  Use [show][ferrum.Chart.show] to display, or [save][ferrum.Chart.save] to write to disk.

        Parameters
        ----------
        raster : bool or None, default None
            Override the auto-raster policy for this render only.
            ``False`` forces per-element rendering.  ``True`` forces raster.
            ``None`` uses the chart's ``RenderConfig`` policy.
        scale : float, default 2.0
            Pixel-density multiplier applied to the chart's intrinsic dimensions.
            ``2.0`` (the default) produces a retina-quality image at twice the
            logical pixel count.  ``1.0`` renders at 1:1 resolution.

        Returns
        -------
        bytes
            PNG-encoded image data.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> png = fm.Chart(df).mark_point().encode(x="x", y="y").to_png()
        >>> png[:4] == b'\\x89PNG'
        True
        """
        from ferrum._core import rasterize_svg

        svg = self.to_svg(raster=raster)
        return bytes(rasterize_svg(svg, scale=scale))

    def to_html(
        self,
        *,
        embed_wasm: bool = True,
        toolbar: bool = True,
        raster: bool | None = None,
        csp_nonce: str | None = None,
    ) -> str:
        """Return the chart as a self-contained interactive HTML document.

        This **returns** the HTML markup; it does not display the chart or
        write it to disk.  The returned string is byte-identical to what
        ``save(path)`` writes for an ``.html`` destination — it embeds the
        WASM-backed interactive renderer rather than a static SVG snapshot.
        Because it bundles that renderer, the document is substantially larger
        than a static export; for a lightweight static image use
        [to_svg][ferrum.Chart.to_svg] / [to_png][ferrum.Chart.to_png].

        Parameters
        ----------
        embed_wasm : bool, default True
            When True, the WASM binary is base64-inlined for single-file
            distribution.  When False, the document references an adjacent
            ``ferrum_wasm_bg.wasm`` sidecar that must be served alongside it.
        toolbar : bool, default True
            When False, the interactive toolbar (zoom / pan controls, export
            button) is hidden in the rendered HTML.
        raster : bool or None, default None
            Override the auto-raster policy for this render only.
            ``False`` forces per-element rendering.  ``True`` forces raster.
            ``None`` uses the chart's ``RenderConfig`` policy.
        csp_nonce : str, optional
            When provided, both the ``<style>`` and ``<script type="module">``
            tags receive a ``nonce="..."`` attribute so they pass strict
            Content-Security-Policy headers.

        Returns
        -------
        str
            A complete, self-contained interactive HTML document.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> html = fm.Chart(df).mark_point().encode(x="x", y="y").to_html()
        >>> html.lstrip().startswith("<!")
        True
        """
        from ferrum.display import html_string

        return html_string(
            self._with_raster_override(raster),
            embed_wasm=embed_wasm,
            toolbar=toolbar,
            csp_nonce=csp_nonce,
        )

    def show_svg(self, *, raster: bool | None = None) -> str:
        """Render the chart to an SVG string.

        .. deprecated:: 0.16.0
            Use [to_svg][ferrum.Chart.to_svg] instead.  ``show_svg`` will be removed in a
            future release.  It now forwards to [to_svg][ferrum.Chart.to_svg].

        Parameters
        ----------
        raster : bool or None, default None
            Override the auto-raster policy for this render only.

        Returns
        -------
        str
            SVG markup for the chart.
        """
        warnings.warn(
            "Chart.show_svg() is deprecated; use Chart.to_svg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_svg(raster=raster)

    def show_png(self, *, raster: bool | None = None, scale: float = 2.0) -> bytes:
        """Render the chart to PNG bytes.

        .. deprecated:: 0.16.0
            Use [to_png][ferrum.Chart.to_png] instead.  ``show_png`` will be removed in a
            future release.  It now forwards to [to_png][ferrum.Chart.to_png].

        Parameters
        ----------
        raster : bool or None, default None
            Override the auto-raster policy for this render only.
        scale : float, default 2.0
            Pixel-density multiplier applied to the chart's intrinsic dimensions.

        Returns
        -------
        bytes
            PNG-encoded image data.
        """
        warnings.warn(
            "Chart.show_png() is deprecated; use Chart.to_png() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_png(raster=raster, scale=scale)

    def save(
        self,
        path,
        *,
        format=None,
        embed_wasm=True,
        raster: bool | None = None,
        scale: float = 2.0,
        toolbar: bool = True,
        csp_nonce: str | None = None,
    ) -> None:
        """Save the chart to a file on disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path.  Extension determines the default format:
            ``.svg`` -> SVG, ``.png`` -> PNG, ``.html`` -> HTML,
            ``.json`` -> JSON, ``.pdf`` -> PDF.
        format : {"svg", "png", "html", "json", "pdf"} or None, optional
            Explicit format override.  ``None`` (default) infers from ``path``.
        embed_wasm : bool
            For ``"html"`` format only.  When True (default), the WASM binary
            is base64-inlined for single-file distribution.
        raster : bool or None, default None
            Override the auto-raster policy for this save only.
            ``False`` forces per-element output.  ``True`` forces raster.
            ``None`` uses the chart's ``RenderConfig`` policy.
        scale : float, default 2.0
            Pixel-density multiplier for PNG and PDF output.  Has no effect
            on SVG, HTML, or JSON exports.
        toolbar : bool, default True
            For ``"html"`` format only.  When False, the interactive toolbar
            is hidden in the rendered HTML.
        csp_nonce : str, optional
            For ``"html"`` format only.  When provided, both the ``<style>``
            and ``<script type="module">`` tags receive a ``nonce="..."``
            attribute so they pass strict Content-Security-Policy headers.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").save("/tmp/chart.svg")
        """
        from ferrum.display import save_chart

        save_chart(
            self._with_raster_override(raster),
            path,
            format=format,
            embed_wasm=embed_wasm,
            scale=scale,
            toolbar=toolbar,
            csp_nonce=csp_nonce,
        )

    def show(self, *, raster: bool | None = None) -> None:
        """Display the chart inline or in a browser.

        Parameters
        ----------
        raster : bool or None, default None
            Override the auto-raster policy for this render only.
            ``False`` forces per-element SVG regardless of mark count.
            ``True`` forces raster aggregation.  ``None`` uses the chart's
            ``RenderConfig`` policy.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> fm.Chart(df).mark_point().encode(x="x", y="y").show()  # doctest: +SKIP
        """
        from ferrum.display import show_chart

        show_chart(self._with_raster_override(raster))

    def _repr_svg_(self) -> str | None:
        """Jupyter SVG rich display hook."""
        try:
            return self.to_svg()
        except Exception:
            _logger.debug("Chart._repr_svg_ failed; falling back to __repr__", exc_info=True)
            return None

    def _repr_html_(self) -> str | None:
        """Jupyter HTML rich display hook -- wraps SVG in a <div>."""
        try:
            return f"<div>{self.to_svg()}</div>"
        except Exception:
            _logger.debug("Chart._repr_html_ failed; falling back to __repr__", exc_info=True)
            return None
