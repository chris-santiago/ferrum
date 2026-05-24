"""Rendering mixin — extracted from chart.py to reduce file size.

The ``_RenderMixin`` class provides rendering, display, and auto-raster
methods that ``Chart`` inherits via mixin.  All methods operate through
``self`` (which is a ``Chart`` instance at runtime) and access ``Chart``
slots like ``_render_config``, ``_mark``, ``_data``, ``_encoding``, etc.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from ferrum._coerce import to_arrow_table
from ferrum.encoding.base import ChannelBase

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


def _sanitize_for_rust(tbl: "pyarrow.Table") -> "pyarrow.Table":
    """Decode or cast any Arrow column types that the Rust CDI boundary rejects.

    This is a render-boundary concern, not a coerce concern — ``to_arrow_table``
    preserves the caller's data as-is so its contract is predictable.  The Rust
    renderer currently rejects two special types:

    - ``Dictionary`` (categorical / dictionary-encoded) — decode to the plain
      value type via ``pyarrow.compute.dictionary_decode()``.
    - ``Null`` (all-None column with unknown type) — cast to ``float64`` so Rust
      can represent it as a numeric column of NaNs.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    new_cols: list = []
    needs_rebuild = False
    for i in range(len(tbl.schema)):
        col = tbl.column(i)
        field_type = tbl.schema.field(i).type
        if pa.types.is_dictionary(field_type):
            col = pc.dictionary_decode(col)
            needs_rebuild = True
        elif pa.types.is_null(field_type):
            col = col.cast(pa.float64())
            needs_rebuild = True
        new_cols.append(col)
    if not needs_rebuild:
        return tbl
    return pa.table({tbl.schema.field(i).name: new_cols[i] for i in range(len(new_cols))})


def _collect_label_maps(chart: Any) -> dict[str, dict[str, str]]:
    """Collect Axis(label_map=...) entries from a chart's encoding.

    Returns a mapping of ``{column_name: {old_value: new_value, ...}}``
    by scanning the chart's ``_encoding`` dict (single-mark path) and
    the ``_layers`` list (layered-mark path).

    Only x and y channels are checked because axis label remapping only
    applies to positional axes.
    """
    from ferrum.axis import Axis as _Axis

    result: dict[str, dict[str, str]] = {}

    def _check_enc(enc_dict: dict) -> None:
        for ch_name in ("x", "y"):
            ch = enc_dict.get(ch_name)
            if not isinstance(ch, ChannelBase):
                continue
            axis_kwarg = ch._kwargs.get("axis")
            if not isinstance(axis_kwarg, _Axis):
                continue
            if axis_kwarg.label_map is None:
                continue
            col = ch.field
            if col is None:
                continue
            if col in result:
                # Merge: later layers override earlier for same column.
                result[col].update(axis_kwarg.label_map)
            else:
                result[col] = dict(axis_kwarg.label_map)

    # Single-mark or top-level encoding
    _check_enc(chart._encoding)

    # Layered chart: each _Layer has its own encoding dict
    if chart._layers:
        for layer in chart._layers:
            _check_enc(layer.encoding)

    return result


def _apply_label_maps(
    data: Any,
    label_maps: dict[str, dict[str, str]],
) -> Any:
    """Apply label remapping to a polars DataFrame.

    For each ``(column_name, mapping)`` pair in *label_maps*, replaces
    string values in that column according to the mapping.  Values not
    present in the mapping are left unchanged.

    If the column does not exist in *data* or the data is not a polars
    DataFrame, the data is returned unchanged.
    """
    if not label_maps:
        return data

    import polars as pl

    if not isinstance(data, pl.DataFrame):
        # Non-polars data: coerce first, but we can't modify Arrow tables
        # in-place easily — convert to polars, remap, return polars.
        try:
            from ferrum._coerce import to_arrow_table as _to_arrow
            import pyarrow as pa

            arrow = _to_arrow(data)
            df = pl.from_arrow(arrow)
        except (ImportError, TypeError, ValueError):
            import warnings

            warnings.warn(
                "Axis label_map could not be applied (data coercion failed); labels unchanged.",
                stacklevel=2,
            )
            return data
    else:
        df = data

    for col, mapping in label_maps.items():
        if col not in df.columns:
            continue
        series = df[col]
        if series.dtype not in (pl.Utf8, pl.String, pl.Categorical):
            continue
        # replace(mapping) without a default leaves unmatched values unchanged.
        df = df.with_columns(series.replace(mapping).alias(col))

    return df


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
        from ferrum.render_config import RenderConfig

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

        substituted._pending_stat_mark = _PendingMark(
            "raster",
            {
                "aggregate": cfg.raster_aggregate,
                "cmap": cfg.raster_cmap,
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
        if self._annotations:
            ann_list = []
            for annotate in self._annotations:
                ann_list.extend(annotate.to_dict_list())
            merged["annotations"] = ann_list
        if self._structural:
            merged["structural"] = [self._serialize_structural(feat) for feat in self._structural]
        return merged

    @staticmethod
    def _serialize_structural(feat) -> dict:
        """Convert a structural feature dataclass to its dict form for the Rust binding."""
        from ferrum.structural import BreakAxis, Inset, SecondaryY

        if isinstance(feat, SecondaryY):
            d: dict = {"type": "secondary_y", "field": feat.field, "mark": feat.mark}
            if feat.color is not None:
                d["color"] = feat.color
            if feat.opacity is not None:
                d["opacity"] = feat.opacity
            return d
        elif isinstance(feat, BreakAxis):
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

            inset_svg = feat.chart.show_svg()
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
        spec = chart.to_spec()
        if _auto_tooltips:
            kw = json.loads(spec.to_json())
            kw = chart._inject_auto_tooltips(kw)
            spec = ChartSpec.from_json(json.dumps(kw))
        # F17: apply Axis(label_map=...) column-value remapping before data
        # reaches Rust so the scale domain uses the display labels.
        label_maps = _collect_label_maps(chart)
        raw_data = _apply_label_maps(chart._data, label_maps) if label_maps else chart._data
        data = _sanitize_for_rust(to_arrow_table(raw_data))
        from ferrum import config as _config

        viewport = (
            chart._width or float(_config.get("width")),
            chart._height or float(_config.get("height")),
        )
        from ferrum.themes._defaults import get_default_theme

        effective_theme = chart._theme or get_default_theme()
        theme_dict = effective_theme.to_spec_dict() if effective_theme else {}
        chart_config_dict = chart._resolve_chart_config()
        return spec, data, viewport, theme_dict, chart_config_dict

    def show_svg(self, *, raster: bool | None = None) -> str:
        """Render the chart to an SVG string.

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
        >>> svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        >>> svg.startswith("<svg")
        True
        """
        from ferrum._core import render_svg

        chart = self._with_raster_override(raster)
        spec, data, viewport, theme_dict, chart_config_dict = chart._render_inputs()
        if data.num_rows == 0:
            w, h = viewport
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
                f"<!-- empty dataset --></svg>"
            )
        return render_svg(
            spec,
            data,
            viewport=viewport,
            theme=theme_dict,
            chart_config=chart_config_dict or None,
        )

    def show_png(self, *, raster: bool | None = None, scale: float = 2.0) -> bytes:
        """Render the chart to PNG bytes.

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
        >>> png = fm.Chart(df).mark_point().encode(x="x", y="y").show_png()
        >>> png[:4] == b'\\x89PNG'
        True
        """
        from ferrum._core import rasterize_svg

        svg = self.show_svg(raster=raster)
        return bytes(rasterize_svg(svg, scale=scale))

    def save(
        self,
        path,
        *,
        format=None,
        embed_wasm=True,
        raster: bool | None = None,
        scale: float = 2.0,
        toolbar: bool = True,
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
            return self.show_svg()
        except Exception:
            _logger.debug("Chart._repr_svg_ failed; falling back to __repr__", exc_info=True)
            return None

    def _repr_html_(self) -> str | None:
        """Jupyter HTML rich display hook -- wraps SVG in a <div>."""
        try:
            return f"<div>{self.show_svg()}</div>"
        except Exception:
            _logger.debug("Chart._repr_html_ failed; falling back to __repr__", exc_info=True)
            return None
