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
            # After decoding, the underlying type may be date32/date64 which
            # Rust also rejects — cast to timestamp[ms] (same as _coerce.py).
            decoded_type = col.type
            if pa.types.is_date32(decoded_type) or pa.types.is_date64(decoded_type):
                col = col.cast(pa.timestamp("ms"))
            needs_rebuild = True
        elif pa.types.is_null(field_type):
            col = col.cast(pa.float64())
            needs_rebuild = True
        new_cols.append(col)
    if not needs_rebuild:
        return tbl
    return pa.table(new_cols, names=[tbl.schema.field(i).name for i in range(len(tbl.schema))])


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


def _build_ordinal_norm_map(domain: list) -> dict[str, float]:
    """Build a category → norm-center mapping for an ordinal domain list.

    Each category occupies an equal band of width ``1/n``.  The band center
    for item at index *i* (0-based) is ``(i + 0.5) / n``.

    Parameters
    ----------
    domain : list
        Ordered list of category values as they appear in the data.

    Returns
    -------
    dict[str, float]
        ``{str(category): norm_center}``

    Examples
    --------
    >>> _build_ordinal_norm_map(["a", "b", "c"])
    {'a': 0.1667, 'b': 0.5, 'c': 0.8333}
    """
    n = len(domain)
    if n == 0:
        return {}
    # NOTE: (i+0.5)/n is a band-center approximation that ignores the Rust band
    # scale's padding_inner/outer (default 0.1), so the annotation lands in the
    # correct band/order but is not pixel-aligned to the bar center.
    return {str(v): (i + 0.5) / n for i, v in enumerate(domain)}


def _extract_ordinal_domain(chart: Any, channel: str) -> list:
    """Return the ordered unique values for *channel* in *chart*, or an empty list.

    Checks the chart's ``_data`` attribute against the field name(s) bound to
    *channel* in ``_encoding`` (single-mark path) and ``_layers`` (layered path).
    Only processes columns whose dtype is ordinal-compatible (string or
    categorical).

    Parameters
    ----------
    chart : Chart
        Chart to inspect.
    channel : str
        Encoding channel, typically ``"x"`` or ``"y"``.

    Returns
    -------
    list
        Unique values in first-appearance order.
    """
    import polars as pl
    from ferrum._scale_share import _chart_bindings

    data = getattr(chart, "_data", None)
    if data is None or not isinstance(data, pl.DataFrame):
        return []

    for field in _chart_bindings(chart, channel):
        if field is None:
            continue
        try:
            col = data[field]
        except (KeyError, AttributeError):
            continue
        if col.dtype not in (pl.Utf8, pl.String, pl.Categorical):
            continue
        # Preserve appearance order using unique(maintain_order=True)
        return col.unique(maintain_order=True).to_list()

    return []


def _resolve_category_coords_in_annotations(
    ann_list: list[dict],
    chart: Any,
) -> list[dict]:
    """Replace ``{"category": value}`` annotation coordinate dicts with ``{"norm": frac}``.

    The Rust annotation renderer's ``CoordValue`` enum only accepts
    ``Data(f64)``, ``Pixel { px }``, and ``Norm { norm }``.  Ordinal category
    coordinates (strings that are not ISO-8601 dates) are serialized by
    ``_coord()`` as ``{"category": value}`` so they can be post-processed here
    using the chart's ordinal domain before the annotation list is sent to Rust.

    Resolution rules:
    - Build the ordinal domain for the x and y axes from the chart's data.
    - For each coordinate dict that is ``{"category": v}``, look up ``v`` in
      the appropriate axis domain and replace the dict with
      ``{"norm": band_center}`` where ``band_center = (index + 0.5) / n``.
    - If the category is not found in the domain (unknown label, or wrong axis),
      fall back to ``{"norm": 0.5}`` (plot center) so the annotation renders
      without crashing.

    Parameters
    ----------
    ann_list : list[dict]
        Annotation spec dicts (as produced by ``to_dict_list()``).
    chart : Chart
        The chart whose ordinal domain is used for resolution.

    Returns
    -------
    list[dict]
        Annotation spec dicts with all ``{"category": ...}`` entries resolved.
    """
    # Lazily build the norm maps only when at least one {"category": ...} entry
    # exists to avoid scanning chart data on every render.
    _x_map: dict[str, float] | None = None
    _y_map: dict[str, float] | None = None

    def _x_norm_map() -> dict[str, float]:
        nonlocal _x_map
        if _x_map is None:
            _x_map = _build_ordinal_norm_map(_extract_ordinal_domain(chart, "x"))
        return _x_map

    def _y_norm_map() -> dict[str, float]:
        nonlocal _y_map
        if _y_map is None:
            _y_map = _build_ordinal_norm_map(_extract_ordinal_domain(chart, "y"))
        return _y_map

    # Coordinate keys that belong to the x-axis vs y-axis.
    _X_KEYS = frozenset({"x", "x1", "x2"})
    _Y_KEYS = frozenset({"y", "y1", "y2"})

    def _resolve_coord(key: str, val: Any) -> Any:
        """Replace a single coordinate value if it is a category dict."""
        if not isinstance(val, dict) or "category" not in val:
            return val
        cat = val["category"]
        if key in _X_KEYS:
            norm_map = _x_norm_map()
            axis_label = "x"
        elif key in _Y_KEYS:
            norm_map = _y_norm_map()
            axis_label = "y"
        else:
            # Unrecognised key — fall back to center.
            norm_map = {}
            axis_label = key
        cat_str = str(cat)
        if cat_str not in norm_map:
            domain_repr = sorted(norm_map.keys()) if norm_map else []
            warnings.warn(
                f"annotation category {cat_str!r} not found in {axis_label}-axis domain "
                f"{domain_repr}; placing at plot center",
                UserWarning,
                stacklevel=2,
            )
        norm_val = norm_map.get(cat_str, 0.5)
        return {"norm": norm_val}

    resolved = []
    for ann in ann_list:
        new_ann = dict(ann)
        for key, val in ann.items():
            new_ann[key] = _resolve_coord(key, val)
        resolved.append(new_ann)
    return resolved


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

    def to_svg(self, *, raster: bool | None = None) -> str:
        """Return the chart rendered as an SVG string.

        This **returns** the SVG markup; it does not display the chart.
        Use :meth:`show` to display inline or in a browser, or
        :meth:`save` to write to disk.

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
                from ferrum._core import compose_svg_vertical

                return compose_svg_vertical([empty_svg], spacing=0.0, caption=figure_caption)
            return empty_svg
        svg = render_svg(
            spec,
            data,
            viewport=viewport,
            theme=theme_dict,
            chart_config=chart_config_dict or None,
        )
        # Post-wrap with a caption band when one has been set via
        # .properties(caption=...).  Using compose_svg_vertical with a single
        # SVG is byte-identical to the plain SVG when no chrome is given, so
        # this branch is only reached when a caption is actually present.
        figure_caption = getattr(chart, "_figure_caption", None)
        if figure_caption is not None:
            from ferrum._core import compose_svg_vertical

            svg = compose_svg_vertical([svg], spacing=0.0, caption=figure_caption)
        return svg

    def to_png(self, *, raster: bool | None = None, scale: float = 2.0) -> bytes:
        """Return the chart rendered as PNG bytes.

        This **returns** the PNG-encoded image data; it does not display the
        chart.  Use :meth:`show` to display, or :meth:`save` to write to disk.

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
    ) -> str:
        """Return the chart as a self-contained interactive HTML document.

        This **returns** the HTML markup; it does not display the chart or
        write it to disk.  The returned string is byte-identical to what
        ``save(path)`` writes for an ``.html`` destination — it embeds the
        WASM-backed interactive renderer rather than a static SVG snapshot.

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
        from ferrum._html import assemble_html
        from ferrum.display import _extract_title_text, _render_scene_json

        chart = self._with_raster_override(raster)
        scene_json, packed_data = _render_scene_json(chart)
        title = _extract_title_text(chart._title)
        return assemble_html(
            scene_json,
            packed_data=packed_data,
            title=title,
            embed_wasm=embed_wasm,
            toolbar=toolbar,
        )

    def show_svg(self, *, raster: bool | None = None) -> str:
        """Render the chart to an SVG string.

        .. deprecated:: 0.16.0
            Use :meth:`to_svg` instead.  ``show_svg`` will be removed in a
            future release.  It now forwards to :meth:`to_svg`.

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
            Use :meth:`to_png` instead.  ``show_png`` will be removed in a
            future release.  It now forwards to :meth:`to_png`.

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
