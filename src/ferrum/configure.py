"""Chart-level configuration dataclasses for Ferrum's declarative config surface."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any

from ferrum._configure_mixin import _MISSING, _resolve_band_alias
from ferrum._validate import validate_choice, validate_pixel_value


_VALID_LEGEND_ORIENTS = frozenset({"right", "left", "top", "bottom", "none"})
_VALID_TITLE_ANCHORS = frozenset({"start", "middle", "end"})

_AXIS_XY_DEPRECATION_MSG = (
    "AxisConfig.x / configure_axis(x=...) has no effect and is deprecated; "
    "use Chart.axis(x=False) / Chart.axis(y=False) to show/hide an axis."
)


def _warn_axis_xy_deprecated(*, x: bool, y: bool, stacklevel: int) -> None:
    """Emit a DeprecationWarning when the vestigial ``x``/``y`` flags are set to ``False``.

    ``x=True``/``y=True`` are the do-nothing defaults and never warn; only the
    meaningful (but no-op) ``False`` intent is flagged. Centralised here so the
    direct-construction path (:meth:`AxisConfig.__init__`) and the
    ``configure_axis`` mixin method share one message.
    """
    if x is False or y is False:
        warnings.warn(_AXIS_XY_DEPRECATION_MSG, DeprecationWarning, stacklevel=stacklevel + 1)


def _to_dict_omit_none(obj: Any) -> dict[str, Any]:
    """Serialize a dataclass to dict, omitting fields whose value is None."""
    result: dict[str, Any] = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is not None:
            result[f.name] = val
    return result


@dataclass(frozen=True, init=False)
class AxisConfig:
    """Chart-level axis configuration applied uniformly to all axes (or a specific one).

    Parameters
    ----------
    x, y : bool, default True
        Deprecated and has no effect. Use ``Chart.axis(x=False)`` /
        ``Chart.axis(y=False)`` to show or hide an axis.
    label_angle : float, optional
        Tick label rotation angle in degrees.
    label_font_size : float, optional
        Tick label font size.
    label_color : str, optional
        Tick label color.
    label_format : str, optional
        Named format preset (see ``ferrum.format_presets``). Must be a
        recognized preset key; raises ``ValueError`` at construction
        otherwise. For a raw d3-format/strftime string, use
        ``label_format_raw`` instead. Mutually exclusive with
        ``label_format_raw``.
    label_format_raw : str, optional
        Raw d3-format or strftime string passed directly to the renderer.
        Mutually exclusive with ``label_format``.
    label_overlap : str, optional
        Label overlap strategy: ``"parity"``, ``"greedy"``, ``"rotate"``, or ``"hide"``.
    tick_count : int, optional
        Suggested number of ticks.
    tick_size : float, optional
        Tick mark length in pixels.
    tick_values : list, optional
        Explicit tick values.
    title_font_size : float, optional
        Axis title font size.
    title_color : str, optional
        Axis title color.
    title_padding : float, optional
        Padding between the axis title and tick labels.
    label_padding : float, optional
        Pixel gap between the tick mark endpoint and the label text baseline.
        Defaults to 2.0 when not set.
    domain : bool, optional
        Show the axis domain line.
    domain_color : str, optional
        Domain line color.
    domain_width : float, optional
        Domain line width.
    grid : bool, optional
        Show grid lines.
    grid_color : str, optional
        Grid line color.
    grid_dash : list[float], optional
        Grid line dash pattern.
    grid_width : float, optional
        Grid line width.
    domain_min : float, optional
        Minimum value of the scale domain.
    domain_max : float, optional
        Maximum value of the scale domain.
    nice : bool, optional
        Round the scale domain to nice round values.
    zero : bool, optional
        Include zero in the scale domain.
    grid_opacity : float, optional
        Grid line opacity (0--1).
    orient : str, optional
        Axis side: ``"top"``/``"bottom"`` (x) or ``"left"``/``"right"`` (y).
        Because chart-level ``axis`` applies to both axes, a single value is
        valid for only one of them; set ``orient`` via ``axis_x`` / ``axis_y``
        (or per-channel ``fm.Axis(orient=...)``) instead of the general ``axis``.
    translate : float, optional
        Pixel translation of the axis group perpendicular to its line.
    min_band : float, optional
        Lower bound for the reserved axis margin band, in pixels.
    max_band : float, optional
        Upper bound for the reserved axis margin band, in pixels.

        .. deprecated:: 0.17.0
            ``min_extent`` and ``max_extent`` are accepted as aliases for
            ``min_band`` and ``max_band`` respectively, but are deprecated and
            will be removed in a future release.
    tick_extra : bool, optional
        Append an extra tick at each domain boundary.
    tick_min_step : float, optional
        Minimum step between ticks in data space.
    title_orient : str, optional
        Side/orientation of the axis title.
    zindex : int, optional
        Coarse draw order of the axis relative to marks.
    """

    x: bool = True
    y: bool = True
    label_angle: float | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    label_format: str | None = None
    label_format_raw: str | None = None
    label_overlap: str | None = None
    tick_count: int | None = None
    tick_size: float | None = None
    tick_values: list | None = None
    title_font_size: float | None = None
    title_color: str | None = None
    title_padding: float | None = None
    domain: bool | None = None
    domain_color: str | None = None
    domain_width: float | None = None
    grid: bool | None = None
    grid_color: str | None = None
    grid_dash: list[float] | None = None
    grid_width: float | None = None
    domain_min: float | None = None
    domain_max: float | None = None
    nice: bool | None = None
    zero: bool | None = None
    label_padding: float | None = None
    grid_opacity: float | None = None
    orient: str | None = None
    translate: float | None = None
    min_band: float | None = None
    max_band: float | None = None
    tick_extra: bool | None = None
    tick_min_step: float | None = None
    title_orient: str | None = None
    zindex: int | None = None

    # ------------------------------------------------------------------
    # Custom __init__ — accepts deprecated min_extent=/max_extent= aliases
    # and maps them to the canonical min_band=/max_band= fields.
    # The dataclass is declared with init=False so we own the full init.
    # We use object.__setattr__ because the dataclass is frozen=True.
    # ------------------------------------------------------------------

    def __init__(
        self,
        x: bool = True,
        y: bool = True,
        label_angle: float | None = None,
        label_font_size: float | None = None,
        label_color: str | None = None,
        label_format: str | None = None,
        label_format_raw: str | None = None,
        label_overlap: str | None = None,
        tick_count: int | None = None,
        tick_size: float | None = None,
        tick_values: list | None = None,
        title_font_size: float | None = None,
        title_color: str | None = None,
        title_padding: float | None = None,
        domain: bool | None = None,
        domain_color: str | None = None,
        domain_width: float | None = None,
        grid: bool | None = None,
        grid_color: str | None = None,
        grid_dash: list[float] | None = None,
        grid_width: float | None = None,
        domain_min: float | None = None,
        domain_max: float | None = None,
        nice: bool | None = None,
        zero: bool | None = None,
        label_padding: float | None = None,
        grid_opacity: float | None = None,
        orient: str | None = None,
        translate: float | None = None,
        min_band: float | None = None,
        max_band: float | None = None,
        tick_extra: bool | None = None,
        tick_min_step: float | None = None,
        title_orient: str | None = None,
        zindex: int | None = None,
        # Deprecated aliases — accepted with a DeprecationWarning.
        min_extent: object = _MISSING,
        max_extent: object = _MISSING,
    ) -> None:
        # Resolve deprecated aliases (min_extent → min_band, max_extent → max_band)
        # through the shared canonical helper so all three surfaces stay in sync.
        min_band = _resolve_band_alias(
            min_band,
            min_extent,
            canonical_name="min_band",
            alias_name="min_extent",
            owner="AxisConfig",
            stacklevel=2,
        )
        max_band = _resolve_band_alias(
            max_band,
            max_extent,
            canonical_name="max_band",
            alias_name="max_extent",
            owner="AxisConfig",
            stacklevel=2,
        )

        # Reuse the existing validation logic from __post_init__.
        _warn_axis_xy_deprecated(x=x, y=y, stacklevel=2)
        if label_format is not None and label_format_raw is not None:
            raise ValueError(
                "AxisConfig: 'label_format' and 'label_format_raw' are mutually exclusive; "
                "provide at most one."
            )
        # Eager preset-name validation (NF-B1, 2026-09-02): AxisConfig's
        # label_format is preset-names-only by contract — it is not one of
        # the four raw-spec-accepting surfaces. A raw d3-format/strftime
        # string belongs in the dedicated, mutually-exclusive
        # label_format_raw field. An unrecognized name is a typed
        # construction-time refusal, not a silently-passed-through raw spec
        # (the exact NF-B1 harm class this surface must stay immune to).
        if label_format is not None:
            from ferrum.format_presets import resolve_format

            resolve_format(label_format)

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "label_angle", label_angle)
        object.__setattr__(self, "label_font_size", label_font_size)
        object.__setattr__(self, "label_color", label_color)
        object.__setattr__(self, "label_format", label_format)
        object.__setattr__(self, "label_format_raw", label_format_raw)
        object.__setattr__(self, "label_overlap", label_overlap)
        object.__setattr__(self, "tick_count", tick_count)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "tick_values", tick_values)
        object.__setattr__(self, "title_font_size", title_font_size)
        object.__setattr__(self, "title_color", title_color)
        object.__setattr__(self, "title_padding", title_padding)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "domain_color", domain_color)
        object.__setattr__(self, "domain_width", domain_width)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_color", grid_color)
        object.__setattr__(self, "grid_dash", grid_dash)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)
        object.__setattr__(self, "nice", nice)
        object.__setattr__(self, "zero", zero)
        object.__setattr__(self, "label_padding", label_padding)
        object.__setattr__(self, "grid_opacity", grid_opacity)
        object.__setattr__(self, "orient", orient)
        object.__setattr__(self, "translate", translate)
        object.__setattr__(self, "min_band", min_band)
        object.__setattr__(self, "max_band", max_band)
        object.__setattr__(self, "tick_extra", tick_extra)
        object.__setattr__(self, "tick_min_step", tick_min_step)
        object.__setattr__(self, "title_orient", title_orient)
        object.__setattr__(self, "zindex", zindex)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values.

        ``label_format`` preset names are resolved to their d3-format/strftime
        strings via :func:`ferrum.format_presets.resolve_format_field` before
        serialization (NF-B1) so the Rust side receives a ready-to-use format
        spec; ``__init__`` already rejected any unrecognized name at
        construction, so resolution here always succeeds. The resolved
        ``format_type`` is threaded onto the wire as ``label_format_type``.
        The deprecated, no-op ``x``/``y`` flags are never emitted — the wire
        schema does not accept them.
        """
        d = _to_dict_omit_none(self)
        d.pop("x", None)
        d.pop("y", None)
        if "label_format" in d:
            from ferrum.format_presets import resolve_format_field

            # AxisConfig has no user-facing label_format_type field; the
            # preset-derived type is always what gets threaded (label_format
            # is preset-names-only by contract, enforced in __init__).
            spec, format_type = resolve_format_field(d["label_format"], None)
            if spec is not None:
                d["label_format"] = spec
            if format_type is not None:
                d["label_format_type"] = format_type
        return d


@dataclass(frozen=True)
class LegendConfig:
    """Chart-level legend configuration.

    Parameters
    ----------
    orient : str, optional
        Legend position: ``"right"``, ``"left"``, ``"top"``, ``"bottom"``, or ``"none"``.
    direction : str, optional
        Layout direction: ``"vertical"`` or ``"horizontal"``.
    columns : int, optional
        Number of columns for multi-column layout.
    title_font_size : float, optional
        Legend title font size.
    label_font_size : float, optional
        Legend label font size.
    label_color : str, optional
        Legend label color.
    label_limit : float, optional
        Maximum legend-label width in pixels; wider labels are truncated with
        an ellipsis.
    symbol_size : float, optional
        Symbol size.
    symbol_stroke_width : float, optional
        Stroke width of legend symbol swatches in pixels.
    symbol_type : str, optional
        Symbol shape type.
    gradient_length : float, optional
        Gradient legend length.
    gradient_thickness : float, optional
        Gradient legend thickness in pixels.
    title_padding : float, optional
        Padding between the legend title and its entries.
    row_padding : float, optional
        Vertical entry spacing in pixels (vertical-direction legends).
    column_padding : float, optional
        Horizontal entry spacing in pixels (horizontal-direction legends).
    clip_height : float, optional
        Cap on the legend group height in pixels; overflow is hard-clipped.
    tick_min_step : float, optional
        Minimum step between colorbar ticks in data units.
    offset : float, optional
        Offset from the plot area.
    padding : float, optional
        Internal padding.
    zindex : int, optional
        Coarse draw order of the legend relative to marks.
    """

    orient: str | None = None
    direction: str | None = None
    columns: int | None = None
    title_font_size: float | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    label_limit: float | None = None
    symbol_size: float | None = None
    symbol_stroke_width: float | None = None
    symbol_type: str | None = None
    gradient_length: float | None = None
    gradient_thickness: float | None = None
    title_padding: float | None = None
    row_padding: float | None = None
    column_padding: float | None = None
    clip_height: float | None = None
    tick_min_step: float | None = None
    offset: float | None = None
    padding: float | None = None
    zindex: int | None = None

    def __post_init__(self) -> None:
        if self.orient is not None:
            validate_choice("LegendConfig.orient", "orient", self.orient, _VALID_LEGEND_ORIENTS)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class TitleConfig:
    """Chart-level title configuration.

    Parameters
    ----------
    font_size : float, optional
        Title font size.
    font_weight : str, optional
        Title font weight (e.g. ``"bold"``, ``"600"``).
    anchor : str, optional
        Horizontal anchor: ``"start"``, ``"middle"``, or ``"end"``.
    color : str, optional
        Title color.
    offset : float, optional
        Pixel offset from the plot area.
    subtitle_font_size : float, optional
        Subtitle font size.
    subtitle_color : str, optional
        Subtitle color.
    """

    font_size: float | None = None
    font_weight: str | None = None
    anchor: str | None = None
    color: str | None = None
    offset: float | None = None
    subtitle_font_size: float | None = None
    subtitle_color: str | None = None

    def __post_init__(self) -> None:
        if self.anchor is not None:
            validate_choice("TitleConfig.anchor", "anchor", self.anchor, _VALID_TITLE_ANCHORS)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class GridConfig:
    """Chart-level grid configuration.

    Parameters
    ----------
    x : bool, optional
        Show x-axis grid lines.
    y : bool, optional
        Show y-axis grid lines.
    color : str, optional
        Grid line color.
    width : float, optional
        Grid line width.
    dash : list[float], optional
        Grid line dash pattern.
    opacity : float, optional
        Grid line opacity.
    band_colors : list[str], optional
        Alternating band fill colors between grid lines.
    """

    x: bool | None = None
    y: bool | None = None
    color: str | None = None
    width: float | None = None
    dash: list[float] | None = None
    opacity: float | None = None
    band_colors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class PaddingConfig:
    """Chart-level padding configuration.

    Parameters
    ----------
    top : float, optional
        Top padding in pixels.
    right : float, optional
        Right padding in pixels.
    bottom : float, optional
        Bottom padding in pixels.
    left : float, optional
        Left padding in pixels.
    auto : bool
        When True and all four sides are None, let the renderer choose
        padding automatically.
    """

    top: float | None = None
    right: float | None = None
    bottom: float | None = None
    left: float | None = None
    auto: bool = True

    def __post_init__(self) -> None:
        # Validate at construction time (NF-B5/B6/B7): each side must be a
        # finite, non-negative pixel value (spec §4.7's pixel contract).
        validate_pixel_value("padding.top", self.top)
        validate_pixel_value("padding.right", self.right)
        validate_pixel_value("padding.bottom", self.bottom)
        validate_pixel_value("padding.left", self.left)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class ColorConfig:
    """Chart-level color scale configuration.

    Parameters
    ----------
    scheme : str, optional
        Named color scheme for categorical encodings.
    sequential_scheme : str, optional
        Named scheme for sequential (single-hue) encodings.
    diverging_scheme : str, optional
        Named scheme for diverging encodings.
    domain : list, optional
        Explicit scale domain values.
    range : list[str], optional
        Explicit list of color strings for the range.
    """

    scheme: str | None = None
    sequential_scheme: str | None = None
    diverging_scheme: str | None = None
    domain: list | None = None
    range: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class Configure:
    """Container for chart-level configuration overrides.

    Each field maps to a specific configuration domain.  Unset fields
    (``None``) mean "use the chart/theme default" — only fields that are
    explicitly set are forwarded to the renderer.

    Parameters
    ----------
    axis : AxisConfig, optional
        Applies to all axes.
    axis_x : AxisConfig, optional
        Applies only to the x axis (overrides ``axis`` for x).
    axis_y : AxisConfig, optional
        Applies only to the y axis (overrides ``axis`` for y).
    axis_y2 : AxisConfig, optional
        Applies only to the secondary y axis.
    legend : LegendConfig, optional
        Legend appearance.
    title : TitleConfig, optional
        Chart title appearance.
    grid : GridConfig, optional
        Grid line appearance.
    padding : PaddingConfig, optional
        Plot-area padding.
    color : ColorConfig, optional
        Default color scale settings.
    """

    axis: AxisConfig | None = None
    axis_x: AxisConfig | None = None
    axis_y: AxisConfig | None = None
    axis_y2: AxisConfig | None = None
    legend: LegendConfig | None = None
    title: TitleConfig | None = None
    grid: GridConfig | None = None
    padding: PaddingConfig | None = None
    color: ColorConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None fields and recursing into sub-configs."""
        result: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val.to_dict()
        return result
