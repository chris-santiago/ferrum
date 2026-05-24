"""Chart-level configuration dataclasses for Ferrum's declarative config surface."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


_VALID_LEGEND_ORIENTS = frozenset({"right", "left", "top", "bottom", "none"})
_VALID_TITLE_ANCHORS = frozenset({"start", "middle", "end"})


def _to_dict_omit_none(obj: Any) -> dict[str, Any]:
    """Serialize a dataclass to dict, omitting fields whose value is None."""
    result: dict[str, Any] = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is not None:
            result[f.name] = val
    return result


@dataclass(frozen=True)
class AxisConfig:
    """Chart-level axis configuration applied uniformly to all axes (or a specific one).

    Parameters
    ----------
    x : bool
        Show the x axis.
    y : bool
        Show the y axis.
    label_angle : float, optional
        Tick label rotation angle in degrees.
    label_font_size : float, optional
        Tick label font size.
    label_color : str, optional
        Tick label color.
    label_format : str, optional
        Named format preset (see ``ferrum.format_presets``). Mutually exclusive
        with ``label_format_raw``.
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

    def __post_init__(self) -> None:
        if self.label_format is not None and self.label_format_raw is not None:
            raise ValueError(
                "AxisConfig: 'label_format' and 'label_format_raw' are mutually exclusive; "
                "provide at most one."
            )
        if self.label_format is not None:
            from ferrum.format_presets import resolve_format

            resolve_format(self.label_format)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


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
    symbol_size : float, optional
        Symbol size.
    symbol_type : str, optional
        Symbol shape type.
    gradient_length : float, optional
        Gradient legend length.
    offset : float, optional
        Offset from the plot area.
    padding : float, optional
        Internal padding.
    """

    orient: str | None = None
    direction: str | None = None
    columns: int | None = None
    title_font_size: float | None = None
    label_font_size: float | None = None
    symbol_size: float | None = None
    symbol_type: str | None = None
    gradient_length: float | None = None
    offset: float | None = None
    padding: float | None = None

    def __post_init__(self) -> None:
        if self.orient is not None and self.orient not in _VALID_LEGEND_ORIENTS:
            raise ValueError(
                f"LegendConfig.orient must be one of {sorted(_VALID_LEGEND_ORIENTS)!r}; "
                f"got {self.orient!r}."
            )

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
        if self.anchor is not None and self.anchor not in _VALID_TITLE_ANCHORS:
            raise ValueError(
                f"TitleConfig.anchor must be one of {sorted(_VALID_TITLE_ANCHORS)!r}; "
                f"got {self.anchor!r}."
            )

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
