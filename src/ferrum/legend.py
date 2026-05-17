"""Legend value class for per-channel legend configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


# Default values that should be omitted from serialization (they match
# the renderer's built-in defaults and would add noise to the spec).
_LEGEND_DEFAULTS: dict[str, Any] = {
    "orient": "right",
    "direction": "vertical",
}


@dataclass(frozen=True, slots=True)
class Legend:
    """Per-channel legend configuration.

    Parameters
    ----------
    title : str, optional
        Legend title text.
    orient : str
        Legend position ("right", "left", "top", "bottom", "none").
    direction : str
        Layout direction ("vertical", "horizontal").
    type : str, optional
        Legend type ("symbol", "gradient").
    tick_count : int, optional
        Number of ticks for gradient legends.
    tick_min_step : float, optional
        Minimum tick step.
    values : list, optional
        Explicit legend values.
    format : str, optional
        Label format string.
    format_type : str, optional
        Format type.
    label_font_size : float, optional
        Label font size.
    label_color : str, optional
        Label color.
    label_limit : float, optional
        Maximum label width.
    symbol_size : float, optional
        Symbol size.
    symbol_stroke_width : float, optional
        Symbol stroke width.
    symbol_type : str, optional
        Symbol shape type.
    gradient_length : float, optional
        Gradient legend length.
    gradient_thickness : float, optional
        Gradient legend thickness.
    columns : int, optional
        Number of columns for multi-column layout.
    column_padding : float, optional
        Padding between columns.
    row_padding : float, optional
        Padding between rows.
    clip_height : float, optional
        Clip height for legend.
    title_font_size : float, optional
        Title font size.
    title_padding : float, optional
        Title padding.
    offset : float, optional
        Offset from plot.
    padding : float, optional
        Internal padding.
    zindex : int, optional
        Z-index.
    """

    title: str | None = None
    orient: str = "right"
    direction: str = "vertical"
    type: str | None = None
    tick_count: int | None = None
    tick_min_step: float | None = None
    values: list | None = None
    format: str | None = None
    format_type: str | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    label_limit: float | None = None
    symbol_size: float | None = None
    symbol_stroke_width: float | None = None
    symbol_type: str | None = None
    gradient_length: float | None = None
    gradient_thickness: float | None = None
    columns: int | None = None
    column_padding: float | None = None
    row_padding: float | None = None
    clip_height: float | None = None
    title_font_size: float | None = None
    title_padding: float | None = None
    offset: float | None = None
    padding: float | None = None
    zindex: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for the renderer, omitting None and default values."""
        result: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            # Skip None values
            if val is None:
                continue
            # Skip values that match the renderer default
            if f.name in _LEGEND_DEFAULTS and val == _LEGEND_DEFAULTS[f.name]:
                continue
            result[f.name] = val
        return result


def _normalize_legend(value: Any) -> dict[str, Any] | None:
    """Normalize a legend kwarg value to a dict or None.

    Accepts:
    - Legend instance -> .to_dict()
    - None or False -> {"disabled": True} (suppress legend)
    - dict -> pass through
    - Other truthy values -> None (not specified; reserved)
    """
    if value is None or value is False:
        return {"disabled": True}
    if isinstance(value, Legend):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    # Other truthy values are reserved — treat as "not specified".
    return None
