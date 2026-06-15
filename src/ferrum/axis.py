"""Axis value class for per-channel axis configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ferrum._title_sentinel import TitleParam, _UNSET, serialize_title


# Default values that should be omitted from serialization (they match
# the renderer's built-in defaults and would add noise to the spec).
_AXIS_DEFAULTS: dict[str, Any] = {
    "ticks": True,
    "tick_extra": False,
    "grid": True,
    "labels": True,
    "label_flush": False,
    "label_overlap": "greedy",
    "domain": True,
}

# Fields that exist only in the Python layer and must not be forwarded to
# the Rust renderer (they have no corresponding key in EncodingSpec.axis.extra).
_PYTHON_ONLY_FIELDS: frozenset[str] = frozenset({"label_map"})


@dataclass(frozen=True, slots=True)
class Axis:
    """Per-channel axis configuration.

    Parameters
    ----------
    title : str, optional
        Axis title text.  Pass ``title=None`` to suppress the axis title
        (the field name will not be shown).  Omitting ``title`` entirely keeps
        the field-name default.
    orient : str, optional
        Axis orientation ("top", "bottom", "left", "right").
    ticks : bool
        Show tick marks.
    tick_count : int, optional
        Suggested number of ticks.
    tick_extra : bool
        Include extra tick at domain boundary.
    tick_min_step : float, optional
        Minimum step between ticks.
    grid : bool
        Show grid lines.
    grid_dash : list[float], optional
        Grid line dash pattern.
    grid_width : float, optional
        Grid line width.
    grid_color : str, optional
        Grid line color.
    grid_opacity : float, optional
        Grid line opacity.
    labels : bool
        Show tick labels.
    label_angle : float, optional
        Tick label rotation angle.
    label_flush : bool
        Flush the first and last tick labels against the axis ends so they do
        not overhang the plot area.  Defaults to ``False`` (no flush), matching
        the renderer default.
    label_overlap : str
        Label overlap strategy ("greedy", "parity", "rotate").  Defaults to
        ``"greedy"`` (the renderer's graduated collision cascade).
    label_format : str, optional
        d3-format string for labels.
    label_format_type : str, optional
        Format type ("number", "time").
    label_font_size : float, optional
        Label font size.
    label_color : str, optional
        Label color.
    domain : bool
        Show axis domain line.
    domain_width : float, optional
        Domain line width.
    domain_color : str, optional
        Domain line color.
    offset : float, optional
        Axis offset from plot area.
    translate : float, optional
        Pixel translation.
    min_extent : float, optional
        Minimum axis extent.
    max_extent : float, optional
        Maximum axis extent.
    title_orient : str, optional
        Title orientation.
    title_font_size : float, optional
        Title font size.
    title_color : str, optional
        Title color.
    title_padding : float, optional
        Title padding from axis.
    values : list, optional
        Explicit tick values.
    zindex : int, optional
        Z-index for layering.
    label_map : dict[str, str], optional
        Mapping from original tick-label text to display text.  Applied in the
        Python layer at render time by renaming the corresponding column values
        in the DataFrame before Rust computes the scale domain.  Keys not
        present in the data are silently ignored.

        Example — rename categorical axis labels::

            fm.Axis(label_map={"a": "Group A", "b": "Group B"})

    Examples
    --------
    >>> import ferrum as fm
    >>> ax = fm.Axis(title="Speed (km/h)", label_angle=-45)
    >>> ax.title
    'Speed (km/h)'
    >>> ax.to_dict()
    {'title': 'Speed (km/h)', 'label_angle': -45}
    """

    title: TitleParam = _UNSET
    orient: str | None = None
    ticks: bool = True
    tick_count: int | None = None
    tick_extra: bool = False
    tick_min_step: float | None = None
    grid: bool = True
    grid_dash: list[float] | None = None
    grid_width: float | None = None
    grid_color: str | None = None
    grid_opacity: float | None = None
    labels: bool = True
    label_angle: float | None = None
    label_flush: bool = False
    label_overlap: str = "greedy"
    label_format: str | None = None
    label_format_type: str | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    domain: bool = True
    domain_width: float | None = None
    domain_color: str | None = None
    offset: float | None = None
    translate: float | None = None
    min_extent: float | None = None
    max_extent: float | None = None
    title_orient: str | None = None
    title_font_size: float | None = None
    title_color: str | None = None
    title_padding: float | None = None
    values: list | None = None
    zindex: int | None = None
    label_map: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for the renderer, omitting None, defaults, and Python-only fields."""
        result: dict[str, Any] = {}
        for f in fields(self):
            # Python-only fields have no Rust counterpart — never forward them.
            if f.name in _PYTHON_ONLY_FIELDS:
                continue
            val = getattr(self, f.name)
            if f.name == "title":
                # Three-way title contract (mirrors base.py and prepare.rs):
                #   _UNSET  → omit key; Rust falls back to field name (default)
                #   None    → emit ""; Rust treats "" as suppress
                #   "Foo"   → emit "Foo" verbatim
                serialized = serialize_title(val)
                if serialized is not None:
                    result["title"] = serialized
                continue
            # Skip None values for all other fields
            if val is None:
                continue
            # Skip values that match the renderer default
            if f.name in _AXIS_DEFAULTS and val == _AXIS_DEFAULTS[f.name]:
                continue
            result[f.name] = val
        return result


def _axis_suppressed_dict() -> dict[str, Any]:
    """Return the dict equivalent of axis=False (suppress all axis elements)."""
    return {
        "domain": False,
        "ticks": False,
        "labels": False,
        "title": None,
        "grid": False,
    }


def _normalize_axis(value: Any) -> dict[str, Any] | None:
    """Normalize an axis kwarg value to a dict or None.

    Accepts:
    - Axis instance -> .to_dict()
    - False -> suppression dict
    - dict -> pass through
    - None -> None (meaning "not specified")
    """
    if value is None:
        return None
    if value is False:
        return _axis_suppressed_dict()
    if isinstance(value, Axis):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return None
