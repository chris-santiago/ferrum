"""RenderConfig — per-chart rendering policy configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RenderConfig:
    """Per-chart rendering configuration.

    Controls the auto-raster policy that transparently substitutes
    ``mark_raster`` for per-element marks when the mark count exceeds
    ``raster_threshold``. This prevents multi-million-row charts from
    producing impractically large SVG output.

    For one-off overrides, prefer the ``raster=`` keyword on output
    methods (``chart.show(raster=False)``, ``chart.save(..., raster=False)``,
    ``chart.show_svg(raster=False)``).  Use ``RenderConfig`` when you want
    to bake the policy into the chart object itself.

    Parameters
    ----------
    raster_threshold : int or None, default 500_000
        Mark count above which auto-raster fires. ``None`` disables
        auto-raster entirely.
    raster_behavior : {"warn", "silent", "error"}, default "warn"
        ``"warn"`` emits a ``UserWarning`` when auto-raster substitutes.
        ``"silent"`` substitutes without warning.
        ``"error"`` raises ``ValueError`` instead of substituting.
    raster_aggregate : str, default "count"
        Aggregation function for the substituted ``mark_raster``.
    raster_cmap : str, default "viridis"
        Colormap for the substituted ``mark_raster``.
    """

    raster_threshold: Optional[int] = 500_000
    raster_behavior: str = "warn"
    raster_aggregate: str = "count"
    raster_cmap: str = "viridis"
