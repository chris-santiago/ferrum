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
    ``chart.to_svg(raster=False)``).  Use ``RenderConfig`` when you want
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
    raster_scheme : str or None, default None
        Colormap (color scheme) for the substituted ``mark_raster``.
        ``None`` defers to the built-in ``"viridis"`` default.
    raster_cmap : str or None, default None
        Back-compat alias for ``raster_scheme``.  Pass at most one of the two;
        after construction it reads back as the resolved ``raster_scheme``.
    """

    raster_threshold: Optional[int] = 500_000
    raster_behavior: str = "warn"
    raster_aggregate: str = "count"
    raster_scheme: Optional[str] = None
    raster_cmap: Optional[str] = None

    def __post_init__(self) -> None:
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        resolved = resolve_cmap_alias(
            scheme=self.raster_scheme, cmap=self.raster_cmap, where="RenderConfig"
        )
        resolved = "viridis" if resolved is None else resolved
        object.__setattr__(self, "raster_scheme", resolved)
        object.__setattr__(self, "raster_cmap", resolved)
