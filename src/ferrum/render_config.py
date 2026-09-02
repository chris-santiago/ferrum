"""RenderConfig — per-chart rendering policy configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# The true accepted set for the substituted mark_raster's aggregate, per the
# Rust Raster transform (crates/ferrum-core/src/transform/raster.rs). "max" /
# "min" / "median" are NOT valid raster aggregates despite once being
# undocumented-but-reachable at render time (F-L07-10) -- construction now
# refuses them instead of deferring to a confusing runtime ValueError.
_RASTER_AGGREGATES = frozenset({"count", "density", "mean", "sum", "any"})
# Aggregates that need a value column to aggregate over.
_RASTER_AGGREGATES_NEEDING_FIELD = frozenset({"mean", "sum"})


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
    raster_aggregate : {"count", "density", "mean", "sum", "any"}, default "count"
        Aggregation function for the substituted ``mark_raster``.
        ``"mean"`` and ``"sum"`` require ``raster_field`` to also be set,
        since they aggregate a value column rather than counting rows.
        Any other value (e.g. ``"max"``/``"min"``/``"median"``, which are
        not valid raster aggregates) raises ``ValueError`` at construction.
    raster_scheme : str or None, default None
        Colormap (color scheme) for the substituted ``mark_raster``.
        ``None`` defers to the built-in ``"viridis"`` default.
    raster_cmap : str or None, default None
        Back-compat alias for ``raster_scheme``.  Pass at most one of the two;
        after construction it reads back as the resolved ``raster_scheme``.
    raster_field : str or None, default None
        Column name to aggregate when ``raster_aggregate`` is ``"mean"``
        or ``"sum"``.  Not required for ``"count"``, ``"density"``, and
        ``"any"`` -- and genuinely ignored for them: it is dropped before
        the substitution, so it is never resolved as a column name and an
        unrelated or nonexistent value here has no effect and never raises.
        The column must be ``Float64``-typed; an integer or other
        non-``Float64`` column raises ``stat_raster: column '...' must be
        Float64`` at render time (a pre-existing constraint of the
        underlying ``Raster`` transform, not new to this field).
    """

    raster_threshold: Optional[int] = 500_000
    raster_behavior: str = "warn"
    raster_aggregate: str = "count"
    raster_scheme: Optional[str] = None
    raster_cmap: Optional[str] = None
    raster_field: Optional[str] = None

    def __post_init__(self) -> None:
        from ferrum._validate import validate_choice
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        validate_choice(
            "RenderConfig", "raster_aggregate", self.raster_aggregate, _RASTER_AGGREGATES
        )
        if self.raster_aggregate in _RASTER_AGGREGATES_NEEDING_FIELD and self.raster_field is None:
            raise ValueError(
                f"RenderConfig: raster_aggregate={self.raster_aggregate!r} requires "
                f"raster_field=... (the column to aggregate); pass raster_field= or "
                f"use raster_aggregate='count'|'density'|'any', which need no field."
            )

        resolved = resolve_cmap_alias(
            scheme=self.raster_scheme, cmap=self.raster_cmap, where="RenderConfig"
        )
        resolved = "viridis" if resolved is None else resolved
        object.__setattr__(self, "raster_scheme", resolved)
        object.__setattr__(self, "raster_cmap", resolved)
