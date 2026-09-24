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
        Aggregation function for the substituted ``mark_raster``, validated
        against the live ``transform_registry("raster")["methods"]``
        vocabulary (spec §4.6). Aggregates in
        ``transform_registry("raster")["methods_needing_field"]`` require
        ``raster_field`` to also be set, since they aggregate a value column
        rather than counting rows. Any other value (e.g. ``"max"``/``"min"``/
        ``"median"``, which are not valid raster aggregates) raises
        ``ValueError`` at construction.

        **Quantitative color and auto-raster.** An over-threshold chart with
        a *quantitative* ``color`` encoding is eligible for auto-raster: the
        substitution defaults to ``aggregate="mean"`` over the color field
        (plus the color channel's own ``scheme=`` when it declared one, else
        ``raster_scheme``) so the color information survives rasterisation
        instead of being discarded. Passing ``raster_aggregate``/``raster_field``
        explicitly here overrides that default wholesale -- the color field
        is then ignored and this config's values are used as-is. A
        *categorical*, *ordinal*, or *temporal* ``color`` encoding is
        excluded from auto-raster **permanently**, not merely "not yet
        implemented" -- rasterising would collapse the category to one
        summary pixel color per bin, discarding the information the color
        channel exists to show. There is no ``RenderConfig`` setting that
        lifts this exclusion; rasterize such a chart explicitly with
        ``chart.mark_raster(aggregate=..., field=...)`` instead, accepting
        that the categories collapse into a chosen numeric aggregate.
    raster_scheme : str or None, default None
        Colormap (color scheme) for the substituted ``mark_raster``.
        ``None`` defers to the built-in ``"viridis"`` default.
    raster_cmap : str or None, default None
        Back-compat alias for ``raster_scheme``.  Pass at most one of the two;
        after construction it reads back as the resolved ``raster_scheme``.
    raster_field : str or None, default None
        Column name to aggregate when ``raster_aggregate`` is one of
        ``transform_registry("raster")["methods_needing_field"]``.  Genuinely
        ignored for every other aggregate: it is dropped before the
        substitution, so it is never resolved as a column name and an
        unrelated or nonexistent value here has no effect and never raises.
        The column must be numeric; integer columns are coerced to
        ``Float64`` automatically (batch-D T2 numeric coercion), so an
        ``Int64``/``Int32``/etc. column works unchanged. A genuinely
        non-numeric column (e.g. a string column) raises ``stat_raster:
        column '...' must be numeric, got <Type>`` at render time (a
        pre-existing constraint of the underlying ``Raster`` transform, not
        new to this field) -- this is narrower than the stale
        "must be Float64" wording this docstring carried before the #136
        probe (2026-09-12): the Float64-only claim predated the coercion
        and no longer holds.
    """

    raster_threshold: Optional[int] = 500_000
    raster_behavior: str = "warn"
    raster_aggregate: str = "count"
    raster_scheme: Optional[str] = None
    raster_cmap: Optional[str] = None
    raster_field: Optional[str] = None

    def __post_init__(self) -> None:
        from ferrum._registry import require_aggregate_field, validate_aggregate
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        validate_aggregate(
            "raster", "RenderConfig", self.raster_aggregate, param="raster_aggregate"
        )
        require_aggregate_field(
            "raster",
            "RenderConfig",
            self.raster_aggregate,
            self.raster_field,
            aggregate_param="raster_aggregate",
            field_param="raster_field",
        )

        resolved = resolve_cmap_alias(
            scheme=self.raster_scheme, cmap=self.raster_cmap, where="RenderConfig"
        )
        resolved = "viridis" if resolved is None else resolved
        object.__setattr__(self, "raster_scheme", resolved)
        object.__setattr__(self, "raster_cmap", resolved)
