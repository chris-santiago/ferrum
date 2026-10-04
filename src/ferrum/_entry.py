"""The one stamp of the public entry a caller invoked onto the chart it got back."""

from __future__ import annotations

from typing import Any


def stamp_entry(chart: Any, entry: str) -> Any:
    """Return *chart* with every leaf chart's layers naming *entry*.

    Shared by every figure function (``ferrum.plots._helpers._figure_entry``)
    and every visualizer (``FerrumVisualizer.show``), so warnings and
    refusals about any layer speak as the call the user made. See
    ``Chart._with_entry`` for what a leaf records and checks.
    """
    from ferrum.chart import Chart

    if isinstance(chart, Chart):
        return chart._with_entry(entry)
    return chart._rebuild_with_charts(lambda child: stamp_entry(child, entry))
