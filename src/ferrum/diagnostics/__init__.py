"""Model-diagnostics adapter + visualizer layer.

Public surface (re-exported here and from ``ferrum``):
    ferrum.ModelSource           — wrap a fitted estimator + data
    ferrum.ComparedModelSource   — multi-model comparison wrapper
    ferrum.*Visualizer           — sklearn-protocol diagnostic visualizers
                                   (see ``visualizers``)

Four-root diagnostic taxonomy
-----------------------------
Each model-diagnostic domain (classification, regression, clustering,
ranking, model-selection, explanation) has four parallel roots, keyed by
domain so siblings line up across the codebase:

- ``ferrum.plots.<domain>``                — figure functions ``f(model, X, y) -> Chart``
- ``ferrum.marks.diagnostic._<domain>``    — composite-mark desugars
- ``ferrum.diagnostics.sources._<domain>`` — derived-data adapters (``ModelSource`` mixins)
- ``ferrum.diagnostics.visualizers._<domain>`` — sklearn-protocol visualizers

Heavy sklearn-boundary internals (lazy-import gates, curve kernels, Rust
rank bridges, the precomputed-source adapter, shared schemas) live under
``ferrum.diagnostics._internal`` and are not part of the public surface.
"""

from __future__ import annotations

from .source import ComparedModelSource, ModelSource

__all__ = ["ComparedModelSource", "ModelSource"]
