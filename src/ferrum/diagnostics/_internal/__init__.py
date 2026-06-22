"""Heavy sklearn-boundary internals for the model-diagnostics package.

These modules are implementation details of :mod:`ferrum.diagnostics`,
not part of the public surface:

- ``deps`` — lazy-import gates for sklearn / shap (keeps ``import ferrum`` light).
- ``_curve_frames`` — sklearn-free curve kernels emitting the chart-builder frames.
- ``_rank_helpers`` — Rust ``rank1d``/``rank2d`` bridges for the ranking adapters.
- ``precomputed`` — the ``_PrecomputedSource`` adapter over already-computed arrays.
- ``schemas`` — the long-form DataFrame schemas the adapters and charts share.

Nothing here is re-exported from ``ferrum`` or ``ferrum.diagnostics``.
"""

from __future__ import annotations
