"""Typing-only contracts for the ``ModelSource`` mixin family.

Two protocols, both consumed only by static type checkers (``pyright`` /
``mypy``); neither changes runtime behavior, MRO, or construction:

``_SourceState``
    The shared instance state + small helpers that every per-domain mixin
    reads off ``self`` (``self._X``, ``self._cache_key(...)``, …). The
    mixins are method-only and rely on :class:`BaseSource` providing these
    at runtime (``BaseSource`` is last in the ``ModelSource`` MRO). Without
    a declared contract, pyright cannot resolve ``self._X`` and emits a
    ``reportAttributeAccessIssue`` warning for every access (GitHub #30,
    the C2 pyright debt). Each mixin selects ``_SourceState`` as its static
    base under ``TYPE_CHECKING`` and ``object`` at runtime, so the accesses
    type-check while the runtime class hierarchy is untouched.

``DiagnosticSource``
    The method/return contract that the ``ferrum.plots.*`` chart builders
    consume. ``ModelSource``, ``ComparedModelSource``, and
    ``_PrecomputedSource`` are three adapters that must keep this surface in
    lockstep; annotating each against ``DiagnosticSource`` lets pyright flag
    future signature or return drift at the seam (it does not "fix" the
    existing confusion-matrix sort-order / X-surface divergences — those are
    behavior changes, logged as carries, just made visible to the checker).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl
import pyarrow as pa


class _SourceState(Protocol):
    """Shared mixin state + helpers, declared for the type checker only.

    Every per-domain mixin operates on a ``self`` that satisfies this
    protocol at runtime (provided by :class:`BaseSource`). The mixins never
    instantiate it; it exists purely so pyright resolves ``self._X`` etc.
    """

    _model: Any
    _X: pl.DataFrame
    _y: pl.Series | None
    _feature_names: list[str]
    _class_names: list[str] | None
    _sample_weight: Any
    _random_state: int | None
    _capabilities: frozenset[str]
    _cache: dict[tuple, pl.DataFrame]

    def _x_record_batch(self) -> pa.RecordBatch: ...

    def _require_capability(self, attr: str, method_name: str) -> None: ...

    def _cache_key(self, method: str, **kwargs: Any) -> tuple: ...

    # Cross-mixin methods reached through ``self`` (e.g. the classification
    # curves read ``self.probabilities()``; declared here so those calls
    # type-check across the mixin split).
    def predictions(self) -> pl.DataFrame: ...

    def probabilities(self) -> pl.DataFrame: ...


@runtime_checkable
class DiagnosticSource(Protocol):
    """Derived-data method surface the chart builders consume.

    The shared contract for the three source adapters (``ModelSource``,
    ``ComparedModelSource``, ``_PrecomputedSource``). Only the methods +
    return type (``pl.DataFrame``) are pinned here; the per-method column
    schemas are documented in ``ferrum.diagnostics._internal.schemas``.

    Marked ``runtime_checkable`` so ``isinstance(src, DiagnosticSource)``
    works for defensive checks, but the diagnostics code never relies on
    that — it is a static contract first.
    """

    # Classification curves
    def roc_curve(self, *, average: str | None = ...) -> pl.DataFrame: ...

    def pr_curve(self, *, average: str | None = ...) -> pl.DataFrame: ...

    def calibration_curve(self, *, n_bins: int = ..., strategy: str = ...) -> pl.DataFrame: ...

    def cumulative_gain(self) -> pl.DataFrame: ...

    def lift_curve(self) -> pl.DataFrame: ...

    def confusion_matrix(self, *, normalize: str | None = ...) -> pl.DataFrame: ...

    def discrimination_threshold(
        self, *, n_thresholds: int = ..., cv: Any = ...
    ) -> pl.DataFrame: ...

    # Predictions
    def predictions(self) -> pl.DataFrame: ...
