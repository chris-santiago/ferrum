"""Internal _Layer value type — frozen dataclass shared between Chart and the
mark-desugar modules (composite, heavy_stat, statistical, diagnostic).

The wire format emitted to Rust by ``Chart._build_layers_list`` still uses
``mark_style`` for backward compat with ``coerce_layers``; ``_Layer``
canonicalises on ``mark_kwargs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class _Layer:
    """Internal layer descriptor consumed by ``Chart._build_layers_list``."""

    mark: Optional[str] = None
    encoding: dict = field(default_factory=dict)
    transforms: list = field(default_factory=list)
    mark_kwargs: Optional[dict] = None
    data_source: Optional[str] = None
    position: Any = None


@dataclass(frozen=True)
class _PendingMark:
    """Sentinel stored on ``Chart._pending_stat_mark`` when a composite or
    diagnostic ``mark_*()`` is called before ``.encode()``.

    ``Chart._resolve_pending`` calls ``desugar_fn(x_field, y_field, **kwargs)``
    once the encoding is known and threads the result back into the chart's
    transforms / layers / encoding.
    """

    kind: str
    kwargs: dict
    desugar_fn: Any  # Callable[[Optional[str], Optional[str], **Any], tuple]
    prior_mark: str | None = None  # Existing primitive mark to preserve as a layer
