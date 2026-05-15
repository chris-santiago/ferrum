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

    name: Optional[str] = None
    mark: Optional[str] = None
    encoding: dict = field(default_factory=dict)
    transforms: list = field(default_factory=list)
    mark_kwargs: Optional[dict] = None
    data_source: Optional[str] = None
    position: Any = None
    blend: Optional[str] = None


@dataclass(frozen=True)
class MarkDesugarResult:
    """Typed return from a desugar function consumed by ``_resolve_pending``.

    Replaces the legacy tuple protocol that used 3-tuple, 4-tuple, and 5-tuple
    shapes to signal different modes.

    Modes
    -----
    **Layered** (``layers`` is not ``None``): multi-layer chart.
    ``mark`` is ignored; ``transforms`` apply at the chart level.

    **Single-mark** (``layers`` is ``None``): single mark with optional
    encoding remap and position adjustment.
    """

    mark: Optional[str] = None
    transforms: list = field(default_factory=list)
    remap: dict = field(default_factory=dict)
    position: Any = None
    layers: Optional[list] = None
    data: Any = None  # synthetic data (e.g. desugar_function)


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
    desugar_fn: Any  # Callable[[Optional[str], Optional[str], **Any], MarkDesugarResult]
    prior_mark: str | None = None  # Existing primitive mark to preserve as a layer
