"""Internal _Layer value type — frozen dataclass shared between Chart and the
mark-desugar modules (composite, heavy_stat, statistical, diagnostic).

Replaces the legacy dict shape ``{"mark": ..., "encoding": ..., "transforms": ...,
"mark_kwargs": ...|"mark_style": ..., "data_source": ..., "position": ...}`` with
explicit fields. The wire format emitted to Rust by ``Chart._build_layers_list``
still uses ``mark_style`` for backward compat with ``coerce_layers``; ``_Layer``
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


def _layer_get(layer, key: str, default=None):
    """Read a layer attribute, accepting either ``_Layer`` or the legacy dict.

    Used during the producer migration so consumers can handle both shapes.
    Once all producers emit ``_Layer``, callers can switch to plain attribute
    access and this shim is removed.
    """
    if isinstance(layer, _Layer):
        if key == "mark_style":
            # Legacy alias — _Layer canonicalises on mark_kwargs.
            return layer.mark_kwargs if layer.mark_kwargs is not None else default
        return getattr(layer, key, default)
    # dict path
    if key == "mark_kwargs":
        # Some legacy producers used "mark_style"; treat them as equivalent.
        return layer.get("mark_kwargs") or layer.get("mark_style") or default
    return layer.get(key, default)
