"""Python-side typed view over a layered ``ChartSpec``."""
from __future__ import annotations

from typing import Optional


class _SpecView:
    """Python-side typed view over a layered ``ChartSpec``.

    Exposes ``.layers`` as a list of ``types.SimpleNamespace`` items so callers
    can write ``spec.layers[0].mark.name`` and ``spec.layers[0].data_source``
    against the spec returned by ``Chart._build_spec()``. All other attribute
    access (``to_json``, ``mark``, ``encoding``, ``transforms``, etc.) and
    serialization fall through to the underlying ``ChartSpec`` instance.

    This is the Python-side typed view, not a parallel Rust type — Rust's
    ``coerce_layers`` already converts the same source dicts into ``Layer``
    structs internally during ``ChartSpec(...)`` construction.
    """

    __slots__ = ("_spec", "_layer_dicts", "_layers_cached")

    def __init__(self, spec, layer_dicts: list) -> None:
        self._spec = spec
        self._layer_dicts = layer_dicts
        self._layers_cached: Optional[list] = None

    @property
    def layers(self) -> list:
        if self._layers_cached is not None:
            return self._layers_cached
        from types import SimpleNamespace
        out = []
        for layer in self._layer_dicts:
            mark_obj = SimpleNamespace(name=layer.mark or "point")
            ns = SimpleNamespace(
                mark=mark_obj,
                encoding=layer.encoding or {},
                mark_kwargs=layer.mark_kwargs if layer.mark_kwargs else None,
                data_source=layer.data_source,
                transforms=list(layer.transforms or []),
            )
            out.append(ns)
        self._layers_cached = out
        return out

    def to_json(self, *args, **kwargs) -> str:
        return self._spec.to_json(*args, **kwargs)

    def __getattr__(self, name: str):
        # Called only if normal attribute lookup fails — delegates everything
        # else to the underlying ChartSpec.
        return getattr(self._spec, name)

    def __repr__(self) -> str:
        return f"_SpecView({self._spec!r}, layers={len(self._layer_dicts)})"
