"""Scale-to-dict conversion helper for encoding channels (internal).

Each ferrum ``*Scale`` pyclass (``BandScale``, ``BinOrdinalScale``,
``DivergingScale``, ``LinearScale``, ``LogScale``, ``OrdinalScale``,
``PointScale``, ``PowScale``, ``QuantileScale``, ``QuantizeScale``,
``SequentialScale``, ``SqrtScale``, ``SymlogScale``, ``ThresholdScale``,
``TimeScale``) is a Rust-backed PyO3 class that exposes
``_to_scale_spec_dict()`` — a method that serialises the instance into the
canonical ``ScaleSpec`` wire dict.  This module delegates to that method for
all pyclass instances; the dict / ``Parameter`` / ``None`` paths are
unchanged.
"""

from __future__ import annotations

from typing import Any


def _scale_to_dict(scale: Any) -> Any:
    """Convert a Python Scale object to a JSON-serializable dict.

    If ``scale`` is a ferrum ``*Scale`` pyclass instance, delegates to its
    ``_to_scale_spec_dict()`` Rust method, which emits the canonical
    ``ScaleSpec`` serialisation.

    If ``scale`` is already a dict, ensures it has a ``type`` key (defaulting
    to ``"linear"`` when absent).  When the dict's ``domain`` is a
    :class:`~ferrum.parameter.Parameter` (a reactive scale domain), the
    literal ``domain`` key is dropped and a sibling ``domainParam`` carrying
    the parameter's name is emitted instead (D6 reactive rescale).  The
    caller's dict is never mutated.  ``None`` is returned unchanged.

    Unknown objects are returned as-is; Rust will raise if they are not
    serialisable.
    """
    if scale is None:
        return scale
    if isinstance(scale, dict):
        from ferrum.parameter import Parameter

        domain = scale.get("domain")
        if isinstance(domain, Parameter):
            out = {k: v for k, v in scale.items() if k != "domain"}
            out.setdefault("type", "linear")
            out["domainParam"] = domain.name
            return out
        if "type" not in scale:
            return {"type": "linear", **scale}
        return scale

    if hasattr(scale, "_to_scale_spec_dict"):
        return scale._to_scale_spec_dict()

    # Unknown scale type — return as-is and let Rust surface the error.
    return scale
