"""Scale-to-dict conversion helper for encoding channels (internal)."""

from __future__ import annotations

from typing import Any


def _scale_to_dict(scale: Any) -> Any:
    """Convert a Python Scale object to a JSON-serializable dict.

    Converts LogScale, LinearScale, TimeScale, SymlogScale, and OrdinalScale
    instances to the dict shape expected by Rust's ScaleSpec serde deserialiser.
    If ``scale`` is already a dict, ensure it has a ``type`` key (defaulting to
    ``"linear"`` when absent) so Rust's tagged-enum deserialiser can match the
    correct variant.  ``None`` is returned unchanged.
    """
    if scale is None:
        return scale
    if isinstance(scale, dict):
        if "type" not in scale:
            return {"type": "linear", **scale}
        return scale

    # Import here to avoid circular imports at module load time.
    try:
        from ferrum._core import (  # type: ignore[attr-defined]
            LogScale,
            LinearScale,
            TimeScale,
            SymlogScale,
            OrdinalScale,
        )
    except ImportError:
        return scale  # can't convert, pass through and let Rust raise

    if isinstance(scale, LogScale):
        d: dict = {"type": "log", "base": scale.base, "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, LinearScale):
        d = {"type": "linear", "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, TimeScale):
        d = {"type": "time", "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, SymlogScale):
        d = {"type": "symlog", "constant": scale.constant, "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, OrdinalScale):
        d = {"type": "ordinal", "padding": scale.padding}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        return d

    # Unknown scale type — return as-is and let Rust surface the error.
    return scale
