"""Scale-to-dict conversion helper for encoding channels (internal)."""

from __future__ import annotations

from typing import Any


def _scale_to_dict(scale: Any) -> Any:
    """Convert a Python Scale object to a JSON-serializable dict.

    Converts all ferrum scale types (LinearScale, LogScale, TimeScale,
    SymlogScale, OrdinalScale, PowScale, SqrtScale, BandScale, PointScale,
    SequentialScale, DivergingScale, QuantizeScale, BinOrdinalScale) to the
    dict shape expected by Rust's ScaleSpec serde deserialiser.

    If ``scale`` is already a dict, ensure it has a ``type`` key (defaulting to
    ``"linear"`` when absent) so Rust's tagged-enum deserialiser can match the
    correct variant.  ``None`` is returned unchanged.

    When the dict's ``domain`` is a :class:`~ferrum.parameter.Parameter` (a
    reactive scale domain), the literal ``domain`` is dropped and a sibling
    ``domainParam`` carrying the parameter's name is emitted instead (D6
    reactive rescale).  The static renderer falls back to data-inferred domains
    for empty selections; variable parameters supply their initial value via
    the params section.  The caller's dict is never mutated.
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

    # Import here to avoid circular imports at module load time.
    try:
        from ferrum._core import (  # type: ignore[attr-defined]
            BandScale,
            BinOrdinalScale,
            DivergingScale,
            LinearScale,
            LogScale,
            OrdinalScale,
            PointScale,
            PowScale,
            QuantizeScale,
            SequentialScale,
            SqrtScale,
            SymlogScale,
            TimeScale,
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
        d = {"type": "utc" if scale.utc else "time", "clamp": scale.clamp}
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
    if isinstance(scale, PowScale):
        d = {"type": "pow", "exponent": scale.exponent, "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, SqrtScale):
        d = {"type": "sqrt", "exponent": scale.exponent, "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, BandScale):
        d = {
            "type": "band",
            "paddingInner": scale.padding_inner,
            "paddingOuter": scale.padding_outer,
            "align": scale.align,
        }
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        return d
    if isinstance(scale, PointScale):
        d = {
            "type": "point",
            "padding": scale.padding,
            "align": scale.align,
            "reverse": scale.reverse,
        }
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        return d
    if isinstance(scale, SequentialScale):
        d = {"type": "sequential", "reverse": scale.reverse}
        if scale.scheme:
            d["scheme"] = scale.scheme
        if scale.domain:
            d["domain"] = list(scale.domain)
        return d
    if isinstance(scale, DivergingScale):
        d = {"type": "diverging"}
        if scale.scheme:
            d["scheme"] = scale.scheme
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.domain_mid is not None:
            d["domainMid"] = scale.domain_mid
        return d
    if isinstance(scale, QuantizeScale):
        d = {"type": "quantize"}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        return d
    if isinstance(scale, BinOrdinalScale):
        d = {"type": "bin-ordinal"}
        if scale.bins:
            d["bins"] = list(scale.bins)
        if scale.scheme:
            d["scheme"] = scale.scheme
        return d

    # Unknown scale type — return as-is and let Rust surface the error.
    return scale
