"""Cross-chart scale sharing — compute union domains and inject explicit
``scale=`` overrides onto a set of charts.

Two public callers today: ``RepeatChart`` (P2.5d, via ``resolve={...}``) and
``Chart.share_scale`` / ``Figure.shared`` (P2.8, K16).  Both reduce to the
same problem: given N charts and a channel name, compute the union extent
across every layer of every chart that binds that channel, then re-emit
each chart with a fixed scale dict (``{"type": ..., "domain": [...]}``)
inserted on the channel.  Letting the Rust renderer fill in the range from
the cell's pixel geometry keeps this module free of layout knowledge.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


def _extract_field_name(ch: Any) -> Optional[str]:
    """Return the field name bound to an encoding value, or ``None``.

    Encoding values are either bare strings (``encode(x="hp")``) or
    ``ChannelBase`` instances (``encode(x=X("hp", scale=...))``).
    ``_RepeatPlaceholder`` cannot reach here because cells run through
    ``RepeatChart._resolve_template`` first.
    """
    if isinstance(ch, str):
        return ch
    field = getattr(ch, "field", None)
    return field if isinstance(field, str) else None


def _chart_bindings(chart, channel: str) -> Iterable[Optional[str]]:
    """Yield every field name bound to *channel* across *chart*'s layers.

    Layered charts (Chart + Chart composites) keep per-layer encoding
    dicts on ``_layers``; unlayered charts keep a single ``_encoding``
    dict at the top level.
    """
    layers = getattr(chart, "_layers", None)
    if layers:
        for layer in layers:
            yield _extract_field_name(layer.encoding.get(channel))
        return
    encoding = getattr(chart, "_encoding", {}) or {}
    yield _extract_field_name(encoding.get(channel))


def _column_minmax(data, field: str) -> Optional[tuple]:
    """Return ``(min, max)`` of *field* in *data* as floats, or ``None``."""
    try:
        col = data[field]
    except (KeyError, AttributeError):
        return None
    lo, hi = col.min(), col.max()
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def _column_unique(data, field: str) -> list:
    """Return the unique values of *field* in *data* as a list, preserving
    appearance order."""
    try:
        col = data[field]
    except (KeyError, AttributeError):
        return []
    return list(col.unique().to_list())


def _classify_field(data, field: str) -> Optional[str]:
    """Return ``"linear"``, ``"ordinal"``, or ``"time"`` for *field*'s dtype.

    Returns ``None`` for unknown dtypes — caller skips sharing on that
    channel rather than guessing a scale type.
    """
    try:
        col = data[field]
    except (KeyError, AttributeError):
        return None
    dtype = col.dtype
    # Lazy import polars to avoid hard-coupling this module to polars
    # initialization order; ferrum already requires polars at runtime.
    import polars as pl

    if dtype.is_numeric():
        return "linear"
    if dtype in (pl.Datetime, pl.Date, pl.Time):
        return "time"
    if dtype in (pl.Utf8, pl.Categorical):
        return "ordinal"
    return None


def compute_union_domain(charts, channel: str) -> Optional[dict]:
    """Compute a ferrum scale dict spanning *channel* across *charts*.

    Walks every layer of every chart, collects ``(field, data)`` pairs,
    detects the scale type from the first binding's dtype, then either
    unions numeric min/max (linear) or unique values (ordinal).  Time
    domains use the same numeric union path but emit ``type="time"``.

    Parameters
    ----------
    charts : iterable of Chart
        Charts whose channel will share a domain.
    channel : str
        Encoding channel name (``"x"``, ``"y"``, ``"color"``, ...).

    Returns
    -------
    dict or None
        ``{"type": "linear" | "ordinal" | "time", "domain": [...]}``
        suitable for passing as ``scale=`` on an encoding channel.
        Returns ``None`` when no chart binds the channel, no data is
        available, or the dtype is unsupported.
    """
    bindings: list[tuple[str, Any]] = []
    for chart in charts:
        data = getattr(chart, "_data", None)
        if data is None:
            continue
        for field in _chart_bindings(chart, channel):
            if field is not None:
                bindings.append((field, data))
    if not bindings:
        return None

    first_field, first_data = bindings[0]
    scale_type = _classify_field(first_data, first_field)
    if scale_type is None:
        return None

    if scale_type in ("linear", "time"):
        lo, hi = float("inf"), float("-inf")
        for field, data in bindings:
            extent = _column_minmax(data, field)
            if extent is None:
                continue
            lo = min(lo, extent[0])
            hi = max(hi, extent[1])
        if lo == float("inf"):
            return None
        return {"type": scale_type, "domain": [lo, hi]}

    # ordinal: union of unique values, preserving first-appearance order
    seen: list = []
    seen_set: set = set()
    for field, data in bindings:
        for v in _column_unique(data, field):
            key = v
            if key not in seen_set:
                seen_set.add(key)
                seen.append(v)
    if not seen:
        return None
    return {"type": "ordinal", "domain": seen}


def inject_scale(chart, channel: str, scale_dict: dict):
    """Return a clone of *chart* with ``scale=scale_dict`` set on *channel*.

    For layered charts each layer's encoding is updated independently.
    Channels not currently bound on the chart (or on a particular layer)
    are left untouched — no implicit binding is added.
    """
    from ferrum._layer import _Layer
    from ferrum.encoding.base import ChannelBase
    from ferrum.chart import _channel_class_for

    def _set_on(value: Any) -> Any:
        if isinstance(value, ChannelBase):
            new_kwargs = dict(value._kwargs)
            new_kwargs["scale"] = scale_dict
            return type(value)(value.field, **new_kwargs)
        cls = _channel_class_for(channel)
        if cls is None:
            return value
        return cls(value, scale=scale_dict)

    new = chart._clone()
    if new._layers:
        new._layers = [
            _Layer(
                mark=layer.mark,
                encoding={
                    k: (_set_on(v) if k == channel else v) for k, v in layer.encoding.items()
                },
                transforms=layer.transforms,
                mark_kwargs=layer.mark_kwargs,
                data_source=layer.data_source,
                position=layer.position,
            )
            for layer in new._layers
        ]
    else:
        if channel in new._encoding:
            new._encoding[channel] = _set_on(new._encoding[channel])
    return new
