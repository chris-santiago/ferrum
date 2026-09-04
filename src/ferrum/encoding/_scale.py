"""Scale-to-dict conversion helper for encoding channels (internal).

Any Rust-backed ``*Scale`` pyclass exposes ``_to_scale_spec_dict()``, which
serialises it into the canonical ``ScaleSpec`` wire dict; this module
delegates to that method.  The dict / ``Parameter`` / ``None`` paths are
unchanged.

``_scale_to_dict`` is the ONE seam every raw ``scale={...}`` dict passes
through before it becomes JSON, on BOTH wire routes: the chart-level channel
path (:func:`ferrum.encoding.base._emit_scale`, called from
``ChannelBase.to_encoding_spec_dict()``) and the layer/composite-mark path
(``SpecBuildMixin._build_layers_list``, which calls the exact same
``ch.to_encoding_spec_dict()`` method on each layer's own channel objects —
see ``src/ferrum/_spec_build.py``). Batch-C task 4 (F-L04-10) added the raw-
dict temporal-domain conversion here for that reason: a Python
``datetime.date``/``datetime.datetime`` cannot survive ``json.dumps`` at
all, and only the chart-level path used to convert it (via a Rust-side hook
at ``EncodingSpec::new``, which the layer path's ``coerce_layers`` ->
``pyo3_serde::from_py`` route never passes through) — so a temporal raw-dict
domain rendered on a bare chart but crashed with an opaque
``TypeError: Object of type date is not JSON serializable`` on any layered
or composite-mark chart. Converting here, at the one shared seam, closes
both routes in one place instead of chasing every JSON-serialization
call site individually.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

# ScaleSpec "type" tags whose `domain` is temporal (epoch-ms based).
_TEMPORAL_SCALE_TYPES = frozenset({"time", "utc"})


def _convert_temporal_domain_elements(domain: list | tuple) -> list:
    """Convert a temporal scale's raw ``domain`` elements to epoch-ms (UTC).

    Every ``datetime.date``/``datetime.datetime``/ISO-8601 string element is
    converted via :func:`ferrum.annotation.coords.temporal_coord_to_epoch_ms`
    — the canonical Python-side converter, cross-language-parity-pinned
    against Rust's own ``TimeScale(domain=...)`` extraction by
    ``tests/test_timescale_domain.py``. Elements already numeric (``float``/
    ``int``, i.e. already epoch-ms) pass through unchanged.

    This performs the CONVERSION half only. An element that is neither
    numeric nor date/datetime/str (a genuinely malformed value, e.g.
    ``object()`` or a ``bool``) is also left unchanged here rather than
    raised — refusing it with a clear "accepted forms" message stays the
    job of the downstream Rust gate
    (``crates/ferrum-core/src/spec/encoding.rs::convert_raw_dict_temporal_domain``),
    which still runs on the chart-level channel path after this conversion
    (a no-op there once the valid elements are already floats). The
    layer/composite-mark path has no equivalent Rust-side refusal today
    (only the conversion gap this function closes) — a malformed element on
    that path still surfaces as ``json.dumps``'s generic
    ``TypeError: Object of type ... is not JSON serializable``, unchanged
    from before this task and not a regression it introduces.

    An unparseable ISO string, by contrast, IS raised here (there is
    nothing downstream to catch it — a bad string still survives
    ``json.dumps`` just fine as a JSON string, so Rust's gate would never
    see it as an error at all, just a wrong epoch value). It is re-raised
    in the exact vocabulary the Rust-side hook uses for the identical
    mistake (``iso8601_string_epoch_ms`` in
    ``crates/ferrum-core/src/scale/time/mod.rs``), not
    ``temporal_coord_to_epoch_ms``'s own "annotation coordinate" wording —
    a bad ISO string in a *scale* domain is a scale-domain mistake, and the
    message must read as one coherent taxonomy regardless of which path
    (Python, here, for a raw dict's date/datetime/str elements; Rust, for
    everything else) happens to catch it, not two subsystems' worth of
    unrelated vocabulary for the same input.
    """
    from ferrum.annotation.coords import temporal_coord_to_epoch_ms

    converted = []
    for value in domain:
        if isinstance(value, (_dt.date, _dt.datetime, str)):
            try:
                converted.append(temporal_coord_to_epoch_ms(value))
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse TimeScale domain value {value!r} as an ISO-8601 date or "
                    "datetime. Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS[.ffffff][±HH:MM]'."
                ) from exc
        else:
            converted.append(value)
    return converted


def _scale_to_dict(scale: Any) -> Any:
    """Convert a Python Scale object to a JSON-serializable dict.

    If ``scale`` is a ferrum ``*Scale`` pyclass instance, delegates to its
    ``_to_scale_spec_dict()`` Rust method, which emits the canonical
    ``ScaleSpec`` serialisation.

    If ``scale`` is already a dict:

    - When ``domain`` is a :class:`~ferrum.parameter.Parameter` (a reactive
      scale domain), the literal ``domain`` key is dropped and a sibling
      ``domainParam`` carrying the parameter's name is emitted instead (D6
      reactive rescale).
    - When the dict's ``"type"`` is ``"time"``/``"utc"`` and ``domain`` is a
      list/tuple, every ``date``/``datetime``/ISO-string element is
      converted to epoch-ms via :func:`_convert_temporal_domain_elements`
      (F-L04-10, batch-C task 4) — see the module docstring for why this is
      the one shared seam for that conversion.
    - The dict is otherwise ensured to have a ``type`` key (defaulting to
      ``"linear"`` when absent).

    The caller's dict is never mutated. ``None`` is returned unchanged.

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
        scale_type = scale.get("type")
        if (
            isinstance(scale_type, str)
            and scale_type in _TEMPORAL_SCALE_TYPES
            and isinstance(domain, (list, tuple))
        ):
            scale = {**scale, "domain": _convert_temporal_domain_elements(domain)}
        if "type" not in scale:
            return {"type": "linear", **scale}
        return scale

    if hasattr(scale, "_to_scale_spec_dict"):
        return scale._to_scale_spec_dict()

    # Unknown scale type — return as-is and let Rust surface the error.
    return scale
