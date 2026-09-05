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
call site individually.  The matching REFUSAL (an element that is neither
numeric nor date/datetime/string) lives here for the same reason — see
:func:`_convert_temporal_domain_elements`.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any


def _temporal_scale_types() -> frozenset[str]:
    """Return the ``ScaleSpec`` ``"type"`` tags whose ``domain`` is temporal.

    Asked of the live schema rather than hand-listed: ``TimeScale`` is the one
    scale class carrying a temporal domain, and its ``utc=`` flag selects which
    of its two wire tags it serialises as, so the tags it can emit ARE the
    temporal tag set.  ``ferrum._core.scale_accepted_keys`` cannot answer this
    one — it publishes key NAMES, and a temporal ``domain`` is spelled
    ``domain`` exactly like every other continuous type's.
    """
    from ferrum._core import TimeScale  # type: ignore[attr-defined]

    return frozenset(TimeScale(utc=utc)._to_scale_spec_dict()["type"] for utc in (False, True))


# ScaleSpec "type" tags whose `domain` is temporal (epoch-ms based).
_TEMPORAL_SCALE_TYPES = _temporal_scale_types()

# The accepted-forms sentence Rust's `temporal_value_to_epoch_ms`
# (`crates/ferrum-core/src/scale/time/mod.rs`) raises for the same mistake.
# Kept byte-identical to it so a malformed element reads the same regardless of
# which side catches it; pinned by
# `tests/test_scale_dict_gate.py::test_non_temporal_element_message_matches_rust_wording`.
_TEMPORAL_ACCEPTED_FORMS = (
    "TimeScale domain values must be float (epoch-ms), datetime.date, "
    "datetime.datetime, or an ISO-8601 date/datetime string"
)


def _convert_temporal_domain_elements(domain: list | tuple) -> list:
    """Convert a temporal scale's raw ``domain`` elements to epoch-ms (UTC), or refuse.

    Every ``datetime.date``/``datetime.datetime``/ISO-8601 string element is
    converted via :func:`ferrum.annotation.coords.temporal_coord_to_epoch_ms`
    — the canonical Python-side converter, cross-language-parity-pinned
    against Rust's own ``TimeScale(domain=...)`` extraction by
    ``tests/test_timescale_domain.py``. Elements already numeric (``float``/
    ``int``, i.e. already epoch-ms) are coerced via ``float()`` to ensure
    consistent serialization across both wire routes (chart-level and layer).

    This seam owns the whole rule, conversion AND refusal, because it is the
    only point BOTH wire routes pass through. Rust's own
    ``convert_raw_dict_temporal_domain`` hook runs at ``EncodingSpec::new``, a
    constructor the layer/composite-mark route never enters, so delegating
    refusal to it left one user mistake with three vocabularies: Rust's
    accepted-forms ``TypeError`` on a bare chart, serde's ``invalid type:
    boolean`` on a layer, and ``json.dumps``'s generic "not JSON serializable"
    for an element serde could not even reach. An element that is neither
    numeric nor ``date``/``datetime``/``str`` is therefore raised here, in
    Rust's own accepted-forms wording (:data:`_TEMPORAL_ACCEPTED_FORMS`) and
    with its exception type (``TypeError``), so the chart-level message is
    unchanged and the layer route now matches it. ``bool`` gets Rust's
    dedicated sub-message: it is an ``int`` subclass, and silently reading
    ``True`` as ``1.0`` epoch-ms is the footgun ``TimeScale``'s own domain
    contract already refuses. Rust's hook stays in place as a belt-and-braces
    no-op on the chart-level route.

    An unparseable ISO string raises ``ValueError``, in the vocabulary Rust's
    ``iso8601_string_epoch_ms`` uses for the identical mistake rather than
    ``temporal_coord_to_epoch_ms``'s own "annotation coordinate" wording — a
    bad ISO string in a *scale* domain is a scale-domain mistake and must read
    as one.
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
        elif isinstance(value, bool):
            raise TypeError(
                f"{_TEMPORAL_ACCEPTED_FORMS}; got bool ({value}), which is not accepted "
                "as a numeric epoch-ms value"
            )
        else:
            # Duck-typed numeric check, matching Rust's `extract::<f64>()`
            # (which goes through `__float__`/`__index__`, so a numpy scalar
            # or a Decimal is a valid epoch-ms element there too). But that
            # `extract::<f64>()` argument only holds on the route that
            # reaches Rust's extractor (the chart-level path); the
            # layer/composite-mark route reaches `json.dumps` instead, which
            # cannot serialize a numpy scalar or a Decimal. Appending
            # `float(value)` widens the layer route up to the chart route
            # (both now accept the same set) rather than narrowing either:
            # Rust deserializes `1` and `1.0` into the same `f64`, so no
            # bytes move on the route that was already correct.
            try:
                converted_value = float(value)
            except (TypeError, ValueError):
                raise TypeError(f"{_TEMPORAL_ACCEPTED_FORMS}; got {type(value).__name__}") from None
            converted.append(converted_value)
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
