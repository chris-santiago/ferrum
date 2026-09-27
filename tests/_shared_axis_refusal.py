"""Shared-axis `LayerCoordUnplaceable` refusal-text builders.

Extracted from ``test_layer_shared_axis_refusal.py`` so sibling test modules
(``test_layer_union_provenance.py``) can build the exact expected refusal text
without importing from another test module (the ``tests/_svg_extents.py``
precedent for cross-cutting test helpers).
"""

from __future__ import annotations

_CHANNEL = "y"


def refusal(layer_index: int, field: str, dtype: str, channel: str = _CHANNEL) -> str:
    """The exact `LayerCoordUnplaceable` text, filled in."""
    return (
        f"layer {layer_index}: column '{field}' ({dtype}) cannot be drawn on the shared "
        f"{channel} axis, which is continuous — a layered chart (fm.layer(...) / chart + chart) "
        "merges its members onto one panel. Cast the column, or give the layer its own scale "
        f'with resolve={{"{channel}": "independent"}}.'
    )


def composite_wrapped_refusal(
    layer_index: int, field: str, dtype: str, channel: str = _CHANNEL
) -> str:
    """The same exact text, wrapped in the composite route's crate-wide
    `CompositeResolveError::DomainResolve` envelope -- the shape a `fm.layer(...)`
    refusal takes, unlike `+`/`.interactive()`'s merged-flat prepare, which raises the
    bare text directly."""
    core = refusal(layer_index, field, dtype, channel)
    return f"composite scale resolution failed: failed to resolve shared {channel} domain for field '{field}': {core}"
