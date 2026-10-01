"""Shared-axis `LayerCoordUnplaceable` refusal-text builders (column and coord-exchange reasons).

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
    return composite_envelope(channel, field, refusal(layer_index, field, dtype, channel))


def composite_envelope(channel: str, field: str, core: str) -> str:
    """Wrap a layer refusal in the composite route's `DomainResolve` envelope."""
    return f"composite scale resolution failed: failed to resolve shared {channel} domain for field '{field}': {core}"


def coord_exchange_refusal(
    layer_index: int, coord: str, field: str, dtype: str, channel: str = _CHANNEL
) -> str:
    """The exact `LayerCoordUnplaceable` text for a member whose own coord exchanges
    its positional channels and lands a column on a continuous panel axis that
    cannot draw it."""
    return (
        f"layer {layer_index}: {coord} leaves this layer's positional channels in the opposite "
        f"space from the panel's axes, and the panel's {channel} axis is continuous while the "
        f"column that lands on it ('{field}': {dtype}) cannot be drawn against a continuous axis "
        "— a layered chart (fm.layer(...) / chart + chart) merges its members onto one panel, "
        "so this coord cannot be placed on one of them. The panel's axes are resolved from the "
        "first member, so a member whose channels sit in the opposite space can only be placed "
        "where both of its columns suit the axes they land on. Move it to the first member "
        "(whose coord IS the panel's), or drop it."
    )
