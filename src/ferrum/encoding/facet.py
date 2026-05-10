"""Facet encoding channels."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


class Facet(ChannelBase):
    _channel_name = "facet"
    _renders_in_phase_8a = True   # rendered via Phase 6 facet pipeline
    _honored_kwargs = frozenset(["type", "title"])


class FacetRow(ChannelBase):
    _channel_name = "facet_row"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "title"])


class FacetCol(ChannelBase):
    _channel_name = "facet_col"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "title"])
