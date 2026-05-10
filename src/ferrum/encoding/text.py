"""Text/Detail/Tooltip/Href/Description/Key channels (all deferred to Phase 9)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


class Text(ChannelBase):
    _channel_name = "text"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "format", "formatType"])


class Detail(ChannelBase):
    _channel_name = "detail"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Tooltip(ChannelBase):
    _channel_name = "tooltip"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])

    def __init__(self, *fields, **kwargs):
        # Tooltip(*fields) is a special case: takes a list of fields, not just one
        if len(fields) == 1:
            super().__init__(fields[0], **kwargs)
            self._field_list = [fields[0]]
        else:
            super().__init__(None, **kwargs)
            self._field_list = list(fields)


class TooltipField(ChannelBase):
    """Helper class used inside Tooltip(*fields). Not used as a channel directly."""
    _channel_name = "tooltip_field"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "title", "format", "formatType"])


class Href(ChannelBase):
    _channel_name = "href"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Description(ChannelBase):
    _channel_name = "description"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Key(ChannelBase):
    _channel_name = "key"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
