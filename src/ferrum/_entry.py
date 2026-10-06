"""The one stamp of the public entry a caller invoked onto the chart it got back."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable


class EntryRefusal(ValueError):
    """A refusal of the caller's style or override input, re-voiceable as any public call.

    Every style-resolution refusal (a mark kwarg, a figure's ``mark=`` or
    ``line_kws=`` dict, a composite's ``layer_kwargs=``) is one. The door
    that refuses may run inside a figure function or visualizer, before the
    public entry is known, so the refusal carries *message_for*, its text
    for any entry, and is raised in the door's own voice (``mark_point``
    for a mark method called directly). :func:`voiced_as` re-raises it in
    the voice of the call the user made (``roc_chart: keyword argument
    'filled' is not honored by mark_line ...``). A figure that builds
    through another figure re-voices it on the way out, so the outermost
    call names it.

    *entry* must name a call; an empty one refuses. The text for another
    voice comes from *message_for* alone, a template or a re-call of the one
    pure validator that refused, never from re-running the door.
    """

    def __init__(self, message_for: Callable[[str], str], entry: str) -> None:
        # A refusal always names a public call: an empty voice would leak a
        # message that starts with ``(mark=...)``, naming nothing.
        if not entry:
            raise ValueError("EntryRefusal needs the public call it speaks as; got an empty entry")
        super().__init__(message_for(entry))
        self.message_for = message_for
        self.entry = entry

    def spoken_as(self, entry: str) -> "EntryRefusal":
        """The same refusal, of the same type, voiced as *entry*."""
        return type(self)(self.message_for, entry)

    def prefixed(self, prefix: str) -> "EntryRefusal":
        """The same refusal, of the same type, with *prefix* ahead of its text in every voice."""
        message_for = self.message_for
        return type(self)(lambda entry: f"{prefix}{message_for(entry)}", self.entry)

    def __reduce__(self) -> tuple[Any, ...]:
        # ``message_for`` is usually a closure, which pickle refuses, and
        # BaseException's default rebuilds from ``args`` (the text alone),
        # which ``__init__`` cannot take. Rebuild from the frozen text
        # instead: the text with a slot wherever the voice goes, so the copy
        # keeps its type, message and entry, and re-voices like the original.
        # Subclasses take ``(message_for, entry)``, as ``spoken_as`` relies on.
        #
        # A refusal can echo raw, un-escaped user input (a color value, a
        # kwarg name), so the base slot is not guaranteed absent from this
        # instance's own text. Lengthen it until it is: `str(self)` is
        # `self.message_for(self.entry)`, so a slot missing from that text
        # cannot collide with anything the template later substitutes.
        state = {k: v for k, v in self.__dict__.items() if k not in ("message_for", "entry")}
        slot = _VOICE_SLOT
        own_text = self.message_for(self.entry)
        while slot in own_text:
            slot += _VOICE_SLOT
        return (
            type(self),
            (_VoiceTemplate(self.message_for(slot), slot), self.entry),
            state or None,
        )


# The base marker for where the voice goes in a refusal's frozen text: a
# private-use character sequence absent from almost every refusal's own
# text. ``__reduce__`` lengthens it per-instance (and stores the lengthened
# form on the ``_VoiceTemplate``) whenever this base slot collides with text
# the refusal echoes from raw user input, so the slot it substitutes on is
# never also literal text.
_VOICE_SLOT = "\ue000voice\ue000"


class _VoiceTemplate:
    """A refusal's frozen text for any voice, the picklable form of its ``message_for``."""

    def __init__(self, text: str, slot: str) -> None:
        self.text = text
        self.slot = slot

    def __call__(self, entry: str) -> str:
        return self.text.replace(self.slot, entry)


def stamp_entry(chart: Any, entry: str) -> Any:
    """Return *chart* with every leaf chart's layers naming *entry*.

    See ``Chart._with_entry`` for what a leaf records.
    """
    from ferrum.chart import Chart

    if isinstance(chart, Chart):
        return chart._with_entry(entry)
    return chart._rebuild_with_charts(lambda child: stamp_entry(child, entry))


@contextmanager
def voiced_as(entry: str) -> Iterator[None]:
    """Re-raise every :class:`EntryRefusal` raised in the block in *entry*'s voice."""
    try:
        yield
    except EntryRefusal as err:
        raise err.spoken_as(entry) from None


def speak_as(entry: str, build: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Build a chart with ``build(*args, **kwargs)`` as the public call *entry*.

    Every refusal the build raises as :class:`EntryRefusal` names *entry*
    (:func:`voiced_as`), and every layer of the chart it returns does too
    (:func:`stamp_entry`), so warnings and refusals about the result speak
    as the call the user made. The wrapper of every figure function
    (``ferrum.plots._helpers._figure_entry``).
    """
    with voiced_as(entry):
        chart = build(*args, **kwargs)
    return stamp_entry(chart, entry)
