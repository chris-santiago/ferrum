"""The one stamp of the public entry a caller invoked onto the chart it got back."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable


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
    pure validator that refused, never from re-running the door. When
    *message_for* also declares :class:`_VoiceSpelledText` (its wording, not
    only the call name, changes per voice -- a bin=/aggregate= refusal
    naming a door's own keyword), the text for a voice it declares comes
    from calling it directly, pickled or not.
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
        return type(self)(_PrefixedText(prefix, self.message_for), self.entry)

    def __reduce__(self) -> tuple[Any, ...]:
        # ``message_for`` is usually a closure, which pickle refuses, and
        # BaseException's default rebuilds from ``args`` (the text alone),
        # which ``__init__`` cannot take. Rebuild from the frozen text
        # instead: the text with a slot wherever the voice goes, plus the
        # whole text for each voice whose door spells it differently, so
        # the copy keeps its type, message and entry, and re-voices like
        # the original. Subclasses take ``(message_for, entry)``, as
        # ``spoken_as`` relies on.
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
        voices = (
            self.message_for.voices
            if isinstance(self.message_for, _VoiceSpelledText)
            else frozenset()
        )
        spelled = {voice: self.message_for(voice) for voice in sorted(voices)}
        return (
            type(self),
            (_VoiceTemplate(self.message_for(slot), slot, spelled), self.entry),
            state or None,
        )


# The base marker for where the voice goes in a refusal's frozen text: a
# private-use character sequence absent from almost every refusal's own
# text. ``__reduce__`` lengthens it per-instance (and stores the lengthened
# form on the ``_VoiceTemplate``) whenever this base slot collides with text
# the refusal echoes from raw user input, so the slot it substitutes on is
# never also literal text.
_VOICE_SLOT = "\ue000voice\ue000"


@runtime_checkable
class _VoiceSpelledText(Protocol):
    """A refusal text whose wording, not only its voice slot, changes with the voice.

    ``EntryRefusal.__reduce__`` freezes a refusal's text into a slot
    template by calling ``message_for`` with a placeholder voice
    (:data:`_VOICE_SLOT`); that rests on the premise that the voice appears
    only as a literal substring of otherwise-fixed text. A message whose
    wording itself depends on the door it is read with (a bin=/aggregate=
    refusal naming that door's own keyword) breaks that premise, so it
    declares every voice its wording actually differs for, letting
    ``__reduce__`` freeze the whole text per voice instead of relying on
    the slot substitution. Structural: a message type satisfies this by
    shape alone, with no import from this module required.
    """

    @property
    def voices(self) -> frozenset[str]: ...

    def __call__(self, voice: str) -> str: ...


class _VoiceTemplate:
    """A refusal's frozen text for any voice, the picklable form of its ``message_for``.

    *spelled* holds the whole text of every voice *text*'s slot template
    cannot reproduce (:class:`_VoiceSpelledText`'s declared ``voices``,
    read once at pickle/copy time); ``__call__`` reads it before falling
    back to the slot substitution, and ``voices`` re-declares it so a
    second copy of this clone keeps the same voices.
    """

    def __init__(self, text: str, slot: str, spelled: dict[str, str] | None = None) -> None:
        self.text = text
        self.slot = slot
        self.spelled = spelled or {}

    def __call__(self, entry: str) -> str:
        if entry in self.spelled:
            return self.spelled[entry]
        return self.text.replace(self.slot, entry)

    @property
    def voices(self) -> frozenset[str]:
        return frozenset(self.spelled)


@dataclass(frozen=True)
class _PrefixedText:
    """A refusal text with *prefix* ahead of *inner*'s own text, in every voice.

    The picklable form :meth:`EntryRefusal.prefixed` builds, replacing a
    closure lambda: a plain lambda wrapper would hide *inner*'s declared
    ``voices`` from ``__reduce__``, so this forwards them instead when
    *inner* is itself :class:`_VoiceSpelledText`.
    """

    prefix: str
    inner: Callable[[str], str]

    def __call__(self, voice: str) -> str:
        return self.prefix + self.inner(voice)

    @property
    def voices(self) -> frozenset[str]:
        if isinstance(self.inner, _VoiceSpelledText):
            return self.inner.voices
        return frozenset()


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


_Checked = TypeVar("_Checked")


def _voiced(entry: str, check: Callable[[str], _Checked]) -> _Checked:
    """Return ``check(entry)``, its refusal an :class:`EntryRefusal` voiced as *entry*.

    *check* is one call of a pure validator, voiced as its argument (a
    :mod:`ferrum._validate` check, :func:`ferrum._core.check_mark_size`).
    The refusal's text for another voice comes from re-calling that same
    validator with the other voice (:func:`_check_refusal_text`), so a
    figure function or visualizer that wrote the style re-voices it as
    itself (:func:`voiced_as`) and a mark method called directly keeps its
    own. Only *check*'s own ``ValueError`` becomes the refusal. The one
    owner of this conversion: every door that turns a pure validator's
    ``ValueError`` into a re-voiceable refusal imports it from here rather
    than each defining its own.
    """
    try:
        return check(entry)
    except ValueError as err:
        raise _validator_refusal(entry, check, str(err)) from None


def _validator_refusal(entry: str, check: Callable[[str], Any], text: str) -> EntryRefusal:
    """*check*'s refusal *text*, voiced as *entry* and re-voiceable by re-calling *check*."""
    return EntryRefusal(lambda voice: _check_refusal_text(check, voice, text), entry)


def _check_refusal_text(check: Callable[[str], Any], voice: str, frozen: str) -> str:
    """The text *check*, a pure validator that refused one voice, refuses *voice* with.

    A pure validator judges the value alone, so it refuses every voice
    alike. Should *check* accept *voice*, the refusal keeps *frozen*, the
    text it was raised with, so re-voicing never turns a refusal into a
    different error.
    """
    try:
        check(voice)
    except ValueError as err:
        return str(err)
    return frozen
