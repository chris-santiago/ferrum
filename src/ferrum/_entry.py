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
    """

    def __init__(self, message_for: Callable[[str], str], entry: str) -> None:
        super().__init__(message_for(entry))
        self.message_for = message_for

    def spoken_as(self, entry: str) -> "EntryRefusal":
        """The same refusal, voiced as *entry*."""
        return EntryRefusal(self.message_for, entry)


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
