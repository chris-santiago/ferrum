"""The one stamp of the public entry a caller invoked onto the chart it got back."""

from __future__ import annotations

from typing import Any, Callable


class EntryRefusal(ValueError):
    """A refusal raised while a figure function or visualizer builds its chart.

    The door that refuses (a ``mark=`` override, a ``line_kws=`` dict) runs
    inside the build, before the public entry is known, so the refusal
    carries *message_for*, its text for any entry, and is raised in a
    provisional voice. :func:`speak_as` re-raises it in the voice of the
    call the user made (``roc_chart: keyword argument 'filled' is not
    honored by mark_line ...``). A figure that builds through another figure
    re-voices it on the way out, so the outermost call names it.
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


def speak_as(entry: str, build: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Build a chart with ``build(*args, **kwargs)`` as the public call *entry*.

    Every refusal the build raises as :class:`EntryRefusal` names *entry*,
    and every layer of the chart it returns does too (:func:`stamp_entry`),
    so warnings and refusals about the result speak as the call the user
    made. Shared by every figure function
    (``ferrum.plots._helpers._figure_entry``) and every visualizer's
    ``fit``.
    """
    try:
        chart = build(*args, **kwargs)
    except EntryRefusal as err:
        raise err.spoken_as(entry) from None
    return stamp_entry(chart, entry)
