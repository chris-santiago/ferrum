"""Registry of desugar parameters that are accepted, threaded through a
``desugar_*`` function's signature, and then never given effect at the
mark layer -- whether explicitly ``del``eted or simply never referenced in
the function body. Both shapes are the identical P9 defect (a declared
parameter whose caller-supplied value is silently discarded, with no
error, no warning, no effect), and the AST guard in
``tests/test_mark_kwargs_no_silent_drop.py`` treats them identically: every
declared desugar parameter must be either genuinely used, wired via a
same-method ``data_transform`` (the ``top_k`` pattern -- verified directly
from the mixin method's AST, no registry needed), or listed here.

The registry is load-bearing, not documentation: :func:`warn_informational_kwarg`
is the *only* place a mixin method may emit the "this parameter is a no-op
here" warning, and it refuses to warn about a ``(mark_name, param)`` pair
that isn't listed in :data:`INFORMATIONAL_KWARGS`. The AST guard verifies
the other half of the link -- that every registered pair has a matching
``warn_informational_kwarg(mark_name, param, ...)`` call in the
correspondingly-named mixin method -- so registry membership and the
runtime warning cannot drift apart: adding a registry entry with no call
site fails the guard, and a call site cannot reference an unregistered
pair (it raises at call time). A parameter listed here is therefore never
a silent drop: someone who reaches for it directly on the mark method is
told so instead of being met with silence.

Keyed by the mark name passed to ``Chart._set_composite_mark`` (e.g.
``"decision_boundary"`` for ``Chart.mark_decision_boundary`` /
``desugar_decision_boundary``); each value is the frozenset of that mark's
informational-only parameter names.

Four entries genuinely have no effect at the mark layer because the real
work already happened upstream, before the data ever reached the mark --
the mark-level parameter is accepted purely so a caller who reaches for it
directly (rather than through the figure function that actually gives it
effect) gets told where the effect lives, instead of silence:

- ``proba`` on ``mark_decision_boundary``: selects which grid ``z`` column
  gets computed (class index vs. predicted probability) -- decided by
  ``decision_boundary_chart``'s grid construction before the mark ever
  sees the data.
- ``n_thresholds`` on ``mark_discrimination_threshold``: the threshold
  sweep is already fixed and the data already pre-melted by the time it
  reaches this mark -- controlled by
  ``ModelSource.discrimination_threshold(n_thresholds=...)`` /
  ``discrimination_threshold_chart(n_thresholds=...)``.
- ``normalize`` on ``mark_confusion``: the cell values are already
  normalized (or not) by the time they reach this mark -- controlled by
  ``ModelSource.confusion_matrix(normalize=...)`` /
  ``confusion_matrix_chart(normalize=...)``.
- ``center`` on ``mark_pdp``: ICE polylines are already re-based to start
  at 0 by the time they reach this mark -- controlled by
  ``pdp_chart(center=...)`` (``_pdp_center_curves``, applied before the
  ``Chart`` is constructed).

All four figure functions above accept and act on their parameter, but no
longer *forward* the already-inert value into the mark method call, so the
mark-level warning fires only for a caller who passes the parameter
directly to the mark method.

``mark_boxen``'s ``palette`` parameter previously lived here as a stopgap
(it was a real, undelivered feature -- no upstream call site honored it
either). It was implemented (design-review residuals batch, #91,
2026-08-27): ``desugar_boxen`` now colors the depth bands directly from
``palette``, so it is genuinely used and no longer needs an entry here.
"""

from __future__ import annotations

INFORMATIONAL_KWARGS: dict[str, frozenset[str]] = {
    "decision_boundary": frozenset({"proba"}),
    "discrimination_threshold": frozenset({"n_thresholds"}),
    "confusion": frozenset({"normalize"}),
    "pdp": frozenset({"center"}),
}


def warn_informational_kwarg(mark_name: str, param: str, message: str) -> None:
    """Warn once that *param* on ``mark_<mark_name>`` is a documented no-op.

    This is the sole runtime consumer of :data:`INFORMATIONAL_KWARGS`: it
    refuses to warn about a pair that isn't registered, which is what makes
    the registry load-bearing rather than descriptive. Every mixin method
    that wants to warn about an informational-only parameter must call this
    helper (not ``ferrum._warn.warn_once`` directly) so the AST guard in
    ``tests/test_mark_kwargs_no_silent_drop.py`` can verify, per registry
    entry, that a matching call site exists.

    The emitted warning is attributed to the *caller's* call site, not to
    this helper or the mixin method: ``ferrum._warn.warn_once`` defaults to
    a ``stacklevel`` tuned for a single intermediary frame (its typical
    caller calls it directly), but the chain here is one frame deeper --
    user code -> ``mark_<mark_name>`` -> this helper -> ``warn_once`` --
    so this helper passes ``stacklevel=4`` to skip both ferrum frames and
    land the warning on the user's own line.

    Parameters
    ----------
    mark_name : str
        The mark name the parameter belongs to (e.g. ``"decision_boundary"``
        for ``Chart.mark_decision_boundary``), matching an
        :data:`INFORMATIONAL_KWARGS` key.
    param : str
        The informational parameter's name (e.g. ``"proba"``).
    message : str
        The warning text passed through to ``warn_once``.

    Raises
    ------
    ValueError
        If ``(mark_name, param)`` is not present in
        :data:`INFORMATIONAL_KWARGS` -- a call site cannot warn about a
        pair the registry doesn't know about.

    Examples
    --------
    >>> warn_informational_kwarg("no_such_mark", "x", "unused")
    Traceback (most recent call last):
        ...
    ValueError: warn_informational_kwarg('no_such_mark', 'x', ...) called for a pair not registered in INFORMATIONAL_KWARGS['no_such_mark'] = []. Add the entry to INFORMATIONAL_KWARGS before warning about it.
    """
    registered = INFORMATIONAL_KWARGS.get(mark_name, frozenset())
    if param not in registered:
        raise ValueError(
            f"warn_informational_kwarg({mark_name!r}, {param!r}, ...) called "
            f"for a pair not registered in INFORMATIONAL_KWARGS[{mark_name!r}] "
            f"= {sorted(registered)}. Add the entry to INFORMATIONAL_KWARGS "
            f"before warning about it."
        )
    from ferrum._warn import warn_once

    warn_once(mark_name, param, message, stacklevel=4)


__all__ = ["INFORMATIONAL_KWARGS", "warn_informational_kwarg"]
