"""``mark_discrimination_threshold``'s two flag defaults have one owner.

``optimum_label`` and ``threshold_line`` are answered one step apart -- the
mark method resolves them to decide whether to inject the *sentinel columns*
a layer would read, ``ferrum._desugar`` resolves them to decide the
explicit-vs-default *disposition* when those columns are missing, and
``desugar_discrimination_threshold``'s own signature restates them a third
time. They used to be spelled at all three sites under a hand-written "keep
both defaults in sync if they ever change" note (#189; whole-wave design
review finding 3, 2026-09-17).

Homed here rather than in ``tests/diagnostics/``: the constant lives in
``ferrum.marks._desugar_helpers`` and every consumer is a marks/desugar
surface -- the diagnostics figure function is one caller of that mark, not
the owner of these values (cycle-2 design review, placement note). The
``_UNSET`` sentinel tests travel with them for the same reason: the sentinel
is ``ferrum._desugar``'s, and it is what makes the explicit-vs-default split
expressible at all.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum


_WITH_F1_LONG_DF = pl.DataFrame(
    {
        "threshold": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "metric": ["precision"] * 3 + ["f1"] * 3,
        "value": [0.5, 0.6, 0.7, 0.4, 0.9, 0.3],
    }
)


def test_the_discrimination_flag_defaults_have_exactly_one_owner():
    """The two consumers of these defaults must read the same constant.

    ``optimum_label``/``threshold_line`` are answered one step apart -- the
    mark method resolves them to decide whether to inject the sentinel
    *columns*, ``ferrum._desugar`` resolves them to decide the
    explicit-vs-default *disposition* when those columns are missing -- and
    they used to be spelled at both sites under a hand-written "keep both
    defaults in sync if they ever change" note (whole-wave design review
    finding 3). A third restatement sat in
    ``desugar_discrimination_threshold``'s own signature. All three now
    read ``DISC_FLAG_DEFAULTS``; this asserts they agree rather than
    trusting the note.
    """
    import inspect

    from ferrum._desugar import _UNSET, resolve_disc_flag
    from ferrum.marks._desugar_helpers import DISC_FLAG_DEFAULTS
    from ferrum.marks.diagnostic import desugar_discrimination_threshold

    assert set(DISC_FLAG_DEFAULTS) == {"optimum_label", "threshold_line"}
    signature = inspect.signature(desugar_discrimination_threshold)
    for flag, default in DISC_FLAG_DEFAULTS.items():
        assert resolve_disc_flag(flag, _UNSET) is default
        assert resolve_disc_flag(flag, True) is True
        assert resolve_disc_flag(flag, False) is False
        assert signature.parameters[flag].default is default


def test_flipping_the_flag_default_owner_moves_both_consumers(monkeypatch):
    """Discriminating check that both consumers really read the owner.

    ``threshold_line`` defaults to ``False``, so nothing draws the rule.
    Flipping the owner's entry must make the mark method inject
    ``_threshold_best`` *and* the desugar emit the rule layer that reads
    it: if only one consumer read the constant, this would render a layer
    with no data behind it (or data with no layer), which is exactly the
    failure the sync note was there to prevent.
    """
    from ferrum.marks._desugar_helpers import DISC_FLAG_DEFAULTS

    before = ferrum.Chart(_WITH_F1_LONG_DF).mark_discrimination_threshold().to_svg()
    assert 'stroke-dasharray="4,4"' not in before

    monkeypatch.setitem(DISC_FLAG_DEFAULTS, "threshold_line", True)
    after = ferrum.Chart(_WITH_F1_LONG_DF).mark_discrimination_threshold().to_svg()
    assert 'stroke-dasharray="4,4"' in after


def test_the_unset_sentinel_keeps_its_identity_through_copies():
    """``_UNSET`` is compared with ``is``, so a copy must be the same object.

    A copied chart whose flag round-tripped into a second ``_UnsetType``
    instance would read as an *explicit* truthy value, flipping #189's
    disposition from "silently downgrade" to "refuse" on a chart nobody
    changed (whole-wave design review finding 8; measured 2026-09-17:
    ``copy.deepcopy(_UNSET) is _UNSET`` was ``False`` before the singleton
    ``__new__``).
    """
    import copy
    import pickle  # round-trips this test's own sentinel; no external input

    from ferrum._desugar import _UNSET, _UnsetType

    assert _UnsetType() is _UNSET
    assert copy.copy(_UNSET) is _UNSET
    assert copy.deepcopy(_UNSET) is _UNSET
    assert pickle.loads(pickle.dumps(_UNSET)) is _UNSET
    assert copy.deepcopy({"optimum_label": _UNSET})["optimum_label"] is _UNSET
