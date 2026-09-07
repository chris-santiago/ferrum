"""Regression: docstring examples in modules nothing else exercises.

Batch D T14 fixed five latent failing doctests in
``ferrum.marks._chart_methods_statistical`` (mark_boxen / mark_boxplot /
mark_errorband / mark_errorbar / mark_histogram expected
``Chart(mark='point', encoding=[])`` where the real repr is
``encoding=['x', 'y']``) and added a grammar doctest to
``ferrum.transforms``. Neither module is covered by ``--doctest-modules``
in ``addopts`` or by the nox doctest session, so those corrections could
silently re-rot. This module runs both modules' doctests inside the
regular suite so a drifting example fails here.
"""

from __future__ import annotations

import doctest
import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "ferrum.marks._chart_methods_statistical",
        "ferrum.transforms",
    ],
)
def test_module_doctests_pass(module_name):
    """Regression: the T14-corrected doctest outputs must keep matching
    real behavior — a drifted example (or a behavior change without a doc
    update) fails loudly here instead of rotting unseen."""
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False)
    assert result.attempted > 0, f"{module_name} has no doctests to run"
    assert result.failed == 0, f"{module_name}: {result.failed}/{result.attempted} doctests failed"


def test_chart_module_doctest_failure_set_does_not_grow():
    """Regression: ``ferrum.chart`` carries 14 known-bad doctests under
    ``doctest.testmod`` (155 examples total; pre-existing failures — mostly
    ``NameError: name 'fm' is not defined`` examples and unset-up composition
    snippets — tracked, not batch E's). Measured A/B across batch E T15's
    docstring-only edits (mark_geoshape's typed refusal, ``Chart.coord()``'s
    accepted types, mark_tick's precedence): 14/155 both before and after,
    proving those edits doctest-neutral. (The narrower
    ``pytest --doctest-modules`` harness collects only a 39-example subset
    with 10 failures — this test deliberately pins the wider testmod corpus.)
    This pins the failure count at the known baseline so a NEW rotting
    example — a future docstring edit or a behavior change without a doc
    update — fails loudly here instead of hiding among the pre-existing
    failures.
    """
    import doctest
    import importlib

    module = importlib.import_module("ferrum.chart")
    result = doctest.testmod(module, verbose=False)
    assert result.attempted >= 100, "ferrum.chart's doctest corpus shrank unexpectedly"
    assert result.failed <= 14, (
        f"ferrum.chart doctest failures grew past the known-bad baseline of 14: "
        f"{result.failed}/{result.attempted} failed — a docstring example rotted"
    )
