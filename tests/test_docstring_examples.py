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
