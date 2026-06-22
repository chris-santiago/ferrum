"""Guard the D-MOD-1 public-home contract for model-diagnostics classes.

The model-diagnostics package was promoted from the private
``ferrum._diagnostics`` to the public ``ferrum.diagnostics`` (the cohesion
campaign's one accepted breaking change). These tests pin the resulting
contract so the private path cannot creep back into a defining-module slot.
"""

from __future__ import annotations

import ferrum
from ferrum.diagnostics.visualizers import __all__ as _VISUALIZER_NAMES

# Every public model-diagnostics class: the two source wrappers plus the
# sklearn-protocol visualizers.
_DIAGNOSTIC_CLASS_NAMES = sorted({"ModelSource", "ComparedModelSource", *_VISUALIZER_NAMES})


def test_diagnostic_classes_define_in_public_package():
    """Every public diagnostic class homes under ``ferrum.diagnostics.*``."""
    offenders = {
        name: getattr(ferrum, name).__module__
        for name in _DIAGNOSTIC_CLASS_NAMES
        if not getattr(ferrum, name).__module__.startswith("ferrum.diagnostics.")
    }
    assert not offenders, f"non-public defining module: {offenders}"


def test_no_private_diagnostics_module_remains():
    """The old private package path is gone, not aliased back in."""
    import importlib

    try:
        importlib.import_module("ferrum._diagnostics")
    except ModuleNotFoundError:
        return
    raise AssertionError("ferrum._diagnostics should no longer be importable")


def test_public_import_surface_unchanged():
    """Each diagnostic class is still importable from the ``ferrum`` namespace."""
    for name in _DIAGNOSTIC_CLASS_NAMES:
        assert name in ferrum.__all__
        assert getattr(ferrum, name) is not None
