"""Regression tests for finding P7 (2026-08-27 batch): canonical choice validation.

Task 1 relocated ``_validate_choice`` from ``plots/_helpers`` to the leaf
module ``ferrum._validate``. These tests pin the two contracts the move must
preserve: the exact message template (plan §4 global constraint) and the
leaf-module import invariant (spec §7).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ferrum._validate as _validate_mod
from ferrum._validate import validate_choice


def test_validate_choice_message_exact():
    """Regression: the P7 message template is a cross-file contract.

    Existing suite assertions are substring matches, so template drift
    (losing the ``{func_name}: `` prefix or the ``; got {value!r}`` tail)
    would survive them. Pin the full string.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_choice("pairplot", "kind", "bogus", {"scatter", "kde"})
    assert str(excinfo.value) == "pairplot: kind must be one of ['kde', 'scatter']; got 'bogus'"


def test_validate_choice_accepts_member():
    validate_choice("pairplot", "kind", "kde", {"scatter", "kde"})  # no raise


def test_validate_module_is_a_leaf():
    """Regression: ``ferrum._validate`` must import nothing from ``ferrum``.

    The leaf invariant (spec §7) is what lets any package adopt it without
    inverting the dependency graph; an added ``from ferrum...`` import would
    silently reintroduce the cycle risk the relocation exists to remove.
    """
    tree = ast.parse(Path(_validate_mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0, e.g. `from . import x`) can carry
            # a `module` that doesn't start with "ferrum" (or is None for
            # `from . import x`) while still reaching back into the ferrum
            # package -- flag it directly rather than relying on name-prefix
            # matching, which misses it entirely.
            assert node.level == 0, (
                f"leaf module uses a relative import (level={node.level}): "
                f"from {'.' * node.level}{node.module or ''} import ..."
            )
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("ferrum"), f"leaf module imports {name!r}"
