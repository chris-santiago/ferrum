"""AST-level regression guard: no ``desugar_*`` function may silently
drop kwargs.

Every ``desugar_<name>`` function in ``src/ferrum/marks/`` must satisfy
ONE of:

1. **No ``**kwargs`` parameter at all** — Python's TypeError for unknown
   kwargs is the enforcement (the cleanest fix, used by most desugars).

2. **Calls ``validate_user_mark_kwargs(...)``** in its body — explicitly
   validates user-supplied kwargs against the renderer-level allowlist
   and raises ``TypeError`` on unknown keys. Used by the four 10e
   desugars that *also* forward known visual kwargs into their layers.

If a future contributor adds a new ``desugar_<name>(*, **kwargs)`` that
silently drops unknown keys, this test fails. The fix is to either
remove ``**kwargs`` (preferred) or wire up validation via
``ferrum.marks._mark_kwargs.validate_user_mark_kwargs``.

The guarantee applies to every module that contributes desugars:
``composite.py``, ``diagnostic.py``, ``heavy_stat.py``, ``statistical.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MARKS_DIR = Path(__file__).parent.parent / "src" / "ferrum" / "marks"
_MARK_FILES = (
    "composite.py",
    "heavy_stat.py",
    "statistical.py",
)


def _find_desugar_functions() -> list[tuple[str, ast.FunctionDef]]:
    """Walk every marks/*.py and marks/diagnostic/*.py module and yield
    (qualname, FunctionDef) pairs for every top-level function named
    ``desugar_*``.
    """
    out: list[tuple[str, ast.FunctionDef]] = []
    # Top-level mark modules.
    for fname in _MARK_FILES:
        path = _MARKS_DIR / fname
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("desugar_"):
                out.append((f"{fname}::{node.name}", node))
    # Diagnostic subpackage domain modules.
    diag_dir = _MARKS_DIR / "diagnostic"
    if diag_dir.is_dir():
        for path in sorted(diag_dir.glob("_*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("desugar_"):
                    out.append((f"diagnostic/{path.name}::{node.name}", node))
    return out


def _has_var_kwargs(fn: ast.FunctionDef) -> bool:
    return fn.args.kwarg is not None


def _calls_validator(fn: ast.FunctionDef) -> bool:
    """True if the function body contains a call to
    ``validate_user_mark_kwargs`` (or its imported alias).
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        # Direct call: validate_user_mark_kwargs(...) or alias _validate(...).
        if isinstance(target, ast.Name) and target.id in {
            "validate_user_mark_kwargs", "_validate",
        }:
            return True
        # Attribute call: _mark_kwargs.validate_user_mark_kwargs(...).
        if isinstance(target, ast.Attribute) and target.attr == "validate_user_mark_kwargs":
            return True
    return False


_DESUGARS = _find_desugar_functions()


def test_marks_dir_discovered():
    """Sanity: the AST walker found a non-trivial set of desugars."""
    assert len(_DESUGARS) >= 25, (
        f"Expected ≥ 25 desugar functions across the marks package; "
        f"found {len(_DESUGARS)}. The discovery glob may be broken."
    )


@pytest.mark.parametrize("qualname,fn", _DESUGARS, ids=[q for q, _ in _DESUGARS])
def test_desugar_does_not_silently_drop_kwargs(qualname: str, fn: ast.FunctionDef):
    """Each desugar must either reject unknown kwargs at the call boundary
    (no ``**kwargs``) or explicitly validate them via
    ``validate_user_mark_kwargs``.
    """
    if not _has_var_kwargs(fn):
        return  # Path 1: Python's TypeError is the enforcement. Pass.
    if _calls_validator(fn):
        return  # Path 2: explicit validation. Pass.
    pytest.fail(
        f"{qualname}: accepts **{fn.args.kwarg.arg} but does not call "
        f"validate_user_mark_kwargs. Either drop the **kwargs parameter "
        f"(preferred) so Python raises TypeError on unknown keys at the "
        f"call boundary, or import "
        f"`ferrum.marks._mark_kwargs.validate_user_mark_kwargs` and call "
        f"it on the kwargs. Silent-drop is forbidden per the Phase 9+ "
        f"no-defer principle."
    )


def test_meta_test_rejects_silent_drop_function():
    """Negative control: feed the meta-test logic a synthetic desugar
    that accepts **kwargs without validating and verify it would fail.

    Without this, the meta-test could pass on a vacuously-empty rule
    (e.g. someone refactors away the `_calls_validator` branch and the
    test would silently let everything through). Verifying both arms of
    the discriminator on a controlled input keeps the test honest.
    """
    silent_drop = ast.parse(
        "def desugar_evil(*, x, **kwargs):\n    return ()\n"
    ).body[0]
    explicit_validate = ast.parse(
        "def desugar_good(*, x, **kwargs):\n"
        "    _validate('good', kwargs)\n"
        "    return ()\n"
    ).body[0]
    no_kwargs = ast.parse(
        "def desugar_clean(*, x):\n    return ()\n"
    ).body[0]

    assert _has_var_kwargs(silent_drop) and not _calls_validator(silent_drop)
    assert _has_var_kwargs(explicit_validate) and _calls_validator(explicit_validate)
    assert not _has_var_kwargs(no_kwargs)


def test_validator_helper_is_importable():
    """The shared validation helper exists at the expected path so future
    desugars have one canonical place to import from.
    """
    from ferrum.marks._mark_kwargs import (
        apply_user_mark_kwargs,
        validate_user_mark_kwargs,
    )

    assert callable(validate_user_mark_kwargs)
    assert callable(apply_user_mark_kwargs)
    # Round-trip: empty dict passes through; unknown raises.
    assert validate_user_mark_kwargs("foo", {}) == {}
    with pytest.raises(TypeError, match="not_a_real_kwarg"):
        validate_user_mark_kwargs("foo", {"not_a_real_kwarg": 1})
