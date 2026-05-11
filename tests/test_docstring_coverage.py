"""Ratcheting docstring coverage test.

Symbols in _DOC_ALLOWLIST must have a non-empty __doc__. The allowlist grows
commit-by-commit during the docstring sweep (see
docs/superpowers/specs/2026-05-11-docstrings-design.md §7). After the final
sweep commit the allowlist covers all of ferrum.__all__ except the three
namespace re-exports (themes, encoding, figure).
"""
from __future__ import annotations

import pytest

import ferrum

# Symbols required to have a non-empty __doc__. Grows commit-by-commit.
# DO NOT add symbols here unless their docstrings have actually landed.
_DOC_ALLOWLIST: set[str] = {
    # Task 2 — top-level Chart class
    "Chart",
}

# Namespace re-exports — exempt forever (they're submodules, not symbols).
_NAMESPACE_EXEMPT: set[str] = {"themes", "encoding", "figure"}


def test_allowlist_symbols_have_docstrings() -> None:
    """Every symbol in _DOC_ALLOWLIST must have a non-empty __doc__."""
    missing: list[str] = []
    for name in sorted(_DOC_ALLOWLIST):
        obj = getattr(ferrum, name, None)
        assert obj is not None, f"ferrum.{name} not found"
        doc = getattr(obj, "__doc__", None)
        if not doc or not doc.strip():
            missing.append(name)
    assert not missing, f"Missing docstrings on: {missing}"


def test_allowlist_covers_all_public_api_after_sweep() -> None:
    """Final assertion: after the final sweep commit the allowlist guards all of __all__.

    Skipped while the allowlist is incomplete. Once the final sweep commit lands, this
    test starts running and stays as a permanent guardrail.
    """
    expected = set(ferrum.__all__) - _NAMESPACE_EXEMPT
    if not _DOC_ALLOWLIST >= expected:
        missing = expected - _DOC_ALLOWLIST
        pytest.skip(f"Allowlist incomplete (sweep in progress); missing: {sorted(missing)}")
    assert _DOC_ALLOWLIST >= expected
