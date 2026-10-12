"""Pytest gate: ``ferrum-spec.md`` can never silently drift from the shipped API.

``scripts/check_spec.py`` is the checker (see its module docstring for the
three modes and what counts as a dated note). This module is its pytest
gate: it runs every mode over the real ``ferrum-spec.md`` against the real
``ferrum`` package and the real ``design-docs/spec-history.md``, and it
proves the gate discriminates by injecting a wrong default into a ``/tmp``
copy of the spec and checking that the checker catches it.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import pytest

from tests._script_loader import load_script

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cs() -> ModuleType:
    return load_script("check_spec", ROOT / "scripts" / "check_spec.py")


def test_spec_contract_passes_every_mode(cs: ModuleType) -> None:
    """The real spec is clean, and all three modes actually ran.

    A mode silently dropped from ``run()`` would report no findings for it
    and look identical to a clean pass, so this also pins each mode's
    summary and that the signature mode found more than zero statements.
    """
    results = cs.run(["signatures", "notes", "history"], cs.SPEC_PATH, cs.HISTORY_PATH, [])
    summaries = [summary for _, summary in results]
    assert len(summaries) == 3, summaries
    assert summaries[0].startswith("signatures: "), summaries
    assert summaries[1].startswith("notes: "), summaries
    assert summaries[2].startswith("history: "), summaries
    stated = int(re.match(r"signatures: (\d+) stated", summaries[0]).group(1))
    assert stated > 0, summaries[0]

    messages = [f.render() for findings, _ in results for f in findings]
    assert messages == []


# The exact fenced signature this test mutates, so a spec rewrite that changes
# JointChart's stated defaults fails loudly here instead of silently defeating
# the RED proof below.
_JOINT_SIGNATURE = "JointChart(center, *, top=None, right=None, ratio=5, spacing=10.0)"
_JOINT_WRONG = "JointChart(center, *, top=None, right=None, ratio=7, spacing=10.0)"


def test_wrong_default_in_a_tmp_spec_copy_fails(cs: ModuleType, tmp_path: Path) -> None:
    """A deliberately wrong default, injected into a /tmp copy, is caught, never the real file."""
    text = cs.SPEC_PATH.read_text(encoding="utf-8")
    assert _JOINT_SIGNATURE in text, "the signature this RED proof mutates is no longer stated"
    spec_copy = tmp_path / "ferrum-spec.md"
    spec_copy.write_text(text.replace(_JOINT_SIGNATURE, _JOINT_WRONG), encoding="utf-8")

    results = cs.run(["signatures"], spec_copy, cs.HISTORY_PATH, [])
    messages = [f.message for findings, _ in results for f in findings]
    assert any("'ratio' default 7, shipped 5" in m for m in messages), messages

    # The real file on disk is untouched; this is the negative control for the
    # positive check above, proving the mutation lived only in the /tmp copy.
    assert cs.SPEC_PATH.read_text(encoding="utf-8") == text


def test_invented_parameter_in_a_tmp_spec_copy_fails(cs: ModuleType, tmp_path: Path) -> None:
    """A deliberately invented parameter, injected into a /tmp copy, is caught."""
    text = cs.SPEC_PATH.read_text(encoding="utf-8")
    assert _JOINT_SIGNATURE in text, "the signature this RED proof mutates is no longer stated"
    bogus = _JOINT_SIGNATURE[:-1] + ", bogus=None)"
    spec_copy = tmp_path / "ferrum-spec.md"
    spec_copy.write_text(text.replace(_JOINT_SIGNATURE, bogus), encoding="utf-8")

    results = cs.run(["signatures"], spec_copy, cs.HISTORY_PATH, [])
    messages = [f.message for findings, _ in results for f in findings]
    assert any("unknown parameter 'bogus'" in m for m in messages), messages
