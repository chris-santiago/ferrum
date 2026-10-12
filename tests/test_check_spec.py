"""Negative controls for ``scripts/check_spec.py``.

Each test builds a small spec or history and checks that the checker fails on
the defect it plants (an invented parameter, a wrong default, an edited or
misplaced note...) and passes on the faithful version. The history tests build
their fixture from the real baseline spec at ``BASELINE_REV``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the class body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cs() -> ModuleType:
    return _load("check_spec", ROOT / "scripts" / "check_spec.py")


def _signatures(cs: ModuleType, body: str) -> tuple[list[str], object]:
    doc = cs.parse_document(f"### 3.12 Compound Views\n\n{body}\n", "spec.md")
    findings, summary = cs.check_signatures(doc, [])
    return [f.message for f in findings], summary


JOINT = "JointChart(center, *, top=None, right=None, ratio=5, spacing=10.0)"


# ── signatures ──────────────────────────────────────────────────────────────


def test_shipped_signature_passes(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, f"```\n{JOINT}\n```")
    assert messages == []


def test_invented_parameter_fails(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, f"```\n{JOINT[:-1]}, bogus=None)\n```")
    assert messages == ["JointChart: unknown parameter 'bogus'"]


def test_wrong_default_fails(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, f"```\n{JOINT.replace('ratio=5', 'ratio=7')}\n```")
    assert messages == ["JointChart: 'ratio' default 7, shipped 5"]


def test_quote_style_does_not_matter(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, "```\nferrum.heatmap(data, *, annot=True, fmt='.2f')\n```")
    assert messages == []


def test_every_code_span_in_a_table_cell_is_read(cs: ModuleType) -> None:
    row = "| `X(field)` | see also `JointChart(center, *, bogus=1)` |"
    messages, _ = _signatures(cs, f"| a | b |\n|---|---|\n{row}")
    assert messages == ["JointChart: unknown parameter 'bogus'"]


def test_tagged_fence_is_counted_not_checked(cs: ModuleType) -> None:
    messages, summary = _signatures(cs, "```python\nJointChart(center, *, bogus=1)\n```")
    assert messages == []
    assert "1 statements in tagged fences" in summary


def test_unclosed_parenthesis_fails(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, "```\nJointChart(center, *, top=None,\n```")
    assert messages == ["JointChart: parenthesis never closes in its fence"]


def test_unresolvable_name_fails_unless_allowlisted(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, "```\nNoSuchThing(a)\nMyStat.apply(data)\n```")
    assert messages == ["NoSuchThing: does not resolve on ferrum (fence)"]


def test_none_means_unset_except_for_title(cs: ModuleType) -> None:
    messages, _ = _signatures(cs, "```\nAxis(title=None, *, ticks=None, grid=None)\n```")
    assert messages == ["Axis: 'title' default None, shipped _UNSET"]


def test_unprovable_default_fails_unless_allowlisted(
    cs: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "```\nAxis(*, ticks=True)\n```"
    messages, _ = _signatures(cs, body)
    assert messages == ["Axis: 'ticks' default True cannot be proved against shipped _UNSET"]
    monkeypatch.setitem(cs.UNPROVABLE_DEFAULTS, ("Axis", "ticks"), "unset draws ticks")
    messages, summary = _signatures(cs, body)
    assert messages == []
    assert "1 allowlisted unprovable defaults" in summary


def test_unclosed_fence_is_a_finding(cs: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "spec.md"
    path.write_text("# Spec\n\n```\nChart(data=None)\n")
    _, findings = cs.load_document(path)
    assert [f.message for f in findings] == ["fence never closes; the rest is unread"]


def test_missing_files_and_baseline_are_findings(cs: ModuleType, tmp_path: Path) -> None:
    results = cs.run(["notes"], tmp_path / "absent.md", tmp_path / "h.md", [])
    assert [f.message for findings, _ in results for f in findings] == ["file does not exist"]
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n")
    results = cs.run(["history"], spec, tmp_path / "h.md", [], baseline_rev="0000000bad")
    messages = [f.message for findings, _ in results for f in findings]
    assert "git cannot read the baseline spec" in messages


# ── notes ───────────────────────────────────────────────────────────────────

NOTES_SPEC = """# Spec

### 3.1 One

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#31-one).

> **Consolidated 2026-10-12.** Earlier notes are elsewhere.

> A callout with no date.

> Changed on 2026-10-01 without a lead.

#### Added 2026-05-11

| a | b |
|---|---|
| `x` | 2026-09-05: row corrected |

- an item (2026-05-15)
  continued

A paragraph that says 2026-09-10
over two lines.

```
Chart(
    data=None,  # 2026-09-01 trailing
    # and a comment block
)
# 2026-09-02 leading
```
"""


def test_notes_finds_every_dated_unit(cs: ModuleType) -> None:
    doc = cs.parse_document(NOTES_SPEC, "spec.md")
    units = cs.dated_units(doc, [])
    found = [(unit.kind, unit.start + 1) for unit in units]
    assert found == [
        ("blockquote", 7),
        ("blockquote", 11),
        ("heading", 13),
        ("table row", 17),
        ("list item", 19),
        ("paragraph", 22),
        ("fence comment", 27),
        ("fence comment", 30),
    ]
    assert units[4].lines == ("- an item (2026-05-15)", "  continued")
    assert units[6].lines == ("    data=None,  # 2026-09-01 trailing", "    # and a comment block")
    findings, summary = cs.check_notes(doc, [])
    assert len(findings) == 8
    assert "1 consolidated lines" in summary


# ── history ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def baseline(cs: ModuleType) -> object:
    text = cs.read_baseline_spec()
    if text is None:
        pytest.skip(f"git cannot read the baseline spec at {cs.BASELINE_REV}")
    return cs.parse_document(text, "baseline")


def _sections(cs: ModuleType, baseline: object) -> list[tuple[str, list[str]]]:
    """Group the baseline notes under their innermost heading of level 3 or above."""
    sections: list[tuple[str, list[str]]] = []
    for unit in cs.dated_units(baseline, []):
        owner = [h for h in baseline.path_at(unit.start) if h.level <= 3][-1]
        if not sections or sections[-1][0] != owner.text:
            sections.append((owner.text, []))
        sections[-1][1].append(cs.format_history_unit(unit))
    return sections


def _render(sections: list[tuple[str, list[str]]]) -> str:
    out = ["# Spec history", "", "Notes moved out of the spec.", ""]
    for heading, entries in sections:
        out += [f"## {heading}", ""]
        for entry in entries:
            out += [entry, ""]
    return "\n".join(out)


def _history_messages(cs: ModuleType, baseline: object, text: str, selectors=()) -> list[str]:
    history = cs.parse_document(text, "history.md")
    findings, _ = cs.check_history(baseline, history, [cs.SectionSelector(s) for s in selectors])
    return [f.message for f in findings]


def test_faithful_history_passes(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    assert _history_messages(cs, baseline, _render(sections)) == []
    assert _history_messages(cs, baseline, _render(sections), ["3.1-3.6"]) == []


def test_appended_entry_passes(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    sections[-1][1].append("> **2026-10-12.** The contract replaces the notes.")
    assert _history_messages(cs, baseline, _render(sections)) == []


def _first_with(sections: list[tuple[str, list[str]]], prefix: str) -> tuple[int, int]:
    for s, (_, entries) in enumerate(sections):
        for e, entry in enumerate(entries):
            if entry.startswith(prefix):
                return s, e
    raise AssertionError(prefix)


@pytest.mark.parametrize("prefix", ["> ", "<!-- from: fence comment", "<!-- from: table row"])
def test_edited_note_fails(cs: ModuleType, baseline: object, prefix: str) -> None:
    sections = _sections(cs, baseline)
    s, e = _first_with(sections, prefix)
    entry = sections[s][1][e]
    # One trailing space on the note's first line of text.
    if entry.startswith(">"):
        sections[s][1][e] = entry.replace("\n", " \n", 1) if "\n" in entry else entry + " "
    else:
        head, body = entry.split("```text\n", 1)
        sections[s][1][e] = head + "```text\n" + body.replace("\n", " \n", 1)
    messages = _history_messages(cs, baseline, _render(sections))
    assert any("missing from history" in m for m in messages)
    assert any("is not a baseline note" in m for m in messages)


def test_deleted_note_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    del sections[3][1][0]
    messages = _history_messages(cs, baseline, _render(sections))
    assert len(messages) == 1 and "missing from history" in messages[0]


def test_misnumbered_provenance_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    s, e = _first_with(sections, "<!-- from: paragraph")
    entry = sections[s][1][e]
    line = int(entry.split(" line ", 1)[1].split(" ", 1)[0])
    sections[s][1][e] = entry.replace(f" line {line} ", f" line {line + 1} ", 1)
    messages = _history_messages(cs, baseline, _render(sections))
    assert any("misnumbered" in m for m in messages)


def test_swapped_notes_fail(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    entries = next(entries for _, entries in sections if len(entries) > 1)
    entries[0], entries[1] = entries[1], entries[0]
    messages = _history_messages(cs, baseline, _render(sections))
    assert any("out of original order" in m for m in messages)


def test_note_under_wrong_section_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    sections[-1][1].append(sections[0][1].pop())
    messages = _history_messages(cs, baseline, _render(sections))
    assert any("a section it was not in" in m for m in messages)


def test_unknown_heading_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    sections[2] = ("Not A Spec Heading", sections[2][1])
    messages = _history_messages(cs, baseline, _render(sections))
    assert "## heading names no spec section: 'Not A Spec Heading'" in messages


def test_duplicate_heading_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    heading, entries = sections[2]
    sections[2:3] = [(heading, entries[:1]), (heading, entries[1:])]
    messages = _history_messages(cs, baseline, _render(sections))
    assert f"duplicate ## heading {heading!r}" in messages


def test_heading_out_of_spec_order_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    first = sections.pop(0)
    sections.insert(2, first)
    messages = _history_messages(cs, baseline, _render(sections))
    assert f"## heading {first[0]!r} is out of spec order" in messages


def test_provenance_without_text_fence_fails(cs: ModuleType, baseline: object) -> None:
    sections = _sections(cs, baseline)
    s, e = _first_with(sections, "<!-- from: heading")
    sections[s][1][e] = sections[s][1][e].replace("```text", "```", 1)
    messages = _history_messages(cs, baseline, _render(sections))
    assert "provenance comment not followed by ```text" in messages
