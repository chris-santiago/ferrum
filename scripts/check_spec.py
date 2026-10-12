#!/usr/bin/env python
"""Check ``ferrum-spec.md`` against the shipped code and against its note history.

Three modes, each scoped by ``--section`` (repeatable):

``signatures``
    Every signature the spec states is resolved on ``ferrum`` and compared with
    ``inspect.signature``, with ``src/ferrum/_core.pyi`` filling in the defaults
    PyO3 classes don't expose. A signature is a standalone ``name(params)``
    statement in an untagged (or ``text``/``pyi``) fence, a ``name(params)``
    code span in a table cell, or a list item that opens with one. It fails on
    a parameter the callable lacks, on a stated default that differs from the
    shipped one (literal comparison, so quote style doesn't matter), on a
    default it cannot prove unless ``UNPROVABLE_DEFAULTS`` lists it with a
    reason, on a name that doesn't resolve unless ``PSEUDO_SIGNATURES`` lists
    it with a reason, and on a statement whose parenthesis never closes. Every
    skipped statement is counted: example calls, chained calls and statements
    in fences tagged with a language (``python``), which hold example code.

``notes``
    Fails on every dated note left in the spec and counts them by kind.

``history``
    Checks that ``design-docs/spec-history.md`` holds every dated note of the
    spec at ``BASELINE_REV``, byte-identical and in original order, each under
    the ``##`` heading of a section it sat in. The history has one ``##``
    heading per section, in spec order, each equal to a spec heading. Entries
    appended later in the ``> **YYYY-MM-DD.** <change>`` form are skipped.

A dated note is any of these units that contains an ISO date (``YYYY-MM-DD``):

- a top-level blockquote outside fences (its form is read from the
  emphasized lead that opens it: ``Dated note``, ``Amended``, a bare date...);
- in a fence of any tag, a maximal run of consecutive lines that carry a
  ``#`` comment (leading or trailing), or a single dated line without one
  (kind ``fence comment``);
- outside fences and blockquotes, a heading, a table row, a list item (with
  its continuation lines) or a paragraph.

The consolidated line in its exact format is not a note. Nor is a date that
appears only inside a single-backtick inline code span (a quoted example date
or a quoted consolidated line shown as a format illustration), for any of the
non-blockquote kinds and for a blockquote's bare-date fallback.

In the history, a blockquote note stays a byte-identical blockquote. Any other
unit is written as::

    <!-- from: <kind>, ferrum-spec.md line N at <BASELINE_REV> -->
    ```text
    <the unit's original lines>
    ```

``--section`` takes a section number (``3.7``), a number range of one depth
(``3.1-3.6``), or a heading title prefix (``"Part I"``). A selector picks
every heading it matches and everything nested under it.

Usage::

    uv run --no-sync python scripts/check_spec.py                 # all modes
    uv run --no-sync python scripts/check_spec.py signatures --section 3.7
    uv run --no-sync python scripts/check_spec.py history --section 3.1-3.6

Exits 1 when any mode reports a failure, 2 when a ``--section`` matches nothing.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "ferrum-spec.md"
HISTORY_PATH = REPO / "design-docs" / "spec-history.md"
STUB_PATH = REPO / "src" / "ferrum" / "_core.pyi"
BASELINE_REV = "5c67fb0f"

# Spec names that are not ferrum callables by design, each with the reason.
PSEUDO_SIGNATURES: dict[str, str] = {}

# ``(resolved name, parameter)`` pairs whose stated default the signature cannot
# prove (a sentinel, an unreadable stub default, or a key only ``**kwargs``
# takes), each with the reason the stated value is the shipped behavior.
UNPROVABLE_DEFAULTS: dict[tuple[str, str], str] = {}

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CONSOLIDATED_LINE = re.compile(
    r"^> \*\*Consolidated \d{4}-\d{2}-\d{2}\.\*\* Earlier dated notes for this section are "
    r"kept verbatim in \[spec history\]\(design-docs/spec-history\.md#[^)\s]+\)\.$"
)
HISTORY_ENTRY = re.compile(r"^> \*\*\d{4}-\d{2}-\d{2}\.\*\* \S")

BLOCKQUOTE = "blockquote"
FENCE_COMMENT = "fence comment"
TABLE_ROW = "table row"
LIST_ITEM = "list item"
HEADING = "heading"
PARAGRAPH = "paragraph"
UNIT_KINDS = (BLOCKQUOTE, FENCE_COMMENT, TABLE_ROW, LIST_ITEM, HEADING, PARAGRAPH)
PROVENANCE = re.compile(
    rf"^<!-- from: ({FENCE_COMMENT}|{TABLE_ROW}|{LIST_ITEM}|{HEADING}|{PARAGRAPH}), "
    rf"ferrum-spec\.md line (\d+) at {re.escape(BASELINE_REV)} -->$"
)

# Lead forms of a blockquote note, matched in order against its dated lead.
NOTE_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Dated note", re.compile(r"^Dated note\b")),
    ("Amended", re.compile(r"^Amended\b")),
    ("Amendment", re.compile(r"^Amendment\b")),
    ("Implementation note", re.compile(r"^Implementation note\b")),
    ("Design decision", re.compile(r"^Design decision\b")),
    ("Note", re.compile(r"^Note\b")),
    ("Refinement", re.compile(r"^Refinement\b")),
    ("Vocabulary change", re.compile(r"^Vocabulary change\b")),
    ("date lead", re.compile(r"^\(?\d{4}-\d{2}-\d{2}")),
)
OTHER_LEAD_FORM = "other dated lead"
NO_LEAD_FORM = "dated, no lead"

SIGNATURE_FENCE_INFO = frozenset({"", "text", "pyi"})
NAME_ALIASES = ("ferrum.", "fm.", "fr.")
INSTANCE_ALIASES = {"chart": "Chart"}


# ── Markdown structure ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Heading:
    """One ATX heading; ``line`` is 0-based and ``parent`` indexes ``Document.headings``."""

    line: int
    level: int
    text: str
    parent: int | None


@dataclass(frozen=True)
class Fence:
    """A fenced block; ``start``/``end`` are the 0-based fence-marker lines."""

    start: int
    end: int
    info: str


@dataclass(frozen=True)
class Blockquote:
    """A maximal run of top-level ``>`` lines outside fences."""

    start: int
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Document:
    """A Markdown file split into lines, with its fences, headings and blockquotes.

    ``unclosed_fence`` is the 0-based line of a fence that never closes; every
    line after it reads as fence content.
    """

    label: str
    lines: tuple[str, ...]
    in_fence: tuple[bool, ...]
    fences: tuple[Fence, ...]
    headings: tuple[Heading, ...]
    blockquotes: tuple[Blockquote, ...]
    unclosed_fence: int | None

    def heading_at(self, line: int) -> int | None:
        """Return the index of the innermost heading whose span contains ``line``."""
        current = None
        for index, heading in enumerate(self.headings):
            if heading.line > line:
                break
            current = index
        return current

    def path(self, heading: int | None) -> tuple[Heading, ...]:
        """Return the heading and its ancestors, outermost first."""
        chain: list[Heading] = []
        while heading is not None:
            chain.append(self.headings[heading])
            heading = self.headings[heading].parent
        return tuple(reversed(chain))

    def path_at(self, line: int) -> tuple[Heading, ...]:
        return self.path(self.heading_at(line))


_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})\s*(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TABLE_LINE = re.compile(r"^\s*\|")
_LIST_START = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_THEMATIC_BREAK = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")


def parse_document(text: str, label: str) -> Document:
    """Split Markdown ``text`` into fences, headings and top-level blockquotes."""
    lines = tuple(text.split("\n"))
    in_fence = [False] * len(lines)
    fences: list[Fence] = []
    opener: tuple[int, str, str] | None = None
    for index, line in enumerate(lines):
        match = _FENCE_OPEN.match(line)
        if opener is None:
            if match:
                opener = (index, match.group(1), match.group(2).strip())
                in_fence[index] = True
            continue
        in_fence[index] = True
        start, marker, info = opener
        if match and match.group(1).startswith(marker) and not match.group(2).strip():
            fences.append(Fence(start, index, info))
            opener = None

    headings: list[Heading] = []
    stack: list[int] = []
    for index, line in enumerate(lines):
        match = None if in_fence[index] else _HEADING.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        while stack and headings[stack[-1]].level >= level:
            stack.pop()
        headings.append(Heading(index, level, match.group(2), stack[-1] if stack else None))
        stack.append(len(headings) - 1)

    blockquotes: list[Blockquote] = []
    run: list[str] = []
    run_start = 0
    for index, line in enumerate((*lines, "")):
        quoted = index < len(lines) and not in_fence[index] and line.startswith(">")
        if quoted:
            if not run:
                run_start = index
            run.append(line)
        elif run:
            blockquotes.append(Blockquote(run_start, tuple(run)))
            run = []
    return Document(
        label,
        lines,
        tuple(in_fence),
        tuple(fences),
        tuple(headings),
        tuple(blockquotes),
        None if opener is None else opener[0],
    )


# ── Section selectors ───────────────────────────────────────────────────────

_SECTION_NUMBER = re.compile(r"^§?(\d+(?:\.\d+)*)\.?(?:\s|$)")


def _heading_number(text: str) -> tuple[int, ...] | None:
    match = _SECTION_NUMBER.match(text.strip())
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


@dataclass(frozen=True)
class SectionSelector:
    """One ``--section`` value: a number, a same-depth number range, or a title prefix."""

    raw: str

    def matches(self, heading_text: str) -> bool:
        number = _heading_number(heading_text)
        value = self.raw.strip().lstrip("§")
        range_match = re.fullmatch(r"(\d+(?:\.\d+)*)\s*-\s*(\d+(?:\.\d+)*)", value)
        if range_match:
            low, high = (tuple(int(p) for p in g.split(".")) for g in range_match.groups())
            return number is not None and len(number) == len(low) and low <= number <= high
        if re.fullmatch(r"\d+(?:\.\d+)*", value):
            return number == tuple(int(p) for p in value.split("."))
        plain = heading_text.replace("`", "").strip()
        return re.match(rf"{re.escape(value)}\b", plain, re.IGNORECASE) is not None


def in_scope(path: tuple[Heading, ...], selectors: list[SectionSelector]) -> bool:
    """Return True when no selector is given or one matches a heading on ``path``."""
    return not selectors or any(s.matches(h.text) for h in path for s in selectors)


def unmatched_selectors(doc: Document, selectors: list[SectionSelector]) -> list[str]:
    """Return the selectors that match no heading of ``doc``, so a typo can't pass as clean."""
    return [s.raw for s in selectors if not any(s.matches(h.text) for h in doc.headings)]


# ── Findings ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One failing line of a report.

    ``fatal`` marks a finding that should abort the whole CLI run with exit
    code 2 (an unmatched ``--section`` selector) rather than fold into the
    usual exit-1 tally alongside ordinary findings.
    """

    location: str
    message: str
    fatal: bool = False

    def render(self) -> str:
        return f"{self.location}: {self.message}"


def _excerpt(text: str, width: int = 100) -> str:
    first = text.strip().split("\n", 1)[0]
    return first if len(first) <= width else first[: width - 3] + "..."


# ── Python-ish text scanning ────────────────────────────────────────────────


def _string_end(text: str, index: int) -> int:
    """Return the index of the quote closing the string literal opening at ``text[index]``."""
    quote = text[index]
    end = text.find(quote, index + 1)
    while end > 0 and text[end - 1] == "\\":
        end = text.find(quote, end + 1)
    return len(text) - 1 if end < 0 else end


def _strip_comments(text: str) -> str:
    """Drop ``#`` comments outside string literals, keeping every newline."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            end = _string_end(text, index)
            out.append(text[index : end + 1])
            index = end + 1
        elif char == "#":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
        else:
            out.append(char)
            index += 1
    return "".join(out)


# ── Dated notes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoteUnit:
    """One dated note: a unit of ``kind`` starting at 0-based ``start``.

    ``form`` is the lead form for a blockquote and the kind for any other unit.
    """

    kind: str
    start: int
    lines: tuple[str, ...]
    form: str

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def key(self) -> tuple[str, int | None, str]:
        """Identity in the history: a blockquote by its text, any other unit by line too."""
        return (self.kind, None if self.kind == BLOCKQUOTE else self.start, self.text)


_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _without_inline_code(text: str) -> str:
    """Drop single-backtick inline code spans, so a quoted date reads as code, not prose.

    A table cell, list item or paragraph that quotes a date-bearing example
    (``2024-01-01`` as sample data, or a consolidated line shown as a format
    illustration) is not itself a dated note.
    """
    return _INLINE_CODE.sub("", text)


def is_consolidated_line(block: Blockquote) -> bool:
    """Return True for a one-line blockquote in the consolidated-line format."""
    return len(block.lines) == 1 and CONSOLIDATED_LINE.match(block.lines[0]) is not None


def lead_form(line: str) -> str | None:
    """Return the form of the dated emphasized lead opening ``line``, or None."""
    lead_match = re.match(r"^> (\*\*|\*)(.*)$", line)
    if lead_match is None:
        return None
    marker, body = lead_match.groups()
    close = body.find(marker)
    lead = body if close < 0 else body[:close]
    if not ISO_DATE.search(lead):
        return None
    for form, pattern in NOTE_FORMS:
        if pattern.match(lead):
            return f"{form} (italic)" if marker == "*" else form
    return OTHER_LEAD_FORM


def note_form(block: Blockquote) -> str | None:
    """Return the dated-note form of ``block``, or None when it is not a dated note."""
    if is_consolidated_line(block):
        return None
    form = lead_form(block.lines[0])
    if form is not None:
        return form
    return NO_LEAD_FORM if ISO_DATE.search(_without_inline_code(block.text)) else None


def _blockquote_units(doc: Document) -> list[NoteUnit]:
    units = []
    for block in doc.blockquotes:
        form = note_form(block)
        if form is not None:
            units.append(NoteUnit(BLOCKQUOTE, block.start, block.lines, form))
    return units


def _fence_units(doc: Document) -> list[NoteUnit]:
    """Return the dated comment runs (and dated uncommented lines) of every fence."""
    units: list[NoteUnit] = []
    for fence in doc.fences:
        run: list[int] = []
        for index in range(fence.start + 1, fence.end + 1):
            line = doc.lines[index] if index < fence.end else ""
            commented = index < fence.end and _strip_comments(line) != line
            if commented:
                run.append(index)
                continue
            if run and any(ISO_DATE.search(doc.lines[i]) for i in run):
                units.append(
                    NoteUnit(FENCE_COMMENT, run[0], tuple(doc.lines[i] for i in run), FENCE_COMMENT)
                )
            run = []
            if index < fence.end and ISO_DATE.search(line):
                units.append(NoteUnit(FENCE_COMMENT, index, (line,), FENCE_COMMENT))
    return units


def _starts_block(line: str) -> bool:
    return bool(
        _HEADING.match(line)
        or _TABLE_LINE.match(line)
        or _LIST_START.match(line)
        or _THEMATIC_BREAK.match(line)
    )


def _markdown_units(doc: Document) -> list[NoteUnit]:
    """Return the dated headings, table rows, list items and paragraphs of ``doc``."""
    skip = {index for index, fenced in enumerate(doc.in_fence) if fenced}
    skip |= {
        block.start + offset for block in doc.blockquotes for offset in range(len(block.lines))
    }
    units: list[NoteUnit] = []
    index = 0
    while index < len(doc.lines):
        line = doc.lines[index]
        if index in skip or not line.strip() or _THEMATIC_BREAK.match(line):
            index += 1
            continue
        if _HEADING.match(line) or _TABLE_LINE.match(line):
            kind, end = (HEADING if _HEADING.match(line) else TABLE_ROW), index + 1
        else:
            kind = LIST_ITEM if _LIST_START.match(line) else PARAGRAPH
            end = index + 1
            while (
                end < len(doc.lines)
                and end not in skip
                and doc.lines[end].strip()
                and not _starts_block(doc.lines[end])
            ):
                end += 1
        lines = doc.lines[index:end]
        if any(ISO_DATE.search(_without_inline_code(text)) for text in lines):
            units.append(NoteUnit(kind, index, lines, kind))
        index = end
    return units


def dated_units(doc: Document, selectors: list[SectionSelector]) -> list[NoteUnit]:
    """Return every in-scope dated note of ``doc``, in line order."""
    units = [*_blockquote_units(doc), *_fence_units(doc), *_markdown_units(doc)]
    units.sort(key=lambda unit: unit.start)
    return [unit for unit in units if in_scope(doc.path_at(unit.start), selectors)]


def _counts(labels: list[str]) -> str:
    counts = Counter(labels)
    order = [*UNIT_KINDS, *(form for form, _ in NOTE_FORMS), OTHER_LEAD_FORM, NO_LEAD_FORM]
    order += sorted(set(counts) - set(order))
    return ", ".join(f"{label}: {counts[label]}" for label in order if counts[label]) or "none"


def _describe(units: list[NoteUnit]) -> str:
    """Summarize ``units`` by kind, and the blockquotes by lead form."""
    quotes = [unit.form for unit in units if unit.kind == BLOCKQUOTE]
    return (
        f"{len(units)} dated notes ({_counts([unit.kind for unit in units])}; "
        f"blockquote forms: {_counts(quotes)})"
    )


def check_notes(doc: Document, selectors: list[SectionSelector]) -> tuple[list[Finding], str]:
    """Fail on every dated note left in ``doc``."""
    units = dated_units(doc, selectors)
    findings = [
        Finding(
            f"{doc.label}:{unit.start + 1}", f"dated {unit.form} remains: {_excerpt(unit.text)}"
        )
        for unit in units
    ]
    consolidated = sum(
        1
        for block in doc.blockquotes
        if is_consolidated_line(block) and in_scope(doc.path_at(block.start), selectors)
    )
    return findings, f"notes: {_describe(units)}; {consolidated} consolidated lines"


@dataclass(frozen=True)
class HistoryEntry:
    """One note held in the history, under the ``##`` heading at ``Document.headings[owner]``."""

    unit: NoteUnit
    line: int
    owner: int


def format_history_unit(unit: NoteUnit) -> str:
    """Return ``unit`` in its history form: a blockquote verbatim, else provenance and fence."""
    if unit.kind == BLOCKQUOTE:
        return unit.text
    return (
        f"<!-- from: {unit.kind}, ferrum-spec.md line {unit.start + 1} at {BASELINE_REV} -->\n"
        f"```text\n{unit.text}\n```"
    )


def _history_entries(history: Document) -> tuple[list[HistoryEntry], int, list[Finding]]:
    """Read the held notes of ``history``; also count appended entries."""
    findings: list[Finding] = []
    entries: list[HistoryEntry] = []
    appended = 0
    fence_at = {fence.start: fence for fence in history.fences}
    found: list[tuple[int, NoteUnit]] = []
    for block in history.blockquotes:
        if HISTORY_ENTRY.match(block.lines[0]):
            appended += 1
        else:
            found.append((block.start, NoteUnit(BLOCKQUOTE, -1, block.lines, BLOCKQUOTE)))
    for index, line in enumerate(history.lines):
        match = None if history.in_fence[index] else PROVENANCE.match(line)
        if match is None:
            continue
        fence = fence_at.get(index + 1)
        if fence is None or fence.info != "text":
            findings.append(
                Finding(
                    f"{history.label}:{index + 1}", "provenance comment not followed by ```text"
                )
            )
            continue
        body = history.lines[fence.start + 1 : fence.end]
        kind, number = match.group(1), int(match.group(2))
        found.append((index, NoteUnit(kind, number - 1, body, kind)))
    for line, unit in sorted(found, key=lambda pair: pair[0]):
        path = history.path_at(line)
        owner = next((history.headings.index(h) for h in reversed(path) if h.level == 2), None)
        if owner is None:
            findings.append(
                Finding(
                    f"{history.label}:{line + 1}",
                    f"note sits under no ## heading: {_excerpt(unit.text)}",
                )
            )
            continue
        entries.append(HistoryEntry(unit, line, owner))
    return entries, appended, findings


def _match_sections(baseline: Document, history: Document) -> tuple[dict[int, int], list[Finding]]:
    """Map each history ``##`` heading to its baseline heading, in spec order, once each."""
    mapping: dict[int, int] = {}
    findings: list[Finding] = []
    seen: set[str] = set()
    last = -1
    for index, heading in enumerate(history.headings):
        if heading.level != 2:
            continue
        location = f"{history.label}:{heading.line + 1}"
        candidates = [i for i, h in enumerate(baseline.headings) if h.text == heading.text]
        later = [i for i in candidates if i > last]
        if heading.text in seen:
            findings.append(Finding(location, f"duplicate ## heading {heading.text!r}"))
        elif not candidates:
            findings.append(
                Finding(location, f"## heading names no spec section: {heading.text!r}")
            )
        elif not later:
            findings.append(Finding(location, f"## heading {heading.text!r} is out of spec order"))
        else:
            mapping[index] = later[0]
            last = later[0]
        seen.add(heading.text)
    return mapping, findings


def check_history(
    baseline: Document, history: Document, selectors: list[SectionSelector]
) -> tuple[list[Finding], str]:
    """Fail unless ``history`` holds the baseline's dated notes verbatim and in order."""
    expected = dated_units(baseline, selectors)
    mapping, findings = _match_sections(baseline, history)
    entries, appended, entry_findings = _history_entries(history)
    findings += entry_findings

    def entry_in_scope(entry: HistoryEntry) -> bool:
        if entry.owner in mapping:
            return in_scope(baseline.path(mapping[entry.owner]), selectors)
        return in_scope((history.headings[entry.owner],), selectors)

    held = [entry for entry in entries if entry_in_scope(entry)]
    want = Counter(unit.key for unit in expected)
    have = Counter(entry.unit.key for entry in held)
    missing, extra = want - have, have - want
    for unit in expected:
        if missing[unit.key]:
            missing[unit.key] -= 1
            findings.append(
                Finding(
                    f"{baseline.label}:{unit.start + 1}",
                    f"{unit.kind} note missing from history: {_excerpt(unit.text)}",
                )
            )
    for entry in held:
        if extra[entry.unit.key]:
            extra[entry.unit.key] -= 1
            findings.append(
                Finding(
                    f"{history.label}:{entry.line + 1}",
                    f"{entry.unit.kind} is not a baseline note (edited, misnumbered or extra): "
                    f"{_excerpt(entry.unit.text)}",
                )
            )

    if not (want - have) and not (have - want):
        for unit, entry in zip(expected, held, strict=True):
            if unit.key != entry.unit.key:
                findings.append(
                    Finding(
                        f"{history.label}:{entry.line + 1}",
                        "notes are out of original order here; expected "
                        f"{baseline.label}:{unit.start + 1} {_excerpt(unit.text)}",
                    )
                )
                break
        sources: dict[tuple[str, int | None, str], list[NoteUnit]] = {}
        for unit in expected:
            sources.setdefault(unit.key, []).append(unit)
        for entry in held:
            source = sources[entry.unit.key].pop(0)
            section = mapping.get(entry.owner)
            if section is None or baseline.headings[section] not in baseline.path_at(source.start):
                findings.append(
                    Finding(
                        f"{history.label}:{entry.line + 1}",
                        f"note from {baseline.label}:{source.start + 1} sits under "
                        f"{history.headings[entry.owner].text!r}, a section it was not in",
                    )
                )

    summary = (
        f"history: {BASELINE_REV} has {_describe(expected)}; "
        f"{len(held)} held in history, {appended} appended entries"
    )
    return findings, summary


# ── Spec signatures ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpecSignature:
    """A ``name(params)`` the spec states; ``line`` is 0-based.

    ``row_owner`` is the name a table row's first code span opens with, which
    a method named elsewhere in the row (``.predictions()``) belongs to.
    """

    line: int
    name: str
    params: str
    origin: str
    row_owner: str | None = None


@dataclass
class SkipCounts:
    """Statements the signature check read but did not compare, by reason.

    ``used_names``/``used_defaults`` record which ``PSEUDO_SIGNATURES`` /
    ``UNPROVABLE_DEFAULTS`` keys were actually matched, so an allowlist entry
    nothing in the spec needs anymore is a finding rather than a silent dead
    entry.
    """

    example_calls: int = 0
    chained_calls: int = 0
    tagged_fence_statements: int = 0
    allowlisted_names: int = 0
    allowlisted_defaults: int = 0
    used_names: set[str] = field(default_factory=set)
    used_defaults: set[tuple[str, str]] = field(default_factory=set)

    def render(self) -> str:
        return (
            f"skipped {self.example_calls} example calls, {self.chained_calls} chained calls, "
            f"{self.tagged_fence_statements} statements in tagged fences, "
            f"{self.allowlisted_names} allowlisted pseudo-signatures, "
            f"{self.allowlisted_defaults} allowlisted unprovable defaults"
        )


_NAME = r"\.?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
_STATEMENT_START = re.compile(rf"^\s*(?:def\s+)?({_NAME})\(")
_CLASS_LINE = re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\s*[:(]")
_CODE_SIGNATURE = re.compile(rf"^({_NAME})\((.*)\)\s*(?:->.*)?$", re.DOTALL)
_LIST_SIGNATURE = re.compile(r"^\s*[-*+]\s+`([^`]+)`")


def _close_paren(text: str, open_index: int) -> int | None:
    """Return the index of the parenthesis closing ``text[open_index]``, outside strings."""
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _string_end(text, index)
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _fence_signatures(
    doc: Document, fence: Fence, skips: SkipCounts, findings: list[Finding]
) -> list[SpecSignature]:
    """Return the standalone ``name(params)`` statements of one fence.

    A ``def`` under a ``class X:`` line is named ``X.method``. A chained call
    (``Chart(df).mark_point()``) is example code: it is counted and skipped.
    An unclosed parenthesis is a finding.
    """
    body = "\n".join(doc.lines[fence.start + 1 : fence.end])
    body_lines = _strip_comments(body).split("\n")
    found: list[SpecSignature] = []
    owner: tuple[int, str] | None = None
    row = 0
    while row < len(body_lines):
        line = body_lines[row]
        class_match = _CLASS_LINE.match(line)
        if class_match:
            owner = (len(class_match.group(1)), class_match.group(2))
            row += 1
            continue
        indent = len(line) - len(line.lstrip())
        if owner is not None and line.strip() and indent <= owner[0]:
            owner = None
        start = _STATEMENT_START.match(line)
        if start is None:
            row += 1
            continue
        text = "\n".join(body_lines[row:])
        open_index = start.end() - 1
        close = _close_paren(text, open_index)
        if close is None:
            findings.append(
                Finding(
                    f"{doc.label}:{fence.start + 2 + row}",
                    f"{start.group(1)}: parenthesis never closes in its fence",
                )
            )
            row += 1
            continue
        rest = text[close + 1 :].split("\n", 1)[0].strip()
        name = start.group(1)
        if owner is not None and line.lstrip().startswith("def "):
            name = f"{owner[1]}.{name}"
        if rest.startswith((".", "(", "[")):
            skips.chained_calls += 1
        else:
            found.append(
                SpecSignature(fence.start + 1 + row, name, text[open_index + 1 : close], "fence")
            )
        row += text[: close + 1].count("\n") + 1
    return found


def _table_cells(line: str) -> list[str]:
    """Split a table row into cells, honoring ``\\|`` escapes and code spans."""
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line) and line[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "`":
            in_code = not in_code
        if char == "|" and not in_code:
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current))
    return cells[1:-1] if line.strip().startswith("|") else cells


def find_signatures(
    doc: Document, selectors: list[SectionSelector], skips: SkipCounts, findings: list[Finding]
) -> list[SpecSignature]:
    """Return every signature ``doc`` states in a fence, a table cell or a list lead."""
    quoted = {
        block.start + offset for block in doc.blockquotes for offset in range(len(block.lines))
    }
    found: list[SpecSignature] = []
    for fence in doc.fences:
        if not in_scope(doc.path_at(fence.start), selectors):
            continue
        statements = _fence_signatures(doc, fence, skips, findings)
        if fence.info.split(" ", 1)[0].lower() in SIGNATURE_FENCE_INFO:
            found.extend(statements)
        else:
            skips.tagged_fence_statements += len(statements)
    for index, line in enumerate(doc.lines):
        if doc.in_fence[index] or index in quoted:
            continue
        candidates: list[tuple[str, str]] = []
        row_owner = None
        if _TABLE_LINE.match(line):
            spans = [code for cell in _table_cells(line) for code in re.findall(r"`([^`]+)`", cell)]
            first = re.match(rf"({_NAME})", spans[0].strip()) if spans else None
            row_owner = first.group(1) if first else None
            candidates += [(code, "table") for code in spans]
        else:
            list_match = _LIST_SIGNATURE.match(line)
            if list_match:
                candidates.append((list_match.group(1), "list"))
        for code, origin in candidates:
            sig_match = _CODE_SIGNATURE.match(code.strip())
            if sig_match:
                found.append(
                    SpecSignature(index, sig_match.group(1), sig_match.group(2), origin, row_owner)
                )
    found.sort(key=lambda sig: sig.line)
    return [sig for sig in found if in_scope(doc.path_at(sig.line), selectors)]


@dataclass(frozen=True)
class SpecParam:
    """One parameter as the spec states it; ``default`` is None when none is stated."""

    name: str
    kind: inspect._ParameterKind
    default: ast.expr | None


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` at commas outside brackets and string literals."""
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _string_end(text, index)
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def parse_spec_params(params: str) -> list[SpecParam] | None:
    """Parse a spec parameter list, dropping ``...`` elisions; None when it isn't one."""
    items = [item for item in _split_top_level(_strip_comments(params)) if item != "..."]
    try:
        tree = ast.parse(f"def _spec({', '.join(items)}): pass")
    except SyntaxError:
        return None
    args = tree.body[0].args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    result = [
        SpecParam(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default)
        for arg, default in zip(positional, defaults, strict=True)
    ]
    if args.vararg:
        result.append(SpecParam(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL, None))
    result += [
        SpecParam(arg.arg, inspect.Parameter.KEYWORD_ONLY, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    ]
    if args.kwarg:
        result.append(SpecParam(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD, None))
    if result and result[0].name in {"self", "cls"}:
        result = result[1:]
    return result


def is_example_call(params: str, stated: list[SpecParam] | None) -> bool:
    """Return True when ``params`` reads as a call with values, not a parameter list.

    A parameter list carries names and literal defaults. A call passes a
    non-literal keyword value (``color=selection.when(...)``) or a positional
    value that is not a bare name (``.save("chart.svg")``).
    """
    if stated is not None:
        return not all(_is_literal_like(p.default) for p in stated if p.default is not None)
    try:
        call = ast.parse(f"_call({_strip_comments(params)})", mode="eval").body
    except SyntaxError:
        return False
    assert isinstance(call, ast.Call)
    positional = [arg.value if isinstance(arg, ast.Starred) else arg for arg in call.args]
    return any(not isinstance(arg, ast.Name) for arg in positional)


def _is_literal_like(node: ast.expr) -> bool:
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.Attribute):
        return _is_literal_like(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_like(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal_like(n) for n in [*node.keys, *node.values] if n is not None)
    return False


class _Unverifiable:
    """A shipped default whose value can't be read (``...`` in a stub or text signature)."""

    def __repr__(self) -> str:
        return "<unreadable default>"


UNVERIFIABLE = _Unverifiable()

# Class attributes naming the keys a ``**kwargs`` constructor accepts, with the
# default each key has. An encoding channel's ``_honored_kwargs`` keys have no
# single default; ``Theme`` drops a None value, so None is each key's default.
ACCEPTED_KEY_SETS: dict[str, object] = {"_honored_kwargs": UNVERIFIABLE, "_KNOWN_KEYS": None}


@dataclass(frozen=True)
class ShippedParam:
    """One parameter of the callable; ``default`` is ``inspect.Parameter.empty`` when required."""

    name: str
    kind: inspect._ParameterKind
    default: object


def _stub_params(node: ast.FunctionDef) -> dict[str, object]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    pairs = list(
        zip(positional[len(positional) - len(args.defaults) :], args.defaults, strict=True)
    )
    pairs += [
        (arg, d) for arg, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d is not None
    ]
    defaults: dict[str, object] = {}
    for arg, default in pairs:
        try:
            value = ast.literal_eval(default)
        except ValueError:
            value = UNVERIFIABLE
        defaults[arg.arg] = UNVERIFIABLE if value is Ellipsis else value
    return defaults


def load_stub_defaults(path: Path = STUB_PATH) -> dict[str, dict[str, object]]:
    """Read parameter defaults from the PyO3 stub, keyed ``Class`` / ``func`` / ``Class.method``."""
    tree = ast.parse(path.read_text())
    table: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            table[node.name] = _stub_params(node)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if not isinstance(member, ast.FunctionDef):
                    continue
                if member.name in {"__init__", "__new__"}:
                    table.setdefault(node.name, {}).update(_stub_params(member))
                else:
                    table[f"{node.name}.{member.name}"] = _stub_params(member)
    return table


def resolve_name(name: str, owner: str | None) -> tuple[object, str] | None:
    """Resolve a spec name on ``ferrum``; the leading-dot form resolves on ``owner``."""
    import ferrum

    for alias in NAME_ALIASES:
        if name.startswith(alias):
            name = name[len(alias) :]
            break
    if name.startswith("."):
        if owner is None:
            return None
        name = owner + name
    parts = name.split(".")
    parts[0] = INSTANCE_ALIASES.get(parts[0], parts[0])
    roots: list[tuple[object, list[str]]] = [(ferrum, parts)]
    if len(parts) == 1:
        roots.append((ferrum.Chart, parts))
    for root, path in roots:
        obj: object = root
        for part in path:
            if hasattr(obj, part):
                obj = getattr(obj, part)
                continue
            if inspect.ismodule(obj):
                try:
                    obj = importlib.import_module(f"{obj.__name__}.{part}")
                    continue
                except ImportError:
                    pass
            break
        else:
            prefix = "" if root is ferrum else "Chart."
            return obj, prefix + ".".join(path)
    return None


def _owner_class(doc: Document, line: int) -> str:
    """Return the class a ``.method(...)`` near ``line`` belongs to.

    That is the ferrum class the nearest enclosing heading names in a code
    span (``#### `ferrum.ModelSource` ``), else ``Chart``.
    """
    import ferrum

    for heading in reversed(doc.path_at(line)):
        for code in re.findall(r"`([^`]+)`", heading.text):
            candidate = code
            for alias in NAME_ALIASES:
                candidate = candidate.removeprefix(alias)
            obj = getattr(ferrum, candidate, None)
            if inspect.isclass(obj):
                return candidate
    return "Chart"


def shipped_params(
    obj: object, qualname: str, stubs: dict[str, dict[str, object]]
) -> list[ShippedParam] | None:
    """Return the callable's parameters, stub defaults standing in for unreadable ones.

    A property reads as its getter. When the callable takes ``**kwargs`` and
    declares the keys it accepts (``ACCEPTED_KEY_SETS``), each key becomes a
    keyword parameter.
    """
    if isinstance(obj, property) and obj.fget is not None:
        obj = obj.fget
    try:
        signature = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    stub = stubs.get(qualname) or stubs.get(qualname.rsplit(".", 1)[-1]) or {}
    is_core = getattr(obj, "__module__", "") == "ferrum._core"
    params = []
    for param in signature.parameters.values():
        default = param.default
        if default is Ellipsis:
            default = stub.get(param.name, UNVERIFIABLE) if is_core else UNVERIFIABLE
        params.append(ShippedParam(param.name, param.kind, default))
    if params and params[0].name in {"self", "cls"}:
        params = params[1:]
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        named = {p.name for p in params}
        for attribute, default in ACCEPTED_KEY_SETS.items():
            keys = getattr(obj, attribute, None) or ()
            params += [
                ShippedParam(key, inspect.Parameter.KEYWORD_ONLY, default)
                for key in sorted(keys)
                if key not in named
            ]
            named.update(keys)
    return params


_LITERAL_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _is_literal_value(value: object) -> bool:
    if isinstance(value, _LITERAL_TYPES):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_is_literal_value(item) for item in value)
    if isinstance(value, dict):
        return all(_is_literal_value(k) and _is_literal_value(v) for k, v in value.items())
    return False


def _same_literal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    numbers = (int, float, complex)
    if isinstance(left, numbers) and isinstance(right, numbers):
        return left == right
    return type(left) is type(right) and left == right


def _is_omit_sentinel(value: object) -> bool:
    """Return True for the sentinel of the omit-vs-explicit contract on Axis/Legend fields."""
    from ferrum._title_sentinel import _UNSET

    return value is _UNSET


def defaults_match(param: str, stated: ast.expr, shipped: object) -> bool | None:
    """Compare a stated default with the shipped one; None when it can't be proved.

    Under ``ferrum._title_sentinel.is_unspecified``, an explicit None means the
    same as omitting a sentinel-defaulted field, except ``title``, where None
    suppresses the title and omitting keeps the field name.
    """
    if shipped is UNVERIFIABLE:
        return None
    text = ast.unparse(stated)
    try:
        stated_value = ast.literal_eval(stated)
    except ValueError:
        stated_value = UNVERIFIABLE
    if _is_omit_sentinel(shipped):
        if text == "_UNSET" or (stated_value is None and param != "title"):
            return True
        return False if stated_value is None else None
    if not _is_literal_value(shipped):
        names = {repr(shipped), getattr(shipped, "__name__", None)}
        return True if text in names or text.rsplit(".", 1)[-1] in names else None
    if stated_value is UNVERIFIABLE:
        return False
    return _same_literal(stated_value, shipped)


def _format_default(value: object) -> str:
    return repr(value) if value is not inspect.Parameter.empty else "<required>"


def compare_signature(
    qualname: str,
    stated: list[SpecParam],
    shipped: list[ShippedParam],
    location: str,
    display: str,
    skips: SkipCounts,
) -> list[Finding]:
    """Compare one stated signature with the shipped parameters."""
    findings: list[Finding] = []
    by_name = {
        p.name: p
        for p in shipped
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    kinds = {p.kind for p in shipped}
    passthrough = inspect.Parameter.VAR_KEYWORD in kinds and not by_name

    def unprovable(param: str, message: str) -> None:
        key = (qualname, param)
        if key in UNPROVABLE_DEFAULTS:
            skips.allowlisted_defaults += 1
            skips.used_defaults.add(key)
        else:
            findings.append(Finding(location, f"{display}: {message}"))

    for param in stated:
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            if inspect.Parameter.VAR_POSITIONAL not in kinds:
                findings.append(
                    Finding(
                        location, f"{display}: spec lists *{param.name}, callable takes no *args"
                    )
                )
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            if inspect.Parameter.VAR_KEYWORD not in kinds:
                findings.append(
                    Finding(
                        location,
                        f"{display}: spec lists **{param.name}, callable takes no **kwargs",
                    )
                )
            continue
        actual = by_name.get(param.name)
        if actual is None:
            if passthrough:
                unprovable(param.name, f"'{param.name}' only reaches **kwargs; cannot prove it")
            else:
                findings.append(Finding(location, f"{display}: unknown parameter '{param.name}'"))
            continue
        if param.default is None:
            continue
        stated_text = ast.unparse(param.default)
        if actual.default is inspect.Parameter.empty:
            findings.append(
                Finding(
                    location, f"{display}: '{param.name}' default {stated_text}, shipped <required>"
                )
            )
            continue
        verdict = defaults_match(param.name, param.default, actual.default)
        shipped_text = _format_default(actual.default)
        if verdict is False:
            findings.append(
                Finding(
                    location,
                    f"{display}: '{param.name}' default {stated_text}, shipped {shipped_text}",
                )
            )
        elif verdict is None:
            unprovable(
                param.name,
                f"'{param.name}' default {stated_text} cannot be proved against shipped "
                f"{shipped_text}",
            )
    return findings


def _resolve_signature(doc: Document, sig: SpecSignature) -> tuple[object, str] | None:
    """Resolve ``sig`` on ferrum, then as a method of the class its table row names."""
    if sig.name.startswith("."):
        resolved = None
    else:
        resolved = resolve_name(sig.name, None)
    if resolved is None and sig.row_owner and sig.row_owner != sig.name:
        resolved = resolve_name("." + sig.name.lstrip("."), sig.row_owner)
    if resolved is None and sig.name.startswith("."):
        resolved = resolve_name(sig.name, _owner_class(doc, sig.line))
    return resolved


def _unused_allowlist_findings(skips: SkipCounts) -> list[Finding]:
    """Fail on a ``PSEUDO_SIGNATURES``/``UNPROVABLE_DEFAULTS`` entry nothing in the spec matched.

    Scoped to a whole-spec run (no ``--section``): a section-scoped run
    legitimately exercises only some entries, so it skips this check rather
    than flagging every allowlist entry outside its scope as dead.
    """
    findings = [
        Finding("PSEUDO_SIGNATURES", f"{name!r} is unused; delete it or restore the signature")
        for name in sorted(set(PSEUDO_SIGNATURES) - skips.used_names)
    ]
    findings += [
        Finding(
            "UNPROVABLE_DEFAULTS",
            f"{key!r} is unused; delete it or restore the unprovable default",
        )
        for key in sorted(set(UNPROVABLE_DEFAULTS) - skips.used_defaults)
    ]
    return findings


def check_signatures(doc: Document, selectors: list[SectionSelector]) -> tuple[list[Finding], str]:
    """Fail on every stated signature whose parameters or defaults differ from the code."""
    stubs = load_stub_defaults()
    findings: list[Finding] = []
    skips = SkipCounts()
    signatures = find_signatures(doc, selectors, skips, findings)
    for sig in signatures:
        location = f"{doc.label}:{sig.line + 1}"
        stated = parse_spec_params(sig.params)
        if is_example_call(sig.params, stated):
            skips.example_calls += 1
            continue
        resolved = _resolve_signature(doc, sig)
        if resolved is None:
            if sig.name in PSEUDO_SIGNATURES:
                skips.allowlisted_names += 1
                skips.used_names.add(sig.name)
            else:
                findings.append(
                    Finding(location, f"{sig.name}: does not resolve on ferrum ({sig.origin})")
                )
            continue
        obj, qualname = resolved
        display = sig.name if sig.name.endswith(qualname) else f"{sig.name} -> {qualname}"
        if stated is None:
            findings.append(
                Finding(location, f"{display}: parameter list is not valid Python syntax")
            )
            continue
        shipped = shipped_params(obj, qualname, stubs)
        if shipped is None:
            findings.append(Finding(location, f"{display}: callable has no readable signature"))
            continue
        findings.extend(compare_signature(qualname, stated, shipped, location, display, skips))
    if not selectors:
        findings.extend(_unused_allowlist_findings(skips))
    origins = Counter(sig.origin for sig in signatures)
    summary = (
        f"signatures: {len(signatures)} stated ({origins['fence']} fenced, "
        f"{origins['table']} tabled, {origins['list']} listed); {skips.render()}"
    )
    return findings, summary


# ── Entry point ─────────────────────────────────────────────────────────────


def read_baseline_spec(rev: str = BASELINE_REV) -> str | None:
    """Return the spec as committed at ``rev``, or None when git cannot read it."""
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:ferrum-spec.md"],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def load_document(path: Path) -> tuple[Document | None, list[Finding]]:
    """Parse ``path``; a missing file or an unclosed fence is a finding."""
    label = _label(path)
    if not path.is_file():
        return None, [Finding(label, "file does not exist")]
    doc = parse_document(path.read_text(encoding="utf-8"), label)
    return doc, _structure_findings(doc)


def _structure_findings(doc: Document) -> list[Finding]:
    if doc.unclosed_fence is None:
        return []
    return [
        Finding(f"{doc.label}:{doc.unclosed_fence + 1}", "fence never closes; the rest is unread")
    ]


def run(
    modes: list[str],
    spec_path: Path,
    history_path: Path,
    selectors: list[SectionSelector],
    baseline_rev: str = BASELINE_REV,
) -> list[tuple[list[Finding], str]]:
    """Run ``modes`` and return each mode's findings and summary line.

    A ``--section`` selector that matches no heading is a finding, under the
    summary ``"section selector"`` and each marked ``Finding.fatal``: a typo
    there must not report a clean run over the empty scope it selected, and
    ``main()`` reads the ``fatal`` marker (not the summary text) to choose
    its distinct exit code.
    """
    spec, spec_findings = load_document(spec_path)
    results: list[tuple[list[Finding], str]] = []
    if spec is None:
        return [(spec_findings, "spec: not checked")]
    if spec_findings:
        results.append((spec_findings, "spec structure"))
    unmatched = unmatched_selectors(spec, selectors)
    if unmatched:
        findings = [
            Finding(spec.label, f"--section matches no heading: {raw!r}", fatal=True)
            for raw in unmatched
        ]
        results.append((findings, "section selector"))
        return results
    if "signatures" in modes:
        results.append(check_signatures(spec, selectors))
    if "notes" in modes:
        results.append(check_notes(spec, selectors))
    if "history" in modes:
        baseline_text = read_baseline_spec(baseline_rev)
        history, history_findings = load_document(history_path)
        if baseline_text is None:
            missing = Finding(f"ferrum-spec.md@{baseline_rev}", "git cannot read the baseline spec")
            results.append(([missing, *history_findings], "history: not checked"))
        elif history is None:
            results.append((history_findings, "history: not checked"))
        else:
            baseline = parse_document(baseline_text, f"ferrum-spec.md@{baseline_rev}")
            findings, summary = check_history(baseline, history, selectors)
            results.append(([*history_findings, *findings], summary))
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the requested modes and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "mode", nargs="?", default="all", choices=["all", "signatures", "notes", "history"]
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="section number (3.7), number range (3.1-3.6) or title prefix; repeatable",
    )
    parser.add_argument("--spec", type=Path, default=SPEC_PATH, help="spec file to check")
    parser.add_argument("--history", type=Path, default=HISTORY_PATH, help="spec history file")
    args = parser.parse_args(argv)
    selectors = [SectionSelector(raw) for raw in args.section]
    modes = ["signatures", "notes", "history"] if args.mode == "all" else [args.mode]

    failed = False
    fatal = False
    for findings, summary in run(modes, args.spec, args.history, selectors):
        for finding in findings:
            print(finding.render())
            fatal = fatal or finding.fatal
        print(f"{summary}; {len(findings)} failing")
        failed = failed or bool(findings)
    if fatal:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
