"""Drift guard: the guide's "Mark style kwargs" section must match the allowlist.

``docs/site/guide/marks-encodings.md`` documents the full set of mark-style
kwargs accepted by ``mark_*()`` methods, and dozens of docstrings across the
codebase point back to that section by name instead of re-listing every
accepted key (see GH #83). If a kwarg some ``mark_*()`` method takes (every
``ferrum.marks.base._VALID_MARK_KWARGS`` entry but the fields only ferrum's
own code writes) is added or removed without updating the guide, this test
fails — preventing the canonical doc list from silently drifting out of sync
with what the renderer actually allows.

The "### Style fields" table's "Drawn by" column is a second, narrower
drift surface: it hand-lists, per field, the primitive marks that draw it.
``test_mark_style_kwargs_doc_drawn_by_matches_table`` pins that column
against Rust's own style table (``ferrum._layer.marks_drawing``, built from
``ferrum._core.mark_style_fields``/``style_table_marks``), so a Rust-side
change to which marks draw a field fails here instead of drifting silently.
"""

from __future__ import annotations

import re
from pathlib import Path

from ferrum._core import style_table_marks
from ferrum._layer import marks_drawing
from ferrum.marks.base import _INTERNAL_STYLE_DOORS, _VALID_MARK_KWARGS

_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "site" / "guide" / "marks-encodings.md"
)

# Matches a markdown table row's first cell containing one or more
# backtick-quoted kwarg names, e.g. "| `size` |" or "| `dx`, `dy` |".
_KWARG_CELL_RE = re.compile(r"^\s*\|\s*((?:`[a-z_]+`,?\s*)+)\|")
_KWARG_NAME_RE = re.compile(r"`([a-z_]+)`")

# Matches a full "### Style fields" table row: kwarg cell, meaning cell,
# "Drawn by" cell. None of the three cells contain a literal "|", so a plain
# split is safe (see test_drawn_by_doc_matches_table's docstring).
_STYLE_FIELD_ROW_RE = re.compile(r"^\s*\|\s*((?:`[a-z_]+`,?\s*)+)\|([^|]*)\|([^|]*)\|\s*$")


def _documented_mark_style_kwargs() -> set[str]:
    """Parse kwarg names out of the "Mark style kwargs" section's tables."""
    text = _GUIDE_PATH.read_text()
    section_match = re.search(
        r"^## Mark style kwargs\n(.*?)(?=\n## )", text, re.DOTALL | re.MULTILINE
    )
    assert section_match is not None, (
        f"{_GUIDE_PATH} has no '## Mark style kwargs' section — "
        "this is the canonical anchor other docstrings link to."
    )
    section = section_match.group(1)

    names: set[str] = set()
    for line in section.splitlines():
        cell_match = _KWARG_CELL_RE.match(line)
        if cell_match is None:
            continue
        names.update(_KWARG_NAME_RE.findall(cell_match.group(1)))
    return names


_ALL_STYLE_TABLE_MARKS = frozenset(style_table_marks())


def _parse_drawn_by_cell(cell: str) -> frozenset[str]:
    """Parse one "Drawn by" cell into the primitive mark names it names.

    Most cells are a plain comma-separated mark list (e.g.
    ``"point, bar, area"``). Two cells use prose shorthand for a field every
    primitive mark draws, or every mark but one: ``"every mark"`` and
    ``"every mark but image"``.
    """
    cell = cell.strip()
    if cell == "every mark":
        return _ALL_STYLE_TABLE_MARKS
    but_match = re.match(r"^every mark but (.+)$", cell)
    if but_match is not None:
        excluded = {
            name.strip() for name in re.split(r",| and ", but_match.group(1)) if name.strip()
        }
        return _ALL_STYLE_TABLE_MARKS - excluded
    return frozenset(name.strip() for name in cell.split(",") if name.strip())


def _documented_drawn_by() -> dict[str, frozenset[str]]:
    """Parse the "### Style fields" table's "Drawn by" column, per kwarg.

    Maps each style-field kwarg name documented in that table to the set of
    primitive mark names its "Drawn by" cell names (expanding the "every
    mark[ but X]" prose shorthand via :func:`_parse_drawn_by_cell`). A row
    naming more than one kwarg (e.g. `` `dx`, `dy` ``) maps every named
    kwarg to that row's same set.
    """
    text = _GUIDE_PATH.read_text()
    section_match = re.search(
        r"^### Style fields\n(.*?)(?=\n### |\n## )", text, re.DOTALL | re.MULTILINE
    )
    assert section_match is not None, (
        f"{_GUIDE_PATH} has no '### Style fields' subsection under 'Mark style "
        "kwargs' — this is the table the 'Drawn by' column lives in."
    )
    section = section_match.group(1)

    drawn_by: dict[str, frozenset[str]] = {}
    for line in section.splitlines():
        row_match = _STYLE_FIELD_ROW_RE.match(line)
        if row_match is None:
            continue
        kwarg_cell, _meaning_cell, drawn_by_cell = row_match.groups()
        marks = _parse_drawn_by_cell(drawn_by_cell)
        for name in _KWARG_NAME_RE.findall(kwarg_cell):
            drawn_by[name] = marks
    return drawn_by


def test_mark_style_kwargs_doc_drawn_by_matches_table():
    documented = _documented_drawn_by()
    assert documented, (
        "parsed no rows out of the '### Style fields' table's 'Drawn by' column — "
        "did the table shape change?"
    )
    for field, documented_marks in documented.items():
        actual_marks = frozenset(marks_drawing(field))
        assert documented_marks == actual_marks, (
            "docs/site/guide/marks-encodings.md 'Drawn by' column for "
            f"`{field}` is out of sync with Rust's style table.\n"
            f"Documented: {sorted(documented_marks)}\n"
            f"Actual (ferrum._layer.marks_drawing({field!r})): {sorted(actual_marks)}"
        )


def test_mark_style_kwargs_doc_matches_allowlist():
    documented = _documented_mark_style_kwargs()
    canonical = set(_VALID_MARK_KWARGS) - set(_INTERNAL_STYLE_DOORS)

    missing_from_doc = canonical - documented
    extra_in_doc = documented - canonical
    assert not missing_from_doc and not extra_in_doc, (
        "docs/site/guide/marks-encodings.md 'Mark style kwargs' section is out "
        "of sync with the kwargs mark_*() methods take.\n"
        f"In the allowlist but not documented: {sorted(missing_from_doc)}\n"
        f"Documented but not in the allowlist: {sorted(extra_in_doc)}"
    )


def test_mark_style_kwargs_doc_parses_a_nonempty_set():
    # Guards the parser itself: if the table format ever changes shape such
    # that the regex stops matching anything, the equality check above would
    # trivially pass on an empty set vs. an empty set only if the allowlist
    # were also empty (it never is). This makes that failure mode explicit.
    #
    # Threshold dropped from 30 to 20 (#243, wave-1 T16): the guide's old
    # "Statistical mark kwargs (forwarded to the transform)" table
    # documented 13 names (method, ci, bandwidth, degree, n, kernel, extent,
    # cumulative, bin_count, bin_width, density, right, multiple) that were
    # never actually part of the `**kwargs` allowlist this section
    # describes -- they are named parameters of specific `mark_*()` methods
    # (`mark_smooth`, `mark_density`, `mark_histogram`, `mark_errorbar`),
    # always intercepted by signature-based partitioning before reaching
    # `MarkBase`/`validate_user_mark_kwargs`. Removing that table (and thus
    # ``_VALID_MARK_KWARGS``'s 13 dead entries) drops the true allowlist
    # from 42 to 29; the threshold is lowered to stay a "did the parser
    # break" sanity check rather than a stale count pin.
    assert len(_documented_mark_style_kwargs()) > 20
