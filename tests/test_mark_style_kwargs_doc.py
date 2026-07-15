"""Drift guard: the guide's "Mark style kwargs" section must match the allowlist.

``docs/site/guide/marks-encodings.md`` documents the full set of mark-style
kwargs accepted by ``mark_*()`` methods, and dozens of docstrings across the
codebase point back to that section by name instead of re-listing every
accepted key (see GH #83). If a kwarg is ever added to (or removed from)
``ferrum.marks.base._VALID_MARK_KWARGS`` without updating the guide, this
test fails — preventing the canonical doc list from silently drifting out
of sync with what the renderer actually allows.
"""

from __future__ import annotations

import re
from pathlib import Path

from ferrum.marks.base import _VALID_MARK_KWARGS

_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "site" / "guide" / "marks-encodings.md"
)

# Matches a markdown table row's first cell containing one or more
# backtick-quoted kwarg names, e.g. "| `size` |" or "| `dx`, `dy` |".
_KWARG_CELL_RE = re.compile(r"^\s*\|\s*((?:`[a-z_]+`,?\s*)+)\|")
_KWARG_NAME_RE = re.compile(r"`([a-z_]+)`")


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


def test_mark_style_kwargs_doc_matches_allowlist():
    documented = _documented_mark_style_kwargs()
    canonical = set(_VALID_MARK_KWARGS)

    missing_from_doc = canonical - documented
    extra_in_doc = documented - canonical
    assert not missing_from_doc and not extra_in_doc, (
        "docs/site/guide/marks-encodings.md 'Mark style kwargs' section is out "
        "of sync with ferrum.marks.base._VALID_MARK_KWARGS.\n"
        f"In the allowlist but not documented: {sorted(missing_from_doc)}\n"
        f"Documented but not in the allowlist: {sorted(extra_in_doc)}"
    )


def test_mark_style_kwargs_doc_parses_a_nonempty_set():
    # Guards the parser itself: if the table format ever changes shape such
    # that the regex stops matching anything, the equality check above would
    # trivially pass on an empty set vs. an empty set only if the allowlist
    # were also empty (it never is). This makes that failure mode explicit.
    assert len(_documented_mark_style_kwargs()) > 30
