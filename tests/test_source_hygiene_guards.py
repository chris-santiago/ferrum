"""Source-hygiene guards over every git-tracked file in ``src/``, ``tests/`` and ``crates/``.

One scanner (``_tracked_lines``) feeds one rule per test. The scanner has its
own control (it must reach known files and lines, so an empty scan fails), and
each rule pairs a scan of the real tree with a negative control proving the
rule flags a synthetic violation.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_ROOTS = ("src", "tests", "crates")

# Built by concatenation so this module never matches its own rule.
_LEDGER_DIR = "." + "sdd/"
LEDGER_CITATION = re.compile(re.escape(_LEDGER_DIR))

# Tracked binary assets the scanner does not read as text. Any other tracked
# file that is not UTF-8 fails the scan rather than escaping every rule.
BINARY_SUFFIXES = frozenset({".png", ".parquet", ".skops", ".ttf", ".woff2", ".wasm"})


def _tracked_files() -> list[Path]:
    """Return every git-tracked text file under ``SCANNED_ROOTS``."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *SCANNED_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [REPO_ROOT / name for name in out.split("\0") if name]


def _tracked_lines() -> Iterator[tuple[str, int, str]]:
    """Yield ``(relative path, 1-based line number, line)`` for every tracked text file.

    Skips ``BINARY_SUFFIXES`` and files deleted in the working tree but still
    in the index; a non-UTF-8 file of any other kind raises ``UnicodeDecodeError``.
    """
    for path in _tracked_files():
        if path.suffix in BINARY_SUFFIXES or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            yield rel, lineno, line


def _ledger_citations(lines: Iterable[tuple[str, int, str]]) -> list[str]:
    return [
        f"{rel}:{lineno}: {line.strip()}"
        for rel, lineno, line in lines
        if LEDGER_CITATION.search(line)
    ]


def test_scanner_reaches_tracked_source_in_every_root() -> None:
    """An empty or partial scan would pass every rule vacuously."""
    files = {path.relative_to(REPO_ROOT).as_posix() for path in _tracked_files()}
    known = {
        "src/ferrum/__init__.py",
        "tests/_snapshots.py",
        "crates/ferrum-core/src/lib.rs",
    }
    assert known <= files, f"scanner missed {sorted(known - files)}"

    lines = {(rel, line) for rel, _, line in _tracked_lines()}
    assert ("tests/_snapshots.py", "from __future__ import annotations") in lines
    assert ("crates/ferrum-core/src/lib.rs", "use pyo3::prelude::*;") in lines


def test_ledger_rule_flags_a_synthetic_citation() -> None:
    synthetic = [
        ("tests/test_example.py", 7, f"# see `{_LEDGER_DIR}task-1-report.md`"),
        ("tests/test_example.py", 8, "# see commit 8b29bb21 and #167"),
    ]
    assert _ledger_citations(synthetic) == [
        f"tests/test_example.py:7: # see `{_LEDGER_DIR}task-1-report.md`"
    ]


def test_no_tracked_source_cites_the_ledger() -> None:
    """The run ledger is gitignored and archived per wave; a citation into it rots on archive."""
    hits = _ledger_citations(_tracked_lines())
    assert not hits, (
        f"{len(hits)} tracked line(s) cite the gitignored run ledger; cite a commit hash, an issue number, "
        "or state the rule instead:\n" + "\n".join(hits)
    )
