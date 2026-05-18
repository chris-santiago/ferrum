"""Docs-site staleness scanner.

Scans docs/site/**/*.md and source code for staleness indicators.
Prints one JSON finding per line to stdout.

Usage:
    uv run python .claude/skills/audit-docs/scripts/scan.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root
DOCS = ROOT / "docs" / "site"
SRC = ROOT / "src" / "ferrum"
PHASES_FILE = ROOT / "docs" / "superpowers" / "ferrum-phases.md"
ZENSICAL = ROOT / "zensical.toml"

KNOWN_RENAMES = [
    ("ferrum.figure", "ferrum.plots"),
    ("ferrum.figures", "ferrum.plots"),
    ("fm.figure", "fm.plots"),
]

STALE_DOCSTRING_PATTERNS = [
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\bnot yet\b", re.IGNORECASE),
    re.compile(r"\bwill be\b", re.IGNORECASE),
    re.compile(r"\bcurrently ignored\b", re.IGNORECASE),
    re.compile(r"\bwhen .{3,30} lands?\b", re.IGNORECASE),
    re.compile(r"\bonce .{3,30} ships?\b", re.IGNORECASE),
    re.compile(r"\bPhase \d+\b"),
]

STUB_PATTERNS = [
    re.compile(r'!!!\s+info\s+"Stub"', re.IGNORECASE),
    re.compile(r"Content lands in a later build phase", re.IGNORECASE),
    re.compile(r"coming soon", re.IGNORECASE),
]

PHASE_FUTURE_PATTERNS = [
    re.compile(r"Phase \d+\s+(will|is planned|is a .* commitment|not (?:yet |currently )shipping)", re.IGNORECASE),
    re.compile(r"\(Phase \d+\)"),
    re.compile(r"Phase \d+(?:\s|$|\.)"),
]


def emit(severity: str, path: str, line: int, message: str, fix: str = ""):
    print(json.dumps({
        "severity": severity,
        "file": path,
        "line": line,
        "message": message,
        "fix": fix,
    }))


def done_phases() -> set[int]:
    if not PHASES_FILE.exists():
        return set()
    text = PHASES_FILE.read_text()
    phases = set()
    for m in re.finditer(r"\|\s*(\d+)\s*\|.*?\|\s*done\s*\|", text, re.IGNORECASE):
        phases.add(int(m.group(1)))
    for m in re.finditer(r"\|\s*(\d+[a-z]?)\s*\|.*?\|\s*done\s*\|", text, re.IGNORECASE):
        num = int(re.match(r"\d+", m.group(1)).group())
        phases.add(num)
    return phases


def scan_docs_phase_refs(done: set[int]):
    for md in sorted(DOCS.rglob("*.md")):
        rel = str(md.relative_to(ROOT))
        for i, line in enumerate(md.read_text().splitlines(), 1):
            for pat in PHASE_FUTURE_PATTERNS:
                for m in pat.finditer(line):
                    phase_num_match = re.search(r"Phase (\d+)", m.group())
                    if phase_num_match:
                        n = int(phase_num_match.group(1))
                        if n in done:
                            emit("STALE", rel, i,
                                 f"Phase {n} referenced as future/planned but is done",
                                 f"Remove or update Phase {n} reference")


def scan_stubs():
    for md in sorted(DOCS.rglob("*.md")):
        rel = str(md.relative_to(ROOT))
        for i, line in enumerate(md.read_text().splitlines(), 1):
            for pat in STUB_PATTERNS:
                if pat.search(line):
                    emit("STUB", rel, i, f"Stub marker: {line.strip()!r}",
                         "Replace with real content")


def scan_stale_docstrings():
    for py in sorted(SRC.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        rel = str(py.relative_to(ROOT))
        in_docstring = False
        for i, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = not in_docstring
                # single-line docstring
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    for pat in STALE_DOCSTRING_PATTERNS:
                        if pat.search(stripped):
                            emit("WARNING", rel, i,
                                 f"Possibly stale docstring: {stripped[:120]}",
                                 "Update docstring language")
                    continue
            if in_docstring:
                for pat in STALE_DOCSTRING_PATTERNS:
                    if pat.search(line):
                        emit("WARNING", rel, i,
                             f"Possibly stale docstring: {stripped[:120]}",
                             "Update docstring language")
                        break


def scan_missing_api_pages():
    init = SRC / "__init__.py"
    if not init.exists():
        return
    text = init.read_text()
    from_imports = set()
    for m in re.finditer(r"from\s+ferrum\.(\w+)\s+import", text):
        from_imports.add(m.group(1))

    public_submodules = set()
    for mod in from_imports:
        mod_path = SRC / mod
        if mod_path.is_dir() or (SRC / f"{mod}.py").exists():
            public_submodules.add(mod)

    api_dir = DOCS / "api"
    existing_pages = set()
    if api_dir.exists():
        for md in api_dir.glob("*.md"):
            existing_pages.add(md.stem)
    existing_pages.discard("ferrum")

    nav_entries = set()
    if ZENSICAL.exists():
        zt = ZENSICAL.read_text()
        for m in re.finditer(r'"ferrum\.(\w+)"', zt):
            nav_entries.add(m.group(1))

    for mod in sorted(public_submodules):
        if mod.startswith("_"):
            continue
        if mod not in existing_pages:
            emit("MISSING", f"docs/site/api/{mod}.md", 0,
                 f"Public module ferrum.{mod} has no API reference page",
                 f"Create api/{mod}.md with mkdocstrings directive")
        if mod not in nav_entries:
            emit("MISSING", "zensical.toml", 0,
                 f"ferrum.{mod} not in zensical.toml nav",
                 f'Add {{ "ferrum.{mod}" = "api/{mod}.md" }} to API Reference nav')


def scan_known_renames():
    for md in sorted(DOCS.rglob("*.md")):
        rel = str(md.relative_to(ROOT))
        for i, line in enumerate(md.read_text().splitlines(), 1):
            for old, new in KNOWN_RENAMES:
                if old in line and new not in line:
                    emit("STALE", rel, i,
                         f"References renamed module {old!r} (now {new!r})",
                         f"Replace {old!r} with {new!r}")


def scan_missing_pngs():
    for md in sorted(DOCS.rglob("*.md")):
        rel = str(md.relative_to(ROOT))
        md_dir = md.parent
        for i, line in enumerate(md.read_text().splitlines(), 1):
            for m in re.finditer(r"!\[.*?\]\((img/[^)]+)\)", line):
                img_path = md_dir / m.group(1)
                if not img_path.exists():
                    emit("MISSING", rel, i,
                         f"Referenced image {m.group(1)} does not exist",
                         "Generate or fix the image reference")


def scan_comparison_drift():
    visualizers = set()
    for py in sorted((SRC / "_diagnostics" / "visualizers").rglob("*.py")):
        if "__pycache__" in str(py) or "__init__" in str(py):
            continue
        for line in py.read_text().splitlines():
            m = re.match(r"class\s+(\w+Visualizer)\(", line)
            if m and m.group(1) != "FerrumVisualizer":
                visualizers.add(m.group(1))

    helpers = set()
    for py in sorted((SRC / "plots").rglob("*.py")):
        if "__pycache__" in str(py) or "__init__" in str(py):
            continue
        for line in py.read_text().splitlines():
            m = re.match(r"def\s+(\w+)\(", line)
            if m and not m.group(1).startswith("_"):
                helpers.add(m.group(1))

    exports = set()
    init_text = (SRC / "__init__.py").read_text()
    for m in re.finditer(r'"(\w+)"', init_text):
        exports.add(m.group(1))

    comparison_dir = DOCS / "comparison"
    if not comparison_dir.exists():
        return
    for md in sorted(comparison_dir.glob("*.md")):
        rel = str(md.relative_to(ROOT))
        text = md.read_text()
        if "not (yet)" in text or "does not (yet)" in text or "not yet" in text.lower():
            for i, line in enumerate(text.splitlines(), 1):
                if "not yet" in line.lower() or "does not" in line.lower():
                    emit("WARNING", rel, i,
                         f"Possible stale gap claim: {line.strip()[:120]}",
                         "Verify this gap still exists against current source")


def main():
    done = done_phases()
    scan_docs_phase_refs(done)
    scan_stubs()
    scan_stale_docstrings()
    scan_missing_api_pages()
    scan_known_renames()
    scan_missing_pngs()
    scan_comparison_drift()


if __name__ == "__main__":
    main()
