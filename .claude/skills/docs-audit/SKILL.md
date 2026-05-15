---
name: docs-audit
description: >
  Audit the ferrum docs site (docs/site/) for staleness against the current source code.
  Checks for stale phase references, outdated "not yet" claims, missing API pages, stub
  pages, stale docstrings, comparison-page drift, and PNG/image staleness. Produces a
  structured report with file paths, line numbers, and suggested fixes. Use when the user
  says "audit the docs", "docs staleness check", "are the docs up to date", /docs-audit,
  "check docs for stale content", "do the docs need updating", or after any phase lands
  on main and the docs branch rebases. Also use proactively after rebasing docs/continue
  onto main — new phases routinely introduce API surface that the docs haven't caught up
  with yet.
---

# Docs-site staleness audit

A read-only audit of `docs/site/` against the current source code. Produces an actionable
report — never modifies files.

## How to run

Run the scanner script first, then interpret its output:

```bash
uv run python .claude/skills/docs-audit/scripts/scan.py
```

The script prints JSON to stdout — one finding per line. Read its output and present
a structured report to the user.

If the script is missing or broken, fall back to the manual checks below.

## What the script checks

The script handles all deterministic scanning:

1. **Stale phase markers** — greps `docs/site/**/*.md` for patterns like `Phase \d+`
   and cross-references against `docs/superpowers/ferrum-phases.md` to see which phases
   are done. Any "Phase N" mentioned as future/planned/upcoming where N is already done
   is a finding.

2. **Stub pages** — finds `!!! info "Stub"`, `"Content lands in a later build phase"`,
   or similar placeholder language.

3. **Stale future-tense language in docstrings** — greps `src/ferrum/**/*.py` for
   `"placeholder"`, `"Phase \d+"`, `"not yet"`, `"will be"`, `"currently ignored"`,
   `"when .* lands"`, `"once .* ships"` inside docstrings. These surface on API
   reference pages via mkdocstrings.

4. **Missing API pages** — compares public submodules exported in `src/ferrum/__init__.py`
   against `docs/site/api/*.md` files and `zensical.toml` nav entries.

5. **Stale module references** — greps docs for known renames (e.g. `ferrum.figure`
   instead of `ferrum.plots`).

6. **Comparison-page drift** — extracts the list of visualizers, figure helpers, and
   marks from source, then checks whether comparison pages' coverage claims match.

7. **PNG staleness** — for each `![...](img/...)` reference in guide pages, checks
   whether the linked PNG exists.

## What needs LLM judgment

After the script runs, review its findings and apply judgment:

- **Phase references that are informational** (e.g. "Added in Phase 10") are not stale
  — only forward-looking references ("Phase 11 will add...") are findings.
- **Comparison coverage tables** may be structurally correct but have stale descriptions.
  Read the table content and cross-reference against actual source to decide.
- **"Not yet" claims** in comparison pages need verification — check whether the claimed
  gap still exists by grepping the source for the relevant function/class.

## Report format

Present findings grouped by severity:

### STALE — content contradicts current source
Highest priority. The docs say something that is no longer true.

### STUB — placeholder content that should be real
Pages with stub markers that can now be filled in.

### MISSING — expected content that doesn't exist
API pages, nav entries, or coverage-table rows that should exist but don't.

### WARNING — possibly stale, needs human review
Future-tense docstrings, PNG age, comparison descriptions that might be outdated.

Each finding should include:

```
[SEVERITY] file:line — what's wrong
  Suggested fix: ...
```

## Known rename history

These renames have happened across phases. The script checks for stale references:

| Old | New | Phase |
|-----|-----|-------|
| `ferrum.figure` | `ferrum.plots` | 10 |
| `ferrum.figures` | `ferrum.plots` | 10 |
| `fm.figure` | `fm.plots` | 10 |

Add new renames to the script's `KNOWN_RENAMES` list as they occur.
