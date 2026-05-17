---
name: code-archaeology
description: Use when surfacing unimplemented features, silently dropped parameters, dead code paths, skipped tests, or spec-vs-impl gaps in the ferrum codebase — especially before marking a phase done, before a major refactor, or when behavior a user expects is mysteriously absent. Trigger phrases: "what's not implemented", "find unfinished work", "what's silently dropped", "audit for stubs", "find deferred features", /code-archaeology.
---

# Code Archaeology

Systematically sweep the codebase for comments, stubs, and structural patterns that reveal planned-but-unbuilt behavior. The most dangerous gaps are not `TODO` comments — they are **silent drops**: parameters accepted, warnings suppressed, match arms that fall through to `_ => {}`.

## Dispatch Strategy

Split by domain and run three agents in parallel. Each reads every file in scope — grep first to find candidates, then read ±15 lines of surrounding context.

| Agent | Scope |
|---|---|
| Python | `src/ferrum/` |
| Rust | `crates/ferrum-core/src/` + `crates/ferrum-wasm/src/` |
| Tests + Docs | `tests/`, `design-docs/`, `design-docs/superpowers/`, `ferrum-spec.md` |

## Python Patterns

**Obvious:** `# TODO`, `# FIXME`, `raise NotImplementedError`, `raise ValueError("not yet...")`, `pass` in non-abstract methods.

**Non-obvious — these matter more:**

- `_SILENT_CHANNELS`-style frozensets with inline comments — channels accepted but never rendered
- `warn_once(...)` calls — anything behind `warn_once` is silently dropped
- `_ = param` — explicit discard of an accepted parameter
- Parameters in the docstring that don't appear in the function body
- Parameters that immediately `raise ValueError` for any non-default value (signature promises, implementation refuses)
- `if False:` or hardcoded-`False` branch guards on feature paths

### Verification required before reporting — do not skip this step

Comments saying `"Reserved for future use"`, `"no-op today"`, `"deferred"`, or `"accepted but not yet wired"` are **candidates**, not confirmed gaps. Before reporting any such comment as a finding, verify the actual code path:

1. **Python kwargs:** does the kwarg appear in `to_mark_kwargs_dict()`? Does a matching field exist in `MarkKwargsSpec`? Does the Rust renderer read that field and produce a visual effect?
2. **Python channels:** is the channel in `_SILENT_CHANNELS`, or does it have a rendering path in the SVG/WASM renderer?
3. **Rust comments:** does the function/field/arm still match the "no-op" description, or has the implementation been added since the comment was written?

If the implementation exists and is wired end-to-end, classify the finding as `STALE_COMMENT` (comment no longer matches code), not `SILENT_DROP`. Stale comments are low-severity cleanup, not bugs. Reporting an implemented kwarg as a gap wastes remediation effort and creates false urgency.

## Rust Patterns

**Obvious:** `todo!()`, `unimplemented!()`, `// TODO`, `// FIXME`.

**Non-obvious — these matter more:**

- `_ => {}` / `_ => None` / `_ => Ok(Vec::new())` in match arms on mark, encoding, or channel dispatch tables — the canonical place where a Python-side variant is accepted but silently dropped on the Rust side
- `#[allow(dead_code)]` on an entire module or struct (not a single field) — blanket suppression hides which helpers are unused
- `let _ = value` — explicit discard in a function body
- `eprintln!` / `println!` in non-test code — debug output left in production
- Struct fields declared but never read anywhere in the crate
- Doc comments saying "no-op", "reserved", or "accepted but not yet wired" — verify the code actually matches

## Tests + Docs Patterns

**Tests:**
- `@pytest.mark.skip(reason=...)` and `@pytest.mark.xfail` — read the reason; is it a known engine bug or a deferred feature?
- `pass` or `assert True` inside a test function body — stubs never filled in
- Commented-out assertions: `# assert result == expected`

**Spec / docs:**
- Spec sections with "not yet", "deferred", "TBD", "pending", "Phase N"
- Cross-references to phases that are done — did the promised wiring actually land?
- Documented public API parameters — cross-check against Python signatures and Rust `EncodingSpec`/`MarkKwargsSpec` fields for silent gaps

## Output Format

Each agent returns structured findings:

```
FILE: <path>
LINE: <number>
TYPE: [TODO | FIXME | STUB | todo!() | SILENT_DROP | DEAD_PARAM | SKIP | SPEC_GAP | STALE_COMMENT | other]
COMMENT/CODE: <exact text>
WHAT'S MISSING: <one sentence — or for STALE_COMMENT: "Comment says X but code already does Y">
VERIFIED: <yes — traced code path | no — comment only>
```

`STALE_COMMENT` is for comments/docstrings that describe absent behavior but the implementation is already present. Do not promote a `STALE_COMMENT` to a higher-severity type without code-path verification.

## Synthesis

After all agents return, group findings into a prioritized report:

1. **Active bugs** — code wired incorrectly (wrong key, missing branch in dispatch)
2. **Skipped tests** — known bugs never fixed
3. **Silent drops** — accepted by API, never forwarded or rendered
4. **Features raise at runtime** — spec-documented, raise on use
5. **Missing spec implementations** — not present in source at all
6. **Stale docs** — comments/docstrings that no longer match the implementation

Save the report to `design-docs/superpowers/followups/YYYY-MM-DD-code-archaeology.md`.
