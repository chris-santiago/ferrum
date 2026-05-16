# Bug Hunt — Skill & Agent Design

**Date:** 2026-05-14
**Status:** approved

---

## Purpose

A repeatable, parallel test-writing campaign that covers the entire ferrum pipeline. Running `/bug-hunt` dispatches one `bug-hunter` agent per subsystem; each agent reads source, identifies uncovered edge cases, writes a test file, runs it, and reports failures as real bugs. The output is both a persistent `BUG_REPORT.md` and committed `tests/test_bug_hunt_*.py` files.

---

## Structure

Two artifacts:

1. **Skill** — `.claude/skills/bug-hunt/SKILL.md`
   The orchestrator. Handles invocation, argument parsing, agent dispatch, result collection, report writing, and summary output.

2. **Agent** — `.claude/agents/bug-hunter.md`
   A single reusable agent definition dispatched once per subsystem. All subsystems share the same workflow; only the scoping prompt differs.

---

## Skill — `/bug-hunt`

### Invocation forms

| Command | Behavior |
|---|---|
| `/bug-hunt` | Full 7-subsystem sweep |
| `/bug-hunt <subsystem>` | Single subsystem (e.g. `/bug-hunt scale-stat`) |

### Orchestrator steps

1. Parse args — determine which subsystems to run (all or named subset).
2. Read `docs/superpowers/ferrum-phases.md` to confirm which phases are active (skip subsystems whose phase is not yet `done`).
3. Dispatch `bug-hunter` agents in parallel, one per subsystem, each with its scoped prompt (source paths + existing test paths + subsystem name).
4. Collect agent verdicts.
5. Run `uv run pytest tests/test_bug_hunt_*.py --tb=short` (or scoped file if single subsystem).
6. Write/update `.claude/skills/bug-hunt/output/BUG_REPORT.md` — append a timestamped run section with one subsystem block each.
7. Print summary table to the session.

### Summary table format

```
Subsystem           | Tests added | Failures (bugs) | Status
--------------------|-------------|-----------------|--------
scale-stat          | 12          | 2               | BUGS
coerce-transport    | 8           | 0               | clean
...
```

---

## Agent — `bug-hunter`

### Inputs (passed in the dispatch prompt)

- Subsystem name
- Source file/directory paths to read
- Existing test file paths to read (to understand what is already covered)

### Workflow

1. Read all source files in scope.
2. Read all existing tests in scope.
3. Identify uncovered edge cases across these categories:
   - **Null / NaN / missing data** — columns with nulls, all-null columns, null in key encoding channel
   - **Empty inputs** — zero-row DataFrame, zero-column DataFrame
   - **Single-row / single-value** — degenerate domain, single category
   - **Extreme values** — very large floats, very small floats near zero, negative values in log scale
   - **Type boundaries** — int vs float columns, boolean columns, string-typed numerics, mixed types
   - **Composition corners** — empty layer stack, layer with no data, facet with one panel, facet with many panels
   - **Spec round-trips** — JSON serialize → deserialize, field presence after round-trip
   - **Contract pins** — key SVG structural properties that should not silently change
4. Write `tests/test_bug_hunt_<subsystem>.py` — one test function per edge case, using `pytest` conventions.
5. Run `uv run pytest tests/test_bug_hunt_<subsystem>.py -x --tb=short`.
6. Return verdict: tests added count, list of failures with error messages, any escalations.

### Test file conventions

- File: `tests/test_bug_hunt_<subsystem>.py`
- Overwritten on each invocation for that subsystem only; other subsystem files are untouched.
- Failing tests are **kept** — they are the bug evidence. Each failing test gets a `# BUG:` comment on the `def` line describing the symptom.
- Tests that pass establish contract baselines.
- No golden SVG comparisons — structural assertions only (element counts, attribute presence, value ranges).
- All tests are written in Python against the Python API regardless of whether the subsystem source is Rust or Python. For Rust-backed subsystems (`marks-rendering`, `phase-11-interactive`), the agent reads Rust source to understand behavior but writes Python tests that call `ferrum.*` and assert on SVG/HTML output.

---

## Subsystem table

| Subsystem key | Source scope | Existing tests read |
|---|---|---|
| `scale-stat` | `crates/ferrum-core/src/scale/`, `crates/ferrum-core/src/transform/` | `tests/test_scales.py`, `tests/test_stat_engine.py` |
| `coerce-transport` | `src/ferrum/_coerce.py`, `src/ferrum/_transport.py` | `tests/test_coerce.py`, `tests/test_transport.py` |
| `marks-rendering` | `crates/ferrum-core/src/render/marks/` | `tests/test_marks.py`, `tests/test_render.py` |
| `composition-facet` | `src/ferrum/composition.py`, `src/ferrum/_layer.py`, `src/ferrum/coord.py` | `tests/test_composition.py`, `tests/test_facet.py`, `tests/test_coord.py` |
| `figure-api` | `src/ferrum/chart.py`, `src/ferrum/marks/`, `src/ferrum/plots/` | `tests/test_phase_9_figures.py`, `tests/test_phase_9_e2e.py` |
| `model-diagnostics` | `src/ferrum/_diagnostics/` | `tests/diagnostics/*.py` |
| `phase-11-interactive` | `src/ferrum/_interactive.py`, `crates/ferrum-wasm/src/` | `tests/test_phase_11d/` |

---

## Output artifacts

| Artifact | Path | Committed? |
|---|---|---|
| Bug report | `.claude/skills/bug-hunt/output/BUG_REPORT.md` | No (gitignored) |
| Test files | `tests/test_bug_hunt_<subsystem>.py` | Yes |

The output directory is gitignored alongside the gallery audit output. Test files are real tests and live in `tests/` permanently.

---

## Repeatability

- `/bug-hunt` reruns the full sweep. Each agent overwrites its test file with a fresh read of the current source.
- `/bug-hunt <subsystem>` reruns just one agent.
- Previously-failing tests that were fixed will now pass — the file updates reflect the current state of the code.
- `BUG_REPORT.md` accumulates timestamped runs; it does not overwrite prior runs.
