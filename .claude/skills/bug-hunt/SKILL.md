---
name: bug-hunt
description: Repeatable parallel test-writing campaign for ferrum. Dispatches one bug-hunter agent per subsystem to find edge-case bugs across the entire pipeline. Use when the user says /bug-hunt, "write edge case tests", "find bugs", "test coverage sweep", or wants a systematic bug-finding pass. Accepts an optional subsystem name to scope: /bug-hunt scale-stat, /bug-hunt coerce-transport, /bug-hunt marks-rendering, /bug-hunt composition-facet, /bug-hunt figure-api, /bug-hunt model-diagnostics, /bug-hunt phase-11-interactive.
---

# Bug Hunt

Parallel edge-case test campaign covering the entire ferrum pipeline. One `bug-hunter` agent per subsystem writes tests, runs them, and reports failures as bugs.

## Subsystem table

| Key | Source scope | Existing tests |
|---|---|---|
| `scale-stat` | `crates/ferrum-core/src/scale/`, `crates/ferrum-core/src/transform/` | `tests/test_scales.py`, `tests/test_stat_engine.py` |
| `coerce-transport` | `src/ferrum/_coerce.py` | `tests/test_coerce.py` |
| `marks-rendering` | `crates/ferrum-core/src/render/marks/` | `tests/test_marks.py`, `tests/test_render.py` |
| `composition-facet` | `src/ferrum/composition.py`, `src/ferrum/_layer.py`, `src/ferrum/coord.py` | `tests/test_composition.py`, `tests/test_facet.py`, `tests/test_coord.py` |
| `figure-api` | `src/ferrum/chart.py`, `src/ferrum/marks/`, `src/ferrum/plots/` | `tests/test_phase_9_figures.py`, `tests/test_phase_9_e2e.py` |
| `model-diagnostics` | `src/ferrum/_diagnostics/` | `tests/diagnostics/` |
| `phase-11-interactive` | `src/ferrum/_interactive.py`, `crates/ferrum-wasm/src/` | `tests/test_phase_11d/` |

## Procedure

### Step 1 — Parse args

If the user provided a subsystem name (e.g. `/bug-hunt scale-stat`), run only that subsystem. Otherwise run all 7.

### Step 2 — Check phases

Read `docs/superpowers/ferrum-phases.md`. Skip any subsystem whose corresponding phase is not `done`:
- `scale-stat` → Phase 4 + 5
- `coerce-transport` → Phase 2
- `marks-rendering` → Phase 7
- `composition-facet` → Phase 8a
- `figure-api` → Phase 9
- `model-diagnostics` → Phase 10
- `phase-11-interactive` → Phase 11

### Step 3 — Dispatch agents in parallel

Send ALL active subsystems in a **single message** as parallel Agent tool calls — one per subsystem. Use `subagent_type: "bug-hunter"` for each.

Prompt template per agent (fill in the blanks):

```
Subsystem: <key>
Source paths: <source scope from table above>
Existing test paths: <existing tests from table above>

You are a bug-hunter agent for the ferrum project. Follow the instructions in your agent definition exactly. Write tests/test_bug_hunt_<key>.py, run it, and return your verdict.
```

### Step 4 — Collect verdicts

Each agent returns: tests added count, list of failures (test name + error), status (clean / bugs-found).

### Step 5 — Run the new tests

Python:
```bash
uv run pytest tests/test_bug_hunt_*.py --tb=short -q
```
(Or `uv run pytest tests/test_bug_hunt_<key>.py --tb=short -q` if scoped.)

Rust (for `scale-stat` and `marks-rendering` only):
```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  cargo test -p ferrum-core --tests -- bug_hunt --nocapture 2>&1
```

### Step 6 — Write the report

Append a timestamped run section to `.claude/skills/bug-hunt/output/BUG_REPORT.md`. Format:

```markdown
## Run — <ISO timestamp>

### <subsystem-key>
- Python tests added: N  |  Failures: N
- Rust tests added:   N  |  Failures: N   (or "n/a")
- Status: clean | BUGS FOUND

<failure details if any — test name, language, error message>
```

Create the file if it does not exist. Never overwrite — always append.

### Step 7 — Print summary table

```
Subsystem             | Py tests | Py fails | Rs tests | Rs fails | Status
----------------------|----------|----------|----------|----------|--------
scale-stat            | 12       | 2        | 8        | 1        | BUGS
coerce-transport      | 8        | 0        | n/a      | n/a      | clean
marks-rendering       | 15       | 1        | 7        | 0        | BUGS
composition-facet     | 10       | 0        | n/a      | n/a      | clean
figure-api            | 14       | 3        | n/a      | n/a      | BUGS
model-diagnostics     | 11       | 0        | n/a      | n/a      | clean
phase-11-interactive  | 9        | 1        | n/a      | n/a      | BUGS
```

Then list every failing test with its error, grouped by subsystem. These are the bugs.

## Output artifacts

| Artifact | Path | Committed? |
|---|---|---|
| Bug report | `.claude/skills/bug-hunt/output/BUG_REPORT.md` | No (gitignored) |
| Python test files | `tests/test_bug_hunt_<subsystem>.py` | Yes |
| Rust test files | `crates/ferrum-core/tests/bug_hunt_<subsystem>.rs` | Yes |

All test files are real and should be committed after review. Failing tests are kept with a `# BUG:` / `// BUG:` comment — they are the bug evidence.

## Repeatability

Running `/bug-hunt` again reruns all agents. Each agent overwrites its test file (fresh read of current source). Previously-fixed bugs will now pass. `BUG_REPORT.md` accumulates all runs with timestamps.
