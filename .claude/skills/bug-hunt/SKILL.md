---
name: bug-hunt
description: Repeatable parallel test-writing campaign for ferrum. Dispatches one bug-hunter agent per subsystem to find edge-case bugs across the entire pipeline. Use when the user says /bug-hunt, "write edge case tests", "find bugs", "test coverage sweep", or wants a systematic bug-finding pass. Accepts an optional subsystem name to scope: /bug-hunt scale-stat, /bug-hunt coerce-transport, /bug-hunt marks-rendering, /bug-hunt composition-facet, /bug-hunt figure-api, /bug-hunt model-diagnostics, /bug-hunt phase-11-interactive, /bug-hunt layout, /bug-hunt draw, /bug-hunt projection, /bug-hunt stats-transforms.
---

# Bug Hunt

Parallel edge-case test campaign covering the entire ferrum pipeline. One `bug-hunter` agent per subsystem writes tests, runs them, and reports failures as bugs.

## Subsystem table

Subsystems have three modes: **Py+Rs** (Python test file + Rust integration test file), **Py** (Python only), **Rs** (Rust only — internal modules with no direct Python API).

| Key | Mode | Source scope | Existing tests |
|---|---|---|---|
| `scale-stat` | Py+Rs | `crates/ferrum-core/src/scale/`, `crates/ferrum-core/src/transform/` | `tests/test_scales.py`, `tests/test_stat_engine.py` |
| `coerce-transport` | Py | `src/ferrum/_coerce.py` | `tests/test_coerce.py` |
| `marks-rendering` | Py+Rs | `crates/ferrum-core/src/render/marks/` | `tests/test_marks.py`, `tests/test_render.py` |
| `composition-facet` | Py | `src/ferrum/composition.py`, `src/ferrum/_layer.py`, `src/ferrum/coord.py` | `tests/test_composition.py`, `tests/test_facet.py`, `tests/test_coord.py` |
| `figure-api` | Py | `src/ferrum/chart.py`, `src/ferrum/marks/`, `src/ferrum/plots/` | `tests/test_phase_9_figures.py`, `tests/test_phase_9_e2e.py` |
| `model-diagnostics` | Py | `src/ferrum/_diagnostics/` | `tests/diagnostics/` |
| `phase-11-interactive` | Py+Rs | `src/ferrum/_interactive.py`, `crates/ferrum-wasm/src/` | `tests/test_phase_11d/` |
| `layout` | Rs | `crates/ferrum-core/src/layout/` | `tests/test_layout_engine.py` |
| `draw` | Rs | `crates/ferrum-core/src/render/draw.rs` | `tests/test_render.py` |
| `projection` | Rs | `crates/ferrum-core/src/projection.rs` | `tests/test_phase_11d/test_coord_and_marks.py` |
| `stats-transforms` | Rs | `crates/ferrum-core/src/transform/stats.rs` | `tests/test_stat_engine.py` |

## Procedure

### Step 1 — Parse args

If the user provided a subsystem name (e.g. `/bug-hunt scale-stat`), run only that subsystem. Otherwise run all 11.

### Step 2 — Check phases

Read `design-docs/superpowers/ferrum-phases.md`. Skip any subsystem whose corresponding phase is not `done`:
- `scale-stat` → Phase 4 + 5
- `coerce-transport` → Phase 2
- `marks-rendering` → Phase 7
- `composition-facet` → Phase 8a
- `figure-api` → Phase 9
- `model-diagnostics` → Phase 10
- `phase-11-interactive` → Phase 11
- `layout` → Phase 6
- `draw` → Phase 7
- `projection` → Phase 11
- `stats-transforms` → Phase 5

### Step 3 — Dispatch agents in parallel

Send ALL active subsystems in a **single message** as parallel Agent tool calls — one per subsystem. Use `subagent_type: "bug-hunter"` for each.

Prompt template per agent (fill in the blanks):

```
Subsystem: <key>
Mode: <Py+Rs | Py | Rs>
Source paths: <source scope from table above>
Existing test paths: <existing tests from table above>

You are a bug-hunter agent for the ferrum project. Follow the instructions in your agent definition exactly. For the mode given, write the appropriate test file(s), run them, and return your verdict.
```

### Step 4 — Collect verdicts

Each agent returns: tests added count, list of failures (test name + error), status (clean / bugs-found).

### Step 5 — Run the new tests

Python:
```bash
uv run pytest tests/test_bug_hunt_*.py --tb=short -q
```
(Or `uv run pytest tests/test_bug_hunt_<key>.py --tb=short -q` if scoped.)

Rust (for all Rs and Py+Rs subsystems):
```bash
# ferrum-core integration tests (scale-stat, marks-rendering, layout, draw, projection, stats-transforms)
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  cargo test -p ferrum-core --tests -- bug_hunt --nocapture 2>&1

# ferrum-wasm unit tests (phase-11-interactive inline #[cfg(test)] blocks)
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  cargo test -p ferrum-wasm -- bug_hunt --nocapture 2>&1
```

### Step 6 — Write the report

Append a timestamped run section to `.claude/output/bug-hunt/BUG_REPORT.md`. Format:

```markdown
## Run — <ISO timestamp>

### <subsystem-key>
- Python tests added: N  |  Failures: N
- Rust tests added:   N  |  Failures: N   (or "n/a")
- Status: ✅ clean | 🐛 BUGS FOUND

<failure details if any — test name, language, error message>
```

Create the file if it does not exist. Never overwrite — always append.

### Step 7 — Print summary table

Use ✅ for clean subsystems and 🐛 for subsystems with failures. Example:

```
| Subsystem            | Mode  | Py tests | Py fails | Rs tests | Rs fails | Status   |
|----------------------|-------|----------|----------|----------|----------|----------|
| scale-stat           | Py+Rs | 12       | 2        | 8        | 1        | 🐛 BUGS  |
| coerce-transport     | Py    | 8        | 0        | —        | —        | ✅ clean |
| marks-rendering      | Py+Rs | 15       | 1        | 7        | 0        | 🐛 BUGS  |
| composition-facet    | Py    | 10       | 0        | —        | —        | ✅ clean |
| figure-api           | Py    | 14       | 3        | —        | —        | 🐛 BUGS  |
| model-diagnostics    | Py    | 11       | 0        | —        | —        | ✅ clean |
| phase-11-interactive | Py+Rs | 9        | 1        | 6        | 2        | 🐛 BUGS  |
| layout               | Rs    | —        | —        | 8        | 0        | ✅ clean |
| draw                 | Rs    | —        | —        | 7        | 1        | 🐛 BUGS  |
| projection           | Rs    | —        | —        | 9        | 0        | ✅ clean |
| stats-transforms     | Rs    | —        | —        | 10       | 2        | 🐛 BUGS  |
```

Then list every failing test with its error, grouped by subsystem. These are the bugs.

## Output artifacts

| Artifact | Path | Committed? |
|---|---|---|
| Bug report | `.claude/output/bug-hunt/BUG_REPORT.md` | No (gitignored) |
| Python test files | `tests/test_bug_hunt_<subsystem>.py` | Yes |
| ferrum-core Rust tests | `crates/ferrum-core/tests/bug_hunt_<subsystem>.rs` | Yes |
| ferrum-wasm Rust tests | Inline `#[cfg(test)]` blocks in `crates/ferrum-wasm/src/*.rs` | Yes |

All test files are real and should be committed after review. Failing tests are kept with a `# BUG:` / `// BUG:` comment — they are the bug evidence.

## Repeatability

Running `/bug-hunt` again reruns all agents. Each agent overwrites its test file (fresh read of current source). Previously-fixed bugs will now pass. `BUG_REPORT.md` accumulates all runs with timestamps.
