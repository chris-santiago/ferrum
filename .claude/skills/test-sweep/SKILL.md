---
name: test-sweep
description: >
  Iterative combinatorial test-and-fix campaign for ferrum. Writes systematic test modules
  targeting cross-cutting dimensions (mark×channel, mark×coord, composite×channel, etc.),
  runs them, fixes all failures via TDD, then uses failure patterns to derive the next
  test suite. Repeats for N rounds or until convergence. Fully autonomous except for
  design-decision pauses. Use when the user says /test-sweep, "sweep for bugs",
  "systematic test campaign", "find gaps across the API surface", or wants a multi-round
  test-driven quality pass.
---

# Test Sweep

Iterative combinatorial test-and-fix campaign. Each round targets a new cross-cutting
dimension of the ferrum API, writes a parametrized test module, fixes all surfaced bugs,
and uses the failure patterns to derive the next round's target.

## Usage

```
/test-sweep [N]
```

N = max rounds (default 3). Stops early if a round produces zero new failures.

---

## Philosophy

- **Fixes must be robust and cohesive** — no band-aids, no isolated patches that ignore
  the surrounding design. Every fix must respect the architecture.
- **End-user experience first** — will this behavior surprise someone reading the API
  docs? Would a data scientist expect this to work?
- **Pause on ambiguity** — when a fix could go multiple directions with different UX
  tradeoffs, stop and present options to the user before implementing.

---

## Core Loop

For each round (up to N):

### 1. Select dimension

Pick the next untested dimension from the seed queue. Skip any dimension whose test file
already exists and passes.

### 2. Write test modules (Python + Rust in parallel)

Dispatch **both** coding agents in parallel for each dimension:

#### 2a. Python integration test → `python-coder`

Write a parametrized test file in `tests/` following established patterns:
- Polars DataFrame fixture with appropriate data shapes
- Per-mark/function config dicts specifying base encodings and applicable channels
- `_make_cases()` → `@pytest.mark.parametrize`
- Assertions: SVG differs (channel/transform has effect) or no-crash (`"<svg" in svg`)
- Run with `--noconftest` to avoid Phase 10 shap guards

#### 2b. Rust unit test → `rust-coder`

Write a `#[cfg(test)]` module or extend an existing one in the relevant
`crates/ferrum-core/src/` module. Rust tests target the computation layer directly:
- Construct `RecordBatch` inputs with edge-case shapes (1 row, all-null, zero-variance)
- Call the transform's `apply()` or internal function directly
- Assert exact output values, column counts, row counts, and error variants
- Faster feedback loop — no Python/rendering overhead

**Why both:** Python tests catch integration bugs (PyO3 boundary, rendering pipeline,
encoding wiring). Rust tests catch computation bugs (off-by-one, NaN propagation, empty
partition handling) with precise value assertions that SVG checks cannot provide.

Do not use `general-purpose`, `Explore`, or write inline for either track.

### 3. Collect failures

Run the test module. Categorize each failure as:
- **Test-design issue** — wrong API call, bad data shape, invalid expectation
- **Real bug** — code silently drops a feature, crashes, or produces wrong output
- **Design decision** — multiple valid fixes with different UX implications

### 4. Fix

- **Test-design issues**: dispatch to `python-coder` to fix the test file.
- **Design decisions**: pause, present options with tradeoffs, get user choice.
- **Real bugs**: dispatch to the language-specific coding agent:
  - Python fixes (`src/ferrum/`, `tests/`) → `python-coder` agent
  - Rust fixes (`crates/ferrum-core/`) → `rust-coder` agent
  - Both → dispatch both agents with clear boundaries

**Never edit `.py` or `.rs` files directly in the orchestrator.** The coding agents
embed review principles from `.claude/skills/{python,rust}-review/` and produce
code that passes the lite-review gate on first attempt.

All fixes must consider:
- Does this change the public API contract?
- Is the fix cohesive with the surrounding subsystem?
- Would a user be surprised by this behavior?

### 5. Verify green

Run both test tracks until green:
- **Python:** Re-run the test module, then the full sweep suite
  (`test_*_matrix.py` / `test_*_handling.py`) for regressions.
- **Rust:** `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core` — run the new Rust tests
  plus the full crate test suite.

Both must pass before proceeding.

### 6. Review-lite gate

Stage changes. Dispatch `python-review-lite` and/or `rust-review-lite` per the
language of files touched. Act on verdict:
- **clean** → proceed
- **block** → fix findings, re-stage, re-dispatch
- **escalate** → surface to user, halt round

### 7. Commit

Commit test module + fixes as a single commit per round:
```
test+fix(sweep-N): <dimension name> — M gaps found, M fixed
```

### 8. Derive follow-up dimensions

Analyze the round's failures:
- If 2+ failures shared a subsystem root cause, generate a targeted follow-up
  dimension scoped to that subsystem (e.g., "Bin transform × all input shapes").
- Append derived dimensions to the queue (they'll be picked up in later rounds).

### 9. Terminate or continue

- If round produced 0 new failures → stop, report convergence.
- If round N reached → stop, report summary.
- Otherwise → next round.

---

## Seed Dimensions (ordered by expected yield)

| # | Dimension | Python test | Rust test scope |
|---|---|---|---|
| 1 | mark × channel | `test_channel_mark_matrix.py` | `render/` mark dispatch |
| 2 | mark × CoordFlip | `test_coord_flip_matrix.py` | `render/` coord flip |
| 3 | mark × position adjustment | `test_position_matrix.py` | `position/` adjustments |
| 4 | composite mark × channel | `test_composite_mark_channels.py` | — (Python-only) |
| 5 | figure function × kwargs passthrough | `test_figure_kwargs_passthrough.py` | — (Python-only) |
| 6 | channel × scale type | `test_channel_scale_types.py` | `scale/` type inference |
| 7 | mark × null/NaN handling | `test_null_nan_handling.py` | `transform/` null paths |
| 8 | figure function × data shapes | `test_figure_data_shapes.py` | — (Python-only) |
| 9 | transform × edge cases | `test_transform_edge_cases.py` | `transform/` edge cases |
| 10 | composition × encoding inheritance | `test_composition_inheritance.py` | `resolve/` inheritance |

Dimensions marked "Python-only" have no meaningful Rust-level unit test because the
logic lives entirely in the Python layer (figure function construction, composite mark
desugaring, kwargs threading).

---

## Pause Conditions

The skill halts and asks the user when:
- A fix requires choosing between multiple valid UX behaviors
- Review-lite returns `block` or `escalate`
- A fix would change existing public API behavior
- A failure reveals a fundamental architecture issue (not a local bug)

---

## Output

After all rounds complete, write `.claude/output/test-sweep/TEST_SWEEP_REPORT.md` (gitignored) with:
- Per-round summary: dimension, tests written, failures found, fixes applied
- Design decisions made (with rationale)
- Derived dimensions discovered (queued for next invocation)
- Remaining xfails with dated reasons

---

## Guardrails

- Never xfail without a dated reason AND a follow-up dimension queued
- All fixes go through review-lite before commit
- **All `.py`/`.rs` writes go through `python-coder` or `rust-coder` agents** — the
  orchestrator reads, analyzes, and dispatches but never edits source directly
- Skip dimensions whose test files already exist and pass
- Do not modify existing passing tests (only add new ones or fix test-design issues)
- Maximum 50 test cases per module (keep parametrize matrices bounded)
