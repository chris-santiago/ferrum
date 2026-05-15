---
name: regression-test
description: >
  Proactively write regression tests after any bug fix. This skill should be
  invoked WITHOUT the user asking whenever you fix a bug, apply a correction,
  resolve unexpected behavior, or make a change that addresses a defect — even
  small one-line fixes. Trigger phrases and contexts include: applying a fix,
  changing code because something "was wrong" or "didn't work", closing a gap
  found by code archaeology, correcting a stale test expectation, wiring a
  missing encoding/channel, or any edit whose intent is "this was broken and now
  it works." Also trigger when the user says "add regression tests", "prevent
  this from regressing", "test this fix", or /regression-test. If you just
  fixed something and are about to report it as done — STOP and invoke this
  skill first. The cost of a missed regression test is a repeat bug; the cost
  of invoking this skill unnecessarily is near zero.
---

# Regression Test Skill

You just fixed a bug. Before reporting the fix as complete, write regression
tests that would **catch this exact bug if it were reintroduced**.

## Why this matters

Bug fixes without regression tests are temporary. The same mistake gets made
again in a refactor, a dependency update, or a parallel feature branch — and
nothing catches it. A regression test encodes the *specific failure mode* so
the test suite acts as institutional memory.

## Workflow

### 1. Identify what changed and why

Read the unstaged diff (or the most recent commit if already committed):

```
git diff          # unstaged
git diff --cached # staged
git show HEAD     # last commit
```

From the diff, answer:
- **What was the bug?** (e.g., "missing `color='y'` encoding made all bars the same color")
- **What was the root cause?** (e.g., "encode() call omitted the color channel")
- **What's the observable symptom?** (e.g., "SVG has only 1 distinct fill color instead of 3")

### 2. Design the test

A good regression test:
- **Reproduces the symptom**, not just the fix. Test the output a user would see (SVG content, DataFrame shape, error message), not internal implementation details.
- **Is minimal**. Smallest input that triggers the bug. Don't reuse large fixtures when 3 rows suffice.
- **Fails on the old code, passes on the new code.** Mentally revert the fix and confirm the test would fail.
- **Has a descriptive name** that explains the regression: `test_class_balance_chart_bars_colored_by_class`, not `test_fix_123`.
- **Includes a docstring** starting with "Regression:" that explains what broke and when.

### 3. Find the right test file

Match the test to the subsystem that was fixed:

- **Python figure functions** → `tests/diagnostics/test_<module>.py`
- **Python chart/encoding** → `tests/test_silent_drop_remediation.py` or `tests/test_pipeline_regression.py`
- **Interactive/WASM rendering** → `tests/test_interactive_regression.py`
- **Rust tessellation/scene** → `crates/ferrum-wasm/src/scene_load.rs` (mod tests)
- **Rust transforms** → `crates/ferrum-core/src/transform/<module>.rs` (mod tests)
- **Rust marks** → `crates/ferrum-core/src/render/marks/<module>.rs` (mod tests)

Read 5-10 lines of the target test file to match the existing patterns (imports, fixtures, assertion style).

### 4. Write the tests

Scale the number of tests to the bug's surface area — don't cap at an arbitrary
number. A one-line missing argument might need 1 test. A pipeline bug that
affects multiple mark types or code paths needs one test per affected path.
Think about *every way this class of bug could manifest*, not just the single
instance that was reported.

**Categories to consider** (write tests for each that applies):

1. **The direct regression test** — reproduces the exact reported symptom.
2. **Sibling paths** — if the fix touched one branch of a match/if, test the
   other branches too. If `mark_hex` was broken, are `mark_geoshape` and other
   polygon marks also covered?
3. **Boundary/edge cases** — degenerate inputs (empty data, single row, NaN,
   zero-area geometry), extreme values, type mismatches the fix now handles.
4. **Property assertions** — verify structural invariants the fix relies on
   (e.g., "all polygon centroids inside plot area", "mesh vertices are finite",
   "fill color is non-null").
5. **Integration/mixed-scene** — when the fix touches infrastructure (a renderer,
   a transform, a layout pass), test that all consumers still work together.
6. **"Still works" tests** — if the fix changed behavior, confirm the happy path
   and any related features didn't break.

Use judgment: a narrow fix gets 1-3 tests, a pipeline fix gets 10-20+.

For **Python tests** (pytest):
```python
def test_<thing>_<expected_behavior>():
    """Regression: <what broke> when <condition>."""
    # Minimal setup
    # Exercise the fixed code path
    # Assert the symptom is absent
```

For **Rust tests** (cargo test):
```rust
#[test]
fn <thing>_<expected_behavior>() {
    // Minimal setup
    // Exercise the fixed code path
    // Assert the symptom is absent
}
```

### 5. Run the tests

```bash
# Python — run just the new tests
uv run pytest <test_file>::<test_name> -v

# Rust — run just the new tests
cargo test -p <crate> -- <test_name>
```

Confirm they pass. If a test fails, fix the test (not the code — the fix is already correct).

### 6. Report

After writing and running the tests, briefly summarize:
- How many tests added
- What each test guards against
- Which file they live in

Then proceed with the commit or whatever the user asked for next.

## Anti-patterns to avoid

- **Testing the implementation, not the symptom.** Don't assert that a specific function was called or a specific line exists. Assert the *output* a user would see.
- **Overly broad tests.** `assert "<svg" in svg` doesn't catch regressions — it passes even when the chart is blank. Assert specific content (distinct fill colors, presence of text labels, non-empty data).
- **Copy-pasting the fix into the test.** The test should exercise the public API, not replicate internal logic.
- **Skipping the run.** Always run the test before reporting it. A test that doesn't pass is worse than no test.

## When NOT to invoke this skill

- Pure refactors with no behavioral change (renaming, reformatting, moving code)
- Documentation-only changes
- Adding new features (that's not a bug fix — use TDD instead)
- The user explicitly says they don't want tests
