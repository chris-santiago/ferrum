# Ferrum: Development Meta-Analysis

## The numbers

| Metric | Value |
|---|---|
| Calendar days | **9** (May 9–17, 2026) |
| Total commits | **911** |
| Python source | **33,288 lines** across 89 files |
| Rust source | **63,651 lines** across 153 files |
| Python tests | **2,278** test functions in 125 files |
| Rust tests | **1,068** `#[test]` functions |
| Design specs | **36** documents |
| Implementation plans | **38** documents |
| Phases completed | **11 of 12** (Phase 12 in-flight) |
| Peak day | **213 commits** (May 11) |

Phases 1–7 (skeleton through first rendered chart) landed on **day 1**. Phase 8a (full grammar API with 31 encoding channels) landed **day 2**. Phase 10 (26 model-diagnostic marks, 21 figure functions, 25 sklearn visualizers) landed **day 3**. Phase 11 (WASM interactive renderer with selections, zoom, pan) landed by **day 6**.

## What made it work

### 1. Spec-first, plan-second, code-third — no exceptions

Every phase followed the same ritual: brainstorm → write design spec → write implementation plan → execute plan → mark done. The spec (`ferrum-spec.md`, 1,653 lines) was written before line one of code. Phase-level specs decompose the concept spec into buildable units. Plans decompose specs into task lists. No phase was ever started without both documents approved.

This front-loaded the hard decisions (Arrow CDI vs. IPC, JSON vs. binary serialization, no-matplotlib constraint, themes-as-values) into a single brainstorming session on day 1. Every subsequent session executed against settled architecture — zero time re-litigating.

### 2. Six-layer automation architecture

The `.claude/` directory is as engineered as the library itself — 9 agent definitions, 12 skills, a shared severity rubric (S1–S5), and explicit dispatch rules:

- **Layer 1 (coding agents):** `python-coder` and `rust-coder` embed the full review principles from their respective heavyweight review skills. Code is written to pass review on the first attempt, not iteratively corrected.
- **Layer 2 (commit gates):** `python-review-lite` and `rust-review-lite` run on every staged diff before commit. Read-only. Three consecutive blocks escalate to heavyweight review. The orchestrator (Opus) never commits without a gate pass.
- **Layer 3 (heavyweight reviews):** Full subsystem audits at phase boundaries. Catch sibling drift, API inconsistency, and structural decay that accumulate across a phase's worth of commits.
- **Layer 4 (quality campaigns):** `/bug-hunt` dispatches 11 parallel agents across subsystems. `/test-sweep` runs multi-round combinatorial TDD. `/gallery-audit` renders 38 plot types against sklearn/seaborn/yellowbrick and judges them with a rubric. `/code-archaeology` sweeps the entire codebase for unimplemented features and spec drift.
- **Layer 5 (remediation agents):** `gallery-fixer`, `schwabish-fixer`, `bug-hunter` — each reads campaign output and closes findings autonomously, delegating code changes back to the Layer 1 coding agents.
- **Layer 6 (utility skills):** `/regression-test` (auto-triggered after every bug fix), `/ferrum-docstrings`, `/docs-audit`, `/release`.

The key insight: **coding agents never commit**. The orchestrator handles staging, gate dispatch, and commit. This separation means the review pipeline is structurally unforgeable — you can't skip it.

### 3. Orchestrator + specialist model split

Opus orchestrates: it reads specs, decomposes work, dispatches agents, interprets results, handles cross-cutting decisions. Sonnet executes: it writes Python, writes Rust, runs tests. This matches the cost/capability curve — architectural judgment is expensive and rare, line-by-line coding is cheap and frequent. A single Opus context window manages the session while parallel Sonnet agents do the mechanical work.

The dispatch rule is enforced in `CLAUDE.md`: "Never use `general-purpose`, `claude`, or `Explore` agents for code that writes or modifies `.py` or `.rs` files." This prevents the orchestrator from doing coding work itself and ensures every line of code goes through an agent that has internalized the review principles.

### 4. The CLAUDE.md as institutional memory

At 245 lines, `CLAUDE.md` is the project's constitution. It encodes:
- Build commands (with known platform gotchas)
- Hard constraints ("no matplotlib, ever")
- Dispatch rules (which agent handles which files)
- Review escalation protocol (when lite → heavyweight)
- The implementation philosophy ("do the work now, do it the right way")
- Where everything lives (a lookup table, not prose)

Every session — every agent — starts by reading this file. It's the mechanism by which decisions made in session 1 are enforced in session 50. The `memory/` system supplements it with cross-session context that doesn't belong in committed code (user preferences, workflow feedback, stale-state warnings).

### 5. Quality campaigns as ratchets

The project didn't just write code and move on. After phases stabilized, systematic sweeps found what human review missed:

- `/test-sweep` wrote **132 combinatorial tests** across 5 rounds (mark×channel, facet×layer, coord×position, theme×mark, encoding×facet×theme), found and fixed **2 bugs**.
- `/bug-hunt` dispatched **11 parallel agents** to write edge-case tests per subsystem.
- `/gallery-audit` rendered **38 plot types** against 4 reference libraries, scored them against a rubric, and fed findings to `gallery-fixer`.
- `/code-archaeology` swept the **entire codebase** for silent drops, dead code, and spec drift — found 4 active bugs, 7 high-severity Rust gaps, 11 silent-drop mark kwargs, and 6 stale doc references. All fixed.

These aren't one-time runs. They're repeatable skills that can be re-invoked after any significant change. Each run either confirms quality or surfaces regressions.

### 6. Feedback loops that actually close

The `memory/` system captures operational lessons: "subagents falsely claimed file deletions" → always verify independently. "Plans with inline code blocks waste tokens" → plans describe WHAT, not HOW. "Integration tests must not mock the database" → test against real state.

These aren't suggestions — they're loaded into every session and shape agent behavior. The feedback loop is: something goes wrong → save a memory → next session reads it → the failure mode is structurally prevented.

## What this architecture produces

A Rust-backed Python visualization library with:
- 31 encoding channels, 20+ mark types, 21 figure functions, 25 sklearn visualizers
- A WASM interactive renderer with selections, zoom, pan, and linked views
- Zero matplotlib dependency (hard constraint from day 1)
- 3,346 tests across Python and Rust
- A docs site (in-progress on a worktree branch)
- A release pipeline with conventional commits, changelog generation, and PyPI publishing

Built in 9 days by one human and an agentic Claude framework.

## The meta-lesson

The velocity didn't come from typing faster. It came from:
1. **Never starting without a spec** — eliminates rework from misunderstood requirements
2. **Enforcing review structurally** — gates on every commit, not periodic audits
3. **Separating judgment from execution** — Opus reasons, Sonnet codes
4. **Making quality campaigns repeatable** — sweeps are skills, not one-time heroics
5. **Treating agent infrastructure as product** — the `.claude/` directory has its own README, architecture diagram, and severity rubric

The 911 commits aren't 911 manual actions. They're the output of a system that was designed to produce correct code at high throughput, with the human providing direction, constraints, and architectural taste — not keystrokes.
