# Lightweight Review Agents — Design

**Date:** 2026-05-11
**Status:** Draft, pending user approval

## Problem

Ferrum has two heavyweight review skills today:

- `.claude/skills/python-review/` — senior Python refactoring & API-design pass
- `.claude/skills/rust-review/` — senior Rust refactoring & API-design pass

Both are interactive, multi-phase (orient → diagnose → propose → execute → review), produce six-section reports, and target whole packages or crates. They are the right tool when a human invokes them to clean up a subsystem.

Separately, ferrum has the `gallery-fixer` agent, which runs autonomously after a gallery audit to close default-output gaps. The fixer writes code; nothing today guards against the fixer introducing code-quality regressions while it closes visual gaps.

**Goal:** add autonomous code-quality guardrails that run with or after `gallery-fixer` and prevent quality regressions from being committed, without requiring human intervention on a clean pass.

## Non-goals

- Replace the heavyweight `python-review` / `rust-review` skills. Those remain the right call for "audit this package" or "this module feels off."
- Cover work outside the `gallery-fixer` flow. The new agents are scoped to diff-level regression checks.
- Provide refactoring suggestions for the user. The output is a binary gate decision (with findings as evidence), not a roadmap.

## Design

### Two new agents

- `.claude/agents/python-review-lite.md`
- `.claude/agents/rust-review-lite.md`

Both follow the existing pattern established by `gallery-fixer.md` and `gallery-judge.md`: read-only autonomous subagents, dispatched by the parent (orchestrator) Claude session.

### Orchestrator flow

Today, `gallery-fixer` makes edits, runs tests, and reports back — it does not commit. The parent (orchestrator) Claude session decides what to commit and when. The lite agents slot in between gallery-fixer's return and the parent's commit decision.

```
parent session
  ├─ Agent(gallery-fixer)
  │     - makes edits in working tree
  │     - runs tests / regenerates goldens / verifies row
  │     - returns: fixer report + list of files changed
  │     (does not stage, does not commit — unchanged from today)
  │
  ├─ Parent stages the gallery-fixer's changes
  │     git add <files from gallery-fixer report>
  │
  ├─ Agent(python-review-lite)       ◄── if any .py files staged
  │  Agent(rust-review-lite)         ◄── if any .rs files staged
  │     - dispatched in parallel when both apply
  │     - each reads `git diff --cached`
  │     - each returns: status ∈ {clean, block, escalate} + verdict file
  │
  └─ Orchestrator interprets:
        clean    → commit (parent runs git commit)
        block    → un-stage; return verdict to gallery-fixer; loop
        escalate → leave staged; surface to user; halt
```

No changes to `gallery-fixer.md` itself. The only existing-doc change is in `CLAUDE.md` (see Integration touchpoints).

### Inputs

Each lite agent reads:

1. `git diff --cached --name-only` — list of staged files
2. `git diff --cached` — the staged change itself
3. Full current contents of each touched file via `Read` (for context around new lines)
4. `CLAUDE.md` (repo root) for ferrum-specific constraints
5. The agent's own `references/checklist.md` (its trimmed idiom list)
6. Optional: `ferrum-spec.md` if a chart factory was touched

The agents do **not** read the wider package, neighbor files, or git history beyond `--cached`.

### Workflow (single phase per agent)

1. **Read the diff.** Categorize each change (mark renderer, new transform, composite expansion, util helper, etc.).
2. **Apply the diff-level idiom checklist** to new and changed lines only. Whole-file architectural assessment is out of scope.
3. **Run linters** if available:
   - python-review-lite: `ruff check <files>` (existing dev dep)
   - rust-review-lite: `cargo clippy --message-format=short -- -D warnings` on the affected crate
4. **Write `verdict.md`** at `.claude/skills/gallery-audit/output/_review_lite/<date>_<agent>.md` (timestamped, so multiple cycles are preserved as audit trail).
5. **Return a one-line summary** plus the status word to the parent.

No architecture map. No proposal-before-execute. No multi-phase review. No write access.

### Severity rubric

Same five levels as the heavyweight skills (S1–S5):

- **S1** — cosmetic inconsistency
- **S2** — readability / maintainability issue
- **S3** — structural cohesion issue
- **S4** — risky design flaw or bug-prone seam
- **S5** — critical correctness or API hazard

Each finding includes:
- Severity (S1–S5)
- Confidence (high / medium / low)
- File + line range
- What / Why it matters / Suggested fix (one to three sentences each)

### Block / escalate rules

| Condition | Status |
|---|---|
| No S3+ findings, all linters pass | **clean** |
| ≥1 S3 finding, OR failing `ruff` / `cargo clippy -D warnings` | **block** |
| ≥1 S4+ finding | **escalate** |
| ≥3 consecutive block cycles on the same row | **escalate** (loop-breaker) |

The cycle counter is the parent orchestrator's responsibility — passed in as part of the dispatch prompt. The agent itself is stateless.

### Verdict file format

```markdown
---
status: clean | block | escalate
agent: python-review-lite | rust-review-lite
date: 2026-05-11
cycle: 1
n_findings: {S1: 0, S2: 1, S3: 0, S4: 0, S5: 0}
files_reviewed:
  - src/ferrum/_diagnostics/charts.py
  - src/ferrum/figures.py
linters:
  ruff: pass
  mypy: not_run
---

## Findings

### S3 — structural cohesion — high confidence — `src/ferrum/_diagnostics/charts.py:200-260`
**What**: `roc_chart` now takes 8 parameters; 4 are mode flags.
**Why it matters**: Boolean parameter smell (heuristic #1). Future fixes will add more.
**Suggested fix**: bundle into a `RocAnnotations` typed options dict, or split into `roc_chart` + `roc_chart_with_ci`.

## Notes (non-blocking)
- Ruff passed clean.
- mypy not run (no mypy config detected in project).
```

### Trimmed checklist content

Each agent ships exactly one `references/checklist.md`. Content is a strict subset of the heavyweight `references/heuristics.md`, restricted to patterns observable in a diff:

**python-review-lite checklist:**
1. Boolean / mode-flag parameter added to a public function
2. Dict-shaped domain data introduced where a dataclass would clarify
3. Hidden side effect newly introduced (env var, filesystem, logging, global mutation)
4. New utility function added to a `utils.py` / `common.py` / `helpers.py` style file
5. New top-level `try/except` that swallows exceptions silently
6. Unused import, dead code, sentinel return values
7. Public API leak (new top-level name not curated via `__all__`)
8. New broad `except Exception` block at a library boundary

**rust-review-lite checklist:**
1. New boolean parameter on a public function
2. New `panic!` / `unwrap` / `expect` on a library-boundary path
3. Inconsistent error type (returning `anyhow::Error` in a crate that uses a typed `Error`)
4. New macro that could be a function
5. New trait with exactly one implementor
6. New `impl` block with only one method that could be inline
7. New `pub` item not exposed via `lib.rs` curation
8. New compatibility shim without a TODO / sunset note

The heavyweight `heuristics.md` files remain unchanged and continue to inform the interactive skills.

### Integration touchpoints

1. **`.claude/agents/gallery-fixer.md`** — no workflow change; gallery-fixer already does not commit. A single new note at the bottom records that the parent now stages and runs review-lite before committing, so the fixer doesn't need to worry about post-fix quality verification.
2. **`CLAUDE.md` (repo root)** — add a new "Code-quality guardrails" section that:
   - Documents the four review surfaces: `python-review` (skill), `rust-review` (skill), `python-review-lite` (agent), `rust-review-lite` (agent). The two heavyweight skills are not currently documented in `CLAUDE.md` either; this section backfills that gap.
   - Explains when each is used: skills for human-invoked subsystem audits; lite agents auto-invoked between `gallery-fixer`'s return and the parent's commit.
   - States the block / escalate semantics.
   - Notes the audit-trail location (`.claude/skills/gallery-audit/output/_review_lite/`).
3. **`README.md`** — no change (these are internal tooling).

### What the lite agents deliberately do not do

- Never write code (no Edit, Write, or NotebookEdit tool access in their frontmatter)
- Never read files outside the staged diff (the workflow constrains them; the file frontmatter permits broader access for the rare case where a referenced file is needed for context, but the workflow says "don't")
- Never analyze whole-file architecture
- Never propose refactors
- Never run the full test suite (only fast linters)
- Never interact with the user — they return to the orchestrator only

## Severity gating: why S3+

S3 ("structural cohesion") is where the heavyweight skills draw the "real problems" line. S1 (cosmetic) and S2 (readability) findings still appear in the verdict for the audit trail, but they don't block — gallery-fixer is allowed to leave behind a slightly-rougher edge if the visual fix is otherwise correct. S4+ (risky design flaws and correctness issues) jumps straight to escalation because those are exactly the cases where the orchestrator should not auto-resolve.

This threshold is a starting point; the verdict file format records the per-severity counts so we can re-tune later from data.

## Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Reviewer is wrong about an S3 finding | gallery-fixer can override in its next-cycle response; ≥3 block cycles escalates to user |
| Reviewer blocks the same finding cycle after cycle | The 3-cycle loop-breaker forces escalation |
| Reviewer takes too long | These agents read only the diff; expected runtime is seconds. If a single dispatch exceeds two minutes, escalate. |
| Linter is missing | Verdict records `linters.ruff: not_available` and downgrades the block threshold to "findings only" for that pass |
| Both agents disagree (e.g. Python diff says clean, Rust says block) | Block wins. The orchestrator surfaces both verdicts together to gallery-fixer. |

## Open questions

None at design time. All major choices were made during brainstorming (see session notes 2026-05-11):

- Action mode: blocking reviewer, no writes
- Scope: only files gallery-fixer touched
- Threshold: S3+ blocks
- Artifact type: agents (not skills)
