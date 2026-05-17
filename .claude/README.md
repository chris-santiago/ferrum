# Ferrum — Claude Automations

This directory contains ferrum's project-specific Claude Code automations: **skills** (invoked interactively with a slash command), **agents** (subagents dispatched by skills or the orchestrator), and **output** (ephemeral artifacts like review verdicts).

---

## System architecture

The automation system has six layers, from innermost (every task) to outermost (periodic campaigns):

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 6: Utility skills (human-invoked, standalone)            │
│  /ferrum-docstrings  /regression-test  /docs-audit  /release    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Remediation agents (dispatched by campaigns)          │
│  gallery-fixer  schwabish-fixer  bug-hunter                     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Quality campaigns (human-invoked, parallel dispatch)  │
│  /bug-hunt  /test-sweep  /gallery-audit  /schwabish-improve     │
│  /code-archaeology                                              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Heavyweight review skills (human-invoked)             │
│  /python-review  /rust-review                                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Commit gates (every commit, read-only)                │
│  python-review-lite  rust-review-lite                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Coding agents (every coding task)                     │
│  python-coder  rust-coder                                       │
└─────────────────────────────────────────────────────────────────┘
```

### End-to-end flow

```
User request (coding task)
  │
  ├─ Python? ──→ python-coder agent ──→ orchestrator stages
  │                                          │
  ├─ Rust?   ──→ rust-coder agent   ──→ orchestrator stages
  │                                          │
  │                                    python-review-lite / rust-review-lite
  │                                          │
  │                                    clean? ──→ commit
  │                                    block? ──→ fix → re-stage → re-gate
  │                                    escalate? ──→ halt, surface to user
  │
  ├─ Phase boundary? ──→ /python-review or /rust-review (heavyweight)
  │
  └─ Quality sweep? ──→ /bug-hunt, /gallery-audit, /schwabish-improve, /test-sweep
```

### Campaign flows

```
/gallery-audit
  └─ audit.py generate          (renders all panels)
  └─ gallery-judge × N          (one per row, parallel)
  └─ audit.py report            (builds REPORT.md)

"fix the gallery findings"
  └─ gallery-fixer agent
       ├─ reads REPORT.md + panel PNGs
       ├─ dispatches python-coder / rust-coder for code changes
       ├─ python-review-lite / rust-review-lite  (pre-commit gate)
       └─ orchestrator commits clean rows

/bug-hunt
  └─ bug-hunter × 11            (one per subsystem, parallel)

/test-sweep
  └─ orchestrator-driven multi-round TDD campaign
       ├─ writes combinatorial test modules
       ├─ dispatches python-coder / rust-coder for fixes
       └─ repeats for N rounds or until convergence

/schwabish-improve --from-audit
  └─ schwabish-judge × N        (one per gallery row, parallel)
  └─ schwabish-fixer × N        (delegates to python-coder for code)
  └─ python-review-lite         (pre-commit gate)
  └─ orchestrator commits clean rows

/code-archaeology
  └─ 3 parallel sweep agents    (Python, Rust, Tests+Docs)
  └─ consolidated report
```

---

## Quick reference

| What you want | Run |
|---|---|
| Implement a Python feature/fix | Orchestrator dispatches `python-coder` |
| Implement a Rust feature/fix | Orchestrator dispatches `rust-coder` |
| Find edge-case bugs across the pipeline | `/bug-hunt` |
| Find edge-case bugs in one subsystem | `/bug-hunt <subsystem>` |
| Multi-round combinatorial test-and-fix | `/test-sweep` |
| Compare ferrum plots to sklearn/seaborn/yellowbrick | `/gallery-audit` |
| Walk through audit results and decide what to fix | `/gallery-feedback` |
| Apply Schwabish text-integration principles | `/schwabish-improve <target>` |
| Apply to the entire gallery autonomously | `/schwabish-improve --from-audit` |
| Senior Python code review on a module/subsystem | `/python-review` |
| Senior Rust code review on a crate/module | `/rust-review` |
| Add or update a docstring | `/ferrum-docstrings` |
| Surface unimplemented features, silent drops, spec gaps | `/code-archaeology` |
| Audit docs site for staleness | `/docs-audit` |
| Write regression tests after a bug fix | `/regression-test` (also auto-triggered) |
| Cut a release | `/release` |

---

## Layer 1 — Coding agents

General-purpose coding agents dispatched by the orchestrator for **all** coding tasks. They embed the review principles from the corresponding heavyweight review skill so code passes the lite-review gate on first attempt.

**Dispatch rule (enforced in CLAUDE.md):** Never use `general-purpose`, `claude`, or `Explore` agents for code that writes or modifies `.py` or `.rs` files.

### `python-coder`

**File:** `agents/python-coder.md`

General-purpose Python coding agent. Handles features, bug fixes, refactors, and tests in `src/ferrum/`, `tests/`, and `scripts/`. Embeds all 9 operating principles, 12 refactoring heuristics, and the S1–S5 self-review checklist from `/python-review`.

**Tools:** Read, Edit, Write, Bash, Glob, Grep, Agent

### `rust-coder`

**File:** `agents/rust-coder.md`

General-purpose Rust coding agent. Handles features, bug fixes, refactors, and tests in `crates/ferrum-core/` and `crates/ferrum-wasm/`. Embeds all 10 operating principles, 10 refactoring heuristics, and the S1–S5 self-review checklist from `/rust-review`.

**Tools:** Read, Edit, Write, Bash, Glob, Grep, Agent

---

## Layer 2 — Commit gates

Read-only agents dispatched by the orchestrator **before every commit** that touches source code. They never write code — only verdict files.

### `python-review-lite`

**File:** `agents/python-review-lite.md`

Reviews staged `*.py` diff against a 10-item idiom checklist + `ruff`. Returns `clean` / `block` / `escalate`.

**Verdict path:** `.claude/output/review-lite/<ISO-timestamp>_python.md`
**Tools:** Read, Grep, Glob, Bash

### `rust-review-lite`

**File:** `agents/rust-review-lite.md`

Reviews staged `*.rs` diff against an 11-item idiom checklist + `cargo clippy -D warnings`. Returns `clean` / `block` / `escalate`.

**Verdict path:** `.claude/output/review-lite/<ISO-timestamp>_rust.md`
**Tools:** Read, Grep, Glob, Bash

---

## Layer 3 — Heavyweight review skills

Full subsystem audits — orient, diagnose, propose, execute, review. Invoked by the user when something "feels off" or at phase boundaries.

### `/python-review`

**File:** `skills/python-review/SKILL.md`

Multi-phase review of a Python package, module, or subsystem. Produces a six-section report: architecture map, drift findings tagged S1–S5, refactor roadmap, proposed first patch.

Trigger when something feels off: utility-module sprawl, dict-shaped domain data, mode-flag creep, inconsistent return shapes, sibling API drift, overgrown classes, leaky internal imports.

### `/rust-review`

**File:** `skills/rust-review/SKILL.md`

Same structure for Rust. Targets a crate, module, or subsystem. Looks for naming inconsistency, boolean/config explosion, leaky APIs, panic-prone library code, unnecessary genericity, parallel-API drift, compatibility scar tissue.

---

## Layer 4 — Quality campaigns

Systematic sweeps that dispatch agents in parallel. Human-invoked.

### `/bug-hunt`

**File:** `skills/bug-hunt/SKILL.md`

Parallel edge-case test campaign. Dispatches one `bug-hunter` agent per subsystem (11 total) to write edge-case tests, run them, and report failures as bugs.

**Subsystems:** `scale-stat`, `coerce-transport`, `marks-rendering`, `composition-facet`, `figure-api`, `model-diagnostics`, `phase-11-interactive`, `layout`, `draw`, `projection`, `stats-transforms`

**Output:** `output/bug-hunt/BUG_REPORT.md`

### `/test-sweep`

**File:** `skills/test-sweep/SKILL.md`

Iterative combinatorial test-and-fix campaign. Writes systematic test modules targeting cross-cutting dimensions (mark×channel, mark×coord, composite×channel, etc.), runs them, fixes all failures via TDD, then uses failure patterns to derive the next test suite. Repeats for N rounds or until convergence. Delegates code fixes to `python-coder` / `rust-coder`.

### `/gallery-audit`

**File:** `skills/gallery-audit/SKILL.md`

Renders the same plot in ferrum and reference libraries (sklearn, seaborn, yellowbrick, scikit-plot) using default settings, then dispatches `gallery-judge` agents in parallel to score each row against a fixed rubric. Produces a prioritized `REPORT.md` punchlist. 38 plot types.

**Output:** `output/gallery-audit/` — `REPORT.md`, per-row `verdict.md`, panel PNGs

### `/gallery-feedback`

**File:** `skills/gallery-feedback/SKILL.md`

Human-in-the-loop follow-up to `/gallery-audit`. Walks through every row, shows ferrum's panel alongside comparator panels, and asks what should change. Compiles decisions into a structured remediation plan.

### `/schwabish-improve`

**File:** `skills/schwabish/SKILL.md`

Applies Schwabish "integrate text and graphics" principles. Four T-categories: active title (T1), direct labels (T2), callouts (T3), inline metrics (T4).

Two modes:
- **Advisory** (`/schwabish-improve <target>`) — dispatches `schwabish-judge` per target, writes verdicts. Read-only.
- **Gallery-autonomous** (`/schwabish-improve --from-audit`) — judges all gallery rows, then dispatches `schwabish-fixer` to apply objective findings, runs lite-review gate.

### `/code-archaeology`

**File:** `skills/code-archaeology/SKILL.md`

Dispatches three parallel agents (Python, Rust, Tests+Docs) to sweep the entire codebase for unimplemented features, silently dropped parameters, dead code paths, skipped tests, and spec-vs-impl gaps. Report saved to `design-docs/superpowers/followups/`.

---

## Layer 5 — Remediation agents

Agents dispatched by campaigns or the user to close findings. All delegate actual code writing to the Layer 1 coding agents.

### `gallery-fixer`

**File:** `agents/gallery-fixer.md`

Dispatched after `/gallery-audit` to close HIGH-severity findings. Reads `REPORT.md` and panel PNGs, plans each fix, then dispatches `python-coder` or `rust-coder` for the code change. Does not commit — orchestrator handles the lite-review gate.

Trigger: "fix the gallery findings", "work the punchlist", "close the ferrum/seaborn gaps"

### `schwabish-fixer`

**File:** `agents/schwabish-fixer.md`

Dispatched by `/schwabish-improve --from-audit`. Reads each row's `schwabish_verdict.md`, filters to `objective: true` findings, checks idempotence, then delegates code edits to `python-coder`. Restricted to gallery panel scripts — never edits `src/ferrum/`.

### `bug-hunter`

**File:** `agents/bug-hunter.md`

Dispatched by `/bug-hunt`, one per subsystem. Writes edge-case test files, runs them, reports failures as bugs. Never fixes bugs — only surfaces them.

**Tools:** Read, Edit, Write, Bash, Glob, Grep

### `gallery-judge`

**File:** `agents/gallery-judge.md`

Dispatched by `/gallery-audit`, one per row. Reads panel PNGs, applies rubric, writes `verdict.md`. Read-only.

### `schwabish-judge`

**File:** `agents/schwabish-judge.md`

Dispatched by `/schwabish-improve`, one per chart. Applies Schwabish T-categories, writes `schwabish_verdict.md`. Read-only.

---

## Layer 6 — Utility skills

Standalone skills for specific tasks. Not part of a campaign flow.

### `/ferrum-docstrings`

**File:** `skills/ferrum-docstrings/SKILL.md`

Reference for adding or updating docstrings. NumPy convention, PyO3 placement rules, ferrum-specific example shapes.

### `/regression-test`

**File:** `skills/regression-test/SKILL.md`

Auto-invoked after any bug fix. Writes regression tests that pin the corrected behavior. Should be triggered without the user asking whenever a bug is fixed.

### `/docs-audit`

**File:** `skills/docs-audit/SKILL.md`

Audits the docs site (`docs/site/`) for staleness against current source code. Checks for stale phase references, outdated claims, missing API pages, stale docstrings, comparison-page drift, and PNG staleness.

### `/release`

**File:** `skills/release/SKILL.md`

Bumps version in `pyproject.toml` and `Cargo.toml`, generates changelog from conventional commits, updates `docs/site/changelog.md`, creates a GitHub release with tag.

---

## Review surface comparison

| Surface | Type | Invoked by | Scope | Writes code? |
|---|---|---|---|---|
| `python-coder` | agent | orchestrator (coding tasks) | any Python file | yes |
| `rust-coder` | agent | orchestrator (coding tasks) | any Rust file | yes |
| `python-review-lite` | agent | orchestrator (before `*.py` commit) | staged diff only | **never** |
| `rust-review-lite` | agent | orchestrator (before `*.rs` commit) | staged diff only | **never** |
| `python-review` | skill | human (`/python-review`) | whole package/subsystem | yes, with approval |
| `rust-review` | skill | human (`/rust-review`) | whole crate/subsystem | yes, with approval |

### Severity rubric (shared across all surfaces)

| Tag | Meaning | Lite-agent behavior |
|---|---|---|
| S1 | cosmetic inconsistency | noted, non-blocking |
| S2 | readability / maintainability | noted, non-blocking |
| S3 | structural cohesion issue | **blocks** commit |
| S4 | risky design flaw or bug-prone seam | **escalates** to user |
| S5 | critical correctness or API hazard | **escalates** to user |

### Verdict audit trail

Lite-agent verdicts are written to `.claude/output/review-lite/<ISO-timestamp>_{python,rust}.md`. Each verdict carries YAML frontmatter (`status`, `cycle`, `n_findings` by severity, `linters`, `files_reviewed`) followed by per-finding prose. The directory is gitignored.

---

## Directory layout

```
.claude/
├── README.md                          ← this file
├── agents/
│   ├── python-coder.md                ← Layer 1: Python coding agent
│   ├── rust-coder.md                  ← Layer 1: Rust coding agent
│   ├── python-review-lite.md          ← Layer 2: Python commit gate
│   ├── rust-review-lite.md            ← Layer 2: Rust commit gate
│   ├── gallery-fixer.md               ← Layer 5: gallery remediation
│   ├── gallery-judge.md               ← Layer 5: gallery audit judge
│   ├── schwabish-fixer.md             ← Layer 5: Schwabish remediation
│   ├── schwabish-judge.md             ← Layer 5: Schwabish audit judge
│   └── bug-hunter.md                  ← Layer 5: edge-case test writer
├── skills/
│   ├── python-review/                 ← Layer 3: heavyweight Python review
│   ├── rust-review/                   ← Layer 3: heavyweight Rust review
│   ├── bug-hunt/                      ← Layer 4: parallel test campaign
│   ├── test-sweep/                    ← Layer 4: combinatorial TDD campaign
│   ├── gallery-audit/                 ← Layer 4: default-output comparison
│   ├── gallery-feedback/              ← Layer 4: interactive audit walkthrough
│   ├── schwabish/                     ← Layer 4: text-integration audit
│   ├── code-archaeology/              ← Layer 4: unimplemented feature sweep
│   ├── ferrum-docstrings/             ← Layer 6: docstring conventions
│   ├── regression-test/               ← Layer 6: post-fix regression tests
│   ├── docs-audit/                    ← Layer 6: docs staleness check
│   └── release/                       ← Layer 6: version bump + changelog
└── output/                                ← all ephemeral output (gitignored)
    ├── review-lite/                   ← commit-gate verdicts
    ├── gallery-audit/                 ← panel PNGs, verdicts, REPORT.md
    ├── bug-hunt/                      ← BUG_REPORT.md
    └── test-sweep/                    ← TEST_SWEEP_REPORT.md
```
