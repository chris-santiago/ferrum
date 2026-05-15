# Ferrum — Claude Automations

This directory contains ferrum's project-specific Claude Code automations: **skills** (invoked interactively with a slash command) and **agents** (subagents dispatched by skills or the orchestrator to do isolated work).

---

## Quick reference

| What you want | Run |
|---|---|
| Find edge-case bugs across the pipeline | `/bug-hunt` |
| Find edge-case bugs in one subsystem | `/bug-hunt <subsystem>` |
| Compare ferrum plots to sklearn/seaborn/yellowbrick | `/gallery-audit` |
| Walk through audit results and decide what to fix | `/gallery-feedback` |
| Apply Schwabish text-integration principles to a chart | `/schwabish-improve <target>` |
| Apply to the entire gallery autonomously | `/schwabish-improve --from-audit` |
| Senior Python code review on a module/subsystem | `/python-review` |
| Senior Rust code review on a crate/module | `/rust-review` |
| Add or update a docstring | `/ferrum-docstrings` |
| Surface unimplemented features, silent drops, spec gaps | `/code-archaeology` |

---

## Skills

Skills are invoked interactively via slash commands. They orchestrate work — often by dispatching agents in parallel.

### `/bug-hunt` — Parallel edge-case test campaign

**File:** `skills/bug-hunt/SKILL.md`

Runs a parallel test-writing sweep across all 11 ferrum subsystems (or one, if you name it). One `bug-hunter` agent per subsystem writes edge-case tests, runs them, and reports failures as real bugs. Results land in `skills/bug-hunt/output/BUG_REPORT.md`.

**Subsystems:** `scale-stat`, `coerce-transport`, `marks-rendering`, `composition-facet`, `figure-api`, `model-diagnostics`, `phase-11-interactive`, `layout`, `draw`, `projection`, `stats-transforms`

**Related agent:** `bug-hunter`

---

### `/gallery-audit` — Default-output comparison against canonical libraries

**File:** `skills/gallery-audit/SKILL.md`

Renders the same plot in ferrum and in each reference library (sklearn, seaborn, yellowbrick, scikit-plot) using **default settings only**, then dispatches `gallery-judge` agents in parallel to score each row against a fixed rubric. Produces a prioritized `REPORT.md` punchlist.

Covers 38 plot types across model-diagnostic (ROC, PR, confusion matrix, calibration, learning curve, residuals, feature importance, SHAP, etc.) and EDA (histogram, boxplot, regression scatter, correlation heatmap, pairplot, etc.) territory.

**Output:** `skills/gallery-audit/output/` — `REPORT.md`, per-row `verdict.md`, panel PNGs, lite-review verdicts under `_review_lite/`

**Related agents:** `gallery-judge` (judging), `gallery-fixer` (remediation)

---

### `/gallery-feedback` — Interactive walkthrough of audit results

**File:** `skills/gallery-feedback/SKILL.md`

Human-in-the-loop follow-up to `/gallery-audit`. Walks through every row one at a time, shows ferrum's panel alongside all comparator panels, and asks what should change. Compiles decisions into a structured remediation plan.

Use this when you want to decide *which* audit findings to adopt before running `gallery-fixer`.

---

### `/schwabish-improve` — Schwabish text-integration audit

**File:** `skills/schwabish/SKILL.md`

Applies the Schwabish "integrate text and graphics" principles to ferrum charts. Judges four T-categories: **T1** active/informative title, **T2** direct data labels, **T3** callouts, **T4** inline metrics.

Two modes:
- **Advisory** (`/schwabish-improve <target>`) — dispatches `schwabish-judge` per target, writes `schwabish_verdict.md` next to each chart. Read-only.
- **Gallery-autonomous** (`/schwabish-improve --from-audit`) — judges all gallery rows in parallel, then dispatches `schwabish-fixer` to apply objective findings to `gallery/plots/<row>/ferrum_panel.py`, runs the lite-review gate, and commits one per row.

**Related agents:** `schwabish-judge`, `schwabish-fixer`

---

### `/python-review` — Senior Python refactoring and API-design review

**File:** `skills/python-review/SKILL.md`

Multi-phase review of a Python package, module, or subsystem: orient → diagnose → propose → execute → review. Produces a six-section report with an architecture map, drift findings tagged S1–S5, a refactor roadmap, and a proposed first patch.

Trigger when something "feels off": utility-module sprawl, dict-shaped domain data, mode-flag creep, inconsistent return shapes, sibling API drift, overgrown classes, leaky internal imports.

**Difference from `python-review-lite`:** This is a full, human-invoked subsystem audit. The lite agent is a per-commit gate.

---

### `/rust-review` — Senior Rust refactoring and API-design review

**File:** `skills/rust-review/SKILL.md`

Same structure as `/python-review` but for Rust: orient → diagnose → propose → execute → review. Targets a crate, module, or subsystem. Looks for naming inconsistency, boolean/config explosion, leaky APIs, panic-prone library code, unnecessary genericity, parallel-API drift, and compatibility scar tissue.

**Difference from `rust-review-lite`:** This is a full, human-invoked subsystem audit. The lite agent is a per-commit gate.

---

### `/ferrum-docstrings` — Docstring conventions

**File:** `skills/ferrum-docstrings/SKILL.md`

Reference for adding or updating docstrings in ferrum. Covers the NumPy convention, PyO3 placement rules (class docstring owns `Parameters`, not `__init__`), and ferrum-specific example shapes. Invoke when adding a new public class, mark, transform, or encoding channel.

Full rationale: `docs/superpowers/specs/2026-05-11-docstrings-design.md`

---

### `/code-archaeology` — Unimplemented feature and silent-drop sweep

**File:** `skills/code-archaeology/SKILL.md`

Dispatches three parallel agents (Python, Rust, Tests+Docs) to sweep the entire codebase for unimplemented features, silently dropped parameters, dead code paths, skipped tests, and spec-vs-impl gaps. Produces a prioritized report saved to `docs/superpowers/followups/YYYY-MM-DD-code-archaeology.md`. The key non-obvious patterns: `_ => {}` match-arm fallthroughs in Rust dispatch tables, `warn_once`-gated silent drops in Python, and `#[allow(dead_code)]` blanket suppressions that hide unused helpers.

---

## Agents

Agents are autonomous subagents dispatched by skills or the orchestrator. They are not invoked directly by the user — the skill or parent session dispatches them.

### `bug-hunter`

**File:** `agents/bug-hunter.md`

Dispatched in parallel by `/bug-hunt`, one per subsystem. Receives a subsystem name, a mode (`Py+Rs` | `Py` | `Rs`), source paths, and existing test paths. Writes the appropriate test file(s), runs them, and reports failures as bugs. Never dispatched directly.

**Tools:** Read, Edit, Write, Bash, Glob, Grep

---

### `gallery-judge`

**File:** `agents/gallery-judge.md`

Dispatched in parallel by `/gallery-audit`, one per row. Reads the rubric, reads 2–4 panel PNGs for the row, applies the rubric category-by-category (A–G), and writes `verdict.md` with YAML frontmatter and prose. One subagent per row keeps the parent context clean. No `ANTHROPIC_API_KEY` required — runs inside the current Claude Code session.

**Tools:** All tools

---

### `gallery-fixer`

**File:** `agents/gallery-fixer.md`

Dispatched after a `/gallery-audit` run to close HIGH-severity findings autonomously. Reads `REPORT.md`, reads panel PNGs, locates the relevant ferrum source (`src/ferrum/figures.py`, `_diagnostics/`, Rust render core), implements the missing default, re-runs the affected rows, and dispatches `python-review-lite` on the staged diff before committing. Composite-mark and annotation defaults are implemented Python-side; Rust renderer changes are only made when unavoidable.

Trigger phrases: "fix the gallery findings", "work the punchlist", "close the ferrum/seaborn gaps"

**Tools:** All tools

---

### `python-review-lite`

**File:** `agents/python-review-lite.md`

Lightweight autonomous gate dispatched by the orchestrator **before every `git commit` that touches `*.py` source**. Reads `git diff --cached`, applies a trimmed diff-level idiom checklist, runs `ruff`, and returns one of three signals:

- **`clean`** — no S3+ findings, linters pass → orchestrator commits
- **`block`** — ≥1 S3 finding or linter failure → orchestrator un-stages, fixes, re-stages, re-dispatches
- **`escalate`** — ≥1 S4+ finding, or 3 consecutive block cycles on the same area → halt and surface to user

Verdicts land at `skills/gallery-audit/output/_review_lite/<ISO-timestamp>_python.md`.

**Tools:** Read, Grep, Glob, Bash — **never writes code**

---

### `rust-review-lite`

**File:** `agents/rust-review-lite.md`

Same protocol as `python-review-lite` but for Rust. Dispatched before every commit touching `*.rs` source. Runs `cargo clippy -D warnings` on the affected crate and applies a Rust-specific idiom checklist. Same three-signal return (`clean` / `block` / `escalate`).

Verdicts land at `skills/gallery-audit/output/_review_lite/<ISO-timestamp>_rust.md`.

**Tools:** Read, Grep, Glob, Bash — **never writes code**

---

### `schwabish-judge`

**File:** `agents/schwabish-judge.md`

Dispatched in parallel by `/schwabish-improve`, one per chart target. Reads the chart artifact (Python panel script, SVG, or directory), applies the four Schwabish T-categories from `skills/schwabish/judge_prompt.md`, and writes `schwabish_verdict.md` with per-finding `severity` and `objective` flags. Never edits chart code.

**Tools:** Read, Grep, Glob, Bash

---

### `schwabish-fixer`

**File:** `agents/schwabish-fixer.md`

Dispatched by `/schwabish-improve --from-audit` after all `schwabish-judge` verdicts are written. Reads each row's `schwabish_verdict.md`, filters to findings where `objective: true`, checks idempotence, and applies eligible actions via `Edit` to `gallery/plots/<row>/ferrum_panel.py`. Regenerates the row's panel via `audit.py generate --row <id>` after each fix. Restricted to gallery panel scripts — never edits `src/ferrum/`.

**Tools:** Read, Grep, Glob, Edit, Bash

---

## Review surfaces reference

### Surface comparison

| Surface | Type | Invoked by | Scope | Writes code? |
|---|---|---|---|---|
| `python-review` | skill | human (`/python-review`) | whole package or named subsystem | yes, with approval |
| `rust-review` | skill | human (`/rust-review`) | whole crate or named subsystem | yes, with approval |
| `python-review-lite` | agent | orchestrator (before any `*.py` commit) | only staged `*.py` diff | **never** |
| `rust-review-lite` | agent | orchestrator (before any `*.rs` commit) | only staged `*.rs` diff | **never** |

The heavyweight skills are interactive: they orient, diagnose, propose, and only execute with user approval. The lite agents are autonomous read-only gates — they never modify code.

### Severity rubric (shared across all four surfaces)

| Tag | Meaning |
|---|---|
| S1 | cosmetic inconsistency; low risk, low impact |
| S2 | readability / maintainability issue; moderate leverage |
| S3 | structural cohesion issue; high leverage — **blocks lite agents** |
| S4 | risky design flaw or bug-prone seam — **escalates lite agents** |
| S5 | critical correctness or API hazard — **escalates lite agents** |

### Audit trail

Lite-agent verdicts land at `skills/gallery-audit/output/_review_lite/<ISO-timestamp>_{python,rust}.md` regardless of trigger (the path is historical — lite started as a post-`gallery-fixer` gate). Each verdict carries YAML frontmatter (`status`, `cycle`, `n_findings` by severity, `linters`, `files_reviewed`) followed by per-finding prose. Gitignored alongside the rest of `output/`.

---

## How the pieces fit together

```
User types /gallery-audit
  └─ gallery-audit skill
       ├─ audit.py generate   (renders all panels)
       ├─ gallery-judge × N   (one per row, parallel)
       └─ audit.py report     (builds REPORT.md)

User types "fix the gallery findings"
  └─ gallery-fixer agent
       ├─ reads REPORT.md + panel PNGs
       ├─ edits ferrum source
       ├─ python-review-lite  (pre-commit gate)
       └─ commits clean rows

User types /bug-hunt
  └─ bug-hunt skill
       └─ bug-hunter × 11    (one per subsystem, parallel)

User types /schwabish-improve --from-audit
  └─ schwabish skill
       ├─ schwabish-judge × N  (one per gallery row, parallel)
       ├─ schwabish-fixer × N  (applies objective findings)
       ├─ python-review-lite   (pre-commit gate)
       └─ commits clean rows

Any commit touching *.py
  └─ python-review-lite (dispatched by orchestrator)

Any commit touching *.rs
  └─ rust-review-lite (dispatched by orchestrator)
```
