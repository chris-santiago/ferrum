---
name: audit-interactive
description: 4-agent parallel wiring audit of the interactive HTML export pipeline. Dispatches one auditor-interactive agent per integration seam (JS-WASM, Python-Rust data flow, Rust state machine, HTML assembly). Each agent traces actual code paths and reports GOOD/WARN/BUG findings. Use when the user says /audit-interactive, "audit the interactive pipeline", "check the wiring", "interactive export audit", or after landing changes to the interactive subsystem.
---

# Interactive Wiring Audit

Parallel audit of the 4 integration seams in the interactive HTML export pipeline. One `auditor-interactive` agent per seam reads the actual source, traces every connection point, and reports findings.

## Seam table

| Seam | What it audits | Key files |
|---|---|---|
| `js-wasm` | JS↔WASM method signatures, D3 routing, standalone adapter, strip correctness, _render contract | `ferrum-anywidget.js`, `d3-interactions.js`, `lib.rs`, `_html.py` |
| `python-rust-data` | Chart.interactive() → scene JSON + packed data flow, composition merging, selection injection, save pipeline | `chart.py`, `_interactive.py`, `composition.py`, `display.py`, `_html.py` |
| `rust-state-machine` | SelectionState transitions, handle_click/drag, toggle_points, conditional resolution, JSON serialization | `selection_state.rs`, `conditional.rs`, `hit_test.rs` |
| `html-assembly` | HTML structure, WASM init, JSON escaping, interaction config, background CSS, font embedding, regex robustness | `_html.py`, `ferrum-interactive.css`, `ferrum-anywidget.js`, `d3-interactions.js` |

## Procedure

### Step 1 — Parse args

If the user provided a seam name (e.g., `/audit-interactive js-wasm`), run only that seam. Otherwise run all 4.

### Step 2 — Dispatch agents

Dispatch one `auditor-interactive` agent per seam **in a single message** so they run in parallel. Use `model: "opus"` — these agents need deep code tracing.

**All 4 seams (default):**

```
Agent(subagent_type="auditor-interactive", model="opus",
      description="Audit JS-WASM wiring",
      prompt="Audit seam: js-wasm")

Agent(subagent_type="auditor-interactive", model="opus",
      description="Audit Python-Rust data flow",
      prompt="Audit seam: python-rust-data")

Agent(subagent_type="auditor-interactive", model="opus",
      description="Audit Rust state machine",
      prompt="Audit seam: rust-state-machine")

Agent(subagent_type="auditor-interactive", model="opus",
      description="Audit HTML assembly",
      prompt="Audit seam: html-assembly")
```

**Single seam (when arg provided):**

Dispatch only the matching agent.

### Step 3 — Consolidate and save

After all agents return, consolidate findings into a single report grouped by severity.

**Write the report to `.claude/output/audit-interactive/YYYY-MM-DD-audit.md`** (use today's date). The report must include:

```markdown
# Interactive Wiring Audit — YYYY-MM-DD

**Totals:** X BUGs, Y WARNs, Z GOODs across N seams.

## BUGs (must fix)
| # | Seam | Finding | File:Line | Impact |
|---|------|---------|-----------|--------|

## WARNs (fix or acknowledge)
| # | Seam | Finding | File:Line | Impact |
|---|------|---------|-----------|--------|

## GOODs (verified correct)
- [seam] description (one line per GOOD)
```

Count totals: X BUGs, Y WARNs, Z GOODs. The file is gitignored (`.claude/output/` is in `.gitignore`).

### Step 4 — Recommend

If BUGs were found:
- List them with recommended fix direction
- Ask the user: "Want me to fix these with TDD?"

If only WARNs:
- Categorize as "fix now" vs "defer" with rationale
- Ask the user which to act on

If all GOOD:
- Report clean audit and move on

## When to run

- After landing changes to the interactive subsystem (`_interactive.py`, `_html.py`, `ferrum-anywidget.js`, `d3-interactions.js`, `lib.rs`, `selection_state.rs`, `conditional.rs`, `composition.py` interactive methods)
- Before marking interactive-related phases as done
- When the user says "audit the interactive pipeline" or "check the wiring"
- After a `/bug-hunt phase-11-interactive` that found issues — this audit covers the integration seams that unit tests miss

## History

First run: 2026-05-18. Found 5 real bugs (B1-B5), 1 theoretical (B6), 8 warnings (W1-W8). All bugs fixed. W1/W3/W6/W8 fixed. W2/W4/W5 deferred to CLAUDE.md. Audit prompts saved to `design-docs/superpowers/audits/2026-05-18-interactive-wiring-audit-prompts.md`.
