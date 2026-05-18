---
name: audit-pyo3
description: 3-agent parallel audit of the PyO3 binding boundary in ferrum. Dispatches one auditor-pyo3-binding per binding group (chart-spec, transforms, scene-types). Each agent reads Rust definitions and Python call sites, verifies types/args/kwargs/returns across the FFI boundary, and reports GOOD/WARN/BUG findings. Use when the user says /audit-pyo3, "audit the bindings", "check the PyO3 boundary", "are kwargs being dropped", or after adding/modifying #[pyfunction]/#[pyclass] definitions.
---

# PyO3 Binding Audit

Parallel audit of the 3 PyO3 binding groups in ferrum. One `auditor-pyo3-binding` agent per group reads Rust bindings and Python callers, verifies every arg/kwarg/return across the FFI boundary, and reports findings.

## Binding group table

| Group | What it audits | Key Rust files | Key Python files |
|---|---|---|---|
| `chart-spec` | ChartSpec fields, render_* functions, selections/conditionals/tooltip kwargs | `spec.rs`, `binding.rs` | `chart.py`, `_render.py`, `display.py` |
| `transforms` | Transform #[pyclass] construction, serialization, named transforms, kwargs | `transform/*.rs`, `spec.rs` | `transforms.py`, `chart.py` |
| `scene-types` | Selection/Conditional serde shapes, EventExpr mapping, scene JSON structure, FieldValue enum | `types.rs`, `selection.rs` | `selection.py`, `composition.py` |

## Procedure

### Step 1 — Parse args

If the user provided a group name (e.g., `/audit-pyo3 chart-spec`), run only that group. Otherwise run all 3.

### Step 2 — Dispatch agents

Dispatch one `auditor-pyo3-binding` agent per group **in a single message** so they run in parallel. Use `model: "opus"` — these agents need deep cross-language tracing.

**All 3 groups (default):**

```
Agent(subagent_type="auditor-pyo3-binding", model="opus",
      description="Audit chart-spec bindings",
      prompt="Audit group: chart-spec")

Agent(subagent_type="auditor-pyo3-binding", model="opus",
      description="Audit transform bindings",
      prompt="Audit group: transforms")

Agent(subagent_type="auditor-pyo3-binding", model="opus",
      description="Audit scene-type bindings",
      prompt="Audit group: scene-types")
```

**Single group (when arg provided):**

Dispatch only the matching agent.

### Step 3 — Consolidate and save

After all agents return, consolidate findings into a single report grouped by severity.

**Write the report to `.claude/output/audit-pyo3/YYYY-MM-DD-audit.md`** (use today's date). The report must include:

```markdown
# PyO3 Binding Audit — YYYY-MM-DD

**Totals:** X BUGs, Y WARNs, Z GOODs across N groups.

## BUGs (must fix)
| # | Group | Finding | Rust file:line | Python file:line | Impact |
|---|-------|---------|----------------|------------------|--------|

## WARNs (fix or acknowledge)
| # | Group | Finding | File:Line | Impact |
|---|-------|---------|-----------|--------|

## GOODs (verified correct)
- [group] description (one line per GOOD)
```

The file is gitignored (`.claude/output/` is in `.gitignore`).

### Step 4 — Recommend

If BUGs were found:
- List them with recommended fix direction
- Ask the user: "Want me to fix these with TDD?"

If only WARNs:
- Categorize as "fix now" vs "defer" with rationale

If all GOOD:
- Report clean audit

## When to run

- After adding or modifying `#[pyfunction]`, `#[pyclass]`, or `#[pymethods]` in `ferrum-core`
- After changing `ChartSpec` fields, `to_spec()`, or the theme dict contract
- Before marking any phase as done that added Rust binding surface
- When the user reports "my kwarg isn't working" or "this parameter seems to be ignored"
