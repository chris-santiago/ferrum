---
name: scene-pipeline-audit
description: 4-agent parallel audit of the ferrum rendering pipeline. Dispatches one scene-pipeline-auditor per stage (spec-to-scene, scene-to-svg, scene-to-wasm, composition-merge). Each agent traces data from input to output, verifies no fields are lost, and reports GOOD/WARN/BUG findings. Use when the user says /scene-pipeline-audit, "audit the pipeline", "trace the rendering", "are encoding channels wired", or after adding new marks, channels, or transforms.
---

# Scene Pipeline Audit

Parallel audit of the 4 stages in the ferrum rendering pipeline. One `scene-pipeline-auditor` agent per stage traces data through every transformation and flags silent data loss.

## Stage table

| Stage | What it audits | Key files |
|---|---|---|
| `spec-to-scene` | ChartSpec → SceneGraph: encoding channels, mark types, tooltips, data_indices | `spec.rs`, `scene_build.rs`, `types.rs` |
| `scene-to-svg` | SceneGraph → SVG: node rendering, style attributes, path commands, clipping | `draw.rs`, `types.rs` |
| `scene-to-wasm` | SceneGraph → GPU: instance loading, color conversion, text elements, packed data | `scene_load.rs`, `tessellate.rs`, `lib.rs` |
| `composition-merge` | Child scenes → merged scene: offsets, panel IDs, selections, packed data | `composition.py`, `_interactive.py` |

## Procedure

### Step 1 — Parse args

If the user provided a stage name (e.g., `/scene-pipeline-audit spec-to-scene`), run only that stage. Otherwise run all 4.

### Step 2 — Dispatch agents

Dispatch one `scene-pipeline-auditor` agent per stage **in a single message** so they run in parallel. Use `model: "opus"` — these agents need deep code tracing.

**All 4 stages (default):**

```
Agent(subagent_type="scene-pipeline-auditor", model="opus",
      description="Audit spec-to-scene",
      prompt="Audit stage: spec-to-scene")

Agent(subagent_type="scene-pipeline-auditor", model="opus",
      description="Audit scene-to-svg",
      prompt="Audit stage: scene-to-svg")

Agent(subagent_type="scene-pipeline-auditor", model="opus",
      description="Audit scene-to-wasm",
      prompt="Audit stage: scene-to-wasm")

Agent(subagent_type="scene-pipeline-auditor", model="opus",
      description="Audit composition-merge",
      prompt="Audit stage: composition-merge")
```

**Single stage (when arg provided):**

Dispatch only the matching agent.

### Step 3 — Consolidate and save

After all agents return, consolidate findings into a single report grouped by severity.

**Write the report to `.claude/output/scene-pipeline-audit/YYYY-MM-DD-audit.md`** (use today's date). The report must include:

```markdown
# Scene Pipeline Audit — YYYY-MM-DD

**Totals:** X BUGs, Y WARNs, Z GOODs across N stages.

## BUGs (must fix)
| # | Stage | Finding | File:Line | Impact |
|---|-------|---------|-----------|--------|

## WARNs (fix or acknowledge)
| # | Stage | Finding | File:Line | Impact |
|---|-------|---------|-----------|--------|

## GOODs (verified correct)
- [stage] description (one line per GOOD)
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

- After adding new mark types, encoding channels, or transforms
- After modifying `scene_build.rs`, `draw.rs`, or `scene_load.rs`
- After changing the composition merge pipeline
- Before marking rendering-related phases as done
- When the user reports "this encoding channel doesn't seem to do anything" or "marks are missing"
