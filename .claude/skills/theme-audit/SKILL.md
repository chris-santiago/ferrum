---
name: theme-audit
description: 4-agent parallel audit of the ferrum theme pipeline. Dispatches one theme-wiring-auditor per segment (python-to-rust, rust-layout, rust-render, cascade). Each agent traces theme keys from Python declaration through Rust consumption and verifies nothing is silently dropped. Use when the user says /theme-audit, "audit the theme", "which theme keys work", "are theme settings dropped", or after adding/modifying theme fields.
---

# Theme Wiring Audit

Parallel audit of the 4 segments in the ferrum theme pipeline. One `theme-wiring-auditor` agent per segment traces every theme key and flags silently dropped settings.

## Segment table

| Segment | What it audits | Key files |
|---|---|---|
| `python-to-rust` | Theme class fields → to_dict → PyO3 boundary → Rust deserialization | `themes.py`, `chart.py`, `_render.py`, `theme.rs` |
| `rust-layout` | Theme keys consumed by layout: margins, padding, font sizes, legend dims | `layout/*.rs`, `theme.rs` |
| `rust-render` | Theme keys consumed by render: colors, fonts, grid, axis, palette, background | `render/*.rs`, `scene_build.rs`, `draw.rs`, `theme.rs` |
| `cascade` | Override hierarchy: per-chart .theme() → set_default_theme() → built-in | `themes.py`, `chart.py`, `_render.py` |

## Procedure

### Step 1 — Parse args

If the user provided a segment name (e.g., `/theme-audit cascade`), run only that segment. Otherwise run all 4.

### Step 2 — Dispatch agents

Dispatch one `theme-wiring-auditor` agent per segment **in a single message** so they run in parallel. Use `model: "opus"` — these agents need deep code tracing.

**All 4 segments (default):**

```
Agent(subagent_type="theme-wiring-auditor", model="opus",
      description="Audit python-to-rust theme",
      prompt="Audit segment: python-to-rust")

Agent(subagent_type="theme-wiring-auditor", model="opus",
      description="Audit rust-layout theme",
      prompt="Audit segment: rust-layout")

Agent(subagent_type="theme-wiring-auditor", model="opus",
      description="Audit rust-render theme",
      prompt="Audit segment: rust-render")

Agent(subagent_type="theme-wiring-auditor", model="opus",
      description="Audit theme cascade",
      prompt="Audit segment: cascade")
```

**Single segment (when arg provided):**

Dispatch only the matching agent.

### Step 3 — Consolidate and save

After all agents return, consolidate findings into a single report grouped by severity.

**Write the report to `.claude/output/theme-audit/YYYY-MM-DD-audit.md`** (use today's date). The report must include:

```markdown
# Theme Wiring Audit — YYYY-MM-DD

**Totals:** X BUGs, Y WARNs, Z GOODs across N segments.

## Key inventory

| Theme key | Python accepts? | Reaches Rust? | Affects output? | Verdict |
|-----------|----------------|---------------|-----------------|---------|

## BUGs (must fix)
| # | Segment | Finding | File:Line | Impact |
|---|---------|---------|-----------|--------|

## WARNs (fix or acknowledge)
| # | Segment | Finding | File:Line | Impact |
|---|---------|---------|-----------|--------|

## GOODs (verified correct)
- [segment] description (one line per GOOD)
```

The key inventory table is unique to the theme audit — it provides a complete map of which theme keys work end-to-end. The file is gitignored (`.claude/output/` is in `.gitignore`).

### Step 4 — Recommend

If BUGs were found (keys silently dropped):
- List them with recommended fix direction
- Ask the user: "Want me to wire these theme keys?"

If only WARNs:
- Categorize as "fix now" vs "defer" with rationale

If all GOOD:
- Report clean audit — all theme keys flow end-to-end

## When to run

- After adding or modifying theme fields in `themes.py` or `theme.rs`
- After the themes overhaul phases (T1-T4) land
- Before marking theme-related phases as done
- When the user reports "this theme setting doesn't seem to do anything"
- Proactively after rebasing a feature branch that touches `theme.rs` or `themes.py`
