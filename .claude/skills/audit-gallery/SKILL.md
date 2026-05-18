---
name: audit-gallery
description: Compare ferrum's default plot output side-by-side against canonical Python implementations (sklearn, seaborn, yellowbrick, scikit-plot) and produce a prioritized punchlist of missing information, missing annotations, and visual-quality gaps. Use when the user says /audit-gallery, "audit our plots", "compare ferrum to seaborn/sklearn/yellowbrick", "what's missing from our default plots", or wants to know if ferrum's defaults look bad next to canonical defaults.
---

# Gallery Audit

A reproducible side-by-side audit: render the same plot in ferrum and in each canonical reference library with **default settings only**, rasterize everything at identical pixel dimensions, run an LLM-as-judge over each row with a fixed rubric, and produce a prioritized REPORT.md.

The output answers one question: **does ferrum's default plot look bad or lack information that a default plot from the canonical libraries would have?**

## Scope

12 canonical plots covering model-diagnostic (ROC, PR, confusion matrix, calibration, learning curve, residuals, feature importance) and EDA (histogram, boxplot, regression scatter, correlation heatmap, bar with error bars) territory. See `plots/` — each row is a directory with one panel script per library.

## Prerequisites

- `uv` on `$PATH` (used to launch isolated PEP 723 envs for each comparator panel).
- Project venv with sklearn installed (ferrum's `models` extra): `uv sync --extra models`.
- maturin-built ferrum: `unset CONDA_PREFIX && uv run --no-sync maturin develop --release` (per project CLAUDE.md).
- No `ANTHROPIC_API_KEY` required — judging runs inside this Claude Code session via the `gallery-judge` subagent (one per row).

## On-invocation: check for unwired rows first

Before running the pipeline, check `RESUME.md` and each row's `config.toml`. If any row has `ferrum_status = "READY"` (or `"PARTIAL"`) but `panels = []`, that row is unwired — the ferrum API exists but no panel scripts were written. Surface this to the user with a short list:

```
The following rows are READY in ferrum but unwired:
  - 05_learning_curve (if its ferrum_status flipped from BLOCKED)
  - <any others>

Wire them before running, or run the audit on the currently wired rows only?
```

If the user says wire them, follow the **Resume protocol** in `RESUME.md`: read the row's `TODO.md`, copy `plots/01_roc/<library>_panel.py` as a template, swap the dataset/model/library call, update the row's `config.toml` (`panels = [...]`, `ferrum_status` if needed), then continue to the generate stage.

For BLOCKED rows (the ferrum API doesn't exist yet), do **not** attempt to wire — surface the gap and skip.

## How to invoke

The pipeline has three stages. **You (Claude) drive all three when the user invokes `/audit-gallery`** — the parent script is mechanical, the judging is delegated to subagents in this session.

### Stage 1 — Generate (mechanical, script)

Run the orchestrator's `generate` subcommand to render all panel PNGs:

```bash
unset CONDA_PREFIX && uv run --no-project --script .claude/skills/audit-gallery/audit.py generate
# Or for a row filter:
unset CONDA_PREFIX && uv run --no-project --script .claude/skills/audit-gallery/audit.py generate --rows 1,3
```

This writes `output/<row>/{ferrum,sklearn,yellowbrick,seaborn,skp}.png` for every wired row. Rows with empty `panels = []` in their config are skipped (those are still on the TODO list — see `RESUME.md`).

### Stage 2 — Judge (in this Claude Code session, parallel subagents)

For each row with at least one panel image in `output/<row>/`, **dispatch a `gallery-judge` subagent** (`subagent_type=gallery-judge`). Send all subagent invocations in a single message so they run in parallel:

```
For each row_id in <wired row ids>:
  Agent(
    description: "Judge row <row_id>",
    subagent_type: "gallery-judge",
    prompt: "Judge row `<row_id>`. Read .claude/output/audit-gallery/<row_id>/, read the rubric and judge_prompt, write verdict.md, return one-line summary."
  )
```

Each subagent reads its row's panel PNGs (via the Read tool, which surfaces images visually), applies `rubric.md`, and writes `output/<row_id>/verdict.md` with YAML frontmatter + prose. The parent (you) gets back one-line summaries.

**Why subagents:** keeps the main session context clean — 30+ panel images would otherwise pile up in the parent. The verdicts are written to disk, so the next stage doesn't need any of the image bytes in context.

### Stage 3 — Report (mechanical, script)

After all judge subagents return, run the report aggregator:

```bash
unset CONDA_PREFIX && uv run --no-project --script .claude/skills/audit-gallery/audit.py report
```

This reads every `output/<row>/verdict.md`, parses the YAML frontmatter, ranks by severity, and writes `output/REPORT.md` (mirrored at the repo-root `gallery/` symlink).

### After the report

Read `output/REPORT.md` and surface the top HIGH-severity findings to the user. Then ask whether to dispatch the **`gallery-fixer`** subagent (`subagent_type=gallery-fixer`) to work through the punchlist autonomously.

### Optional: unattended mode

A legacy `audit.py judge` subcommand exists for unattended runs (cron, CI) — it calls the Anthropic API directly with `ANTHROPIC_API_KEY` and a prompt-cached rubric. Do **not** use this path when invoked from a Claude Code session; use the in-session subagent flow above. The unattended path is documented in `audit.py --help` for completeness.

## Output location

`.claude/output/audit-gallery/`
  - `REPORT.md` — aggregated prioritized punchlist (the main artifact)
  - `<row>/ferrum.png`, `sklearn.png`, `yellowbrick.png`, `skp.png` — rasterized panels
  - `<row>/verdict.md` — YAML-frontmatter verdict for that row

The repo-root `gallery` symlink points at this directory for convenience.

## How the pipeline works

For each row:

1. **Generate panels (deterministic, script)**
   - `ferrum_panel.py` runs in the project venv (needs the maturin-built `ferrum._core`) and writes an SVG.
   - `sklearn_panel.py`, `yellowbrick_panel.py`, `seaborn_panel.py`, `skp_panel.py` each run as isolated PEP 723 scripts via `uv run --no-project --script` — each gets its own pinned deps, never polluting ferrum's venv.
   - All panels render at the same pixel dimensions (declared per-row in `config.toml`), with `MPLBACKEND=Agg`, fixed seed, fixed font, autolayout disabled — passed as env vars from the orchestrator so panels can't drift.
   - The orchestrator rasterizes ferrum's SVG via `resvg-py` at the same pixel size.

2. **Judge (in-session subagents, one per row)**
   - The parent (you) dispatches a `gallery-judge` subagent per row in parallel.
   - Each subagent reads the row's panel PNGs via the Read tool, applies the cached `rubric.md` + `judge_prompt.md`, and writes `verdict.md` with YAML frontmatter + prose.
   - Subagent context is isolated — the parent doesn't pile up 30+ image bytes — and the verdict file is the durable handoff.

3. **Aggregate (script)**
   - `audit.py report` reads every row's `verdict.md`, parses YAML frontmatter, ranks by severity, and writes `output/REPORT.md`.

## Important constraints

- **Never add matplotlib, seaborn, sklearn, yellowbrick, or scikit-plot to ferrum's `pyproject.toml`.** Ferrum's hard constraint forbids matplotlib in any form (deps, dev deps, optional extras). Comparator libraries run in isolated PEP 723 envs, never touching `.venv`. If a comparator panel needs a new dep, add it to that panel's inline `# /// script` block — never to the project.
- **Defaults only.** Each panel script must call its library with default settings; no tweaking to make ferrum look better or worse. The whole point is to compare *what ships out of the box*.
- **Determinism is load-bearing.** If two runs produce different pixels for the same panel, the judge will hallucinate gaps. Always pass the seed/font/figsize/DPI env vars from the orchestrator; don't let individual panel scripts override.
- **No `ANTHROPIC_API_KEY` needed by default.** Judging runs in this Claude Code session via subagents. The legacy `audit.py judge` subcommand (which uses the SDK) is retained for unattended/cron runs only.

## When ferrum panels fail

If the ferrum panel for a row crashes (e.g. a feature isn't implemented yet — `roc_chart(..., annotate_auc=True)` is reserved for Phase 10h), the row's verdict.md should record `ferrum_status: NOT_IMPLEMENTED` and the report should flag it as a separate "blocked" category — not as a visual gap.

## Re-running on a subset

After the `gallery-fixer` agent (or a human) implements a fix, re-run just the affected row: `audit.py all --rows 3`. The report aggregator merges new verdicts with existing ones on disk, so partial re-runs work without invalidating other rows.
