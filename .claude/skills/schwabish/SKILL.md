---
name: schwabish
description: Apply Schwabish "integrate text and graphics" principles to a ferrum chart or the entire gallery. Use when the user says /schwabish-improve, "improve text integration", "add direct labels", "make titles active", or wants a Schwabish-style audit of plots beyond peer-parity.
---

# Schwabish — Text-Integration Audit

A two-mode skill that judges ferrum charts against the Schwabish "integrate text and graphics" rubric — the four T-categories: **T1 active title**, **T2 direct labels**, **T3 callouts**, **T4 inline metrics**. The full principle reference lives at `design-docs/superpowers/specs/2026-05-11-schwabish-principles.md` and is embedded as the cached prefix to the judge prompt.

## Modes

### Advisory mode (default)

```
/schwabish-improve <target> [--context "<string>"] [--out <path>]
```

`<target>` is one of:
- a path to a Python file that builds a ferrum chart (e.g. `gallery/plots/01_roc/ferrum_panel.py`)
- a path to an SVG file
- a path to a directory (recursive scan of all chart artifacts found)

The skill dispatches a `schwabish-judge` subagent per discovered target. Each judge reads the chart artifact, applies the rubric in `judge_prompt.md`, and writes a `schwabish_verdict.md` next to the target (or to `--out <path>`). The verdict contains YAML frontmatter listing per-T-category findings (with `severity` and `objective` flags) plus a prose section with concrete suggestions.

Advisory mode is **read-only**. The judge agent never edits chart code.

### Gallery-autonomous mode

```
/schwabish-improve --from-audit
```

No target argument. The skill walks `gallery/plots/`, dispatches `schwabish-judge` per row in parallel, filters verdicts to findings where `objective: true`, and then dispatches `schwabish-fixer` to apply those findings to `gallery/plots/<row>/ferrum_panel.py` via the `Edit` tool. After fixes land and panels regenerate via `audit.py generate --row <id>`, the orchestrator runs `python-review-lite` on the staged diff (clean → commit one-per-row; block → un-stage and report; escalate → halt). Subjective findings stay in the per-row verdict for the user to review.

A final `.claude/output/gallery-audit/SCHWABISH_REPORT.md` summarizes applied changes and surfaces the subjective findings the user should action.

## Target detection (advisory mode)

When `/schwabish-improve <target>` is invoked, classify `<target>`:

1. **Single Python file** (ends with `.py`): treat as a panel script; dispatch one `schwabish-judge` with `target=<path>` and `out_path=<path>.schwabish_verdict.md`.
2. **Single SVG file** (ends with `.svg`): same as above; judge reads only the SVG.
3. **Directory**: walk for `*.py` and `*.svg` files. Skip anything matching `__pycache__/`, `.venv/`, `node_modules/`. Dispatch one judge per discovered chart artifact, in parallel.
4. **Otherwise**: error out with "target must be a .py file, .svg file, or directory".

## Dispatch (advisory mode)

For each classified target, prompt the `schwabish-judge` agent:

> Read `<target>`. Apply the rubric in `.claude/skills/schwabish/judge_prompt.md` (cached prefix). Write the verdict to `<out_path>`. Context: `<--context value or empty>`.

Use the `Agent` tool with `subagent_type=schwabish-judge`, in parallel when multiple targets exist.

## Aggregation (advisory mode)

After all judges return:
- If `--out` was a single file, the single verdict is already written.
- If targets were a directory, write `<directory>/SCHWABISH_VERDICTS_INDEX.md` listing all per-target verdict paths with severities at-a-glance.

## Gallery-autonomous flow (`--from-audit`)

1. **Discover rows.** Walk `gallery/plots/`. For each row directory:
   - Read `config.toml`. Skip if `ferrum_status` is `BLOCKED` or `NOT_WIRED`.
   - Verify `gallery/plots/<row>/ferrum_panel.py` exists.

2. **Judge in parallel.** Dispatch one `schwabish-judge` per discovered row:
   - `target = gallery/plots/<row>/ferrum_panel.py`
   - `out_path = .claude/output/gallery-audit/<row>/schwabish_verdict.md`
   - `context = ""` (row config files do not carry semantic context)

3. **Filter to objective findings.** For each verdict file, parse the YAML; collect rows where ≥1 finding has `objective: true`.

4. **Apply via fixer (parallel).** For each row from step 3, dispatch `schwabish-fixer`:
   - `row = <id>`
   - `verdict_path = .claude/output/gallery-audit/<row>/schwabish_verdict.md`
   - `eligibility_path = .claude/skills/schwabish/apply_eligibility.md`

5. **Stage + lite-review.** After all fixers return:
   ```bash
   git add gallery/plots/
   ```
   Dispatch `python-review-lite` with the staged diff. Three outcomes:
   - **clean** → proceed to step 6.
   - **block** → un-stage (`git reset HEAD gallery/plots/`), report the review verdict back to the user, halt.
   - **escalate** → un-stage, report, halt.

6. **Commit per row.** For each row touched, one commit:
   ```bash
   git add gallery/plots/<row>/ .claude/output/gallery-audit/<row>/
   git commit -m "feat(gallery): schwabish improvements on row <id>"
   ```

7. **Aggregate.** Write `.claude/output/gallery-audit/SCHWABISH_REPORT.md` summarizing applied changes per row and the subjective findings the user should review. Commit separately:
   ```bash
   git add .claude/output/gallery-audit/SCHWABISH_REPORT.md
   git commit -m "docs(gallery): schwabish report — <ISO>"
   ```

## Cycle tracking

If `python-review-lite` returns `block`, the orchestrator dispatches the same fixer for that row up to 3 times. On the 3rd consecutive block for the same row, escalate to the user and halt.

## Reference docs

- `design-docs/superpowers/specs/2026-05-11-schwabish-principles.md` — canonical reference, embedded as cached prefix in `judge_prompt.md`.
- `design-docs/superpowers/specs/2026-05-11-schwabish-design.md` — full design spec (the *how*).
- `.claude/skills/schwabish/apply_eligibility.md` — objective-only findings the autonomous fixer applies.
