# Docs Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Generate and embed PNG screenshots for every new documentation section that has code examples but no rendered visual output.

## 2. Spec references

- Commit `5b260ec` (docs/gap-fixes branch) — the 25 new sections added
- `scripts/snapshot-goldens.py`, `tests/_snapshots.py` — existing PNG generation infrastructure
- `CLAUDE.md` "Goldens are not blessed until visually inspected" — all PNGs must be Read and verified

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `docs/site/guide/img/recipes_11.png` | CoordFlip horizontal bars |
| Create | `docs/site/guide/img/recipes_12.png` | Annotations (reference lines + text) |
| Create | `docs/site/guide/img/recipes_13.png` | Chart sizing |
| Create | `docs/site/guide/img/recipes_14.png` | Custom category order |
| Create | `docs/site/guide/img/recipes_15.png` | Time-series line chart |
| Create | `docs/site/guide/img/marks-encodings_08.png` | Dodge (grouped bars) |
| Create | `docs/site/guide/img/marks-encodings_09.png` | Stack (stacked bars) |
| Create | `docs/site/guide/img/marks-encodings_10.png` | Axis customization (log scale) |
| Create | `docs/site/guide/img/marks-encodings_11.png` | Axis limits (domain) |
| Create | `docs/site/guide/img/marks-encodings_12.png` | Legend suppression |
| Create | `docs/site/guide/img/composition_08.png` | Shared scales |
| Create | `docs/site/guide/img/figure-helpers_11.png` | regplot |
| Create | `docs/site/guide/img/model-diagnostics_07.png` | Multi-model compare ROC |
| Modify | `docs/site/guide/recipes.md` | Add `![...]` image references after each new recipe |
| Modify | `docs/site/guide/marks-encodings.md` | Add `![...]` image references after new sections |
| Modify | `docs/site/guide/composition.md` | Add `![...]` after shared scales example |
| Modify | `docs/site/guide/figure-helpers.md` | Add `![...]` after regplot example |
| Modify | `docs/site/guide/model-diagnostics.md` | Add `![...]` after compare example |

## 4. Constraints

- Generate PNGs by running the code examples from the docs against the current API via `uv run --no-sync python -c "..."`
- Use `chart.show_png()` to get bytes, write to the target path
- Visually inspect every generated PNG with `Read` before committing
- Use Paper Ink theme (default) unless the example explicitly uses another theme
- Image naming follows existing convention: `<page>_<NN>.png`
- Do not modify any source code — docs and images only

## 5. Tasks

### Task 1: recipes visuals (5 images)
For each new recipe section (CoordFlip, Annotations, Chart sizing, Category order, Time-series), extract the code example from `recipes.md`, run it to generate a PNG, save to `docs/site/guide/img/recipes_<NN>.png`, visually verify, then add `![Description](img/recipes_<NN>.png)` after the code block.
- [ ] recipes_11.png — CoordFlip horizontal bars
- [ ] recipes_12.png — Annotations (mark_rule + mark_text layered)
- [ ] recipes_13.png — Chart sizing (.properties)
- [ ] recipes_14.png — Custom category order
- [ ] recipes_15.png — Time-series line chart
- [ ] Add image references to `recipes.md`
- [ ] Verify: Read each PNG

### Task 2: marks-encodings visuals (5 images)
For each new section (Dodge, Stack, log scale, domain limits, legend suppression), run the code example, generate PNG, verify, embed.
- [ ] marks-encodings_08.png — Dodge grouped bars
- [ ] marks-encodings_09.png — Stack stacked bars
- [ ] marks-encodings_10.png — Log scale axis
- [ ] marks-encodings_11.png — Axis limits via domain
- [ ] marks-encodings_12.png — Legend suppressed
- [ ] Add image references to `marks-encodings.md`
- [ ] Verify: Read each PNG

### Task 3: composition, figure-helpers, model-diagnostics visuals (3 images)
- [ ] composition_08.png — Shared scales (two panels with linked x-axis)
- [ ] figure-helpers_11.png — regplot output
- [ ] model-diagnostics_07.png — Multi-model compare ROC overlay
- [ ] Add image references to each respective guide page
- [ ] Verify: Read each PNG

## 6. Acceptance checks

- 13 new PNGs exist under `docs/site/guide/img/`
- Every new PNG is referenced by a `![...]` tag in its corresponding `.md` file
- Every PNG visually verified via `Read` — charts render correctly with visible marks, axes, labels
- No source code files modified
- `git diff --stat` shows only `docs/site/` changes

## 7. Open questions

- None — all code examples are already verified to run from the previous plan execution.
