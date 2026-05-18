# Audit/Auditor Prefix Rename Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Rename all audit skills and auditor agents from suffix pattern (`*-audit`, `*-auditor`) to prefix pattern (`audit-*`, `auditor-*`).

## 2. Spec references

None — this is a mechanical rename with no behavioral changes.

## 3. Rename mapping

| Old name | New name |
|----------|----------|
| **Skills** | |
| `pyo3-audit` | `audit-pyo3` |
| `gallery-audit` | `audit-gallery` |
| `scene-pipeline-audit` | `audit-scene-pipeline` |
| `theme-audit` | `audit-theme` |
| `interactive-audit` | `audit-interactive` |
| `docs-audit` | `audit-docs` |
| **Agents** | |
| `interactive-auditor` | `auditor-interactive` |
| `pyo3-binding-auditor` | `auditor-pyo3-binding` |
| `scene-pipeline-auditor` | `auditor-scene-pipeline` |
| `theme-wiring-auditor` | `auditor-theme-wiring` |
| **Output dirs** | |
| `.claude/output/pyo3-audit` | `.claude/output/audit-pyo3` |
| `.claude/output/gallery-audit` | `.claude/output/audit-gallery` |
| `.claude/output/scene-pipeline-audit` | `.claude/output/audit-scene-pipeline` |
| `.claude/output/theme-audit` | `.claude/output/audit-theme` |
| `.claude/output/interactive-audit` | `.claude/output/audit-interactive` |

## 4. Constraints

- **Functional names only** — rename the `name:` frontmatter field in each skill/agent file; do NOT change `description:` text (those use natural-language phrases like "audit the theme" which are trigger phrases, not identifiers)
- **`subagent_type=` strings must match the new agent `name:` exactly** — these are the dispatch keys
- **Output dirs are gitignored** — rename them but don't expect git to track them
- **Design docs are historical** — update references in design docs so future readers can find the right skill/agent, but don't change meaning
- **Gallery symlink** — `gallery/` at repo root symlinks to `.claude/output/gallery-audit/`; must update to `.claude/output/audit-gallery/`
- **No behavioral changes** — only names and paths change

## 5. Tasks

### Task 1: Rename skill directories
- [ ] `mv .claude/skills/pyo3-audit .claude/skills/audit-pyo3`
- [ ] `mv .claude/skills/gallery-audit .claude/skills/audit-gallery`
- [ ] `mv .claude/skills/scene-pipeline-audit .claude/skills/audit-scene-pipeline`
- [ ] `mv .claude/skills/theme-audit .claude/skills/audit-theme`
- [ ] `mv .claude/skills/interactive-audit .claude/skills/audit-interactive`
- [ ] `mv .claude/skills/docs-audit .claude/skills/audit-docs`

### Task 2: Rename agent files
- [ ] `mv .claude/agents/interactive-auditor.md .claude/agents/auditor-interactive.md`
- [ ] `mv .claude/agents/pyo3-binding-auditor.md .claude/agents/auditor-pyo3-binding.md`
- [ ] `mv .claude/agents/scene-pipeline-auditor.md .claude/agents/auditor-scene-pipeline.md`
- [ ] `mv .claude/agents/theme-wiring-auditor.md .claude/agents/auditor-theme-wiring.md`

### Task 3: Update frontmatter `name:` fields
- [ ] Each of the 6 skill SKILL.md files: update `name:` to new prefix form
- [ ] Each of the 4 agent .md files: update `name:` to new prefix form

### Task 4: Update `subagent_type=` dispatch strings in skill files
- [ ] `.claude/skills/audit-pyo3/SKILL.md` — 3 occurrences of `subagent_type="pyo3-binding-auditor"` → `"auditor-pyo3-binding"`
- [ ] `.claude/skills/audit-scene-pipeline/SKILL.md` — 4 occurrences of `subagent_type="scene-pipeline-auditor"` → `"auditor-scene-pipeline"`
- [ ] `.claude/skills/audit-theme/SKILL.md` — 4 occurrences of `subagent_type="theme-wiring-auditor"` → `"auditor-theme-wiring"`
- [ ] `.claude/skills/audit-interactive/SKILL.md` — 4 occurrences of `subagent_type="interactive-auditor"` → `"auditor-interactive"`

### Task 5: Update output directory paths in skill files
- [ ] Grep all 6 skill SKILL.md files for old output dir names (e.g. `.claude/output/pyo3-audit`) and replace with new names

### Task 6: Update `CLAUDE.md`
- [ ] Update the "Where things live" table — all skill and agent path entries
- [ ] Update the "Gallery audit" section — skill path reference

### Task 7: Update `.claude/README.md`
- [ ] Update all skill/agent name references, path references, and dispatch tables (50+ occurrences)

### Task 8: Update cross-referencing agent/skill files
- [ ] `gallery-judge.md`, `gallery-fixer.md`, `schwabish-fixer.md` — references to audit skill/output paths
- [ ] `schwabish/SKILL.md`, `schwabish/judge_prompt.md`, `gallery-feedback/SKILL.md` — references to audit paths

### Task 9: Update design docs
- [ ] `design-docs/development-meta-analysis.md`
- [ ] `design-docs/superpowers/plans/2026-05-11-schwabish-plan.md`
- [ ] `design-docs/superpowers/plans/2026-05-11-themes-overhaul-plan.md`
- [ ] `design-docs/superpowers/plans/2026-05-18-interactive-composition-hardening-plan.md`
- [ ] `design-docs/superpowers/specs/2026-05-11-schwabish-design.md`
- [ ] `design-docs/superpowers/specs/2026-05-11-review-lite-agents-design.md`
- [ ] `design-docs/superpowers/specs/2026-05-11-themes-overhaul-design.md`

### Task 10: Rename output directories and update gallery symlink
- [ ] Rename all 5 `.claude/output/*-audit` dirs to `audit-*`
- [ ] Update `gallery` symlink: `rm gallery && ln -s .claude/output/audit-gallery gallery`

### Task 11: Verify
- [ ] `grep -r "pyo3-audit\|gallery-audit\|scene-pipeline-audit\|theme-audit\|interactive-audit\|docs-audit\|interactive-auditor\|pyo3-binding-auditor\|scene-pipeline-auditor\|theme-wiring-auditor" .claude/ CLAUDE.md design-docs/` — should return zero hits
- [ ] Confirm all 6 skill dirs exist under new names
- [ ] Confirm all 4 agent files exist under new names

## 6. Acceptance checks

- Zero grep hits for any old name across `.claude/`, `CLAUDE.md`, `design-docs/`
- `ls .claude/skills/audit-*` shows 6 directories
- `ls .claude/agents/auditor-*` shows 4 files
- `readlink gallery` points to `.claude/output/audit-gallery`

## 7. Open questions

- None — purely mechanical rename.
