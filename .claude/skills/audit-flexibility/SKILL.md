---
name: audit-flexibility
description: Fan out "power-user" agents that behave like experienced data-viz practitioners (fluent in matplotlib, seaborn, Altair, d3, Plotly) who already know the ferrum API and try to build ambitious, beautiful, complicated plots in it — logging what does not work, what is missing, and what is more awkward than the incumbent libraries. Produces a per-category friction report plus a cross-cutting synthesis comparing ferrum's API flexibility and expressive capability against matplotlib/Altair/d3. Use when the user says /audit-flexibility, "compare ferrum's flexibility to matplotlib/Altair/d3", "what can't ferrum do that real viz libraries can", "act like power users and stress the API", "where does our API fall short for custom plots", or wants an expressiveness/capability comparison (distinct from /audit-gallery, which compares default-output quality on a fixed plot set).
---

# Flexibility audit

A repeatable capability comparison. Dispatch one `viz-power-user` agent per plot category. Each agent role-plays an experienced practitioner who **already knows the grammar** and tries to reproduce famous, ambitious chart designs from the matplotlib/seaborn/Altair/d3/Plotly traditions — then reports, candidly and with code, where ferrum's expressiveness falls short of or exceeds the incumbents.

This answers a different question than `/audit-gallery`. Gallery asks "does our **default** output look as good as seaborn's default on a fixed plot set?" This audit asks "**can a power user build the ambitious custom chart they have in their head at all**, and how much does the API fight them versus matplotlib/Altair/d3?"

## When to use

The user wants an honest read on API flexibility / expressive ceiling, not a default-output beauty contest. Trigger phrases are in the description. It is heavier than a single review — it spawns 8 parallel agents that each write and run many throwaway scripts and inspect renders — so it is user-invoked, not automatic.

## Prerequisites

- ferrum built and importable: `unset CONDA_PREFIX && uv run --no-sync maturin develop --release` (per project CLAUDE.md). Confirm with a quick `import ferrum; ferrum.__version__`.
- `resvg-py` available (dev dependency group) so agents can rasterize SVG → PNG for visual inspection.
- Scratch output dir: `/tmp/ferrum-ux-audit/` (agents create their own subdirs).

## Procedure

### Step 1 — Parse scope

Default: run **all 8 categories** (see `personas.md`). If the user names a subset (e.g. `/audit-flexibility scientific interactive`), run only those. The categories are: `distributions`, `explanatory`, `timeseries`, `faceting`, `multivariate`, `scientific`, `categorical`, `interactive`.

### Step 2 — Prep

```bash
mkdir -p /tmp/ferrum-ux-audit
unset CONDA_PREFIX && uv run --no-sync python -c "import ferrum; print('ferrum', ferrum.__version__)"
```

Read `personas.md` to get the per-category brief (incumbents + target plot designs). Optionally skim `src/ferrum/__init__.py` so the dispatch can point agents at any newly-landed public surface.

### Step 3 — Dispatch one `viz-power-user` agent per category, in parallel

**Send all dispatches in a single message** so they run concurrently. For each in-scope category, the prompt is the category's brief from `personas.md` wrapped with the scratch paths and the comparison mandate. Template:

```
YOUR CATEGORY: <Category title from personas.md>
COMPARE AGAINST: <incumbents from personas.md>
SCRATCH DIR: /tmp/ferrum-ux-audit/<slug>/   (throwaway scripts only — do NOT modify ferrum source)
FULL REPORT PATH: /tmp/ferrum-ux-audit/<slug>.md

Attempt the most ambitious ~4-5 of these designs and push them hard:
<bullet list of target designs from personas.md>
<the category's "Push:" line>
<for `interactive`: also paste the "Inspect by reading emitted HTML/JS" and "known deferred limits" notes>

Follow your standing instructions: build each design as a reproducible script in the scratch dir, render AND visually inspect every output (a no-exception render is not a pass), attempt 1-2 workarounds before calling something blocked, pair every "awkward" with the ferrum code and the incumbent equivalent, and write the 5-section report (Attempts table / Blocked-missing / Friction / Wins / Verdict) to the full report path before returning a condensed summary.
```

The agent (`subagent_type: viz-power-user`) already carries the shared context — keep the dispatch focused on the category specifics.

### Step 4 — Synthesize across categories

Once all agents return, read the eight `/tmp/ferrum-ux-audit/<slug>.md` reports (and the returned summaries). Produce a cross-cutting synthesis — the highest-value output, because the same defects recur across categories and the count tells you the priority. Write it to `/tmp/ferrum-ux-audit/SYNTHESIS.md` and present it to the user. Cover:

1. **Cross-cutting defects, ranked by how many categories hit them.** A bug that breaks distributions *and* explanatory *and* categorical is a top-priority fix. For each: the symptom, the categories that hit it, the suspected root cause (file:line if an agent pinned it), and the incumbent that does it as table stakes.
2. **Per-category flexibility verdict** — a one-line ferrum-vs-incumbent grade for each of the 8.
3. **Capability ceiling** — what whole classes of chart are currently inexpressible (not just buggy), with the closest incumbent one-liner.
4. **Where ferrum already wins** — primitives/conveniences that match or beat the incumbents, so a fix campaign doesn't regress them.
5. **Suggested fix order** — the smallest set of changes that unblocks the most chart designs.

Do not file issues, write regression tests, or touch ferrum source as part of this skill — it is a read-only capability audit. If the user wants the findings actioned afterward, that is a separate pass (and any code fix must go through `python-coder`/`rust-coder` + `/regression-test` per project CLAUDE.md).

## Notes

- **Throwaway only.** Agents write scripts under `/tmp/ferrum-ux-audit/`, never under `src/ferrum/` or `crates/`. Pyright diagnostics on the scratch scripts are expected and irrelevant.
- **Inspect, don't assume.** The whole value is in agents *looking at* renders (and at emitted HTML for the interactive category), not just confirming no exception fired. The agent prompt enforces this; keep it that way.
- **Living target lists.** As ferrum's public surface grows, add/swap target designs in `personas.md`. The 8 categories are stable; the ambitions inside them should keep climbing.
- **Repeatability.** Re-running after a fix campaign and diffing the new `SYNTHESIS.md` against the prior one is the natural way to measure flexibility progress release-over-release.
