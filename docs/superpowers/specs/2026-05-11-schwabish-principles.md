# Schwabish Text-Integration Principles for Ferrum

**Date:** 2026-05-11
**Status:** canonical reference
**Scope:** Schwabish's "integrate text and graphics" principle, operationalized for ferrum's statistical gallery.

---

## Source

Jonathan Schwabish, *Better Data Visualizations: A Guide for Scholars, Researchers, and Wonks* (Columbia University Press, 2021), Part I "Visualizing Data Effectively," third principle: *integrate text and graphics*. The book's first two core principles — *show the data* and *reduce the clutter* — are operationalized elsewhere in ferrum: the gallery audit's B-rubric (`/.claude/skills/gallery-audit/rubric.md` §B) covers domain-expected information density, and the themes-overhaul T1–T4 work (`docs/superpowers/specs/2026-05-11-themes-overhaul-design.md`) covers visual polish. This doc covers only the third principle.

The book is the reference; this doc translates the principle into the four T-categories that ferrum's defaults, primitives, and audit pipeline all share.

## Why this principle for ferrum

The gallery audit's rubric is *comparative*: it checks whether ferrum ships, by default, the information that seaborn / sklearn / yellowbrick ship by default. That bar is low. None of those peer libraries ship active titles, callouts, direct labels, or integrated subtitles by default — so the rubric never flags their absence in ferrum either. The audit can return "ferrum is at parity with peers" on a chart that nevertheless fails to communicate.

Themes T1–T4 closed the visual-polish half of the gap: faint gridlines, semibold left-aligned titles, scale padding so marks don't kiss axes, an Observable-Plot-flavored Inter / tableau10 default palette. What remains is the text-integration half — the deliberate weave of words and pictures that distinguishes a chart that *works* from a chart that *speaks*. That is the work this principle governs.

## The four T-categories

The principle decomposes into four checkable categories. These T-IDs are the load-bearing vocabulary shared by the `/schwabish-improve` skill's rubric (`.claude/skills/schwabish/judge_prompt.md`), the autonomous fixer's eligibility list (`apply_eligibility.md`), and the figure-level function defaults (`src/ferrum/figures.py`).

### T1 — Active title

Does the title *communicate a finding*, or does it merely *name the chart type*? `"ROC curve"` names a chart type. `"ROC — AUC 0.94 (good separation)"` communicates a finding. Active titles bias the reader toward the conclusion the chart supports; descriptive titles delegate that work to the caption or to the reader's inference. **Default classification:** subjective — title rewrites depend on the dataset, the audience, and what the chart is being used to argue. Active titles become objective only when a single quantitative metric is computable and a clear template applies (e.g., single-curve ROC → `f"ROC — AUC {auc:.3f}"`); in that case ferrum's figure-level functions assemble the title via f-string and the user receives an informative default. Multi-curve ROC and multi-model overlays fall back to the descriptive title because no single metric can be put there honestly.

### T2 — Direct labels

When a chart has labeled lines or bars and the series count is small (≤4 by convention), the legend can be replaced by text placed at each series' endpoint. The eye then follows a single object — the line and its label — instead of jumping between the plot and a legend square. **Tradeoff:** legends scale to many series; direct labels do not. With more than four series, direct labels overlap and the legend wins. Direct labels are *objective* when the eligibility constraints are met (series count ≤ 4, short string labels); the autonomous fixer applies them and removes the redundant legend in the same patch.

### T3 — Callouts

A callout marks a specific data point with a textual annotation, optionally connected by a leader line. The point being called out is usually a maximum, a threshold crossing, or an anomaly. Callouts are *subjective* — which point deserves emphasis depends on what the chart is being used to argue. Ferrum ships the primitives (`annotate_arrow`, `annotate_text`, `mark_segment`); the `/schwabish-improve` skill surfaces callout opportunities in advisory mode and the user decides where to place them.

### T4 — Inline metrics

Domain-expected numbers — AUC, AP, Brier, R², per-cell counts, importance values — belong *on the plot*, not in a caption or a return value. T4 overlaps with the gallery audit's B-rubric (which calls these "domain-expected annotations") but Schwabish reframes them as text-integration rather than information-density. T4 findings are *objective* whenever a shipped composite annotation (`AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel`) covers the gap or a kwarg flip closes it. Once Schwabish-SB3 lands, the eight figure-level functions ship these annotations by default, and T4 findings against figure-level output drop to zero.

## Objective vs subjective

Every Schwabish finding carries an `objective: true | false` flag. The autonomous gallery mode (`/schwabish-improve --from-audit`) applies only findings with `objective: true`; subjective findings are recorded in the per-row `schwabish_verdict.md` for the user to action manually.

The split is governed by one rule: **an autonomous edit must produce a sensible default for every caller of the affected surface, regardless of dataset or intent**. AUCLabel on a ROC chart is always a defensible default. A specific active-title rewrite ("Model A separates classes well") is not — it bakes intent in. The rule is enforced at two layers: (a) the `apply_eligibility.md` list explicitly enumerates objective finding IDs, and (b) the `schwabish-fixer` agent is sandboxed to `gallery/plots/<row>/ferrum_panel.py` and rejects any finding ID not on the list.

## Where these principles live in ferrum

- **Defaults** — `src/ferrum/figures.py`. Each figure-level function (`roc_chart`, `pr_chart`, `calibration_chart`, `confusion_matrix_chart`, `residuals_chart`, `importance_chart`, `learning_curve_chart`, `validation_curve_chart`) carries Schwabish-compliant kwargs and active-title assembly per Schwabish-SB3.
- **Primitives** — `src/ferrum/annotations.py` (`AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel`, `annotate_arrow`) and `src/ferrum/title.py` (`Title(subtitle=...)`). Each is documented inline with the T-category it serves.
- **Audit pipeline** — `.claude/skills/schwabish/` (advisory + `--from-audit` modes) and `.claude/agents/schwabish-judge.md` / `.claude/agents/schwabish-fixer.md`. The judge prompt embeds this doc as its cached prefix.
- **Override hierarchy** — a user passing `Title("custom string")`, `annotate_auc=False`, or an explicit `legend=` kwarg always wins. Schwabish defaults set the floor; user intent overrides it.

## Out of scope

- Schwabish's chart-type-specific guidance (slope graphs, dot-plot-vs-bar choice, geospatial conventions). Ferrum's gallery is statistical; the chart-type taxonomy from the book does not translate cleanly.
- The first two core principles, *show the data* and *reduce the clutter*. They are covered by the gallery audit's B-rubric and the themes-overhaul T1–T4 work respectively; this doc deliberately does not duplicate them.
- Implementation specifics — those live in the design spec at `docs/superpowers/specs/2026-05-11-schwabish-design.md`. This doc is the *why*; the design doc is the *how*.
