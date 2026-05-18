# Schwabish — Design Spec

**Date:** 2026-05-11
**Status:** approved, awaiting implementation plan
**Slug:** `schwabish`
**Sub-phases:** SB1 (primitives) → SB2 (docs) → SB3 (defaults) → SB4 (advisory skill) → SB5 (gallery-autonomous skill)

---

## Motivation

Three layered observations:

1. **The gallery audit rubric is comparative.** It checks whether ferrum ships what seaborn / sklearn / yellowbrick ship by default. Those peers are themselves weak by Schwabish's bar (active titles, direct labels, callouts, integrated subtitles), so design-quality gaps never surface as findings.
2. **The themes overhaul (T1–T4, on main since `36fdbcb`) closes the visual-polish gap.** Faint grid, semibold left-aligned titles, scale padding, Inter font, tableau10 palette. What remains is *text integration* — Schwabish's third core principle ("integrate text and graphics") — which no current ferrum surface addresses.
3. **Spec drift exists at exactly the surface Schwabish needs.** `ferrum-spec.md §3.11 / §3.19` list `Title(subtitle=...)`, `AUCLabel`, `OutlierLabel`, `annotate_arrow` — none of which are implemented today. Closing this drift is a prerequisite to expressing Schwabish-style improvements cleanly.

This spec operationalizes Schwabish's "integrate text and graphics" principle across ferrum as four connected deliverables — primitives, defaults, principles doc, skill — sharing a single rubric and a single design vocabulary.

### Scope statement

- A canonical **principles doc** for the four text-integration categories (T1–T4).
- A complete set of **text-integration primitives** (closing spec drift in §3.11 + §3.19).
- **Defaults** baked into the 8 existing figure-level functions, so the gallery's default output passes the Schwabish rubric out of the box.
- A `/schwabish-improve` **skill** with an advisory mode (universal, read-only) and a gallery-autonomous mode (applies eligibility-listed objective improvements to panel scripts).

### What this spec does NOT do

- Address Schwabish's "show the data" or "reduce clutter" principles. The first is mostly covered by the existing B-rubric of the gallery audit (domain-expected metrics); the second is mostly covered by themes T1–T4.
- Introduce chart-type-specific Schwabish guidance from the book (slope graphs, dot plots, geospatial, etc.). Ferrum's gallery is statistical; the chart-type taxonomy doesn't translate cleanly.
- Add new figure-level functions. The 8 that exist today are the surface.
- Fabricate subtitles. Functions accept user-supplied subtitle context; they never invent one.
- Auto-rewrite chart titles in the autonomous gallery mode. Title rewriting is subjective and stays advisory-only.

---

## Approach

**Approach B — full Schwabish + missing primitives.** Implement the missing primitives from spec §3.11 / §3.19 (closing drift), bias the 8 existing figure-level functions toward Schwabish-compliant defaults, ship a `/schwabish-improve` skill with advisory + autonomous modes, and author a one-page principles doc that both the skill (cached prefix) and the maintainers cite.

Alternatives considered and rejected:
- **A — scoped to shipped primitives:** subtitle support and the missing composite annotations stay permanently out of scope. Rejected because the spec drift is small and load-bearing for clean Schwabish expression.
- **C — principles-led phasing S1–S4:** similar shape to the chosen approach, but explicitly deferred the missing primitives to a final optional sub-phase. Rejected because the user picked the full-primitives variant; phasing within B accomplishes the same staging without the deferral risk.

---

## Section 1 — Missing primitives

Closing spec drift in `ferrum-spec.md §3.11` and `§3.19`. Four additions; each is additive (no behavioral change to existing charts).

### 1.1 `Title(text, subtitle=...)` — two-line title

Spec §3.19. Today `chart.properties(title="...")` takes a string and `Chart.__init__(title=...)` accepts a string. To support subtitles:

- **Python.** New `ferrum.Title(text, *, subtitle=None, anchor="start", offset=None, font_size=None, font_weight=None, color=None, subtitle_font_size=None, subtitle_color=None)` value class. Accepted everywhere a title string is accepted today: `Chart(title=...)`, `.properties(title=...)`, `HConcat/VConcat/Layer(title=...)`. String titles continue to work — `Title("foo")` is equivalent to passing `"foo"`.
- **Rust.** New `TitleSpec` IR with `subtitle: Option<String>` and subtitle styling fields. Title rendering in `crates/ferrum-core/src/render/marks/title.rs` extends to draw an optional second line. Layout reserves an extra `subtitle_font_size + 2px` strip below the title baseline. When subtitle is `None`, byte-identical to today.

### 1.2 `AUCLabel`, `APLabel`, `BrierLabel` — auto-placed metric labels

Spec §3.11 lists `AUCLabel`. `APLabel` (PR) and `BrierLabel` (calibration) are siblings expanded from the same pattern. Signatures:

- `AUCLabel(*, position="end", format=".3f", prefix="AUC = ")` — placed at the endpoint of each ROC line (or in a corner when `position="corner"`).
- `APLabel(*, position="end", format=".3f", prefix="AP = ")` — placed at the endpoint of each PR line.
- `BrierLabel(*, position="corner", format=".3f", prefix="Brier = ")` — placed at the corner of each calibration curve.

All three are pure Python composites: read the surrounding chart's mark_line data, compute the metric, emit `annotate_text` per series. Added to a chart via `chart + AUCLabel()` (mirrors `chart + annotate_*` pattern). Multi-series → one label per series.

### 1.3 `OutlierLabel` — auto-labeled high-residual / high-leverage points

Spec §3.11. `OutlierLabel(*, threshold=3.0, field=None, label_field=None, max_labels=10)`. Reads the chart's data, identifies points where `|standardized_residual| > threshold` (or `|z(field)| > threshold` when `field` is given), emits `annotate_text` for the top-N (default 10) using `label_field` as the label text.

### 1.4 `annotate_arrow(x1, y1, x2, y2, ...)`

Spec §3.11. Convenience composition of `mark_segment(arrow=True)` + optional `annotate_text` at the `label_side` endpoint. Pure Python; reuses existing primitives.

### 1.5 Where they live

| Symbol | File |
|---|---|
| `Title` | `src/ferrum/title.py` (new) |
| `AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel`, `annotate_arrow` | `src/ferrum/annotations.py` (extend) |
| `TitleSpec` Rust IR + render | `crates/ferrum-core/src/spec/title.rs` + `crates/ferrum-core/src/render/marks/title.rs` |

### 1.6 Tests

- `tests/test_title_subtitle.py` — `Chart(title=Title("foo", subtitle="bar"))` renders both lines; subtitle styling honored; legacy `Chart(title="foo")` byte-identical to today.
- `tests/test_auc_label.py` — single-class + multi-class ROC; correct AUC values; `position="end"` and `position="corner"` both validated.
- `tests/test_ap_label.py` — same shape as AUC, PR.
- `tests/test_brier_label.py` — single-model + multi-model calibration.
- `tests/test_outlier_label.py` — threshold filter + `max_labels` cap.
- `tests/test_annotate_arrow.py` — line segment + arrow + label at correct endpoint.
- Rust unit test for `TitleSpec` serde roundtrip with `subtitle: None | Some(...)`.

---

## Section 2 — Defaults work on the 8 figure-level functions

The principle: **when a quantitative metric exists and the function can compute it, the function should annotate it without the user asking** — at the cost of a flag flip or a small additive change.

### 2.1 Per-function changes

| Function | Verified current state | Schwabish default |
|---|---|---|
| `roc_chart` | `annotate_auc=False` | **Flip to `True`** + active title `f"ROC — AUC {auc:.3f}"` when single-curve. Replace the existing manual injection with `AUCLabel(position="end")`. |
| `pr_chart` | `annotate_ap=False`, `iso_lines=False` | **Flip `annotate_ap=True`** + active title with AP. Also default-on: baseline-prevalence horizontal `annotate_hline(positive_class_rate)` (C2 rubric item). `iso_lines` stays opt-in (clutter risk). |
| `calibration_chart` | No annotation kwarg; perfect-calibration diagonal already shipped per docstring | **New kwarg `annotate_brier=True` default** + `BrierLabel` integration + active title `f"Calibration — Brier {brier:.3f}"` when single-model. |
| `confusion_matrix_chart` | `annotate=True` **already default** ✓ | **No code change** — verify in goldens that cell text actually renders (audit historically flagged it; either the audit was older than this default or rendering is broken). Verification gate in SB3 tests. |
| `residuals_chart` | No metrics kwarg | **New kwarg `annotate_metrics=True` default**: corner annotation `f"R² {r2:.3f}\nRMSE {rmse:.3f}\nMAE {mae:.3f}"`. |
| `importance_chart` | `error_bars=True` ✓ but no value labels | **New kwarg `show_values=True` default**: numeric importance label at end of each bar. |
| `learning_curve_chart` | `ci_style="band"` ✓ | **Add direct-label idiom** (train + val is always ≤2 series). Legend suppressed when direct labels emit; existing legend kwarg as opt-out. |
| `validation_curve_chart` | Same signature shape | Same treatment as `learning_curve_chart`. |

### 2.2 Subtitle convention

For all 8 functions: accept an optional `subtitle: str = None` kwarg. **Default off** — opt-in only. Schwabish encourages subtitles but ferrum cannot infer good subtitle content without context (e.g., "model trained on iris, n=150" requires knowing what the user wants to show). Functions accept a subtitle the user supplies; they do not fabricate one.

### 2.3 Direct-label idiom

A single private helper `_direct_label_endpoint(chart, label_field, position="end")` in a new `src/ferrum/_direct_label.py` that takes a chart + the field whose value labels each series and returns an `annotate_text` overlay placed at line endpoints. Called by `learning_curve_chart`, `validation_curve_chart`, and the gallery-autonomous fixer when it decides to apply direct labels. Not a public mark — composite helper only.

### 2.4 Active title templates

Active titles are computed strings, **not** a template system. Each figure-level function that ships an active title does so via plain f-string assembly inside the function body. No template DSL. No string-substitution primitive. When the function can't compute a clean metric (e.g., `roc_chart` with `per_class=True` — multiple AUCs, no single number to put in the title), it falls back to the descriptive title (`"ROC"`) and lets `AUCLabel` annotate each line.

### 2.5 Golden impact

Every figure-level function in the gallery audit produces a richer panel. Approximately 30–50 goldens regenerate (the 8 figure-function tests, plus any test that exercises them). Standard protocol: `FERRUM_REGENERATE_GOLDENS=1 FERRUM_UPDATE_GOLDENS=1 uv run pytest`, then `python scripts/snapshot-goldens.py`, then Read every PNG. Single inspection batch, single commit per sub-phase.

### 2.6 Tests

- Per-function default-on test: `test_roc_default_annotates_auc.py`, `test_pr_default_annotates_ap.py`, `test_calibration_default_annotates_brier.py`, `test_residuals_default_annotates_metrics.py`, `test_importance_default_shows_values.py`, `test_learning_curve_default_direct_labels.py`, `test_validation_curve_default_direct_labels.py`.
- Per-function opt-out test: `test_roc_annotate_auc_false.py` and equivalents — assert backward-compatibility for users who explicitly set the kwarg to `False`.
- Subtitle test: `test_roc_chart_subtitle.py` — `roc_chart(..., subtitle="iris, n=150")` emits both title lines.
- Confusion-matrix verification: `test_confusion_matrix_cell_counts_render.py` — render with `annotate=True` (default), assert SVG contains the cell-text elements (closes the audit-flagged gap).
- Direct-label exclusivity: `test_learning_curve_no_legend_when_direct_labels.py` — when direct labels emit, legend is suppressed.

---

## Section 3 — Skill `/schwabish-improve` — advisory mode

### 3.1 Trigger and target

`/schwabish-improve <target>` where `<target>` is one of:
- a path to a Python file building a chart (e.g., `gallery/plots/01_roc/ferrum_panel.py`)
- a path to an SVG (e.g., `chart.svg`)
- a path to a directory (recursive advisory scan)

### 3.2 Output

A `schwabish_verdict.md` written next to the target (or to a configurable `--out` path). YAML frontmatter + prose, structured to match the existing `gallery-judge` verdict format so verdicts can be aggregated across the same pipeline.

### 3.3 Verdict structure

```yaml
---
target: <path>
status: <OK | NEEDS_TEXT_INTEGRATION>
findings:
  - id: T1_active_title
    severity: MEDIUM
    objective: false
  - id: T2_direct_labels
    severity: MEDIUM
    objective: true
  - id: T3_callout
    severity: LOW
    objective: false
  - id: T4_inline_metric
    severity: HIGH
    objective: true
---

# Schwabish verdict: <chart description>

## T1 — Active title
<current title> → <suggested title>
**Why:** <one sentence rationale grounded in Schwabish's principle>
**How to apply:** <code snippet>

## T2 — Direct labels
...

## T3 — Callouts
...

## T4 — Inline metrics
...

## Notes
<1–2 sentences qualitative observation>
```

### 3.4 The four T-categories (rubric)

- **T1 — Active title.** Does the title communicate a finding, or just name the chart type? `"ROC curve"` → `"ROC — AUC 0.94 (good separation)"`. **Subjective** by default; objective when a single metric is computable and a clear template exists.
- **T2 — Direct labels.** When ≤4 series and the chart has labeled lines/bars, can the legend be replaced by text at line endpoints? **Objective** when series count is small and labels are short strings.
- **T3 — Callouts.** Is there a specific data point (max, threshold-crossing, anomaly) that deserves an annotated label? **Subjective** — depends on dataset and user intent.
- **T4 — Inline metrics.** Is a domain-expected metric absent (AUC, AP, Brier, R², cell counts)? **Objective** — overlaps with existing audit-gallery B-rubric, but Schwabish reframes it as text-integration. When defaults from Section 2 ship, T4 findings on figure-level-function output drop to zero.

### 3.5 Objectivity flag

Every finding carries `objective: true | false`. The gallery-autonomous mode (Section 4) only applies findings where `objective: true`. Advisory mode reports all four.

### 3.6 Severity rules

- **HIGH** — missing objective metric (T4) where a default exists or is straightforward
- **MEDIUM** — T1 active title or T2 direct labels eligible
- **LOW** — T3 callout opportunity or cosmetic text issue
- **NONE** — chart already satisfies the rubric

### 3.7 How it judges

The skill dispatches a `schwabish-judge` agent (parallel to `gallery-judge`) that reads the chart artifact and a stripped-down rubric (the four T-categories above). The principles doc (Section 5) is the cached prefix to the judge prompt.

### 3.8 What it doesn't do

- Does not run inference — operates on artifacts.
- Does not infer dataset semantics — relies on what's in the chart's title/labels/data.
- Does not propose subtitles unless the user has supplied semantic context via a `--context "<string>"` flag.

### 3.9 Where artifacts live

```
.claude/skills/schwabish/
  ├── SKILL.md
  ├── judge_prompt.md       # rubric + principles doc as cached prefix
  └── apply_eligibility.md  # objective-only list for §4 autonomous mode
.claude/agents/schwabish-judge.md   # per-chart judge agent
.claude/agents/schwabish-fixer.md   # gallery-mode autonomous fixer (§4)
```

Mirrors the `audit-gallery` skill layout.

---

## Section 4 — Skill `/schwabish-improve` — gallery-autonomous mode

### 4.1 Trigger

`/schwabish-improve --from-audit`. No target argument — reads `gallery/output/` and `gallery/plots/<row>/ferrum_panel.py`.

### 4.2 Flow

1. **Discover.** Walk `gallery/plots/`. Skip rows where `config.toml` marks the row as BLOCKED or NOT_WIRED (per RESUME.md convention).
2. **Judge in parallel.** Dispatch one `schwabish-judge` agent per row (mirrors `audit-gallery`'s parallel-dispatch pattern). Each agent reads the row's PNG + panel script + rubric, writes a `schwabish_verdict.md` into `gallery/output/<row>/`.
3. **Filter to objective findings.** For each verdict, keep only findings where `objective: true`. Subjective findings are kept in the verdict (for the user) but not actioned.
4. **Apply via fixer.** Dispatch `schwabish-fixer` agent per row that has ≥1 objective finding. Fixer edits `gallery/plots/<row>/ferrum_panel.py` to add the missing primitive.
5. **Regenerate.** Re-run `audit.py generate --row <id>` for each touched row.
6. **Lite-review gate.** After fixer + regen, dispatch `python-review-lite` agent on the staged diff. Standard cycle protocol: clean → commit; block → re-fix; escalate → halt.
7. **Aggregate report.** Write `gallery/output/SCHWABISH_REPORT.md` summarizing applied changes per row + the subjective findings the user should review.

### 4.3 Eligibility list — what's objective and applies autonomously

| Finding ID | Rubric | Autonomous action |
|---|---|---|
| `T4_auc_label_missing` | T4 | Append `+ AUCLabel()` on ROC panels |
| `T4_ap_label_missing` | T4 | Append `+ APLabel()` on PR panels |
| `T4_brier_label_missing` | T4 | Append `+ BrierLabel()` on calibration panels |
| `T4_residual_metrics_missing` | T4 | Add `annotate_metrics=True` kwarg or append corner annotation on residuals panels |
| `T4_cell_counts_missing` | T4 | Flip `annotate=True` on `confusion_matrix_chart` (or add `mark_text` overlay for hand-rolled panels) |
| `T4_importance_values_missing` | T4 | Flip `show_values=True` on `importance_chart` |
| `T2_direct_labels_eligible` | T2 | Add direct-label `mark_text` at line endpoints AND remove legend, *only when series count ≤ 4 and series labels are short strings* |
| `T4_pr_baseline_missing` | T4 | Append `+ annotate_hline(positive_class_rate, label="baseline")` on PR panels |
| `T4_residual_zero_line_missing` | T4 | Append `+ annotate_hline(0, label=None, stroke_dash=[3,3])` on residuals panels |
| `T4_calibration_diagonal_missing` | T4 | Append calibration y=x diagonal (if not already shipped) |

### 4.4 Non-eligible — advisory only

| Finding ID | Rubric | Why not autonomous |
|---|---|---|
| `T1_active_title_*` | T1 | Title rewriting is subjective; user-facing copy choice |
| `T3_callout_*` | T3 | Where to callout depends on data + intent |
| `T1_subtitle_*` | T1 | Subtitle wording is user-supplied semantic context, not inferable |
| Anything with `objective: false` in the verdict | — | Per-finding judgment by `schwabish-judge` |

### 4.5 Fixer behavior

- **Code edits via `Edit` tool**, not regex. Reads the panel script, identifies the chart construction, appends the new primitive at the appropriate point in the expression chain.
- **Idempotent.** If the panel already has `+ AUCLabel()`, fixer skips. Re-running `--from-audit` produces no diff on a row that's already Schwabish-clean.
- **Restricted to panel scripts.** Does NOT edit `src/ferrum/` source code. Defaults work (Section 2) is a separate concern that lands as code changes to figure-level functions; once those land, `--from-audit` runs find *fewer* T4 findings because the figure functions now default-emit them.
- **No subjective edits.** No title rewrites, no subtitle additions, no callouts. Those stay in the advisory verdict.

### 4.6 Audit trail

- Per-row verdicts: `gallery/output/<row>/schwabish_verdict.md`
- Per-row diff snapshot: `gallery/output/<row>/schwabish_applied.diff`
- Aggregate report: `gallery/output/SCHWABISH_REPORT.md`
- Lite-review verdicts: `.claude/skills/audit-gallery/output/_review_lite/<ISO>_python.md` (reuses existing audit-trail convention)

### 4.7 Interaction with `gallery-fixer`

`gallery-fixer` closes findings from `/audit-gallery`'s REPORT.md (comparative parity). `schwabish-fixer` closes objective findings from `schwabish-judge`'s verdicts (text-integration). They operate on the same panel scripts but with non-overlapping change types — `gallery-fixer` adds primitives the *peer libraries* already ship; `schwabish-fixer` adds primitives *no peer ships* (Schwabish's distinguishing axis).

In practice, after Section 2 defaults land, both fixers find fewer to do — the figure-level functions ship Schwabish-compliant + peer-parity by default, and only hand-rolled panel scripts need either fixer.

### 4.8 Commit shape

One commit per row touched. Message form:
```
feat(gallery): schwabish improvements on row <id>
- T4_auc_label_missing: added AUCLabel()
- T2_direct_labels_eligible: replaced legend with endpoint labels
```

Mirrors `gallery-fixer`'s commit shape. The aggregate `SCHWABISH_REPORT.md` lands in a separate docs commit.

### 4.9 Tests

- `tests/test_schwabish_from_audit_idempotent.py` — running `--from-audit` twice produces no diff on the second run.
- `tests/test_schwabish_eligibility_list.py` — subjective findings are listed in verdicts but never appear in applied diffs.
- `tests/test_schwabish_lite_review_gate.py` — `python-review-lite` returning block status un-stages the change (mocked).

---

## Section 5 — Principles doc

**Location:** `docs/superpowers/specs/2026-05-11-schwabish-principles.md` (alongside this design spec).

**Audience:** ferrum maintainers (informs defaults work, future figure-level functions, future audits) and the `schwabish-judge` agent (cached prefix to the judge prompt).

**Length:** one page, ~600–800 words. No code.

**Structure:**

1. **Source.** One paragraph crediting Schwabish's *Better Data Visualizations* (Columbia, 2021) and naming the third principle ("integrate text and graphics") as the operationalized subset.
2. **Why this principle for ferrum.** Two paragraphs: the comparative-parity blind spot in the gallery audit; what themes T1–T4 already covered (visual polish); what remains (text integration).
3. **The four T-categories.** One paragraph each:
   - T1 active titles — what an active title is, the two-kind split (descriptive baseline OK / active when a metric exists), with worked example contrasting `"ROC curve"` vs `"ROC — AUC 0.94 (good separation)"`.
   - T2 direct labels — when to prefer over a legend (≤4 series, short labels), tradeoff vs. legend (legend handles many series, direct labels lead the eye).
   - T3 callouts — purpose (point at the punchline), implementation note (`annotate_arrow` for leader, `annotate_text` for floating).
   - T4 inline metrics — domain-expected numbers that belong on the plot, not in a caption.
4. **Objective vs subjective.** Half a page on which findings the autonomous gallery mode applies and which are advisory-only, plus the reasoning (auto-applied changes must produce a sensible default for *every* caller; subjective changes depend on the dataset/intent).
5. **Relationship to other surfaces.** Half a page mapping principles to:
   - Where defaults live (figure-level functions in `src/ferrum/figures.py`).
   - Where annotations live (`src/ferrum/annotations.py`, `src/ferrum/title.py`).
   - Where the audit lives (`/schwabish-improve`).
   - Where it does *not* override: a user passing `Title("custom string")` or `legend=...` explicitly always wins.

**Not in the doc:**
- Per-chart-type Schwabish guidance from the book that doesn't apply to ferrum's statistical gallery.
- Anything about "show the data" or "reduce clutter" — out of scope.
- Implementation specifics (those live in this design doc).

**Cross-references:**
- The judge prompt (`.claude/skills/schwabish/judge_prompt.md`) embeds this doc as its cached prefix.
- This design spec references this doc as the *why*; the design doc is the *how*.
- Future figure-level functions (Phase 10+) cite this doc when justifying their default annotation choices.

---

## Section 6 — Sub-phase decomposition

Five sub-phases, each ending in green tests + a committable state.

### SB1 — Missing primitives

Implements Section 1 in full. No goldens regenerate (subtitle renders only when supplied; new composite annotations are additive).

### SB2 — Principles doc + skill scaffolding

Writes:
- `docs/superpowers/specs/2026-05-11-schwabish-principles.md` per Section 5.
- `.claude/skills/schwabish/SKILL.md` (skill entry point).
- `.claude/skills/schwabish/judge_prompt.md` (rubric + principles doc as cached prefix).
- `.claude/agents/schwabish-judge.md` (per-chart judge subagent).

No code changes outside docs/skills.

### SB3 — Defaults work on the 8 figure-level functions

Implements Section 2 in full. Goldens regenerated (~30–50 SVGs). Single inspection batch, single commit.

### SB4 — Skill advisory mode

Implements Section 3 in full:
- Target detection (Python file vs SVG vs directory) in `SKILL.md`.
- `schwabish-judge` dispatch + verdict aggregation.
- No code edits to ferrum source; advisory only.

### SB5 — Skill gallery-autonomous mode

Implements Section 4 in full:
- `--from-audit` flag on `/schwabish-improve`.
- `schwabish-fixer` agent (`.claude/agents/schwabish-fixer.md`).
- Per-row regeneration via `audit.py generate --row <id>`.
- Lite-review gate via existing `python-review-lite` agent on staged diffs.
- Aggregate `gallery/output/SCHWABISH_REPORT.md` writer.

### Cross-phase invariants

- **Worktree:** `.claude/worktrees/schwabish/` on `feat/schwabish` branch based on latest main. Justified by concurrent Python-refactor session in main checkout.
- **Setup cost:** `unset CONDA_PREFIX && uv run --no-sync maturin develop` after worktree creation (Rust subtitle work in SB1 requires a rebuild).
- **Canonical Rust-test command in worktree:** `PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none DYLD_LIBRARY_PATH=$PYTHONHOME/lib cargo test` (per memory note for worktree cargo tests).
- **Test ratchet:** `cargo test` + `uv run pytest` both green at the end of every sub-phase.
- **Spec divergence:** none introduced. Every change closes drift (SB1) or adjusts defaults (SB3); both recorded as dated `2026-05-11` notes in `ferrum-spec.md §3.11 / §3.13 / §3.14 / §3.19` (Section 8).
- **Rebase cadence:** `git fetch && git rebase origin/main` from the worktree before each sub-phase commit.

### Why this order

- SB1 first because every later sub-phase depends on the primitives existing.
- SB2 before SB3 so the figure-function changes cite the canonical principles doc.
- SB3 before SB4/SB5 so the gallery panels show Schwabish-compliant output before the skill runs against them — keeps the skill's surface findings minimal and the regen volume in SB3 isolated to one inspection batch.
- SB4 before SB5 because gallery-autonomous mode is built on advisory mode's judge agent.

### Off-ramps

Each sub-phase is independently committable. Acceptable stopping points:
- After SB1: spec drift closed, no user-visible behavior change.
- After SB2: principles documented; future work can reference them.
- After SB3: gallery panels look better, no new skill machinery.
- After SB4: users can run `/schwabish-improve <path>` advisorily.
- After SB5: full feature shipped.

---

## Section 7 — Test plan and golden regen

### Existing golden footprint (pin at T0)

```
tests/goldens/**/*.svg                          ≈X files
tests/test_phase_9_e2e/goldens/*.svg            ≈Y files
crates/ferrum-core/tests/.../*.svg              (some Rust-side)
```

Exact counts pinned at the start of SB1, recorded in the implementation plan.

### Tests added per sub-phase

Itemized in §1.6, §2.6, §3.x, §4.9. Summary:

| Sub-phase | Test files added |
|---|---|
| SB1 | `test_title_subtitle.py`, `test_auc_label.py`, `test_ap_label.py`, `test_brier_label.py`, `test_outlier_label.py`, `test_annotate_arrow.py`, Rust serde roundtrip |
| SB2 | none (docs + skill scaffolding) |
| SB3 | 7 default-on tests, 7 opt-out tests, subtitle test, confusion-matrix verification, direct-label exclusivity |
| SB4 | `test_schwabish_judge_dispatch.py`, `test_schwabish_rubric_objective_flag.py` |
| SB5 | `test_schwabish_from_audit_idempotent.py`, `test_schwabish_eligibility_list.py`, `test_schwabish_lite_review_gate.py` |

### Golden regen mechanics (SB3 only)

1. Pin golden count at T0: `find tests/goldens -name '*.svg' | wc -l` and `find tests/test_phase_9_e2e/goldens -name '*.svg' | wc -l`. Record in implementation plan.
2. `FERRUM_REGENERATE_GOLDENS=1 FERRUM_UPDATE_GOLDENS=1 uv run pytest` to regenerate everything the 8 figure functions touch.
3. `python scripts/snapshot-goldens.py` — rasterize every regenerated SVG to PNG.
4. **Read every PNG.** Batches of ~10 via `Read` calls. Checks:
   - AUC / AP / Brier text present at expected position; numeric format matches `.3f`.
   - R² / RMSE / MAE corner annotation present on residuals goldens.
   - Cell text visible on confusion-matrix goldens.
   - Direct-label text at line endpoints on learning / validation curve goldens; legend absent.
   - No resvg-py path truncation (cross-check `grep -oE 'd="M' foo.svg | wc -l` for any panel that looks empty).
5. Commit. Message names the sub-phase + visual changes verified.

### CI

- `cargo test` runs at end of every sub-phase. Must be green.
- `uv run pytest` runs at end of every sub-phase. Must be green.
- No new CI jobs.

### Risk register

| Risk | Mitigation |
|---|---|
| Subtitle layout collides with existing chart titles in goldens | SB1 explicitly does not regenerate; subtitle only renders when supplied. Existing goldens stay byte-equal. |
| SB3 flips a default that breaks a downstream test asserting a precise pixel position | Tests should assert via public API; tightly-coupled tests get a one-line update with comment pointing at the dated spec note. |
| `confusion_matrix_chart` cell-text broken (audit flagged it but `annotate=True` is default) | SB3 includes an explicit verification test; if broken, fix as part of SB3. |
| Direct-label idiom clashes with existing legend on learning_curve goldens | SB3 test asserts legend absent when direct labels emit; explicit opt-out via existing legend kwarg if user wants both. |
| Gallery-autonomous fixer over-applies on a row with hand-rolled chart code | Eligibility list + idempotence + lite-review gate; orchestrator un-stages on block status. |
| Worktree `cargo test` fails on documented `DYLD_LIBRARY_PATH` line | Plan uses explicit `PYTHONHOME` form; verified in worktree-setup step. |
| Concurrent main session merges into schwabish-touching code | Worktree isolates; rebase before each sub-phase commit; conflicts surface cleanly. |

---

## Section 8 — Spec updates

CLAUDE.md hard constraint: spec divergence is forbidden; dated notes record evolution. Four spec touches dated `2026-05-11`:

### `ferrum-spec.md §3.11` — Annotations

Appended after the table:

```markdown
> **2026-05-11 (Schwabish SB1):** `AUCLabel`, `OutlierLabel`, and
> `annotate_arrow` are now implemented. Two sibling composites,
> `APLabel(*, position="end", format=".3f", prefix="AP = ")` and
> `BrierLabel(*, position="corner", format=".3f", prefix="Brier = ")`,
> are added — same pattern as `AUCLabel` but for PR (AP) and
> calibration (Brier score). When added to a multi-series chart, each
> composite emits one label per series. All four metric labels read
> the surrounding chart's mark_line data and compute the metric in
> Python (no new Rust IR).
```

### `ferrum-spec.md §3.19` — Utilities (`Title` class)

Appended after the `Title(...)` signature block:

```markdown
> **2026-05-11 (Schwabish SB1):** `Title(text, subtitle=...)` is now
> implemented. Accepted everywhere a title string is accepted today
> (`Chart(title=...)`, `Chart.properties(title=...)`, `HConcat/VConcat/
> Layer(title=...)`). Subtitle renders as a second line below the title
> baseline using `subtitle_font_size` (default `title_font_size * 0.85`)
> and `subtitle_color` (default `theme.label_color`). When subtitle is
> `None`, title rendering is byte-identical to passing a bare string.
```

### `ferrum-spec.md §3.14` — Figure-level functions

Appended at the end of the section:

```markdown
> **2026-05-11 (Schwabish SB3):** Schwabish text-integration defaults
> across the 8 figure-level functions. Default flips:
> - `roc_chart.annotate_auc`: `False → True` (uses new `AUCLabel`).
> - `pr_chart.annotate_ap`: `False → True` (uses new `APLabel`); adds
>   default baseline `annotate_hline(positive_class_rate)`.
> - `importance_chart.show_values`: new kwarg, default `True`.
>
> New kwargs (default-on):
> - `calibration_chart.annotate_brier=True` (uses new `BrierLabel`).
> - `residuals_chart.annotate_metrics=True` (R²/RMSE/MAE corner annotation).
>
> Default behaviors not exposed as kwargs:
> - `learning_curve_chart` and `validation_curve_chart` emit direct
>   labels at line endpoints and suppress the legend when ≤2 series.
>   Existing `legend=` kwarg as opt-out.
> - Single-curve `roc_chart` / `pr_chart` / `calibration_chart` ship an
>   active title (`f"ROC — AUC {auc:.3f}"`, etc.). Per-class /
>   multi-model renders fall back to the descriptive title.
>
> All 8 functions accept an optional `subtitle: str = None` kwarg.
> Default off — functions never fabricate subtitle content. See
> `docs/superpowers/specs/2026-05-11-schwabish-principles.md` for the
> design rationale.
```

### `ferrum-spec.md §3.13` — Themes (cross-reference only)

No new keys. A one-line cross-reference appended:

```markdown
> **2026-05-11 (Schwabish SB1):** `Title(..., subtitle_font_size,
> subtitle_color)` defaults fall back to `title_font_size * 0.85` and
> `theme.label_color` respectively. No new Theme keys introduced.
```

---

## Section 9 — Out of scope

- "Show the data" and "reduce clutter" principles — partially covered by existing B-rubric + themes T1–T4.
- Chart-type taxonomy guidance from Schwabish (slope graphs, dot plots vs bars, geospatial, etc.).
- Subtitle fabrication — figure functions accept a user-supplied subtitle but never invent one.
- Title-rewriting in gallery-autonomous mode — subjective, advisory only.
- Per-finding override flags on individual figure-level functions for nuanced Schwabish behavior — over-engineered; the kwarg + sensible default model is enough.
- New figure-level functions — only the 8 that exist today.
- Vega-Lite output translation for new annotation composites — out of scope; their JSON representation reuses existing `annotate_*` IR, so any translation that supports `annotate_text` + `mark_segment` already covers callouts.
- Iso-F1 lines on `pr_chart` — stay opt-in (clutter risk).
- `--apply` flag on advisory mode — advisory mode stays read-only by design; autonomous behavior is reachable only via `--from-audit`.

---

## Appendix — Decisions captured during brainstorming

| Decision | Choice |
|---|---|
| Motivating gap | All three (rubric blind spot + defaults framework + UX) |
| Schwabish scope | "Integrate text and graphics" only |
| Deliverable surfaces | Defaults + skill + principles doc |
| Pipeline integration | Independent skill with `--from-audit` mode |
| Skill output mode | Universal advisory + gallery-focused autonomous (hybrid) |
| Approach | B — full Schwabish + missing primitives |
| Naming | `schwabish` (skill, docs, agents) |
| Implementation isolation | New worktree at `.claude/worktrees/schwabish/` |
| Sub-phase order | SB1 → SB2 → SB3 → SB4 → SB5 |
| Subtitle handling | Off by default; user supplies semantic context |
| `pr_chart` iso lines | Stay opt-in (clutter risk) |
| Confusion-matrix annotations | `annotate=True` already default; verification gate in SB3 |
| Direct labels | ≤4 series + short labels; replace legend |
| Active titles | Computed strings via f-string assembly; no template DSL |
| Eligibility list | Objective-only findings auto-apply; subjective stay advisory |
