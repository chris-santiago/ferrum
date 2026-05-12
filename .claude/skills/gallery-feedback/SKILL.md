---
name: gallery-feedback
description: >
  Interactive row-by-row walkthrough of gallery audit output, collecting user feedback
  on what concepts from comparator libraries (sklearn, scikit-plot, yellowbrick, seaborn,
  SHAP) should become ferrum defaults. Produces a structured remediation plan.
  Use when the user says "gallery feedback", "walk through the gallery", "review gallery
  plots", "collect feedback on plots", "what should we fix in our defaults", "compare our
  plots and tell me what to change", or wants to interactively decide which comparator
  concepts to adopt. Also use after a /gallery-audit run when the user wants to turn
  visual gaps into an actionable punchlist with their input on each row.
---

# Gallery Feedback — Interactive Audit Walkthrough

Walk through every row in the gallery audit output, one at a time. For each row, show the
user ferrum's panel alongside all comparator panels, present a structured comparison, ask
what should change, and record their answer. After all rows, compile everything into a
remediation plan.

This skill is interactive and human-in-the-loop by design — the user's judgment on each
row is the primary input. Your job is to surface the right visual differences and present
well-targeted options so the user can make fast, informed decisions.

---

## Prerequisites

The gallery audit must have already been run. Verify that the output directory exists and
contains row subdirectories with PNG panels:

```
.claude/skills/gallery-audit/output/
  01_roc/
    ferrum.png
    sklearn.png
    skp.png
    yellowbrick.png
  02_pr/
    ...
```

If the output directory is empty or missing, tell the user to run `/gallery-audit` first.

---

## Workflow

### Phase 1 — Discovery

1. List all subdirectories in `.claude/skills/gallery-audit/output/`.
2. Filter to row directories only (pattern: `NN_name`, e.g. `01_roc`). Skip utility
   directories like `_review_lite`.
3. For each row directory, inventory which PNGs exist. Every row should have `ferrum.png`;
   comparator panels vary (`sklearn.png`, `skp.png`, `yellowbrick.png`, `seaborn.png`,
   `shap.png`). Skip any row that has no `ferrum.png`.
4. Tell the user how many rows you found and which comparators are present, then begin.

### Phase 2 — Row-by-row review

Process rows in numerical order. For each row:

#### Step A — Show the panels

Use the `Read` tool on each PNG file to display it to the user. Read ferrum's panel
first, then all comparator panels that exist for that row. Read them in parallel (multiple
Read calls in one turn) so the user sees everything at once.

#### Step B — Present a comparison table

Write a concise markdown table summarizing what each panel shows. Focus on the differences
that matter for default output quality:

- What information is displayed (metrics, annotations, reference lines, legends)
- Axis labels and titles (human-readable vs raw column names)
- Visual elements (colormaps, markers, error bars, CI bands, outlines)
- Layout choices (grid vs stack, margins, label positioning)

Keep each cell to 1-2 lines. The table is a reading aid, not a comprehensive audit —
highlight the gaps and strengths so the user can decide quickly.

Example:

```markdown
| Panel | What it shows |
|---|---|
| **Ferrum** | Single ROC curve, dashed diagonal, raw `fpr`/`tpr` labels, AUC floating in corner |
| **sklearn** | Full axis labels, AUC in legend text ("LogisticRegression (AUC = 1.00)") |
| **scikit-plot** | Per-class + micro/macro curves, AUC per curve in legend, title "ROC Curves" |
```

#### Step C — Ask for feedback

Use `AskUserQuestion` with `multiSelect: true`. Generate 3-4 options based on the actual
visual differences you observed between the panels. Each option should:

- Have a short label (1-5 words) naming the concept
- Have a description explaining what it means concretely and which comparator does it
- Be actionable — something that could be implemented as a ferrum default

Always include a "fine as-is" or "no changes needed" escape hatch as the last option.
The user can also provide freeform text via the built-in "Other" option.

Use a short header (max 12 chars) that identifies the row, e.g. "01 ROC", "09 Box".

Example:

```
AskUserQuestion({
  questions: [{
    question: "For ROC curve — what concepts from the comparators should become ferrum defaults?",
    header: "01 ROC",
    options: [
      { label: "AUC in legend labels", description: "Show 'ClassName (AUC = 0.XX)' in legend entries" },
      { label: "Per-class + micro/macro", description: "For multiclass, show per-class curves plus averages" },
      { label: "Human-readable axis labels", description: "Use 'False Positive Rate' instead of 'fpr'" },
      { label: "Ferrum is fine as-is", description: "No changes needed for this chart type" }
    ],
    multiSelect: true
  }]
})
```

#### Step D — Record and advance

Note the user's selections and any freeform text they added. You do not need to write
anything to disk yet — hold the feedback in conversation context. Acknowledge briefly
(one line) and move to the next row.

If the user selected "fine as-is" (or equivalent), record that and move on without
further discussion.

### Phase 3 — Compile the remediation plan

After all rows are reviewed, write `gallery_feedback.md` at the repository root. If the
file already exists, read it first (required by the Write tool), then overwrite with the
new content.

Structure the document as follows:

```markdown
# Gallery Feedback — Remediation Plan

Collected <date> via interactive gallery audit walkthrough.

---

## Cross-cutting themes

| Theme | Rows affected | Description |
|---|---|---|
| **Human-readable axis labels** | 01, 02, 05, ... | Replace raw column names with descriptive labels |
| ... | ... | ... |

---

## Per-row feedback

### Row 01 — ROC Curve
- [ ] AUC score in legend labels
- [ ] Per-class curves for multiclass
- [ ] ...

### Row 02 — ...
...

### Row 07 — Feature Importance
- Good as-is. No changes.

---

## Summary statistics

- **Rows reviewed:** N
- **Rows with no changes needed:** M
- **Rows where ferrum is ahead:** K (list which)
- **Most common fix category:** ...
- **Bug fixes identified:** ...
```

#### Cross-cutting themes

After compiling per-row feedback, look for patterns that recur across 3+ rows. Common
themes from prior runs include:

- Human-readable axis labels (replacing raw column names)
- Descriptive default titles (auto-generated from chart type + model name)
- CI/error band styling (no outline stroke, lower alpha)
- Markers on discrete data points in line charts
- Y-axis anchored at 0 for bar charts
- Neutral-colored dashed baselines

Extract these into the themes table with the specific rows affected. This tells the
implementer which fixes to tackle as global defaults vs per-chart changes.

#### Per-row checklists

Each item should be a checkbox (`- [ ]`) so the remediation plan doubles as a trackable
punchlist. Include the user's freeform notes verbatim where they added context beyond the
multi-select options.

For rows marked "fine as-is", write a single line: `- Good as-is. No changes.`

For rows where ferrum is ahead of comparators on some dimension, note it:
`- [ ] Keep Brier score annotation (ferrum is ahead here)`

---

## Tone and pacing

This is a long interactive session (potentially 30+ rows). Keep your per-row commentary
tight — the comparison table and the question do the heavy lifting. Don't editorialize
or add recommendations beyond what the panels show. The user is the decision-maker; you
are the facilitator.

Between rows, a one-line transition is enough: "Got it — all four for ROC. Moving to
**Row 02 — Precision-Recall Curve**."

If the user gives terse answers, that's fine — record and move on. If they add detailed
freeform notes, capture them verbatim in the final document.
