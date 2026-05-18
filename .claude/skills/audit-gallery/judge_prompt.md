You are auditing the **default-settings output** of a Rust-backed statistical plotting library called `ferrum` against canonical Python plotting libraries (sklearn, seaborn, yellowbrick, scikit-plot). For each row you receive:

1. A `ferrum` panel rendered with default settings.
2. One or more **reference panels** from canonical libraries, also at default settings.
3. Metadata describing the plot type, dataset, and model.

Your job is to apply the rubric (above, in the cached prefix) and produce a structured verdict identifying what **default information or visual elements** a domain expert would expect to see — that the reference panels show and the ferrum panel does not.

## What to focus on

- **Information content**, not styling preferences. A different shade of blue is not a finding; a missing AUC annotation is.
- **Defaults only.** Do not penalize ferrum for not showing something that none of the reference libraries show by default either.
- **Symmetry.** If ferrum shows something the references don't (e.g. a confidence band the references skip), call it out under `reference_missing` — this is informative for the user too.

## Output format

Respond with **exactly** this structure, no preamble or trailing prose outside the YAML and the prose section:

```
---
row: <row_id>
severity: <HIGH | MEDIUM | LOW | NONE>
ferrum_status: <OK | NOT_IMPLEMENTED | RENDER_ERROR>
ferrum_missing:
  - <short rubric-item label, e.g. "B1_auc_annotation">
  - <...>
reference_missing:
  - <...>
both_missing:
  - <...>
---

# Verdict: <plot type>

## Ferrum lacks (vs reference defaults)

- **<rubric item>** — <one-sentence concrete description of what's missing and why it matters>
- ...

## Reference lacks (vs ferrum defaults)

- ... (or "None")

## Both lack

- ...

## Notes

<1-3 sentences of qualitative observation: overall visual impression, density, color choices, anything that didn't fit the rubric>
```

## Rubric-item labels

Use the rubric IDs verbatim in the YAML lists: `A1_title`, `A2_xlabel`, `B1_auc_annotation`, `C1_chance_diagonal`, `D1_lc_band`, `E1_legend`, `F1_colorblind`, `G1_aspect_ratio`, etc. This lets the report aggregator group findings across rows.

## Severity rules

Apply the severity rules from the rubric (B = HIGH, C/D/E1 = MEDIUM, A/F/G = LOW). Pick the highest severity present. Use `NONE` only if the ferrum panel matches all reference panels on every rubric item.

If `ferrum_status` is `NOT_IMPLEMENTED` or `RENDER_ERROR`, set `severity: HIGH` and explain in the prose Notes section what the failure was.
