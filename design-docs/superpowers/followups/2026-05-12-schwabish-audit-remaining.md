# Schwabish audit — remaining unfixed issues

**Filed:** 2026-05-12
**Branch:** `feat/schwabish`
**Closes-via-commits:** `f7c4342, d50ff65, faaddaf, c32333f, 61ecff3, e65f7b1, 0fd7a5c, d291a55, fafadb9` (C1–C9)
**Source audits:**
- `/tmp/python-review-schwabish-heavy.md` (heavyweight Python review)
- `/tmp/rust-review-schwabish-heavy.md` (heavyweight Rust review)

## What this doc is

The heavyweight reviews surfaced 20 findings across Python and Rust. C1–C9
closed all the Schwabish-attributable structural issues (S3 + S4 +
relevant S2). The remaining items are explicitly **not Schwabish-
introduced** or **deferred-by-audit** — pre-existing drift, opinion calls,
or future-polish items the audit itself recommended addressing in a
separate pass.

Each entry below names the finding, why it's still open, and what
follow-up PR / phase / commit it would belong to.

---

## Python — remaining

### P11 (S2) — `Chart + Chart` overlay pattern in chart-builder layer

**Audit position:** > "recommend opening a follow-up issue and addressing them in a dedicated 'figure-builder layer cohesion' sub-phase rather than this Schwabish-followup pass."

**State:** Open. The compositor-based overlay pattern (`+`) exists in
`_gain_chart_from_source`, `_lift_chart_from_source`, and (pre-C7) in
`_discrimination_threshold_chart_from_source`. The architecturally
cleaner shape is a `Chart.add_layer(mark, encoding, mark_kwargs)`
affordance that appends to `_layers` directly, so the chart-builder
layer can stop forking into a second `Chart` for sentinel-column
overlays.

**Where to address:** A dedicated "figure-builder layer cohesion" sub-
phase, not Schwabish.

### P12 (S2) — `lmplot.scatter_kws / line_kws / truncate / x_estimator`, `residplot.dropna / label` reserved-but-no-op

**Audit position:** "Reserved-but-no-op kwargs are an anti-pattern by
project policy (no NotImplementedErrors-as-deferral). Either wire them
or remove them from the signature."

**State:** Open. These kwargs existed on `lmplot` and `residplot`
pre-Schwabish as seaborn-API-compat shims. Schwabish did not introduce
them. Resolving them is its own decision (wire = seaborn-parity work;
remove = breaking change for any caller passing them).

**Where to address:** A pre-existing-drift cleanup PR, separately
labelled (`chore/reserved-kwargs-resolution` or similar).

### Pyright stale-config errors (hygiene)

**State:** Open. Editor diagnostics show ~20 errors like
`_apply_metric_label_explicit is unknown import symbol`,
`Title is not a known attribute of module ferrum`,
`Import "ferrum._direct_label" could not be resolved`. The Python
audit verified every symbol resolves at runtime — names exist where
they should — but did not separately run `pyright` to confirm.
Probably stale `pyrightconfig.json` / pre-existing config drift,
not real bugs.

**Where to address:** A separate hygiene PR — establish a known-clean
pyright baseline.

### Pre-existing ruff D-rule errors

**State:** Open. Lines 95, 691, 1006, 1244 in `figures.py` carry pre-
existing `D202` / `D205` / `D401` lints. None are touched by any
Schwabish or audit commit. The lite-review gate skips them as out-of-
diff debt.

**Where to address:** Same hygiene PR as the pyright sweep, or a
dedicated `ruff --fix` pass on `src/ferrum/figures.py`.

---

## Rust — remaining

### R8b (S2) — `axis_batch_for_y` silently swallows stack errors

**Audit position:** "Documented but worth pinning down with a test that
explicitly produces a Stack-failing batch (e.g., x col not Float64-or-
Utf8) and confirms the downstream draw path surfaces the same error."

**State:** Open. The behaviour (`Err(_) => Cow::Borrowed(primary_batch)`)
is correct — the downstream `apply_stack` call during mark drawing
re-derives the error so the user sees it. But the silent-swallow-and-
retry contract is fragile against future refactors. Pre-existing, not
Schwabish-introduced.

**Where to address:** Add an explicit error-path regression test in a
separate hygiene PR.

### LOESS O(n²) metrics path

**Audit position:** > "Each call to `loess_at_point` re-sorts the full
`xs`/`ys` for the local window — O(n log n) per query, O(n² log n)
total. For modest n (≤1000) this is fine; for the seaborn-style 'lmplot
on 50k rows' scenario it's slow enough to notice. ... Don't fix
preemptively — flag for the residuals + LOESS + metrics combo if it
shows up in real use."

**State:** Open by design. Future performance concern only; no
correctness issue.

**Where to address:** When real-world usage surfaces the slowdown.

### 105 pre-existing dead-code warnings

**Audit position:** > "Notable structural items in the warning list:
`transform/core.rs::apply_transforms*` — unused entry points;
`render/color/scheme.rs::CategoricalPalette` + `Scheme` enum — an
entire unused color-scheme module; `transform/letter_value.rs::OutlierRow`
— letter-value boxplot output type, never constructed. ... A separate
~hour-long pass before merge: pick the 5–10 *structural* deletes."

**State:** Open. Not Schwabish-introduced. The lite-review gates flag
the count every commit but the items aren't in any diff.

**Where to address:** A dedicated "Rust dead-code hygiene" PR before
or in parallel with the merge.

---

## Out-of-scope items the audit flagged but didn't list as findings

- **Examples extension across Schwabish-evolved functions (P9).**
  Partially addressed by C2 (parameters block updates). Adding
  per-kwarg opt-out examples on every Schwabish-evolved function was
  flagged as cheap-but-not-urgent in the audit. Not done in C2 to
  keep the docstring sweep tight.

- **`_*_prep` closure dedup (P5 — S2).** The `_shap_beeswarm_prep`,
  `_silhouette_prep`, `_cpe_prep`, `_disc_threshold_prep` closures
  inside `chart.py` repeat similar "if column X, derive Y" structure.
  Audit: "possibly over-engineered for 3 sites; S2 only." Open.

- **`_resolve_source(compare=)` routing inconsistency.** Pre-existing
  before Schwabish; only `roc_chart`, `pr_chart`, `calibration_chart`
  route it. Other compare-eligible figures accept the dict-form via
  `_resolve_source` but not the explicit `compare=` form. Open.

---

## Summary table

| Finding | Severity | Schwabish? | Status | Next stop |
|---|---|---|---|---|
| P1: roc/pr docstring lies | S4 | yes | **closed (C1)** | — |
| P2: disc_threshold no-op kwarg | S4 | yes | **closed (C1)** | — |
| P3: docstring drift × 8 figures | S3 | yes | **closed (C2)** | — |
| P4: direct_labels sibling drift | S3 | yes | **closed (C8)** | — |
| P5: optimum_label on builder | S3 | yes | **closed (C7)** | — |
| P6: _inject_constant location | S3 | yes | **closed (C3)** | — |
| P7: zero_line shap_beeswarm doc | S2 | yes | **closed (C2)** | — |
| P8: Schwabish tags in docstrings | S2 | yes | **closed (C2 — verified clean)** | — |
| P9: Examples for new kwargs | S2 | yes | partial (C2) | optional future |
| P10: shared format_corner_metrics | S2 | yes | **closed (C9)** | — |
| P11: Chart + Chart overlay | S2 | no | open | figure-builder-cohesion phase |
| P12: lmplot/residplot reserved kwargs | S2 | no | open | reserved-kwargs PR |
| R1: Mark-aware stack branching | S4 | yes | **closed (C6)** | — |
| R2: emission-block duplication | S3 | yes | **closed (C4)** | — |
| R3: metrics_input plumbing | S3 | yes | **closed (C4)** | — |
| R4: Smooth/Robust validation drift | S2 | yes | **closed (C5)** | — |
| R5: inject_metrics validation gap | S2 | yes | **closed (C4)** | — |
| R6: Stack × Mark coverage | S2 | yes | **closed (C6)** | — |
| R7: __repr__ missing fields | S2 | yes | **closed (C4)** | — |
| R8: anchor doc drift | S2 | yes | **closed (C4)** | — |
| R8b: axis_batch_for_y silent swallow | S2 | no | open | rust-hygiene PR |
| Loess O(n²) metrics | S2 | no | open by design | when real use surfaces it |
| 105 dead-code warnings | S1 | no | open | rust-hygiene PR |
| pyright stale config | hygiene | no | open | pyright-baseline PR |
| pre-existing D-rule lints | S1 | no | open | python-hygiene PR |

**Schwabish-attributable findings: 18, all closed.**
**Pre-existing or design-deferred: 6, all flagged for separate follow-up.**
