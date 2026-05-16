# Rust coherence pass — retrospective spec audit

**Date:** 2026-05-12
**Trigger:** `docs/superpowers/followups/2026-05-11-rust-pass-spec-audit.md`
**Scope:** 7 high-priority commits from the Rust coherence pass (F16, F20, F21, F5, F3, F2b, F1-family).
**Procedure:** 7 read-only research subagents dispatched in parallel, each cross-referencing `ferrum-spec.md` + `docs/superpowers/specs/` for one commit.
**Status:** Complete. **No reverts required.** A small number of spec amendments and one S2 follow-up surfaced.

---

## TL;DR

The Rust coherence pass landed **clean** against the spec. None of the 7 high-priority commits silently violated `ferrum-spec.md`. The pattern that worried us going in — "intentional spec-committed shapes refactored away under the assumption they were drift" — did not materialize on any audited commit.

| Commit | Verdict | Action item |
|---|---|---|
| **F16** `8cfdc30` color type inference | **KEEP** | Spec already commits to post-commit shape (line 52 + §3.2). Optional §3.5 clarity note. |
| **F20** `d5104f2` grid compositor ratios | **KEEP + amend spec** | Add §3.12 dated note on ratio-weighted slot allocation algorithm + non-uniform-stretch safety caveat. |
| **F21** `6535bef` `share_x`/`share_y` removal | **KEEP** | None — replacement API tracked as Python P2.8 (`Chart.share_scale` / `Figure.shared`). |
| **F5** `a2e424a` `RenderError` typed variants | **KEEP + amend design doc** | Update `2026-05-09-static-renderer-design.md` §7 RenderError variant list. One residual S2 follow-up. |
| **F3** `e0c989a` `ThemeOverridesSpec` serde | **KEEP** | Pre-existing §3.13 spec→impl gap (3 keys) is closed by the in-flight themes overhaul. |
| **F2b** `72cd8c5` `Scale` enum decomposition | **KEEP** | Pure internal — JSON contract byte-identical. |
| **F1-family** macro + PyQQ→PyQq | **KEEP** | Pure internal Rust dispatch. Spec rightly silent on Rust type names. |

Distribution vs the followup's expected outcome:
- **Expected ~80% clean / 15% spec amend / 5% revert/adjust.**
- **Observed: 5/7 (71%) clean, 2/7 (29%) spec/design-doc amend, 0 reverts.**

Two findings worth surfacing beyond the per-commit verdicts:
1. **Spec amendments are concentrated in §3.12 (Compound Views).** Both F20 and F21 (and earlier K9 / P2.6) point at §3.12 as the section most prone to "spec is silent / implementation has invariants nobody documented."
2. **The followup itself mis-cites spec section numbers** in three rows. Corrected in §"Followup row corrections" below.

---

## Per-commit verdicts

### F16 — `8cfdc30` — Color type inference

**Change:** `build_color_scale` narrow dtype check (`Float64 | UInt64`) replaced with `infer_spec_type(c_enc, dtype)` — consults `EncodingSpec.type_` first, then widens to the full numeric+temporal dtype set. New `EncodingTypeMismatch` error path for `type_=Q/T` on non-numeric columns.

**Spec verdict:** `spec commits post-commit shape`.
- `ferrum-spec.md:52` — "No magic inference that silently fails. If Ferrum infers a scale or encoding type incorrectly, it raises a descriptive error with a suggested fix." The pre-commit narrow check is exactly the silent-fail behavior this line prohibits.
- `ferrum-spec.md:307` — `:Q` / `:T` annotation lets users express type intent on any field; pre-commit code ignored it on color.
- `docs/superpowers/specs/2026-05-10-composite-stat-marks-design.md:212-213` — routing rule keys off inferred spec type, not Arrow dtype.

**Cohesion:** Material gain. Eliminates divergence between axis-scale path (already used `infer_spec_type`) and color path. One inference function, one policy.

**Recommendation:** **KEEP.** Optional cheap-clarity follow-up: add a one-line note in §3.5 confirming "continuous color is selected when `type_` is Quantitative/Temporal *or* when `type_` is None and the column dtype is numeric/temporal."

**Carry-forward:** The new `EncodingTypeMismatch` error has descriptive prose but no "suggested fix" hint per line 52's full text. Pre-existing (variant predates F16), but worth tracking.

---

### F20 — `d5104f2` — Grid compositor ratio-weighted sizing

**Change (two coupled shapes):**
1. `compose_svg_grid` slot allocation: `col_widths[c] = col_widths[c].max(p.width)` → `K * ratio[i]` where `K = min_i(native_dim[i] / ratio[i])`; non-dominant cells scaled into slots via nested `<svg viewBox preserveAspectRatio="none">`.
2. `JointChart.show_svg` no longer pre-resizes marginals; relies on compositor-driven ratio enforcement.

**Spec verdict:** `spec doesn't commit` to the algorithm. `ferrum-spec.md:818` documents `JointChart(..., ratio=5, …)` as "size ratio of center panel to each marginal" — behavioral promise only, no algorithm pinned.

**Cohesion:** Material gain. Two named kwargs (`row_ratios`, `col_ratios`) that were validated-and-discarded now have implementation matching their names. Removes a 20-line workaround in `composition.py` that explicitly compensated for the Rust bug.

**Recommendation:** **KEEP + amend spec.**

**Spec amendments needed (§3.12):**
1. Document slot allocation: `K * ratio[i]` with `K` chosen to keep the dominant cell at native and stretch-fit smaller cells via `preserveAspectRatio="none"`.
2. **Caveat sentence** — non-uniform stretch safety: "Compound views whose intrinsic shape constraints (dendrograms, geographic projections, fixed-aspect coordinate systems) cannot tolerate `preserveAspectRatio='none'` stretching must pre-resize their cells to satisfy declared ratios exactly. `ClusterMapChart` is one such caller." This is currently undocumented and the next composite view will rediscover it.

This is the same K9 / `spacing` pattern: spec was silent, implementation made an invariant choice, document it now before the next contributor drifts.

---

### F21 — `6535bef` — Remove `share_x` / `share_y` from `compose_svg_grid`

**Change:** Removed two dead `#[allow(unused_variables)]` parameters from `compose_svg_grid_py`. Three Python call sites (`JointChart`, `RepeatChart`, `ClusterMapChart`) stopped passing them.

**Spec verdict:** `spec doesn't commit`. `compose_svg_grid` is an internal binding; §3.12 references it once as an implementation detail (line 791). The only axis-sharing language in the spec is line 821's behavioral promise for JointChart marginals — preserved structurally post-F21 (marginals run `axis(show=False)`, compositor stretches them into the data-axis strip).

**Cohesion:** Material gain. Two parameters that were silently lying to callers (JointChart passed `[[2,0]]`/`[[2,3]]` index groups that did nothing) are gone.

**Recommendation:** **KEEP.**

**Cross-reference (per advisor pre-dispatch context):** F21's replacement API is tracked as **Python coherence pass P2.8** (`docs/superpowers/plans/2026-05-11-python-coherence-pass-plan.md:244`) — `K16 — Implement Chart.share_scale(other, channel) + Figure.shared(...). F21 follow-up. Public API addition.` This is a planned redesign, not a silent removal. The audit confirms the followup hand-off (`docs/superpowers/followups/2026-05-11-grid-axis-sharing.md`) captures the right scope. When P2.8 lands, an additive §3.12 amendment will document the new sharing API.

---

### F5 — `a2e424a` — `RenderError::Other` retired, typed variants added

**Change:** `RenderError::Other(String)` removed. Three typed variants added: `PositionAdjustFailed { adjustment: &'static str, reason: String }`, `UnsupportedDtype { channel: String, dtype: String }`, `EmptyDomain { channel: String, field: String }`. `numeric_domain_union` signature changed to return `Result<_, RenderError>` directly.

**Spec verdict:** `spec doesn't commit` on RenderError variant taxonomy. `ferrum-spec.md` §3.16 pins `RenderConfig` fields and output methods, not error variants. The Phase 7 static-renderer design (`docs/superpowers/specs/2026-05-09-static-renderer-design.md:544-554`) enumerates an aspirational 9-variant `RenderError` list — needs updating.

**Cohesion:** Material gain overall. 15 `Other(format!(...))` sites collapse to a structured variant; 3 `ScaleResolutionFailed`-with-prefix sites get typed; the `Other` enum hole is closed.

**Recommendation:** **KEEP + amend design doc.**

**Amendments needed:**
- Update `docs/superpowers/specs/2026-05-09-static-renderer-design.md` §7 RenderError block: enumerate the four current variants (`PositionAdjustFailed`, `UnsupportedDtype`, `EmptyDomain`, plus the still-present `ScaleResolutionFailed`), note `Other` was retired in the Phase 9 coherence pass.
- `ferrum-spec.md` §3.16 / §3.13 require **no change** — never named variants; user contract (`PyValueError` with Display string) is preserved.

**Residual S2 follow-up:** In `scale_resolve.rs::build_size_scale` / `build_opacity_scale` / `resolve_continuous_domain_and_range`, `UnsupportedDtype.channel` is overloaded as `"size:{field}"` / `"scale"`. This bends the variant's intent (channel name vs. context tag) and produces slightly worse Display prose than pre-commit (`"column 'size:X' has unsupported dtype: …"` vs. former `"size: column 'X' has unsupported dtype: …"`). Open a small followup to either (a) add `context: Option<&'static str>` field or (b) accept variant-per-site for these three sites.

---

### F3 — `e0c989a` — `ThemeOverridesSpec` serde struct

**Change:** Replaced 230 LOC of hand-rolled `dict.get_item("K")?` extracts + parallel `KNOWN_THEME_KEYS` const list with `#[derive(Deserialize)] #[serde(deny_unknown_fields)] struct ThemeOverridesSpec`. Same 41 keys accepted pre- and post-commit; same enum-validation prose for `title_anchor`/`legend_orient`/`legend_direction`.

**Spec verdict:** `spec doesn't commit` to a parsing mechanism. §3.13 only constrains the key surface. The accepted-key set is byte-identical pre- and post-commit.

**Cohesion:** Material gain. Removes the parallel `KNOWN_THEME_KEYS` const that the original comment explicitly flagged as a silent-drop hazard. Adding a key now requires one struct field + one merge line, not three coordinated edits. Matches the established `pyo3_serde` idiom used in `spec::chart::new`.

**Recommendation:** **KEEP.**

**Spec→impl gap (NOT introduced by F3; logged for the themes overhaul):**

`ferrum-spec.md` §3.13 documents three keys that *neither* the pre- nor post-F3 implementation accepts:
- `width`
- `height`
- `label_font_size` (only the alias `font_size` is accepted, mapping to `ThemeInputs::label_font_size`)

These gaps predate F3 (same omissions in pre-commit `KNOWN_THEME_KEYS` and in Python `Theme._KNOWN_KEYS`). The Python layer rejects them first with a friendlier error pointing at §3.13, so the user-observable behavior is the same in both cases. The 2026-05-11 themes overhaul design (`docs/superpowers/specs/2026-05-11-themes-overhaul-design.md`) is the canonical place to close these — already in flight on `feat/themes` / `.claude/worktrees/themes`. **No new action item from this audit; cross-reference logged.**

Also: `strip_background_color` is accepted by impl (pre- and post-F3) but not in the §3.13 block — same overhaul scope.

**Minor:** Rust-side unknown-key error prose changed from the helpful `"Unknown Theme key: 'X'. See ferrum-spec.md §3.13 for the supported key list."` to serde's auto-prose `"theme: unknown field \`X\`, expected one of …"`. Masked by Python `Theme` validating first; only surfaces when a caller bypasses Python with a raw dict.

---

### F2b — `72cd8c5` — `scale::core::Scale` enum decomposition

**Change:** Deleted `pub(crate) enum Scale` from `crates/ferrum-core/src/scale/core.rs`; replaced with per-variant `XxxScaleData` structs co-located with each PyO3 facade. `core.rs` shrinks ~785 → ~200 LOC. 8 `unreachable!()` arms removed in this commit, 45 across the F2b series.

**Spec verdict:** `spec doesn't commit` to internal Rust enum names. The user-facing JSON contract `ScaleSpec` (in `crates/ferrum-core/src/spec/encoding.rs`) is a completely separate type and is **not touched** by this commit — `git show 72cd8c5 -- crates/ferrum-core/src/spec/` returns no diff. Serde roundtrip tests confirm `{"type":"log",...}` still round-trips byte-identically.

**Cohesion:** Material gain. Co-locates each variant's data, math, and PyO3 facade in one file. Eliminates the `Scale::invert_f64` stub that returned `NaN` for most variants.

**Recommendation:** **KEEP.**

---

### F1-family — `9f3492a` `89f3418` `c07651c` `7350006` `6298bcf` — `for_each_transform!` macro + `PyQQ` → `PyQq`

**Change:** Single source-of-truth `for_each_transform!` table in `transform/core.rs` replaces 5–7 hand-written 24-arm dispatches (`apply`, `spec_name`, `apply_with_context`, `secondary_outputs`, `ChartSpec.transforms` getter, `coerce_transforms`, `lib.rs` registration block). Adding a new transform drops from 5-file lockstep to 1-line table + enum variant. Net ~-180 LOC.

`PyQQ` → `PyQq` rename: only intra-crate (`pub(crate)`); `#[pyclass(name = "QQ")]` preserves Python-visible class name. Zero Python-side edits; zero test edits; zero `ferrum-spec.md` references to the Rust ident.

**Spec verdict:** `spec doesn't commit`. The spec correctly stays at the user-facing API level and never references Rust type names. `grep -n "PyQQ\|PyQq\|TransformSpec\|for_each_transform" ferrum-spec.md` → zero hits.

**Cohesion:** Material gain. Eliminates the documented parallel-API-drift risk. The `PyQq` rename is load-bearing for the macro shape (third column derivable from variant ident).

**Recommendation:** **KEEP** all five commits.

---

## Followup row corrections

The followup table (`docs/superpowers/followups/2026-05-11-rust-pass-spec-audit.md`) cites spec section numbers that don't match the current `ferrum-spec.md` table of contents. The audit found three:

| Row | Followup says | Actual section |
|---|---|---|
| F20 | "§3.15 composition operators" | **§3.12 Compound Views** (§3.15 is "Sklearn-Protocol Visualizers") |
| F21 | "§3.15 composition" | **§3.12 Compound Views** |
| F2b | "§3.10 scale specs" | **§3.6 Scales** (§3.10 is "Selections (Interactivity)") |
| F1-family | "§3.10 transforms list" | **§3.4 / §3.5** (stat / mark tables) |

These are docs typos in the followup; verdicts above use the correct sections. Fix in the followup doc separately if it's kept around as historical context.

---

## Aggregated action items

Tagged for separate handling. None block any active branch.

### S2 — log as new followups

1. **§3.12 amendment (F20).** Document the ratio-weighted slot allocation algorithm with a dated note matching the K9 / `spacing` pattern. Include the `preserveAspectRatio="none"` non-uniform-stretch caveat naming `ClusterMapChart` as the example of a composite view that pre-resizes for this reason.

2. **§7 RenderError list in `2026-05-09-static-renderer-design.md` (F5).** Update the design-doc enum list to match the current four variants (`PositionAdjustFailed`, `UnsupportedDtype`, `EmptyDomain`, `ScaleResolutionFailed`) with a dated note that `Other` was retired in the Phase 9 coherence pass.

3. **`UnsupportedDtype` channel-field conflation (F5 residual).** S2 cosmetic. Three sites in `scale_resolve.rs` (`build_size_scale`, `build_opacity_scale`, `resolve_continuous_domain_and_range`) overload `channel: String` as `"size:{field}"` / `"scale"`. Either add a `context: Option<&'static str>` field or accept variant-per-site.

### Cross-references (no new action, just acknowledgement)

4. **F21 → Python P2.8.** `Chart.share_scale` / `Figure.shared` is the planned replacement; tracked in Python coherence plan.
5. **F3 → themes overhaul.** §3.13 spec→impl gap on `width` / `height` / `label_font_size` and the inverse `strip_background_color` gap close via `feat/themes` / T1–T4.
6. **F16 → line 52 "suggested fix".** The "with a suggested fix" half of `ferrum-spec.md:52` is not yet implemented by `EncodingTypeMismatch` prose; pre-existing, not introduced by F16.

### Optional low-effort clarity

7. **§3.5 inference table note (F16).** Add a one-line statement of the inference rule for color: "continuous color is selected when `type_` is Quantitative/Temporal, or when `type_` is None and the column dtype is numeric/temporal." Optional.

8. **Followup typo fixes.** Correct the four spec section numbers in `docs/superpowers/followups/2026-05-11-rust-pass-spec-audit.md` per the table above. Optional; the followup is mostly historical now that this audit closes it.

---

## Process notes

- **Single audit pass, 7 parallel subagents.** All read-only; no edits, no commits, no working-tree state mutation. Total wall time ≈ 100s for the parallel dispatch.
- **Method validated.** The cross-reference pattern (subagent reads commit + spec sections, returns 5-field verdict) caught the F20 §3.12 spec gap and the F5 residual S2 — both findings that pure cohesion review would have missed because they're spec-side not code-side issues.
- **The advisor's pre-dispatch caveat held.** F21 was the only commit that *looked* like a public-API removal needing revert; the cross-reference to Python P2.8 made the verdict unambiguous on first pass.
- **Medium / low priority commits not audited.** The followup classifies F1-family (already audited above), F4/F6/F7/F8/F9/F10/F11/F12/F13/F14/F15/F17/F18 as low-risk internal-only refactors. By the same "spec doesn't reference Rust internals" logic that exonerated F2b and F1-family, these are very high-confidence KEEP and were skipped to keep this audit focused. Open a second sweep if any specific commit raises a concrete concern.

---

## Disposition

This audit is the end-of-life for `docs/superpowers/followups/2026-05-11-rust-pass-spec-audit.md`. Either:
- Move the followup to `docs/superpowers/followups/resolved/` (or however the project archives closed followups), or
- Add a dated "Resolved by `docs/superpowers/audits/2026-05-12-rust-spec-audit.md` — KEEP all 7; 3 S2 follow-ups logged" line at the top of the followup and leave it in place as a paper trail.
