# Follow-up: audit Rust coherence pass commits against ferrum-spec.md

**Surfaced:** 2026-05-11 during the Python coherence pass verification.
**Verified:** 2026-05-12 — completed via 7 parallel read-only subagents.
**Resolved:** 2026-05-12 — see `docs/superpowers/audits/2026-05-12-rust-spec-audit.md`. All 7 high-priority commits KEEP; 3 S2 follow-ups logged (§3.12 amendment, static-renderer-design §7 update, `UnsupportedDtype` channel-conflation cleanup); 0 reverts.
**Severity:** S2 / MED — unknown risk. The Rust pass landed without
the spec cross-reference discipline the Python pass adopted; some
intentional spec-committed shapes may have been refactored away
under the assumption they were drift.

## TL;DR

The Python pass's pre-flight verification (4 parallel subagents
cross-referencing `ferrum-spec.md` and `docs/superpowers/specs/`)
caught 6 of 12 medium-confidence "drift" findings as actually
INTENTIONAL — spec-committed shapes the refactor would have wrongly
flattened. The Rust pass shipped 29 commits without the same
discipline. We need to retroactively audit the Rust pass for the
same class of error.

## Why this matters

The Python verification pattern (worked example, 2026-05-11):

- Original verdict: "`figures.py::_resolve_source` 4-way dispatch
  with dict-positional is mode-flag-creep — simplify to explicit
  `compare=` keyword."
- Spec cross-reference: `ferrum-spec.md:947–953` literally
  documents `roc_chart({"a": m1, "b": m2}, X, y)` as a primary
  entry form. `tests/diagnostics/test_compare.py:83` exercises it.
- Recalibrated verdict: INTENTIONAL. Drop the refactor.

If the Rust pass made the analogous error on any public-API-touching
finding, we shipped a silent spec deviation. Most Rust commits were
internal (macro-driven dispatch, dead-code elimination, struct field
collapse) and very low risk. A few touched user-visible surfaces.

## What to audit (highest risk first)

### Highest priority — public API or spec-cited surface

These commits changed shapes the user can observe; audit each for
spec divergence:

| Commit | Hash | What changed | Spec sections to check |
|---|---|---|---|
| F16 | `8cfdc30` | Color type inference: narrow `Float64\|UInt64` → consult `EncodingSpec.type_` first, then widen to all numeric dtypes. | §3.2 type inference, §3.5 color encoding, line 52 "no magic inference that silently fails" |
| F20 | `d5104f2` | Grid compositor: implement ratio-weighted row/col sizing; drop JointChart pre-resize workaround. Behavior change. | §3.12 Compound Views, JointChart spec — note **K9 above already caught a related issue (`spacing` units fraction vs pixels)** in `composition.py`, suggesting §3.12 has fragility |
| F21 | `6535bef` | Remove `share_x` / `share_y` from `compose_svg_grid` signature. Public API removal. | §3.12 Compound Views — verify spec doesn't promise these |
| F5 | `a2e424a` | `RenderError::Other` retired; added `PositionAdjustFailed`, `UnsupportedDtype`, `EmptyDomain` typed variants. Error prose changed. | §3.16 error contract, §3.13 theme errors, any docs/tests asserting error strings |
| F3 | `e0c989a` | `ThemeOverridesSpec` (serde struct + `deny_unknown_fields`) replaces ad-hoc theme dict parsing. Error prose changed for invalid theme keys. | §3.13 Theme keys list; verify the serde-derived prose still matches spec promises |
| F2b | `72cd8c5` | Deleted `scale::core::Scale` monolithic enum; per-variant `XxxScaleData` structs replace it. JSON serde verified byte-identical pre/post. | §3.6 Scales; the JSON byte-identity check during F2 was thorough, but verify spec §3.6 doesn't mention the internal enum |

### Medium priority — internal refactors with minor visible effects

| Commit | Hash | What changed | Spec sections to check |
|---|---|---|---|
| F1 family | `9f3492a` `89f3418` `c07651c` `7350006` `6298bcf` | `for_each_transform!` macro across 5 dispatch sites + `PyQQ` → `PyQq` rename. Python-visible class name unchanged via `#[pyclass(name = "QQ")]`. | §3.4 / §3.5 stat-mark + data-transform tables; verify spec doesn't reference the Rust type name `PyQQ` directly |
| F10 | `39e2a6c` | `for_each_mark!` macro for Mark dispatchers. No visible change. | §3.7 marks list |
| F12, F13 | `669b428` `a4360be` | `build_axis_scale` split + `LocatedColumn` invariant type + `build_color_scale` extraction. Internal only. | n/a (pure internal) |
| F11 | `6fbb153` | `build_from_scale_spec` 5× repetition collapsed. Internal only. | n/a |
| F6 | `1345def` | `arrow_cast` shared module. Internal only. | n/a |
| F18 | `75cec65` | Compositor DRY (`write_svg_open` + `write_cell` helpers). Output byte-identical. | n/a |
| F24+F19 | `2b6fe4c` | `CompositorError` typed variants. Error prose changed. | §3.12 Compound Views error contract |

### Low priority — pure cleanup

| Commit | Hash |
|---|---|
| F4 (MarkStyle base) | `b168cfc` |
| F9 (`transformed` dup field) | `ba81468` |
| F8 (pyo3_serde helper) | `849e6f0` |
| F7 (`Encoding::inherit_from`) | `da26dea` |
| F14 (SizeScale/OpacityScale redundant fields) | `01274de` |
| F15 (`axis_batch_for_y` move) | `7753ade` |
| F17 (LogScale underflow tests) | `6b9cdda` |

These are pure internal refactors with no public-facing change. Audit
last (or skip if time-constrained).

## Audit procedure

For each high-priority commit, dispatch a research-only subagent (one
per commit, in parallel) with this template:

> Read commit `<hash>` (`git show <hash>`). Identify every shape the
> commit changed: function signature, struct field, error variant,
> enum variant, JSON key, error message, public API surface, etc.
> For each shape, locate the matching section in
> `/Users/chrissantiago/Dropbox/GitHub/ferrum/ferrum-spec.md` and
> any matching design doc under
> `/Users/chrissantiago/Dropbox/GitHub/ferrum/docs/superpowers/specs/`.
> Report:
> - **Pre-commit shape** (what the spec or code committed to).
> - **Post-commit shape** (what the commit changed it to).
> - **Spec verdict**: spec committed pre-commit / spec committed
>   post-commit / spec doesn't commit.
> - **Cohesion verdict**: refactor produced materially more cohesive
>   code / cosmetic only / regression.
> - **Recommendation**: keep / revert / amend spec.

Run all high-priority commits in one parallel sweep. Medium / low
in a second sweep if time permits.

## Decision rules

- **Spec committed pre-commit + refactor cohesion gain**: KEEP commit
  + amend spec with a dated note (mirrors the Python pass's K9 / D9c
  pattern).
- **Spec committed pre-commit + no cohesion gain**: REVERT commit.
- **Spec doesn't commit + refactor cohesion gain**: KEEP commit
  (status quo).
- **Spec doesn't commit + no cohesion gain**: KEEP commit but note
  the spec gap for future commitment.

The Python K9 verification is the worked example:
`composition.py` `spacing` parameter — spec L806 said "fraction of
total size", Rust compositor implements as pixels, neither produced
the spec'd behavior. Decision was "update spec to pixels, match
implementation, bump defaults" — KEEP + AMEND SPEC.

## Out of scope

This is an audit, not a redo. The 29 commits are landed and tests
green. The audit is a paper exercise that may produce:
- A small number of spec amendments (dated notes).
- A small number of revert commits (only if a commit produced no
  cohesion gain AND deviated from spec).
- A handful of "spec gap" follow-ups for shapes the spec should
  commit to but doesn't.

Expected outcome: 80% of commits land clean (spec doesn't commit,
refactor is a clear cohesion gain), 15% need spec amendment (K9-style),
~5% might warrant adjustment or revert.

## Suggested timing

Run this audit **after the Python coherence pass completes** — the
Python pass is the active work; the Rust audit is bookkeeping. Doing
it after the Python pass also means we have a fresh pattern for the
spec-cross-reference subagent prompt (the Python verification proved
the workflow).

## Files referenced

- All commit hashes above, viewable via `git show <hash>`.
- `/Users/chrissantiago/Dropbox/GitHub/ferrum/ferrum-spec.md` — the
  API contract.
- `/Users/chrissantiago/Dropbox/GitHub/ferrum/docs/superpowers/specs/`
  — per-phase design docs that may commit additional shapes.
- `/Users/chrissantiago/Dropbox/GitHub/ferrum/docs/superpowers/plans/2026-05-11-rust-coherence-pass-plan.md`
  — the original plan (no spec cross-reference was part of it).
- `/Users/chrissantiago/Dropbox/GitHub/ferrum/docs/superpowers/plans/2026-05-11-python-coherence-pass-plan.md`
  — the Python plan (verification pattern documented in
  "Pre-flight verification — RESULTS" section).
