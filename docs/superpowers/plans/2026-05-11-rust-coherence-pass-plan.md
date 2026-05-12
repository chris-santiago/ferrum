# Rust coherence pass — refactor plan

**Date**: 2026-05-11
**Branch**: stacks on `fix/marks-and-composition-wiring`
**Scope**: `crates/ferrum-core/`. No Python source changes. No `ferrum-spec.md` changes.
**Goal**: recover architectural cohesion before first release. Remove 24-variant
boilerplate fan-out, collapse the over-wrapped scale stack, drain accreted
phase-by-phase drift in `render/scale_resolve.rs`, unify Arrow dtype handling,
unify error stories, fix two latent correctness bugs surfaced during the audit.

This is **pure refactoring + two opportunistic correctness fixes**. JSON shape
of `ChartSpec`, Python-visible API, and SVG goldens are preserved on every
commit **except F16**, which is intentionally last and ships its own
re-blessed goldens. Known side effects:

- **Error message wording** changes for invalid theme dicts (F3) and scale
  resolution failures (F5) — the structure becomes typed, the prose tightens.
- **F16 (final commit)** changes runtime behavior: color type inference now
  consults `EncodingSpec.type_` first and treats *all* numeric Arrow dtypes
  (Float32/64, Int8–64, UInt8–64) as continuous when `type_` is unset.
  Integer color columns previously routing to categorical (e.g. `Int64`
  cluster IDs) now route to continuous unless explicitly marked
  `type="nominal"`. Goldens affected by this behavior change are
  regenerated and visually inspected in the F16 commit per CLAUDE.md.

---

## Finding inventory

17 findings, grouped by dependency tier. Each gets at least one commit;
high-fanout ones split into sub-commits as noted.

### Tier 0 — Foundation (no inter-dependencies)

| # | Finding | Severity | Commit(s) |
|---|---|---|---|
| F6 | Arrow dtype trampolines duplicated across 5 sites (~175 LOC). Introduce `render/arrow_cast.rs` with `numeric_iter`, `string_iter`, `min_max_f64`, `distinct_values`. Replace all 5 call sites. | S2 / HIGH | 1 |
| F4 | `resolve_mark_style` builds 7 near-identical `MarkStyle` literals. Extract `MarkStyle::for_mark(theme, mark) -> Self` so per-mark arms become 5-field overrides on a base. | S2 / HIGH | 1 |
| F9 | `PreparedInputs.transformed` duplicates `transform_outputs[FINAL_OUTPUT_KEY]`. Drop the field; have consumers look up the key. | S2 / MED | 1 |
| F8 | PyDict ↔ serde round-trip via `json.dumps`/`json.loads` is open-coded in 4 spots in `spec/chart.rs::new`. Centralize in a `pyo3_serde::from_py_dict::<T>(obj)` helper. Pure prep for F3. | S2 / MED | 1 |
| F1a | Introduce a `for_each_transform!` declarative macro in `transform/core.rs`. Replace `spec_name` and `TransformSpec::apply` with macro-driven dispatch. (Smallest demo slice — covers ~50 LOC.) | S4 / HIGH | 1 |

### Tier 1 — Architectural rollouts (depend on Tier 0)

| # | Finding | Severity | Commit(s) |
|---|---|---|---|
| F1b | Extend `for_each_transform!` to `apply_with_context` and `secondary_outputs`. Encode the "this variant overrides default" matrix declaratively. | S4 / HIGH | 1 |
| F1c-i | Extend `for_each_transform!` to `spec/chart.rs::transforms` accessor (`Py<PyXxx>` projection). Each variant goes from a hand-written `pyo3::Py::new` call to a macro arm. | S4 / HIGH | 1 |
| F1c-ii | Extend `for_each_transform!` to `spec/chart.rs::coerce_transforms` (24 sequential `if let Ok(_) = item.extract::<PyXxx>()` → macro-generated dispatch). | S4 / HIGH | 1 |
| F1c-iii | Extend `for_each_transform!` to `lib.rs::_core` (`m.add_class::<PyXxx>()` × 24 → single macro invocation). | S4 / HIGH | 1 |
| F10 | Apply the same macro pattern to `Mark::dispatch_mark` (12 variants). Trivial follow-up once the macro exists. | S1 / HIGH | 1 |
| F3 | `theme_from_dict`: replace 230 LOC of `if let Some(v) = d.get_item("…")?` with a `ThemeOverridesSpec` struct using `#[serde(deny_unknown_fields)]` consumed via F8's helper. String enums (`TitleAnchor`, `LegendOrient`, `LegendDirection`) become `#[derive(Deserialize)]` enums. | S3 / HIGH | 1 |
| F12 | Split `build_axis_scale` (~150 LOC) into `axis_domain(…)` + `axis_pixel_range(…)` + thin dispatcher. Replace `expect("locate_field_batch guarantees …")` with a `LocatedColumn<'a>` return type that makes the invariant unrepresentable. | S3 / HIGH | 1 |
| F13 | Promote the inline color-channel logic in `resolve_scales_with_outputs` into a `build_color_scale(encoding, batch, outputs, theme) -> Result<(Option<ColorScale>, Option<RenderWarning>)>` to match the parallel shape of `build_size_scale`/`build_opacity_scale`/`build_shape_scale`. | S3 / HIGH | 1 |
| F11 | Collapse `build_from_scale_spec` 5× copy-paste. The Linear/Log/Time/Symlog arms are byte-identical apart from the final constructor call. Extract `resolve_continuous_domain_and_range(…)`. | S2 / HIGH | 1 |

### Tier 2 — Scale architecture (biggest move)

| # | Finding | Severity | Commit(s) |
|---|---|---|---|
| F2a | Introduce a `trait Scale1D { fn to_pixel(…); fn ticks(…); fn range_pair(…); fn repr(…); }` implemented by `LinearScale`, `LogScale`, `SymlogScale`, `TimeScale`, `OrdinalScale`. `ScaleKind::pixel_range` and `tick_labels` collapse from 5-arm matches to one trait call. | S3 / HIGH | 1 |
| F2b | Replace the per-file `XxxScale(crate::scale::core::Scale, …)` newtype + monolithic enum with per-variant inner types (`scale::linear::LinearScaleInner`, `scale::log::LogScaleInner`, …). All `#[allow(unreachable_patterns)] _ => unreachable!()` arms disappear. **Pre-flight JSON serde stability check**: pick 3 serialized scales from existing tests (one Linear, one Log/Symlog, one Time), save the JSON before, run F2b, assert byte-identical JSON after. If `EncodingSpec.scale` round-trip shape changes, every chart with an explicit `scale=fr.LinearScale(...)` would silently break — abort the commit and revisit. | S3 / HIGH | 2 |

### Tier 3 — Error stories, polish, latent bugs

| # | Finding | Severity | Commit(s) |
|---|---|---|---|
| F5 | Drain `RenderError::Other(String)` and the 7 ad-hoc-prefix uses of `RenderError::ScaleResolutionFailed(String)`. Promote to structured variants: `PositionAdjustFailed { stage, message }`, `UnsupportedDtype { channel, dtype }`, `EmptyDomain { channel, field }`. | S3 / MED | 1 |
| F7 | `LayerPrepared::from_chart_and_layer` hand-merges 8 encoding channels with inconsistent `merge_scale` policy. Extract `Encoding::inherit_from(&Encoding)` with a single documented policy. Decide whether `shape`/`opacity`/`x2`/`y2` should also receive `merge_scale` (likely yes — silent asymmetry). | S2 / MED | 1 |
| F14 | `SizeScale`/`OpacityScale` carry redundant `{min,max}_px` / `{min,max}_opacity` fields plus an `inner: ScaleKind::Linear` whose range already encodes those bounds. Collapse into a single `RangedLinearScale { inner }` and read the bounds via `inner.pixel_range()`. | S4 / MED | 1 |
| F15 | `find_stack_for_y` in `scale_resolve.rs` reaches into `crate::render::position::apply_stack` — couples scale resolution to one specific position adjustment. Lift to `position::resolve_axis_batch(spec, channel, primary) -> Cow<RecordBatch>`. | S4 / MED | 1 |
| F16 | **LATENT BUG**: `resolve_scales_with_outputs` line 341 uses `matches!(col.data_type(), Float64 \| UInt64)` to decide continuous-vs-categorical color. Float32/Int32/Int64/etc. silently route to categorical. Widen via F6's `arrow_cast::is_numeric(col)`. Goldens that rely on the old narrow detection must be re-blessed. | S4 / LOW | 1 |
| F17 | **LATENT BUG investigation**: `LogScale::to_pixel_f64` returns `None` only on non-finite output; for input ≤ 0 the underlying scale's NaN path is undocumented. **Procedure**: add a Rust unit test calling `LogScale::to_pixel_f64(0.0)` and `LogScale::to_pixel_f64(-1.0)`; if the result is `Some(NaN)` (silent), that's the bug → fix in a separate commit; if the result is `None` or a finite clamped value, document the contract in the scale's docstring and close. | S5 / LOW | 1 (or 0 if investigation closes) |

**Total**: 24 findings → ~26 commits across 4 tiers.

---

## Dependency graph

```
F6  ─┬─────────────────────────────────────────────────────────────┐
     │                                                             │
     ├─→  F12 (axis scale split needs arrow_cast)                  │
     │                                                             │
     ├─→  F13 (color builder split)                                │
     │                                                             │
     ├─→  F16 (numeric color widening)                             │
     │                                                             │
F4   ┘ (independent)                                               │
F9   (independent)                                                 │
F8   ─→  F3 (theme spec deserializer uses the PyDict helper)       │
F1a  ─→  F1b  ─→  F1c                                              │
                  └─→  F10 (Mark dispatch reuses the macro)        │
F11  (independent of arrow_cast — purely structural)               │
F2a  ─→  F2b                                                       │
F5   ─→  (after F12/F13/F15 land; new error variants are wired in) │
F7   (independent)                                                 │
F14  (independent)                                                 │
F15  (independent of F12; can land before or after)                │
F17  (investigation; standalone)                                   │
```

Critical path: **F6 → F12 → F13 → F5**. The arrow_cast extraction is the
foundation that lets the scale_resolve.rs cleanup land cleanly.

---

## Validation strategy

**After every commit** (the standard sequence):

1. `unset CONDA_PREFIX && uv run --no-sync maturin develop` — rebuild.
2. `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test` — Rust unit tests.
3. `uv run pytest -x` — Python suite (fails fast).
4. **Golden hash check** for refactor-only commits (everything except F16, F20, F22):

   ```bash
   # Before any code change in the commit:
   find tests/goldens tests/test_phase_9_e2e/goldens -name '*.svg' \
     | sort | xargs sha256sum > /tmp/goldens-pre.txt
   # After the commit's changes are staged:
   find tests/goldens tests/test_phase_9_e2e/goldens -name '*.svg' \
     | sort | xargs sha256sum > /tmp/goldens-post.txt
   diff /tmp/goldens-pre.txt /tmp/goldens-post.txt
   ```

   **Empty diff = pass.** Any change on a refactor commit is a regression.

5. For commits that **intentionally** change SVG output (F16, F20, F22): regenerate via `scripts/snapshot-goldens.py <affected>` and **visually inspect each PNG** per CLAUDE.md rule before committing. List affected goldens in the commit message.

**Test-prose updates** (known prior to execution):

- `tests/themes/test_unknown_key_raises.py:11` — `"Unknown Theme key" in msg` → update for F3 serde message.
- `tests/themes/test_binding_roundtrip.py:107` — `"title_anchor must be one of"` → update for F3 enum deserializer message.
- `tests/themes/test_binding_roundtrip.py:114` — `"legend_orient must be one of"` → update for F3 enum deserializer message.

These are updated in the same commit as the F3 structural change. Other prose-asserting tests will be grepped opportunistically before F5 lands.

**After F1c** (largest single change): full pytest, not `-x`, to expose any regression masked by fail-fast.

**After F2b** (scale architecture collapse): `cargo expand` the macros once to confirm the generated code reads as expected; then full pytest.

**After F20** (ratio implementation): expect JointChart goldens to shift because Python's `composition.py:380-401` workaround is being removed in the same/next commit. Re-bless deliberately.

---

## Decisions confirmed by user (2026-05-11)

1. **F16**: stays in this pass. Implemented per the principled spec-line-52
   policy: `EncodingSpec.type_` consultation first, then dtype-driven
   inference where all numeric Arrow dtypes route to continuous. Final
   commit of the pass; goldens re-blessed deliberately.
2. **Branch**: stack on `fix/marks-and-composition-wiring` (user override of
   advisor recommendation; cost acknowledged).
3. **F3 + F5 error-message wording**: marginal prose changes accepted; the
   3 known test assertions are updated in the structural-change commit.
4. **F20 (row/col ratios)**: implement fully with viewBox scaling. Removes
   the Python workaround in `composition.py:379-401`.
5. **F21 (share_x/share_y)**: remove from Rust signature; reimplement at
   the Python layer in a follow-up. Honest cleanup, behaviorally a no-op
   today.
6. **F17 LogScale underflow**: investigation-only commit; if the
   investigation turns up a real fix, it ships in its own commit with
   goldens re-blessed if needed.

---

### Tier 4 — Compositor cleanup (compositor.rs + grid_compose.rs)

Audit (subagent pass, 2026-05-11) found **no parallel-API drift** between
the two files — `grid_compose` correctly imports primitives from
`compositor`. But it surfaced 7 internal findings, two of which are
Phase-9a scar tissue that violates CLAUDE.md's "no defer" rule and one
of which is currently being worked around in Python.

| # | Finding | Severity | Commit(s) |
|---|---|---|---|
| F18 | Per-cell `<g translate>` emission and outer `<svg>` header are duplicated 3× across the two H/V composers and the grid composer. Extract `write_svg_open(out, w, h)` and `write_cell(out, x, y, idx, body, is_first)`. Resolves subagent findings F1+F2+F6. | S3 / HIGH | 1 |
| F19 | `CompositorError::EmptyInput` is overloaded for three distinct grid failures (length mismatch, ratio length mismatch, non-positive ratio sums). Add `LengthMismatch { what, expected, got }` and `InvalidRatios` variants. (Folds into F5's broader error-fragmentation cleanup.) | S2 / HIGH | (shared with F5) |
| F20a | **Implement `row_ratios` / `col_ratios` in Rust.** Currently validated and ignored — see `src/ferrum/composition.py:379-382`, which documents the workaround: JointChart manually resizes marginals to fake ratio behavior. Target: ratio-weighted row/col allocation via per-cell viewBox scaling (`<svg width=W height=H viewBox="0 0 origW origH" preserveAspectRatio="xMidYMid meet">…</svg>`). **Byte-identity expectation at this commit**: Python is *still* pre-sizing cells at ratio-correct dimensions, so the new viewBox transform is identity for those cells → goldens should be unchanged (or trivially different from SVG attribute reorder). Any non-trivial diff is a regression to investigate. | S2 / HIGH | 1 |
| F20b | **Remove the Python workaround.** Drop the `composition.py:380-401` manual marginal resizing now that Rust handles ratios. Python passes native-sized cells; viewBox now actually scales them. **Byte-identity expectation: goldens WILL change.** Regen affected JointChart goldens via `scripts/snapshot-goldens.py` and visually inspect each PNG per CLAUDE.md. | S2 / HIGH | 1 |
| F21 | Remove `share_x` / `share_y` from the Rust signature + PyO3 binding. They're accepted but `#[allow(unused_variables)]` and the right layer for axis-sharing is pre-render (compose_svg_grid sees opaque SVGs, has no coordinate system access). Python caller (`composition.py:412-413`) currently passes them; the Python side already fakes the behavior via `axis(show=False)`. Honest cleanup; follow-up doc captures the proper Python-layer design. | S3 / HIGH | 1 |
| F22 | Grid composer silently top-left-aligns every cell, ignoring the `HorizontalAlign`/`VerticalAlign` enums that the 1D composers support. Either extend grid to accept and honor the alignment, or document grid as deliberately top-left. Recommendation: support both for symmetry with 1D. | S4 / HIGH | 1 |
| F23 | `strip_font_defs` relies on string-search heuristic to locate `<defs>` containing `@font-face`. Cleaner: emit a deterministic marker (`<defs id="ferrum-fontdefs">`) upstream so this becomes a single `replace`. Low priority; opportunistic in same commit as F18. | S4 / MED | (folded with F18) |
| F24 | All-`None` grid emits `<svg width="0" height="0">` rather than raising. Probably benign but inconsistent with 1D `EmptyInput` handling. Either raise `EmptyInput` or document. | S5 / MED | 1 |

---

## Out of scope

- Python source **except** for:
  - test-prose updates (3 known tests; see "Validation strategy" below)
  - removing the F20 workaround in `src/ferrum/composition.py:379-401` after
    F20 lands
  - dropping `share_x` / `share_y` kwargs from JointChart's
    `compose_svg_grid` call in `src/ferrum/composition.py:412-413` after F21
- Build system / Cargo.toml. Already lean.
- `transport.rs` (100 LOC, single-purpose). `diagnostics.rs` (same).
- **Python-layer axis-sharing reimplementation** (the proper home for what
  `share_x`/`share_y` was trying to be). Captured as a follow-up after F21.

---

## Estimated effort

~26 commits across the four tiers:

- **Tier 0** (foundation, 5 commits): F6 arrow_cast, F4 mark-style base,
  F9 prepared-inputs dedupe, F8 PyDict helper, F1a transform macro.
- **Tier 1** (rollouts, 7 commits): F1b/F1c transform macro rollout,
  F10 Mark dispatch, F3 theme serde, F12 axis-scale split, F13 color
  builder, F11 scale-spec collapse.
- **Tier 2** (scale architecture, 3 commits): F2a trait introduction,
  F2b per-variant inner types (split into 2).
- **Tier 3** (errors / latent bugs, 6 commits): F5 typed errors, F7
  encoding inherit, F14 SizeScale collapse, F15 stack-aware position,
  F17 LogScale investigation, F16 color-inference policy (last).
- **Tier 4** (compositor cleanup, 5 commits): F18 DRY extraction (+F23),
  F20 ratio implementation, F20-Python workaround removal,
  F21 share_x/y removal, F22 grid alignment, F24 all-None grid.

Execution discipline: **one commit at a time, run validation, then move
to the next.** On any test failure I stop, diagnose, and report rather
than barreling through. After each tier completes I summarize what
landed and what's next before starting the next tier.
