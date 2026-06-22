# Cohesion Campaign Design Spec

**Date:** 2026-06-21
**Source audit:** `design-docs/superpowers/followups/2026-06-21-cohesion-audit.md` (+ `-findings.md`, 193 findings with stable IDs `CODE-NN`).
**Scope:** Close all 193 cohesion findings by closing their underlying patterns. Behavior-preserving except where a finding *is* a latent bug (Tier 0) or where a public-API vocabulary change is made non-breaking via aliases.

---

## 1. Scope

Drive ferrum to a unified, intentional design by (a) fixing latent bugs surfaced by the audit, (b) finishing unifications already started (adopt extracted helpers everywhere), (c) collapsing dual sources-of-truth to one, (d) unifying public vocabulary via canonical-name-plus-alias, (e) splitting god modules along existing seams, and (f) clearing migration scar tissue. Every one of the 193 finding IDs maps to exactly one plan task (coverage matrix in the plan).

## 2. Goals

- All 10 S4 findings fixed with regression tests; no observable behavior regression elsewhere (goldens byte-stable).
- Each "extracted helper" has zero remaining hand-rolled bypasses among its siblings.
- Each dual-source-of-truth contract has one authority; the other side derives from it or is deleted.
- Each overloaded/competing public name has one canonical spelling; legacy spellings accepted as documented aliases (no breaking change for v0.17.x users).
- `chart.py` and `composition.py` drop below ~1600 LOC each by extracting cohesive modules (pure moves).
- Zero stale `#[allow(dead_code)]` on live code; zero docstring/annotation describing a retired protocol; doctests for `marks/` enabled so the drift cannot reappear.

## 3. Non-goals

- No new user-facing features (missing-feature gaps stay tracked in the archaeology doc; only their *cohesion* symptoms are in scope).
- No breaking public-API changes — every vocabulary change ships canonical + alias.
- Tier 5 (the static-SVG ↔ interactive-WASM dual-emission unification) is **out of scope for this campaign**; it needs its own design pass. Its symptom-level duplications (REND-03/04, RSUP-03) are addressed by shared helpers, not by collapsing the two backends.
- No `cargo test`/golden re-bless without visual inspection (CLAUDE.md hard constraint).

## 4. System behavior (what changes observably)

Only these are intended behavior changes (all Tier 0, each gets a regression test):
- **RMARK-01:** `mark_point(shape="cross"|"vline"|"hline", opacity=p)` now honors `p` (was forced 1.0).
- **XSIB-01 / XFORM-01:** `mark_violin(horizontal=True)` flips orientation (was silently ignored); KDE/smooth/2D-KDE accept Int/UInt/Bool groupby columns (was: raised "must be Utf8").
- **REND-01:** `InteractiveChart.save("x.png")` raises a clear error instead of silently writing HTML into a `.png` file; `save()` no longer swallows `format=`/`scale=`.
- **SEAM-01:** a polars `Duration` column and a pyarrow `Duration` column scale identically (was: divergent axis math).
- **RSUP-02:** a `Date`/`Duration` column is classified numeric consistently across `is_numeric`/`col_as_f64`/`min_max_f64`.

Everything else is behavior-preserving. New alias spellings are additive.

## 5. Architecture (target seams)

**Python `src/ferrum/`:**
- `chart.py` keeps the `Chart` fluent surface + `to_spec` orchestration only. Extract `_desugar.py` (composite-mark expansion + `_resolve_pending` body with a per-kind hook), `_layer_transforms.py` (`_NamedTransform` + per-layer aggregate/bin resolvers + `_transforms_to_json_list*`), and a `SpecBuildMixin`. Primitive `mark_*` may move to a `PrimitiveMarksMixin` to match the existing `Statistical`/`Diagnostic` mixins.
- `composition.py` keeps the composite chart classes + resolve/validate helpers. Extract `_scene_merge.py` (all `_merge_*`/`_offset_*`/`_assemble_*`/`_inject_figure_chrome`), with the four `_merge_child_scenes_*grid` variants unified behind one placement-strategy helper over `_PlacedChild`.
- The two mark mixins move into the `marks/` package (`marks/_chart_methods_{statistical,diagnostic}.py`), eliminating the `_marks_*.py` vs `marks/*` homonym.
- `annotations.py` becomes a thin convenience layer over the `annotation/` dataclass package (one coordinate-coercion implementation, one render path, aligned keyword vocabulary).
- The public model-diagnostics API gets a public defining home (see Decision D-MOD-1).

**Rust `crates/ferrum-core/`:**
- `group_key.rs` gains `group_partition()`; `bin`/`kde`/`kde_2d`/`smooth` all call it.
- One finite-extent helper in `numeric_util.rs` (or `extent.rs`); the ~12 hand-rolled copies call it.
- Mark resolvers (`MarkNodes`, `channels::*`, `OpacityResolver` extended to all opacity channels, `resolve_fill_color`/`stroke`, a completed `FillStroke` builder) adopted by every mark builder.
- One `DASH_PALETTE` const drives both dash maps.
- One `supported_numeric_dtypes()` predicate backs `is_numeric`/`col_as_f64`/`min_max_f64`.
- Layout: a shared measure-band/carve-strip core for x/y axes and the three legend kinds; `AxisLayout`/`LegendLayout` built by one constructor each; `compute_layout` decomposed into its five reservation stages; one `AxisStyleOverrides::fill_from` merge over `AxisStyleSpec`.
- Spine: `resolve_panel_scales()` seam; x/y axis-input resolution looped over a channel; `prepare_render_inputs`/`build_scene` decomposed; `fix_transform_extents_for_facet` + facet partition collapsed to one generic helper each.
- Scales: continuous scales become named-field structs (kills positional-tuple `.3` ambiguity); one canonical `domain_user_set`/`range_user_set`/`utc` representation.

**Rust `crates/ferrum-wasm/`:**
- One typed circle-vs-rect representation; one packed-tooltip parser; one field-value-membership predicate; `ConditionalEncoding` carries channel xor value (not both); dead `wasm_bindgen` methods removed or wired.

## 6. Canonical interfaces / data contracts (the decisions)

These bind multiple tasks. **API-affecting decisions are flagged ⚠️ and require user confirmation before execution.**

- **D-COLOR-1 ⚠️ (ENC-06, XNAME-02, XSIB-07):** Canonical name for a named color set = **`scheme`**. `palette` and `cmap` accepted as aliases, normalized to `scheme` at channel construction. Rust palette registry is the single source of truth; expose `list_palettes()`/`palette_kind(name)`; `color.py` consumes it (no hand-mirrored hex); `scheme=` validated at declaration time with the `palette()` error shape.
- **D-ORIENT-1 ⚠️ (XSIB-01, XSIB-02, RSUP-04):** Canonical orientation kwarg = **`orient`** (`"vertical"`/`"horizontal"`) across all marks. Implement `mark_violin` flip through it. Legacy `horizontal=`/per-mark spellings accepted as aliases.
- **D-FMT-1 ⚠️ (ENC-03):** Canonical = **`format_type`** (snake_case). `formatType` accepted as a Vega-compat alias normalized in `ChannelBase`.
- **D-ASNAME-1 (SEAM-03, ENC-05):** Canonical output-column kwarg = **`as_`** in all `transform_*` wrappers; the inner-dict Vega wire spelling stays `'as'`, documented once as the wire form.
- **D-STACK-1 (ENC-02, ENC-07):** One `STACK_OFFSETS` frozenset + `_validate_stack_offset(value, *, where)` in `position.py`; encoding layers its bool/falsy normalization on top.
- **D-HONORED-1 (ENC-01, ENC-04, XSIB-03, XSIB-05):** A channel's own `_honored_kwargs` is the single truth; `to_encoding_spec_dict` iterates it (with a per-key handler registry). Delete the dead `_RENDERED_HONORED` alias; one honored-kwarg vocabulary module.
- **D-EXTENT-1 ⚠️ (XNAME-01):** Keep **`extent`** = data domain only. Rename the other three uses: layout band → `band`, whisker multiplier → `whisker_mult`, aggregate method → `method`. (Public kwarg renames ship with aliases.)
- **D-PANEL-1 (XNAME-04):** Canonical rendered-sub-region term = **`panel`** (internal/comment/identifier alignment; no public kwarg today, so non-breaking).
- **D-CHROME-1 (XNAME-05):** Canonical figure-level-band term = **`chrome`** (rename stray "header" uses).
- **D-FIRSTPARAM-1 ⚠️ (PLOT-02):** Model-diagnostic family first param = **`model`**; seaborn family = **`data`**; `rank*` = **`data_or_source`** (spec-aligned); fix clustering's intra-family `X`-vs-`model` split. Positional callers unaffected; keyword callers get aliases where a name changes.
- **D-COMPARE-1 ⚠️ (PLOT-01, XSIB-08):** Add `compare: dict|None=None` (+ `random_state`) to every model-diagnostic public function, forwarding to the existing `_resolve_source(..., compare=)`. Document exclusions explicitly.
- **D-THEME-1 (THEME-01, THEME-02, THEME-05, THEME-07):** Rust `ThemeOverridesSpec` is the single key contract. Generate the Python `_KNOWN_KEYS`/`_COLOR_KEYS` from one shared manifest (or from the Rust struct) so they cannot drift; resolve the per-level-grid-key and `font_size` naming drift; complete the `_FALLBACKS` chain.
- **D-SCALE-1 (SPEC-01, SPEC-02, SPEC-03, SPEC-06, SEAM-08, SPEC-07):** Continuous PyO3 scales become named-field structs. One representation of `domain_user_set`/`range_user_set`/`utc`; `TimeScale` gains its own `domain_user_set`; `domain()` getter return shape uniform (`Option<Vec<f64>>`); delete `BandScale.domain_set`.
- **D-GROUPBY-1 (XFORM-03):** One canonical `groupby` field shape across the transform family (the `Vec<String>` form used by the migrated transforms); secondary outputs preserve group dtype.
- **D-DTYPE-1 (SEAM-01, RSUP-02):** One boundary dtype-normalization function (fold `_sanitize_for_rust` into `_coerce`), applying the Duration rule uniformly to polars and pyarrow; one `supported_numeric_dtypes()` predicate in Rust. Decide Duration's canonical form: **cast to ms-like temporal** on both frontends (matches Date handling), so `arrow_cast` Duration arms become the documented single owner or are removed as unreachable.
- **D-MOD-1 ⚠️ (MOD-02, MOD-01):** The public model-diagnostics classes get a public defining home. **Chosen approach: rename `_diagnostics` → public `diagnostics/`** (`sources/`, `visualizers/`), heavy sklearn-boundary internals under `diagnostics/_internal/`. Removes the `gen_api_pages` special-cases and makes the `_`-private convention mean one thing repo-wide. (Changes `__module__` paths; acceptable for a 0.x library, shipped in the changelog.)
- **D-MARKRESULT-1 (MARK-01, MARK-02, MARK-08, XDEAD-04):** `MarkDesugarResult` is the sole return type: all 36 desugars annotated `-> MarkDesugarResult`; all "Returns" docstrings + doctests rewritten to the dataclass shape (template = `desugar_contour`); a constructor/factory enforces the layers-xor-mark mutual exclusion; enable scoped `--doctest-modules src/ferrum/marks`.
- **D-XDEAD-03 (XDEAD-03, [API], user-decided "wire z, drop curve"):** The annotation `z` flag (Text-only, `"above_marks"`/`"below_marks"`) is **wired**, not dropped; the `curve` flag (Arrow-only, dead) is **dropped** from the Python API and the Rust struct. Wiring mechanism: `build_annotations` partitions resolved nodes into `{below_marks, above_marks}` by each Text spec's `z`; `below_marks` route into the panel **`grid`** slot (the pre-marks "below bucket"), `above_marks` stay in the post-marks **`annotations`** slot — the exact mirror of how an above-marks grid/axis (zindex ≥ 1) already routes into the `annotations` bucket. Rust `default_z` aligns to `"above_marks"` (was `"front"`, the vocabulary mismatch the finding flagged). No new `Panel` field: `z` is Text-only, the SVG grid emitter falls through to `emit_node` for non-Line nodes, and WASM `collect_static` handles Text — so a dedicated slot would add a field to 66 `Panel` literals + change the serialized scene shape for zero behavioral gain. Existing goldens stay byte-identical (no existing annotation uses `below_marks`; the flag was dead). Dropping `curve`: only the low-level `ann.arrow()` primitive carried it (the high-level `annotate_arrow` composes `mark_segment` and never used it), and Rust never read it, so removal is a pure dead-flag deletion with no render change.

## 7. Invariants and constraints

- **Hard constraints (CLAUDE.md):** no matplotlib; no new global mutable state; `ferrum-spec.md` stays the contract (update with dated notes where vocabulary changes); `cargo test` green before any phase done; goldens visually inspected before commit; no `git push` without explicit ask; no Claude authorship trailer.
- **Backward compatibility:** every public vocabulary change is additive (canonical + alias). The one accepted module-path change (D-MOD-1) is documented in the changelog.
- **Byte-stability:** non-Tier-0 tasks must keep goldens byte-identical (`uv run pytest -n auto` green; affected golden dirs re-rasterized + visually inspected only if intentionally regenerated).
- **Per-task discipline (CLAUDE.md + subagent-driven-development):** every code task → `python-coder`/`rust-coder`; three review gates (spec-compliance → quality → review-lite); `/regression-test` after every Tier-0 fix; verify subagent reports independently.

## 8. Key decisions and tradeoffs

- **Canonical + alias over breaking renames:** ferrum-viz 0.17.1 is on PyPI; aliases keep existing user code working while unifying the surface. Cost: a one-release alias-maintenance burden, documented for later removal at 1.0.
- **Derive-don't-duplicate for dual sources:** prefer generating one side from the other (theme keys, palette names) over "keep in sync" tests. A sync test still permits drift between releases; derivation makes drift unrepresentable.
- **Pure-move module splits behind goldens:** god-module splits are import-only relocations; goldens are the regression oracle. Done one module at a time to keep each diff reviewable.
- **Tier 5 deferred deliberately, not dropped:** collapsing the SVG/WASM dual emission is a real architectural change with golden-regen risk; it gets its own spec. This is not a no-defer violation — it is scoping a genuinely separate design effort, and its *cohesion symptoms* are still addressed here via shared helpers.
- **D-MOD-1 rename vs re-export:** rename chosen over a re-export shim because a shim leaves `__module__` pointing at the private path (the actual finding); the rename fixes the root cause.

## 9. Acceptance criteria

- `uv run nox` green (lint + full pytest + cargo test + build + docs).
- `cargo test` green; `cargo clippy -D warnings` clean on every touched crate.
- Goldens byte-identical except intentionally-regenerated dirs (each visually inspected per CLAUDE.md).
- A coverage check confirms every one of the 193 finding IDs is closed by a committed task.
- Tier-0 regression tests present and passing; each would fail on the pre-fix code.
- Heavyweight `python-review` + `rust-review` on the final branch report no residual instance of the six patterns in the touched subsystems and no newly-introduced cohesion regression.

## 10. Validation strategy

- Per task: `python-coder`/`rust-coder` runs the task's targeted tests; spec-compliance + quality + review-lite gates; orchestrator independently re-runs the test command and inspects any regenerated golden PNG.
- Per tier: run the matching heavyweight review skill on the subsystems that tier modified (CLAUDE.md escalation trigger #1).
- Final (step 4): full `nox`, then `python-review` + `rust-review` over the whole branch diff scoped to the touched families, plus a `bug-hunt`/`test-sweep` pass to confirm no new defects, plus the coverage-matrix check.

## 11. Open questions

The ⚠️ decisions in §6 (D-COLOR-1, D-ORIENT-1, D-FMT-1, D-EXTENT-1, D-FIRSTPARAM-1, D-COMPARE-1, D-MOD-1) are the user-confirmation gate. All have a recommended resolution above; execution starts once they are confirmed or overridden.
