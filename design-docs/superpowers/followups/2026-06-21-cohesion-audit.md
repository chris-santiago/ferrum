# Cohesion Audit — Consolidated Recommendations

**Date:** 2026-06-21
**Method:** 20 parallel cohesion/architecture reviewers (8 Python subsystems, 7 Rust subsystems, 5 cross-cutting dimensions) applying the `python-review`/`rust-review` lens. 193 findings (0 S5, 10 S4, 77 S3, 94 S2, 12 S1).
**Full findings:** `2026-06-21-cohesion-audit-findings.md` (machine-generated digest, all 193, severity-sorted, with file:line + concrete fix per finding).
**Goal:** push ferrum toward a unified, elegant, intentional design — code that reads as if composed by one mind.

---

## Verdict

ferrum is **fundamentally well-architected.** The reviewers independently confirmed the load-bearing decisions are sound and held: the Python-declares / Rust-computes boundary, the three-crate split (`ferrum-scene` is a clean acyclic serde leaf), the serde-round-trip spec contract, the `for_each_transform!` macro dispatch, the `MarkNodes` accumulator, the theme cascade resolved in one place, `arrow_cast`/`format`/`compositor` as self-aware single-source modules. There is **no critical correctness or API hazard (0 S5)** and **no real dead code** — the named archaeology targets were genuinely removed.

The drift is **concentrated, not pervasive, and it has one signature shape.** Across both languages, the same story repeats: *a good unifying abstraction was introduced, but only half its callers adopted it.* The remaining gap between this codebase and "feels designed by one mind" is almost entirely **finishing unifications that were started** and **collapsing duplicated sources-of-truth that are currently kept in sync by hand-discipline.** Very little requires new design. Most of the highest-leverage work is *deletion*.

---

## The six meta-patterns (the diagnosis)

Every one of the 193 findings is an instance of one of these. Fixing the *patterns* — not the instances — is what produces a unified design.

### 1. Incomplete unification — "the helper exists; half the siblings ignore it" (the dominant pattern)
A shared helper/type was extracted to kill drift, then only some siblings migrated. The others still hand-roll the exact logic the helper encapsulates, and they have *already* diverged.
- **Rust transforms:** `group_key.rs` is the single source of truth for groupby keying; `bin`/`density_data` migrated, but `kde`/`kde_2d`/`smooth` still hand-roll Utf8-only grouping → a grouped histogram accepts an Int64/Boolean group column but the identical KDE/smooth *raises* (S4, FA-7 left half-done).
- **Rust marks:** `MarkNodes` + the extracted `channels::*`/`OpacityResolver`/`resolve_fill_color` resolvers are adopted by 12/15 builders; `point`/`rect`/`arc` still hand-maintain parallel vectors and inline color/opacity/stroke resolution.
- **Python composition:** `_validate_resolve` is called by `_CompositeBase`/`ConcatChart`; `RepeatChart`/`LayerChart` inline a copy with a drifted error message.
- **Python diagnostics:** `_curve_frames` is "exactly one implementation per concept" — except `gain`/`lift`, hand-written in 4 places (`ClassificationCurvesMixin` + `_PrecomputedSource` ×2 branches).
- **Python render:** the canonical `figure_title_text()` accessor is honored by 1 of its 3 intended callers.
- **Rust render-support:** the stroke-dash palette is encoded 3 times (SVG forward map, GPU inverse map, WASM shader) as hand-maintained mutual inverses (S4).

### 2. Two sources of truth for one contract — "correctness by discipline, not by construction"
The same invariant is asserted in two+ places and kept aligned by hand. Several have *already* drifted.
- **Theme keys:** Python `Theme._KNOWN_KEYS` frozenset vs Rust `ThemeOverridesSpec` (`deny_unknown_fields`) — already drifted on per-level grid keys; plus a third `_COLOR_KEYS` list and a Python-only `_FALLBACKS` table.
- **Palettes:** `color.py` hand-mirrors the entire Rust palette registry as literal hex tables and reimplements interpolation; `scheme=` is validated only by Rust at render time, so the two name universes drift silently.
- **Honored-kwargs:** each channel's `_honored_kwargs` (drives the warn guard) vs a hard-coded key tuple in `to_encoding_spec_dict` (drives serialization) — already disagree (a kwarg can work-but-not-warn or warn-but-work).
- **Stack-offset validity:** defined 3× (`encoding/positional.py`, `transforms.py`, `position.py`) with 3 memberships and 3 error formats.
- **Axis style:** two ~28-field `AxisStyleSpec → AxisStyleOverrides` mappers (per-channel fresh-build vs chart-level fill-if-None) over the identical struct (S4).
- **Dtype normalization:** what crosses the CDI boundary is decided in `_coerce.py` (2 branches), `_render.py::_sanitize_for_rust`, and `arrow_cast.rs` — with a polars-vs-pyarrow Duration asymmetry that is a latent axis-scaling bug (S4).

### 3. Sibling drift across API families — parallel members written by different hands
- **Figure functions:** first-positional param named 4 ways (`model`/`source`/`data`/`X`); `compare=` exposed on 1 of 7 model-diagnostic modules though all route a bare dict; two parallel ROC/PR/calibration annotation paths with divergent defaults.
- **Encoding channels:** `format_type` (snake) vs `formatType` (camel) on different siblings; `_RENDERED_HONORED` is two unrelated constants with the same name in sibling files.
- **Marks:** orientation spelled 3 ways; `mark_violin` documents a `horizontal` param it silently drops (S4).
- **Scales:** `TimeScale` repurposes tuple slot `.3` as `utc` while every sibling uses it as `domain_user_set` (S4) — and consequently can't suppress default padding.
- **Visualizers:** `has_score` flag is False on 4+ visualizers whose `score()` actually works.
- **Point shapes:** Cross/VLine/HLine hardcode opacity `1.0`, ignoring the opacity channel the other 6 shapes honor (S4).

### 4. God modules and god functions — mixed abstraction levels in one place
- **`chart.py` (3953 LOC, 0 tests):** fuses the fluent builder, composite desugar (`_resolve_pending`, a 265-line per-kind god-method), layered-transform routing, and spec assembly. The stat/diagnostic marks were *already* extracted to mixins — proving the pattern is sanctioned; the densest layers were left behind.
- **`composition.py` (2988 LOC):** the declarative composite classes AND ~900 lines of low-level scene/packed-byte merge plumbing in one file.
- **Rust:** `prepare/mod.rs` (~1580 code LOC) orchestrates *and* inlines x/y axis resolution 4×; `compute_layout` is a 615-line function mixing 5 reservation stages. (Note: many "huge" Rust files — `scene_load.rs` 4451, `bar.rs` 2077 — are 80%+ tests and are *not* god modules. The reviewers correctly distinguished these.)

### 5. Vocabulary drift — the same concept wears different names
This is half of "feels authored by one mind," and it leaks onto the **public API**:
- **`extent`** is overloaded to mean four unrelated things (data domain, pixel layout band, whisker multiplier, aggregation method) — including on public kwargs.
- **Named color set** has three competing public names: `scheme` / `palette` / `cmap`, plus a duplicated registry.
- **Multi-plot sub-region** is `panel` (Rust layout) vs `cell` (Python composition) vs `facet` (the operation).
- **Value-class idiom:** `position.py` hand-rolls `__slots__`/`__eq__`/`__hash__`/`__setattr__` (~60 lines/class) while sibling value classes in `selection.py`/`configure.py` use `@dataclass(frozen=True)`.

### 6. Migration scar tissue — docs/annotations frozen at a pre-refactor shape
- **`MarkDesugarResult`:** the tuple→dataclass migration updated bodies but not contracts — 24 desugars annotated `-> tuple` (false), every "Returns" docstring + doctest still describes the dead N-tuple protocol (the doctests would `TypeError` if ever run; they survive only because `--doctest-modules` is never set).
- **Stale `#[allow(dead_code)]`:** a family of suppressions carry "future integration / Tasks 3-6 will use" comments for migrations the archaeology doc records as *complete* — the suppression now sits on live code, lying to the next reader.
- **Dead aliases:** the `_RENDERED_HONORED` "back-compat alias" is imported nowhere.

> **The architectural root cause of patterns 1–2 in the render layer:** ferrum maintains **two emission backends** — static SVG and interactive scene/WASM — that encode the same styling/geometry/empty-scene/save-dispatch/dash-palette logic twice, kept aligned by "mirrors the other path" comments rather than shared code. This single fork is the engine behind a large share of the duplication findings. It is the deepest item and the one genuinely worth a design pass (Tier 5).

---

## Prioritized roadmap (by leverage ÷ risk)

### Tier 0 — Latent bugs hiding inside cohesion issues (fix now; tiny, each needs a regression test)
These are the S4s that are *observable wrong behavior*, not just smells:
1. **Point opacity** — Cross/VLine/HLine ignore the opacity channel (`render/marks/point.rs`). Pass resolved opacity into the `to_scene_stroke` calls.
2. **`mark_violin(horizontal=...)`** — documented, silently dropped. Implement the flip (boxplot shows the pattern) or raise.
3. **`InteractiveChart.save('x.png')`** — writes HTML to any extension, silently swallows `format=`/`scale=`. Honor the extension or raise.
4. **Duration dtype asymmetry** — polars Duration → Int64 (raw), pyarrow Duration → typed; same data scales differently by frontend. Apply one rule in both `_coerce` branches.
5. **`arrow_cast::is_numeric`** disagrees with `col_as_f64` on Date/Duration → a Date column parses to f64 but routes categorical. One `supported_numeric_dtypes()` predicate all three branch from.

### Tier 1 — Finish the unifications already started (highest ROI; behavior-preserving, mostly deletion, byte-stable)
This tier alone moves the "single mind" needle the most, at near-zero risk. Each is "adopt the helper that already exists":
- Migrate `kde`/`kde_2d`/`smooth` to `group_key.rs` (extract `group_partition()`; closes the int/bool-groupby gap + smooth's string-only output).
- Migrate `point`/`rect`/`arc` to the extracted mark resolvers (`OpacityResolver`, `resolve_fill_color`/`stroke`, `channels::*`).
- Route `RepeatChart`/`LayerChart` through `_validate_resolve`.
- Add `gain_frame`/`lift_frame` to `_curve_frames`; delete the 4 hand-written copies.
- Route all 3 HTML/title callers through `figure_title_text()` + one `html_string()` helper.
- One `DASH_PALETTE` const; derive both Rust dash maps from it.
- Return `_empty_scene()` (not the hand-typed literal) from all 4 `_merge_child_scenes*` early-returns.
- Move `_merge_layers` from `regression.py` to `_helpers.py`; route plots' DataFrame coercion through `_coerce.to_arrow_table`.
- Convert `position.py` value classes to `@dataclass(frozen=True)` (−150 lines, joins the sibling idiom).

### Tier 2 — Collapse the dual sources of truth (high payoff; removes the latent-drift engine; "by construction not discipline")
- **Theme keys:** one contract. Generate the Python known-key set from the Rust `ThemeOverridesSpec` (or vice versa) so they cannot drift.
- **Palettes:** Rust owns the registry; expose `list_palettes()`/`palette_kind()`; `color.py` consumes it; validate `scheme=` at declaration time with the same error shape as `palette()`.
- **Honored-kwargs:** make `to_encoding_spec_dict` iterate the channel's own `_honored_kwargs` (with a per-key handler registry) — honored set becomes the single truth.
- **Stack offsets:** one `STACK_OFFSETS` + one `_validate_stack_offset(where=)`.
- **Axis style:** one `AxisStyleOverrides::fill_from(spec, fill_only_if_none)` over the shared struct.
- **Dtype normalization:** one boundary-normalization function both render paths call.
- **Scales:** convert the 6 continuous scales from positional tuples (`.3` = `utc`-or-`domain_user_set`-by-comment) to named-field structs.

### Tier 3 — Unify the public vocabulary (some API aliasing; do with a deprecation/alias shim, not a break)
- Canonicalize **`scheme`** for color sets; accept `palette`/`cmap` as documented aliases in one place.
- Disambiguate **`extent`** — rename the non-data-domain uses (layout `band`, whisker `whisker_mult`, agg `method`); keep `extent` for the data-domain meaning only.
- Canonical **`orient`** for orientation across all marks; implement violin's flip through it.
- Canonical **`format_type`** (snake); `formatType` as a Vega-compat alias normalized in `ChannelBase`.
- Canonical first-param **`model`** for the model-diagnostic family (`data` for seaborn family); fix clustering's intra-module `X`-vs-`model` split; add `compare=` to all 7 modules.
- Pick **`panel`** as the canonical rendered-sub-region term repo-wide.

### Tier 4 — Structural splits (larger; do behind goldens, one move at a time)
- **`chart.py`** → extract `_desugar.py` (composite expansion + `_resolve_pending` body, with a per-kind hook so the 265-line god-method becomes ~60 lines of uniform orchestration), `_layer_transforms.py`, and a `SpecBuildMixin`. Target ~1500 LOC.
- **`composition.py`** → extract `_scene_merge.py` (all `_merge_*`/`_offset_*`/`_assemble_*`); unify the 4 `_merge_child_scenes_*grid` variants behind one placement-strategy helper.
- **Rust render spine:** one `resolve_panel_scales()` seam (kills build_scene's per-panel re-resolution duplication); decompose `compute_layout` 615-line god-fn; abstract x/y axis-input assembly over a channel.
- **`_diagnostics` public-home fix:** the entire public model-diagnostics API (`ModelSource` + 29 `*Visualizer`) is homed under a `_`-private package, violating the convention everywhere else and forcing `gen_api_pages` special-cases. Promote to a public `ferrum/diagnostics/` (heavy internals under `diagnostics/_internal/`).
- **Diagnostic taxonomy:** the classification/regression/… partition is replicated across 3 package roots with 3 naming conventions; align the file-prefix convention and document the canonical 4-piece mapping.

### Tier 5 — The dual-emission fork (deepest; needs a design pass, not a patch)
Static-SVG vs interactive-scene/WASM emit the same logic twice. Worth a `brainstorming` session on whether the SVG path can derive from the scene graph (one backend) rather than being a parallel implementation. This is the root of patterns 1–2 in render and the source of the W4/W5 interactive feature-forks. Don't attempt opportunistically — scope it deliberately.

### Tier 6 — Scar-tissue sweep (mechanical; pairs well with enabling doctests)
- Rewrite the 24 `-> tuple` annotations + all `MarkDesugarResult` docstrings/doctests to the dataclass shape; enable `--doctest-modules src/ferrum/marks` so it can't reappear.
- Audit every `#[allow(dead_code)]` whose comment references a now-complete migration; remove the suppression (the code is live) or the code (if truly dead).
- Delete the dead `_RENDERED_HONORED` alias.

---

## Suggested sequencing

**Tier 0 → Tier 1 → Tier 6** is the recommended first campaign: it ships the latent-bug fixes, completes the started unifications, and clears the scar tissue — all behavior-preserving (Tier 0 excepted, which gets regression tests), all byte-stable against goldens, and all *deletion-heavy*. That single campaign closes the majority of the 193 findings by closing their underlying patterns, and it is exactly the work that makes the codebase read as intentional. Tiers 2–4 are the medium-risk follow-ups; Tier 5 is a standalone design effort.

Each tier maps to a heavyweight-review-friendly slice (per CLAUDE.md escalation triggers) and to existing archaeology items (FA-7/11/12/13 are Tier 1; the palette library is Tier 2; `_diagnostics` homing is Tier 4).

---

## How this was produced / how to regenerate

20-agent `cohesion-audit` workflow (run `wf_d46086ec-f0e`), 2.5M analysis tokens. Each reviewer read CLAUDE.md + ARCHITECTURE.md + the archaeology open-items, then its assigned scope, and returned structured findings (severity S1–S5, confidence, file:line, category, concrete fix, `already_tracked`). The full per-finding digest is the companion file `2026-06-21-cohesion-audit-findings.md`.
