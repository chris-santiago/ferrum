# Cohesion Roadmap Design Spec (review findings C1–C14, non-issue subset)

> Source: the consolidated full python-review + rust-review (2026-06-20). Addresses every finding NOT already a GH issue. Excluded as already-filed: C4 (=#12 FA-12), EncodingSpec deny_unknown_fields (=#2), build_color_detail_groups (=#11), per-panel legend (=#16), dead WASM API (=KG-5), legend label_color (=#4). C1 addresses only its non-#11 parts.
>
> Branched from `fix/bugs-5-9-10` (C6/C7 build on the #5 OpacityResolver). Assumes that branch's code is present.

## 1. Scope

Three work packages of cohesion refactors, all behavior-preserving (byte-identical render output unless a finding is an explicit bug fix):
- **WP-A — quick wins** (byte-identical, low-risk): C6, C7, C8, C12.
- **WP-B — typed-modeling + dedup** (behavior-preserving): C5, C9, C10, C11, C13.
- **WP-C — architectural decomposition** (behavior-preserving, high golden/test blast radius): C2, C3, C14, C1.

## 2. Goals

Per finding (the contract each must satisfy):

**WP-A**
- **C6** — `build_size_scale`/`build_opacity_scale` return `(Option<…>, Vec<RenderWarning>)`, matching `build_color_scale`/`build_shape_scale`; `build_auxiliary_scales` collects all four via `warnings.extend`. (Closes the return-type asymmetry KG-8 introduced.)
- **C7** — `tick`/`segment`/`rule` mark builders resolve opacity via the shared `OpacityResolver` instead of hand-rolled per-channel logic.
- **C8** — one `encode_serde_value_for_py(py, &Option<serde_json::Value>) -> PyResult<Option<Py<PyAny>>>` helper; the `condition`/`sort`/`impute`/`legend`/`axis` getters in `encoding.rs` call it.
- **C12** — a Python-side regression test asserts the packed stride constants (`_PACKED_INSTANCE_SIZES = {0:64, 1:72}`) match the bytes the Rust producer actually emits (build a 1-mark chart, parse the packed header/stride), so the hand-synced mirror fails loud on drift.

**WP-B**
- **C5** — encoding appearance channels share one declared "honored kwargs" contract (a base/grouping that makes Color/Shape/Fill/Stroke/Size/Opacity/FillOpacity/StrokeOpacity/StrokeWidth/StrokeDash/Angle consistent); `scheme`/`sort` membership is intentional per channel, not accidental.
- **C9** — Rust: one `color_column_loader(ctx, field) -> (categorical, numeric)` reused by point/bar/rect; one `polar_channel_resolver` reused by arc/bar.
- **C10** — `_Facet` becomes type-safe: either two frozen subclasses (`_FacetWrap{field, wrap_orient, ncols, nrows}` / `_FacetGrid{row, col, ncols, nrows}`) or `wrap_orient` becomes an enum; field-validity is no longer mode-dependent-by-convention.
- **C11** — `OpacityResolver.general_fallback: bool` becomes a small enum (`OpacityFallback::{Standard, BarLike}`); related transform-spec bool clusters are grouped where it clarifies (no behavior change).
- **C13** — silently-dropped encoding kwargs fail loud or warn-once: `stroke` dropped when `color` present (warn-once); ghost positional channels (`X2`/`XError`/…) docstrings match actual `_honored_kwargs` (no "reserved for future use" that actually warns).

**WP-C**
- **C2** — `chart.py` (4041 lines) sheds cohesive units to new modules (e.g. `_facet.py` for `_Facet` + facet routing + `_to_facet_dict`; `_transform_resolve.py` for transform/encoding resolution) with the public `Chart` API unchanged.
- **C3** — the 6 symmetric composite classes (`HConcat`/`VConcat`/`Concat` + the merge variants) collapse toward a layout-strategy: one path parameterized by layout rather than ~70%-duplicated per-class `_render_interactive`/`to_svg`/`_rebuild_with_charts`. Public composite API unchanged.
- **C14** — `prepare.rs` sheds cohesive units (legend/colorbar build, extent-pin, conditional-color) into submodules; `resolve_conditional_color_domain` generalizes to `resolve_conditional_field_names(spec, channel)`.
- **C1** — the figure-function/visualizer family (classification/explanation/clustering/regression/model_selection) routes annotation + legend-suppression + facet-decision + subtitle through shared helpers, eliminating the per-domain drift (metric-label 2-patterns, missing `subtitle=`, scattered facet-decision, direct-label-endpoint ×3, encode-vs-mark legend). The `build_color_detail_groups` sub-part is out of scope (=#11).

## 3. Non-goals

- No public-API changes (Python `Chart`/figure-function/composite signatures and Rust public binding signatures stay stable). All splits are internal module reorganizations.
- No behavior change except the explicit bug-adjacent fixes (C13 warn-once on dropped `stroke`). Render output byte-identical elsewhere — goldens are the oracle.
- Excluded findings (already filed): C4/#12, #2, #4, #11, #16, KG-5 — not addressed here.
- The #5 opacity-composition-semantics question (separate-multiplier vs fallback) remains deferred (its own follow-up); C11 only types the existing flag, it does not change semantics.

## 4. System behavior

Unchanged and observable-equivalent for every chart, except: C13 emits a one-time `UserWarning` when `stroke` is silently dropped (previously no signal). All render output (SVG bytes, packed bytes, interactive) is byte-identical; the full golden + cargo + pytest suites are the binding gate.

## 5. Architecture

- **WP-A/WP-B** are local refactors within existing modules + small new shared helpers.
- **WP-C** is module decomposition. The principle: extract *cohesive units* behind unchanged public surfaces. Target seams:
  - `chart.py` → `_facet.py` (the `_Facet` value type, `facet()` routing helpers, `_to_facet_dict`), `_transform_resolve.py` (transform/encoding spec assembly). `Chart` re-exports/delegates; imports updated.
  - `composition.py` → a layout-strategy: a single composite rendering path that takes a `LayoutStrategy` (horizontal/vertical/wrapping-grid/sparse-grid/nonuniform-grid) instead of one class per layout; the value classes become thin constructors over it. `_PlacedChild`/`_assemble_placed_children` (already extracted) is the shared tail.
  - `prepare.rs` → `prepare/legend.rs` (legend + colorbar build, conditional-color), `prepare/extent.rs` (facet transform-extent pin), keeping `prepare.rs` as the orchestrator.
  - figure family → shared `_charts/_annotate.py`-style helpers for metric-labels, direct-label endpoints, facet-decision, subtitle threading; the `*_chart`/visualizer functions call them.

## 6. Canonical interfaces / data contracts

```rust
// C6: uniform aux-scale return
pub fn build_size_scale(...) -> Result<(Option<SizeScale>, Vec<RenderWarning>), RenderError>;
pub fn build_opacity_scale(...) -> Result<(Option<OpacityScale>, Vec<RenderWarning>), RenderError>;
// build_auxiliary_scales: warnings.extend(size_warns); warnings.extend(opacity_warns);

// C8: one serde→py getter helper (encoding.rs)
fn encode_serde_value_for_py(py: Python, v: &Option<serde_json::Value>) -> PyResult<Option<Py<PyAny>>>;

// C11: typed opacity fallback
enum OpacityFallback { Standard, BarLike }      // replaces general_fallback: bool

// C9: shared loaders
fn color_column_loader<'a>(ctx: &'a DrawCtx, field: &str) -> (Option<Vec<Option<String>>>, Option<Vec<Option<f64>>>);
fn polar_channel_resolver(enc: &Encoding) -> PolarChannels;  // theta→angle, radius→radial + scales

// C14: generalized conditional resolver (prepare)
fn resolve_conditional_field_names(spec: &ChartSpec, channel: ChannelName) -> Vec<String>;
// resolve_conditional_color_domain = resolve_conditional_field_names(spec, Color) + distinct_values_in_order
```

```python
# C10: type-safe facet (one of)
@dataclass(frozen=True)
class _FacetWrap:  field: str; wrap_orient: str | None; ncols: int | None; nrows: int | None
@dataclass(frozen=True)
class _FacetGrid:  row: str | None; col: str | None; ncols: int | None; nrows: int | None
# OR keep _Facet but make wrap_orient an enum and validate field-presence by mode at construction.

# C5: appearance-channel honored-kwargs contract — one source of truth, e.g.
#   class _AppearanceChannel(ChannelBase): _honored_base = frozenset({"type","scale","title","legend"})
#   subclasses extend with their genuine extras ({"sort"}, {"scheme"}, {"sort","scheme"}) explicitly.
```

Public surfaces unchanged: `Chart`, all `*_chart`/visualizers, all composite classes, all PyO3 class/function signatures.

## 7. Invariants and constraints

- **Byte-identical render output** for every refactor (WP-A/WP-B/WP-C), proven by the golden + cargo + pytest suites. Only C13's warn-once is an intended new behavior.
- **No public-API change.** Each WP-C split must keep imports working (`from ferrum import …` and internal call sites) and the `__all__`/re-export surface intact.
- **C6** size/opacity emit empty `Vec` (they have no warnings today) — purely a signature alignment, output unchanged.
- **C7** the resolver outputs must equal tick/segment/rule's current opacity logic exactly (byte-identical).
- **C10/C11** typed remodeling must serialize to the identical JSON/spec the current stringly/bool forms produce.
- **C12** the new assertion must be a real discriminator (a deliberately-wrong stride fails it).
- Goldens regenerated by any WP-C work are visually inspected before commit (orchestrator blesses PNGs).
- No matplotlib; no global mutable state; `cargo test` must pass.

## 8. Key decisions and tradeoffs

- **Staged by risk, not deferred.** Every finding is addressed; WP-A/B are low-risk and land first, WP-C (god-module splits + figure-family) lands as discrete, individually-reviewable refactors because each has a large golden/test blast radius. Staging is execution-ordering, not scope reduction.
- **Refactors preserve behavior.** The splits move code, not semantics; the binding proof is byte-identical goldens. This is why each WP-C item is its own task with a full-suite gate.
- **C10/C11 prefer types over strings/bools** only where it removes a real invalid-state surface (facet field-validity; bar's opacity quirk); transform-spec bool grouping (C11) is applied only where it clarifies, to avoid churn.
- **C1 unifies via shared helpers, not a framework** — extract the repeated annotation/legend/facet logic into plain helpers the figure functions call; no base-class hierarchy imposed (avoids speculative architecture).
- **C2/C3/C14 target cohesive seams** identified by the review (facet, transform-resolve; layout-strategy; legend/extent), not line-count-driven splits.

## 9. Acceptance criteria

- Every in-scope finding (C1,C2,C3,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14) has a landed change satisfying its §2 goal.
- Full suite green throughout: `cargo test -p ferrum-core` + `-p ferrum-wasm` + full pytest; goldens byte-identical (except WP-C goldens regenerated + blessed, if any) ; `ruff`/`clippy` add no new lints.
- Public API unchanged (a stub/signature parity check + `from ferrum import *` smoke).
- C12 assertion fails on a deliberately-wrong stride; C13 warn-once fires on dropped `stroke` and the ghost-channel docstrings match `_honored_kwargs`.

## 10. Validation strategy

WP-A/WP-B: fail-before/pass-after unit tests per finding + byte-identical golden/cargo/pytest suites. WP-C: byte-identical suites are the binding oracle for each split (a refactor that changes any golden byte is wrong unless the change is an intended, blessed regeneration); public-API parity check after each split. The full suite runs after each task; architectural splits (C2/C3/C14/C1) each get their own full-suite gate before commit.

## 11. Open questions

- **C2/C3/C14/C1 are large.** Each is a multi-file refactor with real blast radius; the plan treats each as its own task/stage. If any split's seam proves unsafe mid-refactor (e.g. a circular import in `chart.py`/`_facet.py`), surface it before forcing the split — do not introduce a worse coupling to achieve a line-count win.
- **C5 channel contract**: whether to introduce an `_AppearanceChannel` base or a shared frozenset composed per channel — decided at implementation by which yields the smaller, clearer diff; both satisfy the goal (one source of truth, intentional per-channel extras).
