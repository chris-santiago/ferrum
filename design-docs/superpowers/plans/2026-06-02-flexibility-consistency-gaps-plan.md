# Plan — Flexibility consistency & capability gaps (Phase C)

**Date:** 2026-06-02
**Branch:** continue on `feat/flexibility-new-capabilities` (phase still open)
**Source:** `/tmp/ferrum-ux-audit/SYNTHESIS.md` §C–D (surviving cross-cutting items) + code-archaeology FA-1..FA-6.
**Execution skill:** `chris-code:subagent-driven-development` — every task gets coder → spec review → quality review → review-lite gate. RED test first (TDD). Regression test after each fix. Visual-inspect any render change (rasterize golden → PNG → Read) before commit.

All file:line pins below were verified by read-only Explore agents on 2026-06-02. Coders should re-confirm at edit time (line numbers drift).

---

## Scope — 11 items

Five cross-cutting consistency gaps (C1–C5) + six narrow follow-ups (FA-1..FA-6). Each is "make it work or raise" — no warn-fallbacks, no `NotImplementedError`.

### C1 — `title=None` suppresses an axis title (mirror `legend=None`)
- **Symptom:** `X("hp", title=None)` renders title "hp"; only `Axis(title="")` suppresses. Inconsistent with `legend=None`, which suppresses.
- **Root cause:** `src/ferrum/encoding/base.py:124` drops any kwarg whose value is `None` before serialization, so `title=None` never reaches Rust; `prepare.rs:603,622` then falls back `x_axis_title.or(x_field)`.
- **Change (Python-only):** in `base.py`, special-case `title` the way `legend` is handled (`base.py:118-123`) — when `title` is *present* in `_kwargs` with value `None`, forward an explicit empty-title sentinel (`{"title": ""}`) so Rust emits an empty axis title. Mirror `_normalize_legend`'s suppress contract (`legend.py:137-138`). Leave `title` absent → field-name default unchanged.
- **Test:** `title=None` → no axis title in rendered SVG; `title` omitted → field-name title still present; `title="Foo"` unchanged.

### C2 — Annotations can anchor to a categorical/ordinal axis
- **Symptom:** annotation at a plain category coord (e.g. `"cat_a"`) raises; `_coerce_temporal` force-parses every string as ISO-8601. Ergonomic `fm.annotate_*` wrappers also reject `fm.px`/`fm.norm` that the lower-level `ferrum.annotation.*` accepts.
- **Root cause:** `src/ferrum/annotations.py:24` `_coerce_temporal`; raise at `src/ferrum/annotation/coords.py:112-115`. Wrapper type narrowing at `annotations.py` (`annotate_hline:49`, `annotate_vline:111`, `annotate_rect:172`, `annotate_text:240`, `annotate_arrow:399`) restricts to `_AnnotationCoord`, excluding the `CoordValue` (px/norm) that `annotation/primitives.py:8-14` accepts.
- **Change (Python-only):** (a) make string-coord coercion ISO-detect first — only temporal-parse strings that match an ISO date/datetime shape; otherwise pass the string through as an ordinal category coordinate (resolved against the axis ordinal domain, machinery exists at `_scale_share.py:145-156` `compute_union_domain`). Never raise on a plain label. (b) Widen the `fm.annotate_*` wrapper coordinate types from `_AnnotationCoord` to the `CoordValue` union so `fm.px`/`fm.norm` are accepted, matching the primitives.
- **Test:** annotation on a categorical x-axis renders at the band; `fm.annotate_text(x=fm.px(40), ...)` no longer raises; existing temporal annotations unchanged.

### C3 — Typed `Scale` classes auto-infer `domain` like the dict form
- **Symptom:** `fm.LogScale(range=(0,400))` raises `TypeError: missing 'domain'`, but `scale={"type":"log","range":[0,400]}` auto-infers domain from data. Typed class is *less* capable than the dict.
- **Root cause:** `domain` is a required positional in the PyO3 ctors — `crates/ferrum-core/src/scale/linear.rs:174`, `log.rs:224` (and the sibling continuous scales in `crates/ferrum-core/src/scale/`). The dict path makes `domain: Option<Vec<f64>>` and `scale_resolve/positional.rs:248-275` auto-infers when `None`.
- **Change (Rust + stub):** make `domain` an optional kwarg defaulting to `None` on **all continuous typed Scale ctors** (Linear, Log, and every sibling in `crates/ferrum-core/src/scale/` that today requires it — coder enumerates). A `None` domain must flow through `_scale_to_dict` (`src/ferrum/encoding/_scale.py:62-97`, already gates on truthy domain) to the same render-time inference. Update `src/ferrum/_core.pyi` signatures to `domain` optional.
- **Test:** `LogScale(range=...)` with no domain renders identically to the equivalent dict form; explicit `domain=` still honored.

### C4 — `resolve=` on `vconcat`/`hconcat`; deduped legend in pairplot
- **Symptom:** `resolve=` exists on `concat`/`layer`/`ConcatChart`/`RepeatChart`/`LayerChart` but NOT `vconcat`/`hconcat`; SPLOM/pairplot duplicates the legend per cell.
- **Root cause:** missing param on `HConcatChart.__init__` (`composition.py:427-445`), `VConcatChart.__init__` (`composition.py:478-496`), `hconcat()`/`vconcat()` (`__init__.py:278,304`). `HConcat`/`VConcat` have no `_resolved_charts()` (the `ConcatChart._resolved_charts()` machinery is at `composition.py:1661-1676`). `pairplot` calls `RepeatChart()` without `resolve=` (`plots/matrix.py:293`).
- **Change (Python-only):** (a) add `resolve=None` to both `HConcatChart`/`VConcatChart` ctors and the `hconcat`/`vconcat` free functions, store `self._resolve`, and give the composite-base a `_resolved_charts()` equivalent (factor a shared helper from `ConcatChart` rather than copy-paste). (b) in `pairplot`, pass `resolve={"color":"shared"}` when `hue` is set so the color domain unifies. **Scope honesty:** a single *deduped* legend outside the grid needs compositor layout work — if that is not reachable in this task, unify the domain via `resolve` and **log the residual** (one legend per cell, but consistent colors) as a follow-up; do NOT silently leave per-cell divergent legends.
- **Test:** `vconcat(a, b, resolve={"color":"shared"})` unifies the color scale; `pairplot(hue=)` cells share one color domain.

### C5 — 2-D `mark_density` honors categorical hue
- **Symptom:** `jointplot(kind="kde", hue=)` splits the 1-D marginals by hue but pools the center 2-D surface — internally inconsistent.
- **Root cause:** the 1-D `Kde` transform supports `groupby` (`crates/ferrum-core/src/transform/kde.rs:40`) but `Kde2DSpec` has no `groupby` field (`crates/ferrum-core/src/transform/kde_2d.rs:33-44`, ctor `:264-271`). Python never threads it: `desugar_contour` has no `groupby` param (`src/ferrum/marks/heavy_stat.py:24-33`), `desugar_density` doesn't pass one (`src/ferrum/marks/statistical.py:128-136`), and `jointplot` center omits it (`plots/matrix.py:947`).
- **Change (Rust + Python):** (a) Rust — add `groupby: Option<String>` to `Kde2DSpec` + the `PyKde2D` signature; when set, compute one 2-D surface per group and preserve the group column for downstream color encoding (mirror the per-group logic in `kde.rs`). (b) Python — thread `groupby` through `desugar_contour` → `Kde2D`, pass it from `desugar_density`, and set `groupby=hue` on the `jointplot` center when `hue` is set.
- **Test:** `jointplot(kind="kde", hue=)` center renders one contour set per hue group; no-hue path byte-stable.

### FA-1 (S3) — `mark_arc(theta:N, radius:Q)` Nightingale coxcomb renders blank
- **Root cause:** nominal theta falls into the pie path which `col_as_f64`'s it → empty. Pins: `arc.rs` build gate + Python polar dummy-y remapping (coder locates the remap site).
- **Change:** make `mark_arc` with nominal theta + quantitative radius render equal-band value wedges, OR raise a clear error directing to the `mark_bar`+`CoordPolar` coxcomb path (fixed in G-D7). No silent blank.
- **Test:** the coxcomb design either renders non-empty or raises with an actionable message.

### FA-2 (S2) — Polar-bar angular layout not equal full-circle bands
- **Root cause:** 2-category coxcombs render as two narrow petals in the upper arc, not equal 360°-filling wedges. Angular band-scale/extent under `CoordPolar` — `bar.rs build_polar` angular bands / band-scale padding/extent. Radial stacking (G-D7) is correct; this is the angular axis only.
- **Change:** angular band scale fills the full circle with equal wedges (respect explicit `theta`/`theta2` overrides where present).
- **Test:** N-category polar bar → N equal wedges spanning 360°; visual-inspect the rendered PNG.

### FA-3 (S3) — Rust `stat_aggregate` rejects Int64 groupby
- **Root cause:** aggregate groupby accepts Float64/Utf8 only; an Int64 x/groupby column errors. `crates/ferrum-core/src/transform/` aggregate path.
- **Change:** accept Int64 (and other integer dtypes) as a groupby key (reuse the `col_as_ordinal_category_str` / `numeric_col` patterns already used in `top_k.rs`/`rect.rs`).
- **Test:** aggregate grouped on an Int64 column succeeds; affects both single-chart and layered aggregate.

### FA-4 (S3) — Per-layer `bin=` never runs (twin of the T12 aggregate gap)
- **Root cause:** the layered path resolves per-layer aggregates into named transforms (T12) but NOT per-layer `Bin` sentinels; `_layer_pending_aggregates` keeps only `_PendingAggregate`. A layer with `bin=` silently isn't binned. `src/ferrum/chart.py`.
- **Change:** mirror the T12 named-transform + `data_source` routing for per-layer `Bin` sentinels (factor the shared resolve helper rather than duplicate).
- **Test:** a layered chart with one layer carrying `bin=` actually bins that layer; non-layered binning unchanged.

### FA-5 (S1) — Ordinal/quantitative-color `mark_area` legend swatch ≠ fill
- **Root cause:** after the T11 split fix, legend swatches show categorical colors while area fills use a sequential ramp — legend/fill color source diverges for ordinal area. (Exposed by T11.)
- **Change:** unify the legend swatch color source with the actual fill color source for ordinal/quantitative-color area.
- **Test:** legend swatch color matches the rendered area fill; visual-inspect PNG.

### FA-6 (S1) — Violin box-inner layers don't color-encode while quartile/point do
- **Root cause:** sibling asymmetry from `desugar_boxplot`'s layer contract (its layers never color-encode); cosmetic under the T10 overlay. `src/ferrum/marks/heavy_stat.py`.
- **Change:** make the box-inner layers color-encode consistently with the quartile/point layers when a color field is present.
- **Test:** violin-with-hue box-inner respects the hue color; visual-inspect PNG.

---

## File footprints & staging

Stage by zero-overlap on source AND test files. Rust tasks additionally serialize the `maturin develop` rebuild (race), even when their source files are disjoint.

| Item | Files touched | Lang |
|---|---|---|
| C1 | `encoding/base.py` | Py |
| C2 | `annotations.py`, `annotation/coords.py` | Py |
| C4 | `composition.py`, `__init__.py`, `plots/matrix.py` | Py |
| FA-4 | `chart.py` | Py |
| FA-6 | `marks/heavy_stat.py` | Py |
| C3 | `scale/*.rs`, `_core.pyi` | Rust+stub |
| C5 | `transform/kde_2d.rs` + `marks/heavy_stat.py`, `marks/statistical.py`, `plots/matrix.py` | Rust+Py |
| FA-1 | `render/marks/arc.rs` + polar remap (Py, coder locates) | Rust(+Py) |
| FA-2 | `render/marks/bar.rs` | Rust |
| FA-3 | `transform/` aggregate | Rust |
| FA-5 | `render/marks/area.rs` (+ legend source) | Rust(+Py) |

**Conflicts:** C4 and C5 both touch `plots/matrix.py`; C5 and FA-6 both touch `marks/heavy_stat.py`.

**Suggested stages:**
- **Stage 1 (Python, parallel):** C1, C2, FA-4. (disjoint: base.py / annotations.py+coords.py / chart.py)
- **Stage 2 (Python, parallel):** C4, FA-6. (disjoint: composition.py+__init__.py+matrix.py / heavy_stat.py — both clear of Stage 1)
- **Stage 3 (Rust + the Rust/Py mixed, serialize builds):** C3, FA-1, FA-2, FA-3, FA-5, then C5 last (C5's Python touches matrix.py + heavy_stat.py, so it must follow Stages 1–2). Rust source files are disjoint so coders may write in parallel, but run `maturin develop` serially.

Re-confirm the C5 vs Stage-1/2 ordering at execution: C5 Python edits to `matrix.py`/`heavy_stat.py` must land after C4/FA-6 to avoid clobber.

---

## Done criteria
- All 11 items: RED test → fix → green; full `uv run pytest -n auto` passes; `cargo test` (12 suites) green for any Rust change; wasm untouched (no rebuild expected).
- Every render change visually inspected via `python scripts/snapshot-goldens.py` + Read PNG before commit.
- Code-archaeology doc FA-1..FA-6 status flipped to resolved; SYNTHESIS §C residual updated.
- Each item committed separately via `commit-commands:commit`; final review-lite on `base..HEAD`.
- **Honest-scope rule:** C4 shared-legend-layout and FA-1 (if it lands as a raise rather than a render) get their residual logged, not hidden.
