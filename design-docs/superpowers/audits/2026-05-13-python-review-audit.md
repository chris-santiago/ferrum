# Python Review — `src/ferrum/` coherence audit

**Date:** 2026-05-13
**Status:** draft — awaiting approval before implementation
**Scope:** Full `src/ferrum/` package (72 files, 27,336 lines). Heavyweight `/python-review` pass.
**Trigger:** User invoked `/python-review src/ferrum` after completing Phase 10.

---

## TL;DR

The package is architecturally sound. The `_set_mark` / `_set_composite_mark` dual-dispatch design scales cleanly to 50+ marks, the desugar-fn protocol is a good abstraction, and the immutability-via-clone pattern works. The drift is concentrated in three areas:

1. **Three statistical marks bypass the scaffold** — `mark_density`, `mark_histogram`, `mark_smooth` predate `_set_composite_mark` and inline a ~15-line desugar-or-defer pattern the other 40+ marks solved generically.
2. **`_resolve_pending` accumulated kind-specific branches** — ribbon y2 extraction and smooth prior-mark logic live inline instead of in their desugars.
3. **Composition classes repeat optional-slot unwrapping boilerplate** — `theme()` and `properties()` are manually inlined in each asymmetric composition instead of routing through the existing `_rebuild_with_charts` hook.

No public API changes proposed. No behavior changes. All items are internal structural cleanup.

---

## Findings

### F1 — Statistical marks bypass `_set_composite_mark` [S3, high confidence]

**Location:** `chart.py:891–1178` (`mark_density`, `mark_histogram`, `mark_smooth`)

**Problem:** These three marks were written during Phase 8a, before `_set_composite_mark` existed (Phase 8b). Each inlines the same ~15-line pattern:

1. Detect orientation → pick field channel (`x` or `y`)
2. Check if encoding is set → if not, defer via `_set_composite_mark`
3. If set, call `desugar_*` directly
4. Handle the desugar result (5-tuple layered vs. 3-tuple single-mark)
5. Rewrite encoding slots from the remap dict

The other 40+ marks (composite, heavy-stat, diagnostic) all delegate this entire flow to `_set_composite_mark` + `_resolve_pending`. The three statistical marks are the only exceptions.

**Why it matters:** Three copies of desugar-result handling that `_resolve_pending` already does generically. When the 5-tuple protocol evolved (adding layer names, data_source), these methods had to be updated in parallel. Future changes to the desugar protocol require touching these methods individually instead of the single `_resolve_pending` path.

**Proposed fix:** Replace the inline logic with a `_set_composite_mark` call, using the existing closure adapters (`_resolve_density`, `_resolve_histogram`, `_resolve_smooth`) as the `desugar_fn`. The adapters already match the `desugar_fn(x_field, y_field, **kwargs)` protocol.

**Impact:** ~80 lines deleted from chart.py. No public API change, no behavior change.

**Validation:** `uv run pytest` (all tests), SVG golden byte-comparison for density/histogram/smooth marks. Manual spot-check of deferred path (`Chart(df).mark_density().encode(x="val")`) vs. immediate path (`Chart(df).encode(x="val").mark_density()`).

---

### F2 — `_resolve_pending` mixes three concerns [S3, medium confidence]

**Location:** `chart.py:390–501` (~110 lines, cyclomatic complexity ~12)

**Problem:** The method handles:
- **x/y field extraction** from encoding (lines 411–422) — generic, applies to all marks
- **Kind-specific routing** (lines 424–435) — ribbon y2 extraction, smooth prior-mark name forcing
- **Desugar-result dispatch** (lines 452–501) — 5-tuple vs. 3-tuple, with prior-layer injection and remap rewriting

The kind-specific branches (ribbon, smooth-with-prior) grew organically as features landed. Ribbon's y2 injection could live in `desugar_ribbon` (which already accepts `y2_field`). The smooth prior-mark scatter-layer logic is a distinct responsibility from desugar dispatch.

**Proposed fix:** Two extractions:

1. **Move ribbon y2 into the closure adapter.** Add a `_resolve_ribbon` adapter (like `_resolve_density` / `_resolve_histogram`) that extracts y2 from the encoding and passes it to `desugar_ribbon`. Remove the `if kind == "ribbon"` branch from `_resolve_pending`.

2. **Extract prior-mark layer building to `_build_prior_layer`.** A small helper that takes a mark name, encoding dict, mark_kwargs, and position, and returns a `_Layer`. Called from `_resolve_pending` in both the layered and single-mark branches, replacing the two inline constructions.

**Impact:** `_resolve_pending` drops from ~110 to ~60 lines, cyclomatic complexity from ~12 to ~6. No public API change.

**Validation:** Same test suite. Prior-mark path is exercised by `Chart(df).mark_point().mark_smooth().encode(x="x", y="y")` tests.

---

### F3 — `_PendingMark.prior_mark` accessed via `hasattr` [S2, high confidence]

**Location:** `chart.py:433`

**Problem:** The `prior_mark` attribute is accessed via `hasattr(self._pending_stat_mark, "prior_mark")` instead of being a declared field on the `_PendingMark` dataclass. This is fragile — if the attribute name changes, the `hasattr` check silently returns `False` instead of raising.

**Current `_PendingMark` definition (in `_layer.py`):**
```python
_PendingMark = namedtuple("_PendingMark", ["kind", "kwargs", "desugar_fn"], defaults=[None])
```

The `prior_mark` field was added later and is passed as a keyword argument, but `namedtuple` doesn't support optional fields in the middle of the positional list cleanly.

**Proposed fix:** Convert `_PendingMark` to a `@dataclass` with `prior_mark: str | None = None` as a declared field. Replace the `hasattr` check with a direct attribute access.

**Impact:** One file changed (`_layer.py`), one line changed in `chart.py`. No public API change.

**Validation:** Grep for all `_PendingMark` usage; existing tests cover both paths (with and without prior_mark).

---

### F4 — Composition `theme()`/`properties()` boilerplate [S2, medium confidence]

**Location:** `composition.py` — `JointChart` (388–432), `RepeatChart` (793–843), `ClusterMapChart` (1035–1083)

**Problem:** Each asymmetric composition manually unwraps optional slots in `theme()` and `properties()`:

```python
# JointChart.theme() — 20 lines
def theme(self, t):
    return JointChart(
        self.center.theme(t),
        top=self.top.theme(t) if self.top is not None else None,
        right=self.right.theme(t) if self.right is not None else None,
        ratio=self.ratio,
        spacing=self.spacing,
    )
```

This pattern repeats 6 times across 3 classes (2 methods × 3 classes). The `_rebuild_with_charts` hook already exists on `_ChartLike` and handles `share_scale` — but `theme` and `properties` don't use it.

**Proposed fix:** Each asymmetric composition implements `_rebuild_with_charts(fn)` to apply `fn` to each sub-chart, handling the None-guarding internally. Then `theme()` and `properties()` become one-liners:

```python
# On _ChartLike base
def theme(self, t):
    return self._rebuild_with_charts(lambda c: c.theme(t))

def properties(self, **kwargs):
    return self._rebuild_with_charts(lambda c: c.properties(**kwargs))
```

The symmetric `_CompositeBase` already has this pattern in `_rebuild_with_charts` — the fix just extends it to the three asymmetric classes.

**Complication:** `_CompositeBase` already defines `theme()` and `properties()` as concrete methods. Moving them up to `_ChartLike` requires removing them from `_CompositeBase`. This is clean because `_CompositeBase._rebuild_with_charts` already exists and does the right thing — the concrete `theme`/`properties` implementations are redundant.

**Impact:** ~120 lines deleted across composition.py, replaced by ~30 lines (3 `_rebuild_with_charts` implementations + 2 base methods). No public API change.

**Validation:** Existing tests for `JointChart`, `RepeatChart`, `ClusterMapChart` — they exercise `.theme()` and `.properties()`.

---

### F5 — `to_spec()` channel aliasing is ad-hoc [S2, high confidence]

**Location:** `chart.py:4481–4512`

**Problem:** Channel aliasing (fill→color, stroke→color, fill_opacity→opacity, detail→mark_style) is implemented as nested `if/elif` conditionals. Each alias has subtly different semantics (fill wins over color; stroke falls back to mark_style when color is already mapped; detail injects into mark_style via `setdefault`). The conditionals are correct but hard to audit — adding a new alias channel requires reading the whole block to understand the priority rules.

**Proposed fix:** Extract a `_apply_channel_aliases(enc, mk)` helper that encodes the aliasing rules as a data-driven table with explicit priority ordering:

```python
_CHANNEL_ALIASES = [
    # (source, target, condition, fallback)
    ("fill", "color", lambda enc: "color" not in enc, None),
    ("stroke", "color", lambda enc: "color" not in enc, None),
    ("fill_opacity", "opacity", lambda enc: "opacity" not in enc, None),
]
```

The detail→mark_style injection stays procedural (it writes to a different dict).

**Impact:** ~30 lines replaced by ~20 lines. More importantly, the aliasing rules become a scannable table. No behavior change.

**Validation:** Existing tests that exercise fill, stroke, fill_opacity channels.

---

### ~~F6 — Rendering protocol duplication between `Chart` and `_ChartLike`~~ [DROPPED]

Dropped after review. The duplicated methods (`save`, `show`, `_repr_svg_`, `_repr_mimebundle_`) are ~20 lines of trivial boilerplate each. `show_svg()` and `show_png()` are intentionally different (`Chart` uses single-pass Rust FFI; compositions use two-pass SVG→rasterize because they composite multiple SVGs Python-side). Deduplicating the boilerplate via a mixin or free functions costs more in indirection than the duplication costs in maintenance.

---

### F7 — Inconsistent `data_transform` patterns in diagnostic marks [S1, high confidence]

**Location:** `chart.py:1937–3700` (diagnostic mark methods)

**Problem:** Different diagnostic marks use different patterns for the optional `data_transform` kwarg to `_set_composite_mark`:

| Pattern | Example | Count |
|---|---|---|
| Inline lambda | `lambda df: _inject_constant(df, "_ref_zero", 0.0)` | ~8 |
| Named module-level function | `_disc_threshold_prep` | 2 |
| Conditional lambda | `(lambda df: ...) if flag else None` | ~6 |
| Closure over locals | `lambda df: _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)` | 3 |

All do the same thing: optionally augment the DataFrame before the desugar step. The variance is cosmetic but makes the diagnostic section of chart.py harder to scan.

**Proposed fix:** No code change. This is S1 — the patterns all work correctly and the variance reflects genuine differences in complexity (some transforms need captured locals, some don't). Normalizing to a single pattern would add boilerplate without improving correctness. Documenting the convention ("diagnostic marks use `data_transform=` for pre-desugar DataFrame augmentation") in a one-line comment at the first diagnostic mark is sufficient.

---

## Summary table

| ID | Finding | Severity | Confidence | Public API change | Lines changed (est.) |
|---|---|---|---|---|---|
| F1 | Statistical marks bypass scaffold | S3 | high | none | -80 |
| F2 | `_resolve_pending` mixed concerns | S3 | medium | none | -50, +20 |
| F3 | `_PendingMark.prior_mark` via `hasattr` | S2 | high | none | ~5 |
| F4 | Composition theme/properties boilerplate | S2 | medium | none | -120, +30 |
| F5 | Channel aliasing ad-hoc conditionals | S2 | high | none | -30, +20 |
| ~~F6~~ | ~~Rendering protocol duplication~~ | ~~S2~~ | — | — | DROPPED |
| F7 | `data_transform` pattern variance | S1 | high | none | 0 (document only) |

---

## Proposed implementation order

Ordered by safety × leverage, safest first:

1. **F3** — `_PendingMark` dataclass conversion. Trivial, zero risk, unblocks F2.
2. **F1** — Migrate density/histogram/smooth to `_set_composite_mark`. Mechanical, highest leverage, validated by existing goldens.
3. **F2** — Simplify `_resolve_pending`. Depends on F1 (fewer paths to handle after migration). Medium risk — the prior-mark path is subtle.
4. **F5** — Channel alias table. Self-contained, no dependencies.
5. **F4** — Composition `_rebuild_with_charts`. Self-contained, moderate scope.
6. ~~**F6**~~ — DROPPED.
7. **F7** — No implementation needed (document only).

---

## Decisions

1. **F6** — DROPPED. Duplication is ~20 lines of trivial boilerplate; not worth the indirection.
2. **Scope** — Implement F1–F5. Skip F7 (document only).
3. **Branch** — Single `refactor/python-review` branch.
