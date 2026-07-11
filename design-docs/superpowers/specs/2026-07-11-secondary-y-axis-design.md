# Secondary Y-Axis (Per-Layer Independent Y-Scales) Design Spec

**Date:** 2026-07-11 · **Issue:** GH #52 · **Branch:** `feat/secondary-y-axis`
**Origin:** approved coherent-change defended choice (2026-07-11, this session). Follow-ups: dual-x is GH #55; Joint/ClusterMap native resolve is GH #53.

## 1. Scope

Make `LayerChart(..., resolve={"y": "independent"})` render a real dual-axis chart — each layer positioned by its own y-scale, layer 0's axis on the left, subsequent layers' axes on the right — in both static SVG and interactive output, replacing today's typed `ValueError`. The engine is per-layer y-scale slots in the single-panel chart pipeline (`ChartSpec.layers` → scale resolution → per-layer draw binding → axis layout → WASM runtime). The existing `fm.SecondaryY` structural feature is re-based onto this subsystem and its siloed Rust implementation deleted.

## 2. Goals

- `resolve={"y": "independent"}` on LayerChart produces per-layer y-scales with a left axis (layer 0) and right axis/axes (layers 1..n), n unbounded.
- One implementation serves both output kinds: static SVG and the interactive scene render from the same flat layered spec.
- Layout reserves margin bands for left **and** right y-axes simultaneously; right-side axes never overdraw the plot area.
- Interactive correctness within the one-panel contract: tooltips, hit-testing, zoom/pan, and reactive domain rescale each respect the owning layer's scale.
- `fm.SecondaryY` keeps its public API and gains band reservation, real axis layout, and interactivity by desugaring to a layered independent-y chart.
- Archaeology item S2 (static-vs-interactive shared-resolution divergence for LayerChart) is resolved for the independent-y case by construction (one path).

## 3. Non-goals

- `resolve={"x": "independent"}` (dual x-axis): stays a typed `ValueError`, message citing GH #55.
- Mixed per-layer share groups (e.g. layers 0–1 share left, layer 2 independent right) as *user-facing API*: `resolve` stays all-or-nothing per channel (Vega-Lite semantics). The wire contract supports mixing (§6) because the `SecondaryY` desugar needs it, but no public spelling exposes it.
- Cross-layer gridline/tick alignment ("nice" ticks that coincide across scales).
- Changes to the default/shared LayerChart paths (static overlay tree, interactive merged) — byte-identical output.
- Joint/ClusterMap resolve (GH #53).

## 4. System behavior

**Construction.** `_validate_layer_resolve` accepts `y: "independent"`; `x: "independent"` still raises, message re-pointed to #55. `share_scale(y="independent")` on LayerChart follows the same rule (it is resolve= sugar).

**Rendering (both kinds).** An independent-y LayerChart lowers through the merged flat path (`_build_merged`) — not the overlay composite tree — producing one `ChartSpec` whose `layers` carry per-layer scale-slot metadata. No union-domain injection for y (each layer resolves natively in Rust). Static `to_svg` and `.interactive()` both consume this spec.

**Scales.** Each independent layer's y-scale resolves from that layer's own y encoding (explicit `scale=` wins, else auto from that layer's transform outputs and data), with the same rules the primary y uses today (bar zero-anchor, y2 domain extension, nice-ing, every ScaleSpec type the primary supports). Layer 0's scale is the primary: it drives gridlines and the left axis.

**Axes.** Layer 0's y-axis renders left. Each subsequent independent layer's y-axis renders on the right, stacked outward — each axis reserves its own label band + title gutter beyond the previous. Each axis takes its title from its layer's y field/title and honors that layer's per-encoding `Axis(...)` config and the global theme; there is no automatic color-tinting to the layer's mark color. Right axes render ticks and labels but no gridlines.

**Interactive (one panel).** The chart remains exactly one scene panel. Zoom/pan applies the panel-level screen affine to all layers together; each axis relabels by inverting through its own scale. A reactive domain rescale (`domainParam`) or brush bound to a specific layer's y encoding rescales **only that layer's** marks (per-slot affine composed with the panel affine) and relabels only that layer's axis. Tooltip and selection value readout invert pixel positions through the owning layer's scale.

**Nesting.** An independent-y LayerChart nested in a composite (HConcat/facet/etc.) lowers as one leaf whose spec carries the layer slots. Such a leaf does not participate in cross-panel y sharing; an explicit parent-level `y: "shared"` over a subtree containing one raises a typed `ValueError` (contradictory request — same pattern as the existing overlay-contract error).

**SecondaryY.** `chart + SecondaryY(field, mark, axis, color, opacity, scale)` desugars to: base chart's layer(s) unchanged (retaining their existing shared-y relationship), plus one appended layer — mark `mark`, y encoding on `field` (with `axis`/`scale` attached), inheriting the base chart's x encoding, color literal `color`, opacity `opacity` — marked independent-y. Multiple `SecondaryY` additions append multiple independent layers (stacked right axes). Visual delta vs. today: the plot area narrows to accommodate the reserved right band (an upgrade; goldens re-blessed with visual inspection).

**Degenerate cases.** Single-layer LayerChart with independent y renders normally (left axis only). Empty-data layers keep existing skip/raise semantics. A layer with no y encoding under independent resolve keeps today's merged-chart behavior (inherits chart-level y → joins the primary scale).

## 5. Architecture

- **Python (`composition.py`, `_spec_build.py`):** decides routing (independent-y → merged flat path for both kinds), attaches per-layer slot metadata to the layer dicts, desugars `SecondaryY` at `Chart.__add__` structural handling. No scale math in Python.
- **Rust spec (`spec/layer.rs`, `spec/chart.rs`):** `Layer` carries the independent-y flag; wire-compatible default (absent = shared).
- **Rust scale resolution (`render/scale_resolve`, `scene_build.rs`):** resolves one y `ScaleKind` per independent layer (today only layer 0 seeds resolution); `ResolvedScales` grows per-layer y slots; each layer's `DrawCtx` binds its own slot. Shared layers all bind slot 0.
- **Rust layout (`layout/mod.rs`, `layout/axis.rs`):** `reserve_axis_bands` reserves the left band plus one right band per secondary axis; right axes reuse the existing `AxisOrient::Right` layout machinery.
- **Rust render:** axis nodes emitted per slot; `render/secondary_axis.rs` and `SecondaryYSpec`/`StructuralSpec::SecondaryY` are **deleted** (superseded by the desugar).
- **WASM (`ferrum-wasm`):** the panel's coordinate state carries one y-domain per slot; mark meshes carry their slot id; rescale affines are per (panel, slot); zoom/pan stays panel-level.

Data flows once, Python → Rust, as today; all scale computation stays in Rust.

## 6. Canonical interfaces / data contracts

**Public API (unchanged spellings, new behavior):**

```python
LayerChart(a, b, resolve={"y": "independent"})   # dual axis; was ValueError
chart + SecondaryY(field="temp", mark="line")     # same API, now desugars to a layer
```

**Wire contract — per-layer slot flag** (the seam between Python lowering and Rust):

```python
# ChartSpec.layers[i] layer dict gains one optional key:
{"mark": ..., "encoding": {...}, "independent_y": bool}  # absent/false = shared
```

Semantics: layers with `independent_y: false` (or absent) share the primary y-scale group; each layer with `independent_y: true` resolves its own y-scale and receives its own right-side axis, in layer order. Layer 0 is always the primary/left axis regardless of its flag. `resolve={"y": "independent"}` sets the flag per non-first **member chart**, not per raw layer: a multi-layer non-first member keeps its own internal layers sharing one y-scale (they all get the same `independent_y` value), and a multi-layer member other than the first raises the typed conflict guard rather than silently flattening its layers into one slot — the bool wire has no way to group a multi-layer member's layers into a single secondary slot. A single-layer non-first member is unaffected by this distinction. Composite-mark members (e.g. `mark_line(point=True)`, which expands to two layers) are the tracked follow-up for a slot-group wire extension; the `SecondaryY` desugar sets the flag only on its one appended layer.

**Rust seam:** `ResolvedScales` exposes per-slot y access such that a layer's draw context, axis emission, and the interactive coordinate state all key off the same slot index (slot 0 = primary; slot k = k-th independent layer). Mark geometry, axis ticks/labels, and WASM domain state for a slot must agree by construction — one resolution site.

**Scene/WASM contract:** panel coordinate state = one x-domain + an ordered list of y-domains (index = slot); every mark mesh and y-axis node carries its slot index; per-slot rescale affine composes with the per-panel zoom/pan affine.

**Errors (typed `ValueError`, stable prefixes):**
- `LayerChart` + `x: "independent"` → names the overlay contract, cites GH #55.
- Parent composite `y: "shared"` over a subtree containing an independent-y layered leaf → names the conflict.

## 7. Invariants and constraints

- **One-panel contract preserved:** an independent-y LayerChart is exactly one scene panel; selections/hit-testing operate within it.
- **Shared-path byte-stability:** default and `y: "shared"` LayerChart output (SVG bytes, scene JSON) is unchanged; existing goldens must not regress.
- **No Python-side scale math:** domains, nice-ing, and pixel mapping stay in Rust (no twinx-style data rescaling; tooltips report raw data values).
- **Phase B composite contract:** compositions still render through the composite entries; the merged-flat routing for independent-y LayerChart is a documented extension of the existing interactive exception (CLAUDE.md "Composite rendering"), applied to both kinds for this case.
- **Wire backward compatibility:** specs without `independent_y` deserialize and render exactly as today.
- **No global mutable state; no matplotlib** (standing constraints).
- `cargo test` green before done; new/regenerated goldens rasterized and visually inspected before commit.

## 8. Key decisions and tradeoffs

1. **Engine = per-layer slots in the single-panel pipeline** (vs. growing `SecondaryYSpec`; vs. composite-overlay per-leaf independence; vs. Python twinx emulation). Rationale and full rebuttals: the approved defended choice — one implementation serves both output kinds, honors the one-panel contract, resolves S2, and kills the `secondary_axis.rs` silo instead of deepening it.
2. **Both output kinds route through the merged flat path for independent-y.** The overlay tree cannot serve interactive (one-panel contract), so a composite-side implementation would mean two mechanisms for one feature. Cost: static independent-y bypasses the overlay entry; accepted as the documented LayerChart exception, now symmetric across kinds.
3. **Per-layer boolean flag, not a chart-level mode.** A chart-level "all independent" flag cannot express `layered_base + SecondaryY` (base layers keep sharing; only the appended layer is independent). The public API stays all-or-nothing; the wire supports mixing.
4. **Layer 0 owns left axis + gridlines; secondaries stack right, unbounded n.** Standard dual-axis convention; no arbitrary cap (no-defer rule). Grid from one scale avoids the tick-alignment rabbit hole (explicit non-goal).
5. **Zoom/pan is panel-level; only reactive rescale is per-slot.** Zooming moves all layers together in screen space (axes relabel per scale) — anything else makes overlaid marks shear apart under direct manipulation. `domainParam`/brush rescale is data-domain-driven and layer-owned, so it composes per-slot. This keeps the FA-16 screen-space stroke machinery valid unchanged.
6. **Independent-y leaves are excluded from cross-panel y sharing; explicit conflict raises.** Silent partial sharing (primary-only) would misrepresent the caller's request — same reasoning that produced the original overlay-contract error.
7. **`SecondaryY` unification in scope** (user decision 2026-07-11): one secondary-axis mechanism after this change; its ~10 test files pin desugar compatibility. Accepted visual delta: right band reservation narrows the plot area.
8. **y-only; dual-x deferred to #55** (user decision 2026-07-11): architecturally symmetric but rarer, and would double the WASM interaction scope in one change.
9. **No auto color-tinting of right axes.** Neither Vega-Lite nor the theme cascade has precedent; per-encoding `Axis(...)` config already gives users the knob.

## 9. Acceptance criteria

1. `LayerChart(bars, line, resolve={"y": "independent"}).to_svg()`: two y-axes (left/right), both margin bands reserved, each layer's marks positioned by its own scale, right axis titled from layer 1's y field. Golden committed after PNG inspection.
2. Same chart `.interactive()`: one scene panel; headless WASM capture shows the dual-axis render; tooltips on each layer report that layer's raw data values.
3. Three-layer independent y: left + two right axes stacked outward, no overlap, plot area shrunk accordingly.
4. Default and `y: "shared"` LayerChart: existing goldens and scene JSON byte-identical.
5. `resolve={"x": "independent"}` raises the typed error citing #55; existing rejection tests updated, y rejection tests flipped to feature tests (`tests/test_cohesion_share_scale_resolve_unification.py`).
6. `chart + SecondaryY(...)`: renders via the desugar; `render/secondary_axis.rs` and `SecondaryYSpec` no longer exist in the crate; all pre-existing `SecondaryY` tests pass (updated goldens visually inspected); multiple `SecondaryY` additions produce stacked right axes.
7. Zoom/pan on a dual-axis chart keeps layers visually locked together while both axes relabel correctly (browser/headless-capture validated).
8. A `domainParam` brush bound to one layer rescales only that layer's marks and axis.
9. Independent-y LayerChart nested in an HConcat renders as one leaf; explicit parent `y: "shared"` over it raises the typed conflict error.
10. Temporal y in one layer + numeric y in the other renders with correct per-axis tick formatting (temporal-domain seam already fixed in `5660b62a` — regression stays green).
11. Layer with no y encoding under independent resolve joins the primary scale (no crash, no phantom axis).
12. `cargo test` and `uv run pytest -n auto` green; `grep -rn "#52" src/ crates/` returns no stale pointers; archaeology doc S2 row and #52 references updated.

## 10. Validation strategy

- **Behavioral tests at the seams:** Python-side lowering tests (layer dicts carry the flag; SecondaryY desugar shape), Rust-side scale-resolution tests (per-slot domains from per-layer data), layout tests (band reservation both sides, n-axis stacking), scene JSON assertions (slot indices on meshes/axes).
- **Visual proof:** new goldens rasterized via `scripts/snapshot-goldens.py` and read before commit (hard constraint); dual-axis and SecondaryY-upgrade renders inspected.
- **Interactive proof:** headless WASM capture harness (established pattern) for zoom-lock, per-layer rescale, and tooltip readout.
- **Non-regression:** full existing golden suite byte-checked for the shared paths; the flipped ValueError tests prove the gate moved rather than vanished.
- **RED discipline:** feature tests written to fail on main (ValueError) and pass on the branch; regression tests for each bug found en route per the standing rule.

## 11. Open questions

None blocking; remaining choices (exact band-width computation reuse, axis-node naming) are implementation-level and constrained by §6–§7.
