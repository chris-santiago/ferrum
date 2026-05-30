# Continuous-axis ticks use scale projection — Design Spec

**Date:** 2026-05-30
**Branch:** `feat/render-gaps-17-19-21`
**Status:** Prerequisite for code-archaeology item 18 (`ferrum.Grid` + minor ticks). Must land and be golden-verified before item 18 Tasks 3-5.
**Origin:** A scene-pipeline audit (2026-05-30) found that continuous-axis major ticks/gridlines are placed by uniform slot centers while data marks are placed by scale projection — so gridlines do not align with the data they annotate. Minor gridlines (item 18) cannot align with both grids until this is fixed.

## 1. Scope

Make major axis ticks and gridlines on **continuous** scales (linear, log, time, pow, sqrt, symlog) place by **scale projection** — the same domain→pixel mapping (`scale_internal`, including the existing padding inset) that positions data marks — instead of the current uniform slot centers derived from the tick-label count. Categorical (ordinal/band/point) and discretizing (quantile/threshold/bin-ordinal) axes are unchanged: they keep uniform-slot placement, which is correct band behavior. This aligns continuous gridlines with their data and establishes a single coordinate grid that item 18's scale-projected minor ticks join automatically.

## 2. Goals

- Continuous-axis major tick marks, tick labels, and gridlines sit at the scale-projected pixel of each tick value — coincident with where a data mark of that value lands.
- Categorical/discretizing axes render byte-identical to today.
- The x-axis label-collision cascade operates on the actual (possibly non-uniform) inter-tick gaps rather than a uniform slot width.
- All changed continuous-axis goldens are regenerated and visually inspected; categorical goldens stay byte-identical.

## 3. Non-goals

- Item 18 minor-tick styling, `build_grid` two-level emission, the Python `Grid` class (those resume after this lands).
- Any change to categorical/discretizing tick placement.
- Reconciling the interactive zoom tick metadata (`TickLevel`/`tick_data`) grid — out of scope here; note it for a later pass.
- Changing the scale's padding-inset policy (`DEFAULT_SCALE_PADDING_FRAC`, `SCALE_PADDING_MAX_PX`); ticks must use the *same* inset marks already use, not a new one.

## 4. System behavior

For a continuous x or y axis, each tick (value, label) renders at `scale.to_pixel(value)` — identical to the projection that places a mark of that value, so a gridline passes exactly through data points sharing its value. Tick labels move with their gridlines. End ticks sit at the padded domain extent (the 8px-capped inset), not at half-slot margins. Spacing reflects the scale: linear/time evenly, log/pow/symlog non-uniformly.

For a categorical or discretizing axis, ticks remain at uniform slot centers exactly as today; output is unchanged.

The x-axis collision cascade still chooses wrap/shrink/rotate/cull/elide, but its fit test uses the real per-tick gaps (continuous) instead of `panel.w / n`; for categorical axes it uses the uniform slot as before.

## 5. Architecture

- **Tick pixel source.** The resolved positional scale already exposes scale-projected tick pixels (`scale_resolve::tick_data()` → `{value, label, pixel}`). `prepare.rs` supplies these per-tick pixels to layout for continuous axes; for categorical axes it supplies nothing new (layout keeps deriving slot centers from the label count).
- **Layout placement.** `layout_x_axis` / `layout_y_axis` place each `TickLayout.position` at the supplied scale-projected pixel when present (continuous); otherwise fall back to the existing uniform-slot formula (categorical). The selection is driven by whether scale-projected tick pixels were provided — categorical axes provide none.
- **Gridlines.** `build_grid` is unchanged in structure — it already draws at `TickLayout.position`; it simply now receives projected positions for continuous axes.
- **Collision cascade.** The cascade receives the actual tick pixel positions (or their gaps) so its fit/overlap logic reflects true spacing on continuous axes; the uniform-slot path remains for categorical.
- **Minor ticks (item 18, already built).** `minor_tick_fractions` is scale projection over the same range; once majors are scale-projected, minors and majors share one grid with no further change.

## 6. Canonical interfaces / data contracts

Layout gains a per-axis, optional set of **scale-projected tick pixel positions** (one per tick label, same order). Contract:
- When present (continuous axes): `TickLayout.position` = the provided projected pixel; tick count equals the label count; the collision cascade uses the real gaps between these positions.
- When absent (categorical/discretizing axes): placement and cascade behavior are exactly as today (uniform slot centers).

The projected pixels must come from the **same positional scale and inset** that place data marks, so tick pixel == mark pixel for equal values. No new inset or padding constant is introduced.

## 7. Invariants and constraints

- **Categorical/discretizing byte-identity.** Ordinal/band/point/quantile/threshold/bin-ordinal axes produce byte-identical SVG to pre-change.
- **Tick ↔ mark coincidence.** On a continuous axis, a tick at value `v` and a data mark at value `v` share a pixel coordinate (modulo the shared inset).
- **No new geometry source.** Tick pixels derive from the existing resolved positional scale; do not introduce a parallel projection.
- **Cascade safety.** Non-uniform continuous spacing must not regress label legibility — the cascade must not assume uniform slots for continuous axes.
- **Item-18 gate stays inert.** This change does not enable minor rendering; the `include_minor` gate remains off until item 18 Task 3.

## 8. Key decisions and tradeoffs

- **Continuous → projection, categorical → slots (locked).** Slot centers are correct for categories (a category owns its band) but wrong for continuous values (which have a position, not a slot). Splitting by scale family fixes continuous alignment while preserving correct band behavior and bounding the golden blast radius to continuous axes.
- **Reuse `tick_data` pixels, not a new computation.** The scale-projected pixels already exist alongside the labels; passing them through avoids a second projection path and guarantees tick==mark coincidence. Rejected: recomputing pixels in layout (risks drift from the mark path).
- **Cascade uses real gaps.** Necessary because log/pow/symlog ticks are non-uniform; a uniform-slot assumption would mis-judge collisions. Tradeoff: the cascade is the most golden-sensitive surface and needs visual inspection.
- **Pre-existing major-vs-mark misalignment is the bug being fixed**, accepted as a foundational invariant change with a full golden regeneration, per the root-cause rule. The interactive-tick (`TickLevel`) grid is knowingly left for a follow-up.

## 9. Acceptance criteria

- On a linear axis, major gridline pixels equal the scale-projected pixels of the tick values and coincide with data marks of those values (the audit's 64.9/320.5/576.0 mark positions now match the gridline positions).
- Log/pow/symlog/time continuous axes show non-uniform gridlines at projected tick pixels.
- Categorical/discretizing axis goldens are byte-identical.
- All changed continuous-axis goldens regenerated and visually inspected (rasterize + Read PNG per CLAUDE.md); no blank/misdrawn panels.
- `cargo test`, `uv run pytest -n auto`, `cargo clippy` (incl. wasm target) green.
- Item 18's existing scale-projected minor ticks (gate still off) require no change to align.

## 10. Validation strategy

- **Rust unit:** continuous `layout_x_axis`/`layout_y_axis` place ticks at supplied projected pixels; categorical path unchanged (uniform slots); cascade fit logic exercised with non-uniform gaps.
- **Behavioral (Python/SVG):** a linear chart's gridline x-positions equal its mark cx-positions for shared values; a log chart shows non-uniform gridlines; a categorical chart's SVG is unchanged.
- **Golden:** regenerate continuous-axis goldens, visually inspect each; assert categorical goldens unchanged (byte-equality).
- A regression test pinning tick↔mark coincidence on a linear axis, so this cannot silently regress.

## 11. Open questions

None blocking. (Follow-up, out of scope: reconcile the interactive `TickLevel` zoom-tick grid with the static projected grid.)
