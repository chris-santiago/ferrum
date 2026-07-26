# Enhancement roadmap (2026-07-26)

Triage of all 28 open `enhancement` issues into themed waves. Sizing: **S** = hours, **M** = a session, **L** = multi-session phase (brainstorm → spec → plan). Ordering favors compounding leverage and quick wins first; the big expressiveness features are phased campaigns at the end.

## Wave Q — quick wins (one session, mostly S)

| Issue | What | Size | Note |
|---|---|---|---|
| #23 | Public `mark_polygon` | S-M | Renderer + SceneNode already exist internally; exposure unlocks half-violins/rainclouds and user workarounds for every missing mark below. Highest value-per-hour on the list. |
| #17 | Remove inert `share` dict from Joint/Repeat/ClusterMap `.spec` | S | Dead-API cleanup, zero behavior. |
| #54 | `catplot(dodge="auto")` when hue == categorical axis | S | Off-center single-box fix; surfaced from the catplot golden. |
| #4 | Legend labels fall back to `label_color` | S | Cross-surface theme coherence; one fallback swap. |
| #14 | `configure_axis(label_flush=)` chart-level | S | Field exists in Rust `AxisStyleSpec`; expose in `AxisConfig`. |
| #31 | Eager `scheme=` validation on Scale/Config path + `redblue`/`category10` aliases | S | Completes D-COLOR-1's channel-path work. |
| #59 | Secondary-layer `Axis(orient="left")` silently forced Right | S | **User decision needed first**: warn / typed error / wontfix-by-spec. Repo precedent favors surfacing. |

## Wave H — hygiene & hardening (1–2 sessions)

| Issue | What | Size | Note |
|---|---|---|---|
| #2 | `serde(deny_unknown_fields)` on ~45 transform Specs + ChartConfig | S-M | Kills a silent-drop class; mechanical but wide. |
| #37 | `transform_calculate` wire key `as_field` → `as_` | S | Sibling-drift unification; a pinned test must flip deliberately. |
| #3 | `Option<[f64;2]>` domain replacing sentinel + `domain_user_set` | M | The direct sibling of the shipped #79b `RangeProvenance` pattern — same defended choice ("make unset unrepresentable"), five scale files. |
| #11 | Extract `build_color_detail_groups` (area/line/ribbon) | S-M | Drift-prevention **plus a latent ribbon Int64-color bug risk** (`col_as_str` vs `col_as_ordinal_category_str`) — the one Wave-H item with a possible live defect inside. |
| #12 | Unify group-partition+stack across bin/kde/kde_2d/smooth | M | Extent logic has *already* drifted 4 ways; consolidation is overdue. |

## Wave T — toolchain debt (high compounding leverage)

| Issue | What | Size | Note |
|---|---|---|---|
| #29 | pyo3-deprecation migration → `clippy -D warnings` GREEN | M-L | **Retires the ~166-error baseline every gate in every session works around.** Do this early; every future Rust review gets a real `-D` gate instead of delta-judging. |
| #28 | chart.py decomposition remainder (`_transform_resolve.py`) | M | Completes cohesion-roadmap C2. |
| #30 | Pyright type-debt clusters | M | Background-quality; would quiet the diagnostic noise every session sees. |

## Wave C — composition & dual-axis campaign (the #52 lineage, one spec)

| Issue | What | Size | Note |
|---|---|---|---|
| #56 | Intra-member slot grouping (slot-group id on the wire) | M | Do first — the wire contract #55 also wants; unblocks `mark_line(point=True)` overlays as secondary members. |
| #55 | Dual x-axis (independent per-layer x, top orient) | M-L | Architecturally a mirror of #52; doubles WASM interaction scope (zoom/pan/hit-test across two x domains). |
| #68 | `Resolve(axis=)` — shared/independent axis rendering | M-L | The last unimplemented §3.9 axis; completes the Resolve trio after #16's legend. |
| #33 | Joint/ClusterMap interactive ratio-grid parity (W5 remnant) | M | Needs headless-WASM capture verification (harness in memory). |
| #53 | Native `resolve=` for Joint/ClusterMap | S-M | **Parked by its own text** until a real use case; decide alongside #68. |

## Wave D — diagnostics & ML plots

| Issue | What | Size | Note |
|---|---|---|---|
| #43 | Method-sweep `compare=` for clustering charts | M | The two structurally-excluded charts from #35. |
| #47 | SHAP waterfall anchored at `E[f(x)]` (base_value through schema) | M | Canonical-presentation gap; schema addition. |
| #48 | `_grid_panels` beyond 4 panels | S | **Gated on need** — only do it if #43 (or another caller) actually exceeds 4; otherwise stays parked. |

## Wave V — new visualization capabilities (each a phased campaign: brainstorm → spec → plan)

| Issue | What | Size | Suggested order |
|---|---|---|---|
| #13 + #15 | Grouped filled contours (`level_id` namespacing) + per-cell quantitative color for contour/hex | M each | Pair them — same subsystem; do before #20 (its groundwork overlaps). |
| #20 | Gridded array inputs: `contourf` / `pcolormesh` / `quiver` | L | Highest ML-user demand of the three big ones; pcolormesh is the workhorse. |
| #22 | Treemap / icicle (hierarchical rectangle packing) | L | Rect-based rendering already strong; layout transform is the work. |
| #21 | Sankey / alluvial / variable-width trail | L | Needs a new primitive (variable-width trail); the most novel renderer work — last. |

## Recommended sequence

**Q → H → T(#29) → D → C → V**, with T's #29 pulled as early as tolerable (its payoff compounds across everything after), and Wave V run as separate phase campaigns per feature. Parked pending decision/need: #59 (user call), #53 (use case), #48 (caller demand).

Dependencies worth honoring: #56 before #55 (shared wire contract); #13/#15 before #20 (contour groundwork); #68 and #53 decided together (both are "what does Resolve mean for grid composites"). #67 (refactor, not in this list) remains its own golden-rebless session and is orthogonal to all waves.
