# Refactor batch — spec + plan (2026-07-24)

Source: the 10 open `refactor` issues, run through coherent-change batch mode (consolidated research 2026-07-24; defended choices approved in-session). Scope decision: waves 1–3 now; **#67 deferred to its own session** (must land after #85 — orthogonal factors of the same width formula; #67's mass golden regen re-freezes everything once). #85 ships as **silent reinterpretation + prominent changelog** (no FutureWarning cycle).

Branch: `refactor/open-refactor-batch`. All items byte-stable except W3 (#85, small golden set). Gates per task: coder dispatch → review-lite → commit; whole-change design review + close at the end.

## Dispositions

- **#62 — CLOSE, already resolved** by `601017b8`: `resolve_layer_y_slot_scale` (`scale_resolve/mod.rs:1207`) is the single synthesis site; `prepare/mod.rs:892` and `scene_build.rs:671` both delegate. Close with evidence, no code.
- **#67 — stays open**, annotated: blocked-on/sequenced-after #85; spec direction unchanged (delete `band_extent_or` + 10 callers, `explicit_pixel_range` + gates, `categorical_positions` None-arms `axis.rs:907-911`/`:1380`; add `tick_projection` mutual-exclusivity `debug_assert`; full golden rebless).

## Wave 1 — quick wins, byte-stable

| Task | Issue | Change | Files | Done-check |
|---|---|---|---|---|
| T1 | #61 | Collapse `StructuralOutput` 4-tuple → 2 live fields (`extra_annotations`, `break_results`); fix destructuring at `:352`/`:361` | `crates/ferrum-core/src/render/scene_build.rs:1732-1828` | cargo green; grep shows no `extra_axes`/`extra_mark_batches` |
| T2 | #64 | Per-layer clone gets a self-describing single-slot `y_slots` (add `YScaleSlots::single()` mirroring existing accessor idiom) instead of the stale full-slot copy | `scene_build.rs:1047-1054`, `scale_resolve/mod.rs:712-751` | cargo green; new unit test asserts the clone's `y_slots` is single-slot |
| T3 | #57 | Run existing `_contains_independent_y_layer` (`composition.py:767`) conflict guard in `_build_grid_tree` so Repeat/Joint/ClusterMap grid paths raise the same typed ValueError as `_lower_any:891-900` | `src/ferrum/composition.py:1050+` | RED-proven test: `RepeatChart(indep_y_template, resolve={"y":"shared"})` raises; Joint/ClusterMap siblings covered |
| T4 | #41 | `scripts/regen-scale-wire-baseline.py`: ref-pinned `git worktree` build (default last tag), reuses `_build_baseline_charts()`, writes fixture + `_provenance` key; docstrings at test + loader explaining why working-tree regen defeats the guard | new script; `tests/test_scale_spec_parity.py:94-106`; `tests/_fixtures/scale_wire_baseline.json` | `--check` mode passes against current fixture; provenance key present; test docstring updated |
| T5 | #86 | `BatchPositionMeta` in `position.rs` (3 metadata keys ONLY — offset-column readers stay): `from_batch()`, typed accessors replacing `n_dodge_groups`/`dodge_sub_band_px`/`stack_value_on_x` free fns, `stamp()` used by `apply_dodge_ordinal:620-623` + `apply_stack:980-983`; `_KEY` consts private; migrate consumers (bar/rect/tick/area + test stamp sites) | `crates/ferrum-core/src/render/position.rs:3476-3602`, `marks/{bar,rect,tick,area}.rs` | cargo green byte-stable; grep: no external use of the old free fns or `_KEY` consts |

## Wave 2 — structural, byte-stable

| Task | Issue | Change | Files | Done-check |
|---|---|---|---|---|
| T6 | #79a | `SlotRescaleCtx<'a> { panel_affine, slot_rescales, panel_slot_counts }` threaded through wasm hit-test/render seam (`hit_test_slot_aware:262`, `nearest_slot_aware:300`, `composed_slot_affine:334`, `upload_transform_and_render` render.rs:797, `hit_test.rs` family); context-bundle for `resolve_panel_scales` (10 args, scene_build.rs:536) + `resolve_layer_y_slot_scale` (9 args); retire the 5 `too_many_arguments` allows (mod.rs:1206, scene_build.rs:535/:650, spatial_index.rs:261, render.rs:796). Exemplars: `DrawCtx` draw.rs:17, `LeafScaleContext` seam.rs:70 | ferrum-core `scale_resolve/mod.rs`, `scene_build.rs`; ferrum-wasm `spatial_index.rs`, `hit_test.rs`, `render.rs` | cargo + wasm tests green; zero `too_many_arguments` allows at those fns; wasm clippy green |
| T7 | #63 | Encapsulate, don't reindex: named accessors kill hand-computed `slot - 1` at consumer sites (e.g. `secondary_index(slot)`/lookup on `YScaleSlots` + wasm seam helpers); keep the documented split-convention note (`scale_resolve/mod.rs:772-776`) as contract | same files as T6 (sequence after T6; same seam) | grep: no bare `slot - 1`/`.skip(1)` slot arithmetic at consumers outside the accessors |
| T8 | #79b | Typed range-provenance replacing `range_user_set`+`explicit_pixel_range` dual booleans, modeled on `StackValueAxis` (`spec/position.rs:41`, `Option<Enum>`/default = byte-stable legacy). Writers: `new_internal:239`, pyclass `new:417`, `with_explicit_range:251`; readers: `.range` getter `:460`, `to_scale_spec` `:364-374`, three gated fns `:262/:280/:342`. Replace the interim comment-contract `:193-212`. Wire/`.range` getter behavior byte-identical (scale-parity suite is the guard) | `crates/ferrum-core/src/scale/ordinal.rs`, `render/scale_resolve/positional.rs:263/288/315` | cargo green; `tests/test_scale_spec_parity.py` byte-identical; grep: both boolean fields gone |

## Wave 3 — semantic + small golden set

| Task | Issue | Change | Files | Done-check |
|---|---|---|---|---|
| T9 | #85 | Unify `band_size` on full-width (rect convention), silent reinterpret: tick's 4 ordinal sites drop the `×2` (`tick.rs:86/97, 156/176-178, 210/226-228, 270/282`; reconcile the #66 clamp at `:166/:215`); default `unwrap_or(0.3)`→`0.6`; caps `composite.py:268/275/440/447` 0.3→track `band` (review addendum: caps follow user `size=`); delete median `band / 2` compensation `:289-294`; rewrite `mark_tick` docstring `chart.py:1061-1064`; changelog callout. Rug modes unaffected (use `tick_size`, verified) | `tick.rs`, `composite.py`, `chart.py`, affected goldens (boxplot/errorbar/strip set) | Default boxplot output byte-identical where already correct (median==box width pins hold); regen + visually bless changed goldens; `test_bug_hunt_boxplot_median_width.py` green |

## Close

Per-item review-lite gates; whole-change `python-design-reviewer` + `rust-design-reviewer`; full pytest + cargo + wasm; close #57/#61/#63/#64/#79/#85/#86/#41 with evidence, #62 as already-resolved; annotate #67 with the sequencing note; archaeology doc updated.
