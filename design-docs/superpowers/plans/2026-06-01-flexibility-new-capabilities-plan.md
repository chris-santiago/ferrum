# Flexibility New Capabilities & Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task. Dispatch Rust edits to `rust-coder`, Python edits to `python-coder`; run `/regression-test` before declaring any task done.
>
> **Scope note:** D6 (reactive parameters) is a large, independent subsystem and the riskiest item in either phase. Recommend executing Tasks 1–4 (D7–D10) first as one pass, then D6 (Task 5) as its own pass — optionally promoted to a standalone plan if it grows.

## 1. Objective

Add the five capability/polish items (D6–D10) that lift the expressive ceiling, per the design spec.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-01-flexibility-new-capabilities-design.md` — full spec
- §4 System behavior; §6 Canonical interfaces; §7 Invariants; §8 Key decisions; §9 Acceptance criteria

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/marks/line.rs` | group into one polyline per `detail` value (D8) |
| Modify | `crates/ferrum-core/src/render/marks/axis.rs`, `marks/legend.rs` | derive titles from source field/explicit title, never internal column (D9) |
| Modify | `crates/ferrum-core/src/render/binding.rs` | carry source-field/title through to title resolution (D9) |
| Modify | `src/ferrum/encoding/__init__.py` | `Theta2`/`Radius2` channels (D7) |
| Modify | `crates/ferrum-core/src/spec/coord.rs` | polar second-extent spec (D7) |
| Modify | `crates/ferrum-core/src/render/scale_resolve/positional.rs`, `render/position.rs` | polar radial/angular second extent + radial stack offset (D7) |
| Modify | `crates/ferrum-core/src/render/marks/arc.rs`, `marks/bar.rs` | annular/wedge segments under polar (D7) |
| Modify | `src/ferrum/composition.py` | `.properties(title=, subtitle=, caption=)` figure chrome (D10) |
| Modify | `crates/ferrum-core/src/render/grid_compose.rs`, `render/compositor.rs` | render figure title/caption band once (D10) |
| Modify | `src/ferrum/selection.py` | `Parameter`/`fm.param`; unify selections; `bind="legend"` (D6) |
| Modify | `src/ferrum/transforms.py` | `transform_filter` accepts `Parameter` predicate (D6) |
| Modify | `src/ferrum/chart.py`, `src/ferrum/encoding/_scale.py` | accept `Parameter` in `scale.domain`, `value`, conditional; serialize `params` section (D6) |
| Modify | `crates/ferrum-core/src/spec/` (params ingestion), `render/` (static resolve to initial value) | static-render parameter resolution (D6) |
| Modify | `crates/ferrum-wasm/`, `src/ferrum/_wasm/` interactive runtime | reactive evaluation: rescale, crossfilter, legend toggle (D6) |
| Test | `tests/test_flexibility_caps/` | per-item regression tests (D6–D10) |
| Modify | `tests/goldens/**` | regen polar/line/title/figure-chrome goldens |

## 4. Constraints

- **Goldens not blessed until visually inspected** (`scripts/snapshot-goldens.py` → Read each PNG) — polar, line-detail, title, and figure-chrome changes move goldens.
- **Static determinism:** parameters resolve to their initial value in SVG; a chart using no parameters renders byte-identically to today.
- **Backward compatibility:** `selection_interval`/`selection_point` keep current signatures + behavior; `CoordPolar` charts binding only `theta`/`radius` unchanged; single-chart `.properties(title=)` unchanged.
- Polar stacking must not regress Cartesian stacking.
- `cargo test` green; `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` clean.
- No matplotlib; no global mutable state.
- Generic channels only — no `fm.sunburst`/`fm.wind_rose` helpers, no `mark_trail` (spec §3).

## 5. Tasks

### Task 1: D8 — `detail` splits `mark_line`
- [ ] Failing test: `mark_line(detail="g")` renders N polylines, no color legend; hand-built parallel coordinates render (spec §9)
- [ ] `line.rs`: group by detail key (mirror parallel-coordinates batcher)
- [ ] Regen + inspect affected goldens
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_caps/test_d8_line_detail.py`

### Task 2: D9 — title hygiene
- [ ] Failing test: no chart surfaces `contour_x`/`hex_x`/`lo` as axis/legend title (spec §9)
- [ ] Derive title from source field/explicit `title=` in `binding.rs` + `axis.rs`/`legend.rs`
- [ ] Regen + inspect affected goldens
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_caps/test_d9_titles.py`

### Task 3: D7 — polar `theta2`/`radius2` + radial stacking
- [ ] Failing tests: stacked radial bars accumulate outward (no r=0 overlap); hand-built sunburst renders nested wedges from laid-out rows (spec §9)
- [ ] `Theta2`/`Radius2` channels (Python); polar second extent + radial stack offset (Rust geometry)
- [ ] Regen + inspect polar goldens
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_caps/test_d7_polar.py`

### Task 4: D10 — figure-level title/caption
- [ ] Failing test: `vconcat/hconcat/facet` with `properties(title=,subtitle=,caption=)` renders chrome once around the figure, not per panel (spec §9)
- [ ] `composition.py` `.properties()` slot; `grid_compose.rs`/`compositor.rs` render band
- [ ] Regen + inspect composite goldens
- [ ] Verify: `cargo test` + `uv run pytest tests/test_flexibility_caps/test_d10_figure_title.py`

### Task 5: D6 — reactive parameter system (own pass)
- [ ] Failing tests assert emitted HTML/JS wiring + scene JSON (no browser): param + references + event bindings present and correct (spec §10)
- [ ] 5a `Parameter`/`fm.param`; unify `selection_*` under it; `bind="legend"` (`selection.py`)
- [ ] 5b Reference sites accept `Parameter`: `scale.domain`, `transform_filter`, conditional, `value` (`chart.py`, `_scale.py`, `transforms.py`)
- [ ] 5c Serialize `params` section + reference markers into spec JSON
- [ ] 5d Static resolver: parameters → initial value in SVG (Rust render); param-free output byte-stable
- [ ] 5e Interactive runtime: reactive rescale (overview+detail), crossfilter row removal, legend toggle (`crates/ferrum-wasm`, `_wasm`)
- [ ] Verify: `cargo test` + `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` + `uv run pytest tests/test_flexibility_caps/test_d6_params.py`

### Task 6: Cross-cutting verification
- [ ] Re-run Phase-B audit categories (multivariate, scientific, categorical, interactive, faceting); confirm blocked designs render/verify
- [ ] Golden inspection sweep; confirm param-free static output byte-stable

## 6. Acceptance checks

- `uv run pytest tests/test_flexibility_caps/ -v` — all pass
- `uv run pytest -n auto` — full suite green
- `cargo test` + wasm clippy — clean
- Spec §9 designs render/verify; regenerated goldens visually confirmed; param-free static charts byte-stable

## 7. Open questions

- **Widget surface (5e):** legend/brush/point selections are in scope; HTML slider/dropdown widgets minimal or deferred — if required now, expands Task 5e.
- **Sunburst hierarchy:** rows are user-precomputed this phase; if a built-in rectangling transform is wanted, it is a new task.
- **Parameter scope:** single composed figure only; cross-figure linking would extend 5c/5e.
