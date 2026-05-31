# Flexibility Fix Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task. Dispatch `.py` work to `python-coder`, `.rs` work to `rust-coder`.

## 1. Objective

Close the 9 cross-cutting defects (D1–D9) surfaced by the v0.13.0 flexibility audit, in the synthesis fix order, so power-user chart designs stop breaking on shared root causes.

> **STATUS (2026-05-31): COMPLETE** on branch `fix/flexibility-campaign`.
> D1+D4 `e12ac67` · D3 `7ddaff7` · D2 `e298e66` · D5 `50bfd52` · D6 `103777b` · D7 `d489a6b` · D8 `43b2e05` · D9 `fcab5ef` · Task 9 legends `7d99276` · Task 2b temporal-inference + item 2 µs/ns `6641ccf`.
> Each TDD-first with regression coverage in `tests/test_flexibility_campaign.py` (76 tests), quality-reviewed, committed individually; Gapminder legend visually inspected.
> **Tracked follow-ups (user-accepted, NOT done):** Item 3 full D2 cross-layer color-domain union (rare; needs Rust `build_color_scale`); two Task-9 nits (translucent-color merge round-trip, dead `.min(0.0)` in `layout/legend.rs`); the capabilities docs page (§8).

## 2. Spec references

- `/tmp/ferrum-ux-audit/SYNTHESIS.md` — defect table (D1–D9), per-defect root cause + file:line, fix order, "don't regress" wins.
- Per-category evidence + repro scripts: `/tmp/ferrum-ux-audit/<category>.md` and `/tmp/ferrum-ux-audit/<category>/`.
- `ferrum-spec.md` — API contract; update with a dated note for any behavior these fixes change (D1 scale range typing, D2 layer-merge semantics, D3 format support).

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/scale_resolve/domain.rs` | D5: `_ => {}` no-op at :80 must handle value-sort `'-x'`/`'-y'` + explicit arrays |
| Modify | `crates/ferrum-core/src/render/scale_resolve/` (color scale build) | D1/D4: accept string `range`; route `scheme` into rect/categorical color resolver |
| Modify | `src/ferrum/_core.pyi`, scale value-classes | D1: `OrdinalScale.range` (:177) typed `list[float]` only; widen to colors |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | D3: `label_format_override: None` hardcoded at :538 — wire per-channel `Axis.label_format` |
| Modify | `crates/ferrum-core/src/render/format.rs` | D3: hand-roll full d3-format number grammar; replace hand-rolled date math with `chrono` strftime |
| Modify | `src/ferrum/composition.py`, `src/ferrum/_layer.py` | D2: order-independent `+` color-scale union + axis-title precedence |
| Modify | mark color routing (`src/ferrum/marks/`, Rust draw) | D6: `color=` → `stroke` on line/segment marks |
| Modify | `src/ferrum/annotation/coords.py` (+ serialization) | D7: accept `date`/`datetime` in annotation coords |
| Modify | axis suppression (Python encoding + Rust layout/axis) | D8: `axis=None` actually hides the axis |
| Modify | legend build (`crates/ferrum-core/src/render/` color-only path) + Python `Size`/`Shape` | size/shape legends (fix-order #5) |
| Modify | facet/scale + multi-line/Inset render paths | D9: blank-render class (12-row facet, ordinal-x multi-line, `Inset` parent) |
| Test | `tests/` (Python) + `crates/ferrum-core` integration tests | one failing-first regression test per defect |

## 4. Constraints

- **TDD-first, every task:** write the failing test that reproduces the audit repro, then minimal fix. The repro lives in the matching `/tmp/ferrum-ux-audit/<category>/` script.
- **/regression-test gate is mandatory after every defect fix**, before commit (project CLAUDE.md + PreToolUse hook). Not optional, not batched.
- **No-defer rule (CLAUDE.md):** implement fully — no warn-fallbacks, no `NotImplementedError`, no "later phase". D1 ships real categorical-color mapping; D3 ships real formatting, not a partial stub.
- **Cross-language ownership:** `python-coder` for `.py`, `rust-coder` for `.rs`. D1, D3, D8, D9, legends span both — split the task, Python handles `.py`, Rust handles `.rs`, with an explicit interface note between them.
- **Don't regress the wins:** confidence bands (`mark_area`/`mark_ribbon` y2), bivariate contours, `Annotate`+`px()` callouts, `clustermap`, `parallel_coordinates_chart`, continuous colorbars, offline WASM export, `BreakAxis`. Add a smoke assertion if a fix touches their code path.
- **Goldens:** any new/changed `tests/goldens/**/*.svg` must be rasterized via `scripts/snapshot-goldens.py` and visually inspected before commit (CLAUDE.md hard constraint).
- **Tasks are independently shippable** — each closes one defect and leaves the tree green; commit per defect.

## 5. Tasks

### Task 0: Design decisions (RESOLVED 2026-05-31 — record in `ferrum-spec.md` with dated notes, then proceed)
- **D1 (categorical color API):** Widen `OrdinalScale.range` (and sibling scale value-classes) to accept color strings, paired with `domain` for category→color (Altair-idiomatic `Scale(domain=[...], range=["#ccc","#e4572e"])`). No new value-class. `QuantizeScale` string handling is the precedent.
- **D2 (`+` merge semantics):** Color scales union their domains (symmetric → `a+b == b+a`). On a true scheme conflict the first encoding-bearing layer wins; **annotation-only layers never supply axis titles** (fixes the `annotate_rect` → `_x1`/`_y1` rename). Axis titles come from the first data-bearing layer.
- **D3 (format scope): full grammar, in Rust, single source of truth.**
  - *Time:* use `chrono` (already a dep) for full `strftime` formatting; **delete the hand-rolled `month_short`/`epoch_ms_to_ymdhms` in `format.rs`.**
  - *Numbers:* hand-roll the full d3-format grammar (`[[fill]align][sign][symbol][0][width][,][.precision][type]`, type chars incl. `s % p r g`, plus the `~` trim flag) in `format.rs`. Use `format_num` (MIT/Apache, v0.1.0, unmaintained) as a *reference only* — do **not** take the dependency (it pulls `regex` and lacks `~`/`g`/`r`/`p`). One formatter feeds SVG, PNG, and the Rust-baked interactive strings identically.
- [ ] Verify: the three decisions written into `ferrum-spec.md` (D1 range typing, D2 merge contract, D3 format support) with dated notes before Task 1.

### Task 1: D1 + D4 — categorical/continuous color routing
- [ ] Failing test: ordinal `range=["#ccc","#e4572e",...]` and `scheme=`/`cmap=` on rect/heatmap honored (synthesis D1, D4; faceting B1, explanatory B2/B5, scientific B5).
- [ ] Fix scale serialization + Rust rect/categorical color resolver + `_core.pyi` typing.
- [ ] Verify: `uv run pytest -n auto tests/ -k "scale or color or heatmap"`; `cargo test`; then `/regression-test`.

### Task 2: D3 — axis label formatting
- [ ] Failing test: per-channel `Axis(label_format="%b %Y")` (temporal, via chrono) and `",.0f"`/`"~s"` (numeric, hand-rolled grammar) applied; `tick_count` respected (synthesis D3; explanatory B1, timeseries B3).
- [ ] Wire `prepare.rs:538` `label_format_override` from the encoding's `Axis.label_format`; route time through `chrono` strftime; implement the d3-format number grammar in `format.rs` and delete the hand-rolled date math.
- [ ] Verify: `cargo test`; pytest temporal/numeric axis tests; `/regression-test`.

### Task 3: D2 — order-independent layer merge
- [ ] Failing test: `base + highlight` and `highlight + base` produce identical color scale + axis titles (synthesis D2; distributions, explanatory B3, timeseries, categorical).
- [ ] Verify: pytest composition/layer; `/regression-test`.

### Task 4: D5 — value-sort + composite-mark sort
- [ ] Failing test: `sort='-x'` orders by value; `sort=[...]` and value-sort forwarded to categorical domain on box/violin/swarm (synthesis D5; categorical #1, distributions B2).
- [ ] Fix `domain.rs:80` arm + desugar `sort` forwarding.
- [ ] Verify: `cargo test`; pytest sort/catplot; `/regression-test`.

### Task 5: D6 — line/segment color→stroke
- [ ] Failing test: `mark_line(color=...)`/`mark_segment(color=...)` set stroke, not fill (synthesis D6; timeseries, explanatory B5).
- [ ] Verify: pytest marks; `/regression-test`.

### Task 6: D7 — datetime annotation coords
- [ ] Failing test: `annotate_*` accepts `date`/`datetime` (+ ISO string) without manual epoch-ms (synthesis D7; explanatory B6, timeseries B4).
- [ ] Verify: pytest annotations; `/regression-test`.

### Task 7: D8 — `axis=None` hides axis
- [ ] Failing test: `axis=None` removes the axis single-layer and layered (synthesis D8; distributions, categorical).
- [ ] Verify: pytest axis + golden inspect; `/regression-test`.

### Task 8: D9 — blank-render class
- [ ] Failing tests, one each: 12-row facet at default size, ordinal-x multi-line (>8 x-values), `Inset` parent (synthesis D9; distributions, multivariate B4, timeseries).
- [ ] Verify: pytest + rasterize/inspect affected goldens; `/regression-test`.

### Task 9: size/shape legends
- [ ] Failing test: `Size`/`Shape` encodings emit a graduated/symbol legend (synthesis ceiling; multivariate B1).
- [ ] Extend legend build beyond `color_scale` + Python `legend=` wiring.
- [ ] Verify: `cargo test`; pytest legend + Gapminder repro golden inspect; `/regression-test`.

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass (incl. new per-defect regression tests).
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass.
- Re-run `/audit-flexibility` and diff the new `SYNTHESIS.md` vs the v0.13.0 baseline: D1–D9 no longer appear as cross-cutting defects; no "don't regress" win flipped to broken.
- `ferrum-spec.md` updated with dated notes for D1/D2/D3 behavior changes.

## 7. Open questions

- D1 API shape and D2 merge semantics are the two decisions most likely to change downstream task code — resolve in Task 0 before starting Task 1/Task 3. **(RESOLVED 2026-05-31, see Task 0.)**
- Size/shape legend (Task 9) is the largest single item; if it balloons, split into its own plan rather than blocking D1–D8.

## 8. Surfaced during execution (decisions / follow-ups)

- **D5 sort op (RESOLVED):** `'-y'` shorthand defaults to sum, but sort is NOT sum-only — also support explicit `sort=[...]` arrays and the Vega-Lite `sort={"field":..,"op":"mean"/"max"/..,"order":..}` form for arbitrary ordering.
- **Size legend (RESOLVED):** ~5 nice round representative values. **Multi-legend (RESOLVED):** stack vertically, merge channels that encode the same field.
- **Item 1 — temporal auto-type inference (DECISION 2026-05-31: AUTO-INFER).** A polars `Datetime`/`Date` column without `:T` resolved to a Linear scale (raw epoch ticks). → Becomes **Task 2b**: `_coerce.py` / type inference auto-detects polars `Datetime`/`Date` dtypes → temporal scale (Vega-Lite-consistent). Requires golden refresh + visual inspection (any date columns previously rendered numeric will flip).
- **Item 2 — arrow_cast µs/ns timestamp units (DECISION: FOLD IN as hardening).** `Timestamp(Microsecond/Nanosecond)` read as raw `f64` without normalizing to epoch-ms; safe today only because `_coerce.py` pre-casts to `Datetime("ms")`. Defensive normalization in `crates/ferrum-core/src/render/arrow_cast.rs`, done alongside Task 2b (it lives in the same temporal area).
- **Item 3 — D2 cross-layer color-domain union (DECISION: ACCEPT + TRACK).** The Python fix closes the documented D2 case (gray base + colored highlight). The rarer case — two layers BOTH coloring the same field with disjoint category sets — is NOT a true union (resolves to one layer's categories); full union needs a high-blast-radius Rust `build_color_scale` change. **Not done this campaign — tracked follow-up.**
- **Follow-up — capabilities docs page (user request 2026-05-31):** once the campaign lands, build a docs page showcasing the power-user / well-known-example plots (Gapminder, raincloud, candlestick, slopegraph, ridgeline, contourf, linked brushing, etc.) to highlight the now-working capabilities. Source material: the audit repro scripts in `/tmp/ferrum-ux-audit/<category>/`. Do AFTER fixes land so every example renders correctly.
