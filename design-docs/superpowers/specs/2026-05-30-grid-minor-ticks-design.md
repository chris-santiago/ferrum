# ferrum.Grid value class + minor-tick subsystem — Design Spec

**Date:** 2026-05-30
**Branch:** `feat/render-gaps-17-19-21`
**Tracks:** code-archaeology item 18 (`design-docs/superpowers/followups/2026-05-15-code-archaeology.md`)
**Contract:** `ferrum-spec.md §3.19` (constructor at lines 1607-1611; Theme example at line 1001)
**Constraint:** No-defer rule — `Grid(minor=True)` must produce real minor gridlines on continuous scales, never a silent no-op.

## 1. Scope

Implement the `ferrum.Grid` theme-level value class per `§3.19`, backed by a real minor-tick generation subsystem. Today gridlines render as a single level (no major/minor concept exists anywhere in the scale engine, layout, theme binding, or `build_grid()`). This adds: a Python `Grid` value class, value-object ingestion in `Theme`, a `Tick { position, is_major }` type in the Rust scale engine, minor-tick generation for continuous scales, per-level (`major_*`/`minor_*`) grid styling through the theme binding, and two-level emission in `build_grid()`.

## 2. Goals

- `ferrum.Grid(...)` constructs per the `§3.19` signature, is exported in `__all__`, and is accepted by `Theme.update(grid=fr.Grid(...))`.
- `minor=True` renders real minor gridlines on continuous scales (linear, log, time, pow, sqrt, symlog), lighter/thinner than major by default.
- Log minor ticks land at the standard 2-9 intra-decade multiples; other continuous scales subdivide each major interval in transformed space.
- Categorical (ordinal/band/point) and discretizing (quantile/threshold/bin-ordinal) scales produce no minor ticks — a documented semantic absence, not an error.
- Charts that do not enable minor render byte-identical SVG to today.

## 3. Non-goals

- Chart-level minor control: `GridConfig` gains no `major_*`/`minor_*` and no API change.
- Minor ticks on categorical/discretizing scales.
- Per-axis (x-only / y-only) minor toggles beyond the grid-wide control `GridConfig` already offers.

## 4. System behavior

**Construction & shorthand.** `Grid(major=True, minor=False, *, color=None, width=None, dash=None, opacity=None, major_color=None, minor_color=None, major_dash=None, minor_dash=None, major_width=None, minor_width=None, major_opacity=None, minor_opacity=None)`. The bare `color`/`width`/`dash`/`opacity` are a **fallback that sets both levels**; an explicit per-level value overrides the bare value for that level only. Examples:
- `Grid(color="#f5f5f5")` → `major_color = minor_color = "#f5f5f5"`.
- `Grid(color="#eee", minor_color="#f8f8f8")` → major `#eee`, minor `#f8f8f8`.

**Theme integration.** `fr.themes.<name>.update(grid=fr.Grid(...))` returns a new theme whose grid styling is set by the `Grid` value. `minor=True` enables minor gridline emission; `minor=False` (default) emits only major (today's behavior).

**Minor rendering.** On a continuous-scale axis with `minor=True`, minor gridlines render between major ticks: lighter and thinner than major when unstyled. On a categorical or discretizing axis, `minor=True` renders no minor lines and raises no error.

**Backward compatibility.** A chart with no minor enabled renders exactly as today. Existing `GridConfig(color=…, width=…, dash=…, opacity=…)` calls behave unchanged and now formally style the **major** level.

## 5. Architecture

Five layers, built in this order:

1. **Scale engine (Rust).** Introduce `Tick { position: f64, is_major: bool }` at the tick-generation boundary. Major output is identical to today's `Vec<f64>` (same positions, `is_major = true`). A new minor path generates minor positions: the default subdivides each major interval into nice sub-steps in the scale's transformed space (linear/pow/sqrt/symlog/time); `log` overrides with 2-9 intra-decade multiples. Categorical/discretizing scales return no minor ticks.
2. **Layout (Rust).** `TickLayout` gains `is_major: bool`. Layout carries minor ticks through to render only when minor rendering is enabled.
3. **Styling + emission (Rust).** `ThemeGrid` / `ThemeColors` / `ThemeRenderSizes` gain `major_*` and `minor_*` grid fields; `ThemeOverridesSpec` gains matching keys; `apply_theme_overrides` wires them. The builtin theme sets derived lighter/thinner minor defaults. `build_grid()` emits major and minor gridline `SceneNode::Line` batches, each styled from its level, minor drawn first (under major).
4. **Python value class.** `src/ferrum/grid.py` frozen dataclass; `to_spec_dict()` resolves shorthand → per-level and emits only non-None per-level keys plus `major`/`minor` booleans. Exported in `src/ferrum/__init__.py __all__`.
5. **Theme ingestion.** `Theme.to_spec_dict()` gains value-object handling: if the `grid` prop has a `.to_spec_dict()` method, call it before serialization (mirrors how charts serialize `Title`).

**Precedence cascade:** `builtin theme defaults → theme Grid (sets both major + minor) → chart-level GridConfig (overrides major only)`. Minor styling comes only from the theme `Grid`. `GridConfig` is otherwise untouched (x/y toggle, band_colors, and now formally the major level).

## 6. Canonical interfaces / data contracts

**Python — `Grid` constructor** (signature is the contract; `§3.19` plus the bare shorthand):
```python
Grid(major=True, minor=False, *,
     color=None, width=None, dash=None, opacity=None,
     major_color=None, minor_color=None,
     major_dash=None, minor_dash=None,
     major_width=None, minor_width=None,
     major_opacity=None, minor_opacity=None)
```
`to_spec_dict()` output keys (all optional except the booleans, all per-level — the bare shorthand is never emitted to Rust): `major: bool`, `minor: bool`, `major_color`, `minor_color`, `major_dash`, `minor_dash`, `major_width`, `minor_width`, `major_opacity`, `minor_opacity`.

**Rust — tick type:**
```rust
struct Tick { position: f64, is_major: bool }
```
Scale tick generation returns major ticks (`is_major = true`, positions unchanged from today) and, when requested, minor ticks (`is_major = false`).

**Rust — per-level theme keys.** `ThemeOverridesSpec` (and the Python theme spec dict) carry the per-level keys matching `Grid.to_spec_dict()`: `major_grid_color`/`minor_grid_color`, `major_grid_width`/`minor_grid_width`, `major_grid_dash`/`minor_grid_dash`, `major_grid_opacity`/`minor_grid_opacity`, plus the existing `grid` on/off and the new `minor` enable. (Exact key spelling is an implementation detail provided the Python `Grid.to_spec_dict()` keys and the Rust deserialization agree.)

## 7. Invariants and constraints

- **Byte-identical non-minor output.** Charts that do not enable minor render byte-identical SVG; minor emission is gated on minor being enabled. Existing goldens are unchanged.
- **Major positions preserved.** The `Tick` refactor must not change any scale's major tick positions.
- **GridConfig unchanged.** Existing `GridConfig(color=…/width=…/dash=…/opacity=…)` behavior is preserved (now formally the major level); no API change to `GridConfig`.
- **Categorical/discrete minor is a no-op, not an error.** `minor=True` on ordinal/band/point/quantile/threshold/bin-ordinal renders no minor lines and does not raise.
- **No new dependency.** Minor-tick generation reuses the existing tick helpers; no new crates.

## 8. Key decisions and tradeoffs

- **Minor algorithm = subdivision with a log override.** Default subdivides major intervals in transformed space (correct for linear/pow/sqrt/symlog/time); log overrides with 2-9 per-decade multiples because evenly-subdividing log space is visually wrong. Chosen over full bespoke per-scale locators (more code/test surface) and pure generic subdivision (log looks wrong).
- **Continuous-only minors.** Categorical/discretizing scales have no continuum to subdivide; `minor=True` there is a documented semantic absence. Honors the no-defer rule (it is not an unimplemented path, it is a meaningless one).
- **Per-level cascade, GridConfig = major.** Theme `Grid` owns both levels; chart `GridConfig` overrides the major level only, keeping today's single gridline backward-compatible and avoiding a `GridConfig` API expansion (YAGNI — `§3.19` asks for theme-level minor control, not chart-level).
- **Shorthand = both-levels fallback.** Matches the spec's own example (`Grid(color="#f5f5f5")`); per-level params override. Resolution happens in `Grid.to_spec_dict()` so Rust only ever sees per-level keys.
- **Derived lighter minor default.** Unstyled minors render lighter/thinner (matplotlib/seaborn convention); defaults live in the builtin theme so they cascade and stay overridable.
- **`Tick` struct over parallel Vec<f64> lists.** A single typed list with `is_major` keeps the major/minor relationship local and avoids threading two parallel collections through layout.

## 9. Acceptance criteria

- `ferrum.Grid` constructs per `§3.19`; shorthand resolves to per-level in `to_spec_dict()`; `Theme.update(grid=fr.Grid(...))` ingests the object; `Grid` is in `__all__`.
- `minor=True` on a continuous-scale chart renders minor gridlines (lighter/thinner by default); a new golden captures it and is visually inspected per CLAUDE.md.
- Log minor ticks land at 2-9 intra-decade multiples; linear/time minors subdivide correctly; categorical/discrete produce none.
- Existing SVG goldens are byte-identical; `cargo test`, `uv run pytest -n auto`, and `cargo clippy` (incl. wasm target) are green.
- `ferrum-spec.md §3.19` gains a dated note clarifying the bare `color=`/`width=` shorthand is a both-levels fallback (the full signature omits it today).
- Archaeology doc item-18 row updated to fixed.

## 10. Validation strategy

- **Python:** behavioral tests for `Grid` construction, shorthand→per-level resolution (bare sets both; per-level overrides), `Theme` value-object ingestion, and `minor=True` reaching the Rust boundary.
- **Rust:** unit tests for minor-tick generation per scale type — linear subdivision counts, log 2-9 placement, time subdivision, categorical/discretizing → empty — plus `Tick.is_major` classification and `build_grid()` two-level emission with correct per-level styling and minor-under-major ordering.
- **Goldens:** byte-equality for existing (non-minor) charts; one new `minor=True` golden, rasterized and visually inspected before commit.

## 11. Open questions

None blocking.
