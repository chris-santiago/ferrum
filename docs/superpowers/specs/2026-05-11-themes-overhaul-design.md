# Themes Overhaul — Design Spec

**Date:** 2026-05-11
**Status:** approved, awaiting implementation plan
**Slug:** `themes-overhaul`
**Sub-phases:** T1 (plumbing) → T2 (wiring) → T3 (gridlines + palette) → T4 (defaults + builtins + padding)

---

## Motivation

The gallery audit (`/.claude/skills/gallery-audit/output/`) shows ferrum's default chart output as visually weak next to seaborn / sklearn / yellowbrick. Investigation surfaced three layered causes:

1. **The Theme system is half-wired.** Python's `Theme(...)` class accepts ~35 keys per `ferrum-spec.md §3.13`. The Rust `theme_from_dict` binding (`crates/ferrum-core/src/render/binding.rs:79`) reads only 8 of them — every other key is silently dropped. Four of the eight built-in themes (`dark`, `economist`, `fivethirtyeight`, `solarized_*`) declare `color_scheme=...` keys that never reach the renderer, so they render essentially identical to `default`. The named themes are mostly placebo today.
2. **Gridlines are not implemented.** `theme.grid: true` is the default, and the `grid_color`/`grid_width` keys exist on `ThemeInputs`, but no code in `crates/ferrum-core/src/render/` reads `theme.grid_color` or emits gridline strokes. No chart shows a grid, regardless of theme.
3. **Default visual choices read as weak.** Default `mark_color` is Okabe orange `#E69F00` (washed out against white), no `font_family` is set (resvg falls back to a serif), and quantitative scales pin marks to axis edges (the ROC curve touches the top edge; histograms touch the top).

This spec rebuilds the theme system so it actually delivers the contract `ferrum-spec.md §3.13` already promises, ships a modern Observable Plot–flavored default, and fixes the axis-overshoot polish issue alongside.

### What this spec does NOT do

- Does not address chart-construction defaults that the gallery audit flagged (missing AUC annotations, missing per-cell confusion-matrix counts, missing KDE overlays). Those are gallery-fixer territory.
- Does not introduce new spec keys beyond what `ferrum-spec.md §3.13` already lists. Every plumbed key is already in the spec's `Theme(...)` block.
- Does not add per-axis grid control (`grid_x` / `grid_y`). Decision: stay spec-compliant with the single `grid: bool`.

---

## Approach

**Approach A — Spec-complete Theme overhaul.** Plumb every spec-listed Theme key through `ThemeInputs` and the Rust binding (no silent drops). Implement gridline rendering. Implement a `color_scheme` palette registry. Update `ThemeInputs::default()` and rebuild the 8 builtins so each looks distinct. Fix axis overshoot via scale padding alongside.

Alternatives considered and rejected:
- **B — "Scoped to visible polish":** leave `color_scheme` as silent drop. Rejected because four of the eight named themes rely on it, and the project's `feedback_no_defer_phase_9_plus.md` rule + CLAUDE.md §"Implementation philosophy" forbids "defer to follow-up" as scope reduction.
- **C — "Default-only, defer 7 builtins":** rejected because the user explicitly chose "fix all 8 in this pass" during brainstorming.

---

## Section 1 — Theme key plumbing

Every key listed in `ferrum-spec.md §3.13`'s `Theme(...)` block gets a real home in `ThemeInputs` and the `theme_from_dict` binding. The binding **errors on unknown keys** instead of silently dropping (closes the current placebo trap).

### Key matrix (Python → Rust)

| Python `Theme(...)` key | Rust `ThemeInputs` field | Type | Status |
|---|---|---|---|
| `background` | `background_color` | `Srgba<u8>` | already plumbed |
| `padding` | `padding` | `f64` | already plumbed |
| `font_family` | `font_family` | `String` | **NEW**, threaded into every `TextStyle` |
| `font_size` | `label_font_size` | `f64` | wire existing field |
| `font_weight` | `font_weight` | `String` | **NEW**, default `"normal"` |
| `font_color` | `font_color` | `Srgba<u8>` | in struct, not in binding — wire |
| `title_font_family` | `title_font_family` | `String` | **NEW**, falls back to `font_family` |
| `title_font_size` | `title_font_size` | `f64` | wire existing field |
| `title_font_weight` | `title_font_weight` | `String` | **NEW**, default `"600"` (semibold) |
| `title_color` | `title_color` | `Srgba<u8>` | **NEW**, falls back to `font_color` |
| `title_anchor` | `title_anchor` | `TextAnchor` enum | **NEW**, `Start \| Middle \| End`, default `Start` |
| `title_offset` | `title_offset` | `f64` | **NEW**, default 6.0 |
| `label_font_family` | `label_font_family` | `String` | **NEW**, falls back to `font_family` |
| `label_color` | `label_color` | `Srgba<u8>` | **NEW**, falls back to `font_color` |
| `grid` | `grid` | `bool` | in struct, consumed by new gridline renderer |
| `grid_color` | `grid_color` | `Srgba<u8>` | in struct, consumed by new gridline renderer |
| `grid_dash` | `grid_dash` | `Option<Vec<f64>>` | **NEW**, `None` = solid |
| `grid_width` | `grid_width` | `f64` | in struct, consumed by new gridline renderer |
| `grid_opacity` | `grid_opacity` | `f64` | **NEW**, default 1.0 |
| `axis_line` | `axis_line` | `bool` | **NEW**, default `true`; false suppresses axis stroke |
| `axis_line_width` | `axis_line_width` | `f64` | in struct, wire it |
| `axis_line_color` | `axis_line_color` | `Srgba<u8>` | in struct, wire it |
| `tick_size` | `tick_size` | `f64` | already plumbed |
| `tick_width` | `tick_width` | `f64` | **NEW**, default 1.0 |
| `tick_color` | `tick_color` | `Srgba<u8>` | in struct, wire it |
| `color_scheme` | `color_scheme` | `String` | **NEW**, name resolved against palette registry (Section 5) |
| `mark_color` | `mark_color` | `Srgba<u8>` | already plumbed |
| `point_size` | `point_size` | `f64` | already plumbed |
| `point_opacity` | `point_opacity` | `f64` | **NEW**, default 1.0; distinct from `default_opacity` |
| `line_stroke_width` | `line_stroke_width` | `f64` | already plumbed |
| `bar_corner_radius` | `bar_corner_radius` | `f64` | already plumbed |
| `area_opacity` | `area_opacity` | `f64` | already plumbed |
| `opacity` | `default_opacity` | `f64` | wire alias |
| `legend_orient` | `legend_orient` | `LegendOrient` | already plumbed |
| `legend_direction` | `legend_direction` | `LegendDirection` enum | **NEW**, default `Vertical` for right/left orient, `Horizontal` for top/bottom |
| `legend_title_font_size` | `legend_title_font_size` | `f64` | **NEW**, default `title_font_size` |
| `axis_title_padding` | `axis_title_padding` | `f64` | already plumbed |
| `column_padding` | `column_padding` | `f64` | already plumbed |
| `row_padding` | `row_padding` | `f64` | already plumbed |
| `width` / `height` | (`RenderConfig`, not theme) | `f64` | stays in `RenderConfig`; `Theme.update(width=...)` hands off via `RenderConfig::from_theme()` |

### Behavioral guarantees

- **Unknown keys raise at `Theme(...)` construction time** (Python-side validation against a known-key set), not at first render. Better DX than late failure. A unit test asserts the Python known-key set matches the keys the Rust binding accepts — single source of truth.
- **Fallbacks happen Python-side in `Theme.to_theme_inputs_dict()`** before handing to Rust: if `title_color` is unset but `font_color` is set, `font_color` is copied into `title_color` in the dict. Same chain for `label_color → font_color`, `title_font_family → font_family`, `label_font_family → font_family`. Rust sees a fully-populated dict; no Option fallback chains at the binding layer.
- Adding a new spec key → add it in `ThemeInputs`, in `theme_from_dict`, and to the Python known-key set. Three places, all covered by the round-trip test.

---

## Section 2 — New `ThemeInputs::default()`

The default ships an Observable Plot–flavored visual identity. Below shows the literal Rust struct; comments explain departures from today's defaults.

```rust
impl Default for ThemeInputs {
    fn default() -> Self {
        let mark_blue = Srgba::new(0x4C, 0x78, 0xA8, 0xFF);   // tableau blue
        let text_222  = Srgba::new(0x22, 0x22, 0x22, 0xFF);
        let label_555 = Srgba::new(0x55, 0x55, 0x55, 0xFF);
        let axis_888  = Srgba::new(0x88, 0x88, 0x88, 0xFF);
        let grid_ddd  = Srgba::new(0xDD, 0xDD, 0xDD, 0xFF);
        let bg_white  = Srgba::new(0xFF, 0xFF, 0xFF, 0xFF);
        let strip_bg  = Srgba::new(0xF0, 0xF0, 0xF0, 0xFF);

        Self {
            // Canvas
            padding: 16.0,
            column_padding: 12.0,
            row_padding: 12.0,
            background_color: bg_white,

            // Typography
            font_family: "DejaVu Sans".into(),
            font_size: 11.0,
            font_weight: "normal".into(),
            font_color: text_222,
            label_font_family: "DejaVu Sans".into(),
            label_font_size: 11.0,
            label_color: label_555,
            title_font_family: "DejaVu Sans".into(),
            title_font_size: 13.0,
            title_font_weight: "600".into(),
            title_color: text_222,
            title_anchor: TextAnchor::Start,
            title_offset: 6.0,
            axis_title_padding: 8.0,

            // Grid
            grid: true,
            grid_color: grid_ddd,
            grid_width: 0.5,
            grid_dash: None,
            grid_opacity: 1.0,

            // Axes
            axis_line: true,
            axis_line_color: axis_888,
            axis_line_width: 1.0,
            tick_size: 4.0,
            tick_width: 1.0,
            tick_color: axis_888,

            // Marks
            mark_color: mark_blue,
            point_size: 36.0,
            point_size_min: 4.0,
            point_size_max: 36.0,
            point_opacity: 1.0,
            line_stroke_width: 1.5,
            bar_corner_radius: 0.0,
            area_opacity: 0.35,
            default_opacity: 1.0,
            opacity_min: 0.1,
            opacity_max: 1.0,

            // Color scheme
            color_scheme: "tableau10".into(),

            // Strip (facet headers)
            strip_background_color: strip_bg,
            strip_text_size: 12.0,
            strip_padding: 6.0,

            // Legend
            legend_orient: LegendOrient::Right,
            legend_direction: LegendDirection::Vertical,
            legend_title_font_size: 11.0,
        }
    }
}
```

### Notable departures from current defaults

- `mark_color`: `#E69F00` Okabe orange → `#4C78A8` tableau blue
- `grid_color` / `grid_width`: `#EEEEEE` / 1.0 (invisible) → `#DDDDDD` / 0.5 (faint but visible)
- `font_family`: unset (serif fallback) → `"DejaVu Sans"`
- `title_anchor` / `title_font_weight`: unset / unset → `Start` / `"600"` (semibold left-aligned)
- `point_size`: 30 → 36
- Three-stop text-color ramp: body `#222` / label `#555` / axis `#888`

---

## Section 3 — The 8 built-in themes

Each builtin sets only the keys that differ from `default()`. Unset keys inherit.

### `default`
```python
default = Theme()
```

### `minimal`
```python
minimal = Theme(
    grid=False,
    axis_line=False,
    tick_size=0,
    padding=24,
    label_color="#888888",
)
```

### `dark`
```python
dark = Theme(
    background="#1a1a2e",
    font_color="#e6e6e6",
    label_color="#b8b8c8",
    title_color="#ffffff",
    axis_line_color="#555566",
    tick_color="#555566",
    grid_color="#2a2a3e",
    grid_width=0.5,
    mark_color="#7fb3d5",
    color_scheme="dark2",
    strip_background_color="#252540",
)
```

### `publication`
```python
publication = Theme(
    background="#ffffff",
    grid=False,
    axis_line_color="#000000",
    axis_line_width=1.0,
    tick_color="#000000",
    font_color="#000000",
    label_color="#000000",
    title_color="#000000",
    title_font_weight="bold",
    title_anchor="middle",
    font_family="DejaVu Serif",
    title_font_family="DejaVu Serif",
    label_font_family="DejaVu Serif",
    mark_color="#000000",
    color_scheme="tableau10",
    point_size=24,
)
```

### `economist`
```python
economist = Theme(
    background="#d3e0e6",
    font_color="#1a1a1a",
    title_color="#c00000",
    title_font_weight="bold",
    title_anchor="start",
    axis_line=False,
    grid_color="#b0c4cc",
    grid_width=0.6,
    mark_color="#005a8c",
    color_scheme="set1",
    strip_background_color="#bfd4dc",
)
```

### `fivethirtyeight`
```python
fivethirtyeight = Theme(
    background="#f0f0f0",
    font_color="#333333",
    label_color="#555555",
    axis_line=False,
    tick_color="#999999",
    grid_color="#cbcbcb",
    grid_width=1.0,
    mark_color="#fc4f30",
    color_scheme="set1",
    title_font_weight="bold",
    title_anchor="start",
)
```

### `solarized_light`
```python
solarized_light = Theme(
    background="#fdf6e3",
    font_color="#586e75",
    label_color="#657b83",
    title_color="#073642",
    title_font_weight="bold",
    grid_color="#eee8d5",
    grid_width=0.6,
    axis_line_color="#93a1a1",
    tick_color="#93a1a1",
    mark_color="#268bd2",
    color_scheme="set2",
)
```

### `solarized_dark`
```python
solarized_dark = Theme(
    background="#002b36",
    font_color="#93a1a1",
    label_color="#839496",
    title_color="#fdf6e3",
    title_font_weight="bold",
    grid_color="#073642",
    grid_width=0.6,
    axis_line_color="#586e75",
    tick_color="#586e75",
    mark_color="#268bd2",
    color_scheme="set2",
    strip_background_color="#073642",
)
```

### Distinctness across the 8

| Theme | Background | Font family | Grid | Axis line | Mark | Title anchor |
|---|---|---|---|---|---|---|
| `default` | white | sans | faint | on | tableau-blue | start |
| `minimal` | white | sans | **off** | **off** | tableau-blue | start |
| `dark` | `#1a1a2e` | sans | dark | on | `#7fb3d5` | start |
| `publication` | white | **serif** | **off** | on (bold) | black | **middle** |
| `economist` | `#d3e0e6` | sans | on | **off** | `#005a8c` | start |
| `fivethirtyeight` | `#f0f0f0` | sans | on | **off** | `#fc4f30` | start |
| `solarized_light` | `#fdf6e3` | sans | cream | on | `#268bd2` | start |
| `solarized_dark` | `#002b36` | sans | dark | on | `#268bd2` | start |

A `tests/themes/test_eight_themes_distinct.py` test renders the same bar chart through all 8, asserting all SVGs are byte-distinct, with goldens at `tests/goldens/theme_gallery/{name}.svg`.

---

## Section 4 — Gridline rendering

**Where it lives.** New function `draw_grid()` in `crates/ferrum-core/src/render/marks/axis.rs`. Called once per panel from `render::draw::draw_panel()`.

**Render order inside a panel (load-bearing):**
```
1. background fill
2. gridlines               ← NEW (back layer)
3. mark layers             (existing)
4. axis lines + ticks + tick labels    (existing)
5. axis titles, legend, strip titles   (existing)
```

Gridlines under marks (Plot/Vega convention), but axis lines on top of gridlines so the L-shape stays crisp at intersections.

**What it draws.** For each of the two axes:
- Skip if `!theme.grid` or `axis_layout.ticks.is_empty()`.
- For each `tick.position`, emit a line spanning the opposite axis range.
- Style: `stroke=theme.grid_color`, `stroke-width=theme.grid_width`, `stroke-opacity=theme.grid_opacity`, `stroke-dasharray=theme.grid_dash` (omit when `None`).
- Skip the gridline that coincides with the axis line itself (avoids double-stroke at the origin corner).

**Tick-source guarantee.** Reuses the existing `AxisLayout.ticks` list — no recompute. Gridlines and tick labels are always aligned by construction.

**Faceting.** Each panel calls `draw_grid()` against its own `AxisLayout`. Shared-x/y facets get aligned gridlines for free because tick positions come from the shared scale.

**Sketch:**
```rust
pub fn draw_grid(
    out: &mut SvgWriter,
    plot: Rect,
    axis_x: &AxisLayout,
    axis_y: &AxisLayout,
    theme: &ThemeInputs,
) {
    if !theme.grid { return; }
    let stroke = theme.grid_color;
    let w = theme.grid_width;
    let dash = theme.grid_dash.as_deref();
    let opacity = theme.grid_opacity;

    for tick in &axis_x.ticks {
        if (tick.position - plot.x).abs() < 0.5 { continue; }
        out.line(tick.position, plot.y, tick.position, plot.y + plot.h,
                 stroke, w, dash, opacity);
    }
    for tick in &axis_y.ticks {
        if (tick.position - (plot.y + plot.h)).abs() < 0.5 { continue; }
        out.line(plot.x, tick.position, plot.x + plot.w, tick.position,
                 stroke, w, dash, opacity);
    }
}
```

**Interactions with other theme keys:**
- `grid=false` → no gridlines (used by `minimal`, `publication`).
- `axis_line=false` + `grid=true` → gridlines normal, axis stroke suppressed (used by `economist`, `fivethirtyeight`).
- `grid_dash=[3,3]` → dashed gridlines (no builtin uses this initially; key is wired).

**Per-axis grid (`grid_x`/`grid_y`):** out of scope. Single `grid: bool` per spec. Economist will show both axes' gridlines.

---

## Section 5 — `color_scheme` palette registry

**Where it lives.** New module `crates/ferrum-core/src/render/palette.rs`. Single public function:
```rust
pub fn resolve_scheme(name: &str) -> Result<&'static [Srgba<u8>], PaletteError>;
```

Called from the existing categorical color resolution path in `render/scale_resolve.rs` whenever a color encoding has no explicit `range` and the encoding data type is nominal/ordinal.

### Registry contents

`ferrum-spec.md §3.6` lists 7 categorical + 12 sequential + 6 diverging + 2 cyclical = 27 named schemes. This spec ships **all 7 spec-listed categorical schemes** as new Rust-side `const &[Srgba<u8>]` tables in `render/palette.rs`. Sequential schemes already exist via the existing Rust `ContinuousScheme` infra (`src/ferrum/schemes.py` exposes `viridis/plasma/magma/inferno/cividis`); the palette registry delegates to that infra for `color_scheme="viridis"` etc. Diverging and cyclical schemes are routed through `ScaleDiverging` (existing path), not `color_scheme`, so they're out of scope for this overhaul.

**Categorical (new Rust tables in `palette.rs`):**

| Name (spec §3.6) | Length | Source |
|---|---|---|
| `okabe_ito` | 8 | Okabe-Ito (colorblind-safe; the spec's stated categorical default) |
| `tableau10` | 10 | Tableau 10 (Vega `tableau10`) |
| `set1` | 9 | ColorBrewer Set1 |
| `set2` | 8 | ColorBrewer Set2 |
| `paired` | 12 | ColorBrewer Paired |
| `pastel` | 9 | ColorBrewer Pastel1 |
| `dark2` | 8 | ColorBrewer Dark2 |

**Sequential (delegated to existing `ContinuousScheme`):** `viridis`, `plasma`, `magma`, `inferno`, `cividis`. `resolve_scheme("viridis")` evaluates the existing scheme at 10 stops for categorical use and returns the lookup; for true sequential color encodings the existing `ContinuousScheme` interpolation path is used as today.

**Default flip.** Spec §3.6 names `okabe_ito` as the default categorical scheme. The new `ThemeInputs::default()` (Section 2) sets `color_scheme="tableau10"` instead, matching the Observable Plot aesthetic the user chose during brainstorming. This is a deliberate divergence recorded in the dated `§3.13` spec note (Section 8) — `okabe_ito` remains shipped and accessible via `Theme(color_scheme="okabe_ito")` for users who want the colorblind-safe default back.

### Resolution rules (in `scale_resolve.rs`)

For a categorical color encoding without an explicit `range`:
1. `palette = resolve_scheme(theme.color_scheme)`
2. For each unique category value (stable order from the data): assign `palette[i % palette.len()]`.
3. Wrap-around emits a one-time `RenderWarning::PaletteWrap { scheme, n_categories }`.

For a sequential color encoding (quantitative → color), `viridis` / `magma` are evaluated at the domain extremes and blended between the 10 fixed stops using linear interpolation in sRGB.

**Single-series charts** (no `color=` channel): still use `theme.mark_color`. `color_scheme` is not consulted. Preserves single-color theme identity (e.g. Economist's `#005a8c`).

**Multi-series charts**: `color_scheme` drives categorical assignment; `theme.mark_color` is ignored.

### Errors

`resolve_scheme("does-not-exist")` returns `PaletteError::UnknownScheme(name)`. The binding validates **eagerly in `theme_from_dict`** — fails fast, no silent drops.

### Out of scope

- User-defined inline palettes (`color_scheme=[...]` as a list). Spec defines `color_scheme` as a string name.
- Diverging schemes (RdBu etc.). Covered by explicit `range` on encodings, not by `color_scheme`.
- User-stop interpolated palettes. Each palette is a fixed lookup.

---

## Section 6 — Scale padding (axis overshoot fix)

Sibling to the theme work — not a Theme key, but the same "default polish" pass.

**Problem.** Quantitative scales today nice-extend the domain to `[nice_min, nice_max]` but reserve no visual breathing room. ROC line touches the top edge; histogram bars touch the top; floating `0` / `55` ticks render outside the data extent.

**Change.** For quantitative axes without a user-specified `scale.domain`:
```rust
let pad = (plot.h * 0.05).min(8.0);   // 5% of plot dimension, capped at 8px
y_scale.range = [plot.y_top + pad, plot.y_bottom - pad];
```
Analogously for x. Data domain maps to the inner 90% of the plot range.

**Tick filtering.** After padding is applied, ticks falling in the padding band are dropped:
```rust
ticks.retain(|t| t.position is inside [inner_start, inner_end])
```
A histogram with data `[5, 50]` and nice domain `[0, 55]` keeps `5/10/.../50` and drops `0` and `55` if they fall in the padding band.

**Escape hatches:**
1. User-specified `scale.domain` → padding suppressed; explicit domains honored exactly.
2. Categorical / ordinal axes → no padding (band scales already half-step pad).
3. `include_zero=True` encodings → padding applies, but the `0` tick is preserved if it falls in the padding band.

**Per-axis opt-out** via the existing `Scale.padding` parameter (already listed in `ferrum-spec.md §3.6`). New behavior: the default for quantitative scales flips from `None` → `0.05`. Set `Scale(padding=0.0)` to recover edge-touching behavior.

**Audited-panel impact:**
- ROC: `(0,0)`/`(1,1)` no longer kiss axis lines.
- Histogram: tallest bar clears top edge; floating `0/55` ticks drop.
- Regression scatter: corner points sit inside plot rect.
- Residuals: extreme outliers no longer touch axis lines.

**NOT fixed by this section** (gallery-fixer territory):
- Stray `(1,1)` point with label `1` on ROC (chart-construction bug)
- Missing AUC annotation on ROC
- Missing per-cell counts on confusion matrix
- Missing KDE overlay on histogram

---

## Section 7 — Sub-phase decomposition

Approach A is too large for a linear pass. Four sub-phases, each ending in green tests and a committable state.

### Phase T1 — Theme key plumbing (no visible changes)
- New fields in `ThemeInputs` struct (`crates/ferrum-core/src/layout/mod.rs`).
- Extend `theme_from_dict` in `render/binding.rs`: read every new key, **reject unknown keys**.
- Unit tests: every Python `Theme(**kwargs)` key round-trips to the right `ThemeInputs` field.
- All existing goldens still byte-equal (defaults unchanged at this stage).
- Spec note in `ferrum-spec.md §3.13`.

### Phase T2 — Consumer wiring (theme keys reach the renderer)
- `font_family / font_weight / title_*` threaded through every `TextStyle` constructor.
- `axis_line: bool` conditionally emits axis stroke.
- `tick_width` used for tick strokes.
- `label_color` used for tick label text (distinct from body `font_color`).
- `point_opacity` consulted by point marks.
- `legend_direction / legend_title_font_size` consumed by legend layout.
- `title_anchor / title_offset` honored by chart title placement.
- Existing goldens still byte-equal (defaults still match what was hardcoded).

### Phase T3 — Gridlines + palette registry
- `render/marks/axis.rs::draw_grid()` (Section 4).
- `render/palette.rs` with 8 schemes (Section 5).
- `scale_resolve.rs` categorical color path consults `theme.color_scheme`.
- **Goldens regenerated:** every single-series chart picks up gridlines. Single batch, single commit, every PNG inspected.

### Phase T4 — New defaults + builtins + scale padding
- `ThemeInputs::default()` to Section 2 values.
- `src/ferrum/themes/builtins.py` rewritten to Section 3 value sets.
- `render/scale_resolve.rs` scale padding (Section 6).
- 8-themes-on-one-chart goldens added at `tests/goldens/theme_gallery/`.
- **Goldens regenerated:** every quantitative golden (~95 SVGs). Single inspection batch, single commit.
- Gallery audit re-run + ferrum panels visually inspected.

### Cross-phase

- **Worktree:** `.claude/worktrees/themes/` on a new branch `feat/themes` based on latest `main` (post-Phase-10 merge). User's concurrent main session stays isolated.
- **Worktree setup cost:** uv `.venv` needs creation; `unset CONDA_PREFIX && uv run --no-sync maturin develop` to build the extension. ~30s bounded.
- **Canonical Rust-test command in worktree:** `PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none DYLD_LIBRARY_PATH=$PYTHONHOME/lib cargo test`. The CLAUDE.md `DYLD_LIBRARY_PATH=$(uv run python -c …)` form is fragile in `.claude/worktrees/`.
- **Test ratchet:** `cargo test` + `pytest` green at the end of every sub-phase.
- **Spec updates:** single dated `2026-05-11` note added in T1 covering T1–T4 (see Section 8).
- **Behavior break:** binding rejects unknown keys (T1). Surfaces any caller that passed typo kwargs accidentally.

---

## Section 8 — Spec update strategy

CLAUDE.md hard constraint: spec divergence is forbidden; dated notes record evolution. Three spec touches dated `2026-05-11`:

### `ferrum-spec.md §3.13` — Theme

Appended before the builtins table:

```markdown
> **2026-05-11 (Phase T1–T4):** Every key listed in the `Theme(...)` block above
> is now plumbed end-to-end through the Rust `ThemeInputs` binding and consumed
> by the renderer. Previously, ~20 keys (font family, title styling, axis line,
> grid color/dash/opacity, color_scheme, etc.) were silently dropped by the
> Rust binding while the Python class accepted them. Unknown keys now raise
> `ValueError` at theme construction. The 8 built-in themes have been rebuilt
> to use the newly-plumbed keys; each is visibly distinct from the others on
> the same chart (see `tests/goldens/theme_gallery/`).
>
> New behavior:
> - `theme.grid` now actually draws gridlines (previously a no-op).
> - `theme.color_scheme` resolves against a Rust-side palette registry that
>   ships all 7 categorical schemes listed in §3.6 (`okabe_ito`, `tableau10`,
>   `set1`, `set2`, `paired`, `pastel`, `dark2`) plus delegates sequential
>   names (`viridis`/`plasma`/`magma`/`inferno`/`cividis`) to the existing
>   `ContinuousScheme` infra. Categorical color encodings without an explicit
>   `range` consult this scheme.
> - The default categorical scheme flips from `okabe_ito` (spec §3.6) to
>   `tableau10` to match the Observable Plot aesthetic of the new defaults.
>   `okabe_ito` remains shipped; `Theme(color_scheme="okabe_ito")` restores it.
> - `theme.axis_line: bool` suppresses axis stroke when false (previously the
>   key existed in Python but Rust always drew the axis line).
> - New defaults: `mark_color="#4C78A8"` (tableau blue, was Okabe orange
>   `#E69F00`), `font_family="DejaVu Sans"` (was implicit serif fallback),
>   `grid_color="#DDDDDD"` width `0.5` (was `#EEEEEE` width `1.0` — invisible),
>   `title_anchor="start"`, `title_font_weight="600"`.
```

### `ferrum-spec.md §3.6` Scales — scale padding default

```markdown
> **2026-05-11 (Phase T4):** `Scale.padding` (already listed above) now
> defaults to `0.05` for quantitative scales — the visual mapping reserves 5%
> of the plot dimension on each side (capped at 8px) so marks do not touch
> axis lines. Categorical/ordinal scales are unaffected (band scales already
> half-step pad). User-specified `Scale(domain=...)` suppresses padding. Set
> `Scale(padding=0.0)` to recover edge-touching behavior.
```

### Categorical-default flip (covered in the §3.13 note above)

Spec §3.6 names `okabe_ito` as the default categorical scheme; the new `ThemeInputs::default()` flips it to `tableau10`. The §3.13 dated note above (which already lists "`mark_color="#4C78A8"` (tableau blue, was Okabe orange `#E69F00`)") covers this divergence. `okabe_ito` remains shipped and accessible via `Theme(color_scheme="okabe_ito")`.

### Decisions

- **Do NOT edit existing prose inline.** Project convention is dated-note evolution; fresh readers see the change history.
- **Do NOT add spec keys** beyond what §3.13 + §3.6 already list. Every plumbed key is in the spec today.
- **`Scale.padding` is already in §3.6** — we change the default only, recorded in the §3.6 dated note above.
- **Okabe-Ito mention in builtins table stays.** The dated notes record the default flips.

---

## Section 9 — Test plan & golden regeneration

### Existing golden footprint (pin at T0)

```
tests/goldens/**/*.svg                          ≈80 files
tests/test_phase_9_e2e/goldens/*.svg            ≈15 files
crates/ferrum-core/tests/.../*.svg              (some Rust-side)
```

Exact counts pinned at the start of T1, recorded in the implementation plan.

### Tests added per sub-phase

**T1 — plumbing**
- `tests/themes/test_binding_roundtrip.py` — every spec-listed `Theme(key=value)` key flows through to the right `ThemeInputs` field.
- `tests/themes/test_unknown_key_raises.py` — `Theme(typo="foo")` raises `ValueError` naming the unknown key.
- Rust unit tests in `crates/ferrum-core/src/render/binding.rs` confirming unknown-key rejection.

**T2 — wiring**
- Per-key consumer tests (`test_title_anchor_left_vs_middle.py`, `test_axis_line_off_suppresses_stroke.py`, ...). Assert via SVG attributes, not pixel positions.

**T3 — gridlines + palette**
- `test_gridlines_render.py` — `grid=True` produces `<line>` elements at every tick position; `grid=False` produces none.
- `test_gridline_styling.py` — `grid_dash`/`grid_opacity` emit the right SVG attrs.
- `test_palette_resolution.py` — 5-category bar chart under `tableau10` vs `set1` produces different fill colors.
- `test_unknown_scheme_raises.py` — `Theme(color_scheme="nope")` raises `ValueError`.
- One Rust unit test per palette (length + first-color byte check).

**T4 — defaults + builtins + padding**
- `test_eight_themes_distinct.py` — same bar chart through all 8 builtins, all SVGs byte-distinct. Goldens at `tests/goldens/theme_gallery/{name}.svg`.
- `test_scale_padding_default.py` — quantitative axis with data `[5, 50]` maps to the inner 90% of plot height.
- `test_scale_padding_zero_pinned.py` — `include_zero=True` keeps the `0` tick visible in the padding band.
- `test_scale_padding_categorical_unaffected.py` — bar chart x-axis unchanged.
- `test_scale_padding_user_domain.py` — explicit `scale.domain=[0,100]` suppresses padding.

### Golden regeneration mechanics (T3, T4)

1. `pytest --regen-goldens tests/` (verify flag name at T0).
2. `python scripts/snapshot-goldens.py` — rasterize every regenerated SVG to PNG.
3. **Read every PNG.** Batches of ~10 via `Read` calls, checking:
   - Gridlines present and aligned to ticks (T3).
   - Marks not touching axis lines (T4).
   - Tableau blue replacing Okabe orange (T4).
   - DejaVu Sans replacing serif fallback (T4).
   - No truncated paths (resvg-py many-paths gotcha).
4. Sanity-check SVG path counts for any PNG that looks wrong (`grep -oE 'd="M' foo.svg | wc -l`).
5. Commit. Commit message names the sub-phase + visual changes verified.

Gallery audit re-run after T4 lands; ferrum panels inspected side-by-side against the canonical libraries. Independent of the golden suite; does not gate the merge — a "did the comparison improve" check.

### CI

- `cargo test` runs every sub-phase end. Must be green.
- `pytest` runs every sub-phase end. Must be green.
- No new CI jobs.

### Risk register

| Risk | Mitigation |
|---|---|
| resvg-py path truncation hides a broken golden | Cross-check SVG path counts before declaring a golden broken |
| T1 misses a spec-listed key | Round-trip test asserts every spec key flows through |
| Scale padding breaks a phase-X test that asserts a pixel position | Tests should assert via public API; tightly-coupled tests get a one-line update with comment pointing at the dated spec note |
| Worktree `cargo test` fails on documented `DYLD_LIBRARY_PATH` line | Plan uses explicit `PYTHONHOME` form; verified in worktree-setup step |
| Concurrent main session merges into theme code | Worktree isolates; rebase before each sub-phase commit; conflicts surface cleanly |

---

## Out of scope

- Chart-construction defaults (AUC annotation, confusion matrix counts, KDE overlays) — gallery-fixer territory.
- Per-axis grid control (`grid_x` / `grid_y`) — spec defines single `grid: bool`.
- User-defined inline palettes (`color_scheme=[...]`) — spec defines `color_scheme` as a string name.
- Bundled Inter font in the wheel — user chose DejaVu Sans for golden determinism.
- New spec keys beyond §3.13 — every plumbed key is already in the spec.

---

## Appendix — Decisions captured during brainstorming

| Decision | Choice |
|---|---|
| Scope | Theme + default visual polish (Y-inversion already fixed in `c911bbb`). |
| Aesthetic | Observable Plot — crisp sans, saturated blue, light grid. |
| Builtins | Fix all 8 in this pass. |
| Font | DejaVu Sans (deterministic, no wheel-bundling needed). |
| Approach | A — spec-complete overhaul. |
| Grid control | Single `grid: bool` (spec as-is). |
| Implementation isolation | Worktree at `.claude/worktrees/themes/`. |
