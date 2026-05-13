# Custom Themes & Palettes — Design Spec

**Date:** 2026-05-12
**Branch:** `feat/gallery-defaults` (or new branch if needed)
**Status:** Approved

---

## 1. Goal

Add three custom ferrum themes — **Paper Ink**, **Slate Citrus**, **Arctic Signal** — each with a dedicated 8-color categorical cycle, two sequential ramps, and one diverging ramp. Make Paper Ink the new process default. Preserve the current Observable Plot default as a named theme (`observable`).

Source data: `design-docs/themes/top3_theme_cycles.csv` (categorical) and `design-docs/themes/ferrum_seq_div_palettes.csv` (sequential + diverging).

---

## 2. Rust: Categorical palettes (`palette.rs`)

Add three new `LazyLock<[Color; 8]>` statics:

| Static | Scheme name | Colors (series_1 … series_8 from CSV) |
|---|---|---|
| `PAPER_INK` | `"paper_ink"` | `#2563EB #DC2626 #D4A017 #0F766E #7C3AED #EA580C #4B5563 #DB2777` |
| `SLATE_CITRUS` | `"slate_citrus"` | `#60A5FA #A78BFA #A3E635 #F59E0B #34D399 #F472B6 #F87171 #22D3EE` |
| `ARCTIC_SIGNAL` | `"arctic_signal"` | `#0284C7 #7C3AED #EA580C #16A34A #DC2626 #0891B2 #CA8A04 #DB2777` |

Wire into `CATEGORICAL_SCHEMES`, `categorical_palette()` match arms. The unknown-name fallback changes from `OKABE_ITO` to `PAPER_INK`.

---

## 3. Rust: Sequential & diverging ramps (`continuous.rs`)

Add 9 new `NamedContinuous` variants. Each is implemented as a pre-built 7-stop gradient (using the existing `sample_gradient` machinery) rather than a `colorous` crate dependency.

| Variant | Name string | Type | Stops (step_1 … step_7 from CSV) |
|---|---|---|---|
| `CoolBlue` | `"cool_blue"` | Sequential | `#EFF6FF #DBEAFE #93C5FD #60A5FA #2563EB #1D4ED8 #1E3A8A` |
| `WarmOchre` | `"warm_ochre"` | Sequential | `#FFF7E6 #FDECC8 #F8D88A #D4A017 #B45309 #92400E #78350F` |
| `BlueToRed` | `"blue_to_red"` | Diverging | `#1E3A8A #60A5FA #DBEAFE #FAF7F2 #FDE68A #DC2626 #7F1D1D` |
| `NightBlue` | `"night_blue"` | Sequential | `#1E293B #1D4ED8 #2563EB #60A5FA #93C5FD #BFDBFE #E0F2FE` |
| `ElectricLime` | `"electric_lime"` | Sequential | `#365314 #4D7C0F #65A30D #84CC16 #A3E635 #BEF264 #D9F99D` |
| `CyanToAmber` | `"cyan_to_amber"` | Diverging | `#155E75 #0891B2 #67E8F9 #111827 #FDE68A #F59E0B #B45309` |
| `SignalBlue` | `"signal_blue"` | Sequential | `#F0F9FF #E0F2FE #7DD3FC #38BDF8 #0284C7 #0369A1 #0C4A6E` |
| `EmberOrange` | `"ember_orange"` | Sequential | `#FFF7ED #FED7AA #FDBA74 #EA580C #C2410C #9A3412 #7C2D12` |
| `BlueToViolet` | `"blue_to_violet"` | Diverging | `#0C4A6E #38BDF8 #BAE6FD #F8FAFC #E9D5FF #A78BFA #6D28D9` |

**Implementation approach:** Add a helper on `NamedContinuous` that returns an `Option<Vec<(f64, Color)>>` for custom schemes (stops at t = 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0). The existing `sample()` dispatches to `colorous_gradient()` for colorous-backed schemes and to `sample_gradient()` for custom stop-lists.

Update `SEQUENTIAL_SCHEMES`, `NamedContinuous::from_name()`, and `NamedContinuous::list()`.

---

## 4. Rust: Paper Ink as `ThemeInputs::default()`

Change `ThemeInputs::default()` to Paper Ink's visual identity. Only color properties change; typography (Inter, semibold title, start-anchored), layout (padding, grid enabled, axis lines), and mark sizing stay the same.

| Property | Current (Observable) | New (Paper Ink) |
|---|---|---|
| `background_color` | `#FFFFFF` | `#FAF7F2` |
| `font_color` | `#222222` | `#1F2937` |
| `label_color` | `#555555` | `#6B7280` |
| `title_color` | `#222222` | `#1F2937` |
| `grid_color` | `#DDDDDD` | `#D6D3D1` |
| `axis_line_color` | `#888888` | `#6B7280` |
| `tick_color` | `#888888` | `#6B7280` |
| `mark_color` | `#4C78A8` | `#2563EB` |
| `color_scheme` | `"tableau10"` | `"paper_ink"` |
| `strip_background_color` | `#F0F0F0` | `#EDE9E3` (warm-tinted to match bg) |
| `reference_line_color` | `#AAAAAA` | `#9CA3AF` |

---

## 5. Python: New `Theme` instances (`builtins.py`)

### 5.1 Paper Ink (explicit, for derivation)

Since Paper Ink IS the Rust default, `default = Theme()` stays empty. The explicit instance exists for named access and as a derivation base:

```python
paper_ink = Theme(
    background="#FAF7F2", font_color="#1F2937", label_color="#6B7280",
    title_color="#1F2937", grid_color="#D6D3D1",
    axis_line_color="#6B7280", tick_color="#6B7280",
    mark_color="#2563EB", color_scheme="paper_ink",
    strip_background_color="#EDE9E3",
    reference_line_color="#9CA3AF",
)
```

### 5.2 Slate Citrus

Dark theme — must explicitly set every property that differs from Paper Ink defaults (background, all text colors, grid, strip bg, reference lines):

```python
slate_citrus = Theme(
    background="#111827", font_color="#E5E7EB", label_color="#9CA3AF",
    title_color="#E5E7EB", grid_color="#374151",
    axis_line_color="#9CA3AF", tick_color="#9CA3AF",
    mark_color="#60A5FA", color_scheme="slate_citrus",
    strip_background_color="#1E293B",
    reference_line_color="#6B7280",
)
```

### 5.3 Arctic Signal

Light theme with cool tones — differs from Paper Ink's warm tones:

```python
arctic_signal = Theme(
    background="#F8FAFC", font_color="#0F172A", label_color="#64748B",
    title_color="#0F172A", grid_color="#CBD5E1",
    axis_line_color="#64748B", tick_color="#64748B",
    mark_color="#0284C7", color_scheme="arctic_signal",
    strip_background_color="#E2E8F0",
    reference_line_color="#94A3B8",
)
```

### 5.4 Observable (preserved old default)

Explicitly captures the pre-Paper-Ink identity so users can restore it:

```python
observable = Theme(
    background="#ffffff", font_color="#222222", label_color="#555555",
    title_color="#222222", grid_color="#DDDDDD",
    axis_line_color="#888888", tick_color="#888888",
    mark_color="#4C78A8", color_scheme="tableau10",
    strip_background_color="#F0F0F0",
    reference_line_color="#AAAAAA",
)
```

### 5.5 Re-exports

Update `themes/__init__.py` to re-export `paper_ink`, `slate_citrus`, `arctic_signal`, `observable` and add them to `__all__`.

### 5.6 Docstring updates

- `Theme` class docstring: update `color_scheme` accepted values to include `"paper_ink"` (default), `"slate_citrus"`, `"arctic_signal"` alongside the existing schemes.
- `builtins.py` module docstring: update to reflect Paper Ink as the default identity and list all 12 named themes.

---

## 6. Testing

### 6.1 Rust tests (`palette.rs`)

- Extend `each_named_palette_has_at_least_8_colors` to cover `paper_ink`, `slate_citrus`, `arctic_signal`.
- Extend `categorical_schemes_const_matches_match_arms` — new names must not fall through to default.
- Rename `categorical_palette_unknown_falls_back_to_okabe_ito` → `…_to_paper_ink` and update assertion.

### 6.2 Rust tests (`continuous.rs`)

- Endpoint spot-checks for at least one new sequential (`cool_blue`: sample(0.0) ≈ `#EFF6FF`, sample(1.0) ≈ `#1E3A8A`).
- Midpoint check for one diverging (`blue_to_red`: sample(0.5) ≈ `#FAF7F2`).

### 6.3 Python tests

- Import checks: `fm.themes.paper_ink`, `fm.themes.slate_citrus`, `fm.themes.arctic_signal`, `fm.themes.observable` are `Theme` instances.
- `fm.themes.default` is `Theme()` (empty) and renders as Paper Ink.

### 6.4 Goldens

All existing goldens regenerate after Rust build. Visual inspection per the hard constraint (rasterize → read PNG → confirm correct rendering) before committing.

---

## 7. Future extension (not in scope)

There is currently no `sequential_scheme` / `diverging_scheme` field on `ThemeInputs`. Themes cannot auto-route heatmaps to their matching sequential ramp (e.g., Paper Ink charts don't automatically get Cool Blue for sequential encodings). The ramps are global-by-name per user preference. Adding theme-scoped sequential/diverging defaults is a possible future extension.

---

## 8. Files changed

| File | Change |
|---|---|
| `crates/ferrum-core/src/render/palette.rs` | 3 new categorical palettes, updated schemes list, new fallback |
| `crates/ferrum-core/src/render/color/continuous.rs` | 9 new named ramps, updated schemes list |
| `crates/ferrum-core/src/layout/mod.rs` | `ThemeInputs::default()` → Paper Ink colors |
| `src/ferrum/themes/builtins.py` | 4 new Theme instances, updated module docstring |
| `src/ferrum/themes/__init__.py` | Re-exports, updated class docstring |
| `tests/goldens/**/*.svg` | Regenerated |
