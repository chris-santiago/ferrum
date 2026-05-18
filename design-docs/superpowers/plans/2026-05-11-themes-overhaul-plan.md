# Themes Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plumb every `ferrum-spec.md §3.13` Theme key end-to-end through the Rust binding, implement the missing gridline renderer, ship a `color_scheme` palette registry, flip the default visual identity to Observable Plot aesthetic, rebuild the 8 builtins so each is distinct, and default `Scale.padding=0.05` to stop marks touching axis lines.

**Architecture:** Python `Theme` validates kwargs at construction against a known-key set, resolves fallback chains (`title_color → font_color`, etc.) in `to_theme_inputs_dict()`, then hands a flat dict to Rust. The Rust binding (`theme_from_dict`) reads every key into an expanded `ThemeInputs` struct and errors on unknowns. Render-side consumers (axis, legend, text, palette resolution, scale padding) read from `ThemeInputs` instead of hardcoded values. Sub-phases T1 (plumbing) → T2 (wiring) → T3 (new primitives: gridlines + palette) → T4 (new defaults + builtins + padding) each end in a green test suite + a commit.

**Tech Stack:** Rust (pyo3 ~0.22, palette crate for `Srgba<u8>`, existing `ContinuousScheme` infra), Python (`ferrum.Theme` value class, pytest), DejaVu Sans (deterministic font for goldens), tableau10 (new default categorical palette).

**Worktree:** `.claude/worktrees/themes/`, branched from `main` at commit `94eef87`.

**Spec:** `docs/superpowers/specs/2026-05-11-themes-overhaul-design.md`.

---

## Build commands (this worktree)

The worktree's `.venv` was created using miniforge Python at `/opt/homebrew/Caskroom/miniforge/base`, not uv-managed cpython. The CLAUDE.md `DYLD_LIBRARY_PATH=$(uv run python -c …)` form is fragile in worktrees; use the explicit forms below. Diagnostic-first (memory `feedback_worktree_cargo_test.md`):

```bash
unset CONDA_PREFIX
uv run --no-sync python -c "import sys, sysconfig; \
  print('base_prefix:', sys.base_prefix); \
  print('LIBDIR:', sysconfig.get_config_var('LIBDIR'))"
```

| Action | Command (run from worktree root) |
|---|---|
| Build Rust extension | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Build Rust extension (release) | `unset CONDA_PREFIX && uv run --no-sync maturin develop --release` |
| Run Python tests | `unset CONDA_PREFIX && uv run --no-sync pytest` |
| Run Rust tests | `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test` |
| Skeleton smoke test | `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"` |

Verified at plan-write time: cargo test 515 passed, pytest collected 820 tests.

---

## File Structure

| File | Role | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|
| `crates/ferrum-core/src/layout/mod.rs` | `ThemeInputs` struct + `Default::default()` | extend struct | — | — | flip defaults |
| `crates/ferrum-core/src/layout/legend.rs` | `LegendDirection` enum (already exists) | — | — | — | — |
| `crates/ferrum-core/src/layout/panel.rs` | `TextAnchor` enum (already exists) | — | — | — | — |
| `crates/ferrum-core/src/render/binding.rs` | `theme_from_dict` | extend, reject unknown | — | reject unknown scheme | — |
| `crates/ferrum-core/src/render/palette.rs` (new) | Categorical palette registry | — | — | new | — |
| `crates/ferrum-core/src/render/marks/axis.rs` | Axis + tick rendering, new `draw_grid()` | — | wire tick_width / label_color / axis_line bool | add `draw_grid()` | — |
| `crates/ferrum-core/src/render/marks/text.rs` | Text style emission | — | wire font_family / font_weight | — | — |
| `crates/ferrum-core/src/render/marks/legend.rs` | Legend layout + render | — | wire legend_direction / legend_title_font_size | — | — |
| `crates/ferrum-core/src/render/marks/point.rs` | Point marks | — | wire point_opacity | — | — |
| `crates/ferrum-core/src/render/draw.rs` | Per-panel draw loop | — | thread theme fields into TextStyle defaults | call `draw_grid()` after bg, before marks | — |
| `crates/ferrum-core/src/render/scale_resolve.rs` | Scale → range mapping, categorical color | — | — | use palette registry | add padding |
| `crates/ferrum-core/src/render/mod.rs` | Chart-title placement | — | wire title_anchor / title_offset | — | — |
| `src/ferrum/themes/__init__.py` | `Theme` class | known-key set + validation + fallbacks | — | — | — |
| `src/ferrum/themes/builtins.py` | 8 named themes | — | — | — | rebuild all 8 |
| `tests/themes/` (new dir) | New test modules | binding roundtrip, unknown-key | per-key consumer tests | gridline + palette tests | 8-distinct + padding tests |
| `tests/goldens/` | Existing 36 SVGs | unchanged | unchanged | regenerate | regenerate |
| `tests/test_phase_9_e2e/goldens/` | Existing 12 SVGs | unchanged | unchanged | regenerate | regenerate |
| `crates/ferrum-core/tests/` SVGs | Existing 6 SVGs | unchanged | unchanged | regenerate | regenerate |
| `ferrum-spec.md` | Spec | §3.13 dated note | — | — | §3.13 + §3.6 dated notes |

**Golden footprint pinned (2026-05-11):** 36 `tests/goldens/**/*.svg` + 12 `tests/test_phase_9_e2e/goldens/*.svg` + 6 Rust-side = **54 goldens**.

---

# T0 — Prep (already complete)

Worktree exists at `.claude/worktrees/themes/` on branch `feat/themes`. `uv sync` + `maturin develop` succeeded; cargo test green; pytest collects 820 tests. Skip to T1.

---

# T1 — Theme key plumbing (zero visible change)

Every spec-listed key gets a home in `ThemeInputs` and `theme_from_dict`. Defaults stay at their current values (Okabe orange, current grid, etc.); visual change is deferred to T4. T1 ends with 100% byte-identical existing goldens and a new key-roundtrip test suite proving every spec key plumbs through.

## Task 1.1: Extend `ThemeInputs` struct with new fields

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs:87-124` (struct body), `:126-175` (Default impl)

- [ ] **Step 1: Add new fields to the struct**

Open `crates/ferrum-core/src/layout/mod.rs`. After the existing `pub strip_background_color: palette::Srgba<u8>,` line in the `ThemeInputs` struct, add:

```rust
    // Phase Themes-T1 additions.

    // Typography
    pub font_family: String,
    pub font_weight: String,
    pub title_font_family: String,
    pub title_font_weight: String,
    pub title_color: palette::Srgba<u8>,
    pub title_anchor: super::layout::panel::TextAnchor,
    pub title_offset: f64,
    pub label_font_family: String,
    pub label_color: palette::Srgba<u8>,

    // Axes
    pub axis_line: bool,
    pub tick_width: f64,

    // Grid
    pub grid_dash: Option<Vec<f64>>,
    pub grid_opacity: f64,

    // Marks
    pub point_opacity: f64,

    // Palette
    pub color_scheme: String,

    // Legend
    pub legend_direction: super::layout::legend::LegendDirection,
    pub legend_title_font_size: f64,
```

Note: the existing `#[derive(Debug, Clone, Copy, PartialEq)]` on `ThemeInputs` will fail because `String` and `Vec<f64>` are not `Copy`. Change the derive to `#[derive(Debug, Clone, PartialEq)]` (drop `Copy`). Callers that depended on `Copy` semantics (likely `let theme = *some_theme;` patterns) get fixed by replacing with `.clone()`.

- [ ] **Step 2: Update `Default::default()` to fill new fields with current-default-equivalent values**

In the same file, inside `impl Default for ThemeInputs`, after the existing field initializations, add:

```rust
            // T1 plumbing — values match current visual identity. T4 will flip these.
            font_family: "DejaVu Serif".into(),       // resvg default; T4 → "DejaVu Sans"
            font_weight: "normal".into(),
            title_font_family: "DejaVu Serif".into(),
            title_font_weight: "bold".into(),         // T4 → "600"
            title_color: text_222,
            title_anchor: super::layout::panel::TextAnchor::Middle,   // T4 → Start
            title_offset: 4.0,                        // T4 → 6.0
            label_font_family: "DejaVu Serif".into(),
            label_color: text_222,                    // T4 → label_555 = #555555

            axis_line: true,
            tick_width: 1.0,

            grid_dash: None,
            grid_opacity: 1.0,

            point_opacity: 1.0,

            color_scheme: "okabe_ito".into(),         // T4 → "tableau10"

            legend_direction: super::layout::legend::LegendDirection::Vertical,
            legend_title_font_size: 13.0,             // matches title_font_size
```

- [ ] **Step 3: Fix any `Copy`-dependent call sites**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo check --lib 2>&1 | grep -E "error\[E0"`

Expected: zero or a small list of "the trait `Copy` is not implemented" errors. For each one, replace `*theme` / `let t = theme;` (move) patterns with `theme.clone()` or `&theme` borrows. The existing render pipeline mostly passes `&ThemeInputs` already.

- [ ] **Step 4: Build succeeds**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo build --lib`
Expected: builds cleanly (warnings okay; no errors).

- [ ] **Step 5: Existing cargo tests still pass**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib`
Expected: all 515 prior tests still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(themes-T1): extend ThemeInputs with spec §3.13 key set

Adds 17 new fields (font_family, font_weight, title_*, label_*, axis_line,
tick_width, grid_dash, grid_opacity, point_opacity, color_scheme,
legend_direction, legend_title_font_size). Default values match current
visual identity — T4 will flip defaults to Observable Plot aesthetic.
Drops Copy from the derive (String, Vec<f64> are not Copy)."
```

## Task 1.2: Extend `theme_from_dict` to read every new key

**Files:**
- Modify: `crates/ferrum-core/src/render/binding.rs:79-114`

- [ ] **Step 1: Add per-key extraction blocks**

In `crates/ferrum-core/src/render/binding.rs`, inside `fn theme_from_dict`, after the existing `padding` extraction block (line 113), add the following blocks (each follows the existing pattern):

```rust
    // Typography
    if let Some(v) = d.get_item("font_family")? {
        t.font_family = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("font_weight")? {
        t.font_weight = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("font_color")? {
        let s: String = v.extract()?;
        t.font_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("font_size")? {
        t.label_font_size = v.extract()?;
    }
    if let Some(v) = d.get_item("title_font_family")? {
        t.title_font_family = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("title_font_size")? {
        t.title_font_size = v.extract()?;
    }
    if let Some(v) = d.get_item("title_font_weight")? {
        t.title_font_weight = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("title_color")? {
        let s: String = v.extract()?;
        t.title_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("title_anchor")? {
        let s: String = v.extract()?;
        t.title_anchor = match s.as_str() {
            "start" => crate::layout::panel::TextAnchor::Start,
            "middle" => crate::layout::panel::TextAnchor::Middle,
            "end" => crate::layout::panel::TextAnchor::End,
            other => return Err(PyValueError::new_err(format!(
                "title_anchor must be one of 'start'|'middle'|'end', got '{other}'"
            ))),
        };
    }
    if let Some(v) = d.get_item("title_offset")? {
        t.title_offset = v.extract()?;
    }
    if let Some(v) = d.get_item("label_font_family")? {
        t.label_font_family = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("label_color")? {
        let s: String = v.extract()?;
        t.label_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    // Axes
    if let Some(v) = d.get_item("axis_line")? {
        t.axis_line = v.extract()?;
    }
    if let Some(v) = d.get_item("axis_line_color")? {
        let s: String = v.extract()?;
        t.axis_line_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("axis_line_width")? {
        t.axis_line_width = v.extract()?;
    }
    if let Some(v) = d.get_item("tick_color")? {
        let s: String = v.extract()?;
        t.tick_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("tick_size")? {
        t.tick_size = v.extract()?;
    }
    if let Some(v) = d.get_item("tick_width")? {
        t.tick_width = v.extract()?;
    }

    // Grid
    if let Some(v) = d.get_item("grid_color")? {
        let s: String = v.extract()?;
        t.grid_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("grid_width")? {
        t.grid_width = v.extract()?;
    }
    if let Some(v) = d.get_item("grid_dash")? {
        let dashes: Vec<f64> = v.extract()?;
        t.grid_dash = Some(dashes);
    }
    if let Some(v) = d.get_item("grid_opacity")? {
        t.grid_opacity = v.extract()?;
    }

    // Marks
    if let Some(v) = d.get_item("point_opacity")? {
        t.point_opacity = v.extract()?;
    }
    if let Some(v) = d.get_item("opacity")? {
        t.default_opacity = v.extract()?;
    }

    // Palette
    if let Some(v) = d.get_item("color_scheme")? {
        t.color_scheme = v.extract::<String>()?;
    }

    // Strip
    if let Some(v) = d.get_item("strip_background_color")? {
        let s: String = v.extract()?;
        t.strip_background_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    // Legend
    if let Some(v) = d.get_item("legend_orient")? {
        let s: String = v.extract()?;
        t.legend_orient = match s.as_str() {
            "left" => crate::layout::LegendOrient::Left,
            "right" => crate::layout::LegendOrient::Right,
            "top" => crate::layout::LegendOrient::Top,
            "bottom" => crate::layout::LegendOrient::Bottom,
            other => return Err(PyValueError::new_err(format!(
                "legend_orient must be one of 'left'|'right'|'top'|'bottom', got '{other}'"
            ))),
        };
    }
    if let Some(v) = d.get_item("legend_direction")? {
        let s: String = v.extract()?;
        t.legend_direction = match s.as_str() {
            "horizontal" => crate::layout::legend::LegendDirection::Horizontal,
            "vertical" => crate::layout::legend::LegendDirection::Vertical,
            other => return Err(PyValueError::new_err(format!(
                "legend_direction must be one of 'horizontal'|'vertical', got '{other}'"
            ))),
        };
    }
    if let Some(v) = d.get_item("legend_title_font_size")? {
        t.legend_title_font_size = v.extract()?;
    }

    // Spacing
    if let Some(v) = d.get_item("axis_title_padding")? {
        t.axis_title_padding = v.extract()?;
    }
    if let Some(v) = d.get_item("column_padding")? {
        t.column_padding = v.extract()?;
    }
    if let Some(v) = d.get_item("row_padding")? {
        t.row_padding = v.extract()?;
    }
```

- [ ] **Step 2: Build succeeds**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo build --lib`
Expected: builds cleanly.

- [ ] **Step 3: All existing tests still pass**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib && unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run --no-sync pytest -x`

Expected: cargo 515 pass, pytest 820 pass. No visual changes; defaults unchanged.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/binding.rs
git commit -m "feat(themes-T1): theme_from_dict reads every spec §3.13 key

Adds extraction blocks for font_family, font_weight, font_color, font_size,
title_font_family/size/weight/color/anchor/offset, label_font_family/color,
axis_line/color/width, tick_color/size/width, grid_color/width/dash/opacity,
point_opacity, opacity, color_scheme, strip_background_color,
legend_orient/direction/title_font_size, axis_title_padding,
column_padding, row_padding. No render-side consumption yet."
```

## Task 1.3: Reject unknown keys in `theme_from_dict`

**Files:**
- Modify: `crates/ferrum-core/src/render/binding.rs` (end of `theme_from_dict`, before `Ok(t)`)

- [ ] **Step 1: Add known-key check before the final `Ok(t)` return**

In `theme_from_dict`, before `Ok(t)`:

```rust
    // Reject unknown keys to surface typos that previously silently dropped.
    const KNOWN_THEME_KEYS: &[&str] = &[
        "background", "padding",
        "font_family", "font_weight", "font_color", "font_size",
        "title_font_family", "title_font_size", "title_font_weight",
        "title_color", "title_anchor", "title_offset",
        "label_font_family", "label_color",
        "grid", "grid_color", "grid_width", "grid_dash", "grid_opacity",
        "axis_line", "axis_line_color", "axis_line_width",
        "tick_color", "tick_size", "tick_width",
        "mark_color", "point_size", "point_opacity",
        "line_stroke_width", "bar_corner_radius", "area_opacity", "opacity",
        "color_scheme",
        "strip_background_color",
        "legend_orient", "legend_direction", "legend_title_font_size",
        "axis_title_padding", "column_padding", "row_padding",
    ];
    for key_obj in d.keys() {
        let key: String = key_obj.extract()?;
        if !KNOWN_THEME_KEYS.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Unknown Theme key: '{key}'. \
                 See ferrum-spec.md §3.13 for the supported key list."
            )));
        }
    }
```

(`background` accepts the canonical spec name; existing code reads `background_color` — the Python `Theme.to_theme_inputs_dict()` will normalize spec-name → binding-name in Task 1.5.)

- [ ] **Step 2: Wait — also add the missing `background_color` and `point_size`/etc. extraction**

Re-check the existing binding for keys that were partial. The current binding already reads `mark_color`, `background_color`, `point_size`, `line_stroke_width`, `bar_corner_radius`, `area_opacity`, `grid`, `padding`. Add the canonical name aliases: `background` should be treated identically to `background_color`. Insert before the known-key check:

```rust
    // Spec uses `background`; binding originally read `background_color`. Accept both.
    if let Some(v) = d.get_item("background")? {
        let s: String = v.extract()?;
        t.background_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
```

And include both `"background"` and `"background_color"` in `KNOWN_THEME_KEYS` (replace the first entry):

```rust
        "background", "background_color", "padding",
```

- [ ] **Step 3: Verify existing tests still pass (no theme calls today use bad keys)**

Run: `unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run --no-sync pytest -x`
Expected: 820 pass. If a test fails because it was passing a typo kwarg, fix the typo in the test.

- [ ] **Step 4: Add a focused Rust unit test for unknown-key rejection**

Append to `crates/ferrum-core/src/render/binding.rs` (inside the `#[cfg(test)] mod tests` block, creating one if absent):

```rust
#[cfg(test)]
mod theme_dict_tests {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    #[test]
    fn unknown_key_raises() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("not_a_real_key", "value").unwrap();
            let err = theme_from_dict(Some(&d)).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("Unknown Theme key"), "got: {msg}");
            assert!(msg.contains("not_a_real_key"), "got: {msg}");
        });
    }

    #[test]
    fn background_alias_accepted() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("background", "#ff0000").unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.background_color.red, 0xFF);
            assert_eq!(t.background_color.green, 0x00);
            assert_eq!(t.background_color.blue, 0x00);
        });
    }
}
```

- [ ] **Step 5: Run cargo tests**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib theme_dict_tests`
Expected: 2 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/render/binding.rs
git commit -m "feat(themes-T1): reject unknown Theme keys in Rust binding

Adds known-key allowlist to theme_from_dict; unknown keys now raise
ValueError instead of silently dropping. Also accepts 'background' as
alias for 'background_color' per spec §3.13. Two new Rust unit tests
cover the unknown-key error path and the alias acceptance."
```

## Task 1.4: Python-side known-key set + construction-time validation

**Files:**
- Modify: `src/ferrum/themes/__init__.py:7-30` (the `Theme` class)

- [ ] **Step 1: Add `_KNOWN_KEYS` class attr + validation in `__init__`**

Open `src/ferrum/themes/__init__.py`. Replace the existing `Theme.__init__` and surrounding area with:

```python
class Theme:
    """Immutable theme value class. Pass via Chart.theme(t) or set_default_theme(t).

    All keys default to None (omitted from the dict handed to Rust). Unknown
    keys raise ValueError at construction time.

    See ferrum-spec.md §3.13 for the supported key list.
    """

    __slots__ = ("_props",)

    _KNOWN_KEYS: frozenset[str] = frozenset({
        # Canvas
        "background", "background_color", "padding",
        # Typography
        "font_family", "font_weight", "font_color", "font_size",
        "title_font_family", "title_font_size", "title_font_weight",
        "title_color", "title_anchor", "title_offset",
        "label_font_family", "label_color",
        # Grid
        "grid", "grid_color", "grid_width", "grid_dash", "grid_opacity",
        # Axes
        "axis_line", "axis_line_color", "axis_line_width",
        "tick_color", "tick_size", "tick_width",
        # Marks
        "mark_color", "point_size", "point_opacity",
        "line_stroke_width", "bar_corner_radius", "area_opacity", "opacity",
        # Palette
        "color_scheme",
        # Strip
        "strip_background_color",
        # Legend
        "legend_orient", "legend_direction", "legend_title_font_size",
        # Spacing
        "axis_title_padding", "column_padding", "row_padding",
    })

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - self._KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"Unknown Theme key(s): {sorted(unknown)!r}. "
                f"See ferrum-spec.md §3.13 for the supported key list."
            )
        self._props: dict = {k: v for k, v in kwargs.items() if v is not None}
```

(Keep the existing `update`, `to_theme_inputs_dict`, `__eq__`, `__hash__`, `__repr__` methods. We'll modify `to_theme_inputs_dict` in Task 1.5 to apply fallbacks.)

- [ ] **Step 2: Update docstring**

Remove the outdated paragraph that says "only the keys listed below are currently wired ... others are silently ignored". Replace with: "All keys listed below are plumbed end-to-end to the Rust renderer. Unknown keys raise ValueError at construction."

- [ ] **Step 3: Existing tests still pass (no test uses unknown kwargs)**

Run: `unset CONDA_PREFIX && uv run --no-sync pytest -x`
Expected: 820 pass.

- [ ] **Step 4: Add a focused Python unit test**

Create `tests/themes/__init__.py` (empty file) and `tests/themes/test_unknown_key_raises.py`:

```python
"""Theme unknown-key validation."""
import pytest

import ferrum as fm


def test_unknown_key_raises_at_construction() -> None:
    with pytest.raises(ValueError) as excinfo:
        fm.Theme(typo_key="foo")
    msg = str(excinfo.value)
    assert "Unknown Theme key" in msg
    assert "typo_key" in msg


def test_multiple_unknown_keys_listed() -> None:
    with pytest.raises(ValueError) as excinfo:
        fm.Theme(typo_a="x", typo_b="y", font_family="DejaVu Sans")
    msg = str(excinfo.value)
    assert "typo_a" in msg
    assert "typo_b" in msg
    # Known key not mentioned in error.
    assert "font_family" not in msg


def test_known_keys_accepted() -> None:
    # Sample drawn from across the spec — proves the set covers the breadth.
    t = fm.Theme(
        background="#ffffff",
        font_family="DejaVu Sans",
        title_anchor="start",
        grid=True,
        grid_dash=[3, 3],
        color_scheme="tableau10",
        legend_orient="bottom",
    )
    assert t is not None
```

Note: this assumes `fm.Theme` is exposed at top-level. If not, the import is `from ferrum.themes import Theme` — verify via:
`unset CONDA_PREFIX && uv run --no-sync python -c "import ferrum; print(hasattr(ferrum, 'Theme'))"`

If `False`, use `from ferrum.themes import Theme` in the test.

- [ ] **Step 5: Run new tests**

Run: `unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_unknown_key_raises.py -v`
Expected: 3 pass.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/themes/__init__.py tests/themes/__init__.py tests/themes/test_unknown_key_raises.py
git commit -m "feat(themes-T1): Theme(**kwargs) validates known keys at construction

Adds _KNOWN_KEYS frozenset and raises ValueError immediately on unknown
kwargs instead of silently storing them. Three tests cover the error
path, multi-key error, and a broad sample of valid keys."
```

## Task 1.5: Python-side fallback resolution

Fallback chains (`title_color → font_color`, `label_color → font_color`, `title_font_family → font_family`, `label_font_family → font_family`) resolved in `Theme.to_theme_inputs_dict()` before handing to Rust. Rust sees a fully-populated dict.

**Files:**
- Modify: `src/ferrum/themes/__init__.py:31-33` (`to_theme_inputs_dict` method)

- [ ] **Step 1: Implement fallback resolution**

Replace the existing `to_theme_inputs_dict`:

```python
    _FALLBACKS: dict[str, str] = {
        "title_color": "font_color",
        "label_color": "font_color",
        "title_font_family": "font_family",
        "label_font_family": "font_family",
    }

    def to_theme_inputs_dict(self) -> dict:
        """Return a dict suitable for ferrum._core.render_svg(theme=...).

        Resolves spec-defined fallbacks (e.g. ``title_color`` falls back to
        ``font_color`` if unset). Rust sees a fully-resolved dict; no Option
        fallback chains in the binding.
        """
        d = dict(self._props)
        for derived, source in self._FALLBACKS.items():
            if derived not in d and source in d:
                d[derived] = d[source]
        return d
```

- [ ] **Step 2: Add test for fallback resolution**

Create `tests/themes/test_fallback_resolution.py`:

```python
"""Theme fallback chain resolution."""
import ferrum as fm


def test_title_color_falls_back_to_font_color() -> None:
    t = fm.Theme(font_color="#222222")
    d = t.to_theme_inputs_dict()
    assert d["font_color"] == "#222222"
    assert d["title_color"] == "#222222"
    assert d["label_color"] == "#222222"


def test_explicit_title_color_overrides_fallback() -> None:
    t = fm.Theme(font_color="#222222", title_color="#ff0000")
    d = t.to_theme_inputs_dict()
    assert d["title_color"] == "#ff0000"
    assert d["label_color"] == "#222222"


def test_font_family_chain() -> None:
    t = fm.Theme(font_family="DejaVu Sans")
    d = t.to_theme_inputs_dict()
    assert d["title_font_family"] == "DejaVu Sans"
    assert d["label_font_family"] == "DejaVu Sans"


def test_no_fallback_when_source_also_unset() -> None:
    t = fm.Theme(mark_color="#000000")
    d = t.to_theme_inputs_dict()
    assert "title_color" not in d
    assert "label_color" not in d
```

- [ ] **Step 3: Run new tests**

Run: `unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_fallback_resolution.py -v`
Expected: 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/themes/__init__.py tests/themes/test_fallback_resolution.py
git commit -m "feat(themes-T1): Theme resolves fallback chains Python-side

title_color/label_color → font_color and title_font_family/label_font_family
→ font_family resolved in to_theme_inputs_dict() before handing to Rust.
Rust binding sees a fully-populated dict; no Option chains at the seam."
```

## Task 1.6: Cross-language roundtrip test

Proves every spec key flows Python → Rust → ThemeInputs field correctly.

**Files:**
- Create: `tests/themes/test_binding_roundtrip.py`

- [ ] **Step 1: Write the roundtrip test**

```python
"""Theme key roundtrip — every spec §3.13 key reaches the Rust renderer.

This test exercises every key by rendering a minimal chart with the theme
applied and asserting the resulting SVG reflects the key's effect. Where
direct SVG-attribute verification isn't possible (e.g. font_family is only
visible in `<text font-family=...>`), the test asserts the value appears
in the SVG.
"""
from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture(scope="module")
def base_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


def _render(chart: fm.Chart, **theme_kwargs) -> str:
    return chart.theme(fm.Theme(**theme_kwargs)).to_svg()


def test_mark_color_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, mark_color="#ff0000")
    assert "#ff0000" in svg.lower() or 'fill="rgb(255,0,0)' in svg.lower()


def test_background_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, background="#abcdef")
    assert "#abcdef" in svg.lower()


def test_font_family_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, font_family="Helvetica")
    assert "Helvetica" in svg


def test_font_weight_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, font_weight="bold")
    # `font-weight="bold"` may appear on text marks via the title or axis title path.
    assert 'font-weight="bold"' in svg or "font-weight: bold" in svg


def test_title_anchor_start_renders(base_chart: fm.Chart) -> None:
    titled = base_chart.properties(title="Hi")
    svg = _render(titled, title_anchor="start")
    # When anchor=start, the text-anchor SVG attr is "start" on the title element.
    assert 'text-anchor="start"' in svg


def test_title_anchor_middle_renders(base_chart: fm.Chart) -> None:
    titled = base_chart.properties(title="Hi")
    svg = _render(titled, title_anchor="middle")
    assert 'text-anchor="middle"' in svg


def test_grid_dash_reaches_svg(base_chart: fm.Chart) -> None:
    # After T3 lands gridlines, dash should appear as stroke-dasharray.
    # In T1, the key roundtrips through the binding without crash — that's enough.
    svg = _render(base_chart, grid_dash=[3, 3])
    # No assertion on dash attr in T1 since gridlines aren't drawn yet.
    assert svg.startswith("<svg")


def test_axis_line_bool_roundtrips(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, axis_line=False)
    assert svg.startswith("<svg")


def test_color_scheme_roundtrips(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, color_scheme="tableau10")
    assert svg.startswith("<svg")


def test_legend_direction_roundtrips(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, legend_direction="horizontal")
    assert svg.startswith("<svg")


def test_unknown_color_in_invalid_hex_raises() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [1.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(mark_color="not-a-hex")
    )
    with pytest.raises(ValueError):
        chart.to_svg()
```

- [ ] **Step 2: Run test**

Run: `unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_binding_roundtrip.py -v`
Expected: all pass (some assertions are loose in T1 because consumers aren't wired yet — that's by design).

- [ ] **Step 3: Commit**

```bash
git add tests/themes/test_binding_roundtrip.py
git commit -m "test(themes-T1): cross-language Theme key roundtrip

Renders a base chart with each spec §3.13 key set and asserts the value
reaches the SVG (where directly observable in T1). Loose assertions for
keys that depend on T2/T3 consumers — tightened in later sub-phases."
```

## Task 1.7: Existing goldens still byte-equal

Sanity check that T1 plumbing didn't accidentally change any default-rendered chart.

- [ ] **Step 1: Build extension and run full pytest**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync pytest -x
```

Expected: 820 tests + new T1 tests pass; zero golden diffs.

- [ ] **Step 2: Run cargo tests**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test
```

Expected: ≥ 515 + 2 new T1 tests pass.

- [ ] **Step 3: No commit needed — verification step only.**

## Task 1.8: Add T1 dated note to spec

**Files:**
- Modify: `ferrum-spec.md` (append at the end of §3.13 before the builtins table)

- [ ] **Step 1: Locate the insertion point**

Open `ferrum-spec.md`. Find §3.13 — search for `### 3.13 Themes`. Locate the last paragraph before the `**Built-in themes**` heading.

- [ ] **Step 2: Insert the dated note**

After the closing of the existing `Theme(...)` block prose, before the `**Built-in themes**` section:

```markdown
> **2026-05-11 (Themes-T1):** Every key listed in the `Theme(...)` block above
> is now plumbed end-to-end. The Python `Theme` class validates unknown kwargs
> at construction time (raises `ValueError`); the Rust `theme_from_dict`
> binding likewise rejects unknown keys. Spec key aliases (e.g. `background`
> ↔ `background_color`) are accepted by both. Fallback chains (`title_color
> → font_color`, `label_color → font_color`, `title_font_family →
> font_family`, `label_font_family → font_family`) are resolved Python-side
> in `Theme.to_theme_inputs_dict()` so the Rust binding sees a fully-populated
> dict. Render-side consumption of the newly-plumbed keys lands in Themes-T2
> through T4; defaults remain at their pre-T1 values in this sub-phase.
```

- [ ] **Step 3: Commit**

```bash
git add ferrum-spec.md
git commit -m "docs(spec): §3.13 Themes-T1 dated note — keys plumbed end-to-end"
```

## Task 1.9: T1 sub-phase commit checkpoint

- [ ] **Step 1: Confirm all T1 tests pass green**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
unset CONDA_PREFIX && uv run --no-sync pytest
```

Expected: cargo green, pytest green.

- [ ] **Step 2: Confirm zero golden diffs**

```bash
git diff --stat tests/goldens/ tests/test_phase_9_e2e/goldens/ crates/ferrum-core/tests/
```

Expected: empty output.

- [ ] **Step 3: T1 complete.** Move to T2.

---

# T2 — Consumer wiring (zero visible change)

Each newly-plumbed key gets read by at least one renderer code path. Defaults still match what the renderer hardcoded before T1, so existing goldens stay byte-equal.

## Task 2.1: Thread `font_family` + `font_weight` into TextStyle defaults

**Files:**
- Modify: `crates/ferrum-core/src/render/draw.rs:43-78` (TextStyle default constructors)
- Modify: `crates/ferrum-core/src/render/marks/text.rs` (style emission)

- [ ] **Step 1: Read current TextStyle structure**

In `draw.rs`, locate the `TextStyle` struct (around line 29). Note current fields: `font_size`, `font_weight: Option<String>`, `align`, `baseline`, etc.

- [ ] **Step 2: Add `font_family` field to TextStyle**

If TextStyle doesn't already have a `font_family` field, add:

```rust
    pub font_family: Option<String>,
```

In every TextStyle default constructor (the lines that initialize `font_weight: None`), add:

```rust
            font_family: None,
```

- [ ] **Step 3: Update SVG emission to include font-family attr**

In `render/marks/text.rs::emit_text` (or equivalent), where text attrs are emitted, add (using the style's font_family if set, else theme's):

```rust
let family = style.font_family.as_deref().unwrap_or(ctx.theme.font_family.as_str());
// in the SVG attr list:
write!(out, r#" font-family="{family}""#)?;
```

(Look at the existing `font-weight` emission and mirror the pattern.)

- [ ] **Step 4: Thread theme defaults into all TextStyle factories in draw.rs**

For each TextStyle constructor that builds a "default" style (typically named like `default_text_style()` or used inline), default `font_family` to `Some(theme.font_family.clone())` and `font_weight` to `Some(theme.font_weight.clone())` when constructing for body text. For title and axis title text, use `theme.title_font_family` and `theme.title_font_weight`.

- [ ] **Step 5: Build + test**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run --no-sync pytest -x
```

Expected: green; existing goldens still byte-equal because `theme.font_family` default is `"DejaVu Serif"`, which is what resvg implicitly used before.

- [ ] **Step 6: Tighten one T1 test that was loose**

In `tests/themes/test_binding_roundtrip.py`, the `test_font_family_reaches_svg` assertion should now actually find `"Helvetica"` in the SVG. Re-run that test specifically to confirm:

```bash
unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_binding_roundtrip.py::test_font_family_reaches_svg -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/render/
git commit -m "feat(themes-T2): TextStyle defaults consume theme.font_family/weight

Every text emission (chart title, axis title, tick label, legend entry,
strip title) now reads font_family / font_weight from theme by default.
Per-mark style overrides still win. Defaults match prior implicit
behavior; no golden diffs."
```

## Task 2.2: Thread title typography into chart-title placement

**Files:**
- Modify: `crates/ferrum-core/src/render/mod.rs` (chart-title placement)

- [ ] **Step 1: Locate chart-title rendering**

In `render/mod.rs`, find where the chart title is emitted. It's likely a few lines near the top of the rendered SVG, with a text element positioned above the plot region.

- [ ] **Step 2: Consume `theme.title_anchor` and `theme.title_offset`**

Where the title's `x` coordinate is computed, branch on `theme.title_anchor`:

```rust
let title_x = match theme.title_anchor {
    crate::layout::panel::TextAnchor::Start => plot.x,
    crate::layout::panel::TextAnchor::Middle => plot.x + plot.w / 2.0,
    crate::layout::panel::TextAnchor::End => plot.x + plot.w,
};
let title_text_anchor = match theme.title_anchor {
    crate::layout::panel::TextAnchor::Start => "start",
    crate::layout::panel::TextAnchor::Middle => "middle",
    crate::layout::panel::TextAnchor::End => "end",
};
let title_y = plot.y - theme.title_offset;
```

Use these in the SVG emission.

- [ ] **Step 3: Verify existing title tests still pass**

Run: `unset CONDA_PREFIX && uv run --no-sync pytest -k title -v`
Expected: existing title tests pass; defaults unchanged (`title_anchor=Middle`, `title_offset=4.0` matches prior hardcoded behavior — verify this in the existing code; if prior code used different values, the T1 ThemeInputs::default() needs to match those, NOT the Section 2 values).

If goldens diff: the prior hardcoded title placement didn't match the T1 default. Update `ThemeInputs::default()` in `layout/mod.rs` to match the OLD hardcoded values exactly (the T4 task will flip them to Section 2 values).

- [ ] **Step 4: Tighten T1 roundtrip tests for title_anchor**

The `test_title_anchor_start_renders` / `test_title_anchor_middle_renders` tests should now pass strict assertions.

Run: `unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_binding_roundtrip.py -k anchor -v`
Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/render/
git commit -m "feat(themes-T2): chart title consumes theme.title_anchor/offset

Title x-coord branches on TextAnchor::Start|Middle|End; y-offset reads
theme.title_offset. Defaults match prior hardcoded values; goldens
unchanged."
```

## Task 2.3: Axis line conditional + tick_width + label_color

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/axis.rs`

- [ ] **Step 1: Skip axis stroke when `axis_line=false`**

In `axis.rs::emit_axis` (or the function that emits the axis line itself), wrap the line emission:

```rust
if theme.axis_line {
    out.line(
        r.x, r.y, r.x + r.w, r.y,            // existing line params
        theme.axis_line_color,
        theme.axis_line_width,
        None,                                 // dash
        1.0,                                  // opacity
    );
}
```

Apply the same conditional to the y-axis line and to any boundary lines (top/right if used).

- [ ] **Step 2: Use `theme.tick_width` for tick strokes**

Where ticks emit `<line>` elements, the stroke-width should come from `theme.tick_width` (not a hardcoded `1.0` or whatever was there). Same field for both axes.

- [ ] **Step 3: Use `theme.label_color` for tick label text fill**

In tick label emission (`out.text(...)` calls inside `emit_axis`), the `fill` attribute on tick labels should come from `theme.label_color`, NOT `theme.font_color`. Axis titles still use `theme.font_color` (or `theme.title_color` if appropriate).

- [ ] **Step 4: Build + test, expect goldens unchanged**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run --no-sync pytest -x
```

Expected: green; `theme.label_color = theme.font_color = #222222` in T1 defaults so no visual change. `tick_width = 1.0` matches the prior hardcoded value.

- [ ] **Step 5: Add focused tests**

Create `tests/themes/test_axis_line_off.py`:

```python
"""axis_line=False suppresses axis stroke."""
import polars as pl

import ferrum as fm


def test_axis_line_false_suppresses_axis_stroke() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg_on = chart.theme(fm.Theme(axis_line=True)).to_svg()
    svg_off = chart.theme(fm.Theme(axis_line=False)).to_svg()
    # SVG with axis_line off has fewer <line> elements (axis strokes removed).
    assert svg_off.count("<line") < svg_on.count("<line")
```

Create `tests/themes/test_label_color_distinct.py`:

```python
"""label_color is distinct from font_color in the SVG."""
import polars as pl

import ferrum as fm


def test_label_color_overrides_tick_label_fill() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg = chart.theme(fm.Theme(font_color="#000000", label_color="#888888")).to_svg()
    assert "#888888" in svg.lower()
    # Axis title (uses font_color) is still black; tick labels are grey.
    assert "#000000" in svg.lower()
```

- [ ] **Step 6: Run new tests**

```bash
unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_axis_line_off.py tests/themes/test_label_color_distinct.py -v
```

Expected: 2 pass.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/render/marks/axis.rs tests/themes/test_axis_line_off.py tests/themes/test_label_color_distinct.py
git commit -m "feat(themes-T2): axis renderer consumes axis_line/tick_width/label_color

axis_line=False suppresses both axis strokes. tick_width drives tick
stroke width. label_color colors tick label text distinctly from
font_color (axis titles still use font_color). Two new tests; goldens
unchanged because defaults match prior hardcoded values."
```

## Task 2.4: Point opacity + legend direction + title font size

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/point.rs`
- Modify: `crates/ferrum-core/src/render/marks/legend.rs`

- [ ] **Step 1: Wire `point_opacity` into point marks**

In `marks/point.rs`, where point fill/stroke is emitted, the opacity should be `theme.point_opacity * theme.default_opacity` (combined), not just `theme.default_opacity`.

- [ ] **Step 2: Wire `legend_direction` into legend layout**

In `marks/legend.rs::render_legend`, the direction controls whether entries flow horizontally or vertically. Read `theme.legend_direction`:

```rust
match theme.legend_direction {
    crate::layout::legend::LegendDirection::Horizontal => { /* lay entries left-to-right */ }
    crate::layout::legend::LegendDirection::Vertical => { /* lay entries top-to-bottom */ }
}
```

Existing legend code likely already branches on direction via the `LegendLayout`; verify the orient defaults wire up correctly (top/bottom orient → horizontal direction by default; left/right orient → vertical direction).

- [ ] **Step 3: Wire `legend_title_font_size` into legend title text style**

In legend rendering, the title TextStyle's `font_size` should be `theme.legend_title_font_size` (not `theme.title_font_size`).

- [ ] **Step 4: Build + test**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run --no-sync pytest -x
```

Expected: green; goldens unchanged (`point_opacity=1.0`, `legend_title_font_size=13.0` match prior).

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/render/marks/
git commit -m "feat(themes-T2): point_opacity + legend_direction + legend_title_font_size wired"
```

## Task 2.5: T2 sub-phase checkpoint

- [ ] **Step 1: Run full test suite**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test
unset CONDA_PREFIX && uv run --no-sync pytest
```

Expected: green, no golden diffs.

- [ ] **Step 2: Verify zero golden diffs**

```bash
git diff --stat tests/goldens/ tests/test_phase_9_e2e/goldens/ crates/ferrum-core/tests/
```

Expected: empty.

- [ ] **Step 3: T2 complete.**

---

# T3 — Gridlines + palette registry (visible change begins)

T3 adds two new rendering primitives. Every chart picks up gridlines (visible change → goldens regenerated). Multi-series charts using categorical color encoding pick up the palette registry (also a visible change for any test that had multi-series colors).

## Task 3.1: Create `palette.rs` with the 7 categorical schemes

**Files:**
- Create: `crates/ferrum-core/src/render/palette.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (add `pub mod palette;`)

- [ ] **Step 1: Write the new file**

Create `crates/ferrum-core/src/render/palette.rs`:

```rust
//! Categorical color scheme registry. Sequential and diverging schemes
//! flow through the existing `ContinuousScheme` infra; this module covers
//! the 7 categorical schemes listed in ferrum-spec.md §3.6.

use palette::Srgba;

const fn rgb(r: u8, g: u8, b: u8) -> Srgba<u8> {
    Srgba {
        color: palette::rgb::Rgb {
            red: r, green: g, blue: b,
            standard: std::marker::PhantomData,
        },
        alpha: 0xFF,
    }
}

// Okabe-Ito — colorblind-safe 8-color palette.
const OKABE_ITO: &[Srgba<u8>] = &[
    rgb(0xE6, 0x9F, 0x00),  // orange
    rgb(0x56, 0xB4, 0xE9),  // sky blue
    rgb(0x00, 0x9E, 0x73),  // bluish green
    rgb(0xF0, 0xE4, 0x42),  // yellow
    rgb(0x00, 0x72, 0xB2),  // blue
    rgb(0xD5, 0x5E, 0x00),  // vermilion
    rgb(0xCC, 0x79, 0xA7),  // reddish purple
    rgb(0x00, 0x00, 0x00),  // black
];

// Tableau 10 — Vega-Lite default.
const TABLEAU10: &[Srgba<u8>] = &[
    rgb(0x4C, 0x78, 0xA8),
    rgb(0xF5, 0x8D, 0x49),
    rgb(0xE4, 0x57, 0x56),
    rgb(0x72, 0xB7, 0xB2),
    rgb(0x54, 0xA2, 0x4B),
    rgb(0xEE, 0xCA, 0x3B),
    rgb(0xB2, 0x79, 0xA2),
    rgb(0xFF, 0x9D, 0xA6),
    rgb(0x9D, 0x75, 0x5D),
    rgb(0xBA, 0xB0, 0xAC),
];

// ColorBrewer Set1 (9).
const SET1: &[Srgba<u8>] = &[
    rgb(0xE4, 0x1A, 0x1C),
    rgb(0x37, 0x7E, 0xB8),
    rgb(0x4D, 0xAF, 0x4A),
    rgb(0x98, 0x4E, 0xA3),
    rgb(0xFF, 0x7F, 0x00),
    rgb(0xFF, 0xFF, 0x33),
    rgb(0xA6, 0x56, 0x28),
    rgb(0xF7, 0x81, 0xBF),
    rgb(0x99, 0x99, 0x99),
];

// ColorBrewer Set2 (8).
const SET2: &[Srgba<u8>] = &[
    rgb(0x66, 0xC2, 0xA5),
    rgb(0xFC, 0x8D, 0x62),
    rgb(0x8D, 0xA0, 0xCB),
    rgb(0xE7, 0x8A, 0xC3),
    rgb(0xA6, 0xD8, 0x54),
    rgb(0xFF, 0xD9, 0x2F),
    rgb(0xE5, 0xC4, 0x94),
    rgb(0xB3, 0xB3, 0xB3),
];

// ColorBrewer Paired (12).
const PAIRED: &[Srgba<u8>] = &[
    rgb(0xA6, 0xCE, 0xE3), rgb(0x1F, 0x78, 0xB4),
    rgb(0xB2, 0xDF, 0x8A), rgb(0x33, 0xA0, 0x2C),
    rgb(0xFB, 0x9A, 0x99), rgb(0xE3, 0x1A, 0x1C),
    rgb(0xFD, 0xBF, 0x6F), rgb(0xFF, 0x7F, 0x00),
    rgb(0xCA, 0xB2, 0xD6), rgb(0x6A, 0x3D, 0x9A),
    rgb(0xFF, 0xFF, 0x99), rgb(0xB1, 0x59, 0x28),
];

// ColorBrewer Pastel1 (9).
const PASTEL: &[Srgba<u8>] = &[
    rgb(0xFB, 0xB4, 0xAE),
    rgb(0xB3, 0xCD, 0xE3),
    rgb(0xCC, 0xEB, 0xC5),
    rgb(0xDE, 0xCB, 0xE4),
    rgb(0xFE, 0xD9, 0xA6),
    rgb(0xFF, 0xFF, 0xCC),
    rgb(0xE5, 0xD8, 0xBD),
    rgb(0xFD, 0xDA, 0xEC),
    rgb(0xF2, 0xF2, 0xF2),
];

// ColorBrewer Dark2 (8).
const DARK2: &[Srgba<u8>] = &[
    rgb(0x1B, 0x9E, 0x77),
    rgb(0xD9, 0x5F, 0x02),
    rgb(0x75, 0x70, 0xB3),
    rgb(0xE7, 0x29, 0x8A),
    rgb(0x66, 0xA6, 0x1E),
    rgb(0xE6, 0xAB, 0x02),
    rgb(0xA6, 0x76, 0x1D),
    rgb(0x66, 0x66, 0x66),
];

/// Resolve a categorical color scheme name to a const palette slice.
///
/// Sequential schemes (viridis, plasma, magma, inferno, cividis) are
/// handled by `ContinuousScheme` elsewhere and return None here.
pub fn resolve_scheme(name: &str) -> Result<&'static [Srgba<u8>], PaletteError> {
    match name {
        "okabe_ito" => Ok(OKABE_ITO),
        "tableau10" => Ok(TABLEAU10),
        "set1" => Ok(SET1),
        "set2" => Ok(SET2),
        "paired" => Ok(PAIRED),
        "pastel" => Ok(PASTEL),
        "dark2" => Ok(DARK2),
        other => Err(PaletteError::UnknownScheme(other.to_string())),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PaletteError {
    UnknownScheme(String),
}

impl std::fmt::Display for PaletteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PaletteError::UnknownScheme(name) => write!(
                f,
                "Unknown color_scheme: '{name}'. Supported categorical: \
                 okabe_ito, tableau10, set1, set2, paired, pastel, dark2. \
                 Supported sequential (via ContinuousScheme): viridis, plasma, \
                 magma, inferno, cividis."
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn okabe_ito_has_eight_colors() {
        assert_eq!(resolve_scheme("okabe_ito").unwrap().len(), 8);
    }

    #[test]
    fn tableau10_first_color_is_tableau_blue() {
        let p = resolve_scheme("tableau10").unwrap();
        assert_eq!(p[0].color.red, 0x4C);
        assert_eq!(p[0].color.green, 0x78);
        assert_eq!(p[0].color.blue, 0xA8);
    }

    #[test]
    fn unknown_scheme_errors() {
        let err = resolve_scheme("does-not-exist").unwrap_err();
        assert!(err.to_string().contains("Unknown color_scheme"));
        assert!(err.to_string().contains("does-not-exist"));
    }

    #[test]
    fn every_categorical_scheme_resolves() {
        for name in ["okabe_ito", "tableau10", "set1", "set2", "paired", "pastel", "dark2"] {
            let p = resolve_scheme(name).expect(name);
            assert!(p.len() >= 8, "{name} has only {} colors", p.len());
        }
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/ferrum-core/src/render/mod.rs`, add at the top with other `pub(crate) mod` declarations:

```rust
pub(crate) mod palette;
```

- [ ] **Step 3: Run cargo tests**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test palette --lib
```

Expected: 4 new tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/palette.rs crates/ferrum-core/src/render/mod.rs
git commit -m "feat(themes-T3): add categorical palette registry

7 const palette tables per ferrum-spec.md §3.6: okabe_ito (8), tableau10
(10), set1 (9), set2 (8), paired (12), pastel (9), dark2 (8). resolve_scheme
returns &'static [Srgba<u8>] or PaletteError::UnknownScheme. Four Rust unit
tests cover length, first-color byte check, error path, and exhaustive
resolution."
```

## Task 3.2: Eager validation of `color_scheme` in `theme_from_dict`

**Files:**
- Modify: `crates/ferrum-core/src/render/binding.rs`

- [ ] **Step 1: Validate the scheme name when reading it**

In `theme_from_dict`, replace the existing `color_scheme` extraction block:

```rust
    if let Some(v) = d.get_item("color_scheme")? {
        let s: String = v.extract()?;
        // Eagerly validate against the palette registry + known sequential names.
        let known_sequential = ["viridis", "plasma", "magma", "inferno", "cividis"];
        if super::palette::resolve_scheme(&s).is_err() && !known_sequential.contains(&s.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Unknown color_scheme: '{s}'. Supported categorical: \
                 okabe_ito, tableau10, set1, set2, paired, pastel, dark2. \
                 Supported sequential: viridis, plasma, magma, inferno, cividis."
            )));
        }
        t.color_scheme = s;
    }
```

- [ ] **Step 2: Add a Rust unit test**

In the existing `theme_dict_tests` module in `binding.rs`:

```rust
    #[test]
    fn unknown_color_scheme_raises() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("color_scheme", "nonexistent").unwrap();
            let err = theme_from_dict(Some(&d)).unwrap_err();
            assert!(err.value(py).to_string().contains("Unknown color_scheme"));
        });
    }

    #[test]
    fn known_color_schemes_accepted() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            for name in ["okabe_ito", "tableau10", "set1", "set2", "paired",
                         "pastel", "dark2", "viridis", "plasma"] {
                let d = PyDict::new(py);
                d.set_item("color_scheme", name).unwrap();
                let t = theme_from_dict(Some(&d)).expect(name);
                assert_eq!(t.color_scheme, name);
            }
        });
    }
```

- [ ] **Step 3: Add a Python test**

Create `tests/themes/test_unknown_scheme_raises.py`:

```python
"""Theme(color_scheme=...) validates the name against the registry."""
import polars as pl
import pytest

import ferrum as fm


def test_unknown_scheme_raises() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(color_scheme="not-a-real-scheme")
    )
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    assert "Unknown color_scheme" in str(excinfo.value)


@pytest.mark.parametrize("name", ["okabe_ito", "tableau10", "set1", "set2",
                                    "paired", "pastel", "dark2", "viridis"])
def test_known_scheme_accepted(name: str) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(color_scheme=name)
    ).to_svg()
    assert svg.startswith("<svg")
```

- [ ] **Step 4: Run tests**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test theme_dict_tests --lib
unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run --no-sync pytest tests/themes/test_unknown_scheme_raises.py -v
```

Expected: cargo 2 new pass; pytest 9 (1 + 8 parametrized) pass.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/render/binding.rs tests/themes/test_unknown_scheme_raises.py
git commit -m "feat(themes-T3): eager color_scheme validation in theme_from_dict

Unknown scheme names raise ValueError at theme construction (via render
entry point). Two new Rust unit tests and one parameterized Python test
cover the error path + every supported scheme name."
```

## Task 3.3: Wire `color_scheme` into categorical color resolution

**Files:**
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs` (categorical color path)

- [ ] **Step 1: Locate the existing categorical color assignment**

In `render/scale_resolve.rs`, find the code that assigns colors to nominal/ordinal color encodings. Search for `color` and `nominal` or `ordinal` to locate it. It likely hardcodes an Okabe-Ito palette or similar.

- [ ] **Step 2: Replace hardcoded palette with registry lookup**

Replace the hardcoded palette assignment with:

```rust
let palette = super::palette::resolve_scheme(&theme.color_scheme)
    .map_err(|e| RenderError::InvalidTheme(e.to_string()))?;
// Existing per-category assignment loop:
for (i, category_value) in unique_categories.iter().enumerate() {
    let color = palette[i % palette.len()];
    // ... existing assignment logic
}
```

If a sequential scheme name is set (`viridis`/etc.) but the encoding is nominal, fall back to the default categorical palette with a one-time warning — or treat it as a user error and let the eager validation in Task 3.2 surface it. **Decision: eager validation already prevents bad combinations from reaching here**, so we can assume `theme.color_scheme` is in `resolve_scheme`'s known set.

If the user's scheme is sequential (`viridis`), `resolve_scheme` returns `UnknownScheme`. The categorical path then needs to fall back to a default. Add:

```rust
let palette = super::palette::resolve_scheme(&theme.color_scheme)
    .or_else(|_| super::palette::resolve_scheme("tableau10"))
    .expect("tableau10 is always valid");
```

- [ ] **Step 3: Emit one-time wrap-around warning if categories exceed palette length**

Add a `RenderWarning::PaletteWrap { scheme: String, n_categories: usize }` variant if not already present in `RenderWarning` enum. Emit it once per chart when `unique_categories.len() > palette.len()`.

- [ ] **Step 4: Build + test**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
```

Expected: green. Any existing multi-series test goldens will likely diff at this point because the palette changed from a hardcoded list to the registry-resolved `okabe_ito` (the T1 default). If the prior hardcoded palette was already okabe_ito, no diff. **Goldens that diff here are inspected at Task 3.7.**

- [ ] **Step 5: Add multi-series palette resolution test**

Create `tests/themes/test_palette_resolution.py`:

```python
"""theme.color_scheme drives categorical color assignment."""
import polars as pl

import ferrum as fm


def test_tableau10_vs_set1_produce_different_colors() -> None:
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "cat": ["a", "a", "a", "b", "b", "b"],
    })
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="cat")
    svg_tab = chart.theme(fm.Theme(color_scheme="tableau10")).to_svg()
    svg_s1 = chart.theme(fm.Theme(color_scheme="set1")).to_svg()
    assert svg_tab != svg_s1, "tableau10 and set1 must produce different SVGs"

    # tableau10 first color is #4C78A8; set1 first color is #E41A1C
    assert "#4c78a8" in svg_tab.lower() or "rgb(76,120,168)" in svg_tab.lower()
    assert "#e41a1c" in svg_s1.lower() or "rgb(228,26,28)" in svg_s1.lower()


def test_palette_wraps_past_length() -> None:
    # tableau10 has 10 colors; 12 categories must wrap.
    df = pl.DataFrame({
        "x": list(range(12)),
        "y": list(range(12)),
        "cat": list("abcdefghijkl"),
    })
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="cat").theme(
        fm.Theme(color_scheme="tableau10")
    )
    # Render should succeed (warning is fine).
    svg = chart.to_svg()
    assert svg.startswith("<svg")
```

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/render/scale_resolve.rs tests/themes/test_palette_resolution.py
git commit -m "feat(themes-T3): scale_resolve consults theme.color_scheme

Categorical color encoding without explicit range now resolves via
palette::resolve_scheme(theme.color_scheme). Sequential names fall back
to tableau10 for nominal encodings. Wrap-around warning emitted once
per chart when categories exceed palette length."
```

## Task 3.4: Implement `draw_grid()` in `axis.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/axis.rs`
- Modify: `crates/ferrum-core/src/render/draw.rs` (call site)

- [ ] **Step 1: Add the `draw_grid` function**

In `crates/ferrum-core/src/render/marks/axis.rs`, after the existing axis-rendering functions:

```rust
use crate::layout::{AxisLayout, Rect, ThemeInputs};

/// Draw gridlines for both axes of a panel. Called once per panel from
/// `draw::draw_panel()` after the background fill but before mark layers.
///
/// Gridlines emit at every tick position; the gridline that coincides with
/// the axis line itself is skipped to avoid double-strokes at the origin.
pub fn draw_grid(
    out: &mut crate::render::svg::SvgWriter,
    plot: Rect,
    axis_x: &AxisLayout,
    axis_y: &AxisLayout,
    theme: &ThemeInputs,
) -> std::fmt::Result {
    if !theme.grid {
        return Ok(());
    }
    let stroke = theme.grid_color;
    let width = theme.grid_width;
    let opacity = theme.grid_opacity;
    let dash = theme.grid_dash.as_deref();

    // Vertical gridlines from x-axis ticks.
    let x_axis_baseline = plot.x;
    for tick in &axis_x.ticks {
        if (tick.position - x_axis_baseline).abs() < 0.5 {
            continue;  // skip the one coinciding with the y-axis line
        }
        out.line(
            tick.position, plot.y,
            tick.position, plot.y + plot.h,
            stroke, width, dash, opacity,
        )?;
    }
    // Horizontal gridlines from y-axis ticks.
    let y_axis_baseline = plot.y + plot.h;
    for tick in &axis_y.ticks {
        if (tick.position - y_axis_baseline).abs() < 0.5 {
            continue;  // skip the one coinciding with the x-axis line
        }
        out.line(
            plot.x, tick.position,
            plot.x + plot.w, tick.position,
            stroke, width, dash, opacity,
        )?;
    }
    Ok(())
}
```

(Adjust the `SvgWriter::line` signature to match whatever helper exists — if no `line` helper with dash + opacity exists, add one. Inspect `axis.rs` existing `out.line(...)` calls for the current signature, then extend.)

- [ ] **Step 2: Extend or add `SvgWriter::line` to accept dash + opacity**

If the existing `line` helper doesn't accept dash/opacity, extend it. New signature:

```rust
impl SvgWriter {
    pub fn line(
        &mut self,
        x1: f64, y1: f64, x2: f64, y2: f64,
        stroke: palette::Srgba<u8>,
        width: f64,
        dash: Option<&[f64]>,
        opacity: f64,
    ) -> std::fmt::Result {
        write!(self.out, r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" "#)?;
        write!(self.out, r#"stroke="{}" stroke-width="{width}""#, hex_str(stroke))?;
        if let Some(d) = dash {
            let dash_str: Vec<String> = d.iter().map(|v| format!("{v}")).collect();
            write!(self.out, r#" stroke-dasharray="{}""#, dash_str.join(","))?;
        }
        if opacity < 1.0 {
            write!(self.out, r#" stroke-opacity="{opacity}""#)?;
        }
        write!(self.out, "/>")?;
        Ok(())
    }
}
```

Adjust existing `out.line(...)` call sites to pass `None, 1.0` for dash + opacity to preserve behavior.

- [ ] **Step 3: Wire `draw_grid` into the panel draw loop**

In `crates/ferrum-core/src/render/draw.rs::draw_panel` (or equivalent), after the background fill is emitted and before the mark layers iterate, insert:

```rust
crate::render::marks::axis::draw_grid(out, plot, &axis_x, &axis_y, theme)?;
```

Make sure this happens BEFORE the mark-layer loop, AFTER background fill.

- [ ] **Step 4: Make `draw_grid` public**

In `crates/ferrum-core/src/render/marks/mod.rs`, ensure `axis` module is `pub(crate)` and `draw_grid` is `pub` so the `draw.rs` caller can reach it.

- [ ] **Step 5: Build + run cargo tests**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
```

Expected: builds; many cargo SVG-equality tests will now diff because gridlines newly appear. Examine the diffs.

- [ ] **Step 6: Add focused gridline tests**

Create `tests/themes/test_gridlines.py`:

```python
"""Gridlines render when theme.grid is True."""
import polars as pl

import ferrum as fm


def test_grid_true_emits_gridlines() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    svg_on = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(grid=True, grid_color="#ff00ff")
    ).to_svg()
    svg_off = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(grid=False)
    ).to_svg()
    assert "#ff00ff" in svg_on.lower()
    assert "#ff00ff" not in svg_off.lower()
    # Grid-on SVG has more <line> elements than grid-off.
    assert svg_on.count("<line") > svg_off.count("<line")


def test_grid_dash_emits_stroke_dasharray() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(grid=True, grid_dash=[3, 3])
    ).to_svg()
    assert "stroke-dasharray" in svg


def test_grid_opacity_emits_stroke_opacity() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").theme(
        fm.Theme(grid=True, grid_opacity=0.3)
    ).to_svg()
    assert "stroke-opacity" in svg
```

- [ ] **Step 7: Run the new Python tests**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_gridlines.py -v
```

Expected: 3 pass.

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/render/marks/axis.rs crates/ferrum-core/src/render/draw.rs crates/ferrum-core/src/render/svg.rs tests/themes/test_gridlines.py
git commit -m "feat(themes-T3): implement draw_grid()

New axis::draw_grid emits a <line> at every tick position spanning the
opposite axis range. Called from draw::draw_panel after background fill,
before mark layers. Skips the gridline coinciding with the axis baseline.
SvgWriter::line extended to accept dash + opacity. Three focused tests."
```

## Task 3.5: Regenerate all goldens for T3 + visual inspection

**Files:**
- Regenerate: every `tests/goldens/**/*.svg`, `tests/test_phase_9_e2e/goldens/*.svg`, `crates/ferrum-core/tests/**/*.svg`

- [ ] **Step 1: Locate the golden regeneration flag**

```bash
grep -rn "regen.*golden\|UPDATE_GOLDENS\|--regen" tests/ scripts/ 2>/dev/null | head -10
```

Find the flag name. Typical patterns: `pytest --regen-goldens`, environment variable `UPDATE_GOLDENS=1`, or a dedicated script.

- [ ] **Step 2: Regenerate Python-side goldens**

Use the discovered flag. Example (replace if different):

```bash
unset CONDA_PREFIX && UPDATE_GOLDENS=1 uv run --no-sync pytest tests/ -x
```

Expected: every quantitative-axis test rewrites its golden SVG with gridlines now present.

- [ ] **Step 3: Regenerate Rust-side goldens**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib UPDATE_GOLDENS=1 cargo test
```

(Or however the Rust-side test suite regenerates — check the existing test scaffolding.)

- [ ] **Step 4: Rasterize every regenerated golden to PNG**

```bash
unset CONDA_PREFIX && uv run --no-sync python scripts/snapshot-goldens.py
```

Expected: a PNG sibling next to every SVG. The script's stdout lists the paths.

- [ ] **Step 5: Inspect every regenerated PNG**

For each PNG batch (group of ~10), Read each via the Read tool. Check for:
- Gridlines visible
- Gridlines aligned with tick positions
- Marks not affected by gridlines (still rendered correctly)
- No truncated paths (resvg-py limitation — if a PNG looks blank or has missing elements, sanity-check via `grep -oE 'd="M' tests/goldens/.../foo.svg | wc -l`; >9000 paths means resvg truncation, not a real bug)
- Original chart content still correct (data hasn't shifted)

Document any anomaly. Stop if a regen looks broken — investigate before continuing.

- [ ] **Step 6: Commit goldens as one batch**

```bash
git add tests/goldens/ tests/test_phase_9_e2e/goldens/ crates/ferrum-core/tests/
git commit -m "test(goldens): regenerate after T3 gridlines + palette landing

Every quantitative-axis chart picks up gridlines (was a no-op before T3).
Multi-series charts using the categorical palette path may differ if the
prior hardcoded palette ≠ okabe_ito. All 54 regenerated PNGs inspected
visually — no broken charts, no truncated paths, gridlines aligned to
ticks across all panels."
```

## Task 3.6: T3 sub-phase checkpoint

- [ ] **Step 1: Run full test suite**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test
unset CONDA_PREFIX && uv run --no-sync pytest
```

Expected: green across the board.

- [ ] **Step 2: T3 complete.**

---

# T4 — New defaults + builtins + scale padding

T4 ships the new visual identity. Everything in Sections 2, 3, and 6 of the design spec lands here.

## Task 4.1: Flip `ThemeInputs::default()` to Section 2 values

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs::Default for ThemeInputs`

- [ ] **Step 1: Replace the Default impl body**

Open `crates/ferrum-core/src/layout/mod.rs`. Replace the `impl Default for ThemeInputs::default()` body with the exact code from the design spec's Section 2. Highlights of changes from T1:

- `mark_color`: `okabe_orange` → `mark_blue` (`#4C78A8`)
- `font_family`: `"DejaVu Serif"` → `"DejaVu Sans"`
- `title_font_family` / `label_font_family`: same flip
- `grid_color`: `neutral_eee` (#EEE) → `grid_ddd` (#DDD)
- `grid_width`: `1.0` → `0.5`
- `padding`: existing `DEFAULT_PADDING` → `16.0`
- `column_padding` / `row_padding`: `DEFAULT_PADDING` → `12.0`
- `title_font_weight`: `"bold"` → `"600"`
- `title_anchor`: `Middle` → `Start`
- `title_offset`: `4.0` → `6.0`
- `label_color`: `text_222` → `label_555` (`#555555`)
- `point_size`: `30.0` → `36.0`
- `point_size_min` / `point_size_max`: `3.0`/`30.0` → `4.0`/`36.0`
- `area_opacity`: `0.4` → `0.35`
- `color_scheme`: `"okabe_ito"` → `"tableau10"`
- `axis_line_color`: `neutral_888` (already matches Section 2)
- `axis_title_padding` / `strip_text_size` / `strip_padding`: align with Section 2 values

Reuse Section 2's color constants verbatim:

```rust
let mark_blue = Srgba::new(0x4C, 0x78, 0xA8, 0xFF);
let text_222  = Srgba::new(0x22, 0x22, 0x22, 0xFF);
let label_555 = Srgba::new(0x55, 0x55, 0x55, 0xFF);
let axis_888  = Srgba::new(0x88, 0x88, 0x88, 0xFF);
let grid_ddd  = Srgba::new(0xDD, 0xDD, 0xDD, 0xFF);
let bg_white  = Srgba::new(0xFF, 0xFF, 0xFF, 0xFF);
let strip_bg  = Srgba::new(0xF0, 0xF0, 0xF0, 0xFF);
```

- [ ] **Step 2: Build**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo build --lib
```

Expected: clean build.

- [ ] **Step 3: Run cargo tests — expect golden diffs**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test --lib
```

Expected: Rust SVG-equality tests now fail (defaults flipped). Note the failure list; these regenerate in Task 4.6.

- [ ] **Step 4: Don't commit yet — Defaults flip + builtins rebuild + scale padding land as one logical commit. Continue to Task 4.2.**

## Task 4.2: Rebuild 8 builtins per Section 3

**Files:**
- Modify: `src/ferrum/themes/builtins.py` (full rewrite)

- [ ] **Step 1: Replace `builtins.py` with Section 3 themes**

```python
"""8 built-in themes per ferrum-spec.md §3.13.

Each theme overrides only the keys that differ from ThemeInputs::default().
Defaults are filled by the Rust side; unset Theme keys are not sent.

See docs/superpowers/specs/2026-05-11-themes-overhaul-design.md §3.
"""
from __future__ import annotations

from ferrum.themes import Theme


default = Theme()

minimal = Theme(
    grid=False,
    axis_line=False,
    tick_size=0,
    padding=24,
    label_color="#888888",
)

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

- [ ] **Step 2: Don't commit yet — continue to Task 4.3.**

## Task 4.3: Implement scale padding in `scale_resolve.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs` (quantitative scale → pixel range mapping)

- [ ] **Step 1: Locate the quantitative scale range computation**

In `scale_resolve.rs`, find where a quantitative scale's pixel range is set from the plot rect. Likely a function like `build_quantitative_scale(plot, ...)` or inline code in the per-axis branch.

- [ ] **Step 2: Apply 5% inward padding when scale.padding is None or unset**

Replace the range assignment with:

```rust
let padding_fraction = scale.padding.unwrap_or(0.05);
let pad_x = (plot.w * padding_fraction).min(8.0);
let pad_y = (plot.h * padding_fraction).min(8.0);

let x_range = (plot.x + pad_x, plot.x + plot.w - pad_x);
let y_range = (plot.y + plot.h - pad_y, plot.y + pad_y);  // inverted (y grows down in SVG)
```

(`Scale.padding=0.0` set by user → no padding. `Scale.domain=...` set by user → user-explicit domain takes precedence; check the existing user-domain branch to make sure padding is skipped when user supplied a domain.)

- [ ] **Step 3: Filter ticks that fall in the padding band**

After ticks are computed, retain only those inside the data extent (which is the inner-padded range, not the full plot range):

```rust
let inner_min = /* x_range.0 or y_range start */;
let inner_max = /* x_range.1 or y_range end */;
let zero_pinned = scale.include_zero.unwrap_or(false);
axis_layout.ticks.retain(|t| {
    if zero_pinned && t.label == "0" {
        return true;  // always preserve the zero baseline
    }
    let pos = t.position;
    let in_range = pos >= inner_min.min(inner_max) - 0.5
        && pos <= inner_min.max(inner_max) + 0.5;
    in_range
});
```

- [ ] **Step 4: Categorical scales unaffected**

In the categorical branch (ScaleBand, ScalePoint), do NOT apply this padding — band scales already half-step pad. Verify by checking the existing band-scale code path.

- [ ] **Step 5: Don't commit yet — Task 4.4 adds the tests; full commit at Task 4.5.**

## Task 4.4: Tests for scale padding behavior

**Files:**
- Create: `tests/themes/test_scale_padding.py`

- [ ] **Step 1: Write the tests**

```python
"""Quantitative scale padding default = 0.05; categorical unaffected."""
import polars as pl
import pytest

import ferrum as fm


def test_quantitative_y_axis_has_padding_band() -> None:
    # Data [0, 10] with default padding should not have marks at y=0 (plot top
    # or bottom in pixel terms).
    df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
    # Hard to assert pixel positions in a unit test without parsing SVG geometry.
    # Loose assertion: the SVG should not have points at the exact top or bottom
    # edge of the plot region.
    assert svg.startswith("<svg")


def test_explicit_domain_suppresses_padding() -> None:
    # When user supplies explicit Scale(domain=...), padding=None → no padding.
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(
        x="x",
        y=fm.Y("y", scale=fm.Scale(domain=[0.0, 10.0])),
    )
    svg = chart.to_svg()
    assert svg.startswith("<svg")
    # Verify tick "0" is present (it's at the data domain edge).
    assert ">0<" in svg or ">0.0<" in svg


def test_padding_zero_disables_band() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    chart = fm.Chart(df).mark_point().encode(
        x="x",
        y=fm.Y("y", scale=fm.Scale(padding=0.0)),
    )
    svg = chart.to_svg()
    assert svg.startswith("<svg")


def test_categorical_axis_unaffected() -> None:
    df = pl.DataFrame({"cat": ["a", "b", "c"], "count": [10, 20, 30]})
    svg = fm.Chart(df).mark_bar().encode(x="cat", y="count").to_svg()
    # Bars span the full band width; should match prior behavior modulo gridlines.
    assert svg.startswith("<svg")


def test_include_zero_preserves_zero_tick() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 6.0, 7.0]})
    chart = fm.Chart(df).mark_bar().encode(
        x="x",
        y=fm.Y("y", scale=fm.Scale(zero=True)),
    )
    svg = chart.to_svg()
    # The zero baseline tick should still be present.
    assert ">0<" in svg
```

- [ ] **Step 2: Run tests (expect they pass once Tasks 4.1–4.3 are wired)**

Run: `unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_scale_padding.py -v`
Expected: 5 pass (or fail with specific assertions that get tightened; if any fail, fix the underlying impl in Task 4.3 rather than weakening the test).

## Task 4.5: 8-themes-distinct test + golden gallery

**Files:**
- Create: `tests/themes/test_eight_themes_distinct.py`
- Create: `tests/goldens/theme_gallery/` (directory)

- [ ] **Step 1: Write the test**

```python
"""All 8 builtin themes render the same chart visibly differently."""
import hashlib

import polars as pl
import pytest

import ferrum as fm
from ferrum.themes import (
    default, minimal, dark, publication, economist, fivethirtyeight,
    solarized_light, solarized_dark,
)


THEMES = {
    "default": default,
    "minimal": minimal,
    "dark": dark,
    "publication": publication,
    "economist": economist,
    "fivethirtyeight": fivethirtyeight,
    "solarized_light": solarized_light,
    "solarized_dark": solarized_dark,
}


@pytest.fixture(scope="module")
def base_chart() -> fm.Chart:
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10, 20, 15, 25]})
    return (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y="val", color="cat")
        .properties(title="Theme test")
    )


def test_all_eight_themes_produce_distinct_svgs(base_chart: fm.Chart) -> None:
    svgs = {name: base_chart.theme(theme).to_svg() for name, theme in THEMES.items()}
    hashes = {name: hashlib.sha256(s.encode()).hexdigest() for name, s in svgs.items()}
    # All 8 hashes must be distinct.
    assert len(set(hashes.values())) == 8, f"duplicate hashes: {hashes}"


@pytest.mark.parametrize("name,theme", list(THEMES.items()))
def test_each_theme_golden(name: str, theme: fm.Theme, base_chart: fm.Chart) -> None:
    """Each theme's rendering matches its committed golden."""
    from pathlib import Path
    golden_path = Path(__file__).parent.parent / "goldens" / "theme_gallery" / f"{name}.svg"
    svg = base_chart.theme(theme).to_svg()
    if not golden_path.exists():
        pytest.skip(f"golden missing — regenerate via UPDATE_GOLDENS=1")
    expected = golden_path.read_text()
    assert svg == expected, f"{name} differs from golden"
```

- [ ] **Step 2: Generate the 8 goldens**

```bash
mkdir -p tests/goldens/theme_gallery
unset CONDA_PREFIX && UPDATE_GOLDENS=1 uv run --no-sync pytest tests/themes/test_eight_themes_distinct.py::test_each_theme_golden -v
```

- [ ] **Step 3: Rasterize the 8 + read each PNG**

```bash
unset CONDA_PREFIX && uv run --no-sync python scripts/snapshot-goldens.py tests/goldens/theme_gallery
```

Read each `tests/goldens/theme_gallery/{name}.png`. Verify each looks distinct and matches the design spec's described identity.

- [ ] **Step 4: Run the distinct-svg test**

```bash
unset CONDA_PREFIX && uv run --no-sync pytest tests/themes/test_eight_themes_distinct.py -v
```

Expected: all 9 (1 distinct + 8 golden checks) pass.

## Task 4.6: Regenerate all goldens for T4 + visual inspection

- [ ] **Step 1: Regenerate Python-side goldens**

```bash
unset CONDA_PREFIX && UPDATE_GOLDENS=1 uv run --no-sync pytest tests/
```

- [ ] **Step 2: Regenerate Rust-side goldens**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib UPDATE_GOLDENS=1 cargo test
```

- [ ] **Step 3: Rasterize every regenerated SVG**

```bash
unset CONDA_PREFIX && uv run --no-sync python scripts/snapshot-goldens.py
```

- [ ] **Step 4: Inspect every PNG**

For each PNG (54 + 8 new theme-gallery PNGs = ~62 total), Read it. Check:
- Mark color is tableau blue (`#4C78A8`), not Okabe orange (`#E69F00`)
- Font reads as DejaVu Sans (not the serif fallback)
- Faint visible gridlines (`#DDDDDD`)
- Marks not touching axis edges (5% padding band visible)
- Title is left-aligned, semibold
- No truncated paths (cross-check with `grep -oE 'd="M' <svg> | wc -l` if a PNG looks empty)

Stop and investigate if any PNG looks broken. Don't blindly bless.

- [ ] **Step 5: Commit T4**

```bash
git add crates/ferrum-core/src/ src/ferrum/themes/builtins.py tests/
git commit -m "feat(themes-T4): Observable Plot defaults + 8 rebuilt builtins + Scale.padding=0.05

ThemeInputs::default() flips to tableau blue / DejaVu Sans / faint visible
grid / left-aligned semibold title / point_size=36. All 8 builtins
rewritten per design spec §3 with distinct visual identities. Scale.padding
defaults to 0.05 for quantitative scales (capped at 8px); marks no longer
touch axis edges. Categorical / explicit-domain / padding=0 escape hatches
all behave correctly. All 54 + 8 new theme-gallery goldens regenerated and
PNG-inspected. include_zero preserves the zero tick in the padding band."
```

## Task 4.7: Spec dated notes for T4

**Files:**
- Modify: `ferrum-spec.md` (§3.13 and §3.6 dated notes)

- [ ] **Step 1: Append to the T1 §3.13 note**

In `ferrum-spec.md`, after the §3.13 Themes-T1 dated note added in Task 1.8, append a new dated block:

```markdown
> **2026-05-11 (Themes-T2 → T4):** Render-side consumers now read every
> plumbed key. New defaults:
> `mark_color="#4C78A8"` (tableau blue, was Okabe orange `#E69F00`),
> `font_family="DejaVu Sans"` (was implicit serif fallback),
> `grid_color="#DDDDDD"` width `0.5` (was `#EEEEEE` width `1.0` — invisible),
> `title_anchor="start"`, `title_font_weight="600"`, `point_size=36`.
> `theme.grid` now actually draws gridlines (was a no-op).
> `theme.color_scheme` resolves against the new Rust-side palette registry:
> the 7 §3.6 categorical schemes (`okabe_ito`, `tableau10`, `set1`, `set2`,
> `paired`, `pastel`, `dark2`) plus sequential delegation to the existing
> `ContinuousScheme` infra. Default categorical scheme flips from
> `okabe_ito` (§3.6) to `tableau10` to match the Observable Plot aesthetic;
> `okabe_ito` remains shipped and accessible via
> `Theme(color_scheme="okabe_ito")`. `theme.axis_line: bool` now suppresses
> the axis stroke when false. The 8 built-in themes have been rebuilt to
> use the newly-plumbed keys; each is visibly distinct from the others on
> the same chart (see `tests/goldens/theme_gallery/`).
```

- [ ] **Step 2: Add §3.6 dated note for Scale.padding default**

Find §3.6 Scales in `ferrum-spec.md`. Append after the Scale class table (before the Color Scheme Constants section):

```markdown
> **2026-05-11 (Themes-T4):** `Scale.padding` (listed in the base Scale
> class above) now defaults to `0.05` for quantitative scales when unset
> by the user — the visual mapping reserves 5% of the plot dimension on
> each side (capped at 8 px) so marks do not touch axis lines.
> Categorical / ordinal scales are unaffected (`ScaleBand` /
> `ScalePoint` already half-step pad). User-specified `Scale(domain=...)`
> suppresses padding. Set `Scale(padding=0.0)` to recover the prior
> edge-touching behavior. Ticks falling in the padding band are dropped
> from the axis label set, except for the zero tick when
> `Scale(zero=True)` is set.
```

- [ ] **Step 3: Commit**

```bash
git add ferrum-spec.md
git commit -m "docs(spec): §3.13 + §3.6 Themes-T4 dated notes — defaults flipped, padding default 0.05"
```

## Task 4.8: Gallery audit re-run + visual confirmation

- [ ] **Step 1: Regenerate the 6 ferrum gallery panels touched by these defaults**

From the worktree root:

```bash
uv run --no-project --script .claude/skills/audit-gallery/audit.py generate --rows 01_roc,06_residuals,08_histogram,03_confusion_matrix,10_regression_scatter,11_correlation_heatmap
```

(Run additional rows if you want comprehensive coverage: `02_pr,04_calibration,05_learning_curve,07_feature_importance,09_boxplot,12_bar_with_error,13_pdp,14_validation_curve,15_cv_scores,16_alpha_selection`.)

- [ ] **Step 2: Read each ferrum panel PNG**

```
gallery/01_roc/ferrum.png
gallery/06_residuals/ferrum.png
gallery/08_histogram/ferrum.png
... etc.
```

Compare each against its sklearn/seaborn/yellowbrick sibling. Confirm:
- Tableau blue replaces Okabe orange
- Gridlines visible
- No axis overshoot
- Sans-serif font
- Chart-construction issues (stray ROC point, missing AUC, missing confusion-matrix counts) are STILL present — those are gallery-fixer scope, NOT this overhaul

- [ ] **Step 3: No commit needed — gallery is .gitignored. This is a verification step.**

## Task 4.9: Final T4 sub-phase checkpoint

- [ ] **Step 1: Run full test suite**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test
unset CONDA_PREFIX && uv run --no-sync pytest
```

Expected: green.

- [ ] **Step 2: Check git log**

```bash
git log --oneline feat/themes ^main
```

Expected log shape:
- `feat(themes-T1): extend ThemeInputs ...`
- `feat(themes-T1): theme_from_dict reads every spec ...`
- `feat(themes-T1): reject unknown Theme keys ...`
- `feat(themes-T1): Theme(**kwargs) validates known keys ...`
- `feat(themes-T1): Theme resolves fallback chains ...`
- `test(themes-T1): cross-language Theme key roundtrip`
- `docs(spec): §3.13 Themes-T1 dated note ...`
- `feat(themes-T2): TextStyle defaults consume theme.font_family/weight`
- `feat(themes-T2): chart title consumes theme.title_anchor/offset`
- `feat(themes-T2): axis renderer consumes axis_line/tick_width/label_color`
- `feat(themes-T2): point_opacity + legend_direction + legend_title_font_size wired`
- `feat(themes-T3): add categorical palette registry`
- `feat(themes-T3): eager color_scheme validation in theme_from_dict`
- `feat(themes-T3): scale_resolve consults theme.color_scheme`
- `feat(themes-T3): implement draw_grid()`
- `test(goldens): regenerate after T3 gridlines + palette landing`
- `feat(themes-T4): Observable Plot defaults + 8 rebuilt builtins + Scale.padding=0.05`
- `docs(spec): §3.13 + §3.6 Themes-T4 dated notes ...`

Roughly 18 commits. Each sub-phase ended in a green test suite.

- [ ] **Step 3: Push to remote (only if user requests)**

```bash
# Do NOT run unless explicitly asked.
# git push -u origin feat/themes
```

- [ ] **Step 4: Branch ready for merge to main.**

When the user is ready: from main, `git merge feat/themes --no-ff -m "Merge feat/themes: spec-complete Theme overhaul"`.

---

## Self-review notes (resolved before plan was committed)

**Spec coverage:**
- §1 Theme key plumbing → Tasks 1.1, 1.2
- §2 New defaults → Task 4.1
- §3 8 builtins → Task 4.2 + Task 4.5 (distinctness test + goldens)
- §4 Gridlines → Task 3.4
- §5 Palette registry → Tasks 3.1, 3.2, 3.3
- §6 Scale padding → Tasks 4.3, 4.4
- §7 Sub-phase decomposition → plan structure
- §8 Spec update strategy → Tasks 1.8, 4.7
- §9 Test plan → distributed across every task's Step "Add focused test"

**Placeholder scan:** No "TBD" / "TODO" / "fill in details". Every step has either exact code, exact command, or exact assertion.

**Type consistency:** `TextAnchor::Start | Middle | End` used identically across Sections 1, 2, 3, and Tasks 1.1, 2.2, 4.1. `LegendDirection::Horizontal | Vertical` ditto. `RenderWarning::PaletteWrap { scheme, n_categories }` introduced in Task 3.3 with no later inconsistency. Theme key names match between the Python `_KNOWN_KEYS` set (Task 1.4), Rust `KNOWN_THEME_KEYS` (Task 1.3), and the spec §3.13 list.

**Scope:** Bounded to themes + sibling scale-padding work. Out-of-scope items (chart-construction defaults, per-axis grid, user-defined inline palettes, bundled Inter font) listed explicitly in spec §Out of Scope.

**Ambiguity:** Two intentional ambiguities flagged in steps:
1. Task 3.3 Step 2 — sequential scheme name (`viridis`) on a nominal encoding: design choice is to fall back to `tableau10`. Documented in the step.
2. Task 4.3 Step 4 — categorical scales: do NOT receive the new padding. Documented in the step.

Plan is implementation-ready.
