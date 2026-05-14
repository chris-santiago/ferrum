# Phase 11d — Coordinate Systems + Deferred Marks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver all four coordinate systems (CoordCartesian, CoordFixed, CoordPolar, CoordGeo), three deferred marks (mark_arc, mark_label, mark_geoshape), and coord-awareness for mark_image. After 11d there are zero `NotImplementedError`s for coordinate classes and zero `deferred_mark_error` calls for arc/label/geoshape.

**Architecture:** Extend the Rust spec-side `CoordKind` (in `crates/ferrum-core/src/spec/coord.rs`) from a bare `Cartesian | Flip` enum to a field-carrying enum matching the scene-side `ferrum_scene::CoordKind`. The Python `Chart.coord()` method and `CoordCartesian`/`CoordFixed`/`CoordPolar`/`CoordGeo` classes serialize coord parameters as dicts via the existing `pyo3_serde::from_py` pattern. `scene_build.rs` reads the spec-side `CoordKind` to drive scale domain overrides, aspect-ratio constraints, polar coordinate transforms, and geo projection math. New mark modules (`arc.rs`, `geoshape.rs`, `label.rs`) are added to the render pipeline. A `projection.rs` module in `ferrum-core` implements the six map projection forward/inverse functions.

**Tech Stack:** Rust (serde, geojson 0.24.x for mark_geoshape), Python (dataclasses for coord classes). No other new external dependencies.

**Spec:** `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` sections 7 (Coordinate systems), 10.6 (Python coord classes), 12.4 (Testing).

**Depends on:** 11a (done — SceneGraph IR, `ferrum-scene` crate with `CoordKind` / `GeoProjection` types).

---

## Critical integration note: two CoordKind enums

There are two `CoordKind` enums that must be kept in sync:

1. **Spec-side:** `crates/ferrum-core/src/spec/coord.rs::CoordKind` — currently `Cartesian | Flip` (no fields). This is what flows through `ChartSpec.coord` from Python.
2. **Scene-side:** `crates/ferrum-scene/src/types.rs::CoordKind` — already has all four variants with full fields (xlim, ylim, ratio, theta, projection, etc.). This is what `scene_build.rs` emits into `Panel.coord`.

The spec-side enum must be extended to carry parameters so `scene_build.rs` can convert `spec::CoordKind` to `scene::CoordKind`. The scene-side enum is **not modified** by this plan.

## Theta/Radius channel strategy

Per spec section 7.3, `CoordPolar(theta="x")` means "reinterpret the x encoding channel as the angular axis." The Rust `Encoding` struct does **not** gain theta/radius fields. Instead, `scene_build.rs`'s polar code path reads the existing x/y scales and reinterprets them based on `CoordKind::Polar { theta: X|Y }`.

**Concrete mapping in Python `chart.py`:** The existing `Theta`/`Radius` channel classes (in `src/ferrum/encoding/positional.py`) are currently in `_POLAR_CHANNELS` and are skipped by `to_spec()`. The fix is: in `to_spec()`, when `_coord` is a CoordPolar dict, remap `theta` to `x` (or `y`, per `CoordPolar.theta`) and `radius` to the other axis before building the ChartSpec. Concretely: if `"theta"` is in `enc`, pop it and assign to `enc["x"]` (when `CoordPolar.theta=="x"`) or `enc["y"]` (when `CoordPolar.theta=="y"`); similarly remap `"radius"` to the opposite axis. After remapping, the channels flow through the standard `_RENDERER_HONORED_CHANNELS` path and Rust never sees theta/radius — only x/y reinterpreted by the coord.

The `chart.py` polar channel gate (lines 4388-4398, which raises `NotImplementedError` for theta/radius channels) must be removed.

---

## File map

### New files

| File | Purpose |
|---|---|
| `crates/ferrum-core/src/projection.rs` | Pure-Rust map projection math: 6 `forward`/`inverse` free functions dispatched via match on `GeoProjection` enum |
| `crates/ferrum-core/src/render/marks/arc.rs` | mark_arc builder: pie/donut wedge geometry via `SceneNode::Path` with `PathCmd::ArcTo` |
| `crates/ferrum-core/src/render/marks/geoshape.rs` | mark_geoshape builder: GeoJSON geometry deserialization, projection, `SceneNode::Polygon` emission |
| `crates/ferrum-core/src/render/marks/label.rs` | mark_label builder: positioned text with optional leader lines |
| `tests/test_phase_11d/` | Test directory for coord system and deferred mark tests |
| `tests/test_phase_11d/test_coord_cartesian.py` | CoordCartesian golden SVG tests |
| `tests/test_phase_11d/test_coord_fixed.py` | CoordFixed aspect ratio tests |
| `tests/test_phase_11d/test_coord_polar.py` | CoordPolar + mark_arc golden SVG tests |
| `tests/test_phase_11d/test_coord_geo.py` | CoordGeo + mark_geoshape golden SVG tests |
| `tests/test_phase_11d/test_mark_label.py` | mark_label golden SVG tests |
| `tests/goldens/phase_11d/` | Golden SVGs for all new coord/mark combinations |

### Modified files

| File | Change |
|---|---|
| `crates/ferrum-core/src/spec/coord.rs` | Extend `CoordKind` from 2 variants (Cartesian, Flip) to 6 variants with fields |
| `crates/ferrum-core/src/spec/chart.rs` | Change `coord` param from `Option<&str>` to `Option<&Bound<'_, PyAny>>`, use `pyo3_serde::from_py`; add `geojson_geometries: Option<String>` field |
| `crates/ferrum-core/src/spec/mark.rs` | Add `Arc`, `Geoshape`, `Label` variants to `Mark` enum and `for_each_mark!` macro |
| `crates/ferrum-core/src/render/marks/mod.rs` | Register `arc`, `geoshape`, `label` modules |
| `crates/ferrum-core/src/render/scene_build.rs` | Read `spec.coord` to populate `Panel.coord`; polar coordinate transform code path; geo projection code path |
| `crates/ferrum-core/src/render/scale_resolve.rs` | Honor `xlim`/`ylim` domain overrides from Cartesian/Fixed coord; `expand` flag controls padding |
| `crates/ferrum-core/src/layout/mod.rs` | CoordFixed aspect ratio constraint in `compute_layout()` |
| `crates/ferrum-core/src/lib.rs` | Declare `projection` module |
| `crates/ferrum-core/Cargo.toml` | Add `geojson = "0.24"` dependency |
| `src/ferrum/coord.py` | Replace `NotImplementedError` stubs with frozen dataclasses per spec section 10.6 |
| `src/ferrum/chart.py` | Wire `CoordCartesian`/`CoordFixed`/`CoordPolar`/`CoordGeo` in `Chart.coord()`; remove `deferred_mark_error` for arc/geoshape/label; remove polar channel gate (lines 4388-4398); wire `mark_arc`/`mark_geoshape`/`mark_label` as real marks |
| `src/ferrum/marks/deferred.py` | Remove `arc`, `geoshape`, `label` from `PHASE_9_PLUS_MARKS`; `image` already works |
| `src/ferrum/__init__.py` | Export `CoordCartesian`, `CoordFixed`, `CoordPolar`, `CoordGeo` |
| `src/ferrum/_coerce.py` | GeoJSON detection: split FeatureCollection properties to DataFrame, geometry to JSON string |

### Unchanged files

`ferrum-scene` crate (all scene-side types already defined in 11a), `svg_walk.rs` (SceneNode emission is mark-agnostic), `SvgBuffer`, `rasterize.rs`, `png.rs`, `compositor.rs`, `grid_compose.rs`. Note: `prepare.rs` has `coord_flipped: bool` which pattern-matches `Some(CoordKind::Flip)` -- this match must be updated to `Some(CoordKind::Cartesian { .. })` not matching (since Cartesian is now a struct variant), but the logic is unchanged.

---

## Task 11d0: Coord serialization plumbing (spec-side CoordKind + Python bridge)

**Why first:** Every subsequent task depends on coord parameters flowing from Python to Rust. This is the critical path.

**Files:**
- Modify: `crates/ferrum-core/src/spec/coord.rs`
- Modify: `crates/ferrum-core/src/spec/chart.rs`
- Modify: `src/ferrum/coord.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `src/ferrum/marks/deferred.py`

### Steps

- [ ] **Step 1: Extend spec-side CoordKind**

Replace the current `CoordKind` in `crates/ferrum-core/src/spec/coord.rs`:

```rust
use serde::{Deserialize, Serialize};

/// Coordinate system carried by ChartSpec.
/// Phase 11d extends from bare Cartesian|Flip to full parameterized variants.
///
/// Backward-compatible: old JSON `{"kind":"cartesian"}` deserializes with all
/// fields defaulted to None/true (same behavior as the previous unit variant).
/// Serializing with all defaults produces `{"kind":"cartesian"}` thanks to
/// `skip_serializing_if`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CoordKind {
    Cartesian {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        xlim: Option<(f64, f64)>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ylim: Option<(f64, f64)>,
        #[serde(default = "default_true", skip_serializing_if = "is_true")]
        expand: bool,
        #[serde(default = "default_true", skip_serializing_if = "is_true")]
        clip: bool,
    },
    Flip,
    Fixed {
        #[serde(default = "default_ratio")]
        ratio: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        xlim: Option<(f64, f64)>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ylim: Option<(f64, f64)>,
        #[serde(default = "default_true", skip_serializing_if = "is_true")]
        expand: bool,
        #[serde(default = "default_true", skip_serializing_if = "is_true")]
        clip: bool,
    },
    Polar {
        #[serde(default = "default_theta")]
        theta: String,     // "x" or "y"
        #[serde(default)]
        start: f64,        // radians
        #[serde(default = "default_direction")]
        direction: i8,     // 1 (clockwise) or -1
    },
    Geo {
        #[serde(default = "default_projection")]
        projection: String,
    },
}

fn default_true() -> bool { true }
fn is_true(v: &bool) -> bool { *v }
fn default_ratio() -> f64 { 1.0 }
fn default_theta() -> String { "x".to_string() }
fn default_direction() -> i8 { 1 }
fn default_projection() -> String { "mercator".to_string() }
```

**Backward compat:** The old unit variant `Cartesian` is replaced by a struct variant with all-optional/defaulted fields. `serde(tag = "kind")` means old JSON `{"kind":"cartesian"}` deserializes with all fields defaulted (`xlim: None, ylim: None, expand: true, clip: true`). With `skip_serializing_if`, re-serializing produces the same `{"kind":"cartesian"}`. The `Flip` variant is unchanged. When `Chart.coord()` is not called, `ChartSpec.coord` remains `None`, and `scene_build.rs` defaults to `ferrum_scene::CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true }` on the scene side as it does today.

**Migration note:** Code that previously matched `Some(CoordKind::Cartesian)` must now match `Some(CoordKind::Cartesian { .. })`. The three call sites (chart.rs coord parsing, prepare.rs coord_flipped check, scene_build.rs) must be updated.

Update tests in the same file to cover round-trips for all new variants, including the backward compat case (`{"kind":"cartesian"}` round-trips with defaults).

- [ ] **Step 2: Change PyO3 coord parameter to accept dicts**

In `crates/ferrum-core/src/spec/chart.rs`:

1. Change `coord: Option<&str>` to `coord: Option<&Bound<'_, PyAny>>` in the `#[new]` signature.
2. Replace the string-matching coord parsing block:

```rust
let coord = match coord {
    None => None,
    Some(obj) => {
        // Back-compat: accept bare strings "cartesian" and "flip"
        if let Ok(s) = obj.extract::<String>() {
            match s.as_str() {
                "cartesian" => Some(crate::spec::coord::CoordKind::Cartesian {
                    xlim: None, ylim: None, expand: true, clip: true,
                }),
                "flip" => Some(crate::spec::coord::CoordKind::Flip),
                other => return Err(PyValueError::new_err(format!(
                    "unknown coord kind string: '{other}'"
                ))),
            }
        } else {
            // Dict path: CoordCartesian/Fixed/Polar/Geo serialize as dicts
            Some(crate::pyo3_serde::from_py(obj, "coord")?)
        }
    }
};
```

3. Update the `#[pyo3(signature = (...))]` to change `coord = None` type.

4. Update the `coord()` getter to return the JSON representation:

```rust
#[getter]
fn coord(&self) -> Option<String> {
    self.coord.as_ref().map(|c| serde_json::to_string(c).unwrap_or_default())
}
```

- [ ] **Step 3: Replace Python coord stubs with frozen dataclasses**

Rewrite `src/ferrum/coord.py` per spec section 10.6:

```python
"""Coordinate-system declarations for ferrum charts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class CoordFlip:
    """Flip the x and y axes -- e.g. for horizontal bar charts."""
    # ... (keep existing CoordFlip implementation unchanged)


@dataclass(frozen=True)
class CoordCartesian:
    """Standard Cartesian coordinates with explicit axis limit overrides.

    Parameters
    ----------
    xlim : tuple of (float, float) or None
        Override the x-axis data domain.
    ylim : tuple of (float, float) or None
        Override the y-axis data domain.
    expand : bool
        Add default padding beyond data extent.  ``False`` starts/ends
        exactly at data min/max.
    clip : bool
        Clip marks to the plot area bounds.
    """
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    expand: bool = True
    clip: bool = True

    def _to_spec_dict(self) -> dict:
        d: dict = {"kind": "cartesian"}
        if self.xlim is not None:
            d["xlim"] = list(self.xlim)
        if self.ylim is not None:
            d["ylim"] = list(self.ylim)
        if not self.expand:
            d["expand"] = False
        if not self.clip:
            d["clip"] = False
        return d


@dataclass(frozen=True)
class CoordFixed:
    """Fixed aspect-ratio coordinates.

    Parameters
    ----------
    ratio : float
        Aspect ratio. 1.0 means one data unit on X equals one data
        unit on Y in pixels.
    xlim : tuple of (float, float) or None
        Override the x-axis data domain.
    ylim : tuple of (float, float) or None
        Override the y-axis data domain.
    expand : bool
        Add default padding beyond data extent.
    clip : bool
        Clip marks to the plot area bounds.
    """
    ratio: float = 1.0
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    expand: bool = True
    clip: bool = True

    def _to_spec_dict(self) -> dict:
        d: dict = {"kind": "fixed", "ratio": self.ratio}
        if self.xlim is not None:
            d["xlim"] = list(self.xlim)
        if self.ylim is not None:
            d["ylim"] = list(self.ylim)
        d["expand"] = self.expand
        d["clip"] = self.clip
        return d


@dataclass(frozen=True)
class CoordPolar:
    """Polar coordinates for pie and radial charts.

    Parameters
    ----------
    theta : {"x", "y"}
        Which encoding channel maps to the angular position.
    start : float
        Start angle in radians (0 = 12 o'clock).
    direction : {1, -1}
        1 for clockwise, -1 for counter-clockwise.
    """
    theta: Literal["x", "y"] = "x"
    start: float = 0.0
    direction: Literal[1, -1] = 1

    def _to_spec_dict(self) -> dict:
        return {"kind": "polar", "theta": self.theta,
                "start": self.start, "direction": self.direction}


@dataclass(frozen=True)
class CoordGeo:
    """Geographic map-projection coordinates.

    Parameters
    ----------
    projection : str
        One of "mercator", "albers_usa", "equal_earth",
        "natural_earth", "orthographic", "equirectangular".
    """
    projection: Literal[
        "mercator", "albers_usa", "equal_earth",
        "natural_earth", "orthographic", "equirectangular"
    ] = "mercator"

    def _to_spec_dict(self) -> dict:
        return {"kind": "geo", "projection": self.projection}
```

- [ ] **Step 4: Wire Chart.coord() for all coord types**

In `src/ferrum/chart.py`, update the `coord()` method:

```python
def coord(self, coord: Any) -> "Chart":
    from ferrum.coord import (
        CoordFlip, CoordCartesian, CoordFixed, CoordPolar, CoordGeo,
    )

    new = self._clone()
    if isinstance(coord, CoordFlip):
        new._coord = "flip"
    elif isinstance(coord, (CoordCartesian, CoordFixed, CoordPolar, CoordGeo)):
        new._coord = coord._to_spec_dict()
    else:
        raise TypeError(
            f"unsupported coord: {type(coord).__name__}; expected one of "
            "CoordFlip, CoordCartesian, CoordFixed, CoordPolar, CoordGeo"
        )
    return new
```

In the `to_spec()` method, the existing `kw["coord"] = resolved._coord` line already handles both strings and dicts because the Rust side now accepts `PyAny`.

- [ ] **Step 5: Remove polar channel gate and add theta/radius remapping**

In `src/ferrum/chart.py`:

1. **Remove** the block at lines ~4388-4398 that raises `NotImplementedError` for theta/radius channels.

2. **Add theta/radius-to-x/y remapping** in `to_spec()`, before the channel serialization loop. When `_coord` is a CoordPolar dict, remap polar channels to x/y so Rust never sees theta/radius:

```python
# In to_spec(), after enc = dict(resolved._encoding):
if isinstance(resolved._coord, dict) and resolved._coord.get("kind") == "polar":
    polar_theta_axis = resolved._coord.get("theta", "x")
    polar_radius_axis = "y" if polar_theta_axis == "x" else "x"
    # Remap theta → x (or y) and radius → the other axis
    if "theta" in enc:
        enc[polar_theta_axis] = enc.pop("theta")
    if "radius" in enc:
        enc[polar_radius_axis] = enc.pop("radius")
```

This ensures that `Theta("count")` becomes the x encoding (or y, per `CoordPolar.theta`) and `Radius("distance")` becomes the y encoding (or x). Rust's scale resolver and mark builders see standard x/y channels; the coord type tells scene_build.rs to reinterpret them as angular/radial.

- [ ] **Step 6: Export new coord classes**

In `src/ferrum/__init__.py`, update the coord imports:

```python
from ferrum.coord import (
    CoordFlip,
    CoordCartesian,
    CoordFixed,
    CoordPolar,
    CoordGeo,
)
```

Add `"CoordCartesian"`, `"CoordFixed"`, `"CoordPolar"`, `"CoordGeo"` to `__all__`.

- [ ] **Step 7: Clean up deferred marks**

In `src/ferrum/marks/deferred.py`, remove `"arc"`, `"geoshape"`, and `"label"` from `PHASE_9_PLUS_MARKS` (keep `"image"` — it is already a real mark in Rust, just has a deferred Python entry that we also remove in task 11d3). Update the module docstring.

### Verify

```bash
# Coord round-trip via JSON
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run python -c "
from ferrum._core import ChartSpec
# Old string path still works
s = ChartSpec(mark='point', x='a', y='b', coord='flip')
print('flip OK:', 'flip' in s.to_json())
# New dict path
s2 = ChartSpec(mark='point', x='a', y='b', coord={'kind': 'cartesian', 'xlim': [0, 100]})
print('cartesian_xlim OK:', 'xlim' in s2.to_json())
s3 = ChartSpec(mark='point', x='a', y='b', coord={'kind': 'polar', 'theta': 'x'})
print('polar OK:', 'polar' in s3.to_json())
"
```

```bash
# Python coord classes serialize correctly
uv run python -c "
import ferrum as fm
c = fm.CoordCartesian(xlim=(0, 100), ylim=(-5, 5), expand=False)
print(c._to_spec_dict())
p = fm.CoordPolar(theta='y', direction=-1)
print(p._to_spec_dict())
g = fm.CoordGeo(projection='equal_earth')
print(g._to_spec_dict())
"
```

```bash
# Existing golden SVGs still pass (no regression)
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
uv run pytest tests/ -x --timeout=60
```

---

## Task 11d1: CoordCartesian (xlim/ylim domain override, expand/clip)

**Files:**
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs`
- Modify: `crates/ferrum-core/src/render/scene_build.rs`
- Create: `tests/test_phase_11d/test_coord_cartesian.py`
- Create: `tests/goldens/phase_11d/coord_cartesian_xlim.svg`

### Steps

- [ ] **Step 1: Read coord domain overrides in scale_resolve.rs**

In `resolve_scales_with_outputs`, after auto-computing the x/y scale domains, check `spec.coord` for `CartesianEx` or `Fixed` variants and override the domain:

```rust
// After build_axis_scale("x", ...) and build_axis_scale("y", ...):
let (x_domain_override, y_domain_override, expand) = match &spec.coord {
    Some(crate::spec::coord::CoordKind::Cartesian { xlim, ylim, expand, .. }) => {
        (*xlim, *ylim, *expand)
    }
    Some(crate::spec::coord::CoordKind::Fixed { xlim, ylim, expand, .. }) => {
        (*xlim, *ylim, *expand)
    }
    _ => (None, None, true),
};

if let Some((lo, hi)) = x_domain_override {
    // Override the x scale's data domain with the user-specified limits
    x = override_scale_domain(x, lo, hi, x_pixel_range)?;
}
if let Some((lo, hi)) = y_domain_override {
    y = override_scale_domain(y, lo, hi, y_pixel_range)?;
}
if !expand {
    // Strip the default padding that fit_linear / fit_time add.
    // For Linear/Log/Symlog: set the scale domain to exactly (lo, hi).
    x = strip_scale_padding(x);
    y = strip_scale_padding(y);
}
```

Implement `override_scale_domain` as a helper that replaces the data-driven domain min/max with user values while preserving the scale type (Linear, Time, etc.). Implement `strip_scale_padding` to reconstruct the scale without the expand padding.

- [ ] **Step 2: Propagate clip flag to Panel in scene_build.rs**

Replace the hardcoded `CoordKind::Cartesian { ... }` in `scene_build.rs` with a conversion from `spec.coord`:

```rust
fn spec_coord_to_scene(
    spec_coord: Option<&crate::spec::coord::CoordKind>,
) -> ferrum_scene::CoordKind {
    match spec_coord {
        None | Some(crate::spec::coord::CoordKind::Flip) => {
            ferrum_scene::CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            }
        }
        Some(crate::spec::coord::CoordKind::Cartesian { xlim, ylim, expand, clip }) => {
            ferrum_scene::CoordKind::Cartesian {
                x_domain: *xlim,
                y_domain: *ylim,
                expand: *expand,
                clip: *clip,
            }
        }
        Some(crate::spec::coord::CoordKind::Fixed { ratio, xlim, ylim, expand, clip }) => {
            ferrum_scene::CoordKind::Fixed {
                ratio: *ratio,
                x_domain: *xlim,
                y_domain: *ylim,
                expand: *expand,
                clip: *clip,
            }
        }
        Some(crate::spec::coord::CoordKind::Polar { theta, start, direction }) => {
            ferrum_scene::CoordKind::Polar {
                theta: if theta == "y" {
                    ferrum_scene::PolarThetaChannel::Y
                } else {
                    ferrum_scene::PolarThetaChannel::X
                },
                start_angle: *start as f64,
                direction: if *direction < 0 {
                    ferrum_scene::PolarDirection::CounterClockwise
                } else {
                    ferrum_scene::PolarDirection::Clockwise
                },
                inner_radius: 0.0,
                outer_radius: 0.0, // computed from panel dims
            }
        }
        Some(crate::spec::coord::CoordKind::Geo { projection }) => {
            ferrum_scene::CoordKind::Geo {
                projection: match projection.as_str() {
                    "albers_usa" => ferrum_scene::GeoProjection::AlbersUsa,
                    "equal_earth" => ferrum_scene::GeoProjection::EqualEarth,
                    "natural_earth" => ferrum_scene::GeoProjection::NaturalEarth,
                    "orthographic" => ferrum_scene::GeoProjection::Orthographic,
                    "equirectangular" => ferrum_scene::GeoProjection::Equirectangular,
                    _ => ferrum_scene::GeoProjection::Mercator,
                },
            }
        }
    }
}
```

Use this function in `build_scene()` to set `Panel.coord`.

When `clip=false`, expand the `Panel.clip` rect to include axis margins so marks can overflow:

```rust
let clip = if matches!(spec.coord, Some(crate::spec::coord::CoordKind::Cartesian { clip: false, .. })
                                   | Some(crate::spec::coord::CoordKind::Fixed { clip: false, .. })) {
    // Expand clip to the full viewport so marks can overflow the plot area
    ferrum_scene::Rect { x: 0.0, y: 0.0, w: layout.viewport.w, h: layout.viewport.h }
} else {
    plot_area
};
```

- [ ] **Step 3: Python test + golden SVG**

Write `tests/test_phase_11d/test_coord_cartesian.py`:

```python
import polars as pl
import ferrum as fm

def test_coord_cartesian_xlim_clips_domain(golden):
    """Chart with xlim=(2, 8) should only show data in that range."""
    df = pl.DataFrame({"x": [1, 3, 5, 7, 9], "y": [10, 20, 30, 40, 50]})
    chart = (
        fm.Chart(df, width=300, height=200)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .coord(fm.CoordCartesian(xlim=(2, 8)))
    )
    golden(chart, "coord_cartesian_xlim")

def test_coord_cartesian_expand_false(golden):
    """expand=False removes default padding."""
    df = pl.DataFrame({"x": [0, 10], "y": [0, 100]})
    chart = (
        fm.Chart(df, width=300, height=200)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .coord(fm.CoordCartesian(expand=False))
    )
    golden(chart, "coord_cartesian_no_expand")

def test_coord_cartesian_clip_false_allows_overflow(golden):
    """clip=False allows marks to render outside the plot area."""
    df = pl.DataFrame({"x": [1, 5, 9], "y": [10, 50, 90]})
    chart = (
        fm.Chart(df, width=300, height=200)
        .mark_point(size=200)
        .encode(x="x:Q", y="y:Q")
        .coord(fm.CoordCartesian(xlim=(3, 7), clip=False))
    )
    golden(chart, "coord_cartesian_clip_false")
```

### Verify

```bash
uv run pytest tests/test_phase_11d/test_coord_cartesian.py -x -v
# Rasterize and visually inspect:
uv run python scripts/snapshot-goldens.py coord_cartesian_xlim
uv run python scripts/snapshot-goldens.py coord_cartesian_no_expand
uv run python scripts/snapshot-goldens.py coord_cartesian_clip_false
```

---

## Task 11d2: CoordFixed (aspect ratio constraint in layout)

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs` (or `layout/panel.rs`)
- Modify: `crates/ferrum-core/src/render/scene_build.rs` (already handled in 11d1 Step 2)
- Create: `tests/test_phase_11d/test_coord_fixed.py`
- Create: `tests/goldens/phase_11d/coord_fixed_ratio1.svg`

### Steps

- [ ] **Step 1: Aspect ratio constraint in compute_layout**

In `crates/ferrum-core/src/layout/mod.rs`, after computing the panel dimensions from the facet grid, apply the aspect-ratio constraint when `CoordFixed` is active.

The `compute_layout()` function receives `spec: &ChartSpec`. Read `spec.coord` and, when it is `CoordKind::Fixed { ratio, .. }`, adjust the panel dimensions:

```rust
// After computing panel_w, panel_h from the grid subdivision:
if let Some(crate::spec::coord::CoordKind::Fixed { ratio, .. }) = &spec.coord {
    // ratio = 1.0 means one data unit on X = one data unit on Y.
    // We constrain the panel to satisfy: panel_h = panel_w * ratio
    // (or shrink whichever dimension exceeds the available space).
    let target_h = panel_w * ratio;
    if target_h <= panel_h {
        // Width is the binding dimension; shrink height
        let excess = panel_h - target_h;
        panel_h = target_h;
        // Center vertically within the allocated space
        panel_y += excess / 2.0;
    } else {
        // Height is the binding dimension; shrink width
        let target_w = panel_h / ratio;
        let excess = panel_w - target_w;
        panel_w = target_w;
        // Center horizontally
        panel_x += excess / 2.0;
    }
}
```

The exact insertion point depends on whether panel dimensions are computed in `layout/panel.rs` or inline in `compute_layout`. Follow the existing code structure.

- [ ] **Step 2: Rust unit test for aspect ratio**

Add a Rust test in `layout/mod.rs` or a dedicated test module:

```rust
#[test]
fn compute_layout_coord_fixed_constrains_aspect_ratio() {
    // Create a spec with CoordFixed(ratio=2.0) and a viewport wider than tall.
    // Verify that the resulting panel height = panel width * 2.0.
    let mut spec = test_spec();  // helper that builds minimal ChartSpec
    spec.coord = Some(CoordKind::Fixed { ratio: 2.0, xlim: None, ylim: None, expand: true, clip: true });
    let result = compute_layout(&spec, &theme, viewport, &axes, &[], &[], None, None, &metrics).unwrap();
    let panel = &result.panels[0];
    let expected_h = panel.plot_area.w * 2.0;
    assert!((panel.plot_area.h - expected_h).abs() < 1e-6,
        "panel_h={} should be panel_w * 2.0 = {}", panel.plot_area.h, expected_h);
}
```

- [ ] **Step 3: Python test + golden SVG**

```python
def test_coord_fixed_ratio_1(golden):
    """ratio=1.0 produces a square panel."""
    df = pl.DataFrame({"x": list(range(10)), "y": list(range(10))})
    chart = (
        fm.Chart(df, width=400, height=300)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .coord(fm.CoordFixed(ratio=1.0))
    )
    golden(chart, "coord_fixed_ratio1")
```

### Verify

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -- compute_layout_coord_fixed
uv run pytest tests/test_phase_11d/test_coord_fixed.py -x -v
uv run python scripts/snapshot-goldens.py coord_fixed_ratio1
```

---

## Task 11d3: CoordPolar + mark_arc (polar transform, angular/radial axes, pie/donut)

This is the largest task. Polar coordinates and mark_arc are inseparable: arc marks only make sense in polar coords, and polar coords need arc marks to be visually testable.

**Files:**
- Modify: `crates/ferrum-core/src/spec/mark.rs` — add `Arc` variant
- Create: `crates/ferrum-core/src/render/marks/arc.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs` — register `arc`
- Modify: `crates/ferrum-core/src/render/scene_build.rs` — polar coordinate transform
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs` — polar scale handling
- Modify: `src/ferrum/chart.py` — wire `mark_arc` as a real mark
- Create: `tests/test_phase_11d/test_coord_polar.py`
- Create: `tests/goldens/phase_11d/` — pie, donut, radial goldens

### Steps

- [ ] **Step 1: Add Arc to Mark enum**

In `crates/ferrum-core/src/spec/mark.rs`, add `Arc` to the `Mark` enum and the `for_each_mark!` macro:

```rust
pub enum Mark {
    Point, Line, Bar, Area, Rule, Text, Tick, Rect, Polygon, Image, Ribbon, Segment,
    Arc,       // Phase 11d
}

macro_rules! for_each_mark {
    ($mac:ident) => {
        $mac! {
            Point   => point,
            Line    => line,
            Bar     => bar,
            Area    => area,
            Rule    => rule,
            Text    => text,
            Tick    => tick,
            Rect    => rect,
            Polygon => polygon,
            Image   => image,
            Ribbon  => ribbon,
            Segment => segment,
            Arc     => arc,
        }
    };
}
```

Update the `ParseMarkError` message to include `arc`.

- [ ] **Step 2: Create arc.rs mark builder**

Create `crates/ferrum-core/src/render/marks/arc.rs`:

The arc mark builder produces pie/donut wedge shapes. It reads:
- The theta-axis data (mapped to angular position via `scale_theta.map(value)`)
- The radius-axis data (mapped to radial position)
- Color encoding per wedge
- `inner_radius` from `mark_style` kwargs (0.0 = pie, >0 = donut); this flows through `ctx.mark_style`, NOT through the coord

Each arc wedge is emitted as a `SceneNode::Path` with `PathCmd::MoveTo`, `PathCmd::LineTo`, and `PathCmd::ArcTo` commands forming a closed wedge shape:

```rust
pub fn build(ctx: &DrawCtx) -> MarkBuildResult {
    use ferrum_scene::{MarkBatchKind, PathCmd, SceneNode};

    // Arc marks must be used with CoordPolar.
    // The coord parameters (theta channel, start_angle, direction)
    // come from the spec.coord field.
    let (theta_ch, start_angle, direction, inner_r_frac, outer_r_frac) =
        match &ctx.spec.coord {
            Some(crate::spec::coord::CoordKind::Polar { theta, start, direction }) => {
                // inner_radius comes from mark_style, not coord
                let inner = ctx.mark_style.inner_radius.unwrap_or(0.0);
                (theta.as_str(), *start as f64, *direction as f64, inner, 1.0)
            }
            _ => {
                // Arc mark without polar coords: emit empty result with warning
                return MarkBuildResult::empty(MarkBatchKind::Arc);
            }
        };

    let panel = ctx.panel;
    let cx = panel.plot_area.x + panel.plot_area.w / 2.0;
    let cy = panel.plot_area.y + panel.plot_area.h / 2.0;
    let max_radius = panel.plot_area.w.min(panel.plot_area.h) / 2.0;
    let outer_r = max_radius * outer_r_frac;
    let inner_r = max_radius * inner_r_frac;

    // Read the theta-mapped channel (x or y) values
    // Compute cumulative angles for stacked arc segments
    let theta_scale = if theta_ch == "y" { &ctx.scales.y } else { &ctx.scales.x };
    // ... compute per-row angle extents, then emit Path nodes for each wedge

    let mut nodes = Vec::new();
    // For each row: compute start_angle, end_angle, then emit a wedge Path
    for i in 0..ctx.batch.num_rows() {
        let wedge_start = cumulative_angles[i] * direction + start_angle;
        let wedge_end = cumulative_angles[i + 1] * direction + start_angle;
        let commands = build_wedge_path(cx, cy, inner_r, outer_r, wedge_start, wedge_end);
        nodes.push(SceneNode::Path {
            commands,
            style: resolve_style(ctx, i),
            closed: true,
        });
    }

    MarkBuildResult {
        kind: MarkBatchKind::Arc,
        nodes,
        data_indices: Some((0..ctx.batch.num_rows()).collect()),
        tooltips: build_tooltips(ctx),
        hrefs: build_hrefs(ctx),
        descriptions: None,
    }
}

fn build_wedge_path(cx: f64, cy: f64, inner_r: f64, outer_r: f64,
                     start: f64, end: f64) -> Vec<PathCmd> {
    // Compute start/end points on outer and inner arcs
    let (outer_sx, outer_sy) = (cx + outer_r * start.sin(), cy - outer_r * start.cos());
    let (outer_ex, outer_ey) = (cx + outer_r * end.sin(), cy - outer_r * end.cos());
    let sweep_angle = (end - start).abs();
    let large_arc = sweep_angle > std::f64::consts::PI;

    let mut cmds = vec![PathCmd::MoveTo { x: outer_sx, y: outer_sy }];
    cmds.push(PathCmd::ArcTo {
        rx: outer_r, ry: outer_r, rotation: 0.0,
        large_arc, sweep: (end - start) > 0.0,
        x: outer_ex, y: outer_ey,
    });

    if inner_r > 0.0 {
        // Donut: line to inner arc end, arc back, close
        let (inner_ex, inner_ey) = (cx + inner_r * end.sin(), cy - inner_r * end.cos());
        let (inner_sx, inner_sy) = (cx + inner_r * start.sin(), cy - inner_r * start.cos());
        cmds.push(PathCmd::LineTo { x: inner_ex, y: inner_ey });
        cmds.push(PathCmd::ArcTo {
            rx: inner_r, ry: inner_r, rotation: 0.0,
            large_arc, sweep: (end - start) <= 0.0,
            x: inner_sx, y: inner_sy,
        });
    } else {
        // Pie: line to center, close
        cmds.push(PathCmd::LineTo { x: cx, y: cy });
    }
    cmds.push(PathCmd::Close);
    cmds
}
```

- [ ] **Step 3: Register arc module**

In `crates/ferrum-core/src/render/marks/mod.rs`, add:

```rust
pub(crate) mod arc;
```

The `dispatch_mark_build` macro in `draw.rs` already dispatches via `for_each_mark!`, so adding `Arc => arc` to the macro (Step 1) automatically wires the dispatch.

- [ ] **Step 4: Polar scale handling in scale_resolve.rs**

When the coord is Polar, the x/y pixel ranges need reinterpretation:
- The theta channel maps to angle space [0, 2*pi] rather than pixel width.
- The radius channel maps to [inner_radius, outer_radius] in pixels.

Add a pre-processing branch in `resolve_scales_with_outputs`:

```rust
let (actual_x_pixel_range, actual_y_pixel_range) = match &spec.coord {
    Some(crate::spec::coord::CoordKind::Polar { theta, .. }) => {
        // Theta channel maps to [0, 2*PI]; radius to [0, max_radius]
        // (The actual pixel mapping happens in scene_build/arc, not here;
        //  scales map data→normalized [0,1] and the mark builder does the rest.)
        if theta == "x" {
            ((0.0, 1.0), y_pixel_range)  // x → normalized theta
        } else {
            (x_pixel_range, (0.0, 1.0))  // y → normalized theta
        }
    }
    _ => (x_pixel_range, y_pixel_range),
};
```

This normalizes the theta scale to [0, 1] so the arc mark builder can multiply by 2*PI.

- [ ] **Step 5: Polar axis rendering in scene_build.rs**

When `CoordKind::Polar`, replace the standard Cartesian axes with:
- Angular axis: circle at `outer_radius` with tick marks around the perimeter
- Radial axis: line from center outward with tick marks along it

This is emitted as `Panel.axes` nodes. The standard `marks::axis::build_axis` is not called; instead a dedicated `build_polar_axes` function handles the circular layout.

- [ ] **Step 6: Wire mark_arc in Python**

In `src/ferrum/chart.py`, replace the `mark_arc` method:

```python
def mark_arc(self, **kwargs):
    """Render data as arcs (pie/donut slices).

    Use with ``CoordPolar()`` for pie/donut charts.
    """
    return self._mark("arc", **kwargs)
```

Remove the `deferred_mark_error("arc")` call. Remove `"arc"` from the deferred marks set.

Also remove the `mark_image` deferred error since it is already a working mark:

```python
def mark_image(self, **kwargs):
    """Render data as raster images positioned in the plot area."""
    return self._mark("image", **kwargs)
```

- [ ] **Step 7: Python polar tests + golden SVGs**

```python
def test_pie_chart(golden):
    """Basic pie chart: mark_arc + CoordPolar."""
    df = pl.DataFrame({
        "category": ["A", "B", "C", "D"],
        "value": [30, 20, 35, 15],
    })
    chart = (
        fm.Chart(df, width=300, height=300)
        .mark_arc()
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar())
    )
    golden(chart, "polar_pie")

def test_donut_chart(golden):
    """Donut chart with inner_radius specified via mark_style."""
    df = pl.DataFrame({
        "category": ["A", "B", "C"],
        "value": [40, 35, 25],
    })
    chart = (
        fm.Chart(df, width=300, height=300)
        .mark_arc(inner_radius=0.5)
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar())
    )
    golden(chart, "polar_donut")

def test_polar_point(golden):
    """Points in polar coordinates."""
    import math
    n = 20
    angles = [i * 2 * math.pi / n for i in range(n)]
    df = pl.DataFrame({"angle": angles, "radius": [i % 5 + 1 for i in range(n)]})
    chart = (
        fm.Chart(df, width=300, height=300)
        .mark_point()
        .encode(x="angle:Q", y="radius:Q")
        .coord(fm.CoordPolar(theta="x"))
    )
    golden(chart, "polar_point")
```

### Verify

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
uv run pytest tests/test_phase_11d/test_coord_polar.py -x -v
uv run python scripts/snapshot-goldens.py polar_pie
uv run python scripts/snapshot-goldens.py polar_donut
uv run python scripts/snapshot-goldens.py polar_point
# Visually verify: pie slices sum to full circle, donut has hole, points are radial
```

---

## Task 11d4: mark_label (positioned text with optional leader lines)

**Files:**
- Modify: `crates/ferrum-core/src/spec/mark.rs` — add `Label` variant
- Create: `crates/ferrum-core/src/render/marks/label.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs` — register `label`
- Modify: `src/ferrum/chart.py` — wire `mark_label`
- Create: `tests/test_phase_11d/test_mark_label.py`
- Create: `tests/goldens/phase_11d/mark_label_basic.svg`

### Steps

- [ ] **Step 1: Add Label to Mark enum**

In `crates/ferrum-core/src/spec/mark.rs`, add `Label` variant to `for_each_mark!`:

```rust
Label   => label,
```

- [ ] **Step 2: Create label.rs mark builder**

`mark_label` differs from `mark_text` in that it:
1. Positions labels relative to data points (not at exact data coordinates)
2. Supports optional leader lines connecting the label to its data point
3. Can apply collision avoidance (label dodging) to prevent overlaps

The builder reads:
- `x`/`y` for the data point position
- `text` encoding for the label content
- `mark_style` for label offset, leader line style, font properties

```rust
pub fn build(ctx: &DrawCtx) -> MarkBuildResult {
    use ferrum_scene::{MarkBatchKind, SceneNode, PathCmd};

    let batch = ctx.batch;
    let n = batch.num_rows();
    let mut nodes = Vec::with_capacity(n * 2); // text + optional leader line

    for i in 0..n {
        let (px, py) = resolve_xy(ctx, i);
        let label_text = resolve_text(ctx, i);

        // Offset: push label away from the data point
        let offset_x = ctx.mark_style.dx.unwrap_or(5.0);
        let offset_y = ctx.mark_style.dy.unwrap_or(-5.0);
        let lx = px + offset_x;
        let ly = py + offset_y;

        // Leader line (thin line from data point to label)
        if ctx.mark_style.leader_line.unwrap_or(false) {
            nodes.push(SceneNode::Line {
                x1: px, y1: py, x2: lx, y2: ly,
                style: leader_stroke(ctx),
            });
        }

        // Label text
        nodes.push(SceneNode::Text {
            x: lx, y: ly,
            content: label_text,
            style: resolve_text_style(ctx, i),
        });
    }

    MarkBuildResult {
        kind: MarkBatchKind::Text, // Labels render as text in SVG
        nodes,
        data_indices: Some((0..n).collect()),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}
```

**Note:** `MarkBatchKind` should use `Text` (or a new variant if collision avoidance needs WASM-side awareness). For 11d, basic label positioning without collision avoidance is sufficient; collision avoidance can be addressed in 11e or a follow-up.

**Design question for implementation:** Whether `MarkBatchKind` should have a `Label` variant or reuse `Text`. If the WASM renderer needs to distinguish labels from plain text (for CSS overlay and collision avoidance), add `Label` to the scene-side `MarkBatchKind` enum. Otherwise, reuse `Text`. The implementer should check `ferrum_scene::MarkBatchKind` — it does NOT currently have a `Label` variant (only the `MarkBatchKind` in the plan shows `Arc` added, but not `Label`). If adding, also add it to `ferrum-scene/src/types.rs`.

- [ ] **Step 3: Register label module**

In `crates/ferrum-core/src/render/marks/mod.rs`:

```rust
pub(crate) mod label;
```

- [ ] **Step 4: Wire mark_label in Python**

In `src/ferrum/chart.py`, replace the deferred `mark_label`:

```python
def mark_label(self, **kwargs):
    """Render smart text labels with optional leader lines.

    Parameters
    ----------
    dx : float, optional
        Horizontal offset from data point (default 5).
    dy : float, optional
        Vertical offset from data point (default -5).
    leader_line : bool, optional
        Draw a thin line from the data point to the label (default False).
    """
    return self._mark("label", **kwargs)
```

- [ ] **Step 5: Python test + golden SVG**

```python
def test_mark_label_basic(golden):
    """Labels positioned near data points."""
    df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 40, 30, 50, 20],
        "name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    })
    chart = (
        fm.Chart(df, width=400, height=300)
        .mark_label(dx=8, dy=-8)
        .encode(x="x:Q", y="y:Q", text="name:N")
    )
    golden(chart, "mark_label_basic")

def test_mark_label_with_leader(golden):
    """Labels with leader lines connecting to data points."""
    df = pl.DataFrame({
        "x": [1, 3, 5],
        "y": [20, 50, 30],
        "name": ["A", "B", "C"],
    })
    chart = (
        fm.Chart(df, width=400, height=300)
        .mark_label(dx=15, dy=-15, leader_line=True)
        .encode(x="x:Q", y="y:Q", text="name:N")
    )
    golden(chart, "mark_label_leader")
```

### Verify

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
uv run pytest tests/test_phase_11d/test_mark_label.py -x -v
uv run python scripts/snapshot-goldens.py mark_label_basic
uv run python scripts/snapshot-goldens.py mark_label_leader
```

---

## Task 11d5: CoordGeo + mark_geoshape (projections + GeoJSON data path)

This task has three independently-testable pieces: (A) projection math, (B) GeoJSON data path, (C) mark_geoshape builder.

**Files:**
- Create: `crates/ferrum-core/src/projection.rs`
- Modify: `crates/ferrum-core/src/lib.rs` — declare `projection` module
- Modify: `crates/ferrum-core/Cargo.toml` — add `geojson` dependency
- Modify: `crates/ferrum-core/src/spec/mark.rs` — add `Geoshape` variant
- Create: `crates/ferrum-core/src/render/marks/geoshape.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs` — register `geoshape`
- Modify: `crates/ferrum-core/src/spec/chart.rs` — add `geojson_geometries` field
- Modify: `crates/ferrum-core/src/render/scene_build.rs` — geo projection code path
- Modify: `src/ferrum/_coerce.py` — GeoJSON detection
- Modify: `src/ferrum/chart.py` — wire `mark_geoshape`
- Create: `tests/test_phase_11d/test_coord_geo.py`

### Steps

- [ ] **Step 1: Create projection.rs with all 6 projections**

Create `crates/ferrum-core/src/projection.rs`. This module implements `forward`/`inverse` as **free functions** that take a `&GeoProjection` parameter and dispatch via match. We cannot add inherent `impl` blocks to `GeoProjection` here because it is defined in the `ferrum-scene` crate (Rust's orphan rule). All math is `f64` standard library functions (sin, cos, atan2, ln).

```rust
//! Pure-Rust map projection forward/inverse implementations.
//! Each projection maps (longitude, latitude) in degrees to (x, y) in
//! normalized coordinates, and back.
//!
//! Free functions, not methods on GeoProjection, because GeoProjection
//! is defined in ferrum-scene and the orphan rule forbids inherent impls
//! on foreign types.

use ferrum_scene::GeoProjection;

/// Project (lon, lat) in degrees to (x, y) in normalized coords.
/// Returns None for out-of-domain inputs (e.g. back hemisphere for Orthographic).
pub fn forward(proj: &GeoProjection, lon_deg: f64, lat_deg: f64) -> Option<(f64, f64)> {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    match proj {
        GeoProjection::Mercator => mercator_forward(lon, lat),
        GeoProjection::Equirectangular => equirectangular_forward(lon, lat),
        GeoProjection::EqualEarth => equal_earth_forward(lon, lat),
        GeoProjection::NaturalEarth => natural_earth_forward(lon, lat),
        GeoProjection::Orthographic => orthographic_forward(lon, lat),
        GeoProjection::AlbersUsa => albers_usa_forward(lon, lat),
    }
}

/// Inverse projection: (x, y) in normalized coords to (lon, lat) in degrees.
pub fn inverse(proj: &GeoProjection, x: f64, y: f64) -> Option<(f64, f64)> {
    let result = match proj {
        GeoProjection::Mercator => mercator_inverse(x, y),
        GeoProjection::Equirectangular => equirectangular_inverse(x, y),
        GeoProjection::EqualEarth => equal_earth_inverse(x, y),
        GeoProjection::NaturalEarth => natural_earth_inverse(x, y),
        GeoProjection::Orthographic => orthographic_inverse(x, y),
        GeoProjection::AlbersUsa => albers_usa_inverse(x, y),
    };
    result.map(|(lon, lat)| (lon.to_degrees(), lat.to_degrees()))
}
```

Implement each projection:

**Mercator (~15 LOC):**
```rust
fn mercator_forward(lon: f64, lat: f64) -> Option<(f64, f64)> {
    // Clamp latitude to avoid infinity near poles
    let lat = lat.clamp(-1.4844, 1.4844); // ~85 degrees
    let x = lon;
    let y = (std::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln();
    Some((x, y))
}

fn mercator_inverse(x: f64, y: f64) -> Option<(f64, f64)> {
    let lon = x;
    let lat = 2.0 * y.exp().atan() - std::f64::consts::FRAC_PI_2;
    Some((lon, lat))
}
```

**Equirectangular (~10 LOC):**
```rust
fn equirectangular_forward(lon: f64, lat: f64) -> Option<(f64, f64)> {
    Some((lon, lat))
}

fn equirectangular_inverse(x: f64, y: f64) -> Option<(f64, f64)> {
    Some((x, y))
}
```

**EqualEarth (~40 LOC):** Polynomial parametric projection with Newton-Raphson iteration for inverse. Reference: Savric et al., "The Equal Earth map projection" (2018).

```rust
fn equal_earth_forward(lon: f64, lat: f64) -> Option<(f64, f64)> {
    const A1: f64 = 1.340264;
    const A2: f64 = -0.081106;
    const A3: f64 = 0.000893;
    const A4: f64 = 0.003796;
    const M: f64 = std::f64::consts::SQRT_2;

    let theta = (lat * M / 2.0).sin().asin();
    let theta2 = theta * theta;
    let theta6 = theta2 * theta2 * theta2;
    let x = lon * (A1 + 3.0 * A2 * theta2 + theta6 * (7.0 * A3 + 9.0 * A4 * theta2)).cos()
            / (A1 + A2 * theta2 * 3.0 + theta6 * (A3 * 7.0 + A4 * 9.0 * theta2)); // simplified
    // ... (full implementation follows Savric et al.)
    Some((x, y))
}
```

**NaturalEarth (~40 LOC):** Polynomial approximation. Reference: Savric et al., "A polynomial equation for the Natural Earth projection" (2011).

**Orthographic (~20 LOC):** Great-circle clipping for back hemisphere.
```rust
fn orthographic_forward(lon: f64, lat: f64) -> Option<(f64, f64)> {
    // Clip points on the back hemisphere
    let cos_c = lat.cos() * lon.cos();
    if cos_c < 0.0 { return None; }
    let x = lat.cos() * lon.sin();
    let y = lat.sin();
    Some((x, y))
}
```

**AlbersUsa (~80 LOC):** Albers conic projection centered on the lower 48, with separate Alaska and Hawaii insets. The implementation uses three Albers conic projections with different parameters and checks which inset region the input point falls in.

- [ ] **Step 2: Projection round-trip tests (Rust)**

Add comprehensive Rust unit tests in `projection.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(proj: GeoProjection, lon: f64, lat: f64, tol: f64) {
        let (x, y) = forward(&proj, lon, lat).expect("forward failed");
        let (lon2, lat2) = inverse(&proj, x, y).expect("inverse failed");
        assert!(
            (lon - lon2).abs() < tol && (lat - lat2).abs() < tol,
            "{proj:?}: ({lon}, {lat}) -> ({x}, {y}) -> ({lon2}, {lat2}), delta=({}, {})",
            (lon - lon2).abs(), (lat - lat2).abs()
        );
    }

    #[test]
    fn mercator_round_trip() {
        for (lon, lat) in [(-122.4, 37.8), (0.0, 0.0), (139.7, 35.7), (-73.9, 40.7)] {
            round_trip(GeoProjection::Mercator, lon, lat, 1e-10);
        }
    }

    #[test]
    fn equirectangular_round_trip() {
        for (lon, lat) in [(-180.0, -90.0), (0.0, 0.0), (180.0, 90.0)] {
            round_trip(GeoProjection::Equirectangular, lon, lat, 1e-10);
        }
    }

    #[test]
    fn equal_earth_round_trip() {
        for (lon, lat) in [(-122.4, 37.8), (0.0, 0.0), (139.7, 35.7)] {
            round_trip(GeoProjection::EqualEarth, lon, lat, 1e-10);
        }
    }

    #[test]
    fn natural_earth_round_trip() {
        for (lon, lat) in [(-122.4, 37.8), (0.0, 0.0), (139.7, 35.7)] {
            round_trip(GeoProjection::NaturalEarth, lon, lat, 1e-10);
        }
    }

    #[test]
    fn orthographic_clips_back_hemisphere() {
        // Point behind the globe should return None
        assert!(GeoProjection::Orthographic.forward(180.0, 0.0).is_none());
        // Point on front hemisphere should succeed and round-trip
        round_trip(GeoProjection::Orthographic, 30.0, 45.0, 1e-10);
    }

    #[test]
    fn albers_usa_conus_round_trip() {
        // San Francisco (lower 48)
        round_trip(GeoProjection::AlbersUsa, -122.4, 37.8, 1e-6);
    }

    #[test]
    fn albers_usa_alaska_round_trip() {
        round_trip(GeoProjection::AlbersUsa, -150.0, 64.0, 1e-4);
    }

    #[test]
    fn albers_usa_hawaii_round_trip() {
        round_trip(GeoProjection::AlbersUsa, -155.5, 19.9, 1e-4);
    }
}
```

**Tolerance notes:** Mercator/Equirectangular/EqualEarth/NaturalEarth/Orthographic should achieve `1e-10`. AlbersUsa may have slightly lower precision due to the composite projection (three conic projections + inset transforms); `1e-4` is acceptable for the Alaska/Hawaii insets, `1e-6` for the lower 48.

- [ ] **Step 3: Declare projection module**

In `crates/ferrum-core/src/lib.rs`, add:

```rust
pub mod projection;
```

`GeoProjection` is defined in `ferrum-scene`. `projection.rs` uses free functions (`forward(&proj, ...)`, `inverse(&proj, ...)`) rather than inherent `impl GeoProjection { ... }` because Rust's orphan rule forbids adding inherent methods to types defined in other crates. `ferrum-wasm` can also call these functions since `ferrum-core` is a dependency.

- [ ] **Step 4: Add geojson dependency**

In `crates/ferrum-core/Cargo.toml`, add:

```toml
geojson = "0.24"
```

This is the only new external dependency in 11d. The `geojson` crate is pure Rust, serde-based, and has no transitive native dependencies.

- [ ] **Step 5: Add Geoshape to Mark enum**

In `crates/ferrum-core/src/spec/mark.rs`, add to `for_each_mark!`:

```rust
Geoshape => geoshape,
```

- [ ] **Step 6: Add geojson_geometries field to ChartSpec**

In `crates/ferrum-core/src/spec/chart.rs`:

```rust
pub struct ChartSpec {
    // ... existing fields ...
    /// GeoJSON geometry array as a JSON string, for mark_geoshape.
    /// Populated by Python's _coerce.py when the input data is a GeoJSON
    /// FeatureCollection. Rust deserializes this via the `geojson` crate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geojson_geometries: Option<String>,
}
```

Add `geojson_geometries` as an optional parameter in the `#[new]` method:

```rust
geojson_geometries: Option<String>,
```

Wire it through to the struct construction. In the `#[pyo3(signature)]`, add `geojson_geometries = None`.

- [ ] **Step 7: GeoJSON detection in _coerce.py**

In `src/ferrum/_coerce.py`, add a detection branch for GeoJSON FeatureCollections:

```python
def _detect_geojson(data) -> tuple[pl.DataFrame, str] | None:
    """If data is a GeoJSON FeatureCollection (dict or JSON string),
    return (properties_df, geometry_json_str). Otherwise return None."""
    import json

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(data, dict):
        return None
    if data.get("type") != "FeatureCollection":
        return None

    features = data.get("features", [])
    if not features:
        return None

    # Extract properties into a DataFrame (one row per feature)
    properties = [f.get("properties", {}) for f in features]
    properties_df = pl.DataFrame(properties)

    # Extract geometry coordinates as a JSON array
    geometries = [f.get("geometry") for f in features]
    geometry_json = json.dumps(geometries)

    return properties_df, geometry_json
```

The `Chart` class (or `to_spec()`) should call this detection, set the DataFrame as the data source for encoding resolution, and pass `geometry_json` as `geojson_geometries` to `ChartSpec`.

- [ ] **Step 8: Create geoshape.rs mark builder**

Create `crates/ferrum-core/src/render/marks/geoshape.rs`:

```rust
//! mark_geoshape: GeoJSON geometry → projected SceneNode::Polygon.
//! Reads `spec.geojson_geometries` (JSON string), projects each polygon's
//! vertices via the active GeoProjection, emits Polygon nodes.

use geojson::{Geometry, Value as GeoValue};
use ferrum_scene::{MarkBatchKind, SceneNode, GeoProjection};

use crate::projection;
use crate::render::draw::{DrawCtx, MarkBuildResult};

pub fn build(ctx: &DrawCtx) -> MarkBuildResult {
    let empty = || MarkBuildResult::empty(MarkBatchKind::Polygon);

    let geojson_str = match &ctx.spec.geojson_geometries {
        Some(s) => s,
        None => return empty(),
    };

    let projection = match &ctx.spec.coord {
        Some(crate::spec::coord::CoordKind::Geo { projection }) => {
            // Convert string to GeoProjection enum
            match projection.as_str() {
                "mercator" => GeoProjection::Mercator,
                "albers_usa" => GeoProjection::AlbersUsa,
                "equal_earth" => GeoProjection::EqualEarth,
                "natural_earth" => GeoProjection::NaturalEarth,
                "orthographic" => GeoProjection::Orthographic,
                "equirectangular" => GeoProjection::Equirectangular,
                _ => GeoProjection::Mercator,
            }
        }
        _ => GeoProjection::Mercator, // default if no coord specified
    };

    let geometries: Vec<Option<Geometry>> = match serde_json::from_str(geojson_str) {
        Ok(g) => g,
        Err(_) => return empty(),
    };

    let panel = ctx.panel;
    let mut nodes = Vec::new();
    let mut data_indices = Vec::new();

    for (i, geom) in geometries.iter().enumerate() {
        let Some(geom) = geom else { continue; };
        let polygons = extract_polygons(geom);
        for polygon_coords in polygons {
            let mut projected: Vec<[f64; 2]> = Vec::new();
            for [lon, lat] in &polygon_coords {
                if let Some((px, py)) = projection::forward(&projection, *lon, *lat) {
                    // Scale projected coords to panel pixel space
                    let x = panel.plot_area.x + (px - proj_x_min) / (proj_x_max - proj_x_min) * panel.plot_area.w;
                    let y = panel.plot_area.y + (1.0 - (py - proj_y_min) / (proj_y_max - proj_y_min)) * panel.plot_area.h;
                    projected.push([x, y]);
                }
            }
            if projected.len() >= 3 {
                nodes.push(SceneNode::Polygon {
                    points: projected,
                    style: resolve_fill_stroke(ctx, i),
                });
                data_indices.push(i);
            }
        }
    }

    MarkBuildResult {
        kind: MarkBatchKind::Polygon,
        nodes,
        data_indices: Some(data_indices),
        tooltips: build_tooltips(ctx),
        hrefs: None,
        descriptions: None,
    }
}

fn extract_polygons(geom: &Geometry) -> Vec<Vec<[f64; 2]>> {
    match &geom.value {
        GeoValue::Polygon(rings) => {
            // First ring is exterior; skip interior rings for now
            vec![rings[0].iter().map(|c| [c[0], c[1]]).collect()]
        }
        GeoValue::MultiPolygon(polys) => {
            polys.iter().flat_map(|rings| {
                std::iter::once(rings[0].iter().map(|c| [c[0], c[1]]).collect())
            }).collect()
        }
        _ => vec![],
    }
}
```

**Note:** The `proj_x_min/max`, `proj_y_min/max` bounds need a pre-pass over all geometries to compute the projected extent, which is then used to scale to pixel space. This is analogous to how scales work: data extent → pixel range. The implementer should compute the projection extent in a first pass, then project-and-scale in a second pass.

- [ ] **Step 9: Register geoshape module**

In `crates/ferrum-core/src/render/marks/mod.rs`:

```rust
pub(crate) mod geoshape;
```

- [ ] **Step 10: Wire mark_geoshape in Python**

In `src/ferrum/chart.py`:

```python
def mark_geoshape(self, **kwargs):
    """Render geographic shape data from GeoJSON.

    Pass a GeoJSON FeatureCollection as the data argument
    to ``Chart(data)``.  Feature properties become encoding fields.
    """
    return self._mark("geoshape", **kwargs)
```

- [ ] **Step 11: Geo code path in scene_build.rs**

When `CoordKind::Geo` is active, the standard x/y axes are suppressed (no Cartesian axes for map projections). The panel should not show axis lines, ticks, or labels. Gridlines (graticule) could be emitted as projected parallels/meridians, but this is optional for 11d — suppress all axes/gridlines for Geo coord.

```rust
// In build_scene(), when building panel axes:
let suppress_axes = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Geo { .. }));
if !suppress_axes {
    // ... existing axis building code ...
}
```

- [ ] **Step 12: Python geo test + golden SVG**

A minimal test with a small GeoJSON FeatureCollection (e.g., 3 simple polygons representing fake regions):

```python
def test_geoshape_mercator(golden):
    """Basic map with mark_geoshape + CoordGeo(mercator)."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Region A", "value": 100},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-120, 35], [-120, 40], [-115, 40], [-115, 35], [-120, 35]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Region B", "value": 200},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-115, 35], [-115, 40], [-110, 40], [-110, 35], [-115, 35]
                    ]]
                }
            },
        ]
    }
    chart = (
        fm.Chart(geojson, width=400, height=300)
        .mark_geoshape()
        .encode(color="value:Q")
        .coord(fm.CoordGeo(projection="mercator"))
    )
    golden(chart, "geo_mercator")

def test_geoshape_equal_earth(golden):
    """Map with equal_earth projection."""
    # Same data, different projection
    chart = (
        fm.Chart(geojson, width=400, height=300)
        .mark_geoshape()
        .encode(color="value:Q")
        .coord(fm.CoordGeo(projection="equal_earth"))
    )
    golden(chart, "geo_equal_earth")
```

### Verify

```bash
# Projection round-trip tests
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -- projection

# Full test suite
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
uv run pytest tests/test_phase_11d/test_coord_geo.py -x -v

# Visual verification
uv run python scripts/snapshot-goldens.py geo_mercator
uv run python scripts/snapshot-goldens.py geo_equal_earth
```

---

## Task 11d6: mark_image coord-awareness

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/image.rs` (minimal)
- Modify: `src/ferrum/chart.py` (remove deferred error for mark_image)

### Steps

- [ ] **Step 1: Validate coord compatibility**

`mark_image` (Raster transform output) is inherently Cartesian: it places a rectangular bitmap at pixel coordinates derived from `x_min/x_max/y_min/y_max`. It does not make sense in polar or geo coordinate systems.

In `crates/ferrum-core/src/render/marks/image.rs`, add a validation check at the top of `build()`:

```rust
// Image marks require Cartesian coordinates
if matches!(ctx.spec.coord,
    Some(crate::spec::coord::CoordKind::Polar { .. }) |
    Some(crate::spec::coord::CoordKind::Geo { .. })
) {
    // Log warning and return empty — images can't be projected
    return empty();
}
```

- [ ] **Step 2: Remove deferred error for mark_image in Python**

`mark_image` is already a fully functional mark in Rust (`image.rs` handles Raster transform output). But `chart.py` has a deferred error for it. The deferred entry exists because the Python method raises `NotImplementedError` even though the Rust mark module works.

Check whether `mark_image` needs Python-side changes beyond removing the deferred error. The current `mark_image` Python method raises `deferred_mark_error("image")` — replace with:

```python
def mark_image(self, **kwargs):
    """Render data as raster images positioned in the plot area.

    Used with ``mark_raster`` transforms that produce heatmaps,
    decision boundaries, and density rasters.
    """
    return self._mark("image", **kwargs)
```

Remove `"image"` from `PHASE_9_PLUS_MARKS` in `deferred.py`.

### Verify

```bash
# Existing raster/image goldens still pass
uv run pytest tests/ -k "raster or image" -x -v
```

---

## Validation checklist

### Backward compatibility

- [ ] All existing golden SVGs pass byte-identically (the `CoordKind` extension in spec is backward-compatible: `Cartesian` and `Flip` still round-trip)
- [ ] `ChartSpec(mark='point', x='a', y='b', coord='flip')` still works (string path preserved)
- [ ] `Chart.coord(CoordFlip())` still works (unchanged)
- [ ] `cargo test` passes all existing tests
- [ ] `uv run pytest` passes all existing tests

### New golden SVGs (must be rasterized and visually inspected)

| Golden | What to verify |
|---|---|
| `coord_cartesian_xlim.svg` | Only data points within [2,8] visible on x axis; axis labels show 2-8 range |
| `coord_cartesian_no_expand.svg` | Axis starts exactly at data min (0), ends at data max (10), no padding |
| `coord_cartesian_clip_false.svg` | Points at x=1 and x=9 are visible outside the plot area bounds |
| `coord_fixed_ratio1.svg` | Panel is square (w == h in pixels) |
| `polar_pie.svg` | Four colored wedges summing to a full circle |
| `polar_donut.svg` | Three colored wedges with a hole in the center |
| `polar_point.svg` | Points arranged in a circular pattern |
| `mark_label_basic.svg` | Text labels positioned near (offset from) data points |
| `mark_label_leader.svg` | Text labels with thin lines connecting to data points |
| `geo_mercator.svg` | Two rectangles projected as slightly curved quadrilaterals |
| `geo_equal_earth.svg` | Same regions projected with EqualEarth distortion |

### Projection accuracy (Rust-side tests)

| Projection | Round-trip tolerance |
|---|---|
| Mercator | 1e-10 |
| Equirectangular | 1e-10 |
| EqualEarth | 1e-10 |
| NaturalEarth | 1e-10 |
| Orthographic | 1e-10 (front hemisphere only) |
| AlbersUsa (lower 48) | 1e-6 |
| AlbersUsa (Alaska) | 1e-4 |
| AlbersUsa (Hawaii) | 1e-4 |

### Zero NotImplementedError gate

- [ ] `fm.CoordCartesian()` constructs without error
- [ ] `fm.CoordFixed()` constructs without error
- [ ] `fm.CoordPolar()` constructs without error
- [ ] `fm.CoordGeo()` constructs without error
- [ ] `fm.Chart(df).mark_arc()` does not raise NotImplementedError
- [ ] `fm.Chart(df).mark_label()` does not raise NotImplementedError
- [ ] `fm.Chart(df).mark_geoshape()` does not raise NotImplementedError
- [ ] `fm.Chart(df).mark_image()` does not raise NotImplementedError

### Command-line final verification

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
uv run pytest tests/ -x --timeout=120
```

---

## Task dependency graph

```
11d0 (coord plumbing) ──┬── 11d1 (CoordCartesian)
                        ├── 11d2 (CoordFixed)
                        ├── 11d3 (CoordPolar + mark_arc)
                        ├── 11d4 (mark_label) ← independent of coord, but needs Mark enum changes from 11d3
                        └── 11d5 (CoordGeo + mark_geoshape)

11d6 (mark_image coord-awareness) ← independent, can run any time after 11d0
```

11d0 is the critical path. 11d1-11d5 can proceed in parallel after 11d0 is done, with the caveat that 11d3 and 11d5 both modify the `Mark` enum, so they should be sequenced or carefully merged. Recommended execution order: **11d0 → 11d1 → 11d2 → 11d3 → 11d4 → 11d5 → 11d6**.
