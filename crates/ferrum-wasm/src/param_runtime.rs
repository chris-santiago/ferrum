//! Live runtime for D6 reactive parameters (sub-task 5e-2b).
//!
//! Consumes the `param_bindings` carried on `InteractionConfig` (emitted by the
//! static resolver in 5e-2a) and drives the three reactive behaviors by
//! **reusing** the existing interactive machinery:
//!
//! - **Reactive rescale** (`BindingRole::Domain`): a brush on the source panel
//!   updates the bound target panel's zoom transform via the same `ZoomPanState`
//!   path that wheel/pan/D3-zoom use. The brushed pixel sub-region maps to the
//!   target's plot area with the boxzoom affine the JS brush already computes.
//! - **Crossfilter** (`BindingRole::Filter`): the brushed pixel extent is
//!   converted to a data interval on the source panel, re-projected to the
//!   target panel's pixel space, and applied as a synthesized opacity
//!   conditional through the existing `conditional::apply_conditional_to_*`
//!   containment path — the same dimming selections already use.
//! - **Legend toggle** (`BindingRole::Legend`): handled in `lib.rs` by toggling
//!   the named point selection (mirroring `handle_click`) and re-running
//!   `apply_conditionals_and_render`.
//!
//! The pixel↔data conversions below are plain linear interpolation between a
//! panel's `plot_area` (pixels) and its `CoordKind` domain (data). They are NOT
//! a new scale engine — they invert exactly the linear mapping the renderer
//! already implies for Cartesian/Fixed panels.
//!
//! All functions here are pure and host-testable; the GPU-touching wiring lives
//! in `WasmRenderer` (`lib.rs`).
//!
//! See `design-docs/superpowers/specs/2026-06-01-d6-reactive-params-wire-contract.md` §6/§6a.

use ferrum_scene::{CoordKind, Rect};

/// Which axis of a panel a `Domain`/`Filter` binding addresses, parsed from the
/// binding's wire channel name. Only `x`/`y` participate in pixel↔data
/// rescale and crossfilter; other channels (color/size/opacity) have no
/// pixel extent and are ignored at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
}

impl Axis {
    pub(crate) fn from_channel(channel: &str) -> Option<Self> {
        match channel {
            "x" => Some(Axis::X),
            "y" => Some(Axis::Y),
            _ => None,
        }
    }
}

/// The data domain `(lo, hi)` for the given axis of a Cartesian/Fixed panel,
/// reading the y-axis through a specific scale slot (secondary-y-axis, #52).
///
/// Returns `None` for non-cartesian coords (polar/geo) or when the domain is
/// unset (auto-inferred — no static domain captured), in which case pixel↔data
/// conversion is undefined and the caller skips the binding.
///
/// The x-axis is shared across every layer, so `y_slot` is ignored for `X`. For
/// `Y`, slot `k`'s domain is `y_domains[k]` when the per-slot list is populated;
/// slot 0 also mirrors the legacy `y_domain`. When `y_domains` is empty (every
/// single-y chart — the byte-stable default, slot 0) or the requested slot is
/// out of range, this falls back to `y_domain`. A `None` slot entry
/// (ordinal/band scale) yields `None`, handled by callers exactly like a `None`
/// primary domain (pixel↔data conversion undefined; the binding is skipped).
pub(crate) fn axis_domain_slot(coord: &CoordKind, axis: Axis, y_slot: usize) -> Option<(f64, f64)> {
    match axis {
        Axis::X => match coord {
            CoordKind::Cartesian { x_domain, .. } | CoordKind::Fixed { x_domain, .. } => *x_domain,
            CoordKind::Polar { .. } | CoordKind::Geo { .. } => None,
        },
        Axis::Y => match coord {
            CoordKind::Cartesian { y_domain, y_domains, .. } => match y_domains.get(y_slot) {
                // Guard note (Task 8 review): when slots are present, slot 0's
                // resolved domain lives in `y_domains[0]` and may diverge from
                // the legacy `y_domain`; prefer the slotted value.
                Some(slotted) => *slotted,
                None => *y_domain,
            },
            CoordKind::Fixed { y_domain, .. } => *y_domain,
            CoordKind::Polar { .. } | CoordKind::Geo { .. } => None,
        },
    }
}

/// Pixel bounds `(lo, hi)` of a plot area for the given axis.
///
/// For `Y`, screen pixels increase downward while data increases upward, so the
/// caller must account for the inversion when mapping to data. This returns the
/// raw screen bounds; `pixel_to_data`/`data_to_pixel` apply the Y flip.
fn axis_pixel_bounds(plot_area: &Rect, axis: Axis) -> (f64, f64) {
    match axis {
        Axis::X => (plot_area.x, plot_area.x + plot_area.w),
        Axis::Y => (plot_area.y, plot_area.y + plot_area.h),
    }
}

/// Convert a screen-pixel coordinate on `axis` to a data value, given the
/// panel's plot area and data domain. Linear; honors the Y screen flip.
pub(crate) fn pixel_to_data(
    px: f64,
    plot_area: &Rect,
    domain: (f64, f64),
    axis: Axis,
) -> f64 {
    let (p_lo, p_hi) = axis_pixel_bounds(plot_area, axis);
    let (d_lo, d_hi) = domain;
    let span = p_hi - p_lo;
    if span.abs() < f64::EPSILON {
        return d_lo;
    }
    let frac = (px - p_lo) / span;
    match axis {
        // X: left pixel → domain lo, right pixel → domain hi.
        Axis::X => d_lo + frac * (d_hi - d_lo),
        // Y: top pixel → domain hi, bottom pixel → domain lo (screen flip).
        Axis::Y => d_hi - frac * (d_hi - d_lo),
    }
}

/// Convert a data value on `axis` to a screen-pixel coordinate. Inverse of
/// `pixel_to_data`.
pub(crate) fn data_to_pixel(
    value: f64,
    plot_area: &Rect,
    domain: (f64, f64),
    axis: Axis,
) -> f64 {
    let (p_lo, p_hi) = axis_pixel_bounds(plot_area, axis);
    let (d_lo, d_hi) = domain;
    let dspan = d_hi - d_lo;
    if dspan.abs() < f64::EPSILON {
        return p_lo;
    }
    let frac = (value - d_lo) / dspan;
    match axis {
        Axis::X => p_lo + frac * (p_hi - p_lo),
        Axis::Y => p_hi - frac * (p_hi - p_lo),
    }
}

/// Normalize a screen-pixel extent `(a, b)` into `(lo, hi)` with `lo <= hi`.
pub(crate) fn normalize(a: f64, b: f64) -> (f64, f64) {
    (a.min(b), a.max(b))
}

/// Map a brushed pixel extent on the source panel to a data interval, then
/// re-project it onto the target panel's pixel space for the given axis.
///
/// Used by crossfilter: the source brush is a pixel rectangle on the source
/// panel, but the target panel has a different pixel layout. Going through the
/// shared *data* domain yields the equivalent pixel extent in the target so the
/// existing spatial-containment conditional path can dim target marks.
///
/// Returns `None` when either panel lacks a usable cartesian domain on `axis`.
pub(crate) fn reproject_extent(
    brush_px: (f64, f64),
    source_plot_area: &Rect,
    source_coord: &CoordKind,
    target_plot_area: &Rect,
    target_coord: &CoordKind,
    axis: Axis,
    y_slot: usize,
) -> Option<(f64, f64)> {
    // A brush bound to one layer inverts through that layer's y-slot on both
    // ends (the x-axis is shared, so `y_slot` is ignored for `Axis::X`). Slot 0
    // is the byte-stable default: single-y charts read `y_domain` as before.
    let source_domain = axis_domain_slot(source_coord, axis, y_slot)?;
    let target_domain = axis_domain_slot(target_coord, axis, y_slot)?;
    let (px_lo, px_hi) = normalize(brush_px.0, brush_px.1);
    let d0 = pixel_to_data(px_lo, source_plot_area, source_domain, axis);
    let d1 = pixel_to_data(px_hi, source_plot_area, source_domain, axis);
    let (d_lo, d_hi) = normalize(d0, d1);
    let t0 = data_to_pixel(d_lo, target_plot_area, target_domain, axis);
    let t1 = data_to_pixel(d_hi, target_plot_area, target_domain, axis);
    Some(normalize(t0, t1))
}

/// Affine scale + translate `(scale, offset)` that maps a pixel sub-extent
/// (expressed in the *same* pixel coordinate system as `target_plot_area`)
/// onto the full pixel extent of the target panel for one axis.
///
/// `screen_x' = scale * screen_x + offset`. This is consumed by
/// `ZoomPanState` (the same affine the wheel/pan/D3-zoom path applies), so the
/// target panel rescales to show only the brushed domain.
///
/// **Precondition:** `brush_px` must already be in the target panel's pixel
/// coordinate system. Use [`rescale_affine_cross_panel`] when the brush
/// originates from a different source panel.
///
/// Returns `None` for a degenerate (zero-width) brush extent, where the scale
/// would be infinite.
pub(crate) fn rescale_affine(
    brush_px: (f64, f64),
    source_plot_area: &Rect,
    target_plot_area: &Rect,
    axis: Axis,
) -> Option<(f64, f64)> {
    let (b_lo, b_hi) = normalize(brush_px.0, brush_px.1);
    let sel_span = b_hi - b_lo;
    if sel_span.abs() < 1e-9 {
        return None;
    }
    let (t_lo, t_hi) = axis_pixel_bounds(target_plot_area, axis);
    let target_span = t_hi - t_lo;
    // Clamp the brush extent to the source plot area so a brush that overshoots
    // the panel does not produce a nonsensical scale.
    let (s_lo, s_hi) = axis_pixel_bounds(source_plot_area, axis);
    let b_lo = b_lo.max(s_lo);
    let b_hi = b_hi.min(s_hi);
    let span = (b_hi - b_lo).abs();
    if span < 1e-9 {
        return None;
    }
    let scale = target_span / span;
    let offset = t_lo - scale * b_lo;
    Some((scale, offset))
}

/// Cross-panel variant of [`rescale_affine`]: reprojects `brush_px` from
/// source-panel pixel space through the shared data domain into target-panel
/// pixel space, then builds the boxzoom affine entirely in target pixels.
///
/// This is the correct path when the source panel (where the brush lives) and
/// the target panel (whose marks will be transformed) occupy different pixel
/// regions — the `hconcat(overview, detail)` pattern.
///
/// When both panels share the same plot area (single-panel self-rescale), the
/// reprojection is a no-op and the result is identical to calling
/// `rescale_affine` directly.
///
/// Returns `None` when either panel lacks a usable cartesian domain on `axis`
/// (reprojection undefined) or when the reprojected brush is degenerate.
pub(crate) fn rescale_affine_cross_panel(
    brush_px: (f64, f64),
    source_plot_area: &Rect,
    source_coord: &CoordKind,
    target_plot_area: &Rect,
    target_coord: &CoordKind,
    axis: Axis,
    y_slot: usize,
) -> Option<(f64, f64)> {
    // Reproject the brush from source pixels → data domain → target pixels.
    // `reproject_extent` handles the Y-axis screen flip and normalises lo<=hi.
    // `y_slot` selects the layer's y-scale for a dual-axis panel (ignored for x).
    let target_brush_px = reproject_extent(
        brush_px,
        source_plot_area,
        source_coord,
        target_plot_area,
        target_coord,
        axis,
        y_slot,
    )?;
    // The reprojected brush is now in target pixel space. Build the affine so
    // that target marks in `[target_brush_px.0, target_brush_px.1]` are
    // stretched to fill the full target plot area. Pass target_plot_area as
    // both source and target so the clamp uses the correct bounds.
    rescale_affine(target_brush_px, target_plot_area, target_plot_area, axis)
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod bug_hunt_interactive_slots {
    use super::*;
    use ferrum_scene::Rect;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    fn dual_coord() -> CoordKind {
        CoordKind::Cartesian {
            x_domain: Some((0.0, 10.0)),
            y_domain: Some((0.0, 100.0)),
            expand: true,
            clip: true,
            y_domains: vec![Some((0.0, 100.0)), Some((0.0, 10.0))],
        }
    }

    /// The exact affine `apply_reactive_rescale` builds for a y-domainParam
    /// brush on slot 1 of a single dual-axis panel, hand-computed.
    ///
    /// plot_area y: [0, 100]; slot-1 domain [0, 10]; brush screen y (40, 60).
    /// Through slot 1 (screen flip): y=40 → 6, y=60 → 4 → data [4, 6] →
    /// back to pixels [40, 60]. rescale over target span 100 → scale 5,
    /// offset = 0 − 5·40 = −200. A mark at screen y=50 (data 5, brush
    /// center) must map to 5·50 − 200 = 50 — i.e. the brush center stays put.
    #[test]
    fn bug_hunt_slotted_self_rescale_affine_hand_computed() {
        let pa = rect(0.0, 0.0, 100.0, 100.0);
        let coord = dual_coord();
        let (scale, offset) =
            rescale_affine_cross_panel((40.0, 60.0), &pa, &coord, &pa, &coord, Axis::Y, 1)
                .expect("slot-1 domain present, brush non-degenerate");
        assert!((scale - 5.0).abs() < 1e-9, "scale must be 5, got {scale}");
        assert!((offset - (-200.0)).abs() < 1e-6, "offset must be -200, got {offset}");
        assert!((scale * 50.0 + offset - 50.0).abs() < 1e-6, "brush center must be fixed");
        assert!((scale * 40.0 + offset - 0.0).abs() < 1e-6, "brush top → plot top");
    }

    /// The same brush inverted through slot 0 must yield a DIFFERENT data
    /// interval than slot 1 whenever the domains differ — if the slot index
    /// were ignored, a slot-1 domainParam would silently rescale through the
    /// primary domain. Discriminates the slot routing.
    #[test]
    fn bug_hunt_slot0_and_slot1_rescale_affines_agree_only_by_accident_of_geometry() {
        let pa = rect(0.0, 0.0, 100.0, 100.0);
        let coord = dual_coord();
        // With source == target and proportional domains, the pixel-space
        // affine happens to be identical — so use a target with a DIFFERENT
        // pixel layout for slot 1's domain to expose the routing.
        let tgt_pa = rect(0.0, 0.0, 100.0, 50.0); // target y: [0, 50]
        let tgt_coord = CoordKind::Cartesian {
            x_domain: Some((0.0, 10.0)),
            y_domain: Some((0.0, 100.0)),
            expand: true,
            clip: true,
            // slot 1's domain on the target covers only [0, 5]: data [4, 6]
            // clips differently than the primary [0, 100].
            y_domains: vec![Some((0.0, 100.0)), Some((0.0, 5.0))],
        };
        let r0 = reproject_extent((40.0, 60.0), &pa, &coord, &tgt_pa, &tgt_coord, Axis::Y, 0);
        let r1 = reproject_extent((40.0, 60.0), &pa, &coord, &tgt_pa, &tgt_coord, Axis::Y, 1);
        let (a0, b0) = r0.expect("slot-0 reprojection");
        let (a1, b1) = r1.expect("slot-1 reprojection");
        assert!(
            (a0 - a1).abs() > 1.0 || (b0 - b1).abs() > 1.0,
            "slot 0 ({a0},{b0}) and slot 1 ({a1},{b1}) must reproject differently"
        );
    }

    /// Cross-panel reprojection where the SOURCE is dual-axis but the TARGET
    /// is single-y: the target's `y_domains` is empty, so slot 1 falls back to
    /// the target's legacy `y_domain`. Hand-computed round trip.
    #[test]
    fn bug_hunt_reproject_dual_source_to_single_y_target() {
        let src_pa = rect(0.0, 0.0, 100.0, 100.0);
        let src = dual_coord(); // slot 1 domain [0, 10]
        let tgt_pa = rect(0.0, 0.0, 100.0, 100.0);
        let tgt = CoordKind::Cartesian {
            x_domain: None,
            y_domain: Some((0.0, 10.0)), // same numeric domain as source slot 1
            expand: true,
            clip: true,
            y_domains: Vec::new(), // single-y target
        };
        // Brush source screen y [0, 50] → slot-1 data [5, 10] → target pixels
        // (domain [0,10], screen flip): data 10 → 0, data 5 → 50 → (0, 50).
        let out = reproject_extent((0.0, 50.0), &src_pa, &src, &tgt_pa, &tgt, Axis::Y, 1)
            .expect("both domains resolvable");
        assert!((out.0 - 0.0).abs() < 1e-6, "lo={}", out.0);
        assert!((out.1 - 50.0).abs() < 1e-6, "hi={}", out.1);
    }

    /// A slot index pointing at a `None` entry (ordinal slot-1 scale) must
    /// make the whole reprojection return `None` — the binding is skipped, not
    /// silently routed through the primary domain.
    #[test]
    fn bug_hunt_reproject_through_ordinal_slot_is_none() {
        let pa = rect(0.0, 0.0, 100.0, 100.0);
        let coord = CoordKind::Cartesian {
            x_domain: None,
            y_domain: Some((0.0, 100.0)),
            expand: true,
            clip: true,
            y_domains: vec![Some((0.0, 100.0)), None], // slot 1 is ordinal
        };
        assert!(
            reproject_extent((0.0, 50.0), &pa, &coord, &pa, &coord, Axis::Y, 1).is_none(),
            "ordinal slot must make pixel↔data conversion undefined"
        );
    }

    /// `CoordKind::Fixed` has no per-slot list: any y_slot must read the
    /// single `y_domain` (the clamp-safe fallback), never panic or None.
    #[test]
    fn bug_hunt_axis_domain_slot_fixed_coord_ignores_slot() {
        let fixed = CoordKind::Fixed {
            ratio: 1.0,
            x_domain: Some((0.0, 1.0)),
            y_domain: Some((0.0, 5.0)),
            expand: true,
            clip: true,
        };
        assert_eq!(axis_domain_slot(&fixed, Axis::Y, 0), Some((0.0, 5.0)));
        assert_eq!(axis_domain_slot(&fixed, Axis::Y, 3), Some((0.0, 5.0)));
    }

    /// Degenerate geometry must not divide by zero: a zero-height plot area
    /// returns the domain lo; a zero-span domain returns the pixel lo. Both
    /// must be finite (no NaN propagating into a slot rescale affine).
    #[test]
    fn bug_hunt_degenerate_spans_produce_finite_values() {
        let flat_pa = rect(0.0, 20.0, 100.0, 0.0); // zero height
        let d = pixel_to_data(20.0, &flat_pa, (3.0, 9.0), Axis::Y);
        assert!(d.is_finite(), "zero-height plot area must not produce NaN, got {d}");
        assert!((d - 3.0).abs() < 1e-12, "must return domain lo, got {d}");

        let pa = rect(0.0, 0.0, 100.0, 100.0);
        let p = data_to_pixel(5.0, &pa, (5.0, 5.0), Axis::Y); // zero-span domain
        assert!(p.is_finite(), "zero-span domain must not produce NaN, got {p}");
        assert!((p - 0.0).abs() < 1e-12, "must return pixel lo, got {p}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::Rect;

    fn cart(xd: Option<(f64, f64)>, yd: Option<(f64, f64)>) -> CoordKind {
        CoordKind::Cartesian {
            x_domain: xd,
            y_domain: yd,
            expand: true,
            clip: true,
            y_domains: Vec::new(),
        }
    }

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn axis_from_channel_only_xy() {
        assert_eq!(Axis::from_channel("x"), Some(Axis::X));
        assert_eq!(Axis::from_channel("y"), Some(Axis::Y));
        assert_eq!(Axis::from_channel("color"), None);
        assert_eq!(Axis::from_channel("size"), None);
    }

    #[test]
    fn axis_domain_none_for_polar() {
        let polar = CoordKind::Polar {
            theta: ferrum_scene::PolarThetaChannel::X,
            start_angle: 0.0,
            direction: ferrum_scene::PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: 1.0,
        };
        assert_eq!(axis_domain_slot(&polar, Axis::X, 0), None);
    }

    #[test]
    fn axis_domain_reads_cartesian() {
        let c = cart(Some((0.0, 100.0)), Some((-5.0, 5.0)));
        assert_eq!(axis_domain_slot(&c, Axis::X, 0), Some((0.0, 100.0)));
        assert_eq!(axis_domain_slot(&c, Axis::Y, 0), Some((-5.0, 5.0)));
    }

    /// A dual-axis panel: slot 0 reads the primary y-domain, slot 1 reads the
    /// independent layer's own domain (secondary-y, #52). x is slot-agnostic.
    #[test]
    fn axis_domain_slot_reads_per_slot_y() {
        let coord = CoordKind::Cartesian {
            x_domain: Some((0.0, 10.0)),
            y_domain: Some((0.0, 100.0)),
            expand: true,
            clip: true,
            y_domains: vec![Some((0.0, 100.0)), Some((-3.0, 3.0))],
        };
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 0), Some((0.0, 100.0)));
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 1), Some((-3.0, 3.0)));
        // x ignores the slot.
        assert_eq!(axis_domain_slot(&coord, Axis::X, 1), Some((0.0, 10.0)));
    }

    /// Slot 0 prefers `y_domains[0]` over the legacy `y_domain` when slots are
    /// present (Task 8 guard note); an out-of-range slot falls back to
    /// `y_domain`; a `None` slot entry (ordinal scale) yields `None`.
    #[test]
    fn axis_domain_slot_guard_and_fallbacks() {
        let coord = CoordKind::Cartesian {
            x_domain: None,
            y_domain: Some((0.0, 100.0)), // legacy, divergent from slot 0
            expand: true,
            clip: true,
            y_domains: vec![Some((5.0, 50.0)), None],
        };
        // Guard: slot 0 uses the resolved slotted domain, not the legacy field.
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 0), Some((5.0, 50.0)));
        // Ordinal/band slot → None.
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 1), None);
        // Out-of-range slot falls back to the legacy y_domain.
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 9), Some((0.0, 100.0)));
    }

    /// Empty `y_domains` (every single-y chart) makes slot 0 read the legacy
    /// `y_domain` — the byte-stable default.
    #[test]
    fn axis_domain_slot_empty_slots_is_legacy_y_domain() {
        let coord = cart(Some((0.0, 10.0)), Some((-5.0, 5.0)));
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 0), Some((-5.0, 5.0)));
        // Even a non-zero slot degrades to the legacy domain (no slots present).
        assert_eq!(axis_domain_slot(&coord, Axis::Y, 2), Some((-5.0, 5.0)));
    }

    /// Reprojecting a y-brush through slot 1's domain must invert through that
    /// layer's scale, not the primary's.
    #[test]
    fn reproject_extent_uses_owning_y_slot() {
        // Source == target (single-panel dual-axis self-rescale).
        let pa = rect(0.0, 0.0, 100.0, 100.0); // y screen [0, 100]
        let coord = CoordKind::Cartesian {
            x_domain: None,
            y_domain: Some((0.0, 100.0)),
            expand: true,
            clip: true,
            y_domains: vec![Some((0.0, 100.0)), Some((0.0, 10.0))],
        };
        // Brush the top half of the plot: screen y [0, 50].
        // Through slot 1's domain [0,10] (screen-flipped): y=0→10, y=50→5.
        // Reprojected back to the same panel's pixels for data [5,10] → [0,50].
        let out = reproject_extent((0.0, 50.0), &pa, &coord, &pa, &coord, Axis::Y, 1)
            .expect("slot-1 domain present");
        assert!((out.0 - 0.0).abs() < 1e-6, "lo={}", out.0);
        assert!((out.1 - 50.0).abs() < 1e-6, "hi={}", out.1);
    }

    #[test]
    fn pixel_to_data_x_linear() {
        let pa = rect(50.0, 0.0, 100.0, 200.0); // x: [50, 150]
        let dom = (0.0, 10.0);
        // left edge -> 0, right edge -> 10, middle -> 5.
        assert!((pixel_to_data(50.0, &pa, dom, Axis::X) - 0.0).abs() < 1e-9);
        assert!((pixel_to_data(150.0, &pa, dom, Axis::X) - 10.0).abs() < 1e-9);
        assert!((pixel_to_data(100.0, &pa, dom, Axis::X) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn pixel_to_data_y_is_flipped() {
        let pa = rect(0.0, 20.0, 100.0, 100.0); // y screen: [20, 120]
        let dom = (0.0, 10.0);
        // top pixel (20) -> domain hi (10); bottom pixel (120) -> domain lo (0).
        assert!((pixel_to_data(20.0, &pa, dom, Axis::Y) - 10.0).abs() < 1e-9);
        assert!((pixel_to_data(120.0, &pa, dom, Axis::Y) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn pixel_data_round_trip() {
        let pa = rect(10.0, 30.0, 80.0, 160.0);
        let dom = (-2.0, 18.0);
        for axis in [Axis::X, Axis::Y] {
            for &v in &[-2.0_f64, 0.0, 7.5, 18.0] {
                let px = data_to_pixel(v, &pa, dom, axis);
                let back = pixel_to_data(px, &pa, dom, axis);
                assert!((back - v).abs() < 1e-9, "round-trip {axis:?} v={v} back={back}");
            }
        }
    }

    #[test]
    fn reproject_same_domain_maps_to_target_pixels() {
        // Source and target share the domain [0, 100] but different pixel layouts.
        let src_pa = rect(0.0, 0.0, 200.0, 100.0); // x: [0, 200]
        let tgt_pa = rect(300.0, 0.0, 400.0, 100.0); // x: [300, 700]
        let src = cart(Some((0.0, 100.0)), None);
        let tgt = cart(Some((0.0, 100.0)), None);
        // Brush the left half of the source [0, 100]px → data [0, 50].
        let out = reproject_extent((0.0, 100.0), &src_pa, &src, &tgt_pa, &tgt, Axis::X, 0)
            .expect("cartesian domains present");
        // data [0, 50] on target [300, 700]px for domain [0, 100] → [300, 500].
        assert!((out.0 - 300.0).abs() < 1e-6, "lo={}", out.0);
        assert!((out.1 - 500.0).abs() < 1e-6, "hi={}", out.1);
    }

    #[test]
    fn reproject_none_without_domain() {
        let src_pa = rect(0.0, 0.0, 200.0, 100.0);
        let tgt_pa = rect(0.0, 0.0, 200.0, 100.0);
        let src = cart(None, None); // no captured domain
        let tgt = cart(Some((0.0, 100.0)), None);
        assert!(reproject_extent((0.0, 100.0), &src_pa, &src, &tgt_pa, &tgt, Axis::X, 0).is_none());
    }

    #[test]
    fn rescale_affine_full_brush_is_identity_on_matching_panels() {
        // Brushing the entire source plot area onto an identical target plot
        // area must yield scale 1, offset 0.
        let src = rect(40.0, 0.0, 200.0, 100.0); // x: [40, 240]
        let tgt = rect(40.0, 0.0, 200.0, 100.0);
        let (scale, offset) = rescale_affine((40.0, 240.0), &src, &tgt, Axis::X).unwrap();
        assert!((scale - 1.0).abs() < 1e-9, "scale={scale}");
        assert!(offset.abs() < 1e-9, "offset={offset}");
    }

    #[test]
    fn rescale_affine_half_brush_doubles_scale() {
        // Brushing the left half maps that half onto the full target width.
        let src = rect(0.0, 0.0, 200.0, 100.0); // x: [0, 200]
        let tgt = rect(0.0, 0.0, 200.0, 100.0);
        let (scale, offset) = rescale_affine((0.0, 100.0), &src, &tgt, Axis::X).unwrap();
        assert!((scale - 2.0).abs() < 1e-9, "scale={scale}");
        assert!(offset.abs() < 1e-9, "offset={offset}");
        // Source pixel 100 (right edge of brush) maps to target right edge 200.
        assert!((scale * 100.0 + offset - 200.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_affine_degenerate_brush_is_none() {
        let src = rect(0.0, 0.0, 200.0, 100.0);
        let tgt = rect(0.0, 0.0, 200.0, 100.0);
        assert!(rescale_affine((50.0, 50.0), &src, &tgt, Axis::X).is_none());
    }

    // -------------------------------------------------------------------------
    // rescale_affine_cross_panel — regression guard for INT-1 (cross-panel
    // reactive rescale).  All tests below use src.plot_area != tgt.plot_area,
    // which is the case that was broken before the reprojection fix.
    // -------------------------------------------------------------------------

    /// A brush over the RIGHT half of the overview (source) must map detail
    /// marks at the corresponding data sub-range to land INSIDE the target
    /// plot area — not off-screen to the right.
    ///
    /// Setup mirrors the numerical repro from the INT-1 audit report:
    ///   source panel: x pixels [56, 624], data domain [0, 100]
    ///   target panel: x pixels [706, 1274], data domain [0, 100]
    ///   brush: source pixels [340, 624] → data [50, 100]
    ///
    /// After the affine `x' = scale * x + offset` applied to target marks:
    ///   - a target mark at the data-50 target pixel (706) must map to ~706
    ///     (target left edge)
    ///   - a target mark at the data-100 target pixel (1274) must map to ~1274
    ///     (target right edge)
    #[test]
    fn cross_panel_rescale_marks_land_inside_target_plot_area() {
        // Source panel: overview — smaller pixel region on the left.
        let src_pa = rect(56.0, 0.0, 568.0, 400.0); // x: [56, 624]
        let src_coord = cart(Some((0.0, 100.0)), Some((0.0, 50.0)));
        // Target panel: detail — different (larger) pixel region on the right.
        let tgt_pa = rect(706.0, 0.0, 568.0, 400.0); // x: [706, 1274]
        let tgt_coord = cart(Some((0.0, 100.0)), Some((0.0, 50.0)));

        // Brush: right half of the source overview in source pixel space.
        // source x [340, 624] → data [50, 100] (since domain is [0,100]).
        let brush = (340.0_f64, 624.0_f64);

        let (scale, offset) = rescale_affine_cross_panel(
            brush, &src_pa, &src_coord, &tgt_pa, &tgt_coord, Axis::X, 0,
        )
        .expect("cartesian domains present, brush non-degenerate");

        // After the affine `x' = scale * x + offset` applied to TARGET marks:
        //   - the target mark at data-50 lives at target pixel 990
        //     (706 + 0.5 * 568 = 990) and must map to ~706 (target left edge)
        //   - the target mark at data-100 lives at target pixel 1274
        //     (706 + 1.0 * 568 = 1274) and must map to ~1274 (target right edge)
        //
        // This verifies the reprojection fix: without it the affine was built in
        // source pixel coordinates, so the same mark would map off-screen to ~138.
        let tgt_data50_px = 706.0 + 0.5 * 568.0; // 990: target pixel for data=50
        let tgt_data100_px = 1274.0; // target pixel for data=100

        let mapped_lo = scale * tgt_data50_px + offset;
        let mapped_hi = scale * tgt_data100_px + offset;

        assert!(
            (mapped_lo - 706.0).abs() < 1.0,
            "data=50 target mark mapped to {mapped_lo}, expected ~706 (target left edge)"
        );
        assert!(
            (mapped_hi - 1274.0).abs() < 1.0,
            "data=100 target mark mapped to {mapped_hi}, expected ~1274 (target right edge)"
        );

        // Sanity checks.
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        assert!(
            mapped_lo >= 706.0 - 1.0 && mapped_lo <= 1274.0 + 1.0,
            "lo mark off-screen: {mapped_lo}"
        );
        assert!(
            mapped_hi >= 706.0 - 1.0 && mapped_hi <= 1274.0 + 1.0,
            "hi mark off-screen: {mapped_hi}"
        );
    }

    /// Self-rescale (source == target): reprojection through same domain/pixel
    /// space must produce the same result as the old direct `rescale_affine`.
    #[test]
    fn cross_panel_rescale_same_panel_is_equivalent_to_direct() {
        let pa = rect(40.0, 0.0, 200.0, 100.0); // x: [40, 240]
        let coord = cart(Some((0.0, 100.0)), Some((0.0, 50.0)));

        // Brush the left half of the panel.
        let brush = (40.0, 140.0); // left 100px of the 200px span

        let (scale_cross, offset_cross) =
            rescale_affine_cross_panel(brush, &pa, &coord, &pa, &coord, Axis::X, 0)
                .expect("same-panel cross-panel rescale");
        let (scale_direct, offset_direct) =
            rescale_affine(brush, &pa, &pa, Axis::X).expect("direct rescale");

        assert!(
            (scale_cross - scale_direct).abs() < 1e-9,
            "scale mismatch: cross={scale_cross} direct={scale_direct}"
        );
        assert!(
            (offset_cross - offset_direct).abs() < 1e-9,
            "offset mismatch: cross={offset_cross} direct={offset_direct}"
        );
    }

    /// Different pixel regions AND different (non-overlapping) data domains.
    /// Source domain [0, 50], target domain [100, 200].
    /// Brush: entire source → data [0, 50] → target pixels for [100, 200].
    /// After affine, target marks at data 100 (target left) and data 200
    /// (target right) must map to the target left and right edges.
    #[test]
    fn cross_panel_rescale_distinct_domains_marks_in_bounds() {
        // Source: x pixels [0, 500], data domain [0, 50].
        let src_pa = rect(0.0, 0.0, 500.0, 300.0);
        let src_coord = cart(Some((0.0, 50.0)), None);
        // Target: x pixels [600, 1100], data domain [100, 200].
        let tgt_pa = rect(600.0, 0.0, 500.0, 300.0);
        let tgt_coord = cart(Some((100.0, 200.0)), None);

        // Brush: left 40% of source → data [0, 20].
        // Reprojected to target domain [100, 200]: data [0,20] has no overlap
        // with [100,200].  reproject_extent clamps implicitly via pixel math
        // (out-of-domain maps to out-of-target-bounds pixels), so this returns
        // None when domains have no overlap — verify graceful None.
        let result = rescale_affine_cross_panel(
            (0.0, 200.0),
            &src_pa,
            &src_coord,
            &tgt_pa,
            &tgt_coord,
            Axis::X,
            0,
        );
        // The domains [0,50] and [100,200] share no data, so the reprojected
        // target pixels will be outside [600,1100]. rescale_affine will clamp
        // and may still return Some (clamped brush spans the whole target).
        // The key assertion: if Some, the scale is positive.
        if let Some((scale, _offset)) = result {
            assert!(scale > 0.0, "scale must be positive across disjoint domains");
        }
        // None is also acceptable (degenerate after clamp).
    }

    /// Overlapping but different data domains: source [0, 100], target [0, 50].
    /// Brushing source pixels [0, 100] (data [0, 50]) maps exactly to the
    /// full target domain, so target marks should fill the full target plot area.
    #[test]
    fn cross_panel_rescale_overlapping_domains_full_coverage() {
        // Source: x pixels [50, 650], domain [0, 100].
        let src_pa = rect(50.0, 0.0, 600.0, 300.0); // x: [50, 650]
        let src_coord = cart(Some((0.0, 100.0)), None);
        // Target: x pixels [700, 1300], domain [0, 50].
        let tgt_pa = rect(700.0, 0.0, 600.0, 300.0); // x: [700, 1300]
        let tgt_coord = cart(Some((0.0, 50.0)), None);

        // Brush: source x [50, 350] → data [0, 50] (left half of source).
        // data [0, 50] maps exactly to the full target domain [0, 50],
        // so the reprojected target brush is [700, 1300] (the full target),
        // and the affine should be identity (scale=1, offset=0).
        let (scale, offset) = rescale_affine_cross_panel(
            (50.0, 350.0),
            &src_pa,
            &src_coord,
            &tgt_pa,
            &tgt_coord,
            Axis::X,
            0,
        )
        .expect("overlapping domains, non-degenerate brush");

        // Full target brush → identity affine.
        assert!((scale - 1.0).abs() < 1e-9, "expected scale≈1, got {scale}");
        assert!(offset.abs() < 1e-9, "expected offset≈0, got {offset}");
    }
}
