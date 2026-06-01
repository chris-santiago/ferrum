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

/// The data domain `(lo, hi)` for the given axis of a Cartesian/Fixed panel.
///
/// Returns `None` for non-cartesian coords (polar/geo) or when the domain is
/// unset (auto-inferred — no static domain captured), in which case pixel↔data
/// conversion is undefined and the caller skips the binding.
pub(crate) fn axis_domain(coord: &CoordKind, axis: Axis) -> Option<(f64, f64)> {
    let (xd, yd) = match coord {
        CoordKind::Cartesian { x_domain, y_domain, .. }
        | CoordKind::Fixed { x_domain, y_domain, .. } => (*x_domain, *y_domain),
        CoordKind::Polar { .. } | CoordKind::Geo { .. } => return None,
    };
    match axis {
        Axis::X => xd,
        Axis::Y => yd,
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
) -> Option<(f64, f64)> {
    let source_domain = axis_domain(source_coord, axis)?;
    let target_domain = axis_domain(target_coord, axis)?;
    let (px_lo, px_hi) = normalize(brush_px.0, brush_px.1);
    let d0 = pixel_to_data(px_lo, source_plot_area, source_domain, axis);
    let d1 = pixel_to_data(px_hi, source_plot_area, source_domain, axis);
    let (d_lo, d_hi) = normalize(d0, d1);
    let t0 = data_to_pixel(d_lo, target_plot_area, target_domain, axis);
    let t1 = data_to_pixel(d_hi, target_plot_area, target_domain, axis);
    Some(normalize(t0, t1))
}

/// Affine scale + translate `(scale, offset)` that maps the brushed pixel
/// sub-extent of the source panel onto the full pixel extent of the target
/// panel for one axis — the boxzoom transform, per-axis.
///
/// `screen_x' = scale * screen_x + offset`. This is consumed by
/// `ZoomPanState` (the same affine the wheel/pan/D3-zoom path applies), so the
/// target panel rescales to show only the brushed domain.
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
        assert_eq!(axis_domain(&polar, Axis::X), None);
    }

    #[test]
    fn axis_domain_reads_cartesian() {
        let c = cart(Some((0.0, 100.0)), Some((-5.0, 5.0)));
        assert_eq!(axis_domain(&c, Axis::X), Some((0.0, 100.0)));
        assert_eq!(axis_domain(&c, Axis::Y), Some((-5.0, 5.0)));
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
        let out = reproject_extent((0.0, 100.0), &src_pa, &src, &tgt_pa, &tgt, Axis::X)
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
        assert!(reproject_extent((0.0, 100.0), &src_pa, &src, &tgt_pa, &tgt, Axis::X).is_none());
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
}
