//! Width along a tapered stroke: the one interpolation rule and the one dasher
//! every consumer of a [`SceneNode::Polyline`](crate::SceneNode::Polyline) or
//! [`SceneNode::Path`](crate::SceneNode::Path) `widths` field shares.
//!
//! A tapered centerline is a list of vertices with one full width (scene px)
//! per vertex. Along a straight segment the width is linear in arc length
//! between the segment's two vertex widths ([`width_at`]); a cut at any
//! parameter (a dash boundary, a break-axis crossing) takes its width from that
//! rule, so every backend agrees on the width at every point.
//!
//! A tapered [`SceneNode::Path`](crate::SceneNode::Path) carries one width per
//! command endpoint, and its curve commands are flattened by
//! [`flatten_tapered_path`] before anything else reads it. Along a curve
//! command the width is linear in the curve parameter between the previous
//! endpoint's width and the command's own: each flattened vertex takes
//! [`width_at`] with the curve parameter as its fraction. Once flattened, the
//! centerline is straight segments and the arc-length rule holds along each.
//!
//! Pure f64 arithmetic: deterministic, no randomness, and no allocation beyond
//! the centerlines and pieces the functions return.

use std::fmt;

use crate::PathCmd;

/// A tapered stroke's `widths` that cannot be drawn.
#[derive(Debug, Clone, PartialEq)]
pub enum TaperError {
    /// `widths` does not carry exactly one entry per vertex.
    Misaligned { points: usize, widths: usize },
    /// A width is non-finite or not `> 0`.
    InvalidWidth { index: usize, value: f64 },
    /// A tapered path is closed (`closed == true` or a `Close` command): a
    /// tapered stroke is an open centerline, so a closing segment has no
    /// width to draw.
    Closed,
}

impl fmt::Display for TaperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Misaligned { points, widths } => write!(
                f,
                "tapered stroke has {widths} widths for {points} vertices; it needs one width per vertex"
            ),
            Self::InvalidWidth { index, value } => write!(
                f,
                "tapered stroke width {index} is {value}; every width must be finite and greater than 0"
            ),
            Self::Closed => write!(f, "tapered stroke is a closed path; a tapered stroke must be an open path"),
        }
    }
}

impl std::error::Error for TaperError {}

/// Check that `widths` aligns one for one with `points` vertices and that every
/// width is finite and `> 0`.
pub fn validate_widths(points: usize, widths: &[f64]) -> Result<(), TaperError> {
    if widths.len() != points {
        return Err(TaperError::Misaligned { points, widths: widths.len() });
    }
    match widths.iter().position(|w| !(w.is_finite() && *w > 0.0)) {
        Some(index) => Err(TaperError::InvalidWidth { index, value: widths[index] }),
        None => Ok(()),
    }
}

/// The width at fraction `t` of arc length along segment `segment` (from
/// vertex `segment` to vertex `segment + 1`). `t` is clamped to `[0, 1]`;
/// `None` when the segment has no two vertex widths.
pub fn width_at(widths: &[f64], segment: usize, t: f64) -> Option<f64> {
    let w0 = *widths.get(segment)?;
    let w1 = *widths.get(segment.checked_add(1)?)?;
    Some(w0 + (w1 - w0) * t.clamp(0.0, 1.0))
}

/// One on-piece of a dashed tapered centerline: its vertices and their widths,
/// aligned one for one.
#[derive(Debug, Clone, PartialEq)]
pub struct TaperPiece {
    pub points: Vec<(f64, f64)>,
    pub widths: Vec<f64>,
}

impl TaperPiece {
    fn start(point: (f64, f64), width: f64) -> Self {
        Self { points: vec![point], widths: vec![width] }
    }

    /// Append a vertex, skipping one that repeats the last (a dash boundary
    /// that lands exactly on a vertex is reached from both sides).
    fn push(&mut self, point: (f64, f64), width: f64) {
        if self.points.last() != Some(&point) {
            self.points.push(point);
            self.widths.push(width);
        }
    }
}

/// Lay `dash` (scene px, alternating on and off lengths) along the tapered
/// centerline `points`/`widths` and return the on-pieces.
///
/// SVG `stroke-dasharray` semantics: the pattern starts at phase 0 at the first
/// point and continues across vertices, so an on-dash spanning a vertex is one
/// piece bending through it; an odd-length pattern repeats to even length. A
/// pattern with no positive length, or with a negative or non-finite entry,
/// draws solid (one piece, the whole centerline), as SVG does. Each piece's
/// end widths are interpolated at its cut points by [`width_at`]. A zero-length
/// on-dash yields no piece.
pub fn dash_tapered(
    points: &[(f64, f64)],
    widths: &[f64],
    dash: &[f64],
) -> Result<Vec<TaperPiece>, TaperError> {
    validate_widths(points.len(), widths)?;
    let solid = dash.is_empty()
        || dash.iter().any(|d| !d.is_finite() || *d < 0.0)
        || dash.iter().sum::<f64>() <= 0.0;
    if solid {
        return Ok(vec![TaperPiece { points: points.to_vec(), widths: widths.to_vec() }]);
    }

    let mut pieces = Vec::new();
    let Some((&first, &first_width)) = points.first().zip(widths.first()) else {
        return Ok(pieces);
    };
    // Entry `k` cycles the pattern while on and off alternate at every
    // boundary, which is the even-length repetition SVG gives an odd pattern.
    let mut k = 0usize;
    let mut remaining = dash[0];
    let mut current = Some(TaperPiece::start(first, first_width));

    for (segment, pair) in points.windows(2).enumerate() {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        let length = (x1 - x0).hypot(y1 - y0);
        if !length.is_finite() {
            continue;
        }
        let mut pos = 0.0;
        // Every dash boundary strictly inside this segment cuts it; one that
        // falls exactly on its end is taken at the start of the next.
        while length > 0.0 && remaining < length - pos {
            pos += remaining;
            let t = pos / length;
            let cut = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
            let width = width_at(widths, segment, t).unwrap_or(first_width);
            match current.take() {
                Some(mut piece) => {
                    piece.push(cut, width);
                    if piece.points.len() >= 2 {
                        pieces.push(piece);
                    }
                }
                None => current = Some(TaperPiece::start(cut, width)),
            }
            k += 1;
            remaining = dash[k % dash.len()];
        }
        remaining -= length - pos;
        if let Some(piece) = current.as_mut() {
            piece.push(pair[1], widths[segment + 1]);
        }
    }
    if let Some(piece) = current.filter(|p| p.points.len() >= 2) {
        pieces.push(piece);
    }
    Ok(pieces)
}

/// Straight segments each curve command is flattened into before outlining.
const CURVE_SEGMENTS: usize = 16;

/// Flatten a tapered path into its subpaths' centerlines, one vertex width
/// each, ready for [`dash_tapered`] and outlining. `widths[i]` is the width
/// at command `i`'s endpoint; along a curve the width is [`width_at`] the
/// curve parameter between the previous endpoint's width and the command's
/// own. Quadratic, cubic and arc commands become [`CURVE_SEGMENTS`] straight
/// segments each; a command with no current point starts a subpath at its
/// endpoint. Refuses misaligned or non-positive `widths` and a `Close`
/// command.
pub fn flatten_tapered_path(commands: &[PathCmd], widths: &[f64]) -> Result<Vec<TaperPiece>, TaperError> {
    validate_widths(commands.len(), widths)?;
    let mut subpaths: Vec<TaperPiece> = Vec::new();
    for (i, (cmd, &w)) in commands.iter().zip(widths).enumerate() {
        let current = subpaths.last().and_then(|s| s.points.last().copied());
        let start = match (cmd, current) {
            (PathCmd::Close, _) => return Err(TaperError::Closed),
            (PathCmd::MoveTo { .. }, _) | (_, None) => {
                subpaths.push(TaperPiece { points: vec![command_end(cmd, (0.0, 0.0))], widths: vec![w] });
                continue;
            }
            (_, Some(current)) => current,
        };
        let Some(sub) = subpaths.last_mut() else { continue };
        // A current point means an earlier command, so segment `i - 1` runs
        // from that command's endpoint width to this one's.
        let mut push = |t: f64, p: (f64, f64)| {
            sub.points.push(p);
            // `current` is `Some` only past the first command, so `i >= 1`
            // here and both `widths[i - 1]` and `widths[i]` are in bounds
            // (validate_widths above already sized `widths` to `commands`);
            // `width_at` is never `None`. `w` is an unreachable fallback,
            // kept only because `width_at` returns `Option`.
            sub.widths.push(width_at(widths, i - 1, t).unwrap_or(w));
        };
        let steps = (1..=CURVE_SEGMENTS).map(|k| k as f64 / CURVE_SEGMENTS as f64);
        match *cmd {
            PathCmd::QuadTo { cx, cy, x, y } => {
                for t in steps {
                    let u = 1.0 - t;
                    push(t, (
                        u * u * start.0 + 2.0 * u * t * cx + t * t * x,
                        u * u * start.1 + 2.0 * u * t * cy + t * t * y,
                    ));
                }
            }
            PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
                for t in steps {
                    let u = 1.0 - t;
                    let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    push(t, (
                        a * start.0 + b * c1x + c * c2x + d * x,
                        a * start.1 + b * c1y + c * c2y + d * y,
                    ));
                }
            }
            PathCmd::ArcTo { rx, ry, rotation, large_arc, sweep, x, y } => {
                match arc_center(start, rx, ry, rotation, large_arc, sweep, (x, y)) {
                    Some(arc) => steps.for_each(|t| push(t, arc.at(t))),
                    None => push(1.0, (x, y)),
                }
            }
            _ => push(1.0, command_end(cmd, start)),
        }
    }
    Ok(subpaths)
}

/// The endpoint a command reaches from `current`.
fn command_end(cmd: &PathCmd, current: (f64, f64)) -> (f64, f64) {
    match *cmd {
        PathCmd::MoveTo { x, y }
        | PathCmd::LineTo { x, y }
        | PathCmd::QuadTo { x, y, .. }
        | PathCmd::CubicTo { x, y, .. }
        | PathCmd::ArcTo { x, y, .. } => (x, y),
        PathCmd::HLineTo { x } => (x, current.1),
        PathCmd::VLineTo { y } => (current.0, y),
        PathCmd::Close => current,
    }
}

/// An SVG elliptical arc in centre parameterization.
struct CenterArc {
    center: (f64, f64),
    radii: (f64, f64),
    /// `(cos, sin)` of the ellipse's x-axis rotation.
    axis: (f64, f64),
    start_angle: f64,
    sweep_angle: f64,
}

impl CenterArc {
    /// The point at fraction `t` of the arc's parameter.
    fn at(&self, t: f64) -> (f64, f64) {
        let theta = self.start_angle + self.sweep_angle * t;
        let (ex, ey) = (self.radii.0 * theta.cos(), self.radii.1 * theta.sin());
        let (c, s) = self.axis;
        (self.center.0 + c * ex - s * ey, self.center.1 + s * ex + c * ey)
    }
}

/// SVG's endpoint-to-centre arc conversion (SVG 1.1 implementation notes
/// F.6.5, with F.6.6's out-of-range radii scaled up). `None` where SVG draws
/// the arc as a straight line: a zero radius or coincident endpoints.
fn arc_center(
    from: (f64, f64),
    rx: f64,
    ry: f64,
    rotation_deg: f64,
    large_arc: bool,
    sweep: bool,
    to: (f64, f64),
) -> Option<CenterArc> {
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx == 0.0 || ry == 0.0 || from == to {
        return None;
    }
    let phi = rotation_deg.to_radians();
    let (c, s) = (phi.cos(), phi.sin());
    let (dx, dy) = ((from.0 - to.0) / 2.0, (from.1 - to.1) / 2.0);
    let (x1, y1) = (c * dx + s * dy, -s * dx + c * dy);
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        rx *= lambda.sqrt();
        ry *= lambda.sqrt();
    }
    let num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1;
    let den = rx * rx * y1 * y1 + ry * ry * x1 * x1;
    let sign = if large_arc == sweep { -1.0 } else { 1.0 };
    let coef = sign * (num.max(0.0) / den).sqrt();
    let (cx1, cy1) = (coef * rx * y1 / ry, -coef * ry * x1 / rx);
    let center = (
        c * cx1 - s * cy1 + (from.0 + to.0) / 2.0,
        s * cx1 + c * cy1 + (from.1 + to.1) / 2.0,
    );
    let angle = |u: (f64, f64), v: (f64, f64)| (u.0 * v.1 - u.1 * v.0).atan2(u.0 * v.0 + u.1 * v.1);
    let u = ((x1 - cx1) / rx, (y1 - cy1) / ry);
    let v = ((-x1 - cx1) / rx, (-y1 - cy1) / ry);
    let start_angle = angle((1.0, 0.0), u);
    let mut sweep_angle = angle(u, v);
    if !sweep && sweep_angle > 0.0 {
        sweep_angle -= std::f64::consts::TAU;
    } else if sweep && sweep_angle < 0.0 {
        sweep_angle += std::f64::consts::TAU;
    }
    Some(CenterArc { center, radii: (rx, ry), axis: (c, s), start_angle, sweep_angle })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn assert_piece(piece: &TaperPiece, points: &[(f64, f64)], widths: &[f64]) {
        assert_eq!(piece.points.len(), points.len(), "piece {piece:?} vs points {points:?}");
        for (got, want) in piece.points.iter().zip(points) {
            assert!(close(got.0, want.0) && close(got.1, want.1), "point {got:?} != {want:?} in {piece:?}");
        }
        assert_eq!(piece.widths.len(), widths.len(), "piece {piece:?} vs widths {widths:?}");
        for (got, want) in piece.widths.iter().zip(widths) {
            assert!(close(*got, *want), "width {got} != {want} in {piece:?}");
        }
    }

    #[test]
    fn width_at_is_linear_in_the_segment_fraction() {
        let widths = [2.0, 6.0, 4.0];
        assert_eq!(width_at(&widths, 0, 0.0), Some(2.0));
        assert_eq!(width_at(&widths, 0, 0.25), Some(3.0));
        assert_eq!(width_at(&widths, 0, 1.0), Some(6.0));
        assert_eq!(width_at(&widths, 1, 0.5), Some(5.0));
    }

    #[test]
    fn width_at_clamps_the_fraction_and_refuses_a_missing_segment() {
        let widths = [2.0, 6.0];
        assert_eq!(width_at(&widths, 0, -1.0), Some(2.0));
        assert_eq!(width_at(&widths, 0, 2.0), Some(6.0));
        assert_eq!(width_at(&widths, 1, 0.5), None);
        assert_eq!(width_at(&widths, usize::MAX, 0.5), None);
        assert_eq!(width_at(&[], 0, 0.5), None);
    }

    #[test]
    fn validate_widths_refuses_misaligned_and_non_positive_widths() {
        assert_eq!(validate_widths(2, &[1.0, 2.0]), Ok(()));
        assert_eq!(validate_widths(3, &[1.0, 2.0]), Err(TaperError::Misaligned { points: 3, widths: 2 }));
        assert_eq!(validate_widths(2, &[1.0, 0.0]), Err(TaperError::InvalidWidth { index: 1, value: 0.0 }));
        assert_eq!(validate_widths(2, &[-1.0, 1.0]), Err(TaperError::InvalidWidth { index: 0, value: -1.0 }));
        assert!(matches!(
            validate_widths(2, &[1.0, f64::NAN]),
            Err(TaperError::InvalidWidth { index: 1, .. })
        ));
        assert!(matches!(
            validate_widths(1, &[f64::INFINITY]),
            Err(TaperError::InvalidWidth { index: 0, .. })
        ));
    }

    #[test]
    fn dash_pieces_carry_widths_interpolated_at_their_cuts() {
        // Length 10, widths 2 -> 12: width = 2 + s along the line.
        let pieces = dash_tapered(&[(0.0, 0.0), (10.0, 0.0)], &[2.0, 12.0], &[2.0, 3.0]).unwrap();
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        assert_piece(&pieces[0], &[(0.0, 0.0), (2.0, 0.0)], &[2.0, 4.0]);
        assert_piece(&pieces[1], &[(5.0, 0.0), (7.0, 0.0)], &[7.0, 9.0]);
    }

    #[test]
    fn dash_phase_continues_across_vertices() {
        // An L: 3 px along x, then 4 px along y. The first on-dash (5 px)
        // bends through the corner vertex instead of restarting there.
        let points = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        let widths = [1.0, 4.0, 8.0];
        let pieces = dash_tapered(&points, &widths, &[5.0, 1.0]).unwrap();
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        assert_piece(&pieces[0], &[(0.0, 0.0), (3.0, 0.0), (3.0, 2.0)], &[1.0, 4.0, 6.0]);
        assert_piece(&pieces[1], &[(3.0, 3.0), (3.0, 4.0)], &[7.0, 8.0]);
    }

    #[test]
    fn an_odd_pattern_repeats_to_even_length() {
        // [2] reads as [2, 2]: on 0-2, off 2-4, on 4-6, off 6-8, on 8-10.
        let pieces = dash_tapered(&[(0.0, 0.0), (10.0, 0.0)], &[1.0, 1.0], &[2.0]).unwrap();
        let spans: Vec<(f64, f64)> = pieces.iter().map(|p| (p.points[0].0, p.points[p.points.len() - 1].0)).collect();
        assert_eq!(spans, vec![(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]);
    }

    #[test]
    fn a_boundary_on_a_vertex_does_not_repeat_the_vertex() {
        // The first on-dash ends exactly at the corner; the next on-dash
        // starts 1 px past it.
        let points = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)];
        let pieces = dash_tapered(&points, &[1.0, 1.0, 1.0], &[2.0, 1.0]).unwrap();
        assert_piece(&pieces[0], &[(0.0, 0.0), (2.0, 0.0)], &[1.0, 1.0]);
        assert_piece(&pieces[1], &[(2.0, 1.0), (2.0, 3.0)], &[1.0, 1.0]);
        assert_eq!(pieces.len(), 2, "{pieces:?}");
    }

    #[test]
    fn a_pattern_with_no_positive_length_draws_solid() {
        let points = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        let widths = [1.0, 2.0, 3.0];
        let whole = vec![TaperPiece { points: points.to_vec(), widths: widths.to_vec() }];
        for dash in [&[][..], &[0.0, 0.0], &[2.0, -1.0], &[f64::NAN, 1.0]] {
            assert_eq!(dash_tapered(&points, &widths, dash).unwrap(), whole, "dash {dash:?}");
        }
    }

    #[test]
    fn dashing_refuses_misaligned_widths() {
        assert_eq!(
            dash_tapered(&[(0.0, 0.0), (1.0, 0.0)], &[1.0], &[1.0, 1.0]),
            Err(TaperError::Misaligned { points: 2, widths: 1 })
        );
    }

    #[test]
    fn a_curve_flattens_with_widths_linear_in_its_parameter() {
        let commands = [
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::QuadTo { cx: 5.0, cy: 10.0, x: 10.0, y: 0.0 },
            PathCmd::LineTo { x: 20.0, y: 0.0 },
        ];
        let subpaths = flatten_tapered_path(&commands, &[2.0, 10.0, 4.0]).unwrap();
        assert_eq!(subpaths.len(), 1, "{subpaths:?}");
        let sub = &subpaths[0];
        assert_eq!(sub.points.len(), CURVE_SEGMENTS + 2);
        for k in 0..=CURVE_SEGMENTS {
            let t = k as f64 / CURVE_SEGMENTS as f64;
            let (u, p) = (1.0 - t, &sub.points[k]);
            assert!(close(p.0, 10.0 * u * t + 10.0 * t * t) && close(p.1, 20.0 * u * t), "vertex {k}: {p:?}");
            assert!(close(sub.widths[k], 2.0 + 8.0 * t), "width {k}: {}", sub.widths[k]);
        }
        assert_eq!(sub.points[CURVE_SEGMENTS + 1], (20.0, 0.0));
        assert_eq!(sub.widths[CURVE_SEGMENTS + 1], 4.0);
    }

    #[test]
    fn a_move_starts_a_new_subpath_at_its_own_width() {
        let commands = [
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::HLineTo { x: 5.0 },
            PathCmd::MoveTo { x: 0.0, y: 9.0 },
            PathCmd::VLineTo { y: 3.0 },
        ];
        let subpaths = flatten_tapered_path(&commands, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let want = vec![
            TaperPiece { points: vec![(0.0, 0.0), (5.0, 0.0)], widths: vec![1.0, 2.0] },
            TaperPiece { points: vec![(0.0, 9.0), (0.0, 3.0)], widths: vec![3.0, 4.0] },
        ];
        assert_eq!(subpaths, want);
    }

    #[test]
    fn flattening_refuses_a_close_and_misaligned_widths() {
        let closed = [PathCmd::MoveTo { x: 0.0, y: 0.0 }, PathCmd::LineTo { x: 1.0, y: 0.0 }, PathCmd::Close];
        assert_eq!(flatten_tapered_path(&closed, &[1.0, 1.0, 1.0]), Err(TaperError::Closed));
        assert_eq!(
            flatten_tapered_path(&closed[..2], &[1.0]),
            Err(TaperError::Misaligned { points: 2, widths: 1 })
        );
    }

    #[test]
    fn dashing_is_deterministic() {
        let points: Vec<(f64, f64)> = (0..40).map(|i| (i as f64 * 1.7, (i as f64 * 0.9).sin() * 9.0)).collect();
        let widths: Vec<f64> = (0..40).map(|i| 1.0 + i as f64 * 0.25).collect();
        let a = dash_tapered(&points, &widths, &[4.0, 2.0, 1.0, 2.0]).unwrap();
        let b = dash_tapered(&points, &widths, &[4.0, 2.0, 1.0, 2.0]).unwrap();
        assert_eq!(a, b);
        assert!(a.len() > 5, "{} pieces", a.len());
        for piece in &a {
            assert_eq!(piece.points.len(), piece.widths.len());
        }
    }
}
