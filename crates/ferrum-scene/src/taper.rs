//! Width along a tapered stroke: the one interpolation rule and the one dasher
//! every consumer of a [`SceneNode::Polyline`](crate::SceneNode::Polyline) or
//! [`SceneNode::Path`](crate::SceneNode::Path) `widths` field shares.
//!
//! A tapered centerline is a list of vertices with one full width (scene px)
//! per vertex. Along a segment the width is linear in arc length between the
//! segment's two vertex widths ([`width_at`]); a cut at any parameter (a dash
//! boundary, a break-axis crossing, a curve flattening step) takes its width
//! from that rule, so every backend agrees on the width at every point.
//!
//! Pure f64 arithmetic: deterministic, no randomness, and no allocation beyond
//! the pieces [`dash_tapered`] returns.

use std::fmt;

/// A tapered stroke's `widths` that cannot be drawn.
#[derive(Debug, Clone, PartialEq)]
pub enum TaperError {
    /// `widths` does not carry exactly one entry per vertex.
    Misaligned { points: usize, widths: usize },
    /// A width is non-finite or not `> 0`.
    InvalidWidth { index: usize, value: f64 },
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
