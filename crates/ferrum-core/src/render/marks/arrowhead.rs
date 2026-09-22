//! Arrowhead geometry shared by three consumers: [`super::segment`]'s
//! `arrow=true` path, [`super::super::annotation`]'s `emit_arrow`, and
//! [`super::super::break_axis`]'s `remap_arrowhead`. Moved out of `draw.rs`
//! (which had become the default shared-code dumping ground — see
//! `crates/ferrum-core/src/render/draw.rs`'s remaining contents for the
//! draw-context/dispatch code that stayed behind) into its own home because
//! three consumers across three subsystems, not `draw.rs`'s size, is the
//! reason this geometry needed a dedicated module.

/// Compute the three vertices of a directed arrowhead triangle whose tip
/// sits at `(x2, y2)`, pointing away from `(x1, y1)`. This is the geometry
/// source of truth extracted from `annotation::emit_arrow` (spec §4 H1) —
/// shared by `annotate_arrow`'s unlabelled overlay path and
/// `marks::segment::build`'s `arrow=true` path so the two draw byte-identical
/// triangles for the same endpoints and `head_size`.
///
/// Returns `None` when there is no direction to point in: `head_size <= 0.0`,
/// or the segment has zero length. `head_size` is clamped to the segment's
/// own length first, so an epsilon-length line does not produce a triangle
/// that overshoots its own start point — the same guard `emit_arrow` applied
/// inline before this helper existed.
pub(crate) fn arrowhead_triangle(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    head_size: f64,
) -> Option<[(f64, f64); 3]> {
    if head_size <= 0.0 {
        return None;
    }
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 0.0 {
        return None;
    }
    let ux = dx / len;
    let uy = dy / len;
    let px = -uy;
    let py = ux;
    let head_size = head_size.min(len);
    let half = head_size * 0.5;
    let tip = (x2, y2);
    let left = (x2 - ux * head_size + px * half, y2 - uy * head_size + py * half);
    let right = (x2 - ux * head_size - px * half, y2 - uy * head_size - py * half);
    Some([tip, left, right])
}

/// Build the filled-triangle [`ferrum_scene::SceneNode::Path`] for an
/// arrowhead from [`arrowhead_triangle`]'s three vertices, a fill paint, and
/// full paint-opacity fields (`opacity`, `stroke_opacity`, `fill_opacity`).
/// No stroke.
///
/// `fill` is `Option` so a head inherits its shaft's *cleared* paint: a
/// `mark_segment(stroke="none", arrow=True)` paints neither shaft nor head
/// (`#110`).
///
/// This is the general entry point — [`arrowhead_path`] is the convenience
/// wrapper for callers with no per-row opacity to thread (`annotation.rs`'s
/// `emit_arrow`, which never resolves a per-row opacity). `marks::segment`'s
/// `arrow=true` path calls this one directly so the arrowhead inherits the
/// SAME resolved `opacity`/`stroke_opacity`/`fill_opacity` the row's shaft
/// `Line` carries — mirroring the `PaintChannels::new`/struct-literal split at
/// `draw.rs`'s `to_scene_fill_stroke` (batch H
/// Task 1 quality-review fix: the arrowhead previously always rendered
/// fully opaque regardless of the segment's own resolved opacity, a visible
/// mismatch under `opacity=`/`stroke_opacity=`).
pub(crate) fn arrowhead_path_with_opacity(
    points: [(f64, f64); 3],
    fill: Option<ferrum_scene::Color>,
    opacity: f64,
    stroke_opacity: f64,
    fill_opacity: f64,
) -> ferrum_scene::SceneNode {
    let [tip, left, right] = points;
    ferrum_scene::SceneNode::Path {
        commands: vec![
            ferrum_scene::PathCmd::MoveTo { x: tip.0, y: tip.1 },
            ferrum_scene::PathCmd::LineTo { x: left.0, y: left.1 },
            ferrum_scene::PathCmd::LineTo { x: right.0, y: right.1 },
            ferrum_scene::PathCmd::Close,
        ],
        style: ferrum_scene::FillStroke {
            fill,
            stroke: None,
            stroke_width: 0.0,
            opacity,
            stroke_dash: None,
            stroke_opacity,
            fill_opacity,
            angle: 0.0,
        },
        closed: true,
    }
}

/// Fully opaque convenience wrapper over [`arrowhead_path_with_opacity`] —
/// matches every arrowhead ferrum has ever drawn (`annotation::emit_arrow`'s
/// shape, unchanged by this extraction). Used only where there is no
/// per-row opacity to inherit.
pub(crate) fn arrowhead_path(
    points: [(f64, f64); 3],
    color: ferrum_scene::Color,
) -> ferrum_scene::SceneNode {
    arrowhead_path_with_opacity(points, Some(color), 1.0, 1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Batch H Task 1: `arrowhead_triangle` / `arrowhead_path` ─────────────
    //
    // `arrowhead_triangle` is the geometry source of truth extracted from
    // `annotation::emit_arrow` (which called this exact math inline before
    // this extraction). These tests pin its contract directly so both
    // `emit_arrow` and `marks::segment::build`'s new `arrow=true` path can be
    // proven to call the SAME function for the SAME inputs, rather than two
    // independently-typed copies that could silently drift.

    /// A horizontal segment's arrowhead triangle: tip at (x2,y2), base
    /// centered `head_size` back from the tip, perpendicular half-width
    /// `head_size / 2` on each side.
    #[test]
    fn arrowhead_triangle_horizontal_segment_geometry() {
        let points = arrowhead_triangle(0.0, 0.0, 100.0, 0.0, 8.0)
            .expect("non-zero-length segment with positive head_size must yield a triangle");
        let [tip, left, right] = points;
        assert_eq!(tip, (100.0, 0.0), "tip sits exactly at (x2, y2)");
        // Direction is +x, so the perpendicular is +/-y; base sits at x=92.
        assert!((left.0 - 92.0).abs() < 1e-9);
        assert!((left.1 - 4.0).abs() < 1e-9);
        assert!((right.0 - 92.0).abs() < 1e-9);
        assert!((right.1 + 4.0).abs() < 1e-9);
    }

    /// `head_size` is clamped to the segment's own pixel length — an
    /// epsilon-length shaft cannot produce a head that overshoots its start
    /// point. Mirrors `emit_arrow_clamps_head_size_to_epsilon_shaft` in
    /// `annotation.rs`, which pins the same clamp through the caller that
    /// used to own this math inline.
    #[test]
    fn arrowhead_triangle_clamps_head_size_to_shaft_length() {
        let points = arrowhead_triangle(100.0, 100.0, 100.001, 100.0, 8.0)
            .expect("finite non-zero shaft must yield a triangle");
        for (x, _y) in points {
            assert!(
                x >= 100.0 - 1e-9,
                "clamped arrowhead must not extend past the shaft start; got x={x}"
            );
        }
    }

    /// A zero-length segment has no direction to point in — `None`, not a
    /// triangle with NaN coordinates from a divide-by-zero.
    #[test]
    fn arrowhead_triangle_zero_length_segment_returns_none() {
        assert_eq!(arrowhead_triangle(5.0, 5.0, 5.0, 5.0, 8.0), None);
    }

    /// `head_size <= 0.0` means "no head" — `None`, matching `emit_arrow`'s
    /// original `if head_size > 0.0` guard.
    #[test]
    fn arrowhead_triangle_non_positive_head_size_returns_none() {
        assert_eq!(arrowhead_triangle(0.0, 0.0, 10.0, 0.0, 0.0), None);
        assert_eq!(arrowhead_triangle(0.0, 0.0, 10.0, 0.0, -1.0), None);
    }

    /// `arrowhead_path` builds a closed, filled, unstroked triangle path from
    /// three vertices — the exact `FillStroke` shape `emit_arrow` built
    /// inline before this extraction.
    #[test]
    fn arrowhead_path_builds_closed_filled_triangle() {
        let points = arrowhead_triangle(0.0, 0.0, 10.0, 0.0, 4.0).unwrap();
        let color = ferrum_scene::Color { r: 0x12, g: 0x34, b: 0x56, a: 0xff };
        let node = arrowhead_path(points, color);
        let ferrum_scene::SceneNode::Path { commands, style, closed } = node else {
            panic!("expected Path node");
        };
        assert!(closed);
        assert_eq!(commands.len(), 4, "MoveTo + 2×LineTo + Close");
        assert!(matches!(commands[0], ferrum_scene::PathCmd::MoveTo { .. }));
        assert!(matches!(commands[3], ferrum_scene::PathCmd::Close));
        assert_eq!(style.fill, Some(color));
        assert_eq!(style.stroke, None);
        assert_eq!(style.stroke_width, 0.0);
        assert_eq!(style.opacity, 1.0);
    }
}
