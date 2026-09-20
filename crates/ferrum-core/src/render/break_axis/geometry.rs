//! Polygon and path clipping — the `self`-free geometry
//! [`BreakRemap`](super::BreakRemap) calls to cut a ring, a stroke or a
//! rounded rect at a gap edge, and to read vertices back out of a finished
//! path.
//!
//! Split out of `break_axis/mod.rs` (#229). Every function here already took
//! no `&self`, which is what makes this the seam the code supports: the
//! issue's proposed `clip.rs` would have cut `impl BreakRemap` in half across
//! two modules, forcing six of its private fields to `pub(super)`.
//!
//! Two private types stay in `mod.rs` rather than travelling here with their
//! consumers: `CutEdges`, whose four fields `BreakRemap::clip_node_into`
//! builds directly and whose "exactly two of four are ever set" invariant its
//! own methods depend on, and `PolylineSplit`, which
//! `BreakRemap::split_polyline` both builds and reads. A child module can see
//! its parent's private items, so this file names both without widening
//! either. Precedent for the split: `render/scale_resolve/`, `projection/`,
//! `transform/stats/`, `transform/smooth/`.

use ferrum_scene::{FillStroke, PathCmd, SceneNode};

use super::CutEdges;

/// The three vertices of an arrowhead triangle node, tip first — the shape
/// `marks::arrowhead::arrowhead_path_with_opacity` emits (`MoveTo` tip,
/// `LineTo` left, `LineTo` right, `Close`).
///
/// `None` for any other closed path, which routes the node to the per-vertex
/// remap instead of the arrowhead rebuild.
pub(super) fn triangle_vertices(node: &SceneNode) -> Option<[(f64, f64); 3]> {
    let SceneNode::Path { commands, .. } = node else {
        return None;
    };
    let points = polygon_points(commands)?;
    match points[..] {
        [tip, left, right] => Some([tip, left, right]),
        _ => None,
    }
}

/// Overwrite an arrowhead triangle node's three vertices in place, keeping
/// its style and command shape. The inverse of [`triangle_vertices`].
pub(super) fn set_triangle_vertices(node: &mut SceneNode, points: [(f64, f64); 3]) {
    let SceneNode::Path { commands, .. } = node else {
        return;
    };
    let mut vertices = points.iter();
    for cmd in commands.iter_mut() {
        if let PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } = cmd {
            if let Some(&(nx, ny)) = vertices.next() {
                *x = nx;
                *y = ny;
            }
        }
    }
}

/// Push one `Line` node per piece of a split two-vertex stroke.
///
/// A two-vertex source is one edge, and one edge contributes exactly two points
/// to each piece it reaches (its entry and its exit), so every piece here is a
/// straight `Line` — the node kind the source had. Shared by the plain split and
/// the segment-batch split so a shaft and a rule cannot be re-spelled
/// differently.
///
/// That two-point invariant is ENFORCED rather than assumed: a `Line` is the one
/// node kind with no way to spell an interior vertex, so a three-point piece
/// arriving here could only be rendered by discarding its middle — a silently
/// wrong stroke. The `debug_assert` costs nothing in release and turns a future
/// caller handing in a longer polyline into a test failure instead.
pub(super) fn push_line_pieces(
    pieces: &[Vec<(f64, f64)>],
    style: &ferrum_scene::StrokeStyle,
    out: &mut Vec<SceneNode>,
) {
    for piece in pieces {
        debug_assert_eq!(
            piece.len(),
            2,
            "a `Line` piece is one edge's entry and exit; got {piece:?}"
        );
        let (a, b) = (piece[0], piece[piece.len() - 1]);
        out.push(SceneNode::Line {
            x1: a.0,
            y1: a.1,
            x2: b.0,
            y2: b.1,
            style: style.clone(),
        });
    }
}

/// Spell `points` as a path: a leading `MoveTo`, then `LineTo` per vertex, then
/// `Close` when the path is closed.
///
/// The inverse of [`polygon_points`]/[`straight_path_points`], and the one
/// spelling both rewrites emit — the clipped area ring and the split step line.
pub(super) fn path_commands(points: &[(f64, f64)], closed: bool) -> Vec<PathCmd> {
    let mut commands = Vec::with_capacity(points.len() + usize::from(closed));
    for (i, &(x, y)) in points.iter().enumerate() {
        commands.push(if i == 0 {
            PathCmd::MoveTo { x, y }
        } else {
            PathCmd::LineTo { x, y }
        });
    }
    if closed {
        commands.push(PathCmd::Close);
    }
    commands
}

/// Build one clipped rect piece's scene node — a `Path` with arcs on only
/// the UNCUT corners when a cut edge would otherwise round a corner the gap
/// itself produced, a plain `Rect` in every other case (#230-2).
///
/// A piece stays a `Rect` rather than converting to a `Path` in THREE cases,
/// each because the conversion would either add no information or would
/// actively cost the piece something only `Rect`-ness carries (SVG
/// centre-anchored rotation, WASM hit-testing, R-tree membership, batch
/// packing eligibility):
///
/// - **No cut edge, or no radius.** The common case — an unbroken chart, or
///   a break whose gaps miss the piece entirely — and byte-identical to what
///   this code emitted before #230-2.
/// - **Every surviving corner is squared** (quality-review finding, cycle
///   3): a piece cut on both edges of one axis (the middle slice of a
///   two-gap break) has no corner left to round, so its outline is already a
///   plain four-cornered box. It emits `Rect { corner_radius: 0.0 }` rather
///   than a `Path` that would spell the identical geometry while forfeiting
///   `Rect`-only behavior for no visual gain.
/// - **`style.angle != 0.0`** (quality-review finding, cycle 3):
///   `SceneNode::Rect` rotates about its own centre
///   (`svg_walk.rs::emit_node`'s `Rect` arm), while `SceneNode::Path` rotates
///   about its first `MoveTo` — a converted piece would silently jump to a
///   different rotation anchor than its unbroken rounded sibling (hundreds
///   of pixels, measured), and lose the rotation entirely in WASM, which
///   never reads `style.angle` for a `Path`. A rotated piece keeps the plain
///   `Rect`, uncut radius and all, rather than corrupt its anchor: the
///   un-squared cut edge in that specific rotated+rounded+broken three-way
///   combination is a documented residual (`ferrum-spec.md`'s #230-2 dated
///   note), not a silent regression.
pub(super) fn rounded_rect_piece(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    style: FillStroke,
    corner_radius: f64,
    cut: CutEdges,
) -> SceneNode {
    if corner_radius <= 0.0 || !cut.any() || style.angle != 0.0 {
        return SceneNode::Rect { x, y, w, h, style, corner_radius };
    }
    if cut.corners().iter().all(|&squared| squared) {
        return SceneNode::Rect { x, y, w, h, style, corner_radius: 0.0 };
    }
    SceneNode::Path {
        commands: square_cut_rounded_rect_path(x, y, w, h, corner_radius, cut),
        style,
        closed: true,
    }
}

/// A rounded rect's outline, clockwise from the top edge, with the radius
/// zeroed on every corner `cut` touches — a corner squares whenever EITHER
/// of its two adjacent edges is cut. Never called with all four corners
/// squared; [`rounded_rect_piece`] takes the plain-`Rect` branch for that.
///
/// `radius` is clamped per axis, matching SVG `<rect rx ry>` semantics
/// exactly: `rx` clamps to `w / 2` and `ry` clamps to `h / 2`, INDEPENDENTLY
/// of each other. An SVG `<rect>` gets that clamp for free from the
/// rendering rules, but a hand-built arc path does not, so a piece shorter
/// than the requested radius on only one axis (a thin sliver left after a
/// gap eats most of a bar) needs it spelled out here to match what the
/// plain-`Rect` branch would have rendered for the same dimensions — a
/// single shared scalar clamped by `min(w/2, h/2)` would round the uncut
/// corners of a narrow-but-tall piece more tightly than an unsplit rect of
/// the same size does, since `rx` and `ry` never actually collapse to one
/// number unless `w == h`.
fn square_cut_rounded_rect_path(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    radius: f64,
    cut: CutEdges,
) -> Vec<PathCmd> {
    let radius = radius.max(0.0);
    let rx = radius.min(w / 2.0);
    let ry = radius.min(h / 2.0);
    let [squared_tl, squared_tr, squared_br, squared_bl] = cut.corners();
    let corner = |squared: bool| -> (f64, f64) { if squared { (0.0, 0.0) } else { (rx, ry) } };
    let (rx_tl, ry_tl) = corner(squared_tl);
    let (rx_tr, ry_tr) = corner(squared_tr);
    let (rx_br, ry_br) = corner(squared_br);
    let (rx_bl, ry_bl) = corner(squared_bl);

    let mut cmds = Vec::with_capacity(9);
    cmds.push(PathCmd::MoveTo { x: x + rx_tl, y });
    cmds.push(PathCmd::LineTo { x: x + w - rx_tr, y });
    if rx_tr > 0.0 || ry_tr > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_tr,
            ry: ry_tr,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: x + w,
            y: y + ry_tr,
        });
    }
    cmds.push(PathCmd::LineTo { x: x + w, y: y + h - ry_br });
    if rx_br > 0.0 || ry_br > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_br,
            ry: ry_br,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: x + w - rx_br,
            y: y + h,
        });
    }
    cmds.push(PathCmd::LineTo { x: x + rx_bl, y: y + h });
    if rx_bl > 0.0 || ry_bl > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_bl,
            ry: ry_bl,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x,
            y: y + h - ry_bl,
        });
    }
    cmds.push(PathCmd::LineTo { x, y: y + ry_tl });
    if rx_tl > 0.0 || ry_tl > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_tl,
            ry: ry_tl,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: x + rx_tl,
            y,
        });
    }
    cmds.push(PathCmd::Close);
    cmds
}

/// The vertices of an OPEN path whose every command draws a straight line —
/// `LineTo` plus the `HLineTo`/`VLineTo` shorthand `mark_line`'s three `step*`
/// interpolations are written with.
///
/// `None` for anything a break cannot cut faithfully: a curve command
/// (`QuadTo`/`CubicTo`/`ArcTo`), a second subpath, or a `Close` (a closed ring
/// belongs to [`polygon_points`] and the polygon clip). That is the documented
/// split-vs-fallback boundary of [`BreakRemap::split_nodes`].
///
/// [`BreakRemap::split_nodes`]: super::BreakRemap::split_nodes
pub(super) fn straight_path_points(commands: &[PathCmd]) -> Option<Vec<(f64, f64)>> {
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(commands.len());
    for (i, cmd) in commands.iter().enumerate() {
        let (x, y) = match (i, cmd) {
            (0, PathCmd::MoveTo { x, y }) => (*x, *y),
            (_, PathCmd::LineTo { x, y }) => (*x, *y),
            // The shorthand carries only the coordinate that moves; the other
            // one is the current point's, which is the previous vertex.
            (_, PathCmd::HLineTo { x }) => (*x, points.last()?.1),
            (_, PathCmd::VLineTo { y }) => (points.last()?.0, *y),
            _ => return None,
        };
        points.push((x, y));
    }
    (points.len() >= 2).then_some(points)
}

/// The vertices of a path built only from `MoveTo`/`LineTo` (plus an optional
/// trailing `Close`) — the shape every area fill emits under the linear and
/// step interpolations.
///
/// `None` for anything else (curve commands, multiple subpaths), which routes
/// the node to the per-vertex remap instead of the polygon clip.
pub(super) fn polygon_points(commands: &[PathCmd]) -> Option<Vec<(f64, f64)>> {
    let mut points = Vec::with_capacity(commands.len());
    for (i, cmd) in commands.iter().enumerate() {
        match cmd {
            PathCmd::MoveTo { x, y } if i == 0 => points.push((*x, *y)),
            PathCmd::LineTo { x, y } => points.push((*x, *y)),
            PathCmd::Close if i + 1 == commands.len() => {}
            _ => return None,
        }
    }
    (points.len() >= 3).then_some(points)
}

/// Which side of a boundary a half-plane clip keeps.
#[derive(Clone, Copy)]
enum Side {
    AtLeast,
    AtMost,
}

/// Clip a polygon of `(along, across)` points to `lo <= along <= hi`.
///
/// Sutherland–Hodgman against the two boundaries; `across` is interpolated
/// linearly at each crossing. Returns fewer than three points when nothing
/// survives.
pub(super) fn clip_to_band(points: &[(f64, f64)], lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let lower = clip_half_plane(points, lo, Side::AtLeast);
    if lower.len() < 3 {
        return Vec::new();
    }
    clip_half_plane(&lower, hi, Side::AtMost)
}

fn clip_half_plane(points: &[(f64, f64)], bound: f64, side: Side) -> Vec<(f64, f64)> {
    let inside = |along: f64| match side {
        Side::AtLeast => along >= bound,
        Side::AtMost => along <= bound,
    };
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(points.len() + 2);
    for (i, &cur) in points.iter().enumerate() {
        let prev = points[(i + points.len() - 1) % points.len()];
        let (cur_in, prev_in) = (inside(cur.0), inside(prev.0));
        if cur_in != prev_in {
            let denom = cur.0 - prev.0;
            let t = if denom.abs() < f64::EPSILON {
                0.0
            } else {
                (bound - prev.0) / denom
            };
            out.push((bound, prev.1 + t * (cur.1 - prev.1)));
        }
        if cur_in {
            out.push(cur);
        }
    }
    out
}
