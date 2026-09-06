//! Curve interpolation shared by the path-shaped marks — `mark_line`,
//! `mark_area`, and `mark_ribbon` (spec §6.1, batch E / F-L02-02 + F-L02-03).
//!
//! Before this module the three marks disagreed about what `interpolate=`
//! meant: line and area each carried their own step implementation (with
//! divergent — though geometrically identical — command spellings) and treated
//! every non-step value as `linear`, so the documented `monotone` / `basis` /
//! `cardinal` values rendered straight segments; ribbon read the setting at all
//! only after this batch. All three now trace their edges through a [`Trace`],
//! so the vocabulary, the curve math, and the refusal for an unknown value
//! exist once.
//!
//! Curve semantics follow d3 (spec §6.1):
//! - `monotone` — Fritsch–Carlson monotone cubic: passes through every point
//!   and never overshoots between them. This is d3's *guarantee* but not d3's
//!   arithmetic: `d3.curveMonotoneX` uses the Steffen `slope3`/`slope2`
//!   tangent spelling, which picks different (also monotone-preserving)
//!   tangents, so control-point coordinates are NOT byte-equal to d3's. Do not
//!   treat parity with d3 here as a contract; the monotonicity guarantee is
//!   the contract.
//! - `basis` — uniform cubic B-spline (`d3.curveBasis`): approximating, so it
//!   does NOT pass through the interior points. Exact port of d3's state
//!   machine.
//! - `cardinal` — Catmull–Rom at tension 0 (`d3.curveCardinal`): passes through
//!   every point, may overshoot. Exact port of d3's state machine.
//!
//! All three emit [`PathCmd::CubicTo`], which both walkers already consume
//! (`svg_walk::path_cmds_to_d` → `C`, `ferrum-wasm`'s `pathcmds_to_lyon` →
//! `cubic_bezier_to`). Fewer than three points carry no curvature, so they
//! degrade to `linear` rather than to a degenerate control polygon.
//!
//! Two properties of an edge decide its geometry beyond the method itself, and
//! both live on [`Trace`] because getting either wrong is silent:
//! - **Which coordinate is independent** ([`CurveAxis`]). The math fits the
//!   dependent coordinate as a function of the independent one; a horizontal
//!   (value-on-x) area is parameterized by *y*, and fitting it against x
//!   produces a shape that is not a function at all. d3 splits this into
//!   `curveMonotoneX` / `curveMonotoneY`; this module carries one
//!   implementation and projects through the axis instead.
//! - **Which direction it is walked** ([`Trace::extend_reversed`]). A filled
//!   band closes its ring by walking the bottom edge backwards, and
//!   `step-before`/`step-after` are direction-dependent — each becomes the
//!   other under reversal.
//!
//! Every routine here is pure f64 arithmetic with no randomness, so output is
//! byte-deterministic.

use ferrum_scene::PathCmd;

use crate::render::RenderError;

/// The `interpolate=` vocabulary (spec §6.1).
///
/// [`Interpolate::ALL`] is the single source for both dispatch and the
/// [`RenderError::UnknownInterpolate`] message: a value the enum cannot name
/// cannot be dispatched, and the refusal lists exactly what dispatch accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Interpolate {
    #[default]
    Linear,
    Step,
    StepBefore,
    StepAfter,
    Monotone,
    Basis,
    Cardinal,
}

impl Interpolate {
    /// Every accepted method, in the order the refusal message lists them.
    pub(crate) const ALL: [Self; 7] = [
        Self::Linear,
        Self::Step,
        Self::StepBefore,
        Self::StepAfter,
        Self::Monotone,
        Self::Basis,
        Self::Cardinal,
    ];

    /// The user-facing spelling — the string accepted by `interpolate=`.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Step => "step",
            Self::StepBefore => "step-before",
            Self::StepAfter => "step-after",
            Self::Monotone => "monotone",
            Self::Basis => "basis",
            Self::Cardinal => "cardinal",
        }
    }

    /// Resolve a user-supplied `interpolate=` value, or refuse it.
    ///
    /// Called once per mark from [`crate::render::draw::resolve_mark_style`],
    /// so an unknown value is refused before any geometry is built and the
    /// drawers downstream hold a value that is valid by construction — there
    /// is no "unrecognized method quietly renders straight" path left.
    pub(crate) fn parse(value: &str) -> Result<Self, RenderError> {
        Self::ALL
            .iter()
            .copied()
            .find(|method| method.as_str() == value)
            .ok_or_else(|| RenderError::UnknownInterpolate { value: value.to_string() })
    }

    /// `"linear, step, step-before, step-after, monotone, basis, cardinal"` —
    /// the accepted set as it appears in the refusal message.
    pub(crate) fn accepted_values() -> String {
        Self::ALL
            .iter()
            .map(|method| method.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Whether this method traces straight segments between consecutive
    /// points. `mark_line` uses this to choose its scene node: a straight
    /// series stays a `Polyline`, anything else needs a `Path`.
    pub(crate) fn is_linear(self) -> bool {
        matches!(self, Self::Linear)
    }

    /// The method that traces the SAME function of the independent coordinate
    /// when the point sequence is walked backwards.
    ///
    /// Only the two one-sided steps change. `step-before` holds each interval
    /// at its right-hand point's value; walked backwards that same emission
    /// rule holds the interval at its *left*-hand point's value, which is what
    /// `step-after` means — so the two swap. `step` (midpoint riser),
    /// `linear`, and the three curves are all reversal-symmetric.
    ///
    /// d3 makes this same compensation for its area generator's second edge:
    /// `Step.lineEnd` flips `_t` between `0` and `1` (`d3-shape/src/curve/step.js`).
    fn mirrored(self) -> Self {
        match self {
            Self::StepBefore => Self::StepAfter,
            Self::StepAfter => Self::StepBefore,
            other => other,
        }
    }
}

/// Which screen coordinate a traced series is parameterized by — the
/// independent variable the dependent one is fitted against.
///
/// A vertical area/line advances along x and its value rides y ([`Self::X`]).
/// A horizontal one — `Stack(value_axis=X)`, where the cumulated value lands
/// on x and the category runs down y — advances along y ([`Self::Y`]).
/// Fitting the wrong way round is silent: `push_monotone` would divide by a
/// *value* difference rather than a position difference, and the emitted
/// "curve" would not be a function of the axis it is drawn against.
///
/// The geometry below is written once, in `(u, v)` = (independent, dependent)
/// space; this enum is the only thing that knows which screen coordinate is
/// which. It is d3's `ReflectContext` trick (how `curveMonotoneY` reuses
/// `curveMonotoneX`) applied at emission instead of at the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CurveAxis {
    /// Independent coordinate is x — every vertical line/area/ribbon.
    X,
    /// Independent coordinate is y — a horizontal (value-on-x) area.
    Y,
}

impl CurveAxis {
    /// Screen `(x, y)` → `(independent, dependent)`.
    fn to_uv(self, (x, y): (f64, f64)) -> (f64, f64) {
        match self {
            Self::X => (x, y),
            Self::Y => (y, x),
        }
    }

    /// `(independent, dependent)` → screen `(x, y)`.
    fn to_xy(self, (u, v): (f64, f64)) -> (f64, f64) {
        match self {
            Self::X => (u, v),
            Self::Y => (v, u),
        }
    }

    fn move_to(self, p: (f64, f64)) -> PathCmd {
        let (x, y) = self.to_xy(p);
        PathCmd::MoveTo { x, y }
    }

    fn line_to(self, p: (f64, f64)) -> PathCmd {
        let (x, y) = self.to_xy(p);
        PathCmd::LineTo { x, y }
    }

    /// Shorthand command advancing only the INDEPENDENT coordinate — a step's
    /// tread.
    fn tread_to(self, u: f64) -> PathCmd {
        match self {
            Self::X => PathCmd::HLineTo { x: u },
            Self::Y => PathCmd::VLineTo { y: u },
        }
    }

    /// Shorthand command advancing only the DEPENDENT coordinate — a step's
    /// riser.
    fn riser_to(self, v: f64) -> PathCmd {
        match self {
            Self::X => PathCmd::VLineTo { y: v },
            Self::Y => PathCmd::HLineTo { x: v },
        }
    }

    fn cubic_to(self, c1: (f64, f64), c2: (f64, f64), end: (f64, f64)) -> PathCmd {
        let (c1x, c1y) = self.to_xy(c1);
        let (c2x, c2y) = self.to_xy(c2);
        let (x, y) = self.to_xy(end);
        PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y }
    }
}

/// How step segments are spelled as [`PathCmd`]s.
///
/// The two spellings describe the SAME geometry but serialize to different
/// bytes (`H`/`V` shorthand versus explicit `L` vertices) in both the SVG `d`
/// attribute and the scene JSON, so each consumer keeps the spelling its
/// committed goldens were blessed with. That is the only reason this parameter
/// exists; a future golden re-bless could collapse it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepSpelling {
    /// `HLineTo`/`VLineTo` shorthand — `mark_line`'s spelling.
    Shorthand,
    /// Explicit `LineTo` vertices — `mark_area`'s and `mark_ribbon`'s spelling.
    Vertices,
}

/// Everything a consumer must decide before an edge can be traced: the
/// interpolation method plus the two conventions its geometry depends on.
///
/// Bundled rather than passed as loose arguments because [`axis`](Self::axis)
/// and traversal direction are both silent when wrong — a caller that forgets
/// one gets a plausible-looking path describing the wrong function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Trace {
    /// The user's `interpolate=` method.
    pub(crate) method: Interpolate,
    /// The consumer's step command spelling (byte-compatibility only).
    pub(crate) step: StepSpelling,
    /// Which screen coordinate the series is parameterized by.
    pub(crate) axis: CurveAxis,
}

impl Trace {
    /// Trace `points` as a NEW subpath: a `MoveTo` at `points[0]` followed by
    /// the segments this trace's method prescribes.
    pub(crate) fn subpath(self, points: &[(f64, f64)]) -> Vec<PathCmd> {
        let mut cmds = Vec::with_capacity(points.len() * 3 + 1);
        self.push(&mut cmds, points, Lead::Move);
        cmds
    }

    /// Continue an already-open subpath FORWARD into `points`: a `LineTo` to
    /// `points[0]` followed by the same segments [`subpath`](Self::subpath)
    /// would emit.
    #[cfg(test)]
    pub(crate) fn extend(self, cmds: &mut Vec<PathCmd>, points: &[(f64, f64)]) {
        self.push(cmds, points, Lead::Line);
    }

    /// Continue an already-open subpath BACKWARDS along `forward_points`,
    /// tracing the same function of the independent coordinate that
    /// [`subpath`](Self::subpath) would trace forwards.
    ///
    /// This is how a filled band closes its ring: `mark_area`'s stacked/`y2`
    /// fill and `mark_ribbon` walk the bottom edge from last point to first.
    /// Callers pass the bottom edge in FORWARD order and this owns both the
    /// reversal and the [`Interpolate::mirrored`] compensation, so no consumer
    /// can reverse the points and forget the method swap — the defect this
    /// signature exists to make unrepresentable.
    pub(crate) fn extend_reversed(self, cmds: &mut Vec<PathCmd>, forward_points: &[(f64, f64)]) {
        let reversed: Vec<(f64, f64)> = forward_points.iter().rev().copied().collect();
        let mirrored = Self { method: self.method.mirrored(), ..self };
        mirrored.push(cmds, &reversed, Lead::Line);
    }

    /// Emit the lead command reaching `points[0]`, then the segments.
    fn push(self, cmds: &mut Vec<PathCmd>, points: &[(f64, f64)], lead: Lead) {
        let uv: Vec<(f64, f64)> = points.iter().map(|&p| self.axis.to_uv(p)).collect();
        let Some(&first) = uv.first() else { return };
        cmds.push(match lead {
            Lead::Move => self.axis.move_to(first),
            Lead::Line => self.axis.line_to(first),
        });
        self.push_segments(cmds, &uv);
    }
}

/// How an edge joins what precedes it: a new subpath, or a continuation of the
/// ring already being built.
#[derive(Debug, Clone, Copy)]
enum Lead {
    Move,
    Line,
}

impl Trace {
    /// Append the segments joining `uv[0]` to `uv[len - 1]`, in
    /// `(independent, dependent)` space. The command reaching `uv[0]` has
    /// already been emitted by the caller.
    fn push_segments(self, cmds: &mut Vec<PathCmd>, uv: &[(f64, f64)]) {
        let (points, step, axis) = (uv, self.step, self.axis);
        let method = self.method;
    // A curve needs a neighbourhood to shape itself against: with two points
    // there is exactly one segment and no curvature to describe, so the three
    // curve methods degrade to `linear` (spec §6.1). d3 degrades identically.
    let method = match method {
        Interpolate::Monotone | Interpolate::Basis | Interpolate::Cardinal
            if points.len() < 3 =>
        {
            Interpolate::Linear
        }
        other => other,
    };

        match method {
            Interpolate::Linear => {
                for &p in &points[1..] {
                    cmds.push(axis.line_to(p));
                }
            }
            // The three step methods differ only in WHERE the riser sits: at
            // the segment midpoint, at its start, or at its end.
            Interpolate::Step => {
                for window in points.windows(2) {
                    let ((pu, pv), (cu, cv)) = (window[0], window[1]);
                    let mid_u = (pu + cu) / 2.0;
                    match step {
                        StepSpelling::Shorthand => {
                            cmds.push(axis.tread_to(mid_u));
                            cmds.push(axis.riser_to(cv));
                            cmds.push(axis.tread_to(cu));
                        }
                        StepSpelling::Vertices => {
                            cmds.push(axis.line_to((mid_u, pv)));
                            cmds.push(axis.line_to((mid_u, cv)));
                            cmds.push(axis.line_to((cu, cv)));
                        }
                    }
                }
            }
            Interpolate::StepBefore => {
                for window in points.windows(2) {
                    let ((pu, _), (cu, cv)) = (window[0], window[1]);
                    match step {
                        StepSpelling::Shorthand => {
                            cmds.push(axis.riser_to(cv));
                            cmds.push(axis.tread_to(cu));
                        }
                        StepSpelling::Vertices => {
                            cmds.push(axis.line_to((pu, cv)));
                            cmds.push(axis.line_to((cu, cv)));
                        }
                    }
                }
            }
            Interpolate::StepAfter => {
                for window in points.windows(2) {
                    let ((_, pv), (cu, cv)) = (window[0], window[1]);
                    match step {
                        StepSpelling::Shorthand => {
                            cmds.push(axis.tread_to(cu));
                            cmds.push(axis.riser_to(cv));
                        }
                        StepSpelling::Vertices => {
                            cmds.push(axis.line_to((cu, pv)));
                            cmds.push(axis.line_to((cu, cv)));
                        }
                    }
                }
            }
            Interpolate::Monotone => push_monotone(cmds, points, axis),
            Interpolate::Basis => push_basis(cmds, points, axis),
            Interpolate::Cardinal => push_cardinal(cmds, points, axis),
        }
    }
}

/// Fritsch–Carlson monotone cubic over `(u, v)`, `points.len() >= 3`.
///
/// Tangents start as the averaged neighbouring secant slopes, are zeroed at
/// every local extremum, and are then clamped into the Fritsch–Carlson
/// monotonicity region so the curve cannot overshoot the data it passes
/// through. The Hermite tangents convert to Bézier control points at a third
/// of each interval's width.
///
/// The monotonicity *guarantee* is d3's; the tangent arithmetic is not — see
/// the module header. A repeated independent coordinate gives a zero-width
/// interval; its secant is defined as `0` rather than dividing, which collapses
/// that segment's control points onto its endpoints (a straight join) instead
/// of producing NaN coordinates.
fn push_monotone(cmds: &mut Vec<PathCmd>, points: &[(f64, f64)], axis: CurveAxis) {
    let n = points.len();

    // Interval widths and secant slopes.
    let mut widths = vec![0.0_f64; n - 1];
    let mut secants = vec![0.0_f64; n - 1];
    for i in 0..n - 1 {
        let h = points[i + 1].0 - points[i].0;
        widths[i] = h;
        secants[i] = if h == 0.0 {
            0.0
        } else {
            (points[i + 1].1 - points[i].1) / h
        };
    }

    // Initial tangents: one-sided at the ends, averaged inside, zero wherever
    // the data turns (opposite-signed neighbouring secants).
    let mut tangents = vec![0.0_f64; n];
    tangents[0] = secants[0];
    tangents[n - 1] = secants[n - 2];
    for i in 1..n - 1 {
        tangents[i] = if secants[i - 1] * secants[i] <= 0.0 {
            0.0
        } else {
            (secants[i - 1] + secants[i]) / 2.0
        };
    }

    // Fritsch–Carlson clamp: keep each interval's (α, β) inside the circle of
    // radius 3, the sufficient condition for monotone cubic Hermite output.
    for i in 0..n - 1 {
        if secants[i] == 0.0 {
            tangents[i] = 0.0;
            tangents[i + 1] = 0.0;
            continue;
        }
        let alpha = tangents[i] / secants[i];
        let beta = tangents[i + 1] / secants[i];
        let radius_sq = alpha * alpha + beta * beta;
        if radius_sq > 9.0 {
            let scale = 3.0 / radius_sq.sqrt();
            tangents[i] = scale * alpha * secants[i];
            tangents[i + 1] = scale * beta * secants[i];
        }
    }

    for i in 0..n - 1 {
        let (u0, v0) = points[i];
        let (u1, v1) = points[i + 1];
        let third = widths[i] / 3.0;
        cmds.push(axis.cubic_to(
            (u0 + third, v0 + tangents[i] * third),
            (u1 - third, v1 - tangents[i + 1] * third),
            (u1, v1),
        ));
    }
}

/// Uniform cubic B-spline (`d3.curveBasis`), `points.len() >= 3`.
///
/// The spline is approximating, so the drawn path starts at `points[0]`, runs
/// straight to where the spline hull begins, follows the B-spline through the
/// interior, and finishes with the mirrored straight join back to the last
/// point — d3's exact spelling, kept so `basis` output matches the reference
/// implementation the docs point users at.
fn push_basis(cmds: &mut Vec<PathCmd>, points: &[(f64, f64)], axis: CurveAxis) {
    let n = points.len();
    let (first, second) = (points[0], points[1]);
    cmds.push(axis.line_to((
        (5.0 * first.0 + second.0) / 6.0,
        (5.0 * first.1 + second.1) / 6.0,
    )));
    for i in 2..n {
        push_basis_segment(cmds, points[i - 2], points[i - 1], points[i], axis);
    }
    // Closing segment: the last point is repeated as the spline's trailing
    // control point, then joined straight.
    push_basis_segment(cmds, points[n - 2], points[n - 1], points[n - 1], axis);
    cmds.push(axis.line_to(points[n - 1]));
}

/// One B-spline span over the control triple `(a, b, c)`.
fn push_basis_segment(
    cmds: &mut Vec<PathCmd>,
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    axis: CurveAxis,
) {
    cmds.push(axis.cubic_to(
        ((2.0 * a.0 + b.0) / 3.0, (2.0 * a.1 + b.1) / 3.0),
        ((a.0 + 2.0 * b.0) / 3.0, (a.1 + 2.0 * b.1) / 3.0),
        (
            (a.0 + 4.0 * b.0 + c.0) / 6.0,
            (a.1 + 4.0 * b.1 + c.1) / 6.0,
        ),
    ));
}

/// Catmull–Rom tangent scale at tension 0: `(1 - tension) / 6`.
const CARDINAL_K: f64 = 1.0 / 6.0;

/// Catmull–Rom at tension 0 (`d3.curveCardinal`), `points.len() >= 3`.
///
/// Each interior point's tangent is a sixth of the vector between its
/// neighbours. The two ends have no outside neighbour; d3 substitutes the
/// point ACROSS the end segment, which makes the end tangent vanish and the
/// terminal control point coincide with the endpoint. Mirrored here so
/// `cardinal` matches the reference implementation.
fn push_cardinal(cmds: &mut Vec<PathCmd>, points: &[(f64, f64)], axis: CurveAxis) {
    let n = points.len();
    for i in 0..n - 1 {
        let prev = if i == 0 { points[1] } else { points[i - 1] };
        let next = if i + 2 < n { points[i + 2] } else { points[n - 2] };
        let (u0, v0) = points[i];
        let (u1, v1) = points[i + 1];
        cmds.push(axis.cubic_to(
            (u0 + CARDINAL_K * (u1 - prev.0), v0 + CARDINAL_K * (v1 - prev.1)),
            (u1 + CARDINAL_K * (u0 - next.0), v1 + CARDINAL_K * (v0 - next.1)),
            (u1, v1),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An x-parameterized trace — what every consumer but a horizontal area
    /// uses. The `CurveAxis::Y` cases construct `Trace` directly.
    fn trace(method: Interpolate, step: StepSpelling) -> Trace {
        Trace { method, step, axis: CurveAxis::X }
    }


    /// Four points with a turning point at index 2, so monotone's overshoot
    /// clamp and cardinal's interior tangents both have something to act on.
    const SAMPLE: [(f64, f64); 4] = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0), (30.0, 15.0)];

    fn cubic_count(cmds: &[PathCmd]) -> usize {
        cmds.iter()
            .filter(|c| matches!(c, PathCmd::CubicTo { .. }))
            .count()
    }

    // ── Vocabulary and refusal ───────────────────────────────────────────

    #[test]
    fn every_accepted_value_round_trips_through_parse() {
        for method in Interpolate::ALL {
            assert_eq!(
                Interpolate::parse(method.as_str()).expect("ALL member must parse"),
                method,
                "`as_str` and `parse` must agree for {method:?}"
            );
        }
    }

    #[test]
    fn unknown_value_refuses_and_names_the_accepted_set() {
        let err = Interpolate::parse("cubic").expect_err("unknown value must refuse");
        assert_eq!(
            err,
            RenderError::UnknownInterpolate { value: "cubic".into() }
        );
        let message = err.to_string();
        assert!(
            message.contains("'cubic'"),
            "message must quote the offending value; got {message}"
        );
        for method in Interpolate::ALL {
            assert!(
                message.contains(method.as_str()),
                "message must list the accepted value {:?}; got {message}",
                method.as_str()
            );
        }
    }

    #[test]
    fn accepted_values_lists_the_whole_vocabulary_in_order() {
        assert_eq!(
            Interpolate::accepted_values(),
            "linear, step, step-before, step-after, monotone, basis, cardinal"
        );
    }

    #[test]
    fn default_is_linear() {
        assert_eq!(Interpolate::default(), Interpolate::Linear);
        assert!(Interpolate::Linear.is_linear());
        for method in Interpolate::ALL.iter().filter(|m| **m != Interpolate::Linear) {
            assert!(!method.is_linear(), "{method:?} must not report as linear");
        }
    }

    // ── Step spellings: byte-for-byte what line/area emitted pre-change ──
    //
    // Ported from `line.rs`'s `build_line_cmds_*` tests (themselves ported
    // from `bug_hunt_marks_rendering_r2.rs`) when the step logic moved here.

    #[test]
    fn step_shorthand_emits_hvh_triplet_per_segment() {
        let cmds = trace(Interpolate::Step, StepSpelling::Shorthand).subpath(&SAMPLE);
        assert_eq!(cmds.len(), 10, "MoveTo + 3 * (HLineTo, VLineTo, HLineTo)");
        assert!(matches!(cmds[0], PathCmd::MoveTo { .. }));
        for chunk in cmds[1..].chunks(3) {
            assert!(matches!(chunk[0], PathCmd::HLineTo { .. }));
            assert!(matches!(chunk[1], PathCmd::VLineTo { .. }));
            assert!(matches!(chunk[2], PathCmd::HLineTo { .. }));
        }
        assert_eq!(cmds[1], PathCmd::HLineTo { x: 5.0 }, "riser at the midpoint");
    }

    #[test]
    fn step_vertices_emits_three_linetos_per_segment_at_the_same_places() {
        let cmds = trace(Interpolate::Step, StepSpelling::Vertices).subpath(&SAMPLE);
        assert_eq!(cmds.len(), 10, "MoveTo + 3 * 3 LineTo");
        assert_eq!(
            cmds[1..4],
            [
                PathCmd::LineTo { x: 5.0, y: 0.0 },
                PathCmd::LineTo { x: 5.0, y: 10.0 },
                PathCmd::LineTo { x: 10.0, y: 10.0 },
            ],
            "the `Vertices` spelling names the same corners the shorthand implies"
        );
    }

    #[test]
    fn step_before_spellings_agree_on_geometry() {
        let shorthand = trace(Interpolate::StepBefore, StepSpelling::Shorthand).subpath(&SAMPLE);
        let vertices = trace(Interpolate::StepBefore, StepSpelling::Vertices).subpath(&SAMPLE);
        assert_eq!(shorthand.len(), 7, "MoveTo + 3 * (VLineTo, HLineTo)");
        assert_eq!(vertices.len(), 7, "MoveTo + 3 * 2 LineTo");
        assert_eq!(shorthand[1], PathCmd::VLineTo { y: 10.0 }, "riser first");
        assert_eq!(shorthand[2], PathCmd::HLineTo { x: 10.0 });
        assert_eq!(vertices[1], PathCmd::LineTo { x: 0.0, y: 10.0 });
        assert_eq!(vertices[2], PathCmd::LineTo { x: 10.0, y: 10.0 });
    }

    #[test]
    fn step_after_spellings_agree_on_geometry() {
        let shorthand = trace(Interpolate::StepAfter, StepSpelling::Shorthand).subpath(&SAMPLE);
        let vertices = trace(Interpolate::StepAfter, StepSpelling::Vertices).subpath(&SAMPLE);
        assert_eq!(shorthand.len(), 7, "MoveTo + 3 * (HLineTo, VLineTo)");
        assert_eq!(vertices.len(), 7, "MoveTo + 3 * 2 LineTo");
        assert_eq!(shorthand[1], PathCmd::HLineTo { x: 10.0 }, "tread first");
        assert_eq!(shorthand[2], PathCmd::VLineTo { y: 10.0 });
        assert_eq!(vertices[1], PathCmd::LineTo { x: 10.0, y: 0.0 });
        assert_eq!(vertices[2], PathCmd::LineTo { x: 10.0, y: 10.0 });
    }

    #[test]
    fn linear_emits_one_lineto_per_segment() {
        let cmds = trace(Interpolate::Linear, StepSpelling::Shorthand).subpath(&SAMPLE);
        assert_eq!(cmds.len(), 4, "MoveTo + 3 LineTo");
        assert!(matches!(cmds[0], PathCmd::MoveTo { .. }));
        assert!(cmds[1..].iter().all(|c| matches!(c, PathCmd::LineTo { .. })));
        assert_eq!(
            cmds,
            trace(Interpolate::Linear, StepSpelling::Vertices).subpath(&SAMPLE),
            "the step spelling is inert for a linear trace"
        );
    }

    // ── Curves emit CubicTo ──────────────────────────────────────────────

    #[test]
    fn every_curve_method_emits_cubics() {
        for method in [Interpolate::Monotone, Interpolate::Basis, Interpolate::Cardinal] {
            let cmds = trace(method, StepSpelling::Shorthand).subpath(&SAMPLE);
            assert!(
                cubic_count(&cmds) > 0,
                "{method:?} must emit CubicTo commands, not straight segments"
            );
            assert!(
                !cmds.iter().any(|c| matches!(
                    c,
                    PathCmd::HLineTo { .. } | PathCmd::VLineTo { .. }
                )),
                "{method:?} must not emit step commands"
            );
        }
    }

    #[test]
    fn monotone_emits_one_cubic_per_segment_through_every_point() {
        let cmds = trace(Interpolate::Monotone, StepSpelling::Shorthand).subpath(&SAMPLE);
        assert_eq!(cmds.len(), 4, "MoveTo + 3 CubicTo");
        assert_eq!(cubic_count(&cmds), 3);
        // Monotone interpolates: each cubic ENDS on the next data point.
        let endpoints: Vec<(f64, f64)> = cmds[1..]
            .iter()
            .map(|c| match c {
                PathCmd::CubicTo { x, y, .. } => (*x, *y),
                other => panic!("expected CubicTo, got {other:?}"),
            })
            .collect();
        assert_eq!(endpoints, SAMPLE[1..].to_vec());
    }

    #[test]
    fn monotone_does_not_overshoot_a_local_extremum() {
        // A plateau-then-rise: the classic overshoot case. With unclamped
        // cubic Hermite tangents the control points dip below the flat run.
        let points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 10.0)];
        let cmds = trace(Interpolate::Monotone, StepSpelling::Shorthand).subpath(&points);
        for cmd in &cmds[1..] {
            let PathCmd::CubicTo { c1y, c2y, .. } = cmd else {
                panic!("expected CubicTo, got {cmd:?}");
            };
            assert!(
                *c1y >= -1e-9 && *c2y >= -1e-9,
                "monotone must not dip below the flat run; got c1y={c1y}, c2y={c2y}"
            );
            assert!(
                *c1y <= 10.0 + 1e-9 && *c2y <= 10.0 + 1e-9,
                "monotone must not overshoot the rise; got c1y={c1y}, c2y={c2y}"
            );
        }
    }

    #[test]
    fn monotone_survives_duplicate_x_without_nan() {
        let points = [(0.0, 0.0), (10.0, 5.0), (10.0, 9.0), (20.0, 12.0)];
        let cmds = trace(Interpolate::Monotone, StepSpelling::Shorthand).subpath(&points);
        for cmd in &cmds {
            let coords: Vec<f64> = match cmd {
                PathCmd::MoveTo { x, y } => vec![*x, *y],
                PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
                    vec![*c1x, *c1y, *c2x, *c2y, *x, *y]
                }
                other => panic!("unexpected command {other:?}"),
            };
            assert!(
                coords.iter().all(|v| v.is_finite()),
                "a repeated x must not divide by zero; got {cmd:?}"
            );
        }
    }

    #[test]
    fn basis_is_approximating_not_interpolating() {
        let cmds = trace(Interpolate::Basis, StepSpelling::Shorthand).subpath(&SAMPLE);
        // MoveTo + leading LineTo + (n - 2 + 1) cubics + trailing LineTo.
        assert_eq!(cubic_count(&cmds), SAMPLE.len() - 1);
        assert_eq!(
            *cmds.last().expect("basis emits commands"),
            PathCmd::LineTo { x: 30.0, y: 15.0 },
            "the path still ENDS on the last data point"
        );
        // The interior point (10, 10) is a control point, not a vertex: no
        // emitted command lands on it.
        assert!(
            !cmds.iter().any(|c| matches!(
                c,
                PathCmd::CubicTo { x, y, .. } if (*x - 10.0).abs() < 1e-9 && (*y - 10.0).abs() < 1e-9
            )),
            "basis must not pass through interior points; got {cmds:?}"
        );
    }

    #[test]
    fn cardinal_interpolates_every_point_with_flat_end_tangents() {
        let cmds = trace(Interpolate::Cardinal, StepSpelling::Shorthand).subpath(&SAMPLE);
        assert_eq!(cmds.len(), 4, "MoveTo + 3 CubicTo");
        let PathCmd::CubicTo { c1x, c1y, .. } = cmds[1] else {
            panic!("expected CubicTo");
        };
        assert!(
            (c1x - SAMPLE[0].0).abs() < 1e-12 && (c1y - SAMPLE[0].1).abs() < 1e-12,
            "the leading control point coincides with the first point (d3 end handling)"
        );
        let PathCmd::CubicTo { c2x, c2y, x, y, .. } = cmds[3] else {
            panic!("expected CubicTo");
        };
        assert!(
            (c2x - SAMPLE[3].0).abs() < 1e-12 && (c2y - SAMPLE[3].1).abs() < 1e-12,
            "the trailing control point coincides with the last point"
        );
        assert_eq!((x, y), SAMPLE[3], "cardinal ends on the last data point");
    }

    #[test]
    fn cardinal_interior_tangent_is_a_sixth_of_the_neighbour_span() {
        let cmds = trace(Interpolate::Cardinal, StepSpelling::Shorthand).subpath(&SAMPLE);
        let PathCmd::CubicTo { c1x, c1y, .. } = cmds[2] else {
            panic!("expected CubicTo");
        };
        // Segment p1 -> p2: c1 = p1 + (p2 - p0) / 6.
        assert!((c1x - (10.0 + (20.0 - 0.0) / 6.0)).abs() < 1e-12, "got {c1x}");
        assert!((c1y - (10.0 + (5.0 - 0.0) / 6.0)).abs() < 1e-12, "got {c1y}");
    }

    // ── Degenerate point counts ──────────────────────────────────────────

    #[test]
    fn two_points_degrade_to_linear_for_every_curve_method() {
        let points = [(0.0, 0.0), (10.0, 4.0)];
        let linear = trace(Interpolate::Linear, StepSpelling::Shorthand).subpath(&points);
        for method in [Interpolate::Monotone, Interpolate::Basis, Interpolate::Cardinal] {
            assert_eq!(
                trace(method, StepSpelling::Shorthand).subpath(&points),
                linear,
                "{method:?} with 2 points must be byte-identical to linear"
            );
        }
    }

    #[test]
    fn one_point_emits_only_the_move_and_none_emits_nothing() {
        for method in Interpolate::ALL {
            assert_eq!(
                trace(method, StepSpelling::Shorthand).subpath(&[(3.0, 4.0)]),
                vec![PathCmd::MoveTo { x: 3.0, y: 4.0 }],
                "{method:?} with one point"
            );
            assert!(
                trace(method, StepSpelling::Shorthand).subpath(&[]).is_empty(),
                "{method:?} with no points"
            );
        }
    }

    // ── extend_subpath ───────────────────────────────────────────────────

    #[test]
    fn extend_subpath_joins_the_ring_with_a_lineto_then_traces_identically() {
        for method in Interpolate::ALL {
            let mut extended = vec![PathCmd::MoveTo { x: -1.0, y: -1.0 }];
            trace(method, StepSpelling::Vertices).extend(&mut extended, &SAMPLE);
            let fresh = trace(method, StepSpelling::Vertices).subpath(&SAMPLE);
            assert_eq!(
                extended[1],
                PathCmd::LineTo { x: SAMPLE[0].0, y: SAMPLE[0].1 },
                "{method:?}: the joined edge must NOT start a new subpath"
            );
            assert_eq!(
                extended[2..],
                fresh[1..],
                "{method:?}: everything after the join must match a fresh trace"
            );
        }
    }

    // ── extend_reversed: same function of the independent coordinate ─────

    /// Every vertex a reversed edge emits, read back-to-front, must equal the
    /// forward trace's vertices. This is the property a filled band needs: its
    /// two boundaries describe the same function of x.
    ///
    /// `step-before`/`step-after` are the cases that fail without
    /// [`Interpolate::mirrored`] — reversing the traversal turns each into the
    /// other. They are ALSO the cases a same-command-count assertion cannot
    /// catch, since the wrong shape has the same length as the right one.
    #[test]
    fn extend_reversed_traces_the_same_function_as_a_forward_trace() {
        fn vertices(cmds: &[PathCmd]) -> Vec<(f64, f64)> {
            cmds.iter()
                .filter_map(|c| match c {
                    PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => Some((*x, *y)),
                    PathCmd::CubicTo { x, y, .. } => Some((*x, *y)),
                    _ => None,
                })
                .collect()
        }

        for method in Interpolate::ALL {
            let t = trace(method, StepSpelling::Vertices);
            let mut ring = vec![PathCmd::MoveTo { x: -1.0, y: -1.0 }];
            t.extend_reversed(&mut ring, &SAMPLE);

            let mut walked_back = vertices(&ring[1..]);
            walked_back.reverse();
            let forward = vertices(&t.subpath(&SAMPLE));

            assert_eq!(
                walked_back, forward,
                "{method:?}: a reversed edge, read back-to-front, must land on the same \
                 vertices the forward trace does"
            );
        }
    }

    /// The mirroring is exactly the `step-before` <-> `step-after` swap, and
    /// nothing else moves. Pins the rule rather than only its consequences.
    #[test]
    fn only_the_one_sided_steps_mirror_under_reversal() {
        assert_eq!(Interpolate::StepBefore.mirrored(), Interpolate::StepAfter);
        assert_eq!(Interpolate::StepAfter.mirrored(), Interpolate::StepBefore);
        for method in [
            Interpolate::Linear,
            Interpolate::Step,
            Interpolate::Monotone,
            Interpolate::Basis,
            Interpolate::Cardinal,
        ] {
            assert_eq!(
                method.mirrored(),
                method,
                "{method:?} is reversal-symmetric and must not be swapped"
            );
        }
    }

    // ── CurveAxis: the y-parameterized branch ────────────────────────────

    fn transpose(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        points.iter().map(|&(x, y)| (y, x)).collect()
    }

    /// A `CurveAxis::Y` trace is exactly the `CurveAxis::X` trace of the
    /// transposed points, transposed back — the defining property of the
    /// projection, checked for every method and both step spellings.
    ///
    /// This is what makes one implementation serve both `d3.curveMonotoneX`
    /// and `d3.curveMonotoneY`.
    #[test]
    fn y_axis_trace_is_the_transposed_x_axis_trace() {
        fn transposed_cmd(cmd: &PathCmd) -> PathCmd {
            match *cmd {
                PathCmd::MoveTo { x, y } => PathCmd::MoveTo { x: y, y: x },
                PathCmd::LineTo { x, y } => PathCmd::LineTo { x: y, y: x },
                PathCmd::HLineTo { x } => PathCmd::VLineTo { y: x },
                PathCmd::VLineTo { y } => PathCmd::HLineTo { x: y },
                PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => PathCmd::CubicTo {
                    c1x: c1y,
                    c1y: c1x,
                    c2x: c2y,
                    c2y: c2x,
                    x: y,
                    y: x,
                },
                ref other => other.clone(),
            }
        }

        for method in Interpolate::ALL {
            for step in [StepSpelling::Shorthand, StepSpelling::Vertices] {
                let along_y = Trace { method, step, axis: CurveAxis::Y }.subpath(&SAMPLE);
                let along_x =
                    Trace { method, step, axis: CurveAxis::X }.subpath(&transpose(&SAMPLE));
                let expected: Vec<PathCmd> = along_x.iter().map(transposed_cmd).collect();
                assert_eq!(
                    along_y, expected,
                    "{method:?}/{step:?}: the y-parameterized trace must be the \
                     x-parameterized trace of the transposed points, transposed back"
                );
            }
        }
    }

    /// The observable consequence, stated directly: under `CurveAxis::Y` the
    /// curve is fitted against y, so every control point's y sits a third of
    /// the way into its segment's y-range. Under the x-parameterized fit the
    /// same data puts the thirds on x instead — so this discriminates the two,
    /// which a "does it emit cubics" assertion would not.
    #[test]
    fn y_axis_monotone_fits_against_y_not_x() {
        // A horizontal series: y advances steadily (the category axis), x
        // carries the value and is non-monotone.
        let points = [(10.0, 0.0), (40.0, 30.0), (20.0, 60.0), (50.0, 90.0)];
        let cmds = Trace {
            method: Interpolate::Monotone,
            step: StepSpelling::Vertices,
            axis: CurveAxis::Y,
        }
        .subpath(&points);

        assert_eq!(cmds.len(), 4, "MoveTo + 3 CubicTo; got {cmds:?}");
        for (i, cmd) in cmds[1..].iter().enumerate() {
            let PathCmd::CubicTo { c1y, c2y, .. } = *cmd else {
                panic!("expected CubicTo, got {cmd:?}");
            };
            let (y0, y1) = (points[i].1, points[i + 1].1);
            let third = (y1 - y0) / 3.0;
            assert!(
                (c1y - (y0 + third)).abs() < 1e-9,
                "segment {i}: c1y must sit one third along the Y span ({}), got {c1y}",
                y0 + third
            );
            assert!(
                (c2y - (y1 - third)).abs() < 1e-9,
                "segment {i}: c2y must sit two thirds along the Y span ({}), got {c2y}",
                y1 - third
            );
        }
    }

    /// Under `CurveAxis::Y` a step's tread runs along y and its riser along x
    /// — the shorthand commands swap roles with the axis.
    #[test]
    fn y_axis_step_shorthand_swaps_tread_and_riser() {
        let points = [(10.0, 0.0), (40.0, 30.0)];
        let cmds = Trace {
            method: Interpolate::StepAfter,
            step: StepSpelling::Shorthand,
            axis: CurveAxis::Y,
        }
        .subpath(&points);
        assert_eq!(
            cmds,
            vec![
                PathCmd::MoveTo { x: 10.0, y: 0.0 },
                // tread: advance the independent coordinate (y) first…
                PathCmd::VLineTo { y: 30.0 },
                // …then the riser moves the dependent one (x).
                PathCmd::HLineTo { x: 40.0 },
            ],
            "a y-parameterized step-after treads along y then risers along x"
        );
    }

    // ── WASM parity fixture ──────────────────────────────────────────────

    /// Drift guard for `ferrum-wasm`'s
    /// `tessellate::tests::curve_cubics_from_ferrum_core_tessellate`.
    ///
    /// `ferrum-wasm` cannot depend on `ferrum-core` (it builds for
    /// `wasm32-unknown-unknown`, `ferrum-core` links pyo3), so its parity test
    /// carries this command list as a literal. This test is the other half of
    /// that pairing: it asserts the literal is still what the real curve path
    /// emits, so a change to the math cannot leave the WASM test tessellating
    /// a shape production no longer produces.
    ///
    /// KEEP IN SYNC: if this fixture changes, update the same array in
    /// `crates/ferrum-wasm/src/tessellate.rs`.
    #[test]
    fn wasm_parity_fixture_matches_real_monotone_output() {
        let points = [(0.0, 0.0), (10.0, 30.0), (20.0, 10.0), (30.0, 40.0)];
        let expected = vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::CubicTo {
                c1x: 3.3333333333333335,
                c1y: 10.0,
                c2x: 6.666666666666666,
                c2y: 30.0,
                x: 10.0,
                y: 30.0,
            },
            PathCmd::CubicTo {
                c1x: 13.333333333333334,
                c1y: 30.0,
                c2x: 16.666666666666668,
                c2y: 10.0,
                x: 20.0,
                y: 10.0,
            },
            PathCmd::CubicTo {
                c1x: 23.333333333333332,
                c1y: 10.0,
                c2x: 26.666666666666668,
                c2y: 30.0,
                x: 30.0,
                y: 40.0,
            },
        ];
        assert_eq!(
            trace(Interpolate::Monotone, StepSpelling::Vertices).subpath(&points),
            expected,
            "the WASM parity fixture drifted from the real monotone output — \
             update crates/ferrum-wasm/src/tessellate.rs to match"
        );
    }
}
