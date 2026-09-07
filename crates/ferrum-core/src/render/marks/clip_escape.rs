//! Cross-mark family invariant: `clip=False` means ONE thing across the whole
//! line family (batch-E close remediation, design finding "one public kwarg,
//! two meanings").
//!
//! Test-only module, beside the marks rather than inside any one of them, for
//! the same reason [`super::band_width`] and [`super::color_dtype_parity`] are:
//! the property is a *family* one and it fits in one sentence — `line`, `area`
//! and `ribbon` can all receive `clip`, so all three must honor it the same
//! way. Pinning it per-mark is what let it drift in the first place.
//!
//! # What went wrong
//!
//! `MarkKwargsSpec::clip` is a general mark-style field with no per-mark
//! allowlist on the Rust side, so any spec that sets it — including a layer's
//! own `mark_style` on the wire — reaches `resolve_mark_style` whatever the
//! mark is. But only `line.rs` read `misc.clip == Some(false)` and swapped in
//! the extrapolating scale twin. Area and ribbon therefore got the SVG-group
//! clip escape (T9) while still dropping every out-of-domain vertex to `None`
//! at the scale seam — precisely the half of the feature whose absence made T9
//! non-functional for `mark_line` before T13. Two marks silently kept the
//! broken half of the same field.
//!
//! Python's own mark allowlist is narrower today (`clip=` is surfaced through
//! `mark_function`, which desugars to a *line* layer), so this closes the seam
//! ahead of the public API rather than behind it: no charting call currently
//! reachable from `ferrum` changes its output.
//!
//! The fix routes all three through
//! [`DrawCtx::to_pixel`](crate::render::draw::DrawCtx::to_pixel). This module
//! asserts the three agree, in both directions: unclipped renders geometry past
//! the panel, and clipped is unchanged.
//!
//! # Why the scales are hand-built
//!
//! Mirroring `line.rs`'s own clip tests: `resolve_scales` infers the domain FROM
//! the data, which would make "out of domain" unreachable by construction. These
//! rows pin a y domain of `[0, 10]` onto a range of `[0, 100]` px and then feed a
//! row whose `y = 20`, so the extrapolated pixel is exactly `200` — twice the
//! panel height, unmistakably outside it, and an exact dyadic value.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ferrum_scene::{PathCmd, SceneNode};

use crate::layout::{PanelLayout, Rect, ThemeInputs};
use crate::render::draw::{resolve_mark_style, DrawCtx};
use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
use crate::scale::linear::LinearScale;
use crate::spec::chart::ChartSpec;
use crate::spec::data_ref::DataRef;
use crate::spec::encoding::{DataType as SpecType, Encoding, EncodingSpec};
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;

/// A 100×100 plot area at the origin, matching the scales' `[0, 100]` range so
/// "outside the range" and "outside the panel" are the same pixels.
const PLOT: Rect = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };

/// The data value placed outside the `[0, 10]` domain on the channel under
/// test. Both axes share that domain and a `[0, 100]` px range, so it
/// extrapolates to [`EXTRAPOLATED_PX`] on whichever coordinate it drives.
const OUT_OF_DOMAIN_VALUE: f64 = 20.0;

/// Where [`OUT_OF_DOMAIN_VALUE`] lands: `20 / 10 * 100`, twice the panel
/// extent, unmistakably outside it and an exact dyadic value.
const EXTRAPOLATED_PX: f64 = 200.0;

/// Domains deliberately NARROWER than the batch's data, so the third row is out
/// of domain. `resolve_scales` cannot produce this shape — it infers domains
/// from the data — which is why these are built by hand.
fn narrow_domain_scales() -> ResolvedScales {
    let axis = || {
        ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 10.0],
            vec![0.0, 100.0],
            false,
            false,
        ))
    };
    ResolvedScales {
        x: axis(),
        y: axis(),
        color: None,
        size: None,
        shape: None,
        opacity: None,
        fill_opacity: None,
        stroke_opacity: None,
        stroke_dash: None,
        x2: None,
        y2: None,
        y_slots: Default::default(),
    }
}

/// Which encoding channel carries the out-of-domain value.
///
/// A fixture that varies only one channel discriminates only the reads that
/// consume it: an earlier version of this module put the out-of-domain value on
/// `y` alone, and three `ctx.to_pixel` call sites — `ribbon`'s `x` read,
/// `area`'s `cx`, and `area`'s `y2` bottom edge — were executed by these tests
/// but invisible to them. Reverting any one of the three to `to_pixel_f64` left
/// the whole crate green. That is the same defect class this module exists to
/// catch (a claim asserted wider than its evidence), so the channel is a
/// parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutOfDomain {
    X,
    Y,
    /// The band's second edge. `area` reads it as `BottomEdge::Y2`, `ribbon` as
    /// its lower boundary; `line` binds no `y2` at all, hence no line row.
    Y2,
}

impl OutOfDomain {
    /// The screen coordinate this channel drives — the one an extrapolated
    /// vertex escapes the panel along.
    fn placed(self, nodes: &[SceneNode]) -> Vec<f64> {
        let (xs, ys) = placed_coords(nodes);
        match self {
            Self::X => xs,
            Self::Y | Self::Y2 => ys,
        }
    }

    /// The panel extent along that same coordinate.
    fn panel_extent(self) -> f64 {
        match self {
            Self::X => PLOT.w,
            Self::Y | Self::Y2 => PLOT.h,
        }
    }
}

/// Three rows, with `OUT_OF_DOMAIN_VALUE` on exactly one channel and every
/// other channel in domain — so a dropped vertex is unambiguously attributable
/// to the read under test.
fn out_of_domain_batch(channel: OutOfDomain) -> RecordBatch {
    let pick = |c: OutOfDomain, in_domain: [f64; 3]| -> Float64Array {
        let mut v = in_domain;
        if c == channel {
            v[2] = OUT_OF_DOMAIN_VALUE;
        }
        Float64Array::from(v.to_vec())
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowType::Float64, false),
        Field::new("y", ArrowType::Float64, false),
        Field::new("y2", ArrowType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(pick(OutOfDomain::X, [0.0, 5.0, 10.0])),
            Arc::new(pick(OutOfDomain::Y, [0.0, 5.0, 10.0])),
            Arc::new(pick(OutOfDomain::Y2, [0.0, 1.0, 2.0])),
        ],
    )
    .unwrap()
}

fn spec_for(mark: Mark) -> ChartSpec {
    let quantitative = |field: &str| {
        Some(EncodingSpec {
            field: field.into(),
            type_: Some(SpecType::Quantitative),
            ..Default::default()
        })
    };
    ChartSpec {
        data: DataRef::default(),
        encoding: Encoding {
            x: quantitative("x"),
            y: quantitative("y"),
            // Ribbon requires `y2`; line ignores it, and area's `BottomEdge`
            // reads it, so each mark takes the shape it actually renders.
            y2: match mark {
                Mark::Ribbon | Mark::Area => quantitative("y2"),
                _ => None,
            },
            ..Default::default()
        },
        mark,
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: None,
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
    }
}

/// Render one mark, with the out-of-domain value on `channel`, at one `clip=`
/// setting — through the mark's production builder.
fn render(mark: Mark, channel: OutOfDomain, clip: Option<bool>) -> Vec<SceneNode> {
    let spec = spec_for(mark);
    let batch = out_of_domain_batch(channel);
    let theme = ThemeInputs::default();
    let panel = PanelLayout {
        plot_area: PLOT,
        facet_key: None,
        row: 0,
        col: 0,
        strip_title: None,
        row_strip_title: None,
        row_facet_key: None,
    };
    let scales = narrow_domain_scales();
    let overrides = MarkKwargsSpec { clip, ..Default::default() };
    let mark_style = resolve_mark_style(Some(&overrides), &theme, &spec.mark)
        .expect("clip= alone never fails mark-style resolution");
    let ctx = DrawCtx {
        spec: &spec,
        panel: &panel,
        theme: &theme,
        scales: &scales,
        batch: &batch,
        mark_style: &mark_style,
    };
    match spec.mark {
        Mark::Line => super::line::build(&ctx).nodes,
        Mark::Area => super::area::build(&ctx).nodes,
        Mark::Ribbon => super::ribbon::build(&ctx).nodes,
        ref other => panic!("this module covers the line family only, got {other:?}"),
    }
}

/// Every coordinate the rendered nodes place, split by axis.
///
/// Line emits a `Polyline`, area and ribbon emit closed `Path`s, so a shared
/// assertion needs one reader over both. Returned as a pair rather than as two
/// walkers so the shorthand path commands are handled once: `HLineTo` places an
/// x and no y, `VLineTo` the reverse, and a per-axis copy of this match would
/// have to get that right twice.
fn placed_coords(nodes: &[SceneNode]) -> (Vec<f64>, Vec<f64>) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for node in nodes {
        match node {
            SceneNode::Polyline { points, .. } => {
                xs.extend(points.iter().map(|p| p.0));
                ys.extend(points.iter().map(|p| p.1));
            }
            SceneNode::Path { commands, .. } => {
                for cmd in commands {
                    match *cmd {
                        PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => {
                            xs.push(x);
                            ys.push(y);
                        }
                        PathCmd::HLineTo { x } => xs.push(x),
                        PathCmd::VLineTo { y } => ys.push(y),
                        PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
                            xs.extend([c1x, c2x, x]);
                            ys.extend([c1y, c2y, y]);
                        }
                        PathCmd::QuadTo { cx, cy, x, y } => {
                            xs.extend([cx, x]);
                            ys.extend([cy, y]);
                        }
                        PathCmd::Close | PathCmd::ArcTo { .. } => {}
                    }
                }
            }
            _ => {}
        }
    }
    (xs, ys)
}

/// Every (mark, channel) pair the family actually reads — one row per live
/// `ctx.to_pixel` call site in `line.rs`, `area.rs` and `ribbon.rs`.
///
/// `line` binds no `y2`, so it has no `Y2` row; every other combination is a
/// positional read that the `clip` gate governs. Named so a failure says which
/// mark and which channel drifted.
const COVERAGE: [(&str, Mark, OutOfDomain); 8] = [
    ("mark_line", Mark::Line, OutOfDomain::X),
    ("mark_line", Mark::Line, OutOfDomain::Y),
    ("mark_area", Mark::Area, OutOfDomain::X),
    ("mark_area", Mark::Area, OutOfDomain::Y),
    ("mark_area", Mark::Area, OutOfDomain::Y2),
    ("mark_ribbon", Mark::Ribbon, OutOfDomain::X),
    ("mark_ribbon", Mark::Ribbon, OutOfDomain::Y),
    ("mark_ribbon", Mark::Ribbon, OutOfDomain::Y2),
];

/// `clip=False` must place geometry OUTSIDE the panel for every mark, on every
/// channel it reads — the whole point of the escape.
///
/// RED before the fix for `mark_area` and `mark_ribbon`: both dropped the
/// out-of-domain vertex to `None` at the scale seam, so nothing they emitted
/// ever left the panel and the SVG-group escape they were already granted had
/// nothing to reveal. RED again, per channel, for any single read reverted to
/// `to_pixel_f64` — which the y-only fixture this replaced could not see.
#[test]
fn clip_false_places_geometry_outside_the_panel_for_every_read_in_the_family() {
    for (name, mark, channel) in COVERAGE {
        let placed = channel.placed(&render(mark, channel, Some(false)));
        assert!(
            !placed.is_empty(),
            "{name} / {channel:?}: clip=False must still render something"
        );
        let extreme = placed.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            (extreme - EXTRAPOLATED_PX).abs() < 1e-9,
            "{name} / {channel:?}: the out-of-domain row ({OUT_OF_DOMAIN_VALUE} against domain \
             [0,10] → range [0,100]) must extrapolate to {EXTRAPOLATED_PX}, got {extreme}"
        );
        assert!(
            extreme > channel.panel_extent(),
            "{name} / {channel:?}: clip=False must place a vertex outside the panel \
             ({extreme} vs panel extent {})",
            channel.panel_extent()
        );
    }
}

/// The default (`clip` unset) drops the same vertex it always did: nothing
/// leaves the panel, for any mark on any channel. This is the byte-stability
/// half — the extension is gated strictly on `Some(false)`.
#[test]
fn clip_unset_keeps_every_read_in_the_family_inside_the_panel() {
    for (name, mark, channel) in COVERAGE {
        let placed = channel.placed(&render(mark, channel, None));
        assert!(
            !placed.is_empty(),
            "{name} / {channel:?}: the two in-domain rows must still render with clip unset"
        );
        let extreme = placed.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            extreme <= channel.panel_extent(),
            "{name} / {channel:?}: the default must never place geometry outside the panel; \
             got {extreme}"
        );
    }
}

/// `clip=Some(true)` is bit-identical to leaving `clip` unset, for every mark
/// on every channel: the gate reads `== Some(false)`, never "not false".
/// Compared on the whole node list, so a difference in any style or coordinate
/// fails.
#[test]
fn clip_true_is_byte_identical_to_unset_for_every_read_in_the_family() {
    for (name, mark, channel) in COVERAGE {
        assert_eq!(
            render(mark, channel, Some(true)),
            render(mark, channel, None),
            "{name} / {channel:?}: explicit clip=True must render exactly what unset renders"
        );
    }
}

/// `mark_area`'s two stacked-base reads honor `clip` as well.
///
/// [`COVERAGE`] cannot reach these: `BottomEdge::Y2` takes precedence over
/// `StackBase`, and none of its rows stamp a stack, so `base_x` and `base_y`
/// were *uncovered* rather than covered-but-undiscriminated. Same root cause as
/// the y-only fixture — the rows did not span the reads — so it is closed here
/// rather than left for a later cycle to rediscover.
///
/// The two arms differ in which coordinate the base drives:
/// `Stack(value_axis="x")` builds `(base_x, cy_top)`, so an out-of-domain base
/// escapes along x; `Stack(value_axis="y")` builds `(cx, base_y)` and escapes
/// along y. Both fall back to their panel baseline when the read drops the
/// value, which is what makes the clipped case land exactly ON the panel edge.
#[test]
fn stacked_area_base_reads_honor_clip() {
    /// An area spec with NO `y2`, so `BottomEdge` resolves to `StackBase`.
    fn stacked_area_spec() -> ChartSpec {
        ChartSpec { encoding: Encoding { y2: None, ..spec_for(Mark::Area).encoding }, ..spec_for(Mark::Area) }
    }

    /// Three rows whose top edge is entirely in domain and whose stack base's
    /// third row is not, so the escaping coordinate is unambiguously the base.
    fn stacked_batch(value_on_x: bool) -> RecordBatch {
        let mut metadata = HashMap::new();
        if value_on_x {
            metadata.insert("__stack_value_on_x__".to_string(), "1".to_string());
        }
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("x", ArrowType::Float64, false),
                Field::new("y", ArrowType::Float64, false),
                Field::new("__stack_y_base__", ArrowType::Float64, false),
            ])
            .with_metadata(metadata),
        );
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, OUT_OF_DOMAIN_VALUE])),
            ],
        )
        .unwrap()
    }

    fn render_stacked(value_on_x: bool, clip: Option<bool>) -> Vec<SceneNode> {
        let spec = stacked_area_spec();
        let batch = stacked_batch(value_on_x);
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: PLOT,
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
            row_strip_title: None,
            row_facet_key: None,
        };
        let scales = narrow_domain_scales();
        let overrides = MarkKwargsSpec { clip, ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Area).unwrap();
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        super::area::build(&ctx).nodes
    }

    for (value_on_x, channel) in [(true, OutOfDomain::X), (false, OutOfDomain::Y)] {
        let arm = if value_on_x { "Stack(value_axis=x)" } else { "Stack(value_axis=y)" };

        let unclipped = channel.placed(&render_stacked(value_on_x, Some(false)));
        let extreme = unclipped.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            (extreme - EXTRAPOLATED_PX).abs() < 1e-9,
            "{arm}: clip=False must extrapolate the stack base to {EXTRAPOLATED_PX}, got {extreme}"
        );

        let clipped = channel.placed(&render_stacked(value_on_x, None));
        let clipped_extreme = clipped.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            clipped_extreme <= channel.panel_extent(),
            "{arm}: the default must keep the base inside the panel (it falls back to the \
             baseline); got {clipped_extreme}"
        );
        assert_ne!(
            render_stacked(value_on_x, Some(false)),
            render_stacked(value_on_x, None),
            "{arm}: this row must reach the clip gate"
        );
        assert_eq!(
            render_stacked(value_on_x, Some(true)),
            render_stacked(value_on_x, None),
            "{arm}: explicit clip=True must render exactly what unset renders"
        );
    }
}

/// Every row of [`COVERAGE`] must reach the gate: `clip=False` and the default
/// must NOT produce the same nodes. Without this, a row whose channel the mark
/// happens not to read would pass all three tests above vacuously — which is
/// exactly how the y-only fixture concealed three surviving mutants.
#[test]
fn every_covered_read_actually_separates_clipped_from_unclipped() {
    for (name, mark, channel) in COVERAGE {
        assert_ne!(
            render(mark, channel, Some(false)),
            render(mark, channel, None),
            "{name} / {channel:?}: this row must reach the clip gate — if these agree, the row \
             proves nothing"
        );
    }
}
