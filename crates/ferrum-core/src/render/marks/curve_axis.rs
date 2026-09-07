//! Cross-mark family invariant: the axis a curve is FITTED against must be the
//! axis its point loop actually BUILDS along (batch-E close remediation).
//!
//! Test-only module, beside the marks rather than inside any one of them, for
//! the same reason [`super::band_width`] and [`super::clip_escape`] are: three
//! independent derivations answer one question, and no per-mark test can
//! compare them.
//!
//! - `line` reads [`DrawCtx::series_curve_axis`](crate::render::draw::DrawCtx::series_curve_axis)
//!   — `CoordFlip` or a `__stack_value_on_x__` stamp makes it `Y`.
//! - `area` reads `BottomEdge::axis`, which deliberately DISAGREES with the
//!   series under `CoordFlip` + an explicit `Stack(value_axis="y")`, because the
//!   arm that builds the bottom point is the one that knows what was built.
//! - `ribbon` reads the `RIBBON_CURVE_AXIS` const — always `X`, because a
//!   `y`..`y2` band at a shared x is a vertical idiom by construction.
//!
//! Each is individually well-argued. None of them is checked against the other
//! two, and `curve.rs`'s own doc says getting the axis wrong is *silent*: the
//! emitted path still looks like a curve, it just describes a function of the
//! wrong coordinate. That defect has already shipped twice in this family — a
//! first derivation that knew about the stack stamp and not the flip, and a
//! second that read the stamp while the point loop gave `y2` precedence over it.
//!
//! # How a row proves anything
//!
//! Every row renders through the mark's production builder with
//! `interpolate="monotone"`, then reads the axis back OUT of the emitted
//! control points via [`observe_monotone_trace`]: a monotone cubic puts its
//! control points on exact thirds of the segment's span along the INDEPENDENT
//! coordinate, and (for a non-degenerate series) nowhere near thirds on the
//! dependent one. So the assertion compares the shape actually drawn against the
//! shape the row's geometry demands — not one derivation against another
//! derivation, which would pass just as happily if both were wrong.
//!
//! An ambiguous fixture makes the observer answer `None`, which fails the
//! assertion rather than passing it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ferrum_scene::SceneNode;

use crate::layout::{PanelLayout, Rect, ThemeInputs};
use crate::render::draw::{resolve_mark_style, DrawCtx};
use crate::render::scale_resolve::resolve_scales;
use crate::spec::chart::ChartSpec;
use crate::spec::coord::CoordKind;
use crate::spec::data_ref::DataRef;
use crate::spec::encoding::{DataType as SpecType, Encoding, EncodingSpec};
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;

use super::curve::{observe_monotone_trace, CurveAxis};

const PLOT: Rect = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };

/// Which stack, if any, the batch is stamped with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stack {
    /// No `__stack_y_base__` column and no stamp.
    None,
    /// `Stack(value_axis="x")` — the cumulated value landed on x. Stamped
    /// `__stack_value_on_x__`, which is what `series_curve_axis` reads.
    ValueOnX,
    /// `Stack(value_axis="y")` — a base column with no stamp. The ordinary
    /// vertical stack, and the arm that disagrees with the series under a flip.
    ValueOnY,
}

/// One cell of the matrix: a coordinate system, a stack shape, and whether the
/// `y2` band channel is bound.
#[derive(Debug, Clone, Copy)]
struct Shape {
    name: &'static str,
    flip: bool,
    stack: Stack,
    y2: bool,
}

/// Four rows whose x advances non-monotonically while y advances steadily, so
/// neither coordinate's thirds hold by accident: the observer can always tell
/// the two parameterizations apart on this data. (Same fixture shape as
/// `area.rs`'s own flip test, for continuity.)
fn matrix_batch(stack: Stack) -> RecordBatch {
    let mut fields = vec![
        Field::new("x", ArrowType::Float64, false),
        Field::new("y", ArrowType::Float64, false),
        Field::new("hi", ArrowType::Float64, false),
    ];
    let mut columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![1.0, 4.0, 2.0, 5.0])),
        Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        Arc::new(Float64Array::from(vec![6.0, 7.0, 8.0, 9.0])),
    ];
    let mut metadata = HashMap::new();
    if stack != Stack::None {
        fields.push(Field::new("__stack_y_base__", ArrowType::Float64, false));
        columns.push(Arc::new(Float64Array::from(vec![0.0, 0.0, 0.0, 0.0])));
        if stack == Stack::ValueOnX {
            metadata.insert("__stack_value_on_x__".to_string(), "1".to_string());
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields).with_metadata(metadata)), columns).unwrap()
}

fn matrix_spec(mark: Mark, shape: Shape) -> ChartSpec {
    let quantitative = |field: &str| {
        Some(EncodingSpec {
            field: field.into(),
            type_: Some(SpecType::Quantitative),
            ..Default::default()
        })
    };
    ChartSpec {
        data: DataRef::default(),
        mark,
        encoding: Encoding {
            x: quantitative("x"),
            y: quantitative("y"),
            y2: if shape.y2 { quantitative("hi") } else { None },
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: shape.flip.then_some(CoordKind::Flip),
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

/// Render one cell with `interpolate="monotone"` through the mark's production
/// builder, and return the emitted nodes.
fn render(mark: Mark, shape: Shape) -> Vec<SceneNode> {
    let spec = matrix_spec(mark, shape);
    let batch = matrix_batch(shape.stack);
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
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
    let overrides = MarkKwargsSpec { interpolate: Some("monotone".into()), ..Default::default() };
    let mark_style = resolve_mark_style(Some(&overrides), &theme, &spec.mark).unwrap();
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

/// The axis the emitted geometry is ACTUALLY fitted against — read off the
/// first `Path` node's leading cubics.
fn observed_axis(mark: Mark, shape: Shape) -> CurveAxis {
    let nodes = render(mark, shape);
    let commands = nodes
        .iter()
        .find_map(|n| match n {
            SceneNode::Path { commands, .. } => Some(commands.clone()),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!("{mark:?} / {}: a monotone trace must emit a Path node", shape.name)
        });
    let observed = observe_monotone_trace(&commands);
    assert!(
        observed.cubics >= 3,
        "{mark:?} / {}: expected the edge's three cubics, saw {}",
        shape.name,
        observed.cubics
    );
    observed.axis.unwrap_or_else(|| {
        panic!(
            "{mark:?} / {}: the emitted control points sit on thirds of BOTH coordinates (or \
             neither), so this fixture cannot tell the two parameterizations apart — the row \
             proves nothing until the data discriminates",
            shape.name
        )
    })
}

const PLAIN: Shape = Shape { name: "plain", flip: false, stack: Stack::None, y2: false };
const FLIPPED: Shape = Shape { name: "CoordFlip", flip: true, stack: Stack::None, y2: false };
const STACK_X: Shape =
    Shape { name: "Stack(value_axis=x)", flip: false, stack: Stack::ValueOnX, y2: false };
const STACK_Y: Shape =
    Shape { name: "Stack(value_axis=y)", flip: false, stack: Stack::ValueOnY, y2: false };
const FLIP_STACK_Y: Shape = Shape {
    name: "CoordFlip + Stack(value_axis=y)",
    flip: true,
    stack: Stack::ValueOnY,
    y2: false,
};
const BAND: Shape = Shape { name: "y2 band", flip: false, stack: Stack::None, y2: true };
const BAND_FLIPPED: Shape =
    Shape { name: "y2 band + CoordFlip", flip: true, stack: Stack::None, y2: true };
const BAND_STACK_X: Shape = Shape {
    name: "y2 band + Stack(value_axis=x)",
    flip: false,
    stack: Stack::ValueOnX,
    y2: true,
};

/// `mark_line` advances along x, except where something puts the value on x.
///
/// Both producers of a horizontal series live in `series_curve_axis`, and the
/// flip row is the one the first derivation missed: it enumerated the stack
/// stamp and did not know about `CoordFlip`, so every flipped line was fitted
/// against the wrong axis while the mark's own const asserted it could not
/// happen.
#[test]
fn line_fits_against_the_axis_its_points_advance_along() {
    for (shape, expected) in [
        (PLAIN, CurveAxis::X),
        (FLIPPED, CurveAxis::Y),
        (STACK_X, CurveAxis::Y),
        // A flip plus a vertical stack: the flip alone already makes the series
        // horizontal, and line has no per-arm override to disagree with it.
        (FLIP_STACK_Y, CurveAxis::Y),
    ] {
        assert_eq!(
            observed_axis(Mark::Line, shape),
            expected,
            "mark_line / {}: the drawn curve must be fitted against {expected:?}",
            shape.name
        );
    }
}

/// `mark_area` answers from the arm that BUILDS its bottom point, so it agrees
/// with the series only where it builds no bottom edge at all.
///
/// The two rows that matter are the last three: a `y2` band builds
/// `(cx, by)` at the row's own x and is x-parameterized whatever the coordinate
/// system or the stack stamp says, and a vertical stack builds `(cx, base_y)`
/// and stays x-parameterized even under a flip — the one combination where
/// `series_curve_axis()` would answer `Y` and be wrong.
#[test]
fn area_fits_against_the_axis_its_bottom_edge_arm_builds() {
    for (shape, expected) in [
        // No bottom edge built → the series governs.
        (PLAIN, CurveAxis::X),
        (FLIPPED, CurveAxis::Y),
        // A horizontal stack builds `(base_x, cy_top)` — y-parameterized.
        (STACK_X, CurveAxis::Y),
        // A vertical stack builds `(cx, base_y)` — x-parameterized…
        (STACK_Y, CurveAxis::X),
        // …including under a flip, where the series would wrongly say Y.
        (FLIP_STACK_Y, CurveAxis::X),
        // A `y2` band takes precedence over every stack shape and over the
        // coordinate system: its bottom rides the row's own x.
        (BAND, CurveAxis::X),
        (BAND_FLIPPED, CurveAxis::X),
        (BAND_STACK_X, CurveAxis::X),
    ] {
        assert_eq!(
            observed_axis(Mark::Area, shape),
            expected,
            "mark_area / {}: the drawn curve must be fitted against {expected:?}",
            shape.name
        );
    }
}

/// `mark_ribbon` is x-parameterized unconditionally, and this pins that the
/// const is a statement about the shape rather than an assumption about
/// producers.
///
/// The horizontal-stack row is a DELIBERATE disagreement with `line` and `area`
/// on the identically-stamped batch: ribbon's band spans `y`..`y2` at a shared
/// x and its vertices are sorted by x-pixel before tracing, so the shape it
/// builds is vertical no matter what the stack cumulated. If ribbon ever learns
/// to read `x2`, that is a band on the other axis and this row must change with
/// it.
#[test]
fn ribbon_is_x_parameterized_whatever_the_batch_is_stamped_with() {
    for shape in [BAND, BAND_STACK_X] {
        assert_eq!(
            observed_axis(Mark::Ribbon, shape),
            CurveAxis::X,
            "mark_ribbon / {}: a y..y2 band at a shared x is x-parameterized",
            shape.name
        );
    }
}

/// The premise the ribbon rows rest on: under `CoordFlip`, `flip_positional`
/// has moved the user's `y2` to `x2`, so `encoding.y2` is unbound and ribbon
/// emits nothing. There is no flipped ribbon for the const to be wrong about.
///
/// Stated here as well as in `ribbon.rs` because this matrix would otherwise
/// have a visible hole where every other mark has a flip row.
#[test]
fn ribbon_has_no_flipped_row_because_a_flipped_ribbon_renders_nothing() {
    let unbound_y2 = Shape { name: "CoordFlip (y2 moved to x2)", ..FLIPPED };
    assert!(
        render(Mark::Ribbon, unbound_y2).is_empty(),
        "a flipped ribbon reaches the drawer with y2 unbound and must emit nothing — if this \
         now renders, RIBBON_CURVE_AXIS needs a flip case"
    );
}

/// Where the three marks build the SAME shape they must agree, and where they
/// build different shapes they must differ. Stated as one assertion so a future
/// change that quietly aligns them (or quietly splits them) is visible.
#[test]
fn the_family_agrees_on_shared_shapes_and_diverges_only_where_the_geometry_does() {
    // A plain vertical series is x-parameterized for all three.
    assert_eq!(observed_axis(Mark::Line, PLAIN), CurveAxis::X);
    assert_eq!(observed_axis(Mark::Area, PLAIN), CurveAxis::X);
    assert_eq!(observed_axis(Mark::Ribbon, BAND), CurveAxis::X);

    // Under a horizontal stack, line and area turn with the series; ribbon does
    // not, because its band is not built from the stacked value at all.
    assert_eq!(observed_axis(Mark::Line, STACK_X), CurveAxis::Y);
    assert_eq!(observed_axis(Mark::Area, STACK_X), CurveAxis::Y);
    assert_eq!(observed_axis(Mark::Ribbon, BAND_STACK_X), CurveAxis::X);

    // Under a flip plus a vertical stack, line follows the series and area
    // follows its arm — the one documented disagreement inside the family.
    assert_eq!(observed_axis(Mark::Line, FLIP_STACK_Y), CurveAxis::Y);
    assert_eq!(observed_axis(Mark::Area, FLIP_STACK_Y), CurveAxis::X);
}
