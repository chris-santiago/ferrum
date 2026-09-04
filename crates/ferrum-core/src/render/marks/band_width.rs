//! Cross-mark family invariant: an ordinal mark's width IS the resolved
//! scale's drawn band (F-L04-03, spec §4A).
//!
//! Test-only module, beside the marks rather than inside any one of them, for
//! the same reason [`super::color_dtype_parity`] is: the invariant it pins is a
//! *family* one, and it fits in one sentence — every ordinal mark sizes itself
//! as `ScaleKind::bandwidth() × its own band_size factor` (0.8 bar, 0.6
//! box/tick, 1.0 heatmap cell), never as
//! `panel_extent / distinct-values-in-this-batch`. Nine formulas across `bar`,
//! `rect` and `tick` implement that sentence. Pinning it per-mark would mean
//! four copies of the same resolver fixture and four chances for one member to
//! drift back — and drift is precisely the defect class here, since the
//! pre-fix formulas were nine copies of one expression that had gone stale in
//! two independent ways at once (blind to padding, counting off the batch).
//!
//! Every row resolves through the production resolver ([`resolve_scales`]) and
//! draws through the production dispatcher ([`dispatch_mark_build`]) — no
//! hand-built `ResolvedScales` — so a resolver that stopped threading
//! `padding_inner` fails these rows too.
//!
//! # Where the numbers come from
//!
//! Every expected pixel value is hand-computed from d3's band model, never
//! read off the implementation. For `n` categories over a signed extent:
//! `denom = n − padding_inner + 2·padding_outer`, `step = extent/denom`,
//! `bandwidth = |step|·(1 − padding_inner)`, and a category's center is the
//! middle of its drawn band. The fixtures deliberately choose dyadic
//! parameters — `padding_inner ∈ {0.25, 0.5}`, extents of 350/384/400/700 —
//! so `denom` and `step` are exact in binary and the arithmetic below is
//! reproducible on paper: `4 − 0.5 = 3.5`, `350/3.5 = 100`, `100·0.5 = 50`.
//!
//! Comparisons use a 1e-9 tolerance because the `band_size` factors (0.8, 0.6)
//! are not dyadic; the differences these rows discriminate are 7.5–30px, so
//! the tolerance costs nothing. The one exception is
//! [`unpadded_auto_path_reproduces_the_former_panel_extent_width`], which is
//! the local half of the byte-identity gate and therefore asserts exact
//! equality.

use std::sync::Arc;

use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ferrum_scene::SceneNode;

use crate::layout::{PanelLayout, Rect, ThemeInputs};
use crate::render::draw::{dispatch_mark_build, resolve_mark_style, DrawCtx};
use crate::render::position::apply_position;
use crate::render::scale_resolve::resolve_scales;
use crate::spec::chart::ChartSpec;
use crate::spec::data_ref::DataRef;
use crate::spec::encoding::{DataType as SpecType, Encoding, EncodingSpec, ScaleSpec};
use crate::spec::mark::Mark;
use crate::spec::position::PositionAdjust;

/// A `BandScale` with an explicit pixel range and no outer padding.
fn band(padding_inner: f64, range: [f64; 2]) -> ScaleSpec {
    ScaleSpec::Band {
        domain: None,
        padding: 0.0,
        padding_inner: Some(padding_inner),
        padding_outer: Some(0.0),
        align: 0.5,
        range: Some(range.to_vec()),
    }
}

/// The reference scale for the nine width rows: `padding_inner = 0.5` over
/// `[0, 350]`, four categories.
///
/// `denom = 3.5`, `step = 100`, **`bandwidth = 50`**, centers `25 / 125 / 225 /
/// 325` (the last band's trailing edge lands on 350 exactly). The pre-fix
/// formulas saw the raw slot — `350/4 = 87.5`, the extent over the batch's
/// distinct-value count with no padding term — for this same scale, so every
/// padded row below is RED against them at the source by 87.5-vs-50.
fn reference_band() -> ScaleSpec {
    band(0.5, [0.0, 350.0])
}

/// The drawn band of [`reference_band`]: `|step|·(1 − padding_inner)`.
const BANDWIDTH: f64 = 50.0;

/// The four band centers of [`reference_band`].
const CENTERS: [f64; 4] = [25.0, 125.0, 225.0, 325.0];

/// A 350×350 plot area anchored at the origin, so [`reference_band`]'s
/// explicit range and the panel-extent fallback describe the same pixels: the
/// rows that compare the two compare geometry, not anchoring.
const PLOT: Rect = Rect { x: 0.0, y: 0.0, w: 350.0, h: 350.0 };

/// Tolerance for the hand-computed pixel comparisons (see the module docs).
const EPS: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < EPS,
        "{what}: expected {expected}, got {actual}"
    );
}

/// Count the emitted marks before sweeping them.
///
/// Every sweep in this module goes through here, because a bare
/// `for row in rows` over an empty node list passes silently. T6 added nine
/// `None => return empty…` early returns — one per width site, each
/// structurally unreachable today — and a regression that made any of them
/// fire would otherwise be invisible in exactly the rows that guard the
/// highest-value path: the auto (unpadded, non-explicit) resolver arm every
/// default categorical chart takes, whose assertions are all
/// `rendered == pre-fix expression` and hold vacuously over zero marks.
fn counted<T>(rows: Vec<T>, expected: usize, what: &str) -> Vec<T> {
    assert_eq!(rows.len(), expected, "{what}: expected {expected} marks, got {}", rows.len());
    rows
}

/// One row per category, with quantitative companions for every channel the
/// band marks read (`val` for y/x, `val2` for the y2/x2 span) and `grp` for a
/// Dodge or second-axis split.
fn band_batch(cats: &[&str], groups: &[&str]) -> RecordBatch {
    let n = cats.len();
    assert_eq!(groups.len(), n, "fixture: one group per row");
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", ArrowType::Utf8, false),
        Field::new("val", ArrowType::Float64, false),
        Field::new("val2", ArrowType::Float64, false),
        Field::new("grp", ArrowType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(cats.to_vec())),
            Arc::new(Float64Array::from((0..n).map(|i| 1.0 + i as f64).collect::<Vec<_>>())),
            Arc::new(Float64Array::from((0..n).map(|i| 5.0 + i as f64).collect::<Vec<_>>())),
            Arc::new(StringArray::from(groups.to_vec())),
        ],
    )
    .unwrap()
}

/// The four-category batch every non-sparse row uses.
fn four_cats() -> RecordBatch {
    band_batch(&["a", "b", "c", "d"], &["g1", "g2", "g1", "g2"])
}

fn enc(field: &str, t: SpecType, scale: Option<ScaleSpec>) -> Option<EncodingSpec> {
    Some(EncodingSpec { field: field.into(), type_: Some(t), scale, ..Default::default() })
}

/// Resolve `encoding` over `batch` on a `plot`-sized panel, apply `position`,
/// and return the nodes `mark`'s real builder emits.
///
/// Mirrors the production order in `render::scene_build`: resolve scales over
/// the panel's plot area, then `apply_position` against those scales, then
/// dispatch the mark.
fn render_nodes(
    mark: Mark,
    encoding: Encoding,
    batch: &RecordBatch,
    position: Option<PositionAdjust>,
    plot: Rect,
) -> Vec<SceneNode> {
    let spec = ChartSpec {
        data: DataRef::default(),
        mark,
        encoding,
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: position.clone(),
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
    };
    let theme = ThemeInputs::default();
    let panel = PanelLayout {
        plot_area: plot,
        facet_key: None,
        row: 0,
        col: 0,
        strip_title: None,
        row_strip_title: None,
        row_facet_key: None,
    };
    let (scales, _) = resolve_scales(
        &spec,
        batch,
        (plot.x, plot.x + plot.w),
        (plot.y, plot.y + plot.h),
        &theme,
    )
    .expect("fixture must resolve");
    let adjusted = apply_position(
        batch,
        position.as_ref(),
        &scales,
        &spec.encoding,
        false,
        &mut Vec::new(),
    )
    .expect("fixture position adjustment must apply");
    let mark_style = resolve_mark_style(None, &theme, &spec.mark).expect("fixture mark style");
    let ctx = DrawCtx {
        spec: &spec,
        panel: &panel,
        theme: &theme,
        scales: &scales,
        batch: &adjusted,
        mark_style: &mark_style,
    };
    dispatch_mark_build(&spec.mark, &ctx)
        .expect("fixture must render")
        .nodes
}

/// The `(x, w)` of every emitted rect, in node order.
fn rect_x_w(nodes: &[SceneNode]) -> Vec<(f64, f64)> {
    nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Rect { x, w, .. } => Some((*x, *w)),
            _ => None,
        })
        .collect()
}

/// The `(y, h)` of every emitted rect, in node order.
fn rect_y_h(nodes: &[SceneNode]) -> Vec<(f64, f64)> {
    nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Rect { y, h, .. } => Some((*y, *h)),
            _ => None,
        })
        .collect()
}

/// The stored `(x1, y1, x2, y2)` of every emitted line, in node order.
///
/// The byte-identity rows assert on these rather than on [`line_spans`]'s
/// difference for the two band-axis tick modes, and the distinction is not
/// pedantry: those modes emit `center ∓ half`, so each endpoint rounds, and
/// their difference can miss the product `2·half` by an ulp even when both
/// endpoints are bit-identical to the pre-fix ones. The endpoints are what the
/// SVG serializes and what byte identity is a claim about; the span is a
/// derived quantity. (The crossbar modes anchor one end at the panel edge and
/// add `2·tick_half`, so their span is exact and stays on [`line_spans`].)
fn line_ends(nodes: &[SceneNode]) -> Vec<(f64, f64, f64, f64)> {
    nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        })
        .collect()
}

/// The signed `(x2 − x1, y2 − y1)` of every emitted line, in node order.
fn line_spans(nodes: &[SceneNode]) -> Vec<(f64, f64)> {
    nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((x2 - x1, y2 - y1)),
            _ => None,
        })
        .collect()
}

// ── The nine width formulas ─────────────────────────────────────────────────

/// Vertical bar: `bandwidth × 0.8`, centered on the band's middle.
///
/// RED pre-fix at all four bars: `87.5 × 0.8 = 70`px, so each bar overflowed
/// its own 50px band by 10px on each side and the four bars covered the ground
/// the padding existed to clear.
#[test]
fn vertical_bar_width_is_the_drawn_band_times_zero_point_eight() {
    let nodes = render_nodes(
        Mark::Bar,
        Encoding {
            x: enc("cat", SpecType::Nominal, Some(reference_band())),
            y: enc("val", SpecType::Quantitative, None),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    let rects = counted(rect_x_w(&nodes), 4, "vertical bars");
    for (i, (x, w)) in rects.iter().enumerate() {
        assert_close(*w, 40.0, "bar width = 50px band × 0.8");
        assert_close(*x, CENTERS[i] - 20.0, "bar left edge = center − width/2");
        // Each bar is inside its own drawn band — the assertion the pre-fix
        // width fails outright, since 87.5 × 0.8 = 70px of bar cannot fit in a
        // 50px band however it is centered.
        assert!(
            *x >= CENTERS[i] - BANDWIDTH / 2.0 - EPS
                && *x + *w <= CENTERS[i] + BANDWIDTH / 2.0 + EPS,
            "bar {i} spans [{x}, {}], outside its band [{}, {}]",
            *x + *w,
            CENTERS[i] - BANDWIDTH / 2.0,
            CENTERS[i] + BANDWIDTH / 2.0
        );
    }
}

/// Horizontal bar (ordinal y): the same `bandwidth × 0.8` on the y axis.
#[test]
fn horizontal_bar_height_is_the_drawn_band_times_zero_point_eight() {
    let nodes = render_nodes(
        Mark::Bar,
        Encoding {
            x: enc("val", SpecType::Quantitative, None),
            y: enc("cat", SpecType::Nominal, Some(reference_band())),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    let rects = counted(rect_y_h(&nodes), 4, "horizontal bars");
    for (i, (y, h)) in rects.iter().enumerate() {
        assert_close(*h, 40.0, "bar height = 50px band × 0.8");
        assert_close(*y, CENTERS[i] - 20.0, "bar top edge = center − height/2");
    }
}

/// Box body (ordinal x + a y..y2 span): `bandwidth × band_size`, default 0.6.
#[test]
fn box_body_width_is_the_drawn_band_times_band_size() {
    let nodes = render_nodes(
        Mark::Rect,
        Encoding {
            x: enc("cat", SpecType::Nominal, Some(reference_band())),
            y: enc("val", SpecType::Quantitative, None),
            y2: enc("val2", SpecType::Quantitative, None),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    let rects = counted(rect_x_w(&nodes), 4, "box bodies");
    for (i, (x, w)) in rects.iter().enumerate() {
        assert_close(*w, 30.0, "box width = 50px band × the 0.6 band_size default");
        assert_close(*x, CENTERS[i] - 15.0, "box left edge = center − width/2");
    }
}

/// Box body under CoordFlip (ordinal y + an x..x2 span): the y twin.
#[test]
fn flipped_box_body_height_is_the_drawn_band_times_band_size() {
    let nodes = render_nodes(
        Mark::Rect,
        Encoding {
            x: enc("val", SpecType::Quantitative, None),
            x2: enc("val2", SpecType::Quantitative, None),
            y: enc("cat", SpecType::Nominal, Some(reference_band())),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    let rects = counted(rect_y_h(&nodes), 4, "flipped box bodies");
    for (i, (y, h)) in rects.iter().enumerate() {
        assert_close(*h, 30.0, "box height = 50px band × the 0.6 band_size default");
        assert_close(*y, CENTERS[i] - 15.0, "box top edge = center − height/2");
    }
}

/// Heatmap cells: the drawn band on each axis, with no `band_size` factor.
///
/// Padded, a cell is its 50×128 band inside a 100×256 slot — the padding opens
/// a visible gutter, exactly as it does between bars. Unpadded the cells still
/// tile the panel exactly, which is what keeps every existing heatmap golden
/// byte-identical. The two axes carry different category counts and different
/// ranges so the row discriminates which scale each dimension asks.
#[test]
fn heatmap_cell_is_the_drawn_band_on_each_axis() {
    let plot = Rect { x: 0.0, y: 0.0, w: 350.0, h: 384.0 };
    let batch = four_cats(); // 4 `cat` values on x, 2 `grp` values on y.
    let encoding = |x_scale, y_scale| Encoding {
        x: enc("cat", SpecType::Nominal, x_scale),
        y: enc("grp", SpecType::Nominal, y_scale),
        color: enc("val", SpecType::Quantitative, None),
        ..Default::default()
    };

    // x: 4 cats over [0, 350] at pi = 0.5 → step 100, band 50.
    // y: 2 cats over [0, 384] at pi = 0.5 → denom 1.5, step 256, band 128.
    let padded = render_nodes(
        Mark::Rect,
        encoding(Some(reference_band()), Some(band(0.5, [0.0, 384.0]))),
        &batch,
        None,
        plot,
    );
    for ((_, w), (_, h)) in counted(rect_x_w(&padded), 4, "padded heatmap cells")
        .iter()
        .zip(counted(rect_y_h(&padded), 4, "padded heatmap cells").iter())
    {
        assert_close(*w, 50.0, "padded cell width is the drawn band, not the 100px slot");
        assert_close(*h, 128.0, "padded cell height is the drawn band, not the 256px slot");
    }

    // Unpadded (auto path): cells tile — 350/4 wide, 384/2 tall.
    let plain = render_nodes(Mark::Rect, encoding(None, None), &batch, None, plot);
    for ((_, w), (_, h)) in counted(rect_x_w(&plain), 4, "unpadded heatmap cells")
        .iter()
        .zip(counted(rect_y_h(&plain), 4, "unpadded heatmap cells").iter())
    {
        assert_close(*w, 87.5, "unpadded cells tile the panel width");
        assert_close(*h, 192.0, "unpadded cells tile the panel height");
    }
}

/// Boxplot-median tick (ordinal x + quantitative y): a horizontal line of
/// `bandwidth × band_size`, centered on the band.
#[test]
fn ordinal_x_tick_length_is_the_drawn_band_times_band_size() {
    let nodes = render_nodes(
        Mark::Tick,
        Encoding {
            x: enc("cat", SpecType::Nominal, Some(reference_band())),
            y: enc("val", SpecType::Quantitative, None),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    for (dx, dy) in counted(line_spans(&nodes), 4, "ordinal-x ticks") {
        assert_close(dx, 30.0, "tick length = 50px band × 0.6");
        assert_close(dy, 0.0, "the ordinal-x tick is horizontal");
    }
}

/// Strip tick (ordinal y + quantitative x): the vertical twin.
#[test]
fn ordinal_y_tick_length_is_the_drawn_band_times_band_size() {
    let nodes = render_nodes(
        Mark::Tick,
        Encoding {
            x: enc("val", SpecType::Quantitative, None),
            y: enc("cat", SpecType::Nominal, Some(reference_band())),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    for (dx, dy) in counted(line_spans(&nodes), 4, "ordinal-y ticks") {
        assert_close(dx, 0.0, "the ordinal-y tick is vertical");
        assert_close(dy, 30.0, "tick length = 50px band × 0.6");
    }
}

/// The two ordinal-ONLY crossbar modes, whose line runs along the axis that
/// carries no encoding.
///
/// Their length is keyed by convention to the CROSS-axis panel dimension, so
/// this row uses a non-square panel (700 wide, 350 tall) to prove which term
/// each mode divides — and each divides it by the categorical scale's own slot
/// count and applies its padding fraction.  y-only: `bandwidth_over(700)` for
/// four categories at `pi = 0.5` is `700/3.5 × 0.5 = 100`, so the crossbar is
/// `60`px. x-only: `bandwidth_over(350) = 50`, so the crossbar is `30`px,
/// drawn upward from the panel baseline.
#[test]
fn ordinal_only_crossbars_scale_the_cross_axis_extent_by_the_band_fraction() {
    let wide = Rect { x: 0.0, y: 0.0, w: 700.0, h: 350.0 };

    let y_only = render_nodes(
        Mark::Tick,
        Encoding {
            y: enc("cat", SpecType::Nominal, Some(reference_band())),
            ..Default::default()
        },
        &four_cats(),
        None,
        wide,
    );
    for (dx, dy) in counted(line_spans(&y_only), 4, "ordinal-y-only crossbars") {
        assert_close(dx, 60.0, "y-only crossbars run along x and divide panel.w (700)");
        assert_close(dy, 0.0, "y-only crossbars are horizontal");
    }

    let x_only = render_nodes(
        Mark::Tick,
        Encoding {
            x: enc("cat", SpecType::Nominal, Some(band(0.5, [0.0, 700.0]))),
            ..Default::default()
        },
        &four_cats(),
        None,
        wide,
    );
    for (dx, dy) in counted(line_spans(&x_only), 4, "ordinal-x-only crossbars") {
        assert_close(dx, 0.0, "x-only crossbars are vertical");
        assert_close(dy, -30.0, "x-only crossbars rise from the baseline, dividing panel.h (350)");
    }
}

// ── Domain count, not batch count (spec §4A) ────────────────────────────────

/// A layer whose batch is missing categories sizes by the scale's DOMAIN.
///
/// The deliberate correctness change: an empty facet cell, a filtered layer or
/// a shared-domain composite leaf used to inflate its bars to fill the panel,
/// because the width divided the extent by the categories *this batch*
/// happened to carry. The mechanism is the same wherever the gap comes from —
/// scale domain wider than batch — so it is pinned here with an explicit
/// four-category domain against a two-category batch: the sparse layer must
/// draw the same 80px bars its complete sibling does, where pre-fix it drew
/// 160px ones, double-width and spilling across the empty slots.
#[test]
fn a_batch_missing_categories_sizes_by_the_scale_domain() {
    let full_domain = ScaleSpec::Band {
        domain: Some(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
        padding: 0.0,
        padding_inner: Some(0.0),
        padding_outer: Some(0.0),
        align: 0.5,
        range: Some(vec![0.0, 400.0]),
    };
    let encoding = Encoding {
        x: enc("cat", SpecType::Nominal, Some(full_domain)),
        y: enc("val", SpecType::Quantitative, None),
        ..Default::default()
    };

    let sparse = render_nodes(
        Mark::Bar,
        encoding.clone(),
        &band_batch(&["a", "c"], &["g1", "g1"]),
        None,
        PLOT,
    );
    let complete = render_nodes(Mark::Bar, encoding, &four_cats(), None, PLOT);

    // 400px over the 4-category domain → 100px band → 80px bar, either way.
    let sparse_rects = counted(rect_x_w(&sparse), 2, "sparse-layer bars");
    for (_, w) in &sparse_rects {
        assert_close(*w, 80.0, "the sparse layer sizes by the 4-category domain");
    }
    for (_, w) in counted(rect_x_w(&complete), 4, "complete-layer bars") {
        assert_close(w, 80.0, "the complete sibling draws the same width");
    }
    // And they land in slots 0 and 2 (centers 50 and 250), not 0 and 1.
    assert_close(sparse_rects[0].0, 10.0, "category a sits in domain slot 0");
    assert_close(sparse_rects[1].0, 210.0, "category c sits in domain slot 2");
}

// ── Dodge composition (spec §4A decision 2) ─────────────────────────────────

/// Scale padding and `Dodge` padding compose: the scale defines the drawn
/// band, Dodge subdivides THAT band.
///
/// `BandScale(padding_inner=0.25)` over `[0, 350]` with two categories gives
/// `denom = 1.75`, `step = 200`, a `150`px drawn band, and band middles at 75
/// and 275. `Dodge(padding=0.05)` with two groups takes `150 × 0.05 = 7.5`px
/// off each end of that band and splits the rest into two `67.5`px sub-bands,
/// so group centers sit `±33.75` from the middle. Each bar is
/// `150/2 × 0.8 = 60`px, comfortably inside its sub-band.
///
/// RED pre-fix on the widths: the mark formulas divided the raw `350/2 = 175`
/// slot instead of the 150px band, giving 70px bars — wider than the 67.5px
/// sub-band, so the GH #66 clamp fired and pinned every bar to exactly the
/// sub-band width. The scale's padding was invisible in the bar geometry (it
/// moved only the dodge offsets) and Dodge's own padding was erased by the
/// clamp. That is the double-padding hazard this rule closes.
#[test]
fn dodge_subdivides_the_drawn_band_not_the_slot() {
    let batch = band_batch(&["a", "a", "b", "b"], &["g1", "g2", "g1", "g2"]);
    let nodes = render_nodes(
        Mark::Bar,
        Encoding {
            x: enc("cat", SpecType::Nominal, Some(band(0.25, [0.0, 350.0]))),
            y: enc("val", SpecType::Quantitative, None),
            color: enc("grp", SpecType::Nominal, None),
            ..Default::default()
        },
        &batch,
        Some(PositionAdjust::Dodge { by: Some("grp".into()), padding: 0.05 }),
        PLOT,
    );

    let rects = counted(rect_x_w(&nodes), 4, "dodged bars");
    // Row order is (a,g1) (a,g2) (b,g1) (b,g2); centers are the band middle
    // ∓ half a sub-band, and each left edge is that center − 30.
    for ((x, w), center) in rects.iter().zip([41.25, 108.75, 241.25, 308.75]) {
        assert_close(*w, 60.0, "bar = (150px band / 2 groups) × 0.8");
        assert_close(*x, center - 30.0, "bar centered in its dodge sub-band");
    }

    // The composition, read straight off the geometry: consecutive group
    // centers are one 67.5px sub-band apart, each 60px bar fits inside its
    // sub-band, and category "a"'s whole cluster stays within the scale's
    // drawn band [0, 150].
    assert_close(rects[1].0 - rects[0].0, 67.5, "group stride is the Dodge sub-band");
    assert!(rects[0].1 < 67.5, "the bar fits inside its sub-band");
    assert!(rects[0].0 >= 0.0, "the cluster starts inside the drawn band");
    assert!(rects[1].0 + rects[1].1 <= 150.0 + EPS, "the cluster ends inside the drawn band");
}

// ── The auto resolver arm ───────────────────────────────────────────────────

/// The auto-inferred arm — no `scale=` at all — places and sizes four bars at
/// hand-computed pixels.
///
/// The nine padded rows above all carry an explicit `scale=`, and the two
/// byte-identity rows below assert only `rendered == pre-fix expression`. A
/// regression confined to the auto arm (`DiscreteLayout::UNPADDED` built by
/// `build_axis_scale`'s ordinal branch, not by `build_from_scale_spec`) that
/// moved BOTH the rendered value and the arithmetic it is compared against —
/// or that emitted nothing at all — could satisfy those rows while breaking
/// every default categorical chart in the library. This row pins the arm's
/// absolute geometry instead: over a 350px panel, four categories give a
/// 87.5px band, 70px bars, and centers at `43.75 + i·87.5`.
#[test]
fn auto_resolver_arm_places_and_sizes_bars_at_hand_computed_pixels() {
    let nodes = render_nodes(
        Mark::Bar,
        Encoding {
            x: enc("cat", SpecType::Nominal, None),
            y: enc("val", SpecType::Quantitative, None),
            ..Default::default()
        },
        &four_cats(),
        None,
        PLOT,
    );
    let rects = counted(rect_x_w(&nodes), 4, "auto-arm bars");
    for (i, (x, w)) in rects.iter().enumerate() {
        let center = 43.75 + i as f64 * 87.5;
        assert_close(*w, 70.0, "auto-arm bar width = 87.5px band × 0.8");
        assert_close(*x, center - 35.0, "auto-arm bar left edge = center − width/2");
    }
}

// ── Byte-identity: the unpadded path is the old arithmetic ──────────────────

/// Panel geometries the byte-identity rows sweep: a zero origin, where
/// `(plot.x + plot.w) − plot.x == plot.w` holds trivially, and a fractional one
/// where it is the round-trip that has to hold. Six of the nine formulas divide
/// the scale's `[plot.x, plot.x + plot.w]` range span and so carry the ulp
/// hazard; the two crossbar arms lay out over `[0.0, extent]` and cannot.
const BYTE_IDENTITY_PANELS: [Rect; 2] = [
    Rect { x: 0.0, y: 0.0, w: 360.0, h: 200.0 },
    Rect { x: 47.5, y: 12.25, w: 526.25, h: 301.75 },
];

/// The zero-padding, panel-extent path reproduces the pre-F-L04-03 formula
/// exactly — `assert_eq!`, no tolerance.
///
/// This is the local half of the byte-identity gate (the golden corpora are
/// the other half). It runs the auto-inferred path — no `scale=` at all, what
/// every default categorical chart takes — and compares against the literal
/// pre-fix expression `panel.w / n_categories × 0.8`. `bandwidth()` arrives at
/// the same double: `denom == n` when both paddings are zero, and the
/// `× (1 − 0)` that follows is an exact multiplication by 1.0.
///
/// The extent it divides is the scale's range span, `(plot.x + plot.w) −
/// plot.x`, where the old formula used `plot.w` directly. Those agree in exact
/// arithmetic, and — as both panels here assert — in bits for these origins.
/// Where some future fractional origin does cost an ulp, the residue is
/// ~1e-13px against an SVG that serializes three decimals; the golden corpora
/// are what settle that empirically.
#[test]
fn unpadded_auto_path_reproduces_the_former_panel_extent_width() {
    for plot in BYTE_IDENTITY_PANELS {
        let nodes = render_nodes(
            Mark::Bar,
            Encoding {
                x: enc("cat", SpecType::Nominal, None),
                y: enc("val", SpecType::Quantitative, None),
                ..Default::default()
            },
            &four_cats(),
            None,
            plot,
        );
        let pre_fix_width = (plot.w / 4.0) * 0.8;
        for (_, w) in counted(rect_x_w(&nodes), 4, "auto-path bars") {
            assert_eq!(w, pre_fix_width, "auto path at {plot:?} must keep the pre-fix width");
        }
    }
}

/// The same reduction at the other eight formulas, in one sweep: box body
/// (both orientations), both tick band-axis modes, both ordinal-only crossbar
/// modes, and the heatmap cell all reproduce their pre-fix expressions on the
/// auto path, exactly — at both panel origins.
#[test]
fn unpadded_auto_path_reproduces_the_former_widths_for_every_ordinal_mark() {
    let batch = four_cats();

    for plot in BYTE_IDENTITY_PANELS {
        let slot_w = plot.w / 4.0;
        let slot_h = plot.h / 4.0;
        let render = |encoding| render_nodes(Mark::Rect, encoding, &batch, None, plot);
        let tick = |encoding| render_nodes(Mark::Tick, encoding, &batch, None, plot);

        let box_body = render(Encoding {
            x: enc("cat", SpecType::Nominal, None),
            y: enc("val", SpecType::Quantitative, None),
            y2: enc("val2", SpecType::Quantitative, None),
            ..Default::default()
        });
        for (_, w) in counted(rect_x_w(&box_body), 4, "box bodies") {
            assert_eq!(w, slot_w * 0.6, "box body width at {plot:?}");
        }

        let flipped_box = render(Encoding {
            x: enc("val", SpecType::Quantitative, None),
            x2: enc("val2", SpecType::Quantitative, None),
            y: enc("cat", SpecType::Nominal, None),
            ..Default::default()
        });
        for (_, h) in counted(rect_y_h(&flipped_box), 4, "flipped box bodies") {
            assert_eq!(h, slot_h * 0.6, "flipped box body height at {plot:?}");
        }

        // The band-axis tick modes centre their line on the category, so the
        // pre-fix expression to reproduce is the endpoint pair `center ∓ half`
        // — center from the pre-F-L04-03 symmetric model, half from the old
        // length formula. Both are exact here, and asserting them pins the
        // position as well as the length.
        let x_tick = tick(Encoding {
            x: enc("cat", SpecType::Nominal, None),
            y: enc("val", SpecType::Quantitative, None),
            ..Default::default()
        });
        let half_w = (slot_w * 0.6) / 2.0;
        for (i, (x1, _, x2, _)) in counted(line_ends(&x_tick), 4, "ordinal-x ticks")
            .iter()
            .enumerate()
        {
            let center = plot.x + slot_w / 2.0 + i as f64 * slot_w;
            assert_eq!(*x1, center - half_w, "ordinal-x tick start at {plot:?}");
            assert_eq!(*x2, center + half_w, "ordinal-x tick end at {plot:?}");
        }

        let y_tick = tick(Encoding {
            x: enc("val", SpecType::Quantitative, None),
            y: enc("cat", SpecType::Nominal, None),
            ..Default::default()
        });
        let half_h = (slot_h * 0.6) / 2.0;
        for (i, (_, y1, _, y2)) in counted(line_ends(&y_tick), 4, "ordinal-y ticks")
            .iter()
            .enumerate()
        {
            let center = plot.y + slot_h / 2.0 + i as f64 * slot_h;
            assert_eq!(*y1, center - half_h, "ordinal-y tick start at {plot:?}");
            assert_eq!(*y2, center + half_h, "ordinal-y tick end at {plot:?}");
        }

        // Cross-axis crossbars: each divides the OTHER panel dimension. These
        // anchor one end at a panel edge and step the other by the full
        // length, so the pre-fix expressions are `edge ± 2·half` — again the
        // endpoints, since the free end rounds against the anchor.
        let y_only = tick(Encoding {
            y: enc("cat", SpecType::Nominal, None),
            ..Default::default()
        });
        for (x1, _, x2, _) in counted(line_ends(&y_only), 4, "ordinal-y-only crossbars") {
            assert_eq!(x1, plot.x, "ordinal-y-only crossbar starts at the panel edge");
            assert_eq!(
                x2,
                plot.x + 2.0 * ((slot_w * 0.6) / 2.0),
                "ordinal-y-only crossbar end at {plot:?}"
            );
        }

        let x_only = tick(Encoding {
            x: enc("cat", SpecType::Nominal, None),
            ..Default::default()
        });
        let baseline_y = plot.y + plot.h;
        for (_, y1, _, y2) in counted(line_ends(&x_only), 4, "ordinal-x-only crossbars") {
            assert_eq!(y1, baseline_y, "ordinal-x-only crossbar starts at the baseline");
            assert_eq!(
                y2,
                baseline_y - 2.0 * ((slot_h * 0.6) / 2.0),
                "ordinal-x-only crossbar end at {plot:?}"
            );
        }

        let heatmap = render(Encoding {
            x: enc("cat", SpecType::Nominal, None),
            y: enc("grp", SpecType::Nominal, None),
            color: enc("val", SpecType::Quantitative, None),
            ..Default::default()
        });
        for ((_, w), (_, h)) in counted(rect_x_w(&heatmap), 4, "heatmap cells")
            .iter()
            .zip(counted(rect_y_h(&heatmap), 4, "heatmap cells").iter())
        {
            assert_eq!(*w, slot_w, "heatmap cell width at {plot:?}");
            assert_eq!(*h, plot.h / 2.0, "heatmap cell height (two `grp` categories) at {plot:?}");
        }
    }
}
