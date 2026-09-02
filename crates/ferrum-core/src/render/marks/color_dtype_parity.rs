//! Cross-mark parity: one categorical color column paints the same way on
//! every mark, whatever its Arrow dtype (NF-A3, spec §4.4).
//!
//! Test-only module. It lives beside the marks rather than inside any one of
//! them because the invariant it pins is a *family* invariant, and the defect
//! it guards against is specifically a family one: the NF-A3 color-reader swap
//! landed on `line`/`ribbon`/`area` first and left `point`, `bar`, `rect`,
//! `arc`, `segment`, `rule`, `text`, and `polygon` reading `col_as_str`, whose
//! `Err` on every non-`Utf8` dtype was swallowed by `.ok()`. Those marks then
//! painted every element the palette's first color while the legend — built
//! from the dtype-wide `distinct_values_in_order` — enumerated all the
//! categories. Per-mark tests could not have caught that, and did not: each
//! mark's own suite tested it with a `Utf8` column, where the two readers agree
//! exactly. A table over `mark × dtype` is the shape that fails when one member
//! of the family drifts.
//!
//! The assertion is deliberately the *discriminating* one from the design
//! review's probe: count distinct paint colors **on the mark elements**, not
//! across the document. A document-wide color set is not discriminating,
//! because the legend swatches carry both category colors either way — which is
//! exactly what made the defect invisible on sight.
//!
//! The table's second half (`temporal_color_renders_…` onward) covers the
//! *numeric*-keyed color scales, and it exists because the table alone was not
//! enough: it compares each mark only against its own `Utf8` twin, never marks
//! against each other, and only over dtypes both color readers accept. That let
//! a second family defect through — the stroke-only marks refusing a temporal
//! color encoding that `point` rendered as a gradient. Those rows therefore
//! compare marks *to each other* on one encoding, and pin the outcome class
//! (renders vs. refuses) as well as the colors.
//!
//! Both halves compared *readers*, and every spec in them was bare — no
//! `scale=`. That left the third blind spot: the resolver branch that decides
//! whether a reader is consulted at all. An explicit `scale.domain` supersedes
//! the domain build, so a column no reader can key on reached the mark builders
//! anyway. The explicit-`domain`+`range` rows carry a `scale=` for that reason,
//! and they assert the refusal's *wording* so the stage it comes from (scale
//! resolution, uniform) is pinned rather than just its existence.
//!
//! All three halves call [`dispatch_mark_build`] on a *flat* spec, and that is
//! the fourth blind spot the final section closes: a chart the Python API lowers
//! to LAYERS never reaches a mark builder with the encoding those rows assert
//! on. `mark_ribbon` is such a chart — `desugar_ribbon` emits one ribbon layer
//! and the chart-level `color` reaches it only by inheritance — so its color
//! partition is decided at the layer seam in `render::scene_build`, not by the
//! reader `ribbon.rs` uses. Those rows therefore render the whole chart.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    BooleanArray, Float64Array, Int64Array, StringArray, TimestampMillisecondArray,
};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ferrum_scene::SceneNode;

use crate::layout::{PanelLayout, Rect, ThemeInputs};
use crate::render::draw::{dispatch_mark_build, resolve_mark_style, DrawCtx};
use crate::render::scale_resolve::resolve_scales;
use crate::spec::chart::ChartSpec;
use crate::spec::data_ref::DataRef;
use crate::spec::encoding::{DataType as SpecType, Encoding, EncodingSpec};
use crate::spec::mark::Mark;

/// The four color columns under test. All four carry the SAME two categories in
/// the same row order, so every mark must produce the same 2-color, 2-elements-
/// each partition from each of them.
const COLOR_COLUMNS: [&str; 4] = ["c_utf8", "c_i64", "c_f64", "c_bool"];

/// Four rows, two of each category, on two x positions.
///
/// `t` and `v_nan` serve the *numeric*-keyed color rows below rather than the
/// categorical table: `t` is the one dtype that is a supported numeric column
/// and not a supported category key, and `v_nan` carries a non-finite value in
/// an otherwise ordinary quantitative column.
fn parity_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowType::Float64, false),
        Field::new("y", ArrowType::Float64, false),
        Field::new("x2", ArrowType::Float64, false),
        Field::new("y2", ArrowType::Float64, false),
        Field::new("xo", ArrowType::Utf8, false),
        Field::new("c_utf8", ArrowType::Utf8, false),
        Field::new("c_i64", ArrowType::Int64, false),
        Field::new("c_f64", ArrowType::Float64, false),
        Field::new("c_bool", ArrowType::Boolean, false),
        Field::new("t", ArrowType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        Field::new("v_nan", ArrowType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5, 3.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 0.0, 0.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            Arc::new(StringArray::from(vec!["p", "q", "p", "q"])),
            Arc::new(Int64Array::from(vec![1_i64, 2, 1, 2])),
            Arc::new(Float64Array::from(vec![1.5, 2.5, 1.5, 2.5])),
            Arc::new(BooleanArray::from(vec![true, false, true, false])),
            Arc::new(TimestampMillisecondArray::from(vec![
                0_i64,
                86_400_000,
                172_800_000,
                259_200_000,
            ])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, f64::NAN, 2.0])),
        ],
    )
    .unwrap()
}

fn field(name: &str, t: Option<SpecType>) -> Option<EncodingSpec> {
    Some(EncodingSpec { field: name.into(), type_: t, ..Default::default() })
}

/// The positional channels each mark needs, plus the nominal color channel.
///
/// Every mark gets the SAME color encoding (`type = nominal`, so a numeric
/// column force-resolves a categorical scale exactly as `fm.Color(…,
/// type="nominal")` does); only the geometry channels differ, because the marks
/// genuinely consume different ones.
fn encoding_for(mark: &Mark, color_col: &str) -> Encoding {
    let q = Some(SpecType::Quantitative);
    let n = Some(SpecType::Nominal);
    let color = field(color_col, n);
    match mark {
        // Ordinal x + quantitative y: the band marks.
        Mark::Bar => Encoding { x: field("xo", n), y: field("y", q), color, ..Default::default() },
        // Ordinal x + y..y2 span: rect's ordinal-range (boxplot-body) path.
        Mark::Rect => Encoding {
            x: field("xo", n),
            y: field("y", q),
            y2: field("y2", q),
            color,
            ..Default::default()
        },
        // Both endpoints bound: the diagonal marks.
        Mark::Segment => Encoding {
            x: field("x", q),
            y: field("y", q),
            x2: field("x2", q),
            y2: field("y2", q),
            color,
            ..Default::default()
        },
        // `x` alone → RuleShape::VerticalSpan, one full-height rule per row.
        Mark::Rule => Encoding { x: field("x", q), color, ..Default::default() },
        // Everything else: plain quantitative x/y.
        _ => Encoding { x: field("x", q), y: field("y", q), color, ..Default::default() },
    }
}

fn spec_for(mark: Mark, color_col: &str) -> ChartSpec {
    ChartSpec {
        data: DataRef::default(),
        encoding: encoding_for(&mark, color_col),
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

/// Every paint color a mark element carries, as `#rrggbb`.
///
/// Fill for the filled variants, stroke for the stroke-only ones — i.e. the
/// slot the color channel actually drives on that mark. Nested `Group` nodes
/// are walked so no mark's element is missed by shape.
fn element_colors(nodes: &[SceneNode], out: &mut Vec<String>) {
    fn hex(c: ferrum_scene::Color) -> String {
        format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)
    }
    for node in nodes {
        match node {
            SceneNode::Rect { style, .. }
            | SceneNode::Circle { style, .. }
            | SceneNode::Path { style, .. }
            | SceneNode::Polygon { style, .. } => out.extend(style.fill.map(hex)),
            SceneNode::Line { style, .. } | SceneNode::Polyline { style, .. } => {
                out.push(hex(style.color))
            }
            SceneNode::Text { style, .. } => out.push(hex(style.color)),
            SceneNode::Group { children, .. } => element_colors(children, out),
            SceneNode::Image { .. } | SceneNode::Raw { .. } => {}
        }
    }
}

/// Render `spec` against [`parity_batch`] and return its mark elements' paint
/// colors **in node order** (which is row order for every mark here).
///
/// Both failure stages are surfaced as `Err` rather than a panic, because
/// whether a chart renders or refuses is itself the thing some rows below pin.
fn build_element_colors(spec: &ChartSpec) -> Result<Vec<String>, crate::render::RenderError> {
    let batch = parity_batch();
    let theme = ThemeInputs::default();
    let panel = PanelLayout {
        plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
        facet_key: None,
        row: 0,
        col: 0,
        strip_title: None,
        row_strip_title: None,
        row_facet_key: None,
    };
    let (scales, _) = resolve_scales(spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme)?;
    let mark_style = resolve_mark_style(None, &theme, &spec.mark).unwrap();
    let ctx = DrawCtx {
        spec,
        panel: &panel,
        theme: &theme,
        scales: &scales,
        batch: &batch,
        mark_style: &mark_style,
    };
    let result = dispatch_mark_build(&spec.mark, &ctx)?;
    let mut colors = Vec::new();
    element_colors(&result.nodes, &mut colors);
    Ok(colors)
}

/// `color hex → element count` for one `(mark, color column)` pair.
fn color_histogram(mark: Mark, color_col: &str) -> BTreeMap<String, usize> {
    let spec = spec_for(mark, color_col);
    let colors = build_element_colors(&spec)
        .unwrap_or_else(|e| panic!("{mark:?}/{color_col}: render failed: {e:?}"));
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for c in colors {
        *hist.entry(c).or_default() += 1;
    }
    hist
}

/// NF-A3 family invariant: for every mark, an `Int64` / `Float64` / `Boolean`
/// nominal color column paints exactly the partition its `Utf8` twin does.
///
/// This is the design review's two-render discrimination probe, generalized: it
/// compares each non-`Utf8` dtype's histogram against the `Utf8` one for the
/// SAME mark, so it cannot be satisfied by a mark that collapses every element
/// onto one color (the pre-fix behavior gives `{first_palette_color: 4}` for the
/// three non-`Utf8` columns against `{c1: 2, c2: 2}` for `Utf8`).
///
/// RED before the fix, for the eight marks whose color read still used
/// `col_as_str`: `point`, `bar`, `rect`, `text`, `rule`, `segment` (and `arc`,
/// `polygon`, covered by their own tests since their encodings don't fit this
/// table). GREEN for `line` and `area`, which the batch had already swept —
/// they are the controls that prove the table is measuring the right thing.
#[test]
fn every_mark_paints_non_utf8_categories_like_their_utf8_twin() {
    let marks = [
        Mark::Point,
        Mark::Bar,
        Mark::Line,
        Mark::Area,
        Mark::Rect,
        Mark::Text,
        Mark::Rule,
        Mark::Segment,
    ];
    for mark in marks {
        let reference = color_histogram(mark, "c_utf8");
        assert_eq!(
            reference.len(),
            2,
            "{mark:?}: the Utf8 reference must itself resolve two distinct category \
             colors, else this row proves nothing; got {reference:?}"
        );
        for col in COLOR_COLUMNS.iter().skip(1) {
            let hist = color_histogram(mark, col);
            // The whole `color → count` map, not just the count vector: the
            // palette is indexed by domain position, so every dtype's category
            // *k* must land on palette entry *k*. Comparing only counts would
            // pass a mark that split 2/2 onto the wrong two palette entries.
            assert_eq!(
                hist, reference,
                "{mark:?}: color column `{col}` must paint the same element partition \
                 as the Utf8 twin (expected {reference:?}, got {hist:?}). A single-entry \
                 histogram here is the NF-A3 defect: col_as_str returns Err for this \
                 dtype, .ok() swallows it, and every element falls back to one color \
                 while the legend still lists both categories."
            );
        }
    }
}

/// The same partition, stated as the design review's literal probe shape so a
/// regression reads as "4 marks in the first palette color" rather than as an
/// abstract histogram mismatch.
///
/// `point` and `bar` are the two marks the review probed directly (`{#2563eb: 4,
/// #dc2626: 0}` before the fix vs `{2, 2}` for the Utf8 twin).
#[test]
fn point_and_bar_split_an_int64_nominal_color_two_and_two() {
    for mark in [Mark::Point, Mark::Bar] {
        let hist = color_histogram(mark, "c_i64");
        assert_eq!(
            hist.len(),
            2,
            "{mark:?}: an Int64 nominal color column must resolve TWO colors across \
             four marks, not one; got {hist:?}"
        );
        assert!(
            hist.values().all(|&n| n == 2),
            "{mark:?}: the two categories must take two marks each; got {hist:?}"
        );
    }
}

// ── The numeric half of the family invariant ────────────────────────────────
//
// The table above covers category-keyed color scales. These rows cover the
// numeric-keyed ones (`Continuous` / `Discretizing`), where the parallel defect
// is not "one color for every element" but a *cross-mark* divergence: the
// stroke-only marks resolve color through `row_colors_from_scale` while the
// filled marks resolve it through `resolve_fill_color`, and those two readers
// must reach the same answer from the same encoding. The specs are the exact
// JSON `Chart.to_json()` lowers for the charts named in each test, so a shape
// the Python API cannot produce cannot be pinned here by accident.

/// `mark_rule` + temporal color, exactly as `fm.Chart(df).mark_rule().encode(
/// x="x", y="y", color=fm.Color("t", type="temporal"))` lowers it.
const RULE_TEMPORAL: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"temporal"}}}"#;
/// The same encoding on `mark_point` — the parity reference.
const POINT_TEMPORAL: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"temporal"}}}"#;
/// The same encoding on `mark_segment`, which additionally binds `x2`/`y2`.
const SEGMENT_TEMPORAL: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"segment","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"temporal"},"x2":{"field":"x2"},"y2":{"field":"y2"}}}"#;

/// `mark_rule` + a quantitative color column carrying one `NaN` row.
const RULE_NAN: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"v_nan","type":"quantitative"}}}"#;
/// The same chart with the color channel dropped — the constant-paint reference.
const RULE_NO_COLOR: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"x"},"y":{"field":"y"}}}"#;
/// `mark_point` + the same `NaN`-carrying quantitative color column.
const POINT_NAN: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"v_nan","type":"quantitative"}}}"#;

/// `mark_rule` + a *nominal* temporal column — the category-keyed counterpart.
const RULE_NOMINAL_TS: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"nominal"}}}"#;
/// The same nominal-temporal encoding on `mark_point` and `mark_line`.
const POINT_NOMINAL_TS: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"nominal"}}}"#;
const LINE_NOMINAL_TS: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"line","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"nominal"}}}"#;

/// The same nominal-temporal encoding carrying an explicit ordinal `domain` +
/// `range`, as `fm.Color("t", type="nominal", scale=fm.OrdinalScale(domain=[…],
/// range=["red","blue","green","orange"]))` lowers it. The four domain entries
/// are `t`'s four epoch-millisecond values, so the declared domain is a *valid*
/// domain for the column — the scale that used to resolve here was well-formed,
/// which is precisely why nothing downstream noticed the column was unkeyable.
///
/// This is the shape that skips the domain builder: `scale.domain` supersedes
/// `distinct_values_in_order`, so before the `ensure_category_keyable` gate the
/// column was never read at scale-resolution time.
const POINT_NOMINAL_TS_EXPLICIT_RANGE: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"nominal","scale":{"type":"ordinal","domain":["0","86400000","172800000","259200000"],"range":["red","blue","green","orange"],"padding":0.0}}}}"#;
/// The same explicit-`domain`+`range` encoding on `mark_rule`.
const RULE_NOMINAL_TS_EXPLICIT_RANGE: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"t","type":"nominal","scale":{"type":"ordinal","domain":["0","86400000","172800000","259200000"],"range":["red","blue","green","orange"],"padding":0.0}}}}"#;

fn colors_of(spec_json: &str) -> Result<Vec<String>, crate::render::RenderError> {
    let spec: ChartSpec = serde_json::from_str(spec_json).expect("lowered spec parses");
    build_element_colors(&spec)
}

/// A temporal color encoding renders the same gradient on the stroke-only marks
/// as on `point`, per row.
///
/// `Timestamp` is a supported *numeric* dtype and not a supported *category*
/// key, so this is the encoding that discriminates `row_colors_from_scale`'s
/// dispatch: when the reader forces every scale through the categorical string
/// path, `col_as_ordinal_category_str` refuses the column and — because the
/// read is now correctly propagated with `?` — the whole chart raises
/// `UnsupportedDtype`, while `point` with the identical encoding renders a
/// gradient. Asserting the per-row color *sequence* (not a set or a count)
/// makes "renders something" insufficient: the stroke-only marks must land on
/// the same samples the filled mark does.
#[test]
fn temporal_color_renders_the_same_per_row_gradient_on_rule_and_segment_as_on_point() {
    let reference = colors_of(POINT_TEMPORAL).expect("point renders a temporal color gradient");
    assert_eq!(reference.len(), 4, "one element per row; got {reference:?}");
    assert!(
        reference.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the reference must itself be a gradient, else this pins nothing; got {reference:?}"
    );
    for (name, spec) in [("rule", RULE_TEMPORAL), ("segment", SEGMENT_TEMPORAL)] {
        let colors = colors_of(spec).unwrap_or_else(|e| {
            panic!(
                "{name}: a temporal color encoding must render, not refuse — `point` renders \
                 the identical encoding. Got {e:?}. This is the categorical-string-path \
                 forcing in `row_colors_from_scale`: it must dispatch on ColorScale::input()."
            )
        });
        assert_eq!(
            colors, reference,
            "{name}: per-row colors must match `point`'s for the identical encoding"
        );
    }
}

/// A non-finite row on a continuous color scale falls back to the mark's
/// constant paint, exactly as it does on `point` — it is not a data value and
/// must not be painted like one.
///
/// The pre-fix string path stringified `NaN` to `"NaN"`, which `f64::from_str`
/// accepts, so `normalize_continuous` propagated the NaN through `clamp` and
/// `sample` returned `#000000` — a color no scheme contains, indistinguishable
/// from a real datum. `resolve_fill_color` has carried an `is_finite` filter
/// for exactly this since it was written; this pins that the stroke-only
/// reader carries it too.
#[test]
fn non_finite_row_on_a_continuous_color_scale_takes_the_constant_paint_not_black() {
    let constant = colors_of(RULE_NO_COLOR).expect("uncolored rule renders");
    let fallback = constant.first().expect("uncolored rule paints its rows").clone();

    let rule = colors_of(RULE_NAN).expect("rule renders a quantitative color scale");
    let point = colors_of(POINT_NAN).expect("point renders a quantitative color scale");
    assert_eq!(rule.len(), 4, "one element per row; got {rule:?}");

    // Row 2 is the NaN row (`v_nan = [0.0, 1.0, NaN, 2.0]`).
    assert_ne!(
        rule[2], "#000000",
        "the NaN row must not paint pure black — that is `normalize_continuous(NaN)` \
         reaching the scheme, and it reads as data rather than as a gap. Got {rule:?}"
    );
    assert_eq!(
        rule[2], fallback,
        "the NaN row must take the same constant stroke an uncolored rule paints; got {rule:?}"
    );
    // The finite rows still sample the scale, and sample it exactly as point does.
    let finite: Vec<&String> = [0usize, 1, 3].iter().map(|&i| &rule[i]).collect();
    let point_finite: Vec<&String> = [0usize, 1, 3].iter().map(|&i| &point[i]).collect();
    assert_eq!(
        finite, point_finite,
        "the finite rows must sample the same gradient `point` samples; \
         rule={rule:?} point={point:?}"
    );
    assert!(
        finite.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the finite rows must be a gradient, else the NaN assertion pins nothing; got {rule:?}"
    );
}

/// The category-keyed counterpart, recorded as a policy row: a *nominal*
/// temporal color column is refused at **scale resolution**, uniformly, before
/// any mark builder runs.
///
/// This is what closes the "fatal here, silent there" question the numeric case
/// raised. `build_color_scale`'s categorical branch gates on
/// `ensure_category_keyable`, which admits exactly the dtypes both category
/// readers accept, so a column `row_colors_from_scale`'s category branch cannot
/// key on never reaches a mark builder — on any mark.
///
/// The fourth row is the shape that made the *old* version of this invariant
/// false while this test still passed: an explicit `scale.domain` supersedes
/// `distinct_values_in_order`, so a chart declaring `domain=` **and** `range=`
/// resolved a well-formed `Categorical` scale over a column nothing could key.
/// `rule`/`segment` then raised at mark build while `point`/`bar` swallowed the
/// same error and painted every element one color under a four-swatch legend.
/// Every spec here was bare until then, which is why the branch that skips the
/// builder went untested.
///
/// The wording assertion is what pins *where* the refusal happens: the resolver
/// says "cannot enumerate distinct values", the per-row reader says "cannot
/// convert to ordinal category string". Accepting either would let the fix
/// regress to a mark-build refusal — fatal on two marks, silent on the rest —
/// without failing this test.
#[test]
fn nominal_timestamp_color_is_refused_at_scale_resolution_for_every_mark() {
    for (name, spec) in [
        ("rule", RULE_NOMINAL_TS),
        ("point", POINT_NOMINAL_TS),
        ("line", LINE_NOMINAL_TS),
        ("point + explicit domain/range", POINT_NOMINAL_TS_EXPLICIT_RANGE),
    ] {
        let err = colors_of(spec).expect_err(
            "a nominal temporal color column has no category keys; it must refuse, \
             and it must refuse on every mark alike",
        );
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Timestamp"),
            "{name}: the refusal must name the offending dtype; got {msg}"
        );
        assert!(
            msg.contains("cannot enumerate distinct values"),
            "{name}: the refusal must come from scale resolution, not from a mark \
             builder's per-row read (which says `cannot convert to ordinal category \
             string` and is reached on only two of the marks); got {msg}"
        );
    }
}

/// The explicit-`domain`+`range` shape refuses *uniformly* — it does not paint
/// a legend the marks contradict.
///
/// The RED companion to the fourth row above, stated as the design review's
/// probe: before the `ensure_category_keyable` gate, `point` rendered four
/// circles all in the theme fill (`#2563eb`) beneath a legend carrying the four
/// declared range colors, while `rule` raised — one encoding, two outcome
/// classes, and the rendering one lying. This asserts both halves: `point` must
/// not render at all, and when both refuse it must be with the *same* refusal.
#[test]
fn explicit_domain_and_range_on_a_temporal_color_refuses_uniformly_rather_than_lying() {
    let point = colors_of(POINT_NOMINAL_TS_EXPLICIT_RANGE);
    let rule = colors_of(RULE_NOMINAL_TS_EXPLICIT_RANGE);

    match (point, rule) {
        (Err(point_err), Err(rule_err)) => assert_eq!(
            format!("{point_err:?}"),
            format!("{rule_err:?}"),
            "both marks must refuse the same way; a split here is the divergence \
             this shape reintroduces"
        ),
        (Ok(colors), rule) => {
            let distinct: std::collections::BTreeSet<&String> = colors.iter().collect();
            panic!(
                "point must refuse a nominal temporal color column, not render one: it \
                 painted {} element(s) in {} distinct color(s) ({colors:?}) while the \
                 legend enumerates the four declared range colors — NF-A3's signature. \
                 rule on the identical encoding: {}",
                colors.len(),
                distinct.len(),
                match rule {
                    Ok(c) => format!("rendered {c:?}"),
                    Err(e) => format!("refused with {e:?}"),
                }
            )
        }
        (Err(point_err), Ok(colors)) => panic!(
            "point refused ({point_err:?}) but rule rendered {colors:?} — the two marks \
             must reach the same outcome for one encoding"
        ),
    }
}

// ── The layer seam ──────────────────────────────────────────────────────────
//
// Everything above builds ONE mark from a flat spec. `mark_ribbon` never
// reaches a mark builder that way: `desugar_ribbon` lowers it to a chart with a
// single `ribbon` LAYER whose own encoding is `{x, y, y2}`, so the chart-level
// `color` arrives only through `Encoding::inherit_from` — and whether it
// survives to the builder is decided by `scene_build`'s own-color exemption,
// not by any reader in `marks/`. These rows render the whole chart for that
// reason, and they compare the bands against the LEGEND the same chart drew,
// because a mark contradicting its own legend is NF-A3's signature.

/// `fm.Chart(df).mark_ribbon().encode(x="x:Q", y="y:Q", y2="y2:Q",
/// color="c_utf8:N")` — the exact `Chart.to_spec().to_json()` lowering, layers
/// and inherited color channel included.
const RIBBON_UTF8: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"color":{"field":"c_utf8","type":"nominal"},"y2":{"field":"y2","type":"quantitative"}},"layers":[{"mark":"ribbon","encoding":{"x":{"field":"x"},"y":{"field":"y"},"y2":{"field":"y2"}},"mark_style":{"stroke":"none","opacity":0.3},"name":"ribbon"}]}"#;
/// The same chart on the `Int64` color column.
const RIBBON_I64: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"color":{"field":"c_i64","type":"nominal"},"y2":{"field":"y2","type":"quantitative"}},"layers":[{"mark":"ribbon","encoding":{"x":{"field":"x"},"y":{"field":"y"},"y2":{"field":"y2"}},"mark_style":{"stroke":"none","opacity":0.3},"name":"ribbon"}]}"#;
/// `mark_area` on the identical data and channels — the reference the intent
/// review used, and a flat (unlayered) lowering, so it exercises the seam the
/// ribbon rows do not.
const AREA_UTF8: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"area","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"color":{"field":"c_utf8","type":"nominal"},"y2":{"field":"y2","type":"quantitative"}}}"#;
const AREA_I64: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"area","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"color":{"field":"c_i64","type":"nominal"},"y2":{"field":"y2","type":"quantitative"}}}"#;

/// Every `fill="…"` value carried by an element of the given SVG tag, in
/// document order, as an `#rrggbb` triple with any alpha dropped.
///
/// Alpha is dropped because the band and its legend swatch deliberately differ
/// in it (the band is translucent, the swatch is not) while the hue is the
/// thing that must agree.
fn fills_of_tag(svg: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag} ");
    svg.split(&open)
        .skip(1)
        .filter_map(|rest| {
            let element = rest.split('>').next()?;
            let value = element.split("fill=\"").nth(1)?.split('"').next()?;
            parse_fill_rgb(value)
        })
        .collect()
}

/// `#rrggbb`, `#rrggbbaa` and `rgba(r,g,b,a)` → `#rrggbb`. Anything else
/// (notably `"none"`) is not a color and is skipped.
fn parse_fill_rgb(value: &str) -> Option<String> {
    if let Some(hex) = value.strip_prefix('#') {
        return (hex.len() >= 6).then(|| format!("#{}", &hex[..6]));
    }
    let inner = value.strip_prefix("rgba(").or_else(|| value.strip_prefix("rgb("))?;
    let inner = inner.strip_suffix(')')?;
    let mut parts = inner.split(',');
    let mut channel = || parts.next()?.trim().parse::<u8>().ok();
    let (r, g, b) = (channel()?, channel()?, channel()?);
    Some(format!("#{r:02x}{g:02x}{b:02x}"))
}

/// Render a lowered spec against [`parity_batch`] and return
/// `(mark path fills, legend swatch fills)`.
///
/// The legend swatches are the `<circle>` elements: `render_svg` draws a
/// categorical color legend as one circle + one label per domain entry, and
/// none of the marks in this section draw circles of their own.
fn ribbon_bands_and_legend(spec_json: &str) -> (Vec<String>, Vec<String>) {
    let spec: ChartSpec = serde_json::from_str(spec_json).expect("lowered spec parses");
    let batch = parity_batch();
    let theme = ThemeInputs::default();
    let result = crate::render::render_svg(
        &spec,
        &batch,
        &theme,
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("a nominal color column on ribbon/area must render, not refuse");
    (fills_of_tag(&result.bytes, "path"), fills_of_tag(&result.bytes, "circle"))
}

/// NF-A3 (ribbon half): a categorical `color` on `mark_ribbon` draws one band
/// per category, each in that category's legend color — for `Utf8` and `Int64`
/// alike, and identically to what `mark_area` draws from the same data.
///
/// RED before the fix, both dtypes: `mark_ribbon` emitted ONE merged
/// self-intersecting band in the theme default fill under a two-swatch legend,
/// because `desugar_ribbon`'s `stroke="none"` set `stroke_is_user_set` and so
/// tripped `scene_build`'s own-color exemption, which strips an inherited color
/// channel from any layer carrying its own literal paint. A paint *clear* is
/// not a paint the inherited color could overwrite, so it must not trip it.
///
/// The legend comparison is what makes this discriminating: counting bands
/// alone would pass a ribbon that split into two bands of the same color, and
/// the merged-band defect was invisible precisely because the legend enumerated
/// every category regardless of what drew.
#[test]
fn ribbon_paints_one_legend_colored_band_per_category_like_area() {
    for (dtype, ribbon_spec, area_spec) in [
        ("Utf8", RIBBON_UTF8, AREA_UTF8),
        ("Int64", RIBBON_I64, AREA_I64),
    ] {
        let (area_bands, area_swatches) = ribbon_bands_and_legend(area_spec);
        assert_eq!(
            area_bands, area_swatches,
            "{dtype}: the `mark_area` reference must itself paint one band per legend \
             swatch, in swatch order, else this row proves nothing; \
             bands={area_bands:?} swatches={area_swatches:?}"
        );

        let (bands, swatches) = ribbon_bands_and_legend(ribbon_spec);
        assert_eq!(
            swatches.len(),
            2,
            "{dtype}: the legend must enumerate both categories; got {swatches:?}"
        );
        assert_eq!(
            bands, swatches,
            "{dtype}: `mark_ribbon` must draw one band per category, each in that \
             category's legend color (bands={bands:?}, legend={swatches:?}). A single \
             band here is the NF-A3 defect: the ribbon LAYER's inherited color channel \
             was stripped at the `scene_build` seam, so every category merged into one \
             polygon in the theme default fill while the legend still listed both."
        );
        assert_eq!(
            bands, area_bands,
            "{dtype}: ribbon must reach the same partition `mark_area` reaches from the \
             identical data and channels; ribbon={bands:?} area={area_bands:?}"
        );
    }
}

/// Byte-identity guard (spec §7) for the case the fix does not authorize: a
/// ribbon with NO color encoding still renders exactly one band, and its SVG is
/// unchanged by the presence of the `stroke="none"` the desugar always passes.
///
/// The exemption's strip is reachable only when a chart-level color channel
/// exists to inherit, so an uncolored ribbon must be untouched — this pins that
/// directly rather than trusting the argument.
#[test]
fn uncolored_ribbon_still_draws_exactly_one_band() {
    const RIBBON_NO_COLOR: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"y2":{"field":"y2","type":"quantitative"}},"layers":[{"mark":"ribbon","encoding":{"x":{"field":"x"},"y":{"field":"y"},"y2":{"field":"y2"}},"mark_style":{"stroke":"none","opacity":0.3},"name":"ribbon"}]}"#;
    let (bands, swatches) = ribbon_bands_and_legend(RIBBON_NO_COLOR);
    assert_eq!(bands.len(), 1, "no color encoding → one merged band; got {bands:?}");
    assert!(swatches.is_empty(), "no color encoding → no legend swatches; got {swatches:?}");
}
