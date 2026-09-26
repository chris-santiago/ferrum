use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Per-mark constant style overrides. Phase 8a fields cover all kwargs accepted
/// by the 8 primitive mark_*() Python methods. All None defaults; renderer falls
/// back to theme defaults when None.
///
/// Resolution priority (`#119`, 2026-09-23, one uniform rule): a flat chart
/// resolves `chart.mark_style > theme`; a layered chart resolves
/// `layer.mark_style > theme` — a layer never falls back to a chart-level
/// `mark_style`, paint or otherwise. `render/prepare/mod.rs`'s
/// `LayerPrepared::from_chart_and_layer` reads `layer.mark_style` alone;
/// `ChartSpec.mark_style` is read only by `LayerPrepared::from_chart_only`,
/// the flat (no-`layers`) path. A layer's style is its own.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MarkKwargsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_dash: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dx: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
    /// Polygon mark grouping column (e.g. `hex_id`, `violin_id`, `level_id`).
    /// When None, all rows form a single polygon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Named continuous colormap (e.g. `viridis`, `plasma`). Used by polygon mark
    /// when `color` encoding maps to a quantitative column. Defaults to `viridis`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cmap: Option<String>,
    // ── S1: interpolate (line/area/ribbon) ───────────────────────────────────
    /// Path interpolation method. Supported: "linear" (default), "step",
    /// "step-before", "step-after", "monotone", "basis", "cardinal" — the
    /// vocabulary owned by `render::marks::curve::Interpolate`, which is also
    /// the single source for the refusal message. (Named rather than
    /// intra-doc-linked: this struct is `pub` and that enum is `pub(crate)`, so
    /// the link would not resolve for a reader of the public docs.) An
    /// unrecognized value is REFUSED (`RenderError::UnknownInterpolate`) at
    /// `render::draw::resolve_mark_style`; it does not fall back to linear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolate: Option<String>,
    // ── S2: stroke_cap (line) ────────────────────────────────────────────────
    /// SVG stroke-linecap. Values: "butt" (default), "round", "square".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_cap: Option<String>,
    // ── S3: stroke_join (line/area) ──────────────────────────────────────────
    /// SVG stroke-linejoin. Values: "miter" (default), "round", "bevel".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_join: Option<String>,
    // ── S5: filled (point) ───────────────────────────────────────────────────
    /// When false, points are hollow: fill="none", color applied to stroke.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filled: Option<bool>,
    // ── S6: shape (point, constant) ──────────────────────────────────────────
    /// Constant point shape when shape encoding is absent.
    /// Values: "circle", "square", "diamond", "triangle-up", "cross", "triangle-down".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    // ── S7: limit (text) ─────────────────────────────────────────────────────
    /// Max character length for text labels; truncates with "…" if exceeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    // ── S8: band_size (tick/rect) ────────────────────────────────────────────
    /// Rendered mark length as a fraction of the band width, uniformly a
    /// FULL-length factor across both consumers (GH #85): tick's crossbar/
    /// median/cap length and rect's boxplot IQR width are both
    /// `band_extent * band_size`. Default `0.6` for both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_size: Option<f64>,
    // ── S9: line (area) ──────────────────────────────────────────────────────
    /// When true, draw an additional line border on top of the area fill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<bool>,
    // ── S10: borders (area/errorband) ────────────────────────────────────────
    /// When true, draw border lines on both top and bottom edges of an area.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub borders: Option<bool>,
    // ── S11: leader_line (label) ─────────────────────────────────────────────
    /// When true, draw a thin leader line from each data point (px, py) to the
    /// placed label position. Default false (None = no leader line).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leader_line: Option<bool>,
    // ── mark_image URL-tile sizing ───────────────────────────────────────────
    /// Constant tile width in pixels for mark_image URL tiles. Overridden
    /// per-row by a `width` column in the data. Defaults to 32.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    /// Constant tile height in pixels for mark_image URL tiles. Overridden
    /// per-row by a `height` column in the data. Defaults to 32.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    // ── batch E Task 9: clip escape (mark_function) ──────────────────────────
    /// When `false`, the batch renders outside the panel's clip region
    /// instead of being confined to it (`MarkBatch::clip`, spec §6.3).
    /// `None`/unset resolves to the theme default `true` (clipped) —
    /// byte-stable, matching every mark before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<bool>,
    // ── T5r cycle 2: zero (bar, layered zero-anchor opt-out) ─────────────────
    /// Per-layer mirror of the single-chart `mark_bar(zero=False)` /
    /// `MarkBase.zero_anchor()` concept, consulted by
    /// `render::scale_resolve::domain::numeric_domain_union`'s bar
    /// zero-anchor rule (T5r). `None`/unset resolves to `true` (anchor, when
    /// otherwise eligible — value channel, no bound span field) — byte-stable,
    /// matching every layer before this field existed. `Some(false)` opts a
    /// `Mark::Bar` layer out of the anchor even when eligible, mirroring
    /// `_spec_build.py`'s D3 `not _y2_bound`-style suppression but keyed on
    /// an explicit request rather than a span binding.
    ///
    /// Live and user-reachable today via the generic per-layer `mark=`
    /// passthrough on layered figure functions — e.g.
    /// `fm.catplot(df, x=..., y=..., kind="bar", mark={"bar": {"zero":
    /// False}})` resolves the value axis to the raw data range (`58..61.5`
    /// for data in that range), not `0..61.5`. That passthrough
    /// (`marks/_mark_kwargs.py::apply_user_mark_kwargs`) merges validated
    /// user kwargs straight into a layer's `mark_kwargs` dict, bypassing
    /// `MarkBase.to_mark_kwargs_dict`'s chart-level exclusion entirely — the
    /// ONE path that stays inert is `Chart.mark_bar(zero=False)` composed
    /// via `+` into a layer (`_mark_zero` is consumed and dropped before
    /// that chart becomes a layer, predating this field), which this field
    /// does not change and was never meant to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero: Option<bool>,
    // ── Batch H Task 1: arrow / arrow_size (segment) ─────────────────────────
    /// When `true`, `mark_segment` draws a filled triangle arrowhead at each
    /// segment's `(x2, y2)` end, colored by the segment's resolved stroke and
    /// sized by `arrow_size` (spec §4 H1, `ferrum-spec.md:690`). `None`/unset
    /// resolves to `false` at `render::draw::resolve_mark_style` — byte-stable,
    /// matching every segment rendered before this field existed. The
    /// triangle geometry is shared with `annotate_arrow`'s unlabelled overlay
    /// path (`render::marks::arrowhead::arrowhead_triangle`), extracted from
    /// `render::annotation::emit_arrow`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrow: Option<bool>,
    /// Arrowhead size in pixels when `arrow=true`. `None`/unset resolves to
    /// `8.0` — the same default `annotate_arrow`'s unlabelled overlay uses —
    /// at the draw site, clamped to the segment's own pixel length so a very
    /// short segment cannot produce a head that overshoots its start point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrow_size: Option<f64>,
    // ── `#172`(2): declared value axis (line family) ─────────────────────────
    /// Which positional channel carries this mark's VALUE (the dependent
    /// variable), when the mark's producer put it somewhere `CoordFlip` and
    /// the position adjustment cannot reveal.
    ///
    /// `Some(X)` means the value is on `x`, so the series advances along `y` —
    /// a horizontal series. `Some(Y)` means the ordinary vertical shape.
    /// `None`/unset means "undeclared": the coordinate system and the stack
    /// stamp decide alone, exactly as before this field existed, so every spec
    /// that omits it renders byte-identically.
    ///
    /// Read by [`DrawCtx::series_curve_axis`](crate::render::draw::DrawCtx),
    /// the one place that answers "which coordinate does this series advance
    /// along", so `mark_line`, `mark_area` and every other consumer of that
    /// answer get it from one rule.
    ///
    /// **Why this lives on mark kwargs rather than inside
    /// [`PositionAdjust::Stack`](crate::spec::position::PositionAdjust).**
    /// GH #77 put the identical declaration there because `apply_stack` was
    /// the only consumer that needed it. `mark_density(orientation=
    /// "horizontal")` at its default `multiple="layer"` emits NO position
    /// adjustment at all, so it has no `Stack` to state it in — and its spec is
    /// otherwise indistinguishable from an ordinary vertical area whose rows
    /// arrived unsorted (no `coord`, no `position`, `x` and `y` both
    /// quantitative). `#172`(2)'s first attempt inferred the axis from the
    /// traced edge's monotonicity instead; that misread an unsorted vertical
    /// area as horizontal, which is a silent wrong render, so the inference was
    /// deleted in favour of this declaration. Mark kwargs are the right home
    /// because they already resolve per LAYER (`layer.mark_style > theme` —
    /// a layered chart never falls back to a chart-level `mark_style`, `#119`
    /// 2026-09-23) — a `ChartSpec`-level field would be inherited wholesale
    /// by every layer through `scene_build`'s `..spec.clone()` — and because
    /// a desugar already has a slot for exactly this kind of per-mark
    /// declaration (`MarkDesugarResult.mark_kwargs`, which `desugar_function`
    /// already uses for `clip=False`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_axis: Option<crate::spec::position::StackValueAxis>,
}

/// Every `MarkKwargsSpec` wire key, in declaration order: the field list
/// serde derives for the struct. Read by handing the derived `Deserialize` a
/// deserializer that records the `fields` serde passes to
/// `deserialize_struct` and stops, so the list cannot drift from the struct.
pub(crate) fn mark_kwargs_field_names() -> &'static [&'static str] {
    use serde::de::{Error, Visitor};

    struct FieldTracer<'a>(&'a mut &'static [&'static str]);

    impl<'de> serde::Deserializer<'de> for FieldTracer<'_> {
        type Error = serde::de::value::Error;

        fn deserialize_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Self::Error> {
            Err(Error::custom("FieldTracer reads a struct's field names only"))
        }

        fn deserialize_struct<V: Visitor<'de>>(
            self,
            _name: &'static str,
            fields: &'static [&'static str],
            _visitor: V,
        ) -> Result<V::Value, Self::Error> {
            *self.0 = fields;
            Err(Error::custom("field names recorded"))
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes
            byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct map enum
            identifier ignored_any
        }
    }

    let mut fields: &'static [&'static str] = &[];
    // The tracer always errs once it has the list; the list is the result.
    let _ = MarkKwargsSpec::deserialize(FieldTracer(&mut fields));
    fields
}

/// `ferrum._core.mark_kwargs_fields()`: [`mark_kwargs_field_names`], so
/// Python reads the mark-kwargs wire keys instead of parsing this file.
/// Registered in `lib.rs`, typed in `src/ferrum/_core.pyi`.
#[pyfunction]
pub(crate) fn mark_kwargs_fields() -> Vec<String> {
    mark_kwargs_field_names().iter().map(|f| f.to_string()).collect()
}

/// Every numeric (`f64`/`usize`) `MarkKwargsSpec` field, in declaration
/// order. Derived from the struct, not listed: a field is numeric exactly
/// when its derived `Deserialize` accepts a bare number under its key, which
/// every `Option<f64>`/`Option<usize>` field does and every string, bool,
/// pattern-list or enum field refuses.
pub(crate) fn mark_numeric_kwarg_names() -> Vec<&'static str> {
    mark_kwargs_field_names()
        .iter()
        .copied()
        .filter(|name| {
            serde_json::from_value::<MarkKwargsSpec>(serde_json::json!({ *name: 1 })).is_ok()
        })
        .collect()
}

/// `ferrum._core.mark_numeric_kwargs()`: [`mark_numeric_kwarg_names`], so
/// Python's mark-kwarg funnel coerces exactly the struct's numeric fields
/// instead of hand-listing them. Registered in `lib.rs`, typed in
/// `src/ferrum/_core.pyi`.
#[pyfunction]
pub(crate) fn mark_numeric_kwargs() -> Vec<&'static str> {
    mark_numeric_kwarg_names()
}

/// The `Option<usize>` subset of [`mark_numeric_kwarg_names`], in declaration
/// order. Derived from the numeric set itself, not listed: a numeric field is
/// an integer field exactly when its derived `Deserialize` refuses a
/// non-integer number (`1.5`) under its key, which every `Option<usize>`
/// field does and every `Option<f64>` field accepts. This guarantees the
/// integer and real subsets partition the numeric fields exactly — neither
/// list is hand-maintained against the other.
pub(crate) fn mark_integer_kwarg_names() -> Vec<&'static str> {
    mark_numeric_kwarg_names()
        .into_iter()
        .filter(|name| {
            serde_json::from_value::<MarkKwargsSpec>(serde_json::json!({ *name: 1.5 })).is_err()
        })
        .collect()
}

/// `ferrum._core.mark_integer_kwargs()`: [`mark_integer_kwarg_names`], so
/// Python's mark-kwarg funnel routes exactly the struct's `usize` fields
/// through a non-negative-integer contract instead of the real-number one it
/// gives every other numeric field. Registered in `lib.rs`, typed in
/// `src/ferrum/_core.pyi`.
#[pyfunction]
pub(crate) fn mark_integer_kwargs() -> Vec<&'static str> {
    mark_integer_kwarg_names()
}

#[cfg(test)]
mod tests {
    /// The numeric set is exactly the struct's `Option<f64>`/`Option<usize>`
    /// fields. The exhaustive destructure makes a new field fail to compile
    /// here until it is classified below.
    #[test]
    fn mark_numeric_kwarg_names_are_the_structs_numeric_fields() {
        let super::MarkKwargsSpec {
            size: _,
            stroke: _,
            fill: _,
            opacity: _,
            corner_radius: _,
            stroke_width: _,
            stroke_dash: _,
            font_size: _,
            font_weight: _,
            align: _,
            baseline: _,
            dx: _,
            dy: _,
            angle: _,
            detail: _,
            cmap: _,
            interpolate: _,
            stroke_cap: _,
            stroke_join: _,
            filled: _,
            shape: _,
            limit: _,
            band_size: _,
            line: _,
            borders: _,
            leader_line: _,
            width: _,
            height: _,
            clip: _,
            zero: _,
            arrow: _,
            arrow_size: _,
            value_axis: _,
        } = super::MarkKwargsSpec::default();
        assert_eq!(
            super::mark_numeric_kwarg_names(),
            vec![
                "size",
                "opacity",
                "corner_radius",
                "stroke_width",
                "font_size",
                "dx",
                "dy",
                "angle",
                "limit",
                "band_size",
                "width",
                "height",
                "arrow_size",
            ]
        );
    }

    /// The integer subset is exactly `limit` (the struct's one `Option<usize>`
    /// field) today, and it partitions the numeric fields exactly: every
    /// numeric name is in the integer list or the real list, never both and
    /// never neither.
    #[test]
    fn mark_integer_kwarg_names_partition_the_numeric_fields() {
        let numeric = super::mark_numeric_kwarg_names();
        let integer = super::mark_integer_kwarg_names();
        assert_eq!(integer, vec!["limit"]);
        for name in &integer {
            assert!(numeric.contains(name), "'{name}' must be numeric to be an integer subset");
        }
        let real: Vec<&str> =
            numeric.iter().copied().filter(|n| !integer.contains(n)).collect();
        assert_eq!(
            real,
            vec![
                "size",
                "opacity",
                "corner_radius",
                "stroke_width",
                "font_size",
                "dx",
                "dy",
                "angle",
                "band_size",
                "width",
                "height",
                "arrow_size",
            ]
        );
        assert_eq!(real.len() + integer.len(), numeric.len());
    }

    /// A float or a negative value under `limit`'s key is refused by the
    /// struct's own `Deserialize`, which is the render-time floor
    /// `mark_integer_kwargs()` exists to move earlier: Python routes these
    /// through a non-negative-integer contract at declaration instead of
    /// letting `usize`'s deserializer refuse them at render.
    #[test]
    fn limit_refuses_non_integer_and_negative_values() {
        assert!(
            serde_json::from_value::<super::MarkKwargsSpec>(serde_json::json!({ "limit": 2.5 }))
                .is_err()
        );
        assert!(
            serde_json::from_value::<super::MarkKwargsSpec>(serde_json::json!({ "limit": -1 }))
                .is_err()
        );
        assert!(
            serde_json::from_value::<super::MarkKwargsSpec>(serde_json::json!({ "limit": 2 }))
                .is_ok()
        );
    }

    /// Every traced name is a real field: a wrong-typed value under it is
    /// refused, where an unknown key would be ignored (the struct does not
    /// deny unknown fields). The reverse holds by construction: the list is
    /// serde's own. `size`/`value_axis` bracket the declaration order.
    #[test]
    fn mark_kwargs_field_names_are_the_structs_fields() {
        let names = super::mark_kwargs_field_names();
        assert_eq!(names.first(), Some(&"size"));
        assert_eq!(names.last(), Some(&"value_axis"));
        for name in names {
            let json = serde_json::json!({ *name: [[[]]] });
            assert!(
                serde_json::from_value::<super::MarkKwargsSpec>(json).is_err(),
                "'{name}' is not a MarkKwargsSpec field"
            );
        }
        let unknown = serde_json::json!({ "not_a_field": [[[]]] });
        assert!(serde_json::from_value::<super::MarkKwargsSpec>(unknown).is_ok(), "control");
        assert_eq!(super::mark_kwargs_fields(), names.iter().map(|n| n.to_string()).collect::<Vec<_>>());
    }

    use super::*;
    use crate::spec::position::StackValueAxis;

    // ── `#172`(2) wire contract: `value_axis` on mark kwargs ────────────────
    //
    // Mirrors `spec/position.rs`'s own GH #77 trio
    // (`stack_value_axis_x_round_trips_and_serializes`,
    // `stack_value_axis_absent_when_none`,
    // `stack_value_axis_defaults_to_none_when_absent_from_json`), because this
    // is the same declaration in a second slot and the two must stay spelled
    // identically. Byte-stability is proven by round-tripping a PRE-CHANGE
    // payload, not by reading the serde attributes.

    /// `Some(X)` serializes as a present `"value_axis":"x"` key and round-trips.
    #[test]
    fn value_axis_x_round_trips_and_serializes() {
        let m = MarkKwargsSpec {
            value_axis: Some(StackValueAxis::X),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(r#""value_axis":"x""#), "got {json}");
        assert_eq!(serde_json::from_str::<MarkKwargsSpec>(&json).unwrap(), m);
    }

    /// `Some(Y)` spells `"y"` — the same two values `Stack`'s slot accepts, so
    /// a producer choosing between the slots never has to change the value.
    #[test]
    fn value_axis_y_spells_the_same_token_stacks_slot_does() {
        let m = MarkKwargsSpec {
            value_axis: Some(StackValueAxis::Y),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(r#""value_axis":"y""#), "got {json}");
        assert_eq!(serde_json::from_str::<MarkKwargsSpec>(&json).unwrap(), m);
    }

    /// `None` is absent from the wire entirely, so every spec written before
    /// this field existed serializes to the bytes it always did.
    #[test]
    fn value_axis_absent_when_none() {
        let m = MarkKwargsSpec { fill: Some("#123456".into()), ..Default::default() };
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("value_axis"), "got {json}");
        assert_eq!(serde_json::from_str::<MarkKwargsSpec>(&json).unwrap(), m);
    }

    /// The byte-stability proof proper: a payload captured BEFORE this field
    /// existed still parses, leaves `value_axis` at `None`, and re-serializes
    /// to the identical bytes. A `#[serde(default)]` that had been forgotten
    /// would fail the parse; a missing `skip_serializing_if` would fail the
    /// re-serialization.
    #[test]
    fn a_pre_change_payload_round_trips_to_identical_bytes() {
        const PRE_CHANGE: &str =
            r##"{"size":100.0,"fill":"#2563eb","opacity":0.7,"interpolate":"step","clip":false}"##;
        let parsed: MarkKwargsSpec = serde_json::from_str(PRE_CHANGE).unwrap();
        assert!(parsed.value_axis.is_none(), "an absent key must resolve None, not a default axis");
        assert_eq!(serde_json::to_string(&parsed).unwrap(), PRE_CHANGE);
    }

    #[test]
    fn mark_kwargs_default_omits_all_fields() {
        let m = MarkKwargsSpec::default();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn mark_kwargs_round_trip_with_size_and_stroke() {
        let m = MarkKwargsSpec {
            size: Some(100.0),
            stroke: Some("#ff0000".into()),
            opacity: Some(0.5),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn mark_kwargs_round_trip_with_stroke_dash() {
        let m = MarkKwargsSpec {
            stroke_dash: Some(vec![5.0, 3.0]),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(r#""stroke_dash":[5.0,3.0]"#));
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    // ── Batch E Task 9: clip escape (`mark_function(clip=False)`) ──────────

    #[test]
    fn mark_kwargs_default_clip_is_none_and_omitted() {
        // Covered by `mark_kwargs_default_omits_all_fields` too, but this
        // pins the specific field so a future refactor of that test can't
        // silently stop covering `clip`.
        let m = MarkKwargsSpec::default();
        assert_eq!(m.clip, None);
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("clip"), "unset clip must be omitted: {json}");
    }

    #[test]
    fn mark_kwargs_round_trip_with_clip_false() {
        let m = MarkKwargsSpec { clip: Some(false), ..Default::default() };
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, r#"{"clip":false}"#);
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    // ── T5r cycle 2: zero (bar, layered zero-anchor opt-out) ────────────────

    #[test]
    fn mark_kwargs_default_zero_is_none_and_omitted() {
        // Covered by `mark_kwargs_default_omits_all_fields` too, but this
        // pins the specific field so a future refactor of that test can't
        // silently stop covering `zero`.
        let m = MarkKwargsSpec::default();
        assert_eq!(m.zero, None);
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("zero"), "unset zero must be omitted: {json}");
    }

    #[test]
    fn mark_kwargs_round_trip_with_zero_false() {
        let m = MarkKwargsSpec { zero: Some(false), ..Default::default() };
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, r#"{"zero":false}"#);
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn mark_kwargs_pre_existing_layer_json_without_zero_key_deserializes_default() {
        // Absolute wire back-compat: a layer's `mark_style` dict from before
        // this field existed (no "zero" key at all) must deserialize to
        // `zero: None`, not an error and not a spurious `Some(_)`.
        let json = r#"{"size":50.0}"#;
        let m: MarkKwargsSpec = serde_json::from_str(json).unwrap();
        assert_eq!(m.zero, None);
        assert_eq!(m.size, Some(50.0));
    }

    // ── Batch H Task 1: arrow / arrow_size (segment) ─────────────────────────

    #[test]
    fn mark_kwargs_default_arrow_fields_are_none_and_omitted() {
        // Covered by `mark_kwargs_default_omits_all_fields` too, but this
        // pins the specific fields so a future refactor of that test can't
        // silently stop covering them.
        let m = MarkKwargsSpec::default();
        assert_eq!(m.arrow, None);
        assert_eq!(m.arrow_size, None);
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("arrow"), "unset arrow/arrow_size must be omitted: {json}");
    }

    #[test]
    fn mark_kwargs_round_trip_with_arrow_true_and_size() {
        let m = MarkKwargsSpec { arrow: Some(true), arrow_size: Some(12.0), ..Default::default() };
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, r#"{"arrow":true,"arrow_size":12.0}"#);
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn mark_kwargs_pre_existing_layer_json_without_arrow_keys_deserializes_default() {
        // Same wire back-compat guarantee as `zero`: a `mark_style` dict from
        // before this field existed must deserialize to `arrow: None`,
        // `arrow_size: None`, not an error and not spurious `Some(_)`.
        let json = r##"{"stroke":"#ff0000"}"##;
        let m: MarkKwargsSpec = serde_json::from_str(json).unwrap();
        assert_eq!(m.arrow, None);
        assert_eq!(m.arrow_size, None);
        assert_eq!(m.stroke.as_deref(), Some("#ff0000"));
    }
}
