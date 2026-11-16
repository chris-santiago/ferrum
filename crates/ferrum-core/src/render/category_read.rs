//! The one rule for whether a binding reads a `Timestamp` column as categories.
//!
//! A date column is a time axis unless something says it names categories. The
//! rule lives in [`timestamp_read`] and nowhere else: every reader that decides
//! "category versus time" for a date asks it, and the raw per-row and
//! per-domain readers in [`super::arrow_cast`] are private so no reader can
//! bypass it. Readers that only partition rows by value (a colour or dash key
//! lookup, a dodge or stack bin, an impute series) never choose a scale and
//! read every date as its label through
//! [`super::arrow_cast::col_as_identity_key_str`].
//!
//! Under [`TimestampRead::Temporal`] the category doors return exactly the
//! errors a `Timestamp` column drew before dates could be categories, so every
//! builder's existing error handling (its empty path, a census of zero, a
//! skipped slot, a refusal) reproduces that behavior unchanged. Under
//! [`TimestampRead::Categories`] a date reads as its per-value label
//! ([`super::arrow_cast::timestamp_category_label`]).
//!
//! [`CategoryDomain`] is the ordered category list every union builds: each
//! label carries its instant, so dates come out in time order however many
//! frames contribute them.

use std::collections::HashSet;

use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;

use super::RenderError;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::{DataType as Declared, EncodingSpec, ScaleSpec};
use crate::spec::mark::Mark;

pub(crate) use super::arrow_cast::{category_domain_entries, category_strings};

/// How a binding reads a `Timestamp` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimestampRead {
    /// One category per distinct instant, labelled per value, in time order.
    Categories,
    /// A point on a time axis: the column names no category.
    Temporal,
}

/// The channels whose bindings read categories whatever their declared type.
fn channel_always_reads_categories(channel: &str) -> bool {
    matches!(channel, "shape" | "stroke_dash" | "facet")
}

/// A categorical caller scale on `binding`: a band, point or ordinal scale, or
/// a stated ordinal domain.
fn carries_categorical_scale(binding: &EncodingSpec) -> bool {
    binding.scale.as_ref().is_some_and(|scale| {
        !scale.is_system_injected() && matches!(scale, ScaleSpec::Ordinal { .. } | ScaleSpec::Band { .. } | ScaleSpec::Point { .. })
    }) || super::scale_resolve::explicit_ordinal_domain(binding).is_some()
}

/// Whether `binding` on `channel` of a `mark` layer reads a `Timestamp` column
/// as categories. `x2`/`y2` are read through their primary channel's binding
/// and scale (they share its axis), and `theta`/`radius` count as positional.
///
/// Categories exactly when
/// - (a) the channel always reads categories (`shape`, `stroke_dash`, a facet
///   field, row or column), or
/// - (b) the binding declares nominal or ordinal, or
/// - (c) the binding carries a categorical caller scale (a band, point or
///   ordinal scale, or a stated ordinal domain), or
/// - (d) the channel is `color`, `fill` or `stroke` and the mark is an area
///   (an area groups its colour by category), or
/// - (e) `scale_of_dates`: the panel scale on that channel is a date-category
///   scale, so an instant on it names a category whatever the binding says.
///
/// Temporal otherwise.
pub(crate) fn timestamp_read(
    channel: &str,
    binding: Option<&EncodingSpec>,
    mark: Mark,
    scale_of_dates: bool,
) -> TimestampRead {
    let categories = channel_always_reads_categories(channel)
        || scale_of_dates
        || binding.is_some_and(|binding| {
            matches!(binding.type_, Some(Declared::Nominal | Declared::Ordinal))
                || carries_categorical_scale(binding)
                || (mark == Mark::Area && matches!(channel, "color" | "fill" | "stroke"))
        });
    if categories { TimestampRead::Categories } else { TimestampRead::Temporal }
}

/// One category list: each label with the instant it names, when it names one.
///
/// [`Self::labels`] gives dates in time order. Text keeps first appearance,
/// and a date that joins a text list takes the date slots it occupies, so a
/// text axis a declared date joins keeps its text first.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CategoryDomain {
    entries: Vec<(String, Option<i128>)>,
    seen: HashSet<String>,
    of_dates: bool,
}

impl CategoryDomain {
    /// A list of plain labels, none of them an instant.
    pub(crate) fn from_labels(labels: impl IntoIterator<Item = String>) -> Self {
        let mut domain = Self::default();
        for label in labels {
            domain.push(label, None);
        }
        domain
    }

    /// The categories of one column read under `read`: dates in time order,
    /// every other dtype in first appearance. A `Timestamp` under
    /// [`TimestampRead::Temporal`] is the refusal the column drew before dates
    /// could be categories. The domain is `of_dates` when the column is a date
    /// read as categories.
    pub(crate) fn from_column(batch: &RecordBatch, field: &str, read: TimestampRead) -> Result<Self, RenderError> {
        let entries = category_domain_entries(batch, field, read)?;
        let of_dates = entries.iter().any(|(_, instant)| instant.is_some());
        let mut domain = Self::default();
        for (label, instant) in entries {
            domain.push(label, instant);
        }
        domain.of_dates = of_dates;
        Ok(domain)
    }

    fn push(&mut self, label: String, instant: Option<i128>) {
        if self.seen.insert(label.clone()) {
            self.entries.push((label, instant));
        }
    }

    /// Adds `other`'s categories after this list's, skipping labels already
    /// present. The date fact stays this list's own: it records what
    /// established the axis.
    pub(crate) fn join(&mut self, other: &CategoryDomain) {
        for (label, instant) in &other.entries {
            self.push(label.clone(), *instant);
        }
    }

    /// Marks the list as the categories of a date axis (or not).
    pub(crate) fn with_of_dates(mut self, of_dates: bool) -> Self {
        self.of_dates = of_dates;
        self
    }

    /// Whether the axis this list serves was established by dates.
    pub(crate) fn of_dates(&self) -> bool {
        self.of_dates
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The labels in axis order: the dated entries sorted by instant into the
    /// slots dated entries occupy, every other entry where it stands.
    pub(crate) fn labels(&self) -> Vec<String> {
        let mut dated: Vec<&(String, Option<i128>)> = self.entries.iter().filter(|(_, instant)| instant.is_some()).collect();
        dated.sort_by_key(|(_, instant)| *instant);
        let mut dated = dated.into_iter();
        self.entries
            .iter()
            .map(|entry| if entry.1.is_some() { dated.next().unwrap_or(entry).0.clone() } else { entry.0.clone() })
            .collect()
    }
}

/// Whether a read of `binding` on `channel` of a `mark` layer, over a column
/// that is a `Timestamp` or not, makes the scale it establishes one of dates:
/// the column is a date and the binding reads it as categories with no help
/// from the scale (clause (e) never establishes itself). The one statement of
/// the fact [`crate::scale::ordinal::OrdinalScale::of_dates`] records and the
/// published binding set reads.
pub(crate) fn establishes_dates(channel: &str, binding: &EncodingSpec, mark: Mark, column_is_timestamp: bool) -> bool {
    column_is_timestamp && timestamp_read(channel, Some(binding), mark, false) == TimestampRead::Categories
}

/// Every `(data_source, channel, field)` of a built chart on which
/// [`timestamp_read`] answers Categories, for Python's refusal of a Duration
/// (which crosses as a timestamp) on a category binding.
///
/// `data_source` is the layer's `data_source` or `None` for the chart-level
/// encoding and a facet; `channel` is the spec channel name, a facet field,
/// row or column `"facet"`; `field` is the spec's own spelling. Clause (e) is
/// the engine's own rule: a positional channel's scale is of dates when the
/// read that establishes its axis
/// ([`crate::render::scale_resolve::establishing_axis_read`]) is a date column
/// of `schema` (reached through its branch's renames) its binding reads as
/// categories; any other channel's scale, when its first binding is. A later
/// date on a text axis, or beside a text-established scale, reads as time.
#[pyfunction]
pub(crate) fn category_read_fields(spec: &ChartSpec, schema: pyo3_arrow::PySchema) -> Vec<(Option<String>, String, String)> {
    use crate::render::scale_resolve::{establishing_axis_read, layer_flips};
    let schema = schema.into_inner();
    let renames = branch_renames(spec);
    let is_date = |source: Option<&str>, field: &str| -> bool {
        let column = source.and_then(|s| renames.get(s)).and_then(|r| r.get(field)).map_or(field, String::as_str);
        schema
            .field_with_name(column)
            .is_ok_and(|f| matches!(f.data_type(), arrow::datatypes::DataType::Timestamp(_, _)))
    };
    // The rendering channels' date fact, from the read that establishes each axis.
    let axis_dates = |channel: &str| {
        establishing_axis_read(spec, channel)
            .is_some_and(|(binding, mark, source)| establishes_dates(channel, binding, mark, is_date(source, &binding.field)))
    };
    let (x_dates, y_dates) = (axis_dates("x"), axis_dates("y"));
    // Every scope in layer order: (source, mark, encoding, flipped).
    let mut scopes: Vec<(Option<&str>, Mark, &crate::spec::encoding::Encoding, bool)> =
        vec![(None, spec.mark, &spec.encoding, crate::spec::coord::flips_positional_channels(spec.coord.as_ref()))];
    for layer in spec.layers.iter().flatten() {
        scopes.push((layer.data_source.as_deref(), layer.mark, &layer.encoding, layer_flips(layer, spec.coord.as_ref())));
    }
    let mut found = Vec::new();
    // A non-positional channel's date fact: its first binding's own.
    let mut channel_dates: std::collections::HashMap<&'static str, bool> = std::collections::HashMap::new();
    for (source, mark, encoding, flipped) in &scopes {
        for (channel, binding) in encoding.bindings() {
            let authored = match channel {
                "x2" => "x",
                "y2" => "y",
                other => other,
            };
            if !matches!(authored, "x" | "y" | "color" | "fill" | "stroke" | "shape" | "stroke_dash" | "size" | "opacity" | "theta" | "radius") {
                continue;
            }
            let (primary, of_dates) = match (authored, flipped) {
                ("x", false) | ("y", true) => ("x", x_dates),
                ("y", false) | ("x", true) => ("y", y_dates),
                (other, _) => (other, channel_dates.get(other).copied().unwrap_or(false)),
            };
            if !matches!(primary, "x" | "y") && !channel_dates.contains_key(primary) {
                channel_dates.insert(primary, establishes_dates(primary, binding, *mark, is_date(*source, &binding.field)));
            }
            if timestamp_read(primary, Some(binding), *mark, of_dates) == TimestampRead::Categories {
                found.push((source.map(str::to_string), channel.to_string(), binding.field.clone()));
            }
        }
    }
    if let Some(facet) = &spec.facet {
        for field in std::iter::once(&facet.field).chain(facet.row.iter()) {
            found.push((None, "facet".to_string(), field.clone()));
        }
    }
    found
}

fn branch_renames(spec: &ChartSpec) -> std::collections::HashMap<String, std::collections::BTreeMap<String, String>> {
    use crate::transform::core::TransformSpec;
    fn walk(
        transforms: &[TransformSpec],
        into: &mut std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    ) {
        for transform in transforms {
            if let TransformSpec::Branch(branch) = transform {
                if let Some(frame) = &branch.frame {
                    into.insert(branch.name.clone(), frame.renames.clone());
                }
                walk(&branch.transforms, into);
            }
        }
    }
    let mut into = std::collections::HashMap::new();
    walk(&spec.transforms, &mut into);
    into
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::EncodingSpec;

    fn binding(type_: Option<Declared>, scale: Option<ScaleSpec>) -> EncodingSpec {
        EncodingSpec { field: "k".into(), type_, scale, ..Default::default() }
    }

    /// Categories exactly on clauses (a) to (e), Temporal otherwise.
    #[test]
    fn timestamp_read_follows_its_five_clauses() {
        use TimestampRead::{Categories, Temporal};
        let plain = binding(None, None);
        // Otherwise: undeclared and declared temporal bindings on any channel.
        for channel in ["x", "y", "x2", "color", "fill", "size", "opacity", "theta"] {
            assert_eq!(timestamp_read(channel, Some(&plain), Mark::Point, false), Temporal, "{channel}");
            assert_eq!(timestamp_read(channel, Some(&binding(Some(Declared::Temporal), None)), Mark::Point, false), Temporal);
            assert_eq!(timestamp_read(channel, None, Mark::Point, false), Temporal, "{channel} unbound");
        }
        // (a) channels that always read categories.
        for channel in ["shape", "stroke_dash", "facet"] {
            assert_eq!(timestamp_read(channel, Some(&plain), Mark::Point, false), Categories, "(a) {channel}");
            assert_eq!(timestamp_read(channel, None, Mark::Line, false), Categories, "(a) {channel} unbound");
        }
        // (b) declared nominal or ordinal.
        for declared in [Declared::Nominal, Declared::Ordinal] {
            assert_eq!(timestamp_read("x", Some(&binding(Some(declared), None)), Mark::Bar, false), Categories, "(b) {declared:?}");
        }
        // (c) a categorical caller scale, or a stated ordinal domain; a continuous one is not.
        let ordinal: ScaleSpec = serde_json::from_str(r#"{"type":"ordinal","domain":["a"]}"#).unwrap();
        let band: ScaleSpec = serde_json::from_str(r#"{"type":"band"}"#).unwrap();
        let point: ScaleSpec = serde_json::from_str(r#"{"type":"point"}"#).unwrap();
        let linear: ScaleSpec = serde_json::from_str(r#"{"type":"linear"}"#).unwrap();
        for scale in [ordinal, band, point] {
            assert_eq!(timestamp_read("x", Some(&binding(None, Some(scale))), Mark::Point, false), Categories, "(c)");
        }
        assert_eq!(timestamp_read("x", Some(&binding(None, Some(linear))), Mark::Point, false), Temporal, "(c) linear");
        // (d) an area groups its colour, fill and stroke; another mark does not, and position is unaffected.
        for channel in ["color", "fill", "stroke"] {
            assert_eq!(timestamp_read(channel, Some(&plain), Mark::Area, false), Categories, "(d) {channel}");
            assert_eq!(timestamp_read(channel, Some(&plain), Mark::Line, false), Temporal, "(d) {channel} on a line");
        }
        assert_eq!(timestamp_read("x", Some(&plain), Mark::Area, false), Temporal, "(d) x of an area");
        // (e) a scale of dates reads every binding on its channel as categories.
        assert_eq!(timestamp_read("x", Some(&plain), Mark::Point, true), Categories, "(e)");
        assert_eq!(timestamp_read("x", None, Mark::Point, true), Categories, "(e) unbound");
    }

    /// The published binding set: an area's colour, a band scale, a shape, a dash,
    /// a facet field and row, a declared nominal, and (clause (e)) a later layer's
    /// date on an axis a declared date established, through a branch's renames;
    /// an untyped point colour or position is absent.
    #[test]
    fn category_read_fields_lists_every_binding_the_engine_reads_as_categories() {
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "area",
            "encoding": {
                "x": {"field": "t", "type": "nominal"},
                "y": {"field": "v"},
                "color": {"field": "c"}
            },
            "facet": {"field": "f", "row": "g", "mode": {"kind": "wrap", "ncols": 2}},
            "transforms": [{
                "type": "branch", "name": "_b",
                "frame": {"marker": "m", "renames": {"u": "u__rhs"}, "columns": ["u"]},
                "transforms": []
            }],
            "layers": [
                {"mark": "tick", "encoding": {"x": {"field": "t", "type": "nominal"}, "y": {"field": "v"}}},
                {"mark": "point", "encoding": {"x": {"field": "u"}, "y": {"field": "v"}, "color": {"field": "c2"}, "shape": {"field": "s"}, "stroke_dash": {"field": "d"}}, "data_source": "_b"},
                {"mark": "point", "encoding": {"x": {"field": "k", "scale": {"type": "band"}}, "y": {"field": "v"}}}
            ]
        }))
        .unwrap();
        let schema = Schema::new(vec![
            Field::new("t", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("u__rhs", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("v", DataType::Float64, true),
        ]);
        let found = category_read_fields(&spec, pyo3_arrow::PySchema::new(Arc::new(schema)));
        let expected: Vec<(Option<String>, String, String)> = [
            (None, "x", "t"),      // declared nominal
            (None, "color", "c"),  // an area groups its colour
            (None, "x", "t"),      // the layer that establishes the axis, declared nominal
            (Some("_b"), "x", "u"), // clause (e): the axis that layer established, `u` renamed in its branch
            (Some("_b"), "shape", "s"),
            (Some("_b"), "stroke_dash", "d"),
            (None, "x", "k"),      // a band scale
            (None, "facet", "f"),
            (None, "facet", "g"),
        ]
        .into_iter()
        .map(|(source, channel, field)| (source.map(str::to_string), channel.to_string(), field.to_string()))
        .collect();
        assert_eq!(found, expected);
        // A temporal chart-level binding on the channel (a composite mark's own
        // lowering) does not stop a later declared date from establishing the axis.
        let lowered: ChartSpec = serde_json::from_value(serde_json::json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "encoding": {"x": {"field": "t", "type": "temporal"}, "y": {"field": "v"}},
            "layers": [
                {"mark": "tick", "encoding": {"x": {"field": "t", "type": "nominal"}, "y": {"field": "v"}}},
                {"mark": "point", "encoding": {"x": {"field": "u"}, "y": {"field": "v"}}}
            ]
        }))
        .unwrap();
        let schema = Schema::new(vec![
            Field::new("t", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("u", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        ]);
        let found = category_read_fields(&lowered, pyo3_arrow::PySchema::new(Arc::new(schema)));
        assert_eq!(
            found,
            [
                // the chart-level binding a layer without its own x would inherit, on the axis the tick established
                (None, "x".to_string(), "t".to_string()),
                (None, "x".to_string(), "t".to_string()),
                (None, "x".to_string(), "u".to_string()),
            ],
            "the tick's declared date and the point's undeclared date on the axis it established"
        );
        // The untyped point colour `c2` and the untyped `y` readings are absent: a time or value axis.
        assert!(!found.iter().any(|(_, channel, field)| (channel == "color" && field == "c2") || channel == "y"));
    }

    /// The published set reads clause (e) as the engine does: a channel is one of
    /// dates only when the read that ESTABLISHES its axis is a date read as
    /// categories. A text column establishing the axis, a declared-date layer
    /// after it and an undeclared date layer third: the third reads as time, so it
    /// is not listed, and the engine says the same ([`axis_of_dates`]); with the
    /// declared date first, the third is listed and the engine agrees. Fails when
    /// the publisher and the engine disagree on the establishing read.
    #[test]
    fn category_read_fields_reads_clause_e_from_the_establishing_read_as_the_engine_does() {
        use arrow::array::{StringArray, TimestampMillisecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        let ts = DataType::Timestamp(TimeUnit::Millisecond, None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("k", ts.clone(), true),
            Field::new("u", ts.clone(), true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a"])),
                Arc::new(TimestampMillisecondArray::from(vec![0_i64])),
                Arc::new(TimestampMillisecondArray::from(vec![0_i64])),
            ],
        )
        .unwrap();
        let spec_of = |first: &str, second: &str| -> ChartSpec {
            serde_json::from_value(serde_json::json!({
                "data": {"kind": "named", "name": "default"},
                "mark": "bar",
                "encoding": {"x": {"field": first, "type": "nominal"}, "y": {"field": "v"}},
                "layers": [
                    {"mark": "bar", "encoding": {"x": {"field": first, "type": "nominal"}, "y": {"field": "v"}}},
                    {"mark": "bar", "encoding": {"x": {"field": second, "type": "nominal"}, "y": {"field": "v"}}},
                    {"mark": "point", "encoding": {"x": {"field": "u"}, "y": {"field": "v"}}}
                ]
            }))
            .unwrap()
        };
        for (first, second, third_listed) in [("cat", "k", false), ("k", "cat", true)] {
            let spec = spec_of(first, second);
            let found = category_read_fields(&spec, pyo3_arrow::PySchema::new(schema.clone()));
            let listed = found.iter().any(|(_, channel, field)| channel == "x" && field == "u");
            assert_eq!(listed, third_listed, "{first} first, {second} second: the undeclared third layer");
            let rendering = EncodingSpec { field: first.into(), type_: Some(Declared::Nominal), ..Default::default() };
            let engine = crate::render::scale_resolve::axis_of_dates("x", &rendering, &spec, &batch, &Default::default());
            assert_eq!(engine, listed, "the published set and the engine read clause (e) alike for {first} first");
        }
    }

    /// Dates join in time order, two units holding one instant dedupe, text
    /// keeps first appearance, and a date joining text takes the date slots.
    #[test]
    fn a_category_domain_joins_dates_in_time_order_and_text_in_first_appearance() {
        const DAY: i128 = 86_400_000_000_000;
        let dates = |items: &[(&str, i128)]| {
            let mut domain = CategoryDomain::default();
            for (label, instant) in items {
                domain.push((*label).to_string(), Some(*instant));
            }
            domain
        };
        let mut a = dates(&[("2020-01-03", 3 * DAY), ("2020-01-01", DAY)]);
        a.join(&dates(&[("2020-01-02", 2 * DAY), ("2020-01-01", DAY)]));
        assert_eq!(a.labels(), ["2020-01-01", "2020-01-02", "2020-01-03"]);

        // The same instant read in two units is one label, so it dedupes.
        let mut seconds = dates(&[("2020-01-01", DAY)]);
        seconds.join(&dates(&[("2020-01-01", DAY)]));
        assert_eq!(seconds.len(), 1);

        // Text keeps first appearance.
        let mut text = CategoryDomain::from_labels(["c".to_string(), "a".to_string()]);
        text.join(&CategoryDomain::from_labels(["b".to_string(), "a".to_string()]));
        assert_eq!(text.labels(), ["c", "a", "b"]);

        // A date list joining text keeps the text first and sorts the dates.
        let mut mixed = CategoryDomain::from_labels(["c".to_string(), "a".to_string()]);
        mixed.join(&dates(&[("2020-01-03", 3 * DAY), ("2020-01-01", DAY)]));
        assert_eq!(mixed.labels(), ["c", "a", "2020-01-01", "2020-01-03"]);

        // Text, then a date, then text again: the date stays in its own slot.
        let mut spread = CategoryDomain::from_labels(["c".to_string()]);
        spread.join(&dates(&[("2020-01-03", 3 * DAY)]));
        spread.join(&CategoryDomain::from_labels(["z".to_string()]));
        spread.join(&dates(&[("2020-01-01", DAY)]));
        assert_eq!(spread.labels(), ["c", "2020-01-01", "z", "2020-01-03"]);

        // The date fact stays the establishing list's own.
        let established = dates(&[("2020-01-01", DAY)]).with_of_dates(true);
        let mut text_first = CategoryDomain::from_labels(["a".to_string()]);
        text_first.join(&established);
        assert!(!text_first.of_dates());
    }
}
