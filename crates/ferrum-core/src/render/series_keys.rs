//! The fields a mark splits its rows into series by, described once.
//!
//! A mark's series are the groups of rows its builder draws one path, band or
//! symbol set from: a colour (when the colour scale draws it as categories or
//! as bins), a `shape`, a `stroke_dash` and a `detail` field. The mark
//! builders partition by them ([`crate::render::marks::channels`]), the
//! null-key drop counts the nulls in them
//! ([`crate::render::prepare::null_keys`]) and the encoding impute completes
//! each series they name ([`crate::render::prepare::impute`]); all three read
//! this one description, so a layer's series cannot be one thing to the drop
//! and another to the draw.
//!
//! The one input the description cannot read from the layer alone is the
//! colour decision ([`ColorSplit`]): whether the colour scale draws a colour
//! column as categories depends on the scale (an area that draws colour makes
//! it categorical for every layer of the panel; an earlier layer's continuous
//! scale leaves a later layer's text colour unsplit), so the draw reads it from
//! the resolved scale and prepare reads it from the same decision
//! ([`crate::render::scale_resolve::color_split`]).

use arrow::datatypes::{DataType, Schema};

use crate::render::draw::binned_color_scale;
use crate::render::scale_resolve::{dtype_reads_categorical, is_categorical_scale, ColorInput, ColorScale, ResolvedScales};
use crate::spec::encoding::{DataType as SpecDataType, Encoding, EncodingSpec};
use crate::spec::mark::Mark;

/// How a layer's colour binding splits its marks into series, as the colour
/// scale draws it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::render) enum ColorSplit {
    /// One series per category: the scale reads the column as categories
    /// (a text column, an integer column an area colours), or no scale
    /// resolved.
    Categories,
    /// One series per bin: the colour carries a bin rule and the scale is the
    /// discretizing scale the rule cut.
    Bins,
    /// No split: a continuous (or otherwise numeric) scale colours each mark
    /// by its own value, so the colour keys no series.
    Unsplit,
}

impl ColorSplit {
    /// The split `binding` draws under `scale`, the resolved colour scale:
    /// [`Self::Bins`] when [`binned_color_scale`] answers, [`Self::Categories`]
    /// for a category-keyed or absent scale, [`Self::Unsplit`] for any other
    /// numeric scale.
    pub(in crate::render) fn of_scale(binding: Option<&EncodingSpec>, scale: Option<&ColorScale>) -> Self {
        if binned_color_scale(binding, scale).is_some() {
            return Self::Bins;
        }
        match scale.map(ColorScale::input) {
            Some(ColorInput::Numeric) => Self::Unsplit,
            Some(ColorInput::Category) | None => Self::Categories,
        }
    }

    /// The split a layer draws when the panel's colour scale splits as `self`
    /// for the establishing binding: a bin-cut scale splits only a layer whose
    /// own binding carries the bin rule, and leaves any other layer's colour
    /// unsplit, as [`Self::of_scale`] answers for that layer.
    pub(in crate::render) fn for_binding(self, binding: Option<&EncodingSpec>) -> Self {
        match self {
            Self::Bins if binding.is_none_or(|binding| binding.color_bin().is_none()) => Self::Unsplit,
            split => split,
        }
    }
}

/// What decides whether a colour or stroke-dash field splits series as
/// categories.
#[derive(Debug, Clone, Copy)]
pub(in crate::render) enum SplitSource<'s> {
    /// A draw: the scales the panel resolved decide. A colour splits as
    /// [`ColorSplit::of_scale`] answers; a stroke-dash field splits as
    /// categories exactly when a stroke-dash scale resolved (that scale is
    /// categorical or absent).
    Resolved(&'s ResolvedScales),
    /// Prepare, before any scale resolves: the colour split is the one the
    /// colour scale's kind decides ([`crate::render::scale_resolve::color_split`]),
    /// and a stroke-dash field reads as categories by its own column.
    Decided(ColorSplit),
}

/// Which channel a series key is bound on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::render) enum SplitChannel {
    Color,
    Shape,
    StrokeDash,
    Detail,
}

/// One field a mark splits its series by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::render) struct SeriesKey {
    field: String,
    channel: SplitChannel,
    partitions: bool,
}

impl SeriesKey {
    pub(in crate::render) fn field(&self) -> &str {
        &self.field
    }
}

/// A layer's series keys, in the column gate's read order (colour, shape,
/// stroke dash, detail), each field once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::render) struct SeriesKeys {
    keys: Vec<SeriesKey>,
    color: ColorSplit,
}

impl SeriesKeys {
    /// The keys of a `mark` that draws `drawn`, splitting `detail` when the
    /// mark does, over rows of `schema`; a field the rows do not carry is no
    /// key. `source` says how the colour and the dash field split. An area
    /// groups by its colour's categories whatever the scale reads, so its
    /// colour is never [`ColorSplit::Unsplit`].
    pub(in crate::render) fn of(
        mark: Mark,
        drawn: &Encoding,
        detail: Option<&str>,
        source: SplitSource<'_>,
        schema: &Schema,
    ) -> Self {
        let (color, dash_scale) = match source {
            SplitSource::Resolved(scales) => {
                (ColorSplit::of_scale(drawn.color.as_ref(), scales.color.as_ref()), Some(scales.stroke_dash.is_some()))
            }
            SplitSource::Decided(color) => (color, None),
        };
        let color = match (mark, color) {
            (Mark::Area, ColorSplit::Unsplit) => ColorSplit::Categories,
            (_, color) => color,
        };
        let dtype_of = |field: &str| schema.field_with_name(field).ok().map(|field| field.data_type());
        let mut keys: Vec<SeriesKey> = Vec::new();
        let mut push = |field: &str, channel: SplitChannel, partitions: bool| {
            if !keys.iter().any(|key| key.field == field) {
                keys.push(SeriesKey { field: field.to_string(), channel, partitions });
            }
        };
        if let Some(binding) = &drawn.color {
            if color != ColorSplit::Unsplit && dtype_of(&binding.field).is_some() {
                push(&binding.field, SplitChannel::Color, true);
            }
        }
        // A shape binding draws a symbol per level whatever the field's dtype
        // (a numeric field's values are its levels).
        if let Some(binding) = &drawn.shape {
            if dtype_of(&binding.field).is_some() {
                push(&binding.field, SplitChannel::Shape, true);
            }
        }
        // A dash binding is a key where the mark draws levels from the field:
        // a categorical field on any mark that draws the channel, a numeric
        // index on the marks that dash each row by it. Only a categorical one
        // splits a draw partition: a series mark (line, ribbon) reads a
        // numeric one from a series' first row, and a per-row mark dashes each
        // row by its index without grouping rows.
        if let Some(binding) = &drawn.stroke_dash {
            if let Some(dtype) = dtype_of(&binding.field) {
                let categorical = dash_scale.unwrap_or_else(|| reads_categorical(binding, dtype));
                if categorical || mark.dashes_each_row_by_index() {
                    push(&binding.field, SplitChannel::StrokeDash, categorical);
                }
            }
        }
        if let Some(detail) = detail.filter(|_| mark.splits_series_by_detail()) {
            if dtype_of(detail).is_some() {
                push(detail, SplitChannel::Detail, true);
            }
        }
        Self { keys, color }
    }

    /// Every key: the fields whose nulls the null-key drop counts and the
    /// impute completes by.
    pub(in crate::render) fn levels(&self) -> &[SeriesKey] {
        &self.keys
    }

    /// The keys a draw partition splits by.
    pub(in crate::render) fn partitions(&self) -> impl Iterator<Item = &SeriesKey> {
        self.keys.iter().filter(|key| key.partitions)
    }

    /// The partition key on `channel`, when the draw splits by one.
    pub(in crate::render) fn partition_on(&self, channel: SplitChannel) -> Option<&SeriesKey> {
        self.partitions().find(|key| key.channel == channel)
    }

    /// How the colour binding splits the series, as the keys were read.
    pub(in crate::render) fn color_split(&self) -> ColorSplit {
        self.color
    }

    /// The colour field, when the colour draws one series per bin.
    pub(in crate::render) fn binned_color(&self) -> Option<&str> {
        self.keys
            .iter()
            .find(|key| key.channel == SplitChannel::Color)
            .filter(|_| self.color == ColorSplit::Bins)
            .map(SeriesKey::field)
    }
}

/// Whether `enc`'s column of `dtype` reads as categories: typed nominal or
/// ordinal, carrying a categorical scale, or a dtype the continuous reader
/// cannot place. An untyped all-null column is a quantitative position.
pub(in crate::render) fn reads_categorical(enc: &EncodingSpec, dtype: &DataType) -> bool {
    matches!(enc.type_, Some(SpecDataType::Nominal | SpecDataType::Ordinal))
        || enc.scale.as_ref().is_some_and(is_categorical_scale)
        || (*dtype != DataType::Null && dtype_reads_categorical(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;

    fn schema(columns: &[(&str, DataType)]) -> Schema {
        Schema::new(columns.iter().map(|(name, dtype)| Field::new(*name, dtype.clone(), true)).collect::<Vec<_>>())
    }

    fn binding(field: &str) -> Option<EncodingSpec> {
        Some(EncodingSpec { field: field.into(), ..Default::default() })
    }

    fn fields(keys: &SeriesKeys) -> Vec<&str> {
        keys.levels().iter().map(SeriesKey::field).collect()
    }

    #[test]
    fn a_continuous_colour_keys_nothing() {
        let drawn = Encoding { color: binding("q"), ..Default::default() };
        let schema = schema(&[("q", DataType::Float64)]);
        let keys = SeriesKeys::of(Mark::Line, &drawn, None, SplitSource::Decided(ColorSplit::Unsplit), &schema);
        assert!(keys.levels().is_empty());
        assert_eq!(keys.partitions().count(), 0);
        let keys = SeriesKeys::of(Mark::Line, &drawn, None, SplitSource::Decided(ColorSplit::Categories), &schema);
        assert_eq!(fields(&keys), ["q"], "the same column keys a series once the scale reads it as categories");
    }

    #[test]
    fn an_area_groups_its_colour_whatever_the_scale_reads() {
        let drawn = Encoding { color: binding("h"), ..Default::default() };
        let schema = schema(&[("h", DataType::Int64)]);
        let area = SeriesKeys::of(Mark::Area, &drawn, None, SplitSource::Decided(ColorSplit::Unsplit), &schema);
        assert_eq!(fields(&area), ["h"]);
        let line = SeriesKeys::of(Mark::Line, &drawn, None, SplitSource::Decided(ColorSplit::Unsplit), &schema);
        assert!(line.levels().is_empty());
    }

    #[test]
    fn a_numeric_dash_is_a_level_of_a_per_row_mark_but_never_a_partition() {
        let drawn = Encoding { stroke_dash: binding("n"), ..Default::default() };
        let schema = schema(&[("n", DataType::Float64)]);
        for mark in [Mark::Point, Mark::Bar, Mark::Rect, Mark::Rule] {
            let keys = SeriesKeys::of(mark, &drawn, None, SplitSource::Decided(ColorSplit::Unsplit), &schema);
            assert_eq!(fields(&keys), ["n"], "{mark:?}: the drop counts a null dash and the impute completes by it");
            assert_eq!(keys.partitions().count(), 0, "{mark:?}: rows are dashed by index, never grouped");
        }
    }

    #[test]
    fn a_numeric_dash_on_a_series_mark_is_no_key() {
        let drawn = Encoding { stroke_dash: binding("n"), ..Default::default() };
        let schema = schema(&[("n", DataType::Float64)]);
        for mark in [Mark::Line, Mark::Ribbon] {
            let keys = SeriesKeys::of(mark, &drawn, None, SplitSource::Decided(ColorSplit::Unsplit), &schema);
            assert!(keys.levels().is_empty(), "{mark:?}");
        }
    }

    #[test]
    fn a_categorical_dash_is_a_level_and_a_partition_on_every_mark_that_draws_it() {
        let drawn = Encoding { stroke_dash: binding("s"), ..Default::default() };
        let schema = schema(&[("s", DataType::Utf8)]);
        for mark in [Mark::Line, Mark::Ribbon, Mark::Point, Mark::Rule] {
            let keys = SeriesKeys::of(mark, &drawn, None, SplitSource::Decided(ColorSplit::Unsplit), &schema);
            assert_eq!(fields(&keys), ["s"], "{mark:?}");
            assert!(keys.partition_on(SplitChannel::StrokeDash).is_some(), "{mark:?}");
        }
    }

    #[test]
    fn keys_follow_the_gate_order_each_field_once_and_skip_columns_the_rows_lack() {
        let drawn = Encoding {
            color: binding("g"),
            shape: binding("g"),
            stroke_dash: binding("d"),
            ..Default::default()
        };
        let schema = schema(&[("g", DataType::Utf8), ("d", DataType::Utf8), ("e", DataType::Utf8)]);
        let keys = SeriesKeys::of(Mark::Point, &drawn, Some("e"), SplitSource::Decided(ColorSplit::Categories), &schema);
        assert_eq!(fields(&keys), ["g", "d"], "a point splits by no detail, and 'g' is one key");
        let keys = SeriesKeys::of(Mark::Line, &drawn, Some("e"), SplitSource::Decided(ColorSplit::Categories), &schema);
        assert_eq!(fields(&keys), ["g", "d", "e"], "a line splits by detail, after the dash");
        let lacking = Encoding { color: binding("missing"), ..Default::default() };
        assert!(SeriesKeys::of(Mark::Line, &lacking, Some("also_missing"), SplitSource::Decided(ColorSplit::Categories), &schema)
            .levels()
            .is_empty());
    }

    #[test]
    fn a_binned_colour_is_reported_only_under_a_bin_split() {
        let binned: EncodingSpec = serde_json::from_value(
            serde_json::json!({"field": "w", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 3, "nice": false}}}),
        )
        .unwrap();
        let drawn = Encoding { color: Some(binned.clone()), ..Default::default() };
        let schema = schema(&[("w", DataType::Float64)]);
        let bins = SeriesKeys::of(Mark::Line, &drawn, None, SplitSource::Decided(ColorSplit::Bins), &schema);
        assert_eq!(bins.binned_color(), Some("w"));
        let categories = SeriesKeys::of(Mark::Line, &drawn, None, SplitSource::Decided(ColorSplit::Categories), &schema);
        assert_eq!(categories.binned_color(), None);
        assert_eq!(ColorSplit::Bins.for_binding(Some(&binned)), ColorSplit::Bins);
        assert_eq!(ColorSplit::Bins.for_binding(binding("w").as_ref()), ColorSplit::Unsplit);
        assert_eq!(ColorSplit::Categories.for_binding(binding("w").as_ref()), ColorSplit::Categories);
    }
}
