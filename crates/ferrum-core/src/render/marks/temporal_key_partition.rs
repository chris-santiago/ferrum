//! Test-only: the series partition fixture `line`, `area` and `ribbon` share to
//! pin how a series builder reads a temporal key column.

use crate::render::color::Color;
use crate::render::draw::DrawCtx;
use crate::render::marks::channels::SeriesPartition;
use crate::render::scale_resolve::ColorScale;
use crate::spec::encoding::Encoding;

/// Which channel [`temporal_key_partition`] binds the `Timestamp` key column to.
pub(crate) enum TemporalKeyChannel {
    Detail,
    Color,
}

/// A series mark's own partition of four rows whose key column `k` is a
/// `Timestamp` holding two instants (rows 0 and 1, then rows 2 and 3), bound
/// to `key`. A color key carries a hand-built categorical scale over the two
/// instants' label strings (a hand-built scale, so the builder's own read of the
/// key is what is pinned, not the resolver's domain).
/// Shared by `line`, `area` and `ribbon`, whose `series_groups` take the same
/// context, so each pins its own key reads with one fixture.
pub(crate) fn temporal_key_partition(
    mark: crate::spec::mark::Mark,
    key: TemporalKeyChannel,
    partition: fn(&DrawCtx) -> SeriesPartition,
) -> SeriesPartition {
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::encoding::EncodingSpec;
    use crate::spec::mark_style::MarkKwargsSpec;
    use crate::spec::chart::ChartSpec;
    use arrow::array::{Float64Array, TimestampMillisecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use std::borrow::Cow;
    use std::sync::Arc;

    let field = |name: &str| Some(EncodingSpec { field: name.into(), type_: None, ..Default::default() });
    let spec = ChartSpec {
        mark: mark.clone(),
        encoding: Encoding {
            x: field("x"),
            y: field("y"),
            y2: matches!(mark, crate::spec::mark::Mark::Ribbon).then(|| field("y2")).flatten(),
            ..Default::default()
        },
        ..Default::default()
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("y2", DataType::Float64, false),
        Field::new("k", DataType::Timestamp(TimeUnit::Millisecond, None), false),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(vec![0.0, 1.0, 0.0, 1.0])),
        Arc::new(Float64Array::from(vec![0.0, 1.0, 0.0, 1.0])),
        Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 2.0])),
        Arc::new(TimestampMillisecondArray::from(vec![1_000i64, 1_000, 2_000, 2_000])),
    ]).unwrap();
    let theme = ThemeInputs::default();
    let panel = PanelLayout {
        plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
        facet_key: None, row: 0, col: 0,
        strip_title: None, row_strip_title: None, row_facet_key: None,
    };
    let (mut scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
    let mut draw_spec = spec.clone();
    let overrides = match key {
        TemporalKeyChannel::Detail => MarkKwargsSpec { detail: Some("k".into()), ..Default::default() },
        TemporalKeyChannel::Color => {
            draw_spec.encoding.color = field("k");
            scales.color = Some(ColorScale::Categorical {
                domain: vec!["1970-01-01T00:00:01".into(), "1970-01-01T00:00:02".into()],
                palette: Cow::Owned(vec![Color::new(200, 0, 0, 255), Color::new(0, 0, 200, 255)]),
            });
            MarkKwargsSpec::default()
        }
    };
    let mark_style = resolve_mark_style(Some(&overrides), &theme, &mark).unwrap();
    let ctx = DrawCtx { spec: &draw_spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
    partition(&ctx)
}
