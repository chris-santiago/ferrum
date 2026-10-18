//! Domain computation: numeric extent union, sort application, field location.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use super::column_min_max_f64;
use crate::render::{RenderError, RenderWarning};
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::sort::{parse_sort_field_object, SortOp, SortOrder};
use crate::spec::layer::Layer;
use crate::spec::mark::Mark;
use crate::transform::core::FINAL_OUTPUT_KEY;

/// Result of looking up a field across the primary batch and named
/// transform outputs. Carries both the source batch and the resolved
/// column so callers don't have to re-`column_by_name(...).expect(...)`
/// after the lookup — the "field is present in this batch" invariant
/// lives in the type, not in a comment.
pub(crate) struct LocatedColumn<'a> {
    pub(crate) batch: &'a RecordBatch,
    pub(crate) col: &'a dyn Array,
}

/// The column `field` of an unlayered spec's one batch, `primary_batch`, the
/// batch its one mark draws; `None` when that batch lacks it. A layered
/// spec reads each layer's column from the batch that layer draws
/// ([`LayerSource`]), never by searching batches for a column of that name:
/// a same-named column elsewhere is data no mark of the layer draws.
pub(in crate::render) fn locate_field<'a>(field: &str, primary_batch: &'a RecordBatch) -> Option<LocatedColumn<'a>> {
    primary_batch.column_by_name(field).map(|c| LocatedColumn { batch: primary_batch, col: c.as_ref() })
}

/// Where the batch a layer draws lives: the chart's data (the primary batch,
/// joined on a faceted shared axis by the all-panels batch at
/// [`FINAL_OUTPUT_KEY`]), the one named output its `data_source` names, or,
/// for a member of an overlay group of leaves, the batch that member draws.
///
/// A layer's positional contribution to an axis is read from this batch and
/// no other: a same-named column in another batch is data the layer never
/// draws (a stat layer's fitted `y` beside the raw `y`, an operand's own rows
/// beside the merged frame).
#[derive(Clone, Copy, PartialEq)]
enum LayerSource<'a> {
    Chart,
    Output(&'a str),
    Batch(&'a RecordBatch),
}

impl<'a> LayerSource<'a> {
    fn of(layer: &'a Layer) -> Self {
        layer.data_source.as_deref().map_or(LayerSource::Chart, LayerSource::Output)
    }

    /// The batch itself, or `None` for an output nothing published (refused
    /// by `prepare` before any render reads it).
    fn batch<'b>(
        self,
        primary_batch: &'b RecordBatch,
        transform_outputs: &'b HashMap<String, RecordBatch>,
    ) -> Option<&'b RecordBatch>
    where
        'a: 'b,
    {
        match self {
            LayerSource::Chart => Some(primary_batch),
            LayerSource::Output(key) => transform_outputs.get(key),
            LayerSource::Batch(batch) => Some(batch),
        }
    }

    fn locate<'b>(
        self,
        field: &str,
        primary_batch: &'b RecordBatch,
        transform_outputs: &'b HashMap<String, RecordBatch>,
    ) -> Option<LocatedColumn<'b>>
    where
        'a: 'b,
    {
        let batch = self.batch(primary_batch, transform_outputs)?;
        let col = batch.column_by_name(field)?;
        Some(LocatedColumn { batch, col: col.as_ref() })
    }
}

/// One read of a layered axis's seed: the batch, and the fields a layer draws
/// there on the axis: the value `field`, its paired `x2`/`y2` `span`, and the
/// `opposite` positional field the ordinal null-band rule judges rows by.
/// `inheritor` is the index of the later layer whose inherited field this
/// read is, judged like that layer's own column; `None` for the bound field.
#[derive(Clone, Copy)]
struct SeedRead<'a> {
    source: LayerSource<'a>,
    field: &'a str,
    span: Option<&'a str>,
    opposite: Option<&'a str>,
    inheritor: Option<usize>,
}

/// The reads that seed a layered axis, in layer order, one per batch and
/// field. The first is the bound field (the rendering encoding's) from the
/// batch of the layer that establishes the axis ([`establishing_axis_layer`]):
/// layer 0's own, with the caller's `span` and `opposite`, when layer 0 draws
/// the channel, else the first later layer's that does, with that layer's
/// own `span` and `opposite` where the caller asks for them. Then,
/// for each later layer on the chart's data that binds nothing on `channel`
/// but draws it, the chart's data with the field that layer inherits: the
/// chart-level binding in that layer's own coordinate space, as its completed
/// encoding carries it ([`drawn_channel_slots`]), with the span and opposite
/// fields it draws.
fn seed_reads<'a>(
    spec: &'a ChartSpec,
    layers: &'a [Layer],
    channel: &str,
    bound: (&'a str, Option<&'a str>, Option<&'a str>),
) -> Vec<SeedRead<'a>> {
    let (field, span, opposite) = bound;
    let Some((index, establisher)) = establishing_axis_layer(spec, layers, channel) else { return Vec::new() };
    let first = if index == 0 {
        SeedRead { source: LayerSource::of(establisher), field, span, opposite, inheritor: None }
    } else {
        let drawn = drawn_channel_slots(spec, establisher, channel);
        SeedRead {
            source: LayerSource::of(establisher),
            field,
            span: span.and(drawn.span),
            opposite: opposite.and(drawn.opposite),
            inheritor: None,
        }
    };
    let mut reads = vec![first];
    for (index, layer) in layers.iter().enumerate().skip(1) {
        if layer.data_source.is_some() || (channel == "y" && layer.independent_y) {
            continue;
        }
        let own = channel_slots(&layer.encoding, layer_flips(layer, spec.coord.as_ref()), channel);
        if own.value.is_some() {
            continue;
        }
        let drawn = drawn_channel_slots(spec, layer, channel);
        let Some(field) = drawn.value else { continue };
        if reads.iter().any(|r| r.source == LayerSource::Chart && r.field == field) {
            continue;
        }
        reads.push(SeedRead { source: LayerSource::Chart, field, span: drawn.span, opposite: drawn.opposite, inheritor: Some(index) });
    }
    reads
}

/// The positional bindings `layer` draws on the shared `channel`: the
/// channels of its completed encoding ([`Layer::completed_encoding`]), which
/// fills a chart-data layer's unbound slots from the chart-level binding and
/// keeps only the inherited ones its mark draws. A layer on its own output
/// inherits no positional channel, and an `independent_y` layer draws its own
/// y axis rather than this one, so it draws nothing on the shared y.
fn drawn_channel_slots<'a>(spec: &'a ChartSpec, layer: &'a Layer, channel: &str) -> ChannelSlots<'a> {
    use crate::spec::layer::PositionalChannel::{X, X2, Y, Y2};
    let none = ChannelSlots { value: None, span: None, opposite: None };
    if channel == "y" && layer.independent_y {
        return none;
    }
    let (value, span, opposite) = match (channel, layer_flips(layer, spec.coord.as_ref())) {
        ("x", false) | ("y", true) => (X, X2, Y),
        ("y", false) | ("x", true) => (Y, Y2, X),
        _ => return none,
    };
    let completed = layer.completed_encoding(&spec.encoding);
    let field = |c: crate::spec::layer::PositionalChannel| -> Option<&'a str> {
        if c.slot(&completed).is_none() {
            return None;
        }
        c.slot(&layer.encoding).as_ref().or(c.slot(&spec.encoding).as_ref()).map(|e| e.field.as_str())
    };
    ChannelSlots { value: field(value), span: field(span), opposite: field(opposite) }
}

/// The rows a data-aware sort of a layered `channel` aggregates over, and the
/// fields `x_field`/`y_field` a channel shorthand (`sort="-y"`) names: the
/// establishing layer's ([`establishing_axis_layer`]) pre-statistic rows
/// (`prepare::pre_statistic_rows`) and the fields that layer draws on the
/// rendered x and y. So a sort orders the axis by the establishing layer's
/// own data, whichever operand's frame the chart rows hold. `None` for a
/// spec with no layers, which sorts over its one batch.
pub(in crate::render) struct SortRows<'a> {
    pub(in crate::render) batch: &'a RecordBatch,
    pub(in crate::render) x_field: Option<&'a str>,
    pub(in crate::render) y_field: Option<&'a str>,
}

pub(in crate::render) fn establishing_sort_rows<'a>(
    channel: &str,
    spec: &'a ChartSpec,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
    include_final: bool,
) -> Option<SortRows<'a>> {
    let (_, layer) = establishing_axis_layer(spec, spec.layers.as_deref()?, channel)?;
    let batch = crate::render::prepare::pre_statistic_rows(
        layer.data_source.as_deref(),
        &spec.transforms,
        primary_batch,
        transform_outputs,
        include_final,
    );
    Some(SortRows {
        batch,
        x_field: drawn_channel_slots(spec, layer, "x").value,
        y_field: drawn_channel_slots(spec, layer, "y").value,
    })
}

/// `spec` with the primary y axis owned by a layer that draws y: when no
/// layer sharing the primary y axis draws y ([`drawn_channel_slots`], the
/// predicate [`establishing_axis_layer`] reads), the first `independent_y`
/// layer that draws y has its flag cleared, so it establishes the primary
/// axis. Independence with no shared axis to differ from is vacuous, and the
/// establishing rule (the first layer drawing a channel owns it) then holds
/// for y as for every channel. Applied where a render's spec enters: in
/// `prepare::prepare_render_inputs`, in `scene_build::build_scene`, and at a
/// composite leaf's entry (`composite_render::render_composite_scene`, so the
/// composite resolve pass reads the leaf's normalized flags too), so the
/// slot plan, the axes and every domain read see one flag set. `spec` itself
/// (borrowed) whenever a sharing layer draws y, or no layer does.
pub(in crate::render) fn with_primary_y_owner(spec: &ChartSpec) -> std::borrow::Cow<'_, ChartSpec> {
    use std::borrow::Cow;
    let Some(layers) = spec.layers.as_deref() else { return Cow::Borrowed(spec) };
    if layers.iter().any(|layer| drawn_channel_slots(spec, layer, "y").value.is_some()) {
        return Cow::Borrowed(spec);
    }
    let shared = |layer: &Layer| Layer { independent_y: false, ..layer.clone() };
    let Some(owner) = layers
        .iter()
        .position(|layer| layer.independent_y && drawn_channel_slots(spec, &shared(layer), "y").value.is_some())
    else {
        return Cow::Borrowed(spec);
    };
    let mut normalized = spec.clone();
    if let Some(layer) = normalized.layers.as_mut().and_then(|layers| layers.get_mut(owner)) {
        layer.independent_y = false;
    }
    Cow::Owned(normalized)
}

/// The layer, with its index, whose batch a layered axis's bound field is
/// read from first: layer 0 when it draws `channel`
/// ([`drawn_channel_slots`]), else the first later layer that does, the same
/// establishing rule the auxiliary channels' domains follow ([`aux_reads`]).
/// Layer 0 when no layer draws the channel, so a spec binding it nowhere
/// keeps layer 0's read and its `UnknownColumn` precondition. `None` only for
/// an empty layer list.
fn establishing_axis_layer<'a>(
    spec: &'a ChartSpec,
    layers: &'a [Layer],
    channel: &str,
) -> Option<(usize, &'a Layer)> {
    layers
        .iter()
        .enumerate()
        .find(|(_, layer)| drawn_channel_slots(spec, layer, channel).value.is_some())
        .or_else(|| layers.first().map(|lead| (0, lead)))
}

/// One read of a chart-level non-positional channel's domain (color, size,
/// the opacity family, shape, stroke_dash) on a layered spec: the layer's
/// index, its batch ([`LayerSource`]), the field it draws there, and what
/// the read holds ([`AuxReadKind`]). `pub` (its fields are not) because the
/// crate-visible scale builders take a slice of it; this module is private
/// to `scale_resolve`.
#[derive(Clone, Copy)]
pub struct AuxRead<'a> {
    layer: usize,
    /// The mark of the layer that draws this read.
    mark: Mark,
    source: LayerSource<'a>,
    field: &'a str,
    kind: AuxReadKind<'a>,
}

/// What an [`AuxRead`] holds: a column of its batch, bound by `binding`; or
/// a raster image's cells, the values its Raster batch's `pixel_data` packs
/// (no column holds them), which the image colors in its colormap `cmap`.
/// `entry` is the image layer's public call (`mark_raster`), when a call
/// built it.
#[derive(Clone, Copy)]
enum AuxReadKind<'a> {
    Column { binding: &'a crate::spec::encoding::EncodingSpec },
    RasterCells { cmap: Option<&'a str>, entry: Option<&'a str> },
}

impl<'a> AuxRead<'a> {
    /// The index of the layer this read is (its position among the members'
    /// merged layers for an overlay group's read).
    pub(in crate::render) fn layer(&self) -> usize {
        self.layer
    }

    /// The column this read draws, in the batch its layer draws; `None` when
    /// that batch lacks the field, and for a raster image's cells, which no
    /// column holds.
    pub(in crate::render) fn locate<'b>(
        &self,
        primary_batch: &'b RecordBatch,
        transform_outputs: &'b HashMap<String, RecordBatch>,
    ) -> Option<LocatedColumn<'b>>
    where
        'a: 'b,
    {
        match self.kind {
            AuxReadKind::Column { .. } => self.source.locate(self.field, primary_batch, transform_outputs),
            AuxReadKind::RasterCells { .. } => None,
        }
    }

    /// Whether this read is a raster image's cells.
    pub(in crate::render) fn is_raster_cells(&self) -> bool {
        matches!(self.kind, AuxReadKind::RasterCells { .. })
    }

    /// The field this read draws.
    pub(in crate::render) fn field(&self) -> &'a str {
        self.field
    }

    /// The refusal of a raster image's cells under a categorical color scale
    /// ([`RenderError::RasterColorNotQuantitative`]), voiced by the image
    /// layer's public call (ruling F: `mark_raster`, else `layer {i}:
    /// mark_image`); `None` for a column read.
    pub(in crate::render) fn raster_color_refusal(&self, reason: crate::render::RasterColorReason) -> Option<RenderError> {
        let AuxReadKind::RasterCells { entry, .. } = self.kind else { return None };
        let entry = match entry.filter(|entry| *entry != "mark_image") {
            Some(entry) => entry.to_string(),
            None => format!("layer {}: mark_image", self.layer),
        };
        Some(RenderError::RasterColorNotQuantitative(Box::new(crate::render::RasterColorRefusal {
            layer: self.layer,
            entry,
            field: self.field.to_string(),
            reason,
        })))
    }

    /// The colormap a raster image's cells draw in (`mark_raster(cmap=)`);
    /// `None` for a column read or an image naming none.
    pub(in crate::render) fn cmap(&self) -> Option<&'a str> {
        match self.kind {
            AuxReadKind::RasterCells { cmap, .. } => cmap,
            AuxReadKind::Column { .. } => None,
        }
    }

    /// The transform a raster image's cells map through, which its Raster
    /// batch names ([`crate::render::marks::image::raster_value_transform`]);
    /// `None` for a column read.
    pub(in crate::render) fn raster_transform(
        &self,
        primary_batch: &RecordBatch,
        transform_outputs: &HashMap<String, RecordBatch>,
    ) -> Option<super::ColorTransform> {
        match self.kind {
            AuxReadKind::RasterCells { .. } => self
                .source
                .batch(primary_batch, transform_outputs)
                .map(crate::render::marks::image::raster_value_transform),
            AuxReadKind::Column { .. } => None,
        }
    }

    /// The Raster batch a raster image's cells are read from; `None` for a
    /// column read.
    pub(in crate::render) fn raster_batch<'b>(
        &self,
        primary_batch: &'b RecordBatch,
        transform_outputs: &'b HashMap<String, RecordBatch>,
    ) -> Option<&'b RecordBatch>
    where
        'a: 'b,
    {
        match self.kind {
            AuxReadKind::RasterCells { .. } => self.source.batch(primary_batch, transform_outputs),
            AuxReadKind::Column { .. } => None,
        }
    }

    /// Whether this read's values read as categories: a column by its
    /// binding's declared type or its dtype ([`column_reads_categorical`]);
    /// a raster image's cells never, they are numbers.
    pub(in crate::render) fn reads_categorical(&self, primary_batch: &RecordBatch, transform_outputs: &HashMap<String, RecordBatch>) -> bool {
        match self.kind {
            AuxReadKind::Column { binding } => {
                let located = self.source.locate(self.field, primary_batch, transform_outputs);
                located.is_some() && column_reads_categorical(binding, located)
            }
            AuxReadKind::RasterCells { .. } => false,
        }
    }

    /// The numeric extent this read contributes, and whether it holds any
    /// value: a column's `extent` (on a faceted chart, a read of the chart's
    /// data unioned with the all-panels batch), or a raster image's cells'
    /// finite extent as the image draws it
    /// ([`crate::render::marks::image::raster_cell_domain`]). `None` when
    /// there is nothing to read (a batch lacking the field, a raster with no
    /// finite cell).
    fn extent(
        &self,
        primary_batch: &RecordBatch,
        transform_outputs: &HashMap<String, RecordBatch>,
        facet_shared: bool,
        extent: &impl Fn(&dyn Array, &str) -> Result<(f64, f64), RenderError>,
    ) -> Result<Option<ReadExtent>, RenderError> {
        use crate::render::marks::image::{raster_cell_domain, raster_cell_extent};
        match self.kind {
            AuxReadKind::RasterCells { .. } => Ok(self
                .source
                .batch(primary_batch, transform_outputs)
                .and_then(raster_cell_extent)
                .map(|cells| ReadExtent { extent: raster_cell_domain(cells), has_values: true })),
            AuxReadKind::Column { .. } => {
                let Some(located) = self.source.locate(self.field, primary_batch, transform_outputs) else {
                    return Ok(None);
                };
                let mut read_extent = extent(located.col, self.field)?;
                if facet_shared && self.source == LayerSource::Chart {
                    read_extent = super::union_panel_with_global_extent(read_extent, self.field, transform_outputs);
                }
                Ok(Some(ReadExtent { extent: read_extent, has_values: located.col.null_count() != located.col.len() }))
            }
        }
    }
}

/// The extent one [`AuxRead`] contributes, and whether it holds any value
/// (a column all null contributes its extent only when no read has one).
struct ReadExtent {
    extent: (f64, f64),
    has_values: bool,
}

/// The reads that make up a layered spec's non-positional `channel` domain,
/// in layer order, one per layer: every layer that draws the channel,
/// reading the field it draws from the batch it draws ([`LayerSource::of`]).
/// A layer draws the channel when its mark draws it
/// ([`Mark::draws_channel`]) and its batch carries the column it binds there.
/// The column gate (`prepare::validate_layer_columns`) has already refused
/// every drawn binding whose column a layer's batch lacks, but for an
/// inherited one a statistic's output drops, which that layer draws without:
/// so a layer whose batch lacks the column is exactly a layer that does not
/// draw it. A layer binding the channel itself draws its own field; a layer
/// binding nothing there draws the chart-level binding, which `prepare` gives
/// a layer on the chart's data (`inherit_from`) and a layer on its own output
/// (`inherit_non_positional`) alike. The same layer-to-batch rule the
/// positional union reads by ([`seed_reads`]), without its positional-only
/// skip of `data_source` layers.
///
/// A raster image draws `color` through its cells when its binding names the
/// field they hold ([`crate::render::marks::image::image_draws_color`]): its
/// read is the cells of its Raster batch ([`AuxReadKind::RasterCells`]).
///
/// The first read is the establishing layer's: the first layer that draws
/// the channel, layer 0 when it does, the layer whose binding the rendering
/// encoding carries (`prepare`'s `establishing_encoding`). Empty when no layer
/// draws the channel, which then gets no scale and no legend.
pub(in crate::render) fn aux_reads<'a>(
    layers: &'a [Layer],
    chart_encoding: &'a crate::spec::encoding::Encoding,
    channel: crate::spec::encoding::AuxiliaryScaleChannel,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Vec<AuxRead<'a>> {
    let mut reads: Vec<AuxRead<'a>> = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let Some(binding) = channel.binding(&layer.encoding).as_ref().or(channel.binding(chart_encoding).as_ref())
        else {
            continue;
        };
        let field = binding.field.as_str();
        let source = LayerSource::of(layer);
        if channel.drawn_as() == crate::spec::layer::InheritedChannel::Color {
            if let Some(cmap) = raster_cells_drawn(layer, chart_encoding, primary_batch, transform_outputs) {
                let kind = AuxReadKind::RasterCells { cmap, entry: layer.entry.as_deref() };
                reads.push(AuxRead { layer: index, mark: layer.mark, source, field, kind });
                continue;
            }
        }
        if !layer.mark.draws_channel(channel.drawn_as()) || source.locate(field, primary_batch, transform_outputs).is_none() {
            continue;
        }
        reads.push(AuxRead { layer: index, mark: layer.mark, source, field, kind: AuxReadKind::Column { binding } });
    }
    reads
}

/// The colormap a raster image `layer` draws its cells in, when it draws
/// color through them ([`crate::render::marks::image::image_draws_color`]):
/// `Some(cmap)` (itself `None` when the layer names no colormap); `None` for
/// every other layer.
fn raster_cells_drawn<'a>(
    layer: &'a Layer,
    chart_encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> Option<Option<&'a str>> {
    let batch = LayerSource::of(layer).batch(primary_batch, transform_outputs)?;
    let cells = crate::render::marks::image::raster_value_field(batch);
    crate::render::marks::image::image_draws_color(layer.mark, &layer.encoding, chart_encoding, cells)
        .then(|| layer.mark_style.as_ref().and_then(|style| style.cmap.as_deref()))
}

/// The reads of the cells of every raster image that draws no `color` (its
/// color unbound, or bound to another field, which warns): one
/// [`AuxReadKind::RasterCells`] read per url-less image layer whose batch is
/// a Raster output, named by the field its cells hold; for a spec with no
/// layers, its one batch when the chart is such an image. The scale these
/// reads make (`scale_resolve`'s `raster_cells` scale) colors every such
/// image, so one cell scale spans every panel of a faceted raster and every
/// image of a chart, as the bound color scale does.
///
/// Whether an image draws color is one answer, its own: the layer's color
/// binding, else the one it inherits from the authored `chart_encoding`
/// ([`aux_reads`] asks the same of the same pair, and the image builder of
/// the layer's completed encoding). A sibling layer's color binding, which
/// the rendering encoding may carry as the establishing one, never makes an
/// image draw color.
pub(in crate::render) fn raster_cell_reads<'a>(
    spec: &'a ChartSpec,
    chart_encoding: &crate::spec::encoding::Encoding,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Vec<AuxRead<'a>> {
    use crate::render::marks::image::{image_draws_color, raster_value_field, url_tile_field};
    let undrawn = |mark: Mark,
                   encoding: &crate::spec::encoding::Encoding,
                   parent: &crate::spec::encoding::Encoding,
                   style: Option<&'a crate::spec::mark_style::MarkKwargsSpec>,
                   entry: Option<&'a str>,
                   source: LayerSource<'a>,
                   index: usize| {
        let url = encoding.url.as_ref().or(parent.url.as_ref());
        if mark != Mark::Image || url_tile_field(url).is_some() {
            return None;
        }
        let batch = source.batch(primary_batch, transform_outputs)?;
        let field = raster_value_field(batch)?;
        if image_draws_color(mark, encoding, parent, Some(field)) {
            return None;
        }
        let cmap = style.and_then(|style| style.cmap.as_deref());
        Some(AuxRead { layer: index, mark, source, field, kind: AuxReadKind::RasterCells { cmap, entry } })
    };
    match spec.layers.as_deref() {
        Some(layers) => layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                let style = layer.mark_style.as_ref();
                undrawn(layer.mark, &layer.encoding, chart_encoding, style, layer.entry.as_deref(), LayerSource::of(layer), index)
            })
            .collect(),
        None => undrawn(spec.mark, &spec.encoding, &spec.encoding, spec.mark_style.as_ref(), spec.entry.as_deref(), LayerSource::Chart, 0)
            .into_iter()
            .collect(),
    }
}

/// The reads of an overlay group's auxiliary `channel`: every member's reads
/// of it, in member order, each read from the batch it draws and named by the
/// index its layer has in the layer list the members merged with `+` form
/// (each member's layers in order, an unlayered member one layer). A layered
/// member reads as [`aux_reads`] reads its layers; an unlayered one draws its
/// rendering encoding's binding, which carries only a channel its mark draws,
/// from its one batch ([`locate_field`]), as its own scale reads it; a
/// binding to a column that batch lacks is refused as that chart alone
/// refuses it ([`RenderError::UnknownColumn`]), since the group resolves the
/// member's scale in its place. A layered member's bindings were judged by
/// its own column gate when it was prepared.
pub(in crate::render) fn overlay_aux_reads<'a>(
    members: &[super::LeafResolveInput<'a>],
    channel: crate::spec::encoding::AuxiliaryScaleChannel,
) -> Result<Vec<AuxRead<'a>>, RenderError> {
    let mut reads: Vec<AuxRead<'a>> = Vec::new();
    let mut first_layer = 0;
    for input in members {
        let (final_batch, transform_outputs) = (input.final_batch, input.transform_outputs);
        match input.spec.layers.as_deref() {
            Some(layers) => {
                for read in aux_reads(layers, &input.spec.encoding, channel, final_batch, transform_outputs) {
                    let Some(batch) = read.source.batch(final_batch, transform_outputs) else { continue };
                    reads.push(AuxRead { layer: first_layer + read.layer, source: LayerSource::Batch(batch), ..read });
                }
                first_layer += layers.len();
            }
            None => {
                first_layer += 1;
                let Some(binding) = channel.binding(input.encoding).as_ref() else { continue };
                let field = binding.field.as_str();
                let located =
                    locate_field(field, final_batch).ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
                reads.push(AuxRead {
                    layer: first_layer - 1,
                    mark: input.spec.mark,
                    source: LayerSource::Batch(located.batch),
                    field,
                    kind: AuxReadKind::Column { binding },
                });
            }
        }
    }
    Ok(reads)
}

/// The raster cell reads of an overlay group ([`raster_cell_reads`] over
/// each member, in member order, numbered as [`overlay_aux_reads`] numbers
/// its layers), so the group's images share one cell scale, as the same
/// layers merged with `+` do.
pub(in crate::render) fn overlay_raster_cell_reads<'a>(members: &[super::LeafResolveInput<'a>]) -> Vec<AuxRead<'a>> {
    let mut reads = Vec::new();
    let mut first_layer = 0;
    for input in members {
        for read in raster_cell_reads(input.spec, &input.spec.encoding, input.final_batch, input.transform_outputs) {
            let Some(batch) = read.source.batch(input.final_batch, input.transform_outputs) else { continue };
            reads.push(AuxRead { layer: first_layer + read.layer, source: LayerSource::Batch(batch), ..read });
        }
        first_layer += input.spec.layers.as_ref().map_or(1, Vec::len);
    }
    reads
}

/// The layer of the area that groups a color scale by category (FA-5): the
/// first area layer among the layers drawing color (`reads`, a layered
/// spec's color reads), or layer 0 for an unlayered area chart, which has no
/// reads. Only an area that draws color groups it, so an area beside the
/// layers that do changes nothing, in whatever order they are layered.
pub(in crate::render) fn color_grouping_area(chart_mark: Mark, reads: &[AuxRead<'_>]) -> Option<usize> {
    match reads {
        [] => (chart_mark == Mark::Area).then_some(0),
        reads => reads.iter().find(|read| read.mark == Mark::Area).map(|read| read.layer),
    }
}

/// The reads a chart-level `channel` scale admits into its domain: those whose
/// column reads as the scale does, categorical when `scale_reads_categorical`.
/// The establishing-scale rule of the layered axes ([`AxisMembers`]): the
/// first read, the establishing layer's ([`aux_reads`]), draws the rendering
/// encoding's field, which decided the scale's kind, so it is always admitted
/// and establishes it. Every later
/// read is classified by [`column_reads_categorical`] over the column it
/// draws; one that reads the other way stays out of the domain, and one
/// [`RenderWarning::LayerScaleExcluded`] names its layer, its marks drawing
/// through the shared scale ([`crate::render::ExcludedLayerPlacement::SharedAxis`]).
/// A raster image's cells under a categorical scale would draw nothing, so
/// they are refused ([`RenderError::RasterColorNotQuantitative`]) rather
/// than left out.
pub(in crate::render) fn admitted_aux_reads<'a>(
    reads: &[AuxRead<'a>],
    channel: &str,
    scale_reads_categorical: bool,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<AuxRead<'a>>, RenderError> {
    let Some((lead, rest)) = reads.split_first() else { return Ok(Vec::new()) };
    let mut admitted = vec![*lead];
    for read in rest {
        let conflicts = read.reads_categorical(primary_batch, transform_outputs) != scale_reads_categorical
            && (read.is_raster_cells() || read.locate(primary_batch, transform_outputs).is_some());
        if conflicts && scale_reads_categorical {
            let reason = crate::render::RasterColorReason::CategoricalScale { layer: lead.layer, field: lead.field.to_string() };
            if let Some(refusal) = read.raster_color_refusal(reason) {
                return Err(refusal);
            }
        }
        if conflicts {
            warnings.push(RenderWarning::LayerScaleExcluded {
                layer: read.layer,
                establishing: lead.layer,
                channel: channel.to_string(),
                placement: crate::render::ExcludedLayerPlacement::SharedAxis,
            });
        } else {
            admitted.push(*read);
        }
    }
    Ok(admitted)
}

/// The numeric domain of a chart-level non-positional channel on a layered
/// spec: the union of `extent` over each read's column, read from the batch
/// that layer draws. On a faceted chart a read of the chart's data also takes
/// in the all-panels batch, as the unlayered read does
/// ([`super::union_panel_with_global_extent`]); a layer's own output is
/// already the all-panels one. A column with no value joins only when no
/// read has one, so the result is exactly the single read's extent whenever
/// one layer draws the channel. A raster image's cells join as the image
/// draws them ([`crate::render::marks::image::raster_cell_domain`]: from
/// zero when every cell is non-negative), so the scale every drawing layer
/// reads, and its colorbar, span the domain the image draws. Every read
/// [`aux_reads`] lists draws its
/// field, so the result is `None` only for no reads: an unlayered spec, which
/// reads its one batch instead.
pub(in crate::render) fn aux_numeric_union(
    reads: &[AuxRead<'_>],
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    extent: impl Fn(&dyn Array, &str) -> Result<(f64, f64), RenderError>,
) -> Result<Option<(f64, f64)>, RenderError> {
    let mut union: Option<(f64, f64)> = None;
    let mut first: Option<(f64, f64)> = None;
    for read in reads {
        let Some(ReadExtent { extent: read_extent, has_values }) =
            read.extent(primary_batch, transform_outputs, facet_shared, &extent)?
        else {
            continue;
        };
        first.get_or_insert(read_extent);
        if !has_values {
            continue;
        }
        union = Some(union.map_or(read_extent, |(lo, hi)| (lo.min(read_extent.0), hi.max(read_extent.1))));
    }
    Ok(union.or(first))
}

/// The categorical domain of a chart-level color, shape or stroke-dash on a
/// layered spec: each read's distinct values in first-appearance order,
/// unioned in layer order, as [`positional_category_union`] unions an axis's
/// categories. A read of
/// the chart's data takes the all-panels batch on a faceted chart
/// ([`super::shared_categorical_batch`]). Returns the domain and the first
/// contributing batch, which a data-aware `sort=` aggregates over; `None`
/// for no reads (an unlayered spec, which reads its one batch instead).
pub(in crate::render) fn aux_category_union<'b>(
    reads: &[AuxRead<'b>],
    primary_batch: &'b RecordBatch,
    transform_outputs: &'b HashMap<String, RecordBatch>,
    facet_shared: bool,
) -> Result<Option<(Vec<String>, &'b RecordBatch)>, RenderError> {
    let mut union: Option<(Vec<String>, &'b RecordBatch)> = None;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for read in reads.iter().filter(|read| !read.is_raster_cells()) {
        let batch = match read.source {
            LayerSource::Chart => {
                super::shared_categorical_batch(primary_batch, read.field, transform_outputs, facet_shared)
            }
            LayerSource::Output(_) | LayerSource::Batch(_) => {
                let Some(batch) = read.source.batch(primary_batch, transform_outputs) else { continue };
                batch
            }
        };
        if batch.column_by_name(read.field).is_none() {
            continue;
        }
        let values = super::distinct_values_in_order(batch, read.field)?;
        let (domain, _) = union.get_or_insert_with(|| (Vec::new(), batch));
        for value in values {
            if seen.insert(value.clone()) {
                domain.push(value);
            }
        }
    }
    Ok(union)
}

/// The positional bindings of one encoding for the post-flip `channel`: the
/// value, its paired `x2`/`y2` span and the opposite channel's value.
struct ChannelSlots<'a> {
    value: Option<&'a str>,
    span: Option<&'a str>,
    opposite: Option<&'a str>,
}

/// `encoding`'s [`ChannelSlots`] for the post-flip `channel`, read through
/// the flip when `flipped` (its authored `y` then occupies the x axis).
fn channel_slots<'a>(encoding: &'a crate::spec::encoding::Encoding, flipped: bool, channel: &str) -> ChannelSlots<'a> {
    let field = |e: &'a Option<crate::spec::encoding::EncodingSpec>| e.as_ref().map(|e| e.field.as_str());
    let (x, y) = (
        ChannelSlots { value: field(&encoding.x), span: field(&encoding.x2), opposite: field(&encoding.y) },
        ChannelSlots { value: field(&encoding.y), span: field(&encoding.y2), opposite: field(&encoding.x) },
    );
    match (channel, flipped) {
        ("x", false) | ("y", true) => x,
        ("y", false) | ("x", true) => y,
        _ => ChannelSlots { value: None, span: None, opposite: None },
    }
}

/// Whether `layer`'s effective coord exchanges its positional channels.
fn layer_flips(layer: &Layer, chart_coord: Option<&crate::spec::coord::CoordKind>) -> bool {
    crate::spec::coord::flips_positional_channels(layer.effective_coord(chart_coord))
}

/// The column an axis's bound `field` is typed and ordered by: in a layered
/// spec, the column the layer that establishes `channel` draws
/// ([`establishing_axis_layer`]: layer 0 when it draws the channel, else the
/// first layer that does), read from the batch it draws. A raster image draws
/// its axis through its extent columns, so its first one types the axis
/// ([`crate::render::marks::image::drawn_columns`]). A spec with
/// no layers reads its one batch ([`locate_field`]). `None` when that batch
/// lacks the column: no other batch's same-named column describes what the
/// establishing layer draws.
pub(in crate::render) fn locate_seed_field<'a>(
    channel: &str,
    field: &str,
    spec: &'a ChartSpec,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Option<LocatedColumn<'a>> {
    let Some((_, layer)) = spec.layers.as_deref().and_then(|layers| establishing_axis_layer(spec, layers, channel)) else {
        return locate_field(field, primary_batch);
    };
    let url = layer.encoding.url.as_ref().or(spec.encoding.url.as_ref());
    let drawn = crate::render::marks::image::drawn_columns(layer.mark, url, channel, field);
    LayerSource::of(layer).locate(drawn.first()?, primary_batch, transform_outputs)
}

/// Aggregate operation used by data-aware sort forms (channel shorthand and
/// sort-field object). Vega-Lite's default `op` is `"sum"`.
/// The explicit continuous extent an encoding's own `scale=` declares, or
/// `None` when it declares none.
///
/// The single answer to "did the user pin this channel's domain at the
/// encoding level" — the question the documented cascade turns on. Asked by
/// the color resolver (`color.rs::scale_explicit_domain`, which seeds a
/// colorbar's extent) and by the chart-level scale-domain config
/// (`prepare::apply_axis_domain_config_for`, whose four fields lose to it
/// silently, D3/spec §4.2).
///
/// Delegates to [`ScaleSpec::positional_extent`](crate::spec::encoding::ScaleSpec::positional_extent),
/// which is compiler-enforced exhaustive over the scale variants and already
/// encodes the distinctions that matter here: `Quantile`/`Threshold`/
/// `BinOrdinal` domains are binning artifacts rather than extents, ordinal
/// families have no continuous extent at all, and a continuous scale carrying
/// no `domain` (e.g. `LinearScale(clamp=True)`) declares nothing — so it must
/// NOT out-rank a chart-level `domain_max`. Testing `scale.is_some()` instead
/// gets that last case wrong, which is exactly how the config fields went
/// silently inert for a common encoding shape.
pub(in crate::render) fn encoding_explicit_extent(
    enc: &crate::spec::encoding::EncodingSpec,
) -> Option<(f64, f64)> {
    let extent = enc.scale.as_ref()?.positional_extent()?;
    (extent.len() >= 2).then(|| (extent[0], extent[extent.len() - 1]))
}

/// Extract the declared `domain` from a `ScaleSpec::Ordinal` encoding.
///
/// Returns `Some(Vec<String>)` when the encoding has `scale = ScaleSpec::Ordinal`
/// with an explicit non-empty `domain`. Returns `None` for all other scale types
/// and when no domain is declared (the caller falls back to data first-appearance
/// order).
///
/// Shared by every ordinal auxiliary channel's D1 explicit-domain path — color
/// (`color.rs`'s categorical resolver), shape, and stroke-dash
/// (`auxiliary.rs`'s `build_shape_scale`/`build_stroke_dash_scale`, #123) — so
/// the one resolution rule lives once rather than three times.
pub(in crate::render) fn explicit_ordinal_domain(
    enc: &crate::spec::encoding::EncodingSpec,
) -> Option<Vec<String>> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Ordinal { domain: Some(d), .. } if !d.is_empty() => Some(d.clone()),
        _ => None,
    }
}

/// Extract the declared `range` from a `ScaleSpec::Ordinal` encoding,
/// whatever its entries' types.
///
/// Returns `Some` for a non-empty declared range, `None` for every other
/// scale type and when no range is declared (or it is declared empty). It
/// does not require string entries: it answers "was a range declared at
/// all", so `build_stroke_dash_scale` can refuse any explicit range (its
/// output is a fixed palette index) and `build_shape_scale` can tell "no
/// range" (default palette) from "a declared range with a non-shape entry"
/// (refuse, #123). Color's all-string question is
/// [`super::color`]'s `explicit_string_range`, built on this.
pub(in crate::render) fn explicit_ordinal_range(
    enc: &crate::spec::encoding::EncodingSpec,
) -> Option<&[crate::scale::ordinal::OrdinalRangeValue]> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Ordinal { range: Some(r), .. } if !r.is_empty() => Some(r.as_slice()),
        _ => None,
    }
}

/// Data needed to resolve the *data-aware* sort forms (channel shorthand
/// `"x"`/`"-x"`/`"y"`/`"-y"` and the sort-field object). The alphabetical and
/// explicit-array forms ignore this context entirely.
///
/// `category_field` is the categorical axis field whose distinct values make up
/// the domain. `value_field_for_other_channel` resolves the field bound to the
/// *opposite* positional channel (used by the channel-shorthand form: `"-y"` on
/// an x-categorical axis sorts by the aggregate of the y field). `batch` is the
/// record batch the domain was built from — the same batch carries both the
/// category column and the candidate value columns.
pub(in crate::render) struct SortContext<'a> {
    pub(in crate::render) category_field: &'a str,
    pub(in crate::render) batch: &'a RecordBatch,
    /// Field bound to the RENDERED x channel (if any), for resolving `"x"`/`"-x"`.
    pub(in crate::render) x_field: Option<&'a str>,
    /// Field bound to the RENDERED y channel (if any), for resolving `"y"`/`"-y"`.
    pub(in crate::render) y_field: Option<&'a str>,
    /// Whether the chart carries `CoordFlip` (`spec.coord`), i.e. whether the
    /// two spaces this struct straddles differ at all (#95).
    ///
    /// `x_field`/`y_field` are built from the RESOLVED (post-flip) rendering
    /// encoding — `positional.rs`'s `PositionalFields`, `auxiliary.rs`'s
    /// rendering `Encoding`. The channel token inside a `sort="-y"` shorthand,
    /// by contrast, is the text the user AUTHORED. This flag is what lets
    /// [`apply_channel_shorthand_sort`] move that authored token into the
    /// fields' own space before the lookup, so both sides of it live in one
    /// space. `false` for every unflipped chart → byte-identical.
    pub(in crate::render) coord_flipped: bool,
}

/// Apply `encoding.sort` to an ordinal domain in place.
///
/// Accepted forms (mirrors the Vega-Lite `sort` field):
/// - `"ascending"` — sort alphabetically ascending (locale-independent byte order).
/// - `"descending"` — sort alphabetically descending.
/// - Channel shorthand `"x"`/`"-x"`/`"y"`/`"-y"` — sort the categorical domain by
///   the per-category aggregate of the *other* positional channel's field. The
///   leading `-` selects descending; the default aggregate is `sum`. The token
///   names the channel the user AUTHORED, so under `CoordFlip` a chart sorts
///   identically to its unflipped twin (#95) — see
///   [`apply_channel_shorthand_sort`].
/// - Sort-field object `{"field": "...", "op": "sum"|"mean"|"max"|"min"|
///   "count"|"median", "order": "ascending"|"descending"}` — sort the domain by
///   the per-category aggregate of `field`. `op` defaults to `sum`, `order` to
///   `ascending`.
/// - JSON array of strings — replace domain with that explicit order. Values not
///   present in the original domain are silently ignored; values in the original
///   domain that are absent from the array are appended at the end in their
///   original relative order so no data disappears from the scale.
/// - Absent — no-op (preserves insertion order).
///
/// When a data-aware form references a missing field or an unsupported dtype,
/// the domain is left in insertion order and a
/// [`RenderWarning::SortSpecIgnored`] is pushed onto `warnings` — never a silent
/// no-op and never a panic. A malformed sort-field object refuses with
/// [`RenderError::SortSpecMalformed`].
pub(in crate::render) fn apply_sort_to_domain(
    domain: &mut Vec<String>,
    sort: Option<&serde_json::Value>,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    match sort {
        None => {}
        Some(serde_json::Value::String(s)) if s == "ascending" => {
            domain.sort_unstable();
        }
        Some(serde_json::Value::String(s)) if s == "descending" => {
            domain.sort_unstable_by(|a, b| b.cmp(a));
        }
        Some(serde_json::Value::String(s))
            if matches!(s.as_str(), "x" | "-x" | "y" | "-y") =>
        {
            apply_channel_shorthand_sort(domain, s, ctx, warnings);
        }
        Some(serde_json::Value::String(other)) => {
            // A bare string that isn't a recognized keyword/shorthand. Treating
            // it as a field name would be ambiguous with the keyword forms, so
            // surface it rather than silently no-op.
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("unrecognized sort spec {other:?}"),
            });
        }
        Some(serde_json::Value::Array(arr)) => {
            let explicit: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            // Keep only values that appear in the domain, in explicit order.
            // Then append any domain values not covered by the explicit list.
            let mut reordered: Vec<String> = explicit
                .iter()
                .filter(|v| domain.contains(v))
                .cloned()
                .collect();
            for v in domain.iter() {
                if !explicit.contains(v) {
                    reordered.push(v.clone());
                }
            }
            *domain = reordered;
        }
        Some(serde_json::Value::Object(obj)) => {
            apply_sort_field_object(domain, obj, ctx, warnings)?;
        }
        Some(other) => {
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("unsupported sort spec {other}"),
            });
        }
    }
    Ok(())
}

/// Resolve the channel-shorthand form (`"x"`/`"-x"`/`"y"`/`"-y"`). The domain is
/// reordered by the per-category aggregate (default op = sum) of the field bound
/// to the named channel.
///
/// **The shorthand names an AUTHORED channel** (#95, user-adjudicated; spec
/// §5/§8). `sort="-y"` means "descending by whatever the user's own
/// `.encode(y=...)` bound", which under `CoordFlip` is the field that RENDERS
/// on x. Before this was pinned, the flipped spelling aggregated the category
/// column itself, fell back to insertion order, and warned — so the only
/// spelling that worked under flip was `"-x"`, which names a channel the user
/// never wrote. The sort-FIELD object form (`{"field": …, "op": …}`) names a
/// field rather than a channel and was always flip-invariant; the two forms
/// agreeing is the observable contract, pinned by
/// `prepare`'s `sort_field_object_is_flip_invariant_and_agrees_with_the_shorthand`.
fn apply_channel_shorthand_sort(
    domain: &mut Vec<String>,
    shorthand: &str,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) {
    let (order, channel) = if let Some(ch) = shorthand.strip_prefix('-') {
        (SortOrder::Descending, ch)
    } else {
        (SortOrder::Ascending, shorthand)
    };
    // #95 (AUTHORED semantics, spec §5/§8): `channel` is the literal substring
    // of the `shorthand` STRING the user wrote (`sort="-y"` → `"y"`), so it
    // names the channel their own `.encode(y=...)` bound — the PRE-flip space.
    // `ctx.x_field`/`ctx.y_field` name the RENDERED channels. Move the
    // authored token into the fields' space ONCE, here at the boundary where
    // it enters resolution, rather than re-mapping the fields at the read site
    // (the rule `#113` reinforced): `user_facing_channel` is an involution on
    // `x`/`y`, so the crate's single flip expression serves both directions and
    // there is no second mapping to drift from it. Unflipped, this is identity
    // and the lookup is byte-identical to what it always was.
    let rendered_channel = crate::render::prepare::user_facing_channel(channel, ctx.coord_flipped);
    let value_field = match rendered_channel {
        "x" => ctx.x_field,
        "y" => ctx.y_field,
        _ => None,
    };
    // R3 (SortSpecIgnored family): the message keeps naming `channel`, NOT
    // `rendered_channel` — it is the user's own written text, echoed back
    // verbatim, and translating it would contradict the `sort={shorthand:?}`
    // clause right next to it in the same sentence. This is why the two tokens
    // stay separate bindings rather than one shadowed variable; see
    // `sort_shorthand_missing_channel_names_users_own_text_{unflipped,under_flip}`
    // below, which pin the message under both coordinate systems.
    let Some(value_field) = value_field else {
        warnings.push(RenderWarning::SortSpecIgnored {
            reason: format!("sort={shorthand:?} but no field is bound to channel {channel:?}"),
        });
        return;
    };
    sort_domain_by_aggregate(domain, value_field, SortOp::Sum, order, ctx, warnings);
}

/// Resolve the sort-field object form
/// `{"field": ..., "op": ..., "order": ...}`.
///
/// A malformed shape refuses with [`RenderError::SortSpecMalformed`] naming
/// the category field it was sorting; every chart-level, layer and facet
/// slot is already refused with its own location by `prepare`'s up-front
/// walk, so this is the typed backstop for a caller that skips that walk.
fn apply_sort_field_object(
    domain: &mut Vec<String>,
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    let (field, op, order) =
        parse_sort_field_object(obj).map_err(|reason| RenderError::SortSpecMalformed {
            location: ctx.category_field.to_string(),
            reason,
            value: Box::new(serde_json::Value::Object(obj.clone())),
        })?;
    sort_domain_by_aggregate(domain, &field, op, order, ctx, warnings);
    Ok(())
}

/// Shared driver for both data-aware forms: compute the per-category aggregate
/// of `value_field` (using the category column from `ctx`) and reorder `domain`
/// accordingly. On any data failure, leave the domain untouched and warn.
fn sort_domain_by_aggregate(
    domain: &mut [String],
    value_field: &str,
    op: SortOp,
    order: SortOrder,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) {
    let categories = match crate::render::arrow_cast::col_as_str(ctx.batch, ctx.category_field) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("cannot read category field {:?}: {e}", ctx.category_field),
            });
            return;
        }
    };
    // `count` does not read the value column; every other op does.
    let values: Vec<Option<f64>> = if matches!(op, SortOp::Count) {
        vec![None; categories.len()]
    } else {
        match crate::render::arrow_cast::col_as_f64(ctx.batch, value_field) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(RenderWarning::SortSpecIgnored {
                    reason: format!("cannot read sort value field {value_field:?}: {e}"),
                });
                return;
            }
        }
    };
    if categories.len() != values.len() {
        warnings.push(RenderWarning::SortSpecIgnored {
            reason: format!(
                "row-count mismatch sorting {:?} by {value_field:?}",
                ctx.category_field
            ),
        });
        return;
    }

    // Bucket each row's value under its category, skipping null categories and
    // (for value-reading ops) null/non-finite values.
    let mut buckets: HashMap<&str, Vec<f64>> = HashMap::new();
    for (cat, val) in categories.iter().zip(values.iter()) {
        let Some(cat) = cat.as_deref() else { continue };
        let entry = buckets.entry(cat).or_default();
        if matches!(op, SortOp::Count) {
            // Count of rows in the category — the pushed sentinel is unused by
            // `SortOp::Count::aggregate` (it uses slice length).
            entry.push(0.0);
        } else if let Some(v) = val {
            if v.is_finite() {
                entry.push(*v);
            }
        }
    }

    let key_for = |cat: &str| -> f64 {
        match buckets.get(cat) {
            Some(vals) => op.aggregate(vals),
            None => f64::NEG_INFINITY,
        }
    };
    domain.sort_by(|a, b| {
        let ka = key_for(a);
        let kb = key_for(b);
        let ord = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
        match order {
            SortOrder::Ascending => ord,
            SortOrder::Descending => ord.reverse(),
        }
    });
}

/// Would Python's own single-chart zero-anchor injection have anchored this
/// layer on this channel? The eligibility half of the bar zero-anchor rule
/// (T5r; narrowed cycle 2 — spec-review issue 1; extended cycle 2 —
/// rust-quality-review findings 1-3), extracted from `numeric_domain_union`'s
/// union loop so the concept is nameable and directly unit-testable rather
/// than reachable only end-to-end through a 285-line function (batch-F close
/// review). `numeric_domain_union` owns the *arithmetic* (the two guarded
/// assignments that widen the resolved extent to zero); this owns the
/// *question*.
///
/// `channel`/`layer_fields` are read in the channel's own POST-flip
/// coordinate space — the caller normalizes the layer's encoding via
/// `Encoding::flip_positional` before deriving `layer_fields`, so both sides
/// of every read below live in one space. `value_field_contributed` says
/// whether the layer's value-slot field actually resolved to data in some
/// batch, so a merely-DECLARED field (wrong `data_source`, not loaded yet)
/// cannot force a zero-anchor on the strength of its name alone (spec-review
/// finding 3).
///
/// The terms mirror the single-chart injection's guards
/// (`_spec_build.py::_build_encoding_specs`'s zero-anchor block), not just
/// its arithmetic:
///   (a) the value axis there — the anchor is the bar's EXTENT axis only,
///       never its POSITION axis. A bar's value channel follows its
///       orientation ([`bar_value_on_x`]): `y`, or `x` when the bar is
///       horizontal (a categorical `y`, a native horizontal bar), and the
///       other one under `CoordFlip`, which the caller's
///       `coord_flipped`/`encoding` normalization already applies. Dropping
///       this guard is what fired the anchor on a histogram bar's `x`
///       (bin-range position) channel too, squeezing `displot(kind="hist")`
///       bars into a fraction of the panel when the data doesn't straddle
///       zero. A floating horizontal bar such as
///       `_explanation.py::desugar_shap_waterfall`'s (`x=x0`, `x2=x1`,
///       `y=feature`) has its value on `x` and is excluded by guard (b): its
///       value channel carries `x2`.
///   (b) `not _y2_bound` there (the D3 rule) — when this layer's OWN value
///       channel carries a paired span field (`y2`, or `x2` on an x value
///       channel — `layer_fields[1]`), the extent is taken literally from
///       `[value, span]` and force-anchoring to zero would distort a domain
///       that was never meant to touch zero. `desugar_shap_waterfall` is the
///       concrete horizontal instance (its value channel carries `x2`);
///       `y2_bound_bar_layer_never_zero_extends` below pins the vertical one.
///   (c) the layer's own opt-out ([`bar_wants_zero`]) — `layer.mark_style.zero == Some(false)`,
///       the per-layer mirror of `mark_bar(zero=False)` / `_mark_zero`
///       (`spec::mark_style::MarkKwargsSpec::zero`, T5r cycle 2). Live and
///       user-reachable today via the generic per-layer `mark=` passthrough
///       on layered figure functions — e.g. `fm.catplot(..., kind="bar",
///       mark={"bar": {"zero": False}})` resolves the value axis to the raw
///       data range (measured: `58..61.5`, not `0..61.5`) — see that field's
///       doc comment for the plumbing and the one path that stays inert.
///
/// The Python rule's remaining guard (`_y_is_categorical`) needs no mirror:
/// `numeric_domain_union` is only ever called for a quantitative/temporal
/// axis (`build_axis_scale`'s ordinal/nominal branch never reaches here), so
/// an ordinal position channel is excluded structurally, not by an extra
/// flag.
fn layer_wants_bar_zero_anchor(
    layer: &Layer,
    channel: &str,
    value_on_x: bool,
    layer_fields: [Option<&str>; 2],
    value_field_contributed: bool,
) -> bool {
    let is_value_channel = if value_on_x { channel == "x" } else { channel == "y" };
    is_value_channel
        && bar_wants_zero(layer.mark, layer.mark_style.as_ref())
        && value_field_contributed
        && layer_fields[1].is_none()
}

/// Whether a bar draws its value along x, in the coordinates it renders in.
///
/// A bar's value channel follows its orientation: a categorical `y` makes it
/// horizontal (its value on `x`), and a `CoordFlip` exchanges the two. The
/// categorical reading is the one Python's zero-anchor injection asks through
/// the published `ferrum._core.column_reads_categorical`, so both pick the
/// value channel the bar builders draw along.
fn bar_value_on_x(coord_flipped: bool, y_categorical: bool) -> bool {
    coord_flipped != y_categorical
}

/// Whether a mark is a bar that keeps its zero anchor: any bar whose own
/// `mark_bar(zero=False)` (`MarkKwargsSpec::zero`) has not opted out.
fn bar_wants_zero(
    mark: Mark,
    mark_style: Option<&crate::spec::mark_style::MarkKwargsSpec>,
) -> bool {
    mark == Mark::Bar && mark_style.and_then(|m| m.zero).unwrap_or(true)
}

/// Compute the unioned numeric/temporal extent for an axis field across
/// the relevant batches.
///
/// Extent rule:
///   - Every field is read from the batch that draws it, and from no other
///     ([`LayerSource`]): a layer's `data_source` output when it names one,
///     else the chart's data (`primary_batch`). A same-named column elsewhere
///     is data no layer draws: `mark_smooth(ci=)`'s raw `y` beside its
///     fitted `y`, a user's own `lower` beside `mark_errorbar`'s, the merged
///     frame beside each `+` operand's own rows.
///   - On a layered spec the bound field and its paired endpoint are read
///     from the batch of the layer establishing the axis (layer 0 when it
///     draws the channel, else the first layer that does), plus the chart's
///     data when another layer on it binds nothing on this channel and so
///     draws the inherited field ([`seed_reads`]). A batch with no rows contributes nothing,
///     and the reader never looks elsewhere for a field a layer's batch
///     lacks. Reaching past the primary batch is still the point of the
///     per-layer read: `calibration_chart`'s `ReferenceLine` diagonal, whose
///     endpoints [0, 1] live in its own output, widens the axis past a
///     curve inside (0.05, 0.95).
///   - On an unlayered spec the one mark draws `primary_batch`. When the
///     field is absent from it, every named output holding it is read
///     (Phase 8b composite-mark rule — e.g. boxplot whisker fields living in
///     the `box` named output).
///   - When `include_final` is set (faceted chart, this channel resolves
///     `ResolveMode::Shared`), the chart's data also takes in the global
///     all-panels batch at [`FINAL_OUTPUT_KEY`], even though the field is
///     present in `primary_batch` (the per-panel partition). This makes a
///     faceted raw field's positional domain span every panel, so marks
///     scale through the same global domain the shared axis displays.
///     A layer on its own named output reads that output, which already
///     spans every panel. Non-faceted and `Independent`-mode callers pass
///     `false`, keeping the per-panel-only behavior byte-identical.
///   - Every OTHER layer's fields for this channel are unioned too, read in
///     the channel's own coordinate space — see the `spec.layers` loop below.
///   - **Bar zero-anchor (T5r, catplot aggregation; narrowed cycle 2 per
///     spec-review issue 1 and rust-quality-review findings 1-3).** When a
///     `Mark::Bar` layer binds this channel's VALUE slot (never its position
///     slot — `"y"` unflipped, `"x"` under `CoordFlip`, in the channel's own
///     coordinate space, same `coord_flipped` normalization the layer-union
///     loop already applies) with data that actually contributed to the
///     union, AND that layer does not itself bind the paired span field on
///     this channel (`y2`/`x2`), AND the layer does not explicitly opt out
///     (`mark_style.zero == Some(false)`), the resolved extent is widened to
///     include zero — mirroring `resolve_continuous_domain_and_range`'s
///     `zero: true` block (`d[0] > 0.0 → 0.0`, `d[1] < 0.0 → 0.0`) exactly,
///     just applied on the AUTO-INFERRED path instead of an explicit
///     `ScaleSpec`. The eligibility question — all four conditions, and why
///     each one mirrors a guard of the single-chart injection
///     (`_spec_build.py::_build_encoding_specs`: `axis == "y"`,
///     `_y2_bound`'s D3 rule, `_mark_zero`) — lives in
///     [`layer_wants_bar_zero_anchor`], which owns the per-term rationale;
///     this function owns only the arithmetic. A single (non-layered)
///     `mark_bar` never reaches this function at all: Python's
///     `_spec_build.py::_build_encoding_specs` already injects an explicit
///     `scale={"zero": True}` onto that chart's y encoding, which routes
///     through `build_from_scale_spec` and returns before this function is
///     even called. This rule exists for the case Python's injection cannot
///     serve without silently clipping other layers: a layered spec
///     (`catplot(kind="bar")`'s bar + CI-whisker `rule`), where an explicit
///     per-layer `ScaleSpec` would replace this cross-layer union with a
///     single-column domain and clip whichever layer's field extends
///     furthest (measured: 3 of 4 upper CI whiskers vanished off-panel).
///     Scoped to `spec.layers` only (never `spec.mark`), so a genuinely
///     unlayered spec's auto path is untouched — byte-identical, because a
///     single-chart bar never reaches the auto path in the first place.
///
///     One entry path is newer than the paragraph above and not implied by it
///     (#247A): a FLAT layered spec whose chart-level encoding carries the
///     injected bar scale used to return at `build_axis_scale`'s explicit-scale
///     bypass and never arrive here at all. That bypass now yields when the
///     scale is `system_injected`, so `bar + point` reaches this union where it
///     previously could not. The gate that decides it, and the reason
///     provenance rather than deleting the injection was the route, are
///     documented at that bypass in
///     [`super::positional::build_axis_scale`] — read the two together; this
///     doc owns the arithmetic and its eligibility terms, not the question of
///     which specs get here.
pub(in crate::render) fn numeric_domain_union(
    channel: &str,
    field: &str,
    paired_field: Option<&str>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
) -> Result<(f64, f64), RenderError> {
    numeric_domain_union_over(
        channel,
        field,
        paired_field,
        primary_batch,
        transform_outputs,
        spec,
        include_final,
        LayerUnion::Every,
    )
}

/// Which of `spec.layers` [`numeric_domain_union_over`] reads beyond the bound
/// field and its paired endpoint.
#[derive(Clone, Copy)]
pub(in crate::render) enum LayerUnion<'a> {
    /// The auto path: every layer, and an eligible bar layer anchors the
    /// extent at zero.
    Every,
    /// A caller's scale that states no `domain`: the layers whose own
    /// encoding for this channel carries this same scale, and the layers that
    /// declare no scale there (none at all, or only the bar anchor ferrum
    /// injects). A composite mark desugars into several layers that each hold
    /// a copy of the one scale the caller wrote on it, so the scale spans all
    /// of them; a scale-less sibling draws against this axis, so its data
    /// joins it. A layer carrying a different scale, or one stating its own
    /// `domain`, answers to that scale and stays out. The scale's own `zero`
    /// decides the zero anchor, so the auto path's bar rule does not apply,
    /// and the extent is returned without the auto path's single-value
    /// widening, exactly as an unlayered caller scale reads its column: the
    /// scale's own resolution handles a degenerate domain either way. An
    /// all-null extent still takes the auto path's `(0, 1)` fallback, so no
    /// infinite endpoint leaves this function.
    CarryingScale(&'a crate::spec::encoding::ScaleSpec),
}

impl LayerUnion<'_> {
    /// Whether a layer, or a composite overlay leaf, whose own encoding for
    /// the channel declares `own` joins this union.
    pub(in crate::render) fn admits(self, own: Option<&crate::spec::encoding::ScaleSpec>) -> bool {
        match self {
            LayerUnion::Every => true,
            LayerUnion::CarryingScale(scale) => {
                own.filter(|s| !s.is_system_injected()).is_none_or(|s| s == scale)
            }
        }
    }
}

/// One member of a layered axis on one channel.
#[derive(Clone, Copy)]
pub(in crate::render) struct AxisMember<'a> {
    /// The member's index on its route: its layer index on the flat layered
    /// route, its position in the group on a composite `Overlay`.
    pub(in crate::render) index: usize,
    /// The member's own scale on the channel, if its encoding binds one.
    pub(in crate::render) own_scale: Option<&'a crate::spec::encoding::ScaleSpec>,
}

/// The members of one layered axis, in member order, with the one judgement
/// the establishing-scale rule makes about the first of them. Each layering
/// route builds it from one member list ([`layered_axis`] for the flat
/// layered route, `composite::overlay_axis` for a composite `Overlay` group),
/// and [`establishing_caller_scale`] and [`warn_members_off_caller_scale`]
/// both read it, so they cannot judge different leads.
pub(in crate::render) struct AxisMembers<I> {
    members: I,
    /// The first member's index when it decides its own axis: it carries no
    /// scale of a caller's, and its column there reads categorical
    /// ([`column_reads_categorical`]).
    deciding_lead: Option<usize>,
}

impl<'a, I: Iterator<Item = AxisMember<'a>> + Clone> AxisMembers<I> {
    /// `members`, with the first one classified by `reads_categorical`, which
    /// is asked of that member's index only when it carries no caller scale.
    pub(in crate::render) fn new(members: I, reads_categorical: impl FnOnce(usize) -> bool) -> Self {
        let deciding_lead = members
            .clone()
            .next()
            .filter(|member| member.own_scale.is_none_or(|s| s.is_system_injected()))
            .map(|member| member.index)
            .filter(|&idx| reads_categorical(idx));
        Self { members, deciding_lead }
    }
}

/// Which member's scale establishes a layered axis on one channel, for both
/// the flat layered route and a composite `Overlay` group.
///
/// The establishing scale is the first one a caller wrote, in member order,
/// whatever its kind; the bar anchor ferrum injects is no scale. When it is
/// continuous, it and the index of the member carrying it are returned. The
/// members [`LayerUnion::CarryingScale`] admits (those declaring no scale, and
/// those carrying an equal copy) resolve on it; a member carrying another
/// scale, or its own `domain`, stays out. Only a scale stating no `domain`
/// spans the admitted members' data: a stated `domain` is the axis as
/// written. A categorical scale takes its categories from its own column and
/// keeps the categorical axis's own sharing, and a group with no caller scale
/// takes the auto path; both return `None`.
///
/// The first member decides its own axis when it carries no scale and its
/// column there reads categorical ([`AxisMembers`]): the axis is that
/// member's categorical one, so a later member's scale establishes nothing
/// and `None` is returned. "First" means the first member of the channel's
/// member list, not the first layer: on `y` the flat route's list skips
/// `independent_y` layers ([`layered_axis`]), so there the deciding member
/// is the first layer sharing the axis.
pub(in crate::render) fn establishing_caller_scale<'a, I: Iterator<Item = AxisMember<'a>> + Clone>(
    axis: &AxisMembers<I>,
) -> Option<(usize, &'a crate::spec::encoding::ScaleSpec)> {
    if axis.deciding_lead.is_some() {
        return None;
    }
    let (idx, scale) = first_caller_scale(axis.members.clone())?;
    (!is_categorical_scale(scale)).then_some((idx, scale))
}

pub(in crate::render) fn is_categorical_scale(scale: &crate::spec::encoding::ScaleSpec) -> bool {
    use crate::spec::encoding::ScaleSpec;
    matches!(scale, ScaleSpec::Ordinal { .. } | ScaleSpec::Band { .. } | ScaleSpec::Point { .. })
}

/// Whether a member whose own scale is `own` stays out of a categorical axis
/// whichever member owns it: a continuous scale a caller wrote is never
/// admitted to one, by a deciding lead ([`establishing_caller_scale`]) or by
/// a categorical caller scale ([`LayerUnion::CarryingScale`]). The bar
/// anchor ferrum injects is no scale.
fn stays_off_categorical_axis(own: Option<&crate::spec::encoding::ScaleSpec>) -> bool {
    own.is_some_and(|s| !s.is_system_injected() && !is_categorical_scale(s))
}

/// Whether a member's column on a layered axis reads categorical: one the
/// continuous-axis refusal cannot place
/// ([`crate::render::arrow_cast::is_numeric`]), or one its encoding types
/// nominal or ordinal. `located` is the member's column in the batch it
/// draws; a column that batch does not hold reads as not categorical.
pub(in crate::render) fn column_reads_categorical(
    enc: &crate::spec::encoding::EncodingSpec,
    located: Option<LocatedColumn<'_>>,
) -> bool {
    use crate::spec::encoding::DataType;
    matches!(enc.type_, Some(DataType::Nominal | DataType::Ordinal))
        || located.is_some_and(|located| dtype_reads_categorical(located.col.data_type()))
}

/// Whether an undeclared encoding over a column of `dtype` reads categorical:
/// any dtype the continuous-axis reader cannot place. The dtype half of
/// [`column_reads_categorical`], published to Python as
/// `ferrum._core.column_reads_categorical`.
pub(in crate::render) fn dtype_reads_categorical(dtype: &arrow::datatypes::DataType) -> bool {
    !crate::render::arrow_cast::is_numeric(dtype)
}

/// The first scale a caller wrote among `members`, in member order, with the
/// index of the member carrying it, whatever its kind; the bar anchor ferrum
/// injects is no scale.
fn first_caller_scale<'a>(
    members: impl IntoIterator<Item = AxisMember<'a>>,
) -> Option<(usize, &'a crate::spec::encoding::ScaleSpec)> {
    members
        .into_iter()
        .find_map(|member| Some((member.index, member.own_scale.filter(|s| !s.is_system_injected())?)))
}

/// A layer's own encoding for the post-flip `channel`, with the layer's coord
/// flip applied: under a layer whose effective coord flips, its authored `y`
/// occupies the x axis (see the coordinate-space note in
/// [`numeric_domain_union_over`]). Only a binding its completed encoding
/// keeps ([`Layer::completed_encoding`]): a channel Python filled that the
/// mark never draws is not the layer's.
fn layer_channel_encoding<'a>(
    layer: &'a Layer,
    spec: &'a ChartSpec,
    channel: &str,
) -> Option<&'a crate::spec::encoding::EncodingSpec> {
    use crate::spec::layer::PositionalChannel::{X, Y};
    let flipped = crate::spec::coord::flips_positional_channels(layer.effective_coord(spec.coord.as_ref()));
    let authored = match (channel, flipped) {
        ("x", false) | ("y", true) => X,
        ("y", false) | ("x", true) => Y,
        _ => return None,
    };
    authored.slot(&layer.completed_encoding(&spec.encoding)).as_ref()?;
    authored.slot(&layer.encoding).as_ref()
}

/// A warning for each member left out of the axis the first caller-written
/// scale among the members owns: one carrying a scale
/// [`LayerUnion::CarryingScale`] does not admit. Such a member is partially
/// honored: its marks still draw, where `placement` says. A continuous first
/// scale establishes the axis ([`establishing_caller_scale`]); a categorical
/// one keeps the categorical axis's own sharing, which a scale-less member
/// joins and a member carrying another scale stays out of all the same.
/// Nothing for the owning member, an admitted one, or a group with no caller
/// scale. A deciding first member ([`AxisMembers`]) owns its axis itself, so
/// every later member that [`stays_off_categorical_axis`] is left out of it;
/// the categorical scales among the rest follow the rule above.
pub(in crate::render) fn warn_members_off_caller_scale<'a, I: Iterator<Item = AxisMember<'a>> + Clone>(
    axis: &AxisMembers<I>,
    channel: &str,
    placement: crate::render::ExcludedLayerPlacement,
    warnings: &mut Vec<RenderWarning>,
) {
    let warning = |layer, establishing| RenderWarning::LayerScaleExcluded {
        layer,
        establishing,
        channel: channel.to_string(),
        placement,
    };
    let deciding_lead = axis.deciding_lead;
    let off_lead =
        move |member: &AxisMember<'a>| deciding_lead.is_some() && stays_off_categorical_axis(member.own_scale);
    if let Some(lead) = deciding_lead {
        warnings.extend(axis.members.clone().filter(off_lead).map(|member| warning(member.index, lead)));
    }
    let rest = axis.members.clone().filter(move |member| !off_lead(member));
    let Some((establishing, scale)) = first_caller_scale(rest.clone()) else { return };
    let union = LayerUnion::CarryingScale(scale);
    warnings.extend(
        rest.filter(|member| !union.admits(member.own_scale)).map(|member| warning(member.index, establishing)),
    );
}

/// The members of a flat layered spec's axis on `channel`: each layer with
/// its own scale there, in layer order. An `independent_y` layer owns its own
/// y axis, so it is no member of the shared `y`, as in
/// [`numeric_domain_union_over`].
fn layered_axis<'a>(
    spec: &'a ChartSpec,
    channel: &'a str,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> AxisMembers<impl Iterator<Item = AxisMember<'a>> + Clone + 'a> {
    let layers = spec.layers.as_deref().unwrap_or_default();
    let members = layers
        .iter()
        .enumerate()
        .filter(move |(_, layer)| !(channel == "y" && layer.independent_y))
        .map(move |(index, layer)| AxisMember {
            index,
            own_scale: layer_channel_encoding(layer, spec, channel).and_then(|e| e.scale.as_ref()),
        });
    AxisMembers::new(members, |idx| {
        layer_channel_encoding(&layers[idx], spec, channel)
            .is_some_and(|enc| {
                let located = LayerSource::of(&layers[idx]).locate(&enc.field, primary_batch, transform_outputs);
                column_reads_categorical(enc, located)
            })
    })
}

/// The scale a flat layered spec's axis on `channel` establishes by
/// [`establishing_caller_scale`] over its layers, with the establishing
/// layer's index.
pub(in crate::render) fn layered_caller_scale<'a>(
    spec: &'a ChartSpec,
    channel: &'a str,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> Option<(usize, &'a crate::spec::encoding::ScaleSpec)> {
    establishing_caller_scale(&layered_axis(spec, channel, primary_batch, transform_outputs))
}

/// [`warn_members_off_caller_scale`] over a flat layered spec's layers on
/// `channel`. One scale serves each axis, so a layer left out of it draws on
/// the shared axis.
pub(in crate::render) fn warn_layers_off_caller_scale(
    spec: &ChartSpec,
    channel: &str,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<RenderWarning>,
) {
    warn_members_off_caller_scale(
        &layered_axis(spec, channel, primary_batch, transform_outputs),
        channel,
        crate::render::ExcludedLayerPlacement::SharedAxis,
        warnings,
    );
}

/// Refuse a flat layered spec whose axis on `channel` a continuous caller
/// scale establishes (see [`layered_caller_scale`]) when a layer's column on
/// that axis cannot sit on it. Every layer draws on the one axis, so every
/// layer is judged against the establishing one, the rule a composite
/// `Overlay` group applies to its members, whether or not the scale states a
/// `domain`. An `independent_y` layer owns its own y axis, and a later layer
/// whose own coord exchanges its channels relative to the panel is judged by
/// `scene_build::validate_layer_coord_exchange_is_servable`; neither is
/// judged here. Asked once per spec, by `prepare::prepare_render_inputs`,
/// before any scale resolution reads a column.
pub(in crate::render) fn require_layers_placeable_on_caller_scale(
    spec: &ChartSpec,
    channel: &str,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> Result<(), RenderError> {
    let (Some(layers), Some((establishing, _))) = (
        spec.layers.as_deref(),
        layered_caller_scale(spec, channel, primary_batch, transform_outputs),
    ) else {
        return Ok(());
    };
    let chart_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    for (idx, layer) in layers.iter().enumerate() {
        let exchanges = idx != 0
            && crate::spec::coord::flips_positional_channels(layer.effective_coord(spec.coord.as_ref())) != chart_flipped;
        if idx == establishing || exchanges || (channel == "y" && layer.independent_y) {
            continue;
        }
        let Some(enc) = layer_channel_encoding(layer, spec, channel) else { continue };
        if let Some(located) = LayerSource::of(layer).locate(&enc.field, primary_batch, transform_outputs) {
            crate::render::arrow_cast::require_placeable_on_continuous_axis(
                idx,
                channel,
                &enc.field,
                located.col.data_type(),
            )?;
        }
    }
    Ok(())
}

/// [`numeric_domain_union`] with the layer selection spelled out.
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn numeric_domain_union_over(
    channel: &str,
    field: &str,
    paired_field: Option<&str>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
    layer_union: LayerUnion<'_>,
) -> Result<(f64, f64), RenderError> {
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    // T5r bar zero-anchor: set when a `Mark::Bar` layer binds this channel's
    // primary (non-paired) field, in the channel's own coordinate space. See
    // this function's doc comment ("Bar zero-anchor") for the full rationale.
    let mut bar_zero_wanted = false;
    let mut accumulate = |c: &dyn Array, source_field: &str| -> Result<(), RenderError> {
        let (a, b) = column_min_max_f64(c).map_err(|_| RenderError::UnsupportedDtype {
            field: source_field.to_string(),
            dtype: format!("{:?}", c.data_type()),
            context: None,
        })?;
        // A column with no value contributes nothing; an integer column's
        // fold would otherwise report its type's `MAX`/`MIN` seeds.
        if c.null_count() == c.len() {
            return Ok(());
        }
        if a < mn { mn = a; }
        if b > mx { mx = b; }
        Ok(())
    };

    // Reads `f` from one source and returns whether that source holds it.
    // `Ok(false)` is the normal "this batch has no such column" outcome, not an
    // error. The bar zero-anchor rule below needs the bool so a layer's
    // merely-DECLARED value field, absent from the batch it draws, cannot
    // force a zero-anchor on the strength of its name alone.
    //
    // `None` is an unlayered spec, whose one mark draws the primary batch:
    // there a field the primary batch lacks is read from every named output
    // holding it (a composite mark's whisker fields in its `box` output).
    //
    // Faceted + Shared: the chart's data also takes in the global all-panels
    // batch so the per-panel domain spans every panel. `FINAL_OUTPUT_KEY` is
    // the concatenated all-panels output already present at the per-panel
    // callsite. A layer on its own named output reads that output alone,
    // which is already the all-panels one.
    let mut read = |source: Option<LayerSource<'_>>, f: &str| -> Result<bool, RenderError> {
        let mut found = false;
        let primary_has = primary_batch.column_by_name(f).is_some();
        let mut take = |batch: &RecordBatch, found: &mut bool| -> Result<(), RenderError> {
            if let Some(c) = batch.column_by_name(f) {
                accumulate(c.as_ref(), f)?;
                *found = true;
            }
            Ok(())
        };
        match source {
            Some(LayerSource::Output(key)) => {
                if let Some(batch) = transform_outputs.get(key) {
                    take(batch, &mut found)?;
                }
            }
            Some(LayerSource::Batch(batch)) => take(batch, &mut found)?,
            Some(LayerSource::Chart) | None => {
                take(primary_batch, &mut found)?;
                for (key, batch) in transform_outputs {
                    let key_is_final = include_final && key == FINAL_OUTPUT_KEY;
                    if key_is_final || (source.is_none() && !primary_has) {
                        take(batch, &mut found)?;
                    }
                }
            }
        }
        Ok(found)
    };

    // The bound field and its paired endpoint: from the primary batch on an
    // unlayered spec, from the batches and fields that draw them on a layered
    // one (see [`seed_reads`]). Which of those reads found its field is kept
    // for the layer loop below, so a layer drawing a seed read's field from
    // its batch is not read twice.
    let seeds: Vec<SeedRead<'_>> = spec
        .layers
        .as_deref()
        .map(|layers| seed_reads(spec, layers, channel, (field, paired_field, None)))
        .unwrap_or_default();
    let mut seed_found: Vec<(SeedRead<'_>, bool)> = Vec::with_capacity(seeds.len());
    if spec.layers.is_none() {
        read(None, field)?;
        if let Some(p) = paired_field {
            read(None, p)?;
        }
    }
    let chart_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    for &seed in &seeds {
        // An inherited field is a later layer's column, judged as that
        // layer's own column is in the loop below: placeable on this axis
        // first, or, for a layer whose own coord exchanges its channels,
        // left to `validate_layer_coord_exchange_is_servable`.
        let exchanges = seed.inheritor.zip(spec.layers.as_deref()).is_some_and(|(index, layers)| {
            layer_flips(&layers[index], spec.coord.as_ref()) != chart_flipped
        });
        if exchanges {
            let found = read(Some(seed.source), seed.field).unwrap_or(false);
            seed_found.push((seed, found));
            if let Some(span) = seed.span {
                read(Some(seed.source), span).unwrap_or(false);
            }
            continue;
        }
        if let (Some(index), Some(located)) =
            (seed.inheritor, seed.source.locate(seed.field, primary_batch, transform_outputs))
        {
            crate::render::arrow_cast::require_placeable_on_continuous_axis(
                index,
                channel,
                seed.field,
                located.col.data_type(),
            )?;
        }
        seed_found.push((seed, read(Some(seed.source), seed.field)?));
        if let Some(span) = seed.span {
            read(Some(seed.source), span)?;
        }
    }

    // Also union domains from other layers' fields for the same channel, each
    // read from the batch that layer draws ([`LayerSource`]). When two Charts
    // with different DataFrames are composed via `+`, the RHS layer reads a
    // named output of its own, so the bound field (layer 0's) covers the LHS
    // data only; the RHS layer's field joins it so the shared scale domain
    // spans both layers' data ranges. A same-named column in a batch a layer
    // does not draw (the raw `y` beside a smooth layer's fitted `y`) widens
    // nothing.
    //
    // Also covers each OTHER layer's paired channel field (x2 for "x", y2 for
    // "y") — batch-A Task 13. Before this, a layer's OWN x2/y2 was never
    // unioned in here (only its bare x/y was), which this module's own
    // doc comment above already names as required "when a layer's
    // data_source points at a named transform whose output extends past the
    // primary batch" (the `calibration_chart` `ReferenceLine` example) — that
    // prior example happens to need only bare x/y (its endpoints are two
    // ROWS of one field, not one row's x2/y2), so the gap stayed latent until
    // `mark_qq(line=True)`'s reference-diagonal `rule` layer, whose single-row
    // endpoints live in x/y AND x2/y2 (`qq_line_x_start`/`_end`,
    // `qq_line_y_start`/`_end`), all four on a named `data_source` distinct
    // from the primary/chart field. Without this, `qq_line_y_end` was never
    // reachable by any domain union, so `to_pixel_f64` returned `None` for
    // the far endpoint and the reference diagonal silently drew 0 nodes even
    // though `marks/rule.rs`'s four-numeric branch resolved correctly.
    if let Some(layers) = &spec.layers {
        // COORDINATE SPACE (spec §4.4, "Extended 2026-09-02, T13 quality
        // review", ruling 1). `channel`/`field` name the axis being resolved,
        // i.e. the POST-flip space: the caller derived them from the rendering
        // encoding `prepare::build_layers` already swapped. `spec.layers`, by
        // contrast, is the spec as AUTHORED — `prepare` swaps its own
        // `LayerPrepared` copies, never the spec — so under `CoordFlip` a
        // layer's `y`/`y2` is what post-flip occupies the x axis. Reading the
        // layer's like-named channel therefore unions the WRONG axis: a rule
        // layer's `y2` widening the y domain when its endpoint lives on x is a
        // visible axis blowout (the y axis running to the far endpoint while
        // the endpoint itself is unreachable on x). Move the layer encoding
        // into the channel's own space first, via the same
        // `Encoding::flip_positional` `LayerPrepared::flip_coords` applies, and
        // both sides of every read below live in one space.
        //
        // WHOSE space is per layer (`#172`(1)): the layer `build_layers` swaps
        // is the one whose OWN effective coord flips, so that is the layer whose
        // authored `y` occupies this x axis. Reading the chart's flip for a
        // layer that carries its own coord unions the wrong channel — the same
        // blowout described above, one level down.
        //
        // Flipping the layer's encoding costs a clone, so it is skipped for the
        // (overwhelmingly common) unflipped layer exactly as before.
        //
        // The panel's own flip state (spec.coord alone, not any layer's
        // effective one) — needed below to tell a layer whose OWN coord
        // exchanges its channels relative to the panel from every other
        // layer, the same distinction
        // `scene_build::validate_layer_coord_exchange_is_servable` draws
        // between its `chart_flipped`/`layer_flipped` parameters.
        let chart_flipped =
            crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
        for (layer_index, layer) in layers.iter().enumerate() {
            let coord_flipped = crate::spec::coord::flips_positional_channels(
                layer.effective_coord(spec.coord.as_ref()),
            );
            // A layer beyond the first whose own flip state DIFFERS from the
            // panel's exchanges its positional channels relative to the axes
            // every layer shares (`#172`(1)). That is a distinct question
            // from "is this column numeric or temporal" — the column may be
            // perfectly fine on the axis it was AUTHORED for, just not on the
            // one it lands on post-exchange — and it already has its own
            // answer, asked once the scales this union produces are in hand
            // (`scene_build::validate_layer_coord_exchange_is_servable`,
            // `LayerCoordUnplaceableReason::ExchangedChannelUnservable`). This
            // union defers to it rather than reaching the same verdict first
            // with a message that does not name the coord: a swallowed
            // contribution from an exchanging layer is exactly the pre-#172
            // reading (a categorical column on this layer's own authored
            // channel says nothing about whether it fits the axis it lands
            // on here), and letting the later gate answer first is what keeps
            // its message reachable at all.
            let layer_exchanges_channels = layer_index != 0 && coord_flipped != chart_flipped;
            // An `independent_y` layer resolves its own y-slot (secondary-y-axis,
            // GH #52) and must not widen the primary/shared y domain — that would
            // couple the left axis to a scale it does not own. Its x stays shared,
            // so only the y channel skips. Byte-stable for all-shared charts: the
            // flag defaults false, so this never fires without an independent layer.
            // Names the axis SLOT (the layer owns a second y axis), not an
            // encoding channel, so it reads `channel` directly and is unaffected
            // by the coordinate-space normalization below.
            if channel == "y" && layer.independent_y {
                continue;
            }
            let mut drawn = layer.drawn_own_encoding(&spec.encoding);
            if coord_flipped {
                drawn.flip_positional();
            }
            let encoding = &drawn;
            let (value_enc, paired_enc) = match channel {
                "x" => (encoding.x.as_ref(), encoding.x2.as_ref()),
                "y" => (encoding.y.as_ref(), encoding.y2.as_ref()),
                _ => (None, None),
            };
            if !layer_union.admits(value_enc.and_then(|e| e.scale.as_ref())) {
                continue;
            }
            // A raster image draws over the extent its Raster batch records,
            // not over the source columns its encoding names.
            let url = layer.encoding.url.as_ref().or(spec.encoding.url.as_ref());
            let layer_fields: [Option<&str>; 2] =
                match crate::render::marks::image::raster_extent_columns(layer.mark, url, channel) {
                    Some([lo, hi]) => [Some(lo), Some(hi)],
                    None => [value_enc.map(|e| e.field.as_str()), paired_enc.map(|e| e.field.as_str())],
                };
            // Union every declared field for this layer on this channel, and
            // remember whether the VALUE slot (index 0) actually resolved to
            // data in some batch — `value_field_contributed` below. One pass
            // replaces what used to be a separate "union everything, then
            // separately ask `.is_some()`" step: a layer's field can be
            // absent from every batch (wrong `data_source`, not loaded yet),
            // and the bar zero-anchor rule below must not fire on the
            // strength of the field's declared NAME alone when it never
            // contributed a single value (spec-review finding 3).
            let source = LayerSource::of(layer);
            let mut value_field_contributed = false;
            for (i, lf) in layer_fields.into_iter().enumerate() {
                let Some(lf) = lf else { continue };
                let seeded = seed_found
                    .iter()
                    .find(|(s, _)| s.source == source && s.field == lf)
                    .map(|&(_, found)| found);
                let contributed = if let Some(found) = seeded {
                    found
                } else if layer_exchanges_channels {
                    // An exchanging layer's field is judged by
                    // `validate_layer_coord_exchange_is_servable` (see this
                    // loop's own doc above), not by this union: a failure to
                    // find or read it here belongs to that gate, not this one.
                    read(Some(source), lf).unwrap_or(false)
                } else {
                    match source.locate(lf, primary_batch, transform_outputs) {
                        // Absent from the batch this layer draws is the
                        // ordinary "not in this output" outcome, not an error
                        // — the same reading the ordinal sibling below gives
                        // a layer field it cannot find.
                        None => false,
                        Some(located) => {
                            // A present column must be numeric or temporal
                            // before it can widen a continuous shared axis —
                            // asked as a positive question before the read,
                            // mirroring the ordinal sibling's
                            // `ensure_category_keyable` gate later in this
                            // file. Every OTHER read failure
                            // propagates with `?`. This is the read guard for
                            // every column the union reads, on the auto path
                            // and under a caller's scale alike; the spec-level
                            // `require_layers_placeable_on_caller_scale` judges
                            // each layer's value column only, so a layer's
                            // `x2`/`y2` endpoint is judged here alone.
                            crate::render::arrow_cast::require_placeable_on_continuous_axis(
                                layer_index,
                                channel,
                                lf,
                                located.col.data_type(),
                            )?;
                            read(Some(source), lf)?
                        }
                    }
                };
                if i == 0 {
                    value_field_contributed = contributed;
                }
            }
            // The authored `y` decides the bar's orientation: categorical by
            // its declared type or by the column it reads in this layer's batch.
            let y_categorical = layer.encoding.y.as_ref().is_some_and(|y| {
                column_reads_categorical(y, source.locate(&y.field, primary_batch, transform_outputs))
            });
            if matches!(layer_union, LayerUnion::Every)
                && layer_wants_bar_zero_anchor(
                    layer,
                    channel,
                    bar_value_on_x(coord_flipped, y_categorical),
                    layer_fields,
                    value_field_contributed,
                )
            {
                bar_zero_wanted = true;
            }
        }
    }

    if !mn.is_finite() || !mx.is_finite() {
        // All values were null/NaN — return a default domain so the chart
        // renders with axes but no marks, instead of raising an error.
        return Ok((0.0, 1.0));
    }
    if matches!(layer_union, LayerUnion::CarryingScale(_)) {
        return Ok((mn, mx));
    }
    if bar_zero_wanted {
        // The SAME widening `resolve_continuous_domain_and_range` applies for a
        // `ScaleSpec::Linear { zero: true }`, through the one helper both call
        // (`positional::extend_extent_to_zero`) rather than a second copy of its
        // two comparisons — this block used to restate them, and the restatement
        // was already drifting: it claimed to mirror a block that had itself
        // become a call to the helper. Applied here as well as there because an
        // auto-path spec reaches this function instead of that one (see this
        // function's doc comment).
        //
        // Runs BEFORE the degenerate-domain widening below so a single-row or
        // all-equal bar value (e.g. one category) resolves to a real `[0, v]`
        // span instead of the synthetic `±0.5` band that widening produces for
        // non-bar data — `single_row_bar_value_zero_extends_before_degenerate_widening`
        // pins that order, which the extraction does not change.
        (mn, mx) = crate::render::scale_resolve::extend_extent_to_zero(mn, mx);
    }
    // Degenerate domain: a single row or all-equal values produce mn == mx.
    // As of GH #104, `scale::core::degenerate_ratio` already makes every
    // continuous scale's `scale()` finite (a centered midpoint) for a
    // zero-span domain, so this expansion is no longer needed to keep marks
    // from vanishing — `to_pixel_f64` no longer returns NaN there. It stays
    // for a reason #104 doesn't touch: this is the *domain itself* (the
    // axis's displayed extent, tick generation, "nice" rounding), not the
    // per-value pixel projection. A literal `[d, d]` domain still collapses
    // `nice_ticks`/tick generation to a single degenerate tick and gives the
    // axis no visible span to lay out against; expanding to a symmetric band
    // around `mn == mx` gives the auto-inferred axis a real (if synthetic)
    // extent to tick and label, which #104's per-value guard has no way to
    // provide. This widening serves only the auto path (`LayerUnion::Every`):
    // a caller's scale, layered or not, never reaches it (`CarryingScale`
    // returns after the all-null fallback, and an
    // unlayered or stated-domain scale resolves in `build_from_scale_spec` →
    // `resolve_continuous_domain_and_range`), and relies on #104's per-scale
    // guard to keep a degenerate domain finite instead.
    if mn == mx {
        if mn == 0.0 {
            mn = -1.0;
            mx = 1.0;
        } else {
            mn -= 0.5;
            mx += 0.5;
        }
    }
    Ok((mn, mx))
}

/// Distinct positional (x:N / y:N) categories for an ordinal/band axis, unioning
/// the per-panel partition with the global all-panels batch when faceted+Shared.
///
/// - `panel_batch` is the located per-panel batch the non-shared path uses.
/// - When `include_final` is false this is byte-identical to
///   [`crate::render::arrow_cast::distinct_positional_categories`] on
///   `panel_batch` — non-faceted and `Independent`-mode callers see no change.
/// - When `include_final` is true (faceted, this channel resolves
///   `ResolveMode::Shared`) the domain is the union of the panel's categories
///   with the global set at [`FINAL_OUTPUT_KEY`], **in global first-appearance
///   order**. Deriving the order from the single global batch (a superset of
///   every panel) is what makes the categorical domain identical in every panel
///   — the shared-axis invariant. Per-panel-only categories (which cannot occur,
///   since the global batch is the concat of all panels) would still be honored
///   by the trailing union, never dropped.
///
/// Order semantics for the non-shared path are unchanged: it returns the
/// per-panel first-appearance order untouched.
///
/// `paired_positional` is the opposite positional channel's field, threaded
/// straight through to [`crate::render::arrow_cast::distinct_positional_categories`]'s
/// null-band occupancy rule (`#247`B) — see
/// `arrow_cast::null_category_band_is_occupied`. It is resolved against each
/// batch separately (panel and global), because absence in one says nothing about
/// the other and a batch that cannot answer keeps the band.
///
/// Private to this module: [`positional_category_union`] is the one entry the
/// scale paths call, so "the ordinal domain of this axis" has a single answer
/// that always includes the layer union (task 12c).
fn distinct_positional_categories_shared(
    panel_batch: &RecordBatch,
    field: &str,
    paired_positional: Option<&str>,
    transform_outputs: &HashMap<String, RecordBatch>,
    include_final: bool,
) -> Result<Vec<String>, RenderError> {
    let panel_domain = crate::render::arrow_cast::distinct_positional_categories(
        panel_batch,
        field,
        paired_positional,
    )?;
    if !include_final {
        return Ok(panel_domain);
    }
    let Some(global_batch) = transform_outputs.get(FINAL_OUTPUT_KEY) else {
        return Ok(panel_domain);
    };
    if global_batch.column_by_name(field).is_none() {
        return Ok(panel_domain);
    }
    // Global first-appearance order is the shared domain: a superset of every
    // panel, so it is identical across panels and preserves the global order.
    let mut domain = crate::render::arrow_cast::distinct_positional_categories(
        global_batch,
        field,
        paired_positional,
    )?;
    // Defensive union: append any panel category absent from the global set so a
    // category can never disappear from a panel's scale (no-op in practice).
    for cat in panel_domain {
        if !domain.contains(&cat) {
            domain.push(cat);
        }
    }
    Ok(domain)
}

/// The ordinal/nominal counterpart of [`numeric_domain_union`]: the category
/// ladder for an axis, covering the chart-level column **and every layer's own
/// column for this channel**.
///
/// The shared-axis invariant is "the resolved domain covers every layer's
/// extent". The numeric side has enforced it since long before this function
/// existed — [`numeric_domain_union`]'s `spec.layers` loop, whose own doc
/// explains why (a cross-DataFrame `a + b` merge renames the RHS columns, so the
/// chart-level field covers the LHS data ALONE). The ordinal side had no
/// equivalent, so the same shape lost the RHS layer's categories entirely: no
/// band existed for them, and every mark of that layer was silently dropped for
/// want of one. Measured before this function: `Chart({cat:[a,b,c]}) +
/// Chart({cat:[x,y,z]})`, both `mark_point().encode(x="cat:N", y="v:Q")`, drew
/// **3 circles for 6 rows** under x labels `['a','b','c']`.
///
/// **Order: first appearance, in layer order.** Walk the chart-level column,
/// then the layers in spec order, appending each previously-unseen category.
/// This is not a new semantic: [`crate::render::arrow_cast::distinct_values_in_order`]
/// already orders categories by first appearance *within* a column, and this
/// extends that same rule across the layer boundary rather than introducing a
/// sort. Nothing here sorts, and a user `sort=` still applies afterwards at the
/// caller ([`apply_sort_to_domain`]), to the unioned ladder.
///
/// **Each field from the batch that draws it** ([`LayerSource`]), the rule
/// [`numeric_domain_union`] follows: the seed is the bound field, read from
/// the batch of the layer establishing the axis and from the chart's data
/// when another layer on it inherits the field ([`seed_reads`]); each layer's own field is read
/// from its own `data_source` output, else the chart's data. Two `+`
/// operands binding the same field name on their own rows therefore both
/// contribute, and a merged frame's padding rows, which no layer draws, open
/// no band.
///
/// A layer contributes only when its OWN encoding binds this channel, to a
/// different field from the seed or from a batch the seed was not read from.
/// Reading the layer's own encoding rather than the inherited one (the reading
/// `numeric_domain_union`'s loop also takes) loses nothing: a layer that binds
/// nothing here inherits the chart-level field, which the seed already reads.
/// It is why a layered chart whose layers share their positional columns and
/// their batch resolves byte-identically to before.
///
/// …and only when that column can be a category **key**, which is the same
/// question as "can a mark address a band on it" and is asked through
/// [`crate::render::arrow_cast::ensure_category_keyable`] — see the gate in the
/// loop for why that makes the skip lossless, why the enumeration below still
/// propagates every other failure with `?`, and why this is deliberately not a
/// `Quantitative`/`Temporal` type test.
///
/// An unlayered spec reads its seed from its one batch ([`locate_field`]).
///
/// `include_final` and `paired_positional` mean what they mean for
/// [`distinct_positional_categories_shared`], and every contribution goes
/// through that same helper. A faceted+Shared axis takes the chart data's
/// categories from the global all-panels batch, keeping the ladder identical in
/// every panel rather than panel-local; a layer on its own named output reads
/// that output, which already spans every panel.
///
/// # A layer never opens the null band on an inherited coordinate
///
/// `#247`B decides the null band by occupancy: it exists iff some null-category
/// row carries a non-null value on the opposite positional channel, and where no
/// opposite channel is bound at all the band is kept (nothing can be missing, so
/// every null row draws). That reading is sound for the axis's OWN column and
/// unsound for a layer's, because a layer's opposite channel is frequently
/// *inherited* — its values are the chart-level frame's, and say nothing about
/// whether this layer places a mark. So a layer's contribution is judged against
/// the field the layer itself binds on the opposite channel, and when it binds
/// none, [`crate::render::arrow_cast::NULL_CATEGORY`] is dropped from what that
/// layer contributes: with no coordinate pair of its own, the layer has no
/// evidence to open a band with, and the axis's own column still decides the
/// band exactly as `#247`B left it.
///
/// This is not a hypothetical. `_inject_constant`'s reference-line column is one
/// authored value in a sea of structural nulls (`z = ["b", null, null, null]`),
/// bound on one positional channel with the other inherited. Judging it against
/// the inherited channel opens the null band and the reference-line layer draws
/// a line per null row instead of the one line the user asked for — pinned by
/// `scene_build`'s `coord_flipped_layered_rule_draws_its_own_channel_not_the_inherited_one`.
/// A cross-frame `a + b` merge, by contrast, binds BOTH channels on every layer,
/// so its genuine null-category-with-a-value row still opens the band and its
/// padding rows still do not — `#247`B applies to a layer's own column with full
/// force wherever the layer has the pair to answer with.
pub(in crate::render) fn positional_category_union(
    channel: &str,
    field: &str,
    paired_positional: Option<&str>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
) -> Result<Vec<String>, RenderError> {
    // The one precondition, layered or not: the locator the caller types this
    // axis with finds the bound field somewhere.
    let seed_batch = locate_seed_field(channel, field, spec, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?
        .batch;
    let Some(layers) = &spec.layers else {
        return distinct_positional_categories_shared(
            seed_batch,
            field,
            paired_positional,
            transform_outputs,
            include_final,
        );
    };
    // Membership set rather than `Vec::contains`: the ladder can run to hundreds
    // of categories and every layer would re-scan it.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut domain: Vec<String> = Vec::new();
    let mut join = |cats: Vec<String>| {
        for cat in cats {
            if seen.insert(cat.clone()) {
                domain.push(cat);
            }
        }
    };
    // The seed: the bound field from the establishing layer's batch, then each
    // field a later layer inherits from the chart's data ([`seed_reads`]), in
    // that order. Only the chart's data takes in the facet-shared all-panels
    // batch; a named output is already the all-panels one. An inherited field
    // is a later layer's column, so a column that cannot be a category key
    // contributes nothing, as it does from that layer's own binding below.
    let seeds = seed_reads(spec, layers, channel, (field, None, paired_positional));
    for seed in &seeds {
        let Some(located) = seed.source.locate(seed.field, primary_batch, transform_outputs) else { continue };
        if seed.inheritor.is_some()
            && crate::render::arrow_cast::ensure_category_keyable(seed.field, located.col.data_type()).is_err()
        {
            continue;
        }
        join(distinct_positional_categories_shared(
            located.batch,
            seed.field,
            seed.opposite,
            transform_outputs,
            include_final && seed.source == LayerSource::Chart,
        )?);
    }
    for layer in layers {
        // An `independent_y` layer owns its own y axis (GH #52) and must not
        // widen the primary y ladder, exactly as it does not widen the primary
        // numeric y extent. Its x stays shared, so only `"y"` skips.
        if channel == "y" && layer.independent_y {
            continue;
        }
        // COORDINATE SPACE — the same reconciliation `numeric_domain_union`'s
        // layer loop documents at length: `channel`/`field` are POST-flip
        // (derived from the rendering encoding), while `spec.layers` is the spec
        // as AUTHORED, so the layer encoding is moved into the channel's own
        // space before any read. One flipped clone answers both reads below,
        // which is what keeps "this layer's category column" and "this layer's
        // own opposite coordinate" in one space.
        //
        // WHOSE space is per layer, and the read is therefore INSIDE the loop
        // (`#172`(1), close review F1): the layer `prepare::build_layers` swaps
        // is the one whose OWN effective coord flips, so that is the layer whose
        // authored `y` occupies this x axis. This loop read the CHART's flip for
        // every layer until then — hoisted above the loop, where a per-layer
        // answer cannot be expressed — and a member carrying its own `CoordFlip`
        // therefore contributed its AUTHORED channel: measured, `bar(x="cat:N")
        // + point(x="v2:Q", y="cat2:N").coord(CoordFlip())` enumerated the
        // point layer's value column as three spurious bands (`['a','b','c',
        // '4','5','6']`) while the layer's real post-flip categories never
        // reached the ladder at all, so it drew 0 of its 3 circles. That is
        // precisely the defect this function exists to fix, reopened one level
        // down for the shape per-layer coord made expressible.
        let coord_flipped = crate::spec::coord::flips_positional_channels(
            layer.effective_coord(spec.coord.as_ref()),
        );
        let mut drawn = layer.drawn_own_encoding(&spec.encoding);
        if coord_flipped {
            drawn.flip_positional();
        }
        let encoding = &drawn;
        let (layer_enc, own_paired_enc) = match channel {
            "x" => (encoding.x.as_ref(), encoding.y.as_ref()),
            "y" => (encoding.y.as_ref(), encoding.x.as_ref()),
            _ => (None, None),
        };
        let Some(layer_enc) = layer_enc else { continue };
        let layer_field = layer_enc.field.as_str();
        let source = LayerSource::of(layer);
        // A layer reading the seed field from a batch the seed already read
        // adds nothing. A layer carrying its own continuous scale answers to
        // that scale ([`stays_off_categorical_axis`], the one predicate
        // [`warn_members_off_caller_scale`] reports it by).
        if seeds.iter().any(|s| s.source == source && s.field == layer_field)
            || stays_off_categorical_axis(layer_enc.scale.as_ref())
        {
            continue;
        }
        // Absent from the batch this layer draws is the normal "not in this
        // output" outcome, not an error — the same reading
        // `numeric_domain_union` gives a layer field it cannot find.
        let Some(located) = source.locate(layer_field, primary_batch, transform_outputs) else {
            continue;
        };
        // A column that cannot be a category KEY cannot be a category BAND
        // either, so it contributes nothing and that is lossless (close review
        // F2). The two halves of that sentence are one dtype set by
        // construction: [`crate::render::arrow_cast::ensure_category_keyable`]
        // admits exactly what `distinct_values_in_order` (this ladder) and
        // `col_as_ordinal_category_str` (the per-row key every mark builder
        // reads a band position through) accept — pinned by
        // `category_readers_accept_exactly_the_dtypes_ensure_category_keyable_does`.
        // So for a layer whose own column this rejects, no band the ladder could
        // have grown would ever have been addressed by a mark.
        //
        // Asked as a positive statement about what this axis can serve, BEFORE
        // the read, rather than by swallowing the read's error: the enumeration
        // below keeps its `?`, so a failure from any OTHER cause still refuses
        // the render instead of quietly shortening the ladder. That distinction
        // is the point — this skip is provably lossless, an `unwrap_or` would
        // also have hidden failures that are not.
        //
        // Without it the ordinal side REFUSED where its numeric sibling shrugs
        // (`numeric_domain_union`'s `union_field(lf).unwrap_or(false)`, "the
        // field may not exist in all batches"): measured,
        // `bar(x="cat:N") + line(x="when:T")` raised `column 'when' has
        // unsupported dtype: Timestamp(...)` for a composition that rendered
        // before this function existed. A temporal-x layer beside a
        // categorical-x layer is not malformed, so REFUSE MALFORMED does not
        // reach it; the layer simply places no mark on a band axis, exactly as
        // it did before, and the mark builders reach the same verdict about the
        // same column independently (`col_as_positional_category_str(..)
        // Err(_) => empty`).
        //
        // Deliberately NOT a `type_`/dtype INFERENCE gate
        // (`infer_spec_type(layer_enc, ..)`), which is a different question —
        // "which scale would this channel build" — already answered upstream for
        // this axis. Answering it again here would drop a QUANTITATIVE layer
        // column whose values the band ladder can and does serve: measured,
        // `bar(x="cat:N") + point(x="n:Q")` draws its 2 points on the two bands
        // `'10'`/`'20'` this loop adds for them, and an unannotated integer
        // column (`x="yr"` over `Int64`, which infers Quantitative) is the same
        // shape — both would silently lose every mark, which is the defect this
        // function exists to prevent.
        if crate::render::arrow_cast::ensure_category_keyable(layer_field, located.col.data_type())
            .is_err()
        {
            continue;
        }
        let own_paired = own_paired_enc.map(|e| e.field.as_str());
        let mut cats = distinct_positional_categories_shared(
            located.batch,
            layer_field,
            own_paired,
            transform_outputs,
            include_final && source == LayerSource::Chart,
        )?;
        if own_paired.is_none() {
            // The layer binds no opposite coordinate of its own, so it has no
            // evidence to open the null band with — see this function's doc. The
            // helper's own "unanswerable → keep the band" arm is right for the
            // axis's column and wrong for a layer's, and this is where the two
            // readings part.
            cats.retain(|c| c != crate::render::arrow_cast::NULL_CATEGORY);
        }
        join(cats);
    }
    Ok(domain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn group_batch(groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("group", DataType::Utf8, true)]));
        let arr: Vec<Option<&str>> = groups.into_iter().map(Some).collect();
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(arr))]).unwrap()
    }

    /// An unlayered spec's one mark draws its one batch: `locate_field` reads
    /// that batch alone, so a field it lacks is not found, whichever other
    /// batch holds a column of that name.
    #[test]
    fn locate_field_reads_only_the_unlayered_batch() {
        let primary = group_batch(vec!["A", "B"]);
        assert!(locate_field("value", &primary).is_none());
        assert!(locate_field("group", &primary).is_some_and(|located| std::ptr::eq(located.batch, &primary)));
    }

    // --- layer_wants_bar_zero_anchor: per-term unit tests --------------------
    //
    // The rule used to be a five-term boolean inlined in `numeric_domain_union`,
    // reachable only end-to-end through a whole spec + batch fixture (the
    // `*_SPEC_JSON` tests below still cover that integration). Extracted
    // (batch-F close review), each term gets a direct negative case here: flip
    // exactly one input away from the eligible baseline and the answer must
    // flip too. That is what makes "would Python's injection have anchored
    // this layer?" a checkable question rather than an emergent one.

    fn zero_anchor_layer(mark: Mark, zero: Option<bool>) -> Layer {
        Layer {
            mark,
            encoding: Default::default(),
            mark_style: zero.map(|z| crate::spec::mark_style::MarkKwargsSpec {
                zero: Some(z),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// The eligible baseline every negative case below perturbs: an unflipped
    /// `y` channel, a `Mark::Bar` layer with no `zero` opt-out, a value field
    /// that contributed data, and no paired span field.
    #[test]
    fn bar_zero_anchor_baseline_is_eligible() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(layer_wants_bar_zero_anchor(&layer, "y", false, [Some("v"), None], true));
    }

    /// Term (a), position half: the bar's POSITION channel is never anchored.
    /// This is the term whose absence squeezed `displot(kind="hist")` bars into
    /// a corner by zero-anchoring their `x` bin-range axis.
    #[test]
    fn bar_zero_anchor_refuses_position_channel() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(!layer_wants_bar_zero_anchor(&layer, "x", false, [Some("v"), None], true));
    }

    /// Term (a), flip half: under `CoordFlip` the value channel is `x`, not
    /// `y` — both readings move together, so the flipped `y` is now the
    /// position channel and is refused.
    #[test]
    fn bar_zero_anchor_follows_the_flip() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(layer_wants_bar_zero_anchor(&layer, "x", true, [Some("v"), None], true));
        assert!(!layer_wants_bar_zero_anchor(&layer, "y", true, [Some("v"), None], true));
    }

    /// Mark term: only `Mark::Bar` anchors. A point/line/rule layer on the
    /// same value channel leaves the union alone.
    #[test]
    fn bar_zero_anchor_refuses_non_bar_marks() {
        for mark in [Mark::Point, Mark::Line, Mark::Rule, Mark::Area] {
            let layer = zero_anchor_layer(mark, None);
            assert!(
                !layer_wants_bar_zero_anchor(&layer, "y", false, [Some("v"), None], true),
                "only Mark::Bar anchors, got an anchor for {mark:?}"
            );
        }
    }

    /// Term (c): the per-layer opt-out. `zero: Some(false)` refuses;
    /// `Some(true)` and an absent `mark_style` are both the default-on case.
    #[test]
    fn bar_zero_anchor_honors_the_per_layer_opt_out() {
        let opted_out = zero_anchor_layer(Mark::Bar, Some(false));
        assert!(!layer_wants_bar_zero_anchor(&opted_out, "y", false, [Some("v"), None], true));
        let opted_in = zero_anchor_layer(Mark::Bar, Some(true));
        assert!(layer_wants_bar_zero_anchor(&opted_in, "y", false, [Some("v"), None], true));
    }

    /// Term (b): a value channel carrying a paired span field (`y2`, or `x2`
    /// under flip) takes its extent literally from `[value, span]`.
    #[test]
    fn bar_zero_anchor_refuses_a_span_bound_value_channel() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(!layer_wants_bar_zero_anchor(
            &layer,
            "y",
            false,
            [Some("lo"), Some("hi")],
            true
        ));
    }

    /// The data-contribution term: a merely-DECLARED value field that never
    /// resolved in any batch (wrong `data_source`, not loaded yet) must not
    /// force an anchor on the strength of its name alone.
    #[test]
    fn bar_zero_anchor_refuses_a_field_that_contributed_no_data() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(!layer_wants_bar_zero_anchor(&layer, "y", false, [Some("v"), None], false));
    }

    /// A malformed sort-field object reaching `apply_sort_to_domain`
    /// directly (a caller that skipped `prepare`'s refusal walk) refuses with
    /// the typed error rather than leaving the domain silently unsorted.
    #[test]
    fn apply_sort_to_domain_malformed_object_refuses() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "group",
            batch: &batch,
            x_field: None,
            y_field: None,
            coord_flipped: false,
        };
        let mut domain = vec!["a".to_string(), "b".to_string()];
        let mut warnings = Vec::new();
        let err = apply_sort_to_domain(
            &mut domain,
            Some(&serde_json::json!({"op": "mean"})),
            &ctx,
            &mut warnings,
        )
        .expect_err("a sort object without a field must refuse");
        assert_eq!(
            err.to_string(),
            r#"group: sort={"op":"mean"} is malformed (sort object missing string "field")"#
        );
        assert!(warnings.is_empty(), "{warnings:?}");
    }


    /// A well-formed sort-field object
    /// naming a field the batch does not carry still warns exactly as
    /// before -- only the MALFORMED-SHAPE arm was deleted.
    #[test]
    fn apply_sort_to_domain_unresolvable_well_formed_object_still_warns() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "group",
            batch: &batch,
            x_field: None,
            y_field: None,
            coord_flipped: false,
        };
        let mut domain = vec!["a".to_string(), "b".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(
            &mut domain,
            Some(&serde_json::json!({"field": "does_not_exist"})),
            &ctx,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(warnings.len(), 1, "an unresolvable well-formed object still warns: {warnings:?}");
        assert!(matches!(&warnings[0], RenderWarning::SortSpecIgnored { .. }));
    }

    // --- R3: SortSpecIgnored (channel-shorthand family) ---------------------

    /// R3, unflipped shape: `SortContext.x_field`/`y_field` reflect an ordinary
    /// (unflipped) chart — category on x, y unbound. `sort="-y"` references the
    /// unbound channel, so the warning fires; its message must echo exactly the
    /// literal shorthand text the user wrote.
    #[test]
    fn sort_shorthand_missing_channel_names_users_own_text_unflipped() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: Some("price"),
            y_field: None,
            coord_flipped: false,
        };
        let mut domain = vec!["a".to_string(), "b".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("-y")), &ctx, &mut warnings).unwrap();
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            RenderWarning::SortSpecIgnored { reason } => {
                assert_eq!(reason, "sort=\"-y\" but no field is bound to channel \"y\"");
            }
            other => panic!("expected SortSpecIgnored, got {other:?}"),
        }
        assert_eq!(domain, vec!["a".to_string(), "b".to_string()], "insertion order preserved on failure");
    }

    /// R3, post-`CoordFlip` shape: `SortContext.x_field`/`y_field` are
    /// populated from the RESOLVED (post-flip) rendering encoding on the
    /// prepare path (`positional.rs`'s `PositionalFields`), so the SAME
    /// "the user's own y channel is unbound" situation the unflipped case
    /// above expresses as `y_field: None` is expressed HERE as `x_field:
    /// None` — post-flip x is where the authored y lands. The message is then
    /// asserted byte-identical to the unflipped one: `channel` is parsed
    /// straight out of the `shorthand` STRING the user wrote, never derived
    /// from which resolved slot was consulted.
    ///
    /// Re-reasoned for `#95` (not re-baselined): the fields were previously
    /// `x_field: Some("weight"), y_field: None`, which under AUTHORED
    /// semantics no longer describes an unbound authored `y` at all — it
    /// describes an authored `y` bound to `weight`, which now RESOLVES.
    /// Keeping the old fixture would have made this test pass on a different
    /// message ("cannot read sort value field") and quietly stopped pinning
    /// R3's actual claim. The asserted string is unchanged, because R3's
    /// claim is unchanged.
    #[test]
    fn sort_shorthand_missing_channel_names_users_own_text_under_flip() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: None,            // post-flip x holds the user's y — unbound
            y_field: Some("species"), // post-flip y holds the user's x (the category)
            coord_flipped: true,
        };
        let mut domain = vec!["a".to_string(), "b".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("-y")), &ctx, &mut warnings).unwrap();
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            RenderWarning::SortSpecIgnored { reason } => {
                assert_eq!(
                    reason, "sort=\"-y\" but no field is bound to channel \"y\"",
                    "must echo the user's own written shorthand — never a resolved/internal token"
                );
            }
            other => panic!("expected SortSpecIgnored, got {other:?}"),
        }
        assert_eq!(
            domain,
            vec!["a".to_string(), "b".to_string()],
            "insertion order preserved on failure"
        );
    }

    /// `#95`, at the unit seam: the authored channel token is resolved against
    /// the field the user's OWN channel bound, which under `CoordFlip` is the
    /// field sitting in the opposite `SortContext` slot.
    ///
    /// Both rows aggregate the same column (`val`) and must therefore produce
    /// the same order; the only difference between them is which slot that
    /// column occupies and whether `coord_flipped` is set. A broken
    /// implementation reads the like-named slot instead: unflipped that is
    /// still `val` (so the control passes either way — it is the control), but
    /// flipped it is the category column, which yields no usable aggregate and
    /// leaves insertion order plus a warning.
    #[test]
    fn sort_shorthand_resolves_the_authored_channel_not_the_rendered_one() {
        use arrow::array::Float64Array;
        use arrow::datatypes::Fields;
        let schema = Arc::new(Schema::new(Fields::from(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("val", DataType::Float64, true),
        ])));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
                Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0])),
            ],
        )
        .unwrap();

        // Unflipped: the user authored the value on y, and y is where it renders.
        let unflipped = SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: Some("cat"),
            y_field: Some("val"),
            coord_flipped: false,
        };
        let mut domain = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("-y")), &unflipped, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "control must resolve: {warnings:?}");
        assert_eq!(domain, ["a", "c", "b"], "descending by val");

        // Flipped: the same authored encoding, so the value now renders on x.
        let flipped = SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: Some("val"),
            y_field: Some("cat"),
            coord_flipped: true,
        };
        let mut flipped_domain = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut flipped_warnings = Vec::new();
        apply_sort_to_domain(
            &mut flipped_domain,
            Some(&serde_json::json!("-y")),
            &flipped,
            &mut flipped_warnings,
        )
        .unwrap();
        assert!(
            flipped_warnings.is_empty(),
            "the authored spelling must resolve under flip too: {flipped_warnings:?}"
        );
        assert_eq!(
            flipped_domain, domain,
            "one authored spelling, one order, whatever the coordinate system"
        );
    }

    // ── Task 13 (spec §4.4, "Extended 2026-09-02, T13 quality review",
    // ruling 1): the layer union reads each layer's fields in the coordinate
    // space of the channel being resolved ─────────────────────────────────────
    //
    // Two properties, pinned together because one is the other's control:
    //   1. a layer's `x2`/`y2` past its bare `x`/`y` widens THAT channel's
    //      domain (the batch-A T13 union fix — `mark_qq(line=True)`'s reference
    //      diagonal was unreachable without it), and
    //   2. under `CoordFlip` it widens the OTHER axis, because `spec.layers`
    //      is the AUTHORED encoding while `channel`/`field` name the rendered
    //      (post-flip) axis.
    //
    // Both `*_SPEC_JSON`s are the VERBATIM `chart.to_dict()` of the public-API
    // call in their doc comment (captured 2026-09-02), parsed through the same
    // serde path `ChartSpec.from_json` uses, so they cannot drift from what
    // Python lowers.

    /// `fm.Chart(df).mark_point().encode(x="b", y="a")` layered (`+`) with
    /// `fm.Chart(df).mark_rule().encode(y="a", y2="big")`.
    const LAYERED_Y2_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "rule", "encoding": {"y": {"field": "a"}, "y2": {"field": "big"}}}]}"#;

    /// The same layering with `.coord(fm.CoordFlip())` on the primary chart —
    /// the reviewer's A/B probe. `prepare::build_layers` swaps the RENDERING
    /// encodings, so the resolved x axis holds `a` and the y axis holds `b`,
    /// while `layers` below still reads exactly as authored.
    const LAYERED_Y2_FLIPPED_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "rule", "encoding": {"y": {"field": "a"}, "y2": {"field": "big"}}}], "coord": {"kind": "flip"}}"#;

    /// `pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "big": [1000.0, 2000.0]})`
    /// — three disjoint numeric ranges, so which field reached the union is
    /// readable straight off the returned extent.
    fn abc_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("big", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
                Arc::new(Float64Array::from(vec![1000.0, 2000.0])),
            ],
        )
        .unwrap()
    }

    /// Resolve one axis's auto-inferred extent for a lowered spec, exactly as
    /// `build_axis_scale` does (no named transform outputs, no facet union).
    fn axis_extent(spec_json: &str, channel: &str, field: &str) -> (f64, f64) {
        let spec: ChartSpec =
            serde_json::from_str(spec_json).expect("lowered spec JSON must deserialize");
        let batch = abc_batch();
        numeric_domain_union(channel, field, None, &batch, &HashMap::new(), &spec, false)
            .expect("every field in the fixture is Float64")
    }

    /// Unflipped: the rule layer's `y2` (1000..2000) extends past its bare `y`
    /// (`a`, 1..2), so the y domain must reach 2000 — otherwise the layer's own
    /// far endpoint projects to `None` and the reference line silently draws
    /// nothing. The x axis is the control: `big` belongs to the y channel, so
    /// it must not leak there.
    #[test]
    fn layer_paired_endpoint_widens_its_own_channel_only() {
        assert_eq!(
            axis_extent(LAYERED_Y2_SPEC_JSON, "y", "a"),
            (1.0, 2000.0),
            "the layer's y2 must widen the y domain past its bare y"
        );
        assert_eq!(
            axis_extent(LAYERED_Y2_SPEC_JSON, "x", "b"),
            (3.0, 4.0),
            "a y-channel field must never widen the x domain"
        );
    }

    /// Flipped: the SAME authored `y2` now lives on the rendered x axis, so it
    /// must widen x and leave y at `b`'s own extent — the exact mirror of the
    /// unflipped case above.
    ///
    /// RED (verified in place): read `layer.encoding` instead of its
    /// flip-normalized copy and this fails at the first assertion with
    /// `(1.0, 4.0)` — x picks up the sibling layer's authored `x` (`b`) and
    /// still never reaches the rule's own far endpoint on the axis that
    /// endpoint actually occupies, while y (unasserted once x fails, `(1.0,
    /// 2000.0)` when checked alone) blows out to 2000 on an axis no mark uses
    /// there. That pair is the reviewer's live A/B probe.
    #[test]
    fn coord_flipped_layer_paired_endpoint_widens_the_other_axis() {
        assert_eq!(
            axis_extent(LAYERED_Y2_FLIPPED_SPEC_JSON, "x", "a"),
            (1.0, 2000.0),
            "under CoordFlip the authored y2 occupies the x axis and must widen it"
        );
        assert_eq!(
            axis_extent(LAYERED_Y2_FLIPPED_SPEC_JSON, "y", "b"),
            (3.0, 4.0),
            "under CoordFlip the y axis holds the authored x field; an authored \
             y/y2 must not widen it"
        );
    }

    // ── #120(9) + N1: a layer column that cannot serve the shared numeric
    // axis is refused ─────────────────────────────────────────────────────
    //
    // `numeric_domain_union_over`'s per-layer loop reads every OTHER layer's
    // own field for this channel (`lf != field`) through `union_field`. The
    // dtype gate below asks, before that read, whether the located column can
    // serve a continuous axis at all — the numeric counterpart of the ordinal
    // union's `ensure_category_keyable` gate further down this file.

    /// `fm.layer(point(y="a"), line(y="s"))`: `s` is `Utf8`. Reaches this
    /// function exactly as `scale_resolve::positional`'s merged-flat prepare
    /// path and `composite::leaf_positional_domain`'s static overlay path
    /// both call it.
    const LAYER_UTF8_Y_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "line", "encoding": {"y": {"field": "s"}}}]}"#;

    /// The same shape with the second layer's `y` naming a field present in
    /// no batch at all — the ordinary "not loaded here" outcome, unaffected
    /// by the gate above.
    const LAYER_ABSENT_Y_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "line", "encoding": {"y": {"field": "missing"}}}]}"#;

    /// The same shape with the second layer's `y` naming `t`, a `Timestamp`
    /// column — the positive control proving the gate admits temporal
    /// columns rather than only numeric ones.
    const LAYER_TEMPORAL_Y_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "line", "encoding": {"y": {"field": "t"}}}]}"#;

    /// `a`/`b` numeric, `s` a string column disjoint in name from either.
    fn ab_s_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
                Arc::new(StringArray::from(vec!["p", "q"])),
            ],
        )
        .unwrap()
    }

    /// `a`/`b` numeric, `t` a millisecond `Timestamp` column well past `a`'s
    /// own extent, so a widened union is only explainable by `t` joining it.
    fn ab_t_batch() -> RecordBatch {
        use arrow::array::{Float64Array, TimestampMillisecondArray};
        use arrow::datatypes::TimeUnit;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("t", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
                Arc::new(TimestampMillisecondArray::from(vec![500, 9000])),
            ],
        )
        .unwrap()
    }

    /// A non-numeric, non-temporal layer column refuses rather than joining
    /// the union at arbitrary coerced heights.
    #[test]
    fn layer_utf8_column_on_shared_axis_refuses() {
        let spec: ChartSpec =
            serde_json::from_str(LAYER_UTF8_Y_SPEC_JSON).expect("spec JSON must deserialize");
        let batch = ab_s_batch();
        let err = numeric_domain_union("y", "a", None, &batch, &HashMap::new(), &spec, false)
            .expect_err("a Utf8 layer column must refuse the shared numeric axis");
        assert_eq!(
            err,
            RenderError::LayerAxisColumnUnplaceable {
                layer: 1,
                detail: Box::new(crate::render::LayerAxisColumnDetail {
                    channel: "y".to_string(),
                    field: "s".to_string(),
                    dtype: "Utf8".to_string(),
                }),
            }
        );
        assert_eq!(
            err.to_string(),
            "layer 1: column 's' (Utf8) cannot be drawn on the shared y axis, which is \
             continuous — a layered chart (fm.layer(...) / chart + chart) merges its members \
             onto one panel. Cast the column, or give the layer its own scale with \
             resolve={\"y\": \"independent\"}."
        );
    }

    /// Control: a layer field present in no batch at all contributes
    /// nothing and does not refuse — the same reading the ordinal union
    /// gives the identical case (`a_layer_field_in_no_batch_is_skipped_not_raised`
    /// below).
    #[test]
    fn layer_field_absent_from_every_batch_is_skipped_not_refused() {
        let spec: ChartSpec =
            serde_json::from_str(LAYER_ABSENT_Y_SPEC_JSON).expect("spec JSON must deserialize");
        let batch = ab_s_batch();
        let extent = numeric_domain_union("y", "a", None, &batch, &HashMap::new(), &spec, false)
            .expect("an absent layer field must not refuse");
        assert_eq!(extent, (1.0, 2.0), "the absent field must not widen or shrink the union");
    }

    /// Control: a layer's own `Timestamp` column still joins the union —
    /// the gate admits numeric OR temporal, not numeric alone.
    #[test]
    fn layer_temporal_column_still_widens_the_shared_axis() {
        let spec: ChartSpec = serde_json::from_str(LAYER_TEMPORAL_Y_SPEC_JSON)
            .expect("spec JSON must deserialize");
        let batch = ab_t_batch();
        let extent = numeric_domain_union("y", "a", None, &batch, &HashMap::new(), &spec, false)
            .expect("a Timestamp layer column must be admitted, not refused");
        assert_eq!(
            extent,
            (1.0, 9000.0),
            "the layer's Timestamp column must widen the union past a's own extent"
        );
    }

    /// A layer field can be located as numeric in `primary_batch` (passing
    /// the gate) yet still fail downstream when the SAME field also lives,
    /// with an unsupported dtype, in the facet-shared global batch
    /// (`include_final`). That failure is a genuinely different
    /// `union_field` outcome from the gate's own refusal and must still
    /// reach the caller via `?`, never be swallowed back to `Ok(false)`.
    #[test]
    fn a_non_gate_union_field_error_still_propagates() {
        use arrow::array::Float64Array;
        let spec_json = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "line", "encoding": {"y": {"field": "z"}}}]}"#;
        let spec: ChartSpec = serde_json::from_str(spec_json).expect("spec JSON must deserialize");
        let primary_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ]));
        let primary = RecordBatch::try_new(
            primary_schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
                Arc::new(Float64Array::from(vec![5.0, 6.0])),
            ],
        )
        .unwrap();
        let final_schema = Arc::new(Schema::new(vec![Field::new("z", DataType::Utf8, false)]));
        let final_batch = RecordBatch::try_new(
            final_schema,
            vec![Arc::new(StringArray::from(vec!["x", "y"]))],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), final_batch);
        let err = numeric_domain_union("y", "a", None, &primary, &outputs, &spec, true)
            .expect_err("z's Utf8 copy in the global batch must still refuse the render");
        assert_eq!(
            err,
            RenderError::UnsupportedDtype {
                field: "z".to_string(),
                dtype: format!("{:?}", DataType::Utf8),
                context: None,
            },
            "the propagated error must be union_field's own UnsupportedDtype, not the gate's \
             LayerAxisColumnUnplaceable — the gate saw only z's Float64 copy in primary_batch"
        );
    }

    // ── T5r: bar zero-anchor on the layered/auto-inferred path ──────────────
    //
    // `catplot(kind="bar")` renders a `bar` layer (one mean per category) plus
    // a `rule` "whisker" layer (the bootstrap CI), both reading a named
    // `Summary` transform output via `data_source`. Before this fix the
    // auto-inferred cross-layer union (this function) had no zero rule at
    // all, so the value axis stopped at the union of `mean`/`lower`/`upper`
    // and bars looked truncated at the panel bottom. An explicit per-layer
    // `scale={"zero": True}` (the mechanism a plain single-chart `mark_bar`
    // uses, injected by `_spec_build.py`) was measured and rejected: it
    // routes through `build_from_scale_spec`, which resolves the domain from
    // ONE column, so it silently clipped the whisker layer's own extent (3 of
    // 4 upper CI whiskers vanished off-panel) — see commit 32117b00 and
    // this function's own doc comment
    // ("Bar zero-anchor") above.

    /// Verbatim `chart.to_dict()` JSON for `fm.catplot(df, x="cat", y="val",
    /// kind="bar")` (captured 2026-09-10, T5r, with `df` mapping `cat` to
    /// two groups and `val` to values whose bootstrap mean/CI land around
    /// 50-90). Two layers: `bar` on the transform's `mean` output, `whisker`
    /// (a `rule`) on `lower`/`upper`, both via `data_source: "summary"`.
    const CATPLOT_BAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}]}"#;

    /// The same call with `orient="h"` (captured 2026-09-10, T5r). The
    /// authored encoding is untouched (the value field is still authored on
    /// `y`); only `coord` gains `{"kind": "flip"}`, so the RENDERED x axis is
    /// where the bar's value lives and where the zero-anchor must land.
    const CATPLOT_BAR_FLIPPED_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}],"coord":{"kind":"flip"}}"#;

    /// `CATPLOT_BAR_SPEC_JSON` with the `bar` layer's mark swapped to
    /// `"point"` — the ONLY difference from the spec above, isolating mark
    /// type as the sole variable. Proves the anchor is gated on `Mark::Bar`
    /// specifically, not merely on "a layered spec with a `rule` sibling".
    const CATPLOT_NONBAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}]}"#;

    /// A non-layered `mark_bar` spec — `spec.layers` is absent entirely. This
    /// case never reaches `numeric_domain_union` in real rendering (Python's
    /// `_spec_build.py::_build_encoding_specs` already injects an explicit
    /// `scale={"zero": True}` onto a single-chart bar's y encoding, which
    /// resolves through `build_from_scale_spec` instead), but the function
    /// itself must not ALSO add a zero rule keyed on `spec.mark` — only on
    /// `spec.layers` — so this pins the auto path stays untouched if it were
    /// ever reached directly.
    const SINGLE_BAR_SPEC_JSON: &str =
        r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"val"}}}"#;

    /// The shape a real `Summary` transform emits: `mean`, `lower`, `upper`,
    /// all positive and well clear of zero (50..90), so a resolved domain
    /// that reaches zero can only be explained by the bar zero-anchor, never
    /// by the data itself.
    fn catplot_summary_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("mean", DataType::Float64, false),
            Field::new("lower", DataType::Float64, false),
            Field::new("upper", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![50.0, 80.0])),
                Arc::new(Float64Array::from(vec![40.0, 70.0])),
                Arc::new(Float64Array::from(vec![60.0, 90.0])),
            ],
        )
        .unwrap()
    }

    /// Resolve one axis's auto-inferred extent for a catplot-shaped lowered
    /// spec: the primary batch carries an unrelated `group` column (mirroring
    /// the real primary batch, which the Summary transform's `mean`/`lower`/
    /// `upper` output columns are absent from), and those three fields live
    /// in the named `"summary"` transform output every layer's `data_source`
    /// references — the same wiring `catplot`'s lowered spec carries.
    fn catplot_axis_extent(spec_json: &str, channel: &str, field: &str) -> (f64, f64) {
        let spec: ChartSpec =
            serde_json::from_str(spec_json).expect("lowered spec JSON must deserialize");
        let primary = group_batch(vec!["a", "b"]);
        let mut outputs = HashMap::new();
        outputs.insert("summary".to_string(), catplot_summary_batch());
        numeric_domain_union(channel, field, None, &primary, &outputs, &spec, false)
            .expect("every field in the fixture is Float64")
    }

    /// Before this fix, the union of `mean` (50..80), `lower` (40..70), and
    /// `upper` (60..90) is `(40.0, 90.0)` — the domain the coder's report
    /// found truncated relative to zero. The `bar` layer's own field
    /// (`mean`) participates in the y-channel union, so the zero-anchor
    /// widens the lower bound to 0, while the whisker's own `upper` extent
    /// (90) stays reachable — the exact property the report's rejected
    /// explicit-`ScaleSpec` fix could not preserve.
    ///
    /// RED (verified in place): commenting out the `if bar_zero_wanted`
    /// block in `numeric_domain_union` makes this fail with `(40.0, 90.0)`.
    #[test]
    fn layered_bar_and_whisker_value_domain_includes_zero() {
        assert_eq!(
            catplot_axis_extent(CATPLOT_BAR_SPEC_JSON, "y", "mean"),
            (0.0, 90.0),
            "a bar-mark layer on the y channel must widen the domain to include \
             zero while keeping the whisker's own extent (90) reachable"
        );
    }

    /// Control: the identical layer shape with the bar layer's mark swapped
    /// to `"point"` must NOT gain the zero-anchor — the union is exactly what
    /// `layered_bar_and_whisker_value_domain_includes_zero` computed BEFORE
    /// the zero-anchor widened it, `(40.0, 90.0)`.
    #[test]
    fn nonbar_layered_spec_value_domain_unchanged() {
        assert_eq!(
            catplot_axis_extent(CATPLOT_NONBAR_SPEC_JSON, "y", "mean"),
            (40.0, 90.0),
            "a layered spec with no bar-mark layer must not gain a zero-anchor"
        );
    }

    /// CoordFlip twin: the authored encoding is unchanged (the bar's value
    /// field is still authored on `y`), but under `.coord(CoordFlip())` that
    /// value is RENDERED on the x axis, so the zero-anchor must land there —
    /// same union arithmetic as the unflipped case (`(40.0, 90.0)` before the
    /// anchor), mirroring `coord_flipped_layer_paired_endpoint_widens_the_other_axis`
    /// above.
    ///
    /// RED (verified in place): commenting out the `if bar_zero_wanted`
    /// block makes this fail with `(40.0, 90.0)` too — proving the anchor
    /// follows the coordinate-space normalization, not just the unflipped
    /// case. Also genuinely discriminating on the flip normalization itself
    /// (rust-quality-review finding, re-measured T5r cycle 2): dropping
    /// `flip_positional` from the detection (reading `layer.encoding`
    /// directly) makes this fail too, with `(50.0, 80.0)` — the bar's `cat`
    /// field is absent from every batch in this fixture, so the mutant's
    /// `value_field_contributed` check for the (wrongly-read) unflipped
    /// value slot fails and the anchor never fires. See
    /// `coord_flipped_bar_position_channel_never_zero_extends` below for the
    /// companion fixture that discriminates the same mutant with real data
    /// on both channels instead of an absent field.
    #[test]
    fn coord_flipped_bar_widens_the_rendered_x_axis() {
        assert_eq!(
            catplot_axis_extent(CATPLOT_BAR_FLIPPED_SPEC_JSON, "x", "mean"),
            (0.0, 90.0),
            "under CoordFlip the bar's authored y (the value field) occupies \
             the rendered x axis and must carry the zero-anchor there"
        );
    }

    /// Single-chart (non-layered) `mark_bar`: this function must not add a
    /// zero-anchor on its own when `spec.layers` is absent — pinned to catch
    /// a future widening of the bar-zero rule from `spec.layers` to
    /// `spec.mark`, which would silently diverge from the explicit-scale
    /// path's `nice`/padding treatment that single-chart bars actually get
    /// in real rendering.
    #[test]
    fn single_chart_bar_auto_path_unchanged() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![50.0, 80.0]))])
                .unwrap()
        };
        let extent = numeric_domain_union("y", "val", None, &batch, &HashMap::new(), &spec, false)
            .expect("val is Float64");
        assert_eq!(
            extent,
            (50.0, 80.0),
            "a non-layered mark_bar spec must not gain a zero-anchor from this \
             function; Python's explicit scale={{\"zero\": true}} injection owns \
             that case"
        );
    }

    // ── T5r cycle 2 (spec-review issue 1): the two dropped guards ───────────
    //
    // The cycle-1 rule fired for whatever channel was being resolved, as long
    // as a `Mark::Bar` layer bound it — no distinction between the bar's
    // VALUE axis and its POSITION axis, and no exclusion for a layer whose
    // own value channel already carries a paired span field. Both guards
    // mirror `_spec_build.py::_build_encoding_specs`'s single-chart
    // injection exactly (`axis == "y"`, `not _y2_bound`). Missing them let a
    // `displot(kind="hist")` histogram bar's `x`/`x2` bin-range POSITION
    // widen to zero — a live regression in this same batch's T4 feature,
    // caught by spec review, not by the cycle-1 test suite (none of the four
    // cycle-1 tests exercised a bar layer's position channel or a `y2`-bound
    // bar).

    /// Verbatim `chart.to_dict()` JSON for `fm.displot(df, x="val",
    /// kind="hist", kde=True)` (captured 2026-09-10, T5r cycle 2) — the exact
    /// shape the spec reviewer measured live: a `bar` histogram layer whose
    /// POSITION channel is `x`/`x2` (the bin range) and whose VALUE channel
    /// is `y` (`count`). The sibling `kde` line layer reads a named
    /// `"kde_overlay"` output this test's fixture omits — its own field,
    /// `value`, unions to nothing and is silently skipped (`union_field`'s
    /// documented "ignore errors" behavior), so omitting it doesn't affect
    /// the assertion below.
    const DISPLOT_HIST_KDE_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"transforms":[{"type":"kde","field":"val","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":512,"cumulative":false,"shared_extent":false,"kernel":"gaussian","count_scale":{"kind":"sturges"},"name":"kde_overlay"},{"type":"bin","field":"val","nice":true,"cumulative":false,"shared_extent":false}],"layers":[{"mark":"bar","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"position":{"type":"identity"},"name":"histogram"},{"mark":"line","encoding":{"x":{"field":"value"},"y":{"field":"density"}},"data_source":"kde_overlay","position":{"type":"identity"},"name":"kde"}],"position":{"type":"identity"}}"#;

    /// Bin edges offset well clear of zero (50..89), matching the spec
    /// reviewer's reproduction data — a domain that reaches zero can only be
    /// explained by the (wrong) position-channel anchor, never by the data.
    fn displot_hist_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("bin_end", DataType::Float64, false),
            Field::new("count", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![50.0, 70.0])),
                Arc::new(Float64Array::from(vec![70.0, 89.0])),
                Arc::new(Float64Array::from(vec![4.0, 5.0])),
            ],
        )
        .unwrap()
    }

    /// Before the value-channel guard (cycle 1), this incorrectly returned
    /// `(0.0, 89.0)` — the bar layer's POSITION channel (`x`/`x2`, the bin
    /// range) got the anchor only its VALUE channel (`y`, `count`) is
    /// supposed to get. Measured live by the spec reviewer:
    /// `displot(x="val", kind="hist", kde=True)` with data in `[50, 89]`
    /// rendered its x axis `0…90`, squeezing every bar into the right ~44%
    /// of the panel.
    ///
    /// This fixture's position channel also carries `x2` (`bin_end`), so
    /// guard (b) (span-suppression) independently protects it too —
    /// dropping guard (a) alone does NOT turn this test red (verified: the
    /// span guard still catches it). `bar_layer_own_position_channel_without_span_never_zero_extends`
    /// below isolates guard (a) with a position channel that carries no
    /// span field at all, so only guard (a) protects it.
    #[test]
    fn bar_position_channel_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(DISPLOT_HIST_KDE_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = displot_hist_batch();
        let extent = numeric_domain_union(
            "x",
            "bin_start",
            Some("bin_end"),
            &batch,
            &HashMap::new(),
            &spec,
            false,
        )
        .expect("bin_start/bin_end are Float64");
        assert_eq!(
            extent,
            (50.0, 89.0),
            "a bar layer's POSITION channel (the bin range) must never gain \
             the value-channel zero-anchor"
        );
    }

    /// Verbatim `chart.to_dict()` JSON for
    /// `fm.Chart(df).mark_bar().encode(x="cat", y="lo", y2="hi") +
    /// fm.Chart(df).mark_point().encode(x="cat", y="lo")` (captured
    /// 2026-09-10, T5r cycle 2) — a bar layer whose own VALUE channel
    /// carries a paired span field, reachable from the public `Chart` API
    /// (`.encode(y2=...)` is a generically honored channel on any mark).
    /// Mirrors the D3 rule (`_spec_build.py::_build_encoding_specs`'s
    /// `_y2_bound`/`_zero_anchor_wanted`) already applies to the
    /// single-chart case: an explicit `[y, y2]` extent is taken literally
    /// and must not be force-anchored to zero.
    const Y2_BOUND_BAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"lo"},"y2":{"field":"hi"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"lo"},"y2":{"field":"hi"}}},{"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"lo"}}}]}"#;

    fn y2_bound_bar_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![50.0, 60.0])),
                Arc::new(Float64Array::from(vec![80.0, 90.0])),
            ],
        )
        .unwrap()
    }

    /// The D3 rule: a bar layer that binds its OWN value channel's paired
    /// span field (`y2` here) must not be force-anchored to zero — the
    /// extent stays taken literally from `[lo, hi]` (50..90), exactly the
    /// pre-anchor union.
    ///
    /// RED (verified in place): removing the `layer_fields[1].is_none()`
    /// guard makes this fail with `(0.0, 90.0)`.
    #[test]
    fn y2_bound_bar_layer_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(Y2_BOUND_BAR_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = y2_bound_bar_batch();
        let extent =
            numeric_domain_union("y", "lo", Some("hi"), &batch, &HashMap::new(), &spec, false)
                .expect("lo/hi are Float64");
        assert_eq!(
            extent,
            (50.0, 90.0),
            "a bar layer that binds its own value channel's paired span \
             field must not be force-anchored to zero"
        );
    }

    // ── T5r cycle 2 (rust-quality-review): discriminating flip test, the
    // remaining two arithmetic arms, and the `zero=False` opt-out ──────────

    /// `DISPLOT_HIST_KDE_SPEC_JSON` with `.coord(fm.CoordFlip())` appended —
    /// same shape as `LAYERED_Y2_FLIPPED_SPEC_JSON` above ("the same
    /// layering with `.coord(fm.CoordFlip())`"), not itself reachable
    /// through `displot`'s own `orient=` (histograms have none), but a real,
    /// generically composable shape (`CoordFlip` applies to any chart).
    /// Under this flip the histogram's VALUE channel (`y`=`count`,
    /// authored) is RENDERED on `x`, and its POSITION channel (`x`/`x2` =
    /// the bin range, authored) is rendered on `y`/`y2`.
    const DISPLOT_HIST_KDE_FLIPPED_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"transforms":[{"type":"kde","field":"val","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":512,"cumulative":false,"shared_extent":false,"kernel":"gaussian","count_scale":{"kind":"sturges"},"name":"kde_overlay"},{"type":"bin","field":"val","nice":true,"cumulative":false,"shared_extent":false}],"layers":[{"mark":"bar","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"position":{"type":"identity"},"name":"histogram"},{"mark":"line","encoding":{"x":{"field":"value"},"y":{"field":"density"}},"data_source":"kde_overlay","position":{"type":"identity"},"name":"kde"}],"position":{"type":"identity"},"coord":{"kind":"flip"}}"#;

    /// Companion to `coord_flipped_bar_widens_the_rendered_x_axis` above,
    /// with a bar layer that carries REAL numeric data on BOTH channels
    /// (`count` and the `bin_start`/`bin_end` bin range) rather than relying
    /// on the sibling test's category field (`"cat"`) being absent from
    /// every fixture batch to make a coordinate-space mutant visible.
    ///
    /// RED (verified in place, measured — not predicted): reverting the
    /// `let flipped = ...` / `let encoding = ...` pair above to read
    /// `&layer.encoding` directly (dropping the `flip_positional`
    /// normalization) makes the `x` assertion below fail with `(4.0, 89.0)`
    /// — `count`'s own 4..5 range unions with the RAW (unflipped) `x`/`x2`
    /// fields (`bin_start`/`bin_end`, 50..89) because the mutant reads them
    /// onto the wrong slot, and the span-suppression guard (still correctly
    /// seeing an `x2`-shaped pair, just the WRONG pair) blocks the anchor —
    /// so the x axis ends up neither zero-anchored nor at `count`'s own
    /// range. The `y` assertion does not independently discriminate this
    /// mutant (`is_value_channel` is computed from `coord_flipped`
    /// directly, not from the encoding read, so it stays correctly `false`
    /// for `channel == "y"` either way) — kept for its own sake, as the
    /// position-channel-untouched pin under `CoordFlip`. The SAME mutant is
    /// also caught by the pre-existing (non-bar) T13 test
    /// `coord_flipped_layer_paired_endpoint_widens_the_other_axis`, which
    /// exercises the identical normalization for the x2/y2 union.
    #[test]
    fn coord_flipped_bar_position_channel_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(DISPLOT_HIST_KDE_FLIPPED_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = displot_hist_batch();
        let x_extent =
            numeric_domain_union("x", "count", None, &batch, &HashMap::new(), &spec, false)
                .expect("count is Float64");
        assert_eq!(
            x_extent,
            (0.0, 5.0),
            "under CoordFlip the histogram's VALUE channel (count, authored \
             y) is rendered on x and must carry the zero-anchor"
        );
        let y_extent = numeric_domain_union(
            "y",
            "bin_start",
            Some("bin_end"),
            &batch,
            &HashMap::new(),
            &spec,
            false,
        )
        .expect("bin_start/bin_end are Float64");
        assert_eq!(
            y_extent,
            (50.0, 89.0),
            "under CoordFlip the histogram's POSITION channel (the bin \
             range, authored x/x2) is rendered on y/y2 and must never gain \
             the value-channel zero-anchor"
        );
    }

    /// Verbatim `chart.to_dict()` JSON for `fm.Chart(df).mark_rank1d(orient="vertical")`
    /// (captured 2026-09-10, T5r cycle 2) — a REAL single-bar-layer
    /// production shape (`marks/diagnostic/_ranking.py`,
    /// `desugar_rank1d`'s vertical orientation: `x=feature`, `y=score`).
    /// Reused (with different fixture batches) for the `mx < 0.0` arm and
    /// the zero-before-degenerate-widening ordering test below.
    const SINGLE_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"bar","encoding":{"x":{"field":"feature"},"y":{"field":"score"}},"name":"bar"}]}"#;

    /// Isolates guard (a) alone: `SINGLE_BAR_LAYER_SPEC_JSON`'s bar layer
    /// has NO span field on either channel, so guard (b)
    /// (`layer_fields[1].is_none()`) is trivially satisfied regardless of
    /// which channel is being resolved — unlike
    /// `bar_position_channel_never_zero_extends` and the `x`/`y` pair in
    /// `coord_flipped_bar_position_channel_never_zero_extends`, whose
    /// position channels carry `x2`/`y2` and so are ALSO (redundantly)
    /// protected by guard (b). Querying the POSITION channel (`x` =
    /// `feature`) directly with real, contributing Float64 data means ONLY
    /// guard (a) stands between this layer and a wrong zero-anchor.
    ///
    /// RED (verified in place): replacing `is_value_channel` with `true` in
    /// the detection condition makes this fail with `(0.0, 90.0)`.
    #[test]
    fn bar_layer_own_position_channel_without_span_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema =
                Arc::new(Schema::new(vec![Field::new("feature", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![50.0, 90.0]))])
                .unwrap()
        };
        let extent =
            numeric_domain_union("x", "feature", None, &batch, &HashMap::new(), &spec, false)
                .expect("feature is Float64 in this fixture");
        assert_eq!(
            extent,
            (50.0, 90.0),
            "a bar layer's own POSITION channel must never gain the \
             value-channel zero-anchor, even with no span field to fall \
             back on for protection"
        );
    }

    /// A caller's scale carried only by layers whose column is entirely null
    /// resolves the auto path's `(0, 1)` fallback, never `(inf, -inf)`.
    #[test]
    fn carrying_scale_over_all_null_layers_falls_back_to_unit_domain() {
        const SPEC: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v","scale":{"type":"linear"}}}},{"mark":"line","encoding":{"x":{"field":"x"},"y":{"field":"v","scale":{"type":"linear"}}}}]}"#;
        let spec: ChartSpec = serde_json::from_str(SPEC).expect("layered spec JSON must deserialize");
        let scale = spec.layers.as_ref().unwrap()[0].encoding.y.as_ref().unwrap().scale.clone().unwrap();
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![None, None]))]).unwrap()
        };
        let extent = numeric_domain_union_over(
            "y",
            "v",
            None,
            &batch,
            &HashMap::new(),
            &spec,
            false,
            LayerUnion::CarryingScale(&scale),
        )
        .expect("an all-null Float64 column is readable");
        assert_eq!(extent, (0.0, 1.0));
    }

    /// An all-null Int64 column contributes no extent to either union: the
    /// auto path and a carried caller scale both resolve `(0, 1)`, not the
    /// integer fold's `(i64::MAX, i64::MIN)` seeds.
    #[test]
    fn all_null_integer_column_takes_the_unit_fallback() {
        const SPEC: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v","scale":{"type":"linear"}}}}]}"#;
        let spec: ChartSpec = serde_json::from_str(SPEC).expect("layered spec JSON must deserialize");
        let scale = spec.layers.as_ref().unwrap()[0].encoding.y.as_ref().unwrap().scale.clone().unwrap();
        let batch = {
            use arrow::array::Int64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![None, None]))]).unwrap()
        };
        for union in [LayerUnion::Every, LayerUnion::CarryingScale(&scale)] {
            let extent = numeric_domain_union_over(
                "y",
                "v",
                None,
                &batch,
                &HashMap::new(),
                &spec,
                false,
                union,
            )
            .expect("an all-null Int64 column is readable");
            assert_eq!(extent, (0.0, 1.0));
        }
    }

    /// Every value negative (`-3.0`, `-8.0`) — a bar chart of negative
    /// scores anchors from ABOVE, not below: `mx < 0.0 → mx = 0.0`, `mn`
    /// (already `<= 0.0`) untouched.
    ///
    /// RED (verified in place): deleting the `if mx < 0.0 { mx = 0.0; }`
    /// arm makes this fail with `(-8.0, -3.0)`.
    #[test]
    fn negative_bar_values_anchor_from_above() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![-3.0, -8.0]))])
                .unwrap()
        };
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false)
            .expect("score is Float64");
        assert_eq!(
            extent,
            (-8.0, 0.0),
            "an all-negative bar layer must anchor from above (mx -> 0.0), \
             not below"
        );
    }

    /// A single row (one category, one score `5.0`): pre-anchor `mn == mx
    /// == 5.0`, the degenerate case. Running the zero-anchor BEFORE the
    /// degenerate-domain widening (as `numeric_domain_union` does) resolves
    /// this to the real `[0, 5]` span; running it AFTER (widening first,
    /// `4.5..5.5`, THEN zero-extending `4.5 -> 0.0`) would instead yield the
    /// wrong `(0.0, 5.5)`.
    ///
    /// RED (verified in place): moving the `if bar_zero_wanted { ... }`
    /// block below the "Degenerate domain" widening block makes this fail
    /// with `(0.0, 5.5)`.
    #[test]
    fn single_row_bar_value_zero_extends_before_degenerate_widening() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![5.0]))]).unwrap()
        };
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false)
            .expect("score is Float64");
        assert_eq!(
            extent,
            (0.0, 5.0),
            "the zero-anchor must run BEFORE the degenerate-domain widening, \
             not after"
        );
    }

    /// `SINGLE_BAR_LAYER_SPEC_JSON` with `"mark_style": {"zero": false}`
    /// hand-added to the bar layer — the per-layer mirror of the
    /// single-chart `mark_bar(zero=False)` opt-out (`MarkKwargsSpec::zero`,
    /// T5r cycle 2). Hand-authored rather than captured from a live call on
    /// THIS fixture's own shape (`mark_rank1d`), but the field itself is
    /// live and user-reachable on the wire this test pins: see
    /// `MarkKwargsSpec::zero`'s doc comment for the verbatim, measured
    /// `fm.catplot(..., kind="bar", mark={"bar": {"zero": False}})` shape
    /// that reaches this exact `"mark_style":{"zero":false}` key today.
    const ZERO_FALSE_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"bar","encoding":{"x":{"field":"feature"},"y":{"field":"score"}},"mark_style":{"zero":false},"name":"bar"}]}"#;

    /// Otherwise-eligible bar layer (value channel, no paired span, data
    /// contributes) but explicitly opts out via `mark_style.zero == false`
    /// — the extent must stay the raw data range, not zero-anchored.
    ///
    /// RED (verified in place): dropping the `layer_wants_zero` guard makes
    /// this fail with `(0.0, 80.0)`.
    #[test]
    fn zero_false_layer_opts_out_of_anchor() {
        let spec: ChartSpec = serde_json::from_str(ZERO_FALSE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![50.0, 80.0]))])
                .unwrap()
        };
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false)
            .expect("score is Float64");
        assert_eq!(
            extent,
            (50.0, 80.0),
            "mark_style.zero == Some(false) must suppress the layered bar \
             zero-anchor, mirroring mark_bar(zero=False)"
        );
    }

    /// A `point` layer with a real field (`existing`) plus a sibling `bar`
    /// layer whose declared value field (`phantom`) is absent from every
    /// batch (no `data_source` resolves it, and it isn't in the primary
    /// batch either) — the bar layer's NAME alone must not force a
    /// zero-anchor when its own field never contributed a value
    /// (rust-quality-review finding 3).
    const PHANTOM_FIELD_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"point","encoding":{"y":{"field":"existing"}}},{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"phantom"}}}]}"#;

    /// RED (verified in place): replacing `value_field_contributed` with
    /// `layer_fields[0].is_some()` (the cycle-1 shape, "declared" instead of
    /// "contributed") makes this fail with `(0.0, 20.0)` — the bar layer's
    /// merely-declared `phantom` field forces the anchor even though it
    /// never resolves to data anywhere.
    #[test]
    fn phantom_field_bar_layer_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(PHANTOM_FIELD_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema =
                Arc::new(Schema::new(vec![Field::new("existing", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![10.0, 20.0]))])
                .unwrap()
        };
        let extent =
            numeric_domain_union("y", "existing", None, &batch, &HashMap::new(), &spec, false)
                .expect("existing is Float64");
        assert_eq!(
            extent,
            (10.0, 20.0),
            "a bar layer whose declared value field never resolves to data \
             must not force a zero-anchor from its name alone"
        );
    }

    // ── positional_category_union: the ordinal cross-layer ladder ────────────
    //
    // Same fixture discipline as the numeric layer-union tests above: lowered
    // spec JSON through the serde path Python actually writes, over one batch
    // whose columns make the answer readable straight off the returned vector.
    // The merged-flat lowering's renamed RHS column (`cat__rhs_0001`) is
    // reproduced literally, because the rename is precisely why the chart-level
    // column cannot answer for the other layer.

    /// The merged batch `Chart({cat:[c,a], grp:[p,q]}) + Chart({cat:[b], grp:[r]})`
    /// lowers to: six columns, diagonal-concat padded, the RHS's renamed.
    ///
    /// TWO categorical columns per side, on the two different positional
    /// channels, so "which channel did this category come from" is readable
    /// straight off the returned vector — an x/y mix-up in the channel match
    /// would otherwise be silent.
    fn merged_category_batch() -> RecordBatch {
        use arrow::array::{Float64Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("val", DataType::Float64, true),
            Field::new("grp", DataType::Utf8, true),
            Field::new("cat__rhs_0001", DataType::Utf8, true),
            Field::new("val__rhs_0001", DataType::Float64, true),
            Field::new("grp__rhs_0001", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("c"), Some("a"), None])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None])),
                Arc::new(StringArray::from(vec![Some("p"), Some("q"), None])),
                Arc::new(StringArray::from(vec![None, None, Some("b")])),
                Arc::new(Float64Array::from(vec![None, None, Some(40.0)])),
                Arc::new(StringArray::from(vec![None, None, Some("r")])),
            ],
        )
        .unwrap()
    }

    /// The ladder for `spec_json` over [`merged_category_batch`]. The RHS
    /// layers read `_ident_0`, the named Identity of the merged frame the `+`
    /// lowering publishes, so the fixture publishes it too.
    fn category_ladder(spec_json: &str, channel: &str, field: &str, paired: Option<&str>) -> Vec<String> {
        let spec: ChartSpec =
            serde_json::from_str(spec_json).expect("lowered spec JSON must deserialize");
        let batch = merged_category_batch();
        let outputs = HashMap::from([("_ident_0".to_string(), batch.clone())]);
        positional_category_union(channel, field, paired, &batch, &outputs, &spec, false)
            .expect("every category column in the fixture is Utf8")
    }

    /// The merged-flat lowering of `a + b` across two frames: chart-level
    /// encoding is the LHS's, the RHS layer's columns are renamed.
    const MERGED_LAYERS_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"val__rhs_0001"}},"data_source":"_ident_0"}]}"#;

    /// The ruled order, at the unit seam: first appearance, in layer order.
    ///
    /// `c, a` from the chart-level column then `b` from the RHS layer's. A
    /// sorting union answers `["a","b","c"]`; a union that walked the layers
    /// before the seed answers `["b","c","a"]`; the pre-fix code answers
    /// `["c","a"]`.
    #[test]
    fn layer_categories_join_the_ladder_in_first_appearance_order() {
        assert_eq!(
            category_ladder(MERGED_LAYERS_SPEC_JSON, "x", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string(), "b".to_string()],
        );
    }

    /// The `null`s the merge's padding leaves in BOTH category columns open no
    /// band: each layer binds its own paired value column, and every
    /// null-category row of each is null there too (`#247`B).
    #[test]
    fn merge_padding_opens_no_null_band_on_either_column() {
        let ladder = category_ladder(MERGED_LAYERS_SPEC_JSON, "x", "cat", Some("val"));
        assert!(
            !ladder.iter().any(|c| c == crate::render::arrow_cast::NULL_CATEGORY),
            "padding nulls carry no paired value and must claim no band; got {ladder:?}"
        );
    }

    /// Both positional channels categorical: each one widens from its OWN
    /// channel's layer field and never the other's. An x/y mix-up in the channel
    /// match would swap these two answers wholesale and nothing else would
    /// notice.
    #[test]
    fn a_layer_widens_only_the_channel_it_binds() {
        const BOTH_CATEGORICAL: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"grp__rhs_0001","type":"nominal"}},"data_source":"_ident_0"}]}"#;
        assert_eq!(
            category_ladder(BOTH_CATEGORICAL, "x", "cat", Some("grp")),
            vec!["c".to_string(), "a".to_string(), "b".to_string()],
            "the x ladder takes the layers' x fields"
        );
        assert_eq!(
            category_ladder(BOTH_CATEGORICAL, "y", "grp", Some("cat")),
            vec!["p".to_string(), "q".to_string(), "r".to_string()],
            "the y ladder takes the layers' y fields, never the x ones"
        );
    }

    /// Layers naming the SAME positional field as chart level (every
    /// composite-mark chart, every same-frame `+`) contribute nothing — the
    /// byte-identity control for the common case.
    #[test]
    fn same_field_layers_leave_the_ladder_exactly_as_seeded() {
        const SAME_FIELD: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"rule","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}}]}"#;
        assert_eq!(
            category_ladder(SAME_FIELD, "x", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string()],
        );
    }

    /// An unlayered spec returns the seed untouched — the `spec.layers == None`
    /// arm, which every flat chart takes.
    #[test]
    fn an_unlayered_spec_is_the_seed_alone() {
        const FLAT: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}}"#;
        assert_eq!(
            category_ladder(FLAT, "x", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string()],
        );
    }

    /// A layer binding ONE positional channel opens no null band, even though
    /// its category column carries nulls and the helper's unanswerable arm would
    /// keep them. This is `_inject_constant`'s reference-line shape: the layer's
    /// only other coordinate would be an inherited one, whose values say nothing
    /// about this layer's rows.
    ///
    /// RED (verified in place): drop the `own_paired.is_none()` retain and this
    /// returns `["c", "a", "b", "null"]`, and the rule layer draws a line per
    /// structural null row.
    #[test]
    fn a_single_channel_layer_opens_no_null_band() {
        const RULE_LAYER: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"rule","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"}}}]}"#;
        assert_eq!(
            category_ladder(RULE_LAYER, "x", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string(), "b".to_string()],
        );
    }

    /// Under `CoordFlip` the authored `y` occupies the rendered x axis, so the
    /// layer's authored `y` field is the one that widens x — and its authored
    /// `x` is the opposite coordinate the null band is judged against. Reading
    /// `layer.encoding` unflipped would union the wrong column onto the wrong
    /// axis, silently: the two answers below would simply trade places.
    #[test]
    fn coord_flipped_layer_widens_the_axis_its_authored_channel_occupies() {
        const FLIPPED: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"grp__rhs_0001","type":"nominal"}}}],"coord":{"kind":"flip"}}"#;
        // `channel`/`field` arrive POST-flip: rendered x holds the authored `grp`.
        assert_eq!(
            category_ladder(FLIPPED, "x", "grp", Some("cat")),
            vec!["p".to_string(), "q".to_string(), "r".to_string()],
            "the authored y field must widen the axis it occupies post-flip"
        );
        assert_eq!(
            category_ladder(FLIPPED, "y", "cat", Some("grp")),
            vec!["c".to_string(), "a".to_string(), "b".to_string()],
            "the authored x field must widen the post-flip y axis"
        );
    }

    /// The flip is per LAYER, not per chart (`#172`(1), close review F1): a
    /// member carrying its OWN `CoordFlip` inside an unflipped chart widens each
    /// axis from the channel it occupies POST-flip, exactly as the chart-level
    /// case above does.
    ///
    /// RED (verified in place): hoist the `coord_flipped` read back out of the
    /// loop — where only the chart's flip can be read — and the two answers
    /// below trade places, i.e. the flipped member's value column is enumerated
    /// as bands on the axis its categories were supposed to reach. The two
    /// assertions are the two halves of that defect: no column from the wrong
    /// channel on the ladder, AND the layer's own post-flip column present.
    #[test]
    fn a_layer_with_its_own_flip_widens_the_axis_its_post_flip_channel_occupies() {
        // `coord` sits beside the layer's `encoding`, not inside it — a
        // mis-nested key deserializes to "no coord" and would pass this test for
        // the wrong reason, which is what the premise below rules out.
        const LAYER_FLIP: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"grp__rhs_0001","type":"nominal"}},"coord":{"kind":"flip"}}]}"#;
        let spec: ChartSpec = serde_json::from_str(LAYER_FLIP).unwrap();
        assert!(
            spec.coord.is_none() && spec.layers.as_ref().unwrap()[1].coord.is_some(),
            "premise: an unflipped chart whose second member carries its own flip"
        );

        assert_eq!(
            category_ladder(LAYER_FLIP, "x", "cat", Some("grp")),
            vec!["c".to_string(), "a".to_string(), "r".to_string()],
            "the flipped member's authored y (`grp__rhs`) is what occupies x"
        );
        assert_eq!(
            category_ladder(LAYER_FLIP, "y", "grp", Some("cat")),
            vec!["p".to_string(), "q".to_string(), "b".to_string()],
            "…and its authored x (`cat__rhs`) is what occupies y"
        );
    }

    /// A chart-level flip still reaches a layer that declares no coord of its
    /// own — the `effective_coord` fallback, asserted here because the per-layer
    /// read above could satisfy its own test by ignoring `spec.coord` entirely.
    #[test]
    fn a_layer_without_its_own_coord_still_takes_the_charts_flip() {
        const CHART_FLIP_ONLY: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"grp","type":"nominal"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"grp__rhs_0001","type":"nominal"}}}],"coord":{"kind":"flip"}}"#;
        assert_eq!(
            category_ladder(CHART_FLIP_ONLY, "x", "grp", Some("cat")),
            vec!["p".to_string(), "q".to_string(), "r".to_string()],
        );
    }

    /// A layer whose own column cannot be a category KEY contributes nothing,
    /// and says so by skipping rather than by refusing the render (close review
    /// F2). A `Timestamp` layer field beside a categorical axis is the measured
    /// shape: `bar(x="cat:N") + line(x="when:T")` raised
    /// `UnsupportedDtype` for a composition that rendered before this function
    /// existed, where `numeric_domain_union`'s layer loop shrugs at the same
    /// class of unreadable field.
    ///
    /// The `Err` is still reachable — the enumeration below the gate keeps its
    /// `?`. What the gate removes is the one case where refusing is provably
    /// pointless: no mark could have addressed a band on this column either,
    /// since the per-row keyer admits exactly the dtypes the gate does.
    #[test]
    fn a_layer_field_that_cannot_be_a_category_key_is_skipped_not_raised() {
        use arrow::array::{Float64Array, TimestampMillisecondArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("val", DataType::Float64, true),
            Field::new(
                "when",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("c"), Some("a")])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])),
                Arc::new(TimestampMillisecondArray::from(vec![Some(0_i64), Some(86_400_000)])),
            ],
        )
        .unwrap();
        const TEMPORAL_LAYER: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"line","encoding":{"x":{"field":"when","type":"temporal"},"y":{"field":"val"}}}]}"#;
        let spec: ChartSpec = serde_json::from_str(TEMPORAL_LAYER).unwrap();
        assert_eq!(
            positional_category_union("x", "cat", Some("val"), &batch, &HashMap::new(), &spec, false)
                .expect("a temporal layer field must skip, not refuse the whole ladder"),
            vec!["c".to_string(), "a".to_string()],
        );
    }

    /// The gate is about the COLUMN, not about the declared type: a quantitative
    /// layer channel over a numeric column still contributes, because the band
    /// ladder can serve it and its marks do land on those bands. Measured:
    /// `bar(x="cat:N") + point(x="n:Q")` draws its 2 points on the two bands
    /// this adds. An `infer_spec_type`-based gate would drop both — and would
    /// also drop an unannotated integer categorical, which infers Quantitative.
    #[test]
    fn a_quantitative_layer_column_the_ladder_can_serve_still_contributes() {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("val", DataType::Float64, true),
            Field::new("n", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("c"), Some("a")])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])),
                Arc::new(arrow::array::Int64Array::from(vec![Some(10), Some(20)])),
            ],
        )
        .unwrap();
        const QUANTITATIVE_LAYER: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"point","encoding":{"x":{"field":"n","type":"quantitative"},"y":{"field":"val"}}}]}"#;
        let spec: ChartSpec = serde_json::from_str(QUANTITATIVE_LAYER).unwrap();
        assert_eq!(
            positional_category_union("x", "cat", Some("val"), &batch, &HashMap::new(), &spec, false)
                .unwrap(),
            vec!["c".to_string(), "a".to_string(), "10".to_string(), "20".to_string()],
        );
    }

    /// An `independent_y` layer resolves its own y slot (GH #52) and must not
    /// widen the primary y ladder, exactly as it does not widen the primary
    /// numeric y extent. Its x still does.
    #[test]
    fn an_independent_y_layer_widens_x_but_never_the_primary_y() {
        const INDEPENDENT: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"val__rhs_0001"}},"independent_y":true}]}"#;
        assert_eq!(
            category_ladder(INDEPENDENT, "x", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string(), "b".to_string()],
        );
        // A categorical y with an independent-y layer: the layer's own y field
        // must stay out of the primary ladder.
        const INDEPENDENT_CAT_Y: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"val"},"y":{"field":"cat","type":"nominal"}},"layers":[{"mark":"point","encoding":{"x":{"field":"val"},"y":{"field":"cat","type":"nominal"}}},{"mark":"point","encoding":{"x":{"field":"val__rhs_0001"},"y":{"field":"cat__rhs_0001","type":"nominal"}},"independent_y":true}]}"#;
        assert_eq!(
            category_ladder(INDEPENDENT_CAT_Y, "y", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string()],
        );
    }

    /// A layer field absent from every batch is the normal "wrong `data_source`
    /// / not loaded yet" outcome, not an error — the same reading
    /// `numeric_domain_union` gives one.
    #[test]
    fn a_layer_field_in_no_batch_is_skipped_not_raised() {
        const PHANTOM: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"point","encoding":{"x":{"field":"nowhere","type":"nominal"},"y":{"field":"val"}}}]}"#;
        assert_eq!(
            category_ladder(PHANTOM, "x", "cat", Some("val")),
            vec!["c".to_string(), "a".to_string()],
        );
    }

    // --- per-layer domain source ---------------------------------------------
    //
    // Each layer's positional contribution is read from the batch the layer
    // draws: its `data_source` output when it names one, else the chart's
    // data. Every fixture below gives the chart's data a same-named column
    // with values far outside what any layer draws, so a reader that takes a
    // layer's field from another batch shows up in the returned domain.

    fn columns_batch(cols: Vec<(&str, arrow::array::ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> =
            cols.iter().map(|(name, col)| Field::new(*name, col.data_type().clone(), true)).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols.into_iter().map(|(_, c)| c).collect())
            .unwrap()
    }

    fn f64_col(values: &[f64]) -> arrow::array::ArrayRef {
        Arc::new(arrow::array::Float64Array::from(values.to_vec()))
    }

    fn opt_f64_col(values: &[Option<f64>]) -> arrow::array::ArrayRef {
        Arc::new(arrow::array::Float64Array::from(values.to_vec()))
    }

    fn opt_str_col(values: &[Option<&str>]) -> arrow::array::ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    fn spec_from(json: &str) -> ChartSpec {
        serde_json::from_str(json).expect("layered spec JSON must deserialize")
    }

    /// `mark_smooth(ci=0.95)`'s lowering: the chart encoding is the raw
    /// `x`/`y`, and both drawing layers read the `smooth` output, the ribbon
    /// on `ci_lower`/`ci_upper` and the line on the fitted `y`.
    const SMOOTH_CI_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"}},"layers":[{"mark":"ribbon","encoding":{"x":{"field":"x"},"y":{"field":"ci_lower"},"y2":{"field":"ci_upper"}},"data_source":"smooth"},{"mark":"line","encoding":{"x":{"field":"x"},"y":{"field":"y"}},"data_source":"smooth"}]}"#;

    /// The raw frame reaches -4.15..22; the fit and its band stay inside
    /// -0.2..20.
    fn smooth_ci_batches() -> (RecordBatch, HashMap<String, RecordBatch>) {
        let raw = columns_batch(vec![("x", f64_col(&[0.0, 5.0, 10.0])), ("y", f64_col(&[-4.15, 10.0, 22.0]))]);
        let smooth = columns_batch(vec![
            ("x", f64_col(&[0.0, 5.0, 10.0])),
            ("y", f64_col(&[0.5, 10.0, 19.5])),
            ("ci_lower", f64_col(&[-0.2, 9.0, 18.0])),
            ("ci_upper", f64_col(&[1.0, 11.0, 20.0])),
        ]);
        let outputs = HashMap::from([
            ("smooth".to_string(), smooth),
            (FINAL_OUTPUT_KEY.to_string(), raw.clone()),
        ]);
        (raw, outputs)
    }

    /// The line layer's `y` is the fitted column of its own output, not the
    /// raw column of the same name, so the axis stops at the band's extent.
    /// The rejected reading (a layer field found in the chart's data first)
    /// returns the raw `(-4.15, 22.0)`.
    #[test]
    fn a_stat_layer_reads_its_own_output_not_the_raw_column_of_the_same_name() {
        let spec = spec_from(SMOOTH_CI_SPEC_JSON);
        let (raw, outputs) = smooth_ci_batches();
        let extent = numeric_domain_union("y", "ci_lower", Some("ci_upper"), &raw, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (-0.2, 20.0));
    }

    /// A caller's scale stating no `domain` spans the same per-layer data as
    /// the auto path does.
    #[test]
    fn a_caller_scale_spans_each_layers_own_batch() {
        let spec = spec_from(SMOOTH_CI_SPEC_JSON);
        let scale: crate::spec::encoding::ScaleSpec =
            serde_json::from_str(r#"{"type":"linear"}"#).expect("a linear scale dict");
        let (raw, outputs) = smooth_ci_batches();
        let extent = numeric_domain_union_over(
            "y",
            "ci_lower",
            Some("ci_upper"),
            &raw,
            &outputs,
            &spec,
            false,
            LayerUnion::CarryingScale(&scale),
        )
        .expect("every column is Float64");
        assert_eq!(extent, (-0.2, 20.0));
    }

    /// `mark_errorbar` over a frame that has a column of its own named
    /// `lower`: the bar layer draws the `lower`/`upper` its own output holds.
    const ERRORBAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}}},{"mark":"rule","encoding":{"x":{"field":"x"},"y":{"field":"lower"},"y2":{"field":"upper"}},"data_source":"eb"}]}"#;

    fn errorbar_batches() -> (RecordBatch, HashMap<String, RecordBatch>) {
        let raw = columns_batch(vec![
            ("x", f64_col(&[1.0, 2.0, 3.0])),
            ("v", f64_col(&[0.5, 1.0, 1.5])),
            ("lower", f64_col(&[-100.0, -100.0, -100.0])),
        ]);
        let eb = columns_batch(vec![
            ("x", f64_col(&[1.0, 2.0, 3.0])),
            ("lower", f64_col(&[-1.2, -0.5, 0.0])),
            ("upper", f64_col(&[1.0, 1.5, 2.0])),
        ]);
        let outputs =
            HashMap::from([("eb".to_string(), eb), (FINAL_OUTPUT_KEY.to_string(), raw.clone())]);
        (raw, outputs)
    }

    /// The user's own `lower` column (-100) is drawn by no layer, so it
    /// widens nothing. The rejected reading returns `(-100.0, 2.0)`.
    #[test]
    fn a_user_column_sharing_a_layer_field_name_is_not_read_for_that_layer() {
        let spec = spec_from(ERRORBAR_SPEC_JSON);
        let (raw, outputs) = errorbar_batches();
        let extent = numeric_domain_union("y", "v", None, &raw, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (-1.2, 2.0));
    }

    /// Faceted and shared, the chart's data also takes in the all-panels
    /// batch, and a layer on its own output still reads only that output:
    /// the global batch's `lower` (-100) is the user's column, not the bar's.
    #[test]
    fn the_facet_shared_batch_widens_only_the_layers_on_the_charts_data() {
        let spec = spec_from(ERRORBAR_SPEC_JSON);
        let (raw, mut outputs) = errorbar_batches();
        let global = columns_batch(vec![
            ("x", f64_col(&[1.0, 2.0, 3.0, 4.0])),
            ("v", f64_col(&[0.5, 1.0, 1.5, 9.0])),
            ("lower", f64_col(&[-100.0, -100.0, -100.0, -100.0])),
        ]);
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), global);
        let extent = numeric_domain_union("y", "v", None, &raw, &outputs, &spec, true)
            .expect("every column is Float64");
        assert_eq!(extent, (-1.2, 9.0), "the panel-spanning `v` reaches 9, the user's `lower` reaches nothing");
    }

    /// Both operands of a different-frame `+`, each on its own rows: the
    /// chart's data is the merged frame, whose `v` (100..200) neither layer
    /// draws.
    const OPERAND_BRANCHES_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}},"data_source":"lhs"},{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}},"data_source":"rhs"}]}"#;

    fn operand_batches(lhs_v: &[f64]) -> (RecordBatch, HashMap<String, RecordBatch>) {
        let merged = columns_batch(vec![("x", f64_col(&[1.0, 2.0])), ("v", f64_col(&[100.0, 200.0]))]);
        let lhs_x: Vec<f64> = (0..lhs_v.len()).map(|i| i as f64).collect();
        let lhs = columns_batch(vec![("x", f64_col(&lhs_x)), ("v", f64_col(lhs_v))]);
        let rhs = columns_batch(vec![("x", f64_col(&[1.0, 2.0])), ("v", f64_col(&[3.0, 4.0]))]);
        let outputs = HashMap::from([
            ("lhs".to_string(), lhs),
            ("rhs".to_string(), rhs),
            (FINAL_OUTPUT_KEY.to_string(), merged.clone()),
        ]);
        (merged, outputs)
    }

    /// Layer 0's field is the seed, and it is judged against layer 0's own
    /// batch. The rejected reading seeds from the chart's data:
    /// `(1.0, 200.0)`.
    #[test]
    fn the_seed_field_is_read_from_layer_zeros_batch() {
        let spec = spec_from(OPERAND_BRANCHES_SPEC_JSON);
        let (merged, outputs) = operand_batches(&[1.0, 2.0]);
        let extent = numeric_domain_union("y", "v", None, &merged, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 4.0));
    }

    /// A layer whose batch has no rows draws nothing and widens nothing; the
    /// reader does not go looking for the field elsewhere.
    #[test]
    fn an_empty_layer_batch_contributes_nothing() {
        let spec = spec_from(OPERAND_BRANCHES_SPEC_JSON);
        let (merged, outputs) = operand_batches(&[]);
        let extent = numeric_domain_union("y", "v", None, &merged, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (3.0, 4.0));
    }

    /// A layer on the chart's data that binds no `y` draws the chart's `y`
    /// from that data, so the chart's data still widens the axis even when
    /// layer 0 reads an output of its own. A reader that took the seed from
    /// layer 0's batch alone would return `(1.0, 2.0)`.
    #[test]
    fn a_layer_inheriting_the_seed_reads_it_from_the_charts_data() {
        const INHERITING: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"v"}},"data_source":"lhs"},{"mark":"point","encoding":{"x":{"field":"x"}}}]}"#;
        let spec = spec_from(INHERITING);
        let (merged, outputs) = operand_batches(&[1.0, 2.0]);
        let extent = numeric_domain_union("y", "v", None, &merged, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 200.0));
    }

    /// The ordinal counterpart: two operands on the same field name, each on
    /// its own rows, over a merged frame whose `g` is null on the right
    /// operand's rows while its `v` is not.
    const OPERAND_CATEGORIES_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}},"data_source":"lhs"},{"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}},"data_source":"rhs"}]}"#;

    fn operand_category_batches() -> (RecordBatch, HashMap<String, RecordBatch>) {
        let merged = columns_batch(vec![
            ("g", opt_str_col(&[Some("a"), Some("b"), Some("c"), None, None, None])),
            ("v", opt_f64_col(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0), Some(6.0)])),
        ]);
        let lhs = columns_batch(vec![
            ("g", opt_str_col(&[Some("a"), Some("b"), Some("c")])),
            ("v", opt_f64_col(&[Some(1.0), Some(2.0), Some(3.0)])),
        ]);
        let rhs = columns_batch(vec![
            ("g", opt_str_col(&[Some("x"), Some("y"), Some("z")])),
            ("v", opt_f64_col(&[Some(4.0), Some(5.0), Some(6.0)])),
        ]);
        let outputs = HashMap::from([
            ("lhs".to_string(), lhs),
            ("rhs".to_string(), rhs),
            (FINAL_OUTPUT_KEY.to_string(), merged.clone()),
        ]);
        (merged, outputs)
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// Each operand's categories come from its own rows, in layer order, and
    /// no `null` band opens for rows neither layer draws. The rejected
    /// reading seeds from the merged frame and skips the right operand as a
    /// same-named field: `["a", "b", "c", "null"]`.
    #[test]
    fn each_layers_categories_come_from_its_own_batch() {
        let spec = spec_from(OPERAND_CATEGORIES_SPEC_JSON);
        let (merged, outputs) = operand_category_batches();
        let ladder = positional_category_union("x", "g", Some("v"), &merged, &outputs, &spec, false)
            .expect("every category column is Utf8");
        assert_eq!(ladder, strings(&["a", "b", "c", "x", "y", "z"]));
    }

    /// A different-frame `+` whose layer 0 (a rule on its own rows) draws no
    /// x: the chart-level x names the field the right operand (a bar on its
    /// own rows) draws, and layer 0's own rows carry a same-named column of
    /// their own frame.
    const NON_DRAWING_LEAD_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"w"}},"layers":[{"mark":"rule","encoding":{"y":{"field":"w"}},"data_source":"lhs"},{"mark":"bar","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"y"}},"data_source":"rhs"}]}"#;

    fn non_drawing_lead_batches(lhs_g: arrow::array::ArrayRef) -> (RecordBatch, HashMap<String, RecordBatch>) {
        let merged = columns_batch(vec![
            ("g", opt_str_col(&[Some("a"), Some("b"), Some("c"), Some("a"), Some("b")])),
            ("w", opt_f64_col(&[Some(1.0), Some(2.0), Some(3.0), None, None])),
        ]);
        let lhs = columns_batch(vec![("g", lhs_g), ("w", f64_col(&[1.0, 2.0, 3.0]))]);
        let rhs = columns_batch(vec![
            ("g", opt_str_col(&[Some("a"), Some("b")])),
            ("y", f64_col(&[4.0, 5.0])),
        ]);
        let outputs = HashMap::from([
            ("lhs".to_string(), lhs),
            ("rhs".to_string(), rhs),
            (FINAL_OUTPUT_KEY.to_string(), merged.clone()),
        ]);
        (merged, outputs)
    }

    /// The x ladder comes from the layer that draws x, the bar: `a, b`. The
    /// rejected reading seeds from layer 0's batch, which draws no x, and
    /// returns its frame's `["a", "b", "c"]`.
    #[test]
    fn an_axis_layer_zero_does_not_draw_is_seeded_from_the_first_layer_that_does() {
        let spec = spec_from(NON_DRAWING_LEAD_SPEC_JSON);
        let (merged, outputs) =
            non_drawing_lead_batches(opt_str_col(&[Some("a"), Some("b"), Some("c")]));
        let ladder = positional_category_union("x", "g", Some("y"), &merged, &outputs, &spec, false)
            .expect("every category column is Utf8");
        assert_eq!(ladder, strings(&["a", "b"]));
    }

    /// The axis is typed from the drawing layer's column too: layer 0's own
    /// `g` is numeric, the bar's is text. The rejected reading types the
    /// axis from layer 0's Float64 column.
    #[test]
    fn an_axis_layer_zero_does_not_draw_is_typed_from_the_first_layer_that_does() {
        let spec = spec_from(NON_DRAWING_LEAD_SPEC_JSON);
        let (merged, outputs) = non_drawing_lead_batches(f64_col(&[7.0, 8.0, 9.0]));
        let located = locate_seed_field("x", "g", &spec, &merged, &outputs).expect("the bar draws g");
        assert_eq!(located.col.data_type(), &arrow::datatypes::DataType::Utf8);
    }

    /// The numeric counterpart: layer 0's own `g` (7..9) is not drawn, so it
    /// widens nothing, and the point layer's own rows are where the seed is
    /// read. The rejected reading seeds from layer 0's rows and returns
    /// `(1.0, 9.0)`.
    #[test]
    fn a_numeric_axis_layer_zero_does_not_draw_is_seeded_from_the_drawing_layer() {
        const NUMERIC: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"x":{"field":"g"},"y":{"field":"w"}},"layers":[{"mark":"rule","encoding":{"y":{"field":"w"}},"data_source":"lhs"},{"mark":"point","encoding":{"x":{"field":"g"},"y":{"field":"w"}},"data_source":"rhs"}]}"#;
        let spec = spec_from(NUMERIC);
        let merged = columns_batch(vec![("g", f64_col(&[7.0, 9.0])), ("w", f64_col(&[1.0, 2.0]))]);
        let outputs = HashMap::from([
            ("lhs".to_string(), columns_batch(vec![("g", f64_col(&[7.0, 9.0])), ("w", f64_col(&[1.0, 2.0]))])),
            ("rhs".to_string(), columns_batch(vec![("g", f64_col(&[1.0, 2.0])), ("w", f64_col(&[3.0, 4.0]))])),
            (FINAL_OUTPUT_KEY.to_string(), merged.clone()),
        ]);
        let extent = numeric_domain_union("x", "g", None, &merged, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 2.0));
    }

    /// A categorical scale the caller wrote takes its categories from the
    /// seed's own column, which is layer 0's batch, not the merged frame.
    /// The rejected reading returns the merged frame's `["a", "b", "c",
    /// "null"]`.
    #[test]
    fn a_caller_categorical_scale_reads_layer_zeros_batch() {
        const BAND: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"g","type":"nominal","scale":{"type":"band"}},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"g","type":"nominal","scale":{"type":"band"}},"y":{"field":"v"}},"data_source":"lhs"},{"mark":"point","encoding":{"x":{"field":"g","type":"nominal","scale":{"type":"band"}},"y":{"field":"v"}},"data_source":"rhs"}]}"#;
        let spec = spec_from(BAND);
        let (merged, outputs) = operand_category_batches();
        let enc = spec.encoding.x.clone().expect("the chart binds x");
        let mut warnings = Vec::new();
        let scale = super::super::build_axis_scale(
            "x",
            &enc,
            None,
            super::super::PositionalFields { x: Some("g"), y: Some("v") },
            &merged,
            &outputs,
            (0.0, 300.0),
            &spec,
            false,
            None,
            &mut warnings,
        )
        .expect("a band scale over a Utf8 column resolves");
        let crate::render::scale_resolve::ScaleKind::Ordinal(ordinal) = scale else {
            panic!("a band scale resolves to an ordinal scale");
        };
        assert_eq!(ordinal.ticks_internal(), strings(&["a", "b", "c"]));
    }

    /// `mark_raster`'s lowering: one image layer on the Raster output, whose
    /// encoding names the source columns the transform binned. The image is
    /// drawn over the extent its one-row batch records, so that extent is its
    /// contribution; here it is pinned inside the raw range (as a facet pin
    /// does) so the two readings differ. Reading the encoded `x` from the
    /// layer's batch finds nothing and falls back to `(0.0, 1.0)`, which
    /// places the image off the plot.
    #[test]
    fn a_raster_image_contributes_the_extent_its_batch_records() {
        const RASTER: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"}},"layers":[{"mark":"image","encoding":{"x":{"field":"x"},"y":{"field":"y"}},"data_source":"raster"}]}"#;
        let spec = spec_from(RASTER);
        let raw = columns_batch(vec![("x", f64_col(&[-2.0, 0.5, 4.0])), ("y", f64_col(&[10.0, 11.0, 12.0]))]);
        let raster = columns_batch(vec![
            ("x_min", f64_col(&[-1.5])),
            ("x_max", f64_col(&[3.5])),
            ("y_min", f64_col(&[10.5])),
            ("y_max", f64_col(&[11.5])),
        ]);
        let outputs =
            HashMap::from([("raster".to_string(), raster), (FINAL_OUTPUT_KEY.to_string(), raw.clone())]);
        let x = numeric_domain_union("x", "x", None, &raw, &outputs, &spec, false).expect("Float64 columns");
        let y = numeric_domain_union("y", "y", None, &raw, &outputs, &spec, false).expect("Float64 columns");
        assert_eq!((x, y), ((-1.5, 3.5), (10.5, 11.5)));
    }

    /// A filtered left operand over one frame `+` a right operand over
    /// another: the right layer reads the merged source under its renamed
    /// column, and the merged source also holds the left operand's
    /// unfiltered `a` (40), which no layer draws. The rejected reading takes
    /// the same-named `a` from every output a layer references: `(1.0, 40.0)`.
    #[test]
    fn a_referenced_output_widens_only_through_the_field_its_layer_binds() {
        const CROSS_FRAME: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"tick","encoding":{"x":{"field":"a"},"y":{"field":"b"}},"layers":[{"mark":"tick","encoding":{"x":{"field":"a"},"y":{"field":"b"}}},{"mark":"tick","encoding":{"x":{"field":"a__rhs_0001"},"y":{"field":"b__rhs_0001"}},"data_source":"_ident_0"}]}"#;
        let spec = spec_from(CROSS_FRAME);
        let filtered = columns_batch(vec![("a", f64_col(&[1.0, 2.0, 3.0])), ("b", f64_col(&[1.0, 2.0, 3.0]))]);
        let source = columns_batch(vec![
            ("a", opt_f64_col(&[Some(1.0), Some(2.0), Some(3.0), Some(40.0), None, None])),
            ("b", opt_f64_col(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), None, None])),
            ("a__rhs_0001", opt_f64_col(&[None, None, None, None, Some(5.0), Some(6.0)])),
            ("b__rhs_0001", opt_f64_col(&[None, None, None, None, Some(7.0), Some(8.0)])),
        ]);
        let outputs = HashMap::from([
            ("_ident_0".to_string(), source),
            (FINAL_OUTPUT_KEY.to_string(), filtered.clone()),
        ]);
        let extent = numeric_domain_union("x", "a", None, &filtered, &outputs, &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 6.0));
    }

    /// `(P(df).encode(x="a", y="b") + P(df).encode(y="b")).encode(x="c")`:
    /// layer 0 binds its own `x="a"`, the second point layer binds no `x` and
    /// so draws the chart-level `c` from the chart's data. The axis covers
    /// both. The rejected reading takes the inheriting layer's field to be
    /// layer 0's and returns `(1.0, 3.0)`, leaving the points at 100..300 off
    /// the panel.
    const INHERITED_DIFFERENT_FIELD_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"c"},"y":{"field":"b"}},"layers":[{"mark":"point","encoding":{"x":{"field":"a"},"y":{"field":"b"}}},{"mark":"point","encoding":{"y":{"field":"b"}}}]}"#;

    #[test]
    fn an_inheriting_layer_contributes_the_chart_field_it_draws() {
        let spec = spec_from(INHERITED_DIFFERENT_FIELD_SPEC_JSON);
        let batch = columns_batch(vec![
            ("a", f64_col(&[1.0, 2.0, 3.0])),
            ("b", f64_col(&[1.0, 2.0, 3.0])),
            ("c", f64_col(&[100.0, 200.0, 300.0])),
        ]);
        let extent = numeric_domain_union("x", "a", None, &batch, &HashMap::new(), &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 300.0));
    }

    /// A one-axis tick writing only `y` takes no inherited `x`
    /// (`Mark::inherited_positional`), so the chart-level `c` it never draws
    /// widens nothing. The rejected reading, an inheritance that ignores the
    /// mark, returns `(1.0, 300.0)`.
    #[test]
    fn a_one_axis_tick_does_not_widen_the_axis_with_a_field_it_never_draws() {
        let spec = spec_from(&INHERITED_DIFFERENT_FIELD_SPEC_JSON.replace(
            r#"{"mark":"point","encoding":{"y":{"field":"b"}}}"#,
            r#"{"mark":"tick","encoding":{"y":{"field":"b"}}}"#,
        ));
        let batch = columns_batch(vec![
            ("a", f64_col(&[1.0, 2.0, 3.0])),
            ("b", f64_col(&[1.0, 2.0, 3.0])),
            ("c", f64_col(&[100.0, 200.0, 300.0])),
        ]);
        let extent = numeric_domain_union("x", "a", None, &batch, &HashMap::new(), &spec, false)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 3.0));
    }

    /// A one-axis reference rule over its own rows, `x=ref`, with a `y`
    /// Python filled from the bar and listed as inherited: the rule draws no
    /// `y` (`Mark::inherited_positional`), so the band judge does not read
    /// that filled `val`, which the rule's rows lack. The rejected reading,
    /// the authored encoding with the filled `y`, opens a `null` band.
    #[test]
    fn a_one_axis_rule_judges_its_null_band_without_a_filled_opposite_channel() {
        const SPEC: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"data_source":"lhs"},{"mark":"rule","encoding":{"x":{"field":"ref","type":"nominal"},"y":{"field":"val"}},"data_source":"rhs","inherited_positional":["y"]}]}"#;
        let spec = spec_from(SPEC);
        let lhs = columns_batch(vec![
            ("cat", opt_str_col(&[Some("a"), Some("b"), Some("c")])),
            ("val", f64_col(&[1.0, 2.0, 3.0])),
        ]);
        let rhs = columns_batch(vec![("ref", opt_str_col(&[Some("b"), None, None]))]);
        let outputs = HashMap::from([
            ("lhs".to_string(), lhs.clone()),
            ("rhs".to_string(), rhs),
            (FINAL_OUTPUT_KEY.to_string(), lhs.clone()),
        ]);
        let ladder = positional_category_union("x", "cat", Some("val"), &lhs, &outputs, &spec, false)
            .expect("every category column is Utf8");
        assert_eq!(ladder, strings(&["a", "b", "c"]));
    }

    /// The ordinal counterpart: the inheriting layer's categories are those of
    /// the chart-level field it draws, after layer 0's own. The rejected
    /// reading returns layer 0's `["p", "q"]` alone.
    #[test]
    fn an_inheriting_layer_contributes_the_chart_categories_it_draws() {
        const INHERITED: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"h","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}}},{"mark":"point","encoding":{"y":{"field":"v"}}}]}"#;
        let spec = spec_from(INHERITED);
        let batch = columns_batch(vec![
            ("g", opt_str_col(&[Some("p"), Some("q")])),
            ("h", opt_str_col(&[Some("r"), Some("s")])),
            ("v", opt_f64_col(&[Some(1.0), Some(2.0)])),
        ]);
        let ladder = positional_category_union("x", "g", Some("v"), &batch, &HashMap::new(), &spec, false)
            .expect("every category column is Utf8");
        assert_eq!(ladder, strings(&["p", "q", "r", "s"]));
    }

    /// A layered spec whose bound field is in no batch is refused as an
    /// unlayered one is, with the same `UnknownColumn`. The rejected reading
    /// skips every seed read and returns an empty ladder.
    #[test]
    fn a_layered_axis_field_in_no_batch_is_an_unknown_column() {
        const PHANTOM_SEED: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"nowhere","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"nowhere","type":"nominal"},"y":{"field":"v"}}}]}"#;
        let spec = spec_from(PHANTOM_SEED);
        let batch = columns_batch(vec![("v", f64_col(&[1.0, 2.0]))]);
        let err = positional_category_union("x", "nowhere", Some("v"), &batch, &HashMap::new(), &spec, false)
            .expect_err("a bound field no batch holds is refused");
        assert_eq!(err, RenderError::UnknownColumn { name: "nowhere".to_string() });
    }

    /// A layered axis is typed from the column its establishing layer draws,
    /// in the batch it draws: a raster image draws `x` through its extent
    /// columns, so `x_min` types the axis, not the source column `x` the
    /// chart rows hold. Rejects typing the axis from a same-named column of
    /// another batch.
    #[test]
    fn a_raster_image_types_its_axis_from_its_extent_columns() {
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let x = Some(EncodingSpec { field: "x".into(), ..Default::default() });
        let image = Layer {
            mark: Mark::Image,
            encoding: Encoding { x: x.clone(), ..Default::default() },
            data_source: Some("raster".into()),
            ..Default::default()
        };
        let spec = ChartSpec { encoding: Encoding { x, ..Default::default() }, layers: Some(vec![image]), ..Default::default() };
        let column = |name: &str, values: Vec<f64>| {
            let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(arrow::array::Float64Array::from(values))]).unwrap()
        };
        let rows = column("x", vec![1.0, 2.0]);
        let mut outputs = HashMap::new();
        outputs.insert("raster".to_string(), column("x_min", vec![-3.0]));
        let located = locate_seed_field("x", "x", &spec, &rows, &outputs).expect("the image draws x_min");
        assert!(std::ptr::eq(located.batch, &outputs["raster"]), "read from the batch the image draws");
        outputs.insert("raster".to_string(), column("x", vec![0.0]));
        assert!(
            locate_seed_field("x", "x", &spec, &rows, &outputs).is_none(),
            "no other column of the batch types the axis, nor the chart rows' x"
        );
    }

    /// A layered channel's reads list only the layers that draw it: a layer
    /// whose mark does not draw the channel, and a layer whose batch lacks
    /// the column (a statistic dropped it), contribute nothing, and the first
    /// read is the first layer that does draw it.
    #[test]
    fn aux_reads_list_only_the_layers_that_draw_the_channel() {
        use crate::spec::encoding::{Encoding, EncodingSpec, AUXILIARY_SCALE_CHANNELS};
        let color = |field: &str| Encoding { color: Some(EncodingSpec { field: field.into(), ..Default::default() }), ..Default::default() };
        let layers = vec![
            Layer { mark: Mark::Image, encoding: color("c"), ..Default::default() },
            Layer { mark: Mark::Point, encoding: Encoding::default(), data_source: Some("stats".into()), ..Default::default() },
            Layer { mark: Mark::Point, encoding: color("c"), ..Default::default() },
        ];
        let chart = color("c");
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("c", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["A"]))],
        )
        .unwrap();
        // The statistic's output carries no `c`.
        let mut outputs = HashMap::new();
        outputs.insert("stats".to_string(), group_batch(vec!["B"]));
        let channel = AUXILIARY_SCALE_CHANNELS.into_iter().find(|c| c.name == "color").expect("color");
        let reads = aux_reads(&layers, &chart, channel, &rows, &outputs);
        assert_eq!(reads.iter().map(AuxRead::layer).collect::<Vec<_>>(), [2]);
    }
}
