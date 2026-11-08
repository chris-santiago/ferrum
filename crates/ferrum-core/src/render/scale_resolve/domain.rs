//! Domain computation: numeric extent union, sort application, field location.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use super::{sided_min_max_f64, LeftOutTally, SidedExtent};
use crate::render::position::AdjustedRows;
use crate::render::prepare::LayerPrepared;
use crate::render::{RenderError, RenderWarning};
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::sort::{parse_sort_field_object, SortOp, SortOrder};
use crate::spec::layer::Layer;
use crate::spec::mark::Mark;
use crate::spec::position::StackValueAxis;
use crate::transform::run::{OutputLineage, OutputLineages};
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
/// [`FINAL_OUTPUT_KEY`]), the one named output its `data_source` names, a
/// layer's own position-adjusted preview batch, or, for a member of an
/// overlay group of leaves, the batch that member draws together with the
/// [`OutputLineage`] the read's own key named before the group collapsed it
/// ([`Self::lineage_key`]) — `None` when that key's run published no
/// lineage, same as every other source's lookup.
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
    Member(&'a RecordBatch, Option<&'a OutputLineage>),
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
            LayerSource::Batch(batch) | LayerSource::Member(batch, _) => Some(batch),
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

    /// The lineage key this source names before an overlay group collapses
    /// it to [`LayerSource::Member`] ([`overlay_aux_reads`],
    /// [`overlay_raster_cell_reads`]): [`FINAL_OUTPUT_KEY`] for the chart's
    /// own data, the named output otherwise. `None` for a source that never
    /// names a key directly (a position preview's [`LayerSource::Batch`], or
    /// an already-collapsed [`LayerSource::Member`]).
    fn lineage_key(self) -> Option<&'a str> {
        match self {
            LayerSource::Chart => Some(FINAL_OUTPUT_KEY),
            LayerSource::Output(key) => Some(key),
            LayerSource::Batch(_) | LayerSource::Member(..) => None,
        }
    }
}

/// What `read()` does with `field`'s column for the off-side TALLY, once
/// located (the EXTENT always reads the column `source` locates, whichever
/// variant this is): a plain caller column, folded through the ordinary
/// content-deduplicated `tally_only` ([`Column`](Self::Column)); a
/// previewed layer's [`crate::render::position::AdjustedRows`] pair, folded
/// through [`fold_adjusted_tally`] ([`Adjusted`](Self::Adjusted)) — the
/// caller's own rows plus the rows the adjustment itself moves off the
/// side; or neither — a span read that is the adjusted layer's own stack
/// base column, which feeds the extent only, since a stack's base is not
/// caller data ([`ExtentOnly`](Self::ExtentOnly)). `Adjusted` carries its
/// own CLONE of the layer's [`AdjustedRows`] pair (cheap: an `Arc`-backed
/// `RecordBatch` and a plain reference) rather than a borrow of it, since a
/// borrow tied to [`AxisReads`]'s own (function-local) lifetime cannot
/// outlive the method call that reads it — [`Clone`], not [`Copy`], follows
/// from that; [`SeedRead`]'s own derive matches.
#[derive(Clone)]
enum TallyRead<'a> {
    Column,
    Adjusted(AdjustedRows<'a>),
    ExtentOnly,
}

/// One read of a layered axis's seed: the batch, and the fields a layer draws
/// there on the axis: the value `field`, its paired `x2`/`y2` `span`, and the
/// `opposite` positional field the ordinal null-band rule judges rows by.
/// `inheritor` is the index of the later layer whose inherited field this
/// read is, judged like that layer's own column; `None` for the bound field.
/// `tally` is [`TallyRead`] for this seed's own layer's VALUE field;
/// `span_tally` the same for `span` ([`AxisReads::span_tally`] — `Column`
/// for a caller's own `x2`/`y2`, `ExtentOnly` when `span` is the layer's own
/// stack-base column instead).
#[derive(Clone)]
struct SeedRead<'a> {
    source: LayerSource<'a>,
    tally: TallyRead<'a>,
    field: &'a str,
    span: Option<&'a str>,
    span_tally: TallyRead<'a>,
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
/// fields it draws. Each read comes from where `reads` places that layer's
/// reads on the axis ([`AxisReads`]): a layer that stacks its value there
/// reads the stack it draws, with the segment bases as its span.
fn seed_reads<'a>(
    spec: &'a ChartSpec,
    layers: &'a [Layer],
    channel: &str,
    bound: (&'a str, Option<&'a str>, Option<&'a str>),
    reads: &AxisReads<'a>,
) -> Vec<SeedRead<'a>> {
    let (field, span, opposite) = bound;
    let Some((index, establisher)) = establishing_axis_layer(spec, layers, channel) else { return Vec::new() };
    let first = if index == 0 {
        let span = reads.span(index, span);
        SeedRead { source: reads.source(index), tally: reads.tally(index), field, span, span_tally: reads.span_tally(index, span), opposite, inheritor: None }
    } else {
        let drawn = drawn_channel_slots(spec, establisher, channel);
        let span = reads.span(index, span.and(drawn.span));
        SeedRead {
            source: reads.source(index),
            tally: reads.tally(index),
            field,
            span,
            span_tally: reads.span_tally(index, span),
            opposite: opposite.and(drawn.opposite),
            inheritor: None,
        }
    };
    let mut seeds = vec![first];
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
        let source = reads.source(index);
        if seeds.iter().any(|r| r.source == source && r.field == field) {
            continue;
        }
        let span = reads.span(index, drawn.span);
        seeds.push(SeedRead {
            source,
            tally: reads.tally(index),
            field,
            span,
            span_tally: reads.span_tally(index, span),
            opposite: drawn.opposite,
            inheritor: Some(index),
        });
    }
    seeds
}

/// Where each layer's reads on one axis come from, by layer index: the batch
/// the layer draws ([`LayerSource::of`]), or, for a layer whose own position
/// preview produced one, the preview it draws ([`axis_previews`]). `base` is
/// the per-layer stack-base pairing: `Some` only for a layer whose own
/// effective position is `Stack` on this axis
/// ([`crate::render::position::layer_prepared_stack_axis`]), so a dodged or
/// jittered layer's contribution carries no span partner, as `axis_batch`'s
/// unlayered `base` already decides it.
struct AxisReads<'a> {
    sources: Vec<LayerSource<'a>>,
    base: Vec<Option<&'static str>>,
    /// Each layer's own [`AdjustedRows`] pair — `Some` exactly when
    /// `sources[i]` is a preview `Batch` (`axis_previews` builds one for
    /// every layer with a preview; `unstacked`'s layers, on a categorical
    /// axis no layer adjusts, carry none). [`Self::tally`] reads this
    /// instead of `sources[i]` for the off-side tally, so a previewed
    /// `Dodge`/`Jitter`/`Stack` layer counts the caller's own rows (plus
    /// the rows the adjustment itself moves off the side) rather than
    /// re-reading its own adjusted preview as if it were caller data.
    adjusted: Vec<Option<AdjustedRows<'a>>>,
}

impl<'a> AxisReads<'a> {
    /// Every layer read from the batch it draws, as an axis no layer adjusts
    /// on (a categorical one) reads them.
    fn unstacked(layers: &'a [Layer]) -> Self {
        let sources: Vec<LayerSource<'a>> = layers.iter().map(LayerSource::of).collect();
        let len = sources.len();
        Self { sources, base: vec![None; len], adjusted: (0..len).map(|_| None).collect() }
    }

    /// Each layer read from its own preview in `previews` where it has one,
    /// else from the batch it draws; `base` is each layer's own stack-base
    /// pairing and `adjusted` its own tally pair, both [`axis_previews`]'s
    /// own output.
    fn over(layers: &'a [Layer], previews: &'a [Option<RecordBatch>], base: Vec<Option<&'static str>>, adjusted: Vec<Option<AdjustedRows<'a>>>) -> Self {
        Self {
            sources: layers
                .iter()
                .zip(previews)
                .map(|(layer, preview)| preview.as_ref().map_or(LayerSource::of(layer), LayerSource::Batch))
                .collect(),
            base,
            adjusted,
        }
    }

    fn source(&self, layer: usize) -> LayerSource<'a> {
        self.sources[layer]
    }

    /// What the off-side tally reads for `layer`'s own VALUE field (never a
    /// span — a caller's `x2`/`y2` stays [`TallyRead::Column`] unconditionally,
    /// and a stack-base span is [`TallyRead::ExtentOnly`], both decided at the
    /// call site since this answer only ever concerns the value slot):
    /// [`TallyRead::Adjusted`] when `layer` has its own `AdjustedRows` pair
    /// (a previewed `Dodge`/`Jitter`/`Stack` layer), else
    /// [`TallyRead::Column`] (an unadjusted layer's plain caller column).
    fn tally(&self, layer: usize) -> TallyRead<'a> {
        match &self.adjusted[layer] {
            Some(rows) => TallyRead::Adjusted(rows.clone()),
            None => TallyRead::Column,
        }
    }

    /// The stack's base column for a layer that stacks on the axis: a
    /// stacked segment draws from its base to its top.
    fn base(&self, layer: usize) -> Option<&'static str> {
        self.base[layer]
    }

    /// The span a layer's seed read pairs with: `span`, its own `x2`/`y2`,
    /// else the stack's base column for a stacking layer ([`Self::base`]),
    /// the precedence an unlayered stack's axis gives them
    /// (`position::axis_batch`).
    fn span<'s>(&self, layer: usize, span: Option<&'s str>) -> Option<&'s str> {
        span.or(self.base(layer))
    }

    /// What the off-side tally reads for a SPAN field at `span` (the value
    /// field's own answer is [`Self::tally`]): [`TallyRead::ExtentOnly`]
    /// when `span` IS `layer`'s own stack-base column — the span
    /// [`Self::span`]'s `.or` fallback filled in, not a field the caller
    /// wrote — since a stack's base is not caller data; [`TallyRead::Column`]
    /// for every other span, a caller's own `x2`/`y2`.
    fn span_tally(&self, layer: usize, span: Option<&str>) -> TallyRead<'a> {
        if span.is_some() && span == self.base(layer) {
            TallyRead::ExtentOnly
        } else {
            TallyRead::Column
        }
    }
}

/// [`axis_previews`]'s per-layer result: each layer's own preview batch
/// (`None` for a layer that adjusts nothing on the axis) paired with its
/// stack-base column name (`Some` only for a layer whose effective
/// position is `Stack`, [`AxisReads::over`]'s doc) and, for the off-side
/// tally, its own [`crate::render::position::AdjustedRows`] pair.
type LayerAxisPreviews<'a> = (Vec<Option<RecordBatch>>, Vec<Option<&'static str>>, Vec<Option<crate::render::position::AdjustedRows<'a>>>);

/// The position adjustment each layer draws on the numeric axis `channel`,
/// by layer index: [`crate::render::position::layer_position_preview`] over
/// the batch the layer draws, or `None` for a layer that adjusts nothing
/// there. `Stack` previews its own stacked batch; a layer's continuous `Dodge` or
/// data-space `Jitter` previews as its adjusted rows alone, so the layer's
/// contributed batch spans exactly the positions it draws. A layer on the
/// chart's data on a facet-shared axis (`include_final`) previews the
/// all-panels batch too, panel by panel, so every panel's axis spans the
/// widest panel's adjustment. A layer whose own coord exchanges its
/// channels relative to the panel is left to
/// `scene_build::validate_layer_coord_exchange_is_servable`, and an
/// `independent_y` layer adjusts on its own y axis, not this one; neither
/// adjusts here. Refuses a facet field the panel split cannot read
/// (`position::position_preview`). `color` is the panel's one resolved
/// color scale, threaded to every layer's `Dodge` preview so its sub-band
/// order never diverges from the draw's.
fn axis_previews<'a>(
    spec: &ChartSpec,
    layers: &'a [Layer],
    channel: &str,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
    include_final: bool,
    color: Option<&crate::render::scale_resolve::ColorScale>,
) -> Result<LayerAxisPreviews<'a>, RenderError> {
    use crate::spec::position::StackValueAxis;
    let axis = match channel {
        "x" => StackValueAxis::X,
        "y" => StackValueAxis::Y,
        _ => return Ok((vec![None; layers.len()], vec![None; layers.len()], vec![None; layers.len()])),
    };
    let chart_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    let mut previews = Vec::with_capacity(layers.len());
    let mut bases = Vec::with_capacity(layers.len());
    let mut adjusted = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let exchanges = index != 0 && layer_flips(layer, spec.coord.as_ref()) != chart_flipped;
        if exchanges || (axis == StackValueAxis::Y && layer.independent_y) {
            previews.push(None);
            bases.push(None);
            adjusted.push(None);
            continue;
        }
        // Read the same way `axis_batch`'s own inline `base` check reads it
        // for the unlayered case: this layer's effective position is
        // `Stack` or it isn't, from its own post-flip encoding and
        // position — never from whether its preview happened to return
        // `Some` (a `Dodge`/`Jitter` preview is `Some` too: its own adjusted
        // rows alone, never a union with the unadjusted ones).
        let drawn = crate::render::prepare::LayerPrepared::from_chart_and_layer(spec, layer).in_its_coord();
        let base = crate::render::position::layer_prepared_stack_axis(&drawn)?
            .is_some()
            .then_some(crate::render::position::STACK_BASE_COLUMN);
        let preview =
            |batch: &RecordBatch| crate::render::position::layer_position_preview(spec, layer, axis, batch, color);
        // This layer's own drawn field on `channel`, for pairing its
        // whole-frame unadjusted batch ([`whole_frame_batch`]) with its
        // preview below — `None` for a layer whose completed encoding
        // draws nothing on this channel, which already took the `None`
        // branches above or below.
        let field = drawn_channel_slots(spec, layer, channel).value;
        let (result, layer_adjusted) = match LayerSource::of(layer) {
            LayerSource::Chart => match preview(primary_batch)? {
                None => (None, None),
                Some(rows) => {
                    // The all-panels preview of `FINAL_OUTPUT_KEY`, computed
                    // whenever the chart publishes one — not only under
                    // `include_final` (the facet-SHARED case the EXTENT
                    // concat below still gates on, unchanged): the TALLY's
                    // own `AdjustedRows` pairing needs it whenever
                    // `whole_frame_batch` picks `FINAL` over `primary_batch`,
                    // which happens on an independent facet and on no facet
                    // at all too (`field` published only in `FINAL`).
                    let final_preview = match transform_outputs.get(FINAL_OUTPUT_KEY) {
                        Some(all_panels) => preview(all_panels)?,
                        None => None,
                    };
                    let extent_result = if include_final {
                        match final_preview.clone() {
                            // `concat_by_name`, not a positional `concat_batches`: a
                            // chart-data layer's own-panel preview and the all-panels
                            // preview can disagree on dtype exactly the way two facet
                            // panels can (an Int64/Float32/Timestamp no-op `Dodge`
                            // beside a `Float64` dodged one), and a plain
                            // `concat_batches` either errors or, via `.ok()`, drops
                            // the whole layer's preview silently on any mismatch —
                            // including a genuine column-count difference, which it
                            // would otherwise align POSITIONALLY. Reconciles the
                            // same way `position::adjusted_across_panels` does; a
                            // real failure refuses instead of leaving the axis
                            // unwidened with no warning.
                            Some(all_panels) => Some(
                                crate::transform::concat::concat_by_name("position preview", vec![rows.clone(), all_panels])
                                    .map_err(RenderError::ScaleResolutionFailed)?,
                            ),
                            None => Some(rows.clone()),
                        }
                    } else {
                        Some(rows.clone())
                    };
                    // An `AdjustedRows` is a batch beside its OWN preview:
                    // when `whole_frame` is `primary_batch`, `rows` (its own
                    // preview) pairs with it directly; when `whole_frame` is
                    // `FINAL` instead, only `final_preview` (FINAL's own
                    // preview, computed above) is `FINAL`'s own — `rows` is
                    // a preview of `primary_batch`, a DIFFERENT, narrower
                    // batch (this panel's own rows), so pairing it with
                    // `whole_frame` would misalign row `i` of one against
                    // row `i` of the other. A missing `final_preview` (the
                    // adjustment failed on a later panel inside
                    // `adjusted_across_panels`) gives no pair at all, the
                    // same outcome `axis_batch`'s own handling of this case
                    // gives.
                    let tally_pair = field.and_then(|field| {
                        let whole_frame = whole_frame_batch(primary_batch, transform_outputs, field);
                        if std::ptr::eq(whole_frame, primary_batch) {
                            Some(crate::render::position::AdjustedRows { unadjusted: whole_frame, adjusted: rows, base })
                        } else {
                            final_preview.clone().map(|adjusted| crate::render::position::AdjustedRows { unadjusted: whole_frame, adjusted, base })
                        }
                    });
                    (extent_result, tally_pair)
                }
            },
            LayerSource::Output(key) => match transform_outputs.get(key) {
                Some(batch) => match preview(batch)? {
                    None => (None, None),
                    Some(rows) => {
                        // An Output-source layer's own named output IS its
                        // whole frame — there is no wider batch to prefer.
                        let tally_pair = Some(crate::render::position::AdjustedRows { unadjusted: batch, adjusted: rows.clone(), base });
                        (Some(rows), tally_pair)
                    }
                },
                None => (None, None),
            },
            // This match's scrutinee is `LayerSource::of(layer)`, which
            // returns only `Chart`/`Output` — never `Batch` (a position
            // preview is this function's own OUTPUT, built below, not an
            // input to it here) or `Member` (only `overlay_aux_reads`/
            // `overlay_raster_cell_reads` construct it). Both arms of this
            // pattern are unreachable for this scrutinee; kept, folded
            // together, to satisfy the compiler's exhaustiveness check over
            // the shared enum. A `Member`'s embedded lineage plays no role
            // in a position preview either way, so folding it with `Batch`
            // costs nothing even if it were reached.
            LayerSource::Batch(batch) | LayerSource::Member(batch, _) => (preview(batch)?, None),
        };
        previews.push(result);
        bases.push(base);
        adjusted.push(layer_adjusted);
    }
    Ok((previews, bases, adjusted))
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
/// establishing layer's ([`establishing_axis_layer`]) own rows
/// ([`layer_sort_rows`]) and the fields that layer draws on the rendered x
/// and y. So a sort orders the axis by the establishing layer's own data,
/// whichever operand's frame the chart rows hold. `None` for a spec with no
/// layers, which sorts over its one batch.
pub(in crate::render) struct SortRows<'a> {
    pub(in crate::render) batch: &'a RecordBatch,
    /// The layer's final (post-statistic) rows: read instead of `batch` for
    /// whichever field a sort actually names, ONLY when that field is the
    /// statistic's own output (an aggregate like `mean(y)`, a boxplot's
    /// `median`) rather than one of its raw inputs -- `batch` (the
    /// pre-statistic rows) and `fallback` can name entirely different
    /// fields (an explicit sort-field object's `field` is independent of
    /// whatever the layer draws on x/y), so the choice is made per field at
    /// the point a sort resolves one, in [`sort_domain_by_aggregate`], not
    /// here. `None` for an unlayered chart, which already resolves the one
    /// batch appropriate to it.
    /// Read by two choosers: `resolve_sort_batch` (the axis/legend/aux-domain
    /// sort path) and the facet panel sort
    /// (`prepare::facet_panels::sort_batch`), both of which pick per field
    /// through the shared [`batch_carrying`].
    pub(in crate::render) fallback: Option<&'a RecordBatch>,
    pub(in crate::render) x_field: Option<&'a str>,
    pub(in crate::render) y_field: Option<&'a str>,
}

/// `Ok(None)` only for an unlayered spec (or an empty layer list) -- there
/// is no establishing layer to read. A layered spec always has one
/// ([`establishing_axis_layer`] falls back to layer 0 when no layer draws
/// the channel), so this propagates a `data_source` naming no published
/// output as the same refusal [`layer_sort_rows`] raises, rather than
/// resolving to the chart's own rows.
pub(in crate::render) fn establishing_sort_rows<'a>(
    channel: &str,
    spec: &'a ChartSpec,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
    lineages: &'a OutputLineages,
    include_final: bool,
) -> Result<Option<SortRows<'a>>, RenderError> {
    let Some(layers) = spec.layers.as_deref() else { return Ok(None) };
    let Some((index, layer)) = establishing_axis_layer(spec, layers, channel) else { return Ok(None) };
    layer_sort_rows(spec, index, layer, primary_batch, transform_outputs, lineages, include_final).map(Some)
}

/// The rows a data-aware sort reads for one `layer`, found at `layer_index`
/// in `spec`'s layers: its pre-statistic rows (`prepare::pre_statistic_rows`)
/// -- what a sort naming a raw field its own statistic consumed reads --
/// paired with its final rows (`prepare::layer_final_rows`) as the fallback
/// a sort naming the statistic's own output reads instead
/// ([`SortRows::fallback`]'s doc). Shared by the establishing axis sort and
/// the facet panel sort, which reads each panel value's first drawing
/// layer. A `data_source` naming no published output refuses with the same
/// text `prepare_render_inputs` raises for that key, rather than silently
/// sorting over the chart's own rows.
pub(in crate::render) fn layer_sort_rows<'a>(
    spec: &'a ChartSpec,
    layer_index: usize,
    layer: &'a Layer,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
    lineages: &'a OutputLineages,
    include_final: bool,
) -> Result<SortRows<'a>, RenderError> {
    let batch = crate::render::prepare::pre_statistic_rows(
        layer_index,
        layer.data_source.as_deref(),
        lineages,
        primary_batch,
        transform_outputs,
        include_final,
    )?;
    let fallback =
        crate::render::prepare::layer_final_rows(layer_index, layer.data_source.as_deref(), primary_batch, transform_outputs, include_final)?;
    Ok(SortRows {
        batch,
        fallback: Some(fallback),
        x_field: drawn_channel_slots(spec, layer, "x").value,
        y_field: drawn_channel_slots(spec, layer, "y").value,
    })
}

/// The two batches an unlayered chart's data-aware sort reads: its
/// pre-statistic rows ([`OutputLineage::pre_statistic_rows`]) when its own
/// statistics lowered a step, paired with `drawn` as the rows a sort naming
/// the statistic's own output (an aggregate's result) reads instead. A
/// chart with no statistic reads `drawn` for both, since `pre_statistic_rows`
/// returns exactly the batch it is given — byte-identical to reading
/// `drawn` alone.
fn chart_sort_pair<'a>(drawn: &'a RecordBatch, lineage: Option<&'a OutputLineage>) -> (&'a RecordBatch, &'a RecordBatch) {
    (lineage.map_or(drawn, |l| l.pre_statistic_rows(drawn)), drawn)
}

/// [`SortRows`] for an unlayered spec, built from [`chart_sort_pair`]: a
/// sort naming a raw field its own statistic consumed reads `drawn`'s
/// pre-statistic rows; a sort naming the statistic's own output (`y="mean(y)"`
/// → `mean_y`) reads `drawn` itself, as the fallback. `lineage` is the
/// chart's own lineage ([`FINAL_OUTPUT_KEY`]), read by every caller the same
/// way [`AuxRead::published_origin`] does.
pub(in crate::render) fn chart_sort_rows<'a>(
    drawn: &'a RecordBatch,
    lineage: Option<&'a OutputLineage>,
    x_field: Option<&'a str>,
    y_field: Option<&'a str>,
) -> SortRows<'a> {
    let (batch, fallback) = chart_sort_pair(drawn, lineage);
    SortRows { batch, fallback: Some(fallback), x_field, y_field }
}

/// The [`SortContext`] a chart-level categorical legend's (color, shape,
/// stroke-dash) data-aware `sort=` reads: a layered spec's `lead` read (the
/// first contributing one [`aux_category_union`] returns) pairs its own
/// pre-statistic rows with `domain_batch` as the fallback
/// ([`AuxRead::pre_statistic_rows`]); an unlayered spec (`lead: None`) pairs
/// the chart's pre-statistic rows with `domain_batch` the same way
/// [`chart_sort_rows`] does for a positional axis.
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn legend_sort_context<'a>(
    category_field: &'a str,
    lead: Option<AuxRead<'a>>,
    domain_batch: &'a RecordBatch,
    transform_lineages: &'a OutputLineages,
    x_field: Option<&'a str>,
    y_field: Option<&'a str>,
    coord_flipped: bool,
) -> SortContext<'a> {
    let batch = match lead {
        Some(read) => read.pre_statistic_rows(domain_batch, transform_lineages),
        None => chart_sort_pair(domain_batch, transform_lineages.get(FINAL_OUTPUT_KEY)).0,
    };
    SortContext { category_field, batch, fallback: Some(domain_batch), x_field, y_field, coord_flipped }
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
pub(in crate::render) fn establishing_axis_layer<'a>(
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

    /// The binding this read draws by when it carries a binned color rule
    /// ([`crate::spec::encoding::EncodingSpec::color_bin`]): the binding
    /// whose column a binned color scale cuts. `None` for an unbinned
    /// binding, and for a raster image's cells, which no binding names.
    pub(in crate::render) fn binned_binding(&self) -> Option<&'a crate::spec::encoding::EncodingSpec> {
        match self.kind {
            AuxReadKind::Column { binding } => binding.color_bin().is_some().then_some(binding),
            AuxReadKind::RasterCells { .. } => None,
        }
    }

    /// The refusal of a raster image's cells under a categorical color scale
    /// ([`RenderError::RasterColorNotQuantitative`]), voiced by the image
    /// layer's public call (`mark_raster`, else `layer {i}:
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

    /// The batch a column read's values are read from for a domain: the
    /// batch its layer draws, or, for a read of the chart's data on a
    /// faceted chart, the all-panels batch
    /// ([`super::shared_categorical_batch`]). `None` for a raster image's
    /// cells and when that batch lacks the field.
    fn column_batch<'b>(
        &self,
        primary_batch: &'b RecordBatch,
        transform_outputs: &'b HashMap<String, RecordBatch>,
        facet_shared: bool,
    ) -> Option<&'b RecordBatch>
    where
        'a: 'b,
    {
        if self.is_raster_cells() {
            return None;
        }
        let batch = match self.source {
            LayerSource::Chart => {
                super::shared_categorical_batch(primary_batch, self.field, transform_outputs, facet_shared)
            }
            LayerSource::Output(_) | LayerSource::Batch(_) | LayerSource::Member(..) => {
                self.source.batch(primary_batch, transform_outputs)?
            }
        };
        batch.column_by_name(self.field).map(|_| batch)
    }

    /// The lineage of the output this read draws: a `Member`'s own carried
    /// lineage (resolved once, against the member's own run, before the
    /// overlay group collapsed its source), else `transform_lineages`'s
    /// entry for the key this source names ([`LayerSource::lineage_key`]).
    /// `None` for a [`LayerSource::Batch`] (a position preview; never this
    /// read's own source) and for a `Member` whose run published none.
    fn lineage<'b>(&self, transform_lineages: &'b OutputLineages) -> Option<&'b OutputLineage>
    where
        'a: 'b,
    {
        match self.source {
            LayerSource::Chart | LayerSource::Output(_) => transform_lineages.get(self.source.lineage_key()?),
            LayerSource::Member(_, lineage) => lineage,
            LayerSource::Batch(_) => None,
        }
    }

    /// The rows a bin-grouped statistic consumed for this read: what
    /// [`Self::lineage`] recorded (`OutputLineage::binned_rows`; the chart
    /// rows' for a layer with no `data_source`). `None` when no statistic of
    /// that output's own grouped the read's field, or `lineage` has none.
    fn published_origin<'b>(&self, transform_lineages: &'b OutputLineages) -> Option<&'b RecordBatch>
    where
        'a: 'b,
    {
        self.lineage(transform_lineages)?.binned_rows(self.field)
    }

    /// The rows a data-aware legend sort reads for this read's own value,
    /// when the sort's field is this read's statistic's raw input rather
    /// than its aggregated output: [`Self::lineage`], read through
    /// `OutputLineage::pre_statistic_rows` instead of `binned_rows`.
    /// `fallback` (the read's own, already-drawn batch) when `lineage` has
    /// none — a `Member` reads only its own carried lineage here, never the
    /// `transform_lineages` map a caller passes for the whole group (which
    /// `build_overlay_auxiliary_scales` fills with the group's first
    /// member's lineages, not necessarily this read's own).
    fn pre_statistic_rows<'b>(&self, fallback: &'b RecordBatch, transform_lineages: &'b OutputLineages) -> &'b RecordBatch
    where
        'a: 'b,
    {
        self.lineage(transform_lineages).map_or(fallback, |lineage| lineage.pre_statistic_rows(fallback))
    }

    /// The batch a binned color's edges read this column read's values from:
    /// the rows its bin-grouped statistic consumed ([`Self::published_origin`]),
    /// which are the same before and after the statistic groups by the edges,
    /// else the batch it draws ([`Self::column_batch`]), which no bin-grouped
    /// step produced. `None` for a raster image's cells and when that batch
    /// lacks the field.
    fn binned_sample_batch<'b>(
        &self,
        primary_batch: &'b RecordBatch,
        transform_outputs: &'b HashMap<String, RecordBatch>,
        transform_lineages: &'b OutputLineages,
        facet_shared: bool,
    ) -> Option<&'b RecordBatch>
    where
        'a: 'b,
    {
        if self.is_raster_cells() {
            return None;
        }
        self.published_origin(transform_lineages)
            .filter(|origin| origin.column_by_name(self.field).is_some())
            .or_else(|| self.column_batch(primary_batch, transform_outputs, facet_shared))
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
    /// data combined with the all-panels batch per `policy`), or a raster
    /// image's cells' finite extent as the image draws it
    /// ([`crate::render::marks::image::raster_cell_domain`]). `None` when
    /// there is nothing to read (a batch lacking the field, a raster with no
    /// finite cell).
    fn extent(
        &self,
        primary_batch: &RecordBatch,
        transform_outputs: &HashMap<String, RecordBatch>,
        facet_shared: bool,
        policy: FacetExtentPolicy,
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
                if facet_shared && self.source == LayerSource::Chart {
                    if let FacetExtentPolicy::GlobalAlone = policy {
                        // The all-panels column ALONE is
                        // the shared extent for a numeric auxiliary channel —
                        // not `union_panel_with_global_extent`'s panel-union-
                        // global, which fabricates a value for an all-null
                        // panel via its own `(0.0, 1.0)` default. Falls back
                        // to the panel's own read when the global batch or
                        // the field is absent.
                        if let Some(global_col) = transform_outputs.get(FINAL_OUTPUT_KEY).and_then(|b| b.column_by_name(self.field)) {
                            let global_extent = extent(global_col.as_ref(), self.field)?;
                            return Ok(Some(ReadExtent {
                                extent: global_extent,
                                has_values: global_col.null_count() != global_col.len(),
                            }));
                        }
                    }
                }
                let mut read_extent = extent(located.col, self.field)?;
                if facet_shared && self.source == LayerSource::Chart && policy == FacetExtentPolicy::UnionWithPanel {
                    read_extent = super::union_panel_with_global_extent(read_extent, self.field, transform_outputs);
                }
                Ok(Some(ReadExtent { extent: read_extent, has_values: located.col.null_count() != located.col.len() }))
            }
        }
    }
}

/// How a faceted `Chart`-source read's panel extent relates to the all-panels
/// (`FINAL_OUTPUT_KEY`) one, when both are available.
/// Continuous color keeps unioning the two
/// (`union_panel_with_global_extent`'s original rule — color has no `Log`
/// channel to drop or warn on, and its own all-null default stays an
/// invented `(0.0, 1.0)` colorbar, unchanged). Every numeric auxiliary
/// channel ([`sided_aux_numeric_union`]) reads the all-panels column ALONE,
/// so an all-null faceted column draws no scale and no legend, matching the
/// same column's unfaceted render.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::render) enum FacetExtentPolicy {
    UnionWithPanel,
    GlobalAlone,
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
/// ([`crate::render::drawn::DrawFacts::draws`]) and its batch carries the
/// column it binds there. The column gate (`prepare::validate_layer_columns`)
/// has already refused every drawn binding whose column a layer's batch
/// lacks, but for an inherited one a statistic's output drops, which that
/// layer draws without: so a layer whose batch lacks the column is exactly a
/// layer that does not draw it. A layer binding the channel itself draws its
/// own field; a layer binding nothing there draws the chart-level binding,
/// which `prepare` gives a layer on the chart's data (`inherit_from`) and a
/// layer on its own output (`inherit_non_positional`) alike, unless its mark
/// draws only a binding it owns ([`Mark::draws_inherited`]: a text's color),
/// it is a line yielding an inherited width to one it owns, or it is a
/// segment with no arrowhead reading `fill_opacity`. The same layer-to-batch
/// rule the positional union reads by ([`seed_reads`]), without its
/// positional-only skip of `data_source` layers.
///
/// A raster image draws `color` through its cells when its binding names the
/// field they hold ([`crate::render::marks::image::image_draws_color`]): its
/// read is the cells of its Raster batch ([`AuxReadKind::RasterCells`]).
///
/// The first read is the establishing layer's: the first layer that draws
/// the channel, layer 0 when it does, the layer whose binding the rendering
/// encoding carries (`prepare`'s `establishing_encoding`). Empty when no layer
/// draws the channel, which then gets no scale and no legend. The column gate
/// judges a color's binned reads ([`AuxRead::binned_binding`]) before any
/// scale reads them.
pub(in crate::render) fn aux_reads<'a>(
    layers: &'a [Layer],
    chart_encoding: &'a crate::spec::encoding::Encoding,
    channel: crate::spec::encoding::AuxiliaryScaleChannel,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Vec<AuxRead<'a>> {
    use crate::spec::layer::InheritedChannel;
    let mut reads: Vec<AuxRead<'a>> = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let drawn = channel.drawn_as();
        // A layer's own binding, or, when its mark takes an inherited
        // binding of the channel ([`Mark::draws_inherited`]), the one Python
        // listed on it or the chart's.
        let Some(binding) = channel.binding(&layer.encoding).as_ref().or(channel.binding(chart_encoding).as_ref())
        else {
            continue;
        };
        let field = binding.field.as_str();
        let source = LayerSource::of(layer);
        // A raster image's cells, read only for `color`: the one field
        // [`crate::render::drawn::DrawFacts::draws`] needs to tell a
        // drawing raster image from one whose binding names another field.
        let cells = (drawn == InheritedChannel::Color)
            .then(|| source.batch(primary_batch, transform_outputs).and_then(crate::render::marks::image::raster_value_field))
            .flatten();
        let facts = crate::render::drawn::DrawFacts::of_layer(layer, chart_encoding, cells);
        // A text yields no read for a color it does not own (declined); a
        // contrast-ink text's own color reads the scale without painting a
        // hue from it; a yielding line's inherited width and an arrowless
        // segment's `fill_opacity` are no read either — all judged by the
        // one rule, so none of them establishes the scale, widens its
        // domain, or contributes to its legend
        // (`undrawn_channel_warnings` separately counts each as
        // bound-but-undrawn).
        if facts.draws(drawn).is_err() {
            continue;
        }
        if drawn == InheritedChannel::Color {
            if let Some(cmap) = raster_cells_drawn(layer, chart_encoding, cells) {
                let kind = AuxReadKind::RasterCells { cmap, entry: layer.entry.as_deref() };
                reads.push(AuxRead { layer: index, mark: layer.mark, source, field, kind });
                continue;
            }
        }
        if source.locate(field, primary_batch, transform_outputs).is_none() {
            continue;
        }
        reads.push(AuxRead { layer: index, mark: layer.mark, source, field, kind: AuxReadKind::Column { binding } });
    }
    reads
}

/// The colormap a raster image `layer` draws its cells in, when it draws
/// color through them ([`crate::render::marks::image::image_draws_color`]):
/// `Some(cmap)` (itself `None` when the layer names no colormap); `None` for
/// every other layer. `cells` is the field this layer's cells hold, already
/// read once from its batch by the one caller, `aux_reads`, which also
/// builds its `DrawFacts` from it — so the batch is read once per color
/// read, not once here and once there for the same layer.
fn raster_cells_drawn<'a>(
    layer: &'a Layer,
    chart_encoding: &crate::spec::encoding::Encoding,
    cells: Option<&'a str>,
) -> Option<Option<&'a str>> {
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
                    let lineage = read.source.lineage_key().and_then(|key| input.transform_lineages.get(key));
                    reads.push(AuxRead { layer: first_layer + read.layer, source: LayerSource::Member(batch, lineage), ..read });
                }
                first_layer += layers.len();
            }
            None => {
                first_layer += 1;
                let Some(binding) = channel.binding(input.encoding).as_ref() else { continue };
                let field = binding.field.as_str();
                let located =
                    locate_field(field, final_batch).ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
                let lineage = input.transform_lineages.get(FINAL_OUTPUT_KEY);
                reads.push(AuxRead {
                    layer: first_layer - 1,
                    mark: input.spec.mark,
                    source: LayerSource::Member(located.batch, lineage),
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
            let lineage = read.source.lineage_key().and_then(|key| input.transform_lineages.get(key));
            reads.push(AuxRead { layer: first_layer + read.layer, source: LayerSource::Member(batch, lineage), ..read });
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
/// than left out. The color scale admits through [`admitted_color_reads`].
pub(in crate::render) fn admitted_aux_reads<'a>(
    reads: &[AuxRead<'a>],
    channel: &str,
    scale_reads_categorical: bool,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<AuxRead<'a>>, RenderError> {
    let reads_categorical = |read: &AuxRead<'a>| read.reads_categorical(primary_batch, transform_outputs);
    let reads_kind = ReadsKind { scale_reads_categorical, reads_categorical: &reads_categorical };
    admit_reads(reads, channel, reads_kind, None, primary_batch, transform_outputs, warnings)
}

/// How a color scale classifies the reads it admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::render) enum ColorReads {
    /// A binned color's scale: it bins numbers whatever type a binding
    /// declares, so each read is classified by its column's dtype alone
    /// ([`dtype_reads_categorical`]). A layer binding the field `ordinal`
    /// (Vega-Lite's binned ordinal legend) adds its values; a layer whose
    /// column holds no numbers is excluded.
    Binned,
    /// A mark that groups color (an area) reads every column as categories,
    /// so no read conflicts with the scale's kind.
    Grouped,
    /// [`admitted_aux_reads`]' rule: each read classified by
    /// [`column_reads_categorical`] against the scale's kind.
    Kind {
        /// The scale reads its columns as categories.
        categorical: bool,
    },
}

/// The reads the color scale `establishing` decides admits, by `kind`
/// ([`ColorReads`]), under [`admitted_aux_reads`]' establishing-scale rule.
/// The color scale also carries its establishing binding's bin rule: a later
/// read whose own binding is binned ([`AuxRead::binned_binding`]) with a scale
/// other than `establishing`'s is excluded too, with the same one
/// [`RenderWarning::LayerScaleExcluded`], decided in the same pass so a read
/// warns at most once. Its marks still draw through the panel's scale; a
/// read with no bin rule simply joins.
pub(in crate::render) fn admitted_color_reads<'a>(
    reads: &[AuxRead<'a>],
    establishing: &crate::spec::encoding::EncodingSpec,
    kind: ColorReads,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<AuxRead<'a>>, RenderError> {
    let column_kind = |read: &AuxRead<'a>| read.reads_categorical(primary_batch, transform_outputs);
    let dtype_kind = |read: &AuxRead<'a>| {
        read.locate(primary_batch, transform_outputs)
            .is_some_and(|located| dtype_reads_categorical(located.col.data_type()))
    };
    let every_read_categorical = |_: &AuxRead<'a>| true;
    let reads_kind = match kind {
        ColorReads::Binned => ReadsKind { scale_reads_categorical: false, reads_categorical: &dtype_kind },
        ColorReads::Grouped => ReadsKind { scale_reads_categorical: true, reads_categorical: &every_read_categorical },
        ColorReads::Kind { categorical } => ReadsKind { scale_reads_categorical: categorical, reads_categorical: &column_kind },
    };
    admit_reads(reads, "color", reads_kind, Some(establishing), primary_batch, transform_outputs, warnings)
}

/// The kind a scale reads its columns as, and how a read is classified.
struct ReadsKind<'r, 'a> {
    scale_reads_categorical: bool,
    reads_categorical: &'r dyn Fn(&AuxRead<'a>) -> bool,
}

/// The admission rule [`admitted_aux_reads`] and [`admitted_color_reads`]
/// share: the lead always, then each read `reads_kind` classifies as the
/// scale reads and, under a color scale's `bin_rule` binding, whose own bin
/// rule (if any) is that binding's.
fn admit_reads<'a>(
    reads: &[AuxRead<'a>],
    channel: &str,
    reads_kind: ReadsKind<'_, 'a>,
    bin_rule: Option<&crate::spec::encoding::EncodingSpec>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<AuxRead<'a>>, RenderError> {
    let ReadsKind { scale_reads_categorical, reads_categorical } = reads_kind;
    let Some((lead, rest)) = reads.split_first() else { return Ok(Vec::new()) };
    let mut admitted = vec![*lead];
    for read in rest {
        let conflicts = reads_categorical(read) != scale_reads_categorical
            && (read.is_raster_cells() || read.locate(primary_batch, transform_outputs).is_some());
        if conflicts && scale_reads_categorical {
            let reason = crate::render::RasterColorReason::CategoricalScale { layer: lead.layer, field: lead.field.to_string() };
            if let Some(refusal) = read.raster_color_refusal(reason) {
                return Err(refusal);
            }
        }
        let other_bin_rule = bin_rule.is_some_and(|establishing| {
            read.binned_binding().is_some_and(|binding| binding.scale != establishing.scale)
        });
        if conflicts || other_bin_rule {
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
    policy: FacetExtentPolicy,
    extent: impl Fn(&dyn Array, &str) -> Result<(f64, f64), RenderError>,
) -> Result<Option<(f64, f64)>, RenderError> {
    let mut union: Option<(f64, f64)> = None;
    let mut first: Option<(f64, f64)> = None;
    for read in reads {
        let Some(ReadExtent { extent: read_extent, has_values }) =
            read.extent(primary_batch, transform_outputs, facet_shared, policy, &extent)?
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

/// [`aux_numeric_union`], but folding each read through [`sided_min_max_f64`]
/// and returning the union's `(lo, hi)` alongside a content-deduplicated
/// [`LeftOutTally`] of every read —
/// one tally entry per DISTINCT column, so two layers drawing the same
/// column count its left-out rows once, not twice, instead of the plain sum
/// a `(f64, f64)` extent closure would discard the accounting into. A
/// layered `Log` channel (size, or a numeric auxiliary channel) reads its
/// data through this so its left-out rows are counted once, over the WHOLE
/// layer set: the same whole-frame accounting x/y's
/// [`numeric_domain_union_over`] gives a flat chart's layers, reused here
/// rather than rebuilt.
///
/// `side: None` behaves exactly as `aux_numeric_union` itself (the tally
/// stores nothing, since `sided_min_max_f64(col, None)` never leaves a row
/// out).
///
/// The per-read tallying is captured through a `RefCell` so the extent
/// closure `aux_numeric_union` calls can stay an `Fn` (it may be invoked more
/// than once per read, once per panel on a faceted union) while still
/// feeding this function's tally as each call happens.
pub(in crate::render) fn sided_aux_numeric_union(
    reads: &[AuxRead<'_>],
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
    side: Option<crate::scale::log::LogSide>,
    channel: &'static str,
) -> Result<Option<(SidedExtent, LeftOutTally)>, RenderError> {
    let tally = std::cell::RefCell::new(LeftOutTally::new());
    let extent = |col: &dyn Array, field: &str| {
        let e = sided_min_max_f64(col, side).map_err(|_| RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?}", col.data_type()),
            context: Some(channel),
        })?;
        tally.borrow_mut().count(col, e);
        Ok((e.lo, e.hi))
    };
    let Some((lo, hi)) = aux_numeric_union(reads, primary_batch, transform_outputs, facet_shared, FacetExtentPolicy::GlobalAlone, extent)? else {
        return Ok(None);
    };
    Ok(Some((
        SidedExtent { lo, hi, left_out: 0, left_out_zero: false, left_out_opposite: None },
        tally.into_inner(),
    )))
}

/// The categorical domain of a chart-level color, shape or stroke-dash on a
/// layered spec: each read's distinct values in first-appearance order,
/// unioned in layer order, as [`positional_category_union`] unions an axis's
/// categories. A read of
/// the chart's data takes the all-panels batch on a faceted chart
/// ([`super::shared_categorical_batch`]). Returns the domain, the first
/// contributing batch and that batch's read, whose own pre-statistic rows a
/// data-aware `sort=` reads instead of aggregating over the domain batch
/// ([`legend_sort_context`]); `None` for no reads (an unlayered spec, which
/// reads its one batch instead).
#[allow(clippy::type_complexity)]
pub(in crate::render) fn aux_category_union<'b>(
    reads: &[AuxRead<'b>],
    primary_batch: &'b RecordBatch,
    transform_outputs: &'b HashMap<String, RecordBatch>,
    facet_shared: bool,
) -> Result<Option<(Vec<String>, &'b RecordBatch, AuxRead<'b>)>, RenderError> {
    let mut union: Option<(Vec<String>, &'b RecordBatch, AuxRead<'b>)> = None;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for read in reads {
        let Some(batch) = read.column_batch(primary_batch, transform_outputs, facet_shared) else { continue };
        let values = super::distinct_values_in_order(batch, read.field)?;
        let (domain, ..) = union.get_or_insert_with(|| (Vec::new(), batch, *read));
        for value in values {
            if seen.insert(value.clone()) {
                domain.push(value);
            }
        }
    }
    Ok(union)
}

/// The values a layered spec's numeric `color` domain spans, pooled over
/// `reads` in layer order: each column read's values in the batch it draws
/// (on a faceted chart, a read of the chart's data reads the all-panels
/// batch, as [`aux_category_union`] does), and each raster image's cells.
/// Only finite values: a binned color's edges and a quantile color's
/// cut-points are taken from them.
pub(in crate::render) fn aux_numeric_values(
    reads: &[AuxRead<'_>],
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
) -> Vec<f64> {
    let mut values = Vec::new();
    for read in reads {
        if let Some(raster) = read.raster_batch(primary_batch, transform_outputs) {
            values.extend(crate::render::marks::image::raster_cell_values(raster));
        } else if let Some(batch) = read.column_batch(primary_batch, transform_outputs, facet_shared) {
            values.extend(crate::render::arrow_cast::col_as_f64(batch, read.field).unwrap_or_default().into_iter().flatten());
        }
    }
    values.retain(|v| v.is_finite());
    values
}

/// The values a binned color's edges are cut from, one entry per distinct
/// source ([`binned_color_sources`]), pooled.
pub(in crate::render) fn binned_color_values<'a: 'b, 'b>(
    reads: &[AuxRead<'a>],
    primary_batch: &'b RecordBatch,
    transform_outputs: &'b HashMap<String, RecordBatch>,
    transform_lineages: &'b OutputLineages,
    facet_shared: bool,
) -> Vec<f64> {
    binned_color_sources(reads, primary_batch, transform_outputs, transform_lineages, facet_shared).into_iter().flatten().collect()
}

/// The distinct sources of the values a binned color's edges are cut from:
/// [`aux_numeric_values`] with each column read's values taken from the rows
/// its bin-grouped statistic consumes ([`AuxRead::binned_sample_batch`]), the
/// same on the run that groups by the edges as on the run that cuts them.
/// Each source is its finite values, sorted. Reads whose values are the same
/// multiset (a composite mark's sub-layers over one input, siblings or
/// overlay members over the same rows, and so two frames whose column holds
/// the same values) are one source ([`add_distinct_source`]), so the edges
/// are the `Bin` transform's over the data however many layers read it. A
/// raster image's cells are its own source and always count.
pub(in crate::render) fn binned_color_sources<'a: 'b, 'b>(
    reads: &[AuxRead<'a>],
    primary_batch: &'b RecordBatch,
    transform_outputs: &'b HashMap<String, RecordBatch>,
    transform_lineages: &'b OutputLineages,
    facet_shared: bool,
) -> Vec<Vec<f64>> {
    let mut sources: Vec<Vec<f64>> = Vec::new();
    for read in reads {
        if let Some(raster) = read.raster_batch(primary_batch, transform_outputs) {
            sources.push(finite_sorted(crate::render::marks::image::raster_cell_values(raster)));
            continue;
        }
        let Some(batch) = read.binned_sample_batch(primary_batch, transform_outputs, transform_lineages, facet_shared) else { continue };
        let values = crate::render::arrow_cast::col_as_f64(batch, read.field).unwrap_or_default().into_iter().flatten();
        add_distinct_source(&mut sources, finite_sorted(values));
    }
    sources
}

/// Add `source` (finite values, sorted) to `sources` unless an equal one is
/// already there: one rule for counting each source of a binned color's
/// values once, within a chart and across a composite's leaves.
pub(in crate::render) fn add_distinct_source(sources: &mut Vec<Vec<f64>>, source: Vec<f64>) {
    if !sources.contains(&source) {
        sources.push(source);
    }
}

/// `values`' finite values, sorted.
pub(in crate::render) fn finite_sorted(values: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut values: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
    values.sort_by(f64::total_cmp);
    values
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
/// silently).
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
/// Shared by every ordinal auxiliary channel's explicit-domain path — color
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
    /// The layer's final (post-statistic) rows, read instead of `batch` for
    /// a sort's value field only when `batch` does not carry it
    /// ([`SortRows::fallback`]'s doc). `None` wherever `batch` is already
    /// the one rows a sort over this context could ever need (every
    /// context not built from a layered chart's [`SortRows`]).
    pub(in crate::render) fallback: Option<&'a RecordBatch>,
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

/// Whether `ctx.category_field`'s column in `ctx.batch` is numeric (`Int*`,
/// `UInt*`, `Float*`): a sorted ordinal domain over such a column orders by
/// value, not by its stringified text ([`apply_sort_to_domain`]'s doc).
/// `false` when the column is absent from `ctx.batch` (a sort-field object's
/// own domain-batch fallback never needs this; a plain `"ascending"`/
/// `"descending"` always resolves `ctx.category_field` against `ctx.batch`
/// itself) or is `Utf8`/`LargeUtf8`/`Categorical`/any other non-numeric
/// dtype.
fn numeric_category_column(ctx: &SortContext<'_>) -> bool {
    ctx.batch
        .column_by_name(ctx.category_field)
        .is_some_and(|c| crate::render::arrow_cast::is_numeric(c.data_type()))
}

/// One domain entry's sort key under a numeric-valued sort: `(tier, value,
/// text)`. Tier 0 is every entry that parses as a finite-or-infinite `f64`,
/// ordered by `value`; tier 1 is an entry that parses as NaN (the ordinal
/// reader's own NaN formatting, `"NaN"`); tier 2 is an entry that does not
/// parse as a number at all (an explicit scale domain can name one that
/// isn't). `text` breaks a tie within a tier — including putting multiple
/// non-parsing entries in their own text order, the one case plain value
/// comparison cannot order.
fn numeric_domain_sort_key(s: &str) -> (u8, f64, &str) {
    match s.parse::<f64>() {
        Ok(v) if v.is_nan() => (1, 0.0, s),
        Ok(v) => (0, v, s),
        Err(_) => (2, 0.0, s),
    }
}

/// Ascending comparator for [`numeric_domain_sort_key`]: every number (finite
/// or infinite) orders by value, the NaN tier lands after every one of them
/// — including `+inf` — and before the non-parsing tier. `"descending"` sorts
/// ascending through this and then reverses the whole result, so the tier
/// order flips too — the non-parsing tier comes first, then NaN, then every
/// number last. NaN stays sandwiched between the other two tiers in both
/// directions; which tier comes first does not.
fn numeric_domain_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (ta, va, sa) = numeric_domain_sort_key(a);
    let (tb, vb, sb) = numeric_domain_sort_key(b);
    ta.cmp(&tb)
        .then_with(|| va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal))
        .then_with(|| sa.cmp(sb))
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
///
/// `"ascending"`/`"descending"` order a NUMERIC category column (`Int*`,
/// `UInt*`, `Float*`) by value rather than as text, so `[-3, 1, 10, 2]`
/// reads `-3, 1, 2, 10` rather than `-3, 1, 10, 2` — every categorical
/// door (facet panels, ordinal axes, categorical legends) shares this one
/// function, so a sorted numeric facet, axis and legend all order the same
/// way. A `Utf8`/`Categorical` column (or one the explicit-domain fallback
/// batch holds no column for) keeps plain text order, byte-identical.
pub(in crate::render) fn apply_sort_to_domain(
    domain: &mut Vec<String>,
    sort: Option<&serde_json::Value>,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    match sort {
        None => {}
        Some(serde_json::Value::String(s)) if s == "ascending" => {
            if numeric_category_column(ctx) {
                domain.sort_by(|a, b| numeric_domain_cmp(a, b));
            } else {
                domain.sort_unstable();
            }
        }
        Some(serde_json::Value::String(s)) if s == "descending" => {
            if numeric_category_column(ctx) {
                domain.sort_by(|a, b| numeric_domain_cmp(a, b));
                domain.reverse();
            } else {
                domain.sort_unstable_by(|a, b| b.cmp(a));
            }
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
/// **The shorthand names an AUTHORED channel** (#95). `sort="-y"` means
/// "descending by whatever the user's own
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
    // #95 (AUTHORED semantics): `channel` is the literal substring
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

/// Which of two batches carries `field`: `batch` when it does, else
/// `fallback` when `fallback` carries it instead, else `batch` (so a field
/// nowhere still refuses in `batch`'s own words). Shared by
/// [`resolve_sort_batch`] (the axis/legend/aux-domain sort path) and the
/// facet panel sort (`prepare::facet_panels::sort_batch`), both of which
/// read a value's category key from this SAME returned batch, so a row
/// never splits its key and its value across two batches that could
/// disagree about which rows exist or what order they are in.
pub(in crate::render) fn batch_carrying<'a>(batch: &'a RecordBatch, fallback: Option<&'a RecordBatch>, field: &str) -> &'a RecordBatch {
    if batch.schema().column_with_name(field).is_some() {
        return batch;
    }
    match fallback {
        Some(fallback) if fallback.schema().column_with_name(field).is_some() => fallback,
        _ => batch,
    }
}

/// Which of `ctx`'s two batches to read `value_field` -- and, for row
/// alignment, the matching category column -- from: [`batch_carrying`],
/// except for `SortOp::Count`, which never reads a value field and so
/// always stays on `ctx.batch`, the pre-statistic grain -- counting rows
/// at the aggregated grain would count groups, not observations.
fn resolve_sort_batch<'a>(ctx: &SortContext<'a>, value_field: &str, op: SortOp) -> &'a RecordBatch {
    if matches!(op, SortOp::Count) {
        return ctx.batch;
    }
    batch_carrying(ctx.batch, ctx.fallback, value_field)
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
    let batch = resolve_sort_batch(ctx, value_field, op);
    let categories = match crate::render::arrow_cast::col_as_str(batch, ctx.category_field) {
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
        match crate::render::arrow_cast::col_as_f64(batch, value_field) {
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
/// layer on this channel — or, for a line, would a declared
/// `rise_from_zero` riser rise from it? The eligibility half of the
/// zero-anchor rule (T5r), "a mark whose style anchors its value at
/// zero", extracted from `numeric_domain_union`'s
/// union loop so the concept is nameable and directly unit-testable rather
/// than reachable only end-to-end through a 285-line function.
/// `numeric_domain_union` owns the *arithmetic* (the two guarded
/// assignments that widen the resolved extent to zero); this owns the
/// *question*.
///
/// `channel`/`layer_fields` are read in the channel's own POST-flip
/// coordinate space — the caller normalizes the layer's encoding via
/// `Encoding::flip_positional` before deriving `layer_fields`, so both sides
/// of every read below live in one space. `value_field_contributed` says
/// whether the layer's value-slot field actually resolved to data in some
/// batch, so a merely-DECLARED field (wrong `data_source`, not loaded yet)
/// cannot force a zero-anchor on the strength of its name alone.
///
/// The bar terms mirror the single-chart injection's guards
/// (`_spec_build.py::_build_encoding_specs`'s zero-anchor block), not just
/// its arithmetic; a declared line's riser reuses the same four-term shape
/// with its own value-axis orientation
/// ([`crate::render::marks::curve::resolve_curve_axis`], the one rule the
/// builder's own `DrawCtx::series_curve_axis` reads too, so the two can
/// never disagree) and its own eligibility ([`mark_wants_zero_anchor`]):
///   (a) the value axis there — the anchor is the EXTENT axis only, never
///       the POSITION axis. A bar's value channel follows its orientation
///       ([`bar_value_on_x`]): `y`, or `x` when the bar is horizontal (a
///       categorical `y`, a native horizontal bar), and the other one under
///       `CoordFlip`, which the caller's `coord_flipped`/`encoding`
///       normalization already applies. Dropping this guard is what fired
///       the anchor on a histogram bar's `x` (bin-range position) channel
///       too, squeezing `displot(kind="hist")` bars into a fraction of the
///       panel when the data doesn't straddle zero. A floating horizontal
///       bar such as `_explanation.py::desugar_shap_waterfall`'s (`x=x0`,
///       `x2=x1`, `y=feature`) has its value on `x` and is excluded by guard
///       (b): its value channel carries `x2`. A line's value channel is the
///       series' own non-advancing axis (`resolve_curve_axis`'s answer) —
///       the observation axis it advances along is its POSITION axis and is
///       never anchored, the line sibling of the bar's bin-range guard.
///   (b) the value channel's span partner stays unbound there — when
///       this layer's OWN value channel carries a paired span field (`y2`, or `x2` on an x value
///       channel — `layer_fields[1]`), the extent is taken literally from
///       `[value, span]` and force-anchoring to zero would distort a domain
///       that was never meant to touch zero. `desugar_shap_waterfall` is the
///       concrete horizontal instance (its value channel carries `x2`);
///       `y2_bound_bar_layer_never_zero_extends` below pins the vertical one.
///   (c) the layer's own opt-out ([`mark_wants_zero_anchor`]) —
///       `layer.mark_style.zero == Some(false)`, the per-layer mirror of
///       `mark_bar(zero=False)` / `_mark_zero`
///       (`spec::mark_style::MarkKwargsSpec::zero`, T5r). Live and
///       user-reachable today via the generic per-layer `mark=` passthrough
///       on layered figure functions — e.g. `fm.catplot(..., kind="bar",
///       mark={"bar": {"zero": False}})` resolves the value axis to the raw
///       data range (measured: `58..61.5`, not `0..61.5`) — see that field's
///       doc comment for the plumbing and the one path that stays inert. A
///       line's own opt-IN is `rise_from_zero == Some(true)`
///       (`spec::mark_style::MarkKwargsSpec::rise_from_zero`): unlike a
///       bar's default-on opt-out, an undeclared line never anchors.
///
/// The Python rule's remaining guard (`_y_is_categorical`) needs no mirror:
/// `numeric_domain_union` is only ever called for a quantitative/temporal
/// axis (`build_axis_scale`'s ordinal/nominal branch never reaches here), so
/// an ordinal position channel is excluded structurally, not by an extra
/// flag.
fn layer_wants_zero_anchor(
    layer: &Layer,
    channel: &str,
    value_on_x: bool,
    layer_fields: [Option<&str>; 2],
    value_field_contributed: bool,
) -> bool {
    let is_value_channel = if value_on_x { channel == "x" } else { channel == "y" };
    is_value_channel
        && mark_wants_zero_anchor(layer.mark, layer.mark_style.as_ref())
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

/// Whether a line draws its value along x, in the coordinates it renders
/// in: builds `layer` exactly as the draw pass does
/// (`LayerPrepared::from_chart_and_layer(..).in_its_coord()`, the same
/// construction [`axis_previews`] already runs) and reads its orientation
/// off that prepared layer's own post-flip declaration and its coord's real
/// flip — the same two facts
/// [`crate::render::draw::DrawCtx::series_curve_axis`] reads off the drawn
/// batch — plus its declared stack axis
/// ([`crate::render::position::layer_prepared_stack_axis`], read ahead of
/// any batch), which agrees with the builder's own `apply_stack`-stamped
/// reading for every stack `apply_stack` actually cumulates, through the
/// one shared rule ([`crate::render::marks::curve::resolve_curve_axis`])
/// both readers call.
fn curve_value_on_x(spec: &ChartSpec, layer: &Layer) -> Result<bool, RenderError> {
    use crate::render::marks::curve::{resolve_curve_axis, CurveAxis};
    let drawn = crate::render::prepare::LayerPrepared::from_chart_and_layer(spec, layer).in_its_coord();
    let coord_flipped = crate::spec::coord::flips_positional_channels(drawn.coord.as_ref());
    let stack_on_x = crate::render::position::layer_prepared_stack_axis(&drawn)? == Some(StackValueAxis::X);
    Ok(resolve_curve_axis(drawn.mark_style.and_then(|m| m.value_axis), coord_flipped, stack_on_x) == CurveAxis::Y)
}

/// Whether a mark keeps its style's zero anchor on its value channel: a bar
/// whose own `mark_bar(zero=False)` (`MarkKwargsSpec::zero`) has not opted
/// out, or a line whose own `rise_from_zero` (`mark_line`-only, set only by
/// `displot(kind="ecdf")`'s step line) opted in. Every other mark never
/// anchors.
fn mark_wants_zero_anchor(
    mark: Mark,
    mark_style: Option<&crate::spec::mark_style::MarkKwargsSpec>,
) -> bool {
    match mark {
        Mark::Bar => mark_style.and_then(|m| m.zero).unwrap_or(true),
        Mark::Line => mark_style.is_some_and(|m| m.rise_from_zero == Some(true)),
        _ => false,
    }
}

/// One `Mark::Text` row's physical ink requirement on a continuous
/// `channel`, before a scale's own range sign decides which end ("start"
/// or "end") each side names
/// ([`crate::render::scale_resolve::ScaleKind::text_edge_demand`]).
/// `toward_lower`/`toward_higher` are the ink's extent toward the smaller
/// and larger pixel coordinate respectively, with
/// [`crate::layout::geometry::TEXT_INSET_FLOOR_PX`] and the mark's
/// `dx`/`dy` already folded in; either may be negative (the ink is shifted
/// away from that side, posing no constraint there).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TextEdgeRow {
    pub(crate) value: f64,
    pub(crate) toward_lower: f64,
    pub(crate) toward_higher: f64,
    /// Whether the axis these rows are about to demand a floor from is the
    /// one whose own layers drew this row: always `true` as
    /// [`text_edge_rows`] measures it (every row it returns is the
    /// measuring chart's or leaf's own). A composite group's union
    /// (`render::composite::assign_union`) re-stamps a sibling's copy
    /// `false` for every OTHER member while keeping it `true` for its own
    /// contributor, so [`crate::layout::geometry::EdgeNeed::own`] (which
    /// this flows into unchanged through
    /// [`crate::render::scale_resolve::ScaleKind::text_edge_demand`])
    /// answers "does the axis resolving right now draw this mark" rather
    /// than "did some member of the group draw it".
    pub(crate) own: bool,
}

/// Every [`TextEdgeRow`] a continuous `channel`'s `Mark::Text` layers
/// contribute, scale-independent — built once per rendered
/// channel by [`text_edge_rows`] and shared verbatim by a standalone
/// chart's own axis and, unioned, by a composite group's shared axis.
/// `pub(crate)` (not `pub(in crate::render)`, unlike most of this module):
/// carried on [`seam::LeafResolveInput`](super::seam::LeafResolveInput),
/// which is itself `pub(crate)`, so a narrower visibility here would make
/// that field more private than its own containing struct.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TextEdgeRows(pub(crate) Vec<TextEdgeRow>);

impl TextEdgeRows {
    /// Append `other`'s rows, each stamped `own` (see [`TextEdgeRow::own`]):
    /// `true` when appending a member's own measured rows into its own
    /// union, `false` when appending a sibling's.
    pub(in crate::render) fn extend_from(&mut self, other: &TextEdgeRows, own: bool) {
        self.0.extend(other.0.iter().map(|row| TextEdgeRow { own, ..*row }));
    }

    pub(in crate::render) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Every `Mark::Text` row's ink requirement on `channel`, measured with
/// `metrics` and each layer's own resolved mark style, over
/// `layers`: each already-prepared `LayerPrepared` paired with the batch it
/// draws. A `LayerPrepared`'s encoding and `coord` are already resolved to
/// the rendered coordinate space ([`LayerPrepared::in_its_coord`]), so
/// `channel` ("x" or "y") is read straight off it with no flip logic; a
/// caller that means the primary y slot only (excluding an `independent_y`
/// layer) filters `layers` before calling. An adjusted layer's effective
/// batch is read through
/// [`crate::render::position::layer_prepared_position_preview`], the same
/// position preview the axis domain reads: a stacking layer's text
/// measures at its stacked value, and a dodged/jittered layer's
/// text measures at its adjusted positions, so its ink reserve never
/// under-measures wherever the adjustment actually draws it — layered and
/// unlayered alike, since every rendered layer is a `LayerPrepared`.
/// `color` is the panel's one resolved color scale, the same one the
/// draw's `Dodge` reads, so a dodged text layer's ink measures at the same
/// sub-band order the draw places it.
pub(in crate::render) fn text_edge_rows<'a>(
    channel: &str,
    layers: impl IntoIterator<Item = (&'a LayerPrepared, &'a RecordBatch)>,
    facet: Option<&crate::layout::facet::FacetSpec>,
    theme: &crate::layout::ThemeInputs,
    metrics: &crate::render::font::FontdueMetrics,
    color: Option<&crate::render::scale_resolve::ColorScale>,
) -> TextEdgeRows {
    use crate::layout::geometry::TEXT_INSET_FLOOR_PX as FLOOR;
    use crate::render::marks::text::{glyph_edge_extents, numeric_labels, text_glyphs};

    let mut rows = Vec::new();

    // A layer's column read, style resolve or glyph shaping failing here
    // skips that layer's rows rather than raising: the draw pass measures
    // the same layer with the same `resolve_mark_style`/`text_glyphs` calls
    // and raises the identical typed error there, so this is not a silent
    // degrade — it only avoids reporting the error twice.
    let mut measure = |mark_style: Option<&crate::spec::mark_style::MarkKwargsSpec>,
                        batch: &RecordBatch,
                        encoding: &crate::spec::encoding::Encoding,
                        value_field: &str| {
        let Ok(values) = crate::render::arrow_cast::col_as_f64(batch, value_field) else { return };
        let Ok(style) = crate::render::draw::resolve_mark_style(mark_style, theme, &Mark::Text) else { return };
        let dx = style.text.dx.unwrap_or(0.0);
        let dy = style.text.dy.unwrap_or(0.0);
        let fallback: Vec<Option<String>> = encoding
            .y
            .as_ref()
            .and_then(|e| crate::render::arrow_cast::col_as_f64(batch, &e.field).ok())
            .map(|ys| numeric_labels(&ys))
            .unwrap_or_default();
        let Ok(glyphs) = text_glyphs(encoding, batch, &style.text, theme, &fallback) else { return };
        for (i, v) in values.iter().enumerate() {
            let Some(v) = v.filter(|v| v.is_finite()) else { continue };
            let Some(glyph) = glyphs.get(i).and_then(|g| g.as_ref()) else { continue };
            let ext = glyph_edge_extents(glyph, style.text.align.as_deref(), style.text.baseline.as_deref(), metrics);
            let (toward_lower, toward_higher) = match channel {
                "x" => (ext.left - dx, ext.right + dx),
                "y" => (ext.above - dy, ext.below + dy),
                _ => (0.0, 0.0),
            };
            rows.push(TextEdgeRow {
                value: v,
                toward_lower: FLOOR + toward_lower,
                toward_higher: FLOOR + toward_higher,
                own: true,
            });
        }
    };

    let axis = match channel {
        "x" => StackValueAxis::X,
        "y" => StackValueAxis::Y,
        _ => return TextEdgeRows(rows),
    };
    for (layer, batch) in layers {
        if layer.mark != Mark::Text {
            continue;
        }
        let Some(field) = (match channel {
            "x" => layer.encoding.x.as_ref(),
            "y" => layer.encoding.y.as_ref(),
            _ => None,
        })
        .map(|e| e.field.as_str()) else { continue };
        let adjusted = crate::render::position::layer_prepared_position_preview(layer, axis, batch, facet, color)
            .ok()
            .flatten();
        let effective = adjusted.as_ref().unwrap_or(batch);
        measure(layer.mark_style.as_ref(), effective, &layer.encoding, field);
    }
    TextEdgeRows(rows)
}

/// A primary axis's own [`TextEdgeRows`] on `channel`: every layer on `x`,
/// but on `y` only the layers [`super::YSlotPlan::slot_for_layer`] places on
/// the primary (shared) slot. A `y_slot_plan`'s secondary layers draw on
/// their own slot's axis — `crate::render::prepare::build_secondary_y_axis_inputs`
/// measures each one directly, on its own layer — so a primary axis never
/// floors on a label it does not share. The one rule every caller measuring
/// a primary x/y axis applies, instead of each repeating the filter inline.
pub(in crate::render) fn primary_axis_text_rows<'a>(
    channel: &str,
    layers: impl IntoIterator<Item = (&'a LayerPrepared, &'a RecordBatch)>,
    y_slot_plan: &super::YSlotPlan,
    facet: Option<&crate::layout::facet::FacetSpec>,
    theme: &crate::layout::ThemeInputs,
    metrics: &crate::render::font::FontdueMetrics,
    color: Option<&crate::render::scale_resolve::ColorScale>,
) -> TextEdgeRows {
    if channel == "y" {
        let primary_slot = layers
            .into_iter()
            .enumerate()
            .filter(|(i, _)| y_slot_plan.slot_for_layer(*i) == 0)
            .map(|(_, pair)| pair);
        text_edge_rows(channel, primary_slot, facet, theme, metrics, color)
    } else {
        text_edge_rows(channel, layers, facet, theme, metrics, color)
    }
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
///     (the composite-mark rule — e.g. boxplot whisker fields living in
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
///   - **Zero anchor (T5r, catplot aggregation; a declared line's riser
///     reuses the same rule).** When a layer whose mark style anchors its
///     value at zero — an opted-in `Mark::Bar`, or a `Mark::Line` declaring
///     `rise_from_zero` ([`mark_wants_zero_anchor`]) — binds this channel's
///     VALUE slot (never its position slot — `"y"` unflipped, `"x"` under
///     `CoordFlip` for a bar, the series' own non-advancing axis for a line,
///     in the channel's own coordinate space, same `coord_flipped`
///     normalization the layer-union loop already applies) with data that
///     actually contributed to the union, AND that layer does not itself
///     bind the paired span field on this channel (`y2`/`x2`), the resolved
///     extent is widened to include zero — mirroring
///     `resolve_continuous_domain_and_range`'s `zero: true` block
///     (`d[0] > 0.0 → 0.0`, `d[1] < 0.0 → 0.0`) exactly, just applied on the
///     AUTO-INFERRED path instead of an explicit `ScaleSpec`. The
///     eligibility question — every condition, and why each one mirrors a
///     guard of the single-chart injection
///     (`_spec_build.py::_mark_zero_anchor`: the value axis, the value
///     channel's span partner, the mark's own opt-out or opt-in) — lives in
///     [`layer_wants_zero_anchor`], which owns the per-term rationale; this
///     function owns only the arithmetic. A single (non-layered) `mark_bar`
///     or declared line never reaches this function at all: Python's
///     `_spec_build.py::_mark_zero_anchor` already injects an explicit
///     `scale={"zero": True}` onto that chart's value encoding for either
///     mark, which routes through `build_from_scale_spec` and returns
///     before this function is even called. This rule exists for the case
///     Python's injection cannot serve without silently clipping other
///     layers: a layered spec (`catplot(kind="bar")`'s bar + CI-whisker
///     `rule`, or a declared line sharing an axis with a sibling layer),
///     where an explicit per-layer `ScaleSpec` would replace this
///     cross-layer union with a single-column domain and clip whichever
///     layer's field extends furthest (measured: 3 of 4 upper CI whiskers
///     vanished off-panel). Scoped to `spec.layers` only (never
///     `spec.mark`), so a genuinely unlayered spec's auto path is untouched
///     — byte-identical, because a single-chart bar or declared line never
///     reaches the auto path in the first place.
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn numeric_domain_union(
    channel: &str,
    field: &str,
    paired_field: Option<&str>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
    color: Option<&crate::render::scale_resolve::ColorScale>,
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
        None,
        color,
        None,
    )
    .map(|(e, _)| (e.lo, e.hi))
}

/// The batch the off-side tally's whole-frame rule reads for `field`:
/// `FINAL_OUTPUT_KEY`'s batch when it holds `field`, else `primary`. The
/// ONE selection every reader of "the caller's whole frame" shares — the
/// `Chart`-source arm of [`numeric_domain_union_over`]'s own `read`
/// closure, [`crate::render::position::axis_batch`] (the unlayered door),
/// [`axis_previews`] (a Chart-source layer on the layered door), and
/// `composite::leaf_positional_domain` (a composite leaf) all call this
/// instead of each re-deriving "is `FINAL` the right batch to pair a
/// preview against" on its own.
pub(in crate::render) fn whole_frame_batch<'a>(
    primary: &'a RecordBatch,
    outputs: &'a HashMap<String, RecordBatch>,
    field: &str,
) -> &'a RecordBatch {
    outputs.get(FINAL_OUTPUT_KEY).filter(|b| b.column_by_name(field).is_some()).unwrap_or(primary)
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

/// The scale `channel`'s axis resolves with, in precedence order, BEFORE any
/// pixel geometry is read: the caller's own non-system-injected scale, else
/// the scale a sibling layer establishes ([`layered_caller_scale`]), else a
/// system-injected scale (the bar zero-anchor) unless an auto path (a
/// composite leaf's `shared` domain, or a layered spec) would take it over.
/// This is [`build_axis_scale`]'s own selection, extracted so a caller that
/// only needs to know WHICH scale a channel resolves with — not its full
/// resolved domain/range — can ask the identical question `build_axis_scale`
/// answers, rather than re-deriving it (`render::prepare::log_space_reads`,
/// the positional-space demand's reader).
///
/// [`build_axis_scale`]: super::positional::build_axis_scale
pub(in crate::render) fn resolved_explicit_scale<'a>(
    channel: &'a str,
    enc: &'a crate::spec::encoding::EncodingSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &'a ChartSpec,
    shared: Option<&super::SharedDomain>,
) -> Option<&'a crate::spec::encoding::ScaleSpec> {
    let layered = spec.is_layered();
    let yields_to_auto_path = shared.is_some() || layered;
    match (enc.scale.as_ref(), layered_caller_scale(spec, channel, primary_batch, transform_outputs).map(|(_, scale)| scale)) {
        // The caller's own scale on the chart encoding.
        (Some(own), _) if !own.is_system_injected() => Some(own),
        // The scale the layers establish.
        (_, Some(established)) => Some(established),
        // The bar anchor ferrum injected, unless an auto path takes it over.
        (injected, None) => injected.filter(|_| !yields_to_auto_path),
    }
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

/// Folds a previewed layer's off-side rows into `tally`, through
/// [`LeftOutTally::count_adjusted`]: `rows.unadjusted`'s own `field` column
/// (the caller's rows, content-deduplicated the same way a plain
/// [`LeftOutTally::count`] read dedupes, so this layer's caller column and a
/// demanded statistic's own recorded lineage input on the SAME raw column
/// fold into one count) plus every row the adjustment itself moves off the
/// side. A missing `field` column on either side of `rows` is a no-op — the
/// caller's own EXTENT read already found (or did not find) this field, so
/// this fold only adds to the tally when both halves of the pair actually
/// carry it. The only write path for an [`AdjustedRows`] pair; every caller
/// of [`TallyRead::Adjusted`] in `read()`'s own two arms goes through it.
fn fold_adjusted_tally(
    tally: &std::cell::RefCell<LeftOutTally>,
    rows: &AdjustedRows<'_>,
    field: &str,
    side: crate::scale::log::LogSide,
) -> Result<(), RenderError> {
    let (Some(caller), Some(adjusted)) = (rows.unadjusted.column_by_name(field), rows.adjusted.column_by_name(field)) else {
        return Ok(());
    };
    let caller_extent = sided_min_max_f64(caller.as_ref(), Some(side)).map_err(|_| RenderError::UnsupportedDtype {
        field: field.to_string(),
        dtype: format!("{:?}", caller.data_type()),
        context: None,
    })?;
    tally
        .borrow_mut()
        .count_adjusted(caller.as_ref(), caller_extent, adjusted.as_ref(), side)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("{field}: {e}")))
}

/// [`numeric_domain_union`] with the layer selection spelled out.
///
/// `side`: a `Log` channel's side of zero (always `None` for every other
/// continuous type, since only `Log` ever sets it). Every
/// contributing column folds through [`sided_min_max_f64`] instead of
/// [`column_min_max_f64`] (a `None`-side call of the same function, so a
/// non-`Log` caller's `(lo, hi)` is byte-identical); the rows this union's
/// layers/seeds/facet-shared batches leave out on the other side accumulate
/// into the returned [`SidedExtent`]'s `left_out`/`left_out_zero`/
/// `left_out_opposite`, over EVERY contribution the whole-chart union folds
/// — a faceted chart's shared extent (`include_final`) and a layered
/// chart's every layer alike, so the count
/// `RenderWarning::LogScaleRowsLeftOut`/`RenderError::LogScaleNoValueOnSide`
/// report is the whole frame's, on every door. `adjusted` is the unlayered
/// read's own [`AdjustedRows`] pair (`position::axis_batch`'s own build),
/// `Some` only when a position adjustment applies on this axis — a layered
/// spec threads its own per-layer pairs through [`AxisReads`] instead, so
/// every caller but the unlayered one passes `None` here.
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
    side: Option<crate::scale::log::LogSide>,
    // The panel's own resolved color scale, threaded to every layer's
    // `Dodge` preview ([`axis_previews`]) so a sorted or explicitly-domained
    // legend's sub-band order never diverges from the draw's. `None` for
    // every caller that is not itself resolving the axis a `Dodge` reads
    // (a direct test, or a context with no color scale to offer).
    color: Option<&crate::render::scale_resolve::ColorScale>,
    adjusted: Option<&AdjustedRows<'_>>,
) -> Result<(SidedExtent, LeftOutTally), RenderError> {
    // `(mn, mx)` in a `Cell` while the closures below are live: `extent_only`
    // and `read`'s `take_extent_only` both need to update it, and a `Cell`
    // lets them do so without two simultaneous `&mut` captures of the same
    // state (the same reason the tally below uses a `RefCell`). Unpacked
    // back into plain mutable locals once the data-gathering loop ends,
    // below, so the rest of the function is unchanged.
    let mn_mx = std::cell::Cell::new((f64::INFINITY, f64::NEG_INFINITY));
    // The whole-frame left-out tally, content-deduplicated, so a column read twice (a faceted-shared panel
    // against the all-panels batch, or two layers drawing one column) counts
    // once. `RefCell` lets both `accumulate` (every read) and the Chart-source
    // whole-frame read below (independent facets only) share one tally
    // without two simultaneous `&mut` captures.
    let tally = std::cell::RefCell::new(LeftOutTally::new());
    // T5r zero anchor: set when a layer whose mark style anchors its value
    // at zero (an opted-in `Mark::Bar`, or a `Mark::Line` declaring
    // `rise_from_zero`) binds this channel's primary (non-paired) field, in
    // the channel's own coordinate space. See this function's doc comment
    // ("Zero anchor") for the full rationale.
    let mut zero_anchor_wanted = false;
    // `extent_only`: updates `mn`/`mx` alone. `with_tally`: also feeds the
    // tally. Split so the Chart-source branch below can fold a panel's
    // column into the EXTENT without ALSO counting it into the tally —
    // when the whole-frame tally rule reads `FINAL`'s column separately
    // (an independent facet), the panel's own contribution must not be
    // counted a second time alongside it.
    let extent_only = |c: &dyn Array, source_field: &str| -> Result<(), RenderError> {
        let extent = sided_min_max_f64(c, side).map_err(|_| RenderError::UnsupportedDtype {
            field: source_field.to_string(),
            dtype: format!("{:?}", c.data_type()),
            context: None,
        })?;
        // A column with no value contributes nothing; an integer column's
        // fold would otherwise report its type's `MAX`/`MIN` seeds.
        if c.null_count() == c.len() {
            return Ok(());
        }
        let (lo, hi) = mn_mx.get();
        mn_mx.set((lo.min(extent.lo), hi.max(extent.hi)));
        Ok(())
    };
    let tally_only = |c: &dyn Array, source_field: &str| -> Result<(), RenderError> {
        let extent = sided_min_max_f64(c, side).map_err(|_| RenderError::UnsupportedDtype {
            field: source_field.to_string(),
            dtype: format!("{:?}", c.data_type()),
            context: None,
        })?;
        if c.null_count() != c.len() {
            tally.borrow_mut().count(c, extent);
        }
        Ok(())
    };
    let accumulate = |c: &dyn Array, source_field: &str| -> Result<(), RenderError> {
        extent_only(c, source_field)?;
        tally_only(c, source_field)
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
    //
    // The whole-frame tally rule: a Chart-source field is always
    // COUNTED from the all-panels column when the chart publishes one,
    // whatever this read's own facet resolution. `include_final` still
    // governs the EXTENT fold alone (a true per-panel domain for an
    // independent facet): when it is already true and `FINAL` holds the
    // field, `FINAL` is read ALONE (it is a superset of the panel, so the
    // extent is byte-identical, and the panel is never double-counted into
    // the tally); when `include_final` is false (independent facet), the
    // panel's own extent still comes from `primary_batch` alone, but the
    // tally ALSO reads `FINAL`'s column directly, so every panel's warning
    // reports the whole frame's count — the restatements then collapse to
    // one via `panel_pass_warning_restated_by_every_panel`, the same way
    // every other per-panel-restated warning already does.
    let read = |source: Option<LayerSource<'_>>, tally_read: TallyRead<'_>, f: &str| -> Result<bool, RenderError> {
        let mut found = false;
        let primary_has = primary_batch.column_by_name(f).is_some();
        let take = |batch: &RecordBatch, found: &mut bool| -> Result<(), RenderError> {
            if let Some(c) = batch.column_by_name(f) {
                accumulate(c.as_ref(), f)?;
                *found = true;
            }
            Ok(())
        };
        let take_extent_only = |batch: &RecordBatch, found: &mut bool| -> Result<(), RenderError> {
            if let Some(c) = batch.column_by_name(f) {
                extent_only(c.as_ref(), f)?;
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
            // `Batch` is reached here (a layer's own position-adjusted
            // preview batch, built by `axis_previews`, read through
            // `AxisReads::over`'s `seed.source`/`reads.source(index)`);
            // `Member` is not (only `overlay_aux_reads`/
            // `overlay_raster_cell_reads` construct it, never `seed_reads`/
            // `AxisReads`). A `Member`'s embedded lineage plays no role in
            // this numeric union either way, so folding it with `Batch`
            // costs nothing even if it were reached.
            //
            // The extent always reads `batch` (the preview — the extent
            // always comes from the adjusted rows, unconditionally). The
            // tally dispatches on `tally_read`: a plain caller column
            // ([`TallyRead::Column`]) still reads `batch` itself (an
            // unadjusted layer never substitutes a preview for `batch` in
            // the first place); a previewed layer's own
            // [`AdjustedRows`] pair ([`TallyRead::Adjusted`]) folds through
            // [`fold_adjusted_tally`] instead, which reads the CALLER's
            // rows from `rows.unadjusted` plus the rows the adjustment
            // itself moves off the side; a stack-base span
            // ([`TallyRead::ExtentOnly`]) contributes nothing to the tally
            // at all — a stack's base is not caller data.
            Some(LayerSource::Batch(batch) | LayerSource::Member(batch, _)) => {
                if let Some(c) = batch.column_by_name(f) {
                    extent_only(c.as_ref(), f)?;
                    found = true;
                }
                match tally_read {
                    TallyRead::Column => {
                        if let Some(c) = batch.column_by_name(f) {
                            tally_only(c.as_ref(), f)?;
                        }
                    }
                    TallyRead::Adjusted(rows) => {
                        if let Some(side) = side {
                            fold_adjusted_tally(&tally, &rows, f, side)?;
                        }
                    }
                    TallyRead::ExtentOnly => {}
                }
            }
            Some(LayerSource::Chart) | None => {
                let final_col = transform_outputs.get(FINAL_OUTPUT_KEY).and_then(|b| b.column_by_name(f));
                let final_batch = transform_outputs
                    .get(FINAL_OUTPUT_KEY)
                    .filter(|_| include_final && final_col.is_some());
                // Consumed exactly once here: `plain_tally` and
                // `adjusted_rows` cover every later read of `tally_read` in
                // this arm, so no later line touches the moved value again.
                let (plain_tally, adjusted_rows) = match tally_read {
                    TallyRead::Column => (true, None),
                    TallyRead::Adjusted(rows) => (false, Some(rows)),
                    TallyRead::ExtentOnly => (false, None),
                };
                if let Some(batch) = final_batch {
                    if let Some(c) = batch.column_by_name(f) {
                        extent_only(c.as_ref(), f)?;
                        found = true;
                        if plain_tally {
                            tally_only(c.as_ref(), f)?;
                        }
                    }
                } else {
                    // EXTENT only here — the panel's own domain stays
                    // independent (or there is no `FINAL` to union with);
                    // the tally is resolved separately below so the panel's
                    // contribution is never ALSO counted into it when
                    // `FINAL` is read for the tally.
                    take_extent_only(primary_batch, &mut found)?;
                    for (key, batch) in transform_outputs {
                        let key_is_final = include_final && key == FINAL_OUTPUT_KEY;
                        if key_is_final {
                            take_extent_only(batch, &mut found)?;
                        } else if source.is_none() && !primary_has {
                            // an unlayered
                            // spec's field living only in a named output (a
                            // composite mark's own output) is tallied too —
                            // `take`, not `take_extent_only` — the same full
                            // read an `Output` source gets above. These
                            // batches are all-panels batches like `FINAL`'s
                            // own, so the panel double-count the whole-frame
                            // rule below guards against cannot recur,
                            // and `LeftOutTally`'s content dedupe keeps an
                            // output equal to `FINAL` from counting twice.
                            take(batch, &mut found)?;
                        }
                    }
                    if side.is_some() && plain_tally {
                        // The whole-frame tally rule: `FINAL`'s
                        // column when the chart publishes one, else the
                        // panel's own (an unfaceted call, or a field not
                        // yet in `FINAL` — the panel IS the whole frame
                        // there).
                        match final_col {
                            Some(c) => tally_only(c.as_ref(), f)?,
                            None => {
                                if let Some(c) = primary_batch.column_by_name(f) {
                                    tally_only(c.as_ref(), f)?;
                                }
                            }
                        }
                    }
                }
                if let (Some(rows), Some(side)) = (&adjusted_rows, side) {
                    fold_adjusted_tally(&tally, rows, f, side)?;
                }
            }
        }
        Ok(found)
    };

    // The bound field and its paired endpoint: from the primary batch on an
    // unlayered spec (whose caller already resolved its stack,
    // `position::axis_batch`), from the batches and fields that draw them on
    // a layered one (see [`seed_reads`]). A layer that stacks its value on
    // this axis is read from the stack it draws ([`AxisReads`]), so its
    // segments widen the axis whatever batch it draws, while an unstacked
    // sibling reads its own values. Which of those reads found its field is
    // kept for the layer loop below, so a layer drawing a seed read's field
    // from its batch is not read twice.
    let layers = spec.layers.as_deref().unwrap_or_default();
    let (previews, bases, layer_adjusted) = axis_previews(spec, layers, channel, primary_batch, transform_outputs, include_final, color)?;
    let reads = AxisReads::over(layers, &previews, bases, layer_adjusted);
    let seeds: Vec<SeedRead<'_>> = seed_reads(spec, layers, channel, (field, paired_field, None), &reads);
    let mut seed_found: Vec<(SeedRead<'_>, bool)> = Vec::with_capacity(seeds.len());
    // Unlayered ([`ChartSpec::is_layered`]): with no layer there is no
    // seed, so the chart's one batch is the read. `adjusted` pairs the
    // VALUE field with its own caller-row tally; the paired field is
    // `ExtentOnly` when it is literally the stack's base column
    // (`position::axis_batch`'s own `base`), `Column` for a caller's own
    // `x2`/`y2`.
    if !spec.is_layered() {
        let value_tally = adjusted.map_or(TallyRead::Column, |rows| TallyRead::Adjusted(rows.clone()));
        read(None, value_tally, field)?;
        if let Some(p) = paired_field {
            let span_tally = if adjusted.is_some_and(|rows| rows.base == Some(p)) { TallyRead::ExtentOnly } else { TallyRead::Column };
            read(None, span_tally, p)?;
        }
    }
    let chart_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    for seed in &seeds {
        // An inherited field is a later layer's column, judged as that
        // layer's own column is in the loop below: placeable on this axis
        // first, or, for a layer whose own coord exchanges its channels,
        // left to `validate_layer_coord_exchange_is_servable`.
        let exchanges = seed.inheritor.zip(spec.layers.as_deref()).is_some_and(|(index, layers)| {
            layer_flips(&layers[index], spec.coord.as_ref()) != chart_flipped
        });
        if exchanges {
            let found = read(Some(seed.source), seed.tally.clone(), seed.field).unwrap_or(false);
            seed_found.push((seed.clone(), found));
            if let Some(span) = seed.span {
                read(Some(seed.source), seed.span_tally.clone(), span).unwrap_or(false);
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
        let found = read(Some(seed.source), seed.tally.clone(), seed.field)?;
        seed_found.push((seed.clone(), found));
        if let Some(span) = seed.span {
            read(Some(seed.source), seed.span_tally.clone(), span)?;
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
    // "y"). Before this, a layer's OWN x2/y2 was never
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
        // COORDINATE SPACE (T13). `channel`/`field` name the axis being resolved,
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
        // (overwhelmingly common) unflipped layer.
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
            // data in some batch — `value_field_contributed` below. A layer's
            // field can be absent from every batch (wrong `data_source`, not loaded yet),
            // and the bar zero-anchor rule below must not fire on the
            // strength of the field's declared NAME alone when it never
            // contributed a single value.
            let source = reads.source(layer_index);
            let mut value_field_contributed = false;
            for (i, lf) in layer_fields.into_iter().enumerate() {
                let Some(lf) = lf else { continue };
                // Only the VALUE slot (index 0) is the field a `Dodge`/
                // `Jitter`/`Stack` preview pairs with its own `AdjustedRows`
                // — the paired span (`x2`/`y2`, index 1) is always a
                // caller-written field here (a stacking layer's base column
                // is read separately, below, never through `layer_fields`),
                // so it stays `Column` unconditionally.
                let tally_read = if i == 0 { reads.tally(layer_index) } else { TallyRead::Column };
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
                    read(Some(source), tally_read, lf).unwrap_or(false)
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
                            read(Some(source), tally_read, lf)?
                        }
                    }
                };
                if i == 0 {
                    value_field_contributed = contributed;
                }
            }
            // A stacking layer's segments reach down (or up) to their bases:
            // extent only, never the tally — a stack's base is not caller
            // data.
            if let Some(base) = reads.base(layer_index) {
                read(Some(source), TallyRead::ExtentOnly, base)?;
            }
            // The authored `y` decides the bar's orientation: categorical by
            // its declared type or by the column it reads in this layer's batch.
            // A line's own value-axis orientation follows its declaration
            // instead (`curve_value_on_x`). Every other mark, and an
            // undeclared line through the `_` arm below, still reaches
            // `layer_wants_zero_anchor`: its own mark term
            // (`mark_wants_zero_anchor`) refuses it there, so the
            // orientation computed for it here is moot, not unreached.
            let y_categorical = layer.encoding.y.as_ref().is_some_and(|y| {
                column_reads_categorical(y, source.locate(&y.field, primary_batch, transform_outputs))
            });
            // Only a layer the eligibility check could actually accept pays
            // for an orientation answer: a bar's is a plain dtype read, but
            // a line's builds the layer exactly as the draw pass does
            // (`curve_value_on_x`) and can fail the way that pass can, so an
            // ineligible line (undeclared, or a caller scale this channel
            // yields to) never builds one and never reaches that error path.
            if matches!(layer_union, LayerUnion::Every) {
                let value_on_x = match layer.mark {
                    Mark::Line if mark_wants_zero_anchor(layer.mark, layer.mark_style.as_ref()) => {
                        curve_value_on_x(spec, layer)?
                    }
                    _ => bar_value_on_x(coord_flipped, y_categorical),
                };
                if layer_wants_zero_anchor(layer, channel, value_on_x, layer_fields, value_field_contributed) {
                    zero_anchor_wanted = true;
                }
            }
        }
    }

    let (mut mn, mut mx) = mn_mx.get();
    if (!mn.is_finite() || !mx.is_finite()) && side.is_none() {
        // All values were null/NaN — return a default domain so the chart
        // renders with axes but no marks, instead of raising an error. A
        // sided call (`Log`) falls through instead: its caller needs the
        // genuine empty extent, together with `left_out`, to tell "no value
        // at all" (this same default, applied by the `Log` arm itself) from
        // "no value on this side" (`RenderError::LogScaleNoValueOnSide`).
        return Ok((SidedExtent { lo: 0.0, hi: 1.0, left_out: 0, left_out_zero: false, left_out_opposite: None }, tally.into_inner()));
    }
    if matches!(layer_union, LayerUnion::CarryingScale(_)) {
        return Ok((SidedExtent { lo: mn, hi: mx, left_out: 0, left_out_zero: false, left_out_opposite: None }, tally.into_inner()));
    }
    if zero_anchor_wanted {
        // The SAME widening `resolve_continuous_domain_and_range` applies for a
        // `ScaleSpec::Linear { zero: true }`, through the one helper both call
        // (`positional::extend_extent_to_zero`) rather than a second copy of its
        // two comparisons. Applied here as well as there because an
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
    // zero-span domain, so this expansion is not needed to keep marks
    // from vanishing: `to_pixel_f64` does not return NaN there. It stays
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
    Ok((SidedExtent { lo: mn, hi: mx, left_out: 0, left_out_zero: false, left_out_opposite: None }, tally.into_inner()))
}

/// Count every lineage's recorded caller column for `axis`
/// (`crate::transform::run::OutputLineage::log_input`'s `(axis, field)`
/// entries whose axis matches exactly — channel AND y-slot, so a
/// secondary layer's own off-side rows never reach the primary's tally or
/// another secondary slot's) into `tally`, through the SAME
/// content-deduplicated path [`sided_min_max_f64`] feeds a primitive
/// layer's own column read. `numeric_domain_union_over`'s own column reads
/// never reach a demanded statistic's filtered input, so its returned
/// tally alone misses those rows; this is the other half every `Log`
/// caller of that function folds in afterward, only when `side` is `Some`
/// (a `Log` call — every other scale kind's tally stays exactly what
/// `numeric_domain_union_over` returned). Called through
/// [`super::positional::LogAxisReads::fold_into`] — the chart's own axis at
/// slot `0`, or an independent-y layer's own slot, whichever `axis` that
/// view was built for — and directly by `composite::leaf_positional_domain`,
/// always at slot `0` (a composite overlay never carries an independent-y
/// member). A recorded column
/// [`sided_min_max_f64`] rejects is refused rather than skipped, the same
/// `RenderError::UnsupportedDtype` every other caller of that function
/// raises, so this fold can never under-count a left-out total on the
/// strength of a dtype gap.
pub(in crate::render) fn fold_log_input_into_tally(
    lineages: &crate::transform::run::OutputLineages,
    axis: crate::spec::layer::PositionalAxis,
    side: Option<crate::scale::log::LogSide>,
    tally: &mut LeftOutTally,
) -> Result<(), RenderError> {
    let Some(side) = side else { return Ok(()) };
    for lineage in lineages.values() {
        for ((entry_axis, field), column) in &lineage.log_input {
            if *entry_axis != axis {
                continue;
            }
            let extent = sided_min_max_f64(column.as_ref(), Some(side)).map_err(|_| RenderError::UnsupportedDtype {
                field: field.clone(),
                dtype: format!("{:?}", column.data_type()),
                context: None,
            })?;
            tally.count(column.as_ref(), extent);
        }
    }
    Ok(())
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
/// Private to this module: [`positional_category_union`] and
/// [`establishing_categories`] are the entries the scale paths call, so "the
/// ordinal domain of this axis" has a single answer per reading: the layer
/// union on the auto path, the establishing column under a caller's scale.
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

/// The categories of the column an axis's establishing layer draws: the one
/// batch of an unlayered spec, else the batch of the layer that establishes
/// the axis ([`locate_seed_field`]). Under a facet-shared channel
/// (`include_final`) a seed on the chart's rows also reads the all-panels
/// batch, so every panel enumerates the categories the shared axis shows; a
/// seed on a named output reads that output, which already spans every panel.
///
/// This is the category set of an unlayered axis on the auto path
/// ([`positional_category_union`]) and of a caller's categorical scale stating
/// no `domain` on any spec: that scale is the establishing member's, so it
/// takes its categories from that member's column alone, never the other
/// layers'.
pub(in crate::render) fn establishing_categories(
    channel: &str,
    field: &str,
    paired_positional: Option<&str>,
    spec: &ChartSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    include_final: bool,
) -> Result<Vec<String>, RenderError> {
    let seed = locate_seed_field(channel, field, spec, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let reads_chart_rows = spec
        .layers
        .as_deref()
        .and_then(|layers| establishing_axis_layer(spec, layers, channel))
        .is_none_or(|(_, layer)| LayerSource::of(layer) == LayerSource::Chart);
    distinct_positional_categories_shared(
        seed.batch,
        field,
        paired_positional,
        transform_outputs,
        include_final && reads_chart_rows,
    )
}

/// The ordinal/nominal counterpart of [`numeric_domain_union`]: the category
/// ladder for an axis, covering the chart-level column **and every layer's own
/// column for this channel**.
///
/// The shared-axis invariant is "the resolved domain covers every layer's
/// extent". The numeric side already enforces it —
/// [`numeric_domain_union`]'s `spec.layers` loop, whose own doc
/// explains why (a cross-DataFrame `a + b` merge renames the RHS columns, so the
/// chart-level field covers the LHS data ALONE). The ordinal side had no
/// equivalent, so the same shape lost the RHS layer's categories entirely: no
/// band existed for them, and every mark of that layer was silently dropped for
/// want of one.
///
/// **Order: first appearance, in layer order.** Walk the chart-level column,
/// then the layers in spec order, appending each category not yet seen.
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
    // Unlayered ([`ChartSpec::is_layered`]): the chart's own categories.
    let layers = spec.layers.as_deref().unwrap_or_default();
    if !spec.is_layered() {
        return establishing_categories(channel, field, paired_positional, spec, primary_batch, transform_outputs, include_final);
    }
    // The precondition [`establishing_categories`] checks for an unlayered
    // spec: the locator the caller types this axis with finds the bound field
    // somewhere.
    locate_seed_field(channel, field, spec, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
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
    let seeds = seed_reads(spec, layers, channel, (field, None, paired_positional), &AxisReads::unstacked(layers));
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
        // WHOSE space is per layer, and the read is therefore INSIDE the loop:
        // the layer `prepare::build_layers` swaps
        // is the one whose OWN effective coord flips, so that is the layer whose
        // authored `y` occupies this x axis. Reading the CHART's flip once,
        // hoisted above the loop, cannot express a per-layer answer: a
        // member carrying its own `CoordFlip` would contribute its
        // AUTHORED channel instead of the post-flip one —
        // `bar(x="cat:N") + point(x="v2:Q", y="cat2:N").coord(CoordFlip())`
        // would enumerate the point layer's value column as three spurious
        // bands (`['a','b','c', '4','5','6']`) while the layer's real
        // post-flip categories never reach the ladder at all, so it would
        // draw 0 of its 3 circles.
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
        // either, so it contributes nothing and that is lossless. The two
        // halves of that sentence are one dtype set by
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
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn group_batch(groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("group", DataType::Utf8, true)]));
        let arr: Vec<Option<&str>> = groups.into_iter().map(Some).collect();
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(arr))]).unwrap()
    }

    /// An empty layer list is unlayered on the auto ordinal ladder as on
    /// every other positional reader: it enumerates the bound column, exactly
    /// as no layer list does. Rejects keying on `Some(layers)`, which read no
    /// seed and returned an empty ladder.
    #[test]
    fn an_empty_layer_list_reads_the_unlayered_category_ladder() {
        let primary = group_batch(vec!["A", "B", "A"]);
        let outputs = HashMap::new();
        let ladder = |layers: Option<Vec<Layer>>| {
            let spec = ChartSpec { layers, ..Default::default() };
            positional_category_union("x", "group", None, &primary, &outputs, &spec, false)
                .expect("the bound column enumerates")
        };
        assert_eq!(ladder(None), ["A", "B"]);
        assert_eq!(ladder(Some(Vec::new())), ["A", "B"], "an empty layer list is unlayered");
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

    // --- layer_wants_zero_anchor: per-term unit tests --------------------
    //
    // `layer_wants_zero_anchor` is a five-term boolean that the
    // `*_SPEC_JSON` tests below reach only end to end, through a whole
    // spec + batch fixture. Here each term gets a direct negative case:
    // flip exactly one input away from the eligible baseline and the
    // answer must flip too. That is what makes "would Python's injection
    // have anchored this layer?" a checkable question rather than an
    // emergent one.

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
        assert!(layer_wants_zero_anchor(&layer, "y", false, [Some("v"), None], true));
    }

    /// Term (a), position half: the bar's POSITION channel is never anchored.
    /// This is the term whose absence squeezed `displot(kind="hist")` bars into
    /// a corner by zero-anchoring their `x` bin-range axis.
    #[test]
    fn bar_zero_anchor_refuses_position_channel() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(!layer_wants_zero_anchor(&layer, "x", false, [Some("v"), None], true));
    }

    /// Term (a), flip half: under `CoordFlip` the value channel is `x`, not
    /// `y` — both readings move together, so the flipped `y` is now the
    /// position channel and is refused.
    #[test]
    fn bar_zero_anchor_follows_the_flip() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(layer_wants_zero_anchor(&layer, "x", true, [Some("v"), None], true));
        assert!(!layer_wants_zero_anchor(&layer, "y", true, [Some("v"), None], true));
    }

    /// Mark term: a point, rule or area, and a line that does not declare
    /// `rise_from_zero`, leave the union alone on the same value channel.
    #[test]
    fn undeclared_non_bar_marks_never_anchor() {
        for mark in [Mark::Point, Mark::Line, Mark::Rule, Mark::Area] {
            let layer = zero_anchor_layer(mark, None);
            assert!(
                !layer_wants_zero_anchor(&layer, "y", false, [Some("v"), None], true),
                "an undeclared mark must never anchor, got an anchor for {mark:?}"
            );
        }
    }

    /// Term (c): the per-layer opt-out. `zero: Some(false)` refuses;
    /// `Some(true)` and an absent `mark_style` are both the default-on case.
    #[test]
    fn bar_zero_anchor_honors_the_per_layer_opt_out() {
        let opted_out = zero_anchor_layer(Mark::Bar, Some(false));
        assert!(!layer_wants_zero_anchor(&opted_out, "y", false, [Some("v"), None], true));
        let opted_in = zero_anchor_layer(Mark::Bar, Some(true));
        assert!(layer_wants_zero_anchor(&opted_in, "y", false, [Some("v"), None], true));
    }

    /// Term (b): a value channel carrying a paired span field (`y2`, or `x2`
    /// under flip) takes its extent literally from `[value, span]`.
    #[test]
    fn bar_zero_anchor_refuses_a_span_bound_value_channel() {
        let layer = zero_anchor_layer(Mark::Bar, None);
        assert!(!layer_wants_zero_anchor(
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
        assert!(!layer_wants_zero_anchor(&layer, "y", false, [Some("v"), None], false));
    }

    // --- a declared line's riser reuses the same eligibility rule ----------

    fn rise_from_zero_layer(rise: Option<bool>) -> Layer {
        Layer {
            mark: Mark::Line,
            encoding: Default::default(),
            mark_style: rise.map(|r| crate::spec::mark_style::MarkKwargsSpec {
                rise_from_zero: Some(r),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Term (c), the line's own opt-in: `rise_from_zero: Some(true)` anchors;
    /// `Some(false)` and an absent `mark_style` both stay off — the opposite
    /// default from a bar's `zero`, which anchors unless opted OUT.
    #[test]
    fn line_zero_anchor_honors_the_declared_opt_in() {
        let declared = rise_from_zero_layer(Some(true));
        assert!(layer_wants_zero_anchor(&declared, "y", false, [Some("v"), None], true));
        let undeclared = rise_from_zero_layer(None);
        assert!(!layer_wants_zero_anchor(&undeclared, "y", false, [Some("v"), None], true));
        let declared_false = rise_from_zero_layer(Some(false));
        assert!(!layer_wants_zero_anchor(&declared_false, "y", false, [Some("v"), None], true));
    }

    /// Term (a) for a line: only its VALUE channel (the series' own
    /// non-advancing axis) anchors — the observation axis it advances along
    /// never does, the line sibling of `bar_zero_anchor_refuses_position_channel`.
    #[test]
    fn line_zero_anchor_refuses_the_advancing_axis() {
        let layer = rise_from_zero_layer(Some(true));
        assert!(!layer_wants_zero_anchor(&layer, "x", false, [Some("v"), None], true));
    }

    /// The domain's zero-anchor orientation reads `resolve_curve_axis`, the
    /// exact rule `DrawCtx::series_curve_axis` reads too: a declaration
    /// (already post-flip) answers outright, flip or not; undeclared, the
    /// vertical default is exchanged by the coord flip or by a stacked
    /// value landing on x — the two transforms a real input never sets at
    /// once, so the `||` never needs to choose between them.
    #[test]
    fn resolve_curve_axis_reads_the_declared_axis_else_the_flip_or_the_stack() {
        use crate::render::marks::curve::{resolve_curve_axis, CurveAxis};
        assert_eq!(resolve_curve_axis(Some(StackValueAxis::X), false, false), CurveAxis::Y);
        assert_eq!(
            resolve_curve_axis(Some(StackValueAxis::X), true, false),
            CurveAxis::Y,
            "a declaration answers outright, flip or not"
        );
        assert_eq!(resolve_curve_axis(Some(StackValueAxis::Y), true, true), CurveAxis::X);
        assert_eq!(
            resolve_curve_axis(None, false, false),
            CurveAxis::X,
            "undeclared defaults to vertical: value on y"
        );
        assert_eq!(
            resolve_curve_axis(None, true, false),
            CurveAxis::Y,
            "undeclared, flipped: value exchanges onto x"
        );
        assert_eq!(
            resolve_curve_axis(None, false, true),
            CurveAxis::Y,
            "undeclared, unflipped, but a stack cumulates the value onto x"
        );
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
            fallback: None,
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


    /// A well-formed sort-field object naming a field the batch does not
    /// carry still warns.
    #[test]
    fn apply_sort_to_domain_unresolvable_well_formed_object_still_warns() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "group",
            batch: &batch,
            fallback: None,
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

    // --- Numeric ascending/descending orders by value, not as text ---------

    fn int64_category_batch(values: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(arrow::array::Int64Array::from(values))]).unwrap()
    }

    fn uint64_category_batch(values: Vec<u64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::UInt64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(arrow::array::UInt64Array::from(values))]).unwrap()
    }

    fn float64_category_batch(values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(arrow::array::Float64Array::from(values))]).unwrap()
    }

    fn numeric_sort_ctx<'a>(batch: &'a RecordBatch) -> SortContext<'a> {
        SortContext { category_field: "v", batch, fallback: None, x_field: None, y_field: None, coord_flipped: false }
    }

    /// `[-3, 10, 1, 2]` ascending reads `-3, 1, 2, 10` by value, not
    /// `-3, 1, 10, 2` as text would order it.
    #[test]
    fn apply_sort_to_domain_orders_an_int64_column_by_value_ascending() {
        let batch = int64_category_batch(vec![-3, 10, 1, 2]);
        let ctx = numeric_sort_ctx(&batch);
        let mut domain = vec!["-3".to_string(), "10".to_string(), "1".to_string(), "2".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["-3", "1", "2", "10"]);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// `descending` reverses the whole numeric-ordered result: `10, 2, 1, -3`.
    #[test]
    fn apply_sort_to_domain_orders_an_int64_column_by_value_descending() {
        let batch = int64_category_batch(vec![-3, 10, 1, 2]);
        let ctx = numeric_sort_ctx(&batch);
        let mut domain = vec!["-3".to_string(), "10".to_string(), "1".to_string(), "2".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("descending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["10", "2", "1", "-3"]);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A `UInt64` column orders by value the same way.
    #[test]
    fn apply_sort_to_domain_orders_a_uint64_column_by_value() {
        let batch = uint64_category_batch(vec![10, 2, 1]);
        let ctx = numeric_sort_ctx(&batch);
        let mut domain = vec!["10".to_string(), "2".to_string(), "1".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["1", "2", "10"]);
    }

    /// A `Float64` column orders by value, including negatives and
    /// integer-valued floats stringified without a decimal point (`10`,
    /// not `10.0`, matching `col_as_ordinal_category_str`'s own format).
    #[test]
    fn apply_sort_to_domain_orders_a_float64_column_by_value() {
        let batch = float64_category_batch(vec![1.5, -0.5, 10.0]);
        let ctx = numeric_sort_ctx(&batch);
        let mut domain = vec!["1.5".to_string(), "-0.5".to_string(), "10".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["-0.5", "1.5", "10"]);
    }

    /// A `NaN` category (the ordinal reader's own `"NaN"` text for a
    /// Float64 NaN value) sorts after every finite number ascending, but
    /// before them descending: the whole order reverses, not just the
    /// finite tier.
    #[test]
    fn apply_sort_to_domain_flips_nan_position_with_sort_direction() {
        let batch = float64_category_batch(vec![1.0, -2.0]);
        let ctx = numeric_sort_ctx(&batch);
        let mut domain = vec!["NaN".to_string(), "1".to_string(), "-2".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["-2", "1", "NaN"], "ascending: NaN last");
        let mut domain = vec!["NaN".to_string(), "1".to_string(), "-2".to_string()];
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("descending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["NaN", "1", "-2"], "descending: NaN first (the whole order reverses)");
    }

    /// An explicit scale domain can name an entry that isn't a number at
    /// all (`"x"`) alongside a numeric column's own values: the numbers
    /// order by value, and the non-parsing entry sorts after all of them.
    #[test]
    fn apply_sort_to_domain_puts_a_non_numeric_explicit_entry_after_every_number() {
        let batch = int64_category_batch(vec![1, 2]);
        let ctx = numeric_sort_ctx(&batch);
        let mut domain = vec!["x".to_string(), "2".to_string(), "1".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["1", "2", "x"]);
    }

    /// A domain holding a finite number, a `NaN` entry and a non-parsing
    /// entry together: NaN sits between the finite and non-parsing tiers
    /// in ascending order, and descending reverses that whole order rather
    /// than only flipping the finite tier. `numeric_domain_cmp`'s doc names
    /// this exact three-tier shape; this test is the one place all three
    /// tiers appear in a single domain.
    #[test]
    fn apply_sort_to_domain_orders_nan_between_finite_and_non_parsing_both_directions() {
        let batch = float64_category_batch(vec![5.0, -2.0]);
        let ctx = numeric_sort_ctx(&batch);
        let mut warnings = Vec::new();

        let mut domain = vec!["x".to_string(), "NaN".to_string(), "5".to_string()];
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["5", "NaN", "x"], "ascending: finite, then NaN, then non-parsing");

        let mut domain = vec!["x".to_string(), "NaN".to_string(), "5".to_string()];
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("descending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["x", "NaN", "5"], "descending: non-parsing, then NaN, then finite");
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A `Utf8`/`Categorical` column holding digit strings stays in plain
    /// text order (`"10"` before `"2"`): the numeric-value rule applies
    /// only to a numeric column, never to a string column whose VALUES
    /// happen to look numeric.
    #[test]
    fn apply_sort_to_domain_keeps_text_order_for_a_utf8_column_of_digits() {
        let batch = group_batch(vec!["10", "2"]);
        let ctx = SortContext { category_field: "group", batch: &batch, fallback: None, x_field: None, y_field: None, coord_flipped: false };
        let mut domain = vec!["10".to_string(), "2".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("ascending")), &ctx, &mut warnings).unwrap();
        assert_eq!(domain, vec!["10", "2"], "text order: '10' < '2' byte-wise");
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
            fallback: None,
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
    /// For `#95`: a fixture of `x_field: Some("weight"), y_field: None`
    /// would describe an authored `y` bound to `weight`, which RESOLVES.
    /// That fixture would pass on a different message ("cannot read sort
    /// value field") without pinning R3's claim, so this one leaves
    /// post-flip x unbound instead.
    #[test]
    fn sort_shorthand_missing_channel_names_users_own_text_under_flip() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "cat",
            batch: &batch,
            fallback: None,
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
            fallback: None,
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
            fallback: None,
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

    // ── T13: the layer union reads each layer's fields in the coordinate
    // space of the channel being resolved ────────────────────────
    //
    // Two properties, pinned together because one is the other's control:
    //   1. a layer's `x2`/`y2` past its bare `x`/`y` widens THAT channel's
    //      domain (T13 — `mark_qq(line=True)`'s reference
    //      diagonal was unreachable without it), and
    //   2. under `CoordFlip` it widens the OTHER axis, because `spec.layers`
    //      is the AUTHORED encoding while `channel`/`field` name the rendered
    //      (post-flip) axis.
    //
    // Both `*_SPEC_JSON`s are the VERBATIM `chart.to_dict()` of the public-API
    // call in their doc comment, parsed through the same
    // serde path `ChartSpec.from_json` uses, so they cannot drift from what
    // Python lowers.

    /// `fm.Chart(df).mark_point().encode(x="b", y="a")` layered (`+`) with
    /// `fm.Chart(df).mark_rule().encode(y="a", y2="big")`.
    const LAYERED_Y2_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "rule", "encoding": {"y": {"field": "a"}, "y2": {"field": "big"}}}]}"#;

    /// The same layering with `.coord(fm.CoordFlip())` on the primary chart —
    /// an A/B probe. `prepare::build_layers` swaps the RENDERING
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
        numeric_domain_union(channel, field, None, &batch, &HashMap::new(), &spec, false, None)
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
    /// there.
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

    // ── a layer column that cannot serve the shared numeric
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
        let err = numeric_domain_union("y", "a", None, &batch, &HashMap::new(), &spec, false, None)
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
        let extent = numeric_domain_union("y", "a", None, &batch, &HashMap::new(), &spec, false, None)
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
        let extent = numeric_domain_union("y", "a", None, &batch, &HashMap::new(), &spec, false, None)
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
        let err = numeric_domain_union("y", "a", None, &primary, &outputs, &spec, true, None)
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
    // `Summary` transform output via `data_source`. Without a zero rule,
    // the auto-inferred cross-layer union (this function) would stop the
    // value axis at the union of `mean`/`lower`/`upper`, truncating the
    // bars at the panel bottom. An explicit per-layer
    // `scale={"zero": True}` (the mechanism a plain single-chart `mark_bar`
    // uses, injected by `_spec_build.py`) was measured and rejected: it
    // routes through `build_from_scale_spec`, which resolves the domain from
    // ONE column, so it silently clipped the whisker layer's own extent (3 of
    // 4 upper CI whiskers vanished off-panel) — see
    // this function's own doc comment
    // ("Zero anchor") above.

    /// Verbatim `chart.to_dict()` JSON for `fm.catplot(df, x="cat", y="val",
    /// kind="bar")` (T5r, with `df` mapping `cat` to
    /// two groups and `val` to values whose bootstrap mean/CI land around
    /// 50-90). Two layers: `bar` on the transform's `mean` output, `whisker`
    /// (a `rule`) on `lower`/`upper`, both via `data_source: "summary"`.
    const CATPLOT_BAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}]}"#;

    /// The same call with `orient="h"` (T5r). The
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
        numeric_domain_union(channel, field, None, &primary, &outputs, &spec, false, None)
            .expect("every field in the fixture is Float64")
    }

    /// Without the zero rule, the union of `mean` (50..80), `lower`
    /// (40..70), and `upper` (60..90) would be `(40.0, 90.0)`, truncated
    /// relative to zero. The `bar` layer's own field
    /// (`mean`) participates in the y-channel union, so the zero-anchor
    /// widens the lower bound to 0, while the whisker's own `upper` extent
    /// (90) stays reachable — the exact property an explicit per-layer
    /// `scale={"zero": True}` (rejected in the bar zero-anchor section
    /// header above) could not
    /// preserve.
    ///
    /// RED (verified in place): commenting out the `if zero_anchor_wanted`
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
    /// RED (verified in place): commenting out the `if zero_anchor_wanted`
    /// block makes this fail with `(40.0, 90.0)` too — proving the anchor
    /// follows the coordinate-space normalization, not just the unflipped
    /// case. Also genuinely discriminating on the flip normalization itself
    /// (re-measured T5r): dropping
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
        let extent = numeric_domain_union("y", "val", None, &batch, &HashMap::new(), &spec, false, None)
            .expect("val is Float64");
        assert_eq!(
            extent,
            (50.0, 80.0),
            "a non-layered mark_bar spec must not gain a zero-anchor from this \
             function; Python's explicit scale={{\"zero\": true}} injection owns \
             that case"
        );
    }

    // ── T5r: the two dropped guards ───────────
    //
    // The original rule fired for whatever channel was being resolved, as long
    // as a `Mark::Bar` layer bound it — no distinction between the bar's
    // VALUE axis and its POSITION axis, and no exclusion for a layer whose
    // own value channel already carries a paired span field. Both guards
    // mirror `_spec_build.py::_build_encoding_specs`'s single-chart
    // injection exactly (the value axis, the value channel's span partner
    // left unbound). Missing them let a
    // `displot(kind="hist")` histogram bar's `x`/`x2` bin-range POSITION
    // widen to zero — a live regression in the histogram bar's position
    // channel, not covered by the earlier tests (none of them exercised a
    // bar layer's position channel or a `y2`-bound bar).

    /// Verbatim `chart.to_dict()` JSON for `fm.displot(df, x="val",
    /// kind="hist", kde=True)` (T5r) — the exact
    /// shape measured live: a `bar` histogram layer whose
    /// POSITION channel is `x`/`x2` (the bin range) and whose VALUE channel
    /// is `y` (`count`). The sibling `kde` line layer reads a named
    /// `"kde_overlay"` output this test's fixture omits — its own field,
    /// `value`, unions to nothing and is silently skipped (`union_field`'s
    /// documented "ignore errors" behavior), so omitting it doesn't affect
    /// the assertion below.
    const DISPLOT_HIST_KDE_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"transforms":[{"type":"kde","field":"val","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":512,"cumulative":false,"shared_extent":false,"kernel":"gaussian","count_scale":{"kind":"sturges"},"name":"kde_overlay"},{"type":"bin","field":"val","nice":true,"cumulative":false,"shared_extent":false}],"layers":[{"mark":"bar","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"position":{"type":"identity"},"name":"histogram"},{"mark":"line","encoding":{"x":{"field":"value"},"y":{"field":"density"}},"data_source":"kde_overlay","position":{"type":"identity"},"name":"kde"}],"position":{"type":"identity"}}"#;

    /// Bin edges offset well clear of zero (50..89), so a domain that
    /// reaches zero can only be explained by the (wrong) position-channel
    /// anchor, never by the data.
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

    /// With neither the value-channel guard nor the span guard, this would
    /// return `(0.0, 89.0)` — the bar layer's POSITION channel (`x`/`x2`,
    /// the bin range) would get the anchor only its VALUE channel (`y`,
    /// `count`) is supposed to get. On `displot(x="val", kind="hist",
    /// kde=True)` with data in `[50, 89]`, the unguarded x axis would
    /// render `0…90`, squeezing every bar into the right ~44% of the
    /// panel.
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
            None,
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
    /// fm.Chart(df).mark_point().encode(x="cat", y="lo")` (T5r) — a bar
    /// layer whose own VALUE channel carries a paired span field, reachable
    /// from the public `Chart` API (`.encode(y2=...)` is a generically
    /// honored channel on any mark). Mirrors the rule
    /// `_spec_build.py::_build_encoding_specs`'s span-partner guard
    /// already applies to the single-chart case: an explicit `[y, y2]`
    /// extent is taken literally and must not be force-anchored to zero.
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

    /// A bar layer that binds its OWN value channel's paired
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
            numeric_domain_union("y", "lo", Some("hi"), &batch, &HashMap::new(), &spec, false, None)
                .expect("lo/hi are Float64");
        assert_eq!(
            extent,
            (50.0, 90.0),
            "a bar layer that binds its own value channel's paired span \
             field must not be force-anchored to zero"
        );
    }

    /// A layered line declaring `rise_from_zero` on its `mark_style`: its
    /// VALUE channel (`y`, count) end-to-end zero-anchors through the same
    /// rule a bar's does.
    const RISE_FROM_ZERO_LINE_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"line","encoding":{"x":{"field":"v"},"y":{"field":"count"}},"layers":[{"mark":"line","encoding":{"x":{"field":"v"},"y":{"field":"count"}},"mark_style":{"rise_from_zero":true}}]}"#;

    fn rise_from_zero_line_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("count", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![5.0, 6.0, 7.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    /// With counts in `[1, 3]`, an unguarded union would read `(1.0, 3.0)`;
    /// the declared riser's value axis zero-extends it to `(0.0, 3.0)`.
    #[test]
    fn rise_from_zero_line_value_axis_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(RISE_FROM_ZERO_LINE_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = rise_from_zero_line_batch();
        let extent = numeric_domain_union("y", "count", None, &batch, &HashMap::new(), &spec, false, None)
            .expect("count is Float64");
        assert_eq!(extent, (0.0, 3.0), "a declared line's value axis must zero-anchor");
    }

    /// The series' own observation axis (`x`) is the line's POSITION
    /// channel and is never anchored, whatever `rise_from_zero` says.
    #[test]
    fn rise_from_zero_line_position_channel_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(RISE_FROM_ZERO_LINE_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = rise_from_zero_line_batch();
        let extent = numeric_domain_union("x", "v", None, &batch, &HashMap::new(), &spec, false, None)
            .expect("v is Float64");
        assert_eq!(extent, (5.0, 7.0), "a line's observation axis must never gain the value-channel anchor");
    }

    /// A declared line whose own `Stack` cumulates its value onto x (a
    /// horizontal stack, `value_axis: Some(X)`) anchors ITS value axis, x —
    /// not y, where an unflipped undeclared line's value ordinarily rides.
    /// `resolve_curve_axis`'s stack term exists for exactly this: without
    /// it, the domain would anchor y while the builder (reading the same
    /// stamped stack metadata through `DrawCtx::series_curve_axis`) rises
    /// along x.
    #[test]
    fn a_declared_line_stacked_with_its_value_on_x_anchors_x_not_y() {
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::position::{PositionAdjust, StackAnchor, StackOffset};
        // Two rows share one category position ("g"="cat") and stack into
        // separate segments by "s" (the `by` grouping): row s1's top is 10,
        // row s2's cumulates to 30, with a base of 0.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, false),
                Field::new("g", DataType::Utf8, false),
                Field::new("s", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
                Arc::new(StringArray::from(vec!["cat", "cat"])),
                Arc::new(StringArray::from(vec!["s1", "s2"])),
            ],
        )
        .unwrap();
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "v".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "g".into(), ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                rise_from_zero: Some(true),
                ..Default::default()
            }),
            position: Some(PositionAdjust::Stack {
                by: Some("s".into()),
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: Some(StackValueAxis::X),
            }),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer]), ..Default::default() };
        let (extent, _) = numeric_domain_union_over(
            "x", "v", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, None, None, None,
        )
        .expect("the stacked v column resolves");
        assert_eq!(
            (extent.lo, extent.hi),
            (0.0, 30.0),
            "the stack's own value axis (x) must zero-anchor from its stacked (not raw) extent"
        );
    }

    /// A layer whose own `Stack` DECLINES to cumulate (no `by`, color or
    /// group column: `group_keys` finds nothing, `apply_stack` is a no-op,
    /// and `axis_previews`' own preview for either channel comes back
    /// `None`) still anchors the DECLARED stack axis — the vertical default
    /// here, since `value_axis` is unset. Reading the preview's presence
    /// instead of the declaration would answer this wrong: an absent
    /// preview on `y` would read as "the stack's value is on x", anchoring
    /// the wrong channel while the builder, which also sees no stamp
    /// (`DrawCtx::series_curve_axis`'s stack term is `false` whenever the
    /// batch metadata key is absent), still rises along y.
    #[test]
    fn a_declared_line_with_a_declined_vertical_stack_still_anchors_y() {
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::position::{PositionAdjust, StackAnchor, StackOffset};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cat", DataType::Utf8, false),
                Field::new("v", DataType::Float64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                rise_from_zero: Some(true),
                ..Default::default()
            }),
            position: Some(PositionAdjust::Stack {
                by: None,
                offset: StackOffset::Zero,
                anchor: StackAnchor::Top,
                value_axis: None,
            }),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer]), ..Default::default() };
        let (extent, _) = numeric_domain_union_over(
            "y", "v", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, None, None, None,
        )
        .expect("v resolves");
        assert_eq!(
            (extent.lo, extent.hi),
            (0.0, 2.0),
            "a declined vertical Stack must still anchor y, read from its declaration, not from whether a preview happened to exist"
        );
    }

    /// A layered declared line with its value AUTHORED on x (a horizontal
    /// orientation, `value_axis: Some(X)`) plus a point sharing its axes,
    /// under a chart `CoordFlip`: `LayerPrepared::flip_coords` swaps the
    /// declaration together with the encoding, so the value lands on
    /// DRAWN y (the count), and the observation lands on drawn x — the
    /// same axes the flat (unlayered) twin and the builder both draw.
    /// Reading the RAW authored `value_axis` instead would anchor drawn x,
    /// the observation, while the builder rises along drawn y.
    #[test]
    fn a_layered_declared_line_with_a_horizontal_value_axis_anchors_the_drawn_count_axis_under_a_flip() {
        use crate::spec::coord::CoordKind;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cnt", DataType::Float64, false),
                Field::new("obs", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![5.0, 6.0, 7.0])),
            ],
        )
        .unwrap();
        let axes = || Encoding {
            x: Some(EncodingSpec { field: "cnt".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "obs".into(), ..Default::default() }),
            ..Default::default()
        };
        let line = Layer {
            mark: Mark::Line,
            encoding: axes(),
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                rise_from_zero: Some(true),
                value_axis: Some(StackValueAxis::X),
                ..Default::default()
            }),
            ..Default::default()
        };
        let point = Layer { mark: Mark::Point, encoding: axes(), ..Default::default() };
        let spec = ChartSpec { layers: Some(vec![line, point]), coord: Some(CoordKind::Flip), ..Default::default() };
        let (y_extent, _) = numeric_domain_union_over(
            "y", "cnt", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, None, None, None,
        )
        .expect("cnt resolves");
        assert_eq!(
            (y_extent.lo, y_extent.hi),
            (0.0, 3.0),
            "drawn y (the authored value, moved there by the flip) must zero-anchor"
        );
        let (x_extent, _) = numeric_domain_union_over(
            "x", "obs", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, None, None, None,
        )
        .expect("obs resolves");
        assert_eq!(
            (x_extent.lo, x_extent.hi),
            (5.0, 7.0),
            "drawn x (the observation, moved there by the flip) must never anchor"
        );
    }

    /// The same shape, undeclared: the vertical default is what the flip
    /// moves, so drawn x (the authored y, moved there by the flip) anchors
    /// and drawn y (the authored x) does not — the mirror of the declared
    /// case above, guarding that an undeclared line's flip handling still
    /// works once the declared one reads the post-flip style.
    #[test]
    fn a_layered_undeclared_line_anchors_the_drawn_count_axis_under_a_flip() {
        use crate::spec::coord::CoordKind;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("obs", DataType::Float64, false),
                Field::new("cnt", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![5.0, 6.0, 7.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();
        let axes = || Encoding {
            x: Some(EncodingSpec { field: "obs".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "cnt".into(), ..Default::default() }),
            ..Default::default()
        };
        let line = Layer {
            mark: Mark::Line,
            encoding: axes(),
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                rise_from_zero: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let point = Layer { mark: Mark::Point, encoding: axes(), ..Default::default() };
        let spec = ChartSpec { layers: Some(vec![line, point]), coord: Some(CoordKind::Flip), ..Default::default() };
        let (x_extent, _) = numeric_domain_union_over(
            "x", "cnt", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, None, None, None,
        )
        .expect("cnt resolves");
        assert_eq!(
            (x_extent.lo, x_extent.hi),
            (0.0, 3.0),
            "drawn x (the undeclared default value, moved there by the flip) must zero-anchor"
        );
        let (y_extent, _) = numeric_domain_union_over(
            "y", "obs", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, None, None, None,
        )
        .expect("obs resolves");
        assert_eq!(
            (y_extent.lo, y_extent.hi),
            (5.0, 7.0),
            "drawn y (the observation, moved there by the flip) must never anchor"
        );
    }

    /// A caller's explicit `domain=` on a declared line's value axis wins
    /// at domain resolution itself — through `resolve_scales`, the real
    /// resolver every render call goes through. `build_from_scale_spec`
    /// resolves a scale that states a domain FROM that domain, without
    /// asking the data extent, so no cross-layer union and no zero-anchor
    /// eligibility check runs at all for this route; see
    /// `positional::a_scaled_declared_line_takes_its_scales_zero_not_the_line_anchor`
    /// for the sibling pin that reaches `numeric_domain_union_over`'s
    /// `LayerUnion::CarryingScale` early return (a domain-less caller
    /// scale).
    #[test]
    fn an_explicit_domain_wins_through_the_real_resolver() {
        use crate::spec::encoding::{ContinuousScaleCommon, Encoding, EncodingSpec, ScaleSpec};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, false),
                Field::new("count", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![12.0, 15.0, 18.0])),
            ],
        )
        .unwrap();
        let explicit_domain = ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: Some(vec![10.0, 20.0]),
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero: false,
            system_injected: false,
        };
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "v".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "count".into(), scale: Some(explicit_domain), ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                rise_from_zero: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer]), ..Default::default() };
        let theme = crate::layout::ThemeInputs::default();
        let (scales, _) =
            crate::render::scale_resolve::resolve_scales(&spec, &batch, (0.0, 100.0), (100.0, 0.0), &theme)
                .expect("the explicit domain resolves");
        assert_eq!(
            scales.y.data_domain(),
            Some((10.0, 20.0)),
            "a caller's explicit domain= must win, not zero-extend to (0, 20)"
        );
    }

    /// A flat (unlayered) spec whose value channel carries exactly the
    /// wire shape Python's `_mark_zero_anchor` emits for a channel with no
    /// prior scale (`{"type": "linear", "zero": true, "systemInjected":
    /// true}`) resolves a domain that includes 0, so the riser's baseline
    /// is the 0 pixel — the EXISTING generic `ScaleSpec::Linear { zero:
    /// true }` handling, mark-agnostic already, carries a declared line's
    /// riser on every door Python's injection reaches (the flat chart,
    /// here) once Python injects it for a line too.
    #[test]
    fn a_flat_spec_carrying_pythons_injected_zero_anchor_rises_from_zero() {
        use crate::spec::encoding::{ContinuousScaleCommon, Encoding, EncodingSpec, ScaleSpec};
        let injected: ScaleSpec = serde_json::from_str(r#"{"type":"linear","zero":true,"systemInjected":true}"#)
            .expect("Python's exact wire shape must deserialize");
        assert_eq!(
            injected,
            ScaleSpec::Linear {
                common: ContinuousScaleCommon {
                    domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false,
                },
                nice: false,
                zero: true,
                system_injected: true,
            },
            "agree the exact wire shape with _mark_zero_anchor before trusting it resolves"
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, false),
                Field::new("count", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![5.0, 8.0, 10.0])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "v".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "count".into(), scale: Some(injected), ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec { rise_from_zero: Some(true), ..Default::default() }),
            ..Default::default()
        };
        let theme = crate::layout::ThemeInputs::default();
        let (scales, _) =
            crate::render::scale_resolve::resolve_scales(&spec, &batch, (0.0, 100.0), (100.0, 0.0), &theme)
                .expect("the injected scale resolves");
        assert_eq!(
            scales.y.value_baseline_px(100.0, 0.0),
            scales.y.to_pixel_f64(0.0).unwrap(),
            "0 is in the resolved domain, so the riser's baseline is exactly the 0 pixel"
        );
    }

    // ── T5r: discriminating flip test, the
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
            numeric_domain_union("x", "count", None, &batch, &HashMap::new(), &spec, false, None)
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
            None,
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
    /// (T5r) — a REAL single-bar-layer
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
            numeric_domain_union("x", "feature", None, &batch, &HashMap::new(), &spec, false, None)
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
        let (extent, _tally) = numeric_domain_union_over(
            "y",
            "v",
            None,
            &batch,
            &HashMap::new(),
            &spec,
            false,
            LayerUnion::CarryingScale(&scale),
            None,
            None,
            None,
        )
        .expect("an all-null Float64 column is readable");
        assert_eq!((extent.lo, extent.hi), (0.0, 1.0));
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
            let (extent, _tally) = numeric_domain_union_over(
                "y",
                "v",
                None,
                &batch,
                &HashMap::new(),
                &spec,
                false,
                union,
                None,
                None,
                None,
            )
            .expect("an all-null Int64 column is readable");
            assert_eq!((extent.lo, extent.hi), (0.0, 1.0));
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
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false, None)
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
    /// RED (verified in place): moving the `if zero_anchor_wanted { ... }`
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
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false, None)
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
    /// T5r). Hand-authored rather than captured from a live call on
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
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false, None)
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
    /// zero-anchor when its own field never contributed a value.
    const PHANTOM_FIELD_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"point","encoding":{"y":{"field":"existing"}}},{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"phantom"}}}]}"#;

    /// RED (verified in place): replacing `value_field_contributed` with
    /// `layer_fields[0].is_some()` ("declared" instead of
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
            numeric_domain_union("y", "existing", None, &batch, &HashMap::new(), &spec, false, None)
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
    /// before the seed answers `["b","c","a"]`; a union that never reads the
    /// RHS layer's own column at all answers `["c","a"]`.
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

    /// The flip is per LAYER, not per chart: a
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
    /// and says so by skipping rather than by refusing the render.
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
        let extent = numeric_domain_union("y", "ci_lower", Some("ci_upper"), &raw, &outputs, &spec, false, None)
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
        let (extent, _tally) = numeric_domain_union_over(
            "y",
            "ci_lower",
            Some("ci_upper"),
            &raw,
            &outputs,
            &spec,
            false,
            LayerUnion::CarryingScale(&scale),
            None,
            None,
            None,
        )
        .expect("every column is Float64");
        assert_eq!((extent.lo, extent.hi), (-0.2, 20.0));
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
        let extent = numeric_domain_union("y", "v", None, &raw, &outputs, &spec, false, None)
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
        let extent = numeric_domain_union("y", "v", None, &raw, &outputs, &spec, true, None)
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
        let extent = numeric_domain_union("y", "v", None, &merged, &outputs, &spec, false, None)
            .expect("every column is Float64");
        assert_eq!(extent, (1.0, 4.0));
    }

    /// A layer whose batch has no rows draws nothing and widens nothing; the
    /// reader does not go looking for the field elsewhere.
    #[test]
    fn an_empty_layer_batch_contributes_nothing() {
        let spec = spec_from(OPERAND_BRANCHES_SPEC_JSON);
        let (merged, outputs) = operand_batches(&[]);
        let extent = numeric_domain_union("y", "v", None, &merged, &outputs, &spec, false, None)
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
        let extent = numeric_domain_union("y", "v", None, &merged, &outputs, &spec, false, None)
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
        let extent = numeric_domain_union("x", "g", None, &merged, &outputs, &spec, false, None)
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
            &crate::transform::run::OutputLineages::new(),
            (0.0, 300.0),
            &spec,
            &crate::layout::geometry::TextEdgeDemand::default(),
            false,
            None,
            false,
            None,
            None,
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
        let x = numeric_domain_union("x", "x", None, &raw, &outputs, &spec, false, None).expect("Float64 columns");
        let y = numeric_domain_union("y", "y", None, &raw, &outputs, &spec, false, None).expect("Float64 columns");
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
        let extent = numeric_domain_union("x", "a", None, &filtered, &outputs, &spec, false, None)
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
        let extent = numeric_domain_union("x", "a", None, &batch, &HashMap::new(), &spec, false, None)
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
        let extent = numeric_domain_union("x", "a", None, &batch, &HashMap::new(), &spec, false, None)
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

    /// A text reads no color it does not own: neither the chart's nor one
    /// Python listed on it as inherited, so its rows (here a `zz` no mark
    /// draws) never reach the color scale. A text's own color reads.
    #[test]
    fn aux_reads_take_no_color_a_text_does_not_own() {
        use crate::spec::encoding::{Encoding, EncodingSpec, AUXILIARY_SCALE_CHANNELS};
        use crate::spec::layer::InheritedChannel;
        let color = |field: &str| Encoding { color: Some(EncodingSpec { field: field.into(), ..Default::default() }), ..Default::default() };
        let layers = vec![
            Layer { mark: Mark::Point, ..Default::default() },
            Layer { mark: Mark::Text, ..Default::default() },
            Layer { mark: Mark::Text, encoding: color("c"), inherited_channels: vec![InheritedChannel::Color], ..Default::default() },
            Layer { mark: Mark::Text, encoding: color("c"), ..Default::default() },
        ];
        let chart = color("c");
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("c", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["zz"]))],
        )
        .unwrap();
        let channel = AUXILIARY_SCALE_CHANNELS.into_iter().find(|c| c.name == "color").expect("color");
        let outputs = HashMap::new();
        let reads = aux_reads(&layers, &chart, channel, &rows, &outputs);
        assert_eq!(reads.iter().map(AuxRead::layer).collect::<Vec<_>>(), [0, 3]);
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

    /// A contrast-ink text's color read never reaches the reads list, even
    /// when its own rows (a separate `data_source`) hold a category the
    /// drawing layer's cells lack: the domain `aux_category_union` builds
    /// from these reads is the drawing layer's alone, so the text's wider
    /// rows cannot widen it.
    #[test]
    fn aux_reads_excludes_a_contrast_text_even_when_its_rows_hold_a_wider_category() {
        use crate::spec::encoding::{Encoding, EncodingSpec, AUXILIARY_SCALE_CHANNELS};
        use crate::spec::mark_style::MarkKwargsSpec;
        let color = |field: &str| Encoding { color: Some(EncodingSpec { field: field.into(), ..Default::default() }), ..Default::default() };
        let contrast_fill = Some(MarkKwargsSpec { fill: Some("theme:contrast".into()), ..Default::default() });
        let layers = vec![
            Layer { mark: Mark::Rect, encoding: color("c"), ..Default::default() },
            Layer {
                mark: Mark::Text,
                encoding: color("c"),
                mark_style: contrast_fill,
                data_source: Some("labels".into()),
                ..Default::default()
            },
        ];
        let chart = color("c");
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("c", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["A", "B"]))],
        )
        .unwrap();
        // The contrast text's own rows hold "Z", a category the cells
        // (the rect, over `rows`) never draw.
        let mut outputs = HashMap::new();
        outputs.insert(
            "labels".to_string(),
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("c", DataType::Utf8, true)])),
                vec![Arc::new(StringArray::from(vec!["Z"]))],
            )
            .unwrap(),
        );
        let channel = AUXILIARY_SCALE_CHANNELS.into_iter().find(|c| c.name == "color").expect("color");
        let reads = aux_reads(&layers, &chart, channel, &rows, &outputs);
        assert_eq!(
            reads.iter().map(AuxRead::layer).collect::<Vec<_>>(),
            [0],
            "only the rect's read reaches the domain; the contrast text's wider rows are excluded"
        );
    }

    /// A line with its own `stroke_width` yields an inherited `size` to it:
    /// its read never reaches the size domain, so a point beside it (also
    /// inheriting the chart's `size`, drawing it) establishes the size
    /// domain alone, narrow to the point's own rows rather than widened by
    /// the line's.
    #[test]
    fn aux_reads_excludes_a_line_that_yields_an_inherited_size_to_its_own_stroke_width() {
        use crate::spec::encoding::{Encoding, EncodingSpec, AUXILIARY_SCALE_CHANNELS};
        use crate::spec::mark_style::MarkKwargsSpec;
        let own_stroke_width = Some(MarkKwargsSpec { stroke_width: Some(3.0), ..Default::default() });
        let layers = vec![
            Layer { mark: Mark::Line, mark_style: own_stroke_width, ..Default::default() },
            Layer { mark: Mark::Point, ..Default::default() },
        ];
        let chart = Encoding { size: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() };
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)])),
            vec![Arc::new(arrow::array::Float64Array::from(vec![1.0]))],
        )
        .unwrap();
        let channel = AUXILIARY_SCALE_CHANNELS.into_iter().find(|c| c.name == "size").expect("size");
        let outputs = HashMap::new();
        let reads = aux_reads(&layers, &chart, channel, &rows, &outputs);
        assert_eq!(reads.iter().map(AuxRead::layer).collect::<Vec<_>>(), [1], "the yielding line's read is excluded");
    }

    /// Every layer yielding `size` to a width it owns leaves the channel
    /// with no drawing layer at all.
    #[test]
    fn aux_reads_is_empty_when_every_layer_yields_size() {
        use crate::spec::encoding::{Encoding, EncodingSpec, AUXILIARY_SCALE_CHANNELS};
        use crate::spec::mark_style::MarkKwargsSpec;
        let own_stroke_width = |w: f64| Some(MarkKwargsSpec { stroke_width: Some(w), ..Default::default() });
        let layers = vec![
            Layer { mark: Mark::Line, mark_style: own_stroke_width(3.0), ..Default::default() },
            Layer { mark: Mark::Line, mark_style: own_stroke_width(5.0), ..Default::default() },
        ];
        let chart = Encoding { size: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() };
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)])),
            vec![Arc::new(arrow::array::Float64Array::from(vec![1.0]))],
        )
        .unwrap();
        let channel = AUXILIARY_SCALE_CHANNELS.into_iter().find(|c| c.name == "size").expect("size");
        let outputs = HashMap::new();
        let reads = aux_reads(&layers, &chart, channel, &rows, &outputs);
        assert!(reads.is_empty(), "both lines yield size to their own width: {} read(s)", reads.len());
    }

    /// A segment with no arrowhead reads `fill_opacity` to fill nothing (it
    /// has no fill without `arrow=True`), so its read never reaches the
    /// fill_opacity domain.
    #[test]
    fn aux_reads_excludes_an_arrowless_segments_fill_opacity() {
        use crate::spec::encoding::{Encoding, EncodingSpec, AUXILIARY_SCALE_CHANNELS};
        let encoding = Encoding { fill_opacity: Some(EncodingSpec { field: "o".into(), ..Default::default() }), ..Default::default() };
        let layers = vec![Layer { mark: Mark::Segment, encoding: encoding.clone(), ..Default::default() }];
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("o", DataType::Float64, true)])),
            vec![Arc::new(arrow::array::Float64Array::from(vec![0.5]))],
        )
        .unwrap();
        let channel = AUXILIARY_SCALE_CHANNELS.into_iter().find(|c| c.name == "fill_opacity").expect("fill_opacity");
        let chart = Encoding::default();
        let outputs = HashMap::new();
        let reads = aux_reads(&layers, &chart, channel, &rows, &outputs);
        assert!(reads.is_empty(), "an arrowless segment fills nothing: {} read(s)", reads.len());
    }

    /// A faceted-SHARED (`include_final: true`) read of
    /// a `Log` channel reads `FINAL_OUTPUT_KEY` ALONE when it holds the
    /// field, not primary-PLUS-FINAL, so a panel's own off-side rows are
    /// not counted twice against the whole-frame total. This panel's `v`
    /// is `[-5, 1, 10]` (1 off-side); the all-panels `FINAL` batch is
    /// `[-5, 1, 10, 7, 100, -3]` (2 off-side, including this panel's own
    /// rows verbatim, as a real faceted concat would produce). The tally's
    /// `rows()` is exactly 2 (the whole frame's), not 3 (this panel's 1
    /// plus the global's 2 — double-counting this panel's own rows against
    /// the whole frame).
    #[test]
    fn a_shared_facet_log_tally_does_not_double_count_the_panel_against_final() {
        use arrow::array::Float64Array;
        let panel = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0]))],
        )
        .unwrap();
        let global = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0, 7.0, 100.0, -3.0]))],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), global);
        let spec = ChartSpec::default();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("y", "v", None, &panel, &outputs, &spec, true, LayerUnion::Every, side, None, None)
            .expect("v is Float64 in this fixture");
        assert_eq!(tally.rows(), 2, "the whole frame has 2 off-side rows, not 3 (the panel's 1 double-counted): {tally:?}");
    }

    /// A statistic's recorded caller column on a log axis whose values
    /// equal a primitive layer's own column on that axis counts its
    /// off-side rows once: the folded column goes through the same
    /// content-deduplicated tally the primitive read already fed. Rejects
    /// adding the statistic's rows on top of the primitive's.
    #[test]
    fn a_recorded_statistic_input_equal_to_a_primitive_column_counts_once() {
        use arrow::array::Float64Array;
        let column = || -> arrow::array::ArrayRef { Arc::new(Float64Array::from(vec![0.0, -1.0, 1.0, 10.0])) };
        let panel = RecordBatch::try_new(Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])), vec![column()]).unwrap();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, mut tally) = numeric_domain_union_over("x", "v", None, &panel, &HashMap::new(), &ChartSpec::default(), false, LayerUnion::Every, side, None, None)
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 2, "the primitive read alone: {tally:?}");
        let mut lineages: OutputLineages = HashMap::new();
        lineages.insert(
            "hist".to_string(),
            crate::transform::run::OutputLineage {
                swarm: None,
                statistic_input: Some(panel.clone()),
                lowered: Vec::new(),
                written_groupby: Vec::new(),
                source: panel.schema(),
                log_input: HashMap::from([((crate::spec::layer::PositionalAxis::X, "v".to_string()), column())]),
                steps: Vec::new(),
            },
        );
        fold_log_input_into_tally(&lineages, crate::spec::layer::PositionalAxis::X, side, &mut tally).expect("v is Float64 here");
        assert_eq!(tally.rows(), 2, "the statistic's copy of the same column counts once: {tally:?}");
    }

    /// A recorded caller column [`sided_min_max_f64`] cannot fold —
    /// `coerce_to_float64` accepts `Float16` (the transform step that
    /// recorded it already ran successfully over that column), but
    /// `sided_min_max_f64` has no `Float16` arm — surfaces as a typed
    /// error naming the column, not a silent skip that would under-count
    /// the left-out total.
    #[test]
    fn fold_log_input_into_tally_refuses_a_column_sided_min_max_f64_cannot_fold() {
        use arrow::array::Float64Array;
        let f64_column: arrow::array::ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let column = arrow::compute::cast(&f64_column, &DataType::Float16).expect("Float64 casts to Float16");
        let mut lineages: OutputLineages = HashMap::new();
        lineages.insert(
            "stat".to_string(),
            crate::transform::run::OutputLineage {
                swarm: None,
                statistic_input: None,
                lowered: Vec::new(),
                written_groupby: Vec::new(),
                source: Arc::new(Schema::new(vec![Field::new("v", DataType::Float16, false)])),
                log_input: HashMap::from([((crate::spec::layer::PositionalAxis::X, "v".to_string()), column)]),
                steps: Vec::new(),
            },
        );
        let mut tally = LeftOutTally::new();
        let side = Some(crate::scale::log::LogSide::Positive);
        let err = fold_log_input_into_tally(&lineages, crate::spec::layer::PositionalAxis::X, side, &mut tally)
            .expect_err("Float16 is not among sided_min_max_f64's supported dtypes");
        assert!(
            matches!(&err, RenderError::UnsupportedDtype { field, .. } if field == "v"),
            "expected UnsupportedDtype naming 'v'; got {err:?}"
        );
        assert_eq!(tally.rows(), 0, "a refused column folds nothing into the tally: {tally:?}");
    }

    /// An unlayered spec with no `AdjustedRows` (`adjusted: None` — no
    /// position adjustment applies on this axis) reads `primary_batch`
    /// directly (the `Chart | None` arm's whole-frame tally rule, no
    /// `FINAL_OUTPUT_KEY` published): `_over` counts every off-side row of
    /// WHATEVER unadjusted batch it is given, with no transform outputs at
    /// all.
    #[test]
    fn an_unlayered_tally_counts_every_off_side_row_of_the_batch_it_is_given() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![5.0, -1.0, -2.0, 10.0, -3.0]))],
        )
        .unwrap();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("x", "v", None, &batch, &HashMap::new(), &ChartSpec::default(), false, LayerUnion::Every, side, None, None)
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 3, "{tally:?}");
    }

    /// The same two batches, but INDEPENDENT facet resolution
    /// (`include_final: false`): the panel's own extent still comes from
    /// its own batch alone (unchanged, independent-panel domain), but the
    /// tally STILL reads the whole frame's `FINAL` column directly,
    /// whatever its facet resolution — so an independent
    /// panel's warning reports the SAME whole-frame count a shared panel's
    /// would, not just its own panel's.
    #[test]
    fn an_independent_facet_log_tally_still_reads_the_whole_frame() {
        use arrow::array::Float64Array;
        let panel = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0]))],
        )
        .unwrap();
        let global = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0, 7.0, 100.0, -3.0]))],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), global);
        let spec = ChartSpec::default();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (extent, tally) = numeric_domain_union_over("y", "v", None, &panel, &outputs, &spec, false, LayerUnion::Every, side, None, None)
            .expect("v is Float64 in this fixture");
        assert_eq!((extent.lo, extent.hi), (1.0, 10.0), "the panel's own on-side extent stays independent");
        assert_eq!(tally.rows(), 2, "the tally reads the whole frame regardless of facet resolution: {tally:?}");
    }

    /// An unlayered spec whose
    /// bound field lives only in a named output (a composite mark's `box`,
    /// say) was read EXTENT-ONLY in that loop, so its off-side rows never
    /// reached the tally — a reachable-by-spec silent drop, not yet hit by
    /// any live Python lowering (every composite mark today counts via its
    /// layered form), but a latent gap in a loop this function's sibling
    /// readers all tally.
    #[test]
    fn an_unlayered_reads_named_output_counts_its_off_side_rows() {
        use arrow::array::Float64Array;
        let primary = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("other", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![0.0]))],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(
            "box".to_string(),
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)])),
                vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0]))],
            )
            .unwrap(),
        );
        let spec = ChartSpec::default();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (extent, tally) = numeric_domain_union_over("y", "f", None, &primary, &outputs, &spec, false, LayerUnion::Every, side, None, None)
            .expect("f is Float64 in the 'box' output");
        assert_eq!((extent.lo, extent.hi), (1.0, 10.0));
        assert_eq!(tally.rows(), 1, "the one off-side row (-5) must be counted: {tally:?}");
    }

    // ── `AdjustedRows`: the position-adjustment tally ───────────────────────
    //
    // `position::adjusted_rows`'s pairing of a layer's whole-frame caller
    // batch with its own preview, folded through `fold_adjusted_tally` /
    // `LeftOutTally::count_adjusted`: the one mechanism for pairing
    // adjusted and unadjusted rows that stays correct across different
    // batches and row orders, rather than matching them up positionally.

    /// A caller value of exactly `0.0` — off-side on the positive side,
    /// never on-side — "lifted" by a position adjustment (a stack's base,
    /// say) to an on-side value: the row stays counted via the CALLER'S
    /// OWN off-side extent, not dropped just because its adjusted
    /// counterpart now looks on-side. `count_adjusted`'s `moved` fold never
    /// touches it either (it only tracks a caller row that starts ON-side
    /// and is pushed OFF by the adjustment — the opposite direction), so
    /// the row's single count comes from `caller_extent.left_out` alone.
    #[test]
    fn an_unlayered_adjusted_tally_still_counts_a_lifted_zero_via_the_callers_own_extent() {
        let caller = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![0.0, 5.0, -2.0]))],
        )
        .unwrap();
        let lifted = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![3.0, 5.0, -2.0]))],
        )
        .unwrap();
        let adjusted = AdjustedRows { unadjusted: &caller, adjusted: lifted, base: None };
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("y", "v", None, &caller, &HashMap::new(), &ChartSpec::default(), false, LayerUnion::Every, side, None, Some(&adjusted))
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 2, "both the lifted zero and the plain negative row must stay counted: {tally:?}");
    }

    /// The opposite direction from the lifted-zero case above: a caller
    /// row that starts ON-side (`5.0`) is pushed OFF-side by the
    /// adjustment itself (to `-3.0`) — a tally that reads only the
    /// caller's own rows (ignoring what the adjustment moved) would see
    /// no off-side row at all here (both caller values are on-side) and
    /// undercount to 0; the correct count is 1, the moved row.
    #[test]
    fn an_unlayered_adjusted_tally_counts_a_row_the_adjustment_moves_off_side() {
        let caller = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![5.0, 8.0]))],
        )
        .unwrap();
        let moved = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-3.0, 8.0]))],
        )
        .unwrap();
        let adjusted = AdjustedRows { unadjusted: &caller, adjusted: moved, base: None };
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("y", "v", None, &caller, &HashMap::new(), &ChartSpec::default(), false, LayerUnion::Every, side, None, Some(&adjusted))
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 1, "a caller-rows-only tally would read 0 here (no caller row is off-side); the adjustment-moved row must still be counted: {tally:?}");
    }

    /// The identical caller column, read once PLAINLY (a fit sibling layer
    /// drawing the same raw field) and once through a jittered layer's own
    /// `AdjustedRows` pair, dedupes to the caller's own off-side count —
    /// `LeftOutTally`'s content-dedup (`count`'s `ArrayData` equality check,
    /// mirrored by `count_adjusted`'s own) treats the two reads as the same
    /// column, not two different ones to sum.
    #[test]
    fn a_jittered_layers_adjusted_tally_dedupes_with_a_plain_siblings_read_of_the_same_caller_column() {
        let caller = Float64Array::from(vec![-5.0, 1.0, 10.0]);
        let side = crate::scale::log::LogSide::Positive;
        let tally = std::cell::RefCell::new(LeftOutTally::new());
        let extent = sided_min_max_f64(&caller, Some(side)).expect("v is Float64 here");
        tally.borrow_mut().count(&caller, extent);
        assert_eq!(tally.borrow().rows(), 1, "the plain read alone counts the one off-side row");

        let unadjusted_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(caller.clone())],
        )
        .unwrap();
        // The jitter moves -5.0 to -3.0 — still off-side, so nothing new
        // moves across the side either; this isolates the dedup itself from
        // any "moved" contribution.
        let adjusted_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-3.0, 1.0, 10.0]))],
        )
        .unwrap();
        let rows = AdjustedRows { unadjusted: &unadjusted_batch, adjusted: adjusted_batch, base: None };
        fold_adjusted_tally(&tally, &rows, "v", side).expect("v is Float64 here");
        assert_eq!(
            tally.borrow().rows(),
            1,
            "the identical caller column, read once plainly and once through the jittered layer's own AdjustedRows, must dedupe to one count: {:?}",
            tally.borrow()
        );
    }

    /// A stacked layer's base column (`AdjustedRows::base`) feeds the
    /// EXTENT only, never the tally: `numeric_domain_union_over`'s paired-
    /// span read takes `TallyRead::ExtentOnly` exactly when the paired
    /// field names the stack's own base column, so a base value that sits
    /// off-side (every stacked segment's own base, by construction, since a
    /// stack starts at zero or the prior segment's own top) never inflates
    /// the off-side count — only the stacked VALUE column's own rows (read
    /// separately, through the value field's own `TallyRead`) can.
    #[test]
    fn an_unlayered_stacked_base_column_feeds_extent_only_never_the_tally() {
        let caller = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, false),
                Field::new("base", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![5.0, 8.0])),
                Arc::new(Float64Array::from(vec![-1.0, -1.0])),
            ],
        )
        .unwrap();
        let stacked = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, false),
                Field::new("base", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![4.0, 7.0])),
                Arc::new(Float64Array::from(vec![-1.0, -1.0])),
            ],
        )
        .unwrap();
        let adjusted = AdjustedRows { unadjusted: &caller, adjusted: stacked, base: Some("base") };
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("y", "v", Some("base"), &caller, &HashMap::new(), &ChartSpec::default(), false, LayerUnion::Every, side, None, Some(&adjusted))
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 0, "the base column's own off-side rows must never reach the tally — extent only: {tally:?}");
    }

    // ── The layered door end to end: axis_previews -> AxisReads::tally ->
    // TallyRead::Adjusted -> fold_adjusted_tally ────────────────────────────

    /// A jittered layer's off-side tally, reached through the full layered
    /// pipeline (`axis_previews` builds its `AdjustedRows`, `AxisReads`
    /// pairs it with the layer via `tally()`, `read()` dispatches
    /// `TallyRead::Adjusted` into `fold_adjusted_tally`), counts the SAME
    /// raw `v` column a plain sibling layer reads from the chart's own
    /// batch — the two content-dedupe into ONE count, not two (the
    /// jittered preview's own rewritten column, read verbatim, would never
    /// equal the raw one by content and so would never dedupe against
    /// it). `v = [-5, 1, 10]` has exactly one off-side row (`-5`).
    #[test]
    fn a_jittered_layer_tallies_the_same_raw_rows_a_plain_sibling_reads() {
        use crate::spec::position::{JitterAxis, PositionAdjust};
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0]))],
        )
        .unwrap();
        let jitter_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() },
            position: Some(PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: Some(7) }),
            ..Default::default()
        };
        let plain_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() },
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![jitter_layer, plain_layer]), ..Default::default() };
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("x", "v", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, side, None, None)
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 1, "the jittered layer's tally must count the SAME raw row the plain sibling counts, not a second one of the jittered preview's own (different) content: {tally:?}");
    }

    /// The same dedupe holds between two layers that BOTH jitter the same
    /// field, with different seeds (so their own adjusted columns differ
    /// from each other too): both tallies still read the layer's own
    /// `AdjustedRows.unadjusted` — the same raw `v` column — so the
    /// whole-frame count stays 1 regardless of which layer's randomness is
    /// read.
    #[test]
    fn two_jittered_layers_on_the_same_field_still_dedupe_to_one_count() {
        use crate::spec::position::{JitterAxis, PositionAdjust};
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0, 10.0]))],
        )
        .unwrap();
        let jitter_layer = |seed: u64| Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() },
            position: Some(PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: Some(seed) }),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![jitter_layer(7), jitter_layer(11)]), ..Default::default() };
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("x", "v", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, side, None, None)
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 1, "two differently-seeded jittered layers on the same field must still dedupe to the one raw off-side row: {tally:?}");
    }

    /// The same mechanism holds on the NEGATIVE side: a jittered layer's
    /// tally, reached through the full layered pipeline, still counts the
    /// raw column a plain sibling reads, not the jittered preview's own
    /// rewritten one.
    #[test]
    fn a_jittered_layer_tallies_the_same_raw_rows_on_the_negative_side() {
        use crate::spec::position::{JitterAxis, PositionAdjust};
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![5.0, -1.0, -10.0]))],
        )
        .unwrap();
        let jitter_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() },
            position: Some(PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: Some(3) }),
            ..Default::default()
        };
        let plain_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "v".into(), ..Default::default() }), ..Default::default() },
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![jitter_layer, plain_layer]), ..Default::default() };
        let side = Some(crate::scale::log::LogSide::Negative);
        let (_, tally) = numeric_domain_union_over("x", "v", None, &batch, &HashMap::new(), &spec, false, LayerUnion::Every, side, None, None)
            .expect("v is Float64 here");
        assert_eq!(tally.rows(), 1, "the one off-side row (5.0) counted once, not doubled against the jittered preview's own content: {tally:?}");
    }

    /// A layer's off-side tally gets NO `AdjustedRows` pair when
    /// `whole_frame` is FINAL (FINAL holds the drawn field "y", so
    /// `whole_frame_batch` selects it over `primary_batch`) but FINAL's
    /// own preview fails: `primary_batch` carries every column `Stack`
    /// needs (its category field "x", its value field "y", its by-group
    /// field "g"), so its own preview succeeds, but the published FINAL
    /// batch carries no "x" column at all, so `apply_stack` fails on it
    /// with `PositionAdjustFailed` and `adjusted_across_panels` returns
    /// `Ok(None)` for FINAL's whole preview. A regression that paired
    /// `whole_frame` with `rows` (`primary_batch`'s own preview, a
    /// DIFFERENT batch) instead of giving up would build a row-misaligned
    /// `AdjustedRows` here.
    #[test]
    fn axis_previews_builds_no_pair_when_finals_own_preview_fails_though_primarys_own_succeeds() {
        use crate::spec::position::{PositionAdjust, StackAnchor, StackOffset};
        use crate::spec::encoding::{Encoding, EncodingSpec};
        let primary = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Utf8, false),
                Field::new("y", DataType::Float64, false),
                Field::new("g", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["p", "p"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let final_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("y", DataType::Float64, false),
                Field::new("g", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), final_batch);
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                ..Default::default()
            },
            position: Some(PositionAdjust::Stack { by: Some("g".into()), offset: StackOffset::Zero, anchor: StackAnchor::Top, value_axis: None }),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer]), ..Default::default() };
        let layers = spec.layers.as_deref().unwrap();
        let (_, _, adjusted) = axis_previews(&spec, layers, "y", &primary, &outputs, true, None)
            .expect("the layer's own panel rows still stack");
        assert_eq!(adjusted.len(), 1);
        assert!(adjusted[0].is_none(), "FINAL's own preview failed, so no AdjustedRows pair must be built: {:?}", adjusted[0].is_some());
    }

    /// The same shape, every row off-side: today's silent `[1, base]` axis
    /// (no row ever counted) must become a refusal once the output's rows
    /// are tallied.
    #[test]
    fn an_unlayered_reads_all_off_side_output_refuses() {
        use arrow::array::Float64Array;
        let primary = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("other", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![0.0]))],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(
            "box".to_string(),
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)])),
                vec![Arc::new(Float64Array::from(vec![-5.0, -7.0]))],
            )
            .unwrap(),
        );
        let spec = ChartSpec::default();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (extent, tally) = numeric_domain_union_over("y", "f", None, &primary, &outputs, &spec, false, LayerUnion::Every, side, None, None)
            .expect("f is Float64 in the 'box' output");
        assert_eq!(tally.rows(), 2);
        assert!(!tally.any_on_side());
        let err = crate::render::scale_resolve::positional::resolve_log_extent("y", Some("f"), false, crate::scale::log::LogSide::Positive, (extent.lo, extent.hi), &tally, 10.0, false, &mut Vec::new())
            .expect_err("nothing on side anywhere refuses");
        assert!(matches!(err, RenderError::LogScaleNoValueOnSide { .. }), "{err:?}");
    }

    /// An output whose content is equal to `FINAL`'s is counted once, not
    /// twice (`LeftOutTally`'s content dedupe).
    #[test]
    fn an_output_equal_to_final_is_counted_once() {
        use arrow::array::Float64Array;
        let primary = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("other", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![0.0]))],
        )
        .unwrap();
        let f_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![-5.0, 1.0]))],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert("box".to_string(), f_batch.clone());
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), f_batch);
        let spec = ChartSpec::default();
        let side = Some(crate::scale::log::LogSide::Positive);
        let (_, tally) = numeric_domain_union_over("y", "f", None, &primary, &outputs, &spec, false, LayerUnion::Every, side, None, None)
            .expect("f is Float64");
        assert_eq!(tally.rows(), 1, "'box' and FINAL are equal by content: counted once, not twice: {tally:?}");
    }

    /// A two-layer, faceted, shared-x (`include_final: true`) spec whose
    /// establishing layer (index 0) binds `x`/`g` under `Dodge`.
    /// `primary_batch` is panel `"A"`'s own one row (`x=1`, one dodge
    /// group, `apply_dodge`'s `n_groups <= 1` no-op — `Int64` untouched);
    /// `FINAL` is the whole (unpartitioned) 5-row batch, whose own
    /// per-panel preview internally reconciles panel `"B"`'s two-group
    /// dodge to `Float64`. `axis_previews`'s `LayerSource::Chart` arm then
    /// concatenates panel `"A"`'s own-panel preview (`Int64`, 1 row) with
    /// that already-reconciled `FINAL` preview (`Float64`, 5 rows) — a
    /// second `concat_by_name` call, at the point `AxisReads::over` hands
    /// the establishing layer's seed read a `LayerSource::Batch` built
    /// from exactly this concatenation. A positional `concat_batches`/
    /// `align_to_schema` cast here would either error or silently drop
    /// the mismatched preview; the numeric union instead covers the full
    /// dodged span.
    #[test]
    fn layered_faceted_dodge_seed_read_reconciles_to_float64() {
        use arrow::array::{Float64Array, Int64Array};
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("f", DataType::Utf8, false),
        ]));
        // Panel A's own no-op x (1) sits INSIDE panel B's dodged span
        // (0.75..2.25), the same fixture shape `position.rs`'s own
        // `int64_dodge_facet_batch` uses, so the reconciled extent and a
        // cast-truncated one are never confusable with "panel A alone
        // happened to set the extent".
        let panel_a = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(StringArray::from(vec!["same"])),
                Arc::new(StringArray::from(vec!["A"])),
            ],
        )
        .unwrap();
        let whole_frame = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![0.0, 10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["same", "p", "q", "p", "q"])),
                Arc::new(StringArray::from(vec!["A", "B", "B", "B", "B"])),
            ],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), whole_frame);

        let dodge_layer = || Layer {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), ..Default::default() }),
                ..Default::default()
            },
            position: Some(crate::spec::position::PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 }),
            ..Default::default()
        };
        let spec = ChartSpec {
            layers: Some(vec![dodge_layer(), dodge_layer()]),
            facet: Some(crate::layout::facet::FacetSpec {
                field: "f".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                repeat: Vec::new(),
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            ..Default::default()
        };

        let (extent, _tally) =
            numeric_domain_union_over("x", "x", None, &panel_a, &outputs, &spec, true, LayerUnion::Every, None, None, None)
                .expect("the reconciled Float64 preview must be readable");
        assert!((extent.lo - 0.75).abs() < 1e-9, "lo={}", extent.lo);
        assert!((extent.hi - 2.25).abs() < 1e-9, "hi={}", extent.hi);
    }

    /// `layer_sort_rows` pairs a layer's pre-statistic rows (`batch`) with
    /// its final rows (`fallback`): a data-aware sort whose value field is
    /// the establishing layer's own aggregate OUTPUT (`y="mean(y)"`) needs
    /// the fallback, since `batch` never carries that output column --
    /// `resolve_sort_batch`'s own test proves the sort actually reads it.
    #[test]
    fn layer_sort_rows_pairs_pre_statistic_rows_with_the_layers_final_rows() {
        let primary = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("y", DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a", "b"])),
                    Arc::new(arrow::array::Float64Array::from(vec![10.0, 20.0, 5.0])),
                ],
            )
            .unwrap()
        };
        let agg = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("mean_y", DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "b"])),
                    Arc::new(arrow::array::Float64Array::from(vec![15.0, 5.0])),
                ],
            )
            .unwrap()
        };
        let mut transform_outputs = HashMap::new();
        transform_outputs.insert("agg".to_string(), agg);
        let mut lineages: OutputLineages = HashMap::new();
        lineages.insert(
            "agg".to_string(),
            crate::transform::run::OutputLineage {
                swarm: None,
                statistic_input: Some(primary.clone()),
                lowered: vec![crate::transform::run::LoweredStep { id: vec![0], groupby: vec!["g".into()], shared: false, space_columns: Vec::new() }],
                written_groupby: Vec::new(),
                source: primary.schema(),
                log_input: HashMap::new(),
                steps: Vec::new(),
            },
        );
        let layer = Layer {
            mark: Mark::Bar,
            encoding: crate::spec::encoding::Encoding {
                x: Some(crate::spec::encoding::EncodingSpec { field: "g".into(), ..Default::default() }),
                y: Some(crate::spec::encoding::EncodingSpec { field: "mean_y".into(), ..Default::default() }),
                ..Default::default()
            },
            data_source: Some("agg".into()),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer.clone()]), ..Default::default() };
        let rows = layer_sort_rows(&spec, 0, &layer, &primary, &transform_outputs, &lineages, false).expect("a published key reads");
        assert_eq!(rows.y_field, Some("mean_y"));
        assert!(
            rows.batch.schema().column_with_name("mean_y").is_none(),
            "batch is the pre-statistic rows, which never carry the aggregate's own output"
        );
        assert_eq!(rows.batch.num_rows(), 3, "batch is the raw pre-statistic rows, one per observation");
        let fallback = rows.fallback.expect("a layer with a data_source always has a fallback");
        assert!(
            fallback.schema().column_with_name("mean_y").is_some(),
            "fallback is the layer's final rows, which carry the aggregate's output"
        );
        assert_eq!(fallback.num_rows(), 2, "the layer's final rows are one per category");
    }

    /// A layer whose `data_source` names no published output refuses with
    /// the same text `prepare_render_inputs` raises for that key, rather
    /// than silently sorting over the chart's own rows.
    #[test]
    fn layer_sort_rows_refuses_a_data_source_naming_no_published_output() {
        let primary = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["a"]))],
        )
        .unwrap();
        let transform_outputs: HashMap<String, RecordBatch> = HashMap::new();
        let lineages: OutputLineages = HashMap::new();
        let layer = Layer {
            mark: Mark::Bar,
            encoding: crate::spec::encoding::Encoding {
                x: Some(crate::spec::encoding::EncodingSpec { field: "g".into(), ..Default::default() }),
                ..Default::default()
            },
            data_source: Some("ghost".into()),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer.clone()]), ..Default::default() };
        let err = match layer_sort_rows(&spec, 0, &layer, &primary, &transform_outputs, &lineages, false) {
            Err(err) => err,
            Ok(_) => panic!("an unpublished data_source key must refuse"),
        };
        assert_eq!(
            err.to_string(),
            "transform failed: layer 0 data_source 'ghost' not found in transform outputs; available keys: []"
        );
    }

    /// An establishing layer whose `data_source` names no published output
    /// never falls back to the chart's own rows through `build_axis_scale`:
    /// `establishing_sort_rows` propagates the same refusal
    /// `layer_sort_rows` raises instead of returning `None` (which its
    /// caller would read as "no establishing layer" and substitute the
    /// chart rows).
    #[test]
    fn establishing_sort_rows_propagates_a_layers_unpublished_data_source_refusal() {
        let primary = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["a"]))],
        )
        .unwrap();
        let transform_outputs: HashMap<String, RecordBatch> = HashMap::new();
        let lineages: OutputLineages = HashMap::new();
        let layer = Layer {
            mark: Mark::Bar,
            encoding: crate::spec::encoding::Encoding {
                x: Some(crate::spec::encoding::EncodingSpec { field: "g".into(), ..Default::default() }),
                ..Default::default()
            },
            data_source: Some("ghost".into()),
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![layer]), ..Default::default() };
        let err = match establishing_sort_rows("x", &spec, &primary, &transform_outputs, &lineages, false) {
            Err(err) => err,
            Ok(_) => panic!("an unpublished data_source key must refuse, not read as \"no establishing layer\""),
        };
        assert_eq!(
            err.to_string(),
            "transform failed: layer 0 data_source 'ghost' not found in transform outputs; available keys: []"
        );
    }

    /// A spec with no layers still reads `None` ("no establishing layer"):
    /// the refusal above is specific to a layer whose own `data_source` is
    /// unpublished, not a blanket change to the "no establishing layer"
    /// case an unlayered chart already hits every render.
    #[test]
    fn establishing_sort_rows_keeps_none_for_no_establishing_layer() {
        let primary =
            RecordBatch::try_new(Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)])), vec![Arc::new(StringArray::from(vec!["a"]))])
                .unwrap();
        let transform_outputs: HashMap<String, RecordBatch> = HashMap::new();
        let lineages: OutputLineages = HashMap::new();
        let spec = ChartSpec { layers: None, ..Default::default() };
        let rows = establishing_sort_rows("x", &spec, &primary, &transform_outputs, &lineages, false)
            .expect("an unlayered spec never reaches the readers' refusal");
        assert!(rows.is_none(), "an unlayered spec has no establishing layer");
    }

    /// `chart_sort_rows` pairs an unlayered chart's pre-statistic rows with
    /// its drawn (final) rows, the same pair `layer_sort_rows` builds for a
    /// layer: `batch` is the chart's lineage's `pre_statistic_rows` (the
    /// raw rows a statistic consumed), `fallback` is `drawn` (the
    /// statistic's own output), so a sort naming the statistic's own
    /// output column still finds it. With no lineage recorded, `batch` is
    /// `drawn` itself, by pointer identity: `pre_statistic_rows` returns
    /// exactly the batch it is given when there is no statistic, so an
    /// unlowered chart's sort reads `drawn` directly.
    #[test]
    fn chart_sort_rows_pairs_the_chart_lineages_pre_statistic_rows_with_the_drawn_rows() {
        let pre = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("y", DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a", "b"])),
                    Arc::new(arrow::array::Float64Array::from(vec![10.0, 20.0, 5.0])),
                ],
            )
            .unwrap()
        };
        let drawn = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("mean_y", DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "b"])),
                    Arc::new(arrow::array::Float64Array::from(vec![15.0, 5.0])),
                ],
            )
            .unwrap()
        };
        let lineage = crate::transform::run::OutputLineage {
            swarm: None,
            statistic_input: Some(pre.clone()),
            lowered: vec![crate::transform::run::LoweredStep { id: vec![0], groupby: vec!["g".into()], shared: false, space_columns: Vec::new() }],
            written_groupby: Vec::new(),
            source: pre.schema(),
            log_input: HashMap::new(),
            steps: Vec::new(),
        };
        let rows = chart_sort_rows(&drawn, Some(&lineage), None, Some("mean_y"));
        assert_eq!(rows.batch.num_rows(), 3, "batch is the chart's pre-statistic rows");
        let fallback = rows.fallback.expect("a chart with a lineage always has a fallback");
        assert_eq!(fallback.num_rows(), 2, "fallback is the drawn rows, one per category");

        let no_lineage = chart_sort_rows(&drawn, None, None, Some("mean_y"));
        assert!(std::ptr::eq(no_lineage.batch, &drawn), "with no lineage, batch is the drawn batch itself");
    }

    /// `resolve_sort_batch` reads a sort's value field from `ctx.batch` when
    /// present there, from `ctx.fallback` only when `ctx.batch` lacks it
    /// (an aggregate's own output, `mean_y`, absent from the pre-statistic
    /// rows), and stays on `ctx.batch` for `SortOp::Count` even when
    /// `fallback` would also carry the field -- count reads row presence at
    /// the pre-statistic grain, not the aggregated one. Rejects always
    /// preferring `batch` alone (which would miss `mean_y` and any other
    /// aggregate output absent from the pre-statistic rows -- deciding by
    /// `x_field`/`y_field` presence instead of the ACTUAL field a sort reads
    /// breaks a boxplot/violin/errorbar sort whose drawn channel differs
    /// from the raw field the sort object names) and rejects always
    /// preferring `fallback` (which would misread `count` and any raw-field
    /// sort whose field only the pre-statistic rows carry).
    #[test]
    fn resolve_sort_batch_reads_whichever_batch_carries_the_value_field() {
        let pre = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("raw", DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "b"])),
                    Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0])),
                ],
            )
            .unwrap()
        };
        let post = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("mean_y", DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "b"])),
                    Arc::new(arrow::array::Float64Array::from(vec![10.0, 20.0])),
                ],
            )
            .unwrap()
        };
        let ctx = SortContext { category_field: "g", batch: &pre, fallback: Some(&post), x_field: None, y_field: None, coord_flipped: false };
        assert!(
            std::ptr::eq(resolve_sort_batch(&ctx, "raw", SortOp::Sum), &pre),
            "a field batch carries stays on batch"
        );
        assert!(
            std::ptr::eq(resolve_sort_batch(&ctx, "mean_y", SortOp::Sum), &post),
            "a field only fallback carries reads from fallback"
        );
        assert!(
            std::ptr::eq(resolve_sort_batch(&ctx, "mean_y", SortOp::Count), &pre),
            "count never needs a value field, so it stays on the pre-statistic grain"
        );
        assert!(
            std::ptr::eq(resolve_sort_batch(&ctx, "nowhere", SortOp::Sum), &pre),
            "a field in neither batch stays on batch, so the existing refusal names it"
        );
    }

    // ── text_edge_rows ──

    use crate::layout::geometry::TEXT_INSET_FLOOR_PX;
    use crate::render::font::FontdueMetrics;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark_style::MarkKwargsSpec;

    fn xy_batch(x: &[f64], y: &[f64]) -> RecordBatch {
        columns_batch(vec![("x", f64_col(x)), ("y", f64_col(y))])
    }

    fn xy_encoding() -> Encoding {
        Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
            ..Default::default()
        }
    }

    fn text_chart_spec(mark_style: Option<MarkKwargsSpec>) -> ChartSpec {
        ChartSpec { mark: Mark::Text, encoding: xy_encoding(), mark_style, ..Default::default() }
    }

    fn theme() -> crate::layout::ThemeInputs {
        crate::layout::ThemeInputs::default()
    }

    /// The `LayerPrepared` list `text_edge_rows` reads, built the same way
    /// `prepare::build_layers` builds it for a real render: the chart's own
    /// layer when `spec.layers` is unset, else each declared layer, each in
    /// its own drawn coordinate space.
    fn prepared_layers(spec: &ChartSpec) -> Vec<LayerPrepared> {
        match &spec.layers {
            None => vec![LayerPrepared::from_chart_only(spec)],
            Some(layer_vec) => layer_vec.iter().map(|l| LayerPrepared::from_chart_and_layer(spec, l)).collect(),
        }
        .into_iter()
        .map(LayerPrepared::in_its_coord)
        .collect()
    }

    /// Every prepared layer of `spec` paired with `batch`, the one batch
    /// every layer in these fixtures draws from (none names a `data_source`).
    fn rows_over(spec: &ChartSpec, batch: &RecordBatch) -> TextEdgeRows {
        let layers = prepared_layers(spec);
        text_edge_rows("x", layers.iter().map(|l| (l, batch)), spec.facet.as_ref(), &theme(), &FontdueMetrics::new(), None)
    }

    #[test]
    fn text_edge_rows_is_empty_without_a_text_layer() {
        let spec = ChartSpec { mark: Mark::Point, encoding: xy_encoding(), ..Default::default() };
        let batch = xy_batch(&[1.0, 5.0], &[0.0, 0.0]);
        let rows = rows_over(&spec, &batch);
        assert!(rows.0.is_empty());
    }

    /// Every row's own need clears the floor, regardless of its value —
    /// `text_edge_rows` measures every row, not only the domain's own
    /// endpoints, so a label anywhere on the axis gets its real need.
    #[test]
    fn text_edge_rows_floors_every_rows_need() {
        let spec = text_chart_spec(None);
        let batch = xy_batch(&[1.0, 3.0, 5.0], &[0.0, 0.0, 0.0]);
        let rows = rows_over(&spec, &batch);
        assert_eq!(rows.0.len(), 3);
        for row in &rows.0 {
            assert!(row.toward_lower >= TEXT_INSET_FLOOR_PX, "toward_lower {} must clear the floor", row.toward_lower);
            assert!(row.toward_higher >= TEXT_INSET_FLOOR_PX, "toward_higher {} must clear the floor", row.toward_higher);
        }
    }

    /// A wider label needs a wider clearance: binding `text=x` so each
    /// row's own glyph widens with its own x value.
    #[test]
    fn text_edge_rows_measures_each_rows_own_label_width() {
        let mut encoding = xy_encoding();
        encoding.text = Some(EncodingSpec { field: "x".into(), ..Default::default() });
        let spec = ChartSpec { mark: Mark::Text, encoding, ..Default::default() };
        let batch = xy_batch(&[2.0, 100000.0], &[0.0, 0.0]);
        let rows = rows_over(&spec, &batch);
        assert_eq!(rows.0.len(), 2);
        let narrow = rows.0.iter().find(|r| r.value == 2.0).unwrap();
        let wide = rows.0.iter().find(|r| r.value == 100000.0).unwrap();
        assert!(wide.toward_lower > narrow.toward_lower, "\"100000\" needs more clearance than \"2\"");
    }

    /// A large positive `dx` shifts the glyph away from the left: the row's
    /// `toward_lower` need drops (may go negative, meaning no constraint on
    /// that side), while `toward_higher` grows by the same shift.
    #[test]
    fn text_edge_rows_dx_moves_the_need_from_one_side_to_the_other() {
        let centered = text_chart_spec(None);
        let shifted = text_chart_spec(Some(MarkKwargsSpec { dx: Some(1_000.0), ..Default::default() }));
        let batch = xy_batch(&[1.0], &[0.0]);
        let centered_rows = rows_over(&centered, &batch);
        let shifted_rows = rows_over(&shifted, &batch);
        assert!(centered_rows.0[0].toward_lower >= TEXT_INSET_FLOOR_PX);
        assert!(shifted_rows.0[0].toward_lower < 0.0, "shifted 1000px right, the left side needs no clearance");
        assert!(shifted_rows.0[0].toward_higher > centered_rows.0[0].toward_higher);
    }

    /// Text layered on points: the point layer contributes no rows (a point
    /// has no glyph), only the text layer's own rows.
    #[test]
    fn text_edge_rows_layered_on_points_reads_only_the_text_layer() {
        let point_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "x".into(), ..Default::default() }), y: Some(EncodingSpec { field: "y".into(), ..Default::default() }), ..Default::default() },
            ..Default::default()
        };
        let text_layer = Layer {
            mark: Mark::Text,
            encoding: Encoding { x: Some(EncodingSpec { field: "x".into(), ..Default::default() }), y: Some(EncodingSpec { field: "y".into(), ..Default::default() }), ..Default::default() },
            ..Default::default()
        };
        let spec = ChartSpec {
            layers: Some(vec![point_layer, text_layer]),
            encoding: xy_encoding(),
            ..Default::default()
        };
        let batch = xy_batch(&[1.0, 5.0], &[0.0, 0.0]);
        let rows = rows_over(&spec, &batch);
        assert_eq!(rows.0.len(), 2, "only the text layer's rows, never the point layer's");
    }

    /// A layer whose mark is not `Text` contributes no rows at all.
    #[test]
    fn text_edge_rows_a_non_text_layer_contributes_nothing() {
        let point_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding { x: Some(EncodingSpec { field: "x".into(), ..Default::default() }), y: Some(EncodingSpec { field: "y".into(), ..Default::default() }), ..Default::default() },
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![point_layer]), encoding: xy_encoding(), ..Default::default() };
        let batch = xy_batch(&[1.0, 5.0], &[0.0, 0.0]);
        let rows = rows_over(&spec, &batch);
        assert!(rows.0.is_empty());
    }

    /// Stacked text: a layered text's own stack preview is measured, not its
    /// raw value. Every rendered layer is a `LayerPrepared`, so the stack
    /// preview applies uniformly whether or not `spec.layers` is set.
    #[test]
    fn text_edge_rows_measures_a_layered_texts_stacked_value_not_its_raw_one() {
        use crate::spec::position::{PositionAdjust, StackAnchor, StackOffset};
        let text_layer = Layer {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(crate::spec::encoding::DataType::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), ..Default::default() }),
                ..Default::default()
            },
            position: Some(PositionAdjust::Stack { by: None, offset: StackOffset::Zero, anchor: StackAnchor::Top, value_axis: None }),
            inherited_position: false,
            ..Default::default()
        };
        let spec = ChartSpec { layers: Some(vec![text_layer]), encoding: Encoding::default(), ..Default::default() };
        let batch = columns_batch(vec![
            ("g", Arc::new(StringArray::from(vec!["a", "a"]))),
            ("y", f64_col(&[1.0, 1.0])),
        ]);
        let layers = prepared_layers(&spec);
        let rows = text_edge_rows("y", layers.iter().map(|l| (l, &batch)), spec.facet.as_ref(), &theme(), &FontdueMetrics::new(), None);
        assert_eq!(rows.0.len(), 2, "both rows measured at their stacked (cumulative) y, not the raw 1.0 twice over");
        let values: Vec<f64> = rows.0.iter().map(|r| r.value).collect();
        assert!(values.iter().any(|v| (*v - 2.0).abs() < 1e-9), "the second row's stacked top should be 2.0, got {values:?}");
    }
}
