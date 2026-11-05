//! prepare_render_inputs(spec, batch, theme, viewport, chart_config, leaf_scales) →
//!   1. Apply Phase 5 transforms.
//!   2. Build provisional ResolvedScales for tick-label generation.
//!   3. Derive AxesInput (titles, tick_labels).
//!   4. Group rows by facet field (if facet).
//!   5. Build LegendEntry list (if color encoding).
//!   6. (Phase 8a) Build per-layer prepared inputs; swap x↔y if CoordFlip.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;

use crate::layout::{
    AxesInput, AxisInput, AxisOrient, ColorbarInput, FacetGroup, FacetKey, LegendEntry,
    TickProjection,
};
use crate::spec::chart::ChartSpec;
use crate::transform::color_bin_group::{BinDemand, ColorBinPlan, LogAxisSpace, RunPlan, SpaceDemand};
use crate::transform::context::{AllNullOutcome, AllNullPartition, StepInstance, TransformContext, TransformWarning};
use crate::transform::core::{apply_transforms_named, SpaceSlot, TransformSpec, FINAL_OUTPUT_KEY};
use crate::transform::run::{step_spec_at, OutputLineage, OutputLineages};

use super::scale_resolve::{OverlayLegends, ResolvedScales};
use std::borrow::Cow;
use super::{RenderError, RenderWarning};

mod extent;
mod facet_panels;
mod legend;

// Re-exported into this module's namespace so the orchestrator and the inline
// `#[cfg(test)] mod tests` (which uses `super::*`) resolve the extracted helpers
// from their cohesive submodules without a path change. `pub(crate)` (batch F,
// quality-review cycle-4 finding B) so `kde.rs`'s own test module can call it
// directly to prove `Kde::count_scale_extent`'s facet pin matches `Bin`'s
// (`crate::render::prepare::fix_transform_extents_for_facet`) — no logic fork,
// same visibility-only pattern as `bin::resolve_bin_extent`'s cycle-1 export.
pub(crate) use extent::{fix_transform_extents_for_facet, fix_transform_extents_for_facet_in_space};

// `pub(crate)` so `lib.rs` can register the `#[pyfunction]` — same shape as
// `fix_transform_extents_for_facet` above: `mod legend` stays private (its
// other items are prepare-internal), only this one Python-visible symbol is
// re-exported through `prepare`'s own namespace (#123: Python reads the
// aux-legend channel list instead of mirroring it).
pub(crate) use legend::aux_legend_channels;

// `pub(in crate::render)` so the compositor's overlay sibling pass reads the
// same size-consumer rule the size legend's swatch choice is built from.
pub(in crate::render) use legend::size_drawing_marks;

// `pub(in crate::render)` so the compositor's overlay color exemption judges
// a line or ribbon by the same rule the flat inert-color check reads.
pub(in crate::render) use legend::draws_numeric_color;

/// Key used when unifying wrap-mode and grid-mode partitions: (col_val, Option<row_val>).
/// `None` row value = wrap mode (single-field facet).
type FacetPartitionKey = (String, Option<String>);

/// Per-layer prepared rendering data. When ChartSpec.layers.is_none(), exactly one
/// LayerPrepared is constructed from the chart-level mark + encoding.
#[derive(Debug, Clone)]
pub struct LayerPrepared {
    pub mark: crate::spec::mark::Mark,
    pub encoding: crate::spec::encoding::Encoding,
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
    /// Name of the chart-level transform output this layer reads from.
    /// `None` ⇒ pipeline final output (resolved via [`FINAL_OUTPUT_KEY`]).
    pub data_source: Option<String>,
    /// Phase 9c — position adjustment for this layer. Merged from
    /// `Layer.position` (preferred) or `ChartSpec.position` (chart-level
    /// fallback for single-layer charts).
    pub position: Option<crate::spec::position::PositionAdjust>,
    /// Whether this layer resolves its own independent y-scale slot instead of
    /// sharing the primary (layer 0) y-scale (secondary-y-axis, GH #52). Mirrors
    /// [`crate::spec::layer::Layer::independent_y`]; always `false` for a
    /// single-layer (chart-only) chart, whose sole layer is the primary.
    pub independent_y: bool,
    /// Whether this layer reads its chart's data: it has no `data_source`
    /// (it reads the pipeline tail), or the wire marks its `data_source` as
    /// the chart's data recomputed
    /// ([`crate::spec::layer::Layer::reads_chart_data`]). Resolved once here
    /// so the column gate ([`validate_layer_columns`]) asks
    /// one question on every route; a missing column gets a chart-data
    /// remedy only on such a layer.
    pub reads_chart_data: bool,
    /// Whether THIS layer binds `color` itself
    /// ([`Layer::owns`](crate::spec::layer::Layer::owns)): bound on its own
    /// encoding and not written there by Python from a parent (a `color`
    /// listed in [`Layer::inherited_channels`](crate::spec::layer::Layer::inherited_channels)
    /// is inherited). `encoding.color` above is the completed one, inherited
    /// whole on every mark that takes an inherited color
    /// ([`Mark::draws_inherited`](crate::spec::mark::Mark::draws_inherited)),
    /// so the legend and dodge/stack grouping read it. This flag decides only
    /// whether a layer's per-row color read takes an inherited color: a layer
    /// carrying its own literal `stroke=`/`fill=` override draws that paint,
    /// not the inherited color ([`layer_geometry_encoding`]), while a layer
    /// with neither keeps the shared color, which varies its own rows (the
    /// `catplot(kind="box", hue=x)` layers). For `from_chart_only` there is
    /// no parent, so the flag is whether the chart's encoding binds `color`.
    pub color_is_own: bool,
    /// Whether THIS layer's own encoding declared `size` / `stroke_width`,
    /// captured before the merge the way [`color_is_own`] is. A line's width
    /// bindings outrank inherited ones: `scene_build` drops an inherited
    /// `size` or `stroke_width` from a line layer's `DrawCtx` encoding
    /// whenever the layer has another width source, so only a layer that
    /// binds both itself meets the width-conflict refusal. A channel listed in
    /// [`Layer::inherited_channels`](crate::spec::layer::Layer::inherited_channels)
    /// is inherited, as for [`color_is_own`].
    ///
    /// [`color_is_own`]: LayerPrepared::color_is_own
    pub size_is_own: bool,
    /// Whether THIS layer declared its own `stroke_width` channel — see
    /// [`size_is_own`].
    ///
    /// [`size_is_own`]: LayerPrepared::size_is_own
    pub stroke_width_is_own: bool,
    /// The coordinate system this layer's marks are placed in, already resolved
    /// against the chart's (`#172`(1)): the layer's own
    /// [`Layer::coord`](crate::spec::layer::Layer::coord) when it declares one,
    /// else `ChartSpec::coord`. Identical to `spec.coord` for every layer of
    /// every spec that declares no per-layer coord, which is why the per-layer
    /// reads downstream are byte-stable.
    ///
    /// Resolved HERE, once per layer, through
    /// [`Layer::effective_coord`](crate::spec::layer::Layer::coord) so no
    /// render-side consumer has to remember the fallback. Read by
    /// [`build_layers`]'s flip decision and by `scene_build`'s per-layer
    /// `DrawCtx` spec, polar projection gate, position adjustment and refusal
    /// gates — all of which previously read the chart-level `spec.coord` and so
    /// could not tell one layer's coordinate system from another's.
    pub coord: Option<crate::spec::coord::CoordKind>,
    /// The inherited reads this layer draws without, set by the column gate
    /// ([`validate_layer_columns`]) when the layer reads a statistic's
    /// output that does not carry them; cleared from
    /// [`Self::drawn_encoding`]. Empty for every other layer.
    pub(in crate::render) dropped_reads: Vec<DroppedRead>,
    /// The field this layer's cells hold when it is a raster image reading
    /// a Raster output ([`crate::render::marks::image::raster_value_field`]),
    /// set by `prepare_render_inputs` once the layer's batch is resolved: the
    /// color binding such an image draws. `None` for every other layer.
    pub(in crate::render) raster_value_field: Option<String>,
    /// The fields this layer's series split by beyond the channels its mark
    /// draws ([`statistic_series_fields`]), set by `prepare_render_inputs`
    /// once the layer's rows are resolved. Empty for every layer whose rows
    /// no statistic grouped by such a field.
    pub(in crate::render) grouping_series: SeriesGrouping,
}

/// The position adjustment `layer` of `spec` draws with, resolved once at
/// its construction: the position the layer wrote itself wins. A position it
/// inherits (its own `position` marked [`Layer::inherited_position`], else the
/// chart-level `position`) yields to the layer's own `stack` shorthand
/// ([`Layer::writes_stack_shorthand`]), and an inherited `Stack` yields to the
/// layer's own `stack=False` ([`Layer::declines_stack`]), which every other
/// adjustment agrees with. So a refusal of a position beside a shorthand only
/// ever judges a position the layer wrote.
///
/// [`Layer::inherited_position`]: crate::spec::layer::Layer::inherited_position
/// [`Layer::writes_stack_shorthand`]: crate::spec::layer::Layer::writes_stack_shorthand
/// [`Layer::declines_stack`]: crate::spec::layer::Layer::declines_stack
fn layer_position(
    spec: &crate::spec::chart::ChartSpec,
    layer: &crate::spec::layer::Layer,
) -> Option<crate::spec::position::PositionAdjust> {
    use crate::spec::position::PositionAdjust;
    if let Some(own) = layer.position.as_ref().filter(|_| !layer.inherited_position) {
        return Some(own.clone());
    }
    let inherited = layer.position.as_ref().or(spec.position.as_ref())?;
    let yields = layer.writes_stack_shorthand() || (layer.declines_stack() && matches!(inherited, PositionAdjust::Stack { .. }));
    (!yields).then(|| inherited.clone())
}

/// The fields the statistics that produced a layer's rows grouped them by
/// beyond the channels its mark draws ([`statistic_series_fields`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub(in crate::render) struct SeriesGrouping {
    /// Fields a statistic the chart planned grouped by because the layer
    /// binds them on `shape` or `stroke_dash`. Recorded for every mark: a
    /// mark that does not draw the channel still draws the statistic's
    /// groups, and the undrawn-channel warning says so.
    pub(in crate::render) by_channel: Vec<String>,
    /// Fields a step the user wrote grouped by, whether or not a channel
    /// binds them; recorded for a series mark only.
    pub(in crate::render) by_step: Vec<String>,
}

impl SeriesGrouping {
    /// Every field the series splits by, each once, the planned ones first.
    pub(in crate::render) fn fields(&self) -> Vec<String> {
        let mut fields = self.by_channel.clone();
        fields.extend(self.by_step.iter().filter(|field| !self.by_channel.contains(field)).cloned());
        fields
    }
}

/// What the statistics that produced `layer`'s rows grouped them by, read
/// from the layer's own completed encoding (its bindings over its parent's)
/// and never from a sibling's:
///
/// - `by_channel`: each field a statistic the chart planned grouped by
///   ([`OutputLineage::lowered`]) that the layer binds on `shape` or
///   `stroke_dash`. A planned statistic groups by every encoded field, value
///   channels and the axis included, so only these channels' fields count.
///   A `detail` needs no entry: every series mark splits by its own
///   `mark_style.detail`.
/// - `by_step`, on a series mark (line, area, ribbon): each field a step
///   grouped by where the grouping was authored explicitly, not derived by
///   the chart's statistic planner ([`OutputLineage::written_groupby`] --
///   a user's own pre-marker step, or a lowered composite statistic's own
///   `groupby=` keyword like `mark_smooth(groupby=[...])`,
///   `transform::run::caller_groupby`'s doc), minus the layer's own color
///   and positional fields, along which the series runs. The series splits
///   by the grouping of the statistic that produced its rows: a field an
///   earlier step grouped by stays a series key past a lowered statistic
///   only when that statistic grouped by it too, so the run has already
///   dropped any field the statistic computed across. A field bound on
///   `shape`/`stroke_dash` reaches `by_channel` too; `fields()`'s dedup
///   keeps it from doubling.
///
/// The layer's own color, binned or not, and its grouping-only `group` key
/// split through the color path and stay out of both.
fn statistic_series_fields(layer: &LayerPrepared, lineage: Option<&OutputLineage>) -> SeriesGrouping {
    use crate::spec::mark::Mark;
    let Some(lineage) = lineage else { return SeriesGrouping::default() };
    let encoding = &layer.encoding;
    fn field(binding: Option<&crate::spec::encoding::EncodingSpec>) -> Option<&str> {
        binding.map(|b| b.field.as_str())
    }
    let grouping: HashSet<&str> = [field(encoding.shape.as_ref()), field(encoding.stroke_dash.as_ref())].into_iter().flatten().collect();
    let colored: HashSet<&str> = [field(encoding.color.as_ref()), field(encoding.group.as_ref())].into_iter().flatten().collect();
    let positional: HashSet<&str> =
        [encoding.x.as_ref(), encoding.y.as_ref(), encoding.x2.as_ref(), encoding.y2.as_ref()].into_iter().filter_map(field).collect();
    let mut by_channel: Vec<String> = Vec::new();
    for field in lineage.lowered.iter().flat_map(|step| &step.groupby) {
        if grouping.contains(field.as_str()) && !colored.contains(field.as_str()) && !by_channel.contains(field) {
            by_channel.push(field.clone());
        }
    }
    let by_step = match layer.mark {
        Mark::Line | Mark::Area | Mark::Ribbon => lineage
            .written_groupby
            .iter()
            .filter(|field| !colored.contains(field.as_str()) && !positional.contains(field.as_str()))
            .cloned()
            .collect(),
        _ => Vec::new(),
    };
    SeriesGrouping { by_channel, by_step }
}

impl LayerPrepared {
    /// Build a single layer from chart-level fields (single-layer mode).
    pub(crate) fn from_chart_only(spec: &crate::spec::chart::ChartSpec) -> Self {
        Self {
            mark: spec.mark,
            encoding: spec.encoding.clone(),
            mark_style: spec.mark_style.clone(),
            data_source: None,
            position: spec.position.clone(),
            independent_y: false,
            reads_chart_data: true,
            color_is_own: spec.encoding.color.is_some(),
            size_is_own: spec.encoding.size.is_some(),
            stroke_width_is_own: spec.encoding.stroke_width.is_some(),
            // A chart with no `layers` has no per-layer coord to resolve: its
            // sole layer's coordinate system IS the chart's.
            coord: spec.coord.clone(),
            dropped_reads: Vec::new(),
            raster_value_field: None,
            grouping_series: SeriesGrouping::default(),
        }
    }

    /// Build a layer by completing its encoding from the chart level
    /// ([`Layer::completed_encoding`](crate::spec::layer::Layer::completed_encoding)).
    pub(crate) fn from_chart_and_layer(
        spec: &crate::spec::chart::ChartSpec,
        layer: &crate::spec::layer::Layer,
    ) -> Self {
        // Which width and color bindings this layer owns, asked of its own
        // encoding before completion fills the rest from the chart.
        use crate::spec::layer::InheritedChannel;
        let color_is_own = layer.owns(InheritedChannel::Color);
        let size_is_own = layer.owns(InheritedChannel::Size);
        let stroke_width_is_own = layer.owns(InheritedChannel::StrokeWidth);
        // Layers routed to their own data via data_source are self-contained —
        // only inherit non-positional channels (color, size, etc.) from the
        // chart level. Inheriting x/y would inject the primary batch's field
        // into a layer that reads a different named output, causing marks like
        // rule to iterate over rows they don't own. Of the positional bindings
        // the layer did not write, it keeps only those its mark draws, so the
        // gate, the mark builder and every domain read judge one encoding.
        let mut encoding = layer.completed_encoding(&spec.encoding);
        // An independent-y layer's y scale is its own, never the chart's: the
        // chart-level y scale belongs to the primary axis (a bar's zero
        // anchor, say), while this layer's axis spans its own data, as
        // `fm.LayerChart(..., resolve={"y": "independent"})` draws it. An
        // independent-y bar carries its own anchor on its own `y`. Both the
        // y-slot resolution and the scene read this encoding.
        if layer.independent_y {
            if let Some(y) = encoding.y.as_mut() {
                y.scale = layer.encoding.y.as_ref().and_then(|own| own.scale.clone());
            }
        }
        // A layer's style is its own: a layered chart resolves
        // `layer.mark_style > theme` and never reads `spec.mark_style`
        // (`build_layers` refuses a layered spec that sets it).
        let mark_style = layer.mark_style.clone();
        Self {
            mark: layer.mark,
            encoding,
            mark_style,
            data_source: layer.data_source.clone(),
            position: layer_position(spec, layer),
            independent_y: layer.independent_y,
            reads_chart_data: layer.data_source.is_none() || layer.reads_chart_data,
            color_is_own,
            size_is_own,
            stroke_width_is_own,
            coord: layer.effective_coord(spec.coord.as_ref()).cloned(),
            dropped_reads: Vec::new(),
            raster_value_field: None,
            grouping_series: SeriesGrouping::default(),
        }
    }

    /// This layer in the coordinate space it draws in: its coordinate-space
    /// fields swapped ([`Self::flip_coords`]) when its own coord flips, as
    /// every layer `build_layers` prepares is, and as the scale-resolve stack
    /// preview reads one.
    pub(crate) fn in_its_coord(mut self) -> Self {
        if crate::spec::coord::flips_positional_channels(self.coord.as_ref()) {
            self.flip_coords();
        }
        self
    }

    /// Swap every coordinate-space field between the x and y roles (CoordFlip).
    ///
    /// The swap lives here as ONE operation over the whole coordinate-space
    /// group rather than as a sequence of field swaps at the call site: a
    /// coordinate-space field added to this struct later is an omission
    /// visible in this single method instead of a silent half-swap somewhere
    /// in the pipeline. `mark_style.value_axis` (`#172`(2)) is the second
    /// member of the group, beside the encoding itself.
    ///
    /// # The one coordinate-space field this method deliberately does NOT swap
    ///
    /// `position`'s `PositionAdjust::Stack::value_axis` names a channel too, and
    /// is authored pre-flip by the same desugars — so by the rule above it is a
    /// member of this group, and the close review was right that it had been
    /// left out of it. It is reconciled at
    /// [`resolve_stack_value_on_x`](crate::render::position::resolve_stack_value_on_x)
    /// instead of here, because unlike `mark_style.value_axis` it has several
    /// consumers, each of which already carries the real `coord_flipped`, and
    /// one of them (`position::axis_batch`, an unlayered spec's domain-widening
    /// preview) reads `spec.position` RAW from the rendering spec. Swapping the
    /// field here would leave that one reading the authored declaration against
    /// the real flip, i.e. one rule in two places, so the reconciliation lives at
    /// the single point all of them already funnel through. `mark_style.value_axis` has
    /// the opposite shape — its consumer (`DrawCtx::series_curve_axis`) reads the
    /// post-flip `LayerPrepared` and holds no authored copy to reconcile — which
    /// is why that one is swapped eagerly, here.
    ///
    /// The invariant this method owns is unchanged: a coordinate-space field is
    /// either swapped here or reconciled at its own single resolver, and a field
    /// that does neither is the omission. Both halves are now named.
    fn flip_coords(&mut self) {
        // The channel swap itself (x↔y with x2↔y2, Phase 10c-pre) is
        // `Encoding::flip_positional` — the crate's single expression of the
        // flip mapping, shared with `numeric_domain_union`'s coordinate-space
        // -aware layer read so the two can never disagree.
        self.encoding.flip_positional();
        // `#172`(2): `mark_style.value_axis` NAMES A CHANNEL, so it is a
        // coordinate-space field and belongs to the group this method owns —
        // the omission this doc comment anticipated.
        //
        // A desugar stamps it while building its PRE-flip encoding
        // (`desugar_density(orientation="horizontal")` remaps the density onto
        // `x` and says `value_axis="x"`), and `DrawCtx::series_curve_axis`
        // reads it from the POST-flip `LayerPrepared` that
        // `scene_build`'s `resolve_mark_style(layer.mark_style, …)` resolves.
        // Left unflipped, `mark_density(orientation="horizontal")
        // .coord(CoordFlip())` reports a horizontal series while its post-flip
        // encoding (`x`=observation, `y`=density) is an ordinary vertical area
        // — measured, that closed its fill on the panel's LEFT edge where its
        // unflipped twin closes on the bottom. Two spellings of one shape, two
        // answers.
        if let Some(mark_style) = self.mark_style.as_mut() {
            if let Some(axis) = mark_style.value_axis {
                use crate::spec::position::StackValueAxis;
                mark_style.value_axis = Some(match axis {
                    StackValueAxis::X => StackValueAxis::Y,
                    StackValueAxis::Y => StackValueAxis::X,
                });
            }
        }
    }

    /// This layer's drawing facts, the borrowed view
    /// [`crate::render::drawn::DrawFacts::draws`] decides every non-positional
    /// channel from: `encoding` and `parent` are the same completed
    /// encoding, since a prepared layer's completion has already folded its
    /// chart-level fallback in.
    pub(in crate::render) fn draw_facts(&self) -> crate::render::drawn::DrawFacts<'_> {
        crate::render::drawn::DrawFacts {
            mark: self.mark,
            encoding: &self.encoding,
            parent: &self.encoding,
            mark_style: self.mark_style.as_ref(),
            color_is_own: self.color_is_own,
            size_is_own: self.size_is_own,
            stroke_width_is_own: self.stroke_width_is_own,
            cells: self.raster_value_field.as_deref(),
        }
    }

    /// The inherited width bindings this layer gives up to a width it owns
    /// ([`crate::render::drawn::DrawFacts::yielded_widths`]).
    ///
    /// A line's `size` and `stroke_width` both set its width, and a line that
    /// binds both itself refuses (`RenderError::LineWidthConflict`). An
    /// inherited binding of either is not the layer's choice: a bubble
    /// scatter's `size` reaches every layer of
    /// `point.encode(size=...) + line(stroke_width=3)` through the chart
    /// level, and the line must keep its own 3 px. So an inherited width
    /// binding yields to any width the layer binds itself, own encoding or
    /// own constant kwarg, on the same channel or the other one (an own
    /// `size=5` beside an inherited `size` field draws the constant), and
    /// between two inherited ones the `stroke_width` (a width in pixels) is
    /// kept over the `size` (a taper). Every other mark yields nothing.
    ///
    /// The one precedence rule: the scene build drops what is yielded from
    /// the layer's drawing encoding, and [`Self::drawn_size_field`] (the size
    /// legend's question) reads the same answer.
    pub(in crate::render) fn yielded_widths(&self) -> YieldedWidths {
        self.draw_facts().yielded_widths()
    }

    /// The `size` field this layer draws, after width precedence: `None` when
    /// its mark draws no encoded size, when it binds no size field, or when
    /// it is a line that yields an inherited `size` to a width of its own
    /// ([`Self::yielded_widths`]).
    pub(in crate::render) fn drawn_size_field(&self) -> Option<&str> {
        let size = self.encoding.size.as_ref()?;
        self.draw_facts()
            .draws(crate::spec::layer::InheritedChannel::Size)
            .is_ok()
            .then_some(size.field.as_str())
    }

    /// The field this layer draws on the opacity-family `channel`: `None`
    /// when it binds no field there or its mark does not draw the channel
    /// ([`crate::spec::mark::Mark::draws_opacity_channel`]); a segment draws
    /// `fill_opacity` only on the arrowheads `arrow=True` adds.
    pub(in crate::render) fn drawn_opacity_field(
        &self,
        channel: crate::render::scale_resolve::OpacityChannel,
    ) -> Option<&str> {
        let binding = channel.slot(&self.encoding).as_ref()?;
        self.draw_facts().draws(channel.as_inherited()).is_ok().then_some(binding.field.as_str())
    }

    /// The `stroke_width` field this layer draws, after width precedence:
    /// `None` when its mark draws no encoded stroke width, when it binds no
    /// stroke-width field, or when it is a line that yields an inherited
    /// `stroke_width` to a width of its own ([`Self::yielded_widths`]).
    pub(in crate::render) fn drawn_stroke_width_field(&self) -> Option<&str> {
        let width = self.encoding.stroke_width.as_ref()?;
        self.draw_facts()
            .draws(crate::spec::layer::InheritedChannel::StrokeWidth)
            .is_ok()
            .then_some(width.field.as_str())
    }

    /// The bindings this layer's mark draws: its completed encoding
    /// ([`Self::encoding`]) after the own-color exemption and the line's
    /// own-width precedence ([`layer_geometry_encoding`]), without a binding,
    /// own or inherited, on a channel its mark does not draw
    /// ([`crate::spec::mark::Mark::draws_channel`]; `size`, `stroke_width`
    /// and the opacity family through [`Self::drawn_size_field`],
    /// [`Self::drawn_stroke_width_field`] and [`Self::drawn_opacity_field`]),
    /// and without the inherited reads it draws without
    /// ([`Self::dropped_reads`]). The encoding every mark builder of this
    /// layer reads, and the one the column gate ([`validate_layer_columns`])
    /// judges.
    pub(in crate::render) fn drawn_encoding(&self) -> crate::spec::encoding::Encoding {
        let mut encoding = layer_geometry_encoding(
            self.encoding.clone(),
            LayerGeometryFlags {
                color_is_own: self.color_is_own,
                has_own_literal_paint: self.has_own_literal_paint(),
                yielded_widths: self.yielded_widths(),
            },
        );
        encoding.retain_bindings(|wire, binding| {
            self.draws_binding(wire)
                && !self.dropped_reads.iter().any(|read| read.wire == Some(wire) && read.field == binding.field)
        });
        encoding
    }

    /// The bindings of this layer that carry a binned color rule: its drawn
    /// color ([`Self::drawn_encoding`]) and its grouping-only `group` key,
    /// each when it carries `bin`. The bindings whose statistics group by
    /// bin ([`binned_color_demand`]); the gate requires their columns to be
    /// numbers, as it does the color scale's binned reads
    /// ([`crate::render::scale_resolve::AuxRead::binned_binding`]).
    pub(in crate::render) fn binned_color_bindings(&self) -> Vec<crate::spec::encoding::EncodingSpec> {
        [self.drawn_encoding().color, self.encoding.group.clone()]
            .into_iter()
            .flatten()
            .filter(|binding| binding.color_bin().is_some())
            .collect()
    }

    /// Whether this layer draws its binding on the channel `wire` names
    /// ([`crate::render::drawn::DrawFacts::reads`]). A positional channel is
    /// judged by layer completion
    /// ([`crate::spec::layer::Layer::completed_encoding`]), so every one it
    /// keeps is drawn. A contrast-ink text's color is still drawn here (its
    /// builder reads the column to pick an ink, [`Self::drawn_encoding`]'s
    /// doc): `reads` treats that case as read, since the scale question it
    /// answers elsewhere (`undrawn_channel_warnings`,
    /// `scale_resolve::domain::aux_reads`) is a different one from "does the
    /// builder read this column".
    fn draws_binding(&self, wire: &str) -> bool {
        use crate::spec::layer::InheritedChannel as C;
        let Some(channel) = C::from_wire(wire) else { return true };
        self.draw_facts().reads(channel)
    }

    /// Whether this layer's own `stroke=`/`fill=` mark kwarg names a real
    /// color: the override `draw::resolve_mark_style` resolves to a
    /// [`PaintSource::UserColor`](crate::render::draw::PaintSource::UserColor)
    /// paint, never a clear like `stroke="none"`.
    fn has_own_literal_paint(&self) -> bool {
        use crate::render::draw::PaintSource;
        let Some(kwargs) = self.mark_style.as_ref() else { return false };
        [kwargs.stroke.as_deref(), kwargs.fill.as_deref()]
            .into_iter()
            .flatten()
            .any(|value| PaintSource::from_override(value) == PaintSource::UserColor)
    }
}

/// The ownership facts [`layer_geometry_encoding`] needs to decide its two
/// normalizations, bundled rather than read off `LayerPrepared` inside that
/// function, so each rule is testable on its own facts;
/// [`LayerPrepared::drawn_encoding`] computes them from the layer.
pub(in crate::render) struct LayerGeometryFlags {
    /// Whether this layer declared its own `color` channel — see
    /// [`LayerPrepared::color_is_own`].
    pub(in crate::render) color_is_own: bool,
    /// Whether this layer's OWN `stroke=`/`fill=` mark kwarg names a real
    /// color (not a clear like `stroke="none"`) — see the own-color doc
    /// below for why a clear does not count.
    pub(in crate::render) has_own_literal_paint: bool,
    /// The inherited width bindings this layer yields — see
    /// [`LayerPrepared::yielded_widths`].
    pub(in crate::render) yielded_widths: YieldedWidths,
}

/// The DrawCtx-local normalizations every per-layer synthetic `ChartSpec`
/// needs before a mark builder sees its encoding: the own-color exemption
/// and the own-width precedence for line. None ever touches
/// `layer.encoding` itself — both apply to a clone, so scale domains, axis
/// chrome, position grouping, and the legend keep reading the merged
/// encoding unchanged, while the geometry sees only the shape THIS layer
/// actually declared.
///
/// **Own paint outranks an inherited color.** A layer whose `color` came
/// from its parent (`!flags.color_is_own`) and that carries its own literal
/// `stroke=`/`fill=` naming a real color (`flags.has_own_literal_paint`) draws
/// that paint, so the inherited color does not reach its per-row read (the
/// `roc_chart` chance diagonal stays grey under a per-class color). A
/// *cleared* paint (`fill="none"`, `stroke="transparent"`) is no own paint:
/// `desugar_ribbon`/`desugar_errorband` always pass `stroke="none"` and group
/// their bands by the inherited color. A layer with neither keeps the shared
/// color, which varies its own rows (`catplot(kind="box", hue=x)`'s IQR
/// rects, tick caps and outlier points). A text never reaches here with an
/// inherited color: its completed encoding holds only a color it owns
/// ([`crate::spec::layer::Layer::completed_encoding`]).
///
/// For a GROUPING mark (line/area/ribbon — `marks/line.rs`'s
/// `build_color_detail_groups`), the cleared channel is not just a paint
/// input: it is also the per-group split key, so clearing it collapses
/// whatever per-color groups that layer's OWN batch would otherwise fan out
/// into back down to a single merged node/polyline over every row — the
/// desired effect for a literal-stroke layer that is conceptually ONE curve
/// (the `roc_chart` chance-diagonal is collinear, so the merge is invisible),
/// but a real topology change in general.
///
/// **Own-width precedence for line.** The inherited width bindings a line
/// yields to a width it owns ([`LayerPrepared::yielded_widths`]) are dropped
/// here. The scale keeps reading the merged `size`, so the other layers keep
/// it, and the size legend asks the same precedence which layers draw it.
pub(in crate::render) fn layer_geometry_encoding(
    merged: crate::spec::encoding::Encoding,
    flags: LayerGeometryFlags,
) -> crate::spec::encoding::Encoding {
    let mut encoding = merged;
    if !flags.color_is_own && flags.has_own_literal_paint {
        encoding.color = None;
    }
    if flags.yielded_widths.size {
        encoding.size = None;
    }
    if flags.yielded_widths.stroke_width {
        encoding.stroke_width = None;
    }
    encoding
}

/// Which inherited width bindings a layer yields ([`LayerPrepared::yielded_widths`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::render) struct YieldedWidths {
    /// The inherited `size` binding is dropped from the drawing encoding.
    pub(in crate::render) size: bool,
    /// The inherited `stroke_width` binding is dropped from the drawing encoding.
    pub(in crate::render) stroke_width: bool,
}

/// Normalize Arrow string columns to `Utf8` (`StringArray`).
///
/// Polars exports string columns as `Utf8View` (`StringViewArray`) or `LargeUtf8`
/// (`LargeStringArray`) depending on version. The rest of the render pipeline
/// (scale_resolve, draw, mark renderers) downcasts to `StringArray`. Converting
/// once here keeps every consumer simple and avoids per-site downcast forks.
///
/// Casts through `arrow::compute::cast` — the crate's precedent
/// (`transform::flatten::cast_to_utf8`). The buffer-level cast has no
/// `i32`-offset-overflow panic past 2 GiB of string data; a cast refusal or
/// a normalized-schema rebuild failure surfaces as a typed
/// [`RenderError::StringNormalizationFailed`].
fn normalize_string_views(batch: &RecordBatch) -> Result<RecordBatch, RenderError> {
    let schema = batch.schema();
    let mut new_fields: Vec<Arc<Field>> = Vec::with_capacity(schema.fields().len());
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut changed = false;
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        if matches!(field.data_type(), DataType::Utf8View | DataType::LargeUtf8) {
            let casted = arrow::compute::cast(col, &DataType::Utf8).map_err(|e| {
                RenderError::StringNormalizationFailed {
                    field: Some(field.name().clone()),
                    reason: e.to_string(),
                }
            })?;
            new_cols.push(casted);
            new_fields.push(Arc::new(Field::new(
                field.name(),
                DataType::Utf8,
                field.is_nullable(),
            )));
            changed = true;
            continue;
        }
        new_cols.push(col.clone());
        new_fields.push(field.clone());
    }
    if !changed {
        return Ok(batch.clone());
    }
    let new_schema = Arc::new(Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_cols).map_err(|e| RenderError::StringNormalizationFailed {
        field: None,
        reason: e.to_string(),
    })
}

/// Re-apply the Utf8 string-normalization invariant to every named transform
/// output.
///
/// `normalize_string_views` (above) runs once on the raw input batch, before
/// the transform pipeline, and every render-layer string reader downstream
/// (`col_as_str`, the mark builders) is documented to assume Utf8-only input
/// as a result. But a transform can MATERIALIZE a new string column — not
/// merely carry one through — at a dtype other than Utf8: `data_flatten`
/// extracts list elements via `arrow::compute::take`, which preserves the
/// source element dtype exactly, so flattening a polars `LargeList<LargeUtf8>`
/// column (F-L05-05) emits a `LargeUtf8` column that never passed through the
/// pre-pipeline normalization pass. `group_key::materialize_groupby_col`
/// echoes a `LargeUtf8` groupby column's dtype the same way when
/// reconstructing it for `aggregate`/`summary`/`box_stats`/`violin`/etc.
/// output — reachable whenever one of those transforms groups by a column a
/// prior transform (e.g. `data_flatten`) introduced. `mark_text`'s `text`
/// channel reader (`col_as_str`) rejects any non-Utf8 dtype outright, so
/// either gap silently dropped every row instead of rendering a label.
///
/// `run_transforms_with_facet` is the sole production call site: every output
/// it returns — the unfaceted pipeline's map and the faceted (partition →
/// per-panel transforms → concat) map alike — passes through here before
/// `prepare_render_inputs` hands batches to scale resolution and the mark
/// builders. Re-normalizing every entry here, rather than teaching each
/// transform to normalize its own output, closes the gap for every current
/// and future transform that materializes string data, not just the two
/// found live.
fn normalize_transform_outputs(
    outputs: HashMap<String, RecordBatch>,
) -> Result<HashMap<String, RecordBatch>, RenderError> {
    outputs
        .into_iter()
        .map(|(key, batch)| Ok((key, normalize_string_views(&batch)?)))
        .collect()
}

/// [`normalize_transform_outputs`] for the rows each output's statistics
/// consumed, which a sort and a binned color read as they read the outputs.
fn normalize_transform_lineages(lineages: OutputLineages) -> Result<OutputLineages, RenderError> {
    lineages
        .into_iter()
        .map(|(key, lineage)| {
            let statistic_input = lineage.statistic_input.as_ref().map(normalize_string_views).transpose()?;
            Ok((key, OutputLineage { statistic_input, ..lineage }))
        })
        .collect()
}

#[derive(Debug)]
pub struct PreparedInputs {
    /// All chart-level transform outputs, keyed by their `name` (when present)
    /// plus `FINAL_OUTPUT_KEY` ("__final__") for the pipeline tail. Layers
    /// with `data_source: Some(name)` look up their input batch here; layers
    /// with `data_source: None` resolve to `FINAL_OUTPUT_KEY` via
    /// [`PreparedInputs::final_batch`].
    pub transform_outputs: HashMap<String, RecordBatch>,
    /// Per `transform_outputs` key, what its rows went through, as the
    /// transform run recorded it
    /// ([`crate::transform::run::OutputLineage`]): the `Swarm` that places a
    /// layer reading it at render time, whether it is a statistic's output
    /// and the rows that statistic consumed (what a data-aware sort and a
    /// binned color's edges read), and the rows its lineage started from.
    pub(in crate::render) transform_lineages: OutputLineages,
    /// Scales resolved once during prepare **for tick-label generation only**.
    /// Their pixel ranges are not panel-specific, so the final per-panel scales —
    /// whose ranges differ per panel — are resolved fresh by
    /// `scene_build::resolve_panel_scales` (SPINE-04). Do not consume these for
    /// per-panel mark/grid positioning; use the per-panel scales there instead.
    pub provisional_scales: ResolvedScales,
    pub axes: AxesInput,
    pub facet_groups: Vec<FacetGroup>,
    /// The facet partitions a statistic's `groupby` null-key drop emptied
    /// ([`FacetPartitions::emptied_by_drop`]): their panels draw empty, frame
    /// and axes over no marks, where a cell no layer has rows for is skipped.
    pub(in crate::render) facet_emptied_by_drop: HashSet<FacetPartitionKey>,
    pub legend_entries: Vec<LegendEntry>,
    /// Legend title (Themes-T2.5b). Defaults to the color encoding's field
    /// name; None when no categorical color encoding drives a legend.
    pub legend_title: Option<String>,
    /// Continuous-colorbar input. Built from a Continuous color scale's
    /// domain + scheme; consumed by `compute_layout` to allocate a colorbar
    /// in the legend gutter. Mutually exclusive with `legend_entries`.
    pub colorbar: Option<ColorbarInput>,
    pub warnings: Vec<RenderWarning>,
    /// One entry per layer. Single-layer charts have len() == 1.
    pub layers: Vec<LayerPrepared>,
    /// The encoding every scale, axis and legend of the chart is built from
    /// ([`establishing_encoding`]): layer 0's resolved (post-`CoordFlip`)
    /// encoding, so layer 0 decides every channel it draws, with each
    /// auxiliary-scale channel layer 0 does not draw taken from the first
    /// layer that does. Computed once at construction, from the same `layers`
    /// list `build_layers` already guarantees is non-empty.
    rendering_encoding: crate::spec::encoding::Encoding,
    /// True when spec.coord == Some(CoordKind::Flip). The draw loop uses this
    /// to know that x/y have already been swapped in each layer's encoding.
    pub coord_flipped: bool,
    /// D13: all per-encoding legend style overrides extracted from
    /// `encoding.color.legend.*`. Grouped here to keep `PreparedInputs`
    /// flat and to make adding new override fields a one-location change.
    pub legend_overrides: LegendPreparedOverrides,
    /// Multivariate B1: auxiliary (size / shape) legend blocks. Built from the
    /// resolved size/shape scales, stacked beneath the color legend by
    /// `compute_layout`. Empty when neither size nor shape is encoded (or both
    /// suppressed via `legend=None`). A size/shape channel that shares its
    /// field with the color channel is merged into the color legend rather than
    /// emitted here (Vega-Lite same-field merge).
    pub aux_legends: Vec<crate::layout::AuxLegendInput>,
    /// The tick-count hints (`Axis(tick_count=N)`, default 10) used to generate
    /// the x/y tick labels + projections. Carried so the post-prepare
    /// tick_extra/tick_min_step adjustment (B5 unit 2) can regenerate the SAME
    /// raw tick values that produced the current labels.
    pub x_tick_count: usize,
    pub y_tick_count: usize,
    /// The resolved y scale behind each entry of `axes.secondary_y`, in the
    /// same order (spec §4.9, extended 2026-09-02). Carried for exactly the
    /// reason `x_tick_count`/`y_tick_count` are: the post-config tick
    /// adjustments run in `render::config_apply::resync_ticks_after_axis_merge`,
    /// after `axis_y2`'s
    /// values have merged onto the axis input, and each needs the scale its
    /// axis was derived from. Empty for every chart with no independent-y
    /// layer.
    pub secondary_y_scales: Vec<crate::render::scale_resolve::ScaleKind>,
    /// The RESOLVED tick count behind each entry of `axes.secondary_y`, in the
    /// same order — `resolve_axis_tick_count`'s output at the `ConfigAxis::Y2`
    /// position, the exact analogue of `x_tick_count`/`y_tick_count`.
    ///
    /// Carried out for the same reason the scales are: the post-config trio in
    /// `render::config_apply::resync_ticks_after_axis_merge` needs it, and `adjust_axis_ticks`
    /// re-derives `scale.tick_values_raw(tick_count)` and bails when the length
    /// does not match `tick_labels`. Deriving the count from `tick_labels.len()`
    /// instead is only accidentally equal — whenever a scale returns a different
    /// number of raw values than labels, `axis_y2`'s `tick_extra`/
    /// `tick_min_step`/`values` would go silently inert for that data shape.
    pub secondary_y_tick_counts: Vec<usize>,
    /// The structural layer→y-slot plan (secondary-y-axis, GH #52 / #72),
    /// derived once from the layers' `independent_y` flags. The single source
    /// for the layer→slot mapping: the prepare axis-input builder, per-panel
    /// [`crate::render::scale_resolve::YScaleSlots`] resolution, and the axis
    /// router all read it instead of re-deriving the mapping. Empty (default)
    /// for every chart with no independent-y layer — the byte-stable path.
    pub y_slot_plan: crate::render::scale_resolve::YSlotPlan,
    /// This chart's own `Mark::Text` rows on `x`/`y`, measured
    /// before any composite union is applied ([`PreparedAxes::x_text_rows`]/
    /// `y_text_rows`). A composite tree's resolve pass reads these to union
    /// a group's members' rows without re-measuring them.
    pub(in crate::render) x_text_rows: crate::render::scale_resolve::TextEdgeRows,
    pub(in crate::render) y_text_rows: crate::render::scale_resolve::TextEdgeRows,
}

/// Per-encoding legend style overrides extracted from `encoding.color.legend.*`.
///
/// All fields are `Option` so `Default` (all `None`) represents "use theme defaults".
/// Populated by `prepare_render_inputs` from the `color_legend_extra` JSON map and
/// consumed by `config_apply`'s legend projection (passes 17–18, into the
/// `layout::LegendOverrides` bundle) and its effective-theme construction
/// (pass 14's per-channel > chart-level cascade).
#[derive(Debug, Clone, Default)]
pub struct LegendPreparedOverrides {
    /// D13: orient override from `encoding.color.legend.orient`.
    /// `None` means use the theme default.
    pub orient: Option<crate::layout::LegendOrient>,
    /// D13: title override from `encoding.color.legend.title`.
    /// `Some(s)` replaces the default field-name legend title.
    pub title: Option<String>,
    /// D13: title font size override from `encoding.color.legend.titleFontSize`.
    pub title_font_size: Option<f64>,
    /// D13: columns override from `encoding.color.legend.columns`.
    /// When `Some`, categorical legend entries are arranged in N columns.
    pub columns: Option<u32>,
    /// D13+: maximum number of colorbar ticks from `encoding.color.legend.tickCount`.
    pub tick_count: Option<usize>,
    /// D13+: label font size from `encoding.color.legend.labelFontSize`.
    pub label_font_size: Option<f64>,
    /// D13+: colorbar gradient bar length in pixels from `encoding.color.legend.gradientLength`.
    pub gradient_length: Option<f64>,
    /// D13+: colorbar gradient bar thickness in pixels from `encoding.color.legend.gradientThickness`.
    pub gradient_thickness: Option<f64>,
    /// D13+: direction override from `encoding.color.legend.direction`.
    pub direction: Option<crate::layout::LegendDirection>,
    /// D13+: explicit tick/entry values from `encoding.color.legend.values`.
    pub values: Option<Vec<String>>,
    /// D13+: legend type override from `encoding.color.legend.type`.
    /// "gradient" forces colorbar rendering; "symbol" forces discrete entries.
    pub legend_type: Option<String>,
    /// B5: symbol shape override from `encoding.color.legend.symbol_type`.
    /// Per-channel wins over chart-level `configure_legend(symbol_type=...)`.
    pub symbol_type: Option<String>,
    /// B5 unit 3: stroke width of legend symbols from
    /// `encoding.color.legend.symbol_stroke_width`.
    pub symbol_stroke_width: Option<f64>,
    /// B5 unit 3: per-legend vertical entry spacing from
    /// `encoding.color.legend.row_padding` (replaces `LEGEND_ENTRY_ROW_PAD`).
    pub row_padding: Option<f64>,
    /// B5 unit 3: per-legend horizontal entry spacing from
    /// `encoding.color.legend.column_padding`.
    pub column_padding: Option<f64>,
    /// B5 unit 3: max legend-label pixel width from
    /// `encoding.color.legend.label_limit`. Labels wider than this are truncated
    /// with an ellipsis.
    pub label_limit: Option<f64>,
    /// B5 unit 3: cap the legend group's total height from
    /// `encoding.color.legend.clip_height`. Overflow is hard-clipped.
    pub clip_height: Option<f64>,
    /// B5 unit 3: minimum step between colorbar ticks (data units) from
    /// `encoding.color.legend.tick_min_step`.
    pub tick_min_step: Option<f64>,
    /// B5 unit 3: coarse draw order from `encoding.color.legend.zindex`.
    /// `>= 1` routes the legend above marks; `<= 0` (default) below.
    pub zindex: Option<i64>,
    /// B5 unit 6a: legend swatch area (px²) from `encoding.color.legend.symbol_size`.
    pub symbol_size: Option<f64>,
    /// B5 unit 6a: legend entry-label fill color from
    /// `encoding.color.legend.label_color`.
    pub label_color: Option<String>,
    /// B5 unit 6a: extra plot→legend gap (px) from `encoding.color.legend.offset`.
    pub offset: Option<f64>,
    /// B5 unit 6a: internal legend box padding (px) from
    /// `encoding.color.legend.padding`.
    pub padding: Option<f64>,
    /// B5 unit 6a: legend title→entry gap (px) from
    /// `encoding.color.legend.title_padding`.
    pub title_padding: Option<f64>,
}

impl PreparedInputs {
    /// Whether the facet panel at `keys` (its column key, then its row key
    /// in grid mode) is one a statistic's null-key drop emptied
    /// ([`Self::facet_emptied_by_drop`]).
    pub(in crate::render) fn panel_emptied_by_drop(&self, keys: (Option<&FacetKey>, Option<&FacetKey>)) -> bool {
        let (Some(col), row) = keys else { return false };
        self.facet_emptied_by_drop.contains(&(col.value.clone(), row.map(|row| row.value.clone())))
    }

    /// The chart's rendering encoding: the `rendering_encoding` field this
    /// struct already carries (see that field's doc for what it holds).
    pub(crate) fn rendering_encoding(&self) -> &crate::spec::encoding::Encoding {
        &self.rendering_encoding
    }

    /// The final transform-pipeline output — i.e. `transform_outputs[FINAL_OUTPUT_KEY]`.
    /// Used by the render orchestrator for facet filtering, the colorbar legend
    /// scale rebuild, and any other consumer that needs the chart-level tail.
    pub fn final_batch(&self) -> &RecordBatch {
        self.transform_outputs
            .get(FINAL_OUTPUT_KEY)
            .expect("apply_transforms_named publishes FINAL_OUTPUT_KEY unconditionally")
    }
}

/// Which positional channel an axis-input derivation is for.
///
/// Carries the two pieces of behavior that differ between the x and y axis
/// derivations so `build_axis_input` can be called once per channel instead of
/// the per-axis block being hand-written twice (SPINE-08):
/// - the **default orient** (`Bottom` for x, `Left` for y), used when no
///   per-channel `Axis(orient=...)` override is present;
/// - the **non-ordinal reverse policy** (only y reverses labels/fractions so
///   high domain values sit at the top), threaded into `build_axis_tick_inputs`
///   and `encoding_axis_style_overrides`' validation channel token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    X,
    Y,
}

impl Channel {
    /// The default axis side when no per-channel `Axis(orient=...)` is set.
    fn default_orient(self) -> AxisOrient {
        match self {
            Channel::X => AxisOrient::Bottom,
            Channel::Y => AxisOrient::Left,
        }
    }

    /// Whether this channel applies the non-ordinal label/fraction reversal.
    /// Only y reverses (the pixel range is already inverted in scale_resolve);
    /// x never does.
    fn reverses(self) -> bool {
        matches!(self, Channel::Y)
    }

    /// The static channel token used for axis-orient/style validation messages.
    fn token(self) -> &'static str {
        match self {
            Channel::X => "x",
            Channel::Y => "y",
        }
    }
}

/// Un-flip a positional channel token for user-facing error/warning text (R3).
///
/// Every error/warning message that names a positional channel downstream of
/// the `CoordFlip` x<->y / x2<->y2 swap (see [`build_layers`]) carries the
/// RESOLVED (post-flip) token internally — that's the token validation and
/// lookup logic must keep acting on. This helper is applied ONLY at
/// message-format time, to turn that resolved token back into the one the
/// user actually wrote in `.encode(...)`:
/// - `!coord_flipped` → identity (every unflipped message is byte-identical).
/// - `coord_flipped` → `x`<->`y` and `x2`<->`y2` swap back; every other token
///   (`"color"`, a compound string like `"x2 and y2"`, a field name, ...) is
///   returned unchanged, since only single positional tokens are swapped by
///   `build_layers`.
///
/// Never applied to error-variant fields, lookup keys, or resolution logic —
/// see the module doc on `build_layers` for where the actual swap happens.
pub(crate) fn user_facing_channel(channel: &str, coord_flipped: bool) -> &str {
    if !coord_flipped {
        return channel;
    }
    match channel {
        "x" => "y",
        "y" => "x",
        "x2" => "y2",
        "y2" => "x2",
        other => other,
    }
}

// The 3-way axis-title idiom (absent → fallback, empty → suppress, non-empty →
// verbatim) lives in `layout::axis` alongside `AxisInput::fill_chart_level_title`,
// the chart-level writer that must apply the identical rule (D12, spec §4.9).
// Hoisted there rather than re-expressed here: two copies of a suppression
// sentinel is exactly how a chart-level fill starts resurrecting per-channel
// suppressions.
use crate::layout::axis::resolve_axis_title;

/// Build one positional axis's [`AxisInput`] from its resolved scale and
/// encoding (SPINE-08).
///
/// This is the single derivation the x and y axes both run; the per-channel
/// differences (orient default + the non-ordinal-y label reversal) are carried
/// by [`Channel`]. It performs, in order: the 3-way title resolution
/// (spec-level title → layer-0 title → field name), the show-toggle extraction,
/// the tick-label/projection build (threading numeric format forward, applying
/// temporal format directly, reversing for non-ordinal y), the theme-gated
/// minor-fraction projection, the per-channel style-override parse, the orient
/// default, and the final `AxisInput` assembly.
///
/// `rendering_enc` is the layer-0 (post-CoordFlip) encoding for this channel;
/// `spec_enc` is the user-facing spec-level encoding (its title wins). Returns
/// the assembled `AxisInput` plus the resolved show-axis-band toggle inputs are
/// kept on the `AxisInput` itself.
///
/// Errors propagate as [`crate::render::UnflippedRenderError`] (R3): this fn
/// has no `coord_flipped` of its own — its two callers (`build_axes`,
/// `build_secondary_y_axis_inputs`) do, and resolve at their own boundary.
fn build_axis_input(
    channel: Channel,
    rendering_enc: Option<&crate::spec::encoding::EncodingSpec>,
    spec_enc: Option<&crate::spec::encoding::EncodingSpec>,
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    theme: &crate::layout::ThemeInputs,
    // The `Mark::Text` ink demand this channel's panel-real axis carries
    // (empty when no `Mark::Text` layer draws it), carried onto
    // the built `TickProjection` so layout applies the SAME inset the marks
    // resolve with.
    text_demand: crate::layout::geometry::TextEdgeDemand,
) -> Result<AxisInput, crate::render::UnflippedRenderError> {
    // Axis title resolution priority:
    //   1. Spec-level encoding title (set by user via .encode(y=Y(..., title=...)))
    //   2. Layer-0 encoding title (set by desugar for internal column names)
    //   3. Field name (fallback)
    // User-explicit titles always win; layer-level titles override the field
    // name for diagnostic charts whose layer-0 encoding references a column
    // with a non-semantic name (e.g. "lower_whisker" / "param_value").
    let field_title = rendering_enc.and_then(|e| {
        // Explicit spec-level title takes highest priority.
        let spec_title = resolve_axis_title(spec_enc.and_then(|p| p.title.as_deref()), None);
        if let Some(spec_title) = spec_title {
            return Some(spec_title);
        }
        if spec_enc.and_then(|p| p.title.as_deref()).is_some() {
            // Spec-level title present but empty → explicit suppress, no fallback.
            return None;
        }
        // Layer-0 title (desugar for diagnostic column names), then field name.
        resolve_axis_title(e.title.as_deref(), Some(e.field.clone()))
    });

    // D7 + D12 (B5-typed): per-axis style fields from the typed `encoding.axis`
    // style spec. The show toggles ride the SAME `AxisStyleOverrides` bundle as
    // every other per-axis field (D12, spec §4.9) — `encoding_axis_style_overrides`
    // below writes them unconditionally (fresh build), and the chart-level path
    // fills them only when still `None`, so per-channel wins without this fn
    // needing a hand-written `unwrap_or(true)` per toggle.
    let enc_axis = rendering_enc.and_then(|e| e.axis.as_ref());
    // Axis(title=...): the outer Option distinguishes "key absent" (fall through
    // to the field-name default) from "key present but empty" (explicit suppress).
    let per_channel_title = enc_axis.and_then(|a| a.title.as_deref());
    // The chart-level `configure_axis(title=)` fill must not overwrite a
    // per-channel title — including the empty-string SUPPRESS spelling, whose
    // resolved value is indistinguishable from "never set" once it lands in
    // `AxisInput.title`. Capture the claim here, where the raw wire value is
    // still visible. `spec_enc.title` (the highest-priority source, resolved
    // into `field_title` above) claims the slot for the same reason.
    let mut title = crate::layout::Claimed::new(resolve_axis_title(per_channel_title, field_title));
    if per_channel_title.is_some() || spec_enc.and_then(|p| p.title.as_deref()).is_some() {
        title.claim();
    }

    // Resolve the tick label format. A per-channel `Axis(label_format=,
    // label_format_type=)` takes precedence over the shorthand
    // `encoding.format`/`format_type`. Chart-level
    // `configure_axis(label_format_raw=...)` is applied later in `render/mod.rs`
    // only when no per-encoding override exists.
    let (tick_format, tick_format_type) = resolve_axis_label_format(rendering_enc);
    // When no `Axis(label_format=)` replaces it, the shorthand pair is in use,
    // but an `Axis(label_format_type=)` may have overridden its type above.
    // The shorthand pair as written still meets the one refusal rule, so
    // `X("n:Q", format=".1f", format_type="time", axis=Axis(label_format_type=
    // "number"))` refuses as `X("n:Q", format=".1f", format_type="time")` does
    // (`#147`). Independent facet axes are covered too:
    // this global axis is built before any per-panel one.
    //
    // Only a TIME-typed shorthand is checked as written: that is the half an
    // `Axis` type can launder (`"time"` retyped `"number"` on a numeric
    // axis). A number-format shorthand an `Axis` retypes `"time"` is read as
    // the merged pair, so `X("d:T", format=".2f", axis=Axis(
    // label_format_type="time"))` falls back to the temporal default exactly
    // as `X("d:T", format=".2f", format_type="time")` does (`#147`: the
    // merged pair gives its one-place spelling's answer).
    let shorthand_is_time_typed = |e: &crate::spec::encoding::EncodingSpec| {
        crate::render::format::FormatKind::resolve_on(
            e.format.as_deref(),
            e.format_type.as_deref(),
            scale_temporality(scale),
        )
        .is_some_and(|kind| kind.is_temporal())
    };
    if let Some(e) = rendering_enc.filter(|e| {
        e.axis.as_ref().is_none_or(|a| a.label_format.is_none()) && shorthand_is_time_typed(e)
    }) {
        refuse_format_temporality_mismatch_scale(
            e.format.as_deref(),
            e.format_type.as_deref(),
            scale,
            channel.token(),
        )
        .map_err(crate::render::UnflippedRenderError::new)?;
    }
    // D8 cascade-inversion fix (spec review cycle 2): capture "did per-channel
    // claim a format for this axis at all" BEFORE `tick_format` is moved into
    // `TickFormatMode::Thread` below — this survives even when the eventual
    // per-channel formatting is applied EAGERLY (temporal case) and threads
    // `label_format_override = None`, which `LabelFormat::spec.is_none()` alone
    // cannot distinguish from "no per-channel format". See
    // `layout::Claimed`'s doc.
    let label_format_claimed = tick_format.is_some();

    // Grid item 18: minor tick fractions from the resolved scale, projected
    // through the same `[0,1]`-range scale that places majors. The
    // `theme.grid.minor` gate (default `false`) is the single source of truth:
    // when off, `minor` is empty → no minors built → default output
    // byte-identical.
    let minor_fractions = if theme.grid.minor {
        scale.minor_tick_fractions()
    } else {
        Vec::new()
    };

    // Derive labels + scale-projected tick fractions + threaded label format
    // through the shared `build_axis_tick_inputs` helper. Global axes use
    // `Thread` mode (numeric format is threaded onto the override for central
    // application; temporal is formatted directly). `channel.reverses()` carries
    // the non-ordinal-y reversal; the theme-gated minor fractions are passed in
    // (not reversed).
    let (tick_labels, tick_anchors, tick_origin, tick_projection, label_format_override) = build_axis_tick_inputs(
        scale,
        tick_count,
        TickFormatMode::Thread {
            format: tick_format,
            format_type: tick_format_type.as_deref(),
        },
        channel.reverses(),
        minor_fractions,
        text_demand,
    )
    .map_err(crate::render::UnflippedRenderError::new)?;

    // B5: per-channel axis STYLING + positioning overrides.
    let mut overrides = encoding_axis_style_overrides(enc_axis.map(Box::as_ref), channel.token())?;
    // Seed the per-channel format (resolved via the temporal/numeric threading
    // above) onto the bundle's `labels.format`. This path OWNS the whole slot:
    // `axis_style_fill_from`'s fresh build deliberately writes neither half, so
    // `format_type` is still `Default` here — writing the value entire keeps
    // that explicit rather than leaving a half-written slot behind.
    overrides.labels.format.set(crate::layout::LabelFormat {
        spec: label_format_override,
        format_type: None,
    });
    // D8 cascade-inversion fix: mark the slot claimed regardless of whether
    // a format string survived to be threaded (see the capture above and
    // `layout::Claimed`'s doc). The title slot takes its own claim where it is
    // built, above.
    if label_format_claimed {
        overrides.labels.format.claim();
    }

    // Per-channel `orient` selects the axis side; absent → the dimension default
    // (Bottom for x, Left for y). The `placement.orient` override input stays in
    // the bundle so a later chart-level `configure_axis(orient=...)` fills it only when
    // still `None` (per-channel wins), then `resolve_orient` re-syncs.
    let orient = overrides
        .placement
        .orient
        .unwrap_or(channel.default_orient());

    Ok(AxisInput {
        orient,
        title,
        tick_labels,
        tick_anchors,
        tick_origin,
        tick_format: None, // already applied above
        tick_format_type: None,
        tick_projection,
        // Categorical axes (F-L04-03, GH #67): carry the scale's own band
        // placement so layout puts tick labels and grid lines on the same pixels
        // the marks get. An explicit `range=` resolves to absolute pixels here; a
        // panel-extent scale carries its d3 model instead, because this pass runs
        // before layout against a `[0, 1]` placeholder range. `None` for
        // continuous axes, which project through `tick_projection`.
        categorical_placement: scale.categorical_placement(),
        overrides,
    })
}

/// Filter out warnings a metadata-only, placeholder-`(0.0, 1.0)`-pixel-range
/// scale resolution cannot honestly judge — currently just
/// `GroupCardinalityDegenerate` (#183), whose entire question ("is the
/// resolved bandwidth sub-pixel") is meaningless over a 1px-wide range for
/// any domain of 2+ categories, regardless of how the chart actually
/// renders. Every other scale warning (color scheme, palette overflow,
/// color-range parse failures, sort-spec-ignored, `ScaleDomainConfigOnOrdinalAxis`,
/// ...) does not depend on the pixel range at all, so it is genuinely true
/// and must survive this filter — dropping it wholesale was the bug this
/// filter replaces (quality-review finding #1, batch F/wave-1 task 11): for
/// a chart whose data is empty after transforms, `build_scene`'s panel loop
/// never resolves a panel to re-emit anything, so a warning discarded here
/// would vanish for good, breaking this whole batch's "never silent"
/// premise for exactly that shape of chart.
///
/// Called at both of this module's placeholder-range scale-resolution call
/// sites (the primary provisional pass below, and
/// `build_secondary_y_axis_inputs`'s per-layer pass) — the two places
/// `prepare_render_inputs` resolves scales before the real per-panel plot
/// area exists.
fn drop_placeholder_only_warnings(warnings: Vec<RenderWarning>) -> Vec<RenderWarning> {
    warnings
        .into_iter()
        .filter(|w| !matches!(w, RenderWarning::GroupCardinalityDegenerate { .. }))
        .collect()
}

/// True for a provisional-pass scale warning that the panel-real pass
/// (`scene_build::resolve_panel_scales`) reports again, byte-identically,
/// once at least one panel actually resolves — a companion predicate to
/// [`drop_placeholder_only_warnings`] for the warnings that survive it.
/// `drop_placeholder_only_warnings` drops what the placeholder
/// `(0.0, 1.0)`-pixel-range pass cannot honestly judge at all
/// (`GroupCardinalityDegenerate`'s bandwidth question). This predicate names
/// a different failure: a warning the placeholder
/// pass judges CORRECTLY (a color scheme name lookup does not depend on the
/// pixel range) but which is a whole-chart fact, not a per-panel one — so
/// once a real panel exists to judge it again, the provisional copy is a
/// pure restatement, not an independent finding. This is the opposite shape
/// from `GroupCardinalityDegenerate`, which stays a genuine per-panel fact
/// and is never in this list at all (it is dropped from the provisional
/// pass entirely, by [`drop_placeholder_only_warnings`] above, because the
/// placeholder pixel range cannot honestly judge it even once).
///
/// Members share one shape: the warning answers a question about the SPEC (a
/// scheme name, a stops list, a sort spec), never about this panel's data or
/// pixels, so every panel that resolves re-derives the identical answer.
/// - `UnknownColorScheme` — a named-but-unrecognized `scheme`/`cmap`, from
///   continuous-color-scheme resolution
///   (`scale_resolve::color::resolve_continuous_scheme`).
/// - `GradientTooFewStops` — a `Sequential` scale's `stops` resolved to
///   fewer than 2 usable entries (cycle-2 close-remediation finding N1: this
///   variant reproduced the exact S3-1 shape in the very change that fixed
///   it for `UnknownColorScheme`, because it is built by the same resolver
///   but was not added to this predicate when it was introduced).
/// - `SortSpecIgnored` — a `sort=` the resolver could not honor (#95, wave-5).
///   Every reason it carries is a property of the spec plus the schema
///   (unrecognized form, unsupported `op`, a `field`/channel naming nothing
///   bound), so the panel-real pass restates it verbatim; a user who wrote one
///   bad `sort=` was getting two identical `UserWarning`s for it. Note this
///   member is NOT produced by the color resolver like the two above — the
///   shared property is "restated per panel, not re-measured per panel", which
///   is what this predicate has always actually meant.
/// - `ShapePaletteOverflowed`, `StrokeDashPaletteOverflowed` — every
///   `RenderWarning` `scale_resolve::auxiliary` pushes. These
///   are whole-chart facts even though they look data-dependent (a
///   distinct-category count): `build_auxiliary_scales`
///   resolves every one of these channels with `facet_aux_shared(spec)` —
///   `spec.facet.is_some()` — which reads the SAME batch on every resolution
///   regardless of which pass or which panel asks (a non-faceted chart's one
///   panel is that batch; a faceted chart's per-panel resolution unions the
///   shared global batch), so the panel-real pass re-derives the identical
///   answer every time, exactly like the color-scheme members above. Their
///   restatement does not stop at the provisional/panel-real boundary this
///   predicate polices: a faceted chart's panel LOOP restates them again,
///   once per panel — [`panel_pass_warning_restated_by_every_panel`] below is
///   the second, narrower predicate that collapses those.
/// - `ColorPaletteOverflowed` (`scale_resolve::color::build_color_scale`,
///   pushed from the same `build_auxiliary_scales` pass, `#120`(7) + N2) is
///   HERE for the same provisional-vs-panel-real reason as the three above —
///   its provisional copy is a pure restatement once any panel resolves — but
///   it is deliberately NOT a member of
///   [`panel_pass_warning_restated_by_every_panel`]: unlike the three
///   auxiliary channels, a panel's OWN color scale can legitimately overflow
///   the palette differently from its siblings, so every panel's real copy
///   stays a distinct fact, not a restatement of the first panel's.
///
/// A future member must be added here explicitly — there is no compile-time
/// forcing function for this list (unlike `apply_with_context`'s exhaustive
/// match), so a new pixel-range-independent, spec-keyed warning needs a human
/// to notice it shares this shape. See "What still feels wrong" in the
/// close-rust design review for the durable fix (an exhaustive
/// `RenderWarning::multiplicity()` classifier), not executed here as out of
/// scope for this remediation.
///
/// `RenderWarning::FacetSortSpecIgnored` (`#176`(11) + N3) is the negative
/// case worth naming explicitly: it looks like `SortSpecIgnored`'s sibling
/// (same reason text, same "resolver could not honor a sort" story) and it
/// is tempting to add here, but it must NOT be — the facet partition that
/// produces it runs exactly ONCE, in `prepare_render_inputs`, before any
/// panel loop exists; `resolve_panel_scales` never re-derives a facet's own
/// panel ordering per panel, so there is no panel-real restatement for this
/// predicate to treat the provisional copy as a duplicate of. Adding it here
/// would silently drop the ONLY report of a bad `Facet(sort=)` on every
/// render with at least one non-empty panel.
///
/// `scene_build::build_scene` calls this to strip the provisional prefix of
/// its warnings vec once it has confirmed at least one panel actually
/// reached `resolve_panel_scales` — see that function's own doc for why the
/// empty-after-transform case (T11, #118) must keep the provisional copy
/// when NO panel ever resolves (the provisional pass is then the only chance
/// this fact is ever reported). That is also why `SortSpecIgnored` belongs
/// HERE and not in [`drop_placeholder_only_warnings`]: dropping it at the
/// source would silence it outright for an all-filtered chart, which is the
/// regression that filter's own doc was written to prevent
/// (`unresolvable_sort_still_warns_when_no_panel_resolves` pins the
/// difference).
pub(crate) fn provisional_scale_warning_superseded_by_panel_pass(w: &RenderWarning) -> bool {
    matches!(
        w,
        RenderWarning::UnknownColorScheme { .. }
            | RenderWarning::GradientTooFewStops { .. }
            | RenderWarning::SortSpecIgnored { .. }
            | RenderWarning::ShapePaletteOverflowed { .. }
            | RenderWarning::StrokeDashPaletteOverflowed { .. }
            | RenderWarning::ColorPaletteOverflowed { .. }
            // A `Log` channel's extent fold reads the whole frame's rows
            // (facet-shared, via the same `include_final`/`FINAL_OUTPUT_KEY`
            // every other channel's shared extent already reads), so every
            // panel's resolution restates the identical count; the panel
            // pass's own copy supersedes the provisional one exactly as the
            // palette-overflow warnings above do.
            | RenderWarning::LogScaleRowsLeftOut { .. }
    ) || excluded_from_a_non_positional_scale(w)
}

/// A layer left out of a chart-level non-positional scale
/// (`scale_resolve::admitted_aux_reads`). Unlike an axis exclusion, which
/// `prepare_render_inputs` reports once before any scale resolves, this one is
/// decided inside scale resolution, so the provisional pass and every panel
/// re-derive it from the same layers and columns: both predicates above and
/// below treat the later copies as restatements.
fn excluded_from_a_non_positional_scale(w: &RenderWarning) -> bool {
    matches!(w, RenderWarning::LayerScaleExcluded { channel, .. } if !crate::render::excluded_channel_is_positional(channel))
}

/// True for a warning `resolve_panel_scales` restates once per panel of a
/// faceted chart even though it answers a whole-chart question, not a
/// per-panel one — the panel-LOOP counterpart to
/// [`provisional_scale_warning_superseded_by_panel_pass`] above (`#120`(7) +
/// N2, faceted extension). That predicate polices the boundary between the
/// provisional pass and the panel loop; this one polices repetition WITHIN
/// the panel loop itself, once it has already run once per panel.
///
/// `ShapePaletteOverflowed`/`StrokeDashPaletteOverflowed` qualify for the
/// reason given in this module's other predicate's doc:
/// `build_auxiliary_scales` resolves both with `facet_aux_shared(spec)`,
/// which reads the SAME shared global batch for every panel of a faceted
/// chart, so `resolve_panel_scales` re-derives an identical answer at each
/// one — beyond the first panel, every later copy is a pure restatement.
///
/// `ColorPaletteOverflowed` is deliberately excluded — see
/// [`provisional_scale_warning_superseded_by_panel_pass`]'s doc: a panel
/// with an independently-resolved color scale can legitimately overflow the
/// palette differently from its siblings, so every panel's own copy is a
/// distinct fact and must survive.
///
/// [`collapse_panel_restated_duplicates`] is the collapsing step this
/// predicate feeds; `scene_build::build_scene` runs it over the panel loop's
/// own appended segment only, never the pre-panel prefix (which the OTHER
/// predicate already handles).
pub(crate) fn panel_pass_warning_restated_by_every_panel(w: &RenderWarning) -> bool {
    matches!(
        w,
        RenderWarning::ShapePaletteOverflowed { .. }
            | RenderWarning::StrokeDashPaletteOverflowed { .. }
            | RenderWarning::LogScaleRowsLeftOut { .. }
    ) || excluded_from_a_non_positional_scale(w)
}

/// Keep only the FIRST copy of any warning
/// [`panel_pass_warning_restated_by_every_panel`] names; every later copy in
/// `warnings` is a pure restatement the panel loop produced once per panel of
/// a faceted chart (see that predicate's doc). Every other warning — an
/// unrelated variant, or a repeated `ColorPaletteOverflowed` (deliberately
/// excluded above) — passes through untouched, in its original order.
pub(crate) fn collapse_panel_restated_duplicates(warnings: Vec<RenderWarning>) -> Vec<RenderWarning> {
    let mut kept: Vec<RenderWarning> = Vec::with_capacity(warnings.len());
    for w in warnings {
        if panel_pass_warning_restated_by_every_panel(&w) && kept.contains(&w) {
            continue;
        }
        kept.push(w);
    }
    kept
}

/// Prepare a single leaf's render inputs (transforms, per-layer encodings,
/// provisional scales/axes, legends).
///
/// `leaf_scales` is the D4b composite seam: `Some` only for a composite leaf,
/// threading a resolved-domain context into the provisional scale pass so the
/// leaf's provisional axes (and thus tick labels) resolve on the same auto path,
/// seeded by the shared domain, that the per-panel final pass will use. `None`
/// for every standalone (flat/facet) render reproduces the pre-D4b behavior
/// byte-for-byte.
pub fn prepare_render_inputs(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &crate::layout::ThemeInputs,
    viewport: crate::layout::Viewport,
    chart_config: &crate::render::chart_config::ChartConfig,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
) -> Result<PreparedInputs, RenderError> {
    // A zero-row batch is real input, not malformed: it takes the exact
    // path a filter-emptied tail already takes (transforms, scale
    // resolution, scene build), which draws the chart's frame and axes and
    // warns once (`RenderWarning::NoRowsToDraw`) rather than refusing.
    // The primary y axis is owned by a layer that draws y (an independent-y
    // layer when no sharing layer does), before any layer, slot or axis
    // reads the flags.
    let spec = &*crate::render::scale_resolve::with_primary_y_owner(spec);

    // Normalize Utf8View columns (e.g. from polars) to Utf8 so downstream
    // downcasts to StringArray succeed uniformly.
    let normalized = normalize_string_views(batch)?;
    // The chart's SOURCE schema, taken HERE — before `run_transforms_with_facet`
    // (below) partitions the batch for a faceted chart — so it is always the
    // one unpartitioned schema the column gate names a source remedy from.
    let source_schema = normalized.schema();

    // Run the Phase 5 transform pipeline, applying the facet-aware partition →
    // per-panel transforms → inject → concat path when a facet is present and
    // the plain single-partition path otherwise. The D10 impute step runs once
    // inside, in both branches (SPINE-12). `viewport`/`theme` seed the M2
    // pre-layout panel-size estimate (see `run_transforms_with_facet`'s doc).
    //
    // #162: `run_transforms_with_facet` also drains every `TransformWarning`
    // its `TransformContext` sink accumulated (NaN-panel degrades, the gam
    // `k` clamp, `LetterValue`'s depth cap, `Impute`'s fallbacks). Hoisted
    // here, above `scale_warnings`' construction below, so both land in the
    // one `PreparedInputs.warnings` vec this function returns.
    let (transform_outputs, transform_warnings, transform_lineages, facet_partitions) =
        run_pipeline(spec, &normalized, viewport, theme, leaf_scales)?;
    let transformed = transform_outputs
        .get(FINAL_OUTPUT_KEY)
        .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY")
        .clone();

    // --- Phase 8a: per-layer inputs + CoordFlip ---

    // Build per-layer prepared inputs (swapping x↔y / x2↔y2 for every layer
    // whose own effective coord flips — `#172`(1)).
    let coord_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    let mut layers = build_layers(spec)?;
    refuse_malformed_sort_specs(spec)?;
    refuse_bin_channel_without_binned_color(spec)?;
    refuse_unstackable_shorthands(spec, &layers)?;

    // Layer→y-slot plan (secondary-y-axis, GH #52 / #72): derive the mapping
    // ONCE here from the layers' `independent_y` flags, then hand it to every
    // consumer (axis-input builder below, per-panel scale resolution, axis
    // router). Empty for every chart with no independent-y layer.
    let y_slot_plan = crate::render::scale_resolve::YSlotPlan::from_layer_flags(
        layers.iter().map(|l| l.independent_y),
    );

    // Resolve every layer's own batch once: its `data_source` output, else the
    // pipeline tail. An unknown `data_source` is refused here, every layer
    // before any gate, so the per-panel render loop can unconditionally
    // `.get()` and the column gate below reads the batch this resolved.
    let mut layer_batches: Vec<&RecordBatch> = Vec::with_capacity(layers.len());
    for (i, layer) in layers.iter().enumerate() {
        let batch = match &layer.data_source {
            None => &transformed,
            Some(name) => match transform_outputs.get(name) {
                Some(batch) => batch,
                None => return Err(unpublished_data_source(i, name, &transform_outputs)),
            },
        };
        layer_batches.push(batch);
    }
    for (layer, batch) in layers.iter_mut().zip(&layer_batches) {
        let url = layer.encoding.url.as_ref();
        if layer.mark == crate::spec::mark::Mark::Image && crate::render::marks::image::url_tile_field(url).is_none() {
            layer.raster_value_field = crate::render::marks::image::raster_value_field(batch).map(str::to_string);
        }
        let key = layer.data_source.as_deref().unwrap_or(FINAL_OUTPUT_KEY);
        layer.grouping_series = statistic_series_fields(layer, transform_lineages.get(key));
    }

    // Every column each layer of a layered spec draws is checked against its
    // own rows here, once, before any scale resolution reads one, so a missing
    // column is refused in the gate's words on every route
    // (`validate_layer_columns`). A statistic sub-layer's dropped inherited
    // reads are cleared from what it draws and warned about once per
    // column, naming the first channel that dropped it.
    let mut column_warnings = Vec::new();
    {
        let pipeline = PipelineBatches { source: source_schema.as_ref(), outputs: &transform_outputs, lineages: &transform_lineages };
        // The color scale's reads of a layered spec, as the scale reads them
        // (the call `establishing_binding` makes): a layer whose own literal
        // paint clears its inherited color from what it draws still reads it
        // into the scale. Empty on an unlayered spec, whose one layer draws
        // the chart's own color.
        let color_channel = crate::spec::encoding::AUXILIARY_SCALE_CHANNELS.iter().find(|channel| channel.name == "color");
        let color_reads = match (spec.layers.as_deref(), color_channel) {
            (Some(raw), Some(color)) => {
                crate::render::scale_resolve::aux_reads(raw, &spec.encoding, *color, &transformed, &transform_outputs)
            }
            _ => Vec::new(),
        };
        for (i, (layer, own_batch)) in layers.iter_mut().zip(&layer_batches).enumerate() {
            // A binned color cuts numbers: each binned binding is judged
            // against its own layer's column, whichever layer establishes
            // the scale, and before the empty-rows skip, since a column's
            // dtype stands without rows. The bindings judged are both sets
            // a binned rule reaches: the ones the layer draws and groups by
            // (`binned_color_bindings`) and the ones the color scale cuts
            // (`color_reads`). A column the rows lack is left to the
            // missing-column refusal below.
            for binding in layer.binned_color_bindings() {
                if let Some(col) = own_batch.column_by_name(&binding.field) {
                    crate::render::scale_resolve::require_binnable(&binding, &binding.field, col.as_ref())?;
                }
            }
            for read in color_reads.iter().filter(|read| read.layer() == i) {
                let Some(binding) = read.binned_binding() else { continue };
                if let Some(located) = read.locate(&transformed, &transform_outputs) {
                    crate::render::scale_resolve::require_binnable(binding, read.field(), located.col)?;
                }
            }
            if own_batch.num_rows() == 0 {
                continue;
            }
            let raw = spec.layers.as_ref().and_then(|raw| raw.get(i));
            let flipped = crate::spec::coord::flips_positional_channels(layer.coord.as_ref());
            let entry = layer_entry(spec, i, layer.mark);
            let voice = public_layer(spec, i, layer.mark).is_none().then(|| entry.clone());
            let dropped = validate_layer_columns(
                i,
                layer,
                raw,
                &layer.drawn_encoding(),
                own_batch,
                flipped,
                &spec.conditionals,
                &pipeline,
                voice,
            )?;
            for (k, read) in dropped.iter().enumerate() {
                if dropped[..k].iter().any(|earlier| earlier.field == read.field) {
                    continue;
                }
                // Sibling sub-layers a desugaring call built carry no index
                // and can raise the same text; it is reported once.
                let warning = RenderWarning::StatisticDroppedChannel {
                    layer: public_layer(spec, i, layer.mark),
                    entry: entry.clone(),
                    mark: layer.mark,
                    channel: read.channel.to_string(),
                    field: read.field.clone(),
                };
                if !column_warnings.contains(&warning) {
                    column_warnings.push(warning);
                }
            }
            layer.dropped_reads = dropped;
        }
    }

    // A layered spec whose axis a continuous caller scale establishes draws
    // every layer on that axis, so each layer's column must be placeable on
    // it, and a layer whose own scale that axis leaves out is reported. Asked
    // once per spec, before any scale resolution reads a column; a geoshape
    // chart resolves no positional axis, so it has none to ask about.
    let mut layer_scale_warnings = Vec::new();
    if !matches!(spec.mark, crate::spec::mark::Mark::Geoshape) {
        for channel in ["x", "y"] {
            crate::render::scale_resolve::require_layers_placeable_on_caller_scale(
                spec,
                channel,
                &transformed,
                &transform_outputs,
            )?;
            crate::render::scale_resolve::warn_layers_off_caller_scale(
                spec,
                channel,
                &transformed,
                &transform_outputs,
                &mut layer_scale_warnings,
            );
        }
    }

    // Build provisional scales and axes from the rendering encoding: the first
    // layer's resolved encoding, which already incorporates CoordFlip, with
    // each auxiliary-scale channel taken from the first layer that draws it
    // (`establishing_encoding`). For single-layer non-flipped specs
    // this is identical to what Phase 7 computed (same encoding, same spec
    // fields).
    //
    // We need a ChartSpec whose encoding reflects the (possibly swapped) channels.
    // Clone spec and substitute the rendering encoding so resolve_scales works
    // correctly. For back-compat (single-layer, no flip), this clone is structurally
    // equal to spec itself — goldens should be byte-identical.
    let rendering_encoding = &establishing_encoding(spec, &layers, &transformed, &transform_outputs, leaf_scales)?;
    let mut rendering_spec = ChartSpec {
        encoding: rendering_encoding.clone(),
        ..spec.clone()
    };
    // The legends read these scales, so a `domainParam` is substituted here as
    // the panel pass substitutes it for the marks.
    crate::render::scale_resolve::resolve_param_domains(&mut rendering_spec);

    // This call resolves over a metadata-only `(0.0, 1.0)` placeholder pixel
    // range (needed only for the DOMAIN, to feed `compute_layout`'s
    // tick/label sizing before the real per-panel plot area exists) — so its
    // warnings are NOT uniformly safe to keep: `GroupCardinalityDegenerate`
    // (#183) asks "is the resolved bandwidth sub-pixel", which is untrue by
    // construction over a 1px-wide placeholder for any domain of 2+
    // categories, regardless of the chart's real rendered size. Every OTHER
    // scale warning (color scheme, palette overflow, color-range parse
    // failures, sort-spec-ignored, ...) does not depend on the pixel range at
    // all and is exactly as real here as it would be from the panel-real
    // pass — dropping those unconditionally was a REGRESSION (quality-review
    // finding #1): `build_scene`'s panel loop `continue`s past every
    // zero-row panel, so a chart whose data is empty after transforms never
    // reaches the panel-real resolution that this comment used to claim
    // would "re-emit" everything. `drop_placeholder_only_warnings` is the
    // fix: filter out only the class this placeholder pass cannot honestly
    // judge, and keep the rest.
    let (provisional_scales, provisional_scale_warnings) =
        crate::render::scale_resolve::resolve_scales_with_leaf_context(
            &rendering_spec,
            &spec.encoding,
            &transformed,
            &transform_outputs,
            &transform_lineages,
            (0.0, 1.0),
            (0.0, 1.0),
            theme,
            &crate::render::scale_resolve::TextEdgeDemands::default(),
            None,
            leaf_scales,
        )?;

    // Derive both axis inputs + their tick counts (SPINE-08/SPINE-12). The
    // rendering encoding is post-CoordFlip; the spec-level encoding (whose
    // explicit title wins) is threaded through for the title 3-way. Also
    // derives the provisional secondary y-axis inputs (secondary-y-axis, GH
    // #52) from `layers`/`transformed`/`transform_outputs`, pushing any scale
    // warnings into `scale_warnings` alongside the primary's.
    // #162: merge the transform layer's own degrade warnings in, wrapped as
    // `RenderWarning::Transform` (mirrors `RenderWarning::Layout`'s
    // newtype-wrap idiom). Order doesn't matter to `emit_warnings` (each
    // becomes its own independent Python `UserWarning`), so these are simply
    // collected as the seed of `scale_warnings`, which every later stage in
    // this function (axis input building, legend construction, ...) keeps
    // accumulating into.
    let mut scale_warnings: Vec<RenderWarning> = transform_warnings
        .into_iter()
        .map(RenderWarning::Transform)
        .chain(column_warnings)
        .chain(undrawn_channel_warnings(spec, &layers))
        .chain(layer_scale_warnings)
        .chain(drop_placeholder_only_warnings(provisional_scale_warnings))
        .collect();
    let mut provisional_scales = provisional_scales;
    // Chart-level `domain_min`/`domain_max`/`nice`/`zero`
    // reshape the resolved positional domains BEFORE the axes derive their
    // ticks from them, so the axis labels and the mark placement describe the
    // same domain.
    apply_axis_domain_configs(
        &mut provisional_scales.x,
        &mut provisional_scales.y,
        rendering_encoding,
        chart_config,
        &mut scale_warnings,
    )?;
    let provisional_scales = provisional_scales;
    let PreparedAxes {
        axes,
        x_tick_count,
        y_tick_count,
        secondary_y_scales,
        secondary_y_tick_counts,
        x_text_rows,
        y_text_rows,
    } = build_axes(
        spec,
        rendering_encoding,
        &provisional_scales,
        theme,
        &layers,
        &layer_batches,
        &y_slot_plan,
        &transformed,
        &transform_outputs,
        &transform_lineages,
        chart_config,
        leaf_scales,
        &mut scale_warnings,
    )?;

    let facet_groups = if let Some(fspec) = &spec.facet {
        let chart_fields = (
            rendering_encoding.x.as_ref().map(|e| e.field.as_str()),
            rendering_encoding.y.as_ref().map(|e| e.field.as_str()),
        );
        let rows = facet_panels::FacetRows::new(
            spec,
            &transformed,
            &transform_outputs,
            &transform_lineages,
            &layer_batches,
            &facet_partitions,
            chart_fields,
            coord_flipped,
        )?;
        facet_panels::facet_groups(&rows, fspec, &mut scale_warnings)?
    } else {
        Vec::new()
    };

    // Color legend / colorbar / aux-legend construction (categorical entries,
    // continuous colorbar, conditional-color fallback, per-channel style
    // overrides, size/shape aux legends, the same-field color+size merge, and
    // the line/ribbon inert-continuous-color suppression — the last needs
    // `layers` (built above) to see the mark set consuming the shared color
    // scale, and `scale_warnings` (already live from `build_axes` above) as
    // its warnings sink. See `legend::build_color_legend` for the full
    // behavior.
    //
    // Under the composite path `layers` above only ever sees THIS leaf's own
    // marks: a sibling leaf elsewhere in an Overlay group (e.g.
    // `fm.layer(line(color=v), point(color=v))`) is invisible to it, since
    // each leaf renders through its own standalone `prepare_render_inputs`
    // call. `leaf_scales` (the composite seam) is the one place that already
    // sees the whole group, so its verdicts reach `build_color_legend` with
    // it. `None` for every standalone (flat/facet) render.
    //
    // Every legend reads its binding off the rendering encoding its scale was
    // built from (`LegendBindings`).
    //
    // A leaf of an overlay group that resolves its auxiliary scales as one
    // chart builds legends as a member of it: the first member builds one
    // legend set, read as a layered chart reads its legends, from the
    // group's bindings (plus its own for the channels the group leaves to
    // its members) over every member's layers; a later member builds only
    // its own legends of the channels left to the members, against its own
    // scales of them.
    let member_scales;
    let legend_source: Option<(Cow<'_, crate::spec::encoding::Encoding>, &[LayerPrepared], &ResolvedScales)> =
        match leaf_scales.and_then(|c| c.overlay_legends(rendering_encoding)) {
            None => Some((Cow::Borrowed(rendering_encoding), layers.as_slice(), &provisional_scales)),
            Some(OverlayLegends::Composed { encoding, layers: group_layers }) => {
                Some((Cow::Owned(encoding), group_layers, &provisional_scales))
            }
            Some(OverlayLegends::LeftToMembers { encoding, channels }) => {
                let mut scales = provisional_scales.clone();
                channels.members_only_scales(&mut scales);
                member_scales = scales;
                Some((Cow::Owned(encoding), layers.as_slice(), &member_scales))
            }
            Some(OverlayLegends::None) => None,
        };
    let legend::ColorLegendBundle {
        legend_entries,
        colorbar,
        legend_title,
        legend_overrides,
        aux_legends,
    } = match legend_source {
        Some((legend_encoding, legend_layers, legend_scales)) => {
            let legend_bindings = legend::LegendBindings::new(spec, &legend_encoding);
            legend::refuse_format_temporality_mismatch_legend_fields(&legend_bindings, &transformed, chart_config)?;
            // The colorbar shows the scale the marks are drawn from, so it is
            // built from the color scale with `configure_color(domain=,
            // range=)` applied, which the pipeline applies to
            // `provisional_scales` only after this.
            let colorbar_color =
                crate::render::config_apply::configured_color_scale(legend_scales.color.as_ref(), chart_config)?;
            legend::build_color_legend(
                &legend_bindings,
                &transformed,
                legend_scales,
                colorbar_color.as_ref(),
                legend_layers,
                leaf_scales,
                &mut scale_warnings,
            )
        }
        None => legend::ColorLegendBundle::none(),
    };

    Ok(PreparedInputs {
        transform_outputs,
        transform_lineages,
        provisional_scales,
        axes,
        facet_groups,
        facet_emptied_by_drop: facet_partitions.emptied_by_drop,
        legend_entries,
        legend_title,
        colorbar,
        warnings: scale_warnings,
        rendering_encoding: rendering_encoding.clone(),
        layers,
        coord_flipped,
        legend_overrides,
        aux_legends,
        x_tick_count,
        y_tick_count,
        secondary_y_scales,
        secondary_y_tick_counts,
        y_slot_plan,
        x_text_rows,
        y_text_rows,
    })
}

/// The pipeline run a render draws from ([`run_transforms_with_facet`]).
///
/// A layer whose drawn color, or whose text grouping-only `group` key,
/// carries a `bin` rule demands that the lowered statistics producing its
/// rows group that column by the color's bins
/// ([`crate::transform::color_bin_group`]), cut by the edges the chart's
/// color scale draws, so the scale, the grouping and the legend share one
/// edge set. Those edges are cut from the rows the binned statistics
/// consume, so such a chart runs twice: the first run groups by raw values
/// and records, per output, the lowered steps its rows passed and the rows
/// they consumed; the demand is read off those records ([`BinDemand::of`]),
/// the color scale resolves its edges over those rows
/// ([`binned_color_edges`]), and the second run groups the demanded steps by
/// them. A chart no demanded step groups, or whose color scale has no edges
/// (an empty or all-null column, a scale another kind establishes), keeps
/// the first run.
fn run_pipeline(
    spec: &ChartSpec,
    normalized: &RecordBatch,
    viewport: crate::layout::Viewport,
    theme: &crate::layout::ThemeInputs,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
) -> Result<TransformRun, RenderError> {
    let first = run_transforms_with_facet(spec, normalized, viewport, theme, None)?;

    // One list of every layer's drawn read into a demanded log axis,
    // computed once — both the
    // residplot-y and sqrt-x gates below read it rather than each
    // re-resolving which layer and scale produced a read, and
    // `SpaceDemand::of` still takes its own plain tuple shape, mapped from
    // this list.
    let space_reads = log_space_reads(spec, &first.0, leaf_scales);
    refuse_residplot_log_y(spec, &space_reads)?;
    refuse_sqrt_on_log_x(spec, &space_reads, &first.2)?;

    let color_reads = binned_color_reads(spec);
    let color_reads: Vec<(&str, &str)> = color_reads.iter().map(|(key, field)| (key.as_str(), field.as_str())).collect();
    let color_demand = BinDemand::of(&first.2, &color_reads);

    let space_demand_reads: Vec<(String, crate::spec::layer::PositionalAxis, String, LogAxisSpace)> =
        space_reads.into_iter().map(LogSpaceRead::into_tuple).collect();
    let space_demand = SpaceDemand::of(&first.2, &space_demand_reads)?;

    if !color_demand.any() && !space_demand.any() {
        return Ok(first);
    }

    let edges = if color_demand.any() {
        binned_color_edges(spec, &first.0, &first.2, theme, leaf_scales).map(std::rc::Rc::from)
    } else {
        None
    };
    // When color is the only demand and its edges cannot resolve, there is
    // no second run: the render's own scale pass handles that case in its
    // place. A step this chart's color demands is otherwise harmless to
    // carry with no edges (`StepBins::bins` gates on `edges.is_some()`),
    // so a space demand alongside still reruns.
    if color_demand.any() && edges.is_none() && !space_demand.any() {
        return Ok(first);
    }

    let plan = RunPlan { color: ColorBinPlan { demand: color_demand, edges }, space: space_demand };
    run_transforms_with_facet(spec, normalized, viewport, theme, Some(&plan))
}

/// One layer's drawn read into a demanded log axis ([`log_space_reads`]):
/// `layer` is that layer's index in [`build_layers`]'s own order, read by
/// both [`refuse_sqrt_on_log_x`] and [`refuse_residplot_log_y`] to voice
/// their refusal in that layer's own [`public_layer`]/[`layer_entry`]
/// rather than re-resolving which layer and scale produced each read a
/// second time: the two gates
/// read this ONE list, computed once in [`run_pipeline`] before
/// [`SpaceDemand::of`], instead of each walking `build_layers`/
/// `establishing_encoding`/`resolved_explicit_scale` on its own.
pub(crate) struct LogSpaceRead {
    pub(crate) layer: usize,
    pub(crate) key: String,
    pub(crate) axis: crate::spec::layer::PositionalAxis,
    pub(crate) column: String,
    pub(crate) space: LogAxisSpace,
}

impl LogSpaceRead {
    /// This read's `(key, axis, column, space)` tuple — [`SpaceDemand::of`]'s
    /// own plain shape, with `layer` dropped (that index is this gate's own
    /// addition; `SpaceDemand::of`'s lineage walk never needed it).
    fn into_tuple(self) -> (String, crate::spec::layer::PositionalAxis, String, LogAxisSpace) {
        (self.key, self.axis, self.column, self.space)
    }
}

/// For each layer (the chart itself when unlayered) whose drawn `x` or `y`
/// resolves to a `LogScale`, the key it draws under and that axis's log
/// space ([`LogAxisSpace`]) — the demand [`SpaceDemand::of`] reads against
/// the first pipeline run's lineages, mirroring [`binned_color_reads`].
///
/// `enc` for each channel is [`establishing_encoding`]'s positional binding
/// (layer 0's field and title); the scale itself is the selection
/// [`crate::render::scale_resolve::domain::resolved_explicit_scale`] makes
/// — the caller's own scale when it states one, else the scale a sibling
/// layer establishes — the same selection
/// [`crate::render::scale_resolve::positional::build_axis_scale`] resolves
/// its real scale from, so every door that reaches a `LogScale` (a
/// channel's own `scale=`, an `override()`-merged scale, or a sibling
/// layer's established scale) reaches this demand too. Empty for a chart
/// whose `x`/`y` channel resolves to anything else, or whose layers cannot
/// be built.
///
/// An independent-y layer owns its own y slot: the shared selection
/// above only ever covers slot `0`'s layers, so each independent-y layer
/// below selects its OWN scale over the exact per-layer spec
/// [`crate::render::scale_resolve::resolve_layer_y_slot_scale`] resolves
/// its real scale from ([`crate::render::scale_resolve::layer_y_slot_spec`]),
/// by the same [`resolved_explicit_scale`] rule — so a secondary histogram
/// bins (or not) in its own slot's space, never the primary's.
fn log_space_reads(
    spec: &ChartSpec,
    outputs: &HashMap<String, RecordBatch>,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
) -> Vec<LogSpaceRead> {
    use crate::spec::encoding::ScaleSpec;
    use crate::spec::layer::PositionalAxis;
    let Some(primary_batch) = outputs.get(FINAL_OUTPUT_KEY) else { return Vec::new() };
    let Ok(layers) = build_layers(spec) else { return Vec::new() };
    let Ok(encoding) = establishing_encoding(spec, &layers, primary_batch, outputs, leaf_scales) else {
        return Vec::new();
    };
    let y_slot_plan = crate::render::scale_resolve::YSlotPlan::from_layer_flags(layers.iter().map(|l| l.independent_y));
    let mut reads = Vec::new();
    for channel in ["x", "y"] {
        let own = if channel == "x" { encoding.x.as_ref() } else { encoding.y.as_ref() };
        // A composite leaf that declares no scale of its own on this
        // channel (or only the injected bar anchor) resolves with its
        // `Overlay` group's own caller scale, exactly as
        // `resolve_scales_with_leaf_context` adopts it for the real
        // scale resolve (`with_group_scale`, the same function) — so a
        // histogram leaf under a group a sibling's `LogScale` establishes
        // is demanded here too, not just the leaf that carries it.
        let group_scale = leaf_scales.and_then(|c| if channel == "x" { c.x_scale.as_ref() } else { c.y_scale.as_ref() });
        let adopted = crate::render::scale_resolve::with_group_scale(own, group_scale);
        let Some(enc) = adopted.as_ref().or(own) else { continue };
        let shared = leaf_scales.and_then(|c| if channel == "x" { c.x.as_ref() } else { c.y.as_ref() });
        let resolved = crate::render::scale_resolve::resolved_explicit_scale(channel, enc, primary_batch, outputs, spec, shared);
        let Some(ScaleSpec::Log { base, common, .. }) = resolved else { continue };
        let side = crate::scale::log::LogSide::of(common.domain.as_deref());
        let space = LogAxisSpace { base: *base, side };
        let axis = if channel == "x" { PositionalAxis::X } else { PositionalAxis::Y };
        for (idx, layer) in layers.iter().enumerate() {
            // An independent-y layer owns its own y slot, selected below
            // over its own per-layer spec; slot 0's selection never reaches
            // it.
            if channel == "y" && y_slot_plan.slot_for_layer(idx) != 0 {
                continue;
            }
            // The drawn column is THIS layer's own post-lowering field (a
            // chart's statistic plan rebinds a layer's x/y encoding to the
            // computed column it draws — "bin_start", "value", "grid_x" —
            // never the chart-level establishing encoding's own field,
            // since an overlay's members can draw different columns).
            let drawn_column = if channel == "x" { layer.encoding.x.as_ref() } else { layer.encoding.y.as_ref() };
            let Some(drawn_column) = drawn_column.map(|e| e.field.clone()) else { continue };
            let key = layer.data_source.clone().unwrap_or_else(|| FINAL_OUTPUT_KEY.to_string());
            reads.push(LogSpaceRead { layer: idx, key, axis, column: drawn_column, space });
        }
    }
    // Each independent-y layer's own slot: its scale is resolved over the
    // same per-layer spec `resolve_layer_y_slot_scale` resolves its real
    // scale from, never the chart's establishing encoding above.
    for (slot_index, &layer_idx) in y_slot_plan.secondary_layers().iter().enumerate() {
        let y_slot = slot_index + 1;
        let layer = &layers[layer_idx];
        let layer_spec = crate::render::scale_resolve::layer_y_slot_spec(spec, layer);
        let Some(enc) = layer_spec.encoding.y.as_ref() else { continue };
        let resolved = crate::render::scale_resolve::resolved_explicit_scale("y", enc, primary_batch, outputs, &layer_spec, None);
        let Some(ScaleSpec::Log { base, common, .. }) = resolved else { continue };
        let side = crate::scale::log::LogSide::of(common.domain.as_deref());
        let space = LogAxisSpace { base: *base, side };
        let Some(drawn_column) = layer.encoding.y.as_ref().map(|e| e.field.clone()) else { continue };
        let key = layer.data_source.clone().unwrap_or_else(|| FINAL_OUTPUT_KEY.to_string());
        reads.push(LogSpaceRead { layer: layer_idx, key, axis: PositionalAxis::secondary_y(y_slot), column: drawn_column, space });
    }
    reads
}

/// `residplot(...)`'s residual axis draws signed residuals around 0, which
/// no `LogScale` can place: refused the moment its y channel resolves to
/// one, through any door (`y=fm.Y(scale=...)`, `encode=`,
/// `.override(y_scale_type="log")`) — `reads` (`log_space_reads`) already
/// folds every door into one resolution, so this gate reads it rather than
/// re-resolving the scale itself.
/// Refuses at the FIRST `reads` entry on a `y` axis (the primary slot `0`
/// or any independent-y secondary slot) whose own layer's effective entry
/// is `"residplot"`. `residplot` stamps every internal layer it builds
/// with its own `entry` directly (it writes no chart-level `spec.entry` at
/// all, [`layer_entry`]'s fallback never triggers here), so this reads
/// that layer's own effective entry rather than the chart's.
fn refuse_residplot_log_y(spec: &ChartSpec, reads: &[LogSpaceRead]) -> Result<(), RenderError> {
    use crate::spec::layer::PositionalChannel;
    let Ok(layers) = build_layers(spec) else { return Ok(()) };
    for read in reads {
        if read.axis.channel != PositionalChannel::Y {
            continue;
        }
        let Some(layer) = layers.get(read.layer) else { continue };
        if layer_entry(spec, read.layer, layer.mark) != "residplot" {
            continue;
        }
        return Err(RenderError::ResidplotLogY.in_layer_context(public_layer(spec, read.layer, layer.mark), &layer_entry(spec, read.layer, layer.mark)));
    }
    Ok(())
}

/// `mark_smooth(method="sqrt")` composes with no other predictor transform:
/// refused the moment a `LogScale` demands the x it fits, in the layer
/// entry's own voice, before the fit ever runs in that space
/// (`transform::smooth::apply_in_log_space` carries a backstop of the same
/// refusal, unvoiced, for the case this gate somehow misses). Reads `reads`
/// (`log_space_reads`) for which layers draw a demanded log `x` at all,
/// then, for the first `x` read, walks that layer's own drawn x back
/// through its lineage
/// ([`OutputLineage::space_walk`], the same walk [`SpaceDemand::of`] takes)
/// to the real `TransformSpec` that computed it ([`step_spec_at`]), since a
/// drawn column's lineage carries no fact about which fit *method* produced
/// it, only that something did.
fn refuse_sqrt_on_log_x(spec: &ChartSpec, reads: &[LogSpaceRead], lineages: &OutputLineages) -> Result<(), RenderError> {
    use crate::spec::layer::PositionalChannel;
    use crate::transform::smooth::SmoothMethod;
    let Ok(layers) = build_layers(spec) else { return Ok(()) };
    for read in reads {
        if read.axis.channel != PositionalChannel::X {
            continue;
        }
        let Some(layer) = layers.get(read.layer) else { continue };
        let Some(lineage) = lineages.get(&read.key) else { continue };
        for step in lineage.space_walk(&read.column) {
            if step.slot != SpaceSlot::X {
                continue;
            }
            let Some(TransformSpec::Smooth(s)) = step_spec_at(&spec.transforms, &lineage.lowered[step.index].id) else { continue };
            if matches!(s.method, SmoothMethod::SqrtX) {
                return Err(RenderError::SmoothSqrtOnLogX
                    .in_layer_context(public_layer(spec, read.layer, layer.mark), &layer_entry(spec, read.layer, layer.mark)));
            }
        }
    }
    Ok(())
}

/// The binned reads a run's demand comes from: for each layer (the chart
/// itself on an unlayered spec) whose drawn color, or whose grouping-only
/// `group` key, carries a `bin` rule, the key it draws
/// ([`FINAL_OUTPUT_KEY`] for a layer with no `data_source`) and the binned
/// column. Empty for a spec whose layers cannot be built, which the
/// render's own refusal reports.
fn binned_color_reads(spec: &ChartSpec) -> Vec<(String, String)> {
    let Ok(layers) = build_layers(spec) else { return Vec::new() };
    layers
        .iter()
        .flat_map(|layer| {
            let key = layer.data_source.clone().unwrap_or_else(|| FINAL_OUTPUT_KEY.to_string());
            layer.binned_color_bindings().into_iter().map(move |binding| (key.clone(), binding.field))
        })
        .collect()
}

/// The edges the chart's color scale cuts when it is a binned color's,
/// resolved over a first pipeline run's `outputs` as the render resolves it
/// (its establishing binding, a composite's shared values or an overlay
/// group's scale), from the rows each drawing layer's statistic consumes.
/// `None` when the color scale is another kind or has no edges. A scale
/// that cannot resolve gives `None` too: the render's own scale pass reads
/// the same rows and raises that refusal in its place among the render's
/// checks.
fn binned_color_edges(
    spec: &ChartSpec,
    outputs: &HashMap<String, RecordBatch>,
    lineages: &OutputLineages,
    theme: &crate::layout::ThemeInputs,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
) -> Option<Vec<f64>> {
    let transformed = outputs.get(FINAL_OUTPUT_KEY)?;
    let layers = build_layers(spec).ok()?;
    let encoding = establishing_encoding(spec, &layers, transformed, outputs, leaf_scales).ok()?;
    encoding.color.as_ref()?.color_bin()?;
    let mut rendering = ChartSpec { encoding, ..spec.clone() };
    crate::render::scale_resolve::resolve_param_domains(&mut rendering);
    let scale = crate::render::scale_resolve::resolve_color_scale(&rendering, &spec.encoding, transformed, outputs, lineages, theme, leaf_scales);
    match scale.ok()?? {
        crate::render::scale_resolve::ColorScale::Discretizing(buckets) => Some(buckets.bounds().to_vec()),
        _ => None,
    }
}

/// What [`run_transforms_with_facet`] produces: every published output batch,
/// the transform warnings drained from the run, each output's lineage
/// ([`PreparedInputs::transform_lineages`]), and a faceted run's partitions
/// (none unfaceted), which order and complete the facet's panels.
type TransformRun = (
    HashMap<String, RecordBatch>,
    Vec<crate::transform::context::TransformWarning>,
    OutputLineages,
    FacetPartitions,
);

/// A faceted run's partitions: every key in the order the run partitioned
/// the facet's input, and the keys a step's `groupby` null-key drop emptied
/// ([`crate::transform::context::TransformContext::refuse_unless_partition`]),
/// whose panels draw empty rather than vanish.
#[derive(Debug, Default)]
pub(super) struct FacetPartitions {
    pub(super) order: Vec<FacetPartitionKey>,
    pub(super) emptied_by_drop: HashSet<FacetPartitionKey>,
}

/// One key's facet-merge accumulator: its lineage (minus the rows its
/// statistics consumed, taken out per panel and merged separately below),
/// each panel's consumed rows to merge into [`OutputLineage::statistic_input`],
/// and each `(axis, field)` a demanded step recorded on some panel, one
/// [`ArrayRef`] per panel to concatenate into [`OutputLineage::log_input`].
type MergedLineage = (OutputLineage, Vec<RecordBatch>, HashMap<(crate::spec::layer::PositionalAxis, String), Vec<ArrayRef>>);

/// Run the Phase 5 transform pipeline, producing the named-output map (SPINE-12).
///
/// When faceting is active, partition the input batch by the facet column(s)
/// BEFORE running transforms so each panel gets its own data subset and
/// transforms execute independently per panel; the facet column(s) are
/// re-injected into every output and per-key batches are concatenated across
/// panels. When there is no facet, the pipeline is the plain single-partition
/// path (one partition = full batch). Both branches apply the D10 impute step
/// (`apply_impute_to_final`) exactly once on `FINAL_OUTPUT_KEY`.
///
/// `viewport`/`theme` seed `TransformContext.panel_pixel_size` with the M2
/// pre-layout estimate (spec §4.2; `layout::estimate_panel_pixel_size`'s doc)
/// — the viewport minus the theme's padding, divided per-axis by this chart's
/// OWN facet grid dimensions when it facets (`(1, 1)` otherwise). This is the
/// single production construction site for `TransformContext`; every other
/// `TransformContext::default()` in this crate is a direct-`apply()` fallback
/// path, unaffected by this estimate.
///
/// The facet branch also sets `TransformContext.is_partition = true` (batch-D
/// T9 round 3): each per-panel `apply_transforms_named` call below runs on a
/// facet panel, not the whole frame, even though `groupby` is empty for that
/// call. Consumers that gate a refusal on "this is a genuine whole-frame
/// call" (e.g. `Smooth`'s rank-deficiency check) read this flag to degrade
/// like a `groupby` group instead of failing the whole render — see
/// `TransformContext::is_partition`'s doc.
fn run_transforms_with_facet(
    spec: &ChartSpec,
    normalized: &RecordBatch,
    viewport: crate::layout::Viewport,
    theme: &crate::layout::ThemeInputs,
    run_plan: Option<&RunPlan>,
) -> Result<TransformRun, RenderError> {
    let run_plan = run_plan.map(|plan| std::rc::Rc::new(plan.clone()));
    let Some(fspec) = &spec.facet else {
        // No facet: unchanged pipeline (single partition = full batch); one
        // panel occupies the whole estimated viewport.
        let ctx = TransformContext {
            panel_pixel_size: Some(crate::layout::estimate_panel_pixel_size(
                viewport,
                theme,
                (1, 1),
            )),
            run_plan: run_plan.clone(),
            ..TransformContext::default()
        };
        let (mut outputs, lineages) = apply_transforms_named(&spec.transforms, normalized, &ctx)
            .map_err(|e| transform_refusal(e.to_string(), &ctx))?;
        apply_impute_to_final(&mut outputs, spec);
        // #162: drain whatever this run's (possibly per-step-cloned, but
        // `Rc`-shared) sink accumulated — see `TransformContext::warnings`'s
        // doc for why `Rc<RefCell<_>>` makes this safe across the per-step
        // `ctx.clone()` in `apply_transforms_named`. `merge_null_key_reports`
        // collapses a composite's own several lowered steps over the same
        // rows and key into one report, as the faceted return does.
        let warnings = crate::transform::context::merge_null_key_reports(
            ctx.warnings.borrow_mut().drain(..).collect(),
        );
        return Ok((normalize_transform_outputs(outputs)?, warnings, normalize_transform_lineages(lineages)?, FacetPartitions::default()));
    };

    // Facet-before-transform: partition → per-panel transforms → inject facet column(s) → concat.
    //
    // Grid mode (fspec.row is set): partition on the composite (col_val, row_val) key so each
    // (row, col) cell is a distinct partition. Both the col field and the row field are injected
    // back into every transform output so per-panel filtering can use either.
    //
    // Wrap mode (fspec.row is None): partition on fspec.field only — behavior unchanged.
    // Each partition carries (col_val, optional_row_val) → RecordBatch.
    // Grid mode populates the row value; wrap mode leaves it None.
    //
    // A row its operand's frame lacks a facet field for repeats along that
    // field in every panel (`FacetSpec::repeat`); every other row with a null
    // facet value belongs to no panel and is dropped, one warning per field.
    refuse_malformed_facet_repeats(fspec)?;
    let col_repeat = facet_repeat_mask(normalized, &fspec.repeat, &fspec.field)?;
    let mut dropped_nulls = vec![(
        fspec.field.as_str(),
        null_facet_rows(&crate::render::facet_key_column(normalized, &fspec.field)?, &col_repeat),
    )];
    let partitions: Vec<(FacetPartitionKey, RecordBatch)> = if normalized.num_rows() == 0 {
        // Zero-row input has no facet VALUE to partition by at all -- there
        // is nothing to group. Rather than refuse, give the per-partition
        // loop below one placeholder partition over the whole (empty)
        // batch, so every step it already runs (the transform chain,
        // per-panel imputation, facet-column re-injection, lineage
        // merging) still runs once and still publishes `FINAL_OUTPUT_KEY`
        // with the right (empty) schema, exactly as it does for a REAL
        // facet value. The placeholder key itself never becomes a panel:
        // `facet_panels::held_values` builds panel membership from the
        // rows a layer actually DRAWS, which is none here, so this
        // reaches the SAME "no layer draws any facet value" state a
        // filter-emptied facet already reaches (via its own, unrelated
        // route: real facet values whose per-partition rows the filter
        // happens to empty) -- one downstream reduction, not two.
        vec![((String::new(), fspec.row.is_some().then(String::new)), normalized.clone())]
    } else if let Some(row_field) = &fspec.row {
        let row_repeat = facet_repeat_mask(normalized, &fspec.repeat, row_field)?;
        dropped_nulls.push((row_field.as_str(), null_facet_rows(&crate::render::facet_key_column(normalized, row_field)?, &row_repeat)));
        partition_batch_by_two_fields(
            normalized,
            (&fspec.field, &col_repeat),
            (row_field, &row_repeat),
        )?
        .into_iter()
        .map(|((col_val, row_val), batch)| ((col_val, Some(row_val)), batch))
        .collect()
    } else {
        partition_batch_by_field(normalized, &fspec.field, &col_repeat)?
            .into_iter()
            .map(|(col_val, batch)| ((col_val, None), batch))
            .collect()
    };
    // Every row's facet key (the col field's, or in grid mode either field's)
    // is null: `dropped_nulls` above already counts every row for its field,
    // so the warning loop below reports the drop in full. This is the
    // drop-and-warn rule at its limit, not a different input class, so it
    // takes the same one-placeholder route the zero-row branch above takes
    // (over an empty slice of the null-dropped batch) rather than refusing
    // a frame the single-null-row case already renders.
    let partitions: Vec<(FacetPartitionKey, RecordBatch)> = if partitions.is_empty() {
        vec![((String::new(), fspec.row.is_some().then(String::new)), normalized.slice(0, 0))]
    } else {
        partitions
    };

    // Grid dims for the per-cell panel-size estimate (M2): grid mode's
    // nrows/ncols are explicit on the facet spec; wrap mode's ncols is
    // explicit and nrows is derived from the observed panel count the same
    // way `FacetGrid::compute_wrap` derives it for the real layout later, so
    // the estimate divides by the same grid the chart will actually lay out.
    let grid_dims: (u32, u32) = match &fspec.mode {
        crate::layout::FacetMode::Grid { nrows, ncols } => (*ncols, *nrows),
        crate::layout::FacetMode::Wrap { ncols } => {
            let ncols = (*ncols).max(1);
            let n_panels = partitions.len() as u32;
            let nrows = n_panels.div_ceil(ncols);
            (ncols, nrows.max(1))
        }
    };
    let ctx = TransformContext {
        panel_pixel_size: Some(crate::layout::estimate_panel_pixel_size(
            viewport, theme, grid_dims,
        )),
        // Batch-D T9 round 3: every partition run through this ctx is one
        // facet panel, not the whole frame — see `TransformContext::is_partition`'s
        // doc and `smooth::apply_one_group`'s degenerate-input policy, the
        // consumer this unblocks (`fm.lmplot(order=3, col=...)` on a sparse
        // facet level must degrade like a `groupby` group, not refuse the
        // whole chart the way an actual whole-frame rank-deficient fit does).
        is_partition: true,
        facet_fields: std::iter::once(&fspec.field).chain(&fspec.row).cloned().collect(),
        run_plan,
        ..TransformContext::default()
    };
    for (field, rows) in dropped_nulls {
        if rows > 0 {
            ctx.warn(crate::transform::context::TransformWarning::NullKeyRowsDropped {
                scope: crate::transform::context::NullKeyScope::Facet,
                field: field.to_string(),
                rows,
            });
        }
    }

    // Pin a shared value-axis extent from the full (pre-partition) batch for
    // every extent-carrying transform (Kde / Bin / Violin / Kde2D / Bin2D)
    // without an explicit extent. Otherwise each partition would use its own
    // range, causing panels (and hue groups within panels) to render on
    // different value scales and making positions non-comparable. By fixing
    // the extent to the global range before partitioning, all panels and
    // groups share the same value axis. This is the correct default for
    // faceted density / histogram / violin / 2-D density / heatmap charts,
    // including Kde2D/Bin2D.
    let space_demand = ctx.run_plan.as_deref().map(|plan| &plan.space).filter(|space| space.any());
    let effective_transforms = match space_demand {
        Some(space) => fix_transform_extents_for_facet_in_space(&spec.transforms, normalized, Some(space)),
        None => fix_transform_extents_for_facet(&spec.transforms, normalized),
    };

    let mut merged: HashMap<String, Vec<RecordBatch>> = HashMap::new();
    // Every partition runs the same transforms, so each key's lineage
    // routes the same `Swarm` and passes the same steps in every panel; the
    // rows its statistics consumed, and a demanded step's recorded caller
    // columns (keyed by `(axis, field)`), are merged across the panels as
    // the outputs are.
    let mut merged_lineages: HashMap<String, MergedLineage> = HashMap::new();
    // The partitions in which each step's every `groupby` key was null, by
    // step in the order they first reported one.
    let mut all_null_steps: Vec<StepAllNull<'_>> = Vec::new();
    for (key, partition_batch) in &partitions {
        let (col_value, row_value) = key;
        let before_warnings = ctx.warnings.borrow().len();
        let (mut panel_outputs, panel_lineages) =
            apply_transforms_named(&effective_transforms, partition_batch, &ctx)
                .map_err(|e| transform_refusal(e.to_string(), &ctx))?;
        // A per-group fit warning (`DegenerateFit`/`ParameterClamped`/
        // `IntervalUndefined`) this partition's own run just pushed names
        // which facet panel it ran in: two sparse panels at the same
        // groupby size would otherwise report two identical,
        // unattributable warnings. Stamped here, once per partition,
        // mirroring `GroupKeyAllNull`'s own panel naming
        // (`settle_all_null_partitions`/`facet_partition_label`) and voice.
        for warning in ctx.warnings.borrow_mut()[before_warnings..].iter_mut() {
            stamp_facet_panel(warning, fspec, key);
        }
        for AllNullPartition { step, outcome } in ctx.all_null_partitions.borrow_mut().drain(..) {
            let index = all_null_steps.iter().position(|seen| seen.step == step).unwrap_or_else(|| {
                all_null_steps.push(StepAllNull { step, outcome, partitions: HashSet::new() });
                all_null_steps.len() - 1
            });
            all_null_steps[index].partitions.insert(key);
        }
        for (key, mut lineage) in panel_lineages {
            let consumed = lineage.statistic_input.take().map(|rows| {
                let rows = inject_facet_column(&rows, &fspec.field, col_value);
                match (&fspec.row, row_value) {
                    (Some(row_field), Some(row_val)) => inject_facet_column(&rows, row_field, row_val),
                    _ => rows,
                }
            });
            let panel_log_input = std::mem::take(&mut lineage.log_input);
            let (_, batches, log_input) = merged_lineages.entry(key).or_insert_with(|| (lineage, Vec::new(), HashMap::new()));
            batches.extend(consumed);
            for (field_key, column) in panel_log_input {
                log_input.entry(field_key).or_default().push(column);
            }
        }
        // D10: per-panel imputation.
        apply_impute_to_final(&mut panel_outputs, spec);
        // Re-inject the col field (and row field when in grid mode) into every
        // output batch. Transforms like Smooth replace the batch entirely, losing
        // any facet columns that were present on the input partition.
        for batch in panel_outputs.values_mut() {
            *batch = inject_facet_column(batch, &fspec.field, col_value);
            if let (Some(row_field), Some(row_val)) = (&fspec.row, row_value) {
                *batch = inject_facet_column(batch, row_field, row_val);
            }
        }
        for (key, batch) in panel_outputs {
            merged.entry(key).or_default().push(batch);
        }
    }
    // Concat per-key batches across all panels into a single map. Batches
    // published under the same key can have DIVERGING schemas across
    // panels when a facet panel's fit is degenerate (rank-deficient, zero-
    // variance x, non-convergent): before batch-D T9 round 3, a rank-deficient
    // fit always refused, so faceted charts never hit this state; round 3 now
    // makes degenerate panels degrade to NaN instead (via `TransformContext::
    // is_partition`). A plain `concat_batches` requires identical schemas and
    // otherwise misaligns columns POSITIONALLY, surfacing as a confusing error
    // naming an unrelated column. `concat_batches_by_name` unifies every
    // batch's fields in first-appearance order and reads each batch's columns
    // BY NAME, padding a miss with `new_null_array` at the appropriate dtype.
    let mut combined: HashMap<String, RecordBatch> = HashMap::new();
    for (key, batches) in merged {
        if batches.len() == 1 {
            combined.insert(key, batches.into_iter().next().unwrap());
        } else {
            let merged_batch = concat_batches_by_name(&key, batches)?;
            combined.insert(key, merged_batch);
        }
    }
    let order: Vec<FacetPartitionKey> = partitions.iter().map(|(key, _)| key.clone()).collect();
    let AllNullSettled { emptied_by_drop, replaced_counts, warnings: all_null_warnings } =
        settle_all_null_partitions(all_null_steps, &order, fspec, &ctx)?;
    // #162: `ctx` is the SAME context across every partition in the loop
    // above (not re-constructed per panel), so its `Rc`-shared sink
    // accumulated every panel's warnings — one drain covers the whole facet.
    // A transform's per-panel null-key drop counts sum to the frame's count,
    // except a step whose every key was null in every panel, which says so
    // once instead, as on the whole frame. The retain removes a replaced
    // step's own per-panel counts BEFORE the one final merge — which runs
    // AFTER `all_null_warnings` joins the set, so a cross-step dedup (one
    // lowering's several grouped steps over the same rows and key) sees the
    // whole frame's warnings, not just the per-panel counts.
    let mut warnings: Vec<TransformWarning> = ctx.warnings.borrow_mut().drain(..).collect();
    warnings.retain(|warning| match warning {
        TransformWarning::NullKeyRowsDropped {
            scope: crate::transform::context::NullKeyScope::Groupby { step, .. },
            ..
        } => !replaced_counts.contains(step),
        _ => true,
    });
    warnings.extend(all_null_warnings);
    let warnings = crate::transform::context::merge_null_key_reports(warnings);
    let mut lineages = OutputLineages::new();
    for (key, (lineage, batches, log_input)) in merged_lineages {
        let statistic_input = match batches.is_empty() {
            true => None,
            false => Some(concat_batches_by_name(&key, batches)?),
        };
        // A demanded step's recorded caller column is merged across facet
        // partitions by concatenation, exactly as `statistic_input` is
        // above: the off-side tally reads the WHOLE FRAME's column, not
        // one panel's, so a faceted chart's count is the same whether the
        // axis is shared or independent across panels.
        let mut merged_log_input = HashMap::with_capacity(log_input.len());
        for (field_key, columns) in log_input {
            let refs: Vec<&dyn Array> = columns.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| RenderError::TransformFailed(format!("{key}: {e}")))?;
            merged_log_input.insert(field_key, concatenated);
        }
        lineages.insert(key, OutputLineage { statistic_input, log_input: merged_log_input, ..lineage });
    }
    let partitions = FacetPartitions { order, emptied_by_drop };
    Ok((normalize_transform_outputs(combined)?, warnings, normalize_transform_lineages(lineages)?, partitions))
}

/// One step's facet partitions whose every `groupby` key was null, and what
/// the step gives in that case ([`AllNullOutcome`]).
struct StepAllNull<'k> {
    step: StepInstance,
    outcome: AllNullOutcome,
    partitions: HashSet<&'k FacetPartitionKey>,
}

/// What a facet run's all-null partitions come to.
struct AllNullSettled {
    /// The partitions whose panels draw empty.
    emptied_by_drop: HashSet<FacetPartitionKey>,
    /// The steps whose per-panel null-key counts one all-null warning
    /// replaces.
    replaced_counts: Vec<StepInstance>,
    /// The warnings the run reports once per step.
    warnings: Vec<TransformWarning>,
}

/// Settle each step's all-null facet partitions as the whole frame settles
/// a step whose every key is null:
///
/// - a refusing step that emptied every panel raises its refusal, with the
///   run's warnings ([`transform_refusal`]);
/// - a step whose grouped output is empty in every panel reports its
///   all-null warning once, in place of its counts, and every panel draws
///   empty, as the unfaceted chart draws its one panel empty;
/// - a step that fit a panel's whole batch reports its all-null warning
///   once, naming the panels it pooled when that is fewer than every one
///   (`order`/`fspec` build each pooled panel's label, in partition order);
///   pooling every panel keeps the whole-frame text;
///
/// and every panel a step left with nothing to draw is drawn empty.
fn settle_all_null_partitions(
    steps: Vec<StepAllNull<'_>>,
    order: &[FacetPartitionKey],
    fspec: &crate::layout::FacetSpec,
    ctx: &TransformContext,
) -> Result<AllNullSettled, RenderError> {
    let mut settled = AllNullSettled { emptied_by_drop: HashSet::new(), replaced_counts: Vec::new(), warnings: Vec::new() };
    for StepAllNull { step, outcome, partitions } in steps {
        let every_panel = partitions.len() == order.len();
        if outcome.empties() {
            settled.emptied_by_drop.extend(partitions.iter().map(|key| (*key).clone()));
        }
        match outcome {
            AllNullOutcome::Refused(refusal) if every_panel => {
                return Err(transform_refusal(refusal.to_string(), ctx));
            }
            AllNullOutcome::Refused(_) => {}
            AllNullOutcome::Emptied(warning) if every_panel => {
                settled.replaced_counts.push(step);
                settled.warnings.push(warning);
            }
            AllNullOutcome::Emptied(_) => {}
            AllNullOutcome::Pooled(mut warning) => {
                if !every_panel {
                    if let TransformWarning::GroupKeyAllNull { panels, .. } = &mut warning {
                        *panels = order
                            .iter()
                            .filter(|key| partitions.contains(*key))
                            .map(|key| facet_partition_label(fspec, key))
                            .collect();
                    }
                }
                settled.warnings.push(warning);
            }
        }
    }
    Ok(settled)
}

/// One pooled partition's label for [`TransformWarning::GroupKeyAllNull`]'s
/// `panels` field: `"f=F2"`, or `"f=F2, r=R1"` in grid mode — the same
/// `col=value` shape [`crate::transform::context::describe_group_keys`]
/// uses for a groupby key.
pub(crate) fn facet_partition_label(fspec: &crate::layout::FacetSpec, (col, row): &FacetPartitionKey) -> String {
    match (&fspec.row, row) {
        (Some(row_field), Some(row_val)) => format!("{}={col}, {row_field}={row_val}", fspec.field),
        _ => format!("{}={col}", fspec.field),
    }
}

/// Stamp `warning`'s `panel` field with `fspec`/`key`'s
/// [`facet_partition_label`], when it is one of the three per-group fit
/// warnings and does not already carry one (a reader of raw warnings from
/// a composite leaf's own faceted run, re-run through this function, must
/// not overwrite a panel its own facet already set). Every other warning
/// kind is left untouched.
fn stamp_facet_panel(
    warning: &mut TransformWarning,
    fspec: &crate::layout::FacetSpec,
    key: &FacetPartitionKey,
) {
    let panel = match warning {
        TransformWarning::DegenerateFit { panel, .. }
        | TransformWarning::ParameterClamped { panel, .. }
        | TransformWarning::IntervalUndefined { panel, .. } => panel,
        _ => return,
    };
    if panel.is_none() {
        *panel = Some(facet_partition_label(fspec, key));
    }
}

/// The refusal a transform run raises, `refusal` as
/// [`RenderError::TransformFailed`], carrying the warnings the run reported
/// before it ([`RenderError::Warned`]): the null-key drop that left a
/// statistic nothing to draw is reported beside the refusal it caused.
fn transform_refusal(refusal: String, ctx: &TransformContext) -> RenderError {
    let source = RenderError::TransformFailed(refusal);
    let warnings: Vec<RenderWarning> =
        crate::transform::context::merge_null_key_reports(ctx.warnings.borrow_mut().drain(..).collect())
            .into_iter()
            .map(RenderWarning::Transform)
            .collect();
    match warnings.is_empty() {
        true => source,
        false => RenderError::Warned { warnings, source: Box::new(source) },
    }
}

/// Concatenate `batches` — all outputs published under the same named-output
/// `key` across facet panels — into one batch, tolerating schema divergence
/// between panels.
///
/// The reconciliation itself lives in [`crate::transform::concat::concat_by_name`],
/// which this thin wrapper only supplies the error voice for: the union-of-fields
/// logic was hoisted there so `transform::smooth::apply_grouped` could stop
/// carrying a weaker second implementation. See that function for the
/// union-of-fields algorithm and its guarantees.
fn concat_batches_by_name(
    key: &str,
    batches: Vec<RecordBatch>,
) -> Result<RecordBatch, RenderError> {
    crate::transform::concat::concat_by_name(
        &format!("concat facet partitions for key '{key}'"),
        batches,
    )
    .map_err(RenderError::TransformFailed)
}

/// Apply the D10 impute step in place on the pipeline's `FINAL_OUTPUT_KEY`
/// batch, replacing it only when imputation added rows (SPINE-12).
///
/// This is the single call site for both the facet (per-panel) and non-facet
/// transform branches; it formerly appeared verbatim in each.
fn apply_impute_to_final(outputs: &mut HashMap<String, RecordBatch>, spec: &ChartSpec) {
    let final_batch = outputs
        .get(FINAL_OUTPUT_KEY)
        .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY");
    let imputed = apply_impute(final_batch, spec);
    if imputed.num_rows() != final_batch.num_rows() {
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), imputed);
    }
}

/// Refuse a `group` grouping key where nothing reads it
/// ([`RenderError::GroupKeyMisplaced`]): on the chart-level encoding, or on a
/// layer whose mark draws the color it inherits
/// ([`crate::spec::mark::Mark::draws_inherited`]). The key is a text's, for a
/// color it inherits but does not draw. This judges whether a layer may
/// carry a `group` key at all, the spec-level completion fact
/// [`crate::spec::layer::Layer::completed_encoding`] applies at
/// construction — not whether a binding in hand draws, which is
/// [`crate::render::drawn::DrawFacts::draws`]'s question.
fn refuse_misplaced_group_keys(spec: &ChartSpec) -> Result<(), RenderError> {
    use crate::spec::layer::InheritedChannel;
    if spec.encoding.group.is_some() {
        return Err(RenderError::GroupKeyMisplaced { layer: None, mark: spec.mark });
    }
    let misplaced = spec.layers.iter().flatten().enumerate().find(|(_, layer)| {
        layer.encoding.group.is_some() && layer.mark.draws_inherited(InheritedChannel::Color)
    });
    match misplaced {
        Some((layer, l)) => Err(RenderError::GroupKeyMisplaced { layer: Some(layer), mark: l.mark }),
        None => Ok(()),
    }
}

/// Build the per-layer prepared inputs, applying [`LayerPrepared::flip_coords`]
/// to every layer whose OWN effective coord flips (SPINE-12) — the whole
/// coordinate-space group (channels AND their provenance flags) swaps together.
///
/// Single-layer charts (`spec.layers == None`) produce exactly one
/// [`LayerPrepared`] from the chart-level mark + encoding; multi-layer charts
/// produce one per `Layer` (inheriting unset channels from chart level).
///
/// The flip decision is per layer, off [`LayerPrepared::coord`] (`#172`(1)).
/// It used to be one chart-level boolean the caller passed in, which is the
/// same answer for every layer of every spec that declares no per-layer coord —
/// so this is byte-stable — but the wrong question for a spec where one layer
/// declares `CoordFlip` and its sibling does not. The chart-level
/// `PreparedInputs::coord_flipped` survives unchanged for the chart-level
/// questions it answers (which axis the chrome describes, how a refusal names
/// the channel the user wrote at chart level).
///
/// A layered spec that also sets a chart-level `mark_style` is refused
/// ([`RenderError::LayeredChartMarkStyle`]): no layer reads it. So is an
/// empty `layers` list ([`RenderError::LayeredChartNoLayers`]): nothing would
/// draw, and the rendering encoding is layer 0's. So is a
/// layer with its own non-empty `transforms`
/// ([`RenderError::LayerTransformsUnsupported`]): no layer runs them. So is
/// a misplaced `group` key ([`refuse_misplaced_group_keys`]).
fn build_layers(spec: &ChartSpec) -> Result<Vec<LayerPrepared>, RenderError> {
    if spec.layers.as_ref().is_some_and(Vec::is_empty) {
        return Err(RenderError::LayeredChartNoLayers);
    }
    refuse_misplaced_group_keys(spec)?;
    let layer_count = spec.layers.as_ref().map_or(1, Vec::len);
    for (index, cond) in spec.conditionals.iter().enumerate() {
        if let Some(&layer) = cond.layers.iter().flatten().find(|&&l| l >= layer_count) {
            return Err(RenderError::ConditionalLayerOutOfRange { index, layer, layers: layer_count });
        }
    }
    let raw: Vec<LayerPrepared> = match &spec.layers {
        None => vec![LayerPrepared::from_chart_only(spec)],
        Some(layer_vec) if spec.mark_style.is_some() => {
            return Err(RenderError::LayeredChartMarkStyle { layers: layer_vec.len() });
        }
        Some(layer_vec) => {
            if let Some((layer, l)) = layer_vec
                .iter()
                .enumerate()
                .find(|(_, l)| !l.transforms.is_empty())
            {
                return Err(RenderError::LayerTransformsUnsupported {
                    layer,
                    count: l.transforms.len(),
                });
            }
            layer_vec
                .iter()
                .map(|l| LayerPrepared::from_chart_and_layer(spec, l))
                .collect()
        }
    };
    Ok(raw.into_iter().map(LayerPrepared::in_its_coord).collect())
}

/// The encoding a chart's scales, axes and legends are built from, over its
/// prepared `layers` (each already completed from the chart-level encoding).
/// Layer 0 owns the axes, so its encoding is the base. Each auxiliary-scale
/// channel is established by the first layer that draws it
/// ([`establishing_binding`]), so its binding (field, type, title, scale,
/// legend) is the one the scale's kind and the legend's title come from; a
/// channel no layer draws is cleared, so it builds no scale and no legend.
/// The scale's domain still spans every drawing layer
/// (`scale_resolve`'s per-layer reads).
///
/// The positional axes follow the same rule: an `x`/`y` layer 0 does not
/// draw (a one-axis reference rule, which takes no inherited opposite
/// channel) is established by the first later layer that draws it, else by
/// the chart-level binding layer 0 reads its data under, so the axis keeps
/// its field when no layer draws the channel. A composite `Overlay` member
/// whose layers draw nothing on an axis takes the binding its group's
/// establishing member imposes there (`leaf_scales`'
/// [`LeafScaleContext::imposed`](crate::render::scale_resolve::LeafScaleContext::imposed)).
fn establishing_encoding(
    spec: &ChartSpec,
    layers: &[LayerPrepared],
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
) -> Result<crate::spec::encoding::Encoding, RenderError> {
    let (lead, rest) = layers.split_first().ok_or(RenderError::LayeredChartNoLayers)?;
    let mut encoding = lead.encoding.clone();
    for channel in crate::spec::encoding::AUXILIARY_SCALE_CHANNELS {
        *channel.binding_mut(&mut encoding) =
            establishing_binding(spec, layers, channel, primary_batch, transform_outputs);
    }
    if spec.layers.is_some() {
        let mut chart = spec.encoding.clone();
        if crate::spec::coord::flips_positional_channels(lead.coord.as_ref()) {
            chart.flip_positional();
        }
        for channel in [crate::spec::layer::PositionalChannel::X, crate::spec::layer::PositionalChannel::Y] {
            if channel.slot(&encoding).is_some() {
                continue;
            }
            let shares_axis = |layer: &&LayerPrepared| !(channel == crate::spec::layer::PositionalChannel::Y && layer.independent_y);
            let drawn = rest.iter().filter(shares_axis).find_map(|layer| channel.slot(&layer.encoding).clone());
            let read_under = || channel.slot(&chart).clone().filter(|_| lead.data_source.is_none());
            *channel.slot_mut(&mut encoding) = drawn.or_else(read_under);
        }
    }
    if let Some(context) = leaf_scales {
        use crate::render::scale_resolve::Channel;
        for (slot, channel) in [(&mut encoding.x, Channel::X), (&mut encoding.y, Channel::Y)] {
            if slot.is_none() {
                *slot = context.imposed(channel).map(|(binding, _)| binding.clone());
            }
        }
    }
    Ok(encoding)
}

/// The binding an auxiliary-scale `channel`'s scale and legend are
/// established from: on a layered spec, that of the first layer drawing the
/// channel, the first of `scale_resolve`'s per-layer reads
/// ([`crate::render::scale_resolve::aux_reads`]), so the binding and the
/// domain read the same layers; on an unlayered spec, the chart's own when
/// its mark draws the channel. `None` when no layer draws it.
fn establishing_binding(
    spec: &ChartSpec,
    layers: &[LayerPrepared],
    channel: crate::spec::encoding::AuxiliaryScaleChannel,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> Option<crate::spec::encoding::EncodingSpec> {
    let drawer = match spec.layers.as_deref() {
        Some(raw) => {
            let reads = crate::render::scale_resolve::aux_reads(raw, &spec.encoding, channel, primary_batch, transform_outputs);
            layers.get(reads.first()?.layer())?
        }
        None => layers.first().filter(|layer| layer.draw_facts().draws(channel.drawn_as()).is_ok())?,
    };
    channel.binding(&drawer.encoding).clone()
}

/// One [`RenderWarning::ChannelNotDrawn`] per auxiliary-scale channel some
/// layer binds but no binding layer's mark draws
/// ([`crate::render::drawn::DrawFacts::draws`]): the channel builds no scale
/// and no legend ([`establishing_binding`]). It names the first layer
/// binding the channel, and says whether a statistic the chart planned
/// grouped by the binding's field ([`SeriesGrouping::by_channel`]): then the
/// groups are drawn, alike, rather than the binding having no effect. A channel a drawing mark's
/// statistic drops is not this case: the column gate warns about that one.
fn undrawn_channel_warnings(spec: &ChartSpec, layers: &[LayerPrepared]) -> Vec<RenderWarning> {
    use crate::render::drawn::Undrawn;
    let mut warnings = Vec::new();
    for channel in crate::spec::encoding::AUXILIARY_SCALE_CHANNELS {
        // Each candidate layer, with the rule's own reason already asked:
        // a layer whose completed encoding carries the binding asks it
        // directly; one that does not (a text declining an inherited
        // color, which `Layer::completed_encoding` already dropped) asks
        // it too, and is a candidate only when the reason is
        // `DeclinedInherited`, naming the field from its `group` key or
        // the chart's own color, as the completion dropped it from.
        let mut candidates = layers.iter().enumerate().filter_map(|(i, layer)| {
            if let Some(enc) = channel.binding(&layer.encoding).as_ref() {
                return Some((i, layer, enc, layer.draw_facts().draws(channel.drawn_as()).err()));
            }
            spec.layers.as_ref()?;
            let Err(Undrawn::DeclinedInherited) = layer.draw_facts().draws(channel.drawn_as()) else { return None };
            let enc = layer.encoding.group.as_ref().or(spec.encoding.color.as_ref())?;
            Some((i, layer, enc, Some(Undrawn::DeclinedInherited)))
        });
        let Some((first, layer, enc, reason)) = candidates.next() else { continue };
        let Some(reason) = reason else { continue };
        if candidates.any(|(_, _, _, r)| r.is_none()) {
            continue;
        }
        let grouped = matches!(reason, Undrawn::NotDrawn) && layer.grouping_series.by_channel.contains(&enc.field);
        warnings.push(RenderWarning::ChannelNotDrawn {
            layer: public_layer(spec, first, layer.mark),
            entry: layer_entry(spec, first, layer.mark),
            mark: layer.mark,
            channel: channel.name.to_string(),
            field: enc.field.clone(),
            reason,
            grouped,
        });
    }
    warnings
}

/// Refuse a binding that writes a binned color's class (`bin_channel`)
/// without the binned color scale it describes, anywhere a hand-built spec
/// could carry one past `EncodingSpec::new`'s refusal of the same shape:
/// `spec.encoding`'s channels, then every `spec.layers[i]`'s own. Walks the
/// raw spec, before inheritance, as [`refuse_malformed_sort_specs`] does: a
/// binding that writes the class writes the scale it describes. Shares
/// [`crate::spec::encoding::EncodingSpec::bin_channel_conflict`] with the
/// constructor, so the two refusals cannot disagree.
fn refuse_bin_channel_without_binned_color(spec: &ChartSpec) -> Result<(), RenderError> {
    let check = |location: String, enc: &crate::spec::encoding::EncodingSpec| {
        enc.bin_channel_conflict().map_err(|reason| RenderError::BinChannelWithoutBinnedColor { location, reason })
    };
    for (name, enc) in spec.encoding.channels() {
        check(name, enc)?;
    }
    for (i, layer) in spec.layers.iter().flatten().enumerate() {
        for (name, enc) in layer.encoding.channels() {
            check(format!("layers[{i}].{name}"), enc)?;
        }
    }
    Ok(())
}

/// Refuse a layer whose `stack` shorthand cannot be drawn, in that layer's
/// voice, before any scale previews the stack it would draw: a shorthand on
/// both x and y ([`RenderError::StackOnBothChannels`]), a shorthand beside
/// an explicit `position=` ([`RenderError::StackShorthandWithPosition`]),
/// since a mark draws one position adjustment, or `stack=False` beside
/// `position=Stack` ([`RenderError::StackDeclinedBesideStack`]), which say
/// opposite things; `stack=False` beside any other position agrees with it.
/// The position judged is the layer's resolved one ([`layer_position`]),
/// so an inherited position that yields to the layer's shorthand is never
/// judged. Asks the one reader the draw
/// pass and the axis preview stack through
/// ([`crate::render::position::stack_shorthand`]) over the encoding they hand
/// it: the layer's completed, post-flip encoding, less a polar theta its mark
/// sweeps itself ([`crate::render::position::position_encoding`]). The
/// conflict names the channels as the caller wrote them, before the layer's
/// own flip.
fn refuse_unstackable_shorthands(spec: &ChartSpec, layers: &[LayerPrepared]) -> Result<(), RenderError> {
    use crate::render::position::{position_encoding, stack_shorthand};
    use crate::spec::position::StackValueAxis;
    for (li, layer) in layers.iter().enumerate() {
        let voiced = |e: RenderError| e.in_layer_context(public_layer(spec, li, layer.mark), &layer_entry(spec, li, layer.mark));
        let encoding = position_encoding(layer.mark, layer.coord.as_ref(), &layer.encoding);
        let shorthand = stack_shorthand(&encoding);
        let Some(position) = layer.position.as_ref() else {
            shorthand.map_err(voiced)?;
            continue;
        };
        let flipped = crate::spec::coord::flips_positional_channels(layer.coord.as_ref());
        let authored = |channel: StackValueAxis| if (channel == StackValueAxis::X) != flipped { "x" } else { "y" };
        let channels = match shorthand {
            Err(_) => "x and y",
            Ok(Some((channel, _))) => authored(channel),
            Ok(None) => {
                let declined = |e: &Option<crate::spec::encoding::EncodingSpec>| e.as_ref().is_some_and(|e| e.declines_stack());
                let channels = match (declined(&encoding.x), declined(&encoding.y)) {
                    (true, true) => "x and y",
                    (true, false) => authored(StackValueAxis::X),
                    (false, true) => authored(StackValueAxis::Y),
                    (false, false) => continue,
                };
                if matches!(position, crate::spec::position::PositionAdjust::Stack { .. }) {
                    return Err(voiced(RenderError::StackDeclinedBesideStack { channels }));
                }
                continue;
            }
        };
        return Err(voiced(RenderError::StackShorthandWithPosition { adjustment: position.class_name(), channels }));
    }
    Ok(())
}

/// Refuse a malformed sort-field object anywhere it could reach render
/// without passing through Python's construction-time gate
/// (`EncodingSpec.__new__`, `parse_sort_field_object`): a chart-level
/// encoding channel, a layer's own encoding channel, or a facet dimension's
/// `sort`/`row_sort`. Reachable only from a hand-built `ChartSpec` --
/// `ChartSpec.from_json`, `ChartSpec(facet=dict)`, `layers[i].encoding`
/// (`coerce_layers`, raw serde `Deserialize`), and `FacetSpec.sort`/
/// `.row_sort` -- none of which call `EncodingSpec::new`. Reuses
/// `parse_sort_field_object`, the SAME parser `EncodingSpec::new`
/// calls, so a shape refused at construction and a shape refused here can
/// never disagree ([`RenderError::LayerTransformsUnsupported`] precedent: an
/// accepted-but-inert wire shape refuses rather than silently degrading).
///
/// Walks `spec.encoding`'s channels via [`crate::spec::encoding::Encoding::channels`]
/// (compiler-enforced coverage over every `Encoding` field, not a hand-typed
/// list -- see that function's own doc), then every `spec.layers[i]`'s own
/// channels, then `spec.facet`'s two sort slots. A non-object sort value
/// (string, list, or anything else `parse_sort_field_object` does not
/// inspect) is out of this check's scope, matching `EncodingSpec::new`'s own
/// narrowing to the object form.
fn refuse_malformed_sort_specs(spec: &ChartSpec) -> Result<(), RenderError> {
    fn check(location: String, sort: &serde_json::Value) -> Result<(), RenderError> {
        let serde_json::Value::Object(obj) = sort else {
            return Ok(());
        };
        crate::spec::encoding::sort::parse_sort_field_object(obj).map(|_| ()).map_err(|reason| {
            RenderError::SortSpecMalformed { location, reason, value: Box::new(sort.clone()) }
        })
    }

    for (name, enc) in spec.encoding.channels() {
        if let Some(sort) = &enc.sort {
            check(name, sort)?;
        }
    }
    if let Some(layer_vec) = &spec.layers {
        for (i, layer) in layer_vec.iter().enumerate() {
            for (name, enc) in layer.encoding.channels() {
                if let Some(sort) = &enc.sort {
                    check(format!("layers[{i}].{name}"), sort)?;
                }
            }
        }
    }
    if let Some(fspec) = &spec.facet {
        if let Some(sort) = &fspec.sort {
            check("facet.sort".to_string(), sort)?;
        }
        if let Some(sort) = &fspec.row_sort {
            if fspec.row.is_none() {
                // A structurally distinct malformation from an invalid
                // sort-field OBJECT (the value itself may be perfectly
                // well-formed, e.g. "descending"): `row_sort` names a
                // dimension this facet does not have. Wrap mode's `row_sort`
                // is otherwise a silent no-op (mirroring `row_title`'s
                // documented scoping), which the sort vocabulary should not
                // be -- an ordering the caller declared and that never
                // applies is malformed input, not inert-by-design chrome.
                return Err(RenderError::SortSpecMalformed {
                    location: "facet.row_sort".to_string(),
                    reason: "the facet has no row dimension (row=) for row_sort to order"
                        .to_string(),
                    value: Box::new(sort.clone()),
                });
            }
            check("facet.row_sort".to_string(), sort)?;
        }
    }
    Ok(())
}

/// Build both positional [`AxisInput`]s and their tick counts (SPINE-08/12).
///
/// `rendering_encoding` is the layer-0 (post-CoordFlip) encoding; `spec`'s
/// encoding supplies the user-facing spec-level title that wins the 3-way title
/// resolution. The per-channel derivation runs once each through
/// [`build_axis_input`], with [`Channel`] carrying the orient default and the
/// non-ordinal-y reversal. Returns `(axes, x_tick_count, y_tick_count)`.
/// One independent-y layer's secondary axis, in three index-correlated vectors:
/// the axis input, the y scale it was resolved from, and the tick count that
/// resolution used.
///
/// A struct rather than a bare tuple because the correspondence is the whole
/// point — `render::config_apply::resync_ticks_after_axis_merge` zips all three, and
/// a mismatch would silently pair an axis with another layer's scale.
pub(crate) struct SecondaryYAxes {
    pub(crate) inputs: Vec<AxisInput>,
    pub(crate) scales: Vec<crate::render::scale_resolve::ScaleKind>,
    pub(crate) tick_counts: Vec<usize>,
}

/// The public entry layer `li` of `spec` names in its refusals: the layer's
/// own, else the chart's, else the mark's.
pub(in crate::render) fn layer_entry(spec: &ChartSpec, li: usize, mark: crate::spec::mark::Mark) -> String {
    spec.layers
        .as_ref()
        .and_then(|layers| layers.get(li))
        .and_then(|l| l.entry.clone())
        .or_else(|| spec.entry.clone())
        .unwrap_or_else(|| format!("mark_{}", mark.as_str()))
}

/// The index a layer-voiced message names layer `li` by: `Some(li)` on a
/// layered spec when [`voiced_layer`] voices it, else `None`.
pub(in crate::render) fn public_layer(spec: &ChartSpec, li: usize, mark: crate::spec::mark::Mark) -> Option<usize> {
    spec.layers.as_ref().and_then(|_| voiced_layer(li, &layer_entry(spec, li, mark), mark))
}

/// Whether a desugaring call, rather than the caller directly, built the
/// layer carrying `entry`: `entry` is not `mark`'s own primitive default
/// (`mark_{mark}`). [`voiced_layer`] reads this to decide whether a layer's
/// index still locates it for the caller; [`crate::render::marks::polygon::naming_columns`]
/// reads it to decide whether a dropped ring is named by the internal
/// grouping id a call bound to `detail`, or by the categories and
/// color a reader actually sees.
pub(in crate::render) fn built_by_call(entry: &str, mark: crate::spec::mark::Mark) -> bool {
    entry != format!("mark_{}", mark.as_str())
}

/// The index a layer-voiced message names a layer by, given its `index` in
/// the numbering the message counts and its `entry`: `Some(index)` when the
/// entry is its own mark's primitive (a layer the caller placed, whose index
/// locates it), else `None`. A layer a desugaring call built carries that
/// call as its entry ([`layer_entry`]), and its index counts the call's
/// internal sub-layers, which the caller never wrote; the entry alone already
/// names the call they did write. The one rule every layer-voiced producer
/// applies: [`public_layer`] on a layered spec, and the composite overlay on
/// its group's member-order numbering.
pub(in crate::render) fn voiced_layer(index: usize, entry: &str, mark: crate::spec::mark::Mark) -> Option<usize> {
    (!built_by_call(entry, mark)).then_some(index)
}

/// Refuse a layer of a layered spec that names a column absent from that
/// layer's own resolved batch, on any channel it draws, and
/// report the inherited reads a statistic sub-layer drops.
///
/// **Where it runs, and why there.** Once per layer, in
/// `prepare_render_inputs`, before any scale is resolved: the auxiliary scale
/// builders read a later layer's column too, and the gate is the only owner
/// of the render-time message, so it answers before any of them can refuse
/// in other words. It judges `encoding`, the bindings the layer's mark draws
/// ([`LayerPrepared::drawn_encoding`]), so a binding on a channel the mark
/// never draws, or an inherited one it yields, is never refused, and because
/// layer completion gives a one-axis `mark_rule`/`mark_tick` no inherited
/// opposite channel, neither is a positional channel no builder reads.
///
/// **What it checks.** Every column the layer draws ([`layer_column_reads`]),
/// in a fixed order so the reported channel is deterministic when a layer
/// gets more than one wrong: the positional bindings, then every other
/// binding in declaration order, then the layer's `detail` when its mark
/// draws one, then each field branch of a conditional the layer declared on
/// a channel its mark draws. A channel is named as `encode(...)` names it.
///
/// **A statistic sub-layer's inherited read is dropped, not refused.** A
/// layer reading a statistic's output (a `data_source` no `Branch`
/// publishes, [`reads_statistic_output`]) that inherited a drawn binding
/// whose column the statistic dropped, though the chart's source holds it,
/// draws without that channel: the input is not malformed, the statistic
/// just does not carry it. The read is returned as a [`DroppedRead`] for the
/// caller to warn about and clear from the layer's drawn encoding. A
/// layer's own binding, a layer over the user's rows (no `data_source`, or
/// a `Branch` output) and a column found nowhere still refuse.
///
/// **Scope (normative).**
/// - Every layer of a layered spec is judged, layer 0 included: a channel's
///   scale, axis and legend are read only from the batch of a layer that
///   draws it, so a layer 0 binding to a column only another layer's rows
///   hold would otherwise draw nothing while an axis or legend described
///   it. A chart with no `layers` (`raw` is `None`) is not judged here: its
///   one mark keeps scale resolution's [`RenderError::UnknownColumn`].
/// - A raster image draws `x`/`y` through its extent columns
///   ([`crate::render::marks::image::drawn_columns`]), so those
///   are the columns judged on its axes. A tooltip entry is judged on its
///   own field, so an `x`/`y` entry on a raster image, whose one-row batch
///   holds the extent rather than the source column, is dropped with a
///   [`RenderWarning::StatisticDroppedChannel`] (the image reads a
///   statistic's output).
/// - The batch is the layer's OWN resolved batch — base data for
///   `data_source: None`, the named transform output otherwise — never a
///   union of batches. A field that exists only in some sibling transform
///   output is precisely the silent-drop shape: the marks cannot draw it.
/// - Zero-row layers never reach here (the caller skips them first), so an
///   empty batch still means "nothing to draw", not "broken encoding".
///
/// **Known boundary.** A geoshape is still refusable on an inherited `x`/`y`
/// it never reads — layer completion keeps every inherited `x`/`y` on it
/// ([`crate::spec::mark::Mark::inherited_positional`]). Left as-is
/// deliberately: every constructible shape was already degenerate.
///
/// `raw` is the layer as the wire wrote it (`None` for a chart with no
/// `layers`), which says which bindings are the layer's own; `conditionals`
/// are the chart's (the gate reads the ones this layer declared). A missing
/// column's remedy ([`RenderError::LayerUnknownColumn::remedy`]) is
/// classified by [`missing_column_remedy`] against `pipeline`. `voice` is
/// how the refusal names the layer: its index for a layer the caller
/// placed, else the public call that built it ([`public_layer`]).
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn validate_layer_columns(
    layer_index: usize,
    layer: &LayerPrepared,
    raw: Option<&crate::spec::layer::Layer>,
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    coord_flipped: bool,
    conditionals: &[crate::spec::chart::ChartConditional],
    pipeline: &PipelineBatches<'_>,
    voice: Option<String>,
) -> Result<Vec<DroppedRead>, RenderError> {
    if raw.is_none() {
        return Ok(Vec::new());
    }
    let statistic_layer = reads_statistic_output(layer, pipeline.lineages);
    let mut dropped = Vec::new();
    for read in layer_column_reads(layer_index, layer, raw, encoding, conditionals) {
        if batch.column_by_name(read.field).is_some() {
            continue;
        }
        if statistic_layer && read.inherited && pipeline.source.index_of(read.field).is_ok() {
            dropped.push(DroppedRead { channel: read.channel, wire: read.wire, field: read.field.to_string() });
            continue;
        }
        return Err(RenderError::LayerUnknownColumn {
            layer: layer_index,
            entry: voice.map(String::into_boxed_str),
            channel: read.channel,
            name: read.field.to_string(),
            coord_flipped,
            remedy: missing_column_remedy(read.field, layer, pipeline),
            independent_y: layer.independent_y,
        });
    }
    Ok(dropped)
}

/// An inherited read a statistic sub-layer draws without
/// ([`validate_layer_columns`]): the channel as `encode(...)` names it, the
/// encoding slot it was bound on (`None` for a conditional's field branch),
/// and the column its statistic's output does not carry.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::render) struct DroppedRead {
    pub(in crate::render) channel: &'static str,
    pub(in crate::render) wire: Option<&'static str>,
    pub(in crate::render) field: String,
}

/// One column a layer draws ([`layer_column_reads`]).
struct ColumnRead<'a> {
    /// The channel as `encode(...)` names it.
    channel: &'static str,
    /// The encoding slot the column is bound on, by wire name; `None` for
    /// `detail` and a conditional's field branch.
    wire: Option<&'static str>,
    field: &'a str,
    /// Whether the layer took the binding from a parent rather than writing
    /// it: a non-positional binding absent from the layer's own wire
    /// encoding or listed in its `inherited_channels`, or a conditional not
    /// declared by this layer alone. A positional binding and `detail` never
    /// are: a layer cannot draw without its position, and its `detail` is
    /// its own style.
    inherited: bool,
}

/// Every column a layer draws, in [`validate_layer_columns`]'s order: the
/// positional bindings of `encoding`, its other bindings, the layer's
/// `detail` when its mark draws one, then the field branches of the
/// conditionals layer `layer_index` declared on a channel its mark draws.
fn layer_column_reads<'a>(
    layer_index: usize,
    layer: &'a LayerPrepared,
    raw: Option<&'a crate::spec::layer::Layer>,
    encoding: &'a crate::spec::encoding::Encoding,
    conditionals: &'a [crate::spec::chart::ChartConditional],
) -> impl Iterator<Item = ColumnRead<'a>> {
    use crate::spec::layer::InheritedChannel;
    let is_positional = |wire: &str| matches!(wire, "x" | "y" | "x2" | "y2");
    let owns = move |wire: &str| {
        let Some(raw) = raw else { return true };
        let written = raw.encoding.bindings().into_iter().any(|(own, _)| own == wire);
        written && InheritedChannel::from_wire(wire).is_none_or(|c| !raw.inherited_channels.contains(&c))
    };
    let url = encoding.url.as_ref();
    let (positional, other): (Vec<_>, Vec<_>) = encoding
        .bindings()
        .into_iter()
        .flat_map(|(wire, enc)| {
            crate::render::marks::image::drawn_columns(layer.mark, url, wire, enc.field.as_str())
                .into_iter()
                .map(move |field| ColumnRead {
                    channel: python_channel(wire),
                    wire: Some(wire),
                    field,
                    inherited: !is_positional(wire) && !owns(wire),
                })
        })
        .partition(|read| read.wire.is_some_and(is_positional));
    let detail = layer
        .mark_style
        .as_ref()
        .and_then(|m| m.detail.as_deref())
        .filter(|_| layer.mark.style_field("detail").is_some())
        .map(|field| ColumnRead { channel: "detail", wire: None, field, inherited: false });
    let branches = conditionals
        .iter()
        .filter(move |c| c.declared_by(layer_index))
        .filter(move |c| {
            // A conditional's value pair is not itself a bound field: every
            // layer that declares a conditional carries its own copy of it
            // whatever it binds or inherits, so this asks only the mark and
            // its style, never a binding fact (a declined inherited color,
            // a yielded width, a raster's cell field) — a conditional
            // `fill_opacity` on an arrowless segment is not a read either,
            // as a bound one is not.
            InheritedChannel::from_wire(c.encoding.channel.as_str())
                .is_none_or(|ch| layer.draw_facts().reads_conditional(ch))
        })
        .flat_map(move |c| {
            let channel = c.encoding.channel.as_str();
            let inherited = c.layers.as_deref() != Some(&[layer_index][..]);
            [&c.encoding.if_selected, &c.encoding.if_not].into_iter().filter_map(move |branch| match branch {
                ferrum_scene::EncodingValue::Field { name } => {
                    Some(ColumnRead { channel, wire: None, field: name.as_str(), inherited })
                }
                _ => None,
            })
        });
    positional.into_iter().chain(other).chain(detail).chain(branches)
}

/// The channel `encode(...)` writes for an [`Encoding`](crate::spec::encoding::Encoding)
/// wire channel: the wire name, except a `tooltip_fields` entry, which
/// `encode(tooltip=[...])` writes.
fn python_channel(wire: &'static str) -> &'static str {
    match wire {
        "tooltip_fields" => "tooltip",
        other => other,
    }
}

/// Whether `layer` reads a statistic's output: the rows under its
/// effective data key passed a lowered step ([`OutputLineage::is_statistic`]).
/// A layer with no `data_source` reads its chart's own pipeline tail
/// ([`FINAL_OUTPUT_KEY`]), the same resolution every other reader of "which
/// lineage does this layer's rows come from" already applies (e.g. the
/// `statistic_series_fields` call just above). An unlayered
/// `mark_histogram()`'s one bar layer, and a figure's own tail layer built
/// through Python's `Chart._with_statistics`, are `data_source: None`
/// exactly because their rows ARE the chart's own tail, not because they
/// read the user's un-transformed rows. A layer naming a key nothing
/// published still reads the user's rows here (`false`): that mismatch is
/// refused elsewhere ([`validate_layer_columns`]/[`layer_final_rows`]), not
/// silently answered by this predicate.
pub(in crate::render) fn reads_statistic_output(layer: &LayerPrepared, lineages: &OutputLineages) -> bool {
    let key = layer.data_source.as_deref().unwrap_or(FINAL_OUTPUT_KEY);
    lineages.get(key).is_some_and(OutputLineage::is_statistic)
}

/// Every batch a chart's pipeline saw: its SOURCE schema, taken before any
/// transform ran (the unpartitioned schema `prepare_render_inputs` takes),
/// every batch the transforms produced, the tail and each named output
/// ([`PreparedInputs::transform_outputs`]), and each one's lineage. What
/// [`validate_layer_columns`] searches to say where a column a layer cannot
/// find does exist.
pub(in crate::render) struct PipelineBatches<'a> {
    pub(in crate::render) source: &'a Schema,
    pub(in crate::render) outputs: &'a HashMap<String, RecordBatch>,
    pub(in crate::render) lineages: &'a OutputLineages,
}

/// The schema of the rows a layer reading its chart's data starts from: for
/// a layer with a `data_source`, the rows the lineage of the batch it draws
/// started from (an operand's framed rows, never the combined frame); else
/// the pipeline's source schema.
fn chart_rows_schema(layer: &LayerPrepared, pipeline: &PipelineBatches<'_>) -> SchemaRef {
    match layer.data_source.as_deref().and_then(|key| pipeline.lineages.get(key)) {
        Some(lineage) => lineage.source.clone(),
        None => Arc::new(pipeline.source.clone()),
    }
}

/// The refusal for a layer's `data_source` naming a key nothing published:
/// built once so every site that could reach it (the per-layer batch
/// resolution loop above, [`layer_final_rows`], [`pre_statistic_rows`])
/// raises the same typed error with the same text.
fn unpublished_data_source(layer_index: usize, name: &str, transform_outputs: &HashMap<String, RecordBatch>) -> RenderError {
    let mut keys: Vec<&str> = transform_outputs.keys().map(|s| s.as_str()).collect();
    keys.sort_unstable();
    RenderError::TransformFailed(format!(
        "layer {layer_index} data_source '{name}' not found in transform outputs; \
         available keys: [{}]",
        keys.join(", ")
    ))
}

/// Where a layer's own rows live, whole: its named data source's output, or
/// the chart rows when it has none. [`pre_statistic_rows`]'s counterpart
/// that keeps a statistic's own OUTPUT columns (an aggregate's result, a
/// bin's edges) instead of stripping back to its input -- read by a
/// data-aware sort whose value field IS that output, since the field the
/// sort names then lives only here. A `data_source` naming no published
/// output is refused with [`unpublished_data_source`], the same refusal
/// `prepare_render_inputs` raises for the same key, rather than silently
/// substituting the chart's own rows.
///
/// Every output already spans every panel, so a faceted channel reads the
/// same key.
pub(in crate::render) fn layer_final_rows<'b>(
    layer_index: usize,
    data_source: Option<&str>,
    primary_batch: &'b RecordBatch,
    transform_outputs: &'b HashMap<String, RecordBatch>,
    include_final: bool,
) -> Result<&'b RecordBatch, RenderError> {
    let chart_rows = || match include_final {
        true => transform_outputs.get(FINAL_OUTPUT_KEY).unwrap_or(primary_batch),
        false => primary_batch,
    };
    match data_source {
        None => Ok(chart_rows()),
        Some(key) => match transform_outputs.get(key) {
            Some(batch) => Ok(batch),
            None => Err(unpublished_data_source(layer_index, key, transform_outputs)),
        },
    }
}

/// The rows a layer draws from before any statistic of its own: what a
/// data-aware sort on the layer's binding aggregates over when the sort
/// names a raw field its own statistic consumed, since a statistic's
/// output (a boxplot's `box`) carries the category but not that raw field.
/// - A layer with no `data_source` starts from the chart rows: `primary_batch`,
///   or the all-panels tail ([`FINAL_OUTPUT_KEY`]) when the channel is
///   shared across facet panels (`include_final`).
/// - A layer with a `data_source` starts from the rows its statistics
///   consumed when it reads a statistic's output, else from the rows it
///   reads ([`OutputLineage::pre_statistic_rows`]). A `data_source` naming
///   no published output refuses the same way [`layer_final_rows`] does.
pub(in crate::render) fn pre_statistic_rows<'b>(
    layer_index: usize,
    data_source: Option<&str>,
    lineages: &'b OutputLineages,
    primary_batch: &'b RecordBatch,
    transform_outputs: &'b HashMap<String, RecordBatch>,
    include_final: bool,
) -> Result<&'b RecordBatch, RenderError> {
    let Some(key) = data_source else {
        return layer_final_rows(layer_index, data_source, primary_batch, transform_outputs, include_final);
    };
    match (transform_outputs.get(key), lineages.get(key)) {
        (Some(batch), Some(lineage)) => Ok(lineage.pre_statistic_rows(batch)),
        (Some(batch), None) => Ok(batch),
        (None, _) => layer_final_rows(layer_index, data_source, primary_batch, transform_outputs, include_final),
    }
}

/// Where a column missing from a layer's own batch exists, for a layer that
/// reads its chart's data ([`LayerPrepared::reads_chart_data`]). Every
/// other layer (a `+` operand's branch reader) reads rows of its own, so
/// neither remedy is true of it and it gets `None`, the plain message.
///
/// Precedence: the source batch first, since a source column the chain
/// dropped is drawable from the source; then a statistic's output (any batch
/// whose rows passed a lowered step), where the column is a statistic
/// another layer's mark computes; otherwise `None`: the column exists
/// nowhere (a typo), or only in rows of the user's that are not this
/// layer's (another operand's own rows), which no remedy reaches. The source
/// is the rows the layer's chart data starts from ([`chart_rows_schema`]): a
/// composition's operand's own rows, never the merged frame, which holds a
/// sibling operand's columns too.
fn missing_column_remedy(
    name: &str,
    layer: &LayerPrepared,
    pipeline: &PipelineBatches<'_>,
) -> Option<crate::render::MissingColumnRemedy> {
    use crate::render::MissingColumnRemedy;
    if !layer.reads_chart_data {
        None
    } else if chart_rows_schema(layer, pipeline).index_of(name).is_ok() {
        Some(MissingColumnRemedy::Source)
    } else if statistic_outputs_hold(name, pipeline) {
        Some(MissingColumnRemedy::Statistic)
    } else {
        None
    }
}

/// Whether a statistic's output holds column `name`: a batch whose rows
/// passed a lowered step ([`OutputLineage::is_statistic`]).
fn statistic_outputs_hold(name: &str, pipeline: &PipelineBatches<'_>) -> bool {
    pipeline.outputs.iter().any(|(key, batch)| {
        pipeline.lineages.get(key).is_some_and(OutputLineage::is_statistic) && batch.column_by_name(name).is_some()
    })
}

/// What [`build_axes`] produces: the axis inputs, the two resolved primary tick
/// counts, and the secondary y axes' own resolved scales.
///
/// The scales are carried out because the secondary axes' post-config tick
/// adjustments (`adjust_axis_ticks` / `apply_label_format_to_axis` /
/// `sync_tick_placement_to_tick_values`) run in
/// `render::config_apply`'s pipeline, AFTER `axis_y2`'s config has been merged onto
/// the axis inputs — and each of those needs the scale the axis was derived
/// from. Without this the secondary axes had no scale to re-derive against,
/// which is exactly why `axis_y2`'s `label_format`/`tick_extra`/`tick_min_step`/
/// `values` had no consumer (spec §4.9, extended 2026-09-02).
pub(crate) struct PreparedAxes {
    pub(crate) axes: AxesInput,
    pub(crate) x_tick_count: usize,
    pub(crate) y_tick_count: usize,
    pub(crate) secondary_y_scales: Vec<crate::render::scale_resolve::ScaleKind>,
    pub(crate) secondary_y_tick_counts: Vec<usize>,
    /// This chart's own `Mark::Text` rows on `x`/`y`, measured
    /// before any composite union is applied — i.e. `leaf_scales`'s own
    /// union when one was given, else what this chart's layers measure on
    /// their own. Carried out so a composite tree's resolve pass
    /// (`render::composite::assign_union`) can union this leaf's rows with
    /// its siblings' without re-measuring them from a reconstructed spec.
    pub(crate) x_text_rows: crate::render::scale_resolve::TextEdgeRows,
    pub(crate) y_text_rows: crate::render::scale_resolve::TextEdgeRows,
}

#[allow(clippy::too_many_arguments)]
fn build_axes(
    spec: &ChartSpec,
    rendering_encoding: &crate::spec::encoding::Encoding,
    provisional_scales: &ResolvedScales,
    theme: &crate::layout::ThemeInputs,
    layers: &[LayerPrepared],
    layer_batches: &[&RecordBatch],
    y_slot_plan: &crate::render::scale_resolve::YSlotPlan,
    transformed: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    transform_lineages: &OutputLineages,
    chart_config: &crate::render::chart_config::ChartConfig,
    // Composite seam: `Some` only for a composite leaf whose
    // positional channel shares a domain across the tree. When that leaf's
    // channel also carries a group `Mark::Text` rows union
    // (`LeafScaleContext::x_text_rows`/`y_text_rows`, built by
    // `composite::assign_union` from every member's own rows), this leaf's
    // demand is built from THAT union instead of its own rows alone, so its
    // marks and whichever member's chrome the overlay keeps protect the
    // same ink. `None` for every standalone (flat/facet) render, and a
    // composite leaf whose channel shares no domain → byte-identical.
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<PreparedAxes, RenderError> {
    // Per-channel `Axis(tick_count=N)` controls the
    // target tick count for continuous + temporal axes (default 10). Limiting it
    // is what de-clutters dense temporal axes (e.g. a 72-row series no longer
    // renders 72 overlapping ticks). Read before tick generation so the count
    // feeds every downstream tick call (labels, fractions, minors).
    use crate::render::chart_config::ConfigAxis;
    let x_tick_count =
        resolve_axis_tick_count(rendering_encoding.x.as_ref(), chart_config, ConfigAxis::X);
    let y_tick_count =
        resolve_axis_tick_count(rendering_encoding.y.as_ref(), chart_config, ConfigAxis::Y);

    // R3: `build_axis_input`'s per-channel `Axis(orient=...)` validation has no
    // flip context of its own (it returns `UnflippedRenderError`); resolve it
    // in here, the boundary that DOES know it, so a rejected orient names the
    // axis the user actually wrote instead of the resolved (post-flip) one.
    let coord_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
    let cartesian_axes_suppressed =
        crate::spec::coord::suppresses_cartesian_axes(spec.coord.as_ref());

    // Parsed once per render: every `Mark::Text` row this pass
    // and `build_secondary_y_axis_inputs` measure shares this one font.
    let metrics = crate::render::font::FontdueMetrics::new();

    let SecondaryYAxes {
        inputs: secondary_y,
        scales: secondary_y_scales,
        tick_counts: secondary_y_tick_counts,
    } = build_secondary_y_axis_inputs(
        spec,
        layers,
        y_slot_plan,
        transformed,
        transform_outputs,
        transform_lineages,
        theme,
        chart_config,
        &metrics,
        warnings,
    )?;
    // The chart's own titles in screen space: a channel's title belongs to
    // the channel, so under a flip it moves with the channel to the axis
    // that channel draws on, as its scale and ticks do (`rendering_encoding`
    // is already post-flip; `spec.encoding` is as authored).
    let mut chart_encoding = spec.encoding.clone();
    if coord_flipped {
        chart_encoding.flip_positional();
    }
    // The primary axes' own text-edge demand: built once here,
    // off the provisional (domain-correct, pixel-placeholder) scale and the
    // chart's REAL batch — a row's ink extent does not depend on the pixel
    // range, so this is the SAME demand the panel-real mark resolve builds
    // later from `prep.axes.x`/`.y`, never a second guess at it. Empty
    // whenever the channel's padding is not the default
    // (`text_inset_applies`) or the chart has no `Mark::Text` layer on it.
    let x_applies = crate::render::scale_resolve::text_inset_applies(
        rendering_encoding.x.as_ref().and_then(|e| e.scale.as_ref()),
        crate::render::scale_resolve::coord_exempts_channel(spec.coord.as_ref(), "x"),
    );
    let y_applies = crate::render::scale_resolve::text_inset_applies(
        rendering_encoding.y.as_ref().and_then(|e| e.scale.as_ref()),
        crate::render::scale_resolve::coord_exempts_channel(spec.coord.as_ref(), "y"),
    );
    // `primary_axis_text_rows` reads every layer on x, but on y only the
    // layers `y_slot_plan` places on the primary (shared) slot, so an
    // `independent_y` layer's own text never floors the primary y it does
    // not share.
    let layer_pairs = || layers.iter().zip(layer_batches.iter().copied());
    let own_x_rows = crate::render::scale_resolve::primary_axis_text_rows(
        "x", layer_pairs(), y_slot_plan, spec.facet.as_ref(), theme, &metrics, provisional_scales.color.as_ref(),
    );
    let own_y_rows = crate::render::scale_resolve::primary_axis_text_rows(
        "y", layer_pairs(), y_slot_plan, spec.facet.as_ref(), theme, &metrics, provisional_scales.color.as_ref(),
    );
    // A composite leaf whose channel shares a group's rows union measures
    // from THAT instead of its own rows alone — never a second guess at
    // what the group's other members already measured.
    let x_rows = leaf_scales.and_then(|c| c.x_text_rows.clone()).unwrap_or_else(|| own_x_rows.clone());
    let y_rows = leaf_scales.and_then(|c| c.y_text_rows.clone()).unwrap_or_else(|| own_y_rows.clone());
    let x_text_demand = provisional_scales.x.text_edge_demand(x_applies.then_some(&x_rows));
    let y_text_demand = provisional_scales.y.text_edge_demand(y_applies.then_some(&y_rows));
    let axes = AxesInput {
        x: build_axis_input(
            Channel::X,
            rendering_encoding.x.as_ref(),
            chart_encoding.x.as_ref(),
            &provisional_scales.x,
            x_tick_count,
            theme,
            x_text_demand,
        )
        .map_err(|e| e.resolve(coord_flipped))?,
        y: build_axis_input(
            Channel::Y,
            rendering_encoding.y.as_ref(),
            chart_encoding.y.as_ref(),
            &provisional_scales.y,
            y_tick_count,
            theme,
            y_text_demand,
        )
        .map_err(|e| e.resolve(coord_flipped))?,
        // `AxesInput::show_x`/`show_y` mean "this Cartesian axis is DRAWN", so
        // this is where both reasons it may not be fold together: the user's
        // `.axis(x=False)` toggle, and a projected coord (`Polar`/`Geo`) that
        // replaces the Cartesian axes outright — `scene_build` emits no axis or
        // gridline node under either, so `layout` must reserve no band for
        // either (#182, closes #94). Resolving it here keeps every layout
        // consumer on one gate instead of each re-asking the coord.
        show_x: spec.axis_x.unwrap_or(true) && !cartesian_axes_suppressed,
        show_y: spec.axis_y.unwrap_or(true) && !cartesian_axes_suppressed,
        secondary_y,
        panels: Vec::new(),
    };
    Ok(PreparedAxes {
        axes,
        x_tick_count,
        y_tick_count,
        secondary_y_scales,
        secondary_y_tick_counts,
        x_text_rows: own_x_rows,
        y_text_rows: own_y_rows,
    })
}

/// Provisional secondary y-axis inputs, one per `independent_y` layer in layer
/// order (secondary-y-axis, GH #52). Mirrors the primary y axis's
/// provisional-then-final pattern directly above: each independent layer's own
/// y-scale is resolved against the full (pre-panel) batch with a `(0.0, 1.0)`
/// placeholder pixel range, using the SAME per-layer encoding merge (chart
/// encoding overlaid by the layer's own, `layers: None` so the y-domain isn't
/// re-unioned with sibling layers) that
/// [`render::scene_build::resolve_layer_y_scale`](crate::render::scene_build)
/// (Task 2) applies per-panel later. Domain-derived tick labels/fractions do
/// not depend on the placeholder pixel range, so the two resolutions agree —
/// one logical scale, computed twice only because layout must run before
/// panels exist. `compute_layout` (Task 3) reserves one right-side margin band
/// per returned axis and places it stacked outward from the primary.
///
/// Empty when the plan carries no independent-y layer — the byte-stable path,
/// so `AxesInput.secondary_y` stays empty and layout/scene output for the shared
/// path is unchanged. Iterates the plan's `secondary_layers` (layer order,
/// slot 1..n) so the axis-band order matches the per-panel [`YScaleSlots`] slot
/// order and the axis router's slot ids by construction, not by parallel
/// re-derivation (GH #72).
// 8 params after Task 8 threaded `chart_config` in (for `axis_y2`'s tick
// count). Mirrors the `#[allow]` its own caller `build_axes` already carries:
// these are the same six resolution inputs `build_axes` holds, forwarded, and
// bundling them into a struct used at exactly one call site would trade a lint
// for an indirection.
#[allow(clippy::too_many_arguments)]
fn build_secondary_y_axis_inputs(
    spec: &ChartSpec,
    layers: &[LayerPrepared],
    y_slot_plan: &crate::render::scale_resolve::YSlotPlan,
    transformed: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    transform_lineages: &OutputLineages,
    theme: &crate::layout::ThemeInputs,
    chart_config: &crate::render::chart_config::ChartConfig,
    metrics: &crate::render::font::FontdueMetrics,
    warnings: &mut Vec<RenderWarning>,
) -> Result<SecondaryYAxes, RenderError> {
    let mut inputs = Vec::new();
    let mut scales = Vec::new();
    let mut tick_counts = Vec::new();
    for (slot_index, &layer_idx) in y_slot_plan.secondary_layers().iter().enumerate() {
        let y_slot = slot_index + 1;
        let layer = &layers[layer_idx];
        let layer_batch: &RecordBatch = match &layer.data_source {
            Some(name) => transform_outputs
                .get(name)
                .expect("layer.data_source validated by prepare_render_inputs"),
            None => transformed,
        };
        // Shared param-aware per-layer y resolution (#72): the same
        // `scale_resolve::resolve_layer_y_slot_scale` scene_build's
        // `resolve_layer_y_scale` (Task 2) consumes for per-panel mark placement,
        // so axis ticks here and mark positions there share one substituted
        // domain. Only the pixel range differs (placeholder here vs. panel-real
        // there); domain-derived tick fractions do not depend on it. The merged
        // encoding below (layer's own overlaying the chart-level) drives axis
        // tick-count/title metadata only — the resolved scale itself comes from
        // the shared helper.
        let mut layer_encoding = spec.encoding.clone();
        layer_encoding.overlay_from(&layer.encoding);
        let layer_ctx = crate::render::scale_resolve::LayerScaleCtx {
            spec,
            transform_outputs,
            theme,
            leaf_scales: None,
            transform_lineages,
        };
        // This call resolves over a metadata-only placeholder `(0.0, 1.0)`
        // pixel range — it exists only to get this layer's y DOMAIN for
        // `compute_layout`'s tick/label sizing, before the real per-panel
        // plot area exists. `drop_placeholder_only_warnings` keeps every
        // warning this placeholder pass can honestly produce (color scheme,
        // palette overflow, sort-spec-ignored, ...) and drops only
        // `GroupCardinalityDegenerate`, whose bandwidth question is
        // meaningless over a 1px range — see that function's doc. The
        // panel-real resolution (`scene_build::resolve_layer_y_scale`)
        // re-runs this exact call over the actual plot area and keeps its
        // own (correctly pixel-scoped) cardinality warnings.
        let (y_scale, layer_scale_warnings) =
            crate::render::scale_resolve::resolve_layer_y_slot_scale(
                &layer_ctx,
                layer,
                layer_batch,
                (0.0, 1.0),
                (0.0, 1.0),
                &Default::default(),
                y_slot,
            )?;
        warnings.extend(drop_placeholder_only_warnings(layer_scale_warnings));
        // Chart-level scale-domain config at the `axis_y2` position: reshape THIS layer's own y
        // domain before `build_axis_input` below derives its ticks from the
        // scale, exactly as the primary pair is reshaped before `build_axes`.
        // Reported here (the one chart-level application);
        // `scene_build::resolve_layer_y_scale` re-applies it silently against
        // the panel-real scale so marks land on the same domain these ticks
        // describe.
        let mut y_scale = y_scale;
        apply_axis_domain_config_for(
            &mut y_scale,
            layer_effective_y(&layer.encoding, &spec.encoding),
            "y2",
            crate::render::chart_config::ConfigAxis::Y2,
            chart_config,
            warnings,
        )?;
        let y_scale = y_scale;
        let y_tick_count = resolve_axis_tick_count(
            layer_encoding.y.as_ref(),
            chart_config,
            crate::render::chart_config::ConfigAxis::Y2,
        );
        // R3: one of the two chains that RESOLVE `coord_flipped` (the other is
        // `build_axes`, just above; the third production chain to
        // `InvalidAxisOrient` — the chart-level `configure_axis` apply block in
        // `render::config_apply`'s `fill_axis_slots_specific_before_shared` — is
        // a deliberate EXEMPTION, not a
        // sibling; see the field doc on `RenderError::InvalidAxisOrient` for the
        // full three-chain account). A per-layer `fm.Axis(orient=...)` on an
        // independent-y layer validates against the RESOLVED `Channel::Y`
        // (secondary axes are always drawn right-side, never flipped
        // themselves), but `layer_encoding` is the post-`CoordFlip` layer
        // encoding (`build_layers`) — the `EncodingSpec` carrying the user's
        // override traveled through the swap — so the same boundary correction
        // `build_axes` applies is needed here too. `build_axis_input` returns
        // `UnflippedRenderError`, so this `.resolve(coord_flipped)` is not
        // optional — the compiler forces it before `?` can propagate a
        // `RenderError`.
        let coord_flipped = crate::spec::coord::flips_positional_channels(spec.coord.as_ref());
        // The secondary axis's own text-edge demand, measured
        // on this layer's REAL batch (unlike the placeholder-range scale
        // resolve above, a row's ink extent does not depend on the pixel
        // range at all): the slot's own layer, over its own data, matching
        // `resolve_layer_y_scale`'s panel-real resolve below.
        let y_applies = crate::render::scale_resolve::text_inset_applies(
            layer_encoding.y.as_ref().and_then(|e| e.scale.as_ref()),
            false,
        );
        let y_rows = crate::render::scale_resolve::text_edge_rows(
            "y", std::iter::once((layer, layer_batch)), spec.facet.as_ref(), theme, metrics, None,
        );
        let y_text_demand = y_scale.text_edge_demand(y_applies.then_some(&y_rows));
        let axis_input = build_axis_input(
            Channel::Y,
            layer_encoding.y.as_ref(),
            layer_encoding.y.as_ref(),
            &y_scale,
            y_tick_count,
            theme,
            y_text_demand,
        )
        .map_err(|e| e.resolve(coord_flipped))?;
        inputs.push(axis_input);
        scales.push(y_scale);
        tick_counts.push(y_tick_count);
    }
    Ok(SecondaryYAxes {
        inputs,
        scales,
        tick_counts,
    })
}

/// D10: fill missing (group × x-value) combinations in the batch with a constant y value.
///
/// When `encoding.y.impute = {"value": N}` is set and the encoding has both an x and
/// color channel, this synthesizes zero-rows for every (x-value, color-group) pair
/// that is absent from the data, ensuring that line charts and area charts connect
/// correctly even when some groups are missing observations at certain x ticks.
///
/// The imputed rows carry the x and color values from the (x, group) key and the
/// impute constant for y. All other columns default to null. No-ops when any of
/// these conditions hold: no x encoding, no color encoding, impute value absent,
/// or the batch is already complete.
fn apply_impute(batch: &RecordBatch, spec: &ChartSpec) -> RecordBatch {
    use arrow::array::{Float64Array, StringArray};

    // Only handle `encoding.y.impute = {"value": <number>}`.
    let impute_value = spec
        .encoding
        .y
        .as_ref()
        .and_then(|y| y.impute.as_ref())
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("value"))
        .and_then(|v| v.as_f64());
    let Some(fill) = impute_value else {
        return batch.clone();
    };

    let x_enc = match spec.encoding.x.as_ref() {
        Some(e) => e,
        None => return batch.clone(),
    };
    let color_enc = match spec.encoding.color.as_ref() {
        Some(e) => e,
        None => return batch.clone(),
    };
    let y_enc = match spec.encoding.y.as_ref() {
        Some(e) => e,
        None => return batch.clone(),
    };

    let x_field = &x_enc.field;
    let color_field = &color_enc.field;
    let y_field = &y_enc.field;

    // Collect distinct x values and groups. Only handles Float64 x + Utf8 color.
    let x_col = match batch.column_by_name(x_field) {
        Some(c) => c,
        None => return batch.clone(),
    };
    let color_col = match batch.column_by_name(color_field) {
        Some(c) => c,
        None => return batch.clone(),
    };
    let x_arr = match x_col.as_any().downcast_ref::<Float64Array>() {
        Some(a) => a,
        None => return batch.clone(),
    };
    let color_arr = match color_col.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return batch.clone(),
    };

    // Collect all (x, group) pairs and the full domain of each.
    use std::collections::HashSet;
    let mut x_vals: Vec<f64> = x_arr.iter().flatten().collect();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    x_vals.dedup();
    let mut groups: Vec<String> = color_arr.iter().flatten().map(str::to_owned).collect();
    groups.sort_unstable();
    groups.dedup();

    if x_vals.is_empty() || groups.is_empty() {
        return batch.clone();
    }

    // Build the set of existing (x, group) keys.
    let mut existing: HashSet<(u64, String)> = HashSet::new();
    for i in 0..batch.num_rows() {
        if x_arr.is_null(i) || color_arr.is_null(i) {
            continue;
        }
        let xv = x_arr.value(i);
        let gv = color_arr.value(i).to_owned();
        existing.insert((xv.to_bits(), gv));
    }

    // Build synthetic rows for missing (x, group) pairs.
    let mut new_x: Vec<Option<f64>> = Vec::new();
    let mut new_group: Vec<Option<String>> = Vec::new();
    let mut new_y: Vec<Option<f64>> = Vec::new();
    for xv in &x_vals {
        for gv in &groups {
            if existing.contains(&(xv.to_bits(), gv.clone())) {
                continue;
            }
            new_x.push(Some(*xv));
            new_group.push(Some(gv.clone()));
            new_y.push(Some(fill));
        }
    }
    if new_x.is_empty() {
        return batch.clone();
    }

    // Append synthetic rows: build a small batch with (x, color, y) and null all other cols.
    let n_new = new_x.len();
    let n_orig = batch.num_rows();
    let schema = batch.schema();
    let mut combined_cols: Vec<ArrayRef> = Vec::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let orig_col = batch.column(col_idx);
        match field.name().as_str() {
            name if name == x_field => {
                let combined: ArrayRef = Arc::new(Float64Array::from(
                    (0..n_orig)
                        .map(|i| {
                            if x_arr.is_null(i) {
                                None
                            } else {
                                Some(x_arr.value(i))
                            }
                        })
                        .chain(new_x.iter().copied())
                        .collect::<Vec<Option<f64>>>(),
                ));
                combined_cols.push(combined);
            }
            name if name == color_field => {
                let orig_str = orig_col.as_any().downcast_ref::<StringArray>();
                let combined: ArrayRef = if let Some(orig_str) = orig_str {
                    Arc::new(
                        (0..n_orig)
                            .map(|i| {
                                if orig_str.is_null(i) {
                                    None
                                } else {
                                    Some(orig_str.value(i))
                                }
                            })
                            .chain(new_group.iter().map(|v| v.as_deref()))
                            .collect::<Vec<Option<&str>>>()
                            .into_iter()
                            .collect::<StringArray>(),
                    )
                } else {
                    return batch.clone();
                };
                combined_cols.push(combined);
            }
            name if name == y_field => {
                let orig_f64 = orig_col.as_any().downcast_ref::<Float64Array>();
                let combined: ArrayRef = if let Some(orig_f64) = orig_f64 {
                    Arc::new(Float64Array::from(
                        (0..n_orig)
                            .map(|i| {
                                if orig_f64.is_null(i) {
                                    None
                                } else {
                                    Some(orig_f64.value(i))
                                }
                            })
                            .chain(new_y.iter().copied())
                            .collect::<Vec<Option<f64>>>(),
                    ))
                } else {
                    return batch.clone();
                };
                combined_cols.push(combined);
            }
            _ => {
                // Append nulls for synthetic rows.
                let extended = arrow::compute::concat(&[
                    orig_col.as_ref(),
                    arrow::array::new_null_array(orig_col.data_type(), n_new).as_ref(),
                ]);
                match extended {
                    Ok(arr) => combined_cols.push(arr),
                    Err(_) => return batch.clone(),
                }
            }
        }
    }
    match RecordBatch::try_new(schema, combined_cols) {
        Ok(b) => b,
        Err(_) => batch.clone(),
    }
}

// ── Tick-label format application ────────────────────────────────────────────
//
// All number/time formatting is delegated to `crate::render::format`, the single
// source of truth for the d3-format grammar and chrono time formatting. This
// module only re-parses pre-computed label strings back to numbers (or epoch-ms
// for temporal axes) and re-applies the requested format.

/// The default major-tick count when neither level asks for one.
pub(crate) const DEFAULT_TICK_COUNT: usize = 10;

/// The effective major-tick count for one axis, resolved at the documented
/// cascade in ONE place:
/// per-channel `fm.Axis(tick_count=)` > chart-level `configure_axis(
/// tick_count=)` (per-axis section before the shared one, via
/// `ChartConfig::axis_field`) > [`DEFAULT_TICK_COUNT`].
///
/// Resolved HERE, before the axes are built, rather than as a fill onto
/// `AxisStyleOverrides` after the fact: the count decides how many ticks the
/// scale generates, so it has to be known at generation time. That is also why
/// `prepare_render_inputs` takes the chart config at all — a chart-level count
/// applied later could only thin an already-built tick vector, not choose it.
fn resolve_axis_tick_count(
    enc: Option<&crate::spec::encoding::EncodingSpec>,
    chart_config: &crate::render::chart_config::ChartConfig,
    axis: crate::render::chart_config::ConfigAxis,
) -> usize {
    enc.and_then(|e| e.axis.as_ref())
        .and_then(|a| a.tick_count)
        .or_else(|| chart_config.axis_field(axis, |cfg| cfg.style.tick_count))
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_TICK_COUNT)
}

/// One axis's chart-level scale-domain config, lifted out of
/// `ChartConfig` into the neutral shape `scale_resolve` accepts.
fn axis_domain_config(
    chart_config: &crate::render::chart_config::ChartConfig,
    axis: crate::render::chart_config::ConfigAxis,
) -> crate::render::scale_resolve::AxisDomainConfig {
    crate::render::scale_resolve::AxisDomainConfig {
        min: chart_config.axis_field(axis, |c| c.domain_min),
        max: chart_config.axis_field(axis, |c| c.domain_max),
        nice: chart_config.axis_field(axis, |c| c.nice),
        zero: chart_config.axis_field(axis, |c| c.zero),
    }
}

/// Apply the chart-level scale-domain config to ONE axis position. The
/// primitive every position routes through, so the
/// section→encoding→gate correspondence is expressed once.
///
/// `enc` is the RESOLVED (post-`CoordFlip`) encoding for this axis: an
/// encoding-level `scale=` **domain** on it wins entirely and silently,
/// because it is the more specific surface and the documented cascade has
/// already answered.
///
/// The gate is `encoding_explicit_extent`, not `scale.is_some()`. A `scale=`
/// that declares no domain — `LinearScale(clamp=True)`, a `Quantile` whose
/// domain is a binning artifact, any ordinal family — pins nothing, so it has
/// no domain for the config to lose to and the config must still apply.
/// Testing mere presence made all four fields silently inert for those
/// encodings — the parsed-but-dropped class this gate exists to prevent.
pub(in crate::render) fn apply_axis_domain_config_for(
    scale: &mut crate::render::scale_resolve::ScaleKind,
    enc: Option<&crate::spec::encoding::EncodingSpec>,
    channel: &'static str,
    axis: crate::render::chart_config::ConfigAxis,
    chart_config: &crate::render::chart_config::ChartConfig,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    crate::render::scale_resolve::apply_axis_domain_config(
        scale,
        &axis_domain_config(chart_config, axis),
        channel,
        enc.is_some_and(|e| crate::render::scale_resolve::encoding_explicit_extent(e).is_some()),
        warnings,
    )
}

/// Apply the chart-level scale-domain config to BOTH primary positional scales
/// of one resolution pass.
///
/// Two passes need this and must agree: this module's provisional resolution
/// (before the axes derive their ticks from the domain) and
/// `scene_build::resolve_panel_scales` (which re-resolves each panel's scales
/// from scratch and would otherwise place marks against the un-shaped
/// domain). One body rather than the same per-channel pairing written twice —
/// the axis→section→encoding correspondence is exactly the kind of mapping
/// that drifts when hand-copied.
///
/// `warnings` is the caller's sink: `scene_build` passes a throwaway, because
/// its per-panel re-application would otherwise repeat this chart's one
/// ordinal-axis warning once per panel.
pub(in crate::render) fn apply_axis_domain_configs(
    x: &mut crate::render::scale_resolve::ScaleKind,
    y: &mut crate::render::scale_resolve::ScaleKind,
    encoding: &crate::spec::encoding::Encoding,
    chart_config: &crate::render::chart_config::ChartConfig,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    use crate::render::chart_config::ConfigAxis;
    for (channel, axis, scale, enc) in [
        ("x", ConfigAxis::X, x, encoding.x.as_ref()),
        ("y", ConfigAxis::Y, y, encoding.y.as_ref()),
    ] {
        apply_axis_domain_config_for(scale, enc, channel, axis, chart_config, warnings)?;
    }
    Ok(())
}

/// The y encoding an `independent_y` layer actually resolves against: its own
/// when the layer binds y, else the chart-level one.
///
/// This is `Encoding::overlay_from`'s rule for a single channel, which is what
/// `scale_resolve::resolve_layer_y_slot_scale` applies internally to build the
/// scale — so the `scale=` gate above is asked of the SAME encoding the scale
/// came from. Both stages that resolve a secondary y scale (this module's
/// axis-input builder and `scene_build::resolve_layer_y_scale`) call this
/// rather than each spelling the merge out.
pub(in crate::render) fn layer_effective_y<'a>(
    layer_encoding: &'a crate::spec::encoding::Encoding,
    chart_encoding: &'a crate::spec::encoding::Encoding,
) -> Option<&'a crate::spec::encoding::EncodingSpec> {
    layer_encoding.y.as_ref().or(chart_encoding.y.as_ref())
}

/// Parse an axis `orient` string into an [`AxisOrient`], validating it against
/// the channel dimension: x accepts top/bottom, y accepts left/right. A
/// cross-dimension value fails loud per the B5 contract. Shared by the
/// per-channel (here) and chart-level (`render::config_apply`) apply paths so the two
/// cannot drift on the accepted token set or the dimension check.
///
/// R3: the `InvalidAxisOrient` constructions below always build
/// `coord_flipped: false` — this function has no flip context of its own (it
/// only ever sees the RESOLVED `channel`) — but the result is wrapped in
/// [`UnflippedRenderError`], whose sole exit,
/// [`resolve`](UnflippedRenderError::resolve), forces the caller to decide,
/// at ITS OWN boundary, whether `false` is correct or needs patching to the
/// real flip flag:
/// - `.resolve(coord_flipped)` when the caller's `channel` reached this
///   function by way of a user-written `EncodingSpec` that traveled through
///   `build_layers`' swap (e.g. `build_axes`'s per-channel
///   `fm.Axis(orient=...)`, `build_secondary_y_axis_inputs`'s per-layer one)
///   — there the resolved token needs translating back to what the user
///   actually wrote;
/// - `.resolve(false)` explicitly when `channel` is resolved-slot/physical
///   vocabulary the user's own config key already names directly (the
///   chart-level `configure_axis` chain — see the field doc on
///   [`crate::render::RenderError::InvalidAxisOrient`] for the full
///   three-chain account and why that one is a deliberate exemption, not a
///   gap).
///
/// **This decision is compiler-enforced**: [`UnflippedRenderError`] has no
/// `From` conversion to [`RenderError`], so a new call chain reaching this
/// function cannot propagate the error via `?` without an explicit
/// `.resolve(...)` somewhere in the chain — it fails to compile instead of
/// silently defaulting. Without it, a call chain can add `build_axes`'
/// `.resolve(coord_flipped)` while leaving a sibling chain (such as
/// `build_secondary_y_axis_inputs`'s per-layer one) resolving `false`,
/// silently wrong under a flip; the compiler check keeps every chain honest
/// instead of relying on a prose rule to catch the omission.
pub(crate) fn parse_axis_orient(
    value: &str,
    channel: &'static str,
) -> Result<AxisOrient, crate::render::UnflippedRenderError> {
    let orient = match value.trim().to_ascii_lowercase().as_str() {
        "top" => AxisOrient::Top,
        "bottom" => AxisOrient::Bottom,
        "left" => AxisOrient::Left,
        "right" => AxisOrient::Right,
        // Placeholder — see the doc comment on this function.
        _ => {
            return Err(crate::render::UnflippedRenderError::new(
                RenderError::InvalidAxisOrient {
                    channel,
                    orient: value.to_owned(),
                    coord_flipped: false,
                },
            ))
        }
    };
    let ok = match channel {
        "x" => matches!(orient, AxisOrient::Top | AxisOrient::Bottom),
        _ => matches!(orient, AxisOrient::Left | AxisOrient::Right),
    };
    if ok {
        Ok(orient)
    } else {
        // Placeholder — see the doc comment on this function.
        Err(crate::render::UnflippedRenderError::new(
            RenderError::InvalidAxisOrient {
                channel,
                orient: value.to_owned(),
                coord_flipped: false,
            },
        ))
    }
}

/// Parse a `label_overlap` token (`fm.Axis(label_overlap=...)` /
/// `configure_axis(label_overlap=...)`) into a [`LabelOverlap`] strategy (B5 unit
/// 6b). Maps the Vega-style values onto the existing cascade primitives:
/// - `"true"` → [`LabelOverlap::ShowAll`] (show every label, may overlap).
/// - `"false"` / `"greedy"` → [`LabelOverlap::Greedy`] (cascade default cull).
/// - `"parity"` → [`LabelOverlap::Parity`] (stride-2 decimation).
/// - `"rotate"` → [`LabelOverlap::Rotate`] (force the rotate stage).
///
/// An unrecognized token is **bounded to the nearest existing behavior**
/// (`Greedy`, the cascade default) rather than failing loud, since
/// `label_overlap` only biases an already-graceful cascade — an unknown value
/// degrades to the default rather than erroring on a render. Returns `None` for
/// an empty/whitespace token so callers leave the cascade unmodified.
pub(crate) fn parse_label_overlap(value: &str) -> Option<crate::layout::LabelOverlap> {
    use crate::layout::LabelOverlap;
    match value.trim().to_ascii_lowercase().as_str() {
        "" => None,
        "true" => Some(LabelOverlap::ShowAll),
        "parity" => Some(LabelOverlap::Parity),
        "rotate" => Some(LabelOverlap::Rotate),
        // "false", "greedy", and any unrecognized token bound to the cascade
        // default (greedy) — the nearest existing behavior.
        _ => Some(LabelOverlap::Greedy),
    }
}

/// Parse a `title_orient` string into an [`AxisOrient`]. Unlike axis `orient`,
/// all four sides are valid (a title may run perpendicular to its axis), so this
/// only rejects an unrecognized token. Shared with the chart-level apply path.
pub(crate) fn parse_title_orient(
    value: &str,
    channel: &'static str,
) -> Result<AxisOrient, crate::render::UnflippedRenderError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "top" => Ok(AxisOrient::Top),
        "bottom" => Ok(AxisOrient::Bottom),
        "left" => Ok(AxisOrient::Left),
        "right" => Ok(AxisOrient::Right),
        // Placeholder — see the doc comment on `parse_axis_orient` above
        // (this function shares the exact same contract).
        _ => Err(crate::render::UnflippedRenderError::new(
            RenderError::InvalidAxisOrient {
                channel,
                orient: value.to_owned(),
                coord_flipped: false,
            },
        )),
    }
}

// ── The canonical AxisStyleSpec → AxisStyleOverrides merge ────────────────
//
// #143 remediation (design review rec 6): this merge lived in `config_apply`,
// which made `prepare` — the EARLIER stage — depend on a later one for a
// primitive that is neutral between them. It is not chart-config application:
// it is the one field mapping both the per-channel (this module) and the
// chart-level (`config_apply`) disciplines share, selected by
// `fill_only_if_none`. Filed here, beside the three `parse_*` validators it
// calls and beside its own per-channel caller below, the dependency points
// one way again: `config_apply` → `prepare`, never back.

/// One canonical [`AxisStyleSpec`](crate::render::chart_config::AxisStyleSpec) →
/// [`AxisStyleOverrides`](crate::layout::AxisStyleOverrides) merge, replacing the
/// two parallel ~28-field mappers that previously turned the SAME source struct
/// into the bundle in two shapes (the per-channel fresh-builder in
/// `prepare::encoding_axis_style_overrides` and the chart-level fill-only-if-`None`
/// `apply_axis_style_to_axis_input`). Both call sites now route through here so a
/// new `AxisStyleSpec` field is wired once, not in two drifting bodies.
///
/// `fill_only_if_none` selects the merge discipline:
/// - `false` (per-channel path): write every field unconditionally — the caller
///   starts from `Default`, so this is the fresh-build the encoding path needs.
/// - `true` (chart-level path): write each field only when the slot is still
///   `None`, so a higher-precedence source (a per-channel spec, or an earlier
///   config layer) always wins.
///
/// Field-ownership exceptions preserved bit-for-bit from the old two-mapper world:
/// - The **`labels.format`** slot is written ONLY on the chart-level
///   (`fill_only_if_none`) path. The per-channel/prepare path deliberately leaves
///   it empty here and seeds it separately from the temporal/numeric format
///   threading (`apply_axis_format_or_thread`) after this merge runs.
/// - The **`show_*`** toggles are `ticks.show`, `labels.show`, `domain.show` and
///   `grid.show` on this bundle, and this merge writes all four (see the body's
///   own note on why they moved here from `AxisInput`). *(Corrected 2026-09-20,
///   pre-existing: this bullet used to say they "live on `AxisInput`, not on this
///   bundle" and that the merge "cannot touch them". Both halves were false at
///   `HEAD` verbatim and contradicted by this same function ~80 lines below.
///   Not a task-11 regression; corrected because it sits inside a block the
///   rename sweep had to cover anyway.)*
///
/// An invalid `orient` (cross-dimension) or `title_orient` token fails loud via
/// [`RenderError::InvalidAxisOrient`]; an unparseable color hex string leaves the
/// slot `None` (theme fallback) on both paths.
///
/// Returns [`UnflippedRenderError`] (R3), not `RenderError` directly: neither
/// this fn nor `parse_axis_orient`/`parse_title_orient` beneath it has access
/// to `coord_flipped` (this fn is shared by both the flip-patched per-channel
/// path and the flip-exempt chart-level path — see the field doc on
/// [`RenderError::InvalidAxisOrient`]), so the decision is deferred to each
/// caller's own boundary via `.resolve(coord_flipped)`.
pub(crate) fn axis_style_fill_from(
    o: &mut crate::layout::AxisStyleOverrides,
    style: &crate::render::chart_config::AxisStyleSpec,
    channel: &'static str,
    fill_only_if_none: bool,
) -> Result<(), crate::render::UnflippedRenderError> {
    // One merge predicate for every field: on the fresh-build (per-channel) path
    // (`fill_only_if_none == false`) the value is written unconditionally; on the
    // chart-level path it is written only when the slot is still `None` so a
    // higher-precedence source wins. Collapsing it here means each field below
    // appears exactly once regardless of which discipline applies. A generic `fn`
    // (not a closure) because the slot type varies across fields.
    fn set<T>(slot: &mut Option<T>, value: Option<T>, fill_only_if_none: bool) {
        if !fill_only_if_none || slot.is_none() {
            *slot = value;
        }
    }
    // Full CSS vocabulary (names, `rgb()`, hex) — `None` for absent or
    // unparseable values, matching the pre-existing "keep the theme default"
    // behavior of every axis-style color override.
    let opt_color = |c: &Option<String>| {
        c.as_deref()
            .and_then(|s| crate::render::color::parse_color(s).ok())
    };
    // ── Positioning / draw-order orphans (B5 unit 2) ─────────────────────────
    // `placement.orient` is the override INPUT (validated against the dimension);
    // the concrete `AxisInput.orient` is re-synced from it by `resolve_orient` after
    // all override layers merge. Validation can fail, so it is resolved eagerly
    // (before `set`) and only assigned under the same fill predicate.
    if !fill_only_if_none || o.placement.orient.is_none() {
        o.placement.orient = style
            .orient
            .as_deref()
            .map(|s| parse_axis_orient(s, channel))
            .transpose()?;
    }
    if !fill_only_if_none || o.title.orient.is_none() {
        o.title.orient = style
            .title_orient
            .as_deref()
            .map(|s| parse_title_orient(s, channel))
            .transpose()?;
    }
    set(
        &mut o.placement.translate,
        style.translate,
        fill_only_if_none,
    );
    set(&mut o.placement.min_band, style.min_band, fill_only_if_none);
    set(&mut o.placement.max_band, style.max_band, fill_only_if_none);
    set(&mut o.grid.opacity, style.grid_opacity, fill_only_if_none);
    set(&mut o.placement.zindex, style.zindex, fill_only_if_none);
    set(&mut o.ticks.size, style.tick_size, fill_only_if_none);
    set(&mut o.ticks.extra, style.tick_extra, fill_only_if_none);
    set(
        &mut o.ticks.min_step,
        style.tick_min_step,
        fill_only_if_none,
    );
    // ── Show toggles (D12, spec §4.9) ────────────────────────────────────────
    // These four used to live as bare `bool`s on `AxisInput`, written ONLY by
    // the per-channel prepare path, because a `bool` cannot say "unset" and a
    // chart-level write would therefore have clobbered the per-channel answer.
    // As `Option<bool>` on this bundle they obey the same one merge predicate
    // as every sibling field, which is what makes chart-level `labels`/`ticks`/
    // `domain`/`grid` honored (they had no consumer at all before) WITHOUT
    // inverting the cascade. `grid`/`domain` additionally carry a theme
    // fallback, folded in by `AxesInput::apply_show_defaults` after every
    // config layer has had its turn.
    set(&mut o.labels.show, style.labels, fill_only_if_none);
    set(&mut o.ticks.show, style.ticks, fill_only_if_none);
    set(&mut o.domain.show, style.domain, fill_only_if_none);
    set(&mut o.grid.show, style.grid, fill_only_if_none);
    // ── Residual positioning/overlap orphans (B5 unit 6b) ────────────────────
    set(&mut o.placement.offset, style.offset, fill_only_if_none);
    set(&mut o.labels.flush, style.label_flush, fill_only_if_none);
    set(
        &mut o.labels.overlap,
        style.label_overlap.as_deref().and_then(parse_label_overlap),
        fill_only_if_none,
    );
    set(&mut o.labels.angle, style.label_angle, fill_only_if_none);
    // The `labels.format` slot is owned by the chart-level path only. The
    // per-channel path threads it separately after this merge, so it must stay
    // empty here.
    //
    // This fallback write is normally shadowed by `apply_axis_config_to_axis_input`'s
    // own (more complete — raw-first via `effective_label_format()`) fill
    // immediately before this fn runs, which makes THIS block a no-op in
    // the common case (it only re-derives the SAME `style.label_format`
    // that caller's fill already found — see that caller's doc). But when
    // the caller's fill is correctly SKIPPED for a per-channel-claimed axis
    // (the format slot's claim, D8 cascade-inversion fix),
    // `o.labels.format`'s spec is still `None` reaching here. Routes through
    // the SAME mechanized `fill_chart_level_label_format` the caller uses
    // rather than hand-re-deriving the
    // "unclaimed" predicate a second time — this is exactly the two-writer
    // duplication that fix closed.
    if fill_only_if_none {
        o.fill_chart_level_label_format(
            style.label_format.clone(),
            style.label_format_type.clone(),
        );
    }
    set(&mut o.ticks.values, style.values.clone(), fill_only_if_none);
    // Title overrides.
    set(
        &mut o.title.font_size,
        style.title_font_size,
        fill_only_if_none,
    );
    set(
        &mut o.title.color,
        opt_color(&style.title_color),
        fill_only_if_none,
    );
    set(&mut o.title.padding, style.title_padding, fill_only_if_none);
    set(
        &mut o.labels.padding,
        style.label_padding,
        fill_only_if_none,
    );
    // ── Per-axis styling overrides (B5): consulted by build_axis/build_grid ──
    set(
        &mut o.labels.color,
        opt_color(&style.label_color),
        fill_only_if_none,
    );
    set(
        &mut o.labels.font_size,
        style.label_font_size,
        fill_only_if_none,
    );
    set(
        &mut o.grid.color,
        opt_color(&style.grid_color),
        fill_only_if_none,
    );
    set(&mut o.grid.dash, style.grid_dash.clone(), fill_only_if_none);
    set(&mut o.grid.width, style.grid_width, fill_only_if_none);
    set(
        &mut o.domain.color,
        opt_color(&style.domain_color),
        fill_only_if_none,
    );
    set(&mut o.domain.width, style.domain_width, fill_only_if_none);
    Ok(())
}

/// Parse a per-channel typed [`AxisStyleSpec`] into the bundled
/// [`AxisStyleOverrides`] ready to drop into an `AxisInput` (B5). A fresh-build
/// over a `Default` bundle via the canonical [`axis_style_fill_from`]
/// merge (`fill_only_if_none = false`), the same merge the chart-level
/// `configure_axis` path uses (`fill_only_if_none = true`) — so the two cannot
/// drift on the field set, color parsing, or orient/title_orient validation.
///
/// The `labels.format` slot is left empty here (the fresh-build path never
/// writes it):
/// prepare resolves it through the temporal/numeric format threading
/// (`apply_axis_format_or_thread`) and seeds it onto the bundle afterward. An
/// unparseable color hex string yields `None` (theme fallback) rather than
/// failing. `placement.orient` is the validated override input; the concrete axis
/// side is resolved into `AxisInput.orient` once all override layers merge.
///
/// Returns [`crate::render::UnflippedRenderError`] (R3), not `RenderError`
/// directly: this fn has no `coord_flipped` of its own — it's `build_axes`
/// (via [`build_axis_input`]) that knows it and resolves at that boundary.
fn encoding_axis_style_overrides(
    axis: Option<&crate::render::chart_config::AxisStyleSpec>,
    channel: &'static str,
) -> Result<crate::layout::AxisStyleOverrides, crate::render::UnflippedRenderError> {
    let mut overrides = crate::layout::AxisStyleOverrides::default();
    if let Some(a) = axis {
        axis_style_fill_from(&mut overrides, a, channel, false)?;
    }
    Ok(overrides)
}

/// Apply the per-axis `tick_min_step` / `tick_extra` adjustments (B5 unit 2) to a
/// continuous axis's already-generated ticks, mutating the axis's `tick_labels`
/// and projected `tick_projection.major` fractions in lockstep.
///
/// - `tick_min_step` (data units): greedily drop any tick whose raw value is
///   within `min_step` of the previously kept tick (in domain order), thinning a
///   dense axis without disturbing the surviving labels/fractions.
/// - `tick_extra`: append a tick at each domain boundary (`scale min`/`max`) when
///   not already present (within a tiny epsilon), labeled via `format_numeric`
///   and projected to its domain fraction via the scale.
///
/// Both are no-ops for ordinal axes (no `tick_projection`), when neither field is
/// set, or when the tick count drifted, so default output is byte-identical. The
/// effective `ticks.extra`/`ticks.min_step` are read off the axis input AFTER the
/// chart-level config merge, so the per-channel value already won when present.
/// `tick_extra` formats boundary labels with `format_numeric`, so it is scoped to
/// non-temporal continuous axes (temporal boundary formatting needs spacing
/// context the scale only has internally — flagged as a bounded interpretation).
///
/// `reversed` mirrors the non-ordinal-y label reversal so the raw values pulled
/// from the scale stay index-aligned with the (reversed) labels/fractions.
pub(crate) fn adjust_axis_ticks(
    axis: &mut AxisInput,
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    reversed: bool,
) {
    let tick_min_step = axis.overrides.ticks.min_step;
    let tick_extra = axis.overrides.ticks.extra.unwrap_or(false);
    if tick_min_step.is_none() && !tick_extra {
        return;
    }
    // Only continuous axes carry a projection; ordinal axes have no numeric
    // domain to thin or bound.
    let Some(proj) = axis.tick_projection.as_mut() else {
        return;
    };
    let Some(raw_src) = scale.tick_values_raw(tick_count) else {
        return;
    };
    let is_temporal = matches!(scale, crate::render::scale_resolve::ScaleKind::Time(_));

    let mut raw: Vec<f64> = raw_src;
    if reversed {
        raw.reverse();
    }
    // Guard against index drift (a stale tick count): only act when the raw
    // values, labels, and projected fractions all line up.
    if raw.len() != axis.tick_labels.len() || raw.len() != proj.major.len() {
        return;
    }
    // `tick_anchors` lines up with `tick_labels`
    // only when it was built from the SAME raw tick count (non-empty, for a
    // Log scale) — carried in lockstep through every rewrite below, exactly
    // as `proj.major` already is. Empty (every non-`Log` axis) stays empty.
    let anchors_aligned = axis.tick_anchors.len() == raw.len();

    // ── tick_min_step: thin ticks closer than min_step in data space ─────────
    if let Some(min_step) = tick_min_step {
        if min_step > 0.0 && raw.len() > 1 {
            // Walk in domain order so "closer than min_step to the last kept" is
            // well-defined regardless of the reversed display order.
            let mut order: Vec<usize> = (0..raw.len()).collect();
            order.sort_by(|&a, &b| raw[a].total_cmp(&raw[b]));
            let mut keep = vec![false; raw.len()];
            let mut last_kept: Option<f64> = None;
            for &idx in &order {
                let v = raw[idx];
                let far_enough = last_kept.is_none_or(|lk| (v - lk).abs() >= min_step);
                if far_enough {
                    keep[idx] = true;
                    last_kept = Some(v);
                }
            }
            let mut k = 0;
            raw.retain(|_| {
                let keep_it = keep[k];
                k += 1;
                keep_it
            });
            let mut k = 0;
            axis.tick_labels.retain(|_| {
                let keep_it = keep[k];
                k += 1;
                keep_it
            });
            let mut k = 0;
            proj.major.retain(|_| {
                let keep_it = keep[k];
                k += 1;
                keep_it
            });
            if anchors_aligned {
                let mut k = 0;
                axis.tick_anchors.retain(|_| {
                    let keep_it = keep[k];
                    k += 1;
                    keep_it
                });
            }

            // A chart-level TIME format
            // (`labels.format`'s `format_type == "time"`) is applied AFTER this fn runs,
            // by `render::config_apply::apply_label_format_to_axis`, which re-derives raw
            // epoch-ms values via `scale.temporal_tick_values(tick_count)` —
            // an UN-thinned, full-length vec that can no longer line up with
            // `tick_labels` once thinning just dropped entries above, so its
            // own length guard silently falls through to a no-op (the
            // temporal labels are already pretty-printed strings that can't
            // be re-parsed), discarding the requested format entirely. `raw`
            // above is the exact correctly-thinned epoch-ms values in
            // lockstep with the tick_labels just retained — the ONE place
            // that lockstep invariant actually holds — so finalize the
            // temporal reformat HERE instead of trying to reconstruct it
            // later. Gated on the slot's `format_type` specifically (chart-
            // level-only-owned, see `LabelFormat::format_type`'s doc) so a
            // per-channel temporal format —
            // already fully applied before thinning ran, see
            // `apply_axis_format_or_thread` — is untouched (thinning simply
            // drops its already-correct label strings by index, same as any
            // other format). `apply_label_format_to_axis`'s own attempt
            // afterward safely no-ops on these now-thinned labels via its
            // existing length guard.
            let fmt_slot = axis.overrides.labels.format.get();
            // `TimeAutodetect::Denied`: the chart-level slot's own declared
            // `format_type` is the gate (see the comment above), not a guess
            // from the spec's content — an untyped `%`-bearing per-channel
            // spec was already applied by `apply_axis_format_or_thread` and
            // must not be applied a second time here.
            let temporal_spec = is_temporal
                .then(|| {
                    crate::render::format::FormatKind::resolve(
                        fmt_slot.spec.as_deref(),
                        fmt_slot.format_type.as_deref(),
                        crate::render::format::TimeAutodetect::Denied,
                    )
                })
                .flatten();
            if let Some(crate::render::format::FormatKind::Strftime(pattern)) = temporal_spec {
                axis.tick_labels = raw
                    .iter()
                    .map(|&ms| crate::render::format::format_time_spec(ms as i64, &pattern))
                    .collect();
            }
        }
    }

    // ── tick_extra: append domain-boundary ticks if missing ──────────────────
    if tick_extra && !is_temporal {
        // Re-checked post-`tick_min_step` (it may have just thinned `raw` in
        // lockstep with the anchors above): aligned only when non-empty and
        // still exactly `raw.len()` long.
        let anchors_aligned = axis.tick_anchors.len() == raw.len();
        if let Some((d_lo, d_hi)) = scale.data_domain() {
            let eps = ((d_hi - d_lo).abs() * 1e-9).max(f64::MIN_POSITIVE);
            for boundary in [d_lo, d_hi] {
                if !boundary.is_finite() {
                    continue;
                }
                if raw.iter().any(|&v| (v - boundary).abs() <= eps) {
                    continue;
                }
                let frac = scale.value_fractions(&[boundary]);
                let Some(&f) = frac.first() else { continue };
                if !f.is_finite() {
                    continue;
                }
                raw.push(boundary);
                axis.tick_labels
                    .push(crate::render::format::format_numeric(boundary));
                proj.major.push(f);
                if anchors_aligned {
                    let anchor = scale.tick_anchors(&[boundary]).first().copied().unwrap_or(false);
                    axis.tick_anchors.push(anchor);
                }
            }
        }
    }

    // `raw` now holds the FINAL tick values, in lockstep with `tick_labels`,
    // after whatever `tick_min_step`/`tick_extra` just did to them — re-derive
    // the origin from them directly, rather than trying to track an index
    // through the retains/pushes above the way `tick_anchors` does.
    axis.tick_origin = scale.tick_origin(&raw);
}

/// Resolve the tick-label format `(spec, type)` for a positional channel.
///
/// Precedence: a per-channel `Axis(label_format=, label_format_type=)` (in the
/// `encoding.axis` map) wins over the shorthand `encoding.format`/`format_type`.
/// Returns `(None, None)` when neither is set so default formatting is preserved.
///
/// An `Axis(label_format_type=)` with no `label_format` still declares the
/// type: it fills the type over the shorthand's, field by field, so a bare
/// `"time"` reaches the one refusal rule instead of being dropped here
/// (`#147` close, wave 6A T21 cycle 2). The merged pair is what
/// `config_apply::validate_chart_format_specs` gates, and `build_axis_input`
/// still refuses the shorthand pair the Axis type overrode (cycle 3), so a
/// pair split across the two spellings meets the checks its one-place
/// spellings meet.
///
/// `pub(crate)` so `scene_build.rs` can re-derive the format spec for
/// independent-axis per-panel label formatting without duplicating the
/// precedence logic.
pub(crate) fn resolve_axis_label_format(
    enc: Option<&crate::spec::encoding::EncodingSpec>,
) -> (Option<String>, Option<String>) {
    let Some(e) = enc else { return (None, None) };
    let axis = e.axis.as_ref();
    if let Some(fmt) = axis.and_then(|a| a.label_format.clone()) {
        let ty = axis.and_then(|a| a.label_format_type.clone());
        return (Some(fmt), ty);
    }
    let ty = axis.and_then(|a| a.label_format_type.clone()).or_else(|| e.format_type.clone());
    (e.format.clone(), ty)
}

/// Apply a resolved tick-label format to one axis, returning the (possibly
/// reformatted) labels and the `label_format_override` to thread into the
/// `AxisInput`.
///
/// Two paths:
/// - **Temporal** axis with a strftime pattern (`format` contains `%` and
///   `format_type` is `"time"` or unset): the raw epoch-ms tick values are
///   pulled from the scale and formatted directly via `chrono` here. No override
///   is threaded (the strings are already final; re-parsing them downstream
///   would fail). Returns `(formatted, None)`.
/// - **Numeric** axis (or any non-temporal): a d3 spec is threaded forward as the
///   `label_format_override`, and the
///   labels are returned unchanged. `render::config_apply::apply_label_format_to_axis`
///   then applies it centrally — lossless because numeric label strings reparse
///   to f64. Threading (not pre-applying) also lets chart-level
///   `configure_axis` defer to the per-channel override via its `is_none()` gate.
///   The ordinal preset takes this path whatever `format_type` says, on a
///   non-temporal scale; on a temporal one it is refused (`#147`).
/// - **Time-typed, naming no pattern** ([`FormatKind::TemporalDefault`], e.g.
///   `".2f"` + `"time"`): nothing is threaded and the scale's own labels stand.
///
/// Errors (#133) when a time-typed format is
/// applied to a scale whose `temporal_tick_values` returns `None` — i.e. the
/// resolved [`crate::render::scale_resolve::ScaleKind`] is not `Time` at all,
/// not merely a length-drift between a fresh tick computation and the cached
/// labels (drift keeps its pre-existing string-reparse fallback, since the
/// scale genuinely IS temporal there); when the ordinal preset (`#147`)
/// or a d3 number format (`#147`) meets a `Time` scale; and when the
/// spec does not parse under the grammar it resolved to, e.g. an undeclared
/// `"%Y"` on a non-`Time` scale, which is d3 there (`InvalidFormatSpec`,
/// `#147`). `channel` is the RESOLVED token this
/// call acted on; `coord_flipped` is always constructed `false` here — the
/// caller boundary (`build_axis_input`'s two call chains) corrects it via R3,
/// the same discipline `InvalidAxisOrient` uses.
fn apply_axis_format_or_thread(
    labels: Vec<String>,
    format: Option<String>,
    format_type: Option<&str>,
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    reversed: bool,
    channel: &'static str,
) -> Result<(Vec<String>, Option<String>), RenderError> {
    use crate::render::format::{format_time_spec, FormatKind};
    // The one refusal every surface applies to the identical pair, decided
    // once for every kind below: a declared time type on a non-temporal scale
    // (`#133`; a bare `format_type="time"` included, `#147` T21 cycle 2), the
    // ordinal preset and a d3 number format on a temporal one (`#147` T22,
    // T23), and a spec its resolved grammar cannot parse.
    refuse_format_temporality_mismatch_scale(format.as_deref(), format_type, scale, channel)?;
    let Some(fmt) = format else {
        // A bare `format_type="time"` names no pattern: the scale's own
        // labels stand.
        return Ok((labels, None));
    };
    // The resolution the refusal above made: an undeclared `%`-bearing spec
    // is strftime on a temporal scale and d3 on any other, so there is no
    // guess left to undo here (`#147` T23; the drift fallback below is the
    // one exception).
    let kind = FormatKind::resolve_on(Some(&fmt), format_type, scale_temporality(scale));
    if let Some(FormatKind::Strftime(pattern)) = &kind {
        if let Some(mut values) = scale.temporal_tick_values(tick_count) {
            if reversed {
                values.reverse();
            }
            // Guard against index drift between the cached labels and a fresh
            // tick computation: only substitute when the counts line up.
            if values.len() == labels.len() {
                let out = values
                    .into_iter()
                    .map(|ms| format_time_spec(ms, pattern))
                    .collect();
                return Ok((out, None));
            }
        }
        // Time scale, but drift: pre-existing string-reparse fallback,
        // unrelated to #133 (the scale genuinely IS temporal here, since a
        // strftime kind resolves only on one). It re-resolves the raw pair
        // under `TimeAutodetect::Denied` inside `apply_tick_format`, so an
        // undeclared pattern is read as d3 there: the one place in this fn
        // that does not use `kind` (pre-existing, ledgered in the `#147` T23
        // report as a follow-up).
        return Ok((apply_tick_format(labels, Some(&fmt), format_type), None));
    }
    if matches!(kind, Some(FormatKind::TemporalDefault)) {
        // Time-typed, naming no pattern (`".2f"` + `"time"`): on a temporal
        // scale (a non-temporal one was refused above, #133) the scale's own
        // spacing-keyed labels stand, the same fallback the colorbar and text
        // mark give the identical pair. Threading the spec instead re-seeds
        // it untyped and applies it as d3 (`2001.00` on year ticks, `#147`).
        return Ok((labels, None));
    }
    // A d3 format (`FormatKind::D3`) or the ordinal preset, on a
    // non-temporal scale: thread the spec to the override, apply later.
    Ok((labels, Some(fmt)))
}

/// #133: refuse when `format_type`
/// unambiguously declares `"time"` but the resolved `scale` is not temporal
/// at all — a non-`Time` [`crate::render::scale_resolve::ScaleKind`] (the
/// case `scale.temporal_tick_values` answers `None`), never a length-drift on
/// a genuinely temporal scale; and, the rule's other half, when the ordinal
/// preset or a d3 number format meets a temporal scale. The axis wrapper of
/// [`refuse_format_temporality_mismatch`], which owns the rule.
///
/// The pair resolves by the scale's temporality, as the axis applier formats
/// it: an undeclared `%`-bearing spec is strftime on a `Time` scale and d3 on
/// any other, so a plain d3 percent format like `".1%"` keeps the numeric
/// grammar on a numeric axis, and the ordinal preset resolves to
/// [`crate::render::format::FormatKind::Ordinal`] whatever the type says (so
/// it is never refused as a time format). Both of
/// `build_axis_tick_inputs`'s mode arms reach this through
/// [`apply_axis_format_or_thread`] (the per-panel independent-facet-axis
/// surface included), so the two format-application
/// disciplines cannot drift on which scale kinds refuse. `channel` is the
/// RESOLVED token; `coord_flipped` is always constructed `false` — corrected
/// by the caller boundary (R3, same discipline as `InvalidAxisOrient`).
///
/// `pub(crate)` (wave-1 close remediation, #133 predicate-sharing finding):
/// `config_apply::apply_label_format_to_axis` shares this same refusal
/// predicate at its own two sites rather than hand-rolling the
/// `format_type == "time"` + `scale.temporal_tick_values(...).is_none()`
/// check a third and fourth time — all four sites that can refuse a
/// time-typed format on a non-temporal scale now build the SAME error
/// through this one function. `fmt` is `None` for a bare
/// `format_type="time"`, refused like any other time-typed pair.
pub(crate) fn refuse_format_temporality_mismatch_scale(
    fmt: Option<&str>,
    format_type: Option<&str>,
    scale: &crate::render::scale_resolve::ScaleKind,
    channel: &'static str,
) -> Result<(), RenderError> {
    refuse_format_temporality_mismatch(
        fmt,
        format_type,
        scale_temporality(scale),
        channel,
        crate::render::FormatSurface::Axis,
    )
}

/// Whether an axis's resolved scale is temporal: `ScaleKind::Time`, exactly
/// when `temporal_tick_values` answers `Some`. The axis half of the
/// temporality [`refuse_format_temporality_mismatch`] and the axis applier
/// both read (the field half is [`field_temporality`]).
fn scale_temporality(
    scale: &crate::render::scale_resolve::ScaleKind,
) -> crate::render::format::Temporality {
    use crate::render::format::Temporality;
    match scale {
        crate::render::scale_resolve::ScaleKind::Time(_) => Temporality::Temporal,
        _ => Temporality::NonTemporal,
    }
}

/// The field-checked sibling of [`refuse_format_temporality_mismatch_scale`], for
/// the surfaces that format a channel's own values rather than an axis's
/// ticks: every legend (`Legend(format=, format_type=)` on color, size, shape
/// and stroke-dash, via
/// [`legend::refuse_format_temporality_mismatch_legend_fields`]) and the text
/// channel (`Text(format=, format_type=)`). Before `#147`'s close (wave
/// 6A T21) those rendered a time-typed format on a quantitative field as a
/// column of `1970-01-01` labels (the epoch read of small numbers) while the
/// axes refused the identical pair.
///
/// "Temporal field" is the scale resolver's own rule
/// ([`crate::render::scale_resolve::infer_spec_type`]: a declared type wins,
/// else the column's dtype), so `"n:T"` over epoch-ms numbers stays
/// temporal exactly as it does on an axis. A field with no declared type
/// that `batch` does not carry has no answer here and is not refused: the
/// read that follows raises its own `UnknownColumn`.
pub(crate) fn refuse_format_temporality_mismatch_field(
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    fmt: Option<&str>,
    format_type: Option<&str>,
    channel: &'static str,
    surface: crate::render::FormatSurface,
) -> Result<(), RenderError> {
    let Some(field) = field_temporality(enc, batch) else {
        return Ok(());
    };
    refuse_format_temporality_mismatch(fmt, format_type, field, channel, surface)
}

/// Whether `enc`'s field is temporal, by the scale resolver's own rule
/// ([`crate::render::scale_resolve::infer_spec_type`]: a declared type wins,
/// else the column's dtype). `None` when the field has no declared type and
/// `batch` does not carry it. The one answer the field-checked refusal and
/// the legend and text appliers ([`crate::render::format::format_value_with_spec`])
/// both read, so the kind a surface refuses and the kind it formats with
/// come from the same temporality (`#147` T23).
pub(in crate::render) fn field_temporality(
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
) -> Option<crate::render::format::Temporality> {
    use crate::render::format::Temporality;
    batch
        .column_by_name(&enc.field)
        .map(|col| crate::render::scale_resolve::infer_spec_type(enc, col.data_type()))
        .or(enc.type_)
        .map(|t| match t {
            crate::spec::encoding::DataType::Temporal => Temporality::Temporal,
            _ => Temporality::NonTemporal,
        })
}

/// The one rule behind every `TimeFormatOnNonTemporalScale`,
/// `OrdinalFormatOnTemporalScale` and `NumberFormatOnTemporalScale`: the
/// `(fmt, format_type)` pair is resolved once, by the value's temporality
/// ([`crate::render::format::FormatKind::resolve_on`], the resolution every
/// applier formats with: an undeclared `%`-bearing spec is strftime on a
/// temporal value and d3 otherwise), and its kind must match that
/// temporality. A time-typed kind (`Strftime` or `TemporalDefault`) refuses
/// on a value that is not temporal. On a value that is, the ordinal preset
/// refuses (`#147`) and so does a d3 number format
/// (`#147`). A matching kind must parse under its
/// grammar (`InvalidFormatSpec`, the declaration-time gate's own error).
/// The two wrappers above answer "is it temporal" for their surface; this
/// answers everything else.
fn refuse_format_temporality_mismatch(
    fmt: Option<&str>,
    format_type: Option<&str>,
    target: crate::render::format::Temporality,
    channel: &'static str,
    surface: crate::render::FormatSurface,
) -> Result<(), RenderError> {
    use crate::render::format::{validate_d3_format_spec, validate_strftime_spec, FormatKind};
    // An empty spec names no format: every applier renders its default.
    let fmt = fmt.filter(|f| !f.is_empty());
    let invalid = |spec: &str, reason| RenderError::InvalidFormatSpec { spec: spec.to_string(), reason };
    let target_is_temporal = target == crate::render::format::Temporality::Temporal;
    match FormatKind::resolve_on(fmt, format_type, target) {
        Some(kind) if kind.is_temporal() && !target_is_temporal => {
            Err(RenderError::TimeFormatOnNonTemporalScale {
                channel,
                surface,
                format: fmt.map(str::to_string),
                coord_flipped: false,
            })
        }
        Some(FormatKind::Ordinal) if target_is_temporal => {
            Err(RenderError::OrdinalFormatOnTemporalScale { channel, surface, coord_flipped: false })
        }
        Some(FormatKind::D3(_)) if target_is_temporal => {
            Err(RenderError::NumberFormatOnTemporalScale {
                channel,
                surface,
                format: fmt.unwrap_or_default().to_string(),
                coord_flipped: false,
            })
        }
        // The kind matches the value; its grammar must parse. The
        // declaration-time gate could not always check it: an undeclared
        // field's `%`-bearing spec is either grammar until the field's
        // temporality is known, which is here.
        Some(FormatKind::Strftime(pattern)) => {
            validate_strftime_spec(&pattern).map_err(|reason| invalid(&pattern, reason))
        }
        Some(FormatKind::D3(_)) => {
            let spec = fmt.unwrap_or_default();
            validate_d3_format_spec(spec).map_err(|reason| invalid(spec, reason))
        }
        _ => Ok(()),
    }
}

/// How [`build_axis_tick_inputs`] applies the resolved tick-label format. The two
/// modes are the two byte-distinct format disciplines the render spine already
/// has — surfaced as one enum so the shared and independent axis paths run the
/// same tick-derivation function while keeping their (intentionally different)
/// format handling.
pub(crate) enum TickFormatMode<'a> {
    /// Shared/global axes: format temporal axes directly, but THREAD a numeric
    /// format forward (returned as the second tuple element) so chart-level
    /// `configure_axis` can still defer to the per-channel override via its
    /// `is_none()` gate, with `apply_label_format_to_axis` applying it centrally
    /// later. Mirrors [`apply_axis_format_or_thread`].
    Thread {
        format: Option<String>,
        format_type: Option<&'a str>,
    },
    /// Independent per-panel axes: resolve the format exactly as [`Thread`]
    /// does, then apply it IMMEDIATELY and thread nothing (the per-panel
    /// layout is built directly in `scene_build`, bypassing
    /// `apply_label_format_to_axis`, so there is no later applier to defer to).
    ///
    /// The two modes differ in TIMING, not in discipline: this arm calls
    /// [`apply_axis_format_or_thread`] — the only function that can re-derive
    /// raw epoch-ms tick values from the scale — and then runs
    /// [`apply_tick_format`] on whatever that returned as threaded, which is
    /// the numeric case only.
    ///
    /// It used to call [`apply_tick_format`] directly, and that WAS the `#147`
    /// defect: `apply_tick_format`'s only input is the already-formatted label
    /// strings, which on a temporal scale cannot be re-parsed as timestamps,
    /// so an independent-x/y facet panel silently dropped every temporal tick
    /// format while the shared axis honored the identical spelling.
    ///
    /// [`Thread`]: TickFormatMode::Thread
    Immediate {
        format: Option<&'a str>,
        format_type: Option<&'a str>,
    },
}

/// `build_axis_tick_inputs`'s return shape: `(labels, anchors, origin,
/// projection, threaded_label_format)` — named to keep the
/// `Result<_, RenderError>` signature (#133) under clippy's type-complexity
/// threshold. `anchors` is index-aligned with `labels`; empty for every
/// non-`Log` scale. `origin` is the index, among `labels`, of the axis's
/// nicest tick ([`crate::render::scale_resolve::ScaleKind::tick_origin`]);
/// `None` for Time/Log/Ordinal and for fewer than two ticks.
type AxisTickInputs = (Vec<String>, Vec<bool>, Option<usize>, Option<TickProjection>, Option<String>);

/// Derive one axis's tick labels and [`TickProjection`] from a resolved scale —
/// the single source of truth for the sequence the shared (global) and
/// independent (per-panel facet) axis paths both run: `tick_labels(count)`, the
/// non-ordinal-y label/fraction reversal, format application, and the
/// fraction-projection build.
///
/// Returns `(labels, anchors, origin, projection, threaded_label_format)`.
/// The threaded format is
/// `Some` only under [`TickFormatMode::Thread`] for a numeric axis (the caller
/// seeds it onto `AxisInput.overrides.labels.format`); it is always `None` under
/// [`TickFormatMode::Immediate`].
///
/// `minor` is supplied by the caller (already projected, NOT reversed) so this fn
/// stays theme-agnostic: the shared path passes the `theme.grid.minor`-gated
/// `minor_tick_fractions()`; the independent path passes `Vec::new()` (per-panel
/// independent axes do not rebuild minor ticks — preserved from the pre-extraction
/// behavior). An ordinal scale yields no projection (`None`), matching uniform-
/// slot placement.
///
/// `text_demand` is likewise supplied by the caller rather than
/// read off `scale`: `scale` here is the metadata-only placeholder `(0, 1)`-range
/// resolution (`scale.padding_fraction()`'s own precondition), over which the
/// demand's pixel needs would be meaningless, so the caller passes the SAME
/// `TextEdgeDemand` `ScaleKind::text_edge_demand` built for the panel-real
/// axis, carried onto [`TickProjection::text_demand`] for layout to apply at
/// the real panel range. Empty for every axis no `Mark::Text` layer draws.
///
/// Errors (#133) when either mode's format is time-typed but the resolved
/// `scale` is not temporal — see [`refuse_format_temporality_mismatch_scale`],
/// which both mode arms share so the surfaces cannot drift on the refusal
/// predicate. `channel` (derived from `is_y`, matching [`Channel::token`])
/// is the RESOLVED token; the caller boundary corrects `coord_flipped` (R3).
pub(crate) fn build_axis_tick_inputs(
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    mode: TickFormatMode<'_>,
    is_y: bool,
    minor: Vec<f64>,
    text_demand: crate::layout::geometry::TextEdgeDemand,
) -> Result<AxisTickInputs, RenderError> {
    use crate::render::scale_resolve::ScaleKind;
    let channel: &'static str = if is_y { "y" } else { "x" };
    // Non-ordinal y axes display high domain values at the top, so labels and the
    // projected fractions are reversed in lockstep (the pixel range is already
    // inverted in scale_resolve). Ordinal y keeps domain order (top-down for
    // heatmaps/confusion matrices).
    let reverse = is_y && !matches!(scale, ScaleKind::Ordinal(_));

    let mut labels = scale.tick_labels(tick_count);
    let mut fractions = scale.tick_fractions(tick_count);
    let padding_frac = scale.padding_fraction();
    let raw_values = scale.tick_values_raw(tick_count);
    // Which of these SAME tick values (same `count_hint`,
    // so index-aligned with `labels`/`fractions`) are `Log` decades — empty
    // for every non-`Log` scale. Reversed in the SAME lockstep as `labels`
    // so index `i` still names the same tick after either order.
    let mut anchors = raw_values.as_ref().map(|v| scale.tick_anchors(v)).unwrap_or_default();
    // The index, among the SAME tick values, of the axis's nicest tick (see
    // `ScaleKind::tick_origin`) — `None` for Time/Log/Ordinal and for fewer
    // than two ticks. Re-indexed in the SAME lockstep as `labels` so it
    // still names the same tick after either order.
    let mut origin = raw_values.as_ref().and_then(|v| scale.tick_origin(v));
    if reverse {
        labels.reverse();
        fractions.reverse();
        anchors.reverse();
        if let Some(o) = origin {
            // `labels.len() - 1 - o` underflows for an out-of-range `o`
            // (or an empty `labels`) — the same `o < labels.len()` guard
            // `AxisInput::set_ticks` applies to a mismatched origin,
            // degrading to `None` instead.
            origin = (o < labels.len()).then(|| labels.len() - 1 - o);
        }
    }

    let (labels, threaded) = match mode {
        TickFormatMode::Thread {
            format,
            format_type,
        } => apply_axis_format_or_thread(
            labels,
            format,
            format_type,
            scale,
            tick_count,
            // The temporal raw values must be reversed in lockstep with the
            // already-reversed labels so index `i` still aligns.
            reverse,
            channel,
        )?,
        TickFormatMode::Immediate {
            format,
            format_type,
        } => {
            // The #133 refusal lives in `apply_axis_format_or_thread` for
            // every time-declaring kind, so this arm refuses exactly what the
            // `Thread` arm refuses (it used to refuse up front on the raw
            // `format_type`, which also refused the ordinal preset).
            //
            // ONE format discipline, two application TIMINGS (`#147`): resolve
            // through the same function `Thread` uses — which is the only one
            // that can re-derive raw epoch-ms tick values from a temporal
            // scale — then immediately apply whatever `Thread` would have
            // threaded forward for its caller to apply later.
            //
            // Before this, `Immediate` went straight to `apply_tick_format`,
            // whose only input is the already-formatted label STRINGS. On a
            // temporal axis those strings cannot be re-parsed as timestamps,
            // so an independent-x/y facet panel silently DROPPED every
            // temporal tick format — `fm.Axis(label_format="%Y/%m/%d")`
            // rendered dates on a shared axis and the scale's own default
            // spacing-keyed strings on an independent one, for the same chart.
            let (labels, threaded) = apply_axis_format_or_thread(
                labels,
                format.map(str::to_string),
                format_type,
                scale,
                tick_count,
                reverse,
                channel,
            )?;
            match threaded {
                Some(fmt) => (apply_tick_format(labels, Some(&fmt), format_type), None),
                None => (labels, None),
            }
        }
    };

    // A non-empty major-fraction vec means a continuous scale → scale-projected
    // placement; an empty vec means ordinal/discretizing → no carrier (uniform
    // slots), byte-identical to the pre-extraction `None`.
    let projection = (!fractions.is_empty()).then_some(TickProjection {
        padding_frac,
        major: fractions,
        minor,
        text_demand,
    });

    Ok((labels, anchors, origin, projection, threaded))
}

/// D12: apply an encoding-level `format` string to pre-computed tick label strings.
///
/// The scale's `tick_labels()` method returns pre-formatted strings (via
/// `format_numeric`). When the encoding carries an explicit `format` string
/// (e.g. `.2f`, `~s`, `,`), we re-parse each label back to f64 and re-format it
/// via the canonical d3-format grammar in `crate::render::format`.
///
/// Resolution is [`format::FormatKind::resolve`] with
/// [`format::TimeAutodetect::Denied`]: this fn's only input is the
/// already-stringified labels, so it holds no scale to fall back on and must
/// not guess that an untyped `%`-bearing spec is a date pattern. Callers that
/// DO hold the scale (`apply_axis_format_or_thread`, which every axis path now
/// routes through first) resolve the temporal case before reaching here.
///
/// When `format_type == "time"`, labels are treated as epoch-ms integers. A
/// format string containing `%` resolves to a `chrono` strftime pattern (e.g.
/// `"%b %Y"`); a time-typed pair naming no pattern falls back to the default
/// spacing-keyed temporal formatter. Labels that fail to parse are left
/// unchanged (ordinal labels, already-formatted strings, etc.).
pub(crate) fn apply_tick_format(
    labels: Vec<String>,
    format: Option<&str>,
    format_type: Option<&str>,
) -> Vec<String> {
    use crate::render::format::{format_time, format_time_spec, FormatKind, TimeAutodetect, DAY_MS};
    // No spec and no `"time"` type is a pure pass-through (byte-identical).
    let Some(kind) = FormatKind::resolve(format, format_type, TimeAutodetect::Denied) else {
        return labels;
    };
    let epoch_ms = |raw: &str| -> Option<i64> {
        raw.parse::<i64>()
            .ok()
            .or_else(|| raw.parse::<f64>().ok().map(|f| f as i64))
    };
    labels
        .into_iter()
        .map(|raw| match &kind {
            FormatKind::Strftime(pattern) => match epoch_ms(&raw) {
                Some(ms) => format_time_spec(ms, pattern),
                None => raw,
            },
            // Time-typed but naming no pattern: the default spacing-keyed
            // temporal formatter, the same fallback `format_value_with_spec`
            // uses for the identical pair.
            FormatKind::TemporalDefault => match epoch_ms(&raw) {
                Some(ms) => format_time(ms, DAY_MS),
                None => raw,
            },
            // A d3 format (`FormatKind::D3`) or the ordinal preset, whatever `format_type` says.
            FormatKind::D3(_) | FormatKind::Ordinal => match raw.parse::<f64>() {
                Ok(v) => kind.format(v),
                Err(_) => raw, // ordinal category name — pass through
            },
        })
        .collect()
}

#[cfg(test)]
mod axis_style_fill_from_tests {
    //! Characterization tests for `axis_style_fill_from` (T2.6 / SPINE-01): the
    //! one merge that replaced the two parallel `AxisStyleSpec → AxisStyleOverrides`
    //! mappers. These pin the exact field ownership both old mappers had so the
    //! collapse stays behavior-preserving.
    use super::*;
    use crate::layout::{AxisStyleOverrides, LabelOverlap};
    use crate::render::chart_config::AxisStyleSpec;
    use crate::render::color;

    /// An `AxisStyleSpec` with every merge-relevant field populated, plus a couple
    /// of fields that exercise color parsing and token validation.
    fn fully_populated_spec() -> AxisStyleSpec {
        AxisStyleSpec {
            label_angle: Some(45.0),
            label_font_size: Some(11.0),
            label_color: Some("#112233".into()),
            label_format: Some(".2f".into()),
            label_format_type: None,
            label_overlap: Some("parity".into()),
            label_flush: Some(true),
            labels: Some(false), // show toggle — NOT an overrides field; must be ignored
            ticks: Some(false),  // show toggle — ignored
            tick_count: Some(7), // not an overrides field — ignored
            tick_size: Some(4.0),
            tick_extra: Some(true),
            tick_min_step: Some(0.5),
            values: Some(vec![0.0, 1.0, 2.0]),
            grid: Some(false), // show toggle — ignored
            grid_color: Some("#445566".into()),
            grid_dash: Some(vec![6.0, 3.0]),
            grid_width: Some(1.5),
            grid_opacity: Some(0.4),
            domain: Some(false), // show toggle — ignored
            domain_color: Some("#778899".into()),
            domain_width: Some(2.0),
            title: Some("ignored-here".into()), // title text — not an overrides field
            title_font_size: Some(13.0),
            title_color: Some("#aabbcc".into()),
            title_padding: Some(8.0),
            title_orient: Some("right".into()),
            label_padding: Some(3.0),
            orient: Some("bottom".into()),
            translate: Some(5.0),
            min_band: Some(10.0),
            max_band: Some(40.0),
            offset: Some(2.0),
            zindex: Some(1),
        }
    }

    /// Per-channel fresh-build (`fill_only_if_none = false`) writes every
    /// overrides field from the spec — EXCEPT the `labels.format` slot, which
    /// the per-channel/prepare path deliberately leaves empty (it is threaded
    /// separately afterward). The non-bundle fields (title text, tick_count) are
    /// never written because they are not on this struct at all.
    /// *(Corrected 2026-09-20, pre-existing: this sentence used to list the
    /// tick size among the never-written non-bundle fields and to group the
    /// show toggles with them. Both are bundle fields — the merge writes
    /// `o.ticks.size` and all four `show` slots. False at `HEAD` verbatim; not a
    /// task-11 regression.)*
    #[test]
    fn fresh_build_writes_all_fields_except_label_format() {
        let spec = fully_populated_spec();
        let mut o = AxisStyleOverrides::default();
        axis_style_fill_from(&mut o, &spec, "x", false).unwrap();

        assert_eq!(o.labels.angle, Some(45.0));
        assert_eq!(o.labels.font_size, Some(11.0));
        assert_eq!(o.labels.color, color::parse_color("#112233").ok());
        // The `labels.format` slot MUST stay empty on the per-channel/fresh-build path.
        assert_eq!(o.labels.format.get().spec, None);
        assert_eq!(o.labels.overlap, Some(LabelOverlap::Parity));
        assert_eq!(o.labels.flush, Some(true));
        assert_eq!(o.ticks.extra, Some(true));
        assert_eq!(o.ticks.min_step, Some(0.5));
        assert_eq!(o.ticks.values, Some(vec![0.0, 1.0, 2.0]));
        assert_eq!(o.grid.color, color::parse_color("#445566").ok());
        assert_eq!(o.grid.dash, Some(vec![6.0, 3.0]));
        assert_eq!(o.grid.width, Some(1.5));
        assert_eq!(o.grid.opacity, Some(0.4));
        assert_eq!(o.domain.color, color::parse_color("#778899").ok());
        assert_eq!(o.domain.width, Some(2.0));
        assert_eq!(o.title.font_size, Some(13.0));
        assert_eq!(o.title.color, color::parse_color("#aabbcc").ok());
        assert_eq!(o.title.padding, Some(8.0));
        assert_eq!(o.title.orient, Some(crate::layout::AxisOrient::Right));
        assert_eq!(o.labels.padding, Some(3.0));
        assert_eq!(o.placement.orient, Some(crate::layout::AxisOrient::Bottom));
        assert_eq!(o.placement.translate, Some(5.0));
        assert_eq!(o.placement.min_band, Some(10.0));
        assert_eq!(o.placement.max_band, Some(40.0));
        assert_eq!(o.placement.offset, Some(2.0));
        assert_eq!(o.placement.zindex, Some(1));
    }

    /// Chart-level fill (`fill_only_if_none = true`) fills only `None` slots
    /// (higher-precedence values survive) AND owns the `labels.format` slot (the
    /// one slot the per-channel path leaves empty).
    #[test]
    fn chart_level_overwrites_no_claimed_slot_and_owns_label_format() {
        let spec = fully_populated_spec();
        // EVERY slot the `set(...)` helper routes is pre-claimed, with a value
        // distinct from `fully_populated_spec()`'s. Pre-setting only two of them
        // (as this test did before the #143 remediation) left ~24 fields free to
        // silently switch to unconditional-overwrite: mutation review M8 flipped
        // `tick_values` and the suite stayed green, while the same flip on
        // `grid_width` — one of the two that WERE pre-set — was caught. The
        // asymmetry was an oversight, not a decision, so the field set is now
        // exhaustive and a discipline flip on any single field fails here.
        let mut o = AxisStyleOverrides {
            ticks: crate::layout::axis::AxisTickOverrides {
                values: Some(vec![7.0]),
                extra: Some(false),
                min_step: Some(0.25),
                size: Some(88.0),
                show: Some(true),
            },
            labels: crate::layout::axis::AxisLabelOverrides {
                angle: Some(90.0),
                format: Default::default(),
                padding: Some(93.0),
                color: crate::render::color::parse_color("#040506").ok(),
                font_size: Some(94.0),
                flush: Some(false),
                overlap: Some(LabelOverlap::Greedy),
                show: Some(true),
            },
            title: crate::layout::axis::AxisTitleOverrides {
                font_size: Some(91.0),
                color: crate::render::color::parse_color("#010203").ok(),
                padding: Some(92.0),
                orient: Some(AxisOrient::Left),
            },
            grid: crate::layout::axis::AxisGridOverrides {
                color: crate::render::color::parse_color("#070809").ok(),
                dash: Some(vec![9.0, 9.0]),
                width: Some(99.0),
                opacity: Some(0.99),
                show: Some(true),
            },
            domain: crate::layout::axis::AxisDomainOverrides {
                color: crate::render::color::parse_color("#0a0b0c").ok(),
                width: Some(95.0),
                show: Some(true),
            },
            placement: crate::layout::axis::AxisPlacementOverrides {
                orient: Some(AxisOrient::Top),
                translate: Some(96.0),
                offset: Some(89.0),
                min_band: Some(97.0),
                max_band: Some(98.0),
                zindex: Some(42),
            },
        };
        let pre_set = o.clone();
        axis_style_fill_from(&mut o, &spec, "x", true).unwrap();

        // Every pre-set slot survives (fill-only-if-None) — the whole field set,
        // asserted as one comparison so a newly added field cannot slip through
        // by being absent from a hand-listed assertion block. The format slot is
        // blanked on both sides because it is the one the chart-level path owns.
        let without_format = |mut ov: AxisStyleOverrides| {
            ov.labels.format = Default::default();
            ov
        };
        assert_eq!(
            without_format(o.clone()),
            without_format(pre_set),
            "a chart-level fill must not overwrite ANY slot a higher-precedence \
             per-channel value already claimed"
        );
        // `labels.format` is the one slot the chart-level path OWNS: the
        // per-channel path leaves it `None`, so it is excluded above and
        // asserted as written here.
        assert_eq!(o.labels.format.get().spec, Some(".2f".into()));
    }

    /// Chart-level fill does NOT clobber a `labels.format` slot that a per-channel
    /// value already set (the threaded override wins).
    #[test]
    fn chart_level_label_format_defers_to_existing() {
        let spec = fully_populated_spec(); // label_format = ".2f"
        let mut o = AxisStyleOverrides::default();
        // per-channel already won
        o.labels.format.set(crate::layout::LabelFormat {
            spec: Some("~s".into()),
            format_type: None,
        });
        axis_style_fill_from(&mut o, &spec, "x", true).unwrap();
        assert_eq!(o.labels.format.get().spec, Some("~s".into()));
    }

    /// A cross-dimension `orient` (left/right on an x axis) fails loud on both
    /// paths, exactly as both old mappers did via `parse_axis_orient`.
    #[test]
    fn cross_dimension_orient_errors() {
        let spec = AxisStyleSpec {
            orient: Some("left".into()),
            ..Default::default()
        };
        let mut o = AxisStyleOverrides::default();
        assert!(axis_style_fill_from(&mut o, &spec, "x", false).is_err());
        let mut o = AxisStyleOverrides::default();
        assert!(axis_style_fill_from(&mut o, &spec, "x", true).is_err());
    }

    /// An unparseable color hex leaves the slot `None` (theme fallback) rather
    /// than failing — preserved from both old mappers.
    #[test]
    fn bad_color_hex_falls_back_to_none() {
        let spec = AxisStyleSpec {
            label_color: Some("not-a-color".into()),
            ..Default::default()
        };
        let mut o = AxisStyleOverrides::default();
        axis_style_fill_from(&mut o, &spec, "x", false).unwrap();
        assert_eq!(o.labels.color, None);
    }
}

#[cfg(test)]
mod tick_format_tests {
    use super::*;

    /// Regression: apply_tick_format with a d3 percent spec must format numerics.
    /// This exercises the code path used by scene_build.rs independent-axis label
    /// formatting — where apply_tick_format is called directly with the raw format
    /// spec from resolve_axis_label_format.
    #[test]
    fn tick_format_percent_spec_applied_to_numerics() {
        let labels = vec!["0".to_string(), "0.5".to_string(), "1".to_string()];
        let out = apply_tick_format(labels, Some(".0%"), None);
        // ".0%" of 0.5 = "50%"
        assert_eq!(out, vec!["0%", "50%", "100%"]);
    }

    fn fmt_labels(labels: &[&str], format: &str) -> Vec<String> {
        apply_tick_format(
            labels.iter().map(|s| s.to_string()).collect(),
            Some(format),
            None,
        )
    }

    #[test]
    fn currency_format_prepends_dollar() {
        // "$,.0f" should produce "$10", "$1,000", etc.
        assert_eq!(
            fmt_labels(&["10", "1000", "50"], "$,.0f"),
            vec!["$10", "$1,000", "$50"]
        );
    }
    #[test]
    fn dotf_two_decimals() {
        assert_eq!(
            fmt_labels(&["2", "2.5", "3"], ".2f"),
            vec!["2.00", "2.50", "3.00"]
        );
    }
    #[test]
    fn dot1f_one_decimal() {
        assert_eq!(
            fmt_labels(&["1", "1.5", "2"], ".1f"),
            vec!["1.0", "1.5", "2.0"]
        );
    }
    #[test]
    fn percent_format() {
        assert_eq!(
            fmt_labels(&["0.1", "0.123", "0.5"], ".1%"),
            vec!["10.0%", "12.3%", "50.0%"]
        );
    }
    #[test]
    fn comma_thousands_integer() {
        assert_eq!(
            fmt_labels(&["1000", "10000", "1234567"], ","),
            vec!["1,000", "10,000", "1,234,567"]
        );
    }
    #[test]
    fn comma_dotf_thousands_decimal() {
        assert_eq!(
            fmt_labels(&["1000", "2000", "3000"], ",.0f"),
            vec!["1,000", "2,000", "3,000"]
        );
    }
    #[test]
    fn d_integer_format() {
        assert_eq!(
            fmt_labels(&["42", "1000", "3.7"], "d"),
            vec!["42", "1000", "4"]
        );
    }
    #[test]
    fn si_kilo() {
        // d3 `s` precision is SIGNIFICANT digits: ".2s" of 1200 → "1.2k".
        assert_eq!(fmt_labels(&["1200", "1500"], ".2s"), vec!["1.2k", "1.5k"]);
        // ".3s" keeps three sig figs → "1.20k".
        assert_eq!(fmt_labels(&["1200"], ".3s"), vec!["1.20k"]);
    }
    #[test]
    fn si_trim_megabyte() {
        // The audit-motivating case: "~s" trims insignificant zeros → "1.5M".
        assert_eq!(fmt_labels(&["1500000"], "~s"), vec!["1.5M"]);
    }
    #[test]
    fn ordinal_passthrough() {
        assert_eq!(
            fmt_labels(&["setosa", "versicolor"], ".2f"),
            vec!["setosa", "versicolor"]
        );
    }
    #[test]
    fn no_format_passthrough() {
        let labels = vec!["1".to_string(), "2".to_string()];
        let out = apply_tick_format(labels.clone(), None, None);
        assert_eq!(out, labels);
    }
    #[test]
    fn time_pattern_via_chrono() {
        // 2020-01-01T00:00:00Z = 1577836800000 ms; "%b %Y" → "Jan 2020".
        let out = apply_tick_format(
            vec!["1577836800000".to_string()],
            Some("%b %Y"),
            Some("time"),
        );
        assert_eq!(out, vec!["Jan 2020"]);
    }
    #[test]
    fn time_default_when_no_pattern() {
        // No `%` pattern → default day-spacing temporal formatter (byte-stable).
        let out = apply_tick_format(vec!["1577836800000".to_string()], Some(","), Some("time"));
        assert_eq!(out, vec!["2020-01-01"]);
    }
}

/// Refuse a `FacetSpec::repeat` entry naming a field the facet does not
/// split by: Python names only the facet's own fields, so any other is a
/// malformed wire, not a field to ignore.
fn refuse_malformed_facet_repeats(fspec: &crate::layout::FacetSpec) -> Result<(), RenderError> {
    for entry in &fspec.repeat {
        if let Some(field) =
            entry.fields.iter().find(|f| **f != fspec.field && Some(*f) != fspec.row.as_ref())
        {
            return Err(RenderError::ScaleResolutionFailed(format!(
                "facet repeat for rows marked '{}' names '{field}', which is not a facet field",
                entry.marker
            )));
        }
    }
    Ok(())
}

/// Per row of `batch`, whether it repeats along `field` in every panel: its
/// marker is true in a `repeats` entry naming `field`. All false when no
/// entry names it. A marker the batch lacks, or one that is not Boolean, is a
/// malformed wire and is refused.
fn facet_repeat_mask(
    batch: &RecordBatch,
    repeats: &[crate::layout::FacetRepeat],
    field: &str,
) -> Result<Vec<bool>, RenderError> {
    let mut mask = vec![false; batch.num_rows()];
    for entry in repeats.iter().filter(|e| e.fields.iter().any(|f| f == field)) {
        let column = batch
            .column_by_name(&entry.marker)
            .ok_or_else(|| RenderError::UnknownColumn { name: entry.marker.clone() })?;
        let marker = column.as_any().downcast_ref::<arrow::array::BooleanArray>().ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!(
                "facet repeat marker '{}' must be Boolean",
                entry.marker
            ))
        })?;
        for (row, repeats) in mask.iter_mut().enumerate() {
            *repeats |= marker.is_valid(row) && marker.value(row);
        }
    }
    Ok(mask)
}

/// How many rows of a facet field's `values` belong to no panel: null and
/// not repeated across panels.
fn null_facet_rows(values: &StringArray, repeat: &[bool]) -> usize {
    repeat.iter().enumerate().filter(|&(row, &repeats)| !repeats && values.is_null(row)).count()
}

/// `panel` with `value` written into its Utf8 `field` column on every row
/// `repeated` marks, so a repeated row carries the panel it is drawn in, as
/// every other row in the panel does. Unchanged when no row repeats.
fn write_repeated_facet_value(
    panel: RecordBatch,
    field: &str,
    value: &str,
    repeated: &[bool],
) -> Result<RecordBatch, RenderError> {
    if !repeated.contains(&true) {
        return Ok(panel);
    }
    let index = panel.schema().index_of(field).map_err(|_| RenderError::UnknownColumn { name: field.to_string() })?;
    let values = crate::render::facet_key_column(&panel, field)?;
    let written: StringArray = values
        .iter()
        .zip(repeated)
        .map(|(own, &repeats)| if repeats { Some(value) } else { own })
        .collect();
    let mut columns: Vec<ArrayRef> = panel.columns().to_vec();
    columns[index] = Arc::new(written);
    RecordBatch::try_new(panel.schema(), columns)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("facet repeat for '{field}': {e}")))
}

/// The rows `mask` keeps, each paired with whether it repeats.
fn kept_repeats(mask: &arrow::array::BooleanArray, repeat: &[bool]) -> Vec<bool> {
    repeat.iter().enumerate().filter(|&(row, _)| mask.value(row)).map(|(_, &repeats)| repeats).collect()
}

/// Partition a RecordBatch by a Utf8 field, returning `(value, filtered_batch)`
/// pairs in first-appearance order. Used by facet-before-transform to split the
/// input into per-panel subsets before running transforms.
///
/// The panel values come from the rows that do not `repeat`; a row that
/// repeats joins every panel, carrying that panel's value in `field`. A
/// non-repeated row with a null value joins none.
fn partition_batch_by_field(
    batch: &RecordBatch,
    field: &str,
    repeat: &[bool],
) -> Result<Vec<(String, RecordBatch)>, RenderError> {
    use arrow::array::BooleanArray;
    use arrow::compute::filter_record_batch;
    let arr = crate::render::facet_key_column(batch, field)?;
    // Collect distinct values in first-appearance order.
    let mut order: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (v, &repeats) in arr.iter().zip(repeat) {
        let Some(v) = v.filter(|_| !repeats) else { continue };
        let s = v.to_string();
        if seen.insert(s.clone()) {
            order.push(s);
        }
    }
    let mut result = Vec::with_capacity(order.len());
    for value in order {
        let mask: BooleanArray = arr
            .iter()
            .zip(repeat)
            .map(|(v, &repeats)| Some(repeats || v == Some(value.as_str())))
            .collect();
        let filtered = filter_record_batch(batch, &mask)
            .map_err(|e| RenderError::ScaleResolutionFailed(format!("partition filter: {e}")))?;
        let filtered = write_repeated_facet_value(filtered, field, &value, &kept_repeats(&mask, repeat))?;
        result.push((value, filtered));
    }
    Ok(result)
}

/// Ensure a RecordBatch has a Utf8 column named `field` with the constant `value`.
/// If the column already exists, return the batch unchanged. Otherwise, append a
/// new Utf8 column filled with `value` repeated for every row. This is used to
/// re-inject the facet column into transform outputs that replace the batch
/// entirely (e.g. Smooth, KDE, Histogram).
fn inject_facet_column(batch: &RecordBatch, field: &str, value: &str) -> RecordBatch {
    if batch.column_by_name(field).is_some() {
        return batch.clone();
    }
    let n = batch.num_rows();
    let constant: ArrayRef = Arc::new(StringArray::from(vec![value; n]));
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(field, DataType::Utf8, false)));
    let new_schema = Arc::new(Schema::new(fields));
    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(constant);
    RecordBatch::try_new(new_schema, columns)
        .expect("inject_facet_column: schema + columns must be consistent")
}

/// `(col_val, row_val)` composite key for a single grid-mode partition.
type GridPartitionKey = (String, String);

/// Partition a RecordBatch by two Utf8 fields (col_field, row_field), returning
/// `((col_val, row_val), filtered_batch)` pairs in row-major first-appearance
/// order. Used by facet-before-transform in grid mode.
///
/// Each field comes with its repeat mask: a row repeating along a field joins
/// every value of it, carrying each cell's value, and is partitioned normally
/// along the other. A row that does not repeat along a field and is null
/// there joins no cell. Distinct values along a field come from the kept rows
/// that do not repeat along it.
fn partition_batch_by_two_fields(
    batch: &RecordBatch,
    (col_field, col_repeat): (&str, &[bool]),
    (row_field, row_repeat): (&str, &[bool]),
) -> Result<Vec<(GridPartitionKey, RecordBatch)>, RenderError> {
    use arrow::array::{Array, BooleanArray};
    use arrow::compute::filter_record_batch;

    let col_arr = crate::render::facet_key_column(batch, col_field)?;
    let row_arr = crate::render::facet_key_column(batch, row_field)?;
    let joins_col = |i: usize| col_repeat[i] || !col_arr.is_null(i);
    let joins_row = |i: usize| row_repeat[i] || !row_arr.is_null(i);

    // Collect distinct (row_val, col_val) pairs in row-major first-appearance order.
    let mut row_order: Vec<String> = Vec::new();
    let mut row_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut col_order: Vec<String> = Vec::new();
    let mut col_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for i in 0..batch.num_rows() {
        if !joins_col(i) || !joins_row(i) {
            continue;
        }
        if !row_repeat[i] {
            let rv = row_arr.value(i).to_string();
            if row_seen.insert(rv.clone()) {
                row_order.push(rv);
            }
        }
        if !col_repeat[i] {
            let cv = col_arr.value(i).to_string();
            if col_seen.insert(cv.clone()) {
                col_order.push(cv);
            }
        }
    }

    let mut result = Vec::with_capacity(row_order.len() * col_order.len());
    for row_v in &row_order {
        for col_v in &col_order {
            let mask: BooleanArray = (0..batch.num_rows())
                .map(|i| {
                    let in_col = col_repeat[i] || (!col_arr.is_null(i) && col_arr.value(i) == col_v.as_str());
                    let in_row = row_repeat[i] || (!row_arr.is_null(i) && row_arr.value(i) == row_v.as_str());
                    Some(in_col && in_row)
                })
                .collect();
            // Skip (row_v, col_v) pairs that have no rows in this batch (sparse cross-products).
            if !mask.iter().any(|v| v == Some(true)) {
                continue;
            }
            let filtered = filter_record_batch(batch, &mask).map_err(|e| {
                RenderError::ScaleResolutionFailed(format!("partition filter: {e}"))
            })?;
            let filtered = write_repeated_facet_value(filtered, col_field, col_v, &kept_repeats(&mask, col_repeat))?;
            let filtered = write_repeated_facet_value(filtered, row_field, row_v, &kept_repeats(&mask, row_repeat))?;
            result.push(((col_v.clone(), row_v.clone()), filtered));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;

// ── Grid-mode composite-key partitioning unit tests ──────────────────────────

#[cfg(test)]
mod grid_partition_tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// No row repeats along either field: every pre-repeat partition.
    fn no_repeat(batch: &RecordBatch) -> Vec<bool> {
        vec![false; batch.num_rows()]
    }

    fn utf8(batch: &RecordBatch, field: &str) -> Vec<Option<String>> {
        crate::render::facet_key_column(batch, field).expect("Utf8 facet column").iter().map(|v| v.map(str::to_string)).collect()
    }

    /// A wrap partition over `c = [p, q, null, null, null]` whose rows 2 and 3
    /// repeat: each panel holds its own row plus both repeated rows, written
    /// with the panel's value; row 4 (null, not repeated) is in no panel and
    /// is the one counted. The rejected reading partitions the repeated rows
    /// by their null value, so they reach no panel.
    #[test]
    fn a_repeated_row_joins_every_wrap_panel_with_its_value() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("c", DataType::Utf8, true),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("p"), Some("q"), None, None, None])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap();
        let repeat = [false, false, true, true, false];
        let partitions = partition_batch_by_field(&batch, "c", &repeat).unwrap();
        let values: Vec<&str> = partitions.iter().map(|(v, _)| v.as_str()).collect();
        assert_eq!(values, ["p", "q"]);
        for (value, panel) in &partitions {
            assert_eq!(utf8(panel, "c"), vec![Some(value.clone()); 3]);
        }
        let v = |panel: &RecordBatch| {
            panel.column_by_name("v").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
        };
        assert_eq!(v(&partitions[0].1), [1.0, 3.0, 4.0]);
        assert_eq!(v(&partitions[1].1), [2.0, 3.0, 4.0]);
        assert_eq!(null_facet_rows(&crate::render::facet_key_column(&batch, "c").unwrap(), &repeat), 1);
    }

    /// A grid whose second frame lacks only the row field: its rows (2, 3)
    /// partition by their column value and repeat down every row value. The
    /// rejected reading drops them for their null row value.
    #[test]
    fn a_row_lacking_only_the_row_field_repeats_down_its_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col_cat", DataType::Utf8, true),
            Field::new("row_cat", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("c1"), Some("c2"), Some("c1"), Some("c2")])),
                Arc::new(StringArray::from(vec![Some("r1"), Some("r2"), None, None])),
            ],
        )
        .unwrap();
        let row_repeat = [false, false, true, true];
        let partitions =
            partition_batch_by_two_fields(&batch, ("col_cat", &no_repeat(&batch)), ("row_cat", &row_repeat)).unwrap();
        let cells: Vec<(&str, &str, usize)> =
            partitions.iter().map(|((c, r), b)| (c.as_str(), r.as_str(), b.num_rows())).collect();
        assert_eq!(cells, [("c1", "r1", 2), ("c2", "r1", 1), ("c1", "r2", 1), ("c2", "r2", 2)]);
        let (_, r2c2) = &partitions[3];
        assert_eq!(utf8(r2c2, "row_cat"), [Some("r2".to_string()), Some("r2".to_string())]);
        assert_eq!(utf8(r2c2, "col_cat"), [Some("c2".to_string()), Some("c2".to_string())]);
    }

    /// Build a 3×3 grid batch: 3 row categories × 3 col categories, 2 data rows each.
    fn grid_3x3_batch() -> RecordBatch {
        let mut col_vals: Vec<&str> = Vec::new();
        let mut row_vals: Vec<&str> = Vec::new();
        let cols = ["c1", "c2", "c3"];
        let rows = ["r1", "r2", "r3"];
        for &r in &rows {
            for &c in &cols {
                // 2 data rows per (row, col) cell.
                col_vals.push(c);
                row_vals.push(r);
                col_vals.push(c);
                row_vals.push(r);
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("col_cat", DataType::Utf8, false),
            Field::new("row_cat", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(col_vals)),
                Arc::new(StringArray::from(row_vals)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn partition_by_two_fields_produces_correct_count() {
        let batch = grid_3x3_batch();
        let partitions = partition_batch_by_two_fields(&batch, ("col_cat", &no_repeat(&batch)), ("row_cat", &no_repeat(&batch))).unwrap();
        // 3 row values × 3 col values = 9 partitions.
        assert_eq!(
            partitions.len(),
            9,
            "Expected 9 composite partitions for a 3×3 grid; got {}",
            partitions.len()
        );
    }

    #[test]
    fn partition_by_two_fields_each_partition_has_correct_row_count() {
        let batch = grid_3x3_batch();
        let partitions = partition_batch_by_two_fields(&batch, ("col_cat", &no_repeat(&batch)), ("row_cat", &no_repeat(&batch))).unwrap();
        for ((col_v, row_v), part_batch) in &partitions {
            assert_eq!(
                part_batch.num_rows(),
                2,
                "Partition ({col_v}, {row_v}) must have 2 rows; got {}",
                part_batch.num_rows()
            );
        }
    }

    #[test]
    fn partition_by_two_fields_row_major_order() {
        let batch = grid_3x3_batch();
        let partitions = partition_batch_by_two_fields(&batch, ("col_cat", &no_repeat(&batch)), ("row_cat", &no_repeat(&batch))).unwrap();
        // Row-major: (r1,c1), (r1,c2), (r1,c3), (r2,c1), ...
        let keys: Vec<(&str, &str)> = partitions
            .iter()
            .map(|((cv, rv), _)| (cv.as_str(), rv.as_str()))
            .collect();
        assert_eq!(keys[0], ("c1", "r1"), "First partition should be (c1, r1)");
        assert_eq!(keys[1], ("c2", "r1"), "Second partition should be (c2, r1)");
        assert_eq!(keys[2], ("c3", "r1"), "Third partition should be (c3, r1)");
        assert_eq!(keys[3], ("c1", "r2"), "Fourth partition should be (c1, r2)");
        assert_eq!(keys[8], ("c3", "r3"), "Last partition should be (c3, r3)");
    }

    // ── facet_key_column contract ──────────────────────────────

    /// A minimal batch with one Utf8 column `"cat"` and one Float64 column `"v"`.
    /// Used by all three `facet_key_column` tests below.
    fn facet_key_column_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    /// Happy path: `facet_key_column` on a present Utf8 column returns the
    /// underlying `StringArray` with the correct values.
    #[test]
    fn facet_key_column_returns_correct_utf8_array() {
        let batch = facet_key_column_batch();
        let arr = crate::render::facet_key_column(&batch, "cat").expect("cat is a Utf8 column");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), "a");
        assert_eq!(arr.value(1), "b");
        assert_eq!(arr.value(2), "a");
    }

    /// Missing column: `facet_key_column` must return `RenderError::UnknownColumn`
    /// (not panic) when the field is absent from the batch.
    #[test]
    fn facet_key_column_missing_column_errors_unknown_column() {
        let batch = facet_key_column_batch();
        let err = crate::render::facet_key_column(&batch, "nonexistent").unwrap_err();
        assert!(
            matches!(err, RenderError::UnknownColumn { .. }),
            "expected UnknownColumn, got: {err:?}"
        );
    }

    /// A numeric facet column keys through the same ordinal-category
    /// reader a dodge/stack `by=` column reads, rather than refusing —
    /// `facet_key_column` on a Float64 column reads its labels.
    #[test]
    fn facet_key_column_numeric_column_reads_ordinal_labels() {
        let batch = facet_key_column_batch();
        let arr = crate::render::facet_key_column(&batch, "v").expect("Float64 facets through the ordinal reader");
        assert_eq!(arr.value(0), "1");
        assert_eq!(arr.value(1), "2");
        assert_eq!(arr.value(2), "3");
    }

    /// An all-null `Null`-typed column (no further dtype to read) also
    /// keys through the ordinal reader rather than refusing; every row
    /// reads as a null facet key, the same shape the drop-and-warn route
    /// already handles for a Utf8 column.
    #[test]
    fn facet_key_column_null_typed_column_reads_as_all_null() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Null, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arrow::array::NullArray::new(2))]).unwrap();
        let arr = crate::render::facet_key_column(&batch, "n").expect("Null-typed facets through the ordinal reader");
        assert_eq!(arr.len(), 2);
        assert!(arr.is_null(0) && arr.is_null(1));
    }

    /// A dtype the ordinal reader itself cannot stringify (e.g. `Timestamp`)
    /// refuses, naming the dtype, the accepted dtypes and the cast remedy.
    #[test]
    fn facet_key_column_unsupported_dtype_names_the_dtype_and_remedy() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "t",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::TimestampMillisecondArray::from(vec![Some(1_000i64)]))],
        )
        .unwrap();
        let err = crate::render::facet_key_column(&batch, "t").unwrap_err();
        match &err {
            RenderError::UnsupportedDtype { field, dtype, context } => {
                assert_eq!(field, "t");
                assert!(dtype.contains("Timestamp"), "{dtype}");
                assert!(dtype.contains("cast 't' to a string"), "{dtype}");
                assert_eq!(*context, Some("facet"));
            }
            other => panic!("expected UnsupportedDtype, got: {other:?}"),
        }
    }
}
