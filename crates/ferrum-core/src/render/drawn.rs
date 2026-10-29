//! The one rule for whether a layer draws a non-positional channel, and
//! whether a conditional branch on that channel is read.
//!
//! A layer carries a binding, own or inherited, on every non-positional
//! channel its completion fills in, but its mark may not draw every one it
//! carries: a text declines an inherited color, a raster image paints only
//! the field its cells hold, a line with its own (or also-inherited) width
//! yields an inherited one, and a segment with no arrowhead has nothing to
//! fill. [`DrawFacts`] is the one borrowed view of a layer's drawing facts,
//! mirroring [`crate::render::marks::image::image_draws_color`]'s shape:
//! built from either side (a prepared layer's own fields, via
//! [`crate::render::prepare::LayerPrepared::draw_facts`], or a raw spec
//! layer plus its chart's encoding, via [`DrawFacts::of_layer`]).
//!
//! [`DrawFacts::draws`] answers the bound question: a real binding is in
//! hand, so a raster's color is compared against the field its cells hold,
//! a declined inherited color and a yielded width are ruled out, before the
//! mark-and-style question is even asked. [`DrawFacts::by_mark_and_style`]
//! answers that binding-independent half — the mark table, the contrast-ink
//! sentinel, and the arrowless segment's fill — and is also what decides
//! whether a conditional branch on the channel is read
//! ([`DrawFacts::reads_conditional`]), since every layer that declares a
//! conditional carries its own copy of it regardless of what it binds or
//! inherits.

use serde::{Deserialize, Serialize};

use crate::spec::encoding::{Encoding, EncodingSpec};
use crate::spec::layer::InheritedChannel;
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;

use super::prepare::YieldedWidths;

/// Why a layer that carries a binding on a non-positional channel does not
/// draw it, named for the message [`crate::render::RenderWarning::ChannelNotDrawn`]
/// builds from it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Undrawn {
    /// The mark never draws this channel.
    NotDrawn,
    /// A text declines an inherited color it does not bind itself
    /// ([`Mark::draws_inherited`]).
    DeclinedInherited,
    /// A contrast-ink text binds color to read it, not to paint a hue from
    /// it ([`crate::render::marks::text::reads_color_for_contrast`]).
    ContrastInk,
    /// A raster image's cells hold another field than this color binding
    /// names ([`crate::render::marks::image::image_draws_color`]).
    RasterCells { cells: String },
    /// A line's own width (`winner`, its value or field as text) outranks
    /// another one: its own, when `inherited` is false, or one it also
    /// only inherits, when `inherited` is true (between two inherited
    /// widths, `stroke_width` outranks `size`).
    YieldedWidth { winner: InheritedChannel, value: String, inherited: bool },
    /// A segment with no arrowhead has nothing to fill, so its
    /// `fill_opacity` paints nothing.
    ArrowlessSegmentFill,
}

impl Undrawn {
    /// Whether the layer's builder still reads the column even though this
    /// reason says the channel paints no scale value from it: true only for
    /// [`Self::ContrastInk`], the one case the builder reads a column for a
    /// purpose other than painting it from a scale
    /// (`crate::render::marks::text::reads_color_for_contrast`'s doc).
    /// Asked by [`DrawFacts::reads`] (the bound question) and
    /// [`DrawFacts::reads_conditional`] (the conditional one), so both
    /// treat a contrast-ink read as read.
    pub(in crate::render) fn still_read(&self) -> bool {
        matches!(self, Undrawn::ContrastInk)
    }
}

/// The facts [`DrawFacts::draws`] and [`DrawFacts::by_mark_and_style`] need
/// to decide whether a layer draws a non-positional channel: borrowed, built
/// from either side of a layer. [`crate::render::prepare::LayerPrepared::draw_facts`]
/// builds it from a prepared layer's own fields (`encoding` and `parent` the
/// same completed encoding, since its completion already folded the
/// chart-level fallback in); [`Self::of_layer`] builds it from a raw spec
/// layer's own encoding and its chart's, so the field lookup that finds a
/// binding to ask about keeps its borrowed lifetime instead of a
/// short-lived completed copy.
pub(in crate::render) struct DrawFacts<'a> {
    pub(in crate::render) mark: Mark,
    /// This layer's own encoding (unmerged for a raw layer; the completed
    /// one for a prepared layer).
    pub(in crate::render) encoding: &'a Encoding,
    /// The chart-level encoding a binding this layer does not own falls
    /// back to (the same value as `encoding` for a prepared layer, whose
    /// completion already folded it in).
    pub(in crate::render) parent: &'a Encoding,
    pub(in crate::render) mark_style: Option<&'a MarkKwargsSpec>,
    /// Whether this layer binds `color` itself, as
    /// [`crate::spec::layer::Layer::owns`] answers: the one fact
    /// [`Mark::draws_inherited`]'s declined-color rule needs that is not
    /// recoverable from `encoding`/`parent` alone (an inherited binding a
    /// layer's own encoding also carries, by wire convention, is still not
    /// owned).
    pub(in crate::render) color_is_own: bool,
    pub(in crate::render) size_is_own: bool,
    pub(in crate::render) stroke_width_is_own: bool,
    /// The field this layer's cells hold, when it is a raster image reading
    /// one ([`crate::render::marks::image::raster_value_field`]).
    pub(in crate::render) cells: Option<&'a str>,
}

impl<'a> DrawFacts<'a> {
    /// This layer's facts built from the raw spec side: its own encoding,
    /// the chart's as the fallback, and `cells` read from whatever batch it
    /// draws. The twin of [`crate::render::prepare::LayerPrepared::draw_facts`],
    /// read by [`crate::render::scale_resolve::domain::aux_reads`].
    pub(in crate::render) fn of_layer(
        layer: &'a crate::spec::layer::Layer,
        chart_encoding: &'a Encoding,
        cells: Option<&'a str>,
    ) -> Self {
        DrawFacts {
            mark: layer.mark,
            encoding: &layer.encoding,
            parent: chart_encoding,
            mark_style: layer.mark_style.as_ref(),
            color_is_own: layer.owns(InheritedChannel::Color),
            size_is_own: layer.owns(InheritedChannel::Size),
            stroke_width_is_own: layer.owns(InheritedChannel::StrokeWidth),
            cells,
        }
    }

    fn size(&self) -> Option<&'a EncodingSpec> {
        self.encoding.size.as_ref().or(self.parent.size.as_ref())
    }

    fn stroke_width(&self) -> Option<&'a EncodingSpec> {
        self.encoding.stroke_width.as_ref().or(self.parent.stroke_width.as_ref())
    }

    /// The inherited width bindings this layer gives up to a width it owns:
    /// a line's `size` and `stroke_width` both set its width, and an
    /// inherited binding of either yields to a width the layer binds
    /// itself (own encoding or own constant kwarg), on the same channel or
    /// the other one. Between two inherited ones, `stroke_width` (pixels)
    /// is kept over `size` (a taper). Every other mark yields nothing.
    pub(in crate::render) fn yielded_widths(&self) -> YieldedWidths {
        if self.mark != Mark::Line {
            return YieldedWidths::default();
        }
        let owns_size = self.size_is_own || self.mark_style.is_some_and(|m| m.size.is_some());
        let owns_stroke_width = self.stroke_width_is_own || self.mark_style.is_some_and(|m| m.stroke_width.is_some());
        let size_inherited = self.size().is_some() && !self.size_is_own;
        let stroke_width_inherited = self.stroke_width().is_some() && !self.stroke_width_is_own;
        YieldedWidths {
            size: size_inherited && (owns_size || owns_stroke_width || stroke_width_inherited),
            stroke_width: stroke_width_inherited && (owns_size || owns_stroke_width),
        }
    }

    /// The width this layer's own or also-inherited `stroke_width`/`size`
    /// outranks another one with, named for [`Undrawn::YieldedWidth`]'s
    /// message: a constant mark kwarg renders as its number, a field as its
    /// name. `stroke_width` is reported when the layer owns it, or, owning
    /// neither, when both are inherited (which `stroke_width` wins, and
    /// `inherited` is then true, since that winner is not owned either).
    fn yield_reason(&self) -> Undrawn {
        let stroke_width_literal = self.mark_style.and_then(|m| m.stroke_width);
        let size_literal = self.mark_style.and_then(|m| m.size);
        let owns_stroke_width = self.stroke_width_is_own || stroke_width_literal.is_some();
        let owns_size = self.size_is_own || size_literal.is_some();
        let (winner, literal, field, inherited) = if owns_stroke_width {
            (InheritedChannel::StrokeWidth, stroke_width_literal, self.stroke_width(), false)
        } else if owns_size {
            (InheritedChannel::Size, size_literal, self.size(), false)
        } else {
            (InheritedChannel::StrokeWidth, None, self.stroke_width(), true)
        };
        let value = literal
            .map(|v| v.to_string())
            .unwrap_or_else(|| field.map(|binding| format!("'{}'", binding.field)).unwrap_or_default());
        Undrawn::YieldedWidth { winner, value, inherited }
    }

    fn generic(&self, channel: InheritedChannel) -> Result<(), Undrawn> {
        if self.mark.draws_channel(channel) {
            Ok(())
        } else {
            Err(Undrawn::NotDrawn)
        }
    }

    /// Whether this layer's own segment has no arrowhead, so its
    /// `fill_opacity` has nothing to fill.
    fn arrowless_segment(&self) -> bool {
        self.mark == Mark::Segment && !self.mark_style.and_then(|m| m.arrow).unwrap_or(false)
    }

    /// The binding-independent half of whether this layer's mark draws
    /// `channel`: the mark table ([`Mark::draws_channel`]), refined only by
    /// the one style fact that removes a read a mark would otherwise have —
    /// a contrast-ink text's sentinel fill, and a segment with no
    /// arrowhead. Neither refinement needs a bound field in hand, which is
    /// why this is also the answer for a conditional branch
    /// ([`Self::reads_conditional`]): every layer that declares a
    /// conditional carries its own copy of it, whatever bindings that layer
    /// owns or inherits (`scene_build::layer_conditionals`), so a binding
    /// fact — an inherited color this layer does not own, a width it
    /// yields to another one, a raster's cell field — cannot decide
    /// whether the conditional is read.
    fn by_mark_and_style(&self, channel: InheritedChannel) -> Result<(), Undrawn> {
        match channel {
            InheritedChannel::Color if crate::render::marks::text::reads_color_for_contrast(self.mark, self.mark_style) => {
                Err(Undrawn::ContrastInk)
            }
            InheritedChannel::FillOpacity if self.arrowless_segment() => Err(Undrawn::ArrowlessSegmentFill),
            _ => self.generic(channel),
        }
    }

    /// Color's binding fact: a raster image draws it only through its cells
    /// ([`crate::render::marks::image::image_draws_color`]), never through
    /// [`Self::by_mark_and_style`]'s mark table (which excludes `Image`
    /// outright).
    fn draws_raster_color(&self) -> Result<(), Undrawn> {
        if crate::render::marks::image::image_draws_color(self.mark, self.encoding, self.parent, self.cells) {
            Ok(())
        } else {
            match self.cells {
                Some(cells) => Err(Undrawn::RasterCells { cells: cells.to_string() }),
                None => Err(Undrawn::NotDrawn),
            }
        }
    }

    /// Whether this layer's mark draws `channel`, given a binding on it is
    /// in hand: the binding facts first (a raster's cells, a declined
    /// inherited color, a yielded width), then
    /// [`Self::by_mark_and_style`]. The one rule
    /// [`crate::render::prepare::LayerPrepared::draws_binding`] (through
    /// [`Self::reads`]), [`crate::render::scale_resolve::domain::aux_reads`],
    /// `establishing_binding`'s unlayered branch and
    /// `undrawn_channel_warnings` all ask.
    pub(in crate::render) fn draws(&self, channel: InheritedChannel) -> Result<(), Undrawn> {
        use crate::spec::layer::InheritedChannel as C;
        match channel {
            C::Color if self.mark == Mark::Image => self.draws_raster_color(),
            C::Color => {
                if !self.color_is_own && !self.mark.draws_inherited(C::Color) {
                    return Err(Undrawn::DeclinedInherited);
                }
                self.by_mark_and_style(channel)
            }
            C::Size => {
                if self.yielded_widths().size {
                    return Err(self.yield_reason());
                }
                self.by_mark_and_style(channel)
            }
            C::StrokeWidth => {
                if self.yielded_widths().stroke_width {
                    return Err(self.yield_reason());
                }
                self.by_mark_and_style(channel)
            }
            other => self.by_mark_and_style(other),
        }
    }

    /// Whether this layer's mark reads `channel`'s column at all, even when
    /// it paints no scale value from it: [`Self::draws`] is `Ok`, or its
    /// refusal [`Undrawn::still_read`]s. [`crate::render::prepare::LayerPrepared::draws_binding`]
    /// asks this, so `drawn_encoding`'s meaning (every column a layer's
    /// builder reads) is unchanged by routing through the rule.
    pub(in crate::render) fn reads(&self, channel: InheritedChannel) -> bool {
        match self.draws(channel) {
            Ok(()) => true,
            Err(reason) => reason.still_read(),
        }
    }

    /// Whether a conditional branch on `channel` is read: the
    /// binding-independent half ([`Self::by_mark_and_style`]) is `Ok`, or
    /// [`Undrawn::still_read`]s. Asked by the conditional-branch column-gate
    /// filter — a conditional is not itself a bound field, so the binding
    /// facts [`Self::draws`] applies first (an image's cells, a declined
    /// color, a yielded width) do not apply to it.
    pub(in crate::render) fn reads_conditional(&self, channel: InheritedChannel) -> bool {
        match self.by_mark_and_style(channel) {
            Ok(()) => true,
            Err(reason) => reason.still_read(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts<'a>(mark: Mark, encoding: &'a Encoding, mark_style: Option<&'a MarkKwargsSpec>) -> DrawFacts<'a> {
        DrawFacts {
            mark,
            encoding,
            parent: encoding,
            mark_style,
            color_is_own: encoding.color.is_some(),
            size_is_own: encoding.size.is_some(),
            stroke_width_is_own: encoding.stroke_width.is_some(),
            cells: None,
        }
    }

    fn field(name: &str) -> Option<EncodingSpec> {
        Some(EncodingSpec { field: name.to_string(), ..Default::default() })
    }

    #[test]
    fn a_line_with_an_inherited_size_and_its_own_stroke_width_yields_size() {
        let encoding = Encoding { size: field("s"), ..Default::default() };
        let style = MarkKwargsSpec { stroke_width: Some(3.0), ..Default::default() };
        let mut facts = facts(Mark::Line, &encoding, Some(&style));
        facts.size_is_own = false;
        match facts.draws(InheritedChannel::Size) {
            Err(Undrawn::YieldedWidth { winner, value, inherited }) => {
                assert_eq!(winner, InheritedChannel::StrokeWidth);
                assert_eq!(value, "3");
                assert!(!inherited, "the stroke_width is the line's own");
            }
            other => panic!("expected a yielded-width refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_line_with_its_own_size_field_and_no_width_draws_size() {
        let encoding = Encoding { size: field("s"), ..Default::default() };
        let facts = facts(Mark::Line, &encoding, None);
        assert_eq!(facts.draws(InheritedChannel::Size), Ok(()));
    }

    #[test]
    fn a_line_inheriting_both_widths_yields_size_to_an_also_inherited_stroke_width() {
        let encoding = Encoding { size: field("s"), stroke_width: field("w"), ..Default::default() };
        let mut facts = facts(Mark::Line, &encoding, None);
        facts.size_is_own = false;
        facts.stroke_width_is_own = false;
        match facts.draws(InheritedChannel::Size) {
            Err(Undrawn::YieldedWidth { winner, value, inherited }) => {
                assert_eq!(winner, InheritedChannel::StrokeWidth);
                assert_eq!(value, "'w'");
                assert!(inherited, "neither width is owned");
            }
            other => panic!("expected a yielded-width refusal, got {other:?}"),
        }
        assert_eq!(facts.draws(InheritedChannel::StrokeWidth), Ok(()));
    }

    #[test]
    fn an_arrowless_segment_does_not_draw_fill_opacity() {
        let encoding = Encoding { fill_opacity: field("o"), ..Default::default() };
        let facts = facts(Mark::Segment, &encoding, None);
        assert_eq!(facts.draws(InheritedChannel::FillOpacity), Err(Undrawn::ArrowlessSegmentFill));
        assert!(!facts.reads_conditional(InheritedChannel::FillOpacity), "no read, bound or conditional");
    }

    #[test]
    fn an_arrow_segment_draws_fill_opacity() {
        let encoding = Encoding { fill_opacity: field("o"), ..Default::default() };
        let style = MarkKwargsSpec { arrow: Some(true), ..Default::default() };
        let facts = facts(Mark::Segment, &encoding, Some(&style));
        assert_eq!(facts.draws(InheritedChannel::FillOpacity), Ok(()));
        assert!(facts.reads_conditional(InheritedChannel::FillOpacity));
    }

    #[test]
    fn a_contrast_ink_text_does_not_draw_color_but_still_reads_it() {
        let encoding = Encoding { color: field("v"), ..Default::default() };
        let style = MarkKwargsSpec { fill: Some(crate::render::draw::CONTRAST_SENTINEL.to_string()), ..Default::default() };
        let facts = facts(Mark::Text, &encoding, Some(&style));
        assert_eq!(facts.draws(InheritedChannel::Color), Err(Undrawn::ContrastInk));
        assert!(facts.reads(InheritedChannel::Color), "the builder still reads the column");
        assert!(facts.reads_conditional(InheritedChannel::Color), "a conditional color is still read");
    }

    #[test]
    fn a_text_declines_an_inherited_color_it_does_not_own_but_its_own_conditional_is_read() {
        let own = Encoding::default();
        let parent = Encoding { color: field("g"), ..Default::default() };
        let facts = DrawFacts {
            mark: Mark::Text,
            encoding: &own,
            parent: &parent,
            mark_style: None,
            color_is_own: false,
            size_is_own: false,
            stroke_width_is_own: false,
            cells: None,
        };
        assert_eq!(facts.draws(InheritedChannel::Color), Err(Undrawn::DeclinedInherited));
        assert!(!facts.reads(InheritedChannel::Color), "no bound read when it is declined");
        assert!(facts.reads_conditional(InheritedChannel::Color), "a conditional does not carry the declined fact");
    }

    #[test]
    fn a_raster_image_draws_color_only_through_its_own_cells_and_never_reads_a_conditional_one() {
        let encoding = Encoding { color: field("v"), ..Default::default() };
        let mut facts = facts(Mark::Image, &encoding, None);
        facts.cells = Some("other");
        assert_eq!(facts.draws(InheritedChannel::Color), Err(Undrawn::RasterCells { cells: "other".to_string() }));
        assert!(
            !facts.reads_conditional(InheritedChannel::Color),
            "the mark table alone (no color for an image) judges a conditional, not the cell field"
        );
        facts.cells = Some("v");
        assert_eq!(facts.draws(InheritedChannel::Color), Ok(()));
    }
}
