use serde::{Deserialize, Serialize};
use std::fmt;

use crate::spec::layer::PositionalChannel;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mark {
    Point,
    Line,
    Bar,
    Area,
    Rule,
    Text,
    Tick,
    Rect,
    Polygon,
    Image,
    Ribbon,
    Segment,
    Arc,
    Label,
    Geoshape,
}

/// Single source of truth for the primitive `Mark` variants. Mirrors the
/// `for_each_transform!` pattern in `transform/core.rs` — the name mappings
/// (`as_str`, `from_str`) drive off this table. The lowercase form is encoded
/// as the second column so it doubles as both the serde tag and the
/// `render::marks` submodule name. `render::draw`'s build dispatch spells its
/// arms out explicitly instead; see the rationale at that match.
#[macro_export]
macro_rules! for_each_mark {
    ($mac:ident) => {
        $mac! {
            Point    => point,
            Line     => line,
            Bar      => bar,
            Area     => area,
            Rule     => rule,
            Text     => text,
            Tick     => tick,
            Rect     => rect,
            Polygon  => polygon,
            Image    => image,
            Ribbon   => ribbon,
            Segment  => segment,
            Arc      => arc,
            Label    => label,
            Geoshape => geoshape,
        }
    };
}

impl Mark {
    pub fn as_str(&self) -> &'static str {
        macro_rules! arm {
            ($($V:ident => $name:ident,)*) => {
                match self { $( Mark::$V => stringify!($name), )* }
            };
        }
        for_each_mark!(arm)
    }

    /// How this mark draws `size`: the one table every question about which
    /// marks draw `size` reads, at render and (through
    /// [`marks_drawing_constant_size`]) at construction.
    pub(crate) fn size_drawing(self) -> SizeDrawing {
        match self {
            Mark::Point | Mark::Line => SizeDrawing::ConstantOrEncoded,
            Mark::Text => SizeDrawing::EncodedOnly,
            Mark::Bar | Mark::Area | Mark::Rule | Mark::Tick | Mark::Rect | Mark::Polygon | Mark::Image
            | Mark::Ribbon | Mark::Segment | Mark::Arc | Mark::Label | Mark::Geoshape => SizeDrawing::Never,
        }
    }

    /// Whether this mark draws an encoded `size` channel: a point's area, a
    /// line's dot-diameter width, a text's font size. Every other mark
    /// carries an inherited `size` binding without drawing it.
    pub(crate) fn draws_size_channel(self) -> bool {
        self.size_drawing() != SizeDrawing::Never
    }

    /// Whether this mark draws an encoded `stroke_width` channel: a per-row
    /// stroke width mapped through the stroke-width scale
    /// (`render::draw::stroke_width_column`'s readers). Every other mark
    /// carries an inherited `stroke_width` binding without drawing it.
    pub(crate) fn draws_stroke_width_channel(self) -> bool {
        match self {
            Mark::Point | Mark::Line | Mark::Bar | Mark::Rule | Mark::Tick | Mark::Rect | Mark::Segment => true,
            Mark::Area | Mark::Text | Mark::Polygon | Mark::Image | Mark::Ribbon | Mark::Arc | Mark::Label
            | Mark::Geoshape => false,
        }
    }

    /// Whether this mark draws the opacity-family `channel` bound to a field,
    /// read from what each mark's builder takes from that column: `opacity`
    /// fades every mark that reads it (bar through its fill), `fill_opacity`
    /// the marks that paint a fill with it, `stroke_opacity` the marks that
    /// paint a stroke with it. A segment's `fill_opacity` fills its
    /// arrowheads, so it draws only with `arrow=True`
    /// (`render::prepare::LayerPrepared::drawn_opacity_field`). Every other
    /// mark carries an inherited binding of the channel without drawing it.
    pub(crate) fn draws_opacity_channel(self, channel: crate::render::scale_resolve::OpacityChannel) -> bool {
        use crate::render::scale_resolve::OpacityChannel::{Fill, Opacity, Stroke};
        match self {
            Mark::Point => true,
            Mark::Bar => true,
            Mark::Rect | Mark::Area => matches!(channel, Opacity | Fill),
            Mark::Line | Mark::Rule | Mark::Tick => matches!(channel, Opacity | Stroke),
            Mark::Segment => true,
            Mark::Text | Mark::Arc | Mark::Polygon => channel == Opacity,
            Mark::Image | Mark::Ribbon | Mark::Label | Mark::Geoshape => false,
        }
    }

    /// Whether this mark draws the non-positional `channel` when it is bound to
    /// a field, read from what each mark's builder takes from that column:
    /// the one table every question about which marks draw a non-positional
    /// channel reads. `size`, `stroke_width` and the opacity family answer
    /// from their own tables ([`Self::draws_size_channel`],
    /// [`Self::draws_stroke_width_channel`], [`Self::draws_opacity_channel`]).
    /// Of the rest: `color` is drawn by every mark that paints from a color
    /// scale (not an image, a label or a geoshape); `text` by the text, label
    /// and heatmap rect builders; `shape` by a point; `angle` by a point, a
    /// bar and a text; `stroke_dash` by the marks reading a dash column
    /// (point, bar, rect, line, ribbon, rule); `url` by an image; and the
    /// tooltip family, `href`, `description` and `key` by every mark. Every
    /// other mark carries a binding of the channel without drawing it.
    pub(crate) fn draws_channel(self, channel: crate::spec::layer::InheritedChannel) -> bool {
        use crate::render::scale_resolve::OpacityChannel;
        use crate::spec::layer::InheritedChannel as C;
        match channel {
            C::Size => self.draws_size_channel(),
            C::StrokeWidth => self.draws_stroke_width_channel(),
            C::Opacity => self.draws_opacity_channel(OpacityChannel::Opacity),
            C::FillOpacity => self.draws_opacity_channel(OpacityChannel::Fill),
            C::StrokeOpacity => self.draws_opacity_channel(OpacityChannel::Stroke),
            C::Color => !matches!(self, Mark::Image | Mark::Label | Mark::Geoshape),
            C::Text => matches!(self, Mark::Text | Mark::Label | Mark::Rect),
            C::Shape => self == Mark::Point,
            C::Angle => matches!(self, Mark::Point | Mark::Bar | Mark::Text),
            C::StrokeDash => {
                matches!(self, Mark::Point | Mark::Bar | Mark::Rect | Mark::Line | Mark::Ribbon | Mark::Rule)
            }
            C::Url => self == Mark::Image,
            C::Tooltip | C::TooltipFields | C::Href | C::Description | C::Key => true,
        }
    }

    /// Whether a layer of this mark takes an INHERITED binding of `channel`
    /// as a binding, so it draws it where [`Self::draws_channel`] says the
    /// mark draws the channel. False only for a text's color: a text draws
    /// only a color it binds itself, so its labels stay legible on the marks
    /// they annotate, and an inherited color reaches it only as a grouping
    /// key, the layer's `group` ([`crate::spec::encoding::Encoding::group`]),
    /// which groups its statistic, dodge and stack as the color groups the
    /// marks beside it. Applied where a layer is completed from its parent
    /// ([`crate::spec::layer::Layer::completed_encoding`]).
    pub(crate) fn draws_inherited(self, channel: crate::spec::layer::InheritedChannel) -> bool {
        !matches!((self, channel), (Mark::Text, crate::spec::layer::InheritedChannel::Color))
    }

    /// Whether this mark draws the span channels `x2`/`y2`, read from what
    /// each mark's builder takes: a bar's, rect's, area's and ribbon's other
    /// edge, a rule's and segment's other end. Every other mark carries an
    /// inherited span binding without drawing it.
    pub(crate) fn draws_span_channels(self) -> bool {
        match self {
            Mark::Bar | Mark::Rect | Mark::Rule | Mark::Segment | Mark::Area | Mark::Ribbon => true,
            Mark::Point | Mark::Line | Mark::Text | Mark::Tick | Mark::Polygon | Mark::Image | Mark::Arc
            | Mark::Label | Mark::Geoshape => false,
        }
    }

    /// Whether this mark draws the positional channels `x`/`y`. Every mark
    /// does but a geoshape, which draws its geometry column through its
    /// projection. An arc draws them as its polar theta and radius.
    pub(crate) fn draws_position_channels(self) -> bool {
        match self {
            Mark::Geoshape => false,
            Mark::Point | Mark::Line | Mark::Bar | Mark::Area | Mark::Rule | Mark::Text | Mark::Tick
            | Mark::Rect | Mark::Polygon | Mark::Image | Mark::Ribbon | Mark::Segment | Mark::Arc
            | Mark::Label => true,
        }
    }

    /// Whether this mark stacks the theta channel of a `CoordPolar` itself:
    /// an arc always sweeps its theta into a pie, and a text with
    /// `Theta(stack=...)` stacks its labels into that sweep
    /// (`render::marks::text::polar_label_placement`). Their theta `stack` is
    /// that sweep's, never a value axis for the position adjustment to stack
    /// (`render::position::position_encoding`).
    pub(crate) fn stacks_polar_theta(self) -> bool {
        matches!(self, Mark::Arc | Mark::Text)
    }

    /// Which of the `offered` positional bindings a layer of this mark takes
    /// from its parent, given the positional channels it writes itself
    /// (`own`): the channels it draws. A span channel (`x2`/`y2`) only on a
    /// mark that draws spans ([`Self::draws_span_channels`]). `x`/`y` on every
    /// mark that draws them ([`Self::draws_position_channels`]), except that a
    /// rule or tick declaring exactly one of `x`/`y`, with no span written or
    /// taken, is a one-axis reference line or rug and draws only the channel
    /// it declared. A rule assembling a ranged or diagonal
    /// shape partly from inherited span channels keeps assembling it. The
    /// result lists the taken channels in wire order; `own` channels are never
    /// in it.
    pub(crate) fn inherited_positional(
        self,
        own: &[PositionalChannel],
        offered: &[PositionalChannel],
    ) -> Vec<PositionalChannel> {
        let candidates = || PositionalChannel::ALL.into_iter().filter(|c| offered.contains(c) && !own.contains(c));
        let spans: Vec<PositionalChannel> =
            candidates().filter(|c| c.is_span() && self.draws_span_channels()).collect();
        let draws_a_span = !spans.is_empty() || own.iter().any(|c| c.is_span());
        let one_axis = matches!(self, Mark::Rule | Mark::Tick)
            && !draws_a_span
            && own.contains(&PositionalChannel::X) != own.contains(&PositionalChannel::Y);
        let takes_position = self.draws_position_channels() && !one_axis;
        candidates().filter(|c| if c.is_span() { spans.contains(c) } else { takes_position }).collect()
    }

    /// Whether this mark draws a constant `size=`.
    pub(crate) fn draws_constant_size(self) -> bool {
        self.size_drawing() == SizeDrawing::ConstantOrEncoded
    }

    /// Every style field (`MarkKwargsSpec` wire key) this mark draws, in the
    /// struct's declaration order: the one table of what each mark draws.
    /// A set field outside it refuses at render
    /// ([`crate::render::RenderError::MarkStyleNotDrawn`]); a conditional
    /// entry warns when its input is absent. `size` appears exactly where
    /// [`Self::draws_constant_size`] holds; its refusal elsewhere stays
    /// [`crate::render::RenderError::MarkSizeNotDrawn`], which names the
    /// mark's own sizing kwarg. `blend` is drawn by every mark.
    ///
    /// The table records what each mark's builder reads. Stroke shape
    /// (`stroke_cap`, `stroke_join`) draws only where the mark strokes an
    /// open path, a corner or a dash: a polygon's and a ribbon's closed,
    /// undashed outlines have no ends for a cap.
    pub(crate) fn style_fields(self) -> &'static [StyleField] {
        use StyleNeed::*;
        const POINT: &[StyleField] = &[
            drawn("size"), drawn("stroke"), drawn("fill"), drawn("opacity"),
            drawn_with("stroke_width", Stroke), drawn_with("stroke_dash", Stroke),
            drawn("stroke_cap"), drawn_with("stroke_join", Stroke), drawn("filled"), drawn("shape"),
            drawn("clip"), drawn("blend"),
        ];
        const LINE: &[StyleField] = &[
            drawn("size"), drawn("stroke"), drawn("opacity"), drawn("stroke_width"), drawn("stroke_dash"),
            drawn("detail"), drawn("interpolate"), drawn("stroke_cap"), drawn("stroke_join"), drawn("clip"),
            drawn("blend"), drawn("value_axis"), drawn("sort_vertices"), drawn("rise_from_zero"),
        ];
        const BAR: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn("corner_radius"),
            drawn_with("stroke_width", Stroke), drawn_with("stroke_dash", Stroke),
            drawn_with("stroke_cap", Stroke), drawn_with("stroke_join", Stroke), drawn("clip"), drawn("blend"),
            drawn_with("zero", LayeredChart),
        ];
        const AREA: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn_with("stroke_width", Stroke),
            drawn("detail"), drawn("interpolate"), drawn_with("stroke_cap", Stroke),
            drawn_with("stroke_join", Stroke), drawn("line"), drawn("borders"), drawn("clip"), drawn("blend"),
            drawn("value_axis"),
        ];
        const RULE: &[StyleField] = &[
            drawn("stroke"), drawn("opacity"), drawn("stroke_width"), drawn("stroke_dash"),
            drawn("stroke_cap"), drawn("stroke_join"), drawn("clip"), drawn("blend"),
        ];
        const TEXT: &[StyleField] = &[
            drawn("fill"), drawn("opacity"), drawn("font_size"), drawn("font_weight"), drawn("align"),
            drawn("baseline"), drawn("dx"), drawn("dy"), drawn("angle"), drawn("limit"),
            drawn_with("radius", PolarLabel), drawn("blend"),
        ];
        const TICK: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn("stroke_width"), drawn("stroke_dash"),
            drawn("stroke_cap"), drawn("stroke_join"), drawn("band_size"), drawn("clip"), drawn("blend"),
        ];
        const RECT: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn("corner_radius"),
            drawn_with("stroke_width", Stroke), drawn_with("stroke_dash", Stroke),
            drawn_with("font_size", TextChannel), drawn_with("stroke_cap", Stroke),
            drawn_with("stroke_join", Stroke), drawn_with("band_size", BandScale), drawn("clip"),
            drawn("blend"),
        ];
        const POLYGON: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn_with("stroke_width", Stroke),
            drawn("detail"), drawn_with("cmap", QuantitativeColor), drawn_with("stroke_join", Stroke),
            drawn("clip"), drawn("blend"),
        ];
        const IMAGE: &[StyleField] = &[
            drawn_with("cmap", RasterData), drawn_with("width", UrlChannel),
            drawn_with("height", UrlChannel), drawn("clip"), drawn("blend"),
        ];
        const RIBBON: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn_with("stroke_width", Stroke),
            drawn("detail"), drawn("interpolate"), drawn_with("stroke_join", Stroke), drawn("clip"), drawn("blend"),
        ];
        const SEGMENT: &[StyleField] = &[
            drawn("stroke"), drawn("opacity"), drawn("stroke_width"), drawn("stroke_dash"),
            drawn("stroke_cap"), drawn("stroke_join"), drawn("clip"), drawn("blend"), drawn("arrow"),
            drawn_with("arrow_size", Arrow),
        ];
        const ARC: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn("corner_radius"),
            drawn_with("stroke_width", Stroke), drawn_with("stroke_dash", Stroke),
            drawn_with("stroke_cap", Stroke), drawn_with("stroke_join", Stroke), drawn("clip"), drawn("blend"),
        ];
        const LABEL: &[StyleField] = &[
            drawn("fill"), drawn("opacity"), drawn("font_size"), drawn("font_weight"),
            drawn_with("dx", LeaderPlacement), drawn_with("dy", LeaderPlacement), drawn("leader_line"),
            drawn("blend"),
        ];
        const GEOSHAPE: &[StyleField] = &[
            drawn("stroke"), drawn("fill"), drawn("opacity"), drawn("stroke_width"), drawn("stroke_cap"),
            drawn("stroke_join"), drawn("clip"), drawn("blend"),
        ];
        match self {
            Mark::Point => POINT,
            Mark::Line => LINE,
            Mark::Bar => BAR,
            Mark::Area => AREA,
            Mark::Rule => RULE,
            Mark::Text => TEXT,
            Mark::Tick => TICK,
            Mark::Rect => RECT,
            Mark::Polygon => POLYGON,
            Mark::Image => IMAGE,
            Mark::Ribbon => RIBBON,
            Mark::Segment => SEGMENT,
            Mark::Arc => ARC,
            Mark::Label => LABEL,
            Mark::Geoshape => GEOSHAPE,
        }
    }

    /// This mark's [`Self::style_fields`] entry for wire key `name`, or
    /// `None` when the mark never draws it.
    pub(crate) fn style_field(self, name: &str) -> Option<StyleField> {
        self.style_fields().iter().copied().find(|f| f.name == name)
    }

    /// The mark's own kwarg that sets how big or thick it draws, named when a
    /// constant `size=` it never draws is refused.
    pub(crate) fn size_knob(self) -> Option<&'static str> {
        match self {
            Mark::Rule | Mark::Segment => Some("stroke_width"),
            Mark::Text | Mark::Label => Some("font_size"),
            Mark::Tick | Mark::Rect => Some("band_size"),
            _ => None,
        }
    }

    /// Every input that turns this mark's stroke on, which is what a
    /// [`StyleNeed::Stroke`] field needs. `render::draw::stroke_is_drawn`
    /// tests exactly these, and the inert-field warning names exactly these,
    /// so the two cannot disagree.
    pub(crate) fn stroke_sources(self) -> &'static [StrokeSource] {
        use StrokeSource::*;
        match self {
            Mark::Point => &[Color, Hollow, WidthColumn],
            Mark::Area => &[Color, AreaBorders],
            Mark::Bar | Mark::Rect => &[Color, WidthColumn],
            _ => &[Color],
        }
    }
}

/// One input that turns a mark's stroke on ([`Mark::stroke_sources`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrokeSource {
    /// A `stroke=` color.
    Color,
    /// `filled=False` on a point, whose hollow outline takes the fill.
    Hollow,
    /// `line=True` or `borders=True` on an area, whose border lines take the
    /// fill.
    AreaBorders,
    /// A bound numeric `stroke_width` column on a filled mark whose stroke is
    /// unset: the stroke takes the fill so the width shows. A cleared stroke
    /// (`stroke="none"`) shuts it.
    WidthColumn,
}

impl StrokeSource {
    /// The input, as the inert-field warning names it.
    fn describe(self) -> &'static str {
        match self {
            Self::Color => "a stroke color (stroke=)",
            Self::Hollow => "a hollow outline (filled=False)",
            Self::AreaBorders => "border lines (line=True or borders=True)",
            Self::WidthColumn => "a numeric stroke_width encoding (which stroke='none' turns off)",
        }
    }
}

/// What a mark draws for `size` ([`Mark::size_drawing`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SizeDrawing {
    /// A constant `size=` or an encoded `size` channel: a point's area, a
    /// line's dot-diameter width.
    ConstantOrEncoded,
    /// An encoded `size` channel only: a text's font size. A constant
    /// `size=` is never read; `font_size=` sizes the mark.
    EncodedOnly,
    /// Nothing: the mark carries an inherited `size` binding without drawing it.
    Never,
}

/// What a mark draws for one style field ([`Mark::style_fields`]): the
/// field's wire key, and the input it needs alongside it, if any. A field
/// with no `needs` draws whenever it is set; one with `needs` draws only
/// when that input is present too, and is reported inert
/// ([`crate::render::RenderWarning::MarkStyleInert`]) when it is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StyleField {
    pub(crate) name: &'static str,
    pub(crate) needs: Option<StyleNeed>,
}

const fn drawn(name: &'static str) -> StyleField {
    StyleField { name, needs: None }
}

const fn drawn_with(name: &'static str, need: StyleNeed) -> StyleField {
    StyleField { name, needs: Some(need) }
}

/// The input a conditionally drawn style field needs ([`StyleField::needs`]).
/// Whether it is present is decided once per layer, after scales resolve,
/// by `render::draw::style_need_met`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleNeed {
    /// A drawn stroke, on a mark whose stroke is off by default: any one of
    /// the mark's [`Mark::stroke_sources`].
    Stroke,
    /// `arrow=True` on a segment.
    Arrow,
    /// A quantitative color encoding, which a colormap maps.
    QuantitativeColor,
    /// A band (ordinal) axis paired with a `y2`/`x2` range: the box-body
    /// shape a rect sizes by `band_size`. A heatmap's cells fill their bands
    /// and read no `band_size`.
    BandScale,
    /// Fixed label placement: both `dx` and `dy` set. A label places itself
    /// by collision avoidance unless both offsets are given.
    LeaderPlacement,
    /// A `text` encoding on a heatmap rect, whose cells then carry labels.
    TextChannel,
    /// A `url` encoding: an image's url tiles, which `width`/`height` size.
    UrlChannel,
    /// Raster data (`mark_raster()`), which a colormap colors: an image
    /// without a `url` encoding.
    RasterData,
    /// A layered chart. A single bar chart anchors at zero through
    /// `mark_bar(zero=)`, which the chart consumes; a layer reads its own.
    LayeredChart,
    /// A text label under `CoordPolar` with a theta encoding and no radius
    /// encoding: the label's distance from the polar center is then the
    /// style's to set.
    PolarLabel,
}

impl StyleNeed {
    /// The wire name: what `mark_style_fields` publishes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stroke => "stroke",
            Self::Arrow => "arrow",
            Self::QuantitativeColor => "quantitative_color",
            Self::BandScale => "band_scale",
            Self::LeaderPlacement => "leader_placement",
            Self::TextChannel => "text_channel",
            Self::UrlChannel => "url_channel",
            Self::RasterData => "raster_data",
            Self::LayeredChart => "layered_chart",
            Self::PolarLabel => "polar_label",
        }
    }

    /// The missing input on `mark`, as the inert-field warning names it. A
    /// stroke need names every one of the mark's [`Mark::stroke_sources`].
    pub(crate) fn describe(self, mark: Mark) -> String {
        let fixed = match self {
            Self::Stroke => {
                let sources = mark.stroke_sources();
                let mut text = String::new();
                for (i, source) in sources.iter().enumerate() {
                    if i > 0 {
                        text.push_str(if i + 1 == sources.len() { " or " } else { ", " });
                    }
                    text.push_str(source.describe());
                }
                return text;
            }
            Self::Arrow => "arrow=True",
            Self::QuantitativeColor => "a quantitative color encoding",
            Self::BandScale => "a band axis paired with a y2 or x2 range",
            Self::LeaderPlacement => "both dx and dy (a label places itself unless both are set)",
            Self::TextChannel => "both a text encoding and the heatmap shape (one cell per x/y band pair, no x2 or y2 range)",
            Self::UrlChannel => "a url encoding",
            Self::RasterData => "raster data from mark_raster()",
            Self::LayeredChart => "a layered chart (a single bar chart sets it through mark_bar(zero=))",
            Self::PolarLabel => "a polar coord (CoordPolar) with a theta encoding and no radius encoding",
        };
        fixed.to_string()
    }
}

/// Every primitive mark, in [`for_each_mark!`] order.
pub(crate) fn all_marks() -> impl Iterator<Item = Mark> {
    macro_rules! names {
        ($($V:ident => $name:ident,)*) => {
            [$( Mark::$V, )*]
        };
    }
    for_each_mark!(names).into_iter()
}

/// The primitive marks that draw a constant `size=`, in [`for_each_mark!`]
/// order: Python's mark constructors read this rather than keeping a list of
/// their own.
#[pyo3::pyfunction]
pub(crate) fn marks_drawing_constant_size() -> Vec<&'static str> {
    all_marks().filter(|m| m.draws_constant_size()).map(|m| m.as_str()).collect()
}

/// `ferrum._core.style_table_marks()`: the marks [`Mark::style_fields`] has a
/// row for, in table order, which is every primitive mark in
/// [`for_each_mark!`] order. Python reads its mark population from here
/// rather than keeping a list of its own. Registered in `lib.rs`.
#[pyo3::pyfunction]
pub(crate) fn style_table_marks() -> Vec<&'static str> {
    all_marks().map(|m| m.as_str()).collect()
}

/// `ferrum._core.mark_style_fields(mark)`: [`Mark::style_fields`] as a dict
/// from each field the mark draws to the name of the input it needs
/// ([`StyleNeed::as_str`]), or `None` when it draws whenever set. Python
/// derives every mark's honored style kwargs from it. Refuses an unknown
/// mark. Registered in `lib.rs`, typed in `src/ferrum/_core.pyi`.
#[pyo3::pyfunction]
pub(crate) fn mark_style_fields<'py>(
    py: pyo3::Python<'py>,
    mark: &str,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDictMethods;
    let mark = parse_mark_arg(mark)?;
    let fields = pyo3::types::PyDict::new(py);
    for field in mark.style_fields() {
        fields.set_item(field.name, field.needs.map(StyleNeed::as_str))?;
    }
    Ok(fields)
}

/// `ferrum._core.mark_inherited_positional(mark, own, offered)`:
/// [`Mark::inherited_positional`] over wire names, the positional channels
/// (`"x"`, `"y"`, `"x2"`, `"y2"`) a layer of `mark` writing `own` takes of the
/// `offered` parent bindings, in wire order. Python's effective encoding
/// filters its inherited positional channels through it rather than keeping
/// a list of its own. Refuses an unknown mark or channel name. Registered in
/// `lib.rs`.
#[pyo3::pyfunction]
pub(crate) fn mark_inherited_positional(
    mark: &str,
    own: Vec<String>,
    offered: Vec<String>,
) -> pyo3::PyResult<Vec<&'static str>> {
    let mark = parse_mark_arg(mark)?;
    let channels = |names: &[String]| -> pyo3::PyResult<Vec<PositionalChannel>> {
        names
            .iter()
            .map(|name| {
                PositionalChannel::from_wire(name).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown positional channel '{name}'; expected one of [x, y, x2, y2]"
                    ))
                })
            })
            .collect()
    };
    let taken = mark.inherited_positional(&channels(&own)?, &channels(&offered)?);
    Ok(taken.into_iter().map(PositionalChannel::as_str).collect())
}

/// `ferrum._core.mark_draws_inherited(mark, channel)`: whether a layer of
/// `mark` takes an inherited binding of the non-positional `channel` (an
/// `Encoding` wire name) as a binding ([`Mark::draws_inherited`]). Python's
/// effective encoding asks it which inherited channels become a grouping key
/// instead of a binding, rather than keeping a list of its own. Refuses an
/// unknown mark or channel name. Registered in `lib.rs`.
#[pyo3::pyfunction]
pub(crate) fn mark_draws_inherited(mark: &str, channel: &str) -> pyo3::PyResult<bool> {
    use crate::spec::layer::InheritedChannel;
    let mark = parse_mark_arg(mark)?;
    let channel = InheritedChannel::from_wire(channel).ok_or_else(|| {
        let names: Vec<&str> = InheritedChannel::ALL.into_iter().map(InheritedChannel::as_str).collect();
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown non-positional channel '{channel}'; expected one of [{}]",
            names.join(", ")
        ))
    })?;
    Ok(mark.draws_inherited(channel))
}

/// `ferrum._core.mark_drawn_channels(mark)`: the non-positional channels
/// (`Encoding` wire names, in declaration order) a layer of `mark` draws when
/// they are bound to a field ([`Mark::draws_channel`]). Python's per-layer
/// auto tooltip lists only these rather than keeping a table of its own.
/// Refuses an unknown mark. Registered in `lib.rs`.
#[pyo3::pyfunction]
pub(crate) fn mark_drawn_channels(mark: &str) -> pyo3::PyResult<Vec<&'static str>> {
    use crate::spec::layer::InheritedChannel;
    let mark = parse_mark_arg(mark)?;
    Ok(InheritedChannel::ALL.into_iter().filter(|c| mark.draws_channel(*c)).map(InheritedChannel::as_str).collect())
}

/// `ferrum._core.mark_stacks_polar_theta(mark)`: whether `mark` stacks the
/// theta channel of a `CoordPolar` itself ([`Mark::stacks_polar_theta`]).
/// Python's stack-honoring decision reads this rather than keeping its own
/// copy of which marks sweep theta into a pie. Refuses an unknown mark.
/// Registered in `lib.rs`.
#[pyo3::pyfunction]
pub(crate) fn mark_stacks_polar_theta(mark: &str) -> pyo3::PyResult<bool> {
    let mark = parse_mark_arg(mark)?;
    Ok(mark.stacks_polar_theta())
}

impl fmt::Display for Mark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct ParseMarkError(pub String);

impl fmt::Display for ParseMarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown mark '{}'; expected one of [point, line, bar, area, rule, text, tick, rect, polygon, image, ribbon, segment, arc, label, geoshape]",
            self.0
        )
    }
}

impl std::error::Error for ParseMarkError {}

/// Parses `mark` into a [`Mark`], refusing an unknown name with the
/// [`pyo3::exceptions::PyValueError`] every mark-name-accepting `#[pyfunction]`
/// raises before it applies its own rule: [`mark_style_fields`],
/// [`crate::render::draw::check_mark_size`] and
/// [`crate::render::draw::mark_size_reason`] all parse through here.
pub(crate) fn parse_mark_arg(mark: &str) -> pyo3::PyResult<Mark> {
    mark.parse().map_err(|e: ParseMarkError| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

impl FromStr for Mark {
    type Err = ParseMarkError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        macro_rules! arm {
            ($($V:ident => $name:ident,)*) => {
                match s {
                    $( s if s == stringify!($name) => Ok(Mark::$V), )*
                    other => Err(ParseMarkError(other.to_string())),
                }
            };
        }
        for_each_mark!(arm)
    }
}

#[cfg(test)]
mod tests {
    use crate::spec::layer::PositionalChannel::{self, X, X2, Y, Y2};

    /// The span-drawing marks, read from each builder: the marks whose
    /// builders read an `x2`/`y2` column. Pinned per mark so a new builder
    /// reading a span is a visible table change.
    #[test]
    fn the_span_drawing_marks_are_the_builders_that_read_a_span() {
        let drawing: Vec<&str> = all_marks().filter(|m| m.draws_span_channels()).map(|m| m.as_str()).collect();
        assert_eq!(drawing, ["bar", "area", "rule", "rect", "ribbon", "segment"]);
    }

    /// Per mark, what a layer writing nothing positional takes of a parent
    /// binding all four channels: `x`/`y` on every mark but a geoshape, the
    /// spans only on a span-drawing mark.
    #[test]
    fn a_layer_writing_nothing_takes_x_y_and_the_spans_its_mark_draws() {
        for mark in all_marks() {
            let taken = mark.inherited_positional(&[], &[X, Y, X2, Y2]);
            let position = if mark.draws_position_channels() { vec![X, Y] } else { vec![] };
            let spans = if mark.draws_span_channels() { vec![X2, Y2] } else { vec![] };
            let expected: Vec<PositionalChannel> = position.into_iter().chain(spans).collect();
            assert_eq!(taken, expected, "{mark}");
        }
    }

    /// Only a geoshape draws no position channel, so it takes no inherited
    /// `x`/`y`. An arc keeps them: it draws them as its polar theta and
    /// radius.
    #[test]
    fn only_a_geoshape_draws_no_position_channel() {
        let not_drawing: Vec<&str> =
            all_marks().filter(|m| !m.draws_position_channels()).map(|m| m.as_str()).collect();
        assert_eq!(not_drawing, ["geoshape"]);
        assert_eq!(Mark::Geoshape.inherited_positional(&[], &[X, Y]), Vec::<PositionalChannel>::new());
        assert_eq!(Mark::Arc.inherited_positional(&[], &[X, Y]), vec![X, Y]);
    }

    /// The marks drawing each non-positional channel outside the size,
    /// stroke-width and opacity tables, as each builder reads it, and those
    /// three answering from their own tables.
    #[test]
    fn each_non_positional_channel_is_drawn_by_the_marks_whose_builders_read_it() {
        use crate::render::scale_resolve::OpacityChannel;
        use crate::spec::layer::InheritedChannel as C;
        let drawing = |channel: C| -> Vec<&str> {
            all_marks().filter(|m| m.draws_channel(channel)).map(|m| m.as_str()).collect()
        };
        let every: Vec<&str> = all_marks().map(|m| m.as_str()).collect();
        assert_eq!(
            drawing(C::Color),
            ["point", "line", "bar", "area", "rule", "text", "tick", "rect", "polygon", "ribbon", "segment", "arc"]
        );
        assert_eq!(drawing(C::Text), ["text", "rect", "label"]);
        assert_eq!(drawing(C::Shape), ["point"]);
        assert_eq!(drawing(C::Angle), ["point", "bar", "text"]);
        assert_eq!(drawing(C::StrokeDash), ["point", "line", "bar", "rule", "rect", "ribbon"]);
        assert_eq!(drawing(C::Url), ["image"]);
        for channel in [C::Tooltip, C::TooltipFields, C::Href, C::Description, C::Key] {
            assert_eq!(drawing(channel), every, "{channel:?}");
        }
        for mark in all_marks() {
            assert_eq!(mark.draws_channel(C::Size), mark.draws_size_channel());
            assert_eq!(mark.draws_channel(C::StrokeWidth), mark.draws_stroke_width_channel());
            assert_eq!(mark.draws_channel(C::Opacity), mark.draws_opacity_channel(OpacityChannel::Opacity));
            assert_eq!(mark.draws_channel(C::FillOpacity), mark.draws_opacity_channel(OpacityChannel::Fill));
            assert_eq!(mark.draws_channel(C::StrokeOpacity), mark.draws_opacity_channel(OpacityChannel::Stroke));
        }
    }

    /// The published table lists a mark's drawn channels by wire name, in
    /// declaration order, and refuses an unknown mark.
    #[test]
    fn mark_drawn_channels_publishes_the_table() {
        assert_eq!(
            mark_drawn_channels("image").unwrap(),
            ["tooltip", "tooltip_fields", "href", "description", "key", "url"]
        );
        assert!(mark_drawn_channels("point").unwrap().contains(&"shape"));
        assert!(mark_drawn_channels("nope").is_err());
    }

    /// The published predicate matches `Mark::stacks_polar_theta`, and refuses
    /// an unknown mark.
    #[test]
    fn mark_stacks_polar_theta_publishes_the_predicate() {
        assert!(mark_stacks_polar_theta("arc").unwrap());
        assert!(mark_stacks_polar_theta("text").unwrap());
        assert!(!mark_stacks_polar_theta("point").unwrap());
        assert!(mark_stacks_polar_theta("nope").is_err());
    }

    /// A rule or tick declaring exactly one of `x`/`y` with no span is a
    /// one-axis line or rug and takes nothing else. A rule that takes an
    /// inherited span keeps assembling its ranged shape. Any other mark, or a
    /// rule or tick declaring both or neither, takes `x`/`y`.
    #[test]
    fn a_one_axis_rule_or_tick_takes_no_opposite_channel() {
        for mark in [Mark::Rule, Mark::Tick] {
            assert_eq!(mark.inherited_positional(&[Y], &[X]), Vec::<PositionalChannel>::new(), "{mark}");
            assert_eq!(mark.inherited_positional(&[X], &[Y]), Vec::<PositionalChannel>::new(), "{mark}");
            assert_eq!(mark.inherited_positional(&[], &[X, Y]), vec![X, Y], "{mark}");
        }
        assert_eq!(Mark::Rule.inherited_positional(&[X], &[Y, X2]), vec![Y, X2]);
        assert_eq!(Mark::Rule.inherited_positional(&[X, X2], &[Y]), vec![Y]);
        assert_eq!(Mark::Tick.inherited_positional(&[X], &[Y, X2]), Vec::<PositionalChannel>::new());
        assert_eq!(Mark::Point.inherited_positional(&[Y], &[X]), vec![X]);
    }

    /// The Python door: wire names in, wire names out, in wire order; an
    /// unknown channel or mark is refused.
    #[test]
    fn the_pyfunction_speaks_wire_names() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            mark_inherited_positional("line", names(&[]), names(&["y2", "x", "x2", "y"])).unwrap(),
            ["x", "y"]
        );
        assert_eq!(mark_inherited_positional("rule", names(&["y"]), names(&["x"])).unwrap(), Vec::<&str>::new());
        pyo3::Python::initialize();
        assert!(mark_inherited_positional("rule", names(&["z"]), names(&[])).is_err());
        assert!(mark_inherited_positional("nope", names(&[]), names(&[])).is_err());
    }

    use super::*;

    #[test]
    fn test_mark_round_trip_each_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect, Mark::Polygon, Mark::Image, Mark::Ribbon, Mark::Segment,
            Mark::Arc, Mark::Label, Mark::Geoshape,
        ] {
            let json = serde_json::to_string(&m).unwrap();
            let parsed: Mark = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, m, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_mark_serde_form_is_lowercase() {
        assert_eq!(serde_json::to_string(&Mark::Point).unwrap(), "\"point\"");
        assert_eq!(serde_json::to_string(&Mark::Bar).unwrap(),   "\"bar\"");
    }

    #[test]
    fn test_mark_from_str_known() {
        assert_eq!(Mark::from_str("point").unwrap(), Mark::Point);
        assert_eq!(Mark::from_str("rect").unwrap(),  Mark::Rect);
    }

    #[test]
    fn test_mark_segment_round_trip() {
        let m = Mark::Segment;
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "\"segment\"");
        let parsed: Mark = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Mark::Segment);
        assert_eq!(Mark::from_str("segment").unwrap(), Mark::Segment);
        assert_eq!(Mark::Segment.as_str(), "segment");
    }

    /// Every table entry names a real `MarkKwargsSpec` wire key, once per
    /// mark, in the struct's declaration order.
    #[test]
    fn style_fields_name_real_wire_keys_in_declaration_order() {
        let keys = crate::spec::mark_style::mark_kwargs_field_names();
        for mark in all_marks() {
            let positions: Vec<usize> = mark
                .style_fields()
                .iter()
                .map(|f| keys.iter().position(|k| *k == f.name).unwrap_or_else(|| panic!("{mark}: '{}' is no wire key", f.name)))
                .collect();
            assert!(positions.windows(2).all(|w| w[0] < w[1]), "{mark}: out of order or repeated: {positions:?}");
        }
    }

    /// `size` is in a mark's table exactly where the size table says the mark
    /// draws a constant size, so the two tables cannot disagree; `blend` is
    /// in every mark's; and some field is outside some mark's (the gate has
    /// something to refuse).
    #[test]
    fn style_fields_agree_with_the_size_table_and_list_blend_everywhere() {
        let keys = crate::spec::mark_style::mark_kwargs_field_names();
        for mark in all_marks() {
            assert_eq!(mark.style_field("size").is_some(), mark.draws_constant_size(), "{mark}");
            assert_eq!(mark.style_field("blend"), Some(StyleField { name: "blend", needs: None }), "{mark}");
            assert!(mark.style_fields().len() < keys.len(), "{mark} draws every field");
        }
        assert_eq!(Mark::Segment.style_field("arrow_size").and_then(|f| f.needs), Some(StyleNeed::Arrow));
        assert_eq!(Mark::Point.style_field("font_size"), None);
    }

    /// Every mark takes every inherited binding but a text's color, and the
    /// published answer is the table's; an unknown mark or channel refuses.
    #[test]
    fn draws_inherited_declines_only_a_texts_color() {
        use crate::spec::layer::InheritedChannel;
        for mark in all_marks() {
            for channel in InheritedChannel::ALL {
                let declined = mark == Mark::Text && channel == InheritedChannel::Color;
                assert_eq!(mark.draws_inherited(channel), !declined, "{mark} {}", channel.as_str());
                assert_eq!(mark_draws_inherited(mark.as_str(), channel.as_str()).unwrap(), !declined);
            }
        }
        pyo3::Python::initialize();
        assert!(mark_draws_inherited("text", "x").is_err(), "a positional channel is no inherited channel");
        assert!(mark_draws_inherited("txt", "color").is_err());
    }

    /// The published dict is the table, in order, each need by its wire name.
    #[test]
    fn mark_style_fields_publishes_the_table() {
        use pyo3::types::{PyAnyMethods, PyDictMethods, PyListMethods};
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let fields = mark_style_fields(py, "segment").unwrap();
            let names: Vec<String> = fields.keys().iter().map(|k| k.extract().unwrap()).collect();
            let expected: Vec<&str> = Mark::Segment.style_fields().iter().map(|f| f.name).collect();
            assert_eq!(names, expected);
            let arrow_size: Option<String> = fields.get_item("arrow_size").unwrap().unwrap().extract().unwrap();
            assert_eq!(arrow_size.as_deref(), Some("arrow"));
            let stroke: Option<String> = fields.get_item("stroke").unwrap().unwrap().extract().unwrap();
            assert_eq!(stroke, None);
            let err = mark_style_fields(py, "pont").unwrap_err().to_string();
            assert!(err.contains("unknown mark 'pont'"), "{err}");
        });
    }

    /// The published marks are the table's rows, in order, each one a name
    /// `mark_style_fields` reads.
    #[test]
    fn style_table_marks_publishes_every_row_in_table_order() {
        let marks = style_table_marks();
        assert_eq!(
            marks,
            [
                "point", "line", "bar", "area", "rule", "text", "tick", "rect", "polygon", "image", "ribbon",
                "segment", "arc", "label", "geoshape",
            ]
        );
        for name in marks {
            let mark: Mark = name.parse().unwrap();
            assert!(!mark.style_fields().is_empty(), "{name} has no table row");
        }
    }

    /// The wire names are the serde spellings, which a warning serializes.
    #[test]
    fn style_need_wire_names_are_the_serde_spellings() {
        use StyleNeed::*;
        for need in [Stroke, Arrow, QuantitativeColor, BandScale, LeaderPlacement, TextChannel, UrlChannel, RasterData, LayeredChart] {
            assert_eq!(serde_json::to_string(&need).unwrap(), format!("\"{}\"", need.as_str()));
        }
    }

    #[test]
    fn test_mark_from_str_unknown_lists_variants() {
        let err = Mark::from_str("pont").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'pont'"), "msg was: {msg}");
        assert!(msg.contains("point"), "msg was: {msg}");
        assert!(msg.contains("rect"),  "msg was: {msg}");
    }
}
