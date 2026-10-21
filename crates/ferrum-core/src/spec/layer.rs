use serde::{Deserialize, Serialize};

use crate::spec::encoding::Encoding;
use crate::spec::mark::Mark;
use crate::transform::core::TransformSpec;

/// A positional encoding channel name: `x`, `y`, `x2` or `y2` — the domain
/// [`Layer::inherited_positional`] ranges over. Deserializes from exactly
/// those four lowercase wire strings; any other value is a deserialize
/// error, not a silently-ignored one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionalChannel {
    X,
    Y,
    X2,
    Y2,
}

impl PositionalChannel {
    /// The four channels, in wire order.
    pub(crate) const ALL: [PositionalChannel; 4] =
        [PositionalChannel::X, PositionalChannel::Y, PositionalChannel::X2, PositionalChannel::Y2];

    /// The wire name: `"x"`, `"y"`, `"x2"` or `"y2"`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            PositionalChannel::X => "x",
            PositionalChannel::Y => "y",
            PositionalChannel::X2 => "x2",
            PositionalChannel::Y2 => "y2",
        }
    }

    /// The channel a wire name names, or `None` for any other string.
    pub(crate) fn from_wire(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_str() == name)
    }

    /// Whether this is a span channel (`x2`/`y2`).
    pub(crate) fn is_span(self) -> bool {
        matches!(self, PositionalChannel::X2 | PositionalChannel::Y2)
    }

    /// This channel's binding slot in `encoding`.
    pub(crate) fn slot(self, encoding: &Encoding) -> &Option<crate::spec::encoding::EncodingSpec> {
        match self {
            PositionalChannel::X => &encoding.x,
            PositionalChannel::Y => &encoding.y,
            PositionalChannel::X2 => &encoding.x2,
            PositionalChannel::Y2 => &encoding.y2,
        }
    }

    /// This channel's binding slot in `encoding`, mutably.
    pub(crate) fn slot_mut(self, encoding: &mut Encoding) -> &mut Option<crate::spec::encoding::EncodingSpec> {
        match self {
            PositionalChannel::X => &mut encoding.x,
            PositionalChannel::Y => &mut encoding.y,
            PositionalChannel::X2 => &mut encoding.x2,
            PositionalChannel::Y2 => &mut encoding.y2,
        }
    }
}

/// A non-positional encoding channel name: one wire key of [`Encoding`]
/// other than the positional `x`, `y`, `x2` and `y2` — the domain
/// [`Layer::inherited_channels`] ranges over. Deserializes from exactly those
/// wire keys (`"color"`, `"stroke_width"`, ...); any other value is a
/// deserialize error, as [`PositionalChannel`] refuses one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InheritedChannel {
    Color,
    Size,
    Shape,
    Opacity,
    Text,
    Tooltip,
    TooltipFields,
    Href,
    Description,
    Key,
    Url,
    StrokeWidth,
    StrokeOpacity,
    StrokeDash,
    Angle,
    FillOpacity,
}

impl InheritedChannel {
    /// Every non-positional channel, in [`Encoding`]'s declaration order.
    pub(crate) const ALL: [InheritedChannel; 16] = [
        InheritedChannel::Color,
        InheritedChannel::Size,
        InheritedChannel::Shape,
        InheritedChannel::Opacity,
        InheritedChannel::Text,
        InheritedChannel::Tooltip,
        InheritedChannel::TooltipFields,
        InheritedChannel::Href,
        InheritedChannel::Description,
        InheritedChannel::Key,
        InheritedChannel::Url,
        InheritedChannel::StrokeWidth,
        InheritedChannel::StrokeOpacity,
        InheritedChannel::StrokeDash,
        InheritedChannel::Angle,
        InheritedChannel::FillOpacity,
    ];

    /// The wire name, the key [`Encoding`] serializes the channel under.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            InheritedChannel::Color => "color",
            InheritedChannel::Size => "size",
            InheritedChannel::Shape => "shape",
            InheritedChannel::Opacity => "opacity",
            InheritedChannel::Text => "text",
            InheritedChannel::Tooltip => "tooltip",
            InheritedChannel::TooltipFields => "tooltip_fields",
            InheritedChannel::Href => "href",
            InheritedChannel::Description => "description",
            InheritedChannel::Key => "key",
            InheritedChannel::Url => "url",
            InheritedChannel::StrokeWidth => "stroke_width",
            InheritedChannel::StrokeOpacity => "stroke_opacity",
            InheritedChannel::StrokeDash => "stroke_dash",
            InheritedChannel::Angle => "angle",
            InheritedChannel::FillOpacity => "fill_opacity",
        }
    }

    /// The channel a wire name names, or `None` for a positional or unknown
    /// name.
    pub(crate) fn from_wire(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_str() == name)
    }
}

/// A single layer within a multi-layer ChartSpec. Inherits chart-level
/// encoding for any field set to None at the layer level.
///
/// An unknown key refuses (`deny_unknown_fields`), as `ChartSpec`'s and
/// `MarkKwargsSpec`'s do: a misspelt or retired layer key (a layer-level
/// `blend`, which is a style field in `mark_style`) is never dropped without
/// a word.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    /// Positional channel names (`"x"`, `"y"`, `"x2"`, `"y2"`) this layer's
    /// `encoding` carries a VALUE for but did not itself declare — filled in
    /// by Python's `+` routing from the RHS operand's chart-level encoding
    /// when a layered operand gets a `data_source` and would otherwise lose
    /// every chart positional channel it never bound itself (a
    /// `_promote_layer_channels`-style copy with provenance). Read only by
    /// [`Self::completed_encoding`]: a channel listed here is an inherited
    /// one, kept only when the layer's mark draws it
    /// ([`Mark::inherited_positional`]), not a layer-declared one. Empty (the default) for
    /// every layer that declares its own channels or inherits none,
    /// including every pre-existing spec, so the field is omitted from the
    /// wire then and round-trips byte-identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inherited_positional: Vec<PositionalChannel>,
    /// Non-positional channel names this layer's `encoding` carries a value
    /// for that it inherited from its parent rather than declared: Python
    /// writes a parent's binding onto the layer's own encoding when the
    /// binding is scoped to that layer (a `+` operand's encoding reaches only
    /// the operand's layers) or must name a renamed or statistic-lowered
    /// column, and lists the channel here. The non-positional counterpart of
    /// [`Self::inherited_positional`], read the same way and only by
    /// [`LayerPrepared::from_chart_and_layer`](crate::render::prepare::LayerPrepared)
    /// to decide `color_is_own`, `size_is_own` and `stroke_width_is_own`: a
    /// listed channel draws as an inherited one does (a text label or a
    /// literal-paint mark does not take an inherited color, and a line
    /// yields an inherited width to one of its own), so a layer draws the
    /// same whether its parent's binding reached it through the chart-level
    /// encoding or through its own. Every other consumer (scales, domains,
    /// legends, titles) reads `encoding` itself. Empty (the default) when the
    /// layer inherited nothing onto its own encoding, which omits the field
    /// from the wire.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inherited_channels: Vec<InheritedChannel>,
    /// A layer does not run its own transforms: it reads the chart's pipeline
    /// output, or a named chart-level output via `data_source`. This field is
    /// read only to refuse a non-empty list at render
    /// ([`RenderError::LayerTransformsUnsupported`](crate::render::RenderError::LayerTransformsUnsupported));
    /// an empty or absent list is accepted and omitted when serializing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub position: Option<crate::spec::position::PositionAdjust>,
    /// Whether this layer's `position` is one it inherited from its parent
    /// rather than wrote: Python carries a `+` operand chart's `position=`
    /// onto the operand's layers that wrote none, and marks it here. The
    /// position counterpart of [`Self::inherited_positional`] and
    /// [`Self::inherited_channels`], read only by
    /// [`LayerPrepared::from_chart_and_layer`](crate::render::prepare::LayerPrepared):
    /// an inherited position (this one, or the chart-level `position` a layer
    /// with none falls back to) yields to a stack shorthand the layer writes
    /// itself ([`Self::writes_stack_shorthand`]), while a position the layer
    /// wrote is its own and is judged beside its shorthand. `false` (the
    /// default) is omitted from the wire, so every spec without an inherited
    /// position round-trips byte-identically.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inherited_position: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Whether this layer resolves its own y-scale slot instead of sharing
    /// the primary (layer 0) y-scale. `false` (absent on the wire) = shared,
    /// today's behavior. Layer 0 is always the primary/left axis regardless
    /// of this flag (spec §6, GH #52). `false` is omitted when serializing so
    /// pre-#52 layer wire JSON stays byte-identical.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub independent_y: bool,
    /// Whether this layer, reading a named `data_source`, reads its chart's
    /// data through it: Python points a layer attached with `Chart.layer` or
    /// `SecondaryY` at the output that recomputes the chart's rows when they
    /// are not the pipeline tail. A layer with no `data_source` reads its
    /// chart's data already, so the flag is only ever set beside a
    /// `data_source`.
    ///
    /// Read only to pick the remedy of a missing column
    /// ([`RenderError::LayerUnknownColumn`](crate::render::RenderError::LayerUnknownColumn)):
    /// without it a layer reading its chart's data through a named output
    /// could not be told apart from a `+` operand's branch reader, whose
    /// missing column no chart-data remedy describes. `false` is omitted
    /// when serializing, so every spec without such a layer is unchanged on
    /// the wire.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reads_chart_data: bool,
    /// This layer's OWN coordinate system (`#172`(1)). `None` (absent on the
    /// wire) = inherit the chart's, which is every pre-`#172` spec, so a layer
    /// that declares none serializes byte-identically to before.
    ///
    /// A `LayerChart` member declares its coord on its own `Chart`, and the
    /// merged-flat path (`.interactive()`, `resolve={"y": "independent"}`, and
    /// plain `chart + chart`) assembles those members into ONE spec's layers —
    /// which had nowhere to put it, so it was dropped: a secondary `CoordFlip`
    /// rendered unflipped with no error, and a secondary `CoordPolar` refused
    /// with a message prescribing the `.coord()` call the caller had already
    /// written. The coordinate system is spec data about where a layer's marks
    /// go, so it travels on the wire beside the mark and the encoding rather
    /// than being inferred at render time.
    ///
    /// Scope, stated because it is asymmetric on purpose: this places THIS
    /// layer's marks. The panel's chrome — Cartesian axes, gridlines, the polar
    /// boundary, the aspect ratio, the scale domains — stays the chart-level
    /// `ChartSpec::coord`'s, matching what the static composite path draws for
    /// the same composition (one chrome, from the primary leaf). A coord that
    /// declares any of those panel properties therefore cannot be honored here
    /// and refuses; see [`crate::spec::coord::panel_scoped_property`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coord: Option<crate::spec::coord::CoordKind>,
    /// The public entry the caller invoked that produced this layer (e.g.
    /// `"mark_smooth"` for the fit line it desugars into). Every refusal
    /// raised while building the layer (`RenderError::InLayer`), and a
    /// layer-voiced warning (`RenderWarning::SwarmOverflowsBand`), names this
    /// entry in place of the mark's own primitive (`mark_line`), so a caller
    /// who never wrote `mark_line` reads the entry they did write. `None`
    /// falls back to [`crate::spec::chart::ChartSpec::entry`], then to
    /// `mark_{mark}`; the field is omitted from the wire then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
}

/// A point layer with every optional field empty, the base test literals
/// spread with `..Default::default()`. Test-only, as `ChartSpec`'s is.
#[cfg(test)]
impl Default for Layer {
    fn default() -> Self {
        Self {
            mark: Mark::Point,
            encoding: Encoding::default(),
            inherited_positional: Vec::new(),
            inherited_channels: Vec::new(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            inherited_position: false,
            name: None,
            independent_y: false,
            reads_chart_data: false,
            coord: None,
            entry: None,
        }
    }
}

impl Layer {
    /// The positional channels this layer writes itself: those its encoding
    /// binds that Python did not fill from a parent
    /// ([`Self::inherited_positional`]).
    pub(crate) fn own_positional(&self) -> Vec<PositionalChannel> {
        PositionalChannel::ALL
            .into_iter()
            .filter(|c| c.slot(&self.encoding).is_some() && !self.inherited_positional.contains(c))
            .collect()
    }

    /// Whether this layer writes the `stack` shorthand itself: an x or y of
    /// its own ([`Self::own_positional`]) whose binding names a stack offset
    /// ([`EncodingSpec::stack_offset`], the reading
    /// [`crate::render::position::stack_shorthand`] stacks by). A position the
    /// layer inherits yields to it.
    ///
    /// [`EncodingSpec::stack_offset`]: crate::spec::encoding::EncodingSpec::stack_offset
    pub(crate) fn writes_stack_shorthand(&self) -> bool {
        self.own_stack_bindings().any(|binding| binding.stack_offset().is_some())
    }

    /// Whether this layer declines stacking itself: an x or y of its own
    /// whose `stack=` is a falsy spelling ([`EncodingSpec::declines_stack`]).
    /// An inherited `Stack` yields to it; any other inherited position agrees
    /// with it and applies.
    ///
    /// [`EncodingSpec::declines_stack`]: crate::spec::encoding::EncodingSpec::declines_stack
    pub(crate) fn declines_stack(&self) -> bool {
        self.own_stack_bindings().any(|binding| binding.declines_stack())
    }

    /// The bindings of this layer's own x and y, the channels a `stack`
    /// shorthand is written on.
    fn own_stack_bindings(&self) -> impl Iterator<Item = &crate::spec::encoding::EncodingSpec> {
        self.own_positional()
            .into_iter()
            .filter(|c| matches!(c, PositionalChannel::X | PositionalChannel::Y))
            .filter_map(|c| c.slot(&self.encoding).as_ref())
    }

    /// Whether this layer binds the non-positional `channel` itself: bound on
    /// its own encoding and not written there by Python from a parent
    /// ([`Self::inherited_channels`]).
    pub(crate) fn owns(&self, channel: InheritedChannel) -> bool {
        self.encoding.bindings().iter().any(|(name, _)| *name == channel.as_str())
            && !self.inherited_channels.contains(&channel)
    }

    /// This layer's encoding completed from `parent`, the chart-level
    /// encoding: the one completion every reader of a layer's channels sees.
    /// A layer reading the chart's data inherits every unbound channel, one
    /// on its own output only the non-positional ones. Of the positional
    /// bindings it did not write itself (inherited here, or filled by Python
    /// and listed in [`Self::inherited_positional`]) it keeps only those its
    /// mark draws ([`Mark::inherited_positional`]), so a mark never carries a
    /// positional binding it never draws. A color its mark takes only when it
    /// binds it ([`Mark::draws_inherited`]: a text's) is dropped unless the
    /// layer owns it, so no reader counts the layer as drawing it; such a
    /// layer groups by its own `group` key instead.
    pub(crate) fn completed_encoding(&self, parent: &Encoding) -> Encoding {
        let mut encoding = self.encoding.clone();
        if self.data_source.is_some() {
            encoding.inherit_non_positional(parent);
        } else {
            encoding.inherit_from(parent);
        }
        if !self.mark.draws_inherited(InheritedChannel::Color) && !self.owns(InheritedChannel::Color) {
            encoding.color = None;
        }
        let own = self.own_positional();
        let offered: Vec<PositionalChannel> = PositionalChannel::ALL
            .into_iter()
            .filter(|c| c.slot(&encoding).is_some() && !own.contains(c))
            .collect();
        let taken = self.mark.inherited_positional(&own, &offered);
        for channel in offered.into_iter().filter(|c| !taken.contains(c)) {
            *channel.slot_mut(&mut encoding) = None;
        }
        encoding
    }

    /// This layer's own encoding without the positional channels Python
    /// filled that its mark does not draw: the channels it binds itself, as
    /// its completed encoding ([`Self::completed_encoding`]) keeps them. The
    /// read a per-layer domain loop takes, so a channel the mark never draws
    /// widens no axis and judges no null band.
    pub(crate) fn drawn_own_encoding(&self, parent: &Encoding) -> Encoding {
        let completed = self.completed_encoding(parent);
        let mut encoding = self.encoding.clone();
        for channel in PositionalChannel::ALL {
            if channel.slot(&completed).is_none() {
                *channel.slot_mut(&mut encoding) = None;
            }
        }
        encoding
    }

    /// The coordinate system this layer's marks are placed in: its own when it
    /// declares one, else the chart's.
    ///
    /// The single expression of the fallback. `LayerPrepared` resolves its
    /// `coord` slot through this call, so every render-side consumer reads an
    /// already-resolved answer, and every spec-side reader that works from
    /// `spec.layers` directly (they run before `prepare`'s layers exist) calls it
    /// rather than spelling `.or()` themselves.
    ///
    /// The set of those readers is exactly this method's call sites — stated as a
    /// rule, not as a list, because the list rotted inside four commits: it named
    /// `numeric_domain_union` and `positional_binding`, and
    /// `scale_resolve::domain::positional_category_union` (added between the two
    /// commits, and reading the CHART's flip for every layer) was neither in it
    /// nor converted alongside them. `rg effective_coord` is the enumeration.
    /// A new reader of `spec.layers` that asks about coordinate space belongs
    /// here; one that reads the chart's `coord` directly for a per-layer question
    /// is the bug this method exists to make unspellable.
    pub fn effective_coord<'a>(
        &'a self,
        chart_coord: Option<&'a crate::spec::coord::CoordKind>,
    ) -> Option<&'a crate::spec::coord::CoordKind> {
        self.coord.as_ref().or(chart_coord)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::EncodingSpec;

    #[test]
    fn layer_round_trips_minimal() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_round_trips_with_encoding() {
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_round_trips_with_mark_style() {
        use crate::spec::mark_style::MarkKwargsSpec;
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            mark_style: Some(MarkKwargsSpec {
                size: Some(50.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_data_source_round_trip_some() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            data_source: Some("box".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            json.contains(r#""data_source":"box""#),
            "expected data_source in JSON: {json}"
        );
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    /// A key `Layer` does not have refuses, naming it, where it used to be
    /// dropped: a layer-level `blend` rendered as `normal` with no error.
    #[test]
    fn an_unknown_layer_key_refuses() {
        let err = serde_json::from_value::<Layer>(serde_json::json!({"mark": "image", "blend": "additive"}))
            .unwrap_err();
        assert!(err.to_string().starts_with("unknown field `blend`, expected one of"), "{err}");
        let blended = serde_json::from_value::<Layer>(serde_json::json!({"mark": "image", "mark_style": {"blend": "additive"}}))
            .unwrap();
        assert_eq!(blended.mark_style.and_then(|s| s.blend).as_deref(), Some("additive"));
    }

    #[test]
    fn layer_position_round_trips() {
        use crate::spec::position::PositionAdjust;
        let layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding::default(),
            position: Some(PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 }),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(json.contains(r#""position""#));
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_position_none_omits_from_json() {
        let layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding::default(),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(!json.contains("position"), "position=None must be omitted: {json}");
    }

    #[test]
    fn layer_data_source_none_omits_from_json() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            !json.contains("data_source"),
            "data_source=None must be omitted: {json}"
        );
    }

    #[test]
    fn layer_independent_y_defaults_false_when_absent_on_wire() {
        // Deserializing a pre-#52 layer dict (no `independent_y` key at all)
        // must default to false — absolute wire back-compat.
        let json = r#"{"mark":"point"}"#;
        let layer: Layer = serde_json::from_str(json).unwrap();
        assert!(!layer.independent_y);
    }

    #[test]
    fn layer_independent_y_round_trips_true() {
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding::default(),
            independent_y: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            json.contains(r#""independent_y":true"#),
            "expected independent_y:true in JSON: {json}"
        );
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_independent_y_round_trips_false() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            !json.contains("independent_y"),
            "independent_y=false must be omitted from the wire: {json}"
        );
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
        assert!(!parsed.independent_y);
    }

    // --- `#172`(1): per-layer coord ---------------------------------------

    fn point_layer() -> Layer {
        Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        }
    }

    /// The additive contract: a layer declaring no coord emits the bytes it
    /// emitted before the field existed. Asserted on the whole payload, not just
    /// on the absence of the key, because the point is byte-stability.
    #[test]
    fn layer_with_no_coord_serializes_without_the_key() {
        let json = serde_json::to_string(&point_layer()).unwrap();
        assert_eq!(json, r#"{"mark":"point","encoding":{}}"#);
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, point_layer());
        assert!(parsed.coord.is_none());
    }

    /// A pre-`#172` payload (no `coord` key at all) deserializes to "inherit".
    #[test]
    fn layer_coord_defaults_to_none_when_absent_on_wire() {
        let layer: Layer = serde_json::from_str(r#"{"mark":"point"}"#).unwrap();
        assert!(layer.coord.is_none());
    }

    #[test]
    fn layer_coord_round_trips_every_placement_kind() {
        use crate::spec::coord::{CoordKind, PolarDirection, PolarThetaChannel};
        for coord in [
            CoordKind::Flip,
            CoordKind::default(),
            CoordKind::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                direction: PolarDirection::Clockwise,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
            },
        ] {
            let layer = Layer { coord: Some(coord.clone()), ..point_layer() };
            let json = serde_json::to_string(&layer).unwrap();
            assert!(json.contains(r#""coord":{"kind":"#), "expected a tagged coord: {json}");
            assert_eq!(serde_json::from_str::<Layer>(&json).unwrap(), layer);
        }
    }

    /// The fallback every consumer reads through: the layer's own coord wins,
    /// and a layer declaring none takes the chart's. Both directions are
    /// asserted because a one-directional implementation (`self.coord` only, or
    /// `chart` only) would still satisfy half the calls and silently break the
    /// other half — a layer's polar projection, or every unflipped layer of a
    /// flipped chart.
    #[test]
    fn effective_coord_prefers_the_layers_own_then_the_charts() {
        use crate::spec::coord::CoordKind;
        let chart = CoordKind::Flip;
        let own = CoordKind::default();

        let declared = Layer { coord: Some(own.clone()), ..point_layer() };
        assert_eq!(declared.effective_coord(Some(&chart)), Some(&own));
        assert_eq!(declared.effective_coord(None), Some(&own));

        let inheriting = point_layer();
        assert_eq!(inheriting.effective_coord(Some(&chart)), Some(&chart));
        assert_eq!(inheriting.effective_coord(None), None);
    }

    /// `inherited_positional` round-trips all four positional channel names,
    /// each as its lowercase wire string, and is omitted entirely when
    /// empty.
    #[test]
    fn inherited_positional_round_trips_every_channel() {
        let layer = Layer {
            inherited_positional: vec![
                PositionalChannel::X,
                PositionalChannel::Y,
                PositionalChannel::X2,
                PositionalChannel::Y2,
            ],
            ..point_layer()
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            json.contains(r#""inherited_positional":["x","y","x2","y2"]"#),
            "expected the four lowercase channel strings: {json}"
        );
        assert_eq!(serde_json::from_str::<Layer>(&json).unwrap(), layer);

        let empty = point_layer();
        let json = serde_json::to_string(&empty).unwrap();
        assert!(
            !json.contains("inherited_positional"),
            "an empty list must be omitted from the wire: {json}"
        );
    }

    /// A channel name outside `{x, y, x2, y2}` is a deserialize error, not a
    /// silently-dropped or silently-accepted value.
    #[test]
    fn inherited_positional_refuses_an_unknown_channel_name() {
        let err = serde_json::from_value::<Layer>(serde_json::json!({
            "mark": "point",
            "inherited_positional": ["color"],
        }))
        .unwrap_err();
        assert!(
            err.to_string().contains("unknown variant `color`"),
            "expected an unknown-variant error naming 'color': {err}"
        );
    }

    /// `reads_chart_data` is read from the wire when set and omitted when
    /// false, so a layer that does not read its chart's data through a named
    /// output serializes exactly as before the field existed.
    #[test]
    fn reads_chart_data_round_trips_and_is_omitted_when_false() {
        let flagged: Layer = serde_json::from_value(serde_json::json!({
            "mark": "point", "data_source": "rows", "reads_chart_data": true,
        }))
        .expect("the flag deserializes");
        let json = serde_json::to_value(&flagged).unwrap();
        assert_eq!(json["reads_chart_data"], serde_json::json!(true));
        assert_eq!(serde_json::from_value::<Layer>(json).unwrap(), flagged);

        let plain: Layer =
            serde_json::from_value(serde_json::json!({"mark": "point"})).unwrap();
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json.get("reads_chart_data").is_none(), "false is omitted: {json}");
    }

    /// `inherited_channels` round-trips under the wire keys `Encoding` itself
    /// uses, and is omitted when empty so a layer that inherited nothing onto
    /// its own encoding serializes exactly as before the field existed.
    #[test]
    fn inherited_channels_round_trips_and_is_omitted_when_empty() {
        let listed = Layer {
            inherited_channels: vec![InheritedChannel::Color, InheritedChannel::StrokeWidth],
            ..Default::default()
        };
        let json = serde_json::to_string(&listed).unwrap();
        assert!(
            json.contains(r#""inherited_channels":["color","stroke_width"]"#),
            "expected the Encoding wire keys: {json}"
        );
        assert_eq!(serde_json::from_str::<Layer>(&json).unwrap(), listed);

        let plain = serde_json::to_value(Layer::default()).unwrap();
        assert!(plain.get("inherited_channels").is_none(), "empty is omitted: {plain}");
    }

    /// A positional or misspelt name is a deserialize error, not a silently
    /// ignored entry: the positional channels have their own list.
    #[test]
    fn inherited_channels_refuses_a_name_that_is_not_a_non_positional_channel() {
        for name in ["x", "strokewidth"] {
            let err = serde_json::from_value::<Layer>(serde_json::json!({
                "mark": "point",
                "inherited_channels": [name],
            }))
            .unwrap_err();
            assert!(
                err.to_string().contains(&format!("unknown variant `{name}`")),
                "expected an unknown-variant error naming {name:?}: {err}"
            );
        }
    }

    /// The channel names are exactly `Encoding`'s non-positional wire keys.
    /// The encoding literal lists every field without a `..Default` spread, so
    /// a new `Encoding` channel stops this test compiling until it is given a
    /// variant here and in `InheritedChannel`.
    #[test]
    fn inherited_channels_name_every_non_positional_encoding_key() {
        let spec = || Some(EncodingSpec { field: "a".into(), ..Default::default() });
        let every_channel = Encoding {
            x: spec(),
            y: spec(),
            color: spec(),
            size: spec(),
            shape: spec(),
            opacity: spec(),
            x2: spec(),
            y2: spec(),
            text: spec(),
            tooltip: spec(),
            tooltip_fields: spec().map(|s| vec![s]),
            href: spec(),
            description: spec(),
            key: spec(),
            url: spec(),
            stroke_width: spec(),
            stroke_opacity: spec(),
            stroke_dash: spec(),
            angle: spec(),
            fill_opacity: spec(),
            group: spec(),
        };
        // `group` is a grouping key, not a channel: never inherited.
        let encoding_keys: std::collections::BTreeSet<String> = serde_json::to_value(&every_channel)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| !["x", "y", "x2", "y2", "group"].contains(&k.as_str()))
            .cloned()
            .collect();
        let channel_names: std::collections::BTreeSet<String> = [
            InheritedChannel::Color,
            InheritedChannel::Size,
            InheritedChannel::Shape,
            InheritedChannel::Opacity,
            InheritedChannel::Text,
            InheritedChannel::Tooltip,
            InheritedChannel::TooltipFields,
            InheritedChannel::Href,
            InheritedChannel::Description,
            InheritedChannel::Key,
            InheritedChannel::Url,
            InheritedChannel::StrokeWidth,
            InheritedChannel::StrokeOpacity,
            InheritedChannel::StrokeDash,
            InheritedChannel::Angle,
            InheritedChannel::FillOpacity,
        ]
        .iter()
        .map(|c| serde_json::to_value(c).unwrap().as_str().unwrap().to_owned())
        .collect();
        assert_eq!(channel_names, encoding_keys);
        let listed: std::collections::BTreeSet<String> =
            InheritedChannel::ALL.iter().map(|c| c.as_str().to_owned()).collect();
        assert_eq!(listed, encoding_keys, "ALL names every channel by its wire name");
        for channel in InheritedChannel::ALL {
            assert_eq!(serde_json::to_value(channel).unwrap().as_str(), Some(channel.as_str()));
            assert_eq!(InheritedChannel::from_wire(channel.as_str()), Some(channel));
        }
        assert_eq!(InheritedChannel::from_wire("x"), None);
    }

    fn bound(field: &str) -> Option<crate::spec::encoding::EncodingSpec> {
        Some(crate::spec::encoding::EncodingSpec { field: field.into(), ..Default::default() })
    }

    /// The chart-level parent the completion tests inherit from: every
    /// positional channel bound.
    fn spanning_parent() -> Encoding {
        Encoding { x: bound("px"), y: bound("py"), x2: bound("px2"), y2: bound("py2"), ..Default::default() }
    }

    fn layer_of(mark: Mark, encoding: Encoding) -> Layer {
        Layer { mark, encoding, ..Default::default() }
    }

    /// A line draws no span: completion gives it the parent's `x`/`y` and
    /// drops the `x2`/`y2` it would never draw. The rejected reading,
    /// `inherit_from` alone, keeps both span channels.
    #[test]
    fn completion_drops_the_span_channels_a_line_does_not_draw() {
        let line = layer_of(Mark::Line, Encoding::default());
        let completed = line.completed_encoding(&spanning_parent());
        assert_eq!((completed.x, completed.y), (bound("px"), bound("py")));
        assert_eq!((completed.x2, completed.y2), (None, None));
    }

    /// A rule writing only `y`, under a parent with no span, is a one-axis
    /// reference line: it takes no inherited `x`. A rule under a spanning
    /// parent takes the spans and so keeps assembling its ranged shape.
    #[test]
    fn completion_gives_a_one_axis_rule_no_opposite_channel() {
        let rule = layer_of(Mark::Rule, Encoding { y: bound("own"), ..Default::default() });
        let flat_parent = Encoding { x: bound("px"), y: bound("py"), ..Default::default() };
        let completed = rule.completed_encoding(&flat_parent);
        assert_eq!((completed.x, completed.y), (None, bound("own")));
        let ranged = rule.completed_encoding(&spanning_parent());
        assert_eq!((ranged.x, ranged.x2, ranged.y2), (bound("px"), bound("px2"), bound("py2")));
    }

    /// A positional channel Python filled and listed in
    /// `inherited_positional` is inherited: a one-axis rule on its own output
    /// drops it. Unlisted, the same value is the layer's own and stays.
    #[test]
    fn completion_treats_a_listed_filled_channel_as_inherited() {
        let filled = Encoding { x: bound("own"), y: bound("filled"), ..Default::default() };
        let mut rule = Layer { data_source: Some("rows".into()), ..layer_of(Mark::Rule, filled) };
        rule.inherited_positional = vec![PositionalChannel::Y];
        assert_eq!(rule.completed_encoding(&Encoding::default()).y, None);
        rule.inherited_positional.clear();
        assert_eq!(rule.completed_encoding(&Encoding::default()).y, bound("filled"));
    }
}
