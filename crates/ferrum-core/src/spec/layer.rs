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
    /// [`LayerPrepared::from_chart_and_layer`] to
    /// decide `x_is_own`/`y_is_own`: a channel listed here is treated
    /// exactly as the standalone-inherited case the own-span normalization
    /// (`scene_build`'s rule/tick own-span rule) already clears, not as a
    /// layer-declared one. No other Rust reader looks at this field — every
    /// other consumer (domain unions, axis titles, legends) reads
    /// `encoding` itself, which already carries the filled value, exactly
    /// as it reads a flat operand's own snapshot. Empty (the default) for
    /// every layer that declares its own channels or inherits none,
    /// including every pre-existing spec, so the field is omitted from the
    /// wire then and round-trips byte-identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inherited_positional: Vec<PositionalChannel>,
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Whether this layer resolves its own y-scale slot instead of sharing
    /// the primary (layer 0) y-scale. `false` (absent on the wire) = shared,
    /// today's behavior. Layer 0 is always the primary/left axis regardless
    /// of this flag (spec §6, GH #52). `false` is omitted when serializing so
    /// pre-#52 layer wire JSON stays byte-identical.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub independent_y: bool,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            name: None,
            independent_y: false,
            coord: None,
            entry: None,
        }
    }
}

impl Layer {
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
}
