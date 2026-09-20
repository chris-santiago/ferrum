use serde::{Deserialize, Serialize};

use crate::spec::encoding::Encoding;
use crate::spec::mark::Mark;
use crate::transform::core::TransformSpec;

/// A single layer within a multi-layer ChartSpec. Inherits chart-level
/// encoding for any field set to None at the layer level.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub position: Option<crate::spec::position::PositionAdjust>,
    /// Pixel-level blending mode for this layer. `None` = Normal (default).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub blend: Option<ferrum_scene::BlendMode>,
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
}

impl Layer {
    /// The coordinate system this layer's marks are placed in: its own when it
    /// declares one, else the chart's.
    ///
    /// The single expression of the fallback. `LayerPrepared` resolves its
    /// `coord` slot through this call, so every render-side consumer reads an
    /// already-resolved answer, and the two spec-side readers that work from
    /// `spec.layers` directly (`scale_resolve::numeric_domain_union` and
    /// `positional_binding`, which run before `prepare`'s layers exist) call it
    /// rather than spelling `.or()` themselves.
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
        position: None, blend: None, name: None, independent_y: false, coord: None,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false, coord: None,
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
            transforms: Vec::new(),
            mark_style: Some(MarkKwargsSpec {
                size: Some(50.0),
                ..Default::default()
            }),
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false, coord: None,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: Some("box".into()),
            position: None, blend: None, name: None, independent_y: false, coord: None,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            json.contains(r#""data_source":"box""#),
            "expected data_source in JSON: {json}"
        );
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_position_round_trips() {
        use crate::spec::position::PositionAdjust;
        let layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: Some(PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 }),
            blend: None,
            name: None,
            independent_y: false, coord: None,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None, blend: None, name: None, independent_y: false, coord: None,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(!json.contains("position"), "position=None must be omitted: {json}");
    }

    #[test]
    fn layer_data_source_none_omits_from_json() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
        position: None, blend: None, name: None, independent_y: false, coord: None,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: true, coord: None,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false, coord: None,
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
            coord: None,
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
}
