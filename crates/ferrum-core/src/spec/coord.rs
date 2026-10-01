use serde::{Deserialize, Serialize};

// Re-export scene-side enums so spec→scene conversion needs no re-mapping.
pub use ferrum_scene::{GeoProjection, PolarDirection, PolarThetaChannel};

/// Coordinate system for the spec (input) layer.
///
/// `Flip` stays here (no scene equivalent) — Python handles channel swapping
/// before emitting the spec; scene_build.rs maps `Flip` → scene `Cartesian`.
/// All other variants mirror `ferrum_scene::CoordKind` field-for-field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CoordKind {
    Cartesian {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_domain: Option<(f64, f64)>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_domain: Option<(f64, f64)>,
        #[serde(default = "default_true")]
        expand: bool,
        #[serde(default = "default_true")]
        clip: bool,
    },
    Flip,
    Fixed {
        #[serde(default = "default_ratio")]
        ratio: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_domain: Option<(f64, f64)>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_domain: Option<(f64, f64)>,
        #[serde(default = "default_true")]
        expand: bool,
        #[serde(default = "default_true")]
        clip: bool,
    },
    Polar {
        #[serde(default = "default_polar_theta")]
        theta: PolarThetaChannel,
        #[serde(default)]
        start_angle: f64,
        #[serde(default = "default_polar_direction")]
        direction: PolarDirection,
        #[serde(default)]
        inner_radius: f64,
        /// `None` means "fill the panel" — resolved to `outer_radius_px` in `to_scene_coord`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outer_radius: Option<f64>,
        /// Angular gap between adjacent slices in radians (default 0).
        #[serde(default)]
        pad_angle: f64,
    },
    Geo {
        #[serde(default = "default_geo_projection")]
        projection: GeoProjection,
    },
}

fn default_true() -> bool { true }
fn default_ratio() -> f64 { 1.0 }
fn default_polar_theta() -> PolarThetaChannel { PolarThetaChannel::X }
fn default_polar_direction() -> PolarDirection { PolarDirection::Clockwise }
fn default_geo_projection() -> GeoProjection { GeoProjection::Mercator }

impl Default for CoordKind {
    fn default() -> Self {
        CoordKind::Cartesian {
            x_domain: None,
            y_domain: None,
            expand: true,
            clip: true,
        }
    }
}

/// Whether this coordinate system replaces the Cartesian x/y axes entirely.
///
/// `Polar` draws a circular boundary with radial ticks and `Geo` draws none at
/// all; in both cases `render::scene_build` emits no Cartesian axis or gridline
/// nodes for the panel. Because nothing is drawn, nothing may be reserved:
/// `render::prepare::build_axes` folds this into `AxesInput::show_x`/`show_y`
/// so `layout` never sets aside a label band or title gutter for an axis a
/// projected coord suppresses (`#94`, the polar half of `#182`'s rule — a
/// gutter exists if and only if something is drawn in it).
///
/// One owner for the question, asked by both subsystems: layout reserves off
/// it, `scene_build` suppresses off it. Two hand-written `matches!` arms is how
/// the reservation and the emission drifted apart in the first place. Takes an
/// `Option` because `ChartSpec::coord` is optional and absent means Cartesian.
pub fn suppresses_cartesian_axes(coord: Option<&CoordKind>) -> bool {
    matches!(coord, Some(CoordKind::Polar { .. }) | Some(CoordKind::Geo { .. }))
}

/// Whether this coordinate system exchanges the positional channels.
///
/// One owner for the question, asked by the five stages that must agree about
/// which slot a channel occupies: `prepare::build_layers` applies the swap,
/// `scale_resolve`'s `numeric_domain_union` / `positional_binding` /
/// `validate_drawing_layers_can_plot` read a layer's encoding in the resolved
/// axis's space, and `scene_build`'s per-layer loop adjusts positions and names
/// refused channels in it.
///
/// It exists because `#172`(1) made the question per LAYER: those five sites
/// each carried their own `matches!(spec.coord, Some(CoordKind::Flip))` against
/// the CHART's coord, which is the wrong subject for a layer that declares its
/// own — and five hand-copies of one predicate is how the swap and its
/// consumers would drift apart one edit at a time. Callers pass
/// `Layer::effective_coord(...)` for a per-layer question and
/// `spec.coord.as_ref()` for a chart-level one (which axis the chrome
/// describes, which channel a chart-level refusal names).
///
/// Takes an `Option` because both slots are optional and absent means
/// Cartesian.
pub fn flips_positional_channels(coord: Option<&CoordKind>) -> bool {
    matches!(coord, Some(CoordKind::Flip))
}

/// The user-facing constructor name for a coord, for refusal messages: the
/// caller wrote `fm.CoordFixed(...)`, not `CoordKind::Fixed { .. }`.
pub(crate) fn public_spelling(coord: &CoordKind) -> &'static str {
    match coord {
        CoordKind::Cartesian { .. } => "fm.CoordCartesian",
        CoordKind::Flip => "fm.CoordFlip",
        CoordKind::Fixed { .. } => "fm.CoordFixed",
        CoordKind::Polar { .. } => "fm.CoordPolar",
        CoordKind::Geo { .. } => "fm.CoordGeo",
    }
}

/// What *this* coord declares that belongs to the whole PANEL rather than to
/// one layer's marks — `None` when the coord is entirely a per-layer placement
/// rule and can therefore ride a single layer of a shared panel.
///
/// `#172`(1)'s refusal predicate. A per-layer coord is honorable exactly when
/// everything it declares is about where that layer's own marks go: the
/// projection (`Polar`/`Geo`, applied per mark batch) or the channel exchange
/// (`Flip`, applied to that layer's encoding). An axis domain, a padding
/// choice, a clip decision, or an aspect ratio is a property of the panel every
/// layer shares, so there is no honest way to apply it to one of them — the
/// render refuses and names it (`RenderError::LayerCoordUnplaceable`) instead
/// of rendering a chart with the declaration silently dropped, which is what
/// the merged-flat path did for EVERY per-layer coord before this.
///
/// The returned string is the user-facing spelling of the property, for the
/// refusal message: the caller wrote `fm.CoordFixed(ratio=…)`, not
/// `CoordKind::Fixed { ratio }`.
pub fn panel_scoped_property(coord: &CoordKind) -> Option<&'static str> {
    match coord {
        // A plain `CoordCartesian()` declares nothing the panel owns — it is
        // the default, spelled out, and is honored as the no-op it is (which
        // is load-bearing: Python stamps exactly this on a secondary layer
        // whose own chart declared no coord under a projected chart-level one,
        // so that layer opts out of the projection without claiming anything).
        CoordKind::Cartesian { x_domain, y_domain, expand, clip } => {
            cartesian_panel_property(*x_domain, *y_domain, *expand, *clip)
        }
        // Fixed is Cartesian plus an aspect ratio, and the ratio alone is
        // already panel-scoped, so the whole variant is — named as the ratio
        // whichever other field the caller also set.
        CoordKind::Fixed { .. } => Some("a fixed aspect ratio (fm.CoordFixed(ratio=...))"),
        CoordKind::Flip | CoordKind::Polar { .. } | CoordKind::Geo { .. } => None,
    }
}

/// The panel-scoped half of a Cartesian coord, in the order a caller would
/// most likely have written it. Split out so the `Fixed` arm above reads as
/// the one-line statement it is rather than repeating four checks it answers
/// the same way regardless.
fn cartesian_panel_property(
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    expand: bool,
    clip: bool,
) -> Option<&'static str> {
    if x_domain.is_some() {
        return Some("an x-axis domain (xlim=)");
    }
    if y_domain.is_some() {
        return Some("a y-axis domain (ylim=)");
    }
    if !expand {
        return Some("axis padding (expand=False)");
    }
    if !clip {
        return Some("panel clipping (clip=False)");
    }
    None
}

/// Convert a spec-side `CoordKind` to the scene-side equivalent.
///
/// `Flip` maps to Cartesian — Python handles the channel swap before this point.
pub fn to_scene_coord(coord: &CoordKind, outer_radius_px: f64) -> ferrum_scene::CoordKind {
    match coord {
        CoordKind::Cartesian { x_domain, y_domain, expand, clip } => {
            ferrum_scene::CoordKind::Cartesian {
                x_domain: *x_domain,
                y_domain: *y_domain,
                expand: *expand,
                clip: *clip,
                // Per-slot y-domains (secondary-y-axis, GH #52 Task 8) are
                // injected downstream in `scene_build.rs`, which has access to
                // the per-panel resolved scales this spec-level conversion
                // does not see.
                y_domains: Vec::new(),
            }
        }
        CoordKind::Flip => ferrum_scene::CoordKind::Cartesian {
            x_domain: None,
            y_domain: None,
            expand: true,
            clip: true,
            y_domains: Vec::new(),
        },
        CoordKind::Fixed { ratio, x_domain, y_domain, expand, clip } => {
            ferrum_scene::CoordKind::Fixed {
                ratio: *ratio,
                x_domain: *x_domain,
                y_domain: *y_domain,
                expand: *expand,
                clip: *clip,
            }
        }
        CoordKind::Polar { theta, start_angle, direction, inner_radius, outer_radius, .. } => {
            ferrum_scene::CoordKind::Polar {
                theta: *theta,
                start_angle: *start_angle,
                direction: *direction,
                inner_radius: *inner_radius,
                outer_radius: outer_radius.unwrap_or(outer_radius_px),
            }
        }
        CoordKind::Geo { projection } => {
            ferrum_scene::CoordKind::Geo { projection: *projection }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_kind_round_trip_cartesian_bare() {
        let json = r#"{"kind":"cartesian"}"#;
        let parsed: CoordKind = serde_json::from_str(json).unwrap();
        assert!(matches!(parsed, CoordKind::Cartesian { expand: true, clip: true, .. }));
    }

    #[test]
    fn coord_kind_round_trip_flip() {
        let c = CoordKind::Flip;
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#"{"kind":"flip"}"#);
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn coord_kind_round_trip_fixed() {
        let c = CoordKind::Fixed { ratio: 1.0, x_domain: None, y_domain: None, expand: true, clip: true };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn coord_kind_round_trip_polar() {
        let c = CoordKind::Polar {
            theta: PolarThetaChannel::X,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: None,
            pad_angle: 0.0,
        };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn coord_kind_polar_explicit_outer_radius() {
        let c = CoordKind::Polar {
            theta: PolarThetaChannel::X,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 20.0,
            outer_radius: Some(100.0),
            pad_angle: 0.0,
        };
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("outer_radius"));
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    /// `#182`/`#94`: exactly the projected coords suppress the Cartesian axes,
    /// and `None` (no `coord=` at all) does not. Enumerated variant by variant
    /// rather than sampled, because both consumers — `layout`'s band
    /// reservation and `scene_build`'s node emission — read this one answer, so
    /// a wrong `true` silently deletes a real chart's axes and a wrong `false`
    /// restores `#94`'s phantom margins.
    #[test]
    fn only_projected_coords_suppress_the_cartesian_axes() {
        let polar = CoordKind::Polar {
            theta: PolarThetaChannel::X,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: None,
            pad_angle: 0.0,
        };
        let geo = CoordKind::Geo { projection: GeoProjection::Mercator };
        assert!(suppresses_cartesian_axes(Some(&polar)));
        assert!(suppresses_cartesian_axes(Some(&geo)));

        for drawn in [
            CoordKind::default(),
            CoordKind::Flip,
            CoordKind::Fixed {
                ratio: 1.0,
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
        ] {
            assert!(
                !suppresses_cartesian_axes(Some(&drawn)),
                "{drawn:?} draws Cartesian axes and must keep its gutters"
            );
        }
        assert!(!suppresses_cartesian_axes(None), "no coord= means Cartesian");
    }

    #[test]
    fn coord_kind_round_trip_geo() {
        let c = CoordKind::Geo { projection: GeoProjection::Mercator };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    /// `#172`(1): exactly `Flip` exchanges the positional channels, and `None`
    /// (no coord) does not. Enumerated variant by variant rather than sampled,
    /// because five stages read this one answer to decide which slot a channel
    /// occupies: a wrong `true` swaps a layer's encoding nothing else swapped,
    /// and a wrong `false` leaves a flipped layer reading its own channels in
    /// the wrong space.
    #[test]
    fn only_flip_exchanges_the_positional_channels() {
        assert!(flips_positional_channels(Some(&CoordKind::Flip)));
        for unswapped in [
            CoordKind::default(),
            CoordKind::Fixed {
                ratio: 1.0,
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
            CoordKind::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                direction: PolarDirection::Clockwise,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
            },
            CoordKind::Geo { projection: GeoProjection::Mercator },
        ] {
            assert!(
                !flips_positional_channels(Some(&unswapped)),
                "{unswapped:?} does not exchange x and y"
            );
        }
        assert!(!flips_positional_channels(None), "no coord= means Cartesian");
    }

    /// `#172`(1)'s refusal predicate, both arms. The placement-only coords must
    /// return `None` — a `Some` for any of them would refuse the very
    /// compositions this field exists to honor — and each panel-scoped field
    /// must be named, since the refusal message quotes it back to the caller.
    #[test]
    fn panel_scoped_property_names_only_what_the_panel_owns() {
        let placement = [
            CoordKind::Flip,
            CoordKind::default(),
            CoordKind::Polar {
                theta: PolarThetaChannel::Y,
                start_angle: 1.0,
                direction: PolarDirection::CounterClockwise,
                inner_radius: 20.0,
                outer_radius: Some(100.0),
                pad_angle: 0.1,
            },
            CoordKind::Geo { projection: GeoProjection::Mercator },
        ];
        for coord in &placement {
            assert_eq!(
                panel_scoped_property(coord),
                None,
                "{coord:?} declares only where this layer's marks go"
            );
        }

        let cartesian = |x_domain, y_domain, expand, clip| CoordKind::Cartesian {
            x_domain,
            y_domain,
            expand,
            clip,
        };
        assert_eq!(
            panel_scoped_property(&cartesian(Some((0.0, 1.0)), None, true, true)),
            Some("an x-axis domain (xlim=)")
        );
        assert_eq!(
            panel_scoped_property(&cartesian(None, Some((0.0, 1.0)), true, true)),
            Some("a y-axis domain (ylim=)")
        );
        assert_eq!(
            panel_scoped_property(&cartesian(None, None, false, true)),
            Some("axis padding (expand=False)")
        );
        assert_eq!(
            panel_scoped_property(&cartesian(None, None, true, false)),
            Some("panel clipping (clip=False)")
        );
        // Fixed is panel-scoped through its ratio alone, so even the
        // otherwise-default construction refuses.
        assert_eq!(
            panel_scoped_property(&CoordKind::Fixed {
                ratio: 1.0,
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            }),
            Some("a fixed aspect ratio (fm.CoordFixed(ratio=...))")
        );
    }

    #[test]
    fn flip_maps_to_scene_cartesian() {
        let scene = to_scene_coord(&CoordKind::Flip, 200.0);
        assert!(matches!(scene, ferrum_scene::CoordKind::Cartesian { .. }));
    }
}
