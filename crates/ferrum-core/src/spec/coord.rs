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

    #[test]
    fn flip_maps_to_scene_cartesian() {
        let scene = to_scene_coord(&CoordKind::Flip, 200.0);
        assert!(matches!(scene, ferrum_scene::CoordKind::Cartesian { .. }));
    }
}
