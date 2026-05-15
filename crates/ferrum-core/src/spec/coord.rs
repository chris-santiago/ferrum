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
            }
        }
        CoordKind::Flip => ferrum_scene::CoordKind::Cartesian {
            x_domain: None,
            y_domain: None,
            expand: true,
            clip: true,
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
