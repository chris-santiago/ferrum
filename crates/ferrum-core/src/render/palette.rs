//! Categorical palettes and scheme registry.

use std::sync::LazyLock;

use super::color::{from_rgb, Color};

/// Okabe-Ito 8-color categorical palette. Lazy-initialized because palette's
/// `Srgba::new` is not const-fn and the internal struct layout (`Alpha<Rgb<...>, u8>`)
/// is not stable enough to literal-construct in a `const`. `LazyLock` (Rust 1.80+)
/// initializes on first access; cost is one-time.
pub static OKABE_ITO: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0xE6, 0x9F, 0x00), // orange
    from_rgb(0x56, 0xB4, 0xE9), // sky blue
    from_rgb(0x00, 0x9E, 0x73), // bluish green
    from_rgb(0xF0, 0xE4, 0x42), // yellow
    from_rgb(0x00, 0x72, 0xB2), // blue
    from_rgb(0xD5, 0x5E, 0x00), // vermillion
    from_rgb(0xCC, 0x79, 0xA7), // reddish purple
    from_rgb(0x00, 0x00, 0x00), // black
]);

pub static TABLEAU10: LazyLock<[Color; 10]> = LazyLock::new(|| [
    from_rgb(0x4C, 0x78, 0xA8), from_rgb(0xF5, 0x8E, 0x18),
    from_rgb(0xE4, 0x57, 0x56), from_rgb(0x72, 0xB7, 0xB2),
    from_rgb(0x54, 0xA2, 0x4B), from_rgb(0xEE, 0xCA, 0x3B),
    from_rgb(0xB2, 0x79, 0xA2), from_rgb(0xFF, 0x9D, 0xA6),
    from_rgb(0x9D, 0x75, 0x5D), from_rgb(0xBA, 0xB0, 0xAC),
]);

pub static SET1: LazyLock<[Color; 9]> = LazyLock::new(|| [
    from_rgb(0xE4, 0x1A, 0x1C), from_rgb(0x37, 0x7E, 0xB8),
    from_rgb(0x4D, 0xAF, 0x4A), from_rgb(0x98, 0x4E, 0xA3),
    from_rgb(0xFF, 0x7F, 0x00), from_rgb(0xFF, 0xFF, 0x33),
    from_rgb(0xA6, 0x56, 0x28), from_rgb(0xF7, 0x81, 0xBF),
    from_rgb(0x99, 0x99, 0x99),
]);

pub static SET2: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0x66, 0xC2, 0xA5), from_rgb(0xFC, 0x8D, 0x62),
    from_rgb(0x8D, 0xA0, 0xCB), from_rgb(0xE7, 0x8A, 0xC3),
    from_rgb(0xA6, 0xD8, 0x54), from_rgb(0xFF, 0xD9, 0x2F),
    from_rgb(0xE5, 0xC4, 0x94), from_rgb(0xB3, 0xB3, 0xB3),
]);

pub static PAIRED: LazyLock<[Color; 12]> = LazyLock::new(|| [
    from_rgb(0xA6, 0xCE, 0xE3), from_rgb(0x1F, 0x78, 0xB4),
    from_rgb(0xB2, 0xDF, 0x8A), from_rgb(0x33, 0xA0, 0x2C),
    from_rgb(0xFB, 0x9A, 0x99), from_rgb(0xE3, 0x1A, 0x1C),
    from_rgb(0xFD, 0xBF, 0x6F), from_rgb(0xFF, 0x7F, 0x00),
    from_rgb(0xCA, 0xB2, 0xD6), from_rgb(0x6A, 0x3D, 0x9A),
    from_rgb(0xFF, 0xFF, 0x99), from_rgb(0xB1, 0x59, 0x28),
]);

pub static PASTEL: LazyLock<[Color; 9]> = LazyLock::new(|| [
    from_rgb(0xFB, 0xB4, 0xAE), from_rgb(0xB3, 0xCD, 0xE3),
    from_rgb(0xCC, 0xEB, 0xC5), from_rgb(0xDE, 0xCB, 0xE4),
    from_rgb(0xFE, 0xD9, 0xA6), from_rgb(0xFF, 0xFF, 0xCC),
    from_rgb(0xE5, 0xD8, 0xBD), from_rgb(0xFD, 0xDA, 0xEC),
    from_rgb(0xF2, 0xF2, 0xF2),
]);

pub static DARK2: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0x1B, 0x9E, 0x77), from_rgb(0xD9, 0x5F, 0x02),
    from_rgb(0x75, 0x70, 0xB3), from_rgb(0xE7, 0x29, 0x8A),
    from_rgb(0x66, 0xA6, 0x1E), from_rgb(0xE6, 0xAB, 0x02),
    from_rgb(0xA6, 0x76, 0x1D), from_rgb(0x66, 0x66, 0x66),
]);

pub static PAPER_INK: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0x25, 0x63, 0xEB), from_rgb(0xDC, 0x26, 0x26),
    from_rgb(0xD4, 0xA0, 0x17), from_rgb(0x0F, 0x76, 0x6E),
    from_rgb(0x7C, 0x3A, 0xED), from_rgb(0xEA, 0x58, 0x0C),
    from_rgb(0x4B, 0x55, 0x63), from_rgb(0xDB, 0x27, 0x77),
]);

pub static SLATE_CITRUS: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0x60, 0xA5, 0xFA), from_rgb(0xA7, 0x8B, 0xFA),
    from_rgb(0xA3, 0xE6, 0x35), from_rgb(0xF5, 0x9E, 0x0B),
    from_rgb(0x34, 0xD3, 0x99), from_rgb(0xF4, 0x72, 0xB6),
    from_rgb(0xF8, 0x71, 0x71), from_rgb(0x22, 0xD3, 0xEE),
]);

pub static ARCTIC_SIGNAL: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0x02, 0x84, 0xC7), from_rgb(0x7C, 0x3A, 0xED),
    from_rgb(0xEA, 0x58, 0x0C), from_rgb(0x16, 0xA3, 0x4A),
    from_rgb(0xDC, 0x26, 0x26), from_rgb(0x08, 0x91, 0xB2),
    from_rgb(0xCA, 0x8A, 0x04), from_rgb(0xDB, 0x27, 0x77),
]);

/// Categorical scheme names recognized by [`categorical_palette`]. Source of
/// truth for theme-side validation in `binding::theme_from_dict`.
pub const CATEGORICAL_SCHEMES: &[&str] = &[
    "okabe_ito", "tableau10", "set1", "set2", "paired", "pastel", "dark2",
    "paper_ink", "slate_citrus", "arctic_signal",
];

/// Sequential/diverging scheme names recognized by `ContinuousScheme` in
/// `render/color`. Listed here so theme-side validation can accept them
/// without depending on the continuous-scale machinery. "blues" is a
/// sequential single-hue ramp; "rdbu" is a diverging red-blue scheme.
///
/// This is the full set of *continuous* (sequential OR diverging) names; it
/// matches `NamedContinuous::list()` one-for-one. `is_sequential_scheme` tests
/// membership in this set (i.e. "is this a continuous scheme?") and is kept
/// unchanged for back-compat with existing callers. To split sequential from
/// diverging for presentation, use [`DIVERGING_SCHEMES`] / [`palette_kind`].
pub const SEQUENTIAL_SCHEMES: &[&str] = &[
    "viridis", "plasma", "magma", "inferno", "cividis", "blues", "rdbu",
    "reds", "greens", "oranges", "purples",
    "cool_blue", "warm_ochre", "blue_to_red",
    "night_blue", "electric_lime", "cyan_to_amber",
    "signal_blue", "ember_orange", "blue_to_violet",
];

/// Continuous schemes that are *diverging* (two hues meeting at a neutral
/// midpoint) rather than single-direction sequential ramps. This is metadata
/// over the existing continuous registry — these names are also present in
/// [`SEQUENTIAL_SCHEMES`] (which means "continuous"); the split here only
/// classifies them for `palette_kind` and matches the partition that the
/// Python `color.py` lookup module exposes (`rdbu` plus the three custom
/// diverging maps with light/neutral midpoints).
pub const DIVERGING_SCHEMES: &[&str] = &[
    "rdbu", "blue_to_red", "cyan_to_amber", "blue_to_violet",
];

/// True when `name` is a recognized diverging continuous scheme.
pub fn is_diverging_scheme(name: &str) -> bool {
    DIVERGING_SCHEMES.contains(&name)
}

/// True when `name` is one of the recognized categorical schemes.
pub fn is_categorical_scheme(name: &str) -> bool {
    CATEGORICAL_SCHEMES.contains(&name)
}

/// True when `name` is one of the recognized sequential schemes (handled by
/// `ContinuousScheme`).
pub fn is_sequential_scheme(name: &str) -> bool {
    SEQUENTIAL_SCHEMES.contains(&name)
}

/// Look up a categorical palette by scheme name. Returns PAPER_INK when the
/// name is unknown (caller may emit a warning). Theme-level validation in
/// `binding::theme_from_dict` rejects unknown names eagerly, so callers
/// reaching this function with an unknown name are using an encoding-level
/// override (e.g. `encoding.color.scheme`) — fallback preserved for that path.
pub fn categorical_palette(name: &str) -> &'static [Color] {
    match name {
        "okabe_ito"      => &*OKABE_ITO,
        "tableau10"      => &*TABLEAU10,
        "set1"           => &*SET1,
        "set2"           => &*SET2,
        "paired"         => &*PAIRED,
        "pastel"         => &*PASTEL,
        "dark2"          => &*DARK2,
        "paper_ink"      => &*PAPER_INK,
        "slate_citrus"   => &*SLATE_CITRUS,
        "arctic_signal"  => &*ARCTIC_SIGNAL,
        _                => &*PAPER_INK,
    }
}

// --- PyO3 accessors: expose the palette registry as the single source of truth ---
//
// These functions let the Python `color.py` lookup module consume the Rust
// registry instead of hand-mirroring hex tables. They are read-only views over
// the existing registry data; they do not change the render-time color path.

use pyo3::prelude::*;

use super::color::{fmt_svg, ContinuousScheme, NamedContinuous};

/// Number of evenly-spaced stops returned by [`palette_colors`] for a
/// continuous (sequential/diverging) scheme. Endpoints are inclusive: the
/// stops are sampled at `t = i / (PALETTE_CONTINUOUS_STOPS - 1)` for
/// `i in 0..PALETTE_CONTINUOUS_STOPS`, so the first stop is `sample(0.0)` and
/// the last is `sample(1.0)`. Seven mirrors the shape of the Python tables.
const PALETTE_CONTINUOUS_STOPS: usize = 7;

/// All palette names the registry knows: the union of the categorical schemes
/// and the continuous (sequential + diverging) schemes.
///
/// Order: categorical names first (in registry order), then continuous names
/// (in registry order). Names are unique across the two sets, so no dedup is
/// needed, but the union is built defensively in case that ever changes.
#[pyfunction]
pub fn list_palettes() -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for &n in CATEGORICAL_SCHEMES {
        names.push(n.to_string());
    }
    for &n in NamedContinuous::list() {
        if !names.iter().any(|existing| existing == n) {
            names.push(n.to_string());
        }
    }
    names
}

/// Classify a palette name: `"categorical"`, `"sequential"`, `"diverging"`,
/// or `None` if the name is not in the registry.
///
/// Continuous schemes are split into sequential vs diverging via
/// [`DIVERGING_SCHEMES`]; everything continuous that is not diverging is
/// reported as `"sequential"`.
#[pyfunction]
pub fn palette_kind(name: &str) -> Option<String> {
    if is_categorical_scheme(name) {
        Some("categorical".to_string())
    } else if is_diverging_scheme(name) {
        Some("diverging".to_string())
    } else if is_sequential_scheme(name) {
        Some("sequential".to_string())
    } else {
        None
    }
}

/// Hex color stops for a palette name, or `None` if the name is unknown.
///
/// - **Categorical** schemes return their discrete fixed stops (the exact
///   colors the render path uses), formatted with the canonical SVG hex
///   formatter (`#rrggbb`, lowercase). These are byte-identical to the colors
///   used by `categorical_palette`.
/// - **Continuous** schemes (sequential/diverging) return
///   [`PALETTE_CONTINUOUS_STOPS`] evenly-spaced samples from the render-path
///   interpolator (`ContinuousScheme::sample`), endpoints inclusive
///   (`t = i / (PALETTE_CONTINUOUS_STOPS - 1)`). These are the *render-truth*
///   colors. For colorous-backed maps (viridis, plasma, magma, inferno,
///   cividis, blues, reds, greens, oranges, purples, rdbu) they are sampled
///   from `colorous`, so they will not necessarily equal any hand-picked
///   approximation. For arbitrary-`t` sampling that matches the render path
///   exactly, use [`palette_sample`].
#[pyfunction]
pub fn palette_colors(name: &str) -> Option<Vec<String>> {
    if is_categorical_scheme(name) {
        return Some(
            categorical_palette(name)
                .iter()
                .map(|&c| fmt_svg(c))
                .collect(),
        );
    }
    let scheme = NamedContinuous::from_name(name).map(ContinuousScheme::Named)?;
    let last = (PALETTE_CONTINUOUS_STOPS - 1) as f64;
    Some(
        (0..PALETTE_CONTINUOUS_STOPS)
            .map(|i| {
                let t = i as f64 / last;
                fmt_svg(scheme.sample(t))
            })
            .collect(),
    )
}

/// Sample a continuous scheme at `t` (clamped to `[0, 1]`), returning the hex
/// color, or `None` if `name` is not a continuous scheme.
///
/// This is the exact value the render path produces for that `(scheme, t)`
/// pair: it delegates to `ContinuousScheme::sample` and formats with the
/// canonical SVG hex formatter. Categorical names return `None` (they are
/// discrete, not sampled). Use this when Python needs to reproduce the
/// render-path interpolation for an arbitrary `t`.
#[pyfunction]
pub fn palette_sample(name: &str, t: f64) -> Option<String> {
    let scheme = NamedContinuous::from_name(name).map(ContinuousScheme::Named)?;
    Some(fmt_svg(scheme.sample(t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_palette_returns_named_palette() {
        assert!(std::ptr::eq(categorical_palette("tableau10").as_ptr(), TABLEAU10.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("set1").as_ptr(), SET1.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("dark2").as_ptr(), DARK2.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("paper_ink").as_ptr(), PAPER_INK.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("slate_citrus").as_ptr(), SLATE_CITRUS.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("arctic_signal").as_ptr(), ARCTIC_SIGNAL.as_ptr()));
    }

    #[test]
    fn categorical_palette_unknown_falls_back_to_paper_ink() {
        assert!(std::ptr::eq(categorical_palette("nonexistent").as_ptr(), (&*PAPER_INK).as_ptr()));
    }

    #[test]
    fn each_named_palette_has_at_least_8_colors() {
        for name in CATEGORICAL_SCHEMES {
            assert!(categorical_palette(name).len() >= 8, "{name} has < 8 colors");
        }
    }

    #[test]
    fn categorical_schemes_const_matches_match_arms() {
        for name in CATEGORICAL_SCHEMES {
            let p = categorical_palette(name).as_ptr();
            let fallback = (&*PAPER_INK).as_ptr();
            if *name == "paper_ink" {
                assert!(std::ptr::eq(p, fallback));
            } else {
                assert!(
                    !std::ptr::eq(p, fallback),
                    "{name} resolved to PAPER_INK fallback",
                );
            }
        }
    }

    #[test]
    fn is_scheme_predicates_partition() {
        for name in CATEGORICAL_SCHEMES {
            assert!(is_categorical_scheme(name));
            assert!(!is_sequential_scheme(name));
        }
        for name in SEQUENTIAL_SCHEMES {
            assert!(is_sequential_scheme(name));
            assert!(!is_categorical_scheme(name));
        }
        assert!(!is_categorical_scheme("nonexistent"));
        assert!(!is_sequential_scheme("nonexistent"));
    }

    #[test]
    fn diverging_schemes_are_a_subset_of_continuous() {
        for name in DIVERGING_SCHEMES {
            assert!(
                is_sequential_scheme(name),
                "{name} is diverging but not in the continuous set",
            );
            assert!(is_diverging_scheme(name));
        }
    }

    #[test]
    fn list_palettes_covers_known_names() {
        let all = list_palettes();
        for name in ["okabe_ito", "tableau10", "viridis", "rdbu", "blues", "cool_blue"] {
            assert!(all.iter().any(|n| n == name), "{name} missing from list_palettes()");
        }
        // Union has no duplicates.
        let mut sorted = all.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "list_palettes() contains duplicates");
        // Size = categorical + continuous (names are disjoint).
        assert_eq!(all.len(), CATEGORICAL_SCHEMES.len() + NamedContinuous::list().len());
    }

    #[test]
    fn palette_kind_classifies_each_family() {
        assert_eq!(palette_kind("okabe_ito").as_deref(), Some("categorical"));
        assert_eq!(palette_kind("viridis").as_deref(), Some("sequential"));
        assert_eq!(palette_kind("blues").as_deref(), Some("sequential"));
        assert_eq!(palette_kind("rdbu").as_deref(), Some("diverging"));
        assert_eq!(palette_kind("blue_to_red").as_deref(), Some("diverging"));
        assert_eq!(palette_kind("nonexistent"), None);
    }

    #[test]
    fn palette_kind_partitions_the_whole_registry() {
        for name in &list_palettes() {
            assert!(
                palette_kind(name).is_some(),
                "{name} is in list_palettes() but palette_kind returned None",
            );
        }
    }

    #[test]
    fn palette_colors_returns_categorical_stops() {
        let tableau = palette_colors("tableau10").expect("tableau10 known");
        assert_eq!(
            tableau,
            vec![
                "#4c78a8", "#f58e18", "#e45756", "#72b7b2", "#54a24b",
                "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
            ],
        );
        // Categorical stops are byte-identical to the render-path colors.
        let from_registry: Vec<String> =
            categorical_palette("tableau10").iter().map(|&c| fmt_svg(c)).collect();
        assert_eq!(tableau, from_registry);
    }

    #[test]
    fn palette_colors_continuous_has_inclusive_endpoints() {
        let viridis = palette_colors("viridis").expect("viridis known");
        assert_eq!(viridis.len(), PALETTE_CONTINUOUS_STOPS);
        // Endpoints are sample(0.0) and sample(1.0) — the render-truth values.
        assert_eq!(viridis.first().unwrap(), &palette_sample("viridis", 0.0).unwrap());
        assert_eq!(viridis.last().unwrap(), &palette_sample("viridis", 1.0).unwrap());
    }

    #[test]
    fn palette_colors_unknown_is_none() {
        assert_eq!(palette_colors("nonexistent"), None);
    }

    #[test]
    fn palette_sample_matches_render_path() {
        let s = ContinuousScheme::Named(NamedContinuous::Viridis);
        assert_eq!(palette_sample("viridis", 0.5).unwrap(), fmt_svg(s.sample(0.5)));
        // Categorical and unknown names are not sampleable.
        assert_eq!(palette_sample("tableau10", 0.5), None);
        assert_eq!(palette_sample("nonexistent", 0.5), None);
    }

    #[test]
    fn palette_sample_clamps_out_of_range_t() {
        assert_eq!(palette_sample("viridis", -1.0), palette_sample("viridis", 0.0));
        assert_eq!(palette_sample("viridis", 2.0), palette_sample("viridis", 1.0));
    }
}
