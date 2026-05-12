//! Hardcoded categorical palette (Okabe-Ito). One palette for Phase 7;
//! Phase 8+ may add a scheme registry.

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

/// Categorical scheme names recognized by [`categorical_palette`]. Source of
/// truth for theme-side validation in `binding::theme_from_dict`.
pub const CATEGORICAL_SCHEMES: &[&str] = &[
    "okabe_ito", "tableau10", "set1", "set2", "paired", "pastel", "dark2",
];

/// Sequential/diverging scheme names recognized by `ContinuousScheme` in
/// `render/color`. Listed here so theme-side validation can accept them
/// without depending on the continuous-scale machinery. "blues" is a
/// sequential single-hue ramp; "rdbu" is a diverging red-blue scheme.
pub const SEQUENTIAL_SCHEMES: &[&str] = &[
    "viridis", "plasma", "magma", "inferno", "cividis", "blues", "rdbu",
];

/// True when `name` is one of the recognized categorical schemes.
pub fn is_categorical_scheme(name: &str) -> bool {
    CATEGORICAL_SCHEMES.contains(&name)
}

/// True when `name` is one of the recognized sequential schemes (handled by
/// `ContinuousScheme`).
pub fn is_sequential_scheme(name: &str) -> bool {
    SEQUENTIAL_SCHEMES.contains(&name)
}

/// Look up a categorical palette by scheme name. Returns OKABE_ITO when the
/// name is unknown (caller may emit a warning). Theme-level validation in
/// `binding::theme_from_dict` rejects unknown names eagerly, so callers
/// reaching this function with an unknown name are using an encoding-level
/// override (e.g. `encoding.color.scheme`) — fallback preserved for that path.
pub fn categorical_palette(name: &str) -> &'static [Color] {
    match name {
        "okabe_ito"  => &*OKABE_ITO,
        "tableau10"  => &*TABLEAU10,
        "set1"       => &*SET1,
        "set2"       => &*SET2,
        "paired"     => &*PAIRED,
        "pastel"     => &*PASTEL,
        "dark2"      => &*DARK2,
        _            => &*OKABE_ITO,   // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_palette_returns_named_palette() {
        assert!(std::ptr::eq(categorical_palette("tableau10").as_ptr(), TABLEAU10.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("set1").as_ptr(), SET1.as_ptr()));
        assert!(std::ptr::eq(categorical_palette("dark2").as_ptr(), DARK2.as_ptr()));
    }

    #[test]
    fn categorical_palette_unknown_falls_back_to_okabe_ito() {
        assert!(std::ptr::eq(categorical_palette("nonexistent").as_ptr(), (&*OKABE_ITO).as_ptr()));
    }

    #[test]
    fn each_named_palette_has_at_least_8_colors() {
        for name in &["okabe_ito", "tableau10", "set1", "set2", "paired", "pastel", "dark2"] {
            assert!(categorical_palette(name).len() >= 8, "{name} has < 8 colors");
        }
    }

    #[test]
    fn categorical_schemes_const_matches_match_arms() {
        // Every entry in CATEGORICAL_SCHEMES must resolve to its own palette,
        // not the fallback. Detect via pointer identity.
        for name in CATEGORICAL_SCHEMES {
            let p = categorical_palette(name).as_ptr();
            let fallback = (&*OKABE_ITO).as_ptr();
            if *name == "okabe_ito" {
                assert!(std::ptr::eq(p, fallback));
            } else {
                assert!(
                    !std::ptr::eq(p, fallback),
                    "{name} resolved to OKABE_ITO fallback",
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
}
