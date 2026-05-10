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

pub fn categorical_color(category_index: usize) -> Color {
    OKABE_ITO[category_index % OKABE_ITO.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_category_is_okabe_orange() {
        let c = categorical_color(0);
        assert_eq!(c.red, 0xE6);
        assert_eq!(c.green, 0x9F);
        assert_eq!(c.blue, 0x00);
    }

    #[test]
    fn overflow_wraps() {
        let c = categorical_color(8);
        assert_eq!(c, OKABE_ITO[0]);
    }
}
