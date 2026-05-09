//! Text width measurement. Phase 6 ships only `HeuristicMetrics` (char_count *
//! font_size * K). Phase 7's renderer will provide a fontdue-backed implementation
//! by implementing this trait. `MockMetrics` is test-only and supports a
//! user-supplied closure for pixel-exact reference values.

pub trait TextMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64;
    fn line_height(&self, font_size: f64) -> f64 {
        font_size * 1.2
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HeuristicMetrics {
    pub k: f64,
}

impl Default for HeuristicMetrics {
    fn default() -> Self {
        Self { k: 0.6 }
    }
}

impl TextMetrics for HeuristicMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars().count() as f64 * font_size * self.k
    }
}

#[cfg(test)]
pub(crate) struct MockMetrics<F: Fn(&str, f64) -> f64> {
    pub measure: F,
    pub line_h_factor: f64,
}

#[cfg(test)]
impl<F: Fn(&str, f64) -> f64> TextMetrics for MockMetrics<F> {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        (self.measure)(text, font_size)
    }
    fn line_height(&self, font_size: f64) -> f64 {
        font_size * self.line_h_factor
    }
}

#[cfg(test)]
pub(crate) fn fixed_width(per_char_px: f64) -> impl Fn(&str, f64) -> f64 {
    move |text: &str, _font_size: f64| text.chars().count() as f64 * per_char_px
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_default_k_is_0_6() {
        let m = HeuristicMetrics::default();
        assert!((m.k - 0.6).abs() < 1e-12);
    }

    #[test]
    fn heuristic_measure_hello_at_12pt_equals_36() {
        let m = HeuristicMetrics::default();
        // 5 chars * 12 * 0.6 = 36.0
        assert!((m.measure_width("hello", 12.0) - 36.0).abs() < 1e-12);
    }

    #[test]
    fn heuristic_line_height_default_is_1_2x() {
        let m = HeuristicMetrics::default();
        assert!((m.line_height(12.0) - 14.4).abs() < 1e-12);
    }

    #[test]
    fn heuristic_handles_unicode_chars() {
        let m = HeuristicMetrics::default();
        // 3 chars (each emoji is one char in this count), font_size 10.
        // The actual char count depends on grapheme/scalar count; we use chars()
        // which counts unicode scalars: "abc" = 3, "héllo" = 5.
        assert!((m.measure_width("abc", 10.0) - 18.0).abs() < 1e-12);
        assert!((m.measure_width("héllo", 10.0) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn mock_metrics_uses_closure() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.5,
        };
        assert_eq!(m.measure_width("abc", 12.0), 30.0);
        assert!((m.line_height(12.0) - 18.0).abs() < 1e-12);
    }
}
