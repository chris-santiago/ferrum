//! Pixel-space geometry primitives. Coordinates are f64; positive-y is downward
//! (consistent with SVG/screen conventions). All `shrink`/`split_*` operations
//! return new values; `Rect` and `Inset` are `Copy`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub const ZERO: Rect = Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };

    /// Shrink by an inset on each side. Returns `Rect::ZERO` if the inset
    /// would collapse either dimension to ≤ 0.
    pub fn shrink(&self, inset: Inset) -> Rect {
        let w = self.w - inset.left - inset.right;
        let h = self.h - inset.top - inset.bottom;
        if w <= 0.0 || h <= 0.0 {
            return Rect::ZERO;
        }
        Rect { x: self.x + inset.left, y: self.y + inset.top, w, h }
    }

    /// Split off a strip of height `h` from the top. Returns `(strip, remainder)`.
    /// If `h >= self.h`, strip == self and remainder == ZERO.
    pub fn split_top(&self, h: f64) -> (Rect, Rect) {
        let h = h.min(self.h).max(0.0);
        let strip = Rect { x: self.x, y: self.y, w: self.w, h };
        let remainder = Rect {
            x: self.x,
            y: self.y + h,
            w: self.w,
            h: self.h - h,
        };
        (strip, remainder)
    }

    pub fn split_bottom(&self, h: f64) -> (Rect, Rect) {
        let h = h.min(self.h).max(0.0);
        let remainder = Rect { x: self.x, y: self.y, w: self.w, h: self.h - h };
        let strip = Rect {
            x: self.x,
            y: self.y + self.h - h,
            w: self.w,
            h,
        };
        (strip, remainder)
    }

    pub fn split_left(&self, w: f64) -> (Rect, Rect) {
        let w = w.min(self.w).max(0.0);
        let strip = Rect { x: self.x, y: self.y, w, h: self.h };
        let remainder = Rect {
            x: self.x + w,
            y: self.y,
            w: self.w - w,
            h: self.h,
        };
        (strip, remainder)
    }

    pub fn split_right(&self, w: f64) -> (Rect, Rect) {
        let w = w.min(self.w).max(0.0);
        let remainder = Rect { x: self.x, y: self.y, w: self.w - w, h: self.h };
        let strip = Rect {
            x: self.x + self.w - w,
            y: self.y,
            w,
            h: self.h,
        };
        (strip, remainder)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Inset {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Inset {
    pub const fn uniform(v: f64) -> Inset {
        Inset { top: v, right: v, bottom: v, left: v }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

impl Viewport {
    pub fn into_rect(self) -> Rect {
        Rect { x: 0.0, y: 0.0, w: self.width, h: self.height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn rect_shrink_normal() {
        let r0 = r(0.0, 0.0, 100.0, 50.0);
        let r1 = r0.shrink(Inset::uniform(5.0));
        assert_eq!(r1, r(5.0, 5.0, 90.0, 40.0));
    }

    #[test]
    fn rect_shrink_collapses_to_zero() {
        let r0 = r(0.0, 0.0, 10.0, 10.0);
        let r1 = r0.shrink(Inset::uniform(10.0));
        assert_eq!(r1, Rect::ZERO);
    }

    #[test]
    fn rect_shrink_collapses_one_dim_to_zero() {
        // Left+right > w but top+bottom fits.
        let r0 = r(0.0, 0.0, 10.0, 100.0);
        let r1 = r0.shrink(Inset { top: 5.0, right: 6.0, bottom: 5.0, left: 6.0 });
        assert_eq!(r1, Rect::ZERO);
    }

    #[test]
    fn split_top_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (top, rest) = r0.split_top(30.0);
        assert_eq!(top, r(10.0, 20.0, 100.0, 30.0));
        assert_eq!(rest, r(10.0, 50.0, 100.0, 50.0));
        // No gap, no overlap.
        assert_eq!(top.y + top.h, rest.y);
    }

    #[test]
    fn split_bottom_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (bottom, rest) = r0.split_bottom(30.0);
        assert_eq!(rest, r(10.0, 20.0, 100.0, 50.0));
        assert_eq!(bottom, r(10.0, 70.0, 100.0, 30.0));
    }

    #[test]
    fn split_left_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (left, rest) = r0.split_left(40.0);
        assert_eq!(left, r(10.0, 20.0, 40.0, 80.0));
        assert_eq!(rest, r(50.0, 20.0, 60.0, 80.0));
    }

    #[test]
    fn split_right_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (right, rest) = r0.split_right(40.0);
        assert_eq!(rest, r(10.0, 20.0, 60.0, 80.0));
        assert_eq!(right, r(70.0, 20.0, 40.0, 80.0));
    }

    #[test]
    fn viewport_into_rect() {
        let v = Viewport { width: 600.0, height: 400.0 };
        assert_eq!(v.into_rect(), r(0.0, 0.0, 600.0, 400.0));
    }

    #[test]
    fn rect_serde_round_trip() {
        let r0 = r(1.0, 2.0, 3.0, 4.0);
        let json = serde_json::to_string(&r0).unwrap();
        let r1: Rect = serde_json::from_str(&json).unwrap();
        assert_eq!(r0, r1);
    }
}
