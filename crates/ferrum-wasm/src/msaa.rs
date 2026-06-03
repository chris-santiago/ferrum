//! MSAA sample-count selection (FA-19).
//!
//! Pure host-testable logic shared by the wasm32-only GPU context. Kept out of
//! `gpu.rs` so the capability-fallback invariant can be unit-tested on the host
//! without a GPU device.

/// Pick the MSAA sample count from backend capability.
///
/// 4× when supported, else 1× (silent FA-19 fallback — no panic, no error).
/// `sample_count == 1` keeps the main pass byte-identical to the pre-MSAA path.
pub(crate) fn choose_sample_count(supports_4x: bool) -> u32 {
    if supports_4x {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::choose_sample_count;

    #[test]
    fn sample_count_prefers_4x_when_supported() {
        assert_eq!(choose_sample_count(true), 4);
    }

    #[test]
    fn sample_count_falls_back_to_1x() {
        assert_eq!(choose_sample_count(false), 1);
    }
}
