//! Canonical stroke-dash pattern palette.
//!
//! The single home both the SVG render path (index → pattern, for the
//! `stroke-dasharray` attribute) and the GPU/interactive path (pattern →
//! index, for the packed instance field the WASM shaders read) derive
//! from, so the two directions cannot silently drift from each other.

/// Ordered stroke-dash pattern table. Index `0` is solid and carries no
/// entry here ([`resolve_stroke_dash`] returns `None` for it); index `i`
/// (`i >= 1`) reads `DASH_PALETTE[i - 1]`.
///
/// The first three entries predate this table and keep their indices, so a
/// categorical `StrokeDash` scale's cycling assignment (bounded to
/// [`STROKE_DASH_LEGACY_SLOTS`], not this table's length) is unaffected by
/// the append below. The remaining four are the named linetypes Python's
/// `_LINETYPE_MAP` exposes (`dashed`, `dotted`, `dashdot`, `longdash`); a
/// Python test asserts every `_LINETYPE_MAP` pattern is a member of this
/// table, read through [`crate`]'s `stroke_dash_palette()` Python binding
/// rather than a hand-mirrored copy.
pub const DASH_PALETTE: &[&[f64]] = &[
    &[6.0, 3.0],           // 1: long dash
    &[2.0, 3.0],           // 2: short dash / dot
    &[6.0, 3.0, 2.0, 3.0], // 3: long-short dash
    &[4.0, 2.0],           // 4: dashed
    &[1.0, 3.0],           // 5: dotted
    &[4.0, 2.0, 1.0, 2.0], // 6: dashdot
    &[8.0, 4.0],           // 7: longdash
];

/// The slot span a stroke-dash *index* can reach: solid plus the three
/// patterns that predate the named-linetype append. Two readers index into
/// it, both through this module: a numeric `stroke_dash` column clamps into
/// it ([`numeric_index_dash`]) and a categorical `StrokeDash` scale cycles its
/// categories through it ([`categorical_slot_dash`]).
///
/// Deliberately fixed independent of `DASH_PALETTE.len()`: both readers'
/// outputs are byte-stable contracts (`StrokeDashPaletteOverflowed` fires past
/// this many categories), and the named linetypes are reached only by a
/// literal pattern (a direct `stroke_dash=` value or a conditional literal),
/// never by an index. So a numeric column value `4.0` resolves to slot 3, not
/// to the `[4, 2]` entry at palette index 4. Appending to `DASH_PALETTE` must
/// not move which pattern an existing index gets.
pub const STROKE_DASH_LEGACY_SLOTS: usize = 4;

/// Map a stroke-dash palette index to its canonical dasharray pattern.
///
/// The index is rounded and clamped to `[0, DASH_PALETTE.len()]` before
/// lookup:
/// - `0` → solid (returns `None`)
/// - `1..=DASH_PALETTE.len()` → `DASH_PALETTE[idx - 1]`
///
/// Derives from [`DASH_PALETTE`] so it cannot drift from the GPU-path
/// inverse, [`stroke_dash_index`].
pub fn resolve_stroke_dash(idx: f64) -> Option<Vec<f64>> {
    let idx = (idx.round() as i64).clamp(0, DASH_PALETTE.len() as i64) as usize;
    if idx == 0 {
        None
    } else {
        DASH_PALETTE.get(idx - 1).map(|pat| pat.to_vec())
    }
}

/// The pattern a numeric `stroke_dash` column value selects: the value is
/// rounded and clamped into the [`STROKE_DASH_LEGACY_SLOTS`] span, so `0` is
/// solid (`None`) and any value at or past the last slot reads that slot.
pub fn numeric_index_dash(value: f64) -> Option<Vec<f64>> {
    let last_slot = (STROKE_DASH_LEGACY_SLOTS - 1) as i64;
    resolve_stroke_dash((value.round() as i64).clamp(0, last_slot) as f64)
}

/// The pattern a categorical `StrokeDash` scale assigns its `ordinal`-th
/// category: categories cycle through the [`STROKE_DASH_LEGACY_SLOTS`] span,
/// so the first category (and every `STROKE_DASH_LEGACY_SLOTS`-th after it)
/// is solid (`None`).
pub fn categorical_slot_dash(ordinal: usize) -> Option<Vec<f64>> {
    resolve_stroke_dash((ordinal % STROKE_DASH_LEGACY_SLOTS) as f64)
}

/// Map a stroke-dash pattern to its palette index.
///
/// `None`, an empty pattern, and a pattern that is not an exact (within
/// floating-point tolerance) member of [`DASH_PALETTE`] all map to `0`
/// (solid) — the non-palette fallback scope: a palette pattern always
/// round-trips, so this is reached only by a pattern this table does not
/// carry.
///
/// Derives from [`DASH_PALETTE`] so it cannot drift from the SVG-path
/// inverse, [`resolve_stroke_dash`].
pub fn stroke_dash_index(dash: Option<&[f64]>) -> u32 {
    let Some(v) = dash else { return 0 };
    if v.is_empty() {
        return 0;
    }
    for (i, entry) in DASH_PALETTE.iter().enumerate() {
        if entry.len() == v.len()
            && entry
                .iter()
                .zip(v.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        {
            return (i + 1) as u32;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_index_resolves_to_no_pattern() {
        assert_eq!(resolve_stroke_dash(0.0), None);
    }

    #[test]
    fn every_palette_entry_round_trips_index_to_pattern_to_index() {
        for idx in 1..=DASH_PALETTE.len() {
            let pattern =
                resolve_stroke_dash(idx as f64).expect("non-zero index must produce a pattern");
            assert_eq!(pattern, DASH_PALETTE[idx - 1].to_vec());
            let recovered = stroke_dash_index(Some(&pattern));
            assert_eq!(
                recovered, idx as u32,
                "pattern {pattern:?} at index {idx} must round-trip through stroke_dash_index"
            );
        }
    }

    #[test]
    fn none_and_empty_pattern_map_to_solid() {
        assert_eq!(stroke_dash_index(None), 0);
        assert_eq!(stroke_dash_index(Some(&[])), 0);
    }

    /// A pattern outside the palette maps to solid — the fallback's actual
    /// scope, non-palette input only (every palette entry itself round-trips
    /// per `every_palette_entry_round_trips_index_to_pattern_to_index`).
    #[test]
    fn nonpalette_pattern_maps_to_solid() {
        assert_eq!(stroke_dash_index(Some(&[5.0, 5.0])), 0);
    }

    #[test]
    fn non_integer_valued_pattern_does_not_fuzzy_match_a_palette_entry() {
        // [6.5, 3.5] must not match [6.0, 3.0] (index 1).
        assert_eq!(stroke_dash_index(Some(&[6.5, 3.5])), 0);
    }

    #[test]
    fn single_element_pattern_matches_no_palette_entry() {
        assert_eq!(stroke_dash_index(Some(&[6.0])), 0);
    }

    #[test]
    fn legacy_slot_count_is_the_solid_slot_plus_the_three_pre_append_patterns() {
        assert_eq!(STROKE_DASH_LEGACY_SLOTS, 4);
        assert!(
            DASH_PALETTE.len() > STROKE_DASH_LEGACY_SLOTS - 1,
            "the named-linetype append must add entries beyond the legacy slots"
        );
    }

    #[test]
    fn numeric_index_clamps_into_the_legacy_slots() {
        assert_eq!(numeric_index_dash(0.0), None);
        assert_eq!(numeric_index_dash(-2.0), None);
        assert_eq!(numeric_index_dash(1.4), Some(DASH_PALETTE[0].to_vec()));
        assert_eq!(numeric_index_dash(3.0), Some(DASH_PALETTE[2].to_vec()));
        // Past the span clamps to the last slot, never reaching a named linetype.
        assert_eq!(numeric_index_dash(4.0), Some(DASH_PALETTE[2].to_vec()));
    }

    #[test]
    fn categorical_ordinals_cycle_through_the_legacy_slots() {
        let cycled: Vec<_> = (0..6).map(categorical_slot_dash).collect();
        assert_eq!(
            cycled,
            vec![
                None,
                Some(DASH_PALETTE[0].to_vec()),
                Some(DASH_PALETTE[1].to_vec()),
                Some(DASH_PALETTE[2].to_vec()),
                None,
                Some(DASH_PALETTE[0].to_vec()),
            ]
        );
    }
}
