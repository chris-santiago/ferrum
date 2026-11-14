//! Which keys a series lacks and where each imputed row goes.
//!
//! A series is completed over the key universe. Each key it lacks is inserted
//! right after the last row of the series whose key comes earlier in the key
//! order, or before the series' first row when none does. A caller's rows never
//! move relative to each other, so a series written out of key order keeps its
//! order and a line draws it as written; a series in key order stays in key
//! order. Imputed rows that land at one place keep key order among themselves.

use std::collections::BTreeMap;

use super::fill::Entry;

/// A key a series lacks, and the place its row goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Missing {
    /// The key's place in the key order.
    pub rank: usize,
    /// The row of the batch the imputed row is inserted before. `rows.len()`
    /// is the end of the batch.
    pub slot: usize,
}

/// The keys of the universe (`universe_len` of them, ranked `0..universe_len`)
/// that none of `rows` holds, in key order, each with its slot. `rows` are the
/// series' rows in ascending batch order and `ranks` every row's key rank.
pub(super) fn missing_keys(
    rows: &[usize],
    ranks: &[Option<usize>],
    universe_len: usize,
) -> Vec<Missing> {
    let Some(&first) = rows.first() else {
        return Vec::new();
    };
    // The last row holding each key: rows ascend, so a later insert wins.
    let mut last_row_with: BTreeMap<usize, usize> = BTreeMap::new();
    for &row in rows {
        if let Some(rank) = ranks[row] {
            last_row_with.insert(rank, row);
        }
    }
    let mut last_earlier: Option<usize> = None;
    let mut missing = Vec::new();
    for rank in 0..universe_len {
        match last_row_with.get(&rank) {
            Some(&row) => last_earlier = Some(last_earlier.map_or(row, |seen| seen.max(row))),
            None => missing.push(Missing { rank, slot: last_earlier.map_or(first, |row| row + 1) }),
        }
    }
    missing
}

/// The series' rows and its missing keys as the positions a window counts, in
/// key order. A row with a null key holds no place in the order and follows
/// every keyed position, in row order.
pub(super) fn key_ordered(
    rows: &[usize],
    ranks: &[Option<usize>],
    missing: &[Missing],
) -> Vec<Entry> {
    // (rank, row): a null key sorts after every rank.
    let mut positioned: Vec<(usize, usize)> = rows
        .iter()
        .map(|&row| (ranks[row].unwrap_or(usize::MAX), row))
        .collect();
    positioned.sort();
    let mut entries: Vec<Entry> = Vec::with_capacity(rows.len() + missing.len());
    let mut next_missing = missing.iter().enumerate().peekable();
    for (rank, row) in positioned {
        while let Some((index, _)) = next_missing.next_if(|(_, m)| m.rank < rank) {
            entries.push(Entry::Imputed(index));
        }
        entries.push(Entry::Caller(row));
    }
    entries.extend(next_missing.map(|(index, _)| Entry::Imputed(index)));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranks(values: &[Option<usize>]) -> Vec<Option<usize>> {
        values.to_vec()
    }

    fn slots(missing: &[Missing]) -> Vec<(usize, usize)> {
        missing.iter().map(|m| (m.rank, m.slot)).collect()
    }

    #[test]
    fn a_key_in_the_middle_goes_right_after_the_last_row_with_an_earlier_key() {
        // keys 0 and 2 held; universe 0..3 lacks 1
        let got = missing_keys(&[0, 1], &ranks(&[Some(0), Some(2)]), 3);
        assert_eq!(slots(&got), vec![(1, 1)]);
    }

    #[test]
    fn a_key_before_every_row_goes_before_the_first_row() {
        // the series holds batch rows 4 and 5; rows 0 to 3 belong to another series
        let all = ranks(&[None, None, None, None, Some(1), Some(2)]);
        let got = missing_keys(&[4, 5], &all, 3);
        assert_eq!(slots(&got), vec![(0, 4)]);
    }

    #[test]
    fn a_key_after_every_row_goes_after_the_last_row_with_an_earlier_key() {
        let got = missing_keys(&[0, 1], &ranks(&[Some(0), Some(1)]), 3);
        assert_eq!(slots(&got), vec![(2, 2)]);
    }

    #[test]
    fn a_series_out_of_key_order_gets_the_row_after_its_last_earlier_row_not_the_first_later_one() {
        // rows written with keys 2, 0: key 1 follows the row holding 0, the last row
        let got = missing_keys(&[0, 1], &ranks(&[Some(2), Some(0)]), 3);
        assert_eq!(slots(&got), vec![(1, 2)]);
    }

    #[test]
    fn a_key_above_both_out_of_order_keys_follows_the_later_row_not_the_earlier() {
        // the universe is keys 1, 3, 4 (ranks 0 to 2); the series is written 3 then 1
        let got = missing_keys(&[0, 1], &ranks(&[Some(1), Some(0)]), 3);
        assert_eq!(slots(&got), vec![(2, 2)], "key 4 goes after the last row, not between the two");
    }

    #[test]
    fn a_key_held_twice_is_anchored_after_the_later_of_its_rows() {
        // keys 0, 2, 0, 2 written: key 1 follows the second 0
        let got = missing_keys(&[0, 1, 2, 3], &ranks(&[Some(0), Some(2), Some(0), Some(2)]), 3);
        assert_eq!(slots(&got), vec![(1, 3)]);
    }

    #[test]
    fn rows_interleaved_with_other_series_anchor_to_their_own_batch_positions() {
        // this series holds batch rows 0, 2, 4 (keys 0, 2, 3); the rows between are another series'
        let all = ranks(&[Some(0), Some(0), Some(2), Some(1), Some(3)]);
        let got = missing_keys(&[0, 2, 4], &all, 4);
        assert_eq!(slots(&got), vec![(1, 1)]);
    }

    #[test]
    fn several_missing_keys_at_one_place_come_out_in_key_order() {
        let got = missing_keys(&[0], &ranks(&[Some(3)]), 6);
        assert_eq!(slots(&got), vec![(0, 0), (1, 0), (2, 0), (4, 1), (5, 1)]);
    }

    #[test]
    fn a_null_key_row_is_ignored_when_placing() {
        let got = missing_keys(&[0, 1], &ranks(&[None, Some(0)]), 2);
        assert_eq!(slots(&got), vec![(1, 2)]);
    }

    #[test]
    fn a_series_with_no_keyed_row_gets_every_key_before_its_first_row() {
        let got = missing_keys(&[3, 4], &ranks(&[None, None, None, None, None]), 2);
        assert_eq!(slots(&got), vec![(0, 3), (1, 3)]);
    }

    #[test]
    fn a_complete_series_and_an_empty_universe_miss_nothing() {
        assert!(missing_keys(&[0, 1], &ranks(&[Some(0), Some(1)]), 2).is_empty());
        assert!(missing_keys(&[0, 1], &ranks(&[None, None]), 0).is_empty());
        assert!(missing_keys(&[], &ranks(&[]), 3).is_empty());
    }

    #[test]
    fn the_entries_of_a_series_are_in_key_order_with_the_missing_keys_among_them() {
        // rows 0, 1, 2 hold keys 3, 0, 2; the series lacks 1 and 4
        let r = ranks(&[Some(3), Some(0), Some(2)]);
        let missing = missing_keys(&[0, 1, 2], &r, 5);
        let entries = key_ordered(&[0, 1, 2], &r, &missing);
        assert_eq!(
            entries,
            vec![
                Entry::Caller(1),
                Entry::Imputed(0),
                Entry::Caller(2),
                Entry::Caller(0),
                Entry::Imputed(1)
            ]
        );
    }

    #[test]
    fn rows_with_a_null_key_follow_every_keyed_position_in_row_order() {
        let r = ranks(&[None, Some(1), None]);
        let missing = missing_keys(&[0, 1, 2], &r, 3);
        let entries = key_ordered(&[0, 1, 2], &r, &missing);
        assert_eq!(
            entries,
            vec![
                Entry::Imputed(0),
                Entry::Caller(1),
                Entry::Imputed(1),
                Entry::Caller(0),
                Entry::Caller(2)
            ]
        );
    }

    #[test]
    fn rows_sharing_a_key_keep_their_row_order() {
        let r = ranks(&[Some(1), Some(0), Some(1)]);
        let entries = key_ordered(&[0, 1, 2], &r, &[]);
        assert_eq!(entries, vec![Entry::Caller(1), Entry::Caller(0), Entry::Caller(2)]);
    }
}
