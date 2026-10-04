//! The shape a [`MarkBatch`]'s per-vertex and per-batch sidecars must have
//! before anything reads them: [`MarkBatch::vertex_rows`] aligned with the
//! batch's width vertices, and [`MarkBatch::size_conditionals`] naming only
//! `size` conditionals of the scene.

use crate::{width_vertices, ConditionalEncoding, MarkBatch, TaperError};

/// Check that `batch`'s [`MarkBatch::vertex_rows`], when present, carries one
/// row list per node and one row per width vertex of that node
/// ([`width_vertices`]).
pub fn validate_vertex_rows(batch: &MarkBatch) -> Result<(), TaperError> {
    let Some(lists) = &batch.vertex_rows else { return Ok(()) };
    if lists.len() != batch.nodes.len() {
        return Err(TaperError::RowListsMisaligned { nodes: batch.nodes.len(), lists: lists.len() });
    }
    for (node, rows) in batch.nodes.iter().zip(lists) {
        let points = width_vertices(node);
        if rows.len() != points {
            return Err(TaperError::RowsMisaligned { points, rows: rows.len() });
        }
    }
    Ok(())
}

/// Check that `batch` carries [`MarkBatch::size_conditionals`] whenever it
/// carries `vertex_rows`, and that every index names one of the scene's
/// `conditionals` that sets `size` ([`ConditionalEncoding::sets_size`]).
pub fn validate_size_conditionals(batch: &MarkBatch, conditionals: &[ConditionalEncoding]) -> Result<(), TaperError> {
    let Some(indices) = &batch.size_conditionals else {
        return match batch.vertex_rows {
            Some(_) => Err(TaperError::SizeConditionalsUnpaired),
            None => Ok(()),
        };
    };
    for &index in indices {
        match conditionals.get(index) {
            None => return Err(TaperError::SizeConditionalOutOfRange { index, conditionals: conditionals.len() }),
            Some(cond) if !cond.sets_size() => return Err(TaperError::NotASizeConditional { index }),
            Some(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChannelName, Color, EncodingValue, SceneNode};

    #[test]
    fn vertex_rows_align_one_list_per_node_and_one_row_per_width_vertex() {
        use crate::VertexRow;
        let style = crate::StrokeStyle {
            color: None,
            width: 1.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        };
        let line = SceneNode::Polyline { points: vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], style, widths: None };
        let rows = |n: usize| (0..n).map(VertexRow::Row).collect::<Vec<_>>();
        let batch = |lists: Option<Vec<Vec<VertexRow>>>| MarkBatch { nodes: vec![line.clone()], vertex_rows: lists, ..MarkBatch::default() };
        assert_eq!(validate_vertex_rows(&batch(None)), Ok(()));
        assert_eq!(validate_vertex_rows(&batch(Some(vec![rows(3)]))), Ok(()));
        assert_eq!(validate_vertex_rows(&batch(Some(vec![rows(2)]))), Err(TaperError::RowsMisaligned { points: 3, rows: 2 }));
        assert_eq!(
            validate_vertex_rows(&batch(Some(vec![rows(3), rows(3)]))),
            Err(TaperError::RowListsMisaligned { nodes: 1, lists: 2 })
        );
        assert_eq!(
            TaperError::RowsMisaligned { points: 3, rows: 2 }.to_string(),
            "tapered stroke has 2 vertex rows for 3 vertices; it needs one row per vertex"
        );
        assert_eq!(
            TaperError::RowListsMisaligned { nodes: 1, lists: 2 }.to_string(),
            "tapered line batch has 2 vertex row lists for 1 lines; it needs one list per line"
        );
    }


    #[test]
    fn size_conditionals_name_only_size_conditionals_and_pair_with_vertex_rows() {
        let cond = |if_selected, if_not| ConditionalEncoding {
            selection_name: "s".into(),
            channel: ChannelName::Size,
            if_selected,
            if_not,
        };
        let size = cond(EncodingValue::Size { value: 40.0 }, EncodingValue::Size { value: 10.0 });
        let black = Color { r: 0, g: 0, b: 0, a: 255 };
        let color = cond(EncodingValue::Color { value: black }, EncodingValue::Color { value: black });
        let conditionals = [size, color];
        let batch = |scope: Option<Vec<usize>>, rows: bool| MarkBatch {
            size_conditionals: scope,
            vertex_rows: rows.then(|| vec![]),
            ..MarkBatch::default()
        };
        // A point batch may carry a scope alone; a line's rows need one.
        assert_eq!(validate_size_conditionals(&batch(None, false), &conditionals), Ok(()));
        assert_eq!(validate_size_conditionals(&batch(Some(vec![0]), false), &conditionals), Ok(()));
        assert_eq!(validate_size_conditionals(&batch(Some(vec![0]), true), &conditionals), Ok(()));
        assert_eq!(validate_size_conditionals(&batch(Some(vec![]), true), &conditionals), Ok(()));
        assert_eq!(validate_size_conditionals(&batch(None, true), &conditionals), Err(TaperError::SizeConditionalsUnpaired));
        assert_eq!(
            validate_size_conditionals(&batch(Some(vec![1]), false), &conditionals),
            Err(TaperError::NotASizeConditional { index: 1 })
        );
        assert_eq!(
            validate_size_conditionals(&batch(Some(vec![0, 2]), true), &conditionals),
            Err(TaperError::SizeConditionalOutOfRange { index: 2, conditionals: 2 })
        );
        assert_eq!(
            TaperError::NotASizeConditional { index: 1 }.to_string(),
            "mark batch answers to size conditional 1, but conditional 1 sets no size"
        );
    }
}
