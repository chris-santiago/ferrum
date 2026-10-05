//! The shape a [`MarkBatch`]'s per-vertex and per-batch sidecars must have
//! before anything reads them: [`MarkBatch::vertex_rows`] aligned with the
//! batch's width vertices, and [`MarkBatch::conditionals`] naming only
//! conditionals of the scene.

use std::fmt;

use crate::{width_vertices, ConditionalEncoding, MarkBatch};

/// A [`MarkBatch`] sidecar whose shape does not match the batch it rides on.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchShapeError {
    /// A line's `vertex_rows` entry does not carry exactly one row per vertex.
    RowsMisaligned { points: usize, rows: usize },
    /// A batch's `vertex_rows` does not carry exactly one row list per node.
    RowListsMisaligned { nodes: usize, lists: usize },
    /// A batch carries `vertex_rows` but lists no `size` conditional for
    /// them to answer to.
    SizeConditionalsUnpaired,
    /// A batch's `conditionals` names conditional `index` of a scene that has
    /// `conditionals`.
    ConditionalOutOfRange { index: usize, conditionals: usize },
}

impl fmt::Display for BatchShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowsMisaligned { points, rows } => write!(
                f,
                "tapered stroke has {rows} vertex rows for {points} vertices; it needs one row per vertex"
            ),
            Self::RowListsMisaligned { nodes, lists } => write!(
                f,
                "tapered line batch has {lists} vertex row lists for {nodes} lines; it needs one list per line"
            ),
            Self::SizeConditionalsUnpaired => write!(
                f,
                "tapered line batch carries vertex_rows but answers to no size conditional"
            ),
            Self::ConditionalOutOfRange { index, conditionals } => write!(
                f,
                "mark batch answers to conditional {index}, but the scene has {conditionals} conditionals"
            ),
        }
    }
}

impl std::error::Error for BatchShapeError {}

/// Check that `batch`'s [`MarkBatch::vertex_rows`], when present, carries one
/// row list per node and one row per width vertex of that node
/// ([`width_vertices`]).
pub fn validate_vertex_rows(batch: &MarkBatch) -> Result<(), BatchShapeError> {
    let Some(lists) = &batch.vertex_rows else { return Ok(()) };
    if lists.len() != batch.nodes.len() {
        return Err(BatchShapeError::RowListsMisaligned { nodes: batch.nodes.len(), lists: lists.len() });
    }
    for (node, rows) in batch.nodes.iter().zip(lists) {
        let points = width_vertices(node);
        if rows.len() != points {
            return Err(BatchShapeError::RowsMisaligned { points, rows: rows.len() });
        }
    }
    Ok(())
}

/// Check that every index `batch` lists in [`MarkBatch::conditionals`] names
/// one of the scene's `conditionals`, and that a batch carrying `vertex_rows`
/// lists at least one that sets `size` ([`ConditionalEncoding::sets_size`]).
pub fn validate_conditionals(batch: &MarkBatch, conditionals: &[ConditionalEncoding]) -> Result<(), BatchShapeError> {
    if let Some(&index) = batch.conditionals.iter().find(|&&index| index >= conditionals.len()) {
        return Err(BatchShapeError::ConditionalOutOfRange { index, conditionals: conditionals.len() });
    }
    let sized = batch.conditionals.iter().any(|&index| conditionals.get(index).is_some_and(ConditionalEncoding::sets_size));
    match batch.vertex_rows {
        Some(_) if !sized => Err(BatchShapeError::SizeConditionalsUnpaired),
        _ => Ok(()),
    }
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
        assert_eq!(validate_vertex_rows(&batch(Some(vec![rows(2)]))), Err(BatchShapeError::RowsMisaligned { points: 3, rows: 2 }));
        assert_eq!(
            validate_vertex_rows(&batch(Some(vec![rows(3), rows(3)]))),
            Err(BatchShapeError::RowListsMisaligned { nodes: 1, lists: 2 })
        );
        assert_eq!(
            BatchShapeError::RowsMisaligned { points: 3, rows: 2 }.to_string(),
            "tapered stroke has 2 vertex rows for 3 vertices; it needs one row per vertex"
        );
        assert_eq!(
            BatchShapeError::RowListsMisaligned { nodes: 1, lists: 2 }.to_string(),
            "tapered line batch has 2 vertex row lists for 1 lines; it needs one list per line"
        );
    }


    #[test]
    fn conditionals_name_scene_conditionals_and_tapered_lines_list_a_size_one() {
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
        let batch = |listed: Vec<usize>, rows: bool| MarkBatch {
            conditionals: listed,
            vertex_rows: rows.then(std::vec::Vec::new),
            ..MarkBatch::default()
        };
        // Any batch may list any channel; a line's rows need a size one.
        assert_eq!(validate_conditionals(&batch(vec![], false), &conditionals), Ok(()));
        assert_eq!(validate_conditionals(&batch(vec![1], false), &conditionals), Ok(()));
        assert_eq!(validate_conditionals(&batch(vec![0, 1], true), &conditionals), Ok(()));
        assert_eq!(validate_conditionals(&batch(vec![], true), &conditionals), Err(BatchShapeError::SizeConditionalsUnpaired));
        assert_eq!(validate_conditionals(&batch(vec![1], true), &conditionals), Err(BatchShapeError::SizeConditionalsUnpaired));
        assert_eq!(
            validate_conditionals(&batch(vec![0, 2], false), &conditionals),
            Err(BatchShapeError::ConditionalOutOfRange { index: 2, conditionals: 2 })
        );
        assert_eq!(
            BatchShapeError::ConditionalOutOfRange { index: 2, conditionals: 2 }.to_string(),
            "mark batch answers to conditional 2, but the scene has 2 conditionals"
        );
    }

    #[test]
    fn a_batch_answers_only_to_the_conditionals_it_lists() {
        let batch = |listed: Vec<usize>| MarkBatch { conditionals: listed, ..MarkBatch::default() };
        assert!(batch(vec![1]).answers_to(1));
        assert!(!batch(vec![1]).answers_to(0));
        assert!(!batch(vec![]).answers_to(0));
    }
}
