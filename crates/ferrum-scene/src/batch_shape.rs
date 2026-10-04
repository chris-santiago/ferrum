//! The shape a [`MarkBatch`]'s per-vertex and per-batch sidecars must have
//! before anything reads them: [`MarkBatch::vertex_rows`] aligned with the
//! batch's width vertices, and [`MarkBatch::size_conditionals`] naming only
//! `size` conditionals of the scene.

use std::fmt;

use crate::{width_vertices, ConditionalEncoding, MarkBatch};

/// A [`MarkBatch`] sidecar whose shape does not match the batch it rides on.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchShapeError {
    /// A line's `vertex_rows` entry does not carry exactly one row per vertex.
    RowsMisaligned { points: usize, rows: usize },
    /// A batch's `vertex_rows` does not carry exactly one row list per node.
    RowListsMisaligned { nodes: usize, lists: usize },
    /// A batch carries `vertex_rows` without the `size_conditionals` they
    /// answer to.
    SizeConditionalsUnpaired,
    /// A batch's `size_conditionals` names conditional `index` of a scene
    /// that has `conditionals`.
    SizeConditionalOutOfRange { index: usize, conditionals: usize },
    /// A batch's `size_conditionals` names conditional `index`, which sets no
    /// `size`.
    NotASizeConditional { index: usize },
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
                "tapered line batch carries vertex_rows without the size_conditionals they answer to"
            ),
            Self::SizeConditionalOutOfRange { index, conditionals } => write!(
                f,
                "mark batch answers to size conditional {index}, but the scene has {conditionals} conditionals"
            ),
            Self::NotASizeConditional { index } => write!(
                f,
                "mark batch answers to size conditional {index}, but conditional {index} sets no size"
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

impl MarkBatch {
    /// Whether this batch's marks answer to the scene's `size` conditional
    /// `index`: the ones [`MarkBatch::size_conditionals`] names, or every one
    /// when it is `None`. The one reading of that field for every consumer.
    pub fn answers_to_size_conditional(&self, index: usize) -> bool {
        self.size_conditionals.as_ref().is_none_or(|own| own.contains(&index))
    }
}

/// Check that `batch` carries [`MarkBatch::size_conditionals`] whenever it
/// carries `vertex_rows`, and that every index names one of the scene's
/// `conditionals` that sets `size` ([`ConditionalEncoding::sets_size`]).
pub fn validate_size_conditionals(batch: &MarkBatch, conditionals: &[ConditionalEncoding]) -> Result<(), BatchShapeError> {
    let Some(indices) = &batch.size_conditionals else {
        return match batch.vertex_rows {
            Some(_) => Err(BatchShapeError::SizeConditionalsUnpaired),
            None => Ok(()),
        };
    };
    for &index in indices {
        match conditionals.get(index) {
            None => return Err(BatchShapeError::SizeConditionalOutOfRange { index, conditionals: conditionals.len() }),
            Some(cond) if !cond.sets_size() => return Err(BatchShapeError::NotASizeConditional { index }),
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
        assert_eq!(validate_size_conditionals(&batch(None, true), &conditionals), Err(BatchShapeError::SizeConditionalsUnpaired));
        assert_eq!(
            validate_size_conditionals(&batch(Some(vec![1]), false), &conditionals),
            Err(BatchShapeError::NotASizeConditional { index: 1 })
        );
        assert_eq!(
            validate_size_conditionals(&batch(Some(vec![0, 2]), true), &conditionals),
            Err(BatchShapeError::SizeConditionalOutOfRange { index: 2, conditionals: 2 })
        );
        assert_eq!(
            BatchShapeError::NotASizeConditional { index: 1 }.to_string(),
            "mark batch answers to size conditional 1, but conditional 1 sets no size"
        );
    }

    #[test]
    fn a_batch_answers_to_the_size_conditionals_it_names_or_to_every_one_when_it_names_none() {
        let batch = |scope: Option<Vec<usize>>| MarkBatch { size_conditionals: scope, ..MarkBatch::default() };
        assert!(batch(None).answers_to_size_conditional(0));
        assert!(batch(None).answers_to_size_conditional(3));
        assert!(batch(Some(vec![1])).answers_to_size_conditional(1));
        assert!(!batch(Some(vec![1])).answers_to_size_conditional(0));
        assert!(!batch(Some(vec![])).answers_to_size_conditional(0));
    }
}
