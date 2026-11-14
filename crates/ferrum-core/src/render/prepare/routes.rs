//! Where each layer's rows live once the null-key drop and the impute step
//! have run.
//!
//! Layers that read one transform output may draw different rows of it: a
//! layer keying a field its siblings do not drops different rows, and a layer
//! whose impute differs completes different series. A layer that draws rows
//! its key-siblings do not gets them under its own key, and that fact is
//! written once, as [`Layer::own_rows`](crate::spec::layer::Layer) on the spec
//! [`PreparedInputs`](super::PreparedInputs) owns. Every reader that fetches a
//! layer's batch asks that spec ([`Layer::rows_key`](crate::spec::layer::Layer)
//! or `PreparedInputs::rows_key`), so no reader can resolve a layer over a
//! different frame than the one it draws. [`LayerRows`] is the one place that
//! writes a route.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::spec::chart::ChartSpec;
use crate::transform::core::FINAL_OUTPUT_KEY;
use crate::transform::run::OutputLineages;

/// The key a layer's own rows are published under in the transform outputs,
/// for a layer that draws rows its key-siblings do not share.
pub(in crate::render) fn own_rows_key(index: usize) -> String {
    format!("__own_rows_layer_{index}__")
}

/// The key a panel's slice of the rows published under `key` is published
/// under, in the per-panel output map the panel pass builds
/// (`scene_build::resolve_panel_scales`).
pub(in crate::render) fn panel_key(key: &str) -> String {
    format!("{key}__panel__")
}

/// The key whose batch layer `index` of `spec` draws: its own rows, else the
/// output its `data_source` names, else the pipeline tail
/// ([`FINAL_OUTPUT_KEY`], which an unlayered spec's one layer reads).
pub(in crate::render) fn spec_rows_key(spec: &ChartSpec, index: usize) -> &str {
    spec.layers.as_deref().and_then(|layers| layers.get(index)).map_or(FINAL_OUTPUT_KEY, |layer| layer.rows_key())
}

/// The transform outputs and the spec's routes, written together: the one
/// publisher of a layer's own rows. Holds the normalised spec `prepare` reads
/// (a clone: a spec carries no row data), so a route written here is visible
/// through [`Self::spec`] the moment it is published.
pub(super) struct LayerRows<'o> {
    spec: ChartSpec,
    outputs: &'o mut HashMap<String, RecordBatch>,
    lineages: &'o mut OutputLineages,
}

impl<'o> LayerRows<'o> {
    pub(super) fn new(spec: &ChartSpec, outputs: &'o mut HashMap<String, RecordBatch>, lineages: &'o mut OutputLineages) -> Self {
        Self { spec: spec.clone(), outputs, lineages }
    }

    /// The spec with every route published so far written on its layers.
    pub(super) fn spec(&self) -> &ChartSpec {
        &self.spec
    }

    pub(super) fn outputs(&self) -> &HashMap<String, RecordBatch> {
        self.outputs
    }

    pub(super) fn lineages(&self) -> &OutputLineages {
        self.lineages
    }

    /// The key whose batch layer `index` draws ([`spec_rows_key`]).
    pub(super) fn rows_key(&self, index: usize) -> &str {
        spec_rows_key(&self.spec, index)
    }

    /// Rewrite the batch published under `key`, in place, for every layer
    /// that reads it.
    pub(super) fn replace(&mut self, key: &str, rows: RecordBatch) {
        self.outputs.insert(key.to_string(), rows);
    }

    /// Give layer `index` the rows `rows` under [`own_rows_key`], with the
    /// lineage of `source_key` (the key it read) copied beside them. A layer
    /// has at most one route: publishing again for the same layer overwrites
    /// its rows.
    pub(super) fn publish_own_rows(&mut self, index: usize, source_key: &str, rows: RecordBatch) {
        let own = own_rows_key(index);
        self.outputs.insert(own.clone(), rows);
        if let Some(lineage) = self.lineages.get(source_key).cloned() {
            self.lineages.insert(own.clone(), lineage);
        }
        if let Some(layer) = self.spec.layers.as_mut().and_then(|layers| layers.get_mut(index)) {
            layer.own_rows = Some(own);
        }
    }

    pub(super) fn into_spec(self) -> ChartSpec {
        self.spec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn rows(values: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)])),
            vec![Arc::new(Float64Array::from(values.to_vec()))],
        )
        .unwrap()
    }

    fn two_layers() -> ChartSpec {
        let layer = crate::spec::layer::Layer::default();
        ChartSpec { layers: Some(vec![layer.clone(), layer]), ..Default::default() }
    }

    #[test]
    fn a_layer_reads_its_published_rows_and_a_second_publish_overwrites_the_one_route() {
        let mut outputs = HashMap::from([(FINAL_OUTPUT_KEY.to_string(), rows(&[1.0, 2.0, 3.0]))]);
        let mut lineages = OutputLineages::default();
        let mut routes = LayerRows::new(&two_layers(), &mut outputs, &mut lineages);
        assert_eq!(routes.rows_key(1), FINAL_OUTPUT_KEY);
        routes.publish_own_rows(1, FINAL_OUTPUT_KEY, rows(&[1.0]));
        routes.publish_own_rows(1, FINAL_OUTPUT_KEY, rows(&[1.0, 2.0]));
        assert_eq!(routes.rows_key(0), FINAL_OUTPUT_KEY, "a sibling keeps the key it read");
        assert_eq!(routes.rows_key(1), own_rows_key(1));
        let spec = routes.into_spec();
        assert_eq!(spec.layers.as_ref().unwrap()[1].own_rows.as_deref(), Some(own_rows_key(1).as_str()));
        assert!(spec.layers.as_ref().unwrap()[0].own_rows.is_none());
        assert_eq!(outputs[&own_rows_key(1)].num_rows(), 2, "the second publish replaced the first");
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn an_unlayered_spec_reads_the_tail() {
        let mut outputs = HashMap::new();
        let mut lineages = OutputLineages::default();
        let routes = LayerRows::new(&ChartSpec::default(), &mut outputs, &mut lineages);
        assert_eq!(routes.rows_key(0), FINAL_OUTPUT_KEY);
    }

    #[test]
    fn replace_rewrites_a_key_in_place() {
        let mut outputs = HashMap::from([(FINAL_OUTPUT_KEY.to_string(), rows(&[1.0]))]);
        let mut lineages = OutputLineages::default();
        let mut routes = LayerRows::new(&two_layers(), &mut outputs, &mut lineages);
        routes.replace(FINAL_OUTPUT_KEY, rows(&[1.0, 2.0]));
        assert_eq!(routes.outputs()[FINAL_OUTPUT_KEY].num_rows(), 2);
        assert_eq!(routes.rows_key(0), FINAL_OUTPUT_KEY);
    }
}
