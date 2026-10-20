//! The input columns each transform step reads, the one answer behind
//! [`TransformSpec::fields_read`](crate::transform::core::TransformSpec::fields_read).
//!
//! Every leaf spec implements [`FieldsRead`], and the dispatch runs through
//! `for_each_transform!`, so a new transform does not compile until it
//! states what it reads. A column is listed when the step looks it up in its
//! input rows: its fields, its groupby, its sort and key columns. Output
//! names (`as_`) are not read, and neither is a column a step reads from a
//! named output rather than its input (`Reorder(from=)`).

use std::collections::BTreeSet;

use crate::transform::aggregate::{AggregateOp, AggregateSpec};
use crate::transform::bin::BinSpec;
use crate::transform::bin_2d::Bin2DSpec;
use crate::transform::box_stats::BoxStatsSpec;
use crate::transform::calculate::CalculateSpec;
use crate::transform::contour::ContourSpec;
use crate::transform::data_aggregate::DataAggregateSpec;
use crate::transform::data_bin::DataBinSpec;
use crate::transform::data_stack::DataStackSpec;
use crate::transform::data_window::DataWindowSpec;
use crate::transform::density_data::DensityDataSpec;
use crate::transform::error_extent::ErrorExtentSpec;
use crate::transform::expr::Expr;
use crate::transform::filter::FilterSpec;
use crate::transform::flatten::FlattenSpec;
use crate::transform::fold::FoldSpec;
use crate::transform::glm::GlmSpec;
use crate::transform::hex::HexSpec;
use crate::transform::identity::IdentitySpec;
use crate::transform::impute::ImputeSpec;
use crate::transform::join_aggregate::{AggSpec, JoinAggregateSpec};
use crate::transform::kde::KdeSpec;
use crate::transform::kde_2d::Kde2DSpec;
use crate::transform::letter_value::LetterValueSpec;
use crate::transform::linkage::LinkageSpec;
use crate::transform::loess_data::LoessDataSpec;
use crate::transform::logistic::LogisticSpec;
use crate::transform::outliers::OutliersSpec;
use crate::transform::pivot::PivotSpec;
use crate::transform::project::ProjectSpec;
use crate::transform::qq::QQSpec;
use crate::transform::raster::RasterSpec;
use crate::transform::reference_line::ReferenceLineSpec;
use crate::transform::regression_data::RegressionDataSpec;
use crate::transform::reorder::ReorderSpec;
use crate::transform::robust::RobustSpec;
use crate::transform::sample::SampleSpec;
use crate::transform::smooth::SmoothSpec;
use crate::transform::summary::SummarySpec;
use crate::transform::swarm::SwarmSpec;
use crate::transform::timeunit::TimeUnitSpec;
use crate::transform::top_k::TopKSpec;
use crate::transform::unpivot::UnpivotSpec;
use crate::transform::violin::ViolinSpec;

/// The input columns one step reads.
pub(crate) trait FieldsRead {
    fn fields_read(&self) -> BTreeSet<String>;
}

/// The owned set of `names`.
fn set<'a>(names: impl IntoIterator<Item = &'a String>) -> BTreeSet<String> {
    names.into_iter().cloned().collect()
}

/// The columns an expression names; an unparseable expression names none,
/// and its step refuses the parse error itself when it runs.
fn expression_fields(source: &str) -> BTreeSet<String> {
    Expr::parse(source).map(|expr| expr.field_refs()).unwrap_or_default()
}

/// The field an aggregate op reads, none for a row count (`count()`).
fn op_fields(ops: &[AggregateOp]) -> impl Iterator<Item = &String> {
    ops.iter().filter(|op| !op.is_count_star()).map(|op| &op.field)
}

fn agg_fields(aggregates: &[AggSpec]) -> impl Iterator<Item = &String> {
    aggregates.iter().map(|agg| &agg.field)
}

impl FieldsRead for AggregateSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(op_fields(&self.ops).chain(&self.groupby))
    }
}

impl FieldsRead for BinSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        let aggregates = self.aggregates.as_deref().unwrap_or_default();
        set(std::iter::once(&self.field).chain(&self.groupby).chain(op_fields(aggregates)).chain(&self.agg_field))
    }
}

impl FieldsRead for Bin2DSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y])
    }
}

impl FieldsRead for KdeSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for Kde2DSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(&self.groupby))
    }
}

impl FieldsRead for SmoothSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(&self.groupby))
    }
}

impl FieldsRead for SummarySpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for OutliersSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for ErrorExtentSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for BoxStatsSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for ViolinSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for ContourSpec {
    /// The `Kde2D` grid columns it contours.
    fn fields_read(&self) -> BTreeSet<String> {
        ["grid_x", "grid_y", "density", "nx", "ny"].into_iter().map(String::from).collect()
    }
}

impl FieldsRead for QQSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for LinkageSpec {
    /// Every numeric column, none by name.
    fn fields_read(&self) -> BTreeSet<String> {
        BTreeSet::new()
    }
}

impl FieldsRead for RasterSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(&self.field))
    }
}

impl FieldsRead for HexSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(&self.field))
    }
}

impl FieldsRead for SwarmSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.category, &self.value].into_iter().chain(&self.dodge))
    }
}

impl FieldsRead for UnpivotSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(self.id_vars.iter().chain(self.value_vars.iter().flatten()))
    }
}

impl FieldsRead for ReorderSpec {
    /// Its index column, from its input unless it reads a named output.
    fn fields_read(&self) -> BTreeSet<String> {
        match self.from {
            Some(_) => BTreeSet::new(),
            None => set([&self.by]),
        }
    }
}

impl FieldsRead for ReferenceLineSpec {
    /// It emits its two endpoints and reads no rows.
    fn fields_read(&self) -> BTreeSet<String> {
        BTreeSet::new()
    }
}

impl FieldsRead for LetterValueSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.value).chain(&self.groupby).chain(&self.group))
    }
}

impl FieldsRead for LogisticSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y])
    }
}

impl FieldsRead for GlmSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y])
    }
}

impl FieldsRead for RobustSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(&self.groupby))
    }
}

impl FieldsRead for IdentitySpec {
    fn fields_read(&self) -> BTreeSet<String> {
        BTreeSet::new()
    }
}

impl FieldsRead for FilterSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        expression_fields(&self.predicate)
    }
}

impl FieldsRead for CalculateSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        expression_fields(&self.expr)
    }
}

impl FieldsRead for ProjectSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(&self.fields)
    }
}

impl FieldsRead for FoldSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(&self.fields)
    }
}

impl FieldsRead for PivotSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.field, &self.value].into_iter().chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for JoinAggregateSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(agg_fields(&self.aggregates).chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for DataWindowSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        let op_fields = self.ops.iter().filter_map(|op| op.field.as_ref()).filter(|field| !field.is_empty());
        set(op_fields.chain(self.sort.iter().flatten()).chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for DensityDataSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for RegressionDataSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for LoessDataSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.x, &self.y].into_iter().chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for ImputeSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(self.groupby.iter().flatten()).chain(&self.key))
    }
}

impl FieldsRead for FlattenSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(&self.fields)
    }
}

impl FieldsRead for SampleSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        BTreeSet::new()
    }
}

impl FieldsRead for TopKSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(self.groupby.iter().flatten()))
    }
}

impl FieldsRead for DataStackSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(std::iter::once(&self.field).chain(&self.groupby))
    }
}

impl FieldsRead for TimeUnitSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.field])
    }
}

impl FieldsRead for DataBinSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set([&self.field])
    }
}

impl FieldsRead for DataAggregateSpec {
    fn fields_read(&self) -> BTreeSet<String> {
        set(agg_fields(&self.aggregates).chain(self.groupby.iter().flatten()))
    }
}
