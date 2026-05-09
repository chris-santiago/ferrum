//! PyO3 binding: `compute_layout(spec, viewport, axes, facet_groups, legend_entries)`
//! returns a Python dict. ThemeInputs and TextMetrics are not yet exposed —
//! Phase 6 always uses HeuristicMetrics + ThemeInputs::default(); Phase 8 will
//! map ferrum.Theme into ThemeInputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::axis::{AxesInput, AxisInput, AxisOrient};
use super::facet::FacetGroup;
use super::legend::{LegendEntry, LegendOrient, SymbolKind};
use super::panel::FacetKey;
use super::text_metrics::HeuristicMetrics;
use super::{compute_layout as compute_layout_internal, ThemeInputs, Viewport};

#[pyfunction]
#[pyo3(signature = (
    spec,
    *,
    viewport,
    x_tick_labels,
    y_tick_labels,
    x_title = None,
    y_title = None,
    facet_groups = None,
    legend_entries = None,
    legend_orient = "right",
    label_angle = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn compute_layout(
    py: Python<'_>,
    spec: &crate::spec::chart::ChartSpec,
    viewport: (f64, f64),
    x_tick_labels: Vec<String>,
    y_tick_labels: Vec<String>,
    x_title: Option<String>,
    y_title: Option<String>,
    facet_groups: Option<Vec<(String, String, u64)>>,
    legend_entries: Option<Vec<(String, String)>>,
    legend_orient: &str,
    label_angle: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let viewport = Viewport { width: viewport.0, height: viewport.1 };
    let mut theme = ThemeInputs::default();
    theme.legend_orient = parse_legend_orient(legend_orient)?;

    let axes = AxesInput {
        x: AxisInput {
            orient: AxisOrient::Bottom,
            title: x_title,
            tick_labels: x_tick_labels,
            label_angle_override: label_angle,
        },
        y: AxisInput {
            orient: AxisOrient::Left,
            title: y_title,
            tick_labels: y_tick_labels,
            label_angle_override: None,
        },
    };

    let groups: Vec<FacetGroup> = facet_groups
        .unwrap_or_default()
        .into_iter()
        .map(|(field, value, n_rows)| FacetGroup {
            key: FacetKey { field, value },
            n_rows,
        })
        .collect();

    let entries: Vec<LegendEntry> = legend_entries
        .unwrap_or_default()
        .into_iter()
        .map(|(label, kind)| {
            let symbol = match kind.as_str() {
                "circle" => SymbolKind::Circle,
                "square" => SymbolKind::Square,
                "line" => SymbolKind::Line,
                _ => SymbolKind::Circle,
            };
            LegendEntry { label, symbol }
        })
        .collect();

    let metrics = HeuristicMetrics::default();
    let result = compute_layout_internal(
        spec, &theme, viewport, &axes, &groups, &entries, &metrics,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let json = serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("internal serde error: {e}")))?;
    let json_module = py.import("json")?;
    let parsed = json_module.call_method1("loads", (json,))?;
    let dict: Py<PyDict> = parsed.extract()?;
    Ok(dict)
}

fn parse_legend_orient(s: &str) -> PyResult<LegendOrient> {
    match s {
        "right" => Ok(LegendOrient::Right),
        "left" => Ok(LegendOrient::Left),
        "top" => Ok(LegendOrient::Top),
        "bottom" => Ok(LegendOrient::Bottom),
        other => Err(PyValueError::new_err(format!(
            "legend_orient must be one of right|left|top|bottom; got '{other}'"
        ))),
    }
}
