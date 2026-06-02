//! PyO3 binding: `compute_layout(spec, viewport, axes, facet_groups, legend_entries)`
//! returns a Python dict. Uses HeuristicMetrics for text measurement and
//! ThemeInputs::default() with `legend_orient` patched from the caller.
//! The render pipeline receives the full theme from Python; this layout
//! entry point does not yet accept the full theme dict.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::axis::{AxesInput, AxisInput, AxisOrient};
use super::facet::FacetGroup;
use super::legend::{LegendEntry, LegendOrient, SymbolKind};
use super::panel::FacetKey;
use super::text_metrics::HeuristicMetrics;
use super::{compute_layout as compute_layout_internal, ThemeInputs, Viewport};

/// Compute a chart's layout geometry and return it as a dict.
///
/// Parameters
/// ----------
/// spec : ChartSpec
///     Compiled chart specification used to determine axis types, facet
///     structure, and legend presence.
/// viewport : tuple[float, float]
///     ``(width, height)`` of the total drawing area in pixels.
/// x_tick_labels : list[str]
///     Formatted tick labels for the x-axis, used to estimate label width.
/// y_tick_labels : list[str]
///     Formatted tick labels for the y-axis, used to estimate label height.
/// x_title : str, optional
///     Axis title for the x-axis.
/// y_title : str, optional
///     Axis title for the y-axis.
/// facet_groups : list[tuple[str, str, int]], optional
///     Sequence of ``(field, value, n_rows)`` triples describing each facet
///     panel. Pass ``None`` for non-faceted charts.
/// legend_entries : list[tuple[str, str]], optional
///     Sequence of ``(label, kind)`` pairs for the legend. ``kind`` is one
///     of ``"circle"``, ``"square"``, or ``"line"``.
/// legend_orient : str, default "right"
///     Legend placement. One of ``"right"``, ``"left"``, ``"top"``,
///     ``"bottom"``.
/// label_angle : float, optional
///     Override the x-axis tick label rotation angle in degrees.
///
/// Returns
/// -------
/// dict
///     Layout geometry serialised as a JSON-derived dict with keys such as
///     ``plot_area``, ``margins``, ``facet_panels``, and ``legend_box``.
///     The exact schema follows the internal ``LayoutResult`` type.
///
/// Raises
/// ------
/// ValueError
///     If ``legend_orient`` is not one of the accepted values, or if the
///     layout engine cannot satisfy the viewport constraints.
///
/// Notes
/// -----
/// Uses ``HeuristicMetrics`` for text measurement and ``ThemeInputs::default()``
/// with ``legend_orient`` patched from the caller argument. The render pipeline
/// receives the full Python-side theme; this layout entry point does not yet
/// accept the full theme dict.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y")
/// >>> layout = fm.compute_layout(chart.to_spec(), viewport=(400, 300),
/// ...     x_tick_labels=[], y_tick_labels=[])
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
    theme.legend.legend_orient = parse_legend_orient(legend_orient)?;

    let axes = AxesInput {
        x: AxisInput::new(AxisOrient::Bottom, x_title, x_tick_labels, label_angle),
        y: AxisInput::new(AxisOrient::Left, y_title, y_tick_labels, None),
        show_x: true,
        show_y: true,
    };

    let groups: Vec<FacetGroup> = facet_groups
        .unwrap_or_default()
        .into_iter()
        .map(|(field, value, n_rows)| FacetGroup {
            key: FacetKey { field, value },
            n_rows,
            row_key: None,
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
        spec, &theme, viewport, &axes, &groups, &entries, None, None, &metrics,
        &super::legend::LegendOverrides::default(),
        &[],
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
