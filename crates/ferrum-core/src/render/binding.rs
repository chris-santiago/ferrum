//! PyO3 bindings: render_svg, render_png. Theme/RenderConfig pass via Python dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3_arrow::PyRecordBatchReader;

use crate::layout::{ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;

use super::config::RenderConfig;
use super::RenderError;
use super::{render_png as render_png_internal, render_svg as render_svg_internal};

#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None))]
pub fn render_svg(
    py: Python<'_>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let vp = Viewport {
        width: viewport.0,
        height: viewport.1,
    };
    let result = render_svg_internal(spec, &batch, &t, vp, &c).map_err(render_err_to_py)?;
    emit_warnings(py, &result.warnings)?;
    Ok(result.bytes)
}

#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None))]
pub fn render_png<'py>(
    py: Python<'py>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let vp = Viewport {
        width: viewport.0,
        height: viewport.1,
    };
    let result = render_png_internal(spec, &batch, &t, vp, &c).map_err(render_err_to_py)?;
    emit_warnings(py, &result.warnings)?;
    Ok(PyBytes::new(py, &result.bytes))
}

fn collect_single_batch(reader: PyRecordBatchReader) -> PyResult<arrow::record_batch::RecordBatch> {
    let iter = reader
        .into_reader()
        .map_err(|e| PyValueError::new_err(format!("arrow reader: {e}")))?;
    let mut all = Vec::new();
    for next in iter {
        all.push(next.map_err(|e| PyValueError::new_err(format!("arrow read: {e}")))?);
    }
    if all.is_empty() {
        return Err(PyValueError::new_err("empty record batch stream"));
    }
    if all.len() == 1 {
        Ok(all.into_iter().next().unwrap())
    } else {
        let schema = all[0].schema();
        arrow::compute::concat_batches(&schema, &all)
            .map_err(|e| PyValueError::new_err(format!("concat batches: {e}")))
    }
}

fn theme_from_dict(d: Option<&Bound<'_, PyDict>>) -> PyResult<ThemeInputs> {
    let mut t = ThemeInputs::default();
    let d = match d {
        Some(x) => x,
        None => return Ok(t),
    };
    if let Some(v) = d.get_item("mark_color")? {
        let s: String = v.extract()?;
        t.mark_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("background_color")? {
        let s: String = v.extract()?;
        t.background_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("point_size")? {
        t.point_size = v.extract()?;
    }
    if let Some(v) = d.get_item("line_stroke_width")? {
        t.line_stroke_width = v.extract()?;
    }
    if let Some(v) = d.get_item("bar_corner_radius")? {
        t.bar_corner_radius = v.extract()?;
    }
    if let Some(v) = d.get_item("area_opacity")? {
        t.area_opacity = v.extract()?;
    }
    if let Some(v) = d.get_item("grid")? {
        t.grid = v.extract()?;
    }
    if let Some(v) = d.get_item("padding")? {
        t.padding = v.extract()?;
    }
    Ok(t)
}

fn config_from_dict(d: Option<&Bound<'_, PyDict>>) -> PyResult<RenderConfig> {
    let mut c = RenderConfig::default();
    let d = match d {
        Some(x) => x,
        None => return Ok(c),
    };
    if let Some(v) = d.get_item("scale")? {
        c.scale = v.extract()?;
    }
    if let Some(v) = d.get_item("embed_fonts")? {
        c.embed_fonts = v.extract()?;
    }
    if let Some(v) = d.get_item("background")? {
        let s: String = v.extract()?;
        c.background = Some(
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    if let Some(v) = d.get_item("width")? {
        c.width = Some(v.extract()?);
    }
    if let Some(v) = d.get_item("height")? {
        c.height = Some(v.extract()?);
    }
    Ok(c)
}

fn render_err_to_py(e: RenderError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// SVG compositor bindings (Task 11)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "compose_svg_horizontal")]
#[pyo3(signature = (svgs, *, spacing = 10.0, align = "top"))]
pub fn compose_svg_horizontal_py(
    svgs: Vec<String>,
    spacing: f64,
    align: &str,
) -> PyResult<String> {
    let align_val = match align {
        "top" => crate::render::compositor::VerticalAlign::Top,
        "center" => crate::render::compositor::VerticalAlign::Center,
        "bottom" => crate::render::compositor::VerticalAlign::Bottom,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "align must be one of 'top'|'center'|'bottom', got '{other}'"
            )))
        }
    };
    crate::render::compositor::compose_svg_horizontal(&svgs, spacing, align_val)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "compose_svg_vertical")]
#[pyo3(signature = (svgs, *, spacing = 10.0, align = "left"))]
pub fn compose_svg_vertical_py(
    svgs: Vec<String>,
    spacing: f64,
    align: &str,
) -> PyResult<String> {
    let align_val = match align {
        "left" => crate::render::compositor::HorizontalAlign::Left,
        "center" => crate::render::compositor::HorizontalAlign::Center,
        "right" => crate::render::compositor::HorizontalAlign::Right,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "align must be one of 'left'|'center'|'right', got '{other}'"
            )))
        }
    };
    crate::render::compositor::compose_svg_vertical(&svgs, spacing, align_val)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "compose_svg_grid")]
#[pyo3(signature = (cells, *, rows, cols, row_ratios, col_ratios, spacing = 10.0,
                     share_x = Vec::<Vec<usize>>::new(), share_y = Vec::<Vec<usize>>::new()))]
#[allow(unused_variables)]
pub fn compose_svg_grid_py(
    cells: Vec<Option<String>>,
    rows: usize,
    cols: usize,
    row_ratios: Vec<f64>,
    col_ratios: Vec<f64>,
    spacing: f64,
    share_x: Vec<Vec<usize>>,
    share_y: Vec<Vec<usize>>,
) -> PyResult<String> {
    crate::render::grid_compose::compose_svg_grid(
        &cells,
        rows,
        cols,
        &row_ratios,
        &col_ratios,
        spacing,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

fn emit_warnings(py: Python<'_>, warnings: &[super::RenderWarning]) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let warnings_mod = py.import("warnings")?;
    for w in warnings {
        let msg = format!("{w:?}");
        warnings_mod.call_method1("warn", (msg,))?;
    }
    Ok(())
}
