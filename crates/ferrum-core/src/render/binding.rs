//! PyO3 bindings: render_svg, render_png. Theme/RenderConfig pass via Python dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3_arrow::PyRecordBatchReader;

use crate::layout::{LegendDirection, LegendOrient, TextAnchor, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;

use super::config::RenderConfig;
use super::RenderError;
use super::{render_png as render_png_internal, render_svg as render_svg_internal};

/// Render a ``ChartSpec`` and Arrow batch to an SVG string.
///
/// Parameters
/// ----------
/// spec : ChartSpec
///     Compiled chart specification produced by ``Chart.to_spec()``.
/// data : pyarrow.RecordBatchReader or compatible
///     Input data stream. Columns must satisfy the encoding fields declared
///     in *spec*. Polars ``DataFrame`` and pyarrow objects are accepted
///     directly via the Arrow C Data Interface (zero copy).
/// viewport : tuple[float, float]
///     ``(width, height)`` of the output SVG canvas in pixels.
/// theme : dict, optional
///     Sparse theme override dict. Accepted keys: ``mark_color``,
///     ``background_color``, ``point_size``, ``line_stroke_width``,
///     ``bar_corner_radius``, ``area_opacity``, ``grid``, ``padding``.
///     Unset keys fall back to ``ThemeInputs`` defaults.
/// config : dict, optional
///     Render-config dict. Accepted keys: ``scale``, ``embed_fonts``,
///     ``background``, ``width``, ``height``.
///
/// Returns
/// -------
/// str
///     Complete SVG document as a UTF-8 string.
///
/// Raises
/// ------
/// ValueError
///     If the data stream is empty, a batch cannot be read, or the spec
///     references a column absent from the data.
///
/// Notes
/// -----
/// Output is byte-deterministic given the same *spec*, *data*, and *theme*
/// inputs. Any stochastic transforms (e.g. ``Jitter``, bootstrap CI) use a
/// seeded ``ChaCha8Rng`` keyed from the transform's ``seed`` field
/// (CLAUDE.md "byte-deterministic randomness"). Render warnings (e.g.
/// unsupported encoding combinations) are forwarded to Python's
/// ``warnings.warn``.
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

/// Render a ``ChartSpec`` and Arrow batch to PNG bytes.
///
/// Parameters
/// ----------
/// spec : ChartSpec
///     Compiled chart specification produced by ``Chart.to_spec()``.
/// data : pyarrow.RecordBatchReader or compatible
///     Input data stream. Columns must satisfy the encoding fields declared
///     in *spec*. Polars ``DataFrame`` and pyarrow objects are accepted
///     directly via the Arrow C Data Interface (zero copy).
/// viewport : tuple[float, float]
///     ``(width, height)`` of the output image in pixels (before the
///     ``config["scale"]`` multiplier is applied).
/// theme : dict, optional
///     Sparse theme override dict. Accepted keys: ``mark_color``,
///     ``background_color``, ``point_size``, ``line_stroke_width``,
///     ``bar_corner_radius``, ``area_opacity``, ``grid``, ``padding``.
///     Unset keys fall back to ``ThemeInputs`` defaults.
/// config : dict, optional
///     Render-config dict. Accepted keys: ``scale`` (pixel ratio, default
///     1.0), ``embed_fonts``, ``background``, ``width``, ``height``.
///
/// Returns
/// -------
/// bytes
///     PNG image as raw bytes suitable for ``IPython.display.Image`` or
///     writing directly to disk.
///
/// Raises
/// ------
/// ValueError
///     If the data stream is empty, a batch cannot be read, or the spec
///     references a column absent from the data.
///
/// Notes
/// -----
/// PNG output is produced by rasterising the SVG pipeline result. Output
/// is byte-deterministic for the same inputs (seeded ``ChaCha8Rng`` for
/// stochastic transforms). Render warnings are forwarded to Python's
/// ``warnings.warn``.
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

    // -------- Themes-T1 additions --------

    // Spec uses `background`; binding originally read `background_color`. Accept both.
    if let Some(v) = d.get_item("background")? {
        let s: String = v.extract()?;
        t.background_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    // Typography
    if let Some(v) = d.get_item("font_family")? {
        t.font_family = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("font_weight")? {
        t.font_weight = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("font_color")? {
        let s: String = v.extract()?;
        t.font_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("font_size")? {
        t.label_font_size = v.extract()?;
    }
    if let Some(v) = d.get_item("title_font_family")? {
        t.title_font_family = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("title_font_size")? {
        t.title_font_size = v.extract()?;
    }
    if let Some(v) = d.get_item("title_font_weight")? {
        t.title_font_weight = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("title_color")? {
        let s: String = v.extract()?;
        t.title_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("title_anchor")? {
        let s: String = v.extract()?;
        t.title_anchor = match s.as_str() {
            "start" => TextAnchor::Start,
            "middle" => TextAnchor::Middle,
            "end" => TextAnchor::End,
            other => {
                return Err(PyValueError::new_err(format!(
                    "title_anchor must be one of 'start'|'middle'|'end', got '{other}'"
                )))
            }
        };
    }
    if let Some(v) = d.get_item("title_offset")? {
        t.title_offset = v.extract()?;
    }
    if let Some(v) = d.get_item("label_font_family")? {
        t.label_font_family = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("label_color")? {
        let s: String = v.extract()?;
        t.label_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    // Axes
    if let Some(v) = d.get_item("axis_line")? {
        t.axis_line = v.extract()?;
    }
    if let Some(v) = d.get_item("axis_line_color")? {
        let s: String = v.extract()?;
        t.axis_line_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("axis_line_width")? {
        t.axis_line_width = v.extract()?;
    }
    if let Some(v) = d.get_item("tick_color")? {
        let s: String = v.extract()?;
        t.tick_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("tick_size")? {
        t.tick_size = v.extract()?;
    }
    if let Some(v) = d.get_item("tick_width")? {
        t.tick_width = v.extract()?;
    }

    // Grid
    if let Some(v) = d.get_item("grid_color")? {
        let s: String = v.extract()?;
        t.grid_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("grid_width")? {
        t.grid_width = v.extract()?;
    }
    if let Some(v) = d.get_item("grid_dash")? {
        let dashes: Vec<f64> = v.extract()?;
        t.grid_dash = Some(dashes);
    }
    if let Some(v) = d.get_item("grid_opacity")? {
        t.grid_opacity = v.extract()?;
    }

    // Marks
    if let Some(v) = d.get_item("point_opacity")? {
        t.point_opacity = v.extract()?;
    }
    if let Some(v) = d.get_item("opacity")? {
        t.default_opacity = v.extract()?;
    }

    // Palette
    if let Some(v) = d.get_item("color_scheme")? {
        t.color_scheme = v.extract::<String>()?;
    }

    // Strip
    if let Some(v) = d.get_item("strip_background_color")? {
        let s: String = v.extract()?;
        t.strip_background_color =
            super::color::from_hex_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    // Legend
    if let Some(v) = d.get_item("legend_orient")? {
        let s: String = v.extract()?;
        t.legend_orient = match s.as_str() {
            "left" => LegendOrient::Left,
            "right" => LegendOrient::Right,
            "top" => LegendOrient::Top,
            "bottom" => LegendOrient::Bottom,
            other => {
                return Err(PyValueError::new_err(format!(
                    "legend_orient must be one of 'left'|'right'|'top'|'bottom', got '{other}'"
                )))
            }
        };
    }
    if let Some(v) = d.get_item("legend_direction")? {
        let s: String = v.extract()?;
        t.legend_direction = match s.as_str() {
            "horizontal" => LegendDirection::Horizontal,
            "vertical" => LegendDirection::Vertical,
            other => {
                return Err(PyValueError::new_err(format!(
                    "legend_direction must be one of 'horizontal'|'vertical', got '{other}'"
                )))
            }
        };
    }
    if let Some(v) = d.get_item("legend_title_font_size")? {
        t.legend_title_font_size = v.extract()?;
    }

    // Spacing
    if let Some(v) = d.get_item("axis_title_padding")? {
        t.axis_title_padding = v.extract()?;
    }
    if let Some(v) = d.get_item("column_padding")? {
        t.column_padding = v.extract()?;
    }
    if let Some(v) = d.get_item("row_padding")? {
        t.row_padding = v.extract()?;
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

/// Compose SVG panels side-by-side into a single horizontal strip.
///
/// Parameters
/// ----------
/// svgs : list[str]
///     SVG document strings to lay out left-to-right. Each must be a valid
///     SVG with a parseable ``viewBox`` or ``width``/``height`` attribute.
/// spacing : float, default 10.0
///     Gap in pixels between adjacent panels.
/// align : str, default "top"
///     Vertical alignment of panels with different heights. One of
///     ``"top"``, ``"center"``, or ``"bottom"``.
///
/// Returns
/// -------
/// str
///     A single SVG document whose width equals the sum of panel widths
///     plus total spacing and whose height equals the tallest panel.
///
/// Raises
/// ------
/// ValueError
///     If *align* is not one of the accepted values, or if any SVG string
///     cannot be parsed.
///
/// Notes
/// -----
/// Used internally by ``HConcatChart`` to combine column-concatenated
/// charts. The returned SVG preserves each panel's coordinate system via
/// nested ``<g transform="translate(...)">`` elements.
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

/// Compose SVG panels stacked top-to-bottom into a single vertical strip.
///
/// Parameters
/// ----------
/// svgs : list[str]
///     SVG document strings to lay out top-to-bottom. Each must be a valid
///     SVG with a parseable ``viewBox`` or ``width``/``height`` attribute.
/// spacing : float, default 10.0
///     Gap in pixels between adjacent panels.
/// align : str, default "left"
///     Horizontal alignment of panels with different widths. One of
///     ``"left"``, ``"center"``, or ``"right"``.
///
/// Returns
/// -------
/// str
///     A single SVG document whose height equals the sum of panel heights
///     plus total spacing and whose width equals the widest panel.
///
/// Raises
/// ------
/// ValueError
///     If *align* is not one of the accepted values, or if any SVG string
///     cannot be parsed.
///
/// Notes
/// -----
/// Used internally by ``VConcatChart`` to combine row-concatenated charts.
/// The returned SVG preserves each panel's coordinate system via nested
/// ``<g transform="translate(...)">`` elements.
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

/// Compose SVG panels into a rectangular grid.
///
/// Parameters
/// ----------
/// cells : list[str | None]
///     Flat list of SVG document strings in row-major order (length must
///     equal *rows* × *cols*). Pass ``None`` for empty cells.
/// rows : int
///     Number of grid rows.
/// cols : int
///     Number of grid columns.
/// row_ratios : list[float]
///     Relative height weight for each row (length must equal *rows*).
///     E.g. ``[2.0, 1.0]`` makes the first row twice as tall as the second.
/// col_ratios : list[float]
///     Relative width weight for each column (length must equal *cols*).
/// spacing : float, default 10.0
///     Gap in pixels between adjacent cells (applied both horizontally and
///     vertically).
/// share_x : list[list[int]], default []
///     Groups of cell indices whose x-axes should share the same scale
///     range. Reserved for future alignment; currently accepted but not
///     applied.
/// share_y : list[list[int]], default []
///     Groups of cell indices whose y-axes should share the same scale
///     range. Reserved for future alignment; currently accepted but not
///     applied.
///
/// Returns
/// -------
/// str
///     A single SVG document containing all cells positioned according to
///     the ratio-weighted grid layout.
///
/// Raises
/// ------
/// ValueError
///     If ``len(cells) != rows * cols``, ratios lists have wrong lengths,
///     or any non-``None`` cell SVG cannot be parsed.
///
/// Notes
/// -----
/// Used internally by ``RepeatChart`` and the figure-level ``pairplot`` /
/// ``clustermap`` combinators. Each cell is embedded via a nested
/// ``<g transform="translate(...)">`` preserving its internal coordinate
/// system.
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
