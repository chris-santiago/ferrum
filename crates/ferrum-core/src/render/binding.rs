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

/// Mirror of the ferrum-spec.md §3.13 Theme keys, derived via serde and
/// `pyo3_serde::from_py` from the user's theme dict. Every field is
/// `Option<_>` so missing keys leave the corresponding `ThemeInputs`
/// field at its default.
///
/// Unknown-key handling: `#[serde(deny_unknown_fields)]` rejects typos
/// with a serde error listing the accepted fields — replaces the prior
/// hand-maintained `KNOWN_THEME_KEYS` parallel list.
///
/// Enum-valued keys (`title_anchor`, `legend_orient`, `legend_direction`)
/// are typed as `String` here so `apply_theme_overrides` can produce the
/// user-friendly `"title_anchor must be one of …"` error rather than
/// serde's `unknown variant` default.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ThemeOverridesSpec {
    // Background / padding
    mark_color: Option<String>,
    background_color: Option<String>,
    /// Alias for `background_color` (ferrum-spec.md uses `background`).
    background: Option<String>,
    padding: Option<f64>,

    // Typography
    font_family: Option<String>,
    font_weight: Option<String>,
    font_color: Option<String>,
    /// Routes to `ThemeInputs::label_font_size`, not a `font_size` field.
    font_size: Option<f64>,
    title_font_family: Option<String>,
    title_font_size: Option<f64>,
    title_font_weight: Option<String>,
    title_color: Option<String>,
    title_anchor: Option<String>,
    title_offset: Option<f64>,
    label_font_family: Option<String>,
    label_color: Option<String>,

    // Axes
    axis_line: Option<bool>,
    axis_line_color: Option<String>,
    axis_line_width: Option<f64>,
    tick_color: Option<String>,
    tick_size: Option<f64>,
    tick_width: Option<f64>,

    // Grid
    grid: Option<bool>,
    grid_color: Option<String>,
    grid_width: Option<f64>,
    grid_dash: Option<Vec<f64>>,
    grid_opacity: Option<f64>,

    // Marks
    point_size: Option<f64>,
    point_opacity: Option<f64>,
    line_stroke_width: Option<f64>,
    bar_corner_radius: Option<f64>,
    area_opacity: Option<f64>,
    /// Routes to `ThemeInputs::default_opacity`.
    opacity: Option<f64>,

    // Palette
    color_scheme: Option<String>,

    // Strip
    strip_background_color: Option<String>,

    // Legend
    legend_orient: Option<String>,
    legend_direction: Option<String>,
    legend_title_font_size: Option<f64>,

    // Spacing
    axis_title_padding: Option<f64>,
    column_padding: Option<f64>,
    row_padding: Option<f64>,
}

fn parse_hex(s: &str) -> PyResult<super::color::Color> {
    super::color::from_hex_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn apply_theme_overrides(t: &mut ThemeInputs, spec: ThemeOverridesSpec) -> PyResult<()> {
    if let Some(s) = spec.mark_color { t.mark_color = parse_hex(&s)?; }
    // `background` is an alias for `background_color`; both populate the
    // same field. Last-write-wins if a user passes both.
    if let Some(s) = spec.background_color { t.background_color = parse_hex(&s)?; }
    if let Some(s) = spec.background { t.background_color = parse_hex(&s)?; }
    if let Some(v) = spec.padding { t.padding = v; }

    // Typography
    if let Some(v) = spec.font_family { t.font_family = v; }
    if let Some(v) = spec.font_weight { t.font_weight = v; }
    if let Some(s) = spec.font_color { t.font_color = parse_hex(&s)?; }
    if let Some(v) = spec.font_size { t.label_font_size = v; }
    if let Some(v) = spec.title_font_family { t.title_font_family = v; }
    if let Some(v) = spec.title_font_size { t.title_font_size = v; }
    if let Some(v) = spec.title_font_weight { t.title_font_weight = v; }
    if let Some(s) = spec.title_color { t.title_color = parse_hex(&s)?; }
    if let Some(s) = spec.title_anchor {
        t.title_anchor = match s.as_str() {
            "start" => TextAnchor::Start,
            "middle" => TextAnchor::Middle,
            "end" => TextAnchor::End,
            other => return Err(PyValueError::new_err(format!(
                "title_anchor must be one of 'start'|'middle'|'end', got '{other}'"
            ))),
        };
    }
    if let Some(v) = spec.title_offset { t.title_offset = v; }
    if let Some(v) = spec.label_font_family { t.label_font_family = v; }
    if let Some(s) = spec.label_color { t.label_color = parse_hex(&s)?; }

    // Axes
    if let Some(v) = spec.axis_line { t.axis_line = v; }
    if let Some(s) = spec.axis_line_color { t.axis_line_color = parse_hex(&s)?; }
    if let Some(v) = spec.axis_line_width { t.axis_line_width = v; }
    if let Some(s) = spec.tick_color { t.tick_color = parse_hex(&s)?; }
    if let Some(v) = spec.tick_size { t.tick_size = v; }
    if let Some(v) = spec.tick_width { t.tick_width = v; }

    // Grid
    if let Some(v) = spec.grid { t.grid = v; }
    if let Some(s) = spec.grid_color { t.grid_color = parse_hex(&s)?; }
    if let Some(v) = spec.grid_width { t.grid_width = v; }
    if let Some(v) = spec.grid_dash { t.grid_dash = Some(v); }
    if let Some(v) = spec.grid_opacity { t.grid_opacity = v; }

    // Marks
    if let Some(v) = spec.point_size { t.point_size = v; }
    if let Some(v) = spec.point_opacity { t.point_opacity = v; }
    if let Some(v) = spec.line_stroke_width { t.line_stroke_width = v; }
    if let Some(v) = spec.bar_corner_radius { t.bar_corner_radius = v; }
    if let Some(v) = spec.area_opacity { t.area_opacity = v; }
    if let Some(v) = spec.opacity { t.default_opacity = v; }

    // Palette
    if let Some(s) = spec.color_scheme {
        if !super::palette::is_categorical_scheme(&s)
            && !super::palette::is_sequential_scheme(&s)
        {
            return Err(PyValueError::new_err(format!(
                "Unknown color_scheme: '{s}'. Supported categorical: {}. \
                 Supported sequential: {}.",
                super::palette::CATEGORICAL_SCHEMES.join(", "),
                super::palette::SEQUENTIAL_SCHEMES.join(", "),
            )));
        }
        t.color_scheme = s;
    }

    // Strip
    if let Some(s) = spec.strip_background_color { t.strip_background_color = parse_hex(&s)?; }

    // Legend
    if let Some(s) = spec.legend_orient {
        t.legend_orient = match s.as_str() {
            "left" => LegendOrient::Left,
            "right" => LegendOrient::Right,
            "top" => LegendOrient::Top,
            "bottom" => LegendOrient::Bottom,
            other => return Err(PyValueError::new_err(format!(
                "legend_orient must be one of 'left'|'right'|'top'|'bottom', got '{other}'"
            ))),
        };
    }
    if let Some(s) = spec.legend_direction {
        t.legend_direction = Some(match s.as_str() {
            "horizontal" => LegendDirection::Horizontal,
            "vertical" => LegendDirection::Vertical,
            other => return Err(PyValueError::new_err(format!(
                "legend_direction must be one of 'horizontal'|'vertical', got '{other}'"
            ))),
        });
    }
    if let Some(v) = spec.legend_title_font_size { t.legend_title_font_size = v; }

    // Spacing
    if let Some(v) = spec.axis_title_padding { t.axis_title_padding = v; }
    if let Some(v) = spec.column_padding { t.column_padding = v; }
    if let Some(v) = spec.row_padding { t.row_padding = v; }

    Ok(())
}

fn theme_from_dict(d: Option<&Bound<'_, PyDict>>) -> PyResult<ThemeInputs> {
    let mut t = ThemeInputs::default();
    let Some(d) = d else { return Ok(t) };
    let spec: ThemeOverridesSpec = crate::pyo3_serde::from_py(d.as_any(), "theme")?;
    apply_theme_overrides(&mut t, spec)?;
    Ok(t)
}

#[cfg(test)]
mod theme_dict_tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn unknown_key_raises() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("not_a_real_key", "value").unwrap();
            let err = theme_from_dict(Some(&d)).unwrap_err();
            let msg = err.value(py).to_string();
            // serde's `deny_unknown_fields` produces "unknown field `foo`,
            // expected one of …" — wrapped by pyo3_serde with a "theme:" prefix.
            assert!(msg.contains("unknown field"), "got: {msg}");
            assert!(msg.contains("not_a_real_key"), "got: {msg}");
        });
    }

    #[test]
    fn background_alias_accepted() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("background", "#ff0000").unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.background_color.red, 0xFF);
            assert_eq!(t.background_color.green, 0x00);
            assert_eq!(t.background_color.blue, 0x00);
        });
    }

    #[test]
    fn unknown_color_scheme_raises() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("color_scheme", "nonexistent").unwrap();
            let err = theme_from_dict(Some(&d)).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("Unknown color_scheme"), "got: {msg}");
            assert!(msg.contains("nonexistent"), "got: {msg}");
        });
    }

    #[test]
    fn known_color_schemes_accepted() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            for name in [
                "okabe_ito", "tableau10", "set1", "set2", "paired", "pastel",
                "dark2", "viridis", "plasma", "magma", "inferno", "cividis",
            ] {
                let d = PyDict::new(py);
                d.set_item("color_scheme", name).unwrap();
                let t = theme_from_dict(Some(&d)).expect(name);
                assert_eq!(t.color_scheme, name);
            }
        });
    }
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
