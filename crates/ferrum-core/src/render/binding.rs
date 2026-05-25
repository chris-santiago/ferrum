//! PyO3 bindings: render_svg, render_png. Theme/RenderConfig pass via Python dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3_arrow::PyRecordBatchReader;

use crate::layout::{LegendDirection, LegendOrient, TextAnchor, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;

use super::chart_config::ChartConfig;
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
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y")
/// >>> svg = fm.render_svg(chart.to_spec(), df, viewport=(400, 300))
#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None, chart_config = None))]
pub fn render_svg(
    py: Python<'_>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let cc = chart_config_from_dict(chart_config)?;
    let vp = Viewport {
        width: viewport.0,
        height: viewport.1,
    };
    let result = render_svg_internal(spec, &batch, &t, vp, &c, &cc).map_err(render_err_to_py)?;
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
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> chart = fm.Chart(df).mark_point().encode(x="x", y="y")
/// >>> png_bytes = fm.render_png(chart.to_spec(), df, viewport=(400, 300))
#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None, chart_config = None))]
pub fn render_png<'py>(
    py: Python<'py>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let cc = chart_config_from_dict(chart_config)?;
    let vp = Viewport {
        width: viewport.0,
        height: viewport.1,
    };
    let result = render_png_internal(spec, &batch, &t, vp, &c, &cc).map_err(render_err_to_py)?;
    emit_warnings(py, &result.warnings)?;
    Ok(PyBytes::new(py, &result.bytes))
}

#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None, chart_config = None))]
pub fn render_interactive(
    py: Python<'_>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(String, Py<PyBytes>)> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let cc = chart_config_from_dict(chart_config)?;
    let vp = Viewport {
        width: viewport.0,
        height: viewport.1,
    };
    let (json, packed_bytes) = super::render_scene_json(spec, &batch, &t, vp, &c, &cc)
        .map_err(render_err_to_py)?;
    let py_bytes = PyBytes::new(py, &packed_bytes);
    Ok((json, py_bytes.unbind()))
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
    sequential_scheme: Option<String>,
    diverging_scheme: Option<String>,

    // Strip
    strip_background_color: Option<String>,
    strip_text_size: Option<f64>,
    strip_padding: Option<f64>,

    // Legend
    legend_orient: Option<String>,
    legend_direction: Option<String>,
    legend_title_font_size: Option<f64>,

    // Reference lines
    reference_line_color: Option<String>,
    reference_line_dash: Option<Vec<f64>>,

    // Spacing
    axis_title_padding: Option<f64>,
    column_padding: Option<f64>,
    row_padding: Option<f64>,

    // Axis label culling
    cull_threshold: Option<u32>,
}

fn parse_color_val(s: &str) -> PyResult<super::color::Color> {
    super::color::parse_color(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn apply_theme_overrides(t: &mut ThemeInputs, spec: ThemeOverridesSpec) -> PyResult<()> {
    if let Some(s) = spec.mark_color { t.colors.mark_color = parse_color_val(&s)?; }
    // `background` is an alias for `background_color`; both populate the
    // same field. Last-write-wins if a user passes both.
    if let Some(s) = spec.background_color { t.colors.background_color = parse_color_val(&s)?; }
    if let Some(s) = spec.background { t.colors.background_color = parse_color_val(&s)?; }
    if let Some(v) = spec.padding { t.padding.padding = v; }

    // Typography
    if let Some(v) = spec.font_family { t.typography.font_family = v; }
    if let Some(v) = spec.font_weight { t.typography.font_weight = v; }
    if let Some(s) = spec.font_color { t.colors.font_color = parse_color_val(&s)?; }
    if let Some(v) = spec.font_size { t.typography.label_font_size = v; }
    if let Some(v) = spec.title_font_family { t.typography.title_font_family = v; }
    if let Some(v) = spec.title_font_size { t.typography.title_font_size = v; }
    if let Some(v) = spec.title_font_weight { t.typography.title_font_weight = v; }
    if let Some(s) = spec.title_color { t.colors.title_color = parse_color_val(&s)?; }
    if let Some(s) = spec.title_anchor {
        t.typography.title_anchor = match s.as_str() {
            "start" => TextAnchor::Start,
            "middle" => TextAnchor::Middle,
            "end" => TextAnchor::End,
            other => return Err(PyValueError::new_err(format!(
                "title_anchor must be one of 'start'|'middle'|'end', got '{other}'"
            ))),
        };
    }
    if let Some(v) = spec.title_offset { t.typography.title_offset = v; }
    if let Some(v) = spec.label_font_family { t.typography.label_font_family = v; }
    if let Some(s) = spec.label_color { t.colors.label_color = parse_color_val(&s)?; }

    // Axes
    if let Some(v) = spec.axis_line { t.axis.axis_line = v; }
    if let Some(s) = spec.axis_line_color { t.colors.axis_line_color = parse_color_val(&s)?; }
    if let Some(v) = spec.axis_line_width { t.sizes.axis_line_width = v; }
    if let Some(s) = spec.tick_color { t.colors.tick_color = parse_color_val(&s)?; }
    if let Some(v) = spec.tick_size { t.sizes.tick_size = v; }
    if let Some(v) = spec.tick_width { t.sizes.tick_width = v; }

    // Grid
    if let Some(v) = spec.grid { t.grid.grid = v; }
    if let Some(s) = spec.grid_color { t.colors.grid_color = parse_color_val(&s)?; }
    if let Some(v) = spec.grid_width { t.sizes.grid_width = v; }
    if let Some(v) = spec.grid_dash { t.grid.grid_dash = Some(v); }
    if let Some(v) = spec.grid_opacity { t.grid.grid_opacity = v; }

    // Marks
    if let Some(v) = spec.point_size { t.sizes.point_size = v; }
    if let Some(v) = spec.point_opacity { t.sizes.point_opacity = v; }
    if let Some(v) = spec.line_stroke_width { t.sizes.line_stroke_width = v; }
    if let Some(v) = spec.bar_corner_radius { t.sizes.bar_corner_radius = v; }
    if let Some(v) = spec.area_opacity { t.sizes.area_opacity = v; }
    if let Some(v) = spec.opacity { t.sizes.default_opacity = v; }

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
        t.palette.color_scheme = s;
    }
    if let Some(s) = spec.sequential_scheme {
        if !super::palette::is_sequential_scheme(&s) {
            return Err(PyValueError::new_err(format!(
                "Unknown sequential_scheme: '{s}'. Supported: {}.",
                super::palette::SEQUENTIAL_SCHEMES.join(", "),
            )));
        }
        t.palette.sequential_scheme = s;
    }
    if let Some(s) = spec.diverging_scheme {
        if !super::palette::is_sequential_scheme(&s) {
            return Err(PyValueError::new_err(format!(
                "Unknown diverging_scheme: '{s}'. Supported: {}.",
                super::palette::SEQUENTIAL_SCHEMES.join(", "),
            )));
        }
        t.palette.diverging_scheme = s;
    }

    // Strip
    if let Some(s) = spec.strip_background_color { t.colors.strip_background_color = parse_color_val(&s)?; }
    if let Some(v) = spec.strip_text_size { t.sizes.strip_text_size = v; }
    if let Some(v) = spec.strip_padding { t.padding.strip_padding = v; }

    // Legend
    if let Some(s) = spec.legend_orient {
        t.legend.legend_orient = match s.as_str() {
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
        t.legend.legend_direction = Some(match s.as_str() {
            "horizontal" => LegendDirection::Horizontal,
            "vertical" => LegendDirection::Vertical,
            other => return Err(PyValueError::new_err(format!(
                "legend_direction must be one of 'horizontal'|'vertical', got '{other}'"
            ))),
        });
    }
    if let Some(v) = spec.legend_title_font_size { t.typography.legend_title_font_size = v; }

    // Reference lines
    if let Some(s) = spec.reference_line_color { t.colors.reference_line_color = parse_color_val(&s)?; }
    if let Some(v) = spec.reference_line_dash { t.reference_line.reference_line_dash = Some(v); }

    // Spacing
    if let Some(v) = spec.axis_title_padding { t.padding.axis_title_padding = v; }
    if let Some(v) = spec.column_padding { t.padding.column_padding = v; }
    if let Some(v) = spec.row_padding { t.padding.row_padding = v; }

    // Axis label culling
    if let Some(v) = spec.cull_threshold { t.cull_threshold = v; }

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
            assert_eq!(t.colors.background_color.red, 0xFF);
            assert_eq!(t.colors.background_color.green, 0x00);
            assert_eq!(t.colors.background_color.blue, 0x00);
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
                assert_eq!(t.palette.color_scheme, name);
            }
        });
    }

    #[test]
    fn unknown_sequential_scheme_raises() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("sequential_scheme", "nonexistent").unwrap();
            let err = theme_from_dict(Some(&d)).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("Unknown sequential_scheme"), "got: {msg}");
            assert!(msg.contains("nonexistent"), "got: {msg}");
        });
    }

    #[test]
    fn unknown_diverging_scheme_raises() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("diverging_scheme", "nonexistent").unwrap();
            let err = theme_from_dict(Some(&d)).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("Unknown diverging_scheme"), "got: {msg}");
            assert!(msg.contains("nonexistent"), "got: {msg}");
        });
    }

    #[test]
    fn known_sequential_scheme_accepted() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            for name in ["cool_blue", "viridis", "blues", "night_blue", "signal_blue"] {
                let d = PyDict::new(py);
                d.set_item("sequential_scheme", name).unwrap();
                let t = theme_from_dict(Some(&d)).expect(name);
                assert_eq!(t.palette.sequential_scheme, name);
            }
        });
    }

    #[test]
    fn known_diverging_scheme_accepted() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            for name in ["rdbu", "blue_to_red", "cyan_to_amber", "blue_to_violet"] {
                let d = PyDict::new(py);
                d.set_item("diverging_scheme", name).unwrap();
                let t = theme_from_dict(Some(&d)).expect(name);
                assert_eq!(t.palette.diverging_scheme, name);
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

fn chart_config_from_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<ChartConfig> {
    match dict {
        None => Ok(ChartConfig::default()),
        Some(d) => {
            let val = crate::pyo3_serde::from_py(d.as_any(), "chart_config")?;
            Ok(val)
        }
    }
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
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> combined = fm.compose_svg_horizontal([svg1, svg2], spacing=10)
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
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> combined = fm.compose_svg_vertical([svg1, svg2], spacing=10)
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
/// ``<g transform="translate(...)">`` (native fit) or
/// ``<svg viewBox preserveAspectRatio="none">`` (scaled fit) preserving
/// its internal coordinate system.
///
/// Axis sharing (the prior `share_x` / `share_y` parameters) belongs at
/// the Python layer pre-render: shared scales are computed before each
/// cell renders so the resulting SVGs already align. The compositor sees
/// opaque SVG strings and has no scale metadata to enforce sharing
/// against; the parameters were never functional. Use
/// `Chart.encode(x=fr.X(field, scale=...))` with a shared scale spec at
/// composition time, or `JointChart`/`ClusterMapChart`'s `axis(show=False)`
/// suppression for marginal/dendrogram cells.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> combined = fm.compose_svg_grid(
/// ...     [svg_a, svg_b, svg_c, svg_d], rows=2, cols=2,
/// ...     row_ratios=[1.0, 1.0], col_ratios=[1.0, 1.0], spacing=8,
/// ... )
#[pyfunction]
#[pyo3(name = "compose_svg_grid")]
#[pyo3(signature = (cells, *, rows, cols, row_ratios, col_ratios, spacing = 10.0))]
pub fn compose_svg_grid_py(
    cells: Vec<Option<String>>,
    rows: usize,
    cols: usize,
    row_ratios: Vec<f64>,
    col_ratios: Vec<f64>,
    spacing: f64,
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

/// Rasterize a complete SVG string to PNG bytes.
///
/// Used by composition types whose ``show_svg()`` produces a complete SVG
/// but that have no single ``ChartSpec`` + data to pass through
/// ``render_png``.
///
/// Parameters
/// ----------
/// svg : str
///     Complete SVG document string.
/// scale : float, default 2.0
///     Pixel-density multiplier applied to the SVG's intrinsic dimensions.
///     Default is 2.0 (retina).
///
/// Returns
/// -------
/// bytes
///     PNG image as raw bytes.
///
/// Raises
/// ------
/// ValueError
///     If the SVG string cannot be parsed or rasterization fails.
#[pyfunction]
#[pyo3(signature = (svg, *, scale = 2.0))]
pub fn rasterize_svg<'py>(
    py: Python<'py>,
    svg: &str,
    scale: f64,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = super::png::rasterize_svg_auto(svg, scale)
        .map_err(render_err_to_py)?;
    Ok(PyBytes::new(py, &bytes))
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
