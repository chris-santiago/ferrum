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
///     Sparse theme override dict. The authoritative set of accepted keys is
///     the ``ThemeOverridesSpec`` struct in this module, which carries
///     ``#[serde(deny_unknown_fields)]`` — that struct IS the contract.
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
    let (batch, t, c, cc, vp) = decode_render_inputs(data, viewport, theme, config, chart_config)?;
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
///     Sparse theme override dict. The authoritative set of accepted keys is
///     the ``ThemeOverridesSpec`` struct in this module, which carries
///     ``#[serde(deny_unknown_fields)]`` — that struct IS the contract.
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
    let (batch, t, c, cc, vp) = decode_render_inputs(data, viewport, theme, config, chart_config)?;
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
    let (batch, t, c, cc, vp) = decode_render_inputs(data, viewport, theme, config, chart_config)?;
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

/// Authoritative list of accepted theme keys — the single source of truth
/// for the Python theme-key contract (D-THEME-1). This MUST match the
/// `ThemeOverridesSpec` field/alias set below exactly (including the
/// `background` alias and the per-level grid keys). The
/// `manifest_matches_serde` test asserts this by round-tripping every key
/// through `ThemeOverridesSpec` serde and rejecting a bogus key, so the
/// const cannot silently drift from the struct.
///
/// Exposed to Python via the `theme_known_keys()` PyO3 accessor; Python
/// derives `Theme._KNOWN_KEYS` from it rather than re-listing the keys.
const THEME_KNOWN_KEYS: &[&str] = &[
    // Background / padding
    "mark_color",
    "background_color",
    "background",
    "padding",
    // Typography
    "font_family",
    "font_weight",
    "font_color",
    "font_size",
    "title_font_family",
    "title_font_size",
    "title_font_weight",
    "title_color",
    "title_anchor",
    "title_offset",
    "label_font_family",
    "label_color",
    // Axes
    "axis_line",
    "axis_line_color",
    "axis_line_width",
    "tick_color",
    "tick_size",
    "tick_width",
    // Grid — legacy single-level keys (major level, back-compat)
    "grid",
    "grid_color",
    "grid_width",
    "grid_dash",
    "grid_opacity",
    // Grid — per-level keys (Grid item 18)
    "minor",
    "major_grid_color",
    "minor_grid_color",
    "major_grid_width",
    "minor_grid_width",
    "major_grid_dash",
    "minor_grid_dash",
    "major_grid_opacity",
    "minor_grid_opacity",
    // Marks
    "point_size",
    "point_opacity",
    "line_stroke_width",
    "bar_corner_radius",
    "area_opacity",
    "opacity",
    // Palette
    "color_scheme",
    "sequential_scheme",
    "diverging_scheme",
    // Strip
    "strip_background_color",
    "strip_text_size",
    "strip_padding",
    // Legend
    "legend_orient",
    "legend_direction",
    "legend_title_font_size",
    // Reference lines
    "reference_line_color",
    "reference_line_dash",
    // Spacing
    "axis_title_padding",
    "column_padding",
    "row_padding",
    // Axis label culling
    "cull_threshold",
];

/// The color-typed subset of [`THEME_KNOWN_KEYS`] — keys whose values are
/// color strings (parsed by `parse_color_val`/`from_hex_str` on the Rust
/// side). Exposed via `theme_color_keys()` so Python can decide whether to
/// keep a color-key list (e.g. for CSS-shorthand expansion) without
/// hand-maintaining a third parallel copy. Every entry MUST also appear in
/// `THEME_KNOWN_KEYS` (asserted by `color_keys_subset_of_known`).
const THEME_COLOR_KEYS: &[&str] = &[
    "mark_color",
    "background_color",
    "background",
    "font_color",
    "title_color",
    "label_color",
    "axis_line_color",
    "tick_color",
    "grid_color",
    "major_grid_color",
    "minor_grid_color",
    "strip_background_color",
    "reference_line_color",
];

/// The complete set of valid Python theme keys (the contract published by
/// `ThemeOverridesSpec`). Python derives `Theme._KNOWN_KEYS` from this.
#[pyfunction]
pub fn theme_known_keys() -> Vec<String> {
    THEME_KNOWN_KEYS.iter().map(|s| s.to_string()).collect()
}

/// The color-typed subset of [`theme_known_keys`] — keys whose values are
/// color strings.
#[pyfunction]
pub fn theme_color_keys() -> Vec<String> {
    THEME_COLOR_KEYS.iter().map(|s| s.to_string()).collect()
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
/// The accepted field/alias set is mirrored in the [`THEME_KNOWN_KEYS`]
/// const (the single source of truth published to Python); the
/// `manifest_matches_serde` test guards them against drift.
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

    // Grid — legacy single-level keys style the MAJOR level (back-compat).
    grid: Option<bool>,
    grid_color: Option<String>,
    grid_width: Option<f64>,
    grid_dash: Option<Vec<f64>>,
    grid_opacity: Option<f64>,
    // Grid — per-level keys (Grid item 18). `major_*` overrides the legacy
    // `grid_*` value for the major level; `minor_*` styles the minor level.
    // `minor` enables minor gridline emission (default off).
    minor: Option<bool>,
    major_grid_color: Option<String>,
    minor_grid_color: Option<String>,
    major_grid_width: Option<f64>,
    minor_grid_width: Option<f64>,
    major_grid_dash: Option<Vec<f64>>,
    minor_grid_dash: Option<Vec<f64>>,
    major_grid_opacity: Option<f64>,
    minor_grid_opacity: Option<f64>,

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

    // Grid — legacy single-level keys feed the MAJOR level (back-compat).
    if let Some(v) = spec.grid { t.grid.grid = v; }
    if let Some(s) = spec.grid_color { t.colors.grid_color = parse_color_val(&s)?; }
    if let Some(v) = spec.grid_width { t.sizes.grid_width = v; }
    if let Some(v) = spec.grid_dash { t.grid.grid_dash = Some(v); }
    if let Some(v) = spec.grid_opacity { t.grid.grid_opacity = v; }

    // Grid — per-level keys (Grid item 18). Applied AFTER the legacy keys so an
    // explicit `major_*` value wins over the corresponding `grid_*` fallback.
    if let Some(v) = spec.minor { t.grid.minor = v; }
    if let Some(s) = spec.major_grid_color { t.colors.grid_color = parse_color_val(&s)?; }
    if let Some(s) = spec.minor_grid_color { t.colors.minor_grid_color = parse_color_val(&s)?; }
    if let Some(v) = spec.major_grid_width { t.sizes.grid_width = v; }
    if let Some(v) = spec.minor_grid_width { t.sizes.minor_grid_width = v; }
    if let Some(v) = spec.major_grid_dash { t.grid.grid_dash = Some(v); }
    if let Some(v) = spec.minor_grid_dash { t.grid.minor_grid_dash = Some(v); }
    if let Some(v) = spec.major_grid_opacity { t.grid.grid_opacity = v; }
    if let Some(v) = spec.minor_grid_opacity { t.grid.minor_grid_opacity = v; }

    // Marks
    if let Some(v) = spec.point_size { t.sizes.point_size = v; }
    if let Some(v) = spec.point_opacity { t.sizes.point_opacity = v; }
    if let Some(v) = spec.line_stroke_width { t.sizes.line_stroke_width = v; }
    if let Some(v) = spec.bar_corner_radius { t.sizes.bar_corner_radius = v; }
    if let Some(v) = spec.area_opacity { t.sizes.area_opacity = v; }
    if let Some(v) = spec.opacity { t.sizes.default_opacity = v; }

    // Palette
    if let Some(s) = spec.color_scheme {
        if !super::color::palette::is_categorical_scheme(&s)
            && !super::color::palette::is_sequential_scheme(&s)
        {
            return Err(PyValueError::new_err(format!(
                "Unknown color_scheme: '{s}'. Supported categorical: {}. \
                 Supported sequential: {}.",
                super::color::palette::CATEGORICAL_SCHEMES.join(", "),
                super::color::palette::SEQUENTIAL_SCHEMES.join(", "),
            )));
        }
        t.palette.color_scheme = s;
    }
    if let Some(s) = spec.sequential_scheme {
        if !super::color::palette::is_sequential_scheme(&s) {
            return Err(PyValueError::new_err(format!(
                "Unknown sequential_scheme: '{s}'. Supported: {}.",
                super::color::palette::SEQUENTIAL_SCHEMES.join(", "),
            )));
        }
        t.palette.sequential_scheme = s;
    }
    if let Some(s) = spec.diverging_scheme {
        if !super::color::palette::is_sequential_scheme(&s) {
            return Err(PyValueError::new_err(format!(
                "Unknown diverging_scheme: '{s}'. Supported: {}.",
                super::color::palette::SEQUENTIAL_SCHEMES.join(", "),
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
    fn per_level_grid_keys_round_trip() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("minor", true).unwrap();
            d.set_item("major_grid_color", "#111111").unwrap();
            d.set_item("minor_grid_color", "#eeeeee").unwrap();
            d.set_item("major_grid_width", 2.0).unwrap();
            d.set_item("minor_grid_width", 0.25).unwrap();
            d.set_item("major_grid_dash", vec![1.0, 2.0]).unwrap();
            d.set_item("minor_grid_dash", vec![3.0, 4.0]).unwrap();
            d.set_item("major_grid_opacity", 0.9).unwrap();
            d.set_item("minor_grid_opacity", 0.4).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert!(t.grid.minor);
            assert_eq!(t.colors.grid_color.red, 0x11);
            assert_eq!(t.colors.minor_grid_color.red, 0xEE);
            assert_eq!(t.sizes.grid_width, 2.0);
            assert_eq!(t.sizes.minor_grid_width, 0.25);
            assert_eq!(t.grid.grid_dash, Some(vec![1.0, 2.0]));
            assert_eq!(t.grid.minor_grid_dash, Some(vec![3.0, 4.0]));
            assert_eq!(t.grid.grid_opacity, 0.9);
            assert_eq!(t.grid.minor_grid_opacity, 0.4);
        });
    }

    #[test]
    fn major_grid_keys_fall_back_to_legacy_grid_keys() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // Only the legacy keys are present → they style the major level.
            let d = PyDict::new(py);
            d.set_item("grid_color", "#abcdef").unwrap();
            d.set_item("grid_width", 1.25).unwrap();
            d.set_item("grid_opacity", 0.75).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.colors.grid_color.red, 0xAB);
            assert_eq!(t.sizes.grid_width, 1.25);
            assert_eq!(t.grid.grid_opacity, 0.75);
            // Minor level untouched (still the derived default).
            assert_eq!(t.colors.minor_grid_color, ThemeInputs::default().colors.minor_grid_color);
        });
    }

    #[test]
    fn major_grid_key_wins_over_legacy_when_both_present() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("grid_color", "#abcdef").unwrap();
            d.set_item("major_grid_color", "#123456").unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            // major_* applied after legacy → wins.
            assert_eq!(t.colors.grid_color.red, 0x12);
            assert_eq!(t.colors.grid_color.green, 0x34);
            assert_eq!(t.colors.grid_color.blue, 0x56);
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

    /// A serde-compatible sample value for a theme key, by name. Keeps the
    /// round-trip test's per-key typing in one place; the bool/numeric/list
    /// key sets mirror the `ThemeOverridesSpec` field types exactly, so a
    /// miscategorized key (or a struct field that changes type) fails the
    /// `manifest_matches_serde` round-trip loudly. Every other key
    /// deserializes as `Option<String>` (color, font family/weight, anchor,
    /// scheme name); serde only checks the field exists here — value
    /// validation happens later in `apply_theme_overrides`.
    fn sample_for(py: Python<'_>, key: &str) -> Py<PyAny> {
        const DASH_KEYS: &[&str] = &[
            "grid_dash",
            "major_grid_dash",
            "minor_grid_dash",
            "reference_line_dash",
        ];
        const BOOL_KEYS: &[&str] = &["axis_line", "grid", "minor"];
        const F64_KEYS: &[&str] = &[
            "padding",
            "font_size",
            "title_font_size",
            "title_offset",
            "axis_line_width",
            "tick_size",
            "tick_width",
            "grid_width",
            "grid_opacity",
            "major_grid_width",
            "minor_grid_width",
            "major_grid_opacity",
            "minor_grid_opacity",
            "point_size",
            "point_opacity",
            "line_stroke_width",
            "bar_corner_radius",
            "area_opacity",
            "opacity",
            "strip_text_size",
            "strip_padding",
            "legend_title_font_size",
            "axis_title_padding",
            "column_padding",
            "row_padding",
        ];
        if DASH_KEYS.contains(&key) {
            vec![2.0_f64, 1.0].into_pyobject(py).unwrap().into_any().unbind()
        } else if BOOL_KEYS.contains(&key) {
            true.into_pyobject(py).unwrap().to_owned().into_any().unbind()
        } else if key == "cull_threshold" {
            7_u32.into_pyobject(py).unwrap().into_any().unbind()
        } else if F64_KEYS.contains(&key) {
            1.5_f64.into_pyobject(py).unwrap().into_any().unbind()
        } else {
            "x".into_pyobject(py).unwrap().into_any().unbind()
        }
    }

    /// The const key manifest must not drift from `ThemeOverridesSpec`:
    /// every `THEME_KNOWN_KEYS` entry deserializes into the struct, and a
    /// bogus key is rejected by `deny_unknown_fields`. This is the guard
    /// that keeps the Python-facing contract honest.
    #[test]
    fn manifest_matches_serde() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // Every advertised key must deserialize successfully (individually,
            // so a type mismatch points at the offending key).
            for &key in THEME_KNOWN_KEYS {
                let d = PyDict::new(py);
                d.set_item(key, sample_for(py, key).bind(py)).unwrap();
                let res: PyResult<ThemeOverridesSpec> =
                    crate::pyo3_serde::from_py(d.as_any(), "theme");
                assert!(res.is_ok(), "known key `{key}` failed to deserialize: {res:?}");
            }

            // All keys together also deserialize (no field-name collisions).
            let all = PyDict::new(py);
            for &key in THEME_KNOWN_KEYS {
                all.set_item(key, sample_for(py, key).bind(py)).unwrap();
            }
            let res: PyResult<ThemeOverridesSpec> =
                crate::pyo3_serde::from_py(all.as_any(), "theme");
            assert!(res.is_ok(), "full manifest failed to deserialize: {res:?}");

            // A bogus key is rejected by `deny_unknown_fields`.
            let bogus = PyDict::new(py);
            bogus.set_item("definitely_not_a_theme_key", "v").unwrap();
            let res: PyResult<ThemeOverridesSpec> =
                crate::pyo3_serde::from_py(bogus.as_any(), "theme");
            assert!(res.is_err(), "bogus key should be rejected by deny_unknown_fields");
        });
    }

    /// Parse the accepted-field set out of a serde `deny_unknown_fields`
    /// error. serde_json renders the message as:
    ///
    /// ```text
    /// unknown field `x`, expected one of `a`, `b`, … `z` at line 1 column 29
    /// ```
    ///
    /// We slice after the `expected one of ` marker, drop the trailing
    /// ` at line …` position suffix serde_json appends, then collect every
    /// backtick-delimited token. Assumption: there is more than one accepted
    /// field, so serde uses the `expected one of` wording rather than the
    /// single-field `expected \`only\`` form — `ThemeOverridesSpec` has dozens
    /// of fields, and `unknown_key_raises` already proves the "one of" form is
    /// produced, so this holds for as long as the struct stays non-trivial.
    fn accepted_fields_from_error(msg: &str) -> std::collections::BTreeSet<String> {
        const MARKER: &str = "expected one of ";
        let after = msg
            .find(MARKER)
            .map(|i| &msg[i + MARKER.len()..])
            .unwrap_or_else(|| panic!("error missing `{MARKER}` marker: {msg}"));
        // Strip serde_json's trailing " at line N column M" position suffix so
        // it cannot be mistaken for a field name; backtick extraction below
        // ignores it regardless, but trimming keeps the parse intent obvious.
        let list = after.split(" at line ").next().unwrap_or(after);
        // Every accepted field is wrapped in a matched pair of backticks.
        let mut fields = std::collections::BTreeSet::new();
        let mut rest = list;
        while let Some(open) = rest.find('`') {
            let tail = &rest[open + 1..];
            let close = tail
                .find('`')
                .unwrap_or_else(|| panic!("unbalanced backticks in: {msg}"));
            fields.insert(tail[..close].to_string());
            rest = &tail[close + 1..];
        }
        fields
    }

    /// The reverse drift guard for D-THEME-1: a new `Option<_>` field added to
    /// `ThemeOverridesSpec` *without* a matching `THEME_KNOWN_KEYS` entry would
    /// deserialize fine (every field is optional) yet never be published to
    /// Python via `theme_known_keys()`. `manifest_matches_serde` only checks
    /// const→struct; this checks struct→const by mining the *complete* accepted
    /// field set out of serde's `deny_unknown_fields` error and asserting it
    /// equals `THEME_KNOWN_KEYS`. A field added with no const entry (or a const
    /// entry removed) makes the two sets differ and fails here.
    #[test]
    fn serde_accepted_fields_match_manifest() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let bogus = PyDict::new(py);
            bogus.set_item("definitely_not_a_theme_key", "v").unwrap();
            let res: PyResult<ThemeOverridesSpec> =
                crate::pyo3_serde::from_py(bogus.as_any(), "theme");
            let err = res.expect_err("bogus key must be rejected by deny_unknown_fields");
            let msg = err.value(py).to_string();

            let serde_fields = accepted_fields_from_error(&msg);
            let manifest: std::collections::BTreeSet<String> =
                THEME_KNOWN_KEYS.iter().map(|s| s.to_string()).collect();

            assert_eq!(
                serde_fields, manifest,
                "ThemeOverridesSpec accepted fields drifted from THEME_KNOWN_KEYS.\n\
                 in serde but not const (struct field missing a manifest entry): {:?}\n\
                 in const but not serde (manifest entry with no struct field): {:?}",
                serde_fields.difference(&manifest).collect::<Vec<_>>(),
                manifest.difference(&serde_fields).collect::<Vec<_>>(),
            );
        });
    }

    /// `theme_color_keys()` must be a subset of `theme_known_keys()`.
    #[test]
    fn color_keys_subset_of_known() {
        for &k in THEME_COLOR_KEYS {
            assert!(
                THEME_KNOWN_KEYS.contains(&k),
                "color key `{k}` is not in THEME_KNOWN_KEYS"
            );
        }
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

/// Shared decode preamble for the three PyO3 render entry points
/// (`render_svg`, `render_png`, `render_interactive`).
///
/// Collects the Arrow record batch stream into a single batch, decodes the
/// theme/config/chart-config dicts, and builds the [`Viewport`] from the
/// `(width, height)` tuple. Any change to the render input contract (a new
/// kwarg, a new batch-collection policy) needs to be made here only.
fn decode_render_inputs(
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(
    arrow::record_batch::RecordBatch,
    crate::layout::ThemeInputs,
    super::config::RenderConfig,
    super::chart_config::ChartConfig,
    Viewport,
)> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let cc = chart_config_from_dict(chart_config)?;
    let vp = Viewport {
        width: viewport.0,
        height: viewport.1,
    };
    Ok((batch, t, c, cc, vp))
}

// ---------------------------------------------------------------------------
// SVG compositor bindings (Task 11)
// ---------------------------------------------------------------------------

use crate::render::figure_chrome::{ChromeAnchor, FigureChrome, DEFAULT_CHROME_INSET};

/// Parse the user-facing chrome-anchor string into a [`ChromeAnchor`].
///
/// `None` and `"start"` both map to the default left alignment. Any other
/// string is rejected with a `ValueError` naming the valid set.
fn parse_chrome_anchor(anchor: Option<&str>) -> PyResult<ChromeAnchor> {
    match anchor {
        None | Some("start") => Ok(ChromeAnchor::Start),
        Some("middle") => Ok(ChromeAnchor::Middle),
        Some("end") => Ok(ChromeAnchor::End),
        Some(other) => Err(PyValueError::new_err(format!(
            "anchor must be one of 'start'|'middle'|'end', got '{other}'"
        ))),
    }
}

/// Apply figure chrome to an already-composed SVG string.
///
/// This is the shared tail of all three `compose_svg_*_py` bindings:
/// build the [`FigureChrome`] from the chrome kwargs and call
/// [`wrap_with_chrome`](crate::render::figure_chrome::wrap_with_chrome).
///
/// `composed` is the raw composed SVG (no chrome yet) produced by the
/// compositor. `title`/`subtitle`/`caption`/`left_inset`/`right_inset`/
/// `anchor` are the same chrome kwargs accepted by every compose binding.
///
/// Errors from [`wrap_with_chrome`] are mapped to `PyValueError` here so
/// callers don't need to repeat the `map_err` boilerplate.
#[allow(clippy::too_many_arguments)]
fn compose_and_wrap(
    composed: String,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<String> {
    let chrome = build_chrome(title, subtitle, caption, left_inset, right_inset, anchor)?;
    crate::render::figure_chrome::wrap_with_chrome(&composed, chrome)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Build a [`FigureChrome`] from the chrome-related keyword params, applying the
/// default inset when not supplied and parsing the anchor string.
fn build_chrome<'a>(
    title: Option<&'a str>,
    subtitle: Option<&'a str>,
    caption: Option<&'a str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<FigureChrome<'a>> {
    Ok(FigureChrome {
        title,
        subtitle,
        caption,
        left_inset: left_inset.unwrap_or(DEFAULT_CHROME_INSET),
        right_inset: right_inset.unwrap_or(DEFAULT_CHROME_INSET),
        anchor: parse_chrome_anchor(anchor)?,
    })
}

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
/// title : str or None, default None
///     Figure-level title rendered as a band above all panels (bold, 16 px).
///     Per-panel child titles are unaffected.
/// subtitle : str or None, default None
///     Figure-level subtitle rendered below the title, above the panels (13 px).
/// caption : str or None, default None
///     Figure-level caption rendered as a band below all panels (muted, 11 px).
/// left_inset : float or None, default None
///     Horizontal inset (px) from the left panel edge for ``"start"``-anchored
///     chrome. Defaults to 16.0 (matching single-chart title inset).
/// right_inset : float or None, default None
///     Horizontal inset (px) from the right panel edge for ``"end"``-anchored
///     chrome. Defaults to 16.0.
/// anchor : str or None, default None
///     Horizontal alignment of all chrome lines. One of ``"start"``,
///     ``"middle"``, or ``"end"``. ``None`` is treated as ``"start"``.
///
/// Returns
/// -------
/// str
///     A single SVG document whose width equals the sum of panel widths
///     plus total spacing and whose height equals the tallest panel (plus
///     header/footer bands when title/subtitle/caption are provided).
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
/// When all of *title*, *subtitle*, and *caption* are ``None`` the output is
/// byte-identical to the previous behavior.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> combined = fm.compose_svg_horizontal([svg1, svg2], spacing=10)
#[pyfunction]
#[pyo3(name = "compose_svg_horizontal")]
#[pyo3(signature = (svgs, *, spacing = 10.0, align = "top", title = None, subtitle = None, caption = None, left_inset = None, right_inset = None, anchor = None))]
#[allow(clippy::too_many_arguments)]
pub fn compose_svg_horizontal_py(
    svgs: Vec<String>,
    spacing: f64,
    align: &str,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<String> {
    let align_val = crate::render::compositor::VerticalAlign::try_from(align)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let composed = crate::render::compositor::compose_svg_horizontal(&svgs, spacing, align_val)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    compose_and_wrap(composed, title, subtitle, caption, left_inset, right_inset, anchor)
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
/// title : str or None, default None
///     Figure-level title rendered as a band above all panels (bold, 16 px).
///     Per-panel child titles are unaffected.
/// subtitle : str or None, default None
///     Figure-level subtitle rendered below the title, above the panels (13 px).
/// caption : str or None, default None
///     Figure-level caption rendered as a band below all panels (muted, 11 px).
/// left_inset : float or None, default None
///     Horizontal inset (px) from the left panel edge for ``"start"``-anchored
///     chrome. Defaults to 16.0 (matching single-chart title inset).
/// right_inset : float or None, default None
///     Horizontal inset (px) from the right panel edge for ``"end"``-anchored
///     chrome. Defaults to 16.0.
/// anchor : str or None, default None
///     Horizontal alignment of all chrome lines. One of ``"start"``,
///     ``"middle"``, or ``"end"``. ``None`` is treated as ``"start"``.
///
/// Returns
/// -------
/// str
///     A single SVG document whose height equals the sum of panel heights
///     plus total spacing and whose width equals the widest panel (plus
///     header/footer bands when title/subtitle/caption are provided).
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
/// When all of *title*, *subtitle*, and *caption* are ``None`` the output is
/// byte-identical to the previous behavior.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> combined = fm.compose_svg_vertical([svg1, svg2], spacing=10)
#[pyfunction]
#[pyo3(name = "compose_svg_vertical")]
#[pyo3(signature = (svgs, *, spacing = 10.0, align = "left", title = None, subtitle = None, caption = None, left_inset = None, right_inset = None, anchor = None))]
#[allow(clippy::too_many_arguments)]
pub fn compose_svg_vertical_py(
    svgs: Vec<String>,
    spacing: f64,
    align: &str,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<String> {
    let align_val = crate::render::compositor::HorizontalAlign::try_from(align)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let composed = crate::render::compositor::compose_svg_vertical(&svgs, spacing, align_val)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    compose_and_wrap(composed, title, subtitle, caption, left_inset, right_inset, anchor)
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
/// title : str or None, default None
///     Figure-level title rendered as a band above all panels (bold, 16 px).
///     Per-panel child titles are unaffected.
/// subtitle : str or None, default None
///     Figure-level subtitle rendered below the title, above the panels (13 px).
/// caption : str or None, default None
///     Figure-level caption rendered as a band below all panels (muted, 11 px).
/// left_inset : float or None, default None
///     Horizontal inset (px) from the left panel edge for ``"start"``-anchored
///     chrome. Defaults to 16.0 (matching single-chart title inset).
/// right_inset : float or None, default None
///     Horizontal inset (px) from the right panel edge for ``"end"``-anchored
///     chrome. Defaults to 16.0.
/// anchor : str or None, default None
///     Horizontal alignment of all chrome lines. One of ``"start"``,
///     ``"middle"``, or ``"end"``. ``None`` is treated as ``"start"``.
///
/// Returns
/// -------
/// str
///     A single SVG document containing all cells positioned according to
///     the ratio-weighted grid layout (plus header/footer bands when
///     title/subtitle/caption are provided).
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
/// When all of *title*, *subtitle*, and *caption* are ``None`` the output is
/// byte-identical to the previous behavior.
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
#[pyo3(signature = (cells, *, rows, cols, row_ratios, col_ratios, spacing = 10.0, title = None, subtitle = None, caption = None, left_inset = None, right_inset = None, anchor = None))]
#[allow(clippy::too_many_arguments)]
pub fn compose_svg_grid_py(
    cells: Vec<Option<String>>,
    rows: usize,
    cols: usize,
    row_ratios: Vec<f64>,
    col_ratios: Vec<f64>,
    spacing: f64,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<String> {
    let composed = crate::render::grid_compose::compose_svg_grid(
        &cells,
        rows,
        cols,
        &row_ratios,
        &col_ratios,
        spacing,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    compose_and_wrap(composed, title, subtitle, caption, left_inset, right_inset, anchor)
}

/// Lay out a figure-level chrome title/subtitle/caption band as scene nodes.
///
/// Produces the interactive (WASM) counterpart of the SVG band emitted by
/// ``compose_svg_horizontal``/``_vertical``/``_grid``. The Python composite
/// merge injects the returned nodes into the merged scene's ``title`` list and
/// offsets the merged child panels down by ``header_h`` — mirroring how the SVG
/// path reserves top space — so the static and interactive renders place the
/// title band at identical coordinates. Layout lives entirely in Rust; Python
/// must not recompute any title geometry.
///
/// Parameters
/// ----------
/// width : float
///     Width (px) of the composed figure (the merged panels' bounding box).
///     Drives horizontal placement: ``start`` -> ``left_inset``,
///     ``middle`` -> ``width / 2``, ``end`` -> ``width - right_inset``.
/// height : float
///     Height (px) of the composed panels (excluding chrome). Used to place the
///     caption baseline below the panels. Has no effect on ``header_h``.
/// title : str or None, default None
///     Figure-level title (bold, 16 px, ``#1f2937``).
/// subtitle : str or None, default None
///     Figure-level subtitle (normal, 13 px, ``#6b7280``).
/// caption : str or None, default None
///     Figure-level caption rendered below the panels (normal, 11 px, ``#6b7280``).
/// left_inset : float or None, default None
///     Horizontal inset (px) for ``start``-anchored chrome. Defaults to 16.0.
/// right_inset : float or None, default None
///     Horizontal inset (px) for ``end``-anchored chrome. Defaults to 16.0.
/// anchor : str or None, default None
///     One of ``"start"``, ``"middle"``, ``"end"``. ``None`` -> ``"start"``.
///
/// Returns
/// -------
/// tuple[str, float, float]
///     ``(nodes_json, header_h, footer_h)``.
///
///     - ``nodes_json`` — a JSON array string of ``SceneNode::Text`` objects
///       (same shape as a scene's ``title`` list:
///       ``[{"type": "text", "x": .., "y": .., "content": "..", "style": {..}}, ..]``);
///       parse with ``json.loads`` and extend ``merged["title"]``.
///     - ``header_h`` — vertical band height (px) reserved **above** the panels
///       (title + subtitle). Offset the merged child scenes downward by this amount
///       before extending ``merged["title"]`` with the returned nodes. ``0.0`` when
///       no title/subtitle is present (e.g. caption-only or all-``None``).
///     - ``footer_h`` — vertical band height (px) reserved **below** the panels
///       (caption). Grow the merged scene's ``height`` by ``footer_h`` so the
///       interactive canvas matches the SVG canvas that ``compose_svg_*`` produces.
///       ``0.0`` when no caption is present.
///
///     The caption node's ``y`` is already absolute in outer-canvas space
///     (``header_h + panel_h + CAPTION_TOP_PAD + CAPTION_FONT_SIZE``), so it must
///     NOT be offset when ``_offset_node`` is applied to child panels. Only child
///     scene nodes (marks, axes, decorations) are shifted by ``header_h``; the
///     returned chrome nodes are injected at their final positions.
///
///     The ``height`` parameter must be the merged panels' bounding-box height
///     **before** chrome is added (i.e. the same ``height`` the SVG compositor uses
///     as ``panel_h`` inside ``wrap_with_chrome``). For horizontal/vertical/grid
///     composites this is the inner SVG root's ``height`` after compositing but
///     before ``wrap_with_chrome`` adds the bands.
///
/// Raises
/// ------
/// ValueError
///     If *anchor* is not one of the accepted values, or node serialization fails.
#[pyfunction]
#[pyo3(name = "figure_title_nodes")]
#[pyo3(signature = (width, height, *, title = None, subtitle = None, caption = None, left_inset = None, right_inset = None, anchor = None))]
#[allow(clippy::too_many_arguments)]
pub fn figure_title_nodes_py(
    width: f64,
    height: f64,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<(String, f64, f64)> {
    let chrome = build_chrome(title, subtitle, caption, left_inset, right_inset, anchor)?;
    let (nodes, header_h, footer_h) = crate::render::figure_chrome::title_nodes(chrome, width, height);
    let json = serde_json::to_string(&nodes)
        .map_err(|e| PyValueError::new_err(format!("figure title node serialization: {e}")))?;
    Ok((json, header_h, footer_h))
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
        // Forward the intentional Display contract (a stable user-facing
        // sentence), not the derived Debug of the enum's internal fields
        // (SEAM-07). See `RenderWarning`'s Display impl in `render/mod.rs`.
        let msg = format!("{w}");
        warnings_mod.call_method1("warn", (msg,))?;
    }
    Ok(())
}
