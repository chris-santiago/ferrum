//! PyO3 bindings: render_svg, render_png. Theme/RenderConfig pass via Python dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3_arrow::PyRecordBatchReader;

use crate::layout::{LegendDirection, LegendOrient, TextAnchor, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;
use crate::spec::composite::CompositeNode;

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
    let (json, packed_bytes, warnings) = super::render_scene_json(spec, &batch, &t, vp, &c, &cc)
        .map_err(render_err_to_py)?;
    emit_warnings(py, &warnings)?;
    let py_bytes = PyBytes::new(py, &packed_bytes);
    Ok((json, py_bytes.unbind()))
}

// ---------------------------------------------------------------------------
// Composite render bindings (Task 5c)
// ---------------------------------------------------------------------------

/// Render a composite chart tree (`hconcat`/`vconcat`/`grid`/`wrap`/`overlay`)
/// to a single SVG string.
///
/// Parameters
/// ----------
/// tree : dict
///     A composite tree node: ``{"kind": "leaf", "spec": ChartSpec, "data": int}``
///     or ``{"kind": "composite", "layout": ..., "children": [...], ...}``. See
///     ``crate::spec::composite`` for the full wire shape.
/// payloads : list[pyarrow.RecordBatchReader or compatible]
///     Arrow data streams, one per leaf's ``data`` index. A leaf's ``data``
///     field selects its batch from this list by position.
/// viewport : tuple[float, float]
///     ``(width, height)`` passed to every leaf's standalone render (the same
///     contract as ``render_svg``'s ``viewport``).
/// theme : dict, optional
///     Sparse theme override dict, applied to every leaf. See ``render_svg``.
/// config : dict, optional
///     Render-config dict, applied to every leaf. See ``render_svg``.
/// chart_config : dict, optional
///     Per-chart configuration dict, applied to every leaf.
///
/// Returns
/// -------
/// str
///     Complete SVG document as a UTF-8 string.
///
/// Raises
/// ------
/// ValueError
///     If the tree is malformed (unknown node kind, missing keys, root-only
///     chrome on a non-root node), a leaf's ``data`` index is out of bounds
///     for *payloads*, a payload stream is empty or unreadable, or rendering
///     any leaf fails.
///
/// Notes
/// -----
/// Byte-deterministic given the same *tree*, *payloads*, and *theme* inputs
/// (same contract as ``render_svg``). Render warnings are forwarded to
/// Python's ``warnings.warn``.
#[pyfunction]
#[pyo3(signature = (tree, payloads, *, viewport, theme = None, config = None, chart_config = None))]
pub fn render_composite_svg(
    py: Python<'_>,
    tree: &Bound<'_, PyAny>,
    payloads: Vec<PyRecordBatchReader>,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let node = crate::spec::composite::composite_tree_from_py(tree)?;
    let (batches, t, c, cc, vp) =
        decode_composite_inputs(payloads, viewport, theme, config, chart_config)?;
    let bindings = collect_leaf_bindings(tree, &node, &t, vp, &c, &cc)?;
    let leaves = build_composite_leaves(&node, &batches, &bindings)?;
    let (scene, warnings) = super::composite_render::render_composite_scene(&node, &leaves, &t)
        .map_err(composite_render_err_to_py)?;
    emit_warnings(py, &warnings)?;
    super::svg_walk::walk_svg(&scene, c.embed_fonts).map_err(render_err_to_py)
}

/// Render a composite chart tree to an interactive scene (JSON + packed bytes).
///
/// Same tree/payload/viewport/theme/config contract as
/// [`render_composite_svg`]; produces the WASM-consumable pair instead of an
/// SVG string, mirroring ``render_interactive``.
///
/// Parameters
/// ----------
/// tree : dict
///     A composite tree node (see [`render_composite_svg`]).
/// payloads : list[pyarrow.RecordBatchReader or compatible]
///     Arrow data streams, one per leaf's ``data`` index.
/// viewport : tuple[float, float]
///     ``(width, height)`` passed to every leaf's standalone render.
/// theme : dict, optional
///     Sparse theme override dict, applied to every leaf.
/// config : dict, optional
///     Render-config dict, applied to every leaf.
/// chart_config : dict, optional
///     Per-chart configuration dict, applied to every leaf.
///
/// Returns
/// -------
/// tuple[str, bytes]
///     ``(scene_json, packed_bytes)`` — the merged scene's JSON plus the
///     packed binary sidecar for large homogeneous mark batches. Panel ids in
///     the JSON and `panel_idx` values in the packed headers share one flat
///     0-based namespace numbered in tree pre-order (D4c): the packed headers
///     are written directly against each leaf's already-renumbered panels, so
///     no post-hoc header rewrite is needed.
///
/// Raises
/// ------
/// ValueError
///     Same conditions as [`render_composite_svg`].
#[pyfunction]
#[pyo3(signature = (tree, payloads, *, viewport, theme = None, config = None, chart_config = None))]
pub fn render_composite_interactive<'py>(
    py: Python<'py>,
    tree: &Bound<'_, PyAny>,
    payloads: Vec<PyRecordBatchReader>,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(String, Py<PyBytes>)> {
    let node = crate::spec::composite::composite_tree_from_py(tree)?;
    let (batches, t, c, cc, vp) =
        decode_composite_inputs(payloads, viewport, theme, config, chart_config)?;
    let bindings = collect_leaf_bindings(tree, &node, &t, vp, &c, &cc)?;
    let leaves = build_composite_leaves(&node, &batches, &bindings)?;
    // GH #50: warning emission used to be deferred here because the flat
    // sibling `render_interactive` had the identical gap (`render_scene_json`
    // didn't surface `RenderWarning`s at all — fixed above, in this same
    // burndown, by adding `warnings` to its return). Both interactive entries
    // now emit through the same `py`/GIL context `render_composite_svg` and
    // `render_svg`/`render_png` already use — no design blocker, just a
    // one-sided fix that's no longer one-sided.
    let (mut scene, warnings) = super::composite_render::render_composite_scene(&node, &leaves, &t)
        .map_err(composite_render_err_to_py)?;
    emit_warnings(py, &warnings)?;
    let packed_bytes = super::pack_instances::extract_packed_bytes(&mut scene);
    let json = serde_json::to_string(&scene)
        .map_err(|e| PyValueError::new_err(format!("composite scene serialization: {e}")))?;
    let py_bytes = PyBytes::new(py, &packed_bytes);
    Ok((json, py_bytes.unbind()))
}

fn composite_render_err_to_py(e: super::composite_render::CompositeRenderError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Collect every payload reader into one Arrow batch each (same per-batch
/// collection rule as [`collect_single_batch`]) and decode the shared
/// theme/config/chart-config/viewport inputs applied to every leaf.
fn decode_composite_inputs(
    payloads: Vec<PyRecordBatchReader>,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
    chart_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(
    Vec<arrow::record_batch::RecordBatch>,
    crate::layout::ThemeInputs,
    super::config::RenderConfig,
    super::chart_config::ChartConfig,
    Viewport,
)> {
    let batches = payloads
        .into_iter()
        .map(collect_single_batch)
        .collect::<PyResult<Vec<_>>>()?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let cc = chart_config_from_dict(chart_config)?;
    let vp = Viewport { width: viewport.0, height: viewport.1 };
    Ok((batches, t, c, cc, vp))
}

/// Fully-resolved per-leaf render inputs for one composite leaf (Task 5d).
///
/// Each leaf either carries its own binding override — decoded exactly like a
/// standalone `render_svg` call (built from defaults) — or inherits the
/// call-level value (a clone). Owned so [`CompositeLeafInput`] can borrow it
/// for the leaf's lifetime; the collection lives in the PyO3 entry alongside
/// the tree and batches.
#[derive(Debug)]
struct LeafBinding {
    theme: crate::layout::ThemeInputs,
    viewport: Viewport,
    config: super::config::RenderConfig,
    chart_config: super::chart_config::ChartConfig,
}

/// Walk the Python composite tree in leaf pre-order and resolve each leaf's
/// render inputs (Task 5d per-leaf binding). A leaf may carry optional
/// `theme`/`viewport`/`config`/`chart_config` keys; an absent key inherits the
/// call-level value, so a homogeneous tree (Task 6/7's) resolves every leaf to
/// a clone of the call-level inputs — byte-identical to the pre-5d
/// single-value path. A `hole` (Task 8a) is skipped outright — it has no
/// `spec`/`data`, so no render inputs to resolve, and it must not appear in
/// the returned bindings.
///
/// `built` is the tree already produced by [`composite_tree_from_py`] for this
/// same `tree` dict; the walk recurses through the raw Python dict AND
/// `built` **together**, dispatching on `built`'s node kind rather than
/// re-reading a `"kind"` string off the dict. This was previously two
/// independent pre-order walks of the same dicts — this one (render-input
/// keys) and `composite_tree_from_py`'s (spec/structure keys) — reconciled
/// only by a leaf-count check in `build_composite_leaves`, which would not
/// catch a same-count reorder. Walking both structures in lockstep makes that
/// structurally impossible: any mismatch between the dict's shape and the
/// already-validated `built` tree (a "leaf" dict paired with a `Composite`
/// node, a children-length mismatch, …) surfaces immediately as a typed error
/// naming the offending node, rather than only a leaf-count guard at the end.
fn collect_leaf_bindings(
    tree: &Bound<'_, PyAny>,
    built: &CompositeNode,
    call_theme: &crate::layout::ThemeInputs,
    call_viewport: Viewport,
    call_config: &super::config::RenderConfig,
    call_chart_config: &super::chart_config::ChartConfig,
) -> PyResult<Vec<LeafBinding>> {
    let mut out = Vec::new();
    collect_leaf_bindings_walk(
        tree,
        built,
        call_theme,
        call_viewport,
        call_config,
        call_chart_config,
        &mut out,
    )?;
    Ok(out)
}

fn collect_leaf_bindings_walk(
    node: &Bound<'_, PyAny>,
    built: &CompositeNode,
    call_theme: &crate::layout::ThemeInputs,
    call_viewport: Viewport,
    call_config: &super::config::RenderConfig,
    call_chart_config: &super::chart_config::ChartConfig,
    out: &mut Vec<LeafBinding>,
) -> PyResult<()> {
    let dict: &Bound<'_, PyDict> = node
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("composite tree node must be a dict"))?;
    match built {
        CompositeNode::Leaf { .. } => {
            let theme = match leaf_override_dict(dict, "theme")? {
                Some(d) => theme_from_dict(Some(&d))?,
                None => call_theme.clone(),
            };
            let viewport = match dict.get_item("viewport")? {
                Some(v) if !v.is_none() => {
                    let (w, h): (f64, f64) = v.extract()?;
                    Viewport { width: w, height: h }
                }
                _ => call_viewport,
            };
            let config = match leaf_override_dict(dict, "config")? {
                Some(d) => config_from_dict(Some(&d))?,
                None => call_config.clone(),
            };
            let chart_config = match leaf_override_dict(dict, "chart_config")? {
                Some(d) => chart_config_from_dict(Some(&d))?,
                None => call_chart_config.clone(),
            };
            out.push(LeafBinding { theme, viewport, config, chart_config });
            Ok(())
        }
        // A hole carries no render inputs of its own (no `spec`/`data`, so no
        // theme/viewport/config/chart_config to resolve) and is not a leaf —
        // it contributes nothing to `out`, keeping this walk's leaf count in
        // lockstep with `flatten_leaves` (Task 8a).
        CompositeNode::Hole { .. } => Ok(()),
        CompositeNode::Composite { children: built_children, .. } => {
            let children = dict.get_item("children")?.ok_or_else(|| {
                PyValueError::new_err("composite node missing required key 'children'")
            })?;
            let list = children
                .cast::<pyo3::types::PyList>()
                .map_err(|_| PyValueError::new_err("composite node: 'children' must be a list"))?;
            if list.len() != built_children.len() {
                return Err(PyValueError::new_err(format!(
                    "internal error: composite node has {} pydict children but {} tree children",
                    list.len(),
                    built_children.len()
                )));
            }
            for (item, built_child) in list.iter().zip(built_children) {
                collect_leaf_bindings_walk(
                    &item,
                    built_child,
                    call_theme,
                    call_viewport,
                    call_config,
                    call_chart_config,
                    out,
                )?;
            }
            Ok(())
        }
    }
}

/// Fetch an optional per-leaf override dict (`theme`/`config`/`chart_config`),
/// treating a present-but-`None` value as absent, and rejecting a non-dict
/// value with a typed error naming the key.
fn leaf_override_dict<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.cast_into::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("composite leaf '{key}' override must be a dict"))
        })?)),
    }
}

/// Pair each of `tree`'s leaves (in pre-order) with its selected Arrow batch
/// and resolved per-leaf [`LeafBinding`], validating every `data` index against
/// `batches`' length up front — a malformed tree fails fast with the offending
/// leaf pinpointed (`CompositeRenderError::LeafDataIndexOutOfBounds`) rather
/// than panicking on an out-of-bounds slice access deeper in the render core.
///
/// `bindings` must be in the same leaf pre-order (as produced by
/// [`collect_leaf_bindings`], which now walks in lockstep with `tree` itself —
/// see its doc comment). The count-equality check here is defense-in-depth,
/// not the primary safeguard: a genuine ordering drift between `tree` and
/// `bindings` can no longer happen without also tripping a typed error inside
/// `collect_leaf_bindings_walk`.
fn build_composite_leaves<'a>(
    tree: &'a crate::spec::composite::CompositeNode,
    batches: &'a [arrow::record_batch::RecordBatch],
    bindings: &'a [LeafBinding],
) -> PyResult<Vec<super::composite_render::CompositeLeafInput<'a>>> {
    let flat = super::composite::flatten_leaves(tree);
    if flat.len() != bindings.len() {
        return Err(PyValueError::new_err(format!(
            "internal error: composite tree has {} leaves but {} per-leaf \
             binding(s) were collected",
            flat.len(),
            bindings.len()
        )));
    }
    flat.into_iter()
        .enumerate()
        .map(|(index, (spec, data))| {
            let batch = batches.get(data).ok_or_else(|| {
                composite_render_err_to_py(
                    super::composite_render::CompositeRenderError::LeafDataIndexOutOfBounds {
                        kind: "leaf",
                        index,
                        data,
                        payload_count: batches.len(),
                    },
                )
            })?;
            let b = &bindings[index];
            Ok(super::composite_render::CompositeLeafInput {
                spec,
                batch,
                theme: &b.theme,
                viewport: b.viewport,
                config: &b.config,
                chart_config: &b.chart_config,
            })
        })
        .collect()
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
    "cull_labels",
];

/// The color-typed subset of [`THEME_KNOWN_KEYS`] — keys whose values are
/// color strings (parsed by `parse_color_val` → `render::color::parse_color` on the Rust
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

    // Axis label culling — magnitude and mode, independently settable
    // (`#146`; see `layout::LabelGapPolicy`).
    cull_threshold: Option<u32>,
    cull_labels: Option<bool>,
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

    // Axis label culling — one resolution for both keys, so `cull_threshold=`
    // alone keeps its pre-split meaning and `cull_labels=` can override the
    // mode at any gap (`#146`).
    t.label_gap = t.label_gap.with_overrides(spec.cull_threshold, spec.cull_labels);

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

    /// `#146` — the two axis-label-culling keys are independent on the wire.
    /// `cull_threshold=` alone still means what it always meant (magnitude
    /// AND mode); `cull_labels=` sets the mode alone; together they reach the
    /// two states the single integer could not spell.
    #[test]
    fn cull_threshold_and_cull_labels_resolve_independently() {
        use crate::layout::LabelGapPolicy;
        pyo3::Python::initialize();
        Python::attach(|py| {
            // Neither key: the built-in default.
            let d = PyDict::new(py);
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.label_gap, LabelGapPolicy::default());

            // `cull_threshold` alone: the legacy coupling, both directions.
            let d = PyDict::new(py);
            d.set_item("cull_threshold", 0_u32).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.label_gap.gap_px(), 0);
            assert!(!t.label_gap.may_cull(), "cull_threshold=0 still disables culling");

            let d = PyDict::new(py);
            d.set_item("cull_threshold", 30_u32).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.label_gap.gap_px(), 30);
            assert!(t.label_gap.may_cull(), "a positive cull_threshold still enables culling");

            // `cull_labels` alone: the mode moves, the default gap does not.
            let d = PyDict::new(py);
            d.set_item("cull_labels", false).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.label_gap.gap_px(), LabelGapPolicy::DEFAULT_GAP_PX);
            assert!(!t.label_gap.may_cull());

            // Both: the new "gap of N AND never cull" state.
            let d = PyDict::new(py);
            d.set_item("cull_threshold", 30_u32).unwrap();
            d.set_item("cull_labels", false).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.label_gap.gap_px(), 30);
            assert!(!t.label_gap.may_cull());

            // And its mirror: no gap requirement, culling permitted.
            let d = PyDict::new(py);
            d.set_item("cull_threshold", 0_u32).unwrap();
            d.set_item("cull_labels", true).unwrap();
            let t = theme_from_dict(Some(&d)).unwrap();
            assert_eq!(t.label_gap.gap_px(), 0);
            assert!(t.label_gap.may_cull());
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
        const BOOL_KEYS: &[&str] = &["axis_line", "grid", "minor", "cull_labels"];
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
    /// error. Hoisted to `chart_config::accepted_fields_from_deny_unknown_error`
    /// (this task, 2026-09-02) — the `chart_config`-side wire-key drift tests
    /// need the same parsing, so this used to be a private copy here; both
    /// `binding.rs` and `chart_config.rs` are in that task's footprint, so the
    /// duplicate is hoisted rather than kept (mirror-by-reference rule).
    use crate::render::chart_config::accepted_fields_from_deny_unknown_error as accepted_fields_from_error;

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
            super::color::parse_color(&s).map_err(|e| PyValueError::new_err(e.to_string()))?,
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
            validate_chart_config_keys(d)?;
            let val = crate::pyo3_serde::from_py(d.as_any(), "chart_config")?;
            Ok(val)
        }
    }
}

/// Wire refusal for `chart_config`'s unknown-key gate (D1, spec §4.1/§6).
/// Pinned verbatim-substring contract, both languages:
/// `chart config: unknown key '<k>' in <section>; accepted: <sorted list>`.
/// Do not reword independently — Python's mirror of this text must match.
fn chart_config_unknown_key_err(key: &str, section: &str, accepted: &[&str]) -> PyErr {
    let mut sorted: Vec<&str> = accepted.to_vec();
    sorted.sort_unstable();
    PyValueError::new_err(format!(
        "chart config: unknown key '{key}' in {section}; accepted: {}",
        sorted.join(", ")
    ))
}

/// The wire-key gate for `chart_config` (D1, spec §4.1/§6): refuses an
/// unknown top-level section, or an unknown key inside a gated section,
/// BEFORE the dict ever reaches `pyo3_serde::from_py`/serde. Runs first in
/// `chart_config_from_dict` — necessary because `AxisConfigSpec`/
/// `LegendConfigSpec` flatten a shared style struct, and serde's own
/// `deny_unknown_fields` diagnostic (naming the accepted set) is defeated by
/// that flatten for a NESTED unknown key (see the crate-level note on
/// `chart_config::AxisStyleSpec`); every section's accepted-key list is
/// schema-derived (`chart_config::accepted_keys_for_section`), so this
/// function owns only the dict walk and the pinned error text.
///
/// `annotations`/`structural` are ARRAY sections: [`validate_chart_config_array_section`]
/// gates each item's keys against its own `type` tag (#92/#148), rather than
/// against one section-wide set the way every other (dict-shaped) section is
/// gated below — matching Task 8's grid-sub-struct nesting shape (one level
/// deeper, keyed by a discriminator) more than it matches the top-level
/// per-section gate.
fn validate_chart_config_keys(d: &Bound<'_, PyDict>) -> PyResult<()> {
    for (key_obj, value) in d.iter() {
        let key: String = key_obj.extract()?;
        if !super::chart_config::CHART_CONFIG_SECTIONS.contains(&key.as_str()) {
            return Err(chart_config_unknown_key_err(
                &key,
                "chart_config",
                super::chart_config::CHART_CONFIG_SECTIONS,
            ));
        }
        if key == "annotations" || key == "structural" {
            validate_chart_config_array_section(&key, &value)?;
            continue;
        }
        let Some(accepted) = super::chart_config::accepted_keys_for_section(&key) else {
            continue;
        };
        let Ok(section) = value.cast::<PyDict>() else {
            // Not a dict (e.g. `None`, or a genuinely malformed value) — let
            // the subsequent serde deserialize surface the type-mismatch
            // error; this gate only walks dict-shaped sections.
            continue;
        };
        for (inner_key_obj, inner_value) in section.iter() {
            let inner_key: String = inner_key_obj.extract()?;
            if !accepted.contains(&inner_key.as_str()) {
                return Err(chart_config_unknown_key_err(&inner_key, &key, &accepted));
            }
            // One level deeper for the nested per-axis objects (`grid.x` /
            // `grid.y`, D4/spec §4.3). Without this the sub-struct's own keys
            // would be gated only by serde's untagged bool-or-object arm,
            // whose failure text names no accepted set — the exact gap the
            // pinned refusal exists to close. `None` (every non-nesting key,
            // and the bool spelling of a nesting one) walks no further.
            let Some(nested) = super::chart_config::accepted_keys_for_nested(&key, &inner_key)
            else {
                continue;
            };
            let Ok(nested_dict) = inner_value.cast::<PyDict>() else {
                continue;
            };
            for (leaf_obj, _) in nested_dict.iter() {
                let leaf: String = leaf_obj.extract()?;
                if !nested.contains(&leaf.as_str()) {
                    let section_path = format!("{key}.{inner_key}");
                    return Err(chart_config_unknown_key_err(&leaf, &section_path, nested));
                }
            }
        }
    }
    Ok(())
}

/// The wire-key gate's array branch for `annotations`/`structural` (#92/#148):
/// walks each item, reads its `type` discriminator, and gates the item's own
/// keys against [`chart_config::accepted_keys_for_array_item`]'s per-variant
/// set — closing the carve-out `accepted_keys_for_section` documents (an
/// array section has no single per-dict key set of its own). Also runs
/// [`validate_break_axis_gaps_value`] for a `break_axis` item, the #92 VALUE
/// half of the same closure.
///
/// A non-list `value` (let serde's type-mismatch error surface it), a
/// non-dict item, or a missing/non-string/unrecognized `type` tag all walk no
/// further here — each of those is left to serde's own diagnostic
/// (`data did not match any variant of internally tagged enum`, or
/// `missing field 'type'`), matching how the dict-section walk above leaves a
/// non-dict section value to serde. This function's job is narrowly the
/// pinned-text refusal for a genuinely unknown key or value on an otherwise
/// well-formed, recognized item.
///
/// Runs entirely on the RAW `Bound<'_, PyDict>`/`PyList`, before
/// `chart_config_from_dict`'s `pyo3_serde::from_py` call — deliberately, not
/// just for ordering. `from_py` bridges through `json.dumps`/`serde_json::
/// from_str` (see `pyo3_serde.rs`'s module doc), and Python's `json.dumps`
/// serializes a non-finite float as the bare `NaN`/`Infinity` token (its
/// `allow_nan=True` default) — a token strict JSON (and so `serde_json`)
/// rejects, so a non-finite gap bound already fails on that path today, but
/// with an opaque `expected value at line 1 column N` parse error naming
/// neither the field nor why. Extracting `f64` straight off the Python object
/// here (`PyFloat_AsDouble`, no JSON in between) preserves `NaN` exactly and
/// lets this gate give the SAME refusal Python's own `BreakAxis.__post_init__`
/// gives, before the JSON bridge ever gets a chance to produce its opaque one.
fn validate_chart_config_array_section(section: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let Ok(list) = value.cast::<PyList>() else {
        return Ok(());
    };
    for (idx, item) in list.iter().enumerate() {
        let Ok(item_dict) = item.cast::<PyDict>() else {
            continue;
        };
        let Some(type_tag) = item_dict
            .get_item("type")?
            .and_then(|v| v.extract::<String>().ok())
        else {
            continue;
        };
        let Some(accepted) = super::chart_config::accepted_keys_for_array_item(section, &type_tag)
        else {
            continue;
        };
        let item_section = format!("{section}[{idx}] ({type_tag})");
        for (inner_key_obj, _) in item_dict.iter() {
            let inner_key: String = inner_key_obj.extract()?;
            if !accepted.contains(&inner_key.as_str()) {
                return Err(chart_config_unknown_key_err(&inner_key, &item_section, accepted));
            }
        }
        if section == "structural" && type_tag == "break_axis" {
            validate_break_axis_axis_value(item_dict, &item_section)?;
            validate_break_axis_gaps_value(item_dict, &item_section)?;
        }
    }
    Ok(())
}

/// The `#229`(2) VALUE half of the array-item gate: refuses a `break_axis`
/// item's `axis` token that is outside the two-token vocabulary
/// [`AlignAxis`](crate::spec::axis::AlignAxis) spells, mirroring
/// `ferrum.structural.BreakAxis.__post_init__`'s
/// `validate_choice("BreakAxis.axis", ..., ("x", "y"))` so the wire path and
/// Python's construction-time refusal agree on what "malformed" means for a
/// break axis — exactly the reason [`validate_break_axis_gaps_value`] below
/// exists for the sibling `gaps` key.
///
/// Without it the refusal still happens, but downstream in the JSON bridge, as
/// serde's `unknown variant \`z\`, expected \`x\` or \`y\`` — which names
/// neither the item nor its index, leaks a JSON column number, and carries a
/// different prefix (`chart_config:`) from every other refusal about this same
/// item (`chart config: ...`).
///
/// # What the coverage claim is quantified over: `json.dumps`, not a shape list
///
/// This gate reads the caller's **pre-JSON Python object**, while the vocabulary
/// it guards is defined over what **`json.dumps` emits** (`pyo3_serde::from_py`
/// bridges through it). So "does this gate see every token a caller can present"
/// is a question about that transform, and the only honest way to answer it is
/// to derive the answer from the transform's own rules — not from a list of
/// shapes someone happened to try. Two earlier cuts of this doc claimed
/// completeness from probed shapes and were false both times: the first checked
/// only `"axis": "y"` and missed the externally-tagged map `{"y": null}`
/// `Deserialize` also accepts; the second covered that map but modelled serde's
/// accepted JSON shapes while the input here is Python objects one transform
/// earlier, missing every non-`str` dict key `json.dumps` COERCES into a key.
///
/// The transform's rules close the space. CPython's `json.encoder` key path
/// accepts exactly five key types and raises `TypeError: keys must be str, int,
/// float, bool or None, not X` for everything else, so:
///
/// | key type | JSON key `json.dumps` emits |
/// |---|---|
/// | `str` | itself |
/// | `float` | `NaN` / `Infinity` / `-Infinity` when non-finite, else `float.__repr__` (`1.5`, `1e+16`) |
/// | `bool` | `true` / `false` — checked BEFORE `int`, since `bool` is an `int` subclass |
/// | `None` | `null` |
/// | `int` | `str(k)`, at arbitrary precision |
/// | anything else | never becomes a key: `json.dumps` raises first |
///
/// [`json_object_key_spelling`] mirrors that table in that order, so this gate
/// sees a token for **every** single-key map a caller can get through the
/// bridge, and the claim below is checkable against `json.dumps`'s rules rather
/// than against examples. The four non-`str` types are out of vocabulary *by
/// construction* — no spelling of a number, `true`/`false` or `null` can equal
/// `x` or `y` — so fidelity of the echoed token is about naming what serde
/// would have seen, never about the accept/reject decision.
///
/// What still walks past, deliberately, each left to deserialization's own
/// "missing field" / type-mismatch error and matching the sibling gate's
/// carve-outs: a missing `axis` key; a value that is neither a string nor a
/// dict (`7`, `null`); a dict whose key count is not exactly one, since "which
/// key did you mean" is not a question this gate answers better than serde; and
/// a dict whose single key is of a non-coercible type, which `json.dumps`
/// refuses before serde ever runs (that refusal is a raw `TypeError` from the
/// bridge — pre-existing for every malformed `chart_config` dict, logged as a
/// follow-up, not introduced here).
///
/// Whether a token is accepted is decided by
/// [`AlignAxis::from_wire_name`](crate::spec::axis::AlignAxis::from_wire_name),
/// which delegates to `Deserialize` — so this gate cannot become narrower than
/// the deserializer it front-runs, by construction rather than by a test over a
/// fixed token corpus.
///
/// The division of labour that leaves, stated exactly rather than as "serde is
/// unreachable" (it is not, and an earlier draft of this doc claimed it was):
/// **this gate owns the VOCABULARY for every key spelling `json.dumps` can
/// produce**, so no out-of-vocabulary token reaches serde through the bridge;
/// serde keeps the SHAPE errors, which it reports better than this gate could —
/// `invalid type: integer \`7\`, expected string or map`, `invalid value: map,
/// expected map with a single key`. Both halves are probed on the delivered
/// path in `tests/test_config_manifest.py`'s break-axis block, the vocabulary
/// half over all five key types.
fn validate_break_axis_axis_value(item_dict: &Bound<'_, PyDict>, item_section: &str) -> PyResult<()> {
    let Some(axis_obj) = item_dict.get_item("axis")? else {
        return Ok(());
    };
    let Some(token) = break_axis_axis_token(&axis_obj) else {
        return Ok(());
    };
    if crate::spec::axis::AlignAxis::from_wire_name(&token).is_some() {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "chart config: {item_section}.axis: axis must be one of [{}]; got '{token}'",
        crate::spec::axis::AlignAxis::wire_token_list()
    )))
}

/// The variant token an `axis` wire value names, in either spelling
/// `Deserialize` accepts for a unit variant: the string itself, or the single
/// key of an externally-tagged one-key map. `None` for every other shape, which
/// [`validate_break_axis_axis_value`] passes to serde untouched.
fn break_axis_axis_token(axis_obj: &Bound<'_, PyAny>) -> Option<String> {
    if let Ok(token) = axis_obj.extract::<String>() {
        return Some(token);
    }
    let dict = axis_obj.cast::<PyDict>().ok()?;
    if dict.len() != 1 {
        return None;
    }
    let (key, _) = dict.iter().next()?;
    json_object_key_spelling(&key)
}

/// The JSON object key `json.dumps` would emit for `key`, or `None` when
/// `json.dumps` would refuse it as a key (and so raise before serde runs).
///
/// A faithful mirror of CPython's `json.encoder` key path, **in its order**:
/// `str`, then `float`, then `True`/`False`, then `None`, then `int`. The order
/// is load-bearing in one place — `bool` is an `int` subclass, and `json` spells
/// `True` as `true`, not `1`. See [`validate_break_axis_axis_value`]'s doc for
/// why this function exists and what its completeness is quantified over.
///
/// The two spellings Rust must not re-derive itself are asked of Python:
/// `int` at arbitrary precision (`str(k)`, not a fixed-width extract, so a
/// 30-digit key is echoed intact) and finite `float` (`float.__repr__`, which
/// is `1e+16` where Rust's own `{}` would write `10000000000000000`).
fn json_object_key_spelling(key: &Bound<'_, PyAny>) -> Option<String> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

    if key.is_instance_of::<PyString>() {
        return key.extract::<String>().ok();
    }
    if key.is_instance_of::<PyFloat>() {
        let v = key.extract::<f64>().ok()?;
        if v.is_nan() {
            return Some("NaN".to_string());
        }
        if v.is_infinite() {
            return Some(if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string());
        }
        // `json`'s `_floatstr` calls `float.__repr__(o)` — not the object's own
        // `__repr__`, which a subclass may override — so ask a plain float.
        return PyFloat::new(key.py(), v).repr().ok()?.extract::<String>().ok();
    }
    // Before `PyInt`: `bool` is an `int` subclass and `json` spells it
    // `true`/`false`.
    if key.is_instance_of::<PyBool>() {
        let b = key.extract::<bool>().ok()?;
        return Some(if b { "true" } else { "false" }.to_string());
    }
    if key.is_none() {
        return Some("null".to_string());
    }
    if key.is_instance_of::<PyInt>() {
        return key.str().ok()?.extract::<String>().ok();
    }
    // Not a key type `json.dumps` coerces: it raises `TypeError` before serde,
    // so there is no token here to validate.
    None
}

/// The #92 VALUE half of the array-item gate: refuses a non-finite or
/// reversed `(start, end)` pair in a `break_axis` item's `gaps`, mirroring
/// `ferrum.structural._validate_break_gaps`'s finite+ordered check
/// (`ferrum._validate.validate_finite_ordered_pair`) so the wire path and
/// Python's construction-time refusal agree on what "malformed" means for a
/// `BreakAxis` gap. Without this, a non-finite gap bound that somehow reached
/// `apply_break_to_scale` would resolve via `f64::max`/`f64::min` to a
/// domain-covering gap that silently erases every datum (BRK-1);
/// `apply_break_to_scale` itself stays infallible per the `chart_config.rs:
/// 184-195` layering doctrine, and `break_axis::normalize_gaps`'s own
/// non-finite filter is separate defense-in-depth for a caller that builds a
/// `BreakAxisSpec` directly, bypassing this wire gate entirely — this
/// function is the primary, well-messaged refusal for every wire caller.
///
/// A missing `gaps` key, a non-list value, or a malformed pair (not exactly 2
/// numeric entries) all walk no further here — each is left to
/// deserialization's own "missing field" / type-mismatch error, matching this
/// gate's existing carve-outs elsewhere.
fn validate_break_axis_gaps_value(item_dict: &Bound<'_, PyDict>, item_section: &str) -> PyResult<()> {
    let Some(gaps_obj) = item_dict.get_item("gaps")? else {
        return Ok(());
    };
    // `try_iter` (not a `PyList` cast) so a tuple-of-tuples wire spec — which
    // `json.dumps` serializes identically to a list-of-lists, so deserialization
    // treats them the same — is walked the same way this gate's own raw-dict
    // read does.
    let Ok(gaps_iter) = gaps_obj.try_iter() else {
        return Ok(());
    };
    for (j, gap_res) in gaps_iter.enumerate() {
        let Ok(gap) = gap_res else { continue };
        let Ok(pair) = gap.extract::<Vec<f64>>() else {
            continue;
        };
        if pair.len() != 2 {
            continue;
        }
        let (start, end) = (pair[0], pair[1]);
        let label = format!("{item_section}.gaps[{j}]");
        for (name, v) in [("start", start), ("end", end)] {
            if !v.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "chart config: {label}: {name} must be finite; received {v}"
                )));
            }
        }
        if start >= end {
            return Err(PyValueError::new_err(format!(
                "chart config: {label}: start ({start:?}) must be less than end ({end:?}); \
                 got ({start:?}, {end:?})"
            )));
        }
    }
    Ok(())
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

#[cfg(test)]
mod chart_config_gate_tests {
    //! Wire-key gate tests (D1, spec §4.1/§6). RED-proof for this gate (run
    //! manually, not committed as a test): comment out the
    //! `validate_chart_config_keys(d)?;` line in `chart_config_from_dict` and
    //! re-run `chart_config_top_level_unknown_key_refused` — it fails, because
    //! `pyo3_serde::from_py` alone (before this task's `deny_unknown_fields`
    //! additions) silently swallowed the bogus key. Restore the line before
    //! committing.

    use super::*;

    #[test]
    fn chart_config_top_level_unknown_key_refused() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("bogus_top_level_key", 1).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("unknown top-level key must be refused");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains("chart config: unknown key 'bogus_top_level_key' in chart_config; accepted:"),
                "message must match the pinned refusal text: {msg}"
            );
        });
    }

    #[test]
    fn chart_config_nested_unknown_key_refused_names_section() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let inner = PyDict::new(py);
            inner.set_item("bogus_nested_key", 1).unwrap();
            let d = PyDict::new(py);
            d.set_item("axis_y2", &inner).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("unknown nested key must be refused");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains("chart config: unknown key 'bogus_nested_key' in axis_y2; accepted:"),
                "message must name the nested section, not the top-level: {msg}"
            );
        });
    }

    #[test]
    fn chart_config_deprecated_axis_xy_keys_refused() {
        // D1/NF-B12: the deprecated `x`/`y` AxisConfig flags are not in the
        // accepted set — a raw-dict caller still emitting them (or an
        // AxisConfig.to_dict() that has not yet dropped them) is refused, not
        // silently swallowed. This is the exact pre-fix repro from F-L07-06's
        // evidence quote.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let inner = PyDict::new(py);
            inner.set_item("x", true).unwrap();
            inner.set_item("y", true).unwrap();
            let d = PyDict::new(py);
            d.set_item("axis_x", &inner).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("deprecated x/y keys must be refused");
            let msg = err.value(py).to_string();
            assert!(msg.contains("unknown key 'x' in axis_x"), "{msg}");
        });
    }

    #[test]
    fn chart_config_axis_y2_valid_keys_accepted() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let inner = PyDict::new(py);
            inner.set_item("label_color", "#654321").unwrap();
            inner.set_item("tick_count", 3).unwrap();
            let d = PyDict::new(py);
            d.set_item("axis_y2", &inner).unwrap();
            let cc = chart_config_from_dict(Some(&d)).expect("valid axis_y2 keys must be accepted");
            let axis_y2 = cc.axis_y2.expect("axis_y2 must deserialize");
            assert_eq!(axis_y2.style.label_color.as_deref(), Some("#654321"));
            assert_eq!(axis_y2.style.tick_count, Some(3));
        });
    }

    #[test]
    fn chart_config_every_gated_section_rejects_a_bogus_key() {
        // Sweeps every section `accepted_keys_for_section` covers, proving the
        // gate is wired uniformly rather than for axis_y2 alone.
        pyo3::Python::initialize();
        Python::attach(|py| {
            for section in ["axis", "axis_x", "axis_y", "axis_y2", "legend", "grid", "padding", "color", "title"] {
                let inner = PyDict::new(py);
                inner.set_item("definitely_not_a_real_key", 1).unwrap();
                let d = PyDict::new(py);
                d.set_item(section, &inner).unwrap();
                let res = chart_config_from_dict(Some(&d));
                let err = match res {
                    Ok(_) => panic!("section `{section}` must refuse a bogus key"),
                    Err(e) => e,
                };
                let msg = err.value(py).to_string();
                assert!(
                    msg.contains(&format!("unknown key 'definitely_not_a_real_key' in {section}")),
                    "section `{section}`: {msg}"
                );
            }
        });
    }

    /// RED-proof for the #148 array-item key gate: comment out the
    /// `if key == "annotations" || key == "structural" { ... }` branch in
    /// `validate_chart_config_keys` and re-run this test — it passes when it
    /// should fail, because `pyo3_serde::from_py`'s `deny_unknown_fields`
    /// (added alongside this gate) is the ONLY thing left refusing the bogus
    /// key, with a raw serde message instead of this gate's pinned text.
    /// Restore the branch before committing.
    #[test]
    fn chart_config_annotation_array_item_rejects_a_bogus_key() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let item = PyDict::new(py);
            item.set_item("type", "text").unwrap();
            item.set_item("x", 1.0).unwrap();
            item.set_item("y", 2.0).unwrap();
            item.set_item("text", "hi").unwrap();
            item.set_item("definitely_not_a_real_key", 1).unwrap();
            let list = PyList::new(py, [item]).unwrap();
            let d = PyDict::new(py);
            d.set_item("annotations", list).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("unknown annotation item key must be refused");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains("chart config: unknown key 'definitely_not_a_real_key' in annotations[0] (text)"),
                "{msg}"
            );
        });
    }

    #[test]
    fn chart_config_structural_array_item_rejects_a_bogus_key() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let item = PyDict::new(py);
            item.set_item("type", "break_axis").unwrap();
            item.set_item("axis", "y").unwrap();
            item.set_item("gaps", vec![vec![10.0, 20.0]]).unwrap();
            item.set_item("definitely_not_a_real_key", 1).unwrap();
            let list = PyList::new(py, [item]).unwrap();
            let d = PyDict::new(py);
            d.set_item("structural", list).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("unknown structural item key must be refused");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains("chart config: unknown key 'definitely_not_a_real_key' in structural[0] (break_axis)"),
                "{msg}"
            );
        });
    }

    #[test]
    fn chart_config_annotation_array_item_valid_keys_accepted() {
        // Positive control for the two refusal tests above — proves the gate
        // does not over-refuse a well-formed item.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let item = PyDict::new(py);
            item.set_item("type", "text").unwrap();
            item.set_item("x", 1.0).unwrap();
            item.set_item("y", 2.0).unwrap();
            item.set_item("text", "hi").unwrap();
            let list = PyList::new(py, [item]).unwrap();
            let d = PyDict::new(py);
            d.set_item("annotations", list).unwrap();
            let cc = chart_config_from_dict(Some(&d)).expect("a well-formed annotation item must be accepted");
            assert_eq!(cc.annotations.len(), 1);
        });
    }

    /// RED-proof for #92's Rust half: comment out the
    /// `validate_chart_config_values(&val)?;` line in `chart_config_from_dict`
    /// and re-run this test — it fails, because deserialization alone accepts
    /// `NaN` as a syntactically valid `f64` gap bound and this refusal is the
    /// only thing that rejects it. Restore the line before committing.
    #[test]
    fn chart_config_break_axis_nan_gap_refused() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let item = PyDict::new(py);
            item.set_item("type", "break_axis").unwrap();
            item.set_item("axis", "y").unwrap();
            item.set_item("gaps", vec![vec![f64::NAN, f64::NAN]]).unwrap();
            let list = PyList::new(py, [item]).unwrap();
            let d = PyDict::new(py);
            d.set_item("structural", list).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("a NaN gap bound must be refused");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains("chart config: structural[0] (break_axis).gaps[0]: start must be finite"),
                "{msg}"
            );
        });
    }

    #[test]
    fn chart_config_break_axis_reversed_gap_refused() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let item = PyDict::new(py);
            item.set_item("type", "break_axis").unwrap();
            item.set_item("axis", "y").unwrap();
            item.set_item("gaps", vec![vec![200.0, 100.0]]).unwrap();
            let list = PyList::new(py, [item]).unwrap();
            let d = PyDict::new(py);
            d.set_item("structural", list).unwrap();
            let err = chart_config_from_dict(Some(&d)).expect_err("a reversed gap must be refused");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains(
                    "chart config: structural[0] (break_axis).gaps[0]: start (200.0) must be less than end (100.0)"
                ),
                "{msg}"
            );
        });
    }

    #[test]
    fn chart_config_break_axis_finite_ordered_gap_accepted() {
        // Positive control for the two refusal tests above.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let item = PyDict::new(py);
            item.set_item("type", "break_axis").unwrap();
            item.set_item("axis", "y").unwrap();
            item.set_item("gaps", vec![vec![50.0, 100.0]]).unwrap();
            let list = PyList::new(py, [item]).unwrap();
            let d = PyDict::new(py);
            d.set_item("structural", list).unwrap();
            let cc = chart_config_from_dict(Some(&d)).expect("a finite ordered gap must be accepted");
            assert_eq!(cc.structural.len(), 1);
        });
    }

    /// Minimal fixture for the byte-identity pin below: the gate itself has
    /// no opinion about mark/encoding, so any renderable chart works — this
    /// mirrors `orchestration_tests::scatter_3` (`render/mod.rs`), a private
    /// helper of that sibling test module and not reachable from here.
    fn minimal_line_chart() -> (ChartSpec, arrow::record_batch::RecordBatch) {
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    #[test]
    fn chart_config_from_dict_populated_axis_matches_direct_rust_construction() {
        // Byte-identity pin (constraint §4), replacing a cycle-2 pin that
        // could not fail for any reason related to this task (spec review):
        // `chart_config_from_dict(None)` returns `ChartConfig::default()`
        // and `Some(&empty dict)` deserializes to the identical value (every
        // field is `Option` + `#[serde(default)]`), so a None-vs-empty-dict
        // render comparison only asserted `render_svg_internal` determinism,
        // not gate neutrality.
        //
        // This pin targets the task's riskiest change directly:
        // `#[serde(deny_unknown_fields)]` added to `AxisConfigSpec`, a
        // struct that also `#[serde(flatten)]`s `AxisStyleSpec` — a
        // combination serde's own docs call unsupported (see the doc on
        // `AxisConfigSpec`). A POPULATED `axis_x` config (label_angle +
        // tick_values, exercising both the flattened style field and its
        // alias resolution) is rendered two ways: once through the real
        // wire path (`chart_config_from_dict` -> PyDict -> deserialize),
        // once via a `ChartConfig` built directly in Rust with no
        // PyO3/serde involvement at all. If `deny_unknown_fields`-on-
        // flatten silently dropped or misread a field on the wire path, the
        // two renders would diverge.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let (spec, batch) = minimal_line_chart();
            let theme = ThemeInputs::default();
            let viewport = Viewport { width: 400.0, height: 300.0 };
            let render_config = RenderConfig::default();

            let inner = PyDict::new(py);
            inner.set_item("label_angle", -30.0).unwrap();
            inner.set_item("tick_values", vec![0.0, 1.0, 2.0]).unwrap();
            let d = PyDict::new(py);
            d.set_item("axis_x", &inner).unwrap();
            let cc_from_dict =
                chart_config_from_dict(Some(&d)).expect("populated axis_x config must be accepted");

            let cc_direct = ChartConfig {
                axis_x: Some(crate::render::chart_config::AxisConfigSpec {
                    style: crate::render::chart_config::AxisStyleSpec {
                        label_angle: Some(-30.0),
                        values: Some(vec![0.0, 1.0, 2.0]),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                ..Default::default()
            };

            let svg_from_dict = render_svg_internal(
                &spec,
                &batch,
                &theme,
                viewport,
                &render_config,
                &cc_from_dict,
            )
            .unwrap();
            let svg_direct = render_svg_internal(
                &spec,
                &batch,
                &theme,
                viewport,
                &render_config,
                &cc_direct,
            )
            .unwrap();

            assert_eq!(
                svg_from_dict.bytes, svg_direct.bytes,
                "chart_config_from_dict's deserialized ChartConfig must render byte-identically \
                 to the same config constructed directly in Rust — a divergence here means \
                 deny_unknown_fields-on-flatten (AxisConfigSpec/AxisStyleSpec) is silently \
                 dropping or misreading a field on the wire path"
            );
        });
    }
}

// ---------------------------------------------------------------------------
// Figure chrome bindings (Task 11; N-ary SVG compositor removed Task 10 stage 3)
// ---------------------------------------------------------------------------

use crate::render::figure_chrome::{ChromeAnchor, FigureChrome, DEFAULT_CHROME_INSET};

/// Parse the user-facing chrome-anchor string into a [`ChromeAnchor`].
///
/// `None` maps to the default left alignment; any other string is delegated
/// to [`ChromeAnchor`]'s `FromStr` impl and rejected with a `ValueError`
/// naming the valid set on failure (shared with the composite tree's root
/// `config` slot parser, `render/composite_render.rs`).
fn parse_chrome_anchor(anchor: Option<&str>) -> PyResult<ChromeAnchor> {
    match anchor {
        None => Ok(ChromeAnchor::Start),
        Some(s) => s.parse::<ChromeAnchor>().map_err(|e| PyValueError::new_err(e.to_string())),
    }
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
        ..Default::default()
    })
}

/// Wrap a single already-rendered SVG with a figure-level chrome band
/// (title/subtitle/caption).
///
/// This is the flat single-chart counterpart of the (now-removed) N-ary SVG
/// compositor's chrome kwargs, extracted (Task 10 stage 3) so
/// `Chart.to_svg()`'s `.properties(caption=)` post-wrap in `_render.py` no
/// longer routes a single SVG through the general N-ary compositor. Not
/// re-exported in `ferrum.__all__` — it is a private `ferrum._core`
/// implementation detail of `_render.py`.
///
/// Parameters
/// ----------
/// svg : str
///     The already-rendered SVG document string to wrap.
/// title : str or None, default None
///     Figure-level title rendered as a band above the chart (bold, 16 px).
/// subtitle : str or None, default None
///     Figure-level subtitle rendered below the title, above the chart (13 px).
/// caption : str or None, default None
///     Figure-level caption rendered as a band below the chart (muted, 11 px).
/// left_inset : float or None, default None
///     Horizontal inset (px) from the left edge for ``"start"``-anchored
///     chrome. Defaults to 16.0.
/// right_inset : float or None, default None
///     Horizontal inset (px) from the right edge for ``"end"``-anchored
///     chrome. Defaults to 16.0.
/// anchor : str or None, default None
///     Horizontal alignment of all chrome lines. One of ``"start"``,
///     ``"middle"``, or ``"end"``. ``None`` is treated as ``"start"``.
///
/// Returns
/// -------
/// str
///     The wrapped SVG document. When *title*, *subtitle*, and *caption* are
///     all ``None`` the output is byte-identical to what the (now-removed)
///     N-ary SVG compositor's vertical-stack entry point produced for a
///     one-element list with ``spacing=0.0`` and the same chrome parameters.
///
/// Raises
/// ------
/// ValueError
///     If *svg* cannot be parsed, or *anchor* is not one of the accepted
///     values.
#[pyfunction]
#[pyo3(name = "wrap_svg_with_chrome")]
#[pyo3(signature = (svg, *, title = None, subtitle = None, caption = None, left_inset = None, right_inset = None, anchor = None))]
#[allow(clippy::too_many_arguments)]
pub fn wrap_svg_with_chrome_py(
    svg: String,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: Option<f64>,
    right_inset: Option<f64>,
    anchor: Option<&str>,
) -> PyResult<String> {
    let chrome = build_chrome(title, subtitle, caption, left_inset, right_inset, anchor)?;
    crate::render::figure_chrome::wrap_svg_with_chrome(&svg, chrome)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Lay out a figure-level chrome title/subtitle/caption band as scene nodes.
///
/// Produces the interactive (WASM) counterpart of the SVG band emitted by
/// ``wrap_svg_with_chrome`` (flat single-chart) and the composite tree's root
/// chrome config (composed figures). The Python composite
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
///       interactive canvas matches the SVG canvas that the chrome-wrap emitters
///       (``wrap_svg_with_chrome`` / the composite chrome band) produce.
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

// ── Task 5c gap-fix (spec-review item 4): LeafDataIndexOutOfBounds ─────────
//
// `build_composite_leaves` bounds-checks each leaf's `data` index against the
// caller's payload list *before* any leaf renders, so a malformed tree fails
// fast with the offending leaf pinpointed rather than panicking on an
// out-of-bounds slice access deeper in `render_composite_scene`. No test
// previously exercised this boundary check directly.
#[cfg(test)]
mod composite_leaf_bounds_tests {
    use super::*;
    use crate::spec::composite::CompositeNode;
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    /// A single-leaf composite tree whose leaf selects payload index `data`.
    fn dummy_leaf(data: usize) -> CompositeNode {
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        };
        CompositeNode::Leaf { spec: Box::new(spec), data, label: None }
    }

    /// One inherited-from-call-level [`LeafBinding`] per leaf — the homogeneous
    /// (pre-5d) shape the bounds tests exercise. Length must match the tree's
    /// leaf count (these single-leaf trees need exactly one).
    fn inherited_bindings(n: usize) -> Vec<LeafBinding> {
        (0..n)
            .map(|_| LeafBinding {
                theme: ThemeInputs::default(),
                viewport: Viewport { width: 100.0, height: 100.0 },
                config: RenderConfig::default(),
                chart_config: ChartConfig::default(),
            })
            .collect()
    }

    #[test]
    fn build_composite_leaves_rejects_out_of_bounds_data_index() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // Leaf #0 asks for payload index 2, but zero payloads are supplied.
            let tree = dummy_leaf(2);
            let batches: Vec<arrow::record_batch::RecordBatch> = vec![];
            let bindings = inherited_bindings(1);

            let result = build_composite_leaves(&tree, &batches, &bindings);
            let err = match result {
                Ok(_) => panic!("data index 2 with 0 payloads must be rejected, not accepted"),
                Err(e) => e,
            };

            assert!(
                err.is_instance_of::<PyValueError>(py),
                "must surface as PyValueError, matching composite_render_err_to_py"
            );
            let msg = err.value(py).to_string();
            assert!(msg.contains("leaf #0"), "must name the offending leaf index, got: {msg}");
            assert!(
                msg.contains("data index 2 out of bounds"),
                "must name the invalid data index, got: {msg}"
            );
            assert!(
                msg.contains("0 payload(s) provided"),
                "must name the actual payload count, got: {msg}"
            );
        });
    }

    #[test]
    fn build_composite_leaves_accepts_in_bounds_data_index() {
        // Discriminating counterpart: the same leaf shape with an in-bounds
        // `data` index (0, with 1 payload) must succeed, proving the bounds
        // check above is not simply rejecting every leaf.
        pyo3::Python::initialize();
        Python::attach(|_py| {
            let tree = dummy_leaf(0);
            let batch = {
                use arrow::array::Float64Array;
                use arrow::datatypes::{DataType, Field, Schema};
                use std::sync::Arc;
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "x",
                    DataType::Float64,
                    false,
                )]));
                arrow::record_batch::RecordBatch::try_new(
                    schema,
                    vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
                )
                .unwrap()
            };
            let batches = vec![batch];
            let bindings = inherited_bindings(1);

            let leaves = build_composite_leaves(&tree, &batches, &bindings)
                .expect("data index 0 with 1 payload must succeed");
            assert_eq!(leaves.len(), 1);
        });
    }
}

// ── Task 5d: per-leaf binding wire (collect_leaf_bindings) ─────────────────
//
// Each composite leaf may carry its own `theme`/`viewport`/`config`/
// `chart_config` override; an absent key inherits the call-level value. These
// tests exercise the binding-layer tree walk directly, against the fused
// walk (burndown item 2): `collect_leaf_bindings` now takes the already-built
// `CompositeNode` (from `composite_tree_from_py`) alongside the raw dict and
// recurses through both together, so every leaf dict here needs a (dummy)
// `spec`/`data` pair for that build to succeed, even though the binding walk
// itself only reads the render-input keys.
#[cfg(test)]
mod composite_leaf_binding_tests {
    use super::*;
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;
    use pyo3::types::{PyDict, PyList};

    fn dummy_chart_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        }
    }

    /// A leaf dict for the fused walk: carries a dummy `spec`/`data` (so the
    /// lockstep tree build in `collect_leaf_bindings` succeeds), optionally
    /// with a `theme` and/or `viewport` override.
    fn leaf_dict<'py>(
        py: Python<'py>,
        theme: Option<&Bound<'py, PyDict>>,
        viewport: Option<(f64, f64)>,
    ) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("kind", "leaf").unwrap();
        d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
        d.set_item("data", 0usize).unwrap();
        if let Some(t) = theme {
            d.set_item("theme", t).unwrap();
        }
        if let Some(vp) = viewport {
            d.set_item("viewport", vp).unwrap();
        }
        d
    }

    /// Build the validated [`CompositeNode`] `collect_leaf_bindings` now walks
    /// in lockstep with, the same way the real PyO3 entries do.
    fn built(tree: &Bound<'_, PyAny>) -> CompositeNode {
        crate::spec::composite::composite_tree_from_py(tree).unwrap()
    }

    #[test]
    fn per_leaf_theme_and_viewport_override_or_inherit() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // Leaf 0 inherits every call-level value; leaf 1 overrides its theme
            // (point_size) and viewport.
            let theme1 = PyDict::new(py);
            theme1.set_item("point_size", 42.0).unwrap();

            let children = PyList::new(
                py,
                [
                    leaf_dict(py, None, None),
                    leaf_dict(py, Some(&theme1), Some((400.0, 250.0))),
                ],
            )
            .unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();

            let call_theme = ThemeInputs::default();
            let call_vp = Viewport { width: 100.0, height: 100.0 };
            let call_config = RenderConfig::default();
            let call_cc = ChartConfig::default();
            let node = built(root.as_any());

            let bindings = collect_leaf_bindings(
                root.as_any(),
                &node,
                &call_theme,
                call_vp,
                &call_config,
                &call_cc,
            )
            .unwrap();
            assert_eq!(bindings.len(), 2, "one binding per leaf, in pre-order");

            // Leaf 0 inherits the call-level defaults verbatim.
            assert_eq!(bindings[0].theme.sizes.point_size, call_theme.sizes.point_size);
            assert_eq!(bindings[0].viewport, call_vp);

            // Leaf 1's overrides applied — and DISTINCT from the inherited leaf,
            // proving the wire carries per-leaf values, not one collapsed value.
            assert_eq!(bindings[1].theme.sizes.point_size, 42.0);
            assert_eq!(bindings[1].viewport, Viewport { width: 400.0, height: 250.0 });
            assert_ne!(
                bindings[1].theme.sizes.point_size,
                bindings[0].theme.sizes.point_size,
                "per-leaf theme override must not leak into the inherited leaf"
            );
        });
    }

    #[test]
    fn absent_overrides_inherit_call_level_for_every_leaf() {
        // Back-compat: a homogeneous tree (no per-leaf keys) resolves every leaf
        // to the call-level value — the pre-5d single-value behavior.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let children =
                PyList::new(py, [leaf_dict(py, None, None), leaf_dict(py, None, None)]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "vconcat").unwrap();
            root.set_item("children", children).unwrap();

            let mut call_theme = ThemeInputs::default();
            call_theme.sizes.point_size = 77.0;
            let call_vp = Viewport { width: 321.0, height: 123.0 };
            let node = built(root.as_any());

            let bindings = collect_leaf_bindings(
                root.as_any(),
                &node,
                &call_theme,
                call_vp,
                &RenderConfig::default(),
                &ChartConfig::default(),
            )
            .unwrap();
            assert_eq!(bindings.len(), 2);
            for b in &bindings {
                assert_eq!(b.theme.sizes.point_size, 77.0, "inherit call-level theme");
                assert_eq!(b.viewport, call_vp, "inherit call-level viewport");
            }
        });
    }

    #[test]
    fn non_dict_theme_override_is_typed_error() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let bad = leaf_dict(py, None, None);
            bad.set_item("theme", "not a dict").unwrap();
            let children = PyList::new(py, [bad]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();
            let node = built(root.as_any());

            let err = collect_leaf_bindings(
                root.as_any(),
                &node,
                &ThemeInputs::default(),
                Viewport { width: 100.0, height: 100.0 },
                &RenderConfig::default(),
                &ChartConfig::default(),
            )
            .unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("theme") && msg.contains("dict"), "msg: {msg}");
        });
    }

    #[test]
    fn hole_contributes_no_binding_and_keeps_leaf_count_in_sync() {
        // A grid with a hole among its children: `collect_leaf_bindings_walk`
        // must skip it (Task 8a) so its binding count stays in lockstep with
        // `flatten_leaves`' leaf count (which also excludes the hole).
        pyo3::Python::initialize();
        Python::attach(|py| {
            let hole = PyDict::new(py);
            hole.set_item("kind", "hole").unwrap();
            let children =
                PyList::new(py, [leaf_dict(py, None, None), hole, leaf_dict(py, None, None)])
                    .unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "grid").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("nrows", 1usize).unwrap();
            root.set_item("ncols", 3usize).unwrap();
            let node = built(root.as_any());

            let bindings = collect_leaf_bindings(
                root.as_any(),
                &node,
                &ThemeInputs::default(),
                Viewport { width: 100.0, height: 100.0 },
                &RenderConfig::default(),
                &ChartConfig::default(),
            )
            .unwrap();
            assert_eq!(bindings.len(), 2, "hole must not produce a binding");
        });
    }
}

// ── Task 5c REVISE fix: composite warning threading ────────────────────────
//
// `render_composite_scene` used to drop every leaf's `RenderWarning`s (they
// lived on `render_leaf`'s local `po.warnings`, which went out of scope
// before the caller ever saw it), and `render_composite_svg`'s docstring
// claimed warnings were "forwarded to Python's `warnings.warn`" without ever
// calling `emit_warnings`. This test exercises the real PyO3 entry end to
// end — a leaf whose nominal color field has more distinct categories than
// the default palette holds — and asserts the warning actually reaches
// Python, mirroring how `render_svg`'s own warning contract is documented
// (this module's `Notes` sections) rather than a Rust-only unit check on
// `render_composite_scene`.
#[cfg(test)]
mod composite_warning_tests {
    use super::*;
    use crate::spec::composite::CompositeNode;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};
    use std::sync::Arc;

    /// A point spec with x/y plus a nominal `color` field — the same shape
    /// `scale_resolve::tests`'s `ColorPaletteOverflowed` fixture uses, minus
    /// the composite wrapping.
    fn color_overflow_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// 13 rows, 13 distinct `g` categories — more than every built-in
    /// categorical palette (the largest is 10 entries), so the default
    /// theme's categorical color scale always overflows regardless of which
    /// scheme resolves.
    fn color_overflow_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let groups: Vec<String> = (0..13).map(|i| format!("g{i}")).collect();
        let groups_str: Vec<&str> = groups.iter().map(String::as_str).collect();
        let xs: Vec<f64> = (0..13).map(|i| i as f64).collect();
        let ys = xs.clone();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(groups_str)),
            ],
        )
        .unwrap()
    }

    fn single_leaf_payload(batch: RecordBatch) -> PyRecordBatchReader {
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        PyRecordBatchReader::new(Box::new(reader))
    }

    /// `warnings.catch_warnings(record=True)` mutates *process-global*
    /// interpreter state (the `warnings` module's filter list); CPython
    /// documents it as not thread-safe. Cargo's default parallel test runner
    /// can run any of the three `catch_warnings`-based tests below on
    /// concurrent OS threads, which was observed to intermittently clobber a
    /// sibling's recorded messages (empty `messages` under `--test-threads`
    /// > 1, reliable under `--test-threads=1`). These are the only
    /// `catch_warnings` users in this crate, so serializing them against each
    /// other with a plain (non-GIL) mutex is sufficient — tolerate poisoning
    /// so one test's panic can't cascade-fail its siblings.
    static WARNING_CAPTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn render_composite_svg_forwards_leaf_warnings_to_python() {
        let _guard = WARNING_CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        pyo3::Python::initialize();
        Python::attach(|py| {
            let tree_dict = PyDict::new(py);
            tree_dict.set_item("kind", "leaf").unwrap();
            tree_dict
                .set_item("spec", Py::new(py, color_overflow_spec()).unwrap())
                .unwrap();
            tree_dict.set_item("data", 0usize).unwrap();
            let tree: &pyo3::Bound<'_, pyo3::PyAny> = tree_dict.as_any();

            let payloads = vec![single_leaf_payload(color_overflow_batch())];

            // Python's `warnings.catch_warnings(record=True)` context manager,
            // driven manually via `__enter__`/`__exit__` since this is plain
            // Rust, not a `#[pyfunction]` test harness — the same idiom
            // `test_scale_rendering.py`'s auto-raster warning tests use from
            // the Python side (`with warnings.catch_warnings(record=True) as
            // caught: warnings.simplefilter("always"); ...`).
            let warnings_mod = py.import("warnings").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("record", true).unwrap();
            let cw = warnings_mod
                .call_method("catch_warnings", (), Some(&kwargs))
                .unwrap();
            let caught = cw.call_method0("__enter__").unwrap();
            warnings_mod.call_method1("simplefilter", ("always",)).unwrap();

            let svg = render_composite_svg(py, tree, payloads, (300.0, 200.0), None, None, None)
                .expect("render_composite_svg should succeed for a valid single-leaf tree");
            assert!(svg.contains("<svg"), "expected a real SVG document, got: {svg}");

            let messages: Vec<String> = caught
                .try_iter()
                .unwrap()
                .map(|w| {
                    w.unwrap()
                        .getattr("message")
                        .unwrap()
                        .call_method0("__str__")
                        .unwrap()
                        .to_string()
                })
                .collect();
            cw.call_method1("__exit__", (py.None(), py.None(), py.None())).unwrap();

            assert!(
                messages.iter().any(|m| m.contains("palette")),
                "expected a color-palette-overflow warning to surface through \
                 render_composite_svg's emit_warnings call, got: {messages:?}"
            );
        });
    }

    /// GH #50 fix (burndown item 6): `render_composite_interactive` used to
    /// drop leaf `RenderWarning`s outright (deferred to match
    /// `render_interactive`'s then-identical gap). Both gaps are closed now
    /// that `render_scene_json` returns its warnings instead of discarding
    /// them — this is the composite counterpart of
    /// `render_composite_svg_forwards_leaf_warnings_to_python`, proving the
    /// interactive entry forwards the same warning through `emit_warnings`.
    #[test]
    fn render_composite_interactive_forwards_leaf_warnings_to_python() {
        let _guard = WARNING_CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        pyo3::Python::initialize();
        Python::attach(|py| {
            let tree_dict = PyDict::new(py);
            tree_dict.set_item("kind", "leaf").unwrap();
            tree_dict
                .set_item("spec", Py::new(py, color_overflow_spec()).unwrap())
                .unwrap();
            tree_dict.set_item("data", 0usize).unwrap();
            let tree: &pyo3::Bound<'_, pyo3::PyAny> = tree_dict.as_any();

            let payloads = vec![single_leaf_payload(color_overflow_batch())];

            let warnings_mod = py.import("warnings").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("record", true).unwrap();
            let cw = warnings_mod
                .call_method("catch_warnings", (), Some(&kwargs))
                .unwrap();
            let caught = cw.call_method0("__enter__").unwrap();
            warnings_mod.call_method1("simplefilter", ("always",)).unwrap();

            let (json, _packed) = render_composite_interactive(
                py,
                tree,
                payloads,
                (300.0, 200.0),
                None,
                None,
                None,
            )
            .expect("render_composite_interactive should succeed for a valid single-leaf tree");
            assert!(json.contains("\"panels\""), "expected scene JSON, got: {json}");

            let messages: Vec<String> = caught
                .try_iter()
                .unwrap()
                .map(|w| {
                    w.unwrap()
                        .getattr("message")
                        .unwrap()
                        .call_method0("__str__")
                        .unwrap()
                        .to_string()
                })
                .collect();
            cw.call_method1("__exit__", (py.None(), py.None(), py.None())).unwrap();

            assert!(
                messages.iter().any(|m| m.contains("palette")),
                "expected a color-palette-overflow warning to surface through \
                 render_composite_interactive's emit_warnings call, got: {messages:?}"
            );
        });
    }

    /// GH #50 fix, flat sibling: `render_interactive` had the identical gap
    /// (`render_scene_json` never returned its computed warnings at all, so
    /// there was nothing for this binding's `emit_warnings` to forward).
    /// Same fixture as the two composite tests above, called through the flat
    /// entry directly (no composite tree).
    #[test]
    fn render_interactive_forwards_leaf_warnings_to_python() {
        let _guard = WARNING_CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        pyo3::Python::initialize();
        Python::attach(|py| {
            let data = single_leaf_payload(color_overflow_batch());
            let spec = color_overflow_spec();

            let warnings_mod = py.import("warnings").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("record", true).unwrap();
            let cw = warnings_mod
                .call_method("catch_warnings", (), Some(&kwargs))
                .unwrap();
            let caught = cw.call_method0("__enter__").unwrap();
            warnings_mod.call_method1("simplefilter", ("always",)).unwrap();

            let (json, _packed) =
                super::render_interactive(py, &spec, data, (300.0, 200.0), None, None, None)
                    .expect("render_interactive should succeed for a valid spec/batch");
            assert!(json.contains("\"panels\""), "expected scene JSON, got: {json}");

            let messages: Vec<String> = caught
                .try_iter()
                .unwrap()
                .map(|w| {
                    w.unwrap()
                        .getattr("message")
                        .unwrap()
                        .call_method0("__str__")
                        .unwrap()
                        .to_string()
                })
                .collect();
            cw.call_method1("__exit__", (py.None(), py.None(), py.None())).unwrap();

            assert!(
                messages.iter().any(|m| m.contains("palette")),
                "expected a color-palette-overflow warning to surface through \
                 render_interactive's emit_warnings call, got: {messages:?}"
            );
        });
    }

    /// GH #162: a transform that degrades (here, `Smooth` on a single-row
    /// batch — below its 2-point fit threshold) must reach Python as a
    /// `UserWarning` through the SAME `render_svg` → `emit_warnings` path
    /// every other render warning uses, not a separate or missing channel.
    /// This is the end-to-end proof that `TransformContext.warnings`
    /// (pushed inside `smooth::apply_one_group`) survives
    /// `run_transforms_with_facet`'s drain, `RenderWarning::Transform`'s
    /// wrap, and `PreparedInputs.warnings`' merge, all the way to the PyO3
    /// boundary — RED on pre-#162 code, which had no channel to carry it.
    #[test]
    fn render_svg_surfaces_a_degenerate_transform_warning() {
        let _guard = WARNING_CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        pyo3::Python::initialize();
        Python::attach(|py| {
            use crate::transform::core::TransformSpec;
            use crate::transform::smooth::{SmoothMethod, SmoothOutput, SmoothSpec};

            let mut spec = color_overflow_spec();
            // Drop the color encoding (irrelevant here, and its own overflow
            // warning would confuse the assertion below) and point x/y at
            // Smooth's canonical output columns.
            spec.encoding.color = None;
            spec.transforms = vec![TransformSpec::Smooth(SmoothSpec {
                x: "x".into(),
                y: "y".into(),
                method: SmoothMethod::Lm,
                ci: None,
                bandwidth: 0.0,
                degree: 1,
                k: crate::transform::gam::DEFAULT_BASIS_DIM,
                n: 5,
                seed: 0,
                x_bins: None,
                x_estimator: None,
                output: SmoothOutput::Fitted,
                inject_zero_ref: false,
                inject_metrics: false,
                groupby: vec![],
                name: None,
                x_range: None,
            })];

            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(Float64Array::from(vec![1.0])), Arc::new(Float64Array::from(vec![1.0]))],
            )
            .unwrap();
            let data = single_leaf_payload(batch);

            let warnings_mod = py.import("warnings").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("record", true).unwrap();
            let cw = warnings_mod
                .call_method("catch_warnings", (), Some(&kwargs))
                .unwrap();
            let caught = cw.call_method0("__enter__").unwrap();
            warnings_mod.call_method1("simplefilter", ("always",)).unwrap();

            let svg = render_svg(py, &spec, data, (300.0, 200.0), None, None, None)
                .expect("render_svg should still render the degenerate (NaN) fit, not fail");
            assert!(svg.contains("<svg"), "expected a real SVG document, got: {svg}");

            let messages: Vec<String> = caught
                .try_iter()
                .unwrap()
                .map(|w| {
                    w.unwrap()
                        .getattr("message")
                        .unwrap()
                        .call_method0("__str__")
                        .unwrap()
                        .to_string()
                })
                .collect();
            cw.call_method1("__exit__", (py.None(), py.None(), py.None())).unwrap();

            assert!(
                messages.iter().any(|m| m.contains("Smooth") && m.contains("degraded")),
                "expected the Smooth transform's DegenerateFit warning to surface through \
                 render_svg's emit_warnings call, got: {messages:?}"
            );
        });
    }
}

/// The `background=` render option's color vocabulary (batch-A Task 8 sweep).
///
/// This is the one swept color site that only exists at the PyO3 boundary, so
/// it needs its own `Python::attach` coverage: pre-existing tests of
/// `background=` are hex-only (`tests/test_bug_hunt_composition_facet.py`), and
/// nothing else would catch a re-narrowing of this site back to hex.
#[cfg(test)]
mod render_config_background_tests {
    use super::*;
    use pyo3::types::PyDict;

    /// `background` accepts the full CSS vocabulary, and every spelling of one
    /// color resolves to the same `Srgba` — `lightgray` is CSS Color 4
    /// `(211, 211, 211)` = `#d3d3d3`.
    #[test]
    fn background_accepts_named_and_rgb_forms_identically_to_hex() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let background = |spelling: &str| {
                let d = PyDict::new(py);
                d.set_item("background", spelling).unwrap();
                config_from_dict(Some(&d))
                    .unwrap_or_else(|e| panic!("{spelling:?} must parse: {e}"))
                    .background
            };
            let expected = Some(super::super::color::from_rgb(0xd3, 0xd3, 0xd3));
            for spelling in ["lightgray", "rgb(211, 211, 211)", "#d3d3d3", "#D3D3D3"] {
                assert_eq!(background(spelling), expected, "{spelling:?} must resolve alike");
            }
        });
    }

    /// An unparseable `background` raises `ValueError` naming the accepted
    /// forms — this site refuses rather than silently keeping the theme
    /// background, because it already did so before the sweep.
    #[test]
    fn background_invalid_raises_value_error_naming_accepted_forms() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("background", "nope").unwrap();
            let err = config_from_dict(Some(&d))
                .expect_err("an unparseable background must raise, not fall back");
            assert!(err.is_instance_of::<PyValueError>(py), "expected ValueError");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains(crate::render::color::primitive::ACCEPTED_COLOR_FORMS),
                "message must quote ACCEPTED_COLOR_FORMS verbatim; got {msg:?}"
            );
            assert!(msg.contains("nope"), "message must echo the offending value; got {msg:?}");
        });
    }

    /// Absent `background` stays `None` — byte-identical to no render option.
    #[test]
    fn background_absent_leaves_config_default() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            assert_eq!(config_from_dict(Some(&d)).unwrap().background, None);
        });
    }
}

/// `ferrum._core.column_reads_categorical(schema, field)`: whether an
/// undeclared encoding over `field` of an Arrow `schema` reads categorical,
/// the answer the render's own categorical reading gives
/// (`scale_resolve::domain::dtype_reads_categorical`). Python's bar
/// zero-anchor injection asks it to pick a bar's value channel from its
/// orientation, so the two halves cannot drift. The caller passes the schema
/// of the frame after `_coerce.normalize_for_rust`, the one normalizer every
/// batch crosses before the render, so the dtypes read here are the ones the
/// render reads; nothing is cast here. A field the schema lacks reads not
/// categorical, as an absent column does in the render. Registered in
/// `lib.rs`, typed in `src/ferrum/_core.pyi`.
#[pyfunction]
pub(crate) fn column_reads_categorical(schema: pyo3_arrow::PySchema, field: &str) -> bool {
    schema
        .into_inner()
        .field_with_name(field)
        .is_ok_and(|f| super::scale_resolve::dtype_reads_categorical(f.data_type()))
}

#[cfg(test)]
mod column_reads_categorical_tests {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;

    fn schema(dtype: DataType) -> pyo3_arrow::PySchema {
        pyo3_arrow::PySchema::new(Arc::new(Schema::new(vec![Field::new("c", dtype, true)])))
    }

    /// The published categorical reading over the dtypes a normalized schema
    /// carries: a string (including a decoded dictionary's value type) or a
    /// boolean reads categorical; a number, an all-null column (normalized to
    /// `Float64`), and an instant (a date or duration is normalized to a
    /// `Timestamp`) do not; a field the schema lacks does not.
    #[test]
    fn column_reads_categorical_publishes_the_render_reading() {
        for dtype in [DataType::Utf8, DataType::LargeUtf8, DataType::Boolean] {
            assert!(super::column_reads_categorical(schema(dtype.clone()), "c"), "{dtype:?} reads categorical");
        }
        for dtype in [
            DataType::Float64,
            DataType::Int32,
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ] {
            assert!(!super::column_reads_categorical(schema(dtype.clone()), "c"), "{dtype:?} reads numeric");
        }
        assert!(!super::column_reads_categorical(schema(DataType::Utf8), "absent"), "an absent field");
    }
}
