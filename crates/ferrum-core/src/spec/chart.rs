use serde::{Deserialize, Serialize};

use crate::spec::coord::CoordKind;
use crate::spec::data_ref::DataRef;
use crate::spec::encoding::Encoding;
use crate::spec::layer::Layer;
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::str::FromStr;

use crate::spec::encoding::EncodingSpec;

/// Intermediate Representation for a chart.
///
/// A tree-structured configuration: top-level mark, encoding channels,
/// optional transforms, optional layers, coordinate system, theme styling.
/// Serializes to JSON via ``to_json`` and round-trips via ``from_json``.
///
/// Parameters
/// ----------
/// mark : {"point", "line", "bar", "area", "rule", "text", "tick", \
///         "rect", "polygon", "image", "ribbon", "segment"}
///     Mark kind.
/// x, y, color, size, shape, opacity, x2, y2 : EncodingSpec or str, optional
///     Encoding channels. Strings auto-wrap as ``EncodingSpec(field)``.
/// data : str, optional
///     Dataset name (referenced by ``Layer.data_source`` for multi-batch
///     specs). Defaults to the primary dataset when omitted.
/// transforms : list of transform objects, optional
///     Stat transforms applied before rendering.
/// facet : dict, optional
///     Faceting configuration.
/// layers : list of Layer, optional
///     When set, top-level ``mark`` + encodings are ignored; renderer
///     iterates the layers.
/// coord : {"cartesian", "flip"}, optional
///     Coordinate system name.
/// mark_style : dict, optional
///     Theme/style overrides applied to the mark.
/// position : dict, optional
///     Position adjustment (e.g. ``Jitter``, ``Dodge``, ``Stack``).
///
/// Notes
/// -----
/// ``ChartSpec`` is the contract between Python and Rust. Python's
/// ``Chart`` class builds a ``ChartSpec`` lazily and renders via
/// ``ferrum._core.render_svg``.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> spec = fm.ChartSpec(mark="point", x="x", y="y")
/// >>> spec.to_json()
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<crate::transform::core::TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<crate::layout::facet::FacetSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<Layer>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coord: Option<CoordKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_style: Option<MarkKwargsSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<crate::spec::position::PositionAdjust>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<crate::spec::title::TitleSpec>,
    /// When `Some(false)`, the x axis (line, ticks, tick labels, title) is
    /// suppressed entirely — used by clustermap dendrogram panels and
    /// JointChart marginal panels to keep their cells label-free. `None` /
    /// `Some(true)` both render the axis normally. Default `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis_x: Option<bool>,
    /// Y-axis variant of `axis_x`. Same semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis_y: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selections: Vec<ferrum_scene::SelectionSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditionals: Vec<ferrum_scene::ConditionalEncoding>,
}

#[pymethods]
impl ChartSpec {
    #[new]
    #[pyo3(signature = (
        *, mark, x = None, y = None, color = None,
        size = None, shape = None, opacity = None,           // NEW (from Task 3 follow-on)
        x2 = None, y2 = None,                                 // NEW Phase 8b Task 22 (ribbon)
        text = None,                                          // Phase 10c (mark_text label channel)
        tooltip = None, href = None, description = None,      // Phase 10 gallery-defaults
        tooltip_fields = None,                                // Phase 11a multi-field tooltip
        data = None, transforms = None,
        layers = None,                                        // from Task 1
        coord = None,                                         // from Task 4 (11d: accepts str or dict)
        facet = None,                                         // NEW here
        mark_style = None,                                    // NEW here
        position = None,                                      // Phase 9c
        title = None,                                         // Themes-T2.5: chart-level title
        axis_x = None,                                        // spec-level axis suppression
        axis_y = None,
        key = None,                                               // Phase 11c (transition key)
        selections = None,                                        // Phase 11c (JSON)
        conditionals = None,                                      // Phase 11c (JSON)
    ))]
    fn new(
        mark: &str,
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        color: Option<&Bound<'_, PyAny>>,
        size: Option<&Bound<'_, PyAny>>,
        shape: Option<&Bound<'_, PyAny>>,
        opacity: Option<&Bound<'_, PyAny>>,
        x2: Option<&Bound<'_, PyAny>>,
        y2: Option<&Bound<'_, PyAny>>,
        text: Option<&Bound<'_, PyAny>>,
        tooltip: Option<&Bound<'_, PyAny>>,
        href: Option<&Bound<'_, PyAny>>,
        description: Option<&Bound<'_, PyAny>>,
        tooltip_fields: Option<&str>,
        data: Option<&str>,
        transforms: Option<&Bound<'_, PyAny>>,
        layers: Option<&Bound<'_, PyAny>>,
        coord: Option<&Bound<'_, PyAny>>,
        facet: Option<&Bound<'_, PyAny>>,
        mark_style: Option<&Bound<'_, PyAny>>,
        position: Option<&Bound<'_, PyAny>>,
        title: Option<&Bound<'_, PyAny>>,
        axis_x: Option<bool>,
        axis_y: Option<bool>,
        key: Option<&Bound<'_, PyAny>>,
        selections: Option<&str>,
        conditionals: Option<&str>,
    ) -> PyResult<Self> {
        let mark = Mark::from_str(mark)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let x = x.map(coerce_encoding).transpose()?;
        let y = y.map(coerce_encoding).transpose()?;
        let color = color.map(coerce_encoding).transpose()?;
        let size = size.map(coerce_encoding).transpose()?;
        let shape = shape.map(coerce_encoding).transpose()?;
        let opacity = opacity.map(coerce_encoding).transpose()?;
        let x2 = x2.map(coerce_encoding).transpose()?;
        let y2 = y2.map(coerce_encoding).transpose()?;
        let text = text.map(coerce_encoding).transpose()?;
        let tooltip = tooltip.map(coerce_encoding).transpose()?;
        let href = href.map(coerce_encoding).transpose()?;
        let description = description.map(coerce_encoding).transpose()?;
        let tooltip_fields: Option<Vec<crate::spec::encoding::EncodingSpec>> = tooltip_fields
            .map(|s| serde_json::from_str(s)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("tooltip_fields JSON: {e}"))))
            .transpose()?;
        let key = key.map(coerce_encoding).transpose()?;

        let data = match data {
            None => DataRef::default(),
            Some(name) if name.is_empty() => {
                return Err(PyValueError::new_err("data name must be non-empty"))
            }
            Some(name) => DataRef::Named { name: name.to_string() },
        };

        let transforms = match transforms {
            None => Vec::new(),
            Some(obj) => coerce_transforms(obj)?,
        };

        let layers = match layers {
            None => None,
            Some(obj) => Some(coerce_layers(obj)?),
        };

        let coord = match coord {
            None => None,
            Some(obj) => {
                // Accept a plain string ("flip", "cartesian") for back-compat,
                // or a dict produced by CoordXxx._to_spec_dict() for new coord types.
                if let Ok(s) = obj.extract::<String>() {
                    Some(match s.as_str() {
                        "cartesian" => crate::spec::coord::CoordKind::Cartesian {
                            x_domain: None,
                            y_domain: None,
                            expand: true,
                            clip: true,
                        },
                        "flip" => crate::spec::coord::CoordKind::Flip,
                        other => return Err(PyValueError::new_err(format!(
                            "unknown coord kind: '{other}'; expected 'cartesian', 'flip', \
                             'fixed', 'polar', or 'geo'"
                        ))),
                    })
                } else {
                    Some(crate::pyo3_serde::from_py(obj, "coord")?)
                }
            }
        };

        let facet = facet
            .map(|obj| crate::pyo3_serde::from_py(obj, "facet"))
            .transpose()?;
        let mark_style = mark_style
            .map(|obj| crate::pyo3_serde::from_py(obj, "mark_style"))
            .transpose()?;
        let position = position
            .map(|obj| crate::pyo3_serde::from_py(obj, "position"))
            .transpose()?;

        // Schwabish SB1 (2026-05-11): accept ``title=`` as either a plain
        // string (back-compat) or a dict shaped by ``Title.to_spec_dict()``.
        // Strings widen to ``TitleSpec { text }`` with default anchor; dicts
        // round-trip through ``pyo3_serde::from_py``.
        let title = match title {
            None => None,
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    Some(crate::spec::title::TitleSpec {
                        text: s,
                        ..Default::default()
                    })
                } else {
                    Some(crate::pyo3_serde::from_py(obj, "title")?)
                }
            }
        };

        let selections = match selections {
            None => Vec::new(),
            Some(s) => serde_json::from_str(s)
                .map_err(|e| PyValueError::new_err(format!("selections: {e}")))?,
        };
        let conditionals = match conditionals {
            None => Vec::new(),
            Some(s) => serde_json::from_str(s)
                .map_err(|e| PyValueError::new_err(format!("conditionals: {e}")))?,
        };

        Ok(ChartSpec {
            data,
            mark,
            encoding: Encoding { x, y, color, size, shape, opacity, x2, y2, text, tooltip, tooltip_fields, href, description, key },
            transforms,
            facet,
            layers,
            coord,
            mark_style,
            position,
            title,
            axis_x,
            axis_y,
            selections,
            conditionals,
        })
    }

    /// Whether the x axis is hidden via spec-level suppression.
    #[getter]
    fn axis_x(&self) -> Option<bool> {
        self.axis_x
    }

    /// Whether the y axis is hidden via spec-level suppression.
    #[getter]
    fn axis_y(&self) -> Option<bool> {
        self.axis_y
    }

    /// Mark kind string (e.g. ``"point"``, ``"bar"``).
    #[getter]
    fn mark(&self) -> &'static str {
        self.mark.as_str()
    }

    /// X encoding channel, or ``None`` if unset.
    #[getter]
    fn x(&self) -> Option<EncodingSpec> {
        self.encoding.x.clone()
    }

    /// Y encoding channel, or ``None`` if unset.
    #[getter]
    fn y(&self) -> Option<EncodingSpec> {
        self.encoding.y.clone()
    }

    /// Color encoding channel, or ``None`` if unset.
    #[getter]
    fn color(&self) -> Option<EncodingSpec> {
        self.encoding.color.clone()
    }

    /// Size encoding channel, or ``None`` if unset.
    #[getter]
    fn size(&self) -> Option<EncodingSpec> {
        self.encoding.size.clone()
    }

    /// Shape encoding channel, or ``None`` if unset.
    #[getter]
    fn shape(&self) -> Option<EncodingSpec> {
        self.encoding.shape.clone()
    }

    /// Opacity encoding channel, or ``None`` if unset.
    #[getter]
    fn opacity(&self) -> Option<EncodingSpec> {
        self.encoding.opacity.clone()
    }

    /// Secondary X endpoint channel (ribbon/errorbar), or ``None`` if unset.
    #[getter]
    fn x2(&self) -> Option<EncodingSpec> {
        self.encoding.x2.clone()
    }

    /// Secondary Y endpoint channel (ribbon/errorbar), or ``None`` if unset.
    #[getter]
    fn y2(&self) -> Option<EncodingSpec> {
        self.encoding.y2.clone()
    }

    /// Dataset name; ``"default"`` when the primary dataset is used.
    #[getter]
    fn data(&self) -> &str {
        match &self.data {
            DataRef::Named { name } => name,
        }
    }

    /// Stat transforms applied before rendering, as a list of transform objects.
    #[getter]
    fn transforms(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        use crate::transform::core::{for_each_transform, TransformSpec};
        let mut out: Vec<Py<PyAny>> = Vec::with_capacity(self.transforms.len());
        for t in &self.transforms {
            macro_rules! arm {
                ($($V:ident => $m:ident : $py:ident,)*) => {
                    match t { $(
                        TransformSpec::$V(_) =>
                            pyo3::Py::new(py, crate::transform::$m::$py(t.clone()))?.into_any(),
                    )* }
                };
            }
            out.push(for_each_transform!(arm));
        }
        Ok(out)
    }

    /// Layers as a list of dicts, or ``None`` for single-layer specs.
    #[getter]
    fn layers(&self, py: Python) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let Some(ref vec) = self.layers else { return Ok(None) };
        vec.iter()
            .map(|layer| crate::pyo3_serde::to_py(py, layer))
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    /// Coordinate system dict (serialized ``CoordKind``), or ``None``.
    #[getter]
    fn coord(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.coord
            .as_ref()
            .map(|c| crate::pyo3_serde::to_py(py, c))
            .transpose()
    }

    /// Position adjustment dict (Jitter, Dodge, Stack, ...), or ``None``.
    #[getter]
    fn position(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.position
            .as_ref()
            .map(|p| crate::pyo3_serde::to_py(py, p))
            .transpose()
    }

    /// Serialize this spec to its canonical JSON form.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON-encoded ``ChartSpec``.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reconstruct a ``ChartSpec`` from JSON.
    ///
    /// Parameters
    /// ----------
    /// s : str
    ///     JSON string produced by ``to_json``.
    ///
    /// Returns
    /// -------
    /// ChartSpec
    ///     Reconstructed spec; ``s == ChartSpec.from_json(s.to_json())``.
    #[classmethod]
    fn from_json<'py>(_cls: &Bound<'py, PyType>, s: &str) -> PyResult<Self> {
        serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return a string representation of this spec.
    fn __repr__(&self) -> String {
        let mark = self.mark.as_str();
        let data = match &self.data {
            DataRef::Named { name } => name.as_str(),
        };
        let x = match &self.encoding.x {
            None => "None".to_string(),
            Some(e) => e.repr_string(),
        };
        let y = match &self.encoding.y {
            None => "None".to_string(),
            Some(e) => e.repr_string(),
        };
        if self.transforms.is_empty() {
            format!("ChartSpec(mark='{mark}', x={x}, y={y}, data='{data}')")
        } else {
            format!(
                "ChartSpec(mark='{mark}', x={x}, y={y}, data='{data}', transforms=[{} item(s)])",
                self.transforms.len()
            )
        }
    }
}

fn coerce_encoding(obj: &Bound<'_, PyAny>) -> PyResult<EncodingSpec> {
    if let Ok(s) = obj.extract::<String>() {
        if s.is_empty() {
            return Err(PyValueError::new_err("encoding field name must be non-empty"));
        }
        return Ok(EncodingSpec { field: s, ..Default::default() });
    }
    if let Ok(spec) = obj.extract::<EncodingSpec>() {
        return Ok(spec);
    }
    Err(PyTypeError::new_err(
        "expected str or EncodingSpec for encoding channel",
    ))
}

fn coerce_layers(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Layer>> {
    use pyo3::types::{PyDict, PyList};
    let list: &Bound<'_, PyList> = obj.downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("layers must be a list"))?;
    // No PyLayer class yet; deserialize Python dicts via the shared
    // pyo3_serde helper until one is added.
    let mut out = Vec::with_capacity(list.len());
    for (i, item) in list.iter().enumerate() {
        let _: &Bound<PyDict> = item.downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("layers[{i}] must be a dict")))?;
        let layer: Layer = crate::pyo3_serde::from_py(&item, &format!("layers[{i}]"))?;
        out.push(layer);
    }
    Ok(out)
}

fn coerce_transforms(obj: &Bound<'_, PyAny>) -> PyResult<Vec<crate::transform::core::TransformSpec>> {
    use pyo3::types::PyList;
    use crate::transform::core::for_each_transform;
    let list: &Bound<'_, PyList> = obj.downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("transforms must be a list"))?;
    let mut out = Vec::with_capacity(list.len());
    'next_item: for (i, item) in list.iter().enumerate() {
        macro_rules! try_extract {
            ($($V:ident => $m:ident : $py:ident,)*) => {{
                $(
                    if let Ok(w) = item.extract::<crate::transform::$m::$py>() {
                        out.push(w.0);
                        continue 'next_item;
                    }
                )*
            }};
        }
        for_each_transform!(try_extract);
        return Err(PyValueError::new_err(format!(
            "transforms[{i}]: unrecognized transform; expected one of \
             Bin | Bin2D | Kde | Smooth | Aggregate | Summary | Outliers | \
             ErrorExtent | BoxStats | Violin | Kde2D | Contour | QQ | Linkage | \
             Raster | Hex | Swarm | Unpivot | Reorder | ReferenceLine | \
             LetterValue | Logistic | Glm | Robust"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::{DataType, EncodingSpec};

    fn minimal_scatter() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: Some(DataType::Quantitative),
                    ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        }
    }

    #[test]
    fn test_chart_spec_round_trip_minimal() {
        let original = minimal_scatter();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_chart_spec_round_trip_idempotent_json() {
        let original = minimal_scatter();
        let json1 = serde_json::to_string(&original).unwrap();
        let parsed: ChartSpec = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2, "two-pass JSON differed");
    }

    #[test]
    fn test_chart_spec_round_trip_each_mark_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect,
            Mark::Polygon, Mark::Image, Mark::Ribbon, Mark::Segment,
        ] {
            let mut spec = minimal_scatter();
            spec.mark = m;
            let json = serde_json::to_string(&spec).unwrap();
            let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, spec, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_data_ref_defaults_when_omitted() {
        let json = r#"{"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data, DataRef::Named { name: "default".into() });
    }

    #[test]
    fn test_unknown_mark_in_json_errors() {
        let json = r#"{"data":{"kind":"named","name":"d"},"mark":"spaghetti","encoding":{}}"#;
        let err = serde_json::from_str::<ChartSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("spaghetti") || msg.contains("variant"), "msg: {msg}");
    }

    #[test]
    fn test_missing_required_field_errors() {
        let json = r#"{"encoding":{}}"#;
        let err = serde_json::from_str::<ChartSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mark"), "expected 'mark' in error, got: {msg}");
    }

    #[test]
    fn test_unknown_field_silently_dropped() {
        let json = r#"{"mark":"point","encoding":{},"future_field":42}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.mark, Mark::Point);
    }

    #[test]
    fn test_canonical_json_shape() {
        let spec = ChartSpec {
            data: DataRef::Named { name: "default".into() },
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: Some(DataType::Quantitative),
                    ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            json,
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}}"#,
        );
    }

    #[test]
    fn test_chart_spec_transforms_default_when_omitted() {
        // Phase 3 JSON shape (no `transforms` field) must still deserialize.
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert!(parsed.transforms.is_empty(), "expected empty transforms by default");
    }

    #[test]
    fn test_chart_spec_transforms_omitted_in_canonical_json_when_empty() {
        let spec = ChartSpec {
            data: DataRef::Named { name: "default".into() },
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("transforms"), "empty transforms should be skipped: {json}");
    }

    #[test]
    fn test_chart_spec_transforms_round_trip_with_one_bin() {
        use crate::transform::bin::BinSpec;
        use crate::transform::core::TransformSpec;
        let spec = ChartSpec {
            data: DataRef::Named { name: "default".into() },
            mark: Mark::Bar,
            encoding: Encoding::default(),
            transforms: vec![TransformSpec::Bin(BinSpec {
                field: "x".into(),
                bin_count: Some(10),
                bin_width: None,
                extent: None,
                nice: true,
                cumulative: false,
                groupby: None,
                name: None,
            })],
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""transforms":["#), "should include transforms array: {json}");
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn test_chart_spec_facet_default_when_omitted() {
        // Pre-Phase-6 JSON shape (no `facet` field) must still deserialize.
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert!(parsed.facet.is_none());
    }

    #[test]
    fn test_chart_spec_facet_omitted_in_canonical_json_when_none() {
        let spec = minimal_scatter();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("facet"), "facet=None should be skipped: {json}");
    }

    #[test]
    fn test_chart_spec_facet_round_trip() {
        use crate::layout::facet::{FacetMode, FacetSpec};
        let mut spec = minimal_scatter();
        spec.facet = Some(FacetSpec {
            field: "species".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 3 },
            spacing: None,
        });
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""facet":{"#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn test_chart_spec_layers_default_when_omitted() {
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert!(parsed.layers.is_none());
    }

    #[test]
    fn test_chart_spec_layers_omitted_in_canonical_json_when_none() {
        let spec = minimal_scatter();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("layers"), "layers=None should be skipped: {json}");
    }

    #[test]
    fn test_chart_spec_layers_round_trip() {
        use crate::spec::layer::Layer;
        let mut spec = minimal_scatter();
        spec.layers = Some(vec![
            Layer { mark: Mark::Point, encoding: Encoding::default(), transforms: Vec::new(), mark_style: None, data_source: None, position: None, blend: None },
            Layer { mark: Mark::Line, encoding: Encoding::default(), transforms: Vec::new(), mark_style: None, data_source: None, position: None, blend: None },
        ]);
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""layers":["#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn chart_spec_position_round_trips() {
        use crate::spec::position::PositionAdjust;
        let mut spec = minimal_scatter();
        spec.position = Some(PositionAdjust::Identity);
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""position""#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn chart_spec_position_none_omits_from_json() {
        let spec = minimal_scatter();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains(r#""position""#), "position=None must be omitted: {json}");
    }

    #[test]
    fn test_existing_phase_7_canonical_json_unchanged() {
        // Phase 3-7 byte-identical JSON shape when layers.is_none()
        let spec = minimal_scatter();
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            json,
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}}"#,
        );
    }

    #[test]
    fn test_chart_spec_coord_default_when_omitted() {
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert!(parsed.coord.is_none());
    }

    #[test]
    fn test_chart_spec_coord_omitted_in_canonical_json_when_none() {
        let spec = minimal_scatter();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("coord"));
    }

    #[test]
    fn test_chart_spec_coord_flip_round_trip() {
        use crate::spec::coord::CoordKind;
        let mut spec = minimal_scatter();
        spec.coord = Some(CoordKind::Flip);
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""coord":{"kind":"flip"}"#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn test_chart_spec_coord_cartesian_with_xlim_round_trip() {
        use crate::spec::coord::CoordKind;
        let mut spec = minimal_scatter();
        spec.coord = Some(CoordKind::Cartesian { x_domain: Some((0.0, 100.0)), y_domain: None, expand: true, clip: true });
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""kind":"cartesian""#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn test_chart_spec_coord_geo_round_trip() {
        use crate::spec::coord::{CoordKind, GeoProjection};
        let mut spec = minimal_scatter();
        spec.coord = Some(CoordKind::Geo { projection: GeoProjection::Mercator });
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""kind":"geo""#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn test_chart_spec_mark_style_default_when_omitted() {
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert!(parsed.mark_style.is_none());
    }

    #[test]
    fn test_chart_spec_mark_style_round_trip() {
        use crate::spec::mark_style::MarkKwargsSpec;
        let mut spec = minimal_scatter();
        spec.mark_style = Some(MarkKwargsSpec {
            size: Some(100.0),
            stroke: Some("#ff0000".into()),
            ..Default::default()
        });
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r##""mark_style":{"size":100.0,"stroke":"#ff0000""##));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }
}
