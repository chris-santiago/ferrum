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
}

#[pymethods]
impl ChartSpec {
    #[new]
    #[pyo3(signature = (
        *, mark, x = None, y = None, color = None,
        size = None, shape = None, opacity = None,           // NEW (from Task 3 follow-on)
        data = None, transforms = None,
        layers = None,                                        // from Task 1
        coord = None,                                         // from Task 4
        facet = None,                                         // NEW here
        mark_style = None,                                    // NEW here
    ))]
    fn new(
        mark: &str,
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        color: Option<&Bound<'_, PyAny>>,
        size: Option<&Bound<'_, PyAny>>,
        shape: Option<&Bound<'_, PyAny>>,
        opacity: Option<&Bound<'_, PyAny>>,
        data: Option<&str>,
        transforms: Option<&Bound<'_, PyAny>>,
        layers: Option<&Bound<'_, PyAny>>,
        coord: Option<&str>,
        facet: Option<&Bound<'_, PyAny>>,
        mark_style: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mark = Mark::from_str(mark)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let x = x.map(coerce_encoding).transpose()?;
        let y = y.map(coerce_encoding).transpose()?;
        let color = color.map(coerce_encoding).transpose()?;
        let size = size.map(coerce_encoding).transpose()?;
        let shape = shape.map(coerce_encoding).transpose()?;
        let opacity = opacity.map(coerce_encoding).transpose()?;

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
            Some("cartesian") => Some(crate::spec::coord::CoordKind::Cartesian),
            Some("flip") => Some(crate::spec::coord::CoordKind::Flip),
            Some(other) => return Err(PyValueError::new_err(format!(
                "unknown coord kind: '{other}'; expected 'cartesian' or 'flip'"
            ))),
        };

        let facet = match facet {
            None => None,
            Some(obj) => {
                let py = obj.py();
                let json_module = py.import("json")?;
                let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
                Some(serde_json::from_str(&s).map_err(|e|
                    PyValueError::new_err(format!("facet: {e}")))?)
            }
        };

        let mark_style = match mark_style {
            None => None,
            Some(obj) => {
                let py = obj.py();
                let json_module = py.import("json")?;
                let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
                Some(serde_json::from_str(&s).map_err(|e|
                    PyValueError::new_err(format!("mark_style: {e}")))?)
            }
        };

        Ok(ChartSpec {
            data,
            mark,
            encoding: Encoding { x, y, color, size, shape, opacity },
            transforms,
            facet,
            layers,
            coord,
            mark_style,
        })
    }

    #[getter]
    fn mark(&self) -> &'static str {
        self.mark.as_str()
    }

    #[getter]
    fn x(&self) -> Option<EncodingSpec> {
        self.encoding.x.clone()
    }

    #[getter]
    fn y(&self) -> Option<EncodingSpec> {
        self.encoding.y.clone()
    }

    #[getter]
    fn color(&self) -> Option<EncodingSpec> {
        self.encoding.color.clone()
    }

    #[getter]
    fn size(&self) -> Option<EncodingSpec> {
        self.encoding.size.clone()
    }

    #[getter]
    fn shape(&self) -> Option<EncodingSpec> {
        self.encoding.shape.clone()
    }

    #[getter]
    fn opacity(&self) -> Option<EncodingSpec> {
        self.encoding.opacity.clone()
    }

    #[getter]
    fn data(&self) -> &str {
        match &self.data {
            DataRef::Named { name } => name,
        }
    }

    #[getter]
    fn transforms(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut out: Vec<Py<PyAny>> = Vec::with_capacity(self.transforms.len());
        for t in &self.transforms {
            let obj: Py<PyAny> = match t {
                crate::transform::core::TransformSpec::Bin(_) =>
                    pyo3::Py::new(py, crate::transform::bin::PyBin(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Kde(_) =>
                    pyo3::Py::new(py, crate::transform::kde::PyKde(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Smooth(_) =>
                    pyo3::Py::new(py, crate::transform::smooth::PySmooth(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Aggregate(_) =>
                    pyo3::Py::new(py, crate::transform::aggregate::PyAggregate(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Summary(_) =>
                    pyo3::Py::new(py, crate::transform::summary::PySummary(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Outliers(_) =>
                    pyo3::Py::new(py, crate::transform::outliers::PyOutliers(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::ErrorExtent(_) =>
                    pyo3::Py::new(py, crate::transform::error_extent::PyErrorExtent(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::BoxStats(_) =>
                    pyo3::Py::new(py, crate::transform::box_stats::PyBoxStats(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Violin(_) =>
                    pyo3::Py::new(py, crate::transform::violin::PyViolin(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Kde2D(_) =>
                    pyo3::Py::new(py, crate::transform::kde_2d::PyKde2D(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Contour(_) =>
                    pyo3::Py::new(py, crate::transform::contour::PyContour(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Qq(_) =>
                    pyo3::Py::new(py, crate::transform::qq::PyQQ(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Raster(_) =>
                    pyo3::Py::new(py, crate::transform::raster::PyRaster(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Hex(_) =>
                    pyo3::Py::new(py, crate::transform::hex::PyHex(t.clone()))?.into_any(),
                crate::transform::core::TransformSpec::Swarm(_) =>
                    pyo3::Py::new(py, crate::transform::swarm::PySwarm(t.clone()))?.into_any(),
            };
            out.push(obj);
        }
        Ok(out)
    }

    #[getter]
    fn layers(&self, py: Python) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let Some(ref vec) = self.layers else { return Ok(None) };
        let mut out: Vec<Py<PyAny>> = Vec::with_capacity(vec.len());
        let json_module = py.import("json")?;
        for layer in vec {
            let s = serde_json::to_string(layer).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_obj = json_module.call_method1("loads", (s,))?;
            out.push(py_obj.unbind());
        }
        Ok(Some(out))
    }

    #[getter]
    fn coord(&self) -> Option<&'static str> {
        match self.coord {
            None => None,
            Some(CoordKind::Cartesian) => Some("cartesian"),
            Some(CoordKind::Flip) => Some("flip"),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[classmethod]
    fn from_json<'py>(_cls: &Bound<'py, PyType>, s: &str) -> PyResult<Self> {
        serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
    }

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
    let py = obj.py();
    let json_module = py.import("json")?;
    // No PyLayer class yet; deserialize Python dicts via JSON round-trip until that's added.
    let mut out = Vec::with_capacity(list.len());
    for (i, item) in list.iter().enumerate() {
        let py_dict: &Bound<PyDict> = item.downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("layers[{i}] must be a dict")))?;
        let s: String = json_module.call_method1("dumps", (py_dict,))?.extract()?;
        let layer: Layer = serde_json::from_str(&s)
            .map_err(|e| PyValueError::new_err(format!("layers[{i}]: {e}")))?;
        out.push(layer);
    }
    Ok(out)
}

fn coerce_transforms(obj: &Bound<'_, PyAny>) -> PyResult<Vec<crate::transform::core::TransformSpec>> {
    use pyo3::types::PyList;
    let list: &Bound<'_, PyList> = obj.downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("transforms must be a list"))?;
    let mut out = Vec::with_capacity(list.len());
    for (i, item) in list.iter().enumerate() {
        if let Ok(b) = item.extract::<crate::transform::bin::PyBin>() {
            out.push(b.0);
            continue;
        }
        if let Ok(k) = item.extract::<crate::transform::kde::PyKde>() {
            out.push(k.0);
            continue;
        }
        if let Ok(s) = item.extract::<crate::transform::smooth::PySmooth>() {
            out.push(s.0);
            continue;
        }
        if let Ok(a) = item.extract::<crate::transform::aggregate::PyAggregate>() {
            out.push(a.0);
            continue;
        }
        if let Ok(s) = item.extract::<crate::transform::summary::PySummary>() {
            out.push(s.0);
            continue;
        }
        if let Ok(o) = item.extract::<crate::transform::outliers::PyOutliers>() {
            out.push(o.0);
            continue;
        }
        if let Ok(e) = item.extract::<crate::transform::error_extent::PyErrorExtent>() {
            out.push(e.0);
            continue;
        }
        if let Ok(b) = item.extract::<crate::transform::box_stats::PyBoxStats>() {
            out.push(b.0);
            continue;
        }
        if let Ok(v) = item.extract::<crate::transform::violin::PyViolin>() {
            out.push(v.0);
            continue;
        }
        if let Ok(k) = item.extract::<crate::transform::kde_2d::PyKde2D>() {
            out.push(k.0);
            continue;
        }
        if let Ok(c) = item.extract::<crate::transform::contour::PyContour>() {
            out.push(c.0);
            continue;
        }
        if let Ok(q) = item.extract::<crate::transform::qq::PyQQ>() {
            out.push(q.0);
            continue;
        }
        if let Ok(r) = item.extract::<crate::transform::raster::PyRaster>() {
            out.push(r.0);
            continue;
        }
        if let Ok(h) = item.extract::<crate::transform::hex::PyHex>() {
            out.push(h.0);
            continue;
        }
        if let Ok(sw) = item.extract::<crate::transform::swarm::PySwarm>() {
            out.push(sw.0);
            continue;
        }
        return Err(PyValueError::new_err(format!(
            "transforms[{i}]: unrecognized transform; expected one of Bin | Kde | Smooth | Aggregate | Summary | Outliers | ErrorExtent | BoxStats | Violin | Kde2D | Contour | QQ | Raster | Hex | Swarm"
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
                name: None,
            })],
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
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
            Layer { mark: Mark::Point, encoding: Encoding::default(), transforms: Vec::new(), mark_style: None, data_source: None },
            Layer { mark: Mark::Line, encoding: Encoding::default(), transforms: Vec::new(), mark_style: None, data_source: None },
        ]);
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""layers":["#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
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
