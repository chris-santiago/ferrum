//! Swarm: greedy beeswarm placement.
//!
//! Placement runs in two stages. This transform validates the columns and
//! passes the rows through; the dot offsets are decided at render time
//! by [`place_offsets`], which packs each category in PIXEL space through the
//! resolved panel scale (called from `render::position::apply_swarm`, the seam
//! where `Jitter` computes its band-pixel offsets). Only there are the plot
//! rect, the scale's end insets, a facet-shared or explicit domain and the
//! scale kind all known, so only there can neighbouring centres sit exactly
//! `point_size + spacing` px apart on every layout. There too, a point layer
//! with no size of its own takes the dot diameter `point_size`
//! ([`SwarmSpec::dot_area`]), so the drawn dots match their placement.
//!
//! Output schema: the input schema, unchanged. The cross-axis offset is a
//! pixel quantity of the rendered panel with no data-unit form before layout,
//! so the transform emits none:
//! render time adds the `__pos_x_offset__` (vertical orient) or
//! `__pos_y_offset__` (horizontal orient) column of placed pixel offsets,
//! which `render/position.rs` adds to the category band centre. Placement
//! packs whichever rows the layer draws, so rows a later transform filtered
//! or aggregated are packed as they are.
//!
//! With `dodge`, rows whose dodge value is null are dropped (they belong to no
//! sub-band).
//!
//! Packing: per category, every dodge group packs against one shared placed
//! set, so no two dots of a category overlap whichever group they belong to.
//! Within a group, points are sorted by `value` ascending (stable on row
//! index, so placements are byte-deterministic). Each point tries cross-axis
//! offsets in the order `[0, +d, -d, +2d, -2d, ...]` (or one side only) with
//! `d = point_size + spacing` px, centred on its dodge group's position, and
//! takes the first whose centre is at least `d` px from every already-placed
//! centre in the same category: dots of diameter `point_size` whose edges are
//! `spacing` px apart. Dodge groups sit side by side, their centres `d` px
//! apart.

use arrow::array::{Array, RecordBatch, StringArray};
use arrow::datatypes::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::transform::group_key::{groupby_key_at, is_groupby_supported_dtype, KeyValue};
use crate::transform::numeric_util::coerce_to_float64;

fn default_point_size() -> f64 {
    4.0
}
fn default_spacing() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SwarmSide {
    #[default]
    Both,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SwarmOrient {
    #[default]
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SwarmSpec {
    pub category: String, // grouping field (Utf8/LargeUtf8, Float, integer, or Boolean)
    pub value: String,    // Float64
    #[serde(default = "default_point_size")]
    pub point_size: f64,
    #[serde(default = "default_spacing")]
    pub spacing: f64,
    #[serde(default)]
    pub side: SwarmSide,
    #[serde(default)]
    pub orient: SwarmOrient,
    /// When set, partition within each category by this field and offset each
    /// sub-group's swarm so the groups appear side by side within the band.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dodge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

impl SwarmSpec {
    /// The pixel step between neighbouring centres: one dot diameter plus the
    /// edge gap.
    fn step_px(&self) -> f64 {
        self.point_size + self.spacing
    }

    /// The area (px²) of a point mark `point_size` px across: the size a
    /// swarm-fed point layer with no size of its own draws its dots at.
    pub(crate) fn dot_area(&self) -> f64 {
        std::f64::consts::PI * (self.point_size / 2.0).powi(2)
    }

}

pub(crate) fn apply(spec: &SwarmSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let filtered;
    let batch = match &spec.dodge {
        Some(dodge_field) => {
            filtered = drop_null_dodge_rows(batch, dodge_field)?;
            &filtered
        }
        None => batch,
    };
    let schema = batch.schema();
    let cat_idx = schema
        .index_of(&spec.category)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_swarm: column '{}' not found", spec.category
        )))?;
    let cat_dt = schema.field(cat_idx).data_type().clone();
    if !is_groupby_supported_dtype(&cat_dt) {
        return Err(PyValueError::new_err(format!(
            "stat_swarm: category column '{}' has unsupported dtype {:?}; \
             supported: Utf8/LargeUtf8, Float64/Float32, \
             Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean",
            spec.category, cat_dt
        )));
    }
    let val_idx = schema
        .index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_swarm: column '{}' not found", spec.value
        )))?;
    coerce_to_float64(batch.column(val_idx), "stat_swarm", &spec.value)?;
    Ok(batch.clone())
}

/// Validate the dodge column and drop the rows it leaves in no sub-band.
fn drop_null_dodge_rows(batch: &RecordBatch, dodge_field: &str) -> PyResult<RecordBatch> {
    let col = batch.column_by_name(dodge_field).ok_or_else(|| {
        PyValueError::new_err(format!("stat_swarm: dodge column '{}' not found", dodge_field))
    })?;
    if col.data_type() != &DataType::Utf8 {
        return Err(PyValueError::new_err(format!(
            "stat_swarm: dodge column '{}' must be Utf8", dodge_field
        )));
    }
    let keep = arrow::compute::is_not_null(col.as_ref())
        .map_err(|e| PyValueError::new_err(format!("stat_swarm dodge filter: {e}")))?;
    arrow::compute::filter_record_batch(batch, &keep)
        .map_err(|e| PyValueError::new_err(format!("stat_swarm dodge filter: {e}")))
}

/// Place every row of a swarm layer's batch: the cross-axis pixel offset of
/// each row from its category band centre.
///
/// `values` is the value column read per row; `to_px` maps a value to its
/// pixel coordinate on the rendered value axis (the resolved panel scale).
/// Rows with no value, no category, or a value the scale cannot place get
/// offset `0.0`.
///
/// With `dodge`, dodge group `i` of `n` (sorted by name) is centred
/// `(i - (n - 1) / 2) * (point_size + spacing)` px from the band centre and
/// packs outward from there, trying its own side of the band first. All the
/// groups of one category pack against the same placed dots, so no two dots of
/// a category overlap whichever group they belong to.
pub(crate) fn place_offsets(
    spec: &SwarmSpec,
    batch: &RecordBatch,
    values: &[Option<f64>],
    to_px: impl Fn(f64) -> Option<f64>,
) -> Result<Vec<f64>, String> {
    let n = batch.num_rows();
    let cat_col = batch
        .column_by_name(&spec.category)
        .ok_or_else(|| format!("category column '{}' not found", spec.category))?;
    let cat_dt = cat_col.data_type().clone();
    let dodge: Option<&StringArray> = match &spec.dodge {
        Some(field) => Some(
            batch
                .column_by_name(field)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| format!("dodge column '{field}' is not a Utf8 column of this batch"))?,
        ),
        None => None,
    };
    let dodge_of = |i: usize| dodge.filter(|a| !a.is_null(i)).map(|a| a.value(i));
    let dodge_groups: Vec<&str> = (0..n).filter_map(dodge_of).collect::<BTreeSet<_>>().into_iter().collect();
    let n_groups = dodge_groups.len().max(1) as f64;
    let step = spec.step_px();
    let centre_px = |name: Option<&str>| -> f64 {
        name.and_then(|name| dodge_groups.iter().position(|g| *g == name))
            .map_or(0.0, |gi| (gi as f64 - (n_groups - 1.0) / 2.0) * step)
    };

    // category -> dodge group -> [(row, value, value_px)]. BTreeMaps for a
    // deterministic order: categories and dodge groups pack in sorted order.
    type Entries = Vec<(usize, f64, f64)>;
    let mut groups: BTreeMap<KeyValue, BTreeMap<Option<&str>, Entries>> = BTreeMap::new();
    for (i, v) in values.iter().enumerate().take(n) {
        let Some(v) = v.filter(|v| !v.is_nan()) else { continue };
        let key = match groupby_key_at(cat_col.as_ref(), &cat_dt, i) {
            Some(KeyValue::Null) | None => continue,
            Some(kv) => kv,
        };
        let Some(px) = to_px(v) else { continue };
        groups.entry(key).or_default().entry(dodge_of(i)).or_default().push((i, v, px));
    }

    let mut offsets = vec![0.0; n];
    for by_dodge in groups.into_values() {
        let mut placed: Vec<(f64, f64)> = Vec::new();
        for (dodge_group, mut entries) in by_dodge {
            // Sort by value ascending; tiebreak by row index (stable, deterministic).
            entries.sort_by(|(ia, va, _), (ib, vb, _)| {
                va.partial_cmp(vb).unwrap_or(Ordering::Equal).then_with(|| ia.cmp(ib))
            });
            let centre = centre_px(dodge_group);
            for (row, _, px) in entries {
                offsets[row] = pack_one(&mut placed, px, centre, step, spec.side);
            }
        }
    }
    Ok(offsets)
}

/// Greedy beeswarm placement of one dot in pixel space.
///
/// Takes the first cross-axis offset in `centre + [0, +step, -step, +2 step,
/// ...]` (restricted to `side`; `-step` first when `centre` is left of the
/// band centre) whose centre is at least `step` px from every dot in `placed`,
/// records it there, and returns it.
fn pack_one(placed: &mut Vec<(f64, f64)>, value_px: f64, centre: f64, step: f64, side: SwarmSide) -> f64 {
    let min_dist_sq = step * step;
    let free = |cand: f64| {
        placed.iter().all(|&(po, pv)| {
            let dx = cand - po;
            let dy = value_px - pv;
            dx * dx + dy * dy >= min_dist_sq
        })
    };
    // Every placed dot blocks at most two slots of this lattice row, so
    // `2 * placed + 1` steps (doubled for a one-sided swarm) always reach a
    // free one.
    let max_steps = 4 * placed.len() + 4;
    let outward = if centre < 0.0 { -1.0 } else { 1.0 };
    let candidates = (0..=max_steps).flat_map(|k| {
        let d = k as f64 * step;
        let pair: [Option<f64>; 2] = match (k, side) {
            (0, _) => [Some(0.0), None],
            (_, SwarmSide::Both) => [Some(outward * d), Some(-outward * d)],
            (_, SwarmSide::Left) => [Some(-d), None],
            (_, SwarmSide::Right) => [Some(d), None],
        };
        pair.into_iter().flatten().map(move |off| centre + off)
    });
    // The bound above always admits a free slot; the sentinel keeps the
    // side invariant if it ever did not.
    let fallback = centre
        + match side {
            SwarmSide::Left => -((max_steps as f64) * step),
            SwarmSide::Right | SwarmSide::Both => (max_steps as f64) * step,
        };
    let off = candidates.into_iter().find(|&c| free(c)).unwrap_or(fallback);
    placed.push((off, value_px));
    off
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Beeswarm (strip-plot) offset computation for non-overlapping point layout.
///
/// Takes a categorical-axis column and a numeric value column and computes
/// per-point x-offsets (for a vertical beeswarm) so that circles of radius
/// ``point_size / 2`` do not overlap. Points are placed symmetrically
/// around the category centre, or on one side only when ``side`` is set.
///
/// Output: the input columns, unchanged. The dots are
/// placed when the chart renders, in pixels through the resolved panel scale,
/// so neighbouring centres sit ``point_size + spacing`` px apart. A point layer drawing this output with
/// no size of its own (no ``size=`` and no ``size`` encoding) draws each dot
/// ``point_size`` px across, so dot edges sit ``spacing`` px apart.
///
/// Parameters
/// ----------
/// category : str
///     Categorical column that defines the swarm axis position.
/// value : str
///     Numeric column whose values are spread along the value axis.
/// point_size : float, default 4.0
///     Diameter of each point in pixels. Must be > 0.
/// spacing : float, default 1.0
///     Minimum gap between point edges, in pixels. Must be ≥ 0.
/// side : {"both", "left", "right"}, default "both"
///     Which side of the category centre to place displaced points.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_swarm().encode(x="val", y="category")
#[pyclass(eq, module = "ferrum._core", name = "Swarm")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PySwarm(pub(crate) TransformSpec);

#[pymethods]
impl PySwarm {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        category, value, *,
        point_size = 4.0,
        spacing = 1.0,
        side = "both",
        orient = "vertical",
        dodge = None,
        name = None,
    ))]
    fn new(
        category: &str,
        value: &str,
        point_size: f64,
        spacing: f64,
        side: &str,
        orient: &str,
        dodge: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if category.is_empty() || value.is_empty() {
            return Err(PyValueError::new_err(
                "Swarm: category and value fields must be non-empty",
            ));
        }
        if !point_size.is_finite() || point_size <= 0.0 {
            return Err(PyValueError::new_err(
                "Swarm: point_size must be a positive finite number",
            ));
        }
        if !spacing.is_finite() || spacing < 0.0 {
            return Err(PyValueError::new_err(
                "Swarm: spacing must be a non-negative finite number",
            ));
        }
        let parsed_side = match side {
            "both" => SwarmSide::Both,
            "left" => SwarmSide::Left,
            "right" => SwarmSide::Right,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Swarm: unknown side '{other}'; expected both|left|right"
                )))
            }
        };
        let parsed_orient = match orient {
            "vertical" => SwarmOrient::Vertical,
            "horizontal" => SwarmOrient::Horizontal,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Swarm: unknown orient '{other}'; expected vertical|horizontal"
                )))
            }
        };
        Ok(PySwarm(TransformSpec::Swarm(SwarmSpec {
            category: category.to_string(),
            value: value.to_string(),
            point_size,
            spacing,
            side: parsed_side,
            orient: parsed_orient,
            dodge,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Swarm(s) => format!(
                "Swarm(category='{}', value='{}', point_size={}, spacing={}, side={:?}, orient={:?}, dodge={:?}, name={:?})",
                s.category, s.value, s.point_size, s.spacing, s.side, s.orient, s.dodge, s.name
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(cats: Vec<&str>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    fn spec(point_size: f64, spacing: f64, side: SwarmSide) -> SwarmSpec {
        SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size,
            spacing,
            side,
            orient: SwarmOrient::Vertical,
            name: None,
            dodge: None,
        }
    }

    fn values(batch: &RecordBatch) -> Vec<Option<f64>> {
        crate::render::arrow_cast::col_as_f64(batch, "v").unwrap()
    }

    /// Place with the value axis mapped 1 data unit -> `px_per_unit` px.
    fn place(spec: &SwarmSpec, batch: &RecordBatch, px_per_unit: f64) -> Vec<f64> {
        place_offsets(spec, batch, &values(batch), |v| Some(v * px_per_unit)).unwrap()
    }

    /// Seeded standard-normal draws (Box-Muller over ChaCha8), scaled.
    fn normal_draws(n: usize, mean: f64, sd: f64) -> Vec<f64> {
        use rand::{RngCore, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut unit = || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (0..n)
            .map(|_| {
                let (u1, u2) = (unit(), unit());
                mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect()
    }

    fn min_pairwise(points: &[(f64, f64)]) -> f64 {
        let mut best = f64::INFINITY;
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                best = best.min(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt());
            }
        }
        best
    }

    /// The wire default and the constructor default are one value, the one
    /// `mark_swarm` draws its dots at, so a `Swarm` transform feeding
    /// `mark_point` and `mark_swarm` place and draw the same dots.
    #[test]
    fn swarm_point_size_defaults_to_four_on_the_wire_and_in_the_constructor() {
        let wire: SwarmSpec = serde_json::from_value(serde_json::json!({"category": "c", "value": "v"})).unwrap();
        assert_eq!(wire.point_size, 4.0);
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let built = py.get_type::<PySwarm>().call1(("c", "v")).unwrap();
            let built = built.extract::<PySwarm>().unwrap();
            let TransformSpec::Swarm(s) = built.0 else { panic!("Swarm builds a Swarm spec") };
            assert_eq!(s.point_size, 4.0);
        });
    }

    #[test]
    fn swarm_side_both_alternates_around_zero() {
        pyo3::Python::initialize();
        let b = make_batch(vec!["A"; 5], vec![1.0; 5]);
        let offsets = place(&spec(5.0, 1.0, SwarmSide::Both), &b, 1.0);
        let mut sorted = offsets.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![-12.0, -6.0, 0.0, 6.0, 12.0]);
    }

    #[test]
    fn swarm_side_left_and_right_stay_on_their_side() {
        pyo3::Python::initialize();
        let b = make_batch(vec!["A"; 5], vec![1.0; 5]);
        let left = place(&spec(5.0, 1.0, SwarmSide::Left), &b, 1.0);
        let right = place(&spec(5.0, 1.0, SwarmSide::Right), &b, 1.0);
        assert_eq!(left, vec![0.0, -6.0, -12.0, -18.0, -24.0]);
        assert_eq!(right, vec![0.0, 6.0, 12.0, 18.0, 24.0]);
    }

    #[test]
    fn swarm_deterministic_tiebreak() {
        pyo3::Python::initialize();
        let b = make_batch(vec!["A", "A", "A", "B", "B", "A"], vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0]);
        let s = spec(5.0, 1.0, SwarmSide::Both);
        let o1 = place(&s, &b, 1.0);
        let o2 = place(&s, &b, 1.0);
        assert_eq!(o1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   o2.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        // Categories pack independently: B's two equal values take 0 and +6.
        assert_eq!((o1[3], o1[4]), (0.0, 6.0));
    }

    /// On a spread column, no two placed centres in a category are closer than
    /// `size + spacing` px in the pixel space `to_px` defines, and a dense run
    /// is packed at exactly that distance.
    #[test]
    fn swarm_no_overlapping_placements_on_a_spread_column() {
        pyo3::Python::initialize();
        let vals = normal_draws(300, 50.0, 15.0);
        let b = make_batch(vec!["A"; vals.len()], vals.clone());
        for (size, spacing) in [(8.0, 0.0), (4.0, 1.0), (5.0, 2.0)] {
            let s = spec(size, spacing, SwarmSide::Both);
            let px_per_unit = 4.2;
            let offsets = place(&s, &b, px_per_unit);
            let pts: Vec<(f64, f64)> =
                offsets.iter().zip(&vals).map(|(&o, &v)| (o, v * px_per_unit)).collect();
            let md = min_pairwise(&pts);
            assert!(md >= size + spacing - 1e-9, "size={size} spacing={spacing}: min centre distance {md}");
            assert!(md < size + spacing + 0.5, "size={size} spacing={spacing}: packing too loose ({md})");
        }
    }

    #[test]
    fn swarm_dodge_sub_band_spacing_equals_point_size_plus_spacing() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "A"])),
                Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0])),
                Arc::new(StringArray::from(vec![Some("g2"), Some("g1"), Some("g3")])),
            ],
        )
        .unwrap();
        let s = SwarmSpec { dodge: Some("grp".into()), ..spec(7.0, 2.0, SwarmSide::Both) };
        // One point per group: each offset is its group's centre, sorted by name.
        assert_eq!(place(&s, &b, 1.0), vec![0.0, -9.0, 9.0]);
    }

    /// Tied values in two dodge groups: each group packs outward from its own
    /// sub-band, so the left group's second dot steps further left rather
    /// than into the right group's sub-band (which it packs first, while
    /// that slot is still free), and the right group's steps further right.
    #[test]
    fn a_dodge_group_left_of_centre_packs_outward_to_the_left_first() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("grp", DataType::Utf8, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A"; 4])),
                Arc::new(Float64Array::from(vec![5.0; 4])),
                Arc::new(StringArray::from(vec![Some("g1"), Some("g1"), Some("g2"), Some("g2")])),
            ],
        )
        .unwrap();
        let s = SwarmSpec { dodge: Some("grp".into()), ..spec(7.0, 2.0, SwarmSide::Both) };
        // Sub-band centres sit at -4.5 and +4.5 (step 9 apart).
        assert_eq!(place(&s, &b, 1.0), vec![-4.5, -13.5, 4.5, 13.5]);
    }

    #[test]
    fn swarm_transform_passes_its_input_through_and_drops_null_dodge_rows() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
            Field::new("grp", DataType::Utf8, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("A"), None, Some("A"), Some("A")])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None, Some(4.0)])),
                Arc::new(StringArray::from(vec![Some("g"), Some("g"), Some("g"), None])),
            ],
        )
        .unwrap();
        let s = SwarmSpec { dodge: Some("grp".into()), ..spec(5.0, 1.0, SwarmSide::Both) };
        let out = apply(&s, &b).unwrap();
        assert_eq!(out.num_rows(), 3, "the null-dodge row is dropped");
        assert_eq!(out, b.slice(0, 3), "every other row passes through unchanged");
    }

    #[test]
    fn swarm_emits_no_offset_column_in_either_orient() {
        pyo3::Python::initialize();
        let b = make_batch(vec!["A"; 5], vec![1.0; 5]);
        for orient in [SwarmOrient::Vertical, SwarmOrient::Horizontal] {
            let s = SwarmSpec { orient, ..spec(5.0, 1.0, SwarmSide::Both) };
            let out = apply(&s, &b).unwrap();
            assert_eq!(out.schema(), b.schema(), "{orient:?}");
            assert!(out.schema().index_of("__pos_y_offset__").is_err());
            assert!(out.schema().index_of("__pos_x_offset__").is_err());
        }
    }

    // ---- Rendered placement: the contract holds on the drawn SVG ----

    /// `(cx, cy)` of every `<circle>` in an SVG.
    fn circle_centres(svg: &str) -> Vec<(f64, f64)> {
        let attr = |tag: &str, name: &str| -> f64 {
            let key = format!(" {name}=\"");
            let start = tag.find(&key).unwrap() + key.len();
            let end = start + tag[start..].find('"').unwrap();
            tag[start..end].parse().unwrap()
        };
        svg.split("<circle ")
            .skip(1)
            .map(|rest| {
                let tag = &rest[..rest.find('>').unwrap()];
                (attr(&format!(" {tag}"), "cx"), attr(&format!(" {tag}"), "cy"))
            })
            .collect()
    }

    fn render_output(spec_json: serde_json::Value, batch: &RecordBatch) -> crate::render::RenderOutput<String> {
        let spec: crate::spec::chart::ChartSpec = serde_json::from_value(spec_json).unwrap();
        crate::render::render_svg(
            &spec,
            batch,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport { width: 640.0, height: 480.0 },
            &crate::render::config::RenderConfig::default(),
            &crate::render::chart_config::ChartConfig::default(),
        )
        .expect("swarm chart renders")
    }

    fn render(spec_json: serde_json::Value, batch: &RecordBatch) -> String {
        render_output(spec_json, batch).bytes
    }

    /// Packed exactly: no two centres closer than `step` px (within the SVG's
    /// 3-decimal coordinates, where two rounded centres can each move by
    /// 0.0005 px per axis), and the dense run is not spread looser than that.
    fn assert_packed(pts: &[(f64, f64)], step: f64, what: &str) {
        let md = min_pairwise(pts);
        assert!(md >= step - 0.0015, "{what}: min rendered centre distance {md:.4} < {step}");
        assert!(md < step + 0.5, "{what}: packing too loose ({md:.4} vs {step})");
    }

    /// The chart-level shape `mark_swarm` lowers to, with `extra` merged into
    /// the top-level spec and `y_scale` onto the value channel.
    fn swarm_chart(size: f64, spacing: f64, horizontal: bool, extra: serde_json::Value,
                   y_scale: Option<serde_json::Value>) -> serde_json::Value {
        use serde_json::json;
        let mut value = json!({"field": "v"});
        if let Some(scale) = y_scale {
            value["scale"] = scale;
        }
        let category = json!({"field": "g", "type": "nominal"});
        let (x, y) = if horizontal { (value, category) } else { (category, value) };
        let mut spec = json!({
            "data": {"kind": "named", "name": "default"},
            "mark": "point",
            "encoding": {"x": {"field": if horizontal { "v" } else { "g" }},
                         "y": {"field": if horizontal { "g" } else { "v" }}},
            "transforms": [{"type": "swarm", "category": "g", "value": "v",
                            "point_size": size, "spacing": spacing, "side": "both",
                            "orient": if horizontal { "horizontal" } else { "vertical" },
                            "name": "swarm"}],
            "layers": [{"mark": "point", "encoding": {"x": x, "y": y},
                        "mark_style": {"size": std::f64::consts::PI * (size / 2.0).powi(2)},
                        "data_source": "swarm", "name": "point"}],
        });
        for (k, v) in extra.as_object().unwrap() {
            spec[k] = v.clone();
        }
        spec
    }

    /// Every rendered circle's radius.
    fn circle_radii(svg: &str) -> Vec<f64> {
        svg.split("<circle ")
            .skip(1)
            .map(|rest| {
                let tag = &rest[..rest.find('>').unwrap()];
                let start = tag.find(" r=\"").unwrap() + 4;
                tag[start..start + tag[start..].find('"').unwrap()].parse().unwrap()
            })
            .collect()
    }

    /// A `Swarm`-fed point layer that sets no size of its own draws each dot
    /// `point_size` px across, the diameter the swarm placed it by, on the
    /// layered route and on the flat `Chart.transform(Swarm(...)).mark_point()`
    /// route; a constant or field size of the layer's own still wins.
    #[test]
    fn a_swarm_fed_point_layer_with_no_size_draws_at_the_placement_diameter() {
        use serde_json::json;
        pyo3::Python::initialize();
        let batch = spread_batch();
        let (size, spacing) = (8.0, 1.0);
        let mut layered = swarm_chart(size, spacing, false, json!({}), None);
        layered["layers"][0].as_object_mut().unwrap().remove("mark_style");
        let mut flat = layered.clone();
        let flat_obj = flat.as_object_mut().unwrap();
        flat_obj.remove("layers");
        flat_obj["encoding"]["x"] = json!({"field": "g", "type": "nominal"});
        flat_obj["transforms"][0].as_object_mut().unwrap().remove("name");
        for (route, chart) in [("layered", layered.clone()), ("flat", flat)] {
            let svg = render(chart, &batch);
            let radii = circle_radii(&svg);
            assert_eq!(radii.len(), 300, "{route}: every dot drawn");
            assert!(radii.iter().all(|r| *r == size / 2.0), "{route}: radii {:?}", &radii[..3]);
            assert_packed(&circle_centres(&svg), size + spacing, route);
        }

        let mut own_constant = layered.clone();
        own_constant["layers"][0]["mark_style"] = json!({"size": 100.0});
        let want = (100.0 / std::f64::consts::PI).sqrt();
        let radii = circle_radii(&render(own_constant, &batch));
        assert!(radii.iter().all(|r| (r - want).abs() < 5e-4), "own constant: {:?}", &radii[..3]);

        let mut own_field = layered;
        own_field["layers"][0]["encoding"]["size"] = json!({"field": "v"});
        let radii = circle_radii(&render(own_field, &batch));
        assert!(radii.iter().any(|r| *r != radii[0]), "a size field still varies the radius");
    }

    fn spread_batch() -> RecordBatch {
        let vals = normal_draws(300, 50.0, 15.0);
        let n = vals.len();
        let facets: Vec<&str> = (0..n).map(|i| ["F1", "F2", "F3"][i % 3]).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("f", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a"; n])),
                Arc::new(Float64Array::from(vals)),
                Arc::new(StringArray::from(facets)),
            ],
        )
        .unwrap()
    }

    /// The minimum rendered centre distance is `size + spacing` (within the
    /// SVG's 3-decimal coordinates) under every layout that changes the value
    /// axis' pixels per data unit: the default, a title, a shared-y facet
    /// grid, an explicit y domain, a log y scale and a horizontal swarm.
    #[test]
    fn rendered_swarm_centres_sit_size_plus_spacing_apart_on_every_layout() {
        use serde_json::json;
        pyo3::Python::initialize();
        let batch = spread_batch();
        let none = json!({});
        let layouts: Vec<(&str, bool, serde_json::Value, Option<serde_json::Value>)> = vec![
            ("default", false, none.clone(), None),
            ("title", false, json!({"title": {"text": "A title"}}), None),
            ("facet shared y", false,
             json!({"facet": {"field": "f", "mode": {"kind": "wrap", "ncols": 3}}}), None),
            ("explicit y domain", false,
             json!({"coord": {"kind": "cartesian", "y_domain": [0.0, 200.0], "expand": true, "clip": true}}),
             None),
            ("log y", false, none.clone(),
             Some(json!({"type": "log", "base": 10.0, "clamp": false, "nice": false}))),
            ("horizontal", true, none.clone(), None),
        ];
        for (size, spacing) in [(8.0, 0.0), (4.0, 1.0)] {
            for (name, horizontal, extra, y_scale) in &layouts {
                let svg = render(swarm_chart(size, spacing, *horizontal, extra.clone(), y_scale.clone()), &batch);
                let pts = circle_centres(&svg);
                assert_eq!(pts.len(), 300, "{name}: every dot drawn");
                assert_packed(&pts, size + spacing, &format!("{name} size={size} spacing={spacing}"));
            }
        }
    }

    /// Dodge sub-bands sit `size + spacing` apart and no two dots of the
    /// rendered chart overlap.
    #[test]
    fn rendered_dodged_swarm_does_not_overlap() {
        use serde_json::json;
        pyo3::Python::initialize();
        let base = spread_batch();
        let n = base.num_rows();
        let hue: Vec<&str> = (0..n).map(|i| if i % 2 == 0 { "p" } else { "q" }).collect();
        let mut cols = base.columns().to_vec();
        cols.push(Arc::new(StringArray::from(hue)));
        let mut fields: Vec<Field> = base.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new("h", DataType::Utf8, false));
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
        let mut chart = swarm_chart(8.0, 0.0, false, json!({}), None);
        chart["transforms"][0]["dodge"] = json!("h");
        let pts = circle_centres(&render(chart, &batch));
        assert_eq!(pts.len(), n);
        let md = min_pairwise(&pts);
        assert!(md >= 8.0 - 0.0015, "dodged swarm min rendered centre distance {md:.4}");
    }

    /// A transform after the `Swarm` keeps its dots placed: the layer's data
    /// descends from the swarm whether the later transform is unnamed (the
    /// layer reads the chain tail) or named over that tail.
    #[test]
    fn rendered_swarm_stays_packed_under_a_later_transform() {
        use serde_json::json;
        pyo3::Python::initialize();
        let batch = spread_batch();
        let later = [
            ("filter", json!({"type": "filter", "predicate": "datum.v > 30"})),
            ("calculate", json!({"type": "calculate", "as_field": "w", "expr": "datum.v * 2"})),
            ("sample", json!({"type": "sample", "n": 200})),
        ];
        for (name, transform) in later {
            for named_tail in [false, true] {
                let mut chart = swarm_chart(8.0, 1.0, false, json!({}), None);
                chart["transforms"][0].as_object_mut().unwrap().remove("name");
                let mut transform = transform.clone();
                if named_tail {
                    transform["name"] = json!("tail");
                    chart["layers"][0]["data_source"] = json!("tail");
                } else {
                    chart["layers"][0].as_object_mut().unwrap().remove("data_source");
                }
                chart["transforms"].as_array_mut().unwrap().push(transform);
                let out = render_output(chart, &batch);
                let pts = circle_centres(&out.bytes);
                let what = format!("swarm + {name} (named tail: {named_tail})");
                assert!(pts.len() >= 100, "{what}: {} dots drawn", pts.len());
                assert_packed(&pts, 9.0, &what);
                assert!(out.warnings.is_empty(), "{what}: {:?}", out.warnings);
            }
        }
    }

    /// A swarm wider than its category's slot spills into the neighbouring
    /// category: the dots stay where they were packed and one typed warning,
    /// voiced as the layer's entry, says how far they reach past the
    /// half-slot. A swarm that fits raises none.
    #[test]
    fn a_swarm_wider_than_its_category_slot_warns_once() {
        use serde_json::json;
        pyo3::Python::initialize();
        let batch = |per: usize| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("v", DataType::Float64, false),
            ]));
            let g: Vec<&str> = (0..2 * per).map(|i| if i < per { "a" } else { "b" }).collect();
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(g)), Arc::new(Float64Array::from(vec![5.0; 2 * per]))])
                .unwrap()
        };
        let overflowing = |out: &crate::render::RenderOutput<String>| -> Vec<crate::render::RenderWarning> {
            out.warnings
                .iter()
                .filter(|w| matches!(w, crate::render::RenderWarning::SwarmOverflowsBand { .. }))
                .cloned()
                .collect()
        };
        let mut chart = swarm_chart(10.0, 1.0, false, json!({}), None);
        chart["layers"][0]["entry"] = json!("mark_swarm");
        let out = render_output(chart.clone(), &batch(40));
        let warned = overflowing(&out);
        assert_eq!(warned.len(), 1, "{:?}", out.warnings);
        let crate::render::RenderWarning::SwarmOverflowsBand { entry, reach_px, half_slot_px, .. } = &warned[0] else {
            unreachable!()
        };
        assert_eq!(entry, "mark_swarm");
        // 40 dots on one value pack 20 to a side, 11 px apart: the outermost
        // centre sits 20 * 11 or 19 * 11 px out, plus a 5 px radius.
        assert!((*reach_px - (20.0 * 11.0 + 5.0)).abs() < 1e-9, "{reach_px}");
        assert!(reach_px > half_slot_px);
        assert!(warned[0].to_string().starts_with("mark_swarm: Swarm(category='g', value='v') packs its dots up to 225.0 px"),
                "{}", warned[0]);
        // Vacuity control: four dots a category fit their slot.
        assert_eq!(overflowing(&render_output(chart, &batch(4))), vec![]);
    }

    /// Placement packs whichever rows the layer draws: an aggregate after the
    /// swarm that keeps the category and value columns has its rows placed
    /// with no warning, while one that drops the value column leaves the dots
    /// unplaced and one warning per chart (not per facet panel) names it.
    #[test]
    fn a_later_transform_keeping_the_columns_is_placed_and_one_dropping_them_warns() {
        use serde_json::json;
        pyo3::Python::initialize();
        let aggregated = |as_: &str| {
            let mut chart = swarm_chart(8.0, 1.0, false,
                                        json!({"facet": {"field": "f", "mode": {"kind": "wrap", "ncols": 3}}}), None);
            chart["transforms"][0].as_object_mut().unwrap().remove("name");
            chart["layers"][0].as_object_mut().unwrap().remove("data_source");
            chart["layers"][0]["encoding"]["y"]["field"] = json!(as_);
            chart["transforms"].as_array_mut().unwrap().push(json!({
                "type": "aggregate", "groupby": ["g", "f"],
                "ops": [{"field": "v", "fn": "mean", "as": as_}],
            }));
            render_output(chart, &spread_batch())
        };
        let swarm_warnings = |out: &crate::render::RenderOutput<String>| -> Vec<String> {
            out.warnings
                .iter()
                .filter(|w| matches!(w, crate::render::RenderWarning::PositionAdjustSkipped { adjustment, .. }
                                     if adjustment == "swarm"))
                .map(|w| w.to_string())
                .collect()
        };
        let kept = aggregated("v");
        assert_eq!(swarm_warnings(&kept), Vec::<String>::new());
        assert_eq!(circle_centres(&kept.bytes).len(), 3, "one placed dot per facet panel");
        assert_eq!(swarm_warnings(&aggregated("m")), vec![
            "swarm could not be applied (column 'v' that Swarm(category='g', value='v') \
             places its dots by is not in the layer's data; a later transform dropped it); \
             marks were not offset".to_string(),
        ]);
    }

    /// A flipped layer whose chain holds an earlier Swarm of the opposite
    /// orientation still packs the dots the layer actually draws by the
    /// later Swarm.
    #[test]
    fn rendered_swarm_stays_packed_under_an_earlier_opposite_orient_swarm_on_a_flipped_layer() {
        use serde_json::json;
        pyo3::Python::initialize();
        let batch = spread_batch();
        let (size, spacing) = (8.0, 1.0);
        let mut chart = swarm_chart(size, spacing, false, json!({"coord": {"kind": "flip"}}), None);
        chart["transforms"][0]["name"] = json!("swarm2");
        chart["layers"][0]["data_source"] = json!("swarm2");
        chart["transforms"].as_array_mut().unwrap().insert(0, json!({
            "type": "swarm", "category": "g", "value": "v",
            "point_size": size, "spacing": spacing, "side": "both", "orient": "horizontal",
        }));
        let out = render_output(chart, &batch);
        let pts = circle_centres(&out.bytes);
        assert_eq!(pts.len(), 300, "every dot drawn");
        assert_packed(&pts, size + spacing, "chained opposite-orient swarm on a flipped layer");
    }
}
