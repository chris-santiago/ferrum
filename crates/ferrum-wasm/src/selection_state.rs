use std::collections::{HashMap, HashSet};

use ferrum_scene::{FieldValue, SelectionSpec};

use crate::hit_test::{self, HitResult};

#[derive(Debug, Clone)]
pub enum SelectionState {
    Empty,
    Point {
        indices: Vec<usize>,
        field_values: Vec<(String, FieldValue)>,
    },
    Interval {
        x_range: Option<(f64, f64)>,
        y_range: Option<(f64, f64)>,
    },
}

impl SelectionState {
    pub fn contains(&self, data_idx: usize) -> bool {
        match self {
            Self::Empty => false,
            Self::Point { indices, .. } => indices.contains(&data_idx),
            Self::Interval { .. } => false,
        }
    }

    /// Returns `true` if the given screen-space point falls within this
    /// selection's spatial extent.
    ///
    /// For `Interval` selections, a `None` range on either axis means that
    /// dimension is unconstrained (always in range). Boundaries are inclusive.
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        match self {
            Self::Empty => false,
            Self::Point { .. } => false, // point selections don't use spatial containment
            Self::Interval { x_range, y_range } => {
                let in_x = x_range.is_none_or(|(lo, hi)| x >= lo && x <= hi);
                let in_y = y_range.is_none_or(|(lo, hi)| y >= lo && y <= hi);
                in_x && in_y
            }
        }
    }
}

pub struct InteractionState {
    pub selections: HashMap<String, SelectionState>,
    pub hover: Option<HitResult>,
}

impl InteractionState {
    pub fn new(specs: &[SelectionSpec]) -> Self {
        let mut selections = HashMap::new();
        for spec in specs {
            let name = match spec {
                SelectionSpec::Point { name, .. } => name,
                SelectionSpec::Interval { name, .. } => name,
            };
            selections.insert(name.clone(), SelectionState::Empty);
        }
        InteractionState {
            selections,
            hover: None,
        }
    }

    pub fn handle_click(
        &mut self,
        panels: &[ferrum_scene::Panel],
        specs: &[SelectionSpec],
        x: f64,
        y: f64,
        zoom: &crate::zoom_pan::ZoomPanState,
        shift_held: bool,
    ) {
        let hit = hit_test::hit_test(panels, x, y, zoom);

        for spec in specs {
            match spec {
                SelectionSpec::Point {
                    name, toggle, fields, ..
                } => {
                    let sel = self.selections.entry(name.clone()).or_insert(SelectionState::Empty);
                    match &hit {
                        Some(h) => {
                            if let Some(data_idx) = h.data_idx {
                                // When the spec declares field constraints, expand the
                                // selection to ALL marks sharing the same field values.
                                let indices = if let Some(field_names) = fields {
                                    collect_matching_indices(panels, h, field_names, data_idx)
                                } else {
                                    vec![data_idx]
                                };
                                // Extract field values from the clicked mark's tooltip
                                // so cross-panel conditional resolution can match by
                                // field content rather than by data index.
                                let field_values = if let Some(field_names) = fields {
                                    extract_field_values(panels, h, field_names)
                                } else {
                                    Vec::new()
                                };
                                let is_toggle = shift_held
                                    && matches!(
                                        toggle,
                                        ferrum_scene::EventExpr::ShiftKey
                                    );
                                if is_toggle {
                                    toggle_points(sel, &indices, &field_values);
                                } else {
                                    *sel = SelectionState::Point {
                                        indices,
                                        field_values,
                                    };
                                }
                            }
                        }
                        None => {
                            *sel = SelectionState::Empty;
                        }
                    }
                }
                SelectionSpec::Interval { .. } => {}
            }
        }

        self.hover = hit;
    }

    pub fn handle_drag(
        &mut self,
        specs: &[SelectionSpec],
        panel_id: usize,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) {
        for spec in specs {
            if let SelectionSpec::Interval { name, .. } = spec {
                let x_lo = x0.min(x1);
                let x_hi = x0.max(x1);
                let y_lo = y0.min(y1);
                let y_hi = y0.max(y1);
                let _ = panel_id;
                self.selections.insert(
                    name.clone(),
                    SelectionState::Interval {
                        x_range: Some((x_lo, x_hi)),
                        y_range: Some((y_lo, y_hi)),
                    },
                );
            }
        }
    }

    pub fn handle_mousemove(
        &mut self,
        panels: &[ferrum_scene::Panel],
        x: f64,
        y: f64,
        zoom: &crate::zoom_pan::ZoomPanState,
    ) -> Option<&HitResult> {
        self.hover = hit_test::hit_test(panels, x, y, zoom);
        self.hover.as_ref()
    }

    pub fn to_json(&self) -> String {
        let mut map = serde_json::Map::new();
        for (name, state) in &self.selections {
            let val = match state {
                SelectionState::Empty => serde_json::json!({"type": "empty"}),
                SelectionState::Point {
                    indices,
                    field_values,
                } => {
                    let fv: Vec<serde_json::Value> = field_values
                        .iter()
                        .map(|(k, v)| {
                            serde_json::json!({
                                "field": k,
                                "value": match v {
                                    FieldValue::String { value } => serde_json::Value::String(value.clone()),
                                    FieldValue::Number { value } => serde_json::json!(value),
                                    FieldValue::Bool { value } => serde_json::Value::Bool(*value),
                                    FieldValue::Null => serde_json::Value::Null,
                                }
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "type": "point",
                        "indices": indices,
                        "field_values": fv,
                    })
                }
                SelectionState::Interval { x_range, y_range } => {
                    serde_json::json!({
                        "type": "interval",
                        "x_range": x_range,
                        "y_range": y_range,
                    })
                }
            };
            map.insert(name.clone(), val);
        }
        serde_json::Value::Object(map).to_string()
    }
}

/// Navigate panels → batch → tooltips to retrieve the tooltip entry for a hit.
/// Returns `None` when any step of the chain is absent.
fn tooltip_for_hit<'a>(
    panels: &'a [ferrum_scene::Panel],
    hit: &crate::hit_test::HitResult,
) -> Option<&'a ferrum_scene::TooltipContent> {
    let panel = panels.get(hit.panel_id)?;
    let batch = panel.marks.get(hit.batch_idx)?;
    batch.tooltips.as_ref()?.get(hit.node_idx)
}

/// Extract field values from the clicked mark's tooltip, converting each
/// tooltip string to a typed `FieldValue` (number if parseable, else string).
fn extract_field_values(
    panels: &[ferrum_scene::Panel],
    hit: &crate::hit_test::HitResult,
    field_names: &[String],
) -> Vec<(String, FieldValue)> {
    let Some(tooltip) = tooltip_for_hit(panels, hit) else {
        return Vec::new();
    };

    field_names
        .iter()
        .filter_map(|fname| {
            tooltip
                .fields
                .iter()
                .find(|f| f.name == *fname)
                .map(|f| {
                    let fv = if let Ok(n) = f.value.parse::<f64>() {
                        FieldValue::Number { value: n }
                    } else {
                        FieldValue::String {
                            value: f.value.clone(),
                        }
                    };
                    (fname.clone(), fv)
                })
        })
        .collect()
}

/// Scan all marks in a panel for rows whose tooltip field values match those
/// of the clicked mark.  Returns the set of data indices to select.
///
/// Falls back to `vec![clicked_data_idx]` when tooltip data is unavailable
/// or the field is not present in any tooltip.
fn collect_matching_indices(
    panels: &[ferrum_scene::Panel],
    hit: &crate::hit_test::HitResult,
    field_names: &[String],
    clicked_data_idx: usize,
) -> Vec<usize> {
    // Get the field values for the clicked mark from its tooltip.
    let clicked_values: Vec<(&str, &str)> = tooltip_for_hit(panels, hit)
        .map(|tip| {
            field_names.iter()
                .filter_map(|fname| {
                    tip.fields.iter()
                        .find(|f| &f.name == fname)
                        .map(|f| (fname.as_str(), f.value.as_str()))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if clicked_values.is_empty() {
        return vec![clicked_data_idx];
    }

    // Scan all mark batches across ALL panels; collect data indices whose
    // tooltip fields match every (field_name, field_value) pair from the
    // clicked mark. This enables cross-panel linked selection.
    //
    // A HashSet tracks already-seen indices so membership checks stay O(1),
    // avoiding an O(n²) scan when many marks match across large batches.
    // Insertion order is preserved by pushing to `matching` only on first
    // encounter (`seen.insert` returns true exactly once per unique index).
    let mut seen: HashSet<usize> = HashSet::new();
    let mut matching: Vec<usize> = Vec::new();
    for p in panels {
        for batch in &p.marks {
            let Some(tooltips) = batch.tooltips.as_ref() else {
                continue;
            };
            let data_indices = batch.data_indices.as_deref();
            for (node_idx, tooltip) in tooltips.iter().enumerate() {
                let all_match = clicked_values.iter().all(|(fname, fval)| {
                    tooltip
                        .fields
                        .iter()
                        .any(|f| f.name == *fname && f.value == *fval)
                });
                if all_match {
                    let data_idx = data_indices
                        .and_then(|di| di.get(node_idx))
                        .copied()
                        .unwrap_or(node_idx);
                    if seen.insert(data_idx) {
                        matching.push(data_idx);
                    }
                }
            }
        }
    }

    if matching.is_empty() { vec![clicked_data_idx] } else { matching }
}

/// Toggle a set of indices: if all are already selected, deselect them;
/// otherwise add all missing ones. Field values are replaced wholesale on
/// add (they describe the most recent click) and cleared on full deselect.
fn toggle_points(
    sel: &mut SelectionState,
    indices: &[usize],
    new_field_values: &[(String, FieldValue)],
) {
    // Build a HashSet from the existing indices once so all membership checks
    // below are O(1) rather than O(n) per query.
    let existing_set: HashSet<usize> = match sel {
        SelectionState::Point { indices: existing, .. } => existing.iter().copied().collect(),
        _ => HashSet::new(),
    };

    let already_all = !indices.is_empty() && indices.iter().all(|i| existing_set.contains(i));

    if already_all {
        // Deselect — remove all toggle indices from the existing set.
        let remove_set: HashSet<usize> = indices.iter().copied().collect();
        if let SelectionState::Point {
            indices: existing,
            field_values,
        } = sel
        {
            existing.retain(|i| !remove_set.contains(i));
            if existing.is_empty() {
                *sel = SelectionState::Empty;
            } else {
                // Keep field_values from the remaining selection; if the
                // deselected entries were the only source, clear them.
                field_values.clear();
            }
        }
    } else {
        // Add missing indices (O(1) membership via existing_set).
        match sel {
            SelectionState::Point {
                indices: existing,
                field_values,
            } => {
                for &idx in indices {
                    if !existing_set.contains(&idx) {
                        existing.push(idx);
                    }
                }
                *field_values = new_field_values.to_vec();
            }
            _ => {
                *sel = SelectionState::Point {
                    indices: indices.to_vec(),
                    field_values: new_field_values.to_vec(),
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_spec(name: &str) -> SelectionSpec {
        SelectionSpec::Point {
            name: name.to_string(),
            fields: None,
            encodings: None,
            nearest: false,
            toggle: ferrum_scene::EventExpr::Click,
            on: ferrum_scene::EventExpr::Click,
            clear: ferrum_scene::EventExpr::Mouseout,
            resolve: ferrum_scene::SelectionResolve::Global,
        }
    }

    #[test]
    fn new_initializes_empty_selections() {
        let specs = vec![point_spec("sel1")];
        let state = InteractionState::new(&specs);
        assert!(matches!(
            state.selections.get("sel1"),
            Some(SelectionState::Empty)
        ));
    }

    #[test]
    fn toggle_adds_and_removes() {
        let mut sel = SelectionState::Point {
            indices: vec![0],
            field_values: Vec::new(),
        };
        toggle_points(&mut sel, &[1], &[]);
        if let SelectionState::Point { indices, .. } = &sel {
            assert_eq!(indices, &[0, 1]);
        }
        toggle_points(&mut sel, &[0], &[]);
        if let SelectionState::Point { indices, .. } = &sel {
            assert_eq!(indices, &[1]);
        }
        toggle_points(&mut sel, &[1], &[]);
        assert!(matches!(sel, SelectionState::Empty));
    }

    #[test]
    fn background_click_clears_existing_selection() {
        // The JS click handler now calls handleClick for ALL clicks (hit or miss).
        // A click at coordinates that hit no mark must clear Point selections to Empty.
        let specs = vec![point_spec("sel1")];
        let mut state = InteractionState::new(&specs);
        // Pre-populate a selection as if a mark had been clicked earlier.
        state.selections.insert(
            "sel1".to_string(),
            SelectionState::Point { indices: vec![0, 1], field_values: Vec::new() },
        );
        // Click on empty panels (no marks) — simulates a background click.
        let zoom = crate::zoom_pan::ZoomPanState::new(0, &ferrum_scene::InteractionConfig::default());
        state.handle_click(&[], &specs, 50.0, 50.0, &zoom, false);
        assert!(
            matches!(state.selections.get("sel1"), Some(SelectionState::Empty)),
            "background click must deselect to Empty"
        );
    }

    #[test]
    fn background_click_with_no_prior_selection_stays_empty() {
        let specs = vec![point_spec("s")];
        let mut state = InteractionState::new(&specs);
        let zoom = crate::zoom_pan::ZoomPanState::new(0, &ferrum_scene::InteractionConfig::default());
        state.handle_click(&[], &specs, 0.0, 0.0, &zoom, false);
        assert!(matches!(state.selections.get("s"), Some(SelectionState::Empty)));
    }

    #[test]
    fn to_json_empty() {
        let state = InteractionState::new(&[point_spec("s")]);
        let json = state.to_json();
        assert!(json.contains("\"empty\""));
    }

    // ── bug_hunt: InteractionState edge cases ─────────────────────────────────

    #[test]
    fn bug_hunt_new_with_empty_specs_has_no_selections() {
        let state = InteractionState::new(&[]);
        assert!(state.selections.is_empty(), "empty specs must produce empty selections map");
    }

    #[test]
    fn bug_hunt_to_json_produces_valid_json_for_point_selection() {
        let specs = vec![point_spec("sel1")];
        let mut state = InteractionState::new(&specs);
        state.selections.insert(
            "sel1".to_string(),
            SelectionState::Point {
                indices: vec![0, 2, 5],
                field_values: Vec::new(),
            },
        );
        let json_str = state.to_json();
        // Must be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("to_json must produce valid JSON");
        let sel = &parsed["sel1"];
        assert_eq!(sel["type"], "point");
        let indices = sel["indices"].as_array().expect("indices must be array");
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn bug_hunt_to_json_produces_valid_json_for_interval_selection() {
        let specs = vec![
            SelectionSpec::Interval {
                name: "brush".to_string(),
                fields: None,
                encodings: None,
                translate: true,
                zoom: true,
                mark: None,
                resolve: ferrum_scene::SelectionResolve::Global,
            }
        ];
        let mut state = InteractionState::new(&specs);
        state.selections.insert(
            "brush".to_string(),
            SelectionState::Interval {
                x_range: Some((10.0, 200.0)),
                y_range: Some((50.0, 300.0)),
            },
        );
        let json_str = state.to_json();
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("to_json must produce valid JSON");
        assert_eq!(parsed["brush"]["type"], "interval");
        let xr = parsed["brush"]["x_range"].as_array().expect("x_range must be array");
        assert!((xr[0].as_f64().unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_handle_drag_clamps_lo_hi_regardless_of_order() {
        // Dragging right-to-left (x0 > x1) must still produce correct lo/hi ordering.
        let specs = vec![
            SelectionSpec::Interval {
                name: "b".to_string(),
                fields: None,
                encodings: None,
                translate: true,
                zoom: true,
                mark: None,
                resolve: ferrum_scene::SelectionResolve::Global,
            }
        ];
        let mut state = InteractionState::new(&specs);
        state.handle_drag(&specs, 0, 300.0, 200.0, 100.0, 50.0);
        match state.selections.get("b") {
            Some(SelectionState::Interval { x_range: Some((lo, hi)), .. }) => {
                assert!(lo <= hi, "x_range lo must be <= hi regardless of drag direction");
                assert!((lo - 100.0).abs() < 1e-10, "lo must be min(x0,x1)=100");
                assert!((hi - 300.0).abs() < 1e-10, "hi must be max(x0,x1)=300");
            }
            other => panic!("expected Interval x_range, got {other:?}"),
        }
    }

    #[test]
    fn bug_hunt_toggle_points_deselects_when_all_already_selected() {
        // Toggling indices that are already all selected must deselect them.
        let mut sel = SelectionState::Point {
            indices: vec![1, 2, 3],
            field_values: Vec::new(),
        };
        toggle_points(&mut sel, &[1, 2, 3], &[]);
        assert!(matches!(sel, SelectionState::Empty), "all-selected toggle must deselect to Empty");
    }

    #[test]
    fn bug_hunt_toggle_points_adds_when_none_selected() {
        // Toggling from Empty must add the indices.
        let mut sel = SelectionState::Empty;
        toggle_points(&mut sel, &[4, 5], &[]);
        match &sel {
            SelectionState::Point { indices, .. } => {
                assert!(indices.contains(&4) && indices.contains(&5));
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn bug_hunt_toggle_points_partial_overlap_adds_missing() {
        // [0, 1] already selected; toggle [1, 2] — should add 2 since not all are selected.
        let mut sel = SelectionState::Point {
            indices: vec![0, 1],
            field_values: Vec::new(),
        };
        toggle_points(&mut sel, &[1, 2], &[]);
        match &sel {
            SelectionState::Point { indices, .. } => {
                assert!(indices.contains(&2), "index 2 must be added");
                // 1 was already there, must still be present
                assert!(indices.contains(&1));
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn bug_hunt_selection_contains_empty_always_false() {
        let sel = SelectionState::Empty;
        assert!(!sel.contains(0));
        assert!(!sel.contains(999));
    }

    #[test]
    fn bug_hunt_selection_contains_point_correct_membership() {
        let sel = SelectionState::Point {
            indices: vec![3, 7, 11],
            field_values: Vec::new(),
        };
        assert!(sel.contains(3));
        assert!(sel.contains(7));
        assert!(!sel.contains(0));
        assert!(!sel.contains(100));
    }

    // ── R1–R6: regression tests for interval/brush selection ────────────────

    fn interval_spec(name: &str) -> SelectionSpec {
        SelectionSpec::Interval {
            name: name.to_string(),
            fields: None,
            encodings: None,
            translate: true,
            zoom: true,
            mark: None,
            resolve: ferrum_scene::SelectionResolve::Global,
        }
    }

    fn shift_toggle_point_spec(name: &str) -> SelectionSpec {
        SelectionSpec::Point {
            name: name.to_string(),
            fields: Some(vec![]),
            encodings: None,
            nearest: false,
            toggle: ferrum_scene::EventExpr::ShiftKey,
            on: ferrum_scene::EventExpr::Click,
            clear: ferrum_scene::EventExpr::Mouseout,
            resolve: ferrum_scene::SelectionResolve::Global,
        }
    }

    // R1: handle_drag updates interval selection state.
    #[test]
    fn r1_handle_drag_updates_interval_selection() {
        let specs = vec![interval_spec("brush")];
        let mut state = InteractionState::new(&specs);
        state.handle_drag(&specs, 0, 10.0, 20.0, 50.0, 60.0);
        match state.selections.get("brush") {
            Some(SelectionState::Interval {
                x_range: Some((x_lo, x_hi)),
                y_range: Some((y_lo, y_hi)),
            }) => {
                assert!(
                    (x_lo - 10.0).abs() < 1e-10 && (x_hi - 50.0).abs() < 1e-10,
                    "x_range must be (10.0, 50.0), got ({x_lo}, {x_hi})"
                );
                assert!(
                    (y_lo - 20.0).abs() < 1e-10 && (y_hi - 60.0).abs() < 1e-10,
                    "y_range must be (20.0, 60.0), got ({y_lo}, {y_hi})"
                );
            }
            other => panic!("expected Interval with both ranges, got {other:?}"),
        }
    }

    // R2: Interval contains_point spatial resolution.

    // Outside x — should return false even with the stub.
    #[test]
    fn r2_contains_point_outside_x() {
        let sel = SelectionState::Interval {
            x_range: Some((10.0, 50.0)),
            y_range: Some((20.0, 60.0)),
        };
        assert!(
            !sel.contains_point(5.0, 40.0),
            "point outside x_range must not be contained"
        );
    }

    // Outside y — should return false even with the stub.
    #[test]
    fn r2_contains_point_outside_y() {
        let sel = SelectionState::Interval {
            x_range: Some((10.0, 50.0)),
            y_range: Some((20.0, 60.0)),
        };
        assert!(
            !sel.contains_point(30.0, 70.0),
            "point outside y_range must not be contained"
        );
    }

    // Inside — contains_point now implemented.
    #[test]
    fn r2_contains_point_inside() {
        let sel = SelectionState::Interval {
            x_range: Some((10.0, 50.0)),
            y_range: Some((20.0, 60.0)),
        };
        assert!(
            sel.contains_point(30.0, 40.0),
            "point inside interval must be contained"
        );
    }

    // On boundary — contains_point now implemented with inclusive bounds.
    #[test]
    fn r2_contains_point_on_boundary() {
        let sel = SelectionState::Interval {
            x_range: Some((10.0, 50.0)),
            y_range: Some((20.0, 60.0)),
        };
        assert!(
            sel.contains_point(10.0, 20.0),
            "point on boundary (lo, lo) must be contained (inclusive)"
        );
    }

    // R4: Point selection shift-click toggle semantics.
    #[test]
    fn r4_point_selection_shift_click_toggle() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect,
        };
        let specs = vec![shift_toggle_point_spec("sel")];
        let mut state = InteractionState::new(&specs);

        let style = FillStroke {
            fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![ferrum_scene::SceneNode::Circle {
                    cx: 100.0,
                    cy: 100.0,
                    r: 10.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
                tooltips: None,
                hrefs: None,
                keys: None,
                blend: BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
                packed_instances: None,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];
        let zoom = crate::zoom_pan::ZoomPanState::new(
            1,
            &ferrum_scene::InteractionConfig::default(),
        );

        // First click on mark with shift held — should select index 0.
        state.handle_click(&panels, &specs, 100.0, 100.0, &zoom, true);
        match state.selections.get("sel") {
            Some(SelectionState::Point { indices, .. }) => {
                assert!(
                    indices.contains(&0),
                    "first click must select index 0, got {indices:?}"
                );
            }
            other => panic!("expected Point selection after first click, got {other:?}"),
        }

        // Second click on same mark with shift held — toggle=ShiftKey means deselect.
        state.handle_click(&panels, &specs, 100.0, 100.0, &zoom, true);
        assert!(
            matches!(
                state.selections.get("sel"),
                Some(SelectionState::Empty)
            ),
            "second click on same mark must toggle to Empty"
        );
    }

    // R5: handle_click with empty panels returns Empty selection.
    #[test]
    fn r5_handle_click_empty_panels_gives_empty_selection() {
        let specs = vec![point_spec("sel")];
        let mut state = InteractionState::new(&specs);
        // Pre-seed a selection to verify it gets cleared.
        state.selections.insert(
            "sel".to_string(),
            SelectionState::Point {
                indices: vec![42],
                field_values: Vec::new(),
            },
        );
        let zoom = crate::zoom_pan::ZoomPanState::new(
            0,
            &ferrum_scene::InteractionConfig::default(),
        );
        state.handle_click(&[], &specs, 50.0, 50.0, &zoom, false);
        assert!(
            matches!(state.selections.get("sel"), Some(SelectionState::Empty)),
            "click with empty panels must produce Empty selection"
        );
    }

    // R6: Skipped — already covered by bug_hunt_to_json_produces_valid_json_for_interval_selection.

    // ── Linked-selection interaction tests ──────────────────────────────

    // Test 1: handle_drag with reversed coordinates normalizes lo < hi.
    #[test]
    fn handle_drag_reversed_coords_normalizes_ranges() {
        let specs = vec![interval_spec("brush")];
        let mut state = InteractionState::new(&specs);
        // Drag from bottom-right to top-left: x0 > x1, y0 > y1.
        state.handle_drag(&specs, 0, 400.0, 300.0, 100.0, 50.0);
        match state.selections.get("brush") {
            Some(SelectionState::Interval {
                x_range: Some((x_lo, x_hi)),
                y_range: Some((y_lo, y_hi)),
            }) => {
                assert!(
                    x_lo <= x_hi,
                    "x_range lo must be <= hi after reversed drag, got ({x_lo}, {x_hi})"
                );
                assert!(
                    y_lo <= y_hi,
                    "y_range lo must be <= hi after reversed drag, got ({y_lo}, {y_hi})"
                );
                assert!(
                    (x_lo - 100.0).abs() < 1e-10,
                    "x_lo must be min(400,100)=100, got {x_lo}"
                );
                assert!(
                    (x_hi - 400.0).abs() < 1e-10,
                    "x_hi must be max(400,100)=400, got {x_hi}"
                );
                assert!(
                    (y_lo - 50.0).abs() < 1e-10,
                    "y_lo must be min(300,50)=50, got {y_lo}"
                );
                assert!(
                    (y_hi - 300.0).abs() < 1e-10,
                    "y_hi must be max(300,50)=300, got {y_hi}"
                );
            }
            other => panic!("expected Interval with both ranges, got {other:?}"),
        }
    }

    // Test 2: contains_point with partial ranges (x only, y unconstrained).
    #[test]
    fn contains_point_partial_range_x_only() {
        let sel = SelectionState::Interval {
            x_range: Some((10.0, 50.0)),
            y_range: None,
        };
        // Point at any y should be contained if x is in range.
        assert!(
            sel.contains_point(30.0, 0.0),
            "x=30 in [10,50], y unconstrained → must be contained"
        );
        assert!(
            sel.contains_point(30.0, 99999.0),
            "x=30 in [10,50], y unconstrained (large y) → must be contained"
        );
        assert!(
            sel.contains_point(10.0, -100.0),
            "x=10 on boundary, y unconstrained → must be contained"
        );
        // Point outside x range should not be contained.
        assert!(
            !sel.contains_point(5.0, 30.0),
            "x=5 outside [10,50] → must NOT be contained"
        );
        assert!(
            !sel.contains_point(55.0, 30.0),
            "x=55 outside [10,50] → must NOT be contained"
        );
    }

    // Test 3: contains_point with both ranges None (unconstrained = everything).
    #[test]
    fn contains_point_both_ranges_none_contains_everything() {
        let sel = SelectionState::Interval {
            x_range: None,
            y_range: None,
        };
        assert!(
            sel.contains_point(0.0, 0.0),
            "unconstrained interval must contain (0, 0)"
        );
        assert!(
            sel.contains_point(1000.0, -500.0),
            "unconstrained interval must contain arbitrary point"
        );
        assert!(
            sel.contains_point(f64::MAX, f64::MIN),
            "unconstrained interval must contain extreme values"
        );
    }

    // Test 4: handle_click clears on background miss (pre-seeded Point selection).
    #[test]
    fn handle_click_clears_preseeded_selection_on_background_miss() {
        let specs = vec![point_spec("sel")];
        let mut state = InteractionState::new(&specs);
        // Pre-seed a Point selection with indices [0, 1, 2].
        state.selections.insert(
            "sel".to_string(),
            SelectionState::Point {
                indices: vec![0, 1, 2],
                field_values: Vec::new(),
            },
        );
        // Click at coordinates that miss all marks (empty panels = no marks to hit).
        let zoom = crate::zoom_pan::ZoomPanState::new(
            0,
            &ferrum_scene::InteractionConfig::default(),
        );
        state.handle_click(&[], &specs, 999.0, 999.0, &zoom, false);
        assert!(
            matches!(state.selections.get("sel"), Some(SelectionState::Empty)),
            "background click on pre-seeded [0,1,2] must clear to Empty"
        );
    }

    // W1: handle_drag stores scene-space coordinates for interval selection.
    //
    // The coordinate conversion from canvas-space to scene-space happens in
    // lib.rs (WasmRenderer::handle_drag) using Affine2::inverse_apply before
    // calling InteractionState::handle_drag. This test verifies that the
    // InteractionState correctly stores whatever coordinates it receives,
    // and that contains_point uses those stored coordinates consistently.
    // The inverse_apply math itself is tested in zoom_pan.rs.
    #[test]
    fn w1_handle_drag_stores_coords_and_contains_point_uses_them() {
        let specs = vec![interval_spec("brush")];
        let mut state = InteractionState::new(&specs);

        // Simulate what lib.rs does after inverse_apply: pass scene-space
        // coordinates (100, 50) to (300, 250) to handle_drag.
        state.handle_drag(&specs, 0, 100.0, 50.0, 300.0, 250.0);

        // The stored interval must match the scene-space coords.
        match state.selections.get("brush") {
            Some(SelectionState::Interval {
                x_range: Some((x_lo, x_hi)),
                y_range: Some((y_lo, y_hi)),
            }) => {
                assert!(
                    (x_lo - 100.0).abs() < 1e-10 && (x_hi - 300.0).abs() < 1e-10,
                    "stored x_range must be scene-space (100, 300), got ({x_lo}, {x_hi})"
                );
                assert!(
                    (y_lo - 50.0).abs() < 1e-10 && (y_hi - 250.0).abs() < 1e-10,
                    "stored y_range must be scene-space (50, 250), got ({y_lo}, {y_hi})"
                );
            }
            other => panic!("expected Interval with both ranges, got {other:?}"),
        }

        // A scene-space point inside the interval must be contained.
        let sel = state.selections.get("brush").expect("brush must exist");
        assert!(sel.contains_point(200.0, 150.0), "scene-space point inside interval must be contained");

        // A canvas-space point that would be inside the brush at 2x zoom
        // (e.g., 400, 300 in canvas-space maps to 200, 150 in scene-space
        // at identity) is OUTSIDE the stored scene-space interval.
        // This is the bug W1 prevents: without inverse_apply, the stored
        // range would be in canvas-space and the comparison would use
        // scene-space mark positions, causing a mismatch.
        assert!(!sel.contains_point(400.0, 300.0), "canvas-space point must be outside scene-space interval");
    }

    // Test 5: handle_click with fields populates field_values.
    #[test]
    fn handle_click_with_fields_populates_field_values() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect,
            TooltipContent, TooltipField,
        };
        let style = FillStroke {
            fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        // Build a panel with one circle that has a tooltip containing group="a".
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![ferrum_scene::SceneNode::Circle {
                    cx: 100.0,
                    cy: 100.0,
                    r: 10.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
                tooltips: Some(vec![TooltipContent {
                    fields: vec![TooltipField {
                        name: "group".to_string(),
                        value: "a".to_string(),
                    }],
                }]),
                hrefs: None,
                keys: None,
                blend: BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
                packed_instances: None,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];

        // Point spec with fields: Some(vec!["group"]).
        let specs = vec![SelectionSpec::Point {
            name: "sel".to_string(),
            fields: Some(vec!["group".to_string()]),
            encodings: None,
            nearest: false,
            toggle: ferrum_scene::EventExpr::Click,
            on: ferrum_scene::EventExpr::Click,
            clear: ferrum_scene::EventExpr::Mouseout,
            resolve: ferrum_scene::SelectionResolve::Global,
        }];
        let mut state = InteractionState::new(&specs);
        let zoom = crate::zoom_pan::ZoomPanState::new(
            1,
            &ferrum_scene::InteractionConfig::default(),
        );

        // Click on the mark.
        state.handle_click(&panels, &specs, 100.0, 100.0, &zoom, false);
        match state.selections.get("sel") {
            Some(SelectionState::Point {
                field_values,
                indices,
                ..
            }) => {
                assert!(
                    !field_values.is_empty(),
                    "field_values must be non-empty after clicking mark with tooltip, got empty"
                );
                assert!(
                    !indices.is_empty(),
                    "indices must be non-empty after clicking mark"
                );
                // Check that field_values contains ("group", String("a")).
                let has_group = field_values.iter().any(|(name, val)| {
                    name == "group"
                        && matches!(val, FieldValue::String { value } if value == "a")
                });
                assert!(
                    has_group,
                    "field_values must contain (\"group\", String(\"a\")), got {field_values:?}"
                );
            }
            other => panic!("expected Point selection with field_values, got {other:?}"),
        }
    }
}
