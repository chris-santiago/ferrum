use std::collections::HashMap;

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
                                let is_toggle = matches!(
                                    toggle,
                                    ferrum_scene::EventExpr::ShiftKey
                                );
                                if is_toggle {
                                    toggle_points(sel, &indices);
                                } else {
                                    *sel = SelectionState::Point {
                                        indices,
                                        field_values: Vec::new(),
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
    let Some(panel) = panels.get(hit.panel_id) else {
        return vec![clicked_data_idx];
    };

    // Get the field values for the clicked mark from its tooltip.
    let hit_batch = panel.marks.get(hit.batch_idx);
    let clicked_values: Vec<(&str, &str)> = hit_batch
        .and_then(|b| b.tooltips.as_ref())
        .and_then(|tips| tips.get(hit.node_idx))
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

    // Scan all mark batches in the panel; collect data indices whose tooltip
    // fields match every (field_name, field_value) pair from the clicked mark.
    let mut matching: Vec<usize> = Vec::new();
    for batch in &panel.marks {
        let Some(tooltips) = batch.tooltips.as_ref() else { continue; };
        let data_indices = batch.data_indices.as_deref();
        for (node_idx, tooltip) in tooltips.iter().enumerate() {
            let all_match = clicked_values.iter().all(|(fname, fval)| {
                tooltip.fields.iter().any(|f| f.name == *fname && f.value == *fval)
            });
            if all_match {
                let data_idx = data_indices
                    .and_then(|di| di.get(node_idx))
                    .copied()
                    .unwrap_or(node_idx);
                if !matching.contains(&data_idx) {
                    matching.push(data_idx);
                }
            }
        }
    }

    if matching.is_empty() { vec![clicked_data_idx] } else { matching }
}

/// Toggle a set of indices: if all are already selected, deselect them;
/// otherwise add all missing ones.
fn toggle_points(sel: &mut SelectionState, indices: &[usize]) {
    let already_all = match sel {
        SelectionState::Point { indices: existing, .. } => {
            indices.iter().all(|i| existing.contains(i))
        }
        _ => false,
    };
    if already_all {
        // Deselect — remove all from set.
        if let SelectionState::Point { indices: existing, .. } = sel {
            existing.retain(|i| !indices.contains(i));
            if existing.is_empty() {
                *sel = SelectionState::Empty;
            }
        }
    } else {
        // Add missing.
        match sel {
            SelectionState::Point { indices: existing, .. } => {
                for &idx in indices {
                    if !existing.contains(&idx) {
                        existing.push(idx);
                    }
                }
            }
            _ => {
                *sel = SelectionState::Point {
                    indices: indices.to_vec(),
                    field_values: Vec::new(),
                };
            }
        }
    }
}

fn toggle_point(sel: &mut SelectionState, data_idx: usize) {
    match sel {
        SelectionState::Point { indices, .. } => {
            if let Some(pos) = indices.iter().position(|&i| i == data_idx) {
                indices.remove(pos);
                if indices.is_empty() {
                    *sel = SelectionState::Empty;
                }
            } else {
                indices.push(data_idx);
            }
        }
        _ => {
            *sel = SelectionState::Point {
                indices: vec![data_idx],
                field_values: Vec::new(),
            };
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
        toggle_point(&mut sel, 1);
        if let SelectionState::Point { indices, .. } = &sel {
            assert_eq!(indices, &[0, 1]);
        }
        toggle_point(&mut sel, 0);
        if let SelectionState::Point { indices, .. } = &sel {
            assert_eq!(indices, &[1]);
        }
        toggle_point(&mut sel, 1);
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
        state.handle_click(&[], &specs, 50.0, 50.0, &zoom);
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
        state.handle_click(&[], &specs, 0.0, 0.0, &zoom);
        assert!(matches!(state.selections.get("s"), Some(SelectionState::Empty)));
    }

    #[test]
    fn to_json_empty() {
        let state = InteractionState::new(&[point_spec("s")]);
        let json = state.to_json();
        assert!(json.contains("\"empty\""));
    }
}
