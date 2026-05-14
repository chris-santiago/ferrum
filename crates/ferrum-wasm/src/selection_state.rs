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
    ) {
        let hit = hit_test::hit_test(panels, x, y);

        for spec in specs {
            match spec {
                SelectionSpec::Point {
                    name, toggle, ..
                } => {
                    let sel = self.selections.entry(name.clone()).or_insert(SelectionState::Empty);
                    match &hit {
                        Some(h) => {
                            if let Some(data_idx) = h.data_idx {
                                let is_toggle = matches!(
                                    toggle,
                                    ferrum_scene::EventExpr::ShiftKey
                                );
                                if is_toggle {
                                    toggle_point(sel, data_idx);
                                } else {
                                    *sel = SelectionState::Point {
                                        indices: vec![data_idx],
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
    ) -> Option<&HitResult> {
        self.hover = hit_test::hit_test(panels, x, y);
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
    fn to_json_empty() {
        let state = InteractionState::new(&[point_spec("s")]);
        let json = state.to_json();
        assert!(json.contains("\"empty\""));
    }
}
