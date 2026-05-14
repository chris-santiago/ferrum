use serde::{Deserialize, Serialize};

use crate::types::Color;

// ── 3.7 SelectionSpec ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SelectionSpec {
    Point {
        name: String,
        fields: Option<Vec<String>>,
        encodings: Option<Vec<ChannelName>>,
        nearest: bool,
        toggle: EventExpr,
        on: EventExpr,
        clear: EventExpr,
        resolve: SelectionResolve,
    },
    Interval {
        name: String,
        fields: Option<Vec<String>>,
        encodings: Option<Vec<ChannelName>>,
        translate: bool,
        zoom: bool,
        mark: Option<SelectionMarkStyle>,
        resolve: SelectionResolve,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionResolve {
    Global,
    Union,
    Intersect,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChannelName {
    X,
    Y,
    Color,
    Size,
    Shape,
    Opacity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectionMarkStyle {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub fill_opacity: f64,
    pub stroke_opacity: f64,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EventExpr {
    Click,
    Mouseout,
    Mouseover,
    ShiftKey,
    Dblclick,
    Custom(String),
}

// ── 3.8 ConditionalEncoding ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalEncoding {
    pub selection_name: String,
    pub channel: ChannelName,
    pub if_selected: EncodingValue,
    pub if_not: EncodingValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EncodingValue {
    Color { value: Color },
    Opacity { value: f64 },
    Size { value: f64 },
    StrokeWidth { value: f64 },
    StrokeDash { value: Vec<f64> },
    Field { name: String },
}

// ── 3.8b FieldValue ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldValue {
    String { value: String },
    Number { value: f64 },
    Bool { value: bool },
    Null,
}

// ── 3.9 InteractionConfig ───────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InteractionConfig {
    pub zoom_enabled: bool,
    pub pan_enabled: bool,
    pub conditionals: Vec<ConditionalEncoding>,
    pub linked_panels: Vec<Vec<usize>>,
    pub tick_levels: Vec<PanelTickLevels>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PanelTickLevels {
    pub panel_id: usize,
    pub x_levels: Vec<TickLevel>,
    pub y_levels: Vec<TickLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TickLevel {
    pub min_zoom: f64,
    pub max_zoom: f64,
    pub ticks: Vec<Tick>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tick {
    pub value: f64,
    pub label: String,
    pub pixel: f64,
}
