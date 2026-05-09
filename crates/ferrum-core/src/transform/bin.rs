use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BinSpec {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(default = "default_true")]
    pub nice: bool,
}

fn default_true() -> bool { true }
