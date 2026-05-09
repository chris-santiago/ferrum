use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum DataRef {
    Named { name: String },
}

impl Default for DataRef {
    fn default() -> Self {
        DataRef::Named { name: "default".into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_ref_named_round_trip() {
        let original = DataRef::Named { name: "my_table".into() };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"kind":"named","name":"my_table"}"#);
        let parsed: DataRef = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_data_ref_default() {
        let d = DataRef::default();
        assert_eq!(d, DataRef::Named { name: "default".into() });
    }
}
