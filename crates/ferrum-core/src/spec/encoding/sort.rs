//! The sort-field-object grammar shared by every `sort`-accepting slot
//! (`EncodingSpec.sort`, `FacetSpec.sort`/`row_sort`): the typed `op`/`order`
//! vocabularies and the one parser that turns a
//! `{"field": ..., "op": ..., "order": ...}` object into them. Construction
//! (`EncodingSpec::new`), the render-time refusal walk and render-time
//! resolution all parse through [`parse_sort_field_object`], so the accepted
//! shapes have a single definition.

/// Aggregate a sort-field object reduces each category's values with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortOp {
    Sum,
    Mean,
    Max,
    Min,
    Count,
    Median,
}

impl SortOp {
    /// Every accepted `"op"` spelling, aliases included (`"average"` for
    /// `Mean`), published as-is by `sort_accepted_vocabulary()`.
    /// [`parse`](Self::parse) matches these same literals directly;
    /// `every_accepted_spelling_parses` keeps the two from drifting apart.
    pub(crate) const ACCEPTED: &'static [&'static str] =
        &["sum", "mean", "average", "max", "min", "count", "median"];

    fn parse(s: &str) -> Option<Self> {
        match s {
            "sum" => Some(Self::Sum),
            "mean" | "average" => Some(Self::Mean),
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "count" => Some(Self::Count),
            "median" => Some(Self::Median),
            _ => None,
        }
    }

    /// Reduce a category's per-row values (already filtered to non-null) to a
    /// single ordering key. An empty slice yields `f64::NEG_INFINITY` so empty
    /// categories sort to the bottom under ascending order (and to the top under
    /// descending), which keeps them out of the visually-meaningful extremes.
    pub(crate) fn aggregate(self, values: &[f64]) -> f64 {
        if matches!(self, Self::Count) {
            return values.len() as f64;
        }
        if values.is_empty() {
            return f64::NEG_INFINITY;
        }
        match self {
            Self::Sum => values.iter().sum(),
            Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Self::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Self::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
            Self::Count => unreachable!("handled above"),
            Self::Median => {
                let mut sorted = values.to_vec();
                sorted
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                }
            }
        }
    }
}

/// Sort order for data-aware forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortOrder {
    Ascending,
    Descending,
}

impl SortOrder {
    /// Every accepted `"order"` spelling, published as-is by
    /// `sort_accepted_vocabulary()`; see [`SortOp::ACCEPTED`].
    pub(crate) const ACCEPTED: &'static [&'static str] = &["ascending", "descending"];

    fn parse(s: &str) -> Option<Self> {
        match s {
            "ascending" => Some(Self::Ascending),
            "descending" => Some(Self::Descending),
            _ => None,
        }
    }
}

/// Parse a sort-field object's three keys into their typed values, or the
/// reason the shape does not conform.
///
/// A well-formed object requires a string `"field"`; `"op"`, if present,
/// must be one of [`SortOp::ACCEPTED`]; `"order"`, if present, must be one
/// of [`SortOrder::ACCEPTED`]. Both default per the Vega-Lite convention
/// (`op` -> sum, `order` -> ascending).
pub(crate) fn parse_sort_field_object(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<(String, SortOp, SortOrder), String> {
    let field = match obj.get("field") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => return Err(format!("sort \"field\" must be a string, got {other}")),
        None => return Err("sort object missing string \"field\"".to_string()),
    };
    // `count` ignores the field's values but still requires the category column.
    let op = match obj.get("op") {
        None => SortOp::Sum,
        Some(serde_json::Value::String(s)) => {
            SortOp::parse(s).ok_or_else(|| format!("unsupported sort op {s:?}"))?
        }
        Some(other) => return Err(format!("sort \"op\" must be a string, got {other}")),
    };
    let order = match obj.get("order") {
        None => SortOrder::Ascending,
        Some(serde_json::Value::String(s)) => {
            SortOrder::parse(s).ok_or_else(|| format!("unsupported sort order {s:?}"))?
        }
        Some(other) => return Err(format!("sort \"order\" must be a string, got {other}")),
    };
    Ok((field, op, order))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: serde_json::Value) -> Result<(String, SortOp, SortOrder), String> {
        let serde_json::Value::Object(m) = value else {
            unreachable!()
        };
        parse_sort_field_object(&m)
    }

    /// Every published `op`/`order` spelling parses (including the
    /// `"average"` alias for `Mean`), and a bare `field` takes both defaults.
    #[test]
    fn every_accepted_spelling_parses() {
        assert_eq!(
            parse(serde_json::json!({"field": "v"})),
            Ok(("v".to_string(), SortOp::Sum, SortOrder::Ascending))
        );
        for op in SortOp::ACCEPTED {
            assert!(
                parse(serde_json::json!({"field": "v", "op": op})).is_ok(),
                "op {op:?}"
            );
        }
        for order in SortOrder::ACCEPTED {
            assert!(
                parse(serde_json::json!({"field": "v", "order": order})).is_ok(),
                "order {order:?}"
            );
        }
    }

    /// No `field`, a non-string `field`, an unknown `op` and an unknown
    /// `order` all refuse.
    #[test]
    fn every_malformed_shape_refuses() {
        let cases = [
            serde_json::json!({"op": "mean"}),
            serde_json::json!({"field": 1}),
            serde_json::json!({"field": "v", "op": "not-a-real-op"}),
            serde_json::json!({"field": "v", "order": "not-a-real-order"}),
        ];
        for case in cases {
            assert!(parse(case.clone()).is_err(), "{case} must refuse");
        }
    }
}
