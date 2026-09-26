//! The one API every closed string vocabulary enum shares (`AggFn`, its
//! declared subsets `HexAggregate`/`XEstimator`, `RasterAggregate`, and the
//! mark-side `Interpolate`).
//!
//! An implementor supplies its variants in declared order and each variant's
//! spellings (canonical first, then aliases); parsing, listing, the registry's
//! filtered lists and the `a|b|c` refusal text all derive from those two, so
//! every vocabulary answers the same questions the same way.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// A closed vocabulary of string spellings. See the module doc.
pub(crate) trait Vocabulary: Copy + PartialEq + 'static {
    /// Every variant, in declared order: the order every listing renders.
    const ALL: &'static [Self];

    /// This variant's accepted spellings: the canonical one first, then its
    /// declared aliases. Implement with an exhaustive match (no wildcard), so
    /// a variant without a spelling fails to compile.
    fn spellings(self) -> &'static [&'static str];

    /// The canonical spelling: what the wire writes and what an alias
    /// normalizes to.
    fn as_str(self) -> &'static str {
        self.spellings()[0]
    }

    /// Parse any spelling, aliases included, to its variant.
    fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|v| v.spellings().contains(&s))
    }

    /// The spellings (aliases included, right after their canonical name) of
    /// every variant `keep` selects, in [`Vocabulary::ALL`] order. How the
    /// registry derives `methods_needing_field`.
    fn all_where(keep: impl Fn(Self) -> bool) -> Vec<&'static str> {
        Self::ALL
            .iter()
            .copied()
            .filter(|&v| keep(v))
            .flat_map(|v| v.spellings().iter().copied())
            .collect()
    }

    /// Every accepted spelling, in [`Vocabulary::all_where`] order.
    fn all() -> Vec<&'static str> {
        Self::all_where(|_| true)
    }

    /// The `a|b|c` rendering of [`Vocabulary::all`] every refusal uses.
    fn accepted_values() -> String {
        Self::all().join("|")
    }

    /// Parse `s`, or the refusal body naming the parameter and the value
    /// received: `unknown <param> '<s>'; expected a|b|c`. The caller prefixes
    /// the door it guards (`Hex: `, `stat_hex: `, ...).
    fn parse_or_refuse(param: &str, s: &str) -> Result<Self, String> {
        Self::parse(s).ok_or_else(|| Self::refusal(param, s))
    }

    /// The refusal body for a value outside the vocabulary:
    /// `unknown <param> '<s>'; expected a|b|c`. For a door that carries the
    /// rejected value in a typed error and renders the text later.
    fn refusal(param: &str, s: &str) -> String {
        format!(
            "unknown {param} '{s}'; expected {}",
            Self::accepted_values()
        )
    }
}

/// An aggregate vocabulary whose door reads a `field` operand for some
/// spellings and runs without one for the rest (`HexAggregate`,
/// `RasterAggregate`).
pub(crate) trait FieldOperand: Vocabulary {
    /// Whether this aggregate reads a `field`. Implement with an exhaustive
    /// match (or an exhaustive fact), so a new variant must decide.
    fn needs_field(self) -> bool;
}

/// The needs-field refusal body shared by the [`FieldOperand`] doors.
/// `spelling` is the value the caller wrote (the canonical spelling for a
/// wire-built spec).
pub(crate) fn require_field<V: FieldOperand>(
    aggregate: V,
    spelling: &str,
    field: Option<&str>,
) -> Result<(), String> {
    if aggregate.needs_field() && field.is_none() {
        return Err(format!("aggregate='{spelling}' requires field"));
    }
    Ok(())
}

/// The pre-check a string-stored vocabulary runs at every door before its
/// dispatch `match`, so an arm accepting an undeclared spelling is
/// unreachable and the door accepts exactly `declared`. Refuses as
/// `<door>: unknown <param> '<s>'; expected a|b|c`, the rendering
/// [`Vocabulary::parse_or_refuse`] uses.
pub(crate) fn check_declared(declared: &[&str], door: &str, param: &str, s: &str) -> PyResult<()> {
    if declared.contains(&s) {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{door}: unknown {param} '{s}'; expected {}",
        declared.join("|")
    )))
}
