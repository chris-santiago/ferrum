//! The positional axis a spec NAMES — one type for every wire field whose
//! value is the token `"x"` or `"y"`.
//!
//! Relocated here from `spec/composite.rs` (`#229`(2)), where it was born as
//! `ExtentAlignment`'s own axis, so the second wire field with the same two
//! values — `render::chart_config::BreakAxisSpec::axis`, which was a `String`
//! compared against `"y"` at a dozen sites — consumes the SAME enum rather than
//! a near-identical twin. The name is the cross-task contract's
//! ([`AlignAxis`]); it reads as "the axis this declaration aligns/applies to",
//! not as "extent alignment only".
//!
//! # What this is NOT
//!
//! Three vocabularies in this crate share the token `"y"`, and only this one is
//! an axis a spec declares:
//!
//! - **Channel names** (`render::marks::axis::AxisNodes::placed_along`, the
//!   `scale_resolve`/`prepare` channel tokens, all from
//!   `Dimension::channel_token()`) name which ENCODING CHANNEL a value came
//!   from, a larger space than two (`x2`, `y2`, `theta`, …). A break's axis is
//!   compared against one of those in two places, and those comparisons go
//!   through [`AlignAxis::wire_name`] deliberately — crossing the vocabularies
//!   explicitly rather than by unifying them.
//! - **`render::annotation::AnnotationSpec::Span.axis`** is a different wire
//!   field on a different spec, with its own accepted-keys entry, and is left
//!   alone.

use serde::{Deserialize, Serialize};

/// The positional axis a spec names: the `axis` of a
/// [`ExtentAlignment`](super::composite::ExtentAlignment) (`X` makes two cells
/// in one grid COLUMN span the same pixels horizontally; `Y` does the same for
/// two cells in one ROW) and the `axis` of a
/// [`BreakAxisSpec`](crate::render::chart_config::BreakAxisSpec) (which
/// coordinate the break removes a range from).
///
/// Two variants, `Copy`, and exhaustively matched at every consumer, so "the
/// other one is x" is a fact the compiler keeps rather than an `else` branch
/// each site re-derives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlignAxis {
    X,
    Y,
}

impl AlignAxis {
    /// Every variant, in wire order — the list a refusal message enumerates.
    ///
    /// Deliberately NOT what decides whether a token is accepted: that is
    /// [`AlignAxis::from_wire_name`], which delegates to `Deserialize`. The
    /// split matters because the two failure modes are not equally bad. A
    /// variant missing from `ALL` under-reports a token in a message, which is
    /// cosmetic; a variant missing from the ACCEPT path would make the wire
    /// gate refuse input the renderer would have honored, which is a refusal of
    /// valid data. `ALL` being a fixed-length array that a new variant does not
    /// break is therefore tolerable here and would not have been there.
    pub const ALL: [AlignAxis; 2] = [AlignAxis::X, AlignAxis::Y];

    /// The variant `token` spells, or `None` for a token outside the
    /// vocabulary — the inverse of [`AlignAxis::wire_name`].
    ///
    /// **Delegates to `Deserialize`** rather than comparing against a list, so
    /// the accept-set of the wire gate built on it *is* serde's accept-set by
    /// construction. A hand-rolled comparison (the first cut of this method
    /// scanned [`AlignAxis::ALL`]) is only as complete as the list it scans: a
    /// variant added to the enum — which `Deserialize` picks up for free — but
    /// not added to that list would make the gate refuse a token the renderer
    /// would then have honored, and no test over a fixed token corpus can be
    /// relied on to notice, since the new token need not be in the corpus.
    /// Delegation removes the possibility instead of testing for it.
    ///
    /// Takes the STRING spelling of a unit variant. `Deserialize` also accepts
    /// the single-key map spelling (`{"y": null}`); the wire gate reduces that
    /// form to its key and asks here, so both spellings are validated against
    /// one vocabulary (see `binding.rs::validate_break_axis_axis_value`).
    pub fn from_wire_name(token: &str) -> Option<Self> {
        use serde::de::IntoDeserializer;
        let de: serde::de::value::StrDeserializer<'_, serde::de::value::Error> =
            token.into_deserializer();
        Self::deserialize(de).ok()
    }

    /// The accepted tokens as a refusal message renders them: `'x', 'y'`.
    pub fn wire_token_list() -> String {
        Self::ALL
            .iter()
            .map(|a| format!("'{}'", a.wire_name()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The axis as it appears on the wire — and, for that reason, as it
    /// appears in the warning a caller reads back
    /// (`RenderWarning::ExtentAlignmentUnresolvable`): the token it is told
    /// about is the token it wrote.
    ///
    /// Also the ONE conversion that may cross into the channel vocabulary (see
    /// the module doc): `scene_build`'s two `placed_along` comparisons read
    /// this rather than retyping `channel_token()`.
    pub fn wire_name(&self) -> &'static str {
        match self {
            AlignAxis::X => "x",
            AlignAxis::Y => "y",
        }
    }

    /// Whether this is the y axis, for a consumer whose two branches are the
    /// two axis orientations and which therefore treats the other answer as
    /// "x".
    ///
    /// Eight production call sites, enumerated so the count stays checkable
    /// against the code rather than drifting in prose:
    ///
    /// - the three break-indicator glyph builders (`render::break_axis::
    ///   indicators`' slash, zigzag and wave);
    /// - `render::break_axis::BreakRemap::is_y`, which forwards to this and is
    ///   itself read by that module's along/across geometry;
    /// - `render::annotation::data_coords_on_axis`, choosing which coordinate
    ///   pair a break's suppression rule tests;
    /// - `render::scene_build::break_grid_node`'s rect-ends restore;
    /// - the two secondary-y (`y_slot >= 1`) break exemptions in
    ///   `render::scene_build` (axis chrome and mark batches).
    ///
    /// ONE exhaustive match answers it for all of them, instead of each site
    /// spelling its own `else`. Be precise about what that buys: a third
    /// variant would compile-error in THIS match, forcing one deliberate
    /// decision about what it answers — but whatever the match then answers,
    /// all eight consumers still inherit silently as "not y", because their
    /// second branch is an `else`. So this centralizes the else-is-x
    /// asymmetry; it does not remove it. If a third axis ever lands, the
    /// honest follow-through is a `match` at the geometry sites (which the
    /// retype makes possible — `axis == "y"` no longer compiles anywhere), not
    /// a second predicate beside this one.
    pub fn is_y(&self) -> bool {
        match self {
            AlignAxis::Y => true,
            AlignAxis::X => false,
        }
    }
}

/// Emits the wire token, so a warning message interpolating `{axis}` reads
/// `on the y axis` exactly as it did when the field was a `String`.
///
/// Deliberately NOT `{:?}`: `Debug` prints the Rust variant (`Y`), and the one
/// message that wants the token QUOTED
/// (`RenderError::BreakAxisIncompatibleCoord`'s `axis="y"`) gets its quotes by
/// applying `{:?}` to [`AlignAxis::wire_name`]'s `&str`, not to the enum.
impl std::fmt::Display for AlignAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire tokens both consuming specs deserialize from, unchanged by the
    /// relocation.
    #[test]
    fn serde_round_trips_the_lowercase_wire_tokens() {
        assert_eq!(serde_json::to_string(&AlignAxis::X).unwrap(), "\"x\"");
        assert_eq!(serde_json::to_string(&AlignAxis::Y).unwrap(), "\"y\"");
        assert_eq!(
            serde_json::from_str::<AlignAxis>("\"y\"").unwrap(),
            AlignAxis::Y
        );
        assert!(serde_json::from_str::<AlignAxis>("\"z\"").is_err());
    }

    /// The delegation is the guarantee; this checks the delegation itself.
    ///
    /// [`AlignAxis::from_wire_name`] routes through `Deserialize`, so its
    /// accept-set IS serde's — not by agreement but by construction, and a
    /// variant added to this enum is picked up by both at once. That is
    /// deliberately NOT what an earlier cut relied on: it scanned
    /// [`AlignAxis::ALL`], a fixed-length array a new variant does not break,
    /// and leaned on a hardcoded token corpus to notice the divergence — which
    /// it could not be relied on to do, since a new variant's token need not be
    /// in the corpus. The corpus below is therefore no longer load-bearing for
    /// completeness. What it still earns: proof that the delegation is wired
    /// correctly at all (a token quoted, escaped or trimmed wrongly on its way
    /// into the deserializer would diverge here), over the shapes the wire gate
    /// actually forwards.
    #[test]
    fn from_wire_name_delegates_to_deserialize() {
        for token in ["x", "y", "z", "X", "Y", "", " x", "theta", "x2", "y2", "\"x\""] {
            let via_serde = serde_json::from_value::<AlignAxis>(serde_json::Value::String(
                token.to_string(),
            ))
            .ok();
            assert_eq!(
                AlignAxis::from_wire_name(token),
                via_serde,
                "delegation diverged from Deserialize on {token:?}"
            );
        }
    }

    /// `ALL` and the message it renders. A variant missing here under-reports a
    /// token in a refusal message, which is cosmetic — the ACCEPT decision does
    /// not come from this list (see
    /// [`from_wire_name_delegates_to_deserialize`]) — so this asserts the list
    /// is self-consistent rather than pretending to be a completeness guard.
    #[test]
    fn all_round_trips_its_tokens_and_renders_the_message_list() {
        for axis in AlignAxis::ALL {
            let expected = match axis {
                AlignAxis::X => "x",
                AlignAxis::Y => "y",
            };
            assert_eq!(axis.wire_name(), expected);
            assert_eq!(AlignAxis::from_wire_name(expected), Some(axis));
        }
        assert_eq!(AlignAxis::wire_token_list(), "'x', 'y'");
    }

    /// The one place "the other axis is x" is decided.
    #[test]
    fn is_y_is_true_for_y_and_false_for_x() {
        assert!(AlignAxis::Y.is_y());
        assert!(!AlignAxis::X.is_y());
    }

    /// `Display` is the token, not the variant — this is what keeps the six
    /// `{axis}` break-family warning arms byte-identical to when `axis` was a
    /// `String`.
    #[test]
    fn display_emits_the_wire_token_and_debug_does_not() {
        assert_eq!(AlignAxis::X.to_string(), "x");
        assert_eq!(AlignAxis::Y.to_string(), "y");
        assert_eq!(format!("on the {} axis", AlignAxis::Y), "on the y axis");
        // The trap this type's consumers must avoid: `{:?}` on the enum is the
        // variant name, which is why `BreakAxisIncompatibleCoord` formats
        // `wire_name()` instead (see `AlignAxis`'s `Display` doc).
        assert_eq!(format!("{:?}", AlignAxis::Y), "Y");
        assert_eq!(format!("{:?}", AlignAxis::Y.wire_name()), "\"y\"");
    }
}
