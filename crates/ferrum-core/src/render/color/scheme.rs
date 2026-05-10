//! Unified Scheme enum: dispatch between categorical and continuous.
//!
//! `CategoricalPalette` is introduced here as scaffolding for 8b+ tasks that need
//! to reference a named categorical palette symbolically. The lookup into the actual
//! color slice is handled by `crate::render::palette::categorical_palette`.

use crate::render::color::continuous::ContinuousScheme;

/// A reference to a named categorical palette (e.g., "okabe_ito", "tableau10").
/// Resolved to a `&'static [Color]` at render time via `render::palette::categorical_palette`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoricalPalette {
    pub name: String,
}

impl CategoricalPalette {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scheme {
    Categorical(CategoricalPalette),
    Continuous(ContinuousScheme),
}
