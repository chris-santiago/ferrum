//! Phase 7 — static renderer. Pure functions: ChartSpec + RecordBatch + ThemeInputs +
//! Viewport -> deterministic SVG/PNG. See docs/superpowers/specs/2026-05-09-static-renderer-design.md.

pub(crate) mod annotation;
pub(crate) mod arrow_cast;
pub(crate) mod chart_config;
pub(crate) mod config;
pub(crate) mod color;
pub(crate) mod font;
pub(crate) mod format;
pub(crate) mod svg;
pub(crate) mod embed_font;
pub(crate) mod scale_resolve;
pub(crate) mod prepare;
pub(crate) mod rasterize;
pub(crate) mod draw;
pub(crate) mod png;
pub(crate) mod binding;
pub(crate) mod mark_nodes;
pub(crate) mod marks;
pub(crate) mod position;
pub(crate) mod figure_chrome;
pub(crate) mod pack_instances;
pub(crate) mod composite;
pub(crate) mod composite_render;
pub(crate) mod scene_build;
pub(crate) mod svg_walk;
pub(crate) mod break_axis;
pub(crate) mod inset;

// Constants (spec §6.1).
pub const FLOAT_PRECISION: usize = 3;
pub const CLIP_ID_PREFIX: &str = "ferrum-clip-";

use serde::{Deserialize, Serialize};

use crate::layout::LayoutWarning;

/// The user-facing detail carried by [`RenderError::PositionAdjustFailed`]
/// (R3 restructure, #89 part C). Most Dodge/Jitter/Stack failures carry a
/// free-form [`Message`](Self::Message) with no positional channel token to
/// un-flip. Stack's four failure modes that DO name a resolved `x`/`y`
/// channel (missing category/value encoding, bad value/category dtype)
/// carry the RESOLVED token structurally instead, so `Display` — not the
/// `position::apply_stack` constructor — un-flips it via
/// `prepare::user_facing_channel` using the `coord_flipped` carried
/// alongside on the error. This replaces the former construction-time bake,
/// where `apply_stack` computed already-un-flipped `cat_token`/`value_token`
/// strings via `user_facing_channel` before building the message.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PositionAdjustReason {
    /// "<role> (<channel>) encoding required" — Stack's missing
    /// category/value encoding pair. `role` is `"category"` or `"value"`.
    MissingEncoding { role: &'static str, channel: &'static str },
    /// "<channel> must be Float64, UInt64, or a signed integer type
    /// (Int8/Int16/Int32/Int64); got <dtype>" — Stack's value-dtype check.
    /// `dtype` is pre-formatted (`{:?}` of the Arrow `DataType`) since it
    /// carries no channel token to un-flip.
    ValueDtype { channel: &'static str, dtype: String },
    /// "<channel> column must be Float64 or Utf8" — Stack's category-dtype
    /// check.
    CategoryDtype { channel: &'static str },
    /// A free-form reason with no positional channel token — every
    /// Dodge/Jitter message, plus Stack's column-not-found and
    /// `RecordBatch`-construction failures.
    Message(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RenderError {
    InvalidViewport { width: f64, height: f64 },
    EmptyBatch,
    UnknownColumn { name: String },
    /// A user-supplied color string that `render::color::parse_color` rejects.
    /// The payload is the `ColorParseError` `parse_color` returned; `Display`
    /// delegates to it verbatim (which itself pairs the offending value with
    /// `ACCEPTED_COLOR_FORMS`, spec §6), so the wrapper sentence exists in
    /// exactly one place rather than being restated here. Raised by
    /// `draw::resolve_mark_style` for `fill=`/`stroke=` mark kwargs — the
    /// color boundaries that must refuse rather than silently keep the theme
    /// default.
    InvalidColor(color::primitive::ColorParseError),
    /// `channel` is the RESOLVED (post-`CoordFlip`) token the caller acted on —
    /// `coord_flipped` (R3) lets `Display` un-flip it back to what the user
    /// wrote; non-positional channels (e.g. `"color"`) are unaffected either way.
    EncodingTypeMismatch { channel: &'static str, expected: &'static str, got: String, coord_flipped: bool },
    TransformFailed(String),
    ScaleResolutionFailed(String),
    LayoutFailed(String),
    ResvgFailed(String),
    /// A position-adjustment pass (Dodge/Jitter/Stack) rejected its inputs.
    /// `adjustment` names the adjustment; `reason` is the user-facing detail
    /// (see [`PositionAdjustReason`]). `coord_flipped` (R3) is a
    /// `Display`-time-only correction: it is read only by
    /// [`PositionAdjustReason::MissingEncoding`],
    /// [`PositionAdjustReason::ValueDtype`], and
    /// [`PositionAdjustReason::CategoryDtype`] (which carry a channel token
    /// to un-flip) and is inert for [`PositionAdjustReason::Message`] (no
    /// token). Every construction site sets it to the real `coord_flipped`
    /// where that context is available (`apply_dodge`, `apply_stack`) and to
    /// `false` where it isn't (`apply_dodge_ordinal`, `apply_jitter` — both
    /// only ever construct token-free `Message` reasons, so the field is
    /// structurally inert there).
    PositionAdjustFailed { adjustment: &'static str, reason: PositionAdjustReason, coord_flipped: bool },
    /// A column carried an Arrow dtype the renderer cannot interpret.
    /// `field` is the column name; `context` is an optional channel /
    /// scale tag (e.g. `"size"`, `"opacity"`, `"scale"`) used to
    /// disambiguate when the same column feeds multiple resolution
    /// passes. Display: `"column '<field>' has unsupported dtype: <dtype>"`
    /// or `"<context>: column '<field>' has unsupported dtype: <dtype>"`.
    UnsupportedDtype { field: String, dtype: String, context: Option<&'static str> },
    /// The unioned numeric/temporal extent for an axis or color channel
    /// produced no finite values (all rows null/NaN or empty after filter).
    EmptyDomain { channel: String, field: String },
    SceneConstruction(String),
    HtmlBundleAssembly(String),
    /// An encoding channel that is not supported by the given mark type was
    /// supplied.  `mark` names the mark (e.g. `"mark_area"`); `channel` names
    /// the unsupported channel (e.g. `"x2"`); `hint` is an actionable suggestion
    /// pointing at the correct mark or channel to use instead.
    UnsupportedChannelCombination {
        mark: &'static str,
        channel: &'static str,
        /// Hint template. A literal `"{alt}"` placeholder, when present, is
        /// substituted with `hint_alt_channel` (un-flipped like `channel` —
        /// R3); hints with no swappable channel (e.g. "both x2= and y2=",
        /// which is flip-symmetric) carry no placeholder.
        hint: &'static str,
        hint_alt_channel: Option<&'static str>,
        /// `Some(coord)` under `CoordFlip` — R3: `channel`/`hint_alt_channel`
        /// are the RESOLVED (post-flip) tokens the validation acted on;
        /// `Display` un-flips them back to what the user wrote.
        coord_flipped: bool,
    },
    /// A per-channel / chart-level `axis(orient=...)` named a side that does not
    /// match the channel's dimension. `channel` is `"x"` or `"y"` — the RESOLVED
    /// (post-`CoordFlip`) geometric axis the validation actually acted on, never
    /// un-flipped itself (top/bottom are only ever valid for the physical x
    /// axis). `orient` is the rejected value. x accepts `top`/`bottom`; y
    /// accepts `left`/`right`. `coord_flipped` (R3) is a `Display`-time-only
    /// correction un-flipping `channel` in the rendered message so it names
    /// the axis token the user actually wrote.
    ///
    /// THREE call chains reach this constructor, and only TWO patch
    /// `coord_flipped` in — the third is a deliberate EXEMPTION, not a gap:
    /// - `prepare::build_axes` (per-channel `fm.Axis(orient=...)`) — PATCHES.
    ///   The `EncodingSpec` carrying the override travels through
    ///   `build_layers`' swap with its channel, so the resolved token needs
    ///   translating back to what the user wrote.
    /// - `prepare::build_secondary_y_axis_inputs` (independent-y layer's own
    ///   `fm.Axis(orient=...)`) — PATCHES, same reasoning as above.
    /// - the chart-level `configure_axis` apply block
    ///   (`apply_axis_config_to_axis_input`, in `prepare_and_layout`) — EXEMPT.
    ///   `channel` there is derived from the axis's PHYSICAL orientation
    ///   (`axis_channel`), never from an `EncodingSpec`, and the config key
    ///   the user actually typed (`axis_x`/`axis_y`) is itself RESOLVED-slot
    ///   vocabulary that Python never remaps under flip — so the resolved
    ///   token already IS what the user wrote; un-flipping it would say the
    ///   opposite of the config key they typed (mirrors the `SortSpecIgnored`
    ///   exemption in `scale_resolve/domain.rs`).
    InvalidAxisOrient { channel: &'static str, orient: String, coord_flipped: bool },
}

/// Boundary correction for [`RenderError::InvalidAxisOrient`] (R3). The error
/// is constructed deep inside `prepare::parse_axis_orient`/`parse_title_orient`,
/// which have no access to whether the chart is flipped — they return it
/// wrapped in [`UnflippedRenderError`], whose sole exit,
/// [`UnflippedRenderError::resolve`], calls this fn. See the field doc on
/// [`RenderError::InvalidAxisOrient`] for the full three-chain account: TWO of
/// the three production call chains resolve this at their own boundary
/// (`prepare::build_axes`, `prepare::build_secondary_y_axis_inputs`); the third
/// (the chart-level `configure_axis` apply block) is a deliberate exemption and
/// resolves with `false` explicitly — `channel`/`orient` (the resolved token
/// validation acted on) are untouched either way; only the `Display`-time
/// un-flip flag is corrected. A no-op for every other variant.
///
/// MAINTENANCE: [`UnflippedRenderError`] compile-enforces that every call
/// chain reaching `parse_axis_orient`/`parse_title_orient` makes this
/// decision explicitly — it cannot become a `RenderError` (and so cannot
/// propagate via `?`) without a `.resolve(coord_flipped)` call somewhere in
/// the chain. Decide whether that chain's `channel` names a user-written
/// encoding channel (patch it with the real flag, like the two chains above)
/// or a resolved/physical/user-typed-literal token (resolve with `false`,
/// like the chart-level chain and `SortSpecIgnored`). This decision was
/// missed once already (the `build_secondary_y_axis_inputs` chain went
/// unpatched through cycle 1, when the placeholder discipline was prose-only)
/// — it must still be made deliberately at review time, not assumed, but a
/// missed call chain now fails to compile instead of silently defaulting.
pub(crate) fn with_coord_flipped(err: RenderError, coord_flipped: bool) -> RenderError {
    match err {
        RenderError::InvalidAxisOrient { channel, orient, .. } => {
            RenderError::InvalidAxisOrient { channel, orient, coord_flipped }
        }
        // Every other variant is listed explicitly (rather than a catch-all)
        // so that a future variant added to `RenderError` is a compile error
        // here, not a silent no-op. `EncodingTypeMismatch`,
        // `UnsupportedChannelCombination`, and `PositionAdjustFailed` also
        // carry a `coord_flipped` field, but each is patched at its own
        // construction site, not through this boundary-correction fn — see
        // their field docs on `RenderError`. The remaining variants carry no
        // user-channel token at all.
        other @ (RenderError::InvalidViewport { .. }
        | RenderError::EmptyBatch
        | RenderError::UnknownColumn { .. }
        | RenderError::InvalidColor(_)
        | RenderError::EncodingTypeMismatch { .. }
        | RenderError::TransformFailed(_)
        | RenderError::ScaleResolutionFailed(_)
        | RenderError::LayoutFailed(_)
        | RenderError::ResvgFailed(_)
        | RenderError::PositionAdjustFailed { .. }
        | RenderError::UnsupportedDtype { .. }
        | RenderError::EmptyDomain { .. }
        | RenderError::SceneConstruction(_)
        | RenderError::HtmlBundleAssembly(_)
        | RenderError::UnsupportedChannelCombination { .. }) => other,
    }
}

/// Wrapper forcing every call chain reaching
/// [`prepare::parse_axis_orient`]/[`prepare::parse_title_orient`] to
/// explicitly decide the `coord_flipped` `Display` correction (R3) before the
/// error can become a [`RenderError`] — compile-enforcing what the
/// MAINTENANCE note on [`with_coord_flipped`] used to police only as prose
/// (see that note for the cycle-1 near-miss this closes). `#[must_use]` and
/// the deliberate absence of `From<UnflippedRenderError> for RenderError`
/// both matter: a new call chain cannot propagate the error via `?` until it
/// calls [`resolve`](Self::resolve).
#[must_use]
#[derive(Debug)]
pub(crate) struct UnflippedRenderError(RenderError);

impl UnflippedRenderError {
    /// Wrap a freshly constructed [`RenderError`] whose `coord_flipped`
    /// field has not yet been decided at this boundary.
    pub(crate) fn new(err: RenderError) -> Self {
        Self(err)
    }

    /// The sole exit. `coord_flipped` is the real flip flag for a call chain
    /// whose `EncodingSpec` traveled through `build_layers`' swap (patches
    /// the un-flip); `false` for a call chain whose channel is already
    /// resolved-slot/physical vocabulary the user typed directly — see the
    /// three-chain account on [`RenderError::InvalidAxisOrient`].
    pub(crate) fn resolve(self, coord_flipped: bool) -> RenderError {
        with_coord_flipped(self.0, coord_flipped)
    }
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidViewport { width, height } =>
                write!(f, "invalid viewport: width={width}, height={height} (both must be > 0)"),
            Self::EmptyBatch =>
                write!(f, "input batch is empty (num_rows == 0)"),
            Self::UnknownColumn { name } =>
                write!(f, "unknown column '{name}' referenced by an encoding"),
            // Delegates to `ColorParseError`'s own `Display` rather than
            // restating the wrapper sentence, so this message and the
            // `ferrum.color.to_hex` one can never drift (spec §6): the
            // sentence quoting `ACCEPTED_COLOR_FORMS` exists in exactly one
            // place, `ColorParseError`'s impl.
            Self::InvalidColor(e) => write!(f, "{e}"),
            Self::EncodingTypeMismatch { channel, expected, got, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                write!(f, "encoding '{channel}' expected {expected}, got {got}")
            }
            Self::TransformFailed(s) =>
                write!(f, "transform failed: {s}"),
            Self::ScaleResolutionFailed(s) =>
                write!(f, "scale resolution failed: {s}"),
            Self::LayoutFailed(s) =>
                write!(f, "layout failed: {s}"),
            Self::ResvgFailed(s) =>
                write!(f, "PNG rasterization failed: {s}"),
            Self::PositionAdjustFailed { adjustment, reason, coord_flipped } => match reason {
                PositionAdjustReason::Message(s) => write!(f, "{adjustment}: {s}"),
                PositionAdjustReason::MissingEncoding { role, channel } => {
                    let channel = prepare::user_facing_channel(channel, *coord_flipped);
                    write!(f, "{adjustment}: {role} ({channel}) encoding required")
                }
                PositionAdjustReason::ValueDtype { channel, dtype } => {
                    let channel = prepare::user_facing_channel(channel, *coord_flipped);
                    write!(
                        f,
                        "{adjustment}: {channel} must be Float64, UInt64, or a signed integer \
                         type (Int8/Int16/Int32/Int64); got {dtype}"
                    )
                }
                PositionAdjustReason::CategoryDtype { channel } => {
                    let channel = prepare::user_facing_channel(channel, *coord_flipped);
                    write!(f, "{adjustment}: {channel} column must be Float64 or Utf8")
                }
            },
            Self::UnsupportedDtype { field, dtype, context } => match context {
                Some(ctx) => write!(f, "{ctx}: column '{field}' has unsupported dtype: {dtype}"),
                None => write!(f, "column '{field}' has unsupported dtype: {dtype}"),
            },
            Self::EmptyDomain { channel, field } =>
                write!(f, "{channel}: no usable values found for field '{field}'"),
            Self::SceneConstruction(msg) =>
                write!(f, "scene construction failed: {msg}"),
            Self::HtmlBundleAssembly(msg) =>
                write!(f, "HTML bundle assembly failed: {msg}"),
            Self::UnsupportedChannelCombination { mark, channel, hint, hint_alt_channel, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                let hint: std::borrow::Cow<str> = match hint_alt_channel {
                    Some(alt) => std::borrow::Cow::Owned(
                        hint.replacen("{alt}", prepare::user_facing_channel(alt, *coord_flipped), 1),
                    ),
                    None => std::borrow::Cow::Borrowed(*hint),
                };
                write!(f, "{mark}: channel '{channel}' is not supported; {hint}")
            }
            Self::InvalidAxisOrient { channel, orient, coord_flipped } => {
                // `allowed`/`physical` reflect the RESOLVED axis (`channel`,
                // pre-un-flip): top/bottom validity is a GEOMETRIC property of
                // an axis drawn horizontally, left/right of one drawn
                // vertically — a property of the physical slot, not of
                // whichever letter names it. `CoordFlip` is implemented as a
                // data swap (`prepare::build_layers`), never a physical
                // re-orientation, so this constraint never changes with flip.
                let (allowed, physical) =
                    if *channel == "x" { ("'top' or 'bottom'", "horizontal") } else { ("'left' or 'right'", "vertical") };
                let user_channel = prepare::user_facing_channel(channel, *coord_flipped);
                if *coord_flipped {
                    // The un-flipped NAME and the physical constraint now
                    // disagree (e.g. "the y axis" paired with "top or
                    // bottom") — spell out why so the message reads as an
                    // explained fact, not an internal contradiction.
                    write!(
                        f,
                        "axis orient '{orient}' is invalid for the {user_channel} axis \
                         (expected {allowed} — under CoordFlip, {user_channel} renders as the {physical} axis)"
                    )
                } else {
                    write!(
                        f,
                        "axis orient '{orient}' is invalid for the {user_channel} axis (expected {allowed})"
                    )
                }
            }
        }
    }
}

impl std::error::Error for RenderError {}

/// Warnings emitted during render. Geometric edge cases or wrapped layout warnings.
///
/// Note (2026-05-09): spec §11 (line ~556) used `#[serde(tag = "kind", ...)]` but
/// that collides with `LayoutWarning`'s own `kind` tag when wrapped via
/// `RenderWarning::Layout(LayoutWarning)` (serde flattens newtype-around-struct
/// variants). Outer tag renamed to `type` to disambiguate; `LayoutWarning`'s
/// `kind` tag is preserved (already pinned by Phase 6 round-trip tests).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RenderWarning {
    Layout(LayoutWarning),
    OutOfDomainRows { mark: String, count: u64 },
    ColorPaletteOverflowed { categories: u32 },
    ShapePaletteOverflowed { categories: u32 },
    /// A categorical `stroke_dash` channel named more categories than the dash
    /// index space has slots (solid plus every
    /// [`DASH_PALETTE`](crate::render::draw::DASH_PALETTE) pattern), so patterns
    /// were recycled and two categories draw with the same dash. Mirrors
    /// [`ShapePaletteOverflowed`](Self::ShapePaletteOverflowed) — the palette is
    /// finite, so the degradation stays (spec §4.3) but is never silent.
    StrokeDashPaletteOverflowed { categories: u32 },
    EmptyPanel { panel_index: usize },
    /// An explicit color range string could not be parsed. The entire range is
    /// discarded and the default theme palette is used instead.
    /// `entry` is the offending color string.
    ColorRangeParseFailure { entry: String },
    /// A chart-level `configure_color(range=…)` named a different number of
    /// colors than the discretizing (Quantize/Quantile/Threshold/BinOrdinal)
    /// color scale has buckets. The bucket count is fixed by the scale's own
    /// thresholds, so the override cannot describe the partition and the
    /// resolved swatches stand (spec §4.2, amended 2026-08-28: reported, never
    /// silently dropped).
    ColorRangeBucketCountMismatch { expected: u32, received: u32 },
    /// A chart-level `configure_color(domain=…)` left some of the data's
    /// categories unlisted. Their marks still draw — in the theme mark color,
    /// with no legend entry — which is the sanctioned behavior but is otherwise
    /// indistinguishable from a rendering bug, so the omitted categories are
    /// named here (spec §4.2, amended 2026-08-28).
    ColorDomainOmitsCategories { categories: Vec<String> },
    /// A data-aware `sort` spec (channel shorthand `"-y"` or a sort-field
    /// object) could not be resolved — the referenced field is missing from the
    /// batch, has an unsupported dtype, or the spec is otherwise malformed. The
    /// categorical domain falls back to insertion order; `reason` explains why.
    SortSpecIgnored { reason: String },
    /// A position adjustment was requested with a grouping channel that could
    /// not yield categories — the named `by`/color column is absent from the
    /// data, or its dtype cannot be turned into category keys (e.g. timestamp /
    /// duration). The marks are left un-offset rather than crashing or silently
    /// no-op-ing. `adjustment` is the adjustment name (e.g. `"dodge"`); `reason`
    /// explains which column failed and why.
    PositionAdjustSkipped { adjustment: String, reason: String },
    /// An overlay group's layers reserved chrome gutters (legend strips, axis
    /// bands, title band) that between them leave no common plot area, so the
    /// per-side-max intersection degenerated and no shared rect could be
    /// imposed (GH #89A). Every layer then keeps its own geometry AND its own
    /// chrome — the honest degradation, since deduplicating chrome would leave
    /// the surviving axes describing a rect the other layers' marks never
    /// used. `layers` is the size of the affected group. Emitted because a
    /// doubled-chrome chart is otherwise hard to attribute: the cause is the
    /// layers' own gutter requests, not the overlay.
    OverlayGuttersDiverged { layers: usize },
    /// A `Continuous`/`Discretizing` (numeric-keyed) color scale was resolved
    /// while EVERY layer consuming it draws with `Mark::Line`/`Mark::Ribbon`
    /// — the two stroke-continuous marks that cannot paint a per-value color
    /// today (spec §4.0, amended 2026-08-28). The channel would otherwise be
    /// silently inert under a colorbar that promises a mapping nothing on
    /// the chart honors, so `render/prepare/legend.rs` suppresses that
    /// colorbar and emits this instead. `marks` names the affected mark(s)
    /// (deduped, encounter order — e.g. `["line"]`, or `["line", "ribbon"]`
    /// for a `mark_smooth(ci=True)`-style band+line pair sharing one field);
    /// `scale_kind` is `"continuous"` or `"discretizing"`. A mixed chart
    /// where another layer's mark (e.g. `point`) shares the same scale does
    /// NOT reach this variant — that layer genuinely renders the mapping, so
    /// the colorbar stays and no warning fires. True gradient-colored
    /// polylines are a logged feature follow-up, not this fix's scope.
    ///
    /// `suppressed` (spec-review 2026-08-28 ruling): the warning itself fires
    /// whenever the channel is inert, regardless of whether a colorbar would
    /// have rendered (`Color(v, legend=None)`, or the same-field color+size
    /// merge whose colorbar was already folded into the size legend, both
    /// still warn — spec §4.0's loudness is a property of the CHANNEL). But
    /// the Display text's claim that "its legend was suppressed" is only
    /// TRUE when a colorbar actually existed to suppress; `suppressed`
    /// records which happened so the message never asserts a suppression
    /// that did not occur.
    UnsupportedColorScaleOnMark {
        marks: Vec<String>,
        scale_kind: String,
        suppressed: bool,
    },
    /// An opacity-family channel (`opacity`/`fill_opacity`/`stroke_opacity`)
    /// carried an explicit `scale=` whose spec is not `Linear` (spec §4.3,
    /// amended 2026-09-01, T6 quality review — ruling 2). The curve, domain,
    /// and range are NOT honored; the channel falls back to the default
    /// linear resolution (data extent onto the theme opacity band) instead.
    /// `channel` is the channel name (`"opacity"` / `"fill_opacity"` /
    /// `"stroke_opacity"`); `scale_kind` is the dropped spec's kind
    /// (`"log"`, `"pow"`, `"sqrt"`, …). Full non-linear opacity-curve support
    /// is a logged campaign follow-up, not this batch.
    UnsupportedOpacityScale { channel: String, scale_kind: String },
}

impl std::fmt::Display for RenderWarning {
    /// User-facing warning text forwarded to Python's ``warnings.warn`` by
    /// [`binding::emit_warnings`](crate::render::binding).
    ///
    /// This is an intentional, stable Display contract — not the derived Debug
    /// of the enum's internal fields. The previous behavior leaked the Rust
    /// variant/struct shape (e.g. `UnsupportedEncodingCombo { channel: "x" }`)
    /// across the Python boundary; these sentences are the supported message
    /// surface instead. `RenderWarning::Layout` delegates to
    /// [`LayoutWarning`]'s own Display so the layout messages live next to that
    /// enum.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderWarning::Layout(w) => write!(f, "{w}"),
            RenderWarning::OutOfDomainRows { mark, count } => write!(
                f,
                "{count} {mark} row(s) fell outside the scale domain and were not drawn"
            ),
            RenderWarning::ColorPaletteOverflowed { categories } => write!(
                f,
                "color palette has fewer entries than the {categories} categories; \
                 colors were recycled"
            ),
            RenderWarning::ShapePaletteOverflowed { categories } => write!(
                f,
                "shape palette has fewer entries than the {categories} categories; \
                 shapes were recycled"
            ),
            RenderWarning::StrokeDashPaletteOverflowed { categories } => write!(
                f,
                "stroke dash palette has fewer entries than the {categories} categories; \
                 dashes were recycled"
            ),
            RenderWarning::EmptyPanel { panel_index } => write!(
                f,
                "panel {panel_index} is too small to render and was left empty"
            ),
            RenderWarning::ColorRangeParseFailure { entry } => write!(
                f,
                "could not parse color '{entry}'; the explicit color range was \
                 discarded in favor of the theme palette"
            ),
            RenderWarning::ColorRangeBucketCountMismatch { expected, received } => write!(
                f,
                "color range names {received} color(s) but the binned color scale has \
                 {expected} bucket(s); the range was not applied"
            ),
            RenderWarning::ColorDomainOmitsCategories { categories } => write!(
                f,
                "color domain does not list {}; those marks paint in the default mark \
                 color with no legend entry",
                categories.join(", ")
            ),
            RenderWarning::SortSpecIgnored { reason } => write!(
                f,
                "sort spec could not be applied ({reason}); categories fall back \
                 to insertion order"
            ),
            RenderWarning::PositionAdjustSkipped { adjustment, reason } => write!(
                f,
                "{adjustment} could not be applied ({reason}); marks were not offset"
            ),
            RenderWarning::OverlayGuttersDiverged { layers } => write!(
                f,
                "overlay gutters diverged; {layers} layers render with independent chrome"
            ),
            RenderWarning::UnsupportedColorScaleOnMark { marks, scale_kind, suppressed } => {
                let marks = marks.join(", ");
                if *suppressed {
                    write!(
                        f,
                        "{scale_kind} color scale is not supported on {marks}; the channel has \
                         no per-mark effect and its legend was suppressed"
                    )
                } else {
                    write!(
                        f,
                        "{scale_kind} color scale is not supported on {marks}; the channel has \
                         no per-mark effect"
                    )
                }
            }
            RenderWarning::UnsupportedOpacityScale { channel, scale_kind } => write!(
                f,
                "{channel}: {scale_kind} scale is not supported on opacity-family channels; \
                 the curve, domain, and range were ignored in favor of the default linear \
                 resolution"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderOutput<T> {
    pub bytes: T,
    pub layout: crate::layout::LayoutResult,
    pub warnings: Vec<RenderWarning>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_config_default_values() {
        let c = config::RenderConfig::default();
        assert_eq!(c.scale, 2.0);
        assert!(c.embed_fonts);
        assert!(c.background.is_none());
        assert!(c.width.is_none());
        assert!(c.height.is_none());
    }

    #[test]
    fn render_warning_round_trip_each_variant() {
        use crate::layout::LayoutWarning;
        for w in [
            RenderWarning::Layout(LayoutWarning::PanelCollapsed { panel_index: 0 }),
            RenderWarning::OutOfDomainRows { mark: "point".into(), count: 3 },
            RenderWarning::ColorPaletteOverflowed { categories: 12 },
            RenderWarning::ShapePaletteOverflowed { categories: 7 },
            RenderWarning::StrokeDashPaletteOverflowed { categories: 6 },
            RenderWarning::EmptyPanel { panel_index: 1 },
            RenderWarning::ColorRangeParseFailure { entry: "#zzz".into() },
            RenderWarning::ColorRangeBucketCountMismatch { expected: 3, received: 2 },
            RenderWarning::ColorDomainOmitsCategories { categories: vec!["c".into()] },
            RenderWarning::SortSpecIgnored { reason: "missing field".into() },
            RenderWarning::PositionAdjustSkipped {
                adjustment: "dodge".into(),
                reason: "by-column 'grp' not found in data".into(),
            },
            RenderWarning::OverlayGuttersDiverged { layers: 2 },
            RenderWarning::UnsupportedColorScaleOnMark {
                marks: vec!["line".into(), "ribbon".into()],
                scale_kind: "continuous".into(),
                suppressed: true,
            },
            RenderWarning::UnsupportedColorScaleOnMark {
                marks: vec!["line".into()],
                scale_kind: "discretizing".into(),
                suppressed: false,
            },
            RenderWarning::UnsupportedOpacityScale {
                channel: "fill_opacity".into(),
                scale_kind: "log".into(),
            },
        ] {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: RenderWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w);
        }
    }

    #[test]
    fn render_warning_display_is_intentional_not_debug() {
        use crate::layout::LayoutWarning;
        // Display must be a human sentence, never the derived-Debug variant
        // shape (SEAM-07). The substrings below are the stable contract the
        // Python warning-filter tests match on.
        let sort = RenderWarning::SortSpecIgnored {
            reason: "missing field".into(),
        };
        let text = format!("{sort}");
        assert!(
            text.contains("sort spec could not be applied"),
            "Display text was: {text}"
        );
        assert!(text.contains("missing field"), "Display text was: {text}");
        // Display must NOT leak the variant name (the old Debug behavior did).
        assert!(!text.contains("SortSpecIgnored"), "Display leaked Debug: {text}");

        let parse_fail = RenderWarning::ColorRangeParseFailure {
            entry: "#zzz".into(),
        };
        assert!(format!("{parse_fail}").contains("could not parse color"));

        let empty = RenderWarning::EmptyPanel { panel_index: 2 };
        assert!(format!("{empty}").contains("too small to render"));

        // `Layout` delegates to LayoutWarning's Display and carries the keys.
        let dropped = RenderWarning::Layout(LayoutWarning::PanelsDropped {
            count: 1,
            keys: vec!["col_cat=c2".into()],
        });
        let dropped_text = format!("{dropped}");
        assert!(dropped_text.contains("facet panel(s) were dropped"));
        assert!(dropped_text.contains("col_cat=c2"));

        // GH #89A: names the CAUSE (the layers' own gutters), not the symptom.
        let diverged = RenderWarning::OverlayGuttersDiverged { layers: 2 };
        let diverged_text = format!("{diverged}");
        assert!(diverged_text.contains("overlay gutters diverged"), "{diverged_text}");
        assert!(diverged_text.contains("independent chrome"), "{diverged_text}");
        assert!(!diverged_text.contains("OverlayGuttersDiverged"), "{diverged_text}");
    }

    /// Spec-review 2026-08-28 (cannot_verify item, resolved this round):
    /// `UnsupportedColorScaleOnMark`'s Display must not claim a legend was
    /// suppressed when none ever existed to suppress (`legend=None`, or the
    /// color+size merge case whose colorbar was folded into the size legend
    /// for a different reason) — the `suppressed` field branches the message
    /// so it only asserts what actually happened.
    #[test]
    fn render_warning_unsupported_color_scale_message_branches_on_suppressed() {
        let with_suppression = RenderWarning::UnsupportedColorScaleOnMark {
            marks: vec!["line".into()],
            scale_kind: "continuous".into(),
            suppressed: true,
        };
        let text = format!("{with_suppression}");
        assert!(text.contains("legend was suppressed"), "{text}");

        let without_suppression = RenderWarning::UnsupportedColorScaleOnMark {
            marks: vec!["line".into()],
            scale_kind: "continuous".into(),
            suppressed: false,
        };
        let text = format!("{without_suppression}");
        assert!(!text.contains("suppressed"), "{text}");
        assert!(text.contains("no per-mark effect"), "{text}");
    }

    #[test]
    fn render_error_display_messages_are_meaningful() {
        let err = RenderError::InvalidViewport { width: 0.0, height: 100.0 };
        let msg = format!("{err}");
        assert!(msg.contains("invalid viewport"), "msg: {msg}");
        assert!(msg.contains("0"), "msg: {msg}");

        let err = RenderError::UnknownColumn { name: "missing".into() };
        let msg = format!("{err}");
        assert!(msg.contains("unknown column"), "msg: {msg}");
        assert!(msg.contains("missing"), "msg: {msg}");
    }
}

#[cfg(test)]
mod axis_style_fill_from_tests {
    //! Characterization tests for `axis_style_fill_from` (T2.6 / SPINE-01): the
    //! one merge that replaced the two parallel `AxisStyleSpec → AxisStyleOverrides`
    //! mappers. These pin the exact field ownership both old mappers had so the
    //! collapse stays behavior-preserving.
    use super::*;
    use crate::layout::{AxisStyleOverrides, LabelOverlap};
    use chart_config::AxisStyleSpec;

    /// An `AxisStyleSpec` with every merge-relevant field populated, plus a couple
    /// of fields that exercise color parsing and token validation.
    fn fully_populated_spec() -> AxisStyleSpec {
        AxisStyleSpec {
            label_angle: Some(45.0),
            label_font_size: Some(11.0),
            label_color: Some("#112233".into()),
            label_format: Some(".2f".into()),
            label_format_type: None,
            label_overlap: Some("parity".into()),
            label_flush: Some(true),
            labels: Some(false), // show toggle — NOT an overrides field; must be ignored
            ticks: Some(false),  // show toggle — ignored
            tick_count: Some(7), // not an overrides field — ignored
            tick_size: Some(4.0),
            tick_extra: Some(true),
            tick_min_step: Some(0.5),
            values: Some(vec![0.0, 1.0, 2.0]),
            grid: Some(false), // show toggle — ignored
            grid_color: Some("#445566".into()),
            grid_dash: Some(vec![6.0, 3.0]),
            grid_width: Some(1.5),
            grid_opacity: Some(0.4),
            domain: Some(false), // show toggle — ignored
            domain_color: Some("#778899".into()),
            domain_width: Some(2.0),
            title: Some("ignored-here".into()), // title text — not an overrides field
            title_font_size: Some(13.0),
            title_color: Some("#aabbcc".into()),
            title_padding: Some(8.0),
            title_orient: Some("right".into()),
            label_padding: Some(3.0),
            orient: Some("bottom".into()),
            translate: Some(5.0),
            min_band: Some(10.0),
            max_band: Some(40.0),
            offset: Some(2.0),
            zindex: Some(1),
        }
    }

    /// Per-channel fresh-build (`fill_only_if_none = false`) writes every
    /// overrides field from the spec — EXCEPT `label_format`, which the
    /// per-channel/prepare path deliberately leaves `None` (it is threaded
    /// separately afterward). show_* toggles and non-overrides fields (title text,
    /// tick_count, tick_size) are never written (they are not bundle fields).
    #[test]
    fn fresh_build_writes_all_fields_except_label_format() {
        let spec = fully_populated_spec();
        let mut o = AxisStyleOverrides::default();
        axis_style_fill_from(&mut o, &spec, "x", false).unwrap();

        assert_eq!(o.label_angle, Some(45.0));
        assert_eq!(o.label_font_size, Some(11.0));
        assert_eq!(o.label_color, color::parse_color("#112233").ok());
        // label_format MUST stay None on the per-channel/fresh-build path.
        assert_eq!(o.label_format, None);
        assert_eq!(o.label_overlap, Some(LabelOverlap::Parity));
        assert_eq!(o.label_flush, Some(true));
        assert_eq!(o.tick_extra, Some(true));
        assert_eq!(o.tick_min_step, Some(0.5));
        assert_eq!(o.tick_values, Some(vec![0.0, 1.0, 2.0]));
        assert_eq!(o.grid_color, color::parse_color("#445566").ok());
        assert_eq!(o.grid_dash, Some(vec![6.0, 3.0]));
        assert_eq!(o.grid_width, Some(1.5));
        assert_eq!(o.grid_opacity, Some(0.4));
        assert_eq!(o.domain_color, color::parse_color("#778899").ok());
        assert_eq!(o.domain_width, Some(2.0));
        assert_eq!(o.title_font_size, Some(13.0));
        assert_eq!(o.title_color, color::parse_color("#aabbcc").ok());
        assert_eq!(o.title_padding, Some(8.0));
        assert_eq!(o.title_orient, Some(crate::layout::AxisOrient::Right));
        assert_eq!(o.label_padding, Some(3.0));
        assert_eq!(o.orient, Some(crate::layout::AxisOrient::Bottom));
        assert_eq!(o.translate, Some(5.0));
        assert_eq!(o.min_band, Some(10.0));
        assert_eq!(o.max_band, Some(40.0));
        assert_eq!(o.offset, Some(2.0));
        assert_eq!(o.zindex, Some(1));
    }

    /// Chart-level fill (`fill_only_if_none = true`) fills only `None` slots
    /// (higher-precedence values survive) AND owns `label_format` (the one field
    /// the per-channel path leaves `None`).
    #[test]
    fn chart_level_fills_only_none_and_owns_label_format() {
        let spec = fully_populated_spec();
        let mut o = AxisStyleOverrides {
            // A higher-precedence per-channel value already claimed these slots.
            label_angle: Some(90.0),
            grid_width: Some(99.0),
            ..AxisStyleOverrides::default()
        };
        axis_style_fill_from(&mut o, &spec, "x", true).unwrap();

        // Pre-set slots survive (fill-only-if-None).
        assert_eq!(o.label_angle, Some(90.0));
        assert_eq!(o.grid_width, Some(99.0));
        // Empty slots filled from the spec.
        assert_eq!(o.label_font_size, Some(11.0));
        assert_eq!(o.tick_min_step, Some(0.5));
        assert_eq!(o.orient, Some(crate::layout::AxisOrient::Bottom));
        // label_format IS written on the chart-level path.
        assert_eq!(o.label_format, Some(".2f".into()));
    }

    /// Chart-level fill does NOT clobber a `label_format` that a per-channel value
    /// already set (the threaded override wins).
    #[test]
    fn chart_level_label_format_defers_to_existing() {
        let spec = fully_populated_spec(); // label_format = ".2f"
        let mut o = AxisStyleOverrides {
            label_format: Some("~s".into()), // per-channel already won
            ..AxisStyleOverrides::default()
        };
        axis_style_fill_from(&mut o, &spec, "x", true).unwrap();
        assert_eq!(o.label_format, Some("~s".into()));
    }

    /// A cross-dimension `orient` (left/right on an x axis) fails loud on both
    /// paths, exactly as both old mappers did via `parse_axis_orient`.
    #[test]
    fn cross_dimension_orient_errors() {
        let spec = AxisStyleSpec { orient: Some("left".into()), ..Default::default() };
        let mut o = AxisStyleOverrides::default();
        assert!(axis_style_fill_from(&mut o, &spec, "x", false).is_err());
        let mut o = AxisStyleOverrides::default();
        assert!(axis_style_fill_from(&mut o, &spec, "x", true).is_err());
    }

    /// An unparseable color hex leaves the slot `None` (theme fallback) rather
    /// than failing — preserved from both old mappers.
    #[test]
    fn bad_color_hex_falls_back_to_none() {
        let spec = AxisStyleSpec { label_color: Some("not-a-color".into()), ..Default::default() };
        let mut o = AxisStyleOverrides::default();
        axis_style_fill_from(&mut o, &spec, "x", false).unwrap();
        assert_eq!(o.label_color, None);
    }
}

// ---------------------------------------------------------------------------
// Task 20 — render_svg full pipeline orchestration (spec §6).
// ---------------------------------------------------------------------------

use crate::layout::{compute_layout, CompositeLayoutSeam, LegendDirection, LegendOrient, LegendOverrides, LegendSuppression, TextAnchor, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;
use chart_config::{AxisConfigSpec, ChartConfig};

/// Apply [`ChartConfig`] overrides to a [`ThemeInputs`] clone.
///
/// This implements "configure > theme" precedence (level 3 > level 4–5). It is
/// called in both `render_svg` and `render_scene_json` after the per-encoding
/// legend overrides have been merged into `effective_theme` but before
/// `compute_layout` and `build_scene` are invoked.
///
/// Per-channel `axis=Axis(...)` overrides live in `AxisInput` and are resolved
/// by `prepare_render_inputs` — they take effect at layout time (level 2) and
/// are never touched here.
fn apply_chart_config(theme: &mut ThemeInputs, config: &ChartConfig) {
    // ── Grid overrides ────────────────────────────────────────────────────────
    if let Some(ref grid_cfg) = config.grid {
        if let Some(enabled) = grid_cfg.x {
            // x-grid is controlled globally via theme.grid.grid; a dedicated x-only
            // flag does not exist in ThemeInputs. When both x and y are present
            // we only flip the global flag when they agree.
            if grid_cfg.y.unwrap_or(enabled) == enabled {
                theme.grid.grid = enabled;
            }
        } else if let Some(enabled) = grid_cfg.y {
            theme.grid.grid = enabled;
        }
        if let Some(ref color_str) = grid_cfg.color {
            if let Ok(c) = color::parse_color(color_str) {
                theme.colors.grid_color = c;
            }
        }
        if let Some(w) = grid_cfg.width {
            theme.sizes.grid_width = w;
        }
        if let Some(ref d) = grid_cfg.dash {
            theme.grid.grid_dash = Some(d.clone());
        }
        if let Some(o) = grid_cfg.opacity {
            theme.grid.grid_opacity = o;
        }
    }

    // ── Padding overrides ─────────────────────────────────────────────────────
    // Per-side padding: map each supplied side directly to ThemeInputs.padding.*.
    // `auto=true` (the Python default) is no longer a guard — it was previously
    // blocking all padding overrides. The `auto` field is reserved for a future
    // "auto-expand to fit labels" semantic; it does not disable explicit values.
    if let Some(ref pad) = config.padding {
        if let Some(top) = pad.top {
            theme.padding.padding_top = Some(top);
        }
        if let Some(right) = pad.right {
            theme.padding.padding_right = Some(right);
        }
        if let Some(bottom) = pad.bottom {
            theme.padding.padding_bottom = Some(bottom);
        }
        if let Some(left) = pad.left {
            theme.padding.padding_left = Some(left);
        }
    }

    // ── Legend overrides ──────────────────────────────────────────────────────
    if let Some(ref legend_cfg) = config.legend {
        let legend = &legend_cfg.style;
        if let Some(ref orient) = legend.orient {
            theme.legend.legend_orient = match orient.as_str() {
                "right"  => LegendOrient::Right,
                "left"   => LegendOrient::Left,
                "top"    => LegendOrient::Top,
                "bottom" => LegendOrient::Bottom,
                _ => theme.legend.legend_orient,
            };
        }
        if let Some(ref dir) = legend.direction {
            theme.legend.legend_direction = match dir.as_str() {
                "vertical"   => Some(LegendDirection::Vertical),
                "horizontal" => Some(LegendDirection::Horizontal),
                _ => theme.legend.legend_direction,
            };
        }
        if let Some(cols) = legend.columns {
            theme.legend.legend_columns = Some(cols);
        }
        if let Some(fs) = legend.title_font_size {
            theme.typography.legend_title_font_size = fs;
        }
        // legend.label_font_size maps to the shared theme.typography.label_font_size,
        // which controls both axis and legend label sizing.
        if let Some(fs) = legend.label_font_size {
            theme.typography.label_font_size = fs;
        }
    }

    // ── Color scheme overrides ────────────────────────────────────────────────
    if let Some(ref color_cfg) = config.color {
        if let Some(ref scheme) = color_cfg.scheme {
            theme.palette.color_scheme = scheme.clone();
        }
        if let Some(ref seq) = color_cfg.sequential_scheme {
            theme.palette.sequential_scheme = seq.clone();
        }
        if let Some(ref div) = color_cfg.diverging_scheme {
            theme.palette.diverging_scheme = div.clone();
        }
    }

    // ── Axis overrides (applied to both axes simultaneously) ──────────────────
    if let Some(ref axis_cfg) = config.axis {
        apply_axis_config_to_theme(theme, axis_cfg);
    }
    // Per-axis overrides run after the combined override so axis_x / axis_y
    // wins over axis when both are specified.
    if let Some(ref axis_x) = config.axis_x {
        apply_axis_x_config_to_theme(theme, axis_x);
    }
    if let Some(ref axis_y) = config.axis_y {
        apply_axis_y_config_to_theme(theme, axis_y);
    }

    // ── Title overrides ───────────────────────────────────────────────────────
    if let Some(ref title) = config.title {
        if let Some(fs) = title.font_size {
            theme.typography.title_font_size = fs;
        }
        if let Some(ref fw) = title.font_weight {
            theme.typography.title_font_weight = fw.clone();
        }
        if let Some(ref anchor) = title.anchor {
            theme.typography.title_anchor = match anchor.as_str() {
                "middle" => TextAnchor::Middle,
                "end"    => TextAnchor::End,
                _        => TextAnchor::Start,
            };
        }
        if let Some(ref c) = title.color {
            if let Ok(parsed) = color::parse_color(c) {
                theme.colors.title_color = parsed;
            }
        }
        if let Some(o) = title.offset {
            theme.typography.title_offset = o;
        }
        if let Some(fs) = title.subtitle_font_size {
            theme.typography.subtitle_font_size = Some(fs);
        }
        if let Some(ref c) = title.subtitle_color {
            if let Ok(parsed) = color::parse_color(c) {
                theme.colors.subtitle_color = Some(parsed);
            }
        }
    }
}

/// Apply axis config fields that affect shared theme state (both axes).
///
/// Fields like `label_font_size`, `tick_size`, and grid color are global in
/// `ThemeInputs` and are applied here regardless of x/y axis distinction.
/// Per-axis-specific overrides (x vs y) are handled in the sibling functions.
fn apply_axis_config_to_theme(theme: &mut ThemeInputs, axis_cfg: &AxisConfigSpec) {
    let style = &axis_cfg.style;
    if let Some(fs) = style.label_font_size {
        theme.typography.label_font_size = fs;
    }
    if let Some(ref c) = style.label_color {
        if let Ok(parsed) = color::parse_color(c) {
            theme.colors.label_color = parsed;
        }
    }
    if let Some(ts) = style.tick_size {
        theme.sizes.tick_size = ts;
    }
    if let Some(enabled) = style.domain {
        theme.axis.axis_line = enabled;
    }
    if let Some(ref c) = style.domain_color {
        if let Ok(parsed) = color::parse_color(c) {
            theme.colors.axis_line_color = parsed;
        }
    }
    if let Some(w) = style.domain_width {
        theme.sizes.axis_line_width = w;
    }
    if let Some(enabled) = style.grid {
        theme.grid.grid = enabled;
    }
    if let Some(ref c) = style.grid_color {
        if let Ok(parsed) = color::parse_color(c) {
            theme.colors.grid_color = parsed;
        }
    }
    if let Some(ref d) = style.grid_dash {
        theme.grid.grid_dash = Some(d.clone());
    }
    if let Some(w) = style.grid_width {
        theme.sizes.grid_width = w;
    }
}

/// Apply `axis_x`-specific config fields that have per-axis theme equivalents.
///
/// Currently `ThemeInputs` has no separate x/y axis theme fields, so x-specific
/// overrides that conflict with y settings cannot be expressed at this level.
/// When axis_x and axis_y specify the same field with different values, the last
/// one applied wins (axis_y runs after axis_x in the caller).
fn apply_axis_x_config_to_theme(theme: &mut ThemeInputs, axis: &AxisConfigSpec) {
    // x-axis-specific overrides reuse the same shared theme fields.
    // Per-channel `axis=Axis(...)` on the x encoding remains the highest-priority
    // override and is handled separately in `prepare_render_inputs`.
    apply_axis_config_to_theme(theme, axis);
}

/// Apply `axis_y`-specific config fields that have per-axis theme equivalents.
fn apply_axis_y_config_to_theme(theme: &mut ThemeInputs, axis: &AxisConfigSpec) {
    apply_axis_config_to_theme(theme, axis);
}

/// Build a [`LegendOverrides`] from a [`prepare::PreparedInputs`].
fn legend_overrides_from_prep(prep: &prepare::PreparedInputs) -> LegendOverrides {
    let lo = &prep.legend_overrides;
    LegendOverrides {
        tick_count:         lo.tick_count,
        gradient_length:    lo.gradient_length,
        gradient_thickness: lo.gradient_thickness,
        direction:          lo.direction,
        values:             lo.values.clone(),
        legend_type:        lo.legend_type.clone(),
        // Per-channel `Legend(symbol_type=...)` (B5); chart-level
        // `configure_legend` fills this only when the per-channel value is absent.
        symbol_type:        lo.symbol_type.clone(),
        tick_min_step:      lo.tick_min_step,
        // 380: the 11 categorical-style fields (B5 unit 3 / 6a orphans —
        // per-channel here; chart-level fills any still None) live nested on
        // `style`. The colorbar path also reads the shared `clip_height` /
        // `label_color` / `label_font_size` from here.
        style: crate::layout::LegendStyleOpts {
            symbol_stroke_width: lo.symbol_stroke_width,
            row_padding:         lo.row_padding,
            column_padding:      lo.column_padding,
            label_limit:         lo.label_limit,
            clip_height:         lo.clip_height,
            padding:             lo.padding,
            title_padding:       lo.title_padding,
            offset:              lo.offset,
            symbol_size:         lo.symbol_size,
            label_color:         lo.label_color.clone(),
            label_font_size:     lo.label_font_size,
        },
    }
}

/// Resolve the effective legend title from a [`prepare::PreparedInputs`].
///
/// Three-way resolution (D13 + v0.15.1), shared by `prepare_and_layout`
/// (single-chart path) and `composite_render::capture_leaf_bundle`
/// (figure-legend seam, design §6): mirrors the axis-title contract in
/// `prepare.rs`.
///   - `legend_overrides.title` absent (`None`)   → fall through to field-name default
///   - `legend_overrides.title = Some("")`        → explicit suppress; no text node, no margin
///   - `legend_overrides.title = Some("Foo")`     → render "Foo" verbatim
///
/// Python forwards `""` only when `Legend(title=None)` is explicitly passed,
/// so `Some("")` here is always the caller's intentional suppress sentinel.
pub(crate) fn effective_legend_title(prep: &prepare::PreparedInputs) -> Option<String> {
    match prep.legend_overrides.title.as_deref() {
        Some(s) if s.trim().is_empty() => None, // explicit suppress — no fallback
        Some(s) => Some(s.to_owned()),           // explicit non-empty title
        None => prep.legend_title.clone(),       // absent — fall through to field-name default
    }
}

/// Whether chart-level `configure_legend(...)` fully suppresses the legend
/// (GH #74). Currently set only via `disabled: true`, which Python's
/// `_resolve_chart_config` derives from a fully-merged `orient="none"` —
/// see the call site in `prepare_and_layout`. Reads the SAME `disabled`
/// field `LegendStyleSpec` already carries for the per-encoding
/// `Color(legend=None)` path (D13/Schwabish SB3); this is the chart-level
/// mirror of that one suppression mechanism, not a second one.
fn chart_config_legend_disabled(chart_config: &ChartConfig) -> bool {
    chart_config.legend.as_ref().and_then(|l| l.style.disabled).unwrap_or(false)
}

/// Apply `ChartConfig.legend` fields to a `LegendOverrides`.
///
/// Per-encoding `legend=Legend(...)` overrides (level 2) are already in the
/// `LegendOverrides` built by `legend_overrides_from_prep`. This function
/// fills in the `configure_legend(...)` overrides (level 3) only when the
/// per-encoding value is absent (level 2 wins over level 3).
fn apply_chart_config_to_legend_overrides(
    overrides: &mut LegendOverrides,
    config: &ChartConfig,
) {
    let Some(ref legend) = config.legend else { return };
    let legend = &legend.style;
    if overrides.gradient_length.is_none() {
        overrides.gradient_length = legend.gradient_length;
    }
    if overrides.symbol_type.is_none() {
        overrides.symbol_type = legend.symbol_type.clone();
    }
    // B5 unit 3 orphans: per-channel (level 2) already in `overrides.style`; fill
    // from `configure_legend` (level 3) only where still None, so per-channel wins.
    let style = &mut overrides.style;
    if style.symbol_stroke_width.is_none() {
        style.symbol_stroke_width = legend.symbol_stroke_width;
    }
    if style.row_padding.is_none() {
        style.row_padding = legend.row_padding;
    }
    if style.column_padding.is_none() {
        style.column_padding = legend.column_padding;
    }
    if style.label_limit.is_none() {
        style.label_limit = legend.label_limit;
    }
    if style.clip_height.is_none() {
        style.clip_height = legend.clip_height;
    }
    if overrides.tick_min_step.is_none() {
        overrides.tick_min_step = legend.tick_min_step;
    }
    // B5 unit 6a orphans: per-channel (level 2) already in `overrides.style`; fill
    // from `configure_legend` (level 3) only where still None, so per-channel wins.
    let style = &mut overrides.style;
    if style.symbol_size.is_none() {
        style.symbol_size = legend.symbol_size;
    }
    if style.label_color.is_none() {
        style.label_color = legend.label_color.clone();
    }
    if style.offset.is_none() {
        style.offset = legend.offset;
    }
    if style.padding.is_none() {
        style.padding = legend.padding;
    }
    if style.title_padding.is_none() {
        style.title_padding = legend.title_padding;
    }
}

/// Apply `ChartConfig.axis` / `axis_x` / `axis_y` per-axis fields to the
/// `AxisInput`. Only fields absent from the input (higher-precedence per-channel
/// or earlier config) are filled, so the cascade is per-channel > axis_x/axis_y >
/// axis > theme.
///
/// Called with the `axis` key (both x and y), then `axis_x`/`axis_y` (which win
/// because they run last). Delegates the per-axis style fields to
/// [`apply_axis_style_to_axis_input`] and handles the chart-only `label_format_raw`
/// d3-format key here (the per-channel path uses `label_format` inside the style).
pub(crate) fn apply_axis_config_to_axis_input(
    axis: &mut crate::layout::AxisInput,
    config: Option<&chart_config::AxisConfigSpec>,
) -> Result<(), RenderError> {
    let Some(cfg) = config else { return Ok(()) };
    // Resolve the chart-level d3-format override BEFORE the shared style apply.
    // `effective_label_format()` is raw-first: `label_format_raw` (the chart-level
    // spelling) wins over the style's `label_format`. Applying it first ensures that
    // precedence holds — the style apply below only fills `label_format_override`
    // when still `None`, so it can no longer override the raw-first resolution. Both
    // keys are mutually exclusive at the Python boundary (`AxisConfig.__post_init__`
    // raises), so in practice at most one is set; this keeps the Rust side
    // self-consistent regardless. Fill only when nothing higher-precedence (a
    // per-channel spec) already set the override.
    if axis.overrides.label_format.is_none() {
        axis.overrides.label_format = cfg.effective_label_format().map(str::to_owned);
    }
    apply_axis_style_to_axis_input(axis, &cfg.style)
}

/// The channel an axis belongs to, inferred from its current orient. x carries a
/// horizontal axis (Top/Bottom); y a vertical one (Left/Right). Used to validate
/// a chart-level `orient`/`title_orient` against the axis's dimension.
fn axis_channel(orient: crate::layout::AxisOrient) -> &'static str {
    use crate::layout::AxisOrient::{Bottom, Top};
    if matches!(orient, Top | Bottom) { "x" } else { "y" }
}

/// One canonical [`AxisStyleSpec`](chart_config::AxisStyleSpec) →
/// [`AxisStyleOverrides`](crate::layout::AxisStyleOverrides) merge, replacing the
/// two parallel ~28-field mappers that previously turned the SAME source struct
/// into the bundle in two shapes (the per-channel fresh-builder in
/// `prepare::encoding_axis_style_overrides` and the chart-level fill-only-if-`None`
/// `apply_axis_style_to_axis_input`). Both call sites now route through here so a
/// new `AxisStyleSpec` field is wired once, not in two drifting bodies.
///
/// `fill_only_if_none` selects the merge discipline:
/// - `false` (per-channel path): write every field unconditionally — the caller
///   starts from `Default`, so this is the fresh-build the encoding path needs.
/// - `true` (chart-level path): write each field only when the slot is still
///   `None`, so a higher-precedence source (a per-channel spec, or an earlier
///   config layer) always wins.
///
/// Field-ownership exceptions preserved bit-for-bit from the old two-mapper world:
/// - **`label_format`** is written ONLY on the chart-level (`fill_only_if_none`)
///   path. The per-channel/prepare path deliberately leaves it `None` here and
///   seeds it separately from the temporal/numeric format threading
///   (`apply_axis_format_or_thread`) after this merge runs.
/// - **`show_*`** toggles (`grid`/`domain`/`labels`/`ticks`) live on `AxisInput`,
///   not on this bundle, and are owned solely by the prepare path. This merge
///   cannot touch them (they are not `AxisStyleOverrides` fields); the chart-level
///   caller documents that single-owner contract at its call site.
///
/// An invalid `orient` (cross-dimension) or `title_orient` token fails loud via
/// [`RenderError::InvalidAxisOrient`]; an unparseable color hex string leaves the
/// slot `None` (theme fallback) on both paths.
///
/// Returns [`UnflippedRenderError`] (R3), not `RenderError` directly: neither
/// this fn nor `parse_axis_orient`/`parse_title_orient` beneath it has access
/// to `coord_flipped` (this fn is shared by both the flip-patched per-channel
/// path and the flip-exempt chart-level path — see the field doc on
/// [`RenderError::InvalidAxisOrient`]), so the decision is deferred to each
/// caller's own boundary via `.resolve(coord_flipped)`.
pub(crate) fn axis_style_fill_from(
    o: &mut crate::layout::AxisStyleOverrides,
    style: &chart_config::AxisStyleSpec,
    channel: &'static str,
    fill_only_if_none: bool,
) -> Result<(), UnflippedRenderError> {
    // One merge predicate for every field: on the fresh-build (per-channel) path
    // (`fill_only_if_none == false`) the value is written unconditionally; on the
    // chart-level path it is written only when the slot is still `None` so a
    // higher-precedence source wins. Collapsing it here means each field below
    // appears exactly once regardless of which discipline applies. A generic `fn`
    // (not a closure) because the slot type varies across fields.
    fn set<T>(slot: &mut Option<T>, value: Option<T>, fill_only_if_none: bool) {
        if !fill_only_if_none || slot.is_none() {
            *slot = value;
        }
    }
    // Full CSS vocabulary (names, `rgb()`, hex) — `None` for absent or
    // unparseable values, matching the pre-existing "keep the theme default"
    // behavior of every axis-style color override.
    let opt_color = |c: &Option<String>| c.as_deref().and_then(|s| color::parse_color(s).ok());
    // ── Positioning / draw-order orphans (B5 unit 2) ─────────────────────────
    // `orient` is the override INPUT (validated against the dimension); the
    // concrete `AxisInput.orient` is re-synced from it by `resolve_orient` after
    // all override layers merge. Validation can fail, so it is resolved eagerly
    // (before `set`) and only assigned under the same fill predicate.
    if !fill_only_if_none || o.orient.is_none() {
        o.orient = style
            .orient
            .as_deref()
            .map(|s| prepare::parse_axis_orient(s, channel))
            .transpose()?;
    }
    if !fill_only_if_none || o.title_orient.is_none() {
        o.title_orient = style
            .title_orient
            .as_deref()
            .map(|s| prepare::parse_title_orient(s, channel))
            .transpose()?;
    }
    set(&mut o.translate, style.translate, fill_only_if_none);
    set(&mut o.min_band, style.min_band, fill_only_if_none);
    set(&mut o.max_band, style.max_band, fill_only_if_none);
    set(&mut o.grid_opacity, style.grid_opacity, fill_only_if_none);
    set(&mut o.zindex, style.zindex, fill_only_if_none);
    set(&mut o.tick_extra, style.tick_extra, fill_only_if_none);
    set(&mut o.tick_min_step, style.tick_min_step, fill_only_if_none);
    // ── Residual positioning/overlap orphans (B5 unit 6b) ────────────────────
    set(&mut o.offset, style.offset, fill_only_if_none);
    set(&mut o.label_flush, style.label_flush, fill_only_if_none);
    set(
        &mut o.label_overlap,
        style.label_overlap.as_deref().and_then(prepare::parse_label_overlap),
        fill_only_if_none,
    );
    set(&mut o.label_angle, style.label_angle, fill_only_if_none);
    // `label_format` is owned by the chart-level path only. The per-channel path
    // threads it separately after this merge, so it must stay `None` here.
    if fill_only_if_none && o.label_format.is_none() {
        o.label_format = style.label_format.clone();
    }
    set(&mut o.tick_values, style.values.clone(), fill_only_if_none);
    // Title overrides.
    set(&mut o.title_font_size, style.title_font_size, fill_only_if_none);
    set(&mut o.title_color, opt_color(&style.title_color), fill_only_if_none);
    set(&mut o.title_padding, style.title_padding, fill_only_if_none);
    set(&mut o.label_padding, style.label_padding, fill_only_if_none);
    // ── Per-axis styling overrides (B5): consulted by build_axis/build_grid ──
    set(&mut o.label_color, opt_color(&style.label_color), fill_only_if_none);
    set(&mut o.label_font_size, style.label_font_size, fill_only_if_none);
    set(&mut o.grid_color, opt_color(&style.grid_color), fill_only_if_none);
    set(&mut o.grid_dash, style.grid_dash.clone(), fill_only_if_none);
    set(&mut o.grid_width, style.grid_width, fill_only_if_none);
    set(&mut o.domain_color, opt_color(&style.domain_color), fill_only_if_none);
    set(&mut o.domain_width, style.domain_width, fill_only_if_none);
    Ok(())
}

/// Apply an [`AxisStyleSpec`](chart_config::AxisStyleSpec) to one `AxisInput`,
/// filling only fields the input has not already set (so a higher-precedence
/// source — a per-channel spec, or an earlier config layer — always wins).
///
/// Shared by the chart-level `configure_axis` path and the per-channel
/// `EncodingSpec.axis` path (B5 fix). Honored styling keys (grid color/dash/width,
/// label color/font-size, domain color/width) flow into per-axis override fields
/// on `AxisInput` that `build_axis`/`build_grid` consult with a theme fallback, so
/// they render per-axis instead of mutating the shared theme. The actual field
/// merge is the canonical [`axis_style_fill_from`] (chart-level discipline:
/// `fill_only_if_none = true`).
///
/// Show toggles (`grid`/`domain`/`labels`/`ticks`) are deliberately NOT written
/// from the chart-level path. The per-channel prepare path
/// (`prepare_render_inputs`) is the sole owner of `AxisInput.show_*`, so a
/// per-channel `Axis(grid=False)` wins over a conflicting chart-level
/// `configure_axis(grid=True)`. The chart-level toggle still takes effect through
/// its global theme/gate path: `configure_axis` maps `grid`→`theme.grid.grid` and
/// `domain`→`theme.axis.axis_line` in `apply_axis_config_to_theme`, and
/// `build_grid`/`build_axis` AND that global gate with the per-axis `show_*` gate.
/// Writing `show_*` here would clobber the per-channel value and invert the
/// precedence — which is why `axis_style_fill_from` (operating on
/// `AxisStyleOverrides`, which has no `show_*` fields) structurally cannot.
pub(crate) fn apply_axis_style_to_axis_input(
    axis: &mut crate::layout::AxisInput,
    style: &chart_config::AxisStyleSpec,
) -> Result<(), RenderError> {
    let channel = axis_channel(axis.orient);
    // R3 EXEMPT chain: `channel` here is the axis's PHYSICAL orientation
    // (`axis_channel`), not a channel that traveled through `build_layers`'
    // swap — resolve with `false` explicitly. See the three-chain account on
    // `RenderError::InvalidAxisOrient` and the `chart_level_orient_error_names_resolved_axis_under_flip`
    // test pinning this.
    axis_style_fill_from(&mut axis.overrides, style, channel, true).map_err(|e| e.resolve(false))
}

/// Re-format tick label strings using a d3-format string override.
///
/// When `tick_values_override` is set, tick_labels are replaced entirely
/// with formatted versions of the explicit tick values. If a
/// `label_format_override` is also provided the values are formatted using
/// that d3-format spec; otherwise they are converted to plain decimal strings.
///
/// When only `label_format_override` is set (no explicit tick values), each
/// existing tick label is parsed as a float and reformatted via the d3-format
/// spec. Non-numeric labels (category names, time strings) are passed through
/// unchanged.
fn apply_label_format_to_axis(axis: &mut crate::layout::AxisInput) {
    if let Some(tick_vals) = axis.overrides.tick_values.clone() {
        // Replace tick_labels with formatted versions of the explicit tick_values.
        let numeric_strings: Vec<String> = tick_vals.iter().map(|v| v.to_string()).collect();
        let fmt = axis.overrides.label_format.as_deref();
        axis.tick_labels = prepare::apply_tick_format(numeric_strings, fmt, None);
    } else if let Some(ref fmt_str) = axis.overrides.label_format.clone() {
        // Re-format existing labels using the d3-format spec.
        axis.tick_labels = prepare::apply_tick_format(
            std::mem::take(&mut axis.tick_labels),
            Some(fmt_str),
            None,
        );
    }
}

/// Continuous-axis scale projection: when `tick_values_override` replaced the
/// axis labels with explicit data values, recompute the `tick_projection`'s
/// `major` fractions from those values via the scale so the carrier matches the
/// new labels. Only acts on continuous axes that already carry a projection
/// (categorical axes have `None` and keep uniform-slot placement). When the
/// scale yields no fractions (e.g. ordinal), the carrier is cleared so layout
/// falls back to uniform slots rather than indexing a stale vec.
fn sync_projected_fractions_to_tick_values(
    axis: &mut crate::layout::AxisInput,
    scale: &scale_resolve::ScaleKind,
) {
    if axis.tick_projection.is_none() {
        return;
    }
    let Some(values) = axis.overrides.tick_values.clone() else {
        return;
    };
    let fractions = scale.value_fractions(&values);
    if fractions.is_empty() {
        // Scale yields no fractions (e.g. ordinal / degenerate domain): clear the
        // carrier so layout falls back to uniform slots rather than indexing a
        // stale vec. The minor carrier is dropped in lockstep — empty
        // `value_fractions` implies an axis with no continuum, so its minors are
        // already empty.
        axis.tick_projection = None;
    } else if let Some(proj) = axis.tick_projection.as_mut() {
        proj.major = fractions;
    }
}

/// Apply `ChartConfig.color.domain` and `color.range` overrides to the
/// resolved color scale. `pub(crate)` so `scene_build` can call it after
/// per-panel scale resolution (which re-resolves the scale independently of
/// the provisional_scales copy patched in `render_svg`).
///
/// Per-encoding `Color(scale=...)` overrides (level 2) have already been
/// applied during scale resolution. This function applies `configure_color(...)`
/// overrides (level 3) only when the per-encoding value is absent (level 2
/// wins over level 3).
///
/// Continuous scales: `domain` pins the data extent; `range` builds an evenly
/// spaced gradient from the provided hex stops.
///
/// Categorical scales: `domain` (a list of strings) replaces the resolved
/// category order — entries are drawn in the listed order, categories absent
/// from the list are dropped from the scale, and listed values absent from the
/// data are kept (an empty legend entry, matching positional explicit-domain
/// behavior). A dropped category's *marks still draw*, in the theme mark color
/// with no legend entry; the omitted names are reported as a
/// [`RenderWarning::ColorDomainOmitsCategories`] so that sanctioned degradation
/// is distinguishable from a rendering bug (spec §4.2, amended 2026-08-28).
/// `range` replaces the palette with a heap-allocated `Cow::Owned` slice of
/// parsed colors, all-or-nothing: one unparseable entry discards the whole
/// range and reports [`RenderWarning::ColorRangeParseFailure`], matching the
/// Discretizing arm and `build_color_scale`'s `explicit_string_range` path.
/// This arm cannot use the "silently skip" convention the Continuous arm keeps:
/// a categorical palette is indexed by domain position, so a dropped entry
/// re-points every later category rather than shortening a cycle (spec §4.2,
/// amended 2026-09-02).
///
/// Discretizing scales: `range` replaces the bucket swatches, all-or-nothing —
/// one unparseable entry discards the whole range and reports
/// [`RenderWarning::ColorRangeParseFailure`], matching `bucket_colors` and the
/// categorical explicit-range path. The bucket count is fixed by the scale's own
/// thresholds, so a range of any other length cannot describe the partition; it
/// is left unapplied and reported as a
/// [`RenderWarning::ColorRangeBucketCountMismatch`] naming both counts (spec
/// §4.2, amended 2026-08-28 — never a silent drop). `domain` is inapplicable
/// here: the boundaries come from the scale spec.
///
/// On the Continuous arm alone, invalid color strings are silently skipped and
/// the override is a no-op when fewer than two stops parse — a pre-existing
/// convention kept as-is, since a dropped stop there re-spaces a gradient
/// (`t = i / (n - 1)` over whatever parsed) rather than shifting a
/// position-indexed mapping. The Categorical and Discretizing arms are both
/// all-or-nothing with a `ColorRangeParseFailure`, because both index by
/// position.
///
/// Returns the warnings the override produced, for the caller to fold into its
/// accumulator. Empty for every chart with no `configure_color(...)`.
#[must_use]
pub(crate) fn apply_color_config_to_color_scale(
    color_scale: &mut Option<scale_resolve::ColorScale>,
    cfg: &chart_config::ColorConfigSpec,
) -> Vec<RenderWarning> {
    let mut warnings = Vec::new();
    let Some(ref mut scale) = color_scale else { return warnings };
    match scale {
        scale_resolve::ColorScale::Continuous { ref mut domain, ref mut scheme, .. } => {
            if let Some(ref d) = cfg.domain {
                let floats: Vec<f64> = d.iter().filter_map(|v| v.as_f64()).collect();
                if floats.len() == 2 {
                    *domain = (floats[0], floats[1]);
                }
            }
            if let Some(ref r) = cfg.range {
                if r.len() >= 2 {
                    let parsed: Vec<color::Color> = r
                        .iter()
                        .filter_map(|s| color::parse_color(s).ok())
                        .collect();
                    if parsed.len() >= 2 {
                        // Build a gradient from the parsed stops, evenly spaced.
                        let stops: Vec<(f64, color::Color)> = parsed
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| {
                                let t = i as f64 / (parsed.len() - 1) as f64;
                                (t, c)
                            })
                            .collect();
                        *scheme = color::ContinuousScheme::Gradient(stops);
                    }
                }
            }
        }
        scale_resolve::ColorScale::Categorical { ref mut domain, ref mut palette } => {
            if let Some(ref d) = cfg.domain {
                let categories: Vec<String> =
                    d.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect();
                if !categories.is_empty() {
                    // Categories the override leaves unlisted keep rendering —
                    // in the theme mark color, with no legend entry (spec §4.2,
                    // amended 2026-08-28). Sanctioned, but indistinguishable
                    // from a rendering bug on sight, so name them.
                    let omitted: Vec<String> = domain
                        .iter()
                        .filter(|c| !categories.contains(c))
                        .cloned()
                        .collect();
                    if !omitted.is_empty() {
                        warnings.push(RenderWarning::ColorDomainOmitsCategories {
                            categories: omitted,
                        });
                    }
                    *domain = categories;
                }
            }
            if let Some(ref range) = cfg.range {
                // All-or-nothing, mirroring the Discretizing arm below and
                // `build_color_scale`'s `explicit_string_range` path. A
                // categorical palette is indexed by DOMAIN POSITION
                // (`ColorScale::lookup` → `palette[i % palette.len()]`), so
                // dropping an unparseable entry does not merely shorten a
                // cycle: it re-points every category after the dropped one.
                // `range=["red", "notacolor", "blue"]` over domain `[a, b, c]`
                // silently rendered `a=red, b=blue, c=red` — a wrong mapping,
                // not a degraded one. One bad entry now discards the whole
                // range and reports the offending string, leaving the resolved
                // palette in place (spec §4.2, amended 2026-09-02).
                let parsed: Result<Vec<color::Color>, &String> = range
                    .iter()
                    .map(|s| color::parse_color(s).map_err(|_| s))
                    .collect();
                match parsed {
                    Ok(colors) if !colors.is_empty() => {
                        *palette = std::borrow::Cow::Owned(colors);
                    }
                    // An empty `range=[]` describes no palette at all; left
                    // unapplied with no warning, as before.
                    Ok(_) => {}
                    Err(entry) => warnings.push(RenderWarning::ColorRangeParseFailure {
                        entry: entry.clone(),
                    }),
                }
            }
        }
        scale_resolve::ColorScale::Discretizing(ref mut buckets) => {
            if let Some(ref range) = cfg.range {
                // All-or-nothing, mirroring every other explicit-range parse in
                // the crate (`bucket_colors`, the categorical explicit-range
                // path). Dropping unparseable entries first would make the
                // count check see a post-filter length: a 4-entry range with one
                // bad entry would "fit" a 3-bucket scale and silently repaint it
                // with a shifted mapping, and a 3-entry range with one bad entry
                // would report a count mismatch for what is really a parse
                // failure.
                let parsed: Result<Vec<color::Color>, &String> = range
                    .iter()
                    .map(|s| color::parse_color(s).map_err(|_| s))
                    .collect();
                match parsed {
                    Ok(colors) => {
                        if let Err(mismatch) = buckets.set_colors(colors) {
                            warnings.push(RenderWarning::ColorRangeBucketCountMismatch {
                                expected: mismatch.expected as u32,
                                received: mismatch.received as u32,
                            });
                        }
                    }
                    Err(entry) => warnings.push(RenderWarning::ColorRangeParseFailure {
                        entry: entry.clone(),
                    }),
                }
            }
        }
    }
    warnings
}

/// Re-derive the color legend's entry labels from the categorical scale domain.
///
/// `prepare_render_inputs` builds the entries from the domain as resolved, but
/// `configure_color(domain=[…])` reorders/subsets that domain afterwards
/// ([`apply_color_config_to_color_scale`]), so without this the swatch colors
/// would follow the new order while the labels kept the old one. Entry symbols
/// are preserved (a category that survives keeps its symbol; a newly listed one
/// takes the first entry's).
///
/// A no-op — and therefore byte-identical — whenever the domain still matches
/// the entries, which is every chart that does not set `configure_color(domain=)`
/// on a categorical color scale.
fn resync_categorical_legend_entries(
    entries: &mut Vec<crate::layout::LegendEntry>,
    color_scale: Option<&scale_resolve::ColorScale>,
) {
    let Some(domain) = color_scale.and_then(scale_resolve::ColorScale::categorical_domain) else {
        return;
    };
    let Some(first) = entries.first() else { return };
    if entries.len() == domain.len() && entries.iter().zip(domain).all(|(e, d)| &e.label == d) {
        return;
    }
    let default_symbol = first.symbol;
    *entries = domain
        .iter()
        .map(|label| crate::layout::LegendEntry {
            label: label.clone(),
            symbol: entries
                .iter()
                .find(|e| &e.label == label)
                .map_or(default_symbol, |e| e.symbol),
        })
        .collect();
}

/// Output of the shared prepare-and-layout pipeline, consumed by both
/// `render_svg` and `render_scene_json`.
struct PipelineOutput {
    prep: prepare::PreparedInputs,
    layout: crate::layout::LayoutResult,
    effective_theme: ThemeInputs,
    warnings: Vec<RenderWarning>,
}

/// Shared pipeline executed by both `render_svg` and `render_scene_json`.
///
/// Performs in order:
///   1. `prepare_render_inputs` — transforms, scale resolution, axis inputs.
///   2. ChartConfig axis overrides (level 3 > level 2 per-encoding).
///   3. Color domain/range overrides (level 3).
///   4. Effective-theme construction: per-encoding legend overrides → ChartConfig overrides.
///   5. Secondary-Y right-padding reservation (fixes missing block in render_scene_json).
///   6. Legend-title resolution.
///   7. `compute_layout`.
///
/// The viewport must already have been validated (both dimensions > 0) and
/// overridden with `RenderConfig.width` / `RenderConfig.height` by the caller.
fn prepare_and_layout(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    chart_config: &ChartConfig,
    // D4b composite seam: the resolved shared-domain context for this leaf,
    // forwarded into the provisional scale pass. `None` for every standalone
    // (flat/facet) render → byte-identical output.
    leaf_scales: Option<&scale_resolve::LeafScaleContext>,
) -> Result<PipelineOutput, RenderError> {
    let mut prep = prepare::prepare_render_inputs(spec, batch, theme, leaf_scales)?;
    let mut warnings = prep.warnings.clone();

    // Chart-level `configure_legend(orient="none")` suppression (GH #74).
    // Python's `_resolve_chart_config` (`_render.py`) maps a fully-merged
    // `orient="none"` onto the same `disabled` signal `Color(legend=None)`
    // sets at the per-channel level (`chart_config_legend_disabled` reads
    // it here) — there is no `LegendOrient::None` variant on the Rust side.
    // `prepare::prepare_render_inputs` only reads the per-channel
    // `encoding.<channel>.legend.disabled` flag (Schwabish SB3), so the
    // chart-level signal is applied here by clearing the already-built
    // legend content — the exact same empty state that flag already
    // produces. One seam covers every consumer: `render_svg` /
    // `render_scene_json` call this function directly for a standalone
    // chart, and `composite_render::render_leaf` calls this SAME function
    // with the leaf's own `chart_config` for every composite leaf. A
    // suppressed leaf therefore draws no per-panel legend AND yields an
    // empty `LeafLegendBundle`, which `apply_legend_band` already treats as
    // "no content to capture" — the identical degrade an all-disabled
    // `Color(legend=None)` composite produces today (design §4/§9.8).
    if chart_config_legend_disabled(chart_config) {
        prep.legend_entries.clear();
        prep.colorbar = None;
        prep.aux_legends.clear();
        prep.legend_title = None;
        prep.legend_overrides.title = None;
    }

    // Apply ChartConfig axis overrides (level 3) to AxisInput (level 2 wins when already set).
    // These styling fields use fill-only-if-`None` (first writer claims the slot), so the
    // MORE-SPECIFIC source must run FIRST: apply the per-axis `axis_x`/`axis_y` keys before
    // the shared `axis` key so `axis_x > axis` precedence holds (documented in configure.py).
    // (This is the OPPOSITE order from the overwrite-semantics theme path in
    // `apply_chart_config`, where last-writer-wins makes `axis` run first then `axis_x`.)
    // R3 (chart-level `configure_axis` chain): deliberately EXEMPT from
    // `with_coord_flipped` — this is the mirror image of the `SortSpecIgnored`
    // exemption (`scale_resolve/domain.rs`'s `apply_channel_shorthand_sort`).
    // `apply_axis_config_to_axis_input` → `apply_axis_style_to_axis_input`
    // derives `channel` from the axis's PHYSICAL orientation (`axis_channel`,
    // just below: `Top|Bottom → "x"`, else `"y"`), never from a user-written
    // encoding channel. And the config KEY the user actually typed here —
    // `axis_x`/`axis_y` (or the shared `axis`) — is itself resolved-slot
    // vocabulary: nothing on the Python side remaps `configure_axis(axis_x=…)`
    // to `axis_y=` under `CoordFlip` (`configure.py`, `_override_apply.py`,
    // `_override_consume.py`). So for THIS chain the resolved token already
    // IS what the user wrote — `prep.axes.x` is unconditionally the physical
    // bottom axis, flip or not (flip is implemented purely as the `x`/`y`
    // encoding swap in `prepare::build_layers`; nothing re-orients axes).
    // Applying `with_coord_flipped` here would translate the user's own typed
    // config key AWAY from what they wrote — see
    // `chart_level_orient_error_names_resolved_axis_under_flip` below.
    apply_axis_config_to_axis_input(&mut prep.axes.x, chart_config.axis_x.as_ref())?;
    apply_axis_config_to_axis_input(&mut prep.axes.y, chart_config.axis_y.as_ref())?;
    apply_axis_config_to_axis_input(&mut prep.axes.x, chart_config.axis.as_ref())?;
    apply_axis_config_to_axis_input(&mut prep.axes.y, chart_config.axis.as_ref())?;
    // Re-sync the concrete axis side from the merged `overrides.orient`: a
    // per-channel `fm.Axis(orient=...)` already set it (so this is a no-op there
    // and per-channel wins), otherwise a chart-level `configure_axis(orient=...)`
    // filled it above and now takes effect.
    prep.axes.x.resolve_orient();
    prep.axes.y.resolve_orient();
    // tick_extra / tick_min_step (B5 unit 2): apply AFTER the config merge so the
    // effective value (per-channel wins, chart-level fallback) is on `AxisInput`,
    // then adjust the generated ticks against the provisional scale. No-op when
    // neither field is set, so default output is byte-identical. The non-ordinal
    // y labels/fractions were reversed in prepare, so the raw values are reversed
    // in lockstep.
    let y_reversed =
        !matches!(prep.provisional_scales.y, scale_resolve::ScaleKind::Ordinal(_));
    let (x_tc, y_tc) = (prep.x_tick_count, prep.y_tick_count);
    prepare::adjust_axis_ticks(&mut prep.axes.x, &prep.provisional_scales.x, x_tc, false);
    prepare::adjust_axis_ticks(&mut prep.axes.y, &prep.provisional_scales.y, y_tc, y_reversed);
    // Apply label_format_override to tick labels (requires axis config to be set first).
    apply_label_format_to_axis(&mut prep.axes.x);
    apply_label_format_to_axis(&mut prep.axes.y);
    // Continuous-axis scale projection: when explicit `tick_values` replaced the
    // auto tick labels, recompute the projected fractions from those same values
    // so the carrier stays index-aligned with the new labels. The explicit
    // labels are NOT reversed for y (unlike auto labels), so the value-order
    // fractions align directly.
    sync_projected_fractions_to_tick_values(&mut prep.axes.x, &prep.provisional_scales.x);
    sync_projected_fractions_to_tick_values(&mut prep.axes.y, &prep.provisional_scales.y);
    // Apply color domain/range overrides (level 3) to resolved color scale.
    // This is the ONE reporting application: `scene_build`'s per-panel and
    // per-legend re-applications run the same config against the same scale, so
    // they discard their (identical) warnings rather than emit one per panel.
    if let Some(ref cfg) = chart_config.color {
        warnings.extend(apply_color_config_to_color_scale(
            &mut prep.provisional_scales.color,
            cfg,
        ));
        resync_categorical_legend_entries(
            &mut prep.legend_entries,
            prep.provisional_scales.color.as_ref(),
        );
    }

    // Build effective_theme: start from the caller-supplied theme, then layer in
    // overrides from lowest to highest priority within the "render" concern:
    //   1. D13 per-encoding legend overrides (from encoding.color.legend.*).
    //   2. ChartConfig overrides (configure_*() calls, level 3 > theme level 4-5).
    // Per-channel axis overrides (level 2) are in AxisInput and applied at layout time.
    let mut effective_theme = theme.clone();
    // D13: per-encoding legend overrides.
    if let Some(orient) = prep.legend_overrides.orient {
        effective_theme.legend.legend_orient = orient;
    }
    if let Some(fs) = prep.legend_overrides.title_font_size {
        effective_theme.typography.legend_title_font_size = fs;
    }
    if let Some(cols) = prep.legend_overrides.columns {
        effective_theme.legend.legend_columns = Some(cols);
    }
    // ChartConfig overrides (configure > theme).
    apply_chart_config(&mut effective_theme, chart_config);

    // D13 + v0.15.1: legend title override (replaces the default field-name title when Some).
    let title_for_layout = effective_legend_title(&prep);

    let mut legend_overrides = legend_overrides_from_prep(&prep);
    // Apply configure_legend overrides (level 3) — only fills in None fields.
    apply_chart_config_to_legend_overrides(&mut legend_overrides, chart_config);
    let metrics = font::FontdueMetrics::new();
    // Composite → layout seam: everything a composite parent imposes on this
    // leaf's layout, threaded through the same `leaf_scales` context the
    // composite already uses for shared-domain resolution (D4b).
    //
    //   - legend suppression (design §6, 2026-07-12): `prep` above was built
    //     from the SAME `leaf_scales`, so its legend bundle
    //     (`legend_entries`/`colorbar`/`legend_title`/`aux_legends`/overrides)
    //     is always fully populated here regardless of suppression — only
    //     this layout call's reservation/draw behavior is gated.
    //   - the overlay group's shared plot region (GH #89A): replaces the
    //     region this leaf's own axis-band reservation would produce, so
    //     every layout product below is computed against the group's one
    //     rect.
    //   - chart-title suppression (GH #89A): set for a leaf whose title the
    //     composite clears, so the band it would reserve cannot inflate that
    //     shared rect.
    //
    // `None` (every standalone render) reproduces today's layout byte-for-byte.
    let seam = leaf_scales
        .map(|ctx| CompositeLayoutSeam {
            legend: LegendSuppression { color: ctx.suppress_color_legend, size: ctx.suppress_size_legend },
            plot_region: ctx.imposed_plot_region,
            suppress_chart_title: ctx.suppress_chart_title,
        })
        .unwrap_or_default();
    let layout = compute_layout(
        spec,
        &effective_theme,
        viewport,
        &prep.axes,
        &prep.facet_groups,
        &prep.legend_entries,
        title_for_layout,
        prep.colorbar.as_ref(),
        &metrics,
        &legend_overrides,
        &prep.aux_legends,
        seam,
    )
    .map_err(|e| RenderError::LayoutFailed(e.to_string()))?;
    for w in &layout.warnings {
        warnings.push(RenderWarning::Layout(w.clone()));
    }

    Ok(PipelineOutput { prep, layout, effective_theme, warnings })
}

pub fn render_svg(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<RenderOutput<String>, RenderError> {
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(RenderError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }

    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };

    let PipelineOutput { prep, layout, effective_theme, mut warnings } =
        prepare_and_layout(spec, batch, theme, viewport, chart_config, None)?;

    let scene = scene_build::build_scene(
        spec, &prep, &layout, &effective_theme, config, &mut warnings, chart_config, None,
    )?;
    let svg_string = svg_walk::walk_svg(&scene, config.embed_fonts);

    Ok(RenderOutput { bytes: svg_string, layout, warnings })
}

pub fn render_png(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<RenderOutput<Vec<u8>>, RenderError> {
    let svg_out = render_svg(spec, batch, theme, viewport, config, chart_config)?;
    let w = (svg_out.layout.viewport.w * config.scale).round() as u32;
    let h = (svg_out.layout.viewport.h * config.scale).round() as u32;
    let bytes = png::svg_string_to_png_bytes(&svg_out.bytes, w, h, config.scale)?;
    Ok(RenderOutput { bytes, layout: svg_out.layout, warnings: svg_out.warnings })
}

/// Render a chart to the interactive scene wire pair (JSON + packed bytes).
///
/// Returns `(json, packed_bytes, warnings)`. The `warnings` were silently
/// dropped from this fn's return until GH #50: they were computed (via
/// `prepare_and_layout` / `build_scene`, same as [`render_svg`]/[`render_png`])
/// but never left the function, so `render_interactive`'s PyO3 binding had no
/// warnings to forward to Python — an asymmetry with `render_svg`/`render_png`,
/// which both surface warnings through [`RenderOutput`]. Returning them here
/// closes that gap; the caller now calls `emit_warnings` the same way the
/// other two entries do.
pub fn render_scene_json(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<(String, Vec<u8>, Vec<RenderWarning>), RenderError> {
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(RenderError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }

    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };

    let PipelineOutput { prep, layout, effective_theme, mut warnings } =
        prepare_and_layout(spec, batch, theme, viewport, chart_config, None)?;

    let mut scene = scene_build::build_scene(
        spec, &prep, &layout, &effective_theme, config, &mut warnings, chart_config, None,
    )?;

    // Extract large homogeneous mark batches as raw packed bytes, clearing
    // their nodes from the scene graph. The JSON stays lightweight; the
    // packed bytes travel as a separate Uint8Array to the WASM renderer.
    let packed_bytes = pack_instances::extract_packed_bytes(&mut scene);

    let json = serde_json::to_string(&scene)
        .map_err(|e| RenderError::LayoutFailed(format!("scene serialization: {e}")))?;
    Ok((json, packed_bytes, warnings))
}

pub(crate) fn filter_batch_by_facet(
    batch: &RecordBatch,
    field: &str,
    value: &str,
) -> Result<RecordBatch, RenderError> {
    use arrow::array::{Array, BooleanArray, StringArray};
    use arrow::compute::filter_record_batch;
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!("facet field '{field}' must be Utf8"))
        })?;
    let mask: BooleanArray = arr
        .iter()
        .map(|v| Some(v.map(|s| s == value).unwrap_or(false)))
        .collect();
    filter_record_batch(batch, &mask)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("filter: {e}")))
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn scatter_3() -> (ChartSpec, RecordBatch) {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    #[test]
    fn render_svg_minimal_scatter() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let svg = result.bytes;
        assert!(svg.starts_with("<svg "));
        assert!(svg.ends_with("</svg>"));
        assert_eq!(svg.matches("<circle ").count(), 3);
        assert!(svg.contains("@font-face"));
    }

    #[test]
    fn render_svg_invalid_viewport_errors() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let result = render_svg(
            &spec,
            &batch,
            &theme,
            Viewport { width: 0.0, height: 100.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        );
        assert!(matches!(result.unwrap_err(), RenderError::InvalidViewport { .. }));
    }

    // ── T12: stroke_dash consumption end-to-end (spec §4.3) ─────────────────
    // Audit F-L01/F-L07: `relplot(style=)` used to warn about dash recycling
    // while drawing zero `stroke-dasharray` attributes — the T6 `StrokeDashScale`
    // had no consumer. These prove the consumer through the full `render_svg`
    // pipeline: a categorical field draws distinct dasharrays with a legend, a
    // numeric field keeps the pre-T12 `DASH_PALETTE` index contract, and the
    // palette-overflow warning now corresponds to real drawn dashes.

    fn line_spec_with_stroke_dash(color_field: Option<&str>) -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: color_field.map(|f| EncodingSpec { field: f.into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        }
    }

    /// Headline fix: a categorical `stroke_dash` field draws N distinct
    /// polylines (one per style category, spec §4.3's (color, detail,
    /// dash-field) partition extension) with N distinct dasharray patterns,
    /// plus a stroke-dash aux legend entry per category.
    #[test]
    fn render_svg_categorical_stroke_dash_draws_distinct_polylines_and_legend() {
        let spec = line_spec_with_stroke_dash(None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sd", DataType::Utf8, false),
        ]));
        // 3 style categories (first-appearance order: solid, dashed, dotted),
        // 3 points each so every polyline connects (points.len() >= 2).
        let xs: Vec<f64> = (0..9).map(|i| (i % 3) as f64).collect();
        let ys: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let sds: Vec<&str> = (0..9).map(|i| match i / 3 { 0 => "solid", 1 => "dashed", _ => "dotted" }).collect();
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(sds)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let svg = &result.bytes;

        assert_eq!(svg.matches("<polyline ").count(), 3,
            "3 distinct style categories must draw 3 polylines, not 1 merged line: {svg}");
        // "solid" cycles to DASH_PALETTE slot 0 (no attribute); "dashed"/"dotted"
        // cycle to slots 1/2, each carrying a distinct dasharray — once on the
        // data polyline, once more on its aux-legend swatch line (4 total).
        assert_eq!(svg.matches("stroke-dasharray=").count(), 4,
            "2 of the 3 categories (dashed, dotted) must carry a stroke-dasharray \
             attribute on both their polyline and legend swatch: {svg}");
        assert!(svg.contains("stroke-dasharray=\"6,3\""), "expected the long-dash pattern: {svg}");
        assert!(svg.contains("stroke-dasharray=\"2,3\""), "expected the short-dash pattern: {svg}");

        // Aux legend: one entry per dash category (not suppressed — no color
        // encoding shares the field).
        assert_eq!(result.layout.aux_legends.len(), 1, "expected one stroke-dash aux legend block");
        let entries = &result.layout.aux_legends[0].entries;
        assert_eq!(entries.len(), 3, "3 dash categories -> 3 legend entries");
        let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, vec!["solid", "dashed", "dotted"]);
        for label in ["solid", "dashed", "dotted"] {
            assert!(svg.contains(&format!(">{label}<")), "legend must render the '{label}' label: {svg}");
        }
    }

    /// Numeric byte-identity (spec §6): a numeric `stroke_dash` column keeps
    /// resolving no scale and reading `DASH_PALETTE` indices directly — the
    /// T12 consumer wiring must not disturb this pre-existing contract.
    #[test]
    fn render_svg_numeric_stroke_dash_keeps_dash_palette_index_contract() {
        let spec = line_spec_with_stroke_dash(Some("g"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("sd", DataType::Float64, false),
        ]));
        // 3 color groups x 3 rows; each group's stroke_dash column carries a
        // single DASH_PALETTE index (0 solid, 1 long dash, 2 short dash) —
        // the group's first row is what the pre-T12 code sampled too.
        let xs: Vec<f64> = (0..9).map(|i| (i % 3) as f64).collect();
        let ys: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let gs: Vec<&str> = (0..9).map(|i| match i / 3 { 0 => "a", 1 => "b", _ => "c" }).collect();
        let sds: Vec<f64> = (0..9).map(|i| (i / 3) as f64).collect();
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(gs)),
            Arc::new(Float64Array::from(sds)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let svg = &result.bytes;

        assert_eq!(svg.matches("<polyline ").count(), 3, "3 color groups must draw 3 polylines: {svg}");
        assert_eq!(svg.matches("stroke-dasharray=").count(), 2,
            "index 0 (solid) carries no attribute; indices 1/2 do: {svg}");
        assert!(svg.contains("stroke-dasharray=\"6,3\""));
        assert!(svg.contains("stroke-dasharray=\"2,3\""));
        // A numeric stroke_dash field never resolves a StrokeDashScale, so no
        // aux legend is built for it.
        assert!(result.layout.aux_legends.is_empty(),
            "a numeric stroke_dash field must not produce a dash aux legend");
    }

    /// Palette overflow (spec §4.3): more style categories than
    /// `DASH_PALETTE` slots (4: solid + 3 patterns) still draws every
    /// polyline — cycling the palette — while emitting
    /// `RenderWarning::StrokeDashPaletteOverflowed`. Before T12 this warning
    /// fired over a chart with zero dasharray attributes; now the warning
    /// corresponds to real (cycled) drawn dashes.
    #[test]
    fn render_svg_stroke_dash_overflow_warns_and_still_draws_cycled_dashes() {
        let spec = line_spec_with_stroke_dash(None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sd", DataType::Utf8, false),
        ]));
        // 5 categories (> 4 slots), 2 rows each.
        let xs: Vec<f64> = (0..10).map(|i| (i % 2) as f64).collect();
        let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let sds: Vec<String> = (0..10).map(|i| format!("cat{}", i / 2)).collect();
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(sds)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();

        assert!(
            result.warnings.iter().any(|w| matches!(
                w,
                RenderWarning::StrokeDashPaletteOverflowed { categories: 5 }
            )),
            "expected a StrokeDashPaletteOverflowed{{categories: 5}} warning; got {:?}",
            result.warnings
        );
        let svg = &result.bytes;
        assert_eq!(svg.matches("<polyline ").count(), 5, "all 5 categories must still draw, cycling the palette: {svg}");
        assert!(svg.matches("stroke-dasharray=").count() >= 1,
            "the overflow warning must correspond to real drawn dashes, not a silent no-op: {svg}");
    }

    /// Fix round (T12 spec review Issue 1, spec §4.3 amended 2026-09-01): a
    /// categorical `stroke_dash` on **ribbon** must partition paths, never
    /// render one merged path under a multi-entry dashed legend. Mirrors
    /// `render_svg_categorical_stroke_dash_draws_distinct_polylines_and_legend`
    /// above but for `Mark::Ribbon`: 3 style categories -> 3 `<path>` bands
    /// with 3 distinct dasharray values (one per non-solid category, doubled
    /// by the legend swatch), and the aux legend's 3 entries/labels agree
    /// with what actually drew.
    #[test]
    fn render_svg_categorical_stroke_dash_on_ribbon_draws_distinct_paths_and_legend() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("sd", DataType::Utf8, false),
        ]));
        // 3 style categories (first-appearance order: solid, dashed, dotted),
        // 3 rows each so every band connects (indices.len() >= 2 after the
        // group's connectivity gate).
        let xs: Vec<f64> = (0..9).map(|i| (i % 3) as f64).collect();
        let ys: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let y2s: Vec<f64> = (0..9).map(|i| i as f64 + 1.0).collect();
        let sds: Vec<&str> = (0..9).map(|i| match i / 3 { 0 => "solid", 1 => "dashed", _ => "dotted" }).collect();
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(Float64Array::from(y2s)),
            Arc::new(StringArray::from(sds)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let svg = &result.bytes;

        assert_eq!(svg.matches("<path ").count(), 3,
            "3 distinct style categories must draw 3 ribbon bands, not 1 merged path: {svg}");
        // "solid" cycles to DASH_PALETTE slot 0 (no attribute); "dashed"/"dotted"
        // cycle to slots 1/2, each carrying a distinct dasharray — once on the
        // data path, once more on its aux-legend swatch line (4 total).
        assert_eq!(svg.matches("stroke-dasharray=").count(), 4,
            "2 of the 3 categories (dashed, dotted) must carry a stroke-dasharray \
             attribute on both their band and legend swatch: {svg}");
        assert!(svg.contains("stroke-dasharray=\"6,3\""), "expected the long-dash pattern: {svg}");
        assert!(svg.contains("stroke-dasharray=\"2,3\""), "expected the short-dash pattern: {svg}");

        // Legend agrees with the drawing: one aux-legend entry per category
        // actually drawn, same labels, same order.
        assert_eq!(result.layout.aux_legends.len(), 1, "expected one stroke-dash aux legend block");
        let entries = &result.layout.aux_legends[0].entries;
        assert_eq!(entries.len(), 3, "3 dash categories -> 3 legend entries");
        let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, vec!["solid", "dashed", "dotted"]);
        for label in ["solid", "dashed", "dotted"] {
            assert!(svg.contains(&format!(">{label}<")), "legend must render the '{label}' label: {svg}");
        }
    }

    /// T8 quality-review finding 2, end-to-end through `render_svg`: a
    /// `mark_ribbon`-shaped chart with `mark_kwargs={"stroke": "none"}`
    /// (exactly what `src/ferrum/marks/composite.py`'s ribbon/errorband
    /// desugar passes) must not emit `stroke="rgba(0,0,0,0.000)"` on the
    /// ribbon's `<path>` — the cleared paint has to normalize away before it
    /// reaches the SVG (or interactive scene JSON) serialization, not just
    /// at the `MarkStyle` level pinned in `draw.rs`/`marks/ribbon.rs`.
    #[test]
    fn render_svg_ribbon_cleared_stroke_emits_no_stroke_attribute() {
        use crate::spec::mark_style::MarkKwargsSpec;
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: Some(MarkKwargsSpec { stroke: Some("none".into()), ..Default::default() }),
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0])),
            ],
        )
        .unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result =
            render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let svg = result.bytes;
        assert!(
            !svg.contains("rgba(0,0,0,0.000)") && !svg.contains("rgba(0, 0, 0, 0.000)"),
            "a cleared stroke must never serialize as an explicit zero-alpha color; svg={svg:?}"
        );
        let path = svg.find("<path ").map(|i| &svg[i..]).expect("ribbon must emit a <path>");
        let path = &path[..path.find('>').unwrap_or(path.len())];
        assert!(
            !path.contains("stroke="),
            "a cleared stroke must omit the stroke attribute entirely; path tag={path:?}"
        );
    }

    /// Build a `mark_point`-shaped `ChartSpec` with the given `fill=`/`opacity=`
    /// `mark_kwargs`, mirroring `ferrum.mark_point(fill=..., opacity=...)`'s
    /// wire shape. Shared by the four T8 quality-review c2 pins below.
    fn point_fill_opacity_spec(fill: Option<&str>, opacity: Option<f64>) -> (ChartSpec, RecordBatch) {
        use crate::spec::mark_style::MarkKwargsSpec;
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: Some(MarkKwargsSpec {
                fill: fill.map(String::from),
                opacity,
                ..Default::default()
            }),
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    /// Extract the first `<circle ...>` tag from an SVG string, for
    /// attribute-level assertions.
    fn first_circle_tag(svg: &str) -> &str {
        let tag = svg.find("<circle ").map(|i| &svg[i..]).expect("must emit a <circle>");
        &tag[..tag.find('>').unwrap_or(tag.len())]
    }

    /// T8 quality-review finding 1 (c2 scoping — provenance, never by value):
    /// `fill="#000000", opacity=0` composes to the same zero-alpha-black
    /// `Color` value as a genuine `"none"`/`"transparent"` clear, but it was
    /// never cleared — the paint must keep serializing as an explicit
    /// `rgba(0,0,0,0.000)`, exactly as pre-batch, per spec §7 byte-identity.
    #[test]
    fn render_svg_point_black_fill_opacity_zero_serializes_by_value_not_no_attribute() {
        let (spec, batch) = point_fill_opacity_spec(Some("#000000"), Some(0.0));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let tag = first_circle_tag(&result.bytes);
        assert!(
            tag.contains("fill=\"rgba(0,0,0,0.000)\""),
            "an un-cleared zero-alpha-black fill must serialize by value, not omit the attribute; circle tag={tag:?}"
        );
    }

    /// T8 quality-review finding 1 (c2 scoping): an explicit `fill="#00000000"`
    /// (8-digit hex) resolves the identical `Color` bytes as a clear sentinel
    /// by value, but is a real parsed color, never a clear — must serialize
    /// exactly as pre-batch.
    #[test]
    fn render_svg_point_explicit_zero_alpha_hex_serializes_by_value_not_no_attribute() {
        let (spec, batch) = point_fill_opacity_spec(Some("#00000000"), None);
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let tag = first_circle_tag(&result.bytes);
        assert!(
            tag.contains("fill=\"rgba(0,0,0,0.000)\""),
            "an explicit #00000000 fill must serialize by value, not omit the attribute; circle tag={tag:?}"
        );
    }

    /// T8 quality-review finding 1: the genuinely cleared-paint case — a
    /// literal `fill="none"` mark_kwargs — normalizes to `FillStroke.fill =
    /// None`, which the SVG walker (`push_fill_stroke`) serializes as the
    /// literal SVG `fill="none"` keyword (its pre-existing, correct rendering
    /// of an absent fill — unlike stroke, which omits the attribute entirely;
    /// `push_fill_stroke`'s fill/stroke asymmetry is unchanged by this fix).
    /// The pin is that it is `fill="none"`, never `fill="rgba(0,0,0,0.000)"`,
    /// distinguishing it from the two same-valued-but-not-cleared cases above
    /// by provenance.
    #[test]
    fn render_svg_point_fill_none_emits_svg_none_not_zero_alpha() {
        let (spec, batch) = point_fill_opacity_spec(Some("none"), None);
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let tag = first_circle_tag(&result.bytes);
        assert!(
            tag.contains("fill=\"none\""),
            "a user-cleared fill must serialize as the literal SVG none keyword; circle tag={tag:?}"
        );
        assert!(
            !result.bytes.contains("rgba(0,0,0,0.000)") && !result.bytes.contains("rgba(0, 0, 0, 0.000)"),
            "a cleared fill must never serialize as an explicit zero-alpha color; svg={:?}",
            result.bytes
        );
    }

    /// Control (T8 quality-review finding 1): a non-black zero-alpha color
    /// (`fill="#ff0000", opacity=0`) is unaffected by the fix either way — it
    /// never aliased `color::TRANSPARENT` by value in the first place (only
    /// zero-alpha *black* collides with the sentinel byte-for-byte) — and must
    /// keep serializing explicitly.
    #[test]
    fn render_svg_point_red_fill_opacity_zero_serializes_by_value_control() {
        let (spec, batch) = point_fill_opacity_spec(Some("#ff0000"), Some(0.0));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let tag = first_circle_tag(&result.bytes);
        assert!(
            tag.contains("fill=\"rgba(255,0,0,0.000)\""),
            "the control (non-black zero-alpha) fill must keep serializing by value; circle tag={tag:?}"
        );
    }

    #[test]
    fn render_svg_unknown_column_errors() {
        let (mut spec, batch) = scatter_3();
        spec.encoding.x = Some(EncodingSpec { field: "missing".into(), type_: None, ..Default::default() });
        let result = render_svg(
            &spec,
            &batch,
            &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        );
        assert!(matches!(result.unwrap_err(), RenderError::UnknownColumn { .. }));
    }

    #[test]
    fn render_svg_faceted_emits_strip_titles() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "c", "b", "c"])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let result = render_svg(
            &spec,
            &batch,
            &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        )
        .unwrap();
        let svg = result.bytes;
        assert!(svg.contains(">a<") || svg.contains(">a</text>"));
        assert!(svg.contains(">b<") || svg.contains(">b</text>"));
        assert!(svg.contains(">c<") || svg.contains(">c</text>"));
    }

    #[test]
    fn render_svg_determinism_two_calls_byte_identical() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let a = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let b = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }

    #[test]
    fn scene_graph_path_matches_old_path_scatter() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let cfg = config::RenderConfig::default();
        let old_svg = render_svg(&spec, &batch, &theme, viewport, &cfg, &ChartConfig::default()).unwrap().bytes;

        let prep = prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        let mut warnings = prep.warnings.clone();

        let mut effective_theme = theme.clone();
        if let Some(orient) = prep.legend_overrides.orient {
            effective_theme.legend.legend_orient = orient;
        }
        if let Some(fs) = prep.legend_overrides.title_font_size {
            effective_theme.typography.legend_title_font_size = fs;
        }
        if let Some(cols) = prep.legend_overrides.columns {
            effective_theme.legend.legend_columns = Some(cols);
        }
        apply_chart_config(&mut effective_theme, &ChartConfig::default());
        let theme_ref = &effective_theme;
        // Same three-way resolution as prepare_and_layout (v0.15.1 suppress fix).
        let effective_legend_title = match prep.legend_overrides.title.as_deref() {
            Some(s) if s.trim().is_empty() => None,
            Some(s) => Some(s.to_owned()),
            None => prep.legend_title.clone(),
        };

        let legend_overrides = legend_overrides_from_prep(&prep);
        let metrics = font::FontdueMetrics::new();
        let vp2 = Viewport {
            width: cfg.width.unwrap_or(viewport.width),
            height: cfg.height.unwrap_or(viewport.height),
        };
        let layout = compute_layout(
            &spec, theme_ref, vp2,
            &prep.axes, &prep.facet_groups, &prep.legend_entries,
            effective_legend_title, prep.colorbar.as_ref(), &metrics,
            &legend_overrides,
            &prep.aux_legends,
            CompositeLayoutSeam::default(),
        ).unwrap();
        for w in &layout.warnings {
            warnings.push(RenderWarning::Layout(w.clone()));
        }

        let scene = scene_build::build_scene(
            &spec, &prep, &layout, theme_ref, &cfg, &mut warnings,
            &ChartConfig::default(), None,
        ).unwrap();
        let new_svg = svg_walk::walk_svg(&scene, cfg.embed_fonts);

        if old_svg != new_svg {
            let old_chars: Vec<char> = old_svg.chars().collect();
            let new_chars: Vec<char> = new_svg.chars().collect();
            let first_diff = old_chars.iter().zip(new_chars.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(old_chars.len().min(new_chars.len()));
            let context_start = first_diff.saturating_sub(80);
            let context_end = (first_diff + 80).min(old_svg.len()).min(new_svg.len());
            panic!(
                "Scene graph SVG differs from old path at byte {}.\n\
                 OLD[{}..{}]: {:?}\n\
                 NEW[{}..{}]: {:?}\n\
                 old len={}, new len={}",
                first_diff,
                context_start, context_end, &old_svg[context_start..context_end.min(old_svg.len())],
                context_start, context_end, &new_svg[context_start..context_end.min(new_svg.len())],
                old_svg.len(), new_svg.len(),
            );
        }
    }

    // ── composite-shared-legend seam (design §6, 2026-07-12) ─────────────────

    /// x/y + a categorical `species` color encoding, 3 rows / 2 categories —
    /// enough for `prepare_render_inputs` to build a non-empty color legend
    /// bundle. `color_disabled` wires `legend={"disabled": true}` onto the
    /// color encoding to exercise the PREPARE-stage (user) suppression path,
    /// as opposed to the LAYOUT-stage (compositor) suppression under test.
    fn species_scatter(color_disabled: bool) -> (ChartSpec, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
            ],
        )
        .unwrap();
        let legend = color_disabled.then(|| {
            Box::new(crate::render::chart_config::LegendStyleSpec {
                disabled: Some(true),
                ..Default::default()
            })
        });
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, legend, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        (spec, batch)
    }

    /// The core seam contract: a leaf rendered with `suppress_color_legend`
    /// set on its `LeafScaleContext` reserves no gutter and draws no legend
    /// node for that channel, yet `prepare_render_inputs`' legend bundle
    /// (entries/title here — colorbar/aux/overrides follow the same code
    /// path) stays fully populated and reachable off `PipelineOutput::prep`,
    /// exactly as a later task needs to capture it for a figure-level legend.
    #[test]
    fn prepare_and_layout_color_suppression_keeps_bundle_but_skips_layout() {
        let (spec, batch) = species_scatter(false);
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let chart_config = ChartConfig::default();

        let ctx = scale_resolve::LeafScaleContext {
            suppress_color_legend: true,
            ..Default::default()
        };
        let suppressed = prepare_and_layout(&spec, &batch, &theme, viewport, &chart_config, Some(&ctx))
            .expect("prepare_and_layout must succeed with a suppressed color legend");

        // Bundle: populated regardless of suppression.
        assert!(
            !suppressed.prep.legend_entries.is_empty(),
            "legend bundle must stay populated under layout-stage suppression"
        );
        assert_eq!(suppressed.prep.legend_title.as_deref(), Some("species"));

        // Layout: no gutter reserved, nothing drawn.
        assert!(suppressed.layout.legend.is_none(), "suppressed color legend must not be laid out/drawn");

        // Contrast against the same leaf with no suppression context (today's
        // behavior, and what every non-composite render already does).
        let baseline = prepare_and_layout(&spec, &batch, &theme, viewport, &chart_config, None)
            .expect("prepare_and_layout must succeed without suppression");
        assert!(baseline.layout.legend.is_some(), "unsuppressed baseline legend must be laid out");
        assert!(
            suppressed.layout.panels[0].plot_area.w > baseline.layout.panels[0].plot_area.w,
            "suppressing the legend must reclaim the gutter the baseline reserved for it"
        );
    }

    /// Distinguishes the two suppression kinds (design §6): a user
    /// `legend={"disabled": true}` on the color encoding suppresses at
    /// PREPARE time — the bundle itself comes back empty — unlike compositor
    /// suppression above, which leaves the bundle populated and only gates
    /// the layout stage.
    #[test]
    fn prepare_and_layout_color_user_disabled_yields_empty_bundle() {
        let (spec, batch) = species_scatter(true);
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let chart_config = ChartConfig::default();

        let po = prepare_and_layout(&spec, &batch, &theme, viewport, &chart_config, None)
            .expect("prepare_and_layout must succeed with a user-disabled color legend");

        assert!(po.prep.legend_entries.is_empty(), "user-disabled legend must yield an empty bundle at prepare time");
        assert!(po.prep.colorbar.is_none());
        assert!(po.layout.legend.is_none(), "no legend to lay out when the bundle is empty");
    }

    /// R3 EXEMPTION: the chart-level `configure_axis(orient=...)` path applies
    /// AFTER `prepare_render_inputs` (the `apply_axis_config_to_axis_input`
    /// calls inside `prepare_and_layout`) and is deliberately NOT patched
    /// through `with_coord_flipped` — the mirror image of the `SortSpecIgnored`
    /// exemption. Chart-level `configure_axis` config keys (`axis_x`/`axis_y`)
    /// are themselves RESOLVED-slot vocabulary Python never remaps under
    /// `CoordFlip` (unlike a per-channel `Axis(...)`, which travels with its
    /// whole `EncodingSpec` through the swap): a user who types
    /// `configure_axis(axis_x=Axis(orient="left"))` always means the physical
    /// x axis, flip or not, so the RESOLVED channel already IS what they
    /// wrote. This test asserts the exemption holds: the message names the
    /// resolved `x` axis (matching the config key the user actually typed),
    /// not an un-flipped `y`, under a flipped chart.
    #[test]
    fn chart_level_orient_error_names_resolved_axis_under_flip() {
        use crate::spec::coord::CoordKind;
        let (mut spec, batch) = scatter_3();
        spec.coord = Some(CoordKind::Flip);
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let chart_config = ChartConfig {
            axis_x: Some(AxisConfigSpec {
                style: chart_config::AxisStyleSpec { orient: Some("left".into()), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        // PipelineOutput isn't Debug, so `expect_err`/`unwrap_err` (which both
        // require `T: Debug`) don't work here — match it out by hand.
        let err = match prepare_and_layout(&spec, &batch, &theme, viewport, &chart_config, None) {
            Ok(_) => panic!("orient='left' on the resolved x axis must fail loud"),
            Err(e) => e,
        };
        match &err {
            RenderError::InvalidAxisOrient { channel, orient, coord_flipped } => {
                assert_eq!(*channel, "x", "resolved/internal token — this chain's own constructor default");
                assert_eq!(orient, "left");
                // NOT patched: this chain is exempt, so the placeholder stays.
                assert!(!coord_flipped);
            }
            other => panic!("expected InvalidAxisOrient, got {other:?}"),
        }
        assert_eq!(
            format!("{err}"),
            "axis orient 'left' is invalid for the x axis (expected 'top' or 'bottom')",
            "message must name the RESOLVED axis ('x') — that's the config key (axis_x=) the \
             user actually typed, not an internal token needing translation"
        );
    }

    /// R3 byte-identity: the same chart-level cross-dimension `orient` error,
    /// unflipped, renders exactly the pre-fix message text.
    #[test]
    fn chart_level_orient_fails_loud_unchanged_when_not_flipped() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let chart_config = ChartConfig {
            axis_x: Some(AxisConfigSpec {
                style: chart_config::AxisStyleSpec { orient: Some("left".into()), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = match prepare_and_layout(&spec, &batch, &theme, viewport, &chart_config, None) {
            Ok(_) => panic!("orient='left' on x must fail loud"),
            Err(e) => e,
        };
        assert_eq!(
            format!("{err}"),
            "axis orient 'left' is invalid for the x axis (expected 'top' or 'bottom')"
        );
    }
}

#[cfg(test)]
mod png_tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn render_png_produces_png_magic_bytes() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        let result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 100.0, height: 80.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        assert_eq!(&result.bytes[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    #[test]
    fn render_png_determinism_two_calls_byte_identical() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 100.0, height: 80.0 };
        let config = config::RenderConfig::default();
        let a = render_png(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let b = render_png(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }
}

#[cfg(test)]
mod golden_tests {
    //! End-to-end goldens. Refresh via `FERRUM_UPDATE_GOLDENS=1 cargo test`.
    //! See spec §9.4 for refresh discipline.

    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn check_golden(name: &str, svg: &str) {
        let path = format!("tests/golden/{name}.svg");
        let abs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&path);
        if std::env::var("FERRUM_UPDATE_GOLDENS").is_ok() {
            std::fs::create_dir_all(abs_path.parent().unwrap()).unwrap();
            std::fs::write(&abs_path, svg).expect("write golden");
            return;
        }
        let expected = std::fs::read_to_string(&abs_path)
            .unwrap_or_else(|e| panic!("read golden {path}: {e} — run FERRUM_UPDATE_GOLDENS=1 to create"));
        assert_eq!(svg, expected, "golden mismatch for {name} — run FERRUM_UPDATE_GOLDENS=1 to refresh");
    }

    fn check_png_hash(name: &str, png: &[u8]) {
        use sha2::Digest;
        use std::io::Write;
        let path = format!("tests/golden/{name}.sha256");
        let abs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&path);
        let mut hasher = sha2::Sha256::new();
        hasher.update(png);
        let hash = format!("{:x}", hasher.finalize());
        if std::env::var("FERRUM_UPDATE_GOLDENS").is_ok() {
            std::fs::create_dir_all(abs_path.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&abs_path).unwrap();
            f.write_all(hash.as_bytes()).unwrap();
            return;
        }
        let expected = std::fs::read_to_string(&abs_path)
            .unwrap_or_else(|e| panic!("read png hash {path}: {e}"));
        assert_eq!(hash.trim(), expected.trim(), "PNG hash mismatch for {name}");
    }

    #[test]
    fn scatter_minimal_golden() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("scatter_minimal", &result.bytes);

        let png_result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_png_hash("scatter_minimal.png", &png_result.bytes);
    }

    #[test]
    fn scatter_color_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            Arc::new(StringArray::from(vec!["a","b","c","a","b","c"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("scatter_color", &result.bytes);
    }

    #[test]
    fn bar_grouped_golden() {
        use crate::spec::encoding::DataType as SDT;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","b","c","d"])),
            Arc::new(Float64Array::from(vec![3.0, 1.0, 4.0, 1.5])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("bar_grouped", &result.bytes);
    }

    #[test]
    fn line_simple_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0, 30.0, 80.0, 60.0])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("line_simple", &result.bytes);
    }

    #[test]
    fn area_filled_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0, 30.0, 80.0, 60.0])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("area_filled", &result.bytes);
    }

    #[test]
    fn faceted_scatter_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0])),
            Arc::new(StringArray::from(vec!["setosa","setosa","setosa","versicolor","versicolor","versicolor","virginica","virginica","virginica"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("faceted_scatter", &result.bytes);
    }
}

#[cfg(test)]
mod chart_config_application_tests {
    //! Unit tests for `apply_chart_config` — verify that ChartConfig overrides
    //! are correctly applied to ThemeInputs.

    use super::*;
    use chart_config::{
        AxisConfigSpec, AxisStyleSpec, ChartConfig, ColorConfigSpec, GridConfigSpec,
        LegendConfigSpec, LegendStyleSpec, PaddingConfigSpec, TitleConfigSpec,
    };

    #[test]
    fn apply_chart_config_noop_on_empty_config() {
        let default_theme = ThemeInputs::default();
        let mut theme = default_theme.clone();
        apply_chart_config(&mut theme, &ChartConfig::default());
        // Empty config must not change anything.
        assert_eq!(theme.grid.grid, default_theme.grid.grid);
        assert_eq!(theme.sizes.grid_width, default_theme.sizes.grid_width);
        assert_eq!(theme.colors.grid_color, default_theme.colors.grid_color);
        assert_eq!(theme.padding.padding, default_theme.padding.padding);
        assert_eq!(theme.legend.legend_orient, default_theme.legend.legend_orient);
        assert_eq!(theme.palette.color_scheme, default_theme.palette.color_scheme);
        assert_eq!(theme.typography.label_font_size, default_theme.typography.label_font_size);
        assert_eq!(theme.typography.title_font_size, default_theme.typography.title_font_size);
    }

    #[test]
    fn apply_chart_config_grid_color_and_width() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                color: Some("#ff0000".to_string()),
                width: Some(2.0),
                opacity: Some(0.5),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.colors.grid_color, color::parse_color("#ff0000").unwrap());
        assert_eq!(theme.sizes.grid_width, 2.0);
        assert_eq!(theme.grid.grid_opacity, 0.5);
    }

    /// Batch A Task 8 sweep: `configure_grid(color=…)` was hex-only (it read
    /// through `from_hex_str`), so a CSS name or `rgb()` string silently kept
    /// the theme default. It now accepts the full vocabulary — and resolves to
    /// exactly the same color the equivalent hex does.
    #[test]
    fn apply_chart_config_grid_color_accepts_named_and_rgb_forms() {
        let grid_color = |spelling: &str| {
            let mut theme = ThemeInputs::default();
            let config = ChartConfig {
                grid: Some(GridConfigSpec {
                    color: Some(spelling.to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            apply_chart_config(&mut theme, &config);
            theme.colors.grid_color
        };
        let expected = color::parse_color("#4682b4").unwrap();
        assert_eq!(grid_color("steelblue"), expected, "CSS name must take effect");
        assert_eq!(grid_color("rgb(70, 130, 180)"), expected, "rgb() must take effect");
        assert_eq!(grid_color("#4682b4"), expected, "hex is unchanged");
        // An unparseable value keeps the theme default, as before the sweep.
        assert_eq!(grid_color("not-a-color"), ThemeInputs::default().colors.grid_color);
    }

    #[test]
    fn apply_chart_config_grid_disabled_via_grid_config() {
        let mut theme = ThemeInputs::default();
        assert!(theme.grid.grid); // default is on
        let config = ChartConfig {
            grid: Some(GridConfigSpec { x: Some(false), y: Some(false), ..Default::default() }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert!(!theme.grid.grid);
    }

    #[test]
    fn apply_chart_config_grid_enabled_via_axis_config() {
        let mut theme = ThemeInputs::default();
        theme.grid.grid = false;
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { grid: Some(true), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert!(theme.grid.grid);
    }

    #[test]
    fn apply_chart_config_padding_per_side() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            padding: Some(PaddingConfigSpec {
                top: Some(10.0),
                right: Some(20.0),
                bottom: Some(30.0),
                left: Some(40.0),
                auto: None,
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        // Each side is set independently.
        assert_eq!(theme.padding.padding_top, Some(10.0));
        assert_eq!(theme.padding.padding_right, Some(20.0));
        assert_eq!(theme.padding.padding_bottom, Some(30.0));
        assert_eq!(theme.padding.padding_left, Some(40.0));
        // Uniform fallback padding is unchanged.
        assert_eq!(theme.padding.padding, 16.0);
    }

    #[test]
    fn apply_chart_config_padding_auto_does_not_block_explicit_sides() {
        // auto=true (the Python default) must NOT block explicit side values.
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            padding: Some(PaddingConfigSpec {
                top: Some(5.0),
                auto: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        // The explicit top value must be applied even when auto=true.
        assert_eq!(theme.padding.padding_top, Some(5.0));
        // Sides not specified remain None.
        assert!(theme.padding.padding_right.is_none());
        assert!(theme.padding.padding_bottom.is_none());
        assert!(theme.padding.padding_left.is_none());
    }

    #[test]
    fn apply_chart_config_legend_orient_and_direction() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            legend: Some(LegendConfigSpec {
                style: LegendStyleSpec {
                    orient: Some("bottom".to_string()),
                    direction: Some("horizontal".to_string()),
                    columns: Some(4),
                    title_font_size: Some(16.0),
                    label_font_size: Some(9.0),
                    ..Default::default()
                },
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.legend.legend_orient, LegendOrient::Bottom);
        assert_eq!(theme.legend.legend_direction, Some(LegendDirection::Horizontal));
        assert_eq!(theme.legend.legend_columns, Some(4));
        assert_eq!(theme.typography.legend_title_font_size, 16.0);
        assert_eq!(theme.typography.label_font_size, 9.0);
    }

    #[test]
    fn apply_chart_config_color_scheme_override() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            color: Some(ColorConfigSpec {
                scheme: Some("tableau10".to_string()),
                sequential_scheme: Some("viridis".to_string()),
                diverging_scheme: Some("rdbu".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.palette.color_scheme, "tableau10");
        assert_eq!(theme.palette.sequential_scheme, "viridis");
        assert_eq!(theme.palette.diverging_scheme, "rdbu");
    }

    #[test]
    fn apply_chart_config_axis_label_font_size() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { label_font_size: Some(14.0), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.typography.label_font_size, 14.0);
    }

    #[test]
    fn apply_chart_config_axis_tick_size_and_domain_visibility() {
        let mut theme = ThemeInputs::default();
        assert!(theme.axis.axis_line); // default on
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec {
                    tick_size: Some(6.0),
                    domain: Some(false),
                    domain_width: Some(2.0),
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.sizes.tick_size, 6.0);
        assert!(!theme.axis.axis_line);
        assert_eq!(theme.sizes.axis_line_width, 2.0);
    }

    #[test]
    fn apply_chart_config_title_overrides() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            title: Some(TitleConfigSpec {
                font_size: Some(22.0),
                font_weight: Some("700".to_string()),
                anchor: Some("end".to_string()),
                color: Some("#123456".to_string()),
                offset: Some(8.0),
                subtitle_font_size: Some(13.0),
                subtitle_color: Some("#ff0000".to_string()),
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.typography.title_font_size, 22.0);
        assert_eq!(theme.typography.title_font_weight, "700");
        assert_eq!(theme.typography.title_anchor, TextAnchor::End);
        assert_eq!(theme.colors.title_color, color::parse_color("#123456").unwrap());
        assert_eq!(theme.typography.title_offset, 8.0);
        assert_eq!(theme.typography.subtitle_font_size, Some(13.0));
        assert_eq!(
            theme.colors.subtitle_color,
            Some(color::parse_color("#ff0000").unwrap())
        );
    }

    #[test]
    fn apply_chart_config_title_subtitle_defaults_unset() {
        // No `title` config → subtitle theme fields stay `None`, preserving the
        // pre-config default subtitle styling (font_color + title*0.85).
        let mut theme = ThemeInputs::default();
        apply_chart_config(&mut theme, &ChartConfig::default());
        assert_eq!(theme.typography.subtitle_font_size, None);
        assert_eq!(theme.colors.subtitle_color, None);
    }

    #[test]
    fn apply_chart_config_invalid_hex_color_silently_ignored() {
        let mut theme = ThemeInputs::default();
        let original_grid_color = theme.colors.grid_color;
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                color: Some("not-a-hex-color".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        // Bad color must not change the existing value.
        assert_eq!(theme.colors.grid_color, original_grid_color);
    }

    #[test]
    fn apply_chart_config_axis_x_wins_over_axis_for_same_field() {
        // When both `axis` and `axis_x` set label_font_size, axis_x wins
        // (applied last). This is the documented behavior for per-axis overrides.
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { label_font_size: Some(10.0), ..Default::default() },
                ..Default::default()
            }),
            axis_x: Some(AxisConfigSpec {
                style: AxisStyleSpec { label_font_size: Some(14.0), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.typography.label_font_size, 14.0);
    }

    #[test]
    fn axis_x_styling_field_wins_over_axis_via_fill_none_ordering() {
        // Per-axis STYLING fields (grid_color/width/dash, label_color, domain_*,
        // title_*, label_padding) flow through `apply_axis_config_to_axis_input`,
        // which fills `AxisInput.overrides` only when still `None` (first writer
        // wins). `prepare_and_layout` therefore applies the MORE-SPECIFIC
        // `axis_x`/`axis_y` BEFORE the shared `axis`, so `axis_x > axis` holds.
        // Reproduce that exact call ordering here.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            Some("X".to_string()),
            vec!["0".to_string(), "1".to_string()],
            None,
        );
        let axis_shared = AxisConfigSpec {
            style: AxisStyleSpec { grid_color: Some("#00ff00".into()), ..Default::default() },
            ..Default::default()
        };
        let axis_x = AxisConfigSpec {
            style: AxisStyleSpec { grid_color: Some("#0000ff".into()), ..Default::default() },
            ..Default::default()
        };
        // Mirror prepare_and_layout: per-axis FIRST, shared `axis` SECOND.
        apply_axis_config_to_axis_input(&mut axis, Some(&axis_x)).unwrap();
        apply_axis_config_to_axis_input(&mut axis, Some(&axis_shared)).unwrap();
        // axis_x blue (0,0,255) must win over axis green.
        assert_eq!(
            axis.overrides.grid_color.map(|c| [c.red, c.green, c.blue]),
            Some([0, 0, 255]),
            "axis_x grid_color must win over the shared axis grid_color",
        );
    }

    #[test]
    fn apply_chart_config_grid_dash() {
        let mut theme = ThemeInputs::default();
        assert!(theme.grid.grid_dash.is_none());
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                dash: Some(vec![4.0, 4.0]),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.grid.grid_dash, Some(vec![4.0, 4.0]));
    }

    // ── New wired fields (configure-punchlist, 2026-05-24) ───────────────────

    #[test]
    fn axis_config_title_font_size_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            Some("X Axis".to_string()),
            vec!["0".to_string(), "50".to_string(), "100".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { title_font_size: Some(16.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.title_font_size, Some(16.0));
    }

    #[test]
    fn chart_config_orient_propagates_when_at_default() {
        // configure_axis(orient="top") on an x-axis with no per-channel orient
        // override fills `overrides.orient`; `resolve_orient` then moves the
        // concrete side to Top. The orphan positioning fields flow too.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            Some("X".to_string()),
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                orient: Some("top".to_string()),
                translate: Some(5.0),
                grid_opacity: Some(0.4),
                zindex: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        axis.resolve_orient();
        assert_eq!(axis.orient, crate::layout::AxisOrient::Top);
        assert_eq!(axis.overrides.translate, Some(5.0));
        assert_eq!(axis.overrides.grid_opacity, Some(0.4));
        assert_eq!(axis.overrides.zindex, Some(1));
    }

    #[test]
    fn chart_config_cross_dimension_orient_fails_loud() {
        // configure_axis(orient="left") on an x-axis is a cross-dimension error.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("left".to_string()), ..Default::default() },
            ..Default::default()
        };
        let err = apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap_err();
        assert!(matches!(err, RenderError::InvalidAxisOrient { channel: "x", .. }));
    }

    #[test]
    fn chart_config_orient_does_not_override_per_channel() {
        // An x-axis already moved to Top by the per-channel path (which sets the
        // `overrides.orient` INPUT, mirroring prepare.rs) must not be moved again
        // by a conflicting chart-level orient — the chart-level fill is gated on
        // `is_none()`, so per-channel wins.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Top,
            None,
            vec![],
            None,
        );
        axis.overrides.orient = Some(crate::layout::AxisOrient::Top); // per-channel already set Top
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("bottom".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        axis.resolve_orient();
        assert_eq!(axis.orient, crate::layout::AxisOrient::Top, "per-channel orient must win");
        assert_eq!(axis.overrides.orient, Some(crate::layout::AxisOrient::Top));
    }

    #[test]
    fn per_channel_orient_at_default_side_still_beats_chart_level() {
        // Regression (B5 follow-up, Issue 1): an EXPLICIT per-channel
        // `fm.Axis(orient="bottom")` lands on the x default side (Bottom). The old
        // value-heuristic treated "at default side" as "unset" and let a
        // chart-level `configure(axis_x=AxisConfig(orient="top"))` win — a
        // precedence inversion. With the `is_none()` sentinel the explicit
        // per-channel Bottom wins and the axis stays at the bottom.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        // Mirror prepare.rs: an explicit per-channel orient sets BOTH the concrete
        // side and the `overrides.orient` input (even when it equals the default).
        axis.overrides.orient = Some(crate::layout::AxisOrient::Bottom);
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("top".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        axis.resolve_orient();
        assert_eq!(
            axis.orient,
            crate::layout::AxisOrient::Bottom,
            "explicit per-channel orient='bottom' must beat chart-level orient='top'",
        );

        // y mirror: explicit per-channel Left must beat chart-level Right.
        let mut yaxis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Left,
            None,
            vec![],
            None,
        );
        yaxis.overrides.orient = Some(crate::layout::AxisOrient::Left);
        let ycfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("right".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut yaxis, Some(&ycfg)).unwrap();
        yaxis.resolve_orient();
        assert_eq!(
            yaxis.orient,
            crate::layout::AxisOrient::Left,
            "explicit per-channel orient='left' must beat chart-level orient='right'",
        );
    }

    #[test]
    fn chart_config_offset_flush_overlap_propagate() {
        // configure_axis(offset=, label_flush=, label_overlap=) flow to AxisInput.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                offset: Some(30.0),
                label_flush: Some(true),
                label_overlap: Some("parity".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.offset, Some(30.0));
        assert_eq!(axis.overrides.label_flush, Some(true));
        assert_eq!(axis.overrides.label_overlap, Some(crate::layout::LabelOverlap::Parity));
    }

    #[test]
    fn chart_config_offset_flush_overlap_do_not_override_per_channel() {
        // Per-channel values already on the AxisInput must win over chart-level.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string()],
            None,
        );
        axis.overrides.offset = Some(5.0);
        axis.overrides.label_flush = Some(false);
        axis.overrides.label_overlap = Some(crate::layout::LabelOverlap::ShowAll);
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                offset: Some(30.0),
                label_flush: Some(true),
                label_overlap: Some("rotate".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.offset, Some(5.0), "per-channel offset must win");
        assert_eq!(axis.overrides.label_flush, Some(false), "per-channel label_flush must win");
        assert_eq!(
            axis.overrides.label_overlap,
            Some(crate::layout::LabelOverlap::ShowAll),
            "per-channel label_overlap must win",
        );
    }

    #[test]
    fn axis_config_title_color_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Left,
            Some("Y".to_string()),
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { title_color: Some("#ff0000".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        let c = axis.overrides.title_color.expect("title_color should be Some");
        assert_eq!(c.red, 0xff);
        assert_eq!(c.green, 0x00);
        assert_eq!(c.blue, 0x00);
    }

    #[test]
    fn axis_config_title_padding_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { title_padding: Some(12.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.title_padding, Some(12.0));
    }

    #[test]
    fn axis_config_label_format_raw_reformats_numeric_labels() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["1000".to_string(), "2000".to_string(), "3000".to_string()],
            None,
        );
        let cfg = AxisConfigSpec { label_format_raw: Some(",.0f".to_string()), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        apply_label_format_to_axis(&mut axis);
        // ",.0f" formats with thousands separator and 0 decimal places.
        assert_eq!(axis.tick_labels, vec!["1,000", "2,000", "3,000"]);
    }

    #[test]
    fn axis_config_label_format_raw_with_tick_values_replaces_labels() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0.0".to_string(), "0.5".to_string(), "1.0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            label_format_raw: Some(".1%".to_string()),
            style: AxisStyleSpec { values: Some(vec![0.0, 0.5, 1.0]), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        apply_label_format_to_axis(&mut axis);
        assert_eq!(axis.tick_labels, vec!["0.0%", "50.0%", "100.0%"]);
    }

    #[test]
    fn axis_config_label_padding_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string(), "50".to_string(), "100".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { label_padding: Some(6.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.label_padding, Some(6.0));
    }

    #[test]
    fn axis_config_per_channel_wins_over_configure() {
        // Per-channel label_angle (level 2) wins over configure (level 3).
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            Some(-45.0), // per-channel override already set (seeds overrides.label_angle)
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { label_angle: Some(-90.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        // -45.0 should win because it was already set (Some).
        assert_eq!(axis.overrides.label_angle, Some(-45.0));
    }

    #[test]
    fn legend_config_gradient_length_fills_legend_overrides() {
        let prep_overrides = legend_overrides_from_prep_default();
        // When prep has no gradient_length, configure() should fill it.
        let mut overrides = prep_overrides;
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec { gradient_length: Some(300.0), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.gradient_length, Some(300.0));
    }

    #[test]
    fn legend_config_symbol_type_fills_legend_overrides() {
        let mut overrides = legend_overrides_from_prep_default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec { symbol_type: Some("square".to_string()), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.symbol_type.as_deref(), Some("square"));
    }

    #[test]
    fn legend_config_per_encoding_wins_over_configure() {
        // Per-encoding gradient_length (already set) must not be overwritten by configure.
        let mut overrides = LegendOverrides {
            gradient_length: Some(150.0), // already set at level 2
            ..Default::default()
        };
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                // configure (level 3) tries to override.
                style: LegendStyleSpec { gradient_length: Some(300.0), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.gradient_length, Some(150.0)); // level 2 wins
    }

    /// B5 unit 3: `configure_legend(symbol_stroke_width=...)` fills the override
    /// only when the per-channel value is absent.
    #[test]
    fn legend_config_symbol_stroke_width_fills_when_absent() {
        let mut overrides = legend_overrides_from_prep_default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec { symbol_stroke_width: Some(2.0), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.symbol_stroke_width, Some(2.0));
    }

    /// B5 unit 3: a per-channel orphan (here `symbol_stroke_width`) beats the
    /// chart-level `configure_legend` value.
    #[test]
    fn legend_orphan_per_channel_wins_over_configure() {
        let mut overrides = LegendOverrides {
            // 380: per-channel (level 2) style fields nest on `style`.
            style: crate::layout::LegendStyleOpts {
                symbol_stroke_width: Some(5.0),
                row_padding: Some(18.0),
                clip_height: Some(40.0),
                // B5 unit 6a orphans, per-channel.
                symbol_size: Some(300.0),
                label_color: Some("#ff0000".into()),
                offset: Some(50.0),
                padding: Some(30.0),
                title_padding: Some(25.0),
                ..Default::default()
            },
            tick_min_step: Some(2.0),
            ..Default::default()
        };
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec {
                    symbol_stroke_width: Some(1.0),
                    row_padding: Some(4.0),
                    clip_height: Some(999.0),
                    tick_min_step: Some(99.0),
                    symbol_size: Some(1.0),
                    label_color: Some("#0000ff".into()),
                    offset: Some(1.0),
                    padding: Some(1.0),
                    title_padding: Some(1.0),
                    ..Default::default()
                },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.symbol_stroke_width, Some(5.0), "per-channel wins");
        assert_eq!(overrides.style.row_padding, Some(18.0), "per-channel wins");
        assert_eq!(overrides.style.clip_height, Some(40.0), "per-channel wins");
        assert_eq!(overrides.tick_min_step, Some(2.0), "per-channel wins");
        assert_eq!(overrides.style.symbol_size, Some(300.0), "per-channel wins");
        assert_eq!(overrides.style.label_color.as_deref(), Some("#ff0000"), "per-channel wins");
        assert_eq!(overrides.style.offset, Some(50.0), "per-channel wins");
        assert_eq!(overrides.style.padding, Some(30.0), "per-channel wins");
        assert_eq!(overrides.style.title_padding, Some(25.0), "per-channel wins");
    }

    /// When per-channel leaves the 6a fields `None`, `configure_legend` fills
    /// them (level 3) — the chart-level fallback.
    #[test]
    fn legend_6a_orphan_chart_level_fills_when_absent() {
        let mut overrides = LegendOverrides::default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec {
                    symbol_size: Some(300.0),
                    label_color: Some("#0000ff".into()),
                    offset: Some(50.0),
                    padding: Some(30.0),
                    title_padding: Some(25.0),
                    ..Default::default()
                },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.symbol_size, Some(300.0));
        assert_eq!(overrides.style.label_color.as_deref(), Some("#0000ff"));
        assert_eq!(overrides.style.offset, Some(50.0));
        assert_eq!(overrides.style.padding, Some(30.0));
        assert_eq!(overrides.style.title_padding, Some(25.0));
    }

    #[test]
    fn color_config_domain_override_updates_continuous_scale() {
        use scale_resolve::ColorScale;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let mut color_scale: Option<ColorScale> = Some(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
        });
        let cfg = ColorConfigSpec { domain: Some(vec![serde_json::json!(10.0), serde_json::json!(90.0)]), ..Default::default() };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        if let Some(ColorScale::Continuous { domain, .. }) = color_scale {
            assert_eq!(domain, (10.0, 90.0));
        } else {
            panic!("expected Continuous color scale");
        }
    }

    #[test]
    fn color_config_range_override_builds_gradient() {
        use scale_resolve::ColorScale;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let mut color_scale: Option<ColorScale> = Some(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
        });
        let cfg = ColorConfigSpec {
            range: Some(vec!["#ffffff".to_string(), "#000000".to_string()]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        if let Some(ColorScale::Continuous { scheme: ContinuousScheme::Gradient(stops), .. }) = color_scale {
            assert_eq!(stops.len(), 2);
            // First stop at t=0, last at t=1.
            assert!((stops[0].0 - 0.0).abs() < 1e-9);
            assert!((stops[1].0 - 1.0).abs() < 1e-9);
        } else {
            panic!("expected Gradient color scheme after range override");
        }
    }

    /// Helper: build a zero-filled LegendOverrides (as if no per-encoding overrides existed).
    fn legend_overrides_from_prep_default() -> LegendOverrides {
        LegendOverrides::default()
    }

    /// A categorical color scale over `domain`, painted from the tableau10
    /// palette (the shape `build_color_scale` produces).
    fn categorical_scale(domain: &[&str]) -> scale_resolve::ColorScale {
        scale_resolve::ColorScale::Categorical {
            domain: domain.iter().map(|s| s.to_string()).collect(),
            palette: std::borrow::Cow::Borrowed(color::palette::categorical_palette("tableau10")),
        }
    }

    /// `configure_color(domain=[…])` on a categorical scale reorders the
    /// category order (and therefore which palette color each category takes).
    #[test]
    fn color_config_domain_reorders_categorical_scale() {
        use scale_resolve::ColorScale;
        let mut color_scale = Some(categorical_scale(&["a", "b", "c"]));
        let before = color_scale.as_ref().unwrap().lookup("c").unwrap();

        let cfg = ColorConfigSpec {
            domain: Some(vec![serde_json::json!("c"), serde_json::json!("a")]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        // Dropping "b" is a reportable degradation — see
        // `color_config_domain_omitting_a_data_category_warns` for the contract.
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorDomainOmitsCategories { categories: vec!["b".into()] }],
        );

        let scale = color_scale.expect("scale survives the override");
        match &scale {
            ColorScale::Categorical { domain, .. } => {
                assert_eq!(domain, &["c", "a"], "listed order wins; unlisted 'b' is dropped");
            }
            other => panic!("expected Categorical, got {other:?}"),
        }
        assert!(scale.lookup("b").is_none(), "'b' is no longer in the domain");
        assert_ne!(
            (scale.lookup("c").unwrap().red, scale.lookup("c").unwrap().green),
            (before.red, before.green),
            "'c' moved to palette slot 0, so its color must change"
        );
    }

    /// Spec §4.2 (amended 2026-08-28): a `configure_color(domain=…)` that omits
    /// a data category keeps that category's marks rendering — in the theme mark
    /// color, with no legend entry — and *names the omission* so the gap is
    /// diagnosable rather than looking like a rendering bug. A domain that
    /// covers every data category is silent.
    #[test]
    fn color_config_domain_omitting_a_data_category_warns() {
        use crate::layout::{LegendEntry, SymbolKind};

        let entry = |label: &str| LegendEntry { label: label.into(), symbol: SymbolKind::Circle };
        let theme_fallback = color::from_rgba(1, 2, 3, 255);
        let data_categories = ["a", "b", "c"];

        // ── Partial domain: two of three categories listed ───────────────────
        let mut color_scale = Some(categorical_scale(&data_categories));
        let mut legend_entries: Vec<LegendEntry> =
            data_categories.iter().map(|c| entry(c)).collect();
        let warnings = apply_color_config_to_color_scale(
            &mut color_scale,
            &ColorConfigSpec {
                domain: Some(vec![serde_json::json!("a"), serde_json::json!("c")]),
                ..Default::default()
            },
        );

        // (1) The warning names the omitted category — and only it.
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorDomainOmitsCategories { categories: vec!["b".into()] }],
        );
        assert_eq!(
            warnings[0].to_string(),
            "color domain does not list b; those marks paint in the default mark color \
             with no legend entry",
        );

        // (2) "b"'s marks still render: the fill resolver falls back to the
        //     theme mark color rather than dropping the row or panicking.
        let scale = color_scale.as_ref().expect("scale survives");
        assert_eq!(
            draw::resolve_fill_color(Some(scale), Some("b"), None, theme_fallback, false).0,
            theme_fallback,
            "an omitted category must still paint, in the fallback color"
        );
        // The listed categories keep a real scale color, distinct from fallback.
        assert_ne!(
            draw::resolve_fill_color(Some(scale), Some("a"), None, theme_fallback, false).0,
            theme_fallback,
        );

        // (3) The legend carries exactly the two listed categories.
        resync_categorical_legend_entries(&mut legend_entries, color_scale.as_ref());
        assert_eq!(legend_entries, vec![entry("a"), entry("c")]);

        // ── Full domain: nothing omitted, nothing reported ───────────────────
        let mut full = Some(categorical_scale(&data_categories));
        let warnings = apply_color_config_to_color_scale(
            &mut full,
            &ColorConfigSpec {
                domain: Some(
                    data_categories.iter().map(|c| serde_json::json!(c)).collect(),
                ),
                ..Default::default()
            },
        );
        assert!(warnings.is_empty(), "a domain covering every category is silent: {warnings:?}");
    }

    /// A category listed in `configure_color(domain=…)` but absent from the data
    /// is kept, matching positional explicit-domain behavior.
    #[test]
    fn color_config_domain_keeps_categories_absent_from_data() {
        use scale_resolve::ColorScale;
        let mut color_scale = Some(categorical_scale(&["a"]));
        let cfg = ColorConfigSpec {
            domain: Some(vec![serde_json::json!("a"), serde_json::json!("ghost")]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        match color_scale.as_ref().unwrap() {
            ColorScale::Categorical { domain, .. } => assert_eq!(domain, &["a", "ghost"]),
            other => panic!("expected Categorical, got {other:?}"),
        }
    }

    /// A numeric `configure_color(domain=…)` (the continuous form) leaves a
    /// categorical domain untouched rather than replacing it with junk.
    #[test]
    fn color_config_numeric_domain_leaves_categorical_domain_alone() {
        use scale_resolve::ColorScale;
        let mut color_scale = Some(categorical_scale(&["a", "b"]));
        let cfg = ColorConfigSpec {
            domain: Some(vec![serde_json::json!(0.0), serde_json::json!(1.0)]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        match color_scale.as_ref().unwrap() {
            ColorScale::Categorical { domain, .. } => assert_eq!(domain, &["a", "b"]),
            other => panic!("expected Categorical, got {other:?}"),
        }
    }

    /// The legend entries follow the reordered domain, so labels and swatch
    /// colors stay in agreement after a `configure_color(domain=…)` override.
    /// The resync is inert when the entries already match the domain.
    #[test]
    fn legend_entries_resync_to_the_overridden_categorical_domain() {
        use crate::layout::{LegendEntry, SymbolKind};
        let entry = |label: &str, symbol| LegendEntry { label: label.into(), symbol };
        let mut entries = vec![
            entry("a", SymbolKind::Square),
            entry("b", SymbolKind::Circle),
            entry("c", SymbolKind::Circle),
        ];
        let unchanged = entries.clone();

        let scale = categorical_scale(&["a", "b", "c"]);
        resync_categorical_legend_entries(&mut entries, Some(&scale));
        assert_eq!(entries, unchanged, "matching domain must leave entries untouched");

        let reordered = categorical_scale(&["c", "a"]);
        resync_categorical_legend_entries(&mut entries, Some(&reordered));
        assert_eq!(
            entries,
            vec![entry("c", SymbolKind::Circle), entry("a", SymbolKind::Square)],
            "entries follow the new domain and each keeps its own symbol"
        );

        // No color scale (conditional-color legends) and no entries: both inert.
        let mut conditional_entries = vec![entry("x", SymbolKind::Circle)];
        resync_categorical_legend_entries(&mut conditional_entries, None);
        assert_eq!(conditional_entries, vec![entry("x", SymbolKind::Circle)]);
        let mut empty: Vec<LegendEntry> = Vec::new();
        resync_categorical_legend_entries(&mut empty, Some(&reordered));
        assert!(empty.is_empty(), "a suppressed legend stays suppressed");
    }

    /// `configure_color(range=…)` repaints a discretizing scale's swatches when
    /// the range describes the same partition. A count mismatch leaves the
    /// swatches alone and is **reported**, naming both counts — spec §4.2
    /// (amended 2026-08-28) forbids a silent drop here.
    #[test]
    fn color_config_range_repaints_discretizing_swatches() {
        use scale_resolve::{ColorScale, DiscretizedColors};
        let black = color::from_rgba(0, 0, 0, 255);
        let buckets = DiscretizedColors::new(vec![0.0, 1.0, 2.0], vec![black, black]).unwrap();
        let mut color_scale = Some(ColorScale::Discretizing(buckets));

        // One color for a 2-bucket partition: swatches stand, and the refusal
        // is warned with both counts.
        let warnings = apply_color_config_to_color_scale(
            &mut color_scale,
            &ColorConfigSpec { range: Some(vec!["#ffffff".into()]), ..Default::default() },
        );
        assert_eq!(color_scale.as_ref().unwrap().lookup_f64(0.5).unwrap().red, 0);
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorRangeBucketCountMismatch { expected: 2, received: 1 }],
        );
        assert_eq!(
            warnings[0].to_string(),
            "color range names 1 color(s) but the binned color scale has 2 bucket(s); \
             the range was not applied",
        );

        let warnings = apply_color_config_to_color_scale(
            &mut color_scale,
            &ColorConfigSpec {
                range: Some(vec!["#ffffff".into(), "#ffffff".into()]),
                ..Default::default()
            },
        );
        assert_eq!(color_scale.as_ref().unwrap().lookup_f64(0.5).unwrap().red, 255);
        assert!(warnings.is_empty(), "a matching range must not warn: {warnings:?}");
    }

    /// Batch A Task 8 sweep: all three `configure_color(range=…)` arms —
    /// continuous gradient, categorical palette, discretizing swatches — were
    /// hex-only. A CSS name or `rgb()` string was dropped by the first two
    /// (shortening the gradient / palette, or making the override a silent
    /// no-op) and reported as a parse failure by the third. All three now
    /// resolve the full vocabulary, and every spelling of one color produces
    /// the identical resolved color.
    ///
    /// One test rather than three: the three arms share a single sweep and a
    /// single parser, so pinning them together is what keeps a future
    /// re-narrowing of *any* of them visible.
    #[test]
    fn color_config_range_accepts_named_and_rgb_forms_on_every_arm() {
        use scale_resolve::{ColorScale, DiscretizedColors};
        use crate::render::color::{ContinuousScheme, NamedContinuous};

        // steelblue / tomato, in the three spellings that must agree.
        let spellings: [[&str; 2]; 3] = [
            ["steelblue", "tomato"],
            ["rgb(70, 130, 180)", "rgb(255, 99, 71)"],
            ["#4682b4", "#ff6347"],
        ];
        let expected = [
            color::from_rgb(0x46, 0x82, 0xb4),
            color::from_rgb(0xff, 0x63, 0x47),
        ];
        let range = |pair: [&str; 2]| ColorConfigSpec {
            range: Some(pair.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };

        // ── Continuous: the range becomes an evenly spaced gradient. ────────
        for pair in spellings {
            let mut scale = Some(ColorScale::Continuous {
                domain: (0.0, 1.0),
                scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
                midpoint: None,
            });
            let warnings = apply_color_config_to_color_scale(&mut scale, &range(pair));
            assert!(warnings.is_empty(), "{pair:?} must not warn: {warnings:?}");
            match scale {
                Some(ColorScale::Continuous {
                    scheme: ContinuousScheme::Gradient(stops), ..
                }) => {
                    assert_eq!(
                        stops.iter().map(|&(_, c)| c).collect::<Vec<_>>(),
                        expected,
                        "{pair:?} must build the same gradient stops"
                    );
                }
                other => panic!("{pair:?}: expected a Gradient scheme, got {other:?}"),
            }
        }

        // ── Categorical: the range replaces the palette. ────────────────────
        for pair in spellings {
            let mut scale = Some(categorical_scale(&["a", "b"]));
            let warnings = apply_color_config_to_color_scale(&mut scale, &range(pair));
            assert!(warnings.is_empty(), "{pair:?} must not warn: {warnings:?}");
            match scale {
                Some(ColorScale::Categorical { palette, .. }) => {
                    assert_eq!(palette.as_ref(), &expected, "{pair:?} must set the same palette");
                }
                other => panic!("{pair:?}: expected Categorical, got {other:?}"),
            }
        }

        // ── Discretizing: the range repaints the bucket swatches. ───────────
        let black = color::from_rgba(0, 0, 0, 255);
        for pair in spellings {
            let mut scale = Some(ColorScale::Discretizing(
                DiscretizedColors::new(vec![0.0, 1.0, 2.0], vec![black; 2]).unwrap(),
            ));
            let warnings = apply_color_config_to_color_scale(&mut scale, &range(pair));
            assert!(warnings.is_empty(), "{pair:?} must not warn: {warnings:?}");
            let scale = scale.unwrap();
            assert_eq!(
                [scale.lookup_f64(0.5).unwrap(), scale.lookup_f64(1.5).unwrap()],
                expected,
                "{pair:?} must repaint both buckets alike"
            );
        }
    }

    /// The discretizing `range` parse is all-or-nothing. Filtering unparseable
    /// entries out *before* the count check would let a too-long range "fit"
    /// after the bad entry is dropped — silently repainting the buckets with a
    /// shifted mapping — and would misreport a too-short one as a count
    /// mismatch. Both shapes must report the parse failure and leave the
    /// swatches untouched.
    #[test]
    fn color_config_range_on_discretizing_scale_parses_all_or_nothing() {
        use scale_resolve::{ColorScale, DiscretizedColors};
        let black = color::from_rgba(0, 0, 0, 255);
        let three_buckets = || {
            Some(ColorScale::Discretizing(
                DiscretizedColors::new(vec![0.0, 1.0, 2.0, 3.0], vec![black; 3]).unwrap(),
            ))
        };
        let parse_failure = |entry: &str| {
            vec![RenderWarning::ColorRangeParseFailure { entry: entry.to_string() }]
        };

        // 4 entries, one unparseable: the survivors would number exactly 3 and
        // would have been accepted, repainting bucket 2 with the 4th color.
        let mut scale = three_buckets();
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec![
                    "#ff0000".into(),
                    "bogus".into(),
                    "#0000ff".into(),
                    "#00ff00".into(),
                ]),
                ..Default::default()
            },
        );
        assert_eq!(warnings, parse_failure("bogus"), "must report the parse, not repaint");
        for probe in [0.5, 1.5, 2.5] {
            assert_eq!(
                scale.as_ref().unwrap().lookup_f64(probe).unwrap().red,
                0,
                "bucket at {probe} must keep its resolved swatch"
            );
        }

        // 3 entries, one unparseable: a post-filter count check would call this
        // a 2-vs-3 mismatch, misattributing a parse failure.
        let mut scale = three_buckets();
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec!["#ff0000".into(), "nope".into(), "#0000ff".into()]),
                ..Default::default()
            },
        );
        assert_eq!(warnings, parse_failure("nope"));
        assert_eq!(scale.as_ref().unwrap().lookup_f64(0.5).unwrap().red, 0);
    }

    /// The categorical `range` parse is all-or-nothing too, for a reason the
    /// Discretizing arm's rationale does not cover: a categorical palette is
    /// indexed by DOMAIN POSITION (`lookup` → `palette[i % palette.len()]`), so
    /// dropping an unparseable entry does not shorten a cycling palette — it
    /// re-points every category after the dropped one.
    ///
    /// RED (verified in place): with the former
    /// `filter_map(|s| parse_color(s).ok())`,
    /// `range=["red", "notacolor", "blue"]` over domain `[a, b, c]` parsed to
    /// `[red, blue]` and rendered `a=red`, `b=blue`, `c=red` — silently wrong,
    /// with no warning, under a doc comment asserting this could not happen.
    /// The assertion is per-category (which category gets which color), not a
    /// palette-length check, so a fix that merely shortened the palette would
    /// still fail it.
    #[test]
    fn color_config_range_on_categorical_scale_parses_all_or_nothing() {
        let red = color::parse_color("red").unwrap();
        let blue = color::parse_color("blue").unwrap();

        // Control: an all-valid range applies, in listed order.
        let mut scale = Some(categorical_scale(&["a", "b", "c"]));
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec!["red".into(), "green".into(), "blue".into()]),
                ..Default::default()
            },
        );
        assert!(warnings.is_empty(), "a valid range must not warn: {warnings:?}");
        let applied = scale.as_ref().unwrap();
        assert_eq!(applied.lookup("a"), Some(red), "a must take the 1st entry");
        assert_eq!(applied.lookup("c"), Some(blue), "c must take the 3rd entry");

        // One unparseable entry: the whole range is discarded and reported.
        let resolved = categorical_scale(&["a", "b", "c"]);
        let (before_a, before_b, before_c) =
            (resolved.lookup("a"), resolved.lookup("b"), resolved.lookup("c"));
        let mut scale = Some(resolved);
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec!["red".into(), "notacolor".into(), "blue".into()]),
                ..Default::default()
            },
        );
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorRangeParseFailure { entry: "notacolor".into() }],
            "one bad entry must be reported, not silently dropped"
        );
        let kept = scale.as_ref().unwrap();
        assert_eq!(
            [kept.lookup("a"), kept.lookup("b"), kept.lookup("c")],
            [before_a, before_b, before_c],
            "the resolved palette must be left untouched; the old filter_map gave \
             a=red, b=blue, c=red — a shifted mapping, not a shortened cycle"
        );
        assert_ne!(
            kept.lookup("b"),
            Some(blue),
            "b must NOT inherit the third entry's color — that is the shift this \
             test exists to catch"
        );
    }
}
