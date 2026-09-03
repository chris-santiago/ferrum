//! Phase 7 — static renderer. Pure functions: ChartSpec + RecordBatch + ThemeInputs +
//! Viewport -> deterministic SVG/PNG. See docs/superpowers/specs/2026-05-09-static-renderer-design.md.

pub(crate) mod annotation;
pub(crate) mod arrow_cast;
pub(crate) mod chart_config;
pub(crate) mod config_apply;
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
    ///   (`config_apply::apply_axis_config_to_axis_input`, called by
    ///   `config_apply::fill_axis_slots_specific_before_shared`) — EXEMPT.
    ///   `channel` there is derived from the axis's PHYSICAL dimension
    ///   (`AxisDimension::channel_token`), never from an `EncodingSpec`, and the config key
    ///   the user actually typed (`axis_x`/`axis_y`) is itself RESOLVED-slot
    ///   vocabulary that Python never remaps under flip — so the resolved
    ///   token already IS what the user wrote; un-flipping it would say the
    ///   opposite of the config key they typed (mirrors the `SortSpecIgnored`
    ///   exemption in `scale_resolve/domain.rs`).
    InvalidAxisOrient { channel: &'static str, orient: String, coord_flipped: bool },
    /// A raw d3-format/strftime spec string reachable from `ChartSpec`
    /// (`EncodingSpec.format`/`.axis.label_format`/`.legend.format`, on any
    /// layer) or `ChartConfig` (`axis`/`axis_x`/`axis_y`/`axis_y2`'s
    /// `label_format`/`label_format_raw`, `legend.format`) could not be
    /// tokenized by the d3-format grammar `format.rs` implements — NF-B1
    /// residual (2026-09-02): an unrecognized preset name legitimately
    /// passes through Python's `resolve_format_or_raw` as a raw spec per
    /// spec §4.5, but a genuine typo (e.g. `"curency"`) previously reached
    /// the lenient per-value tokenizer, which silently discarded trailing
    /// characters it couldn't place — corrupting rendered text with raw
    /// control characters for some inputs (the tokenizer's `c` type char:
    /// "format as Unicode code point"). Raised once per render by
    /// [`validate_chart_format_specs`], before any transform/layout work, so
    /// a malformed spec never reaches per-value formatting. `spec` is the
    /// offending string; `reason` is [`format::validate_d3_format_spec`]'s
    /// message (names the unrecognized trailing token and its position).
    InvalidFormatSpec { spec: String, reason: String },
    /// A chart-level scale-domain config (`configure_axis(domain_min=,
    /// domain_max=, ...)` and its per-axis spellings) resolved to a domain the
    /// scale family refuses (D3, spec §4.2). Today the one such shape is a
    /// DEGENERATE pair — `lo == hi` — which every scale constructor already
    /// rejects with [`crate::scale::core::DEGENERATE_DOMAIN_MESSAGE`]; the
    /// config surface quotes that same sentence so a user who reaches the
    /// contract from `LinearScale(domain=[10, 10])` and from
    /// `configure_axis(domain_min=10, domain_max=10)` reads identical words.
    ///
    /// Refused rather than warned because it is a user error, not a sanctioned
    /// degradation: a zero-width domain clips every mark away and produces a
    /// blank plot that is indistinguishable from a rendering bug. `min > max`
    /// is deliberately NOT refused — `LinearScale(domain=[50, 0])` is an
    /// accepted reversed axis and this surface matches it.
    ///
    /// The primary refusal is at the Python boundary (`AxisConfig
    /// .__post_init__`), matching where the scale constructors refuse; this is
    /// the render-side backstop for a raw-dict `chart_config` that never went
    /// through it, mirroring how `scale_resolve::color` quotes the same
    /// sentence for a raw-dict discretizing scale.
    InvalidScaleDomainConfig { channel: String, reason: String },
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
/// unpatched when the placeholder discipline was prose-only)
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
        | RenderError::UnsupportedChannelCombination { .. }
        | RenderError::InvalidFormatSpec { .. }
        | RenderError::InvalidScaleDomainConfig { .. }) => other,
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
            Self::InvalidScaleDomainConfig { channel, reason } => write!(
                f,
                "{channel} axis scale-domain config: {reason}"
            ),
            Self::InvalidFormatSpec { spec, reason } =>
                write!(f, "invalid format spec {spec:?}: {reason}"),
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
    /// `suppressed`: the warning itself fires
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
    /// amended 2026-09-01). The curve, domain,
    /// and range are NOT honored; the channel falls back to the default
    /// linear resolution (data extent onto the theme opacity band) instead.
    /// `channel` is the channel name (`"opacity"` / `"fill_opacity"` /
    /// `"stroke_opacity"`); `scale_kind` is the dropped spec's kind
    /// (`"log"`, `"pow"`, `"sqrt"`, …). Full non-linear opacity-curve support
    /// is a logged campaign follow-up, not this batch.
    UnsupportedOpacityScale { channel: String, scale_kind: String },
    /// A chart-level config section named a surface the chart does not have.
    /// The sole producer today is `axis_y2` (D2/F-L07-06, spec §4.1): a chart
    /// with no `independent_y` layer has no secondary y-axis input
    /// (`AxesInput.secondary_y` is empty) for the override to fill, so it is
    /// dropped with this warning rather than silently discarded. `section`
    /// names the config key the user set (currently always `"axis_y2"`, but
    /// the variant is not axis_y2-specific by name in case a future config
    /// surface reaches the same "named but absent" shape). `String`, not
    /// `&'static str`, so `#[derive(Deserialize)]` round-trips through owned
    /// input (a `&'static str` field forces `Deserialize<'de>` to require
    /// `'de: 'static`, breaking `serde_json::from_str` on a local `String`).
    ConfigSurfaceNotPresent { section: String },
    /// A per-channel `Legend(values=[...])` on a CATEGORICAL legend named
    /// entries the legend does not have (D6/F-L04-05, spec §4.4). `values`
    /// filters and orders the entries; a name matching none of them cannot be
    /// drawn as a swatch — there is no category, no color, and no scale slot
    /// behind it — so it is skipped and reported here rather than silently
    /// ignored or invented as an empty swatch. `values` names the unmatched
    /// entries in the order the caller wrote them; the matched ones still
    /// filter and order the legend normally.
    LegendValuesUnknown { values: Vec<String> },
    /// A chart-level scale-domain config field (`domain_min`/`domain_max`/
    /// `nice`/`zero`, from `configure_axis(...)` or from a per-axis
    /// `configure(axis_x=/axis_y=/axis_y2=...)` section — note the shared
    /// `axis` key reaches x and y only, never the secondary axis, so `axis_y2`
    /// is the sole way to address that one) was set on an axis whose resolved
    /// scale is ordinal/band
    /// (D3, spec §4.2). Categorical axes have no numeric bounds to clamp,
    /// round, or extend to zero, so the fields describe nothing that exists
    /// — a WRONG SURFACE, not a cascade loss, and therefore reported rather
    /// than silently dropped. (The cascade loss — an encoding-level `scale=`
    /// domain out-ranking these — IS silent by design: the documented
    /// precedence already answers it.) `channel` is `"x"`/`"y"`/`"y2"`;
    /// `fields` names the subset the caller actually set, in schema order.
    ScaleDomainConfigOnOrdinalAxis { channel: String, fields: Vec<String> },
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
            RenderWarning::ConfigSurfaceNotPresent { section } => write!(
                f,
                "chart config names '{section}' but the chart has no matching surface; \
                 the override was not applied"
            ),
            RenderWarning::LegendValuesUnknown { values } => write!(
                f,
                "legend values [{}] match no legend entry and were skipped",
                values.join(", ")
            ),
            RenderWarning::ScaleDomainConfigOnOrdinalAxis { channel, fields } => write!(
                f,
                "{} appl{} to continuous scales; the {channel} axis is ordinal, so {} not applied",
                fields.join(", "),
                if fields.len() == 1 { "ies" } else { "y" },
                if fields.len() == 1 { "it was" } else { "they were" },
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
            RenderWarning::ConfigSurfaceNotPresent { section: "axis_y2".to_string() },
            RenderWarning::LegendValuesUnknown { values: vec!["zzz".into()] },
            RenderWarning::ScaleDomainConfigOnOrdinalAxis {
                channel: "x".into(),
                fields: vec!["domain_min".into(), "nice".into()],
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

        // D2/F-L07-06 (this task): `axis_y2` names a surface the chart lacks.
        let no_surface = RenderWarning::ConfigSurfaceNotPresent { section: "axis_y2".to_string() };
        let no_surface_text = format!("{no_surface}");
        assert!(no_surface_text.contains("axis_y2"), "{no_surface_text}");
        assert!(no_surface_text.contains("no matching surface"), "{no_surface_text}");
        assert!(!no_surface_text.contains("ConfigSurfaceNotPresent"), "{no_surface_text}");

        // D6/F-L04-05 (this task): unknown categorical `Legend(values=)` names.
        let unknown = RenderWarning::LegendValuesUnknown {
            values: vec!["zulu".into(), "yankee".into()],
        };
        let unknown_text = format!("{unknown}");
        assert!(unknown_text.contains("zulu, yankee"), "{unknown_text}");
        assert!(unknown_text.contains("match no legend entry"), "{unknown_text}");
        assert!(!unknown_text.contains("LegendValuesUnknown"), "{unknown_text}");
    }

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

// ---------------------------------------------------------------------------
// Task 20 — render_svg full pipeline orchestration (spec §6).
// ---------------------------------------------------------------------------

use crate::layout::{
    compute_layout, CompositeLayoutSeam, LegendOverrides, LegendSuppression, ThemeInputs, Viewport,
};
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;
use chart_config::ChartConfig;

/// Output of the shared prepare-and-layout pipeline, consumed by both
/// `render_svg` and `render_scene_json`.
struct PipelineOutput {
    prep: prepare::PreparedInputs,
    layout: crate::layout::LayoutResult,
    effective_theme: ThemeInputs,
    warnings: Vec<RenderWarning>,
    /// Passes 17–18 and 16 of the config pipeline
    /// (`config_apply::resolve_leaf_legend_overrides`), carried out rather than
    /// recomputed. `compute_layout` consumes them below; `composite_render`'s
    /// figure-legend seam (`capture_leaf_bundle`) needs the SAME two values for
    /// the same leaf and used to call the projection a second time on this very
    /// `prep`. Carrying them makes the two uses one value, so they cannot
    /// diverge (#143 remediation, design review rec 4).
    legend_overrides: LegendOverrides,
    legend_title: Option<String>,
}

/// Shared pipeline executed by both `render_svg` and `render_scene_json` (and,
/// per composite leaf, by `composite_render::render_leaf`).
///
/// Performs in order:
///   1. `config_apply::validate_chart_format_specs` — the raw format-spec gate.
///   2. `prepare::prepare_render_inputs` — transforms, scale resolution, axis inputs.
///   3. `config_apply::apply_chart_config_pipeline` — every chart-level
///      `configure_*()` override, in the order stated in `config_apply`'s
///      module doc and nowhere else.
///   4. `compute_layout`, against the composite seam.
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
    // NF-B1 residual malformed-spec guard (Task 4, D8) — a separate gate, not a
    // pipeline pass: it runs before any transform/layout work so a bad spec
    // refuses fast, once per render, not once per tick/label. See
    // `config_apply::validate_chart_format_specs`'s doc.
    config_apply::validate_chart_format_specs(spec, chart_config)?;
    let mut prep = prepare::prepare_render_inputs(spec, batch, theme, chart_config, leaf_scales)?;
    let mut warnings = prep.warnings.clone();

    // Every chart-level `configure_*()` override, applied in the one order
    // `config_apply`'s module doc states — legend suppression, axis-slot fill
    // (most specific first), tick re-derivation, color config + legend-entry
    // rebuild, effective theme, legend-overrides projection.
    let config_apply::AppliedChartConfig { effective_theme, legend_overrides, legend_title } =
        config_apply::apply_chart_config_pipeline(&mut prep, theme, chart_config, &mut warnings)?;

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
        legend_title.clone(),
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

    Ok(PipelineOutput { prep, layout, effective_theme, warnings, legend_overrides, legend_title })
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

    let PipelineOutput { prep, layout, effective_theme, mut warnings, .. } =
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

    let PipelineOutput { prep, layout, effective_theme, mut warnings, .. } =
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
    use chart_config::{AxisConfigSpec, AxisStyleSpec, LegendStyleSpec};
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

    // ── NF-B1 residual: malformed-spec refusal, per surface (Task 4, D8) ────
    // The audit repro: a typo'd preset name (`fm.Axis(label_format="curency")`
    // / `fm.X("x", format="curency")`) passes through Python's
    // `resolve_format_or_raw` as an honest raw spec (spec §4.5's
    // unknown-name-passes-raw contract), then previously reached the lenient
    // per-value d3 tokenizer, which parsed it as `type='c'` (the Unicode
    // code-point formatter) and emitted control characters into rendered SVG
    // text. These pin the FULL `render_svg` pipeline refusing before any
    // control characters are ever emitted, on each raw-accepting surface.

    #[test]
    fn render_svg_refuses_malformed_encoding_format() {
        // Surface: encoding `format=` (`fm.X("x", format=...)`).
        let (mut spec, batch) = scatter_3();
        spec.encoding.x.as_mut().unwrap().format = Some("curency".to_string());
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        let msg = err.to_string();
        assert!(msg.contains("curency"), "{msg}");
        assert!(!msg.contains('\u{c}'), "message must not embed control chars: {msg:?}");
    }

    #[test]
    fn render_svg_refuses_malformed_per_channel_axis_label_format() {
        // Surface: per-channel `fm.Axis(label_format=...)`.
        let (mut spec, batch) = scatter_3();
        spec.encoding.x.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            label_format: Some("curency".to_string()),
            ..Default::default()
        }));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        assert!(err.to_string().contains("curency"));
    }

    #[test]
    fn render_svg_refuses_malformed_chart_level_axis_label_format() {
        // Surface: chart-level `configure_axis(label_format_raw=...)` (the
        // `AxisConfig.label_format` preset-name surface always resolves to a
        // real spec at the Python boundary — a genuinely malformed STRING
        // only reaches Rust via the raw spelling, or a raw-dict caller).
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let chart_config = ChartConfig {
            axis: Some(AxisConfigSpec {
                label_format_raw: Some("curency".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err =
            render_svg(&spec, &batch, &theme, viewport, &config, &chart_config).unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        assert!(err.to_string().contains("curency"));
    }

    #[test]
    fn render_svg_refuses_chart_level_raw_strftime_with_diagnosing_message() {
        // A raw strftime-shaped spec on the
        // chart-level axis surface must be refused with a message naming
        // the REAL cause (this surface is numeric-only) rather than
        // restating the misleading "your d3 spec is malformed" complaint —
        // "%b %d" IS a valid d3-format grammar failure (trailing garbage)
        // but that framing hides the actual fix (a time preset name or the
        // per-channel fm.Axis surface).
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let chart_config = ChartConfig {
            axis: Some(AxisConfigSpec {
                label_format_raw: Some("%b %d".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err =
            render_svg(&spec, &batch, &theme, viewport, &config, &chart_config).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        assert!(msg.contains("valid date/time pattern"), "{msg}");
        assert!(msg.contains("only accepts numeric d3-format specs"), "{msg}");
        assert!(msg.contains("fm.Axis(label_format=...)"), "{msg}");
        assert!(
            !msg.contains("unrecognized token"),
            "must not restate the misleading d3-grammar complaint: {msg}"
        );
    }

    #[test]
    fn render_svg_percent_free_typo_on_chart_level_raw_keeps_d3_grammar_message() {
        // The negative control the reviewer required alongside the
        // re-diagnosis fix. `"curency"` — the batch's own headline
        // NF-B1 repro — has NO `%` at all, so `chrono`'s `StrftimeItems`
        // trivially parses it as pure literal text (`validate_strftime_spec`
        // returns `Ok`) even though it is NOT a date/time pattern by any
        // sensible reading. Requiring an actual `%` before treating a spec
        // as a strftime candidate (not just successful parsing) is what
        // keeps this on the d3-grammar message, which names the real defect
        // (an unrecognized trailing token) — the date/time re-diagnosis must
        // never fire for a %-free string.
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let chart_config = ChartConfig {
            axis: Some(AxisConfigSpec {
                label_format_raw: Some("curency".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err =
            render_svg(&spec, &batch, &theme, viewport, &config, &chart_config).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        assert!(msg.contains("unrecognized token"), "expected the d3-grammar message: {msg}");
        assert!(
            !msg.contains("valid date/time pattern"),
            "a %-free typo must never be misdiagnosed as a date/time pattern: {msg}"
        );
    }

    #[test]
    fn render_svg_refuses_malformed_legend_format() {
        // Surface: per-channel `fm.Legend(format=...)` (color legend).
        let (mut spec, batch) = scatter_3();
        spec.encoding.color =
            Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() });
        spec.encoding.color.as_mut().unwrap().legend = Some(Box::new(LegendStyleSpec {
            format: Some("curency".to_string()),
            ..Default::default()
        }));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        assert!(err.to_string().contains("curency"));
    }

    #[test]
    fn render_svg_accepts_valid_but_unusual_specs_on_every_surface() {
        // The refusal must never false-positive on genuinely valid d3 specs.
        // `y` here is NON-temporal (`scatter_3`'s plain `EncodingSpec`, no
        // `type_` set) — pin: "*>8.1%" auto-detects as
        // a TIME candidate by the `%`-containment heuristic (no explicit
        // format_type), but `y`'s declared type is not Temporal, so it must
        // validate as the (valid) d3 percent spec it actually is, matching
        // what `apply_axis_format_or_thread` really does at runtime on a
        // non-temporal scale (falls through to the numeric path regardless
        // of the heuristic's own guess).
        let (mut spec, batch) = scatter_3();
        spec.encoding.x.as_mut().unwrap().format = Some(",.2f".to_string());
        spec.encoding.y.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            label_format: Some("*>8.1%".to_string()),
            ..Default::default()
        }));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let chart_config = ChartConfig {
            axis_y: Some(AxisConfigSpec {
                label_format_raw: Some("~s".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        render_svg(&spec, &batch, &theme, viewport, &config, &chart_config)
            .expect("valid-but-unusual specs must not be refused");
    }

    // ── Batch B design review S4 (2026-09-03): `configure_axis(nice=True)`
    // on a log axis must delegate to `LogScale`'s own `nice()`, not the
    // inline linear `nice_step` rounding every kind used to share. ─────────

    /// RED-proof of the reviewer-reproduced crash: `fm.Chart(df).mark_point()
    /// .encode(x=fm.X("a"), y=fm.Y("v", scale=fm.LogScale()))
    /// .configure_axis(nice=True).to_svg()`. `y`'s data domain `(10, 1000)`
    /// is exactly the shape that trips the pre-fix bug: the OLD
    /// kind-independent `nice_step(10, 1000, 10)` rounds to a step of 100,
    /// and `floor(10 / 100) * 100 == 0` — driving the low bound to 0, which
    /// `LogScale::validate_user_domain` (rightly) refuses, so the whole
    /// chart died with `InvalidScaleDomainConfig` instead of rendering. The
    /// fixed dispatch (`ScaleKind::niced_domain` → `LogScale::nice_domain_pair`)
    /// rounds in LOG space instead (nearest power of 10: `[10, 1000]`,
    /// already exactly a power of the base), so this must render.
    #[test]
    fn render_svg_log_axis_configure_axis_nice_true_renders_instead_of_refusing() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec {
                    field: "y".into(),
                    type_: None,
                    scale: Some(crate::spec::encoding::ScaleSpec::Log {
                        base: 10.0,
                        common: crate::spec::encoding::ContinuousScaleCommon {
                            domain: None,
                            range: None,
                            clamp: false,
                            padding: None,
                            scheme: None,
                            domain_param: None,
                        },
                        nice: false,
                    }),
                    ..Default::default()
                }),
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
                Arc::new(Float64Array::from(vec![10.0, 100.0, 1000.0])),
            ],
        )
        .unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        // `.configure_axis(nice=True)` is the SHARED `axis` key (applies to
        // both x and y, matching the Python surface).
        let chart_config = ChartConfig {
            axis: Some(AxisConfigSpec { nice: Some(true), ..Default::default() }),
            ..Default::default()
        };
        render_svg(&spec, &batch, &theme, viewport, &config, &chart_config).expect(
            "a log y-axis under configure_axis(nice=True) must render, not refuse with \
             InvalidScaleDomainConfig",
        );
    }

    // ── Quality-review S4 (2026-09-03): malformed %-bearing specs on a
    // TEMPORAL channel must be a typed refusal, never a panic. Root cause:
    // `validate_chart_format_specs` exempted every `%`-bearing spec from
    // validation entirely (on the false premise that `chrono` handles a bad
    // pattern "leniently" — it panics instead, via `format::format_time_spec`'s
    // old `.to_string()` on a `DelayedFormat` whose `Display` can error). The
    // fixed `validate_chart_format_specs` validates the `chrono` strftime
    // grammar for a channel whose declared `type_` is `Temporal`, refusing
    // with `RenderError::InvalidFormatSpec` — a typed Rust `Result`, which
    // can NEVER become an unhandled Python panic (PyO3's `render_err_to_py`
    // always converts it to a `ValueError` first). These four reproduce the
    // exact snippets quality review verified crash on the pre-fix build.

    #[test]
    fn render_svg_refuses_curency_percent_typo_on_temporal_axis() {
        // fm.X('t:T', axis=fm.Axis(label_format='curency%')).
        let (mut spec, batch) = scatter_3();
        let x = spec.encoding.x.as_mut().unwrap();
        x.type_ = Some(crate::spec::encoding::DataType::Temporal);
        x.axis = Some(Box::new(AxisStyleSpec {
            label_format: Some("curency%".to_string()),
            ..Default::default()
        }));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
        assert!(err.to_string().contains("curency%"));
    }

    #[test]
    fn render_svg_refuses_unknown_strftime_specifier_on_temporal_axis() {
        // label_format='%J' -- not a real strftime specifier.
        let (mut spec, batch) = scatter_3();
        let x = spec.encoding.x.as_mut().unwrap();
        x.type_ = Some(crate::spec::encoding::DataType::Temporal);
        x.axis =
            Some(Box::new(AxisStyleSpec { label_format: Some("%J".to_string()), ..Default::default() }));
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
    }

    #[test]
    fn render_svg_refuses_ordinary_percent_spec_on_temporal_encoding_format() {
        // fm.X('t:T', format='.1%') -- a perfectly ordinary raw d3 percent
        // spec that is valid on a numeric scale but is not a valid strftime
        // pattern, and `x` here IS declared temporal.
        let (mut spec, batch) = scatter_3();
        let x = spec.encoding.x.as_mut().unwrap();
        x.type_ = Some(crate::spec::encoding::DataType::Temporal);
        x.format = Some(".1%".to_string());
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
    }

    #[test]
    fn render_svg_refuses_bare_percent_on_temporal_encoding_format() {
        // format='%' -- a dangling strftime escape with nothing after it.
        let (mut spec, batch) = scatter_3();
        let x = spec.encoding.x.as_mut().unwrap();
        x.type_ = Some(crate::spec::encoding::DataType::Temporal);
        x.format = Some("%".to_string());
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let err = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap_err();
        assert!(matches!(err, RenderError::InvalidFormatSpec { .. }), "got: {err:?}");
    }

    #[test]
    fn render_svg_accepts_the_same_ordinary_percent_spec_on_a_non_temporal_axis() {
        // The exact mirror-image control: '.1%' on a NON-temporal x (no
        // `type_` set, matching `scatter_3`'s default) must render fine --
        // the fix must not become the opposite over-eager refusal.
        let (mut spec, batch) = scatter_3();
        spec.encoding.x.as_mut().unwrap().format = Some(".1%".to_string());
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .expect("'.1%' on a non-temporal axis must render, not refuse");
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

    /// spec §4.3 (amended 2026-09-01): a
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

    /// End-to-end through `render_svg`: a
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
    /// wire shape. Shared by the four pins below.
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

    /// By provenance, never by value:
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

    /// An explicit `fill="#00000000"`
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

    /// The genuinely cleared-paint case — a
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

    /// Control: a non-black zero-alpha color
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

        let mut prep = prepare::prepare_render_inputs(&spec, &batch, &theme, &ChartConfig::default(), None).unwrap();
        let mut warnings = prep.warnings.clone();

        // Mirror `prepare_and_layout`'s chart-config application by CALLING it,
        // not by re-expressing it: this block used to hand-copy the effective-
        // theme / show-defaults / legend-title / legend-overrides passes, which
        // only kept passing because the test runs with `ChartConfig::default()`
        // (so both expressions coincide). A second, older statement of a
        // precedence rule stops tracking the pipeline the moment the rule moves
        // again — so the assembled path below shares the one statement of it.
        let config_apply::AppliedChartConfig { effective_theme, legend_overrides, legend_title } =
            config_apply::apply_chart_config_pipeline(
                &mut prep,
                &theme,
                &ChartConfig::default(),
                &mut warnings,
            )
            .unwrap();
        let theme_ref = &effective_theme;
        let metrics = font::FontdueMetrics::new();
        let vp2 = Viewport {
            width: cfg.width.unwrap_or(viewport.width),
            height: cfg.height.unwrap_or(viewport.height),
        };
        let layout = compute_layout(
            &spec, theme_ref, vp2,
            &prep.axes, &prep.facet_groups, &prep.legend_entries,
            legend_title, prep.colorbar.as_ref(), &metrics,
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

    // ── axis_y2 effect tests (D2/F-L07-06) ──────────
    //
    // The prior test coverage (chart_config.rs's deserialization tests,
    // binding.rs's wire-gate tests, this file's RenderWarning round-trip)
    // proved `axis_y2` PARSES and the warning variant EXISTS — none of them
    // called `prepare_and_layout` (or any render entry) with `axis_y2` set
    // against a chart that actually HAS a secondary y axis, so the fill
    // branch in `prepare_and_layout` (the code this task added) had zero
    // coverage. These three close that gap.

    /// A primary line layer plus one `independent_y` layer — the same
    /// fixture shape `prepare_render_inputs_independent_y_layer_produces_
    /// secondary_axis_input` (`prepare/mod.rs`) uses to prove
    /// `AxesInput.secondary_y` gets populated; reused here because the
    /// axis_y2 fill branch has nothing to fill without a real secondary axis
    /// input to fill it.
    fn secondary_y_chart() -> (ChartSpec, RecordBatch) {
        use crate::spec::layer::Layer;
        let primary = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y0".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        };
        let secondary = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y1".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: true,
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: Some(vec![primary, secondary]),
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y0", DataType::Float64, false),
            Field::new("y1", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    /// `prepare_and_layout`-level proof that the axis_y2 fill branch fires on
    /// a REAL secondary-y axis input: `chart_config.axis_y2.min_band` lands
    /// on `prep.axes.secondary_y[0].overrides.min_band` — via the SAME
    /// `apply_axis_config_to_axis_input` fill-only path `axis`/`axis_x`/
    /// `axis_y` use — and lands on NEITHER of the primary x/y axes, proving
    /// the fill is scoped to the secondary axis only.
    #[test]
    fn prepare_and_layout_axis_y2_fills_secondary_axis_overrides() {
        let (spec, batch) = secondary_y_chart();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let chart_config = ChartConfig {
            axis_y2: Some(AxisConfigSpec {
                style: chart_config::AxisStyleSpec { min_band: Some(123.0), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };

        let output = prepare_and_layout(&spec, &batch, &theme, viewport, &chart_config, None)
            .expect("prepare_and_layout must succeed with axis_y2 set on a chart with a secondary y axis");

        assert_eq!(output.prep.axes.secondary_y.len(), 1, "fixture must produce exactly one secondary axis");
        assert_eq!(
            output.prep.axes.secondary_y[0].overrides.min_band,
            Some(123.0),
            "axis_y2.min_band must fill the secondary axis's own overrides"
        );
        assert_eq!(output.prep.axes.x.overrides.min_band, None, "axis_y2 must not leak onto the primary x axis");
        assert_eq!(output.prep.axes.y.overrides.min_band, None, "axis_y2 must not leak onto the primary y axis");
    }

    /// The F-L07-06 flip: end to end through `render_svg` (a real render
    /// entry, not the internal `prep` struct), `chart_config.axis_y2.
    /// label_color` now reaches the rendered SVG — the exact repro the
    /// finding used (`AxisConfig(label_color="#654321")` never appearing in
    /// the output). RED-proven by commenting out this task's axis_y2 block
    /// in `prepare_and_layout` (the `if let Some(ref axis_y2_cfg) = chart_
    /// config.axis_y2 { ... }` above) and re-running: this test fails
    /// because `#654321` never reaches the SVG. Restored before committing.
    #[test]
    fn render_svg_axis_y2_label_color_reaches_secondary_axis_svg() {
        let (spec, batch) = secondary_y_chart();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let render_config = config::RenderConfig::default();
        let chart_config = ChartConfig {
            axis_y2: Some(AxisConfigSpec {
                style: chart_config::AxisStyleSpec {
                    label_color: Some("#654321".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };

        let with_override =
            render_svg(&spec, &batch, &theme, viewport, &render_config, &chart_config).unwrap();
        assert!(
            with_override.bytes.contains("fill=\"#654321\""),
            "axis_y2.label_color must reach the rendered secondary-axis tick labels"
        );

        // Contrast: the same chart without the override never emits that color.
        let baseline =
            render_svg(&spec, &batch, &theme, viewport, &render_config, &ChartConfig::default()).unwrap();
        assert!(
            !baseline.bytes.contains("fill=\"#654321\""),
            "the color must not appear absent an explicit axis_y2 override"
        );
    }

    /// The no-secondary-axis case emits `RenderWarning::ConfigSurfaceNotPresent`
    /// — asserted on the actual `warnings` a render entry returns, not just
    /// the variant's Display/round-trip coverage in the `tests` module above.
    #[test]
    fn render_svg_axis_y2_without_secondary_axis_emits_config_surface_warning() {
        let (spec, batch) = scatter_3(); // no independent_y layer -> secondary_y is empty
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let render_config = config::RenderConfig::default();
        let chart_config = ChartConfig {
            axis_y2: Some(AxisConfigSpec {
                style: chart_config::AxisStyleSpec { tick_count: Some(3), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };

        let result = render_svg(&spec, &batch, &theme, viewport, &render_config, &chart_config)
            .expect("axis_y2 on a chart with no secondary axis must warn, not fail");
        assert!(
            result
                .warnings
                .iter()
                .any(|w| matches!(w, RenderWarning::ConfigSurfaceNotPresent { section } if section == "axis_y2")),
            "expected a ConfigSurfaceNotPresent{{section: \"axis_y2\"}} warning, got: {:?}",
            result.warnings
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

