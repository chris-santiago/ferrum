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
use crate::spec::axis::AlignAxis;
use crate::transform::context::TransformWarning;

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
    /// "<channel> must be a numeric dtype (Float64/Float32,
    /// Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, or Timestamp); got
    /// <dtype>" — Stack's value-dtype check. `dtype` is pre-formatted (`{:?}` of
    /// the Arrow `DataType`) since it carries no channel token to un-flip.
    ///
    /// The named set is `arrow_cast::supported_numeric_dtype`'s, because since
    /// #184 `apply_stack` reads its value column through `col_as_f64` like its
    /// two sibling adjustments rather than through a hand-rolled downcast chain.
    /// It previously named only the chain's six dtypes ("Float64, UInt64, or a
    /// signed integer type"), which understated what stack accepts today.
    ValueDtype { channel: &'static str, dtype: String },
    /// "<channel> must be a dtype that can identify a row (Utf8/LargeUtf8,
    /// Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Float32/Float64,
    /// Boolean, or Timestamp); got <dtype>" — Stack's category-dtype check.
    /// `dtype` is pre-formatted (`{:?}` of the Arrow `DataType`), like
    /// [`ValueDtype`](Self::ValueDtype)'s.
    ///
    /// The named set is `arrow_cast::col_as_identity_key_str`'s, because since
    /// #184 `apply_stack` keys its category column through that reader — the
    /// ordinal band ladder's own enumerator, plus the temporal epoch reader for
    /// `Timestamp` — rather than through a two-arm dtype gate. It previously
    /// named only that gate's two dtypes ("column must be Float64 or Utf8"),
    /// which refused every integer, `Float32` and `Boolean` category column that
    /// the same chart's dodge, jitter and bands all accepted, and refused every
    /// stacked time series outright.
    CategoryDtype { channel: &'static str, dtype: String },
    /// A free-form reason with no positional channel token — every
    /// Dodge/Jitter message, plus Stack's column-not-found and
    /// `RecordBatch`-construction failures.
    Message(String),
}

/// Which of the two ways a per-layer `coord` (`#172`(1)) cannot be placed on a
/// layer of a shared panel — the detail carried by
/// [`RenderError::LayerCoordUnplaceable`], whose doc has the policy.
///
/// Same shape and purpose as [`PositionAdjustReason`] above: one refusal whose
/// `Display` composes a common frame around a structured reason, rather than two
/// refusals a caller could tell apart.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayerCoordUnplaceableReason {
    /// The coord declares a property of the whole PANEL — an axis domain
    /// (`xlim=`/`ylim=`), axis padding (`expand=False`), panel clipping
    /// (`clip=False`) or a fixed aspect ratio (`fm.CoordFixed`). The string is
    /// [`crate::spec::coord::panel_scoped_property`]'s answer: the user-facing
    /// spelling of the thing declared.
    ///
    /// Answerable from the coord alone, so it is asked at `build_scene`'s top,
    /// before any geometry.
    PanelProperty(&'static str),
    /// The coord leaves this layer's positional channels in the opposite space
    /// from the panel's axes, and one of the columns that lands on an axis cannot
    /// be drawn against it: a categorical column on a continuous axis, or a
    /// column that cannot be a category key on a categorical one.
    ///
    /// Not answerable from the coord alone — it needs the resolved scales and the
    /// layer's own columns — so it is asked in `build_scene`'s per-layer loop, at
    /// the same site as the sibling positional-column gate
    /// (`validate_layer_positional_columns`).
    ///
    /// This is the class the wave-5 close review found still silent: a member
    /// whose `CoordFlip` puts its categorical column on the panel's quantitative
    /// axis drew ZERO marks, with no warning and no error, on all three
    /// merged-flat routes. The sibling reason above already refused the same
    /// caller mistake spelled with `fm.CoordFixed`, so the two answers to one
    /// question are now one answer.
    ///
    /// `channel` is the RESOLVED (post-flip) axis the column landed on, `column`
    /// its name, `dtype` the pre-formatted Arrow dtype (`{:?}`), and `axis_kind`
    /// the panel axis's kind in user words (`"continuous"` / `"categorical"`).
    /// `channel` is deliberately NOT un-flipped for `Display`: it names the
    /// PANEL's axis, which is the thing the caller has to reconcile, not a
    /// channel they wrote.
    ExchangedChannelUnservable {
        channel: &'static str,
        column: String,
        dtype: String,
        axis_kind: &'static str,
    },
}

/// The boxed payload of [`RenderError::LayerAxisColumnUnplaceable`]. `channel`
/// is the RESOLVED (post-`CoordFlip`) axis token the union acted on, `field`
/// the offending column's name, `dtype` its pre-formatted Arrow dtype.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerAxisColumnDetail {
    pub(crate) channel: String,
    pub(crate) field: String,
    pub(crate) dtype: String,
}

/// What bound one of a line's width sources: a column (rendered `'name'`),
/// a constant mark kwarg (rendered as its number), or a `size` conditional
/// (rendered `fm.when(...)`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WidthBinding {
    Field(String),
    Constant(f64),
    /// The line's own `size` conditional (`size=fm.when(...)`).
    Conditional,
}

impl std::fmt::Display for WidthBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(name) => write!(f, "'{name}'"),
            Self::Constant(value) => write!(f, "{value}"),
            Self::Conditional => write!(f, "fm.when(...)"),
        }
    }
}

/// The boxed payload of [`RenderError::LineWidthConflict`]: the two bindings
/// that each set the line's width.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LineWidthSources {
    pub(crate) size: WidthBinding,
    pub(crate) stroke_width: WidthBinding,
}

/// Where a [`RenderError::LineWidthNotPositive`] width came from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NonPositiveLineWidth {
    /// The row's size `value` in `field` maps to `area` on the size scale.
    Scaled { field: String, value: f64, area: f64 },
    /// A row of `field` with no usable size (null, non-finite, or off the
    /// scale) takes the line's default width `width`.
    Fallback { field: String, width: f64 },
    /// A `size` conditional on `selection` has a branch of `area`.
    Conditional { selection: String, area: f64 },
}

/// The public entry a [`RenderError::TimeFormatOnNonTemporalScale`],
/// [`RenderError::OrdinalFormatOnTemporalScale`] or
/// [`RenderError::NumberFormatOnTemporalScale`] refusal names: where the
/// caller set the format, and so what the temporality was checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatSurface {
    /// A tick-label format (`Axis(label_format=)`, `encoding.format` on a
    /// positional channel, `configure_axis(label_format=)`): checked against
    /// the resolved scale.
    Axis,
    /// `Legend(format=, format_type=)` on a legend-bearing channel (color,
    /// size, shape, stroke-dash): checked against that channel's field.
    Legend,
    /// A mark channel's own `format=`/`format_type=` (`Text(...)`): checked
    /// against that channel's field.
    Channel,
}

impl FormatSurface {
    /// `(noun, needs, subject)` for a format-refusal sentence: what the
    /// surface is called, what it checks, and what was checked. One wording
    /// for `TimeFormatOnNonTemporalScale`, `OrdinalFormatOnTemporalScale` and
    /// `NumberFormatOnTemporalScale`.
    fn wording(self) -> (&'static str, &'static str, &'static str) {
        match self {
            FormatSurface::Axis => ("axis", "scale", "the resolved scale"),
            FormatSurface::Legend => ("legend", "field", "the field"),
            FormatSurface::Channel => ("channel", "field", "the field"),
        }
    }
}

/// The public entry a [`RenderError::InvalidScaleDomainConfig`] refusal
/// names: a `domain`/`range` the caller wrote directly on a scale (a class
/// or a raw `scale=` dict), or `configure_axis`'s chart-level domain config
/// (`domain_min`/`domain_max`/`zero`/`nice`, and the per-axis spellings)
/// resolved against the scale's own data domain. `Display` speaks in each
/// surface's own vocabulary: `Scale` names the scale the caller wrote,
/// `AxisConfig` keeps the existing axis-config wording, since that refusal
/// is about a chart-level config layer rather than the scale itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDomainSurface {
    /// `LinearScale(domain=[5.0])` or `{"type": "linear", "domain": [5.0]}`.
    Scale,
    /// `configure_axis(domain_min=, domain_max=, ...)` and its per-axis
    /// spellings (`axis_x`/`axis_y`/`axis_y2`).
    AxisConfig,
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
    ///
    /// `surface` picks the `Display` wording (see [`ScaleDomainSurface`]):
    /// `validate_explicit_continuous_shape` (a `scale=` dict/class's own
    /// malformed `domain`/`range`) constructs `Scale`; `apply_axis_domain_config`
    /// (the chart-level config layer) constructs `AxisConfig`, keeping the
    /// axis wording this refusal always had, since it is the config layer that
    /// is malformed, not the scale the caller wrote. `channel` is the RESOLVED
    /// (post-`CoordFlip`) token for the `Scale` surface; `coord_flipped` un-flips
    /// it back to what the user wrote. The `AxisConfig` surface always resolves
    /// `false` — `axis_x`/`axis_y`/`axis_y2` are resolved-slot vocabulary the
    /// user typed directly, never remapped under flip.
    InvalidScaleDomainConfig {
        channel: String,
        reason: String,
        surface: ScaleDomainSurface,
        coord_flipped: bool,
    },
    /// A band scale's `padding`/`paddingInner`/`paddingOuter` outside d3's
    /// bounds. The `BandScale` constructor refuses the same values at the
    /// call; this is the refusal for a scale dict (or an override type
    /// switch), which reaches Rust unvalidated. `channel` follows
    /// `InvalidScaleDomainConfig`'s convention; so does `coord_flipped` — the
    /// RESOLVED (post-`CoordFlip`) token un-flipped back to what the user
    /// wrote. Constructed `false` at `scale_resolve::positional::
    /// build_from_scale_spec`'s `ScaleSpec::Band` arm; corrected to the real
    /// flip flag by the one caller that has it in hand
    /// (`scale_resolve::positional::build_axis_scale`'s `map_err` around its
    /// `build_from_scale_spec` call).
    InvalidBandPadding {
        channel: String,
        error: crate::scale::discrete::BandPaddingError,
        coord_flipped: bool,
    },
    /// An `interpolate=` mark kwarg named a curve method outside the accepted
    /// vocabulary (spec §6.1). `value` is the rejected string; the accepted set
    /// in the message comes from
    /// [`marks::curve::Interpolate::accepted_values`], the same array dispatch
    /// runs on, so the two cannot drift.
    ///
    /// Refused rather than silently rendered as `linear`: before batch E,
    /// `mark_line(interpolate="monotone")` and a typo like
    /// `interpolate="mono"` produced the SAME straight-segment output, so a
    /// user had no way to tell an unimplemented curve from a misspelled one.
    /// Raised by [`draw::resolve_mark_style`], before any geometry is built.
    UnknownInterpolate { value: String },
    /// A `stroke_cap=` outside its vocabulary (`butt|round|square`), refused
    /// as [`Self::UnknownInterpolate`] is, where it used to parse to nothing
    /// and draw the default cap. Raised by [`draw::resolve_mark_style`].
    UnknownStrokeCap { value: String },
    /// A `stroke_join=` outside its vocabulary (`miter|round|bevel`), refused
    /// as [`Self::UnknownStrokeCap`] is.
    UnknownStrokeJoin { value: String },
    /// A `blend=` outside its vocabulary (`normal|alpha|additive`), refused
    /// as [`Self::UnknownStrokeCap`] is.
    UnknownBlend { value: String },
    /// A style field the mark never draws ([`crate::spec::mark::Mark::style_fields`]):
    /// accepted, it would render exactly as without it. REFUSE MALFORMED, from
    /// [`draw::resolve_mark_style`] for every producer; the per-layer draw loop
    /// wraps it in [`Self::InLayer`], which names the public entry. A constant
    /// `size` keeps its own refusal, [`Self::MarkSizeNotDrawn`].
    MarkStyleNotDrawn { mark: crate::spec::mark::Mark, field: &'static str },
    /// A mark whose geometry only makes sense under one specific coordinate
    /// system was asked to render under a different one (batch E, spec §6.4).
    /// `mark` names the mark (e.g. `"mark_geoshape"`, `"mark_arc"`);
    /// `required` names the coord kind it needs (`"Geo"`, `"Polar"`).
    ///
    /// **The prescribed fix was removed from this message (`#172`(1)).** Both
    /// producers used to append an actionable `hint` — `"add
    /// .coord(fm.CoordPolar(...))"` / `"add .coord(fm.CoordGeo(...))"` — and on
    /// the merged-flat layer path that hint named a call the caller had already
    /// written: `LayerChart(bar, arc.coord(fm.CoordPolar()))` dropped the arc
    /// layer's coord and then told the caller to add it. The drop is fixed (the
    /// coord rides the wire per layer), and the prescription is gone rather than
    /// left standing beside the fix, because a refusal that names a state
    /// (`requires a Polar coord`) stays true wherever it is raised from, while
    /// one that prescribes a fix is only true on some paths.
    ///
    /// Two marks refuse this way, each refusing something that previously
    /// degraded silently or misleadingly:
    /// - `marks::geoshape::build`: a non-`Geo` coord with `__geometry__` data
    ///   actually present in the batch (an empty/no-geometry batch still
    ///   returns a legitimately empty result — this is not "any non-Geo
    ///   geoshape refuses", only "non-Geo geoshape WITH geometry refuses").
    ///   Before this, `mark_geoshape` under the default Cartesian coord
    ///   silently rendered nothing, with no error and no warning. One
    ///   producer: geoshape only ever builds through this one mark builder.
    /// - `Mark::Arc` under any non-`Polar` coord: raised from TWO producers
    ///   that share one constructor (`marks::arc::requires_polar_coord`), so
    ///   the wording can never drift between them —
    ///   `scale_resolve::resolve_scales_with_leaf_context` raises it first,
    ///   ahead of the generic `EncodingTypeMismatch` fallthrough (`mark_arc`
    ///   without `.coord(fm.CoordPolar(...))` used to surface `encoding 'y'
    ///   expected EncodingSpec, got None`, which names the wrong problem),
    ///   and `marks::arc::build`'s own non-Polar arm raises it again as a
    ///   path-independent backstop — a merged-flat `LayerChart` layer (e.g.
    ///   `LayerChart(bar, arc).interactive()`, or
    ///   `resolve={"y": "independent"}`) resolves scales against the OTHER
    ///   layer's mark, so the scale-resolution guard never runs for the arc
    ///   layer there, and without the builder's own guard it silently
    ///   rendered nothing.
    ///
    /// `mark_image` under Polar/Geo does NOT raise this — unlike geoshape and
    /// arc, both of its build paths (url-tile, raster) place images in
    /// Cartesian pixel space with no polar/geo projection to fall back to, so
    /// there is no coord-independent way to draw anything under either coord.
    /// It instead names the mismatch as a `RenderWarning::MarkBuildEmpty`
    /// (#118) at `marks/image.rs`'s own Polar/Geo arm — a warning, not a
    /// refusal, matching every other #118 site's "this presence-legal
    /// combination genuinely cannot draw" shape rather than a malformed-input
    /// refusal.
    MarkRequiresCoord {
        mark: &'static str,
        required: &'static str,
    },
    /// A layer's own `coord` (`#172`(1)) cannot be placed on one layer of a panel
    /// every layer shares. `layer` is the index in `spec.layers`, `coord` the
    /// constructor the caller wrote, and `reason` which of the two ways it is
    /// unplaceable — see [`LayerCoordUnplaceableReason`].
    ///
    /// ONE refusal with two reasons rather than two refusals, deliberately: a
    /// caller who wrote an unplaceable per-layer coord is owed the same sentence
    /// whichever property of it could not be honored, and must not be able to
    /// tell from the message which gate spoke. The two producers are
    /// `scene_build::build_scene`'s chart-level gate (the panel-property reason,
    /// asked before any geometry) and its per-layer gate (the channel-exchange
    /// reason, which needs the resolved scales); both funnel every standalone
    /// render and every composite leaf, so all three merged-flat routes (static
    /// `chart + chart`, `resolve={"y": "independent"}`, `.interactive()`) refuse
    /// identically.
    ///
    /// REFUSE MALFORMED, and specifically not WARN DEGRADED: rendering with the
    /// declaration dropped is exactly the silent-substitution defect `#172`(1)
    /// closes, and neither reason has a degraded-but-honest version — there is no
    /// partial sense in which one layer of a shared panel has its own axis domain,
    /// or in which a categorical column is drawn against a continuous axis.
    LayerCoordUnplaceable {
        layer: usize,
        coord: &'static str,
        /// Boxed, for `clippy::result_large_err` and nothing else: the reason's
        /// two owned strings would widen `RenderError` past the point where the
        /// composite wrappers that embed it (`CompositeResolveError`,
        /// `CompositeRenderError`) exceed the 128-byte threshold, taxing every
        /// `Result` in those modules for a variant constructed at most once per
        /// failed render. Sibling reasons stay unboxed because they are small.
        reason: Box<LayerCoordUnplaceableReason>,
    },
    /// A layer or Overlay-group member's column bound to a continuous shared
    /// positional axis is neither numeric nor temporal, so the union that
    /// computes the shared domain cannot place it (`#120`(9), N1; spec
    /// §6.8). `layer` names the offending member's position; what it is a
    /// position IN depends on which of the two sites raised this:
    ///
    /// - `scale_resolve::domain::numeric_domain_union_over`'s per-layer loop
    ///   (a single `ChartSpec`'s own `layers` — the merged-flat chart,
    ///   `+`/`.interactive()`, and a hand-built spec whose `layers` is
    ///   `Some`): `layer` is the index in `spec.layers`.
    /// - `composite::refuse_unplaceable_overlay_members` (a bare
    ///   `fm.layer(a, b, ...)`, each member its own composite leaf — the
    ///   shape `LayerChart._composite_tree` actually lowers to): `layer` is
    ///   the member's position among the Overlay group's own leaves, and the
    ///   error reaches Python wrapped in `CompositeResolveError::
    ///   DomainResolve`'s envelope (spec §6.8's amended note) rather than
    ///   raised bare.
    ///
    /// Both sites read the same [`crate::render::arrow_cast::
    /// require_placeable_on_continuous_axis`] predicate and construct this
    /// variant identically, so the check and its text are one rule with two
    /// call sites, not two copies of it.
    ///
    /// `detail.channel` is deliberately left un-flipped for `Display`, for
    /// the same reason
    /// [`LayerCoordUnplaceableReason::ExchangedChannelUnservable`] leaves its
    /// own `channel` alone: it names the panel's axis and the `resolve=` key
    /// a caller would write for it, neither of which is an encoding key the
    /// caller authored on any one member.
    ///
    /// REFUSE MALFORMED: a column that cannot be placed on a continuous axis
    /// has no honest degraded render, the same policy
    /// [`Self::LayerCoordUnplaceable`] states for its own two reasons. A
    /// member whose field is simply absent from its batch is unaffected —
    /// the ordinary "not loaded here" outcome — a member whose own coord
    /// exchanges its channels relative to the establishing member is judged
    /// by the coord-exchange gate instead (`LayerCoordUnplaceableReason::
    /// ExchangedChannelUnservable`, on the merged-flat and composite routes
    /// alike), and `resolve={"<channel>": "independent"}` gives a
    /// member its own scale instead of refusing.
    LayerAxisColumnUnplaceable {
        layer: usize,
        /// Boxed, for `clippy::result_large_err` and nothing else — the same
        /// reason [`Self::LayerCoordUnplaceable`]'s own `reason` field is
        /// boxed: three owned strings here would widen `RenderError` past
        /// the point where the composite wrappers that embed it
        /// (`CompositeResolveError`, `CompositeRenderError`) exceed the
        /// 128-byte threshold.
        detail: Box<LayerAxisColumnDetail>,
    },
    /// A layer beyond the first bound a positional channel (`x`/`y`/`x2`/`y2`)
    /// to a column that is absent from **that layer's own resolved batch** —
    /// the base data, or the named `data_source` output the layer reads from
    /// (F-L10-01 rider, spec §4).
    ///
    /// A layer ≥ 1's encoding reaches scale resolution only where layer 0
    /// leaves the channel unbound (`scene_build::resolve_panel_scales` merges
    /// layer 0 onto the chart encoding, and `scale_resolve::positional_binding`
    /// falls back to the first layer that binds an otherwise-unbound `x`/`y`;
    /// `resolve_layer_y_scale` covers only an `independent_y` layer's `y`), so
    /// a layer ≥ 1 naming a missing positional column that layer 0 also binds
    /// used to reach the mark builders, where the `.ok()`-swallowing readers
    /// (`marks/point.rs`, `marks/text.rs`, `marks/geoshape.rs`) turned it into
    /// an empty mark batch: the layer simply vanished from the render, with no
    /// error and no warning. That is how a broken regression-fit layer shipped
    /// undetected.
    ///
    /// Shape follows the composite `CompositeRenderError::LeafRender { index,
    /// source }` precedent (`composite_render.rs`) — the offending index is
    /// named, so the message points at one layer rather than at the chart.
    /// Standalone single-mark charts are untouched: layer 0 keeps raising the
    /// plain [`RenderError::UnknownColumn`] from scale resolution.
    ///
    /// Scope is deliberately positional-only (#106 ownership): a non-owning
    /// layer that INHERITED a `color` channel naming a column its own batch
    /// does not carry keeps rendering exactly as before, and a zero-row layer
    /// keeps being skipped (an empty batch is not an unknown column).
    ///
    /// `channel` is the RESOLVED (post-`CoordFlip`) token the gate acted on;
    /// `coord_flipped` (R3) is the `Display`-time un-flip back to the token
    /// the user wrote, patched at the one construction site
    /// (`scene_build::validate_layer_positional_columns`) from
    /// `PreparedInputs::coord_flipped`.
    LayerUnknownColumn {
        layer: usize,
        channel: &'static str,
        name: String,
        coord_flipped: bool,
    },
    /// A layer whose mark DRAWS and needs both positional channels has no
    /// binding for one of them — neither on its own encoding nor on the chart
    /// encoding it inherits (wave 4 T6).
    ///
    /// The sibling of [`Self::EncodingTypeMismatch`], one level down. That one
    /// says "nothing in this spec binds this channel"; this one says "the
    /// channel IS bound, but not anywhere this layer can reach". The
    /// distinction became reachable when `scale_resolve::positional_binding`
    /// started sourcing an unbound axis from a later layer: the axis resolves,
    /// so the generic path stops erroring, but a sibling layer that needed a
    /// FIELD (not just a scale) has nothing to plot and its mark builder
    /// returns an empty batch — the same `.ok()`-swallowing silent drop
    /// [`Self::LayerUnknownColumn`] was written to stop, arriving by a
    /// different road.
    ///
    /// Inheritance is deliberately judged against the chart encoding only, NOT
    /// against `positional_binding`'s fallback. That is exactly what separates
    /// the two orderings of the same pair: in `mark_line(x=,y=) +
    /// mark_point(x=)` the line is layer 0, so the merge puts its `y` FIELD
    /// into the chart encoding and the point inherits a real column to plot
    /// (measured: 3 circles, unchanged). In `mark_point(x=) +
    /// mark_line(x=,y=)` the fallback resolves a y SCALE from the line layer
    /// but no `y` field ever reaches the chart encoding, so the point has
    /// nothing — measured pre-guard as a render with 0 circles and no warning.
    ///
    /// Whether the fallback SHOULD also make the field inheritable (which
    /// would render both orderings as 3 circles) is an open rendering-semantics
    /// question, deliberately not decided here; refusing is the conservative
    /// answer and leaves that door open behind a message that names the layer.
    ///
    /// `channel` is the RESOLVED (post-`CoordFlip`) token; `coord_flipped` (R3)
    /// is the `Display`-time un-flip back to what the user wrote, set at the one
    /// construction site (`scale_resolve::validate_drawing_layers_can_plot`).
    LayerMissingPositionalChannel {
        layer: usize,
        mark: &'static str,
        channel: &'static str,
        coord_flipped: bool,
    },
    /// An explicit color-range entry (an encoding's `scale.range=[...]`, a
    /// `Gradient([...])`'s stops, a discretizing `bucket_colors`, or a
    /// chart-level `configure_color(range=...)`) could not be parsed by
    /// [`color::parse_color`] (#126, spec §4.2 amended 2026-09-11). `entry` is
    /// the offending string.
    ///
    /// Refused rather than warned-and-discarded (the pre-#126 behavior): the
    /// parse is all-or-nothing at every one of the six sites this covers — a
    /// range/palette/gradient is indexed by position, so silently dropping the
    /// bad entry re-points every later entry rather than shortening a cycle, a
    /// WRONG mapping rather than a degraded one. The one site that was not
    /// even all-or-nothing — `configure_color(range=...)`'s continuous arm,
    /// which used to `filter_map(...ok())` bad stops out with no warning at
    /// all — is folded in here as a sibling defect of the same root cause.
    ColorRangeParseFailure { entry: String },
    /// A `Shape(scale=...)` explicit range entry does not match the fixed
    /// point-shape name table
    /// ([`marks::point::shape_from_str_strict`](crate::render::marks::point::shape_from_str_strict),
    /// no circle fallback) (#123, spec §4.9). `entry` is the offending entry,
    /// rendered via [`crate::scale::ordinal::OrdinalRangeValue`]'s `Display`
    /// regardless of whether it arrived as a string or (malformed) number —
    /// mirroring [`Self::ColorRangeParseFailure`]'s all-or-nothing rule: one
    /// bad entry refuses the whole range rather than silently drawing every
    /// unmatched glyph as a circle. `Shape(scale=…)`'s primary gate is
    /// Python's own eager `_VALID_POINT_SHAPES` check at construction
    /// (`ferrum.encoding.appearance.Shape._validate`); this is the defense
    /// for a hand-built spec that bypasses it (`ChartSpec::from_json` and
    /// similar raw-spec routes, the same audience
    /// [`marks::point::shape_from_str`](crate::render::marks::point::shape_from_str)'s
    /// own doc comment names), including a range that mixes in a numeric
    /// entry.
    ShapeRangeEntryUnrecognized { entry: String },
    /// A `StrokeDash(scale=...)` carries an explicit `range` at all — string,
    /// numeric, or mixed (#123, spec §4.9). That channel's output is the
    /// categorical scale's fixed legacy dash-index range
    /// (`ferrum_scene::STROKE_DASH_LEGACY_SLOTS`: solid plus the three
    /// patterns that predate the named-linetype append — the full
    /// [`ferrum_scene::DASH_PALETTE`] table is longer, but the
    /// categorical scale never cycles past its own legacy slots), not a
    /// value a scale can redeclare, so ANY explicit range refuses outright
    /// rather than being
    /// parsed entry-by-entry — there is no "one bad entry" to name, every
    /// entry is equally unsupported, unlike [`Self::ShapeRangeEntryUnrecognized`].
    /// `entries` is the received range verbatim (each item rendered via
    /// [`crate::scale::ordinal::OrdinalRangeValue`]'s `Display`), so the
    /// message names the value received. Python's `StrokeDash._validate`
    /// refuses the range at construction; this refuses a hand-built spec that
    /// carries one.
    StrokeDashRangeUnsupported { entries: Vec<String> },
    /// A layered `ChartSpec` (`layers` set) also carries a chart-level
    /// `mark_style`. A layered chart is styled per layer (each layer resolves
    /// `layer.mark_style > theme`), so the chart-level value could only be
    /// ignored; it refuses instead. `layers` is the layer count.
    LayeredChartMarkStyle { layers: usize },
    /// A layered `ChartSpec` whose `layers` list is empty: a chart is drawn
    /// from its layers, and every scale, axis and legend from layer 0's
    /// encoding, so a chart with none has nothing to draw. Python's chart
    /// API never builds one; this refuses a hand-built spec that carries one.
    LayeredChartNoLayers,
    /// A layer carries its own non-empty `transforms` (`layers[layer]`, the
    /// first such layer; `count` is its transform count). A layer never runs
    /// transforms of its own: it reads the chart's pipeline output, or a
    /// named chart-level output via `data_source`, so the list could only be
    /// ignored; it refuses instead. Python's `Layer` refuses the same input at
    /// construction; this refuses a hand-built spec that carries one.
    LayerTransformsUnsupported { layer: usize, count: usize },
    /// `conditionals[index].layers` names layer `layer`, but the chart has
    /// only `layers` layers (one when it is unlayered). The scope could only
    /// be ignored, so it refuses instead. Python stamps a conditional with the
    /// layers that declared it; this refuses a hand-built spec.
    ConditionalLayerOutOfRange { index: usize, layer: usize, layers: usize },
    /// A line mark binds both `size` and `stroke_width` (each a field or a
    /// constant kwarg; a theme width is never bound). The two set the same
    /// stroke's width in different units, `size` as a dot-diameter that can
    /// taper along the line and `stroke_width` in pixels, so there is no
    /// honest way to draw one and ignore the other: REFUSE MALFORMED.
    ///
    /// Decided once, in the line mark build, so every door (a Python chart, a
    /// figure function, a hand-built wire spec) reaches it. The per-layer draw
    /// loop wraps it in [`Self::InLayer`], which names the public entry in
    /// place of `mark_line`.
    LineWidthConflict {
        /// Boxed, for `clippy::result_large_err` and nothing else, the same
        /// reason [`Self::LayerAxisColumnUnplaceable`]'s `detail` is boxed.
        sources: Box<LineWidthSources>,
    },
    /// A constant `size=` (a dot area) that is zero, negative or not finite on
    /// a mark that draws one ([`crate::spec::mark::Mark::draws_constant_size`]):
    /// a point's radius `sqrt(size / PI)` and a line's width (the diameter of
    /// a dot of that area) exist only for a positive area. REFUSE MALFORMED,
    /// through [`draw::check_constant_size`], the one size rule.
    MarkSizeOutOfRange { mark: crate::spec::mark::Mark, value: f64 },
    /// A constant `size=` on a mark that never draws one
    /// ([`crate::spec::mark::Mark::draws_constant_size`]): accepted, it would
    /// render exactly as without it. REFUSE MALFORMED, through
    /// [`draw::check_constant_size`]; the text names the mark's own sizing
    /// kwarg where it has one ([`crate::spec::mark::Mark::size_knob`]).
    MarkSizeNotDrawn { mark: crate::spec::mark::Mark, value: f64 },
    /// A drawn vertex of a sized line would take a width that is not finite
    /// and `> 0`, the scene's `widths` contract: its size value maps to a
    /// non-positive area on the size scale, or a row with no usable size
    /// falls back to a non-positive default width. Neither has a width to
    /// draw, and substituting another would misreport the data, so the line
    /// refuses.
    LineWidthNotPositive {
        /// Boxed for `clippy::result_large_err`, as `LineWidthConflict`'s
        /// `sources` is.
        source: Box<NonPositiveLineWidth>,
    },
    /// A refusal raised while building one layer's marks, with the layer it
    /// came from: `layer` is its index on a layered chart (`None` on an
    /// unlayered one, whose one layer is the chart), and `entry` the public
    /// call that produced it ([`crate::spec::layer::Layer::entry`], else
    /// [`crate::spec::chart::ChartSpec::entry`], else `mark_{mark}`). Applied
    /// to every refusal of the per-layer draw loop's style and build steps, so
    /// naming is a property of where the refusal arose, not of its variant.
    /// `Display` prefixes `layer {i}: ` and voices `source` as `entry`.
    InLayer { layer: Option<usize>, entry: String, source: Box<RenderError> },
    /// A stroke node's `widths` breaks the scene contract the static SVG
    /// outline needs: misaligned with its vertices, a width that is not
    /// finite and `> 0`, or a closed path. The line mark never writes such a
    /// node, so this guards the scene seam rather than user input.
    TaperedStrokeMalformed { source: ferrum_scene::TaperError },
    /// A sort-field object (`{"field": ..., "op": ..., "order": ...}`)
    /// reaches render without a string `"field"`, or naming an `"op"`/
    /// `"order"` outside [`crate::spec::encoding::sort::SortOp::ACCEPTED`]/
    /// [`crate::spec::encoding::sort::SortOrder::ACCEPTED`]. Python's
    /// construction-time gate (`EncodingSpec.__new__`, every sort-accepting
    /// channel) refuses this shape before a spec can exist; this refuses a
    /// hand-built spec that bypasses it — `ChartSpec.from_json`,
    /// `ChartSpec(facet=dict)`, a layer's own `encoding` (`coerce_layers`,
    /// which deserializes through raw serde, never through
    /// `EncodingSpec::new`), and the `FacetSpec` `sort`/`row_sort` fields all
    /// reach render without passing through that gate. `location` names
    /// where the sort sits: a chart channel's wire name (`"x"`, `"color"`,
    /// ...), a layer channel (`"layers[{i}].{channel}"`), or a facet
    /// dimension (`"facet.sort"`/`"facet.row_sort"`); a caller that resolves
    /// a sort without that walk gets the category field it was sorting.
    /// `reason` is [`crate::spec::encoding::sort::parse_sort_field_object`]'s
    /// own refusal text — the SAME parser `EncodingSpec::new` calls, so a shape refused at
    /// construction and a shape refused here can never disagree. `value` is
    /// the malformed value, verbatim. Mirrors [`Self::LayerTransformsUnsupported`]:
    /// an accepted-but-inert wire shape refuses rather than being silently
    /// dropped.
    ///
    /// A second, structurally distinct malformation shares this variant
    /// rather than adding a third one: `FacetSpec.row_sort` set while
    /// `FacetSpec.row` is absent (wrap mode) has no row dimension to order,
    /// so it refuses too — `location` is `"facet.row_sort"`, `value` is the
    /// (possibly well-formed) row_sort value verbatim, and `reason` states
    /// the missing dimension rather than an object-shape complaint.
    SortSpecMalformed {
        location: String,
        reason: String,
        // Boxed (`clippy::result_large_err`): `serde_json::Value` is
        // wide enough that carrying it by value here pushed
        // `CompositeRenderError::LeafRender`'s wrapped `RenderError` past
        // clippy's 128-byte `Result` err-size threshold.
        value: Box<serde_json::Value>,
    },
    /// A time-typed format (`label_format="%b %Y"`, a preset like
    /// `"date_iso"` that resolves to one, or a `%`-free spec declared
    /// `format_type="time"`) was applied where the value it formats is not
    /// temporal (#133, spec §4.2 amended 2026-09-11). `surface` names the
    /// public entry the format was set on: on an [`FormatSurface::Axis`] the
    /// check is the RESOLVED scale (not [`scale_resolve::ScaleKind::Time`]);
    /// on a [`FormatSurface::Legend`] or [`FormatSurface::Channel`] it is the
    /// channel's field (`#147` close, wave 6A T21 — those surfaces used to
    /// render every value as a 1970 epoch date instead). `channel` is the
    /// RESOLVED (post-`CoordFlip`) token the formatting pass acted on;
    /// `format` is the offending spec, `None` when `format_type="time"` was
    /// given with no spec (refused on every surface, an axis included).
    ///
    /// The declaration-time grammar gate ([`config_apply::validate_chart_format_specs`])
    /// cannot catch this: it runs before scale resolution, so it only knows
    /// the DECLARED encoding type, not the RESOLVED scale kind — an
    /// `x="a:N"` axis with an explicit `Time` scale override, or a temporal
    /// column resolved as `Ordinal`, are both legitimate declared/resolved
    /// mismatches the loose grammar gate must tolerate. This refusal runs
    /// AFTER scale resolution, once the real `ScaleKind` is known, closing the
    /// gap the grammar gate leaves open: before this, a strftime pattern
    /// applied to a non-temporal scale silently fell through to the
    /// string-reparse path, which either passed the raw label through
    /// unchanged (ordinal labels) or, for a numeric label, reparsed it as an
    /// epoch-millisecond timestamp and rendered a nonsense 1970-epoch date.
    ///
    /// Three surfaces reach this, all keyed on the same resolved-`ScaleKind`
    /// predicate: the per-channel `Axis(label_format=...)` /
    /// `encoding.format` path (`prepare::apply_axis_format_or_thread`), the
    /// chart-level `configure_axis(label_format=...)` /
    /// `axis_x`/`axis_y`/`axis_y2` path (`config_apply::apply_label_format_to_axis`),
    /// and the per-panel independent-facet-axis path
    /// (`scene_build::independent_panel_axis_inputs`, `TickFormatMode::Immediate`) the
    /// issue's "both paths" phrasing did not name. Two field-checked surfaces
    /// reach it too, through the same constructor: the colorbar and the size,
    /// shape and stroke-dash legends
    /// (`prepare::legend::refuse_format_temporality_mismatch_legend_fields`) and the text channel
    /// (`marks::text::build`). The tooltip column would be a third, but no
    /// Python spelling puts a `format_type` on its wire today (`#147`
    /// follow-up 15c), so it does not call the constructor yet.
    ///
    /// `coord_flipped` follows the same R3 discipline as
    /// [`Self::InvalidAxisOrient`]: the per-channel and per-panel surfaces
    /// PATCH it at their own boundary (the `EncodingSpec` carrying the format
    /// traveled through `build_layers`' swap), while the chart-level surface
    /// is EXEMPT and always resolves `false` — `axis_x`/`axis_y`/`axis_y2` are
    /// resolved-slot vocabulary the user typed directly, never remapped under
    /// flip, so the resolved token already IS what the user wrote.
    TimeFormatOnNonTemporalScale {
        channel: &'static str,
        surface: FormatSurface,
        format: Option<String>,
        coord_flipped: bool,
    },
    /// The ordinal preset (`label_format="ordinal"`, the `"__ordinal__"`
    /// sentinel on the wire) meets a temporal value (`#147` close, wave 6A
    /// T22). The preset formats NUMBERS; on a temporal field there is no
    /// number the caller meant, so the surfaces used to give three different
    /// answers to the identical pair (`2001st` on the shared axis, the format
    /// dropped on an independent facet axis, the ordinal of the epoch-ms count
    /// on the legend and text). The mirror of
    /// [`Self::TimeFormatOnNonTemporalScale`], built by the same single
    /// constructor (`prepare::refuse_format_temporality_mismatch`) at the same
    /// surfaces, with the same `channel`/`surface`/`coord_flipped` discipline.
    OrdinalFormatOnTemporalScale {
        channel: &'static str,
        surface: FormatSurface,
        coord_flipped: bool,
    },
    /// A d3 number format (`".2f"`, `",.0f"`, `"d"`, a `"number"`-typed
    /// preset such as `"percent"`) meets a temporal value (`#147` close, wave
    /// 6A T23, user ruling). The surfaces used to give three answers to the
    /// identical pair: the shared axes formatted the calendar year
    /// (`2001.00`), an independent facet axis dropped the format, and the
    /// legend and text formatted the epoch-millisecond count
    /// (`959817600000.00`). The number-format sibling of
    /// [`Self::OrdinalFormatOnTemporalScale`], built by the same single
    /// constructor at the same surfaces with the same
    /// `channel`/`surface`/`coord_flipped` discipline. A time-typed numeric
    /// spec (`".2f"` + `"time"`) is not this error: it names no pattern and
    /// falls back to the temporal default.
    NumberFormatOnTemporalScale {
        channel: &'static str,
        surface: FormatSurface,
        format: String,
        coord_flipped: bool,
    },
    /// A chart combines a projected coordinate system — `Polar` or, since
    /// 2026-09-12, `Geo` — with a `BreakAxis` structural spec (#230-8, D3:
    /// REFUSE MALFORMED). `coord` names which one (`"Polar"`/`"Geo"`);
    /// `axis` is the offending `BreakAxis.axis` token (`"x"`/`"y"`).
    ///
    /// Pre-fix this was silent geometry destruction, not a suppression: the
    /// false claim (removed from `break_axis::BreakPlacement::for_kind`'s
    /// `Arc` arm doc) was that Cartesian breaks are suppressed under polar
    /// coords. For `Polar`, what actually happened is measured directly:
    /// `mark_arc`'s wedge transform runs during BATCH BUILD, before any
    /// break geometry exists, so the break fold then reverse-mapped the
    /// already-projected pixels through the Cartesian scale it replaced —
    /// destroying the geometry (measured: every polar wedge destroyed, 11
    /// sentinel occurrences in one repro) while still painting a break
    /// indicator over Cartesian axes that `route_panel_axes_and_grid`
    /// (`scene_build.rs`) suppresses under this coord, with zero warnings.
    ///
    /// Adjudicated 2026-09-12: the gate extends from `Polar`-only to also
    /// cover `Geo`, on a structural argument rather than an empirical one.
    /// Under `CoordGeo`, `mark_geoshape` places every vertex by bbox-fitting
    /// its `projection::forward`-projected lon/lat into the plot rect
    /// (`marks/geoshape.rs`'s scaling pass), entirely independent of the
    /// resolved x/y scales. `BreakRemap`'s reverse-map runs through the
    /// Cartesian scale it replaces, so for geo geometry there is no
    /// scale-relative position left for it to correct — the break contract
    /// cannot be honored for `Geo`, and `route_panel_axes_and_grid` suppresses the
    /// very Cartesian axes the break indicator would otherwise annotate.
    /// REFUSE MALFORMED is the sound response here, not silent geometry
    /// loss dressed up as a render.
    ///
    /// Raised at the top of `scene_build::build_scene` — the one funnel
    /// every standalone render AND every composite leaf's render passes
    /// through — before any break geometry, or any other geometry, is
    /// built. A composite leaf combining either coord with a break
    /// therefore refuses per-leaf, exactly like every other per-leaf
    /// render error.
    BreakAxisIncompatibleCoord { coord: &'static str, axis: AlignAxis },
    /// A `Utf8View`/`LargeUtf8` string column (raw input, or a transform
    /// output that materializes one — `data_flatten` on a polars
    /// `LargeList<LargeUtf8>`, a `LargeUtf8` groupby column
    /// `group_key::materialize_groupby_col` echoes back) could not be
    /// normalized to `Utf8` by `prepare::normalize_string_views`. `field`
    /// names the offending column for an `arrow::compute::cast` failure
    /// (e.g. more than `i32::MAX` bytes of string data — `LargeUtf8`'s
    /// reason for existing); empty for a normalized-schema `RecordBatch`
    /// rebuild failure, which is not about any one column.
    StringNormalizationFailed { field: Option<String>, reason: String },
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
        // #133: the per-channel and per-panel-independent-axis surfaces
        // construct this deep inside a fn with no flip context of its own
        // (same shape as `InvalidAxisOrient`), so it is patched through this
        // same boundary-correction fn rather than at its own construction
        // site. The chart-level surface resolves `false` directly at its own
        // (EXEMPT) construction site instead of routing through here — see
        // the field doc on `RenderError::TimeFormatOnNonTemporalScale`.
        RenderError::TimeFormatOnNonTemporalScale { channel, surface, format, .. } => {
            RenderError::TimeFormatOnNonTemporalScale { channel, surface, format, coord_flipped }
        }
        // `#147` T22: the mirror refusal, built by the same constructor at
        // the same surfaces, so it is patched here for the same reason.
        RenderError::OrdinalFormatOnTemporalScale { channel, surface, .. } => {
            RenderError::OrdinalFormatOnTemporalScale { channel, surface, coord_flipped }
        }
        // `#147` T23: the number-format sibling, same constructor and surfaces.
        RenderError::NumberFormatOnTemporalScale { channel, surface, format, .. } => {
            RenderError::NumberFormatOnTemporalScale { channel, surface, format, coord_flipped }
        }
        // A layer context corrects the refusal it carries.
        RenderError::InLayer { layer, entry, source } => {
            RenderError::InLayer { layer, entry, source: Box::new(with_coord_flipped(*source, coord_flipped)) }
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
        | RenderError::UnknownInterpolate { .. }
        | RenderError::UnknownStrokeCap { .. }
        | RenderError::UnknownStrokeJoin { .. }
        | RenderError::UnknownBlend { .. }
        | RenderError::MarkStyleNotDrawn { .. }
        | RenderError::MarkRequiresCoord { .. }
        // Carries a `coord_flipped` field, patched at its own single
        // construction site (`scene_build::validate_layer_positional_columns`,
        // which has `PreparedInputs::coord_flipped` in hand) — same discipline
        // as `EncodingTypeMismatch` and friends above, so this boundary must
        // leave it alone rather than overwrite it.
        | RenderError::LayerUnknownColumn { .. }
        // Wave 4 T6: same discipline as `LayerUnknownColumn` directly above —
        // carries `coord_flipped`, set at its own single construction site
        // (`scale_resolve::validate_drawing_layers_can_plot`, which derives it
        // from `spec.coord` exactly as the `EncodingTypeMismatch` beside it
        // does), so this boundary must leave it alone rather than overwrite it.
        | RenderError::LayerMissingPositionalChannel { .. }
        // Carries `coord_flipped`. The `Scale`-surface construction
        // (`scale_resolve::positional::validate_explicit_continuous_shape`)
        // is constructed `false` and corrected to the real flip flag by the
        // one caller that has it in hand
        // (`scale_resolve::positional::build_axis_scale`'s `map_err` around
        // its `build_from_scale_spec` call, the one place every error this
        // returns is a `scale=` dict's own refusal) — same discipline as
        // `LayerUnknownColumn`/`LayerMissingPositionalChannel` above, so this
        // boundary must leave it alone rather than overwrite it. The sibling
        // `AxisConfig`-surface construction inside `apply_axis_domain_config`
        // (the config-path backstop) resolves `false` directly and is never
        // corrected, like the chart-level `TimeFormatOnNonTemporalScale`
        // surface above.
        | RenderError::InvalidScaleDomainConfig { .. }
        // Carries `coord_flipped`, constructed `false` at
        // `scale_resolve::positional::build_from_scale_spec`'s
        // `ScaleSpec::Band` arm and corrected by the same caller and
        // discipline as `InvalidScaleDomainConfig` directly above.
        | RenderError::InvalidBandPadding { .. }
        // #126: carries no channel token at all — a color string, not a
        // positional channel.
        | RenderError::ColorRangeParseFailure { .. }
        // #230-8: carries a coord name and the `BreakAxis.axis` token, not a
        // positional encoding channel — nothing here to un-flip.
        | RenderError::BreakAxisIncompatibleCoord { .. }
        // `#172`(1): carries a layer index, a coord constructor and the
        // panel-scoped property it declared — no channel token either.
        | RenderError::LayerCoordUnplaceable { .. }
        // `#120`(9)/N1: carries a `channel` field, but like
        // `LayerCoordUnplaceable`'s own `ExchangedChannelUnservable` reason,
        // it deliberately stays RESOLVED (post-flip) for `Display` — it
        // names the panel's axis, not an encoding key the caller wrote —
        // so there is nothing here to un-flip.
        | RenderError::LayerAxisColumnUnplaceable { .. }
        // #123: neither variant carries a positional coordinate channel
        // token (an entry string, or a list of them) — nothing here to
        // un-flip, same as `ColorRangeParseFailure` above.
        | RenderError::ShapeRangeEntryUnrecognized { .. }
        | RenderError::StrokeDashRangeUnsupported { .. }
        // A layer count, no channel token.
        | RenderError::LayeredChartMarkStyle { .. }
        | RenderError::LayeredChartNoLayers
        // A layer index and a transform count, no channel token.
        | RenderError::LayerTransformsUnsupported { .. }
        // Two indices and a layer count, no channel token.
        | RenderError::ConditionalLayerOutOfRange { .. }
        // Two width bindings, no channel token.
        | RenderError::LineWidthConflict { .. }
        // A mark and a number; a field name and a width.
        | RenderError::MarkSizeOutOfRange { .. }
        | RenderError::MarkSizeNotDrawn { .. }
        | RenderError::LineWidthNotPositive { .. }
        // A scene-contract fault, no channel token.
        | RenderError::TaperedStrokeMalformed { .. }
        // `location` is a WIRE channel name (or "layers[i].{channel}"/
        // "facet.sort"/"facet.row_sort"), the authored key the check walked
        // `spec.encoding`/`spec.layers`/`spec.facet` under directly — same
        // AUTHORED-token discipline as `RenderWarning::SortSpecIgnored`'s
        // channel-shorthand family (never the resolved/post-flip slot), so
        // nothing here needs an un-flip correction.
        | RenderError::SortSpecMalformed { .. }
        // Carries a column name (or none, for the schema-rebuild case), not
        // a positional encoding channel token — nothing here to un-flip.
        | RenderError::StringNormalizationFailed { .. }) => other,
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
                write!(f, "invalid viewport: width={width}, height={height} (both must be finite and > 0)"),
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
                        "{adjustment}: {channel} must be a numeric dtype (Float64/Float32, \
                         Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, or Timestamp); \
                         got {dtype}"
                    )
                }
                PositionAdjustReason::CategoryDtype { channel, dtype } => {
                    let channel = prepare::user_facing_channel(channel, *coord_flipped);
                    write!(
                        f,
                        "{adjustment}: {channel} must be a dtype that can identify a row \
                         (Utf8/LargeUtf8, Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, \
                         Float32/Float64, Boolean, or Timestamp); got {dtype}"
                    )
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
            Self::UnsupportedChannelCombination { mark, .. } => self.fmt_as(mark, f),
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
            Self::InvalidScaleDomainConfig { channel, reason, surface, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                match surface {
                    ScaleDomainSurface::Scale => write!(f, "{channel} scale: {reason}"),
                    ScaleDomainSurface::AxisConfig => {
                        write!(f, "{channel} axis scale-domain config: {reason}")
                    }
                }
            }
            Self::InvalidBandPadding { channel, error, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                write!(
                    f,
                    "band scale on {channel}: {} {}",
                    error.key().wire_key(),
                    error.reason()
                )
            }
            Self::UnknownInterpolate { value } => {
                use crate::transform::vocabulary::Vocabulary;
                f.write_str(&marks::curve::Interpolate::refusal("interpolate", value))
            }
            Self::UnknownStrokeCap { value } => {
                use crate::transform::vocabulary::Vocabulary;
                f.write_str(&ferrum_scene::StrokeCap::refusal("stroke_cap", value))
            }
            Self::UnknownStrokeJoin { value } => {
                use crate::transform::vocabulary::Vocabulary;
                f.write_str(&ferrum_scene::StrokeJoin::refusal("stroke_join", value))
            }
            Self::UnknownBlend { value } => {
                use crate::transform::vocabulary::Vocabulary;
                f.write_str(&ferrum_scene::BlendMode::refusal("blend", value))
            }
            Self::InvalidFormatSpec { spec, reason } =>
                write!(f, "invalid format spec {spec:?}: {reason}"),
            Self::MarkRequiresCoord { mark, .. } => self.fmt_as(mark, f),
            // One frame, two reasons (see `LayerCoordUnplaceableReason`): the
            // prefix, the "cannot be placed on one of them" spine and the closing
            // instruction are shared verbatim, so a caller cannot tell which of
            // the two gates refused them. Only the middle clause — what about
            // this coord could not be honored — and the remedy specific to it
            // differ.
            Self::LayerCoordUnplaceable { layer, coord, reason } => {
                let (cause, remedy) = match &**reason {
                    LayerCoordUnplaceableReason::PanelProperty(property) => (
                        format!("{coord} declares {property}, which belongs to the panel every layer shares"),
                        "A per-layer coord may declare placement only: fm.CoordFlip, \
                         fm.CoordPolar, fm.CoordGeo, or a bare fm.CoordCartesian().",
                    ),
                    LayerCoordUnplaceableReason::ExchangedChannelUnservable {
                        channel,
                        column,
                        dtype,
                        axis_kind,
                    } => (
                        format!(
                            "{coord} leaves this layer's positional channels in the opposite \
                             space from the panel's axes, and the panel's {channel} axis is \
                             {axis_kind} while the column that lands on it ('{column}': \
                             {dtype}) cannot be drawn against a {axis_kind} axis"
                        ),
                        "The panel's axes are resolved from the first member, so a member \
                         whose channels sit in the opposite space can only be placed where \
                         both of its columns suit the axes they land on.",
                    ),
                };
                write!(
                    f,
                    "layer {layer}: {cause} — a layered chart (fm.layer(...) / chart + chart) \
                     merges its members onto one panel, so this coord cannot be placed on one \
                     of them. {remedy} Move it to the first member (whose coord IS the \
                     panel's), or drop it."
                )
            }
            Self::LayerAxisColumnUnplaceable { layer, detail } => {
                let LayerAxisColumnDetail { channel, field, dtype } = &**detail;
                write!(
                    f,
                    "layer {layer}: column '{field}' ({dtype}) cannot be drawn on the shared \
                     {channel} axis, which is continuous — a layered chart (fm.layer(...) / \
                     chart + chart) merges its members onto one panel. Cast the column, or give \
                     the layer its own scale with resolve={{\"{channel}\": \"independent\"}}."
                )
            }
            Self::LayerUnknownColumn { layer, channel, name, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                write!(
                    f,
                    "layer {layer}: encoding '{channel}' references unknown column '{name}' \
                     (not present in that layer's data)"
                )
            }
            Self::LayerMissingPositionalChannel { layer, mark, channel, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                write!(
                    f,
                    "layer {layer}: mark_{mark} needs both positional channels, but no \
                     '{channel}' encoding reaches it — a sibling layer binds '{channel}', \
                     which resolves the axis but leaves mark_{mark} with nothing to plot. \
                     Encode '{channel}' on this layer, or on the chart, to draw it."
                )
            }
            Self::ColorRangeParseFailure { entry } => write!(
                f,
                "could not parse color '{entry}' in an explicit color range"
            ),
            Self::StrokeDashRangeUnsupported { entries } => {
                let received = entries.iter().map(|e| format!("'{e}'")).collect::<Vec<_>>().join(", ");
                let slots = ferrum_scene::STROKE_DASH_LEGACY_SLOTS;
                write!(
                    f,
                    "StrokeDash(scale=...) does not honor an explicit range (received [{received}]): \
                     the categorical scale's output is a fixed set of {slots} palette patterns, not a \
                     value a scale can redeclare"
                )
            }
            Self::LayeredChartMarkStyle { layers } => write!(
                f,
                "ChartSpec.mark_style is set on a layered chart ({layers} layers), where it has no \
                 effect: a layered chart is styled per layer. Move the style onto each layer's \
                 own mark_style."
            ),
            Self::LayeredChartNoLayers => write!(
                f,
                "ChartSpec.layers is an empty list: a layered chart draws its layers, and this one \
                 has none. Add at least one layer, or leave layers unset to draw the chart's own mark."
            ),
            Self::ConditionalLayerOutOfRange { index, layer, layers } => write!(
                f,
                "conditionals[{index}].layers names layer {layer}, but the chart has {layers} \
                 layer(s); name only the layers that declared the conditional."
            ),
            Self::LayerTransformsUnsupported { layer, count } => write!(
                f,
                "layer {layer}: layers[{layer}].transforms carries {count} transform(s), but a \
                 layer does not run its own transforms. Give the layer its own chart with \
                 .transform(...) and compose it with +, which routes the transform to that layer \
                 alone."
            ),
            Self::ShapeRangeEntryUnrecognized { entry } => write!(
                f,
                "Shape(scale=...) range entry '{entry}' is not a recognized shape name"
            ),
            Self::SortSpecMalformed { location, reason, value } => write!(
                f,
                "{location}: sort={value} is malformed ({reason})"
            ),
            Self::TimeFormatOnNonTemporalScale { channel, surface, format, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                // The sole constructor (`prepare::refuse_format_temporality_mismatch`)
                // resolves a non-temporal value's pair under
                // `TimeAutodetect::Denied`, so every refusal is
                // a DECLARED `"time"` type: naming it names the cause even for
                // a `%`-free spec like `".2f"`, where the format alone does
                // not say why a temporal value is needed (wave 6A T17).
                let cause = match format {
                    Some(format) => format!("format {format:?} is typed \"time\", which"),
                    None => "format type \"time\"".to_string(),
                };
                let (noun, needs, subject) = surface.wording();
                write!(
                    f,
                    "{cause} on the '{channel}' {noun} requires a temporal {needs}, \
                     but {subject} is not temporal"
                )
            }
            Self::OrdinalFormatOnTemporalScale { channel, surface, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                let (noun, needs, subject) = surface.wording();
                write!(
                    f,
                    "the \"ordinal\" format on the '{channel}' {noun} formats numbers and \
                     requires a non-temporal {needs}, but {subject} is temporal"
                )
            }
            Self::NumberFormatOnTemporalScale { channel, surface, format, coord_flipped } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                let (noun, needs, subject) = surface.wording();
                write!(
                    f,
                    "the number format {format:?} on the '{channel}' {noun} requires a \
                     non-temporal {needs}, but {subject} is temporal; a date pattern \
                     (e.g. \"%Y\") formats a temporal value"
                )
            }
            // `{:?}` over the wire TOKEN, never over `axis` itself: the
            // delivered text quotes the token (`BreakAxis(axis="y")`), which is
            // what `Debug` on the `String` this field used to be produced and
            // what callers' assertions read. `Debug` on `AlignAxis` would print
            // the variant (`BreakAxis(axis=Y)`) — see `#229`(2) and the
            // `Display` impl on [`AlignAxis`]. Pinned verbatim by this file's
            // `tests::break_family_messages_are_byte_identical_after_the_axis_retype`,
            // which asserts the whole sentence on both axis tokens; Python's
            // `tests/test_break_axis_geometry.py::
            // test_polar_coord_with_break_axis_on_x_names_that_axis` pins the
            // quoting from the caller's side.
            Self::BreakAxisIncompatibleCoord { coord, axis } => write!(
                f,
                "BreakAxis(axis={:?}) is not supported under {coord} coordinates; \
                 remove the break or render this chart under Cartesian coords",
                axis.wire_name()
            ),
            Self::StringNormalizationFailed { field, reason } => match field {
                None => write!(f, "could not normalize string columns to Utf8: {reason}"),
                Some(field) => write!(f, "could not normalize column '{field}' to Utf8: {reason}"),
            },
            Self::LineWidthConflict { .. } | Self::LineWidthNotPositive { .. } => self.fmt_as("mark_line", f),
            Self::MarkSizeOutOfRange { mark, .. }
            | Self::MarkSizeNotDrawn { mark, .. }
            | Self::MarkStyleNotDrawn { mark, .. } => self.fmt_as(&format!("mark_{}", mark.as_str()), f),
            Self::InLayer { layer, entry, source } => {
                if let Some(layer) = layer {
                    write!(f, "layer {layer}: ")?;
                }
                source.fmt_as(entry, f)
            }
            Self::TaperedStrokeMalformed { source } => {
                write!(f, "the scene's tapered stroke cannot be drawn: {source}")
            }
        }
    }
}

impl std::error::Error for RenderError {}

impl From<ferrum_scene::TaperError> for RenderError {
    fn from(source: ferrum_scene::TaperError) -> Self {
        Self::TaperedStrokeMalformed { source }
    }
}

/// The size rule's reason clause for a size value [`draw::check_constant_size`]
/// (via [`RenderError::MarkSizeOutOfRange`]) rejects as out of range: the
/// text [`RenderError::fmt_as`] writes after `{entry}(size=...) `, and the
/// text `mark_size_reason` returns to Python with no call name, so the two
/// cannot diverge.
fn mark_size_out_of_range_reason(value: f64) -> String {
    format!("must be a positive number; received {value}")
}

/// The size rule's reason clause for a mark [`draw::check_constant_size`] (via
/// [`RenderError::MarkSizeNotDrawn`]) finds draws no constant size: the text
/// [`RenderError::fmt_as`] writes after `{entry}(size=...) ` (sans that
/// method's own trailing period), and the text `mark_size_reason` returns to
/// Python with no call name, so the two cannot diverge.
fn mark_size_not_drawn_reason(mark: crate::spec::mark::Mark, value: f64) -> String {
    let mut reason = format!("has no effect on mark_{mark}; received {value}");
    if let Some(knob) = mark.size_knob() {
        reason.push_str(&format!(". Set {knob}= instead"));
    }
    reason
}

impl RenderError {
    /// Write this refusal voiced as `entry`, the public call the caller
    /// invoked. A refusal whose sentence is built around the call names it in
    /// place (`{entry}(size=...) ...`, `{entry}: size and stroke_width ...`);
    /// every other one is prefixed `{entry}: `.
    fn fmt_as(&self, entry: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LineWidthConflict { sources } => {
                let LineWidthSources { size, stroke_width } = &**sources;
                write!(
                    f,
                    "{entry}: size and stroke_width both set the line's width (size={size}, \
                     stroke_width={stroke_width}); bind only one. size= sets a dot-diameter width \
                     that can taper along the line; stroke_width= sets a width in pixels."
                )
            }
            Self::MarkSizeOutOfRange { value, .. } => {
                write!(f, "{entry}(size=...) {}", mark_size_out_of_range_reason(*value))
            }
            Self::MarkStyleNotDrawn { mark, field } => {
                let drawn: Vec<&str> = mark.style_fields().iter().map(|f| f.name).collect();
                write!(
                    f,
                    "{entry}: {field} has no effect on a {mark} mark; a {mark} mark draws {}",
                    drawn.join(", ")
                )
            }
            Self::MarkSizeNotDrawn { mark, value } => {
                write!(f, "{entry}(size=...) {}.", mark_size_not_drawn_reason(*mark, *value))
            }
            Self::LineWidthNotPositive { source } => match &**source {
                NonPositiveLineWidth::Scaled { field, value, area } => write!(
                    f,
                    "{entry}: size '{field}' value {value} maps to area {area} on the size \
                     scale, which gives no positive line width. Every width along a line must \
                     be positive; give the size scale a range above 0."
                ),
                NonPositiveLineWidth::Fallback { field, width } => write!(
                    f,
                    "{entry}: size '{field}' has a row with no usable size (null, \
                     non-finite or off the scale), which takes the line's default width \
                     {width}. Every width along a line must be positive; set a positive \
                     Theme(line_stroke_width=)."
                ),
                NonPositiveLineWidth::Conditional { selection, area } => write!(
                    f,
                    "{entry}: size=fm.when(...) on selection '{selection}' has a branch of \
                     size {area}, which gives no positive line width. Every width along a line \
                     must be positive; give every branch a size above 0."
                ),
            },
            Self::MarkRequiresCoord { required, .. } => write!(f, "{entry} requires a {required} coord"),
            Self::UnsupportedChannelCombination { channel, hint, hint_alt_channel, coord_flipped, .. } => {
                let channel = prepare::user_facing_channel(channel, *coord_flipped);
                let hint: std::borrow::Cow<str> = match hint_alt_channel {
                    Some(alt) => std::borrow::Cow::Owned(
                        hint.replacen("{alt}", prepare::user_facing_channel(alt, *coord_flipped), 1),
                    ),
                    None => std::borrow::Cow::Borrowed(*hint),
                };
                write!(f, "{entry}: channel '{channel}' is not supported; {hint}")
            }
            other => write!(f, "{entry}: {other}"),
        }
    }

    /// Wrap a refusal raised while building one layer in [`Self::InLayer`],
    /// naming its `layer` index (on a layered chart) and its public `entry`.
    /// A refusal already wrapped keeps its context.
    pub(crate) fn in_layer_context(self, layer: Option<usize>, entry: &str) -> Self {
        match self {
            wrapped @ Self::InLayer { .. } => wrapped,
            source => Self::InLayer { layer, entry: entry.to_owned(), source: Box::new(source) },
        }
    }

    /// Name layer `index` on a refusal [`Self::in_layer_context`] wrapped
    /// without one: a composite overlay renders each member as an unlayered
    /// chart and knows the member's layer index only outside it. Every other
    /// error passes through unchanged.
    pub(crate) fn in_layer(self, index: usize) -> Self {
        match self {
            Self::InLayer { layer: None, entry, source } => Self::InLayer { layer: Some(index), entry, source },
            other => other,
        }
    }

    /// The refusal inside any [`Self::InLayer`] context.
    #[cfg(test)]
    pub(crate) fn root(&self) -> &Self {
        match self {
            Self::InLayer { source, .. } => source.root(),
            other => other,
        }
    }

    /// This error's size-rule reason clause, naming no call: `Some` for
    /// [`Self::MarkSizeOutOfRange`] and [`Self::MarkSizeNotDrawn`], the two
    /// variants [`draw::check_constant_size`] raises; `None` for every other
    /// variant. The sole caller is `mark_size_reason`.
    pub(crate) fn size_rule_reason(&self) -> Option<String> {
        match self {
            Self::MarkSizeOutOfRange { value, .. } => Some(mark_size_out_of_range_reason(*value)),
            Self::MarkSizeNotDrawn { mark, value } => Some(mark_size_not_drawn_reason(*mark, *value)),
            _ => None,
        }
    }
}

/// Which resolution an opacity-family channel fell back to when its `scale=`
/// carried a non-`Linear` spec (see [`RenderWarning::UnsupportedOpacityScale`]).
/// Decided at the same early return that picks between the two behaviors in
/// `scale_resolve::auxiliary::build_opacity_channel_scale`, so the warning
/// names the fallback the channel actually took rather than always claiming
/// the linear one. Two variants, `Copy`, matched exhaustively in `Display`.
///
/// Not the same concept as `marks::opacity::OpacityFallback` (that one picks
/// the *default* an absent value falls back to; this one names which *scale
/// resolution* a present-but-unsupported `scale=` fell back to).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpacityScaleFallback {
    /// Data extent (or the `scale=` domain) onto the theme opacity band (or
    /// the `scale=` range) — the channel's ordinary default resolution.
    Linear,
    /// `fill_opacity`/`stroke_opacity` over a constant column with no
    /// explicit `scale=` domain or range: no scale is built, and each row
    /// keeps its own value as a literal alpha. An explicit domain or range
    /// over the same constant column is a real scale instead (spec §6.10),
    /// so this fallback is only reported when the channel's `scale=` is
    /// non-`Linear` (which never populates domain/range overrides) and the
    /// column is otherwise a literal alpha.
    LiteralAlpha,
}

/// Where a layer left out of the layered axis a caller's scale establishes
/// draws its marks (see [`RenderWarning::LayerScaleExcluded`]). The two
/// layering routes differ here: a flat layered chart (`+`, `Chart.layer`)
/// has one scale per axis, while each layer of a composite overlay
/// (`fm.layer`) is its own leaf with its own scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcludedLayerPlacement {
    /// The flat route: the layer's own scale is not applied, and its marks are
    /// placed on the one shared axis, whose extent does not include them.
    SharedAxis,
    /// The composite overlay route: the layer resolves its own scale, which
    /// no axis in the panel labels.
    OwnScale,
}

/// A chart-level config section that addresses a surface only some charts
/// have ([`RenderWarning::ConfigSurfaceNotPresent`]). Serialized as the config
/// key the user set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSurface {
    /// `configure_axis_y2` / `axis_y2=`: the secondary y axis.
    AxisY2,
}

impl ConfigSurface {
    /// The config key the user set.
    pub fn key(self) -> &'static str {
        match self {
            Self::AxisY2 => "axis_y2",
        }
    }
}

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
    /// A transform degraded rather than refused (GH #162) — an all-NaN/
    /// empty fit, a silently clamped parameter, or a whole-column fallback.
    /// Wraps [`TransformWarning`], whose own `Display` is the delivered
    /// text, mirroring how [`RenderWarning::Layout`] delegates to
    /// [`LayoutWarning`]'s. Produced by `transform::context::TransformContext::warnings`,
    /// drained by `render::prepare::run_transforms_with_facet` into the
    /// render pipeline's own warnings vec.
    Transform(TransformWarning),
    OutOfDomainRows { mark: String, count: u64 },
    ColorPaletteOverflowed { categories: u32 },
    ShapePaletteOverflowed { categories: u32 },
    /// A categorical `stroke_dash` channel named more categories than the dash
    /// index space has slots (`ferrum_scene::STROKE_DASH_LEGACY_SLOTS`: solid
    /// plus the three patterns that predate the named-linetype append, not
    /// [`DASH_PALETTE`](ferrum_scene::DASH_PALETTE)'s full length), so
    /// patterns were recycled and two categories draw with the same dash. Mirrors
    /// [`ShapePaletteOverflowed`](Self::ShapePaletteOverflowed) — the palette is
    /// finite, so the degradation stays (spec §4.3) but is never silent.
    StrokeDashPaletteOverflowed { categories: u32 },
    /// A categorical axis (Ordinal/Band/Point) resolved with so many
    /// categories relative to its pixel range that each category's band or
    /// point allotment rounds down under one pixel (#183) — sibling to
    /// [`ColorPaletteOverflowed`](Self::ColorPaletteOverflowed): the scale
    /// still resolves (there is no invalid input to refuse), so the
    /// degradation stays, but a chart where every category silently collapses
    /// to a sliver is otherwise indistinguishable from a rendering bug.
    /// `categories` is the domain length; `bandwidth_px` is the resolved
    /// per-category pixel width that triggered the warning.
    GroupCardinalityDegenerate { categories: u32, bandwidth_px: f64 },
    /// A mark builder discovered a genuine defect — a presence-legal channel
    /// combination (`validate_mark_encoding` already passed it) that still
    /// lacks something it needs to draw anything — and produced zero marks
    /// (#118). `layer` is the layer's index in declaration order; `mark` is
    /// its kind; `reason` is a ready-made sentence fragment naming what was
    /// missing (see `crate::render::draw::MarkBuildResult::empty_reason`'s
    /// doc for the exemption boundary this crosses — a legitimate empty
    /// never reaches this variant).
    MarkBuildEmpty {
        layer: usize,
        mark: String,
        reason: String,
    },
    EmptyPanel { panel_index: usize },
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
    /// The facet-dimension sibling of [`Self::SortSpecIgnored`]: `Facet`/
    /// `FacetRow`/`FacetCol(sort=)` named a well-formed sort (a channel
    /// reference, or a sort-field object naming a field absent from the
    /// batch) that the facet partition could not resolve. Facet panels fall
    /// back to data first-appearance order; `reason` explains why, in the
    /// SAME words [`Self::SortSpecIgnored`] would use for the identical
    /// underlying failure (both are produced by
    /// `scale_resolve::apply_sort_to_domain`, the one ordinal-domain sort
    /// resolver).
    ///
    /// A distinct variant from `SortSpecIgnored`, not a reuse of it, because
    /// [`prepare::provisional_scale_warning_superseded_by_panel_pass`]
    /// depends on the variant to decide whether a warning produced before
    /// `scene_build::build_scene`'s panel loop gets restated (and so can be
    /// de-duplicated) once a panel resolves: every `SortSpecIgnored` IS
    /// restated, because it comes from POSITIONAL/color scale resolution,
    /// which the panel-real pass re-runs; a facet's own sort resolution runs
    /// exactly ONCE, in `prepare::prepare_render_inputs`, before any panel
    /// loop exists to restate it — collapsing the two variants would silently
    /// drop this warning on every render with at least one non-empty panel
    /// (which is nearly every render), since the shared predicate would
    /// strip it as an assumed duplicate that was never actually re-derived.
    FacetSortSpecIgnored { reason: String },
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
    /// A grid declared that one of its cells spans another's extent
    /// (`CompositeNode::Composite::extent_alignments`, `#182`), and one of the
    /// two named cells is a nested **composite** — a subtree of leaves laid
    /// out in its own frame, with no single plot rect to copy from or onto.
    /// The declaration is dropped and that cell renders as if nothing were
    /// declared.
    ///
    /// Not a refusal: the caller wrote something coherent (a `LayerChart` as a
    /// jointplot's centre panel is a legitimate figure) and the render
    /// proceeds, which makes it degraded rather than malformed — WARN
    /// DEGRADED. Silence is the one response excluded, because an
    /// unaligned marginal is indistinguishable from the alignment being
    /// broken. `follower`/`leader` are the declared CELL positions (row-major
    /// in the grid's `children`, holes counted) and `axis` is `"x"`/`"y"`, so
    /// a grid carrying several declarations says WHICH one was ignored.
    ExtentAlignmentUnresolvable {
        follower: usize,
        leader: usize,
        axis: String,
    },
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
    /// amended 2026-09-01). The curve, domain, and range are NOT honored;
    /// `fallback` names which resolution the channel actually took instead
    /// (spec §4.2, amended 2026-09-24: the message no longer always claims
    /// the linear fallback when a constant `fill_opacity`/`stroke_opacity`
    /// column took the literal-alpha path). `channel` is the channel name
    /// (`"opacity"` / `"fill_opacity"` / `"stroke_opacity"`); `scale_kind` is
    /// the dropped spec's kind (`"log"`, `"pow"`, `"sqrt"`, …). Full
    /// non-linear opacity-curve support is a logged campaign follow-up, not
    /// this batch.
    UnsupportedOpacityScale {
        channel: String,
        scale_kind: String,
        fallback: OpacityScaleFallback,
    },
    /// A chart-level config section named a surface the chart does not have:
    /// `axis_y2` (D2/F-L07-06, spec §4.1) on a chart with no `independent_y`
    /// layer has no secondary y-axis input (`AxesInput.secondary_y` is empty)
    /// for the override to fill, so it is dropped with this warning rather
    /// than silently discarded. `section` names the config key the user set.
    ConfigSurfaceNotPresent { section: ConfigSurface },
    /// A per-channel `Legend(values=[...])` on a CATEGORICAL legend named
    /// entries the legend does not have (D6/F-L04-05, spec §4.4). `values`
    /// filters and orders the entries; a name matching none of them cannot be
    /// drawn as a swatch — there is no category, no color, and no scale slot
    /// behind it — so it is skipped and reported here rather than silently
    /// ignored or invented as an empty swatch. `values` names the unmatched
    /// entries in the order the caller wrote them; the matched ones still
    /// filter and order the legend normally.
    LegendValuesUnknown { values: Vec<String> },
    /// Two channels sharing one legend surface each stated an explicit title,
    /// and they disagree (#111, spec §4.6). A `shape`/`stroke_dash` channel
    /// encoding the same field as `color` is folded INTO the color legend
    /// rather than given its own block, which leaves one title slot for two
    /// statements. The color channel owns the surface so its title stands
    /// (`kept`); `None` there means it explicitly asked for no title at all.
    /// The folded channel's title cannot be shown, so it is named here rather
    /// than silently discarded — which is precisely what shipped before, and
    /// is indistinguishable from the title having been ignored.
    LegendTitleConflict {
        kept: Option<String>,
        discarded: Vec<String>,
    },
    /// A `shape` AND a `stroke_dash` channel both encode the same field as
    /// `color`, so both folded into the one color legend (#111, spec §4.6) and
    /// landed on the same entries. A swatch cannot be a point glyph and a dash
    /// pattern at once, so one of the two identities is not drawn.
    ///
    /// `kept` is the channel the swatches depict (`"shape"` — what a
    /// standalone `shape` aux block already draws, so a folded legend agrees
    /// with an unfolded one); `dropped` is the channel that could not be shown
    /// (`"stroke_dash"`). Sibling to
    /// [`LegendTitleConflict`](Self::LegendTitleConflict), which reports the
    /// same loss for the merge's title: one legend surface, two channels
    /// addressing it, and the one that cannot be shown is named rather than
    /// silently discarded. Rolled up per chart, not per category — both
    /// domains come from the same field, so a collision on one entry is a
    /// collision on all of them.
    LegendFoldNotShown { kept: String, dropped: String },
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
    /// An annotation was DROPPED because a `BreakAxis` gap contains one of its
    /// data coordinates and left nothing of it to draw (spec §4 H5, F-L08-07;
    /// narrowed by #228's stratified clip). Three populations reach here: a
    /// point-anchored kind (`text`, `callout`, `image`) whose coordinate lands
    /// inside a gap — it has no extent to cut, so in-gap is the whole story —
    /// an extent-shaped kind with an in-gap endpoint whose surviving portion is
    /// EMPTY, and a `Cut`-routed extent kind whose coordinates on THIS axis
    /// degrade to a single point (one of its two bounds is pixel-/norm-space
    /// rather than data, e.g. `ann.rect(y1=<data>, y2=fm.px(...))`): with only
    /// one data-space bound there is no interval to clip, so an in-gap value
    /// there drops the whole annotation exactly as a point-anchored kind's
    /// would, even though `rect`'s own routing is `Cut`. An extent with any
    /// surviving portion is clipped per visible segment instead, silently,
    /// exactly as the equivalent mark geometry is; `bracket` is the one extent
    /// kind still dropped whenever a gap reaches it (its endpoint tips admit
    /// no honest split), and reports here when the in-gap coordinate is one of
    /// its own endpoints.
    ///
    /// What is dropped is the whole annotation — not just the offending
    /// coordinate — because a partially-suppressed annotation (a callout whose
    /// text moved but whose leader line vanished, say) is worse than none. An
    /// annotation no gap reaches is remapped through the break like marks are,
    /// and pushes no warning.
    ///
    /// Rolled up per axis (#240-3): this used to fire once per suppressed
    /// annotation ELEMENT — the internal per-primitive `AnnotationSpec`, not
    /// the user-facing `annotate_*()` call — so a chart whose annotations
    /// desugar to multiple primitive specs per call (`annotate_hline`'s line
    /// plus its own label Text, both landing on the same in-gap value)
    /// doubled the count on top of that — emitting N (or 2N) near-identical
    /// warnings for one underlying fact ("this axis's break gap swallowed
    /// some annotation elements"). The frozen contract ("a chart with N
    /// in-gap annotations emits ONE rolled-up suppression warning carrying
    /// the count") treats every suppressed element as one bucket per axis,
    /// mirroring [`AnnotationSpansBreakGap`](Self::AnnotationSpansBreakGap)'s
    /// and [`LegendValuesUnknown`](Self::LegendValuesUnknown)'s "bundle every
    /// offender into one `Vec`, not one warning each" shape — genuinely, not
    /// just in the field list: the Display arm prints
    /// [`dedup_kinds_for_display`]'s first-appearance-order-deduped join of
    /// `kinds`, the same way `LegendValuesUnknown` prints its own `values`
    /// (quality-review finding: the field must not be collected and then
    /// dropped). The dedup is Display-only — `kinds.len()` (undeduped) is
    /// still the count the message's leading number reports, so 4 suppressed
    /// `annotate_hline`s (8 elements: 4 `line` + 4 `text`) read "8 annotation
    /// elements (line, text)", not "8 annotation elements (line, text, line,
    /// text, line, text, line, text)". `kinds` is each suppressed element's
    /// wire `type` tag (`"text"`, `"line"`, `"span"`, …), in suppression
    /// order — the Display text says "annotation element(s)", not
    /// "annotation(s)", because the count is honestly of elements: this
    /// layer has no attribution back to which user call produced which
    /// element, and claiming a user-call count here would be dishonest for
    /// exactly the multi-primitive desugars this rollup exists to cover.
    AnnotationSuppressedByBreak {
        axis: AlignAxis,
        kinds: Vec<String>,
    },
    /// An EXTENT-shaped annotation (rect, span, line, arrow, bracket) on a
    /// broken axis had no coordinate inside a gap but *straddled* one — its two
    /// data coordinates bound an interval that contains the gap (spec §4 H5) —
    /// and was dropped rather than clipped. Post-#228 that is the narrow case:
    /// a straddling extent is normally CLIPPED into one piece per visible
    /// segment, silently, so this fires only when the interval cannot be split
    /// honestly (`bracket`, whose endpoint tips are what mark the bracketed
    /// range's ends) or when it contributes no length to any visible segment
    /// (its bounds are exactly a gap's bounds — both endpoints visible values,
    /// nothing between them). Drawing either would assert continuity exactly
    /// where the axis says it is broken.
    ///
    /// Rolled up per axis (#240-3): this used to fire once per suppressed
    /// annotation element, so a chart with N such elements emitted N
    /// near-identical warnings — the per-instance `extent`/`gap` values were
    /// never independently actionable (there is exactly one fix regardless of
    /// which element: move the gap, or move the annotation), so `kinds`
    /// bundles every suppressed element's kind into ONE warning per axis,
    /// mirroring [`LegendValuesUnknown`](Self::LegendValuesUnknown)'s
    /// "bundle every offender into one `Vec`, not one warning each" shape —
    /// genuinely: the Display arm prints [`dedup_kinds_for_display`]'s
    /// deduped join of `kinds` (the stored field itself keeps every element,
    /// undeduped — see the sibling
    /// [`AnnotationSuppressedByBreak`](Self::AnnotationSuppressedByBreak)'s
    /// doc for why that matters for the leading count).
    AnnotationSpansBreakGap {
        axis: AlignAxis,
        kinds: Vec<String>,
    },
    /// A `BreakAxis` resolved to zero effective gaps, so the chart rendered
    /// exactly as if the break had not been declared: no indicator, no
    /// compressed scale, no moved mark. Either every gap fell outside the
    /// axis's data domain, or the axis is ordinal and has no numeric domain
    /// for a gap to clip against. `reason` says which. Emitted because a
    /// structural feature that silently does nothing is indistinguishable
    /// from one the renderer failed to honor.
    BreakAxisIneffective { axis: AlignAxis, reason: String },
    /// A `BreakAxis` was declared on an axis whose resolved scale is
    /// continuous but NOT linear in data (`log`, `pow`/`sqrt`, `symlog`).
    /// The break is applied post-hoc in pixel space and reverse-maps
    /// pixel→data LINEARLY (`break_axis::BreakRemap::data_at`), which is not
    /// the inverse of those scales' mapping, so the removed region and the
    /// indicator position do not correspond to the requested gap: ticks
    /// outside the gap can hide while marks inside it draw. The chart still
    /// renders — placement is unreliable, not absent — and the scale-level
    /// inverse that would fix it is a filed follow-up. `scale_kind` is the
    /// offending kind's name (`"pow"` covers `Sqrt`, which resolves as
    /// `Pow(0.5)`; the one exact case a kind test cannot exclude is a
    /// hand-written `Pow(exponent=1)`, whose transform IS linear — hence the
    /// message hedges with "can differ" rather than asserting a mis-mapping).
    ///
    /// The message also names the second consequence: annotation suppression
    /// asks a DATA question (`gap_containing` on the spec's own value) while
    /// mark placement asks a PIXEL one, and on these scales the two answers
    /// diverge — an annotation can be suppressed as in-gap in the same render
    /// where a mark at a nominally in-gap value draws.
    BreakAxisOnNonlinearScale { axis: AlignAxis, scale_kind: String },
    /// A `BreakAxis` clip split one mark batch's nodes (1→N) while one of
    /// that batch's per-node sidecars (`data_indices`, tooltips, hrefs,
    /// descriptions, keys, polar spans) had the wrong length, so it could not
    /// be rebuilt in lockstep and was passed through unchanged — leaving that
    /// batch's tooltips/links/keys attributed to the wrong marks. This is a
    /// builder bug (a `debug_assert` catches it in a debug build); the
    /// warning is how a release build says so instead of shipping the
    /// misattribution silently. `mark` is the batch kind.
    BreakClipSidecarMisaligned { mark: String },
    /// Two or more `BreakAxis` specs shared an axis and disagreed on
    /// `break_style` or `break_size` (#230-4). Same-axis specs merge into
    /// one gap list and render as if declared as one `BreakAxis` with the
    /// union of gaps; a cosmetic field the merged specs disagree on can only
    /// keep one value, so the FIRST declaration's value is kept and every
    /// later, differing value for that field is dropped. `field` names which
    /// knob conflicted (`"break_style"` or `"break_size"`); merging the gaps
    /// themselves is the documented semantics of a same-axis break and never
    /// warns on its own. Its `{axis, field}` payload is a pure function of
    /// the leaf's own declared `BreakAxis` specs, with no data or scale
    /// dependence, so `composite_render.rs`'s `is_figure_level_dedupable`
    /// joins it explicitly (quality-review cycle-3): a `RepeatChart`
    /// replicating one conflicting template across N panels reports the
    /// authored mistake once at the composite boundary, not once per panel.
    BreakAxisMergeConflict { axis: AlignAxis, field: String },
    /// A continuous color encoding named a `scheme`/`cmap` (via `encoding.scheme`
    /// or a scale spec's own `scheme` field) that is not a recognized continuous
    /// scheme name (#240-4b, routed rider). Falls back to the theme's default
    /// sequential/diverging scheme rather than the requested one — the render
    /// is still meaningful, just not painted with the asked-for palette, so this
    /// is a WARN-DEGRADED case, not a refusal. `name` is the unrecognized
    /// scheme name.
    ///
    /// The primary refusal for this lives at Python desugar time
    /// (`resolve_cmap_alias`, #240-4a): this warning is the belt for a
    /// hand-built wire spec (or a channel-less mark site not yet routed
    /// through that desugar validation) that reaches Rust with an
    /// unvalidated name.
    UnknownColorScheme { name: String },
    /// A `Sequential` scale spec's `stops` parsed but produced fewer than 2
    /// entries — not enough to form a gradient (wave-1 close remediation,
    /// #240-4b sibling). The typed `fm.Gradient(...)` constructor already
    /// refuses this at construction (spec reviewer cycle 3, finding 2), so
    /// this is reachable only from a hand-written wire spec that bypasses
    /// that constructor; the resolver falls back to the theme's default
    /// scheme instead of silently degrading with no diagnostic (the last
    /// silent arm `#240-4b`'s own de-silencing left in
    /// `resolve_continuous_scheme`). `count` is the number of stops that
    /// survived parsing.
    GradientTooFewStops { count: usize },
    /// A swarm's packed dots reach `reach_px` from their category's centre,
    /// past `half_slot_px`, half the distance between neighbouring
    /// categories: they spill into the next category (or past the plot
    /// edge). The dots are drawn where they were packed, keeping their
    /// spacing; WARN DEGRADED. `entry` is the public call that built the
    /// layer. Raised once per swarm, not per facet panel.
    SwarmOverflowsBand { entry: String, category: String, value: String, reach_px: f64, half_slot_px: f64 },
    /// A layer of a layered chart was left out of the axis another layer's
    /// scale establishes on one channel, so its marks draw somewhere other
    /// than that axis: `placement` says where. `layer` is the excluded
    /// layer's position; `establishing` is the position of the layer whose
    /// scale owns the axis; `channel` is `"x"`/`"y"`. One per excluded layer
    /// per channel. Which scale establishes the axis, and which layers it
    /// admits, is the rule `scale_resolve::establishing_caller_scale` and
    /// `scale_resolve::warn_members_off_caller_scale` decide and document;
    /// this variant only reports what they decided. `layer` and
    /// `establishing` count the way the shared-axis refusal counts:
    /// positions among a flat chart's layers in declaration order, or
    /// positions among an `Overlay` group's members on the composite route
    /// (a nested `+` member numbers its own layers separately).
    LayerScaleExcluded {
        layer: usize,
        establishing: usize,
        channel: String,
        placement: ExcludedLayerPlacement,
    },
    /// A style field the mark draws only alongside another input
    /// ([`crate::spec::mark::Mark::style_fields`]) was set without that
    /// input, so it drew nothing: WARN DEGRADED. Decided after scales
    /// resolve (`draw::inert_style_fields`), per panel, deduplicated so a
    /// facet chart raises it once rather than once per panel. Voiced as the
    /// layer's refusals are ([`RenderError::InLayer`]): `layer` is its index
    /// on a layered chart, `entry` the public call that built it.
    MarkStyleInert {
        layer: Option<usize>,
        entry: String,
        mark: crate::spec::mark::Mark,
        field: String,
        needs: crate::spec::mark::StyleNeed,
    },
}

/// First-appearance-order dedup of a break-suppression rollup's `kinds`, for
/// Display only. The stored field keeps every suppressed element (its
/// length is the COUNT the message's leading number reports — deduping the
/// field itself would silently shrink that count), but a chart with 4
/// suppressed `annotate_hline`s (each a `line` + its own label `text`)
/// listing "line, text, line, text, line, text, line, text" is noise, not
/// information: only which DISTINCT kinds were involved is worth reading.
fn dedup_kinds_for_display(kinds: &[String]) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for k in kinds {
        if !seen.contains(&k.as_str()) {
            seen.push(k.as_str());
        }
    }
    seen.join(", ")
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
            RenderWarning::Transform(w) => write!(f, "{w}"),
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
            RenderWarning::GroupCardinalityDegenerate { categories, bandwidth_px } => {
                let n = *categories;
                write!(
                    f,
                    "{n} categor{} {} placed on an axis with only {bandwidth_px:.2}px per \
                     category; {} may not render visibly",
                    if n == 1 { "y" } else { "ies" },
                    if n == 1 { "was" } else { "were" },
                    if n == 1 { "it" } else { "some" },
                )
            }
            RenderWarning::MarkBuildEmpty { layer, mark, reason } => write!(
                f,
                "{mark} layer {layer} drew nothing: {reason}"
            ),
            RenderWarning::EmptyPanel { panel_index } => write!(
                f,
                "panel {panel_index} is too small to render and was left empty"
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
            RenderWarning::FacetSortSpecIgnored { reason } => write!(
                f,
                "facet sort spec could not be applied ({reason}); panels fall back \
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
            RenderWarning::ExtentAlignmentUnresolvable { follower, leader, axis } => write!(
                f,
                "grid cell {follower} cannot span cell {leader}'s {axis} extent: one of them is a \
                 nested composite with no single plot rect; that panel lays out on its own terms"
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
            RenderWarning::UnsupportedOpacityScale { channel, scale_kind, fallback } => match fallback {
                OpacityScaleFallback::Linear => write!(
                    f,
                    "{channel}: {scale_kind} scale is not supported on opacity-family channels; \
                     the curve, domain, and range were ignored in favor of the default linear \
                     resolution"
                ),
                OpacityScaleFallback::LiteralAlpha => write!(
                    f,
                    "{channel}: {scale_kind} scale is not supported on opacity-family channels; \
                     the column is constant, so its values were used as literal alphas instead \
                     of a scale"
                ),
            },
            RenderWarning::ConfigSurfaceNotPresent { section } => write!(
                f,
                "chart config names '{}' but the chart has no matching surface; \
                 the override was not applied",
                section.key()
            ),
            RenderWarning::LegendValuesUnknown { values } => write!(
                f,
                "legend values [{}] match no legend entry and were skipped",
                values.join(", ")
            ),
            RenderWarning::LegendTitleConflict { kept, discarded } => {
                let held = match kept {
                    Some(k) => format!("the color channel's title '{k}' stands"),
                    None => "the color channel suppressed the legend title".to_string(),
                };
                write!(
                    f,
                    "{held}; a same-field channel titles the same merged legend [{}], \
                     which could not be shown",
                    discarded.join(", ")
                )
            }
            RenderWarning::LegendFoldNotShown { kept, dropped } => write!(
                f,
                "{kept} and {dropped} encode the same field, so both merged into one legend; \
                 its swatches show {kept} and the {dropped} patterns could not be shown"
            ),
            RenderWarning::ScaleDomainConfigOnOrdinalAxis { channel, fields } => write!(
                f,
                "{} appl{} to continuous scales; the {channel} axis is ordinal, so {} not applied",
                fields.join(", "),
                if fields.len() == 1 { "ies" } else { "y" },
                if fields.len() == 1 { "it was" } else { "they were" },
            ),
            RenderWarning::AnnotationSuppressedByBreak { axis, kinds } => {
                let n = kinds.len();
                write!(
                    f,
                    "{n} annotation element{} ({}) on the {axis} axis fell inside a BreakAxis \
                     gap and {} suppressed",
                    if n == 1 { "" } else { "s" },
                    dedup_kinds_for_display(kinds),
                    if n == 1 { "was" } else { "were" },
                )
            }
            RenderWarning::AnnotationSpansBreakGap { axis, kinds } => {
                let n = kinds.len();
                write!(
                    f,
                    "{n} annotation element{} ({}) on the {axis} axis span{} a BreakAxis gap \
                     and {} suppressed",
                    if n == 1 { "" } else { "s" },
                    dedup_kinds_for_display(kinds),
                    if n == 1 { "s" } else { "" },
                    if n == 1 { "was" } else { "were" },
                )
            }
            RenderWarning::BreakAxisIneffective { axis, reason } => write!(
                f,
                "BreakAxis on the {axis} axis had no effect: {reason}"
            ),
            RenderWarning::BreakAxisOnNonlinearScale { axis, scale_kind } => write!(
                f,
                "BreakAxis on the {axis} axis targets a {scale_kind} scale; break \
                 placement is computed with a linear pixel-to-data map, so the removed \
                 range and the indicator position can differ from the gap you asked for, \
                 and annotation suppression (which reads the data value directly) can \
                 disagree with where marks are placed"
            ),
            RenderWarning::BreakClipSidecarMisaligned { mark } => write!(
                f,
                "{mark} marks were split by a BreakAxis clip while their per-node \
                 metadata had the wrong length; tooltips, links, and keys on that \
                 batch may name the wrong marks"
            ),
            RenderWarning::BreakAxisMergeConflict { axis, field } => write!(
                f,
                "multiple BreakAxis specs on the {axis} axis disagree on {field}; \
                 the first declaration's value was kept"
            ),
            RenderWarning::UnknownColorScheme { name } => write!(
                f,
                "color scheme '{name}' is not recognized; the theme's default scheme \
                 was used instead"
            ),
            RenderWarning::GradientTooFewStops { count } => write!(
                f,
                "the gradient's {count} stop(s) cannot form a gradient (at least 2 are \
                 required); the theme's default scheme was used instead"
            ),
            RenderWarning::SwarmOverflowsBand { entry, category, value, reach_px, half_slot_px } => write!(
                f,
                "{entry}: Swarm(category='{category}', value='{value}') packs its dots up to {reach_px:.1} px \
                 from their category's centre, past the {half_slot_px:.1} px half-slot each category has, so \
                 they spill into the neighbouring categories; reduce the dot size or spacing, or give the \
                 chart more room along the category axis"
            ),
            RenderWarning::MarkStyleInert { layer, entry, mark, field, needs } => {
                if let Some(layer) = layer {
                    write!(f, "layer {layer}: ")?;
                }
                write!(
                    f,
                    "{entry}: {field} has no effect on this {mark} mark without {}; it was ignored",
                    needs.describe(*mark)
                )
            }
            RenderWarning::LayerScaleExcluded { layer, establishing, channel, placement } => {
                let draws = match placement {
                    ExcludedLayerPlacement::SharedAxis => {
                        "its marks are placed on that axis and may fall outside it"
                    }
                    ExcludedLayerPlacement::OwnScale => {
                        "its marks draw on the layer's own scale, not the labelled axis"
                    }
                };
                write!(
                    f,
                    "layer {layer}: its {channel} scale differs from the one layer {establishing} \
                     establishes the {channel} axis with, so it stays out of that axis; {draws}"
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderOutput<T> {
    pub bytes: T,
    pub layout: crate::layout::LayoutResult,
    pub warnings: Vec<RenderWarning>,
}

/// `(kind, size)` per circle or line of a scene's legend, in order: a
/// circle's radius, a line's stroke width.
#[cfg(test)]
pub(crate) fn legend_swatches(scene: &ferrum_scene::SceneGraph) -> Vec<(&'static str, f64)> {
    use ferrum_scene::SceneNode;
    fn walk(nodes: &[SceneNode], out: &mut Vec<(&'static str, f64)>) {
        for node in nodes {
            match node {
                SceneNode::Circle { r, .. } => out.push(("circle", *r)),
                SceneNode::Line { style, .. } => out.push(("line", style.width)),
                SceneNode::Group { children, .. } => walk(children, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(&scene.legend, &mut out);
    out
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
            RenderWarning::Transform(TransformWarning::DegenerateFit {
                transform: "Smooth".into(),
                group: Some("g=a".into()),
                reason: "fewer than 2 usable points".into(),
            }),
            RenderWarning::OutOfDomainRows { mark: "point".into(), count: 3 },
            RenderWarning::ColorPaletteOverflowed { categories: 12 },
            RenderWarning::ShapePaletteOverflowed { categories: 7 },
            RenderWarning::StrokeDashPaletteOverflowed { categories: 6 },
            RenderWarning::GroupCardinalityDegenerate { categories: 500, bandwidth_px: 0.4 },
            RenderWarning::MarkBuildEmpty {
                layer: 0,
                mark: "rect".into(),
                reason: "mark_rect() needs an ordinal x or y axis".into(),
            },
            RenderWarning::EmptyPanel { panel_index: 1 },
            RenderWarning::ColorRangeBucketCountMismatch { expected: 3, received: 2 },
            RenderWarning::ColorDomainOmitsCategories { categories: vec!["c".into()] },
            RenderWarning::SortSpecIgnored { reason: "missing field".into() },
            RenderWarning::FacetSortSpecIgnored { reason: "missing field".into() },
            RenderWarning::PositionAdjustSkipped {
                adjustment: "dodge".into(),
                reason: "by-column 'grp' not found in data".into(),
            },
            RenderWarning::OverlayGuttersDiverged { layers: 2 },
            RenderWarning::ExtentAlignmentUnresolvable {
                follower: 0,
                leader: 2,
                axis: "x".into(),
            },
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
                fallback: OpacityScaleFallback::Linear,
            },
            RenderWarning::UnsupportedOpacityScale {
                channel: "fill_opacity".into(),
                scale_kind: "log".into(),
                fallback: OpacityScaleFallback::LiteralAlpha,
            },
            RenderWarning::ConfigSurfaceNotPresent { section: ConfigSurface::AxisY2 },
            RenderWarning::LegendValuesUnknown { values: vec!["zzz".into()] },
            RenderWarning::LegendTitleConflict {
                kept: Some("species".into()),
                discarded: vec!["Style".into()],
            },
            RenderWarning::LegendTitleConflict {
                kept: None,
                discarded: vec!["Style".into()],
            },
            RenderWarning::LegendFoldNotShown {
                kept: "shape".into(),
                dropped: "stroke_dash".into(),
            },
            RenderWarning::ScaleDomainConfigOnOrdinalAxis {
                channel: "x".into(),
                fields: vec!["domain_min".into(), "nice".into()],
            },
            RenderWarning::AnnotationSuppressedByBreak {
                axis: AlignAxis::Y,
                kinds: vec!["text".into(), "line".into()],
            },
            RenderWarning::AnnotationSpansBreakGap {
                axis: AlignAxis::Y,
                kinds: vec!["rect".into(), "span".into()],
            },
            RenderWarning::BreakAxisIneffective {
                axis: AlignAxis::Y,
                reason: "no gap overlaps the axis domain (0, 100)".into(),
            },
            RenderWarning::BreakAxisOnNonlinearScale {
                axis: AlignAxis::Y,
                scale_kind: "log".into(),
            },
            RenderWarning::BreakClipSidecarMisaligned { mark: "bar".into() },
            RenderWarning::BreakAxisMergeConflict {
                axis: AlignAxis::Y,
                field: "break_style".into(),
            },
            RenderWarning::UnknownColorScheme { name: "not-a-scheme".into() },
            RenderWarning::GradientTooFewStops { count: 1 },
            RenderWarning::LayerScaleExcluded {
                layer: 2,
                establishing: 0,
                channel: "y".into(),
                placement: ExcludedLayerPlacement::SharedAxis,
            },
            RenderWarning::LayerScaleExcluded {
                layer: 1,
                establishing: 0,
                channel: "x".into(),
                placement: ExcludedLayerPlacement::OwnScale,
            },
        ] {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: RenderWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w);
        }
    }

    /// The config surface a `ConfigSurfaceNotPresent` names crosses the wire,
    /// and is printed, as the config key the user set.
    #[test]
    fn config_surface_is_its_config_key_on_the_wire() {
        let w = RenderWarning::ConfigSurfaceNotPresent { section: ConfigSurface::AxisY2 };
        assert_eq!(serde_json::to_string(&w).unwrap(), r#"{"type":"config_surface_not_present","section":"axis_y2"}"#);
        assert_eq!(
            w.to_string(),
            "chart config names 'axis_y2' but the chart has no matching surface; the override was not applied"
        );
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

        let facet_sort = RenderWarning::FacetSortSpecIgnored {
            reason: "missing field".into(),
        };
        let facet_text = format!("{facet_sort}");
        assert!(
            facet_text.contains("facet sort spec could not be applied"),
            "Display text was: {facet_text}"
        );
        assert!(facet_text.contains("missing field"), "Display text was: {facet_text}");
        assert!(!facet_text.contains("FacetSortSpecIgnored"), "Display leaked Debug: {facet_text}");

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

        // #182: a dropped extent-alignment names WHICH declaration was ignored
        // (follower cell, leader cell, axis), because a grid may carry several
        // and "something was ignored" is not actionable.
        let unresolvable = RenderWarning::ExtentAlignmentUnresolvable {
            follower: 3,
            leader: 2,
            axis: "y".into(),
        };
        let unresolvable_text = format!("{unresolvable}");
        assert!(unresolvable_text.contains("cell 3"), "{unresolvable_text}");
        assert!(unresolvable_text.contains("cell 2"), "{unresolvable_text}");
        assert!(unresolvable_text.contains("y extent"), "{unresolvable_text}");
        assert!(unresolvable_text.contains("nested composite"), "{unresolvable_text}");
        assert!(
            !unresolvable_text.contains("ExtentAlignmentUnresolvable"),
            "{unresolvable_text}"
        );

        // A layer left out of a caller scale's axis: one sentence, with the
        // clause the route's placement decides, asserted verbatim.
        let excluded = |placement| {
            format!(
                "{}",
                RenderWarning::LayerScaleExcluded { layer: 2, establishing: 0, channel: "y".into(), placement }
            )
        };
        assert_eq!(
            excluded(ExcludedLayerPlacement::SharedAxis),
            "layer 2: its y scale differs from the one layer 0 establishes the y axis with, so it stays \
             out of that axis; its marks are placed on that axis and may fall outside it"
        );
        assert_eq!(
            excluded(ExcludedLayerPlacement::OwnScale),
            "layer 2: its y scale differs from the one layer 0 establishes the y axis with, so it stays \
             out of that axis; its marks draw on the layer's own scale, not the labelled axis"
        );

        // D2/F-L07-06 (this task): `axis_y2` names a surface the chart lacks.
        let no_surface = RenderWarning::ConfigSurfaceNotPresent { section: ConfigSurface::AxisY2 };
        let no_surface_text = format!("{no_surface}");
        assert!(no_surface_text.contains("axis_y2"), "{no_surface_text}");
        assert!(no_surface_text.contains("no matching surface"), "{no_surface_text}");
        assert!(!no_surface_text.contains("ConfigSurfaceNotPresent"), "{no_surface_text}");

        // #183: the singular arm — a 1-category sub-pixel axis must read
        // "1 category was placed ... it may not render visibly", not the
        // no-op-plural "1 categories were placed ... some may not render
        // visibly" every other numbered variant in this match avoids.
        let one_category = RenderWarning::GroupCardinalityDegenerate {
            categories: 1,
            bandwidth_px: 0.5,
        };
        let one_category_text = format!("{one_category}");
        assert!(one_category_text.contains("1 category "), "{one_category_text}");
        assert!(!one_category_text.contains("1 categories"), "{one_category_text}");
        assert!(one_category_text.contains("was placed"), "{one_category_text}");
        assert!(one_category_text.contains("it may not render visibly"), "{one_category_text}");
        assert!(
            !one_category_text.contains("GroupCardinalityDegenerate"),
            "{one_category_text}"
        );

        // D6/F-L04-05 (this task): unknown categorical `Legend(values=)` names.
        let unknown = RenderWarning::LegendValuesUnknown {
            values: vec!["zulu".into(), "yankee".into()],
        };
        let unknown_text = format!("{unknown}");
        assert!(unknown_text.contains("zulu, yankee"), "{unknown_text}");
        assert!(unknown_text.contains("match no legend entry"), "{unknown_text}");
        assert!(!unknown_text.contains("LegendValuesUnknown"), "{unknown_text}");

        // F-L08-07 (spec §4 H5), rolled up per #240-3: names the axis and the
        // count of suppressed ANNOTATION ELEMENTS (the internal per-primitive
        // spec, not the user-facing annotate_*() call) — the guide's promise
        // ("names the annotation and its position") narrows to "names the
        // axis and how many elements" once the rollup replaces per-instance
        // value/gap detail. "element(s)", not "annotation(s)", because this
        // layer has no attribution back to which user call produced which
        // element (a single `annotate_hline()` desugars to 2 elements).
        // Four kinds with a repeat (`line`/`text` twice each — the exact
        // shape 2 in-gap `annotate_hline`s produce), so this pin exercises
        // BOTH halves of the contract: the leading count stays the true
        // element count (4), and the parenthetical dedups to first-appearance
        // order (quality-review findings: `kinds` was collected and never
        // displayed at all, then — once displayed — repeated per primitive
        // instead of deduping; this must be able to catch either regressing).
        let suppressed = RenderWarning::AnnotationSuppressedByBreak {
            axis: AlignAxis::Y,
            kinds: vec!["line".into(), "text".into(), "line".into(), "text".into()],
        };
        let suppressed_text = format!("{suppressed}");
        assert!(suppressed_text.contains("4 annotation elements"), "{suppressed_text}");
        assert!(suppressed_text.contains("(line, text)"), "{suppressed_text}");
        assert!(
            !suppressed_text.contains("(line, text, line, text)"),
            "kinds must be deduped in Display, not repeated per primitive: {suppressed_text}"
        );
        assert!(suppressed_text.contains("y axis"), "{suppressed_text}");
        assert!(suppressed_text.contains("suppressed"), "{suppressed_text}");
        assert!(
            !suppressed_text.contains("AnnotationSuppressedByBreak"),
            "{suppressed_text}"
        );

        // The sibling mechanism (an extent-shaped annotation spanning a gap,
        // rather than a point landing inside one) gets the same "element(s)"
        // noun and the same genuine, deduped `kinds` display for the same
        // reason — see `AnnotationSuppressedByBreak`'s doc.
        let spans = RenderWarning::AnnotationSpansBreakGap {
            axis: AlignAxis::Y,
            kinds: vec!["rect".into(), "span".into(), "rect".into()],
        };
        let spans_text = format!("{spans}");
        assert!(spans_text.contains("3 annotation elements"), "{spans_text}");
        assert!(spans_text.contains("(rect, span)"), "{spans_text}");
        assert!(
            !spans_text.contains("(rect, span, rect)"),
            "kinds must be deduped in Display, not repeated per primitive: {spans_text}"
        );
        assert!(spans_text.contains("y axis"), "{spans_text}");
        assert!(spans_text.contains("suppressed"), "{spans_text}");
        assert!(!spans_text.contains("AnnotationSpansBreakGap"), "{spans_text}");

        // #240-4b (routed rider): an unrecognized scheme/cmap name.
        let unknown_scheme = RenderWarning::UnknownColorScheme { name: "not-a-scheme".into() };
        let unknown_scheme_text = format!("{unknown_scheme}");
        assert!(unknown_scheme_text.contains("not-a-scheme"), "{unknown_scheme_text}");
        assert!(unknown_scheme_text.contains("not recognized"), "{unknown_scheme_text}");
        assert!(
            !unknown_scheme_text.contains("UnknownColorScheme"),
            "{unknown_scheme_text}"
        );

        // Wave-1 close remediation (#240-4b's last silent arm): a <2-stop
        // gradient falls back to the theme default instead of degrading with
        // no diagnostic.
        let too_few_stops = RenderWarning::GradientTooFewStops { count: 1 };
        let too_few_stops_text = format!("{too_few_stops}");
        assert!(too_few_stops_text.contains('1'), "{too_few_stops_text}");
        assert!(
            too_few_stops_text.contains("theme's default scheme was used instead"),
            "{too_few_stops_text}"
        );
        assert!(
            !too_few_stops_text.contains("GradientTooFewStops"),
            "{too_few_stops_text}"
        );
    }

    /// `#229`(2): retyping the break axis from `String` to [`AlignAxis`] left
    /// every break-family message byte-for-byte what it was.
    ///
    /// The whole sentence is pinned, not a substring, because the risk this
    /// guards is a formatting change that still contains the substrings: the
    /// six `{axis}` arms would print `Y` (the variant) rather than `y` (the
    /// token) without [`AlignAxis`]'s `Display`, and
    /// `BreakAxisIncompatibleCoord` would print `axis=Y` rather than
    /// `axis="y"` if its `{:?}` were applied to the enum instead of to
    /// `wire_name()`. The quoted `axis="y"` is the delivered text
    /// `Debug`-on-a-`String` produced before the retype; `tests/
    /// test_break_axis_geometry.py::test_polar_coord_with_break_axis_on_x_names_that_axis`
    /// pins the same quoting from Python for the x token.
    #[test]
    fn break_family_messages_are_byte_identical_after_the_axis_retype() {
        assert_eq!(
            RenderError::BreakAxisIncompatibleCoord { coord: "Polar", axis: AlignAxis::Y }
                .to_string(),
            "BreakAxis(axis=\"y\") is not supported under Polar coordinates; \
             remove the break or render this chart under Cartesian coords"
        );
        assert_eq!(
            RenderError::BreakAxisIncompatibleCoord { coord: "Geo", axis: AlignAxis::X }
                .to_string(),
            "BreakAxis(axis=\"x\") is not supported under Geo coordinates; \
             remove the break or render this chart under Cartesian coords"
        );

        assert_eq!(
            RenderWarning::AnnotationSuppressedByBreak {
                axis: AlignAxis::Y,
                kinds: vec!["line".into(), "text".into()],
            }
            .to_string(),
            "2 annotation elements (line, text) on the y axis fell inside a \
             BreakAxis gap and were suppressed"
        );
        assert_eq!(
            RenderWarning::AnnotationSpansBreakGap {
                axis: AlignAxis::X,
                kinds: vec!["bracket".into()],
            }
            .to_string(),
            "1 annotation element (bracket) on the x axis spans a BreakAxis gap \
             and was suppressed"
        );
        assert_eq!(
            RenderWarning::BreakAxisIneffective {
                axis: AlignAxis::Y,
                reason: "no gap overlaps the axis domain (10, 950)".into(),
            }
            .to_string(),
            "BreakAxis on the y axis had no effect: no gap overlaps the axis \
             domain (10, 950)"
        );
        assert_eq!(
            RenderWarning::BreakAxisOnNonlinearScale {
                axis: AlignAxis::X,
                scale_kind: "log".into(),
            }
            .to_string(),
            "BreakAxis on the x axis targets a log scale; break placement is \
             computed with a linear pixel-to-data map, so the removed range and \
             the indicator position can differ from the gap you asked for, and \
             annotation suppression (which reads the data value directly) can \
             disagree with where marks are placed"
        );
        assert_eq!(
            RenderWarning::BreakAxisMergeConflict {
                axis: AlignAxis::Y,
                field: "break_style".into(),
            }
            .to_string(),
            "multiple BreakAxis specs on the y axis disagree on break_style; \
             the first declaration's value was kept"
        );
    }

    /// #126/#133: the two new `RenderError` variants follow the same Display
    /// contract as `RenderWarning` — a human sentence, never the derived
    /// Debug shape, and (for the time-format variant) the same `coord_flipped`
    /// un-flip discipline `InvalidAxisOrient` already pins.
    #[test]
    fn render_error_display_is_intentional_not_debug() {
        let parse_fail = RenderError::ColorRangeParseFailure { entry: "#zzz".into() };
        let text = format!("{parse_fail}");
        assert!(text.contains("could not parse color '#zzz'"), "{text}");
        assert!(!text.contains("ColorRangeParseFailure"), "Display leaked Debug: {text}");

        let not_flipped = RenderError::TimeFormatOnNonTemporalScale {
            channel: "y",
            surface: FormatSurface::Axis,
            format: Some("%b %Y".into()),
            coord_flipped: false,
        };
        let not_flipped_text = format!("{not_flipped}");
        assert!(not_flipped_text.contains("'y' axis"), "{not_flipped_text}");
        assert!(not_flipped_text.contains("%b %Y"), "{not_flipped_text}");
        assert!(not_flipped_text.contains("requires a temporal scale"), "{not_flipped_text}");
        assert!(not_flipped_text.contains(r#"typed "time""#), "{not_flipped_text}");
        assert!(
            !not_flipped_text.contains("TimeFormatOnNonTemporalScale"),
            "Display leaked Debug: {not_flipped_text}"
        );

        // R3: under `coord_flipped`, the RESOLVED "y" reads back as the "x"
        // axis the user actually wrote.
        let flipped = RenderError::TimeFormatOnNonTemporalScale {
            channel: "y",
            surface: FormatSurface::Axis,
            format: Some("%b %Y".into()),
            coord_flipped: true,
        };
        assert!(format!("{flipped}").contains("'x' axis"));

        // `#147` close (wave 6A T21): the field-checked surfaces name
        // themselves and what was checked; a bare `format_type="time"` has
        // no spec to quote, so the message names the type instead.
        let legend = RenderError::TimeFormatOnNonTemporalScale {
            channel: "color",
            surface: FormatSurface::Legend,
            format: Some(".2f".into()),
            coord_flipped: false,
        };
        assert_eq!(
            format!("{legend}"),
            r#"format ".2f" is typed "time", which on the 'color' legend requires a temporal field, but the field is not temporal"#
        );
        let text = RenderError::TimeFormatOnNonTemporalScale {
            channel: "text",
            surface: FormatSurface::Channel,
            format: None,
            coord_flipped: false,
        };
        assert_eq!(
            format!("{text}"),
            r#"format type "time" on the 'text' channel requires a temporal field, but the field is not temporal"#
        );

        // `#147` T23: a d3 number format on a temporal value, and R3's
        // un-flip on the axis surface.
        let number = RenderError::NumberFormatOnTemporalScale {
            channel: "y",
            surface: FormatSurface::Axis,
            format: ".2f".into(),
            coord_flipped: true,
        };
        assert_eq!(
            format!("{}", with_coord_flipped(number, true)),
            r#"the number format ".2f" on the 'x' axis requires a non-temporal scale, but the resolved scale is temporal; a date pattern (e.g. "%Y") formats a temporal value"#
        );
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

    /// [`RenderError::MarkSizeNotDrawn`] names the call in the entry
    /// position and the mark in the mark position. A `mark_*()` door's own
    /// refusal (no `InLayer` wrap, entry falls back to `mark_{mark}` at
    /// dispatch) stays byte-identical because entry and mark position then
    /// name the same mark; `fm.Layer`'s refusal (wrapped in `InLayer` with
    /// `entry = "Layer"`) must still name the actual mark rather than
    /// repeating "Layer" — the regression this test pins (entry doubled into
    /// both slots, dropping the mark name entirely).
    #[test]
    fn mark_size_not_drawn_names_the_call_and_the_mark_separately() {
        use crate::spec::mark::Mark;

        let direct = RenderError::MarkSizeNotDrawn { mark: Mark::Area, value: 5.0 };
        assert_eq!(
            format!("{direct}"),
            "mark_area(size=...) has no effect on mark_area; received 5."
        );

        let via_layer = RenderError::MarkSizeNotDrawn { mark: Mark::Area, value: 5.0 }
            .in_layer_context(None, "Layer");
        assert_eq!(
            format!("{via_layer}"),
            "Layer(size=...) has no effect on mark_area; received 5."
        );

        let via_layer_with_knob = RenderError::MarkSizeNotDrawn { mark: Mark::Text, value: 5.0 }
            .in_layer_context(None, "Layer");
        assert_eq!(
            format!("{via_layer_with_knob}"),
            "Layer(size=...) has no effect on mark_text; received 5. Set font_size= instead."
        );
    }

    /// [`RenderError::size_rule_reason`] is the clause the full refusal
    /// places after `{entry}(size=...) `, for both of the size rule's
    /// variants, so the full message and the reason exposed to Python
    /// (`mark_size_reason`) cannot drift apart: this pins the reason's exact
    /// text and that the full message contains it verbatim.
    #[test]
    fn size_rule_reason_is_contained_in_the_full_refusal_for_both_variants() {
        use crate::spec::mark::Mark;

        let not_drawn = RenderError::MarkSizeNotDrawn { mark: Mark::Area, value: 5.0 };
        let reason = not_drawn.size_rule_reason().expect("MarkSizeNotDrawn has a reason");
        assert_eq!(reason, "has no effect on mark_area; received 5");
        let full = format!("{}", not_drawn.in_layer_context(None, "relplot"));
        assert_eq!(full, format!("relplot(size=...) {reason}."));
        assert!(full.contains(&reason), "full: {full}, reason: {reason}");

        let not_drawn_with_knob = RenderError::MarkSizeNotDrawn { mark: Mark::Text, value: 5.0 };
        let reason =
            not_drawn_with_knob.size_rule_reason().expect("MarkSizeNotDrawn has a reason");
        assert_eq!(reason, "has no effect on mark_text; received 5. Set font_size= instead");
        let full = format!("{}", not_drawn_with_knob.in_layer_context(None, "relplot"));
        assert_eq!(full, format!("relplot(size=...) {reason}."));
        assert!(full.contains(&reason), "full: {full}, reason: {reason}");

        let out_of_range = RenderError::MarkSizeOutOfRange { mark: Mark::Point, value: -1.0 };
        let reason = out_of_range.size_rule_reason().expect("MarkSizeOutOfRange has a reason");
        assert_eq!(reason, "must be a positive number; received -1");
        let full = format!("{}", out_of_range.in_layer_context(None, "relplot"));
        assert_eq!(full, format!("relplot(size=...) {reason}"));
        assert!(full.contains(&reason), "full: {full}, reason: {reason}");

        // Vacuity control: a variant the size rule never raises has no reason.
        assert_eq!(RenderError::UnknownColumn { name: "x".into() }.size_rule_reason(), None);
    }
}

// ---------------------------------------------------------------------------
// Task 20 — render_svg full pipeline orchestration (spec §6).
// ---------------------------------------------------------------------------

use crate::layout::{compute_layout, CompositeLayoutSeam, LegendSuppression, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;
use chart_config::ChartConfig;

/// Output of the shared prepare-and-layout pipeline, consumed by both
/// `render_svg` and `render_scene_json`.
struct PipelineOutput {
    prep: prepare::PreparedInputs,
    layout: crate::layout::LayoutResult,
    warnings: Vec<RenderWarning>,
    /// The config pipeline's own product, carried WHOLE rather than unpacked
    /// into three sibling fields here: `effective_theme` (passes 13–15) plus the
    /// legend overrides and title (passes 17, 18, 16). `compute_layout` consumes
    /// all three below, and `composite_render`'s figure-legend seam
    /// (`capture_leaf_bundle`) needs the same values for the same leaf — it used
    /// to recompute the legend pair from this very `prep`. Keeping the bundle
    /// intact means `config_apply` owns its shape in one place: adding a field
    /// there does not require a matching field and a copy line here
    /// (#143 remediation, design review recs 4 and 5).
    applied: config_apply::AppliedChartConfig,
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
    let mut prep = prepare::prepare_render_inputs(spec, batch, theme, viewport, chart_config, leaf_scales)?;
    let mut warnings = prep.warnings.clone();

    // Every chart-level `configure_*()` override, applied in the one order
    // `config_apply`'s module doc states — legend suppression, axis-slot fill
    // (most specific first), tick re-derivation, color config + legend-entry
    // rebuild, effective theme, legend-overrides projection.
    let applied =
        config_apply::apply_chart_config_pipeline(&mut prep, theme, chart_config, &mut warnings)?;

    // An independent-x/y facet panel draws its own axis, from its own scale:
    // resolve those axis inputs now, so layout reserves each axis band for
    // what the panels draw.
    prep.axes.panels = scene_build::independent_panel_axis_inputs(
        &scene_build::PanelResolveCtx {
            spec,
            prep: &prep,
            theme: &applied.effective_theme,
            chart_config,
            leaf_scales,
        },
        &prep.facet_groups,
    )?;

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
        &applied.effective_theme,
        viewport,
        &prep.axes,
        &prep.facet_groups,
        &prep.legend_entries,
        applied.legend_title.clone(),
        prep.colorbar.as_ref(),
        &metrics,
        &applied.legend_overrides,
        &prep.aux_legends,
        seam,
    )
    .map_err(|e| RenderError::LayoutFailed(e.to_string()))?;
    for w in &layout.warnings {
        warnings.push(RenderWarning::Layout(w.clone()));
    }

    Ok(PipelineOutput { prep, layout, warnings, applied })
}

pub fn render_svg(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<RenderOutput<String>, RenderError> {
    if !viewport.is_drawable() {
        return Err(RenderError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }

    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };

    let PipelineOutput { prep, layout, applied, mut warnings } =
        prepare_and_layout(spec, batch, theme, viewport, chart_config, None)?;

    let scene = scene_build::build_scene(
        spec, &prep, &layout, &applied.effective_theme, config, &mut warnings, chart_config, None,
    )?;
    let svg_string = svg_walk::walk_svg(&scene, config.embed_fonts)?;

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
    if !viewport.is_drawable() {
        return Err(RenderError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }

    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };

    let PipelineOutput { prep, layout, applied, mut warnings } =
        prepare_and_layout(spec, batch, theme, viewport, chart_config, None)?;

    let mut scene = scene_build::build_scene(
        spec, &prep, &layout, &applied.effective_theme, config, &mut warnings, chart_config, None,
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
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use chart_config::{AxisConfigSpec, AxisStyleSpec, LegendStyleSpec};
    use std::sync::Arc;

    fn scatter_3() -> (ChartSpec, RecordBatch) {
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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

    /// F-L04-06 (batch-C task 3): `utc=True` and `utc=False` render
    /// byte-identical SVG through the FULL `render_svg` pipeline — not just
    /// the resolved scale in isolation — pinning the "UTC by contract"
    /// promise in `TimeScale`'s struct doc and `format_time`'s doc. Both
    /// renders use an explicit domain so the x column's dtype (plain
    /// `Float64` from `scatter_3`) never needs to matter — only the
    /// `ScaleSpec::Time` vs `ScaleSpec::Utc` tag differs between the two
    /// resolves.
    #[test]
    fn time_scale_utc_true_and_false_render_byte_identical_svg() {
        use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};

        let (mut spec, batch) = scatter_3();
        let common = ContinuousScaleCommon {
            domain: Some(vec![0.0, 3.0]),
            range: None,
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse: false,
        };
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();

        spec.encoding.x.as_mut().unwrap().scale =
            Some(ScaleSpec::Time { common: common.clone(), nice: false });
        let time_svg = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap()
            .bytes;

        spec.encoding.x.as_mut().unwrap().scale = Some(ScaleSpec::Utc { common, nice: false });
        let utc_svg = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap()
            .bytes;

        assert_eq!(
            time_svg, utc_svg,
            "utc=True/False must render byte-identical SVG by contract (F-L04-06)"
        );
    }

    // ── F-L04-03 / GH #67: one centers path for categorical axes ────────────

    /// Four categories with a quantitative companion, for the band-axis rows.
    fn four_categories() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )
        .unwrap()
    }

    /// A bar chart of [`four_categories`] whose x scale is `band`.
    fn band_bar_spec(band: Option<crate::spec::encoding::ScaleSpec>) -> ChartSpec {
        ChartSpec {
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "cat".into(),
                    type_: Some(crate::spec::encoding::DataType::Nominal),
                    scale: band,
                    ..Default::default()
                }),
                y: Some(EncodingSpec { field: "val".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Every `Text` node under `nodes`, flattened through `Group`s.
    fn text_nodes(nodes: &[ferrum_scene::SceneNode]) -> Vec<(f64, String)> {
        let mut out = Vec::new();
        for n in nodes {
            match n {
                ferrum_scene::SceneNode::Text { x, content, .. } => {
                    out.push((*x, content.clone()))
                }
                ferrum_scene::SceneNode::Group { children, .. } => {
                    out.extend(text_nodes(children))
                }
                _ => {}
            }
        }
        out
    }

    /// Run the production orchestration for `spec` over `batch` —
    /// `prepare_and_layout` then `build_scene`, so anything that aborts the
    /// pipeline surfaces here rather than only at a user's `to_svg()`.
    fn scene_and_layout_over(
        spec: &ChartSpec,
        chart_config: &ChartConfig,
        batch: &RecordBatch,
    ) -> (ferrum_scene::SceneGraph, crate::layout::LayoutResult) {
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let PipelineOutput { prep, layout, applied, mut warnings } =
            prepare_and_layout(spec, batch, &theme, viewport, chart_config, None)
                .expect("fixture must prepare");
        let scene = scene_build::build_scene(
            spec,
            &prep,
            &layout,
            &applied.effective_theme,
            &config,
            &mut warnings,
            chart_config,
            None,
        )
        .expect("fixture must build a scene");
        (scene, layout)
    }

    /// [`scene_and_layout_over`] over [`four_categories`] — the fixture every
    /// row in this section but the equal-length explicit-`range=` pins uses.
    fn scene_and_layout(
        spec: &ChartSpec,
        chart_config: &ChartConfig,
    ) -> (ferrum_scene::SceneGraph, crate::layout::LayoutResult) {
        scene_and_layout_over(spec, chart_config, &four_categories())
    }

    /// Panel 0's rendered ticks on `orient`, as `(position, label)` in tick
    /// order — the placement `build_scene` draws its labels and grid lines at.
    fn panel_ticks(
        layout: &crate::layout::LayoutResult,
        orient: crate::layout::AxisOrient,
    ) -> Vec<(f64, String)> {
        let axis = layout
            .axes
            .iter()
            .find(|a| a.panel_index == 0 && a.orient == orient)
            .expect("panel 0 must carry this axis");
        axis.ticks.iter().map(|t| (t.position, t.label.clone())).collect()
    }

    /// Build the scene for `spec` over `four_categories`, and return
    /// `(bar centers, x-axis tick-label pixels)` — both in category order.
    fn bar_centers_and_tick_pixels(spec: &ChartSpec) -> (Vec<f64>, Vec<f64>) {
        let (scene, _) = scene_and_layout(spec, &ChartConfig::default());
        let panel = &scene.panels[0];
        let mut centers: Vec<f64> = panel
            .marks
            .iter()
            .flat_map(|b| b.nodes.iter())
            .filter_map(|n| match n {
                ferrum_scene::SceneNode::Rect { x, w, .. } => Some(x + w / 2.0),
                _ => None,
            })
            .collect();
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let labels: Vec<f64> = text_nodes(&panel.axes)
            .into_iter()
            .filter(|(_, content)| ["a", "b", "c", "d"].contains(&content.as_str()))
            .map(|(x, _)| x)
            .collect();

        assert_eq!(centers.len(), 4, "expected four bars");
        assert_eq!(labels.len(), 4, "expected four x tick labels");
        (centers, labels)
    }

    /// THE observable F-L04-03 owed: a **padded, no-`range=`** `BandScale`'s
    /// axis tick labels land exactly on its bar centers.
    ///
    /// This is the case the old centers gate could not serve. The marks came
    /// from the d3 band model resolved over the real panel; the labels came
    /// from layout's own `(i + 0.5)·slot`, because the scale's centers were
    /// offered to the axis only when the user had supplied an explicit `range=`
    /// — and the panel-extent scale reaching the axis pass carries a `[0, 1]`
    /// placeholder range, so absolute pixels were not available to offer. The
    /// two models disagreed by ~1.6–4.8px per label at these parameters
    /// (`tests/test_bug_hunt_band_point_range.py` pins the same repro from the
    /// Python side).
    ///
    /// The tick labels are asserted against the **hand-computed d3 model** with
    /// `assert_eq!` (the axis resolves that model over the same
    /// `(plot_area.x, plot_area.x + plot_area.w)` interval
    /// `resolve_panel_scales` gives the mark scale, so it is bit-exact or the
    /// paths have forked); the bars are compared with a 1e-9 tolerance, because
    /// a bar stores `center − w/2` and `w`, so recovering its center costs two
    /// roundings. The divergence this row exists to catch is ~1.6–4.8px.
    #[test]
    fn padded_no_range_band_axis_labels_land_on_bar_centers() {
        let spec = band_bar_spec(Some(padded_band()));
        let (centers, labels) = bar_centers_and_tick_pixels(&spec);

        // d3, by hand: `padding = 0.1` sets both inner and outer, so
        // denom = 4 − 0.1 + 0.2 = 4.1, step = w/4.1, start = x + 0.1·step, and
        // each center is the middle of its 0.9-of-a-step drawn band.
        let (x, w) = scene_plot_area(&spec);
        let step = w / 4.1;
        let expected: Vec<f64> = (0..4)
            .map(|i| x + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step)
            .collect();
        assert_eq!(labels, expected, "tick labels must be the d3 band centers");
        for (bar, label) in centers.iter().zip(&labels) {
            assert!(
                (bar - label).abs() < 1e-9,
                "bar center {bar} does not sit under its tick label {label}"
            );
        }

        // Discriminating: the padded centers are NOT the uniform slots the old
        // fallback produced, so the row cannot pass by both paths agreeing on
        // the unpadded model.
        let uniform: Vec<f64> = (0..4).map(|i| x + (i as f64 + 0.5) * (w / 4.0)).collect();
        assert_ne!(labels, uniform, "padding must move the centers off the uniform slots");
    }

    /// The unpadded (auto-inferred) categorical axis takes the same one path
    /// and still lands on the uniform slots — the local half of the
    /// byte-identity gate for tick placement, and why no golden moved when the
    /// gate collapsed.
    #[test]
    fn unpadded_categorical_axis_labels_still_land_on_the_uniform_slots() {
        let spec = band_bar_spec(None);
        let (centers, labels) = bar_centers_and_tick_pixels(&spec);
        for (bar, label) in centers.iter().zip(&labels) {
            assert!(
                (bar - label).abs() < 1e-9,
                "bar center {bar} does not sit under its tick label {label}"
            );
        }

        let (x, w) = scene_plot_area(&spec);
        let uniform: Vec<f64> = (0..4).map(|i| x + (i as f64 + 0.5) * (w / 4.0)).collect();
        assert_eq!(labels, uniform, "the unpadded model must reduce to the uniform slots");
    }

    /// This spec's panel-0 plot area as `(x, w)`, for the rows above to compute
    /// the uniform-slot comparison against the layout's real rect rather than a
    /// hardcoded guess.
    fn scene_plot_area(spec: &ChartSpec) -> (f64, f64) {
        let (_, layout) = scene_and_layout(spec, &ChartConfig::default());
        let plot = layout.panels[0].plot_area;
        (plot.x, plot.w)
    }

    /// A horizontal bar chart of [`four_categories`]: the transpose of
    /// [`band_bar_spec`], with the band on **y** and the quantitative field
    /// on x.
    fn band_bar_spec_y(band: Option<crate::spec::encoding::ScaleSpec>) -> ChartSpec {
        let mut spec = band_bar_spec(None);
        spec.encoding.x = Some(EncodingSpec { field: "val".into(), ..Default::default() });
        spec.encoding.y = Some(EncodingSpec {
            field: "cat".into(),
            type_: Some(crate::spec::encoding::DataType::Nominal),
            scale: band,
            ..Default::default()
        });
        spec
    }

    /// A `BandScale(padding=0.1)` with the four categories as its domain and no
    /// `range=` — the panel-extent case F-L04-03 exists for.
    fn padded_band() -> crate::spec::encoding::ScaleSpec {
        crate::spec::encoding::ScaleSpec::Band {
            domain: Some(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
            padding: crate::scale::discrete::BandPadding::new(Some(0.1), None, None)
                .expect("in-bounds band padding"),
            align: 0.5,
            range: None,
        }
    }

    /// The **y transpose** of [`padded_no_range_band_axis_labels_land_on_bar_centers`].
    ///
    /// The x row alone cannot catch a lo/hi swap in the y placement: an
    /// unpadded model is symmetric about the panel's midpoint, so an inverted
    /// resolution reproduces the same positions. Padded, it does not — the
    /// outer padding sits at one end, and the whole ladder of centers would
    /// mirror. `layout::axis::tests::y_axis_resolves_panel_extent_placement_against_this_panel`
    /// pins `layout_y_axis` in isolation; this row pins the resolver → layout →
    /// scene agreement, which is where an interval handed over backwards would
    /// show up.
    #[test]
    fn padded_no_range_band_y_axis_labels_land_on_bar_centers() {
        let spec = band_bar_spec_y(Some(padded_band()));
        let (scene, layout) = scene_and_layout(&spec, &ChartConfig::default());
        let plot = layout.panels[0].plot_area;

        // The same d3 model as the x row, over the panel's vertical interval:
        // denom = 4 − 0.1 + 0.2 = 4.1, top → bottom in category order.
        let step = plot.h / 4.1;
        let expected: Vec<f64> = (0..4)
            .map(|i| plot.y + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step)
            .collect();
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Left);
        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();
        assert_eq!(labels, ["a", "b", "c", "d"], "ordinal y is not reversed");
        assert_eq!(positions, expected, "y tick labels must be the d3 band centers");

        let mut centers: Vec<f64> = scene.panels[0]
            .marks
            .iter()
            .flat_map(|b| b.nodes.iter())
            .filter_map(|n| match n {
                ferrum_scene::SceneNode::Rect { y, h, .. } => Some(y + h / 2.0),
                _ => None,
            })
            .collect();
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(centers.len(), 4, "expected four bars");
        for (bar, label) in centers.iter().zip(&positions) {
            assert!(
                (bar - label).abs() < 1e-9,
                "bar center {bar} does not sit beside its tick label {label}"
            );
        }

        // Discriminating, exactly as on x: padded centers are NOT the uniform
        // slots, so a mirrored or slot-based resolution cannot pass this row.
        let uniform: Vec<f64> = (0..4).map(|i| plot.y + (i as f64 + 0.5) * (plot.h / 4.0)).collect();
        assert_ne!(positions, uniform, "padding must move the centers off the uniform slots");
    }

    // ── tick_values over a categorical axis (batch-C T7 review, finding 1) ───
    // An explicit tick-value list replaces `tick_labels` wholesale, at any
    // length, while the categorical placement carrier is sized by the scale's
    // DOMAIN. Layout indexes that carrier by tick position, so before
    // `config_apply::sync_tick_placement_to_tick_values` cleared it these three
    // spellings aborted the render with an index-out-of-bounds panic — raised
    // across the PyO3 boundary as a `PanicException`, not a typed
    // `RenderError`. Each row asserts the *placement* that survives (the
    // uniform slots, i.e. the pre-F-L04-03 outcome for an overridden
    // categorical axis), not merely that nothing panicked, and each carries the
    // override's own labels so a row cannot pass by the override being dropped.

    /// The uniform-slot center of tick `i` of `n` over `[lo, lo + extent]`.
    fn uniform_slots(lo: f64, extent: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| lo + (i as f64 + 0.5) * (extent / n as f64)).collect()
    }

    /// `configure_axis(tick_values=[…])` with MORE values than the domain has
    /// categories — the spelling and shape from the review's repro.
    #[test]
    fn categorical_x_axis_under_a_longer_tick_values_override_keeps_uniform_slots() {
        let spec = band_bar_spec(Some(padded_band()));
        let chart_config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec {
                    values: Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        let (_, layout) = scene_and_layout(&spec, &chart_config);
        let plot = layout.panels[0].plot_area;
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Bottom);

        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(labels, ["0", "1", "2", "3", "4", "5"], "the override must reach the axis");
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();
        assert_eq!(
            positions,
            uniform_slots(plot.x, plot.w, 6),
            "six overridden labels take the six uniform slots — there are only four band centers"
        );
    }

    /// Per-channel `fm.Axis(values=[…])` with FEWER values than the domain has
    /// categories. The other half of the length mismatch: a short carrier
    /// cannot be caught by an index guard alone, so the pin is the placement.
    #[test]
    fn categorical_x_axis_under_a_shorter_tick_values_override_keeps_uniform_slots() {
        let mut spec = band_bar_spec(Some(padded_band()));
        spec.encoding.x.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            values: Some(vec![10.0, 20.0]),
            ..Default::default()
        }));
        let (_, layout) = scene_and_layout(&spec, &ChartConfig::default());
        let plot = layout.panels[0].plot_area;
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Bottom);

        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(labels, ["10", "20"], "the override must reach the axis");
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();
        assert_eq!(
            positions,
            uniform_slots(plot.x, plot.w, 2),
            "two overridden labels take the two uniform slots, not the first two band centers"
        );
    }

    /// The y transpose of both rows above, on a categorical y axis (the second
    /// live repro the review reported, at `layout/axis.rs`'s y-side index).
    #[test]
    fn categorical_y_axis_under_a_tick_values_override_keeps_uniform_slots() {
        let ticks_for = |values: Vec<f64>| {
            let mut spec = band_bar_spec_y(Some(padded_band()));
            spec.encoding.y.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
                values: Some(values),
                ..Default::default()
            }));
            let (_, layout) = scene_and_layout(&spec, &ChartConfig::default());
            let plot = layout.panels[0].plot_area;
            (plot, panel_ticks(&layout, crate::layout::AxisOrient::Left))
        };

        let (plot, longer) = ticks_for(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            longer.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
            ["0", "1", "2", "3", "4", "5"]
        );
        assert_eq!(
            longer.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            uniform_slots(plot.y, plot.h, 6),
            "six overridden y labels take the six uniform slots"
        );

        let (plot, shorter) = ticks_for(vec![10.0, 20.0]);
        assert_eq!(
            shorter.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
            ["10", "20"]
        );
        assert_eq!(
            shorter.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            uniform_slots(plot.y, plot.h, 2),
            "two overridden y labels take the two uniform slots"
        );
    }

    // ── equal-length tick_values on a categorical axis (batch-C T7 review,
    //    round 3 — the equal-length finding) ──────────────────────────────
    // When the override supplies exactly one label per category, the pairing
    // between tick position and category still holds, so
    // `sync_tick_placement_to_tick_values` keeps the placement carrier instead
    // of clearing it: the relabeled ticks land on the aligned centers, not the
    // uniform slots. Mismatched-length overrides (above) are unaffected.

    /// Three categories with a quantitative companion — the spec reviewer's
    /// exact repro shape (`BandScale(range=[50, 400])`, 3 categories).
    fn three_categories() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap()
    }

    /// An explicit-`range=` `BandScale` over `domain`, at
    /// `padding=0.1` and the default align — the shape the spec reviewer measured.
    fn explicit_range_band(
        domain: Vec<&str>,
        range: [f64; 2],
    ) -> crate::spec::encoding::ScaleSpec {
        crate::spec::encoding::ScaleSpec::Band {
            domain: Some(domain.into_iter().map(String::from).collect()),
            padding: crate::scale::discrete::BandPadding::new(Some(0.1), None, None)
                .expect("in-bounds band padding"),
            align: 0.5,
            range: Some(vec![range[0], range[1]]),
        }
    }

    /// Equal-length `tick_values` on an explicit-`range=` band restores the
    /// pre-F-L04-03 outcome exactly, byte-for-byte with the spec reviewer's own
    /// measurement: `BandScale(range=[50, 400])` over three categories centers
    /// each band at 112.097 / 225 / 337.903. Round 2's unconditional clear
    /// moved these to uniform slots (156.453 / 343.472 / 530.491), the last of
    /// which sits OUTSIDE the declared `[50, 400]` range — this pin is what
    /// catches a regression back to that behavior.
    #[test]
    fn categorical_x_axis_under_an_equal_length_tick_values_override_lands_on_explicit_range_band_centers()
    {
        let mut spec = band_bar_spec(Some(explicit_range_band(vec!["a", "b", "c"], [50.0, 400.0])));
        spec.encoding.x.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            values: Some(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        }));
        let (_, layout) = scene_and_layout_over(&spec, &ChartConfig::default(), &three_categories());
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Bottom);

        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(labels, ["1", "2", "3"], "the override's own labels must reach the axis");
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();

        // d3, by hand: denom = 3 − 0.1 + 0.2 = 3.1, step = 350/3.1, over
        // [50, 400] directly (an explicit range is used verbatim, not
        // re-derived from the panel — GH #39).
        let step = 350.0_f64 / 3.1;
        let expected: Vec<f64> =
            (0..3).map(|i| 50.0 + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step).collect();
        assert_eq!(positions, expected, "an equal-length relabel must land on the explicit-range band centers");
        for (got, want) in positions.iter().zip([112.097, 225.0, 337.903]) {
            assert!((got - want).abs() < 1e-2, "expected band center {want}, got {got}");
        }
        assert!(
            positions.iter().all(|p| (50.0..=400.0).contains(p)),
            "band centers must stay inside the declared range: {positions:?}"
        );
    }

    /// The y transpose of the row above.
    #[test]
    fn categorical_y_axis_under_an_equal_length_tick_values_override_lands_on_explicit_range_band_centers()
    {
        let mut spec = band_bar_spec_y(Some(explicit_range_band(vec!["a", "b", "c"], [50.0, 400.0])));
        spec.encoding.y.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            values: Some(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        }));
        let (_, layout) = scene_and_layout_over(&spec, &ChartConfig::default(), &three_categories());
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Left);

        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(labels, ["1", "2", "3"], "the override's own labels must reach the axis");
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();

        let step = 350.0_f64 / 3.1;
        let expected: Vec<f64> =
            (0..3).map(|i| 50.0 + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step).collect();
        assert_eq!(positions, expected, "an equal-length relabel must land on the explicit-range band centers");
        assert!(
            positions.iter().all(|p| (50.0..=400.0).contains(p)),
            "band centers must stay inside the declared range: {positions:?}"
        );
    }

    /// Equal-length `tick_values` on a PADDED no-`range=` band lands on the
    /// same d3 centers [`padded_no_range_band_axis_labels_land_on_bar_centers`]
    /// pins for the unoverridden case — in range, and off the uniform slots.
    #[test]
    fn categorical_x_axis_under_an_equal_length_tick_values_override_lands_on_padded_band_centers() {
        let mut spec = band_bar_spec(Some(padded_band()));
        spec.encoding.x.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            values: Some(vec![10.0, 20.0, 30.0, 40.0]),
            ..Default::default()
        }));
        let (_, layout) = scene_and_layout(&spec, &ChartConfig::default());
        let plot = layout.panels[0].plot_area;
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Bottom);

        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(labels, ["10", "20", "30", "40"], "the override's own labels must reach the axis");
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();

        let step = plot.w / 4.1;
        let expected: Vec<f64> = (0..4)
            .map(|i| plot.x + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step)
            .collect();
        assert_eq!(
            positions, expected,
            "an equal-length relabel must land on the padded band centers, not uniform slots"
        );
        assert_ne!(
            positions,
            uniform_slots(plot.x, plot.w, 4),
            "must not fall back to uniform slots when the override pairs one-to-one with the categories"
        );
        assert!(
            positions.iter().all(|p| (plot.x..=plot.x + plot.w).contains(p)),
            "band centers must stay inside the plot area: {positions:?}"
        );
    }

    /// The y transpose of the row above.
    #[test]
    fn categorical_y_axis_under_an_equal_length_tick_values_override_lands_on_padded_band_centers() {
        let mut spec = band_bar_spec_y(Some(padded_band()));
        spec.encoding.y.as_mut().unwrap().axis = Some(Box::new(AxisStyleSpec {
            values: Some(vec![10.0, 20.0, 30.0, 40.0]),
            ..Default::default()
        }));
        let (_, layout) = scene_and_layout(&spec, &ChartConfig::default());
        let plot = layout.panels[0].plot_area;
        let ticks = panel_ticks(&layout, crate::layout::AxisOrient::Left);

        let labels: Vec<String> = ticks.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(labels, ["10", "20", "30", "40"], "the override's own labels must reach the axis");
        let positions: Vec<f64> = ticks.iter().map(|(p, _)| *p).collect();

        let step = plot.h / 4.1;
        let expected: Vec<f64> = (0..4)
            .map(|i| plot.y + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step)
            .collect();
        assert_eq!(
            positions, expected,
            "an equal-length relabel must land on the padded band centers, not uniform slots"
        );
        assert_ne!(
            positions,
            uniform_slots(plot.y, plot.h, 4),
            "must not fall back to uniform slots when the override pairs one-to-one with the categories"
        );
        assert!(
            positions.iter().all(|p| (plot.y..=plot.y + plot.h).contains(p)),
            "band centers must stay inside the plot area: {positions:?}"
        );
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

    /// A NaN or infinite width or height is refused like a zero one, with
    /// `padding.auto` on (the default) and off, and the message names the
    /// value and what a viewport must be.
    #[test]
    fn render_svg_refuses_a_non_finite_viewport() {
        let (spec, batch) = scatter_3();
        for auto in [true, false] {
            let mut theme = ThemeInputs::default();
            theme.padding.padding_auto = auto;
            for (width, height) in [
                (f64::NAN, 100.0),
                (100.0, f64::NAN),
                (f64::INFINITY, 100.0),
                (100.0, f64::INFINITY),
                (f64::NEG_INFINITY, 100.0),
            ] {
                let err = render_svg(
                    &spec,
                    &batch,
                    &theme,
                    Viewport { width, height },
                    &config::RenderConfig::default(),
                    &ChartConfig::default(),
                )
                .unwrap_err();
                assert!(matches!(err, RenderError::InvalidViewport { .. }), "{width}x{height}: {err:?}");
                assert!(format!("{err}").contains("both must be finite and > 0"), "{err}");
            }
        }
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
                            reverse: false,
                        },
                        nice: false,
                    }),
                    ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: color_field.map(|f| EncodingSpec { field: f.into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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

    /// Palette overflow (spec §4.3): more style categories than the
    /// categorical scale's legacy slots (`STROKE_DASH_LEGACY_SLOTS`, 4:
    /// solid + 3 patterns) still draws every polyline — cycling the legacy
    /// range — while emitting
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
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(MarkKwargsSpec { stroke: Some("none".into()), ..Default::default() }),
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(MarkKwargsSpec {
                fill: fill.map(String::from),
                opacity,
                ..Default::default()
            }),
            ..Default::default()
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

    // ── `#110` / spec §4.1: an explicit paint clear outranks the color channel ──
    //
    // The defect these pin: `mark_line(stroke="none").encode(x=…, y=…, color=…)`
    // rendered BYTE-IDENTICALLY to the same chart without the clear, painting the
    // categorical palette over it; and with no color channel the same clear
    // serialized as `stroke="rgba(0,0,0,0.000)"` instead of omitting the
    // attribute. Both halves are asserted at the SVG-attribute level, on the
    // marks only, for every mark that carries a stroke-only paint.
    //
    // The negative control lives in `render_svg_line_explicit_zero_alpha_stroke_*`
    // below: a paint the caller set to a transparent COLOR is not a cleared paint
    // and must still serialize, which is what stops any of these from being
    // satisfiable by a value test on the resolved color.

    use crate::spec::mark_style::MarkKwargsSpec;

    /// Build a spec for `mark`, optionally with a `color` channel, carrying the
    /// given `mark_kwargs` — the wire shape of
    /// `ferrum.mark_<m>(**kwargs).encode(x=…, y=…[, x2=…, y2=…][, color="g"])`.
    fn paint_clear_spec(
        mark: Mark,
        style: MarkKwargsSpec,
        with_color: bool,
    ) -> (ChartSpec, RecordBatch) {
        let ranged = matches!(mark, Mark::Segment);
        let mut encoding = Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        if ranged {
            encoding.x2 = Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() });
            encoding.y2 = Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() });
        }
        if with_color {
            encoding.color = Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() });
        }
        let spec = ChartSpec {
            mark,
            encoding,
            mark_style: Some(style),
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![1.0, 4.0, 2.0, 5.0])),
                Arc::new(Float64Array::from(vec![2.0, 3.0, 4.0, 5.0])),
                Arc::new(Float64Array::from(vec![2.0, 5.0, 3.0, 6.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    fn render_to_svg(spec: &ChartSpec, batch: &RecordBatch) -> String {
        render_svg(
            spec,
            batch,
            &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        )
        .expect("render must succeed")
        .bytes
    }

    /// The marks-only slice of an SVG: everything inside the panel's clip group.
    ///
    /// Gridlines, axis lines and strip titles are emitted BEFORE `use_clip_open`
    /// and annotations/text batches AFTER `use_clip_close` (see `svg_walk::walk_svg`'s
    /// order), so this is exactly the mark batches — which is what lets a stroke
    /// assertion below be about the mark rather than about chrome that paints
    /// `<line>` elements of its own. The fixtures here emit no nested `<g>` inside
    /// the clip group; each caller asserts on a tag it expects to find, so a
    /// truncated slice fails rather than passing vacuously.
    fn mark_group(svg: &str) -> &str {
        let open = svg.find("<g clip-path=").expect("a panel must open a clip group");
        let rest = &svg[open..];
        let close = rest.find("</g>").expect("the clip group must close");
        &rest[..close]
    }

    /// Every stroke-only mark: a `stroke="none"` under a color channel paints no
    /// stroke, and the same chart without the clear does paint one.
    #[test]
    fn render_svg_stroke_clear_beats_a_color_channel_on_every_stroke_only_mark() {
        // (label, mark, the element tag that mark emits, `interpolate=` kwarg)
        //
        // `line` appears twice because it emits two different NODE SHAPES: the
        // default linear spelling is a `StrokeStyle`-shaped `Polyline`, and a
        // non-linear `interpolate=` is a `FillStroke`-shaped `Path`. At HEAD those
        // branches built their strokes SEPARATELY, which is how they could both
        // leak; they now read one `stroke_paint` value, but that coupling is by
        // construction rather than by type — a future re-split would be exactly
        // this defect's original shape — so both shapes are pinned rather than one
        // standing in for the other.
        let cases: [(&str, Mark, &str, Option<&str>); 5] = [
            ("line", Mark::Line, "<polyline ", None),
            ("line(interpolate=monotone)", Mark::Line, "<path ", Some("monotone")),
            ("rule", Mark::Rule, "<line ", None),
            ("tick", Mark::Tick, "<line ", None),
            ("segment", Mark::Segment, "<line ", None),
        ];
        for (label, mark, tag, interpolate) in cases {
            let painted_kwargs = MarkKwargsSpec {
                interpolate: interpolate.map(String::from),
                ..Default::default()
            };
            let cleared_kwargs = MarkKwargsSpec {
                stroke: Some("none".into()),
                interpolate: interpolate.map(String::from),
                ..Default::default()
            };
            let (spec_p, batch) = paint_clear_spec(mark.clone(), painted_kwargs, true);
            let (spec_c, _) = paint_clear_spec(mark.clone(), cleared_kwargs, true);
            let painted_svg = render_to_svg(&spec_p, &batch);
            let cleared_svg = render_to_svg(&spec_c, &batch);
            let painted = mark_group(&painted_svg);
            let cleared = mark_group(&cleared_svg);

            // Positive control: without the clear, the color channel paints.
            assert!(
                painted.contains(tag) && painted.contains("stroke=\""),
                "{label} with a color channel and no clear must paint a stroke; marks={painted:?}"
            );
            // The clear wins — the geometry is still emitted, unpainted.
            assert!(
                cleared.contains(tag),
                "{label} must still emit its {tag:?} element for a cleared stroke; marks={cleared:?}"
            );
            assert!(
                !cleared.contains("stroke=\""),
                "{label}: an explicit stroke=\"none\" must beat the color channel and emit NO \
                 stroke attribute; marks={cleared:?}"
            );
            assert_ne!(
                painted, cleared,
                "{label}: the cleared render must differ from the uncleared one — the \
                 pre-`#110` bug was that they were byte-identical"
            );
            assert!(
                !cleared_svg.contains("rgba(0,0,0,0.000)"),
                "{label}: a cleared stroke must never serialize a zero-alpha color; svg={cleared_svg:?}"
            );
        }
    }

    /// `mark_area(fill="none")` under a color channel: the area path paints no
    /// fill, where before `#110` the channel's palette color overwrote the clear.
    ///
    /// Area's cleared *stroke* is deliberately not pinned here. It is not a
    /// stroke-only mark: both of its paints travel through the one
    /// `to_scene_fill_stroke` → `push_fill_stroke` path that point/rect/bar/ribbon
    /// already pin for a cleared stroke (`bar_cleared_stroke_normalizes_to_none_not_zero_alpha`,
    /// `render_svg_ribbon_cleared_stroke_emits_no_stroke_attribute`), so area's
    /// stroke cannot serialize differently from theirs without those failing.
    /// What is area-specific is the *fill* resolution — the group-keyed chain this
    /// task rewrote — which is what this pins.
    #[test]
    fn render_svg_area_fill_clear_beats_a_color_channel() {
        let (spec_p, batch) = paint_clear_spec(Mark::Area, MarkKwargsSpec::default(), true);
        let (spec_c, _) = paint_clear_spec(
            Mark::Area,
            MarkKwargsSpec { fill: Some("none".into()), ..Default::default() },
            true,
        );
        let painted_svg = render_to_svg(&spec_p, &batch);
        let cleared_svg = render_to_svg(&spec_c, &batch);
        let painted = mark_group(&painted_svg);
        let cleared = mark_group(&cleared_svg);

        assert!(
            painted.contains("<path ") && painted.contains("fill=\"rgba("),
            "an area with a color channel and no clear must paint a palette fill; marks={painted:?}"
        );
        assert!(
            cleared.contains("<path "),
            "the area path must still be emitted for a cleared fill; marks={cleared:?}"
        );
        assert!(
            !cleared.contains("fill=\"rgba(") && cleared.contains("fill=\"none\""),
            "an explicit fill=\"none\" must beat the color channel: the path takes SVG's own \
             `none` keyword, never a palette rgba(); marks={cleared:?}"
        );
        assert!(
            !cleared_svg.contains("rgba(0,0,0,0.000)"),
            "a cleared area fill must never serialize a zero-alpha color; svg={cleared_svg:?}"
        );
    }

    /// The fill clear stops at the fill: `mark_area(fill="none")` with `borders=`
    /// or `line=` under a color channel is an outline-only area colored by group.
    /// The border paths take the channel's palette stroke while the fill path
    /// keeps `fill="none"` (T12 quality-review c3: the borders once reused the
    /// fill's resolved paint, so the clear blanked them and the chart rendered
    /// no mark ink at all).
    #[test]
    fn render_svg_area_fill_clear_keeps_palette_borders_under_a_color_channel() {
        let modes = [
            ("borders=True", MarkKwargsSpec { borders: Some(true), ..Default::default() }),
            ("line=True", MarkKwargsSpec { line: Some(true), ..Default::default() }),
        ];
        for (label, border_kwargs) in modes {
            let (spec, batch) = paint_clear_spec(
                Mark::Area,
                MarkKwargsSpec { fill: Some("none".into()), ..border_kwargs },
                true,
            );
            let svg = render_to_svg(&spec, &batch);
            let marks = mark_group(&svg);
            assert!(
                marks.contains("fill=\"none\"") && !marks.contains("fill=\"rgba("),
                "{label}: the cleared area fill must stay `none`; marks={marks:?}"
            );
            for palette in ["#2563eb", "#dc2626"] {
                assert!(
                    marks.contains(&format!("stroke=\"{palette}\"")),
                    "{label}: each group's border must paint its palette stroke {palette} \
                     over the fill clear; marks={marks:?}"
                );
            }
        }
    }

    /// Point's `cross`/`vline`/`hline` glyphs paint their lines in the FILL slot
    /// (`point.rs::emit_shape_nodes`), so `mark_point(shape="cross", fill="none")`
    /// is a cleared *fill* reaching a `StrokeStyle`. It must paint nothing.
    #[test]
    fn render_svg_point_stroke_shaped_glyph_honors_a_cleared_fill() {
        for shape in ["cross", "vline", "hline"] {
            let (spec_p, batch) = paint_clear_spec(
                Mark::Point,
                MarkKwargsSpec { shape: Some(shape.into()), ..Default::default() },
                false,
            );
            let (spec_c, _) = paint_clear_spec(
                Mark::Point,
                MarkKwargsSpec {
                    shape: Some(shape.into()),
                    fill: Some("none".into()),
                    ..Default::default()
                },
                false,
            );
            let painted_svg = render_to_svg(&spec_p, &batch);
            let cleared_svg = render_to_svg(&spec_c, &batch);
            let painted = mark_group(&painted_svg);
            let cleared = mark_group(&cleared_svg);

            assert!(
                painted.contains("<line ") && painted.contains("stroke=\""),
                "{shape}: an uncleared glyph must paint its lines; marks={painted:?}"
            );
            assert!(
                cleared.contains("<line "),
                "{shape}: the glyph lines must still be emitted; marks={cleared:?}"
            );
            assert!(
                !cleared.contains("stroke=\""),
                "{shape}: a cleared FILL must leave the glyph lines unpainted; marks={cleared:?}"
            );
            assert!(
                !cleared_svg.contains("rgba(0,0,0,0.000)"),
                "{shape}: must not serialize a zero-alpha stroke; svg={cleared_svg:?}"
            );
        }
    }

    /// The stroke-side negative control, mirroring
    /// `render_svg_point_explicit_zero_alpha_hex_serializes_by_value_not_no_attribute`
    /// on the fill side: `stroke="#00000000"` resolves the same `Color` bytes as a
    /// clear but was never cleared, so it must keep serializing
    /// `rgba(0,0,0,0.000)`. This is what makes the pins above provenance tests
    /// rather than value tests.
    #[test]
    fn render_svg_line_explicit_zero_alpha_stroke_serializes_by_value_not_no_attribute() {
        let (spec, batch) = paint_clear_spec(
            Mark::Line,
            MarkKwargsSpec { stroke: Some("#00000000".into()), ..Default::default() },
            false,
        );
        let svg = render_to_svg(&spec, &batch);
        let marks = mark_group(&svg);
        assert!(
            marks.contains("stroke=\"rgba(0,0,0,0.000)\""),
            "an explicit #00000000 stroke must serialize by value, not omit the attribute; \
             marks={marks:?}"
        );
    }

    // ── `#110` on the text family: `mark_text` / `mark_label` ──
    //
    // Text nodes paint through `TextStyle.color`, not a `FillStroke`, so the
    // rule above did not reach them until `TextStyle.color` became
    // `Option<Color>`: `mark_text(fill="none").encode(color=…)` rendered
    // byte-identically to the same chart without the clear, and a clear with no
    // channel serialized `fill="rgba(0,0,0,0.000)"` on both marks. Text batches
    // paint OUTSIDE the clip group, so these read the mark's own `<text>`
    // elements by a content marker instead of `mark_group`.

    /// [`paint_clear_spec`] plus a `text` channel whose values (`m0`…`m3`)
    /// no axis tick or legend label can collide with.
    fn text_clear_spec(mark: Mark, style: MarkKwargsSpec, with_color: bool) -> (ChartSpec, RecordBatch) {
        let (mut spec, batch) = paint_clear_spec(mark, style, with_color);
        spec.encoding.text = Some(EncodingSpec { field: "t".into(), type_: None, ..Default::default() });
        let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new("t", DataType::Utf8, false));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(StringArray::from(vec!["m0", "m1", "m2", "m3"])));
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        (spec, batch)
    }

    /// The `fill` attribute of every mark `<text>` element (content `m0`…`m3`),
    /// in document order. Asserts all four were found, so a slice that missed
    /// the mark text fails instead of passing vacuously.
    fn mark_text_fills(svg: &str) -> Vec<&str> {
        let fills: Vec<&str> = svg
            .split("<text")
            .skip(1)
            .filter(|el| {
                let body = &el[..el.find("</text>").expect("a text element must close")];
                body.split_once('>').is_some_and(|(_, content)| {
                    content.len() == 2 && content.starts_with('m')
                })
            })
            .map(|el| {
                let start = el.find("fill=\"").expect("a text element must carry fill") + 6;
                &el[start..start + el[start..].find('"').unwrap()]
            })
            .collect();
        assert_eq!(fills.len(), 4, "expected the four mark text elements; svg={svg:?}");
        fills
    }

    /// `fill="none"` on `mark_text` / `mark_label` paints no glyphs, with or
    /// without a color channel, and never serializes a zero-alpha color.
    #[test]
    fn render_svg_text_fill_clear_beats_a_color_channel_on_text_and_label() {
        for mark in [Mark::Text, Mark::Label] {
            let label = format!("{mark:?}");
            // `leader_line` only exists on label (text refuses it); it rides
            // along there so the leader stroke (painted in the fill) is
            // covered by the zero-alpha check.
            let cleared_kwargs = || MarkKwargsSpec {
                fill: Some("none".into()),
                leader_line: matches!(mark, Mark::Label).then_some(true),
                ..Default::default()
            };
            let (spec_p, batch) = text_clear_spec(mark.clone(), MarkKwargsSpec::default(), true);
            let painted_svg = render_to_svg(&spec_p, &batch);
            let painted = mark_text_fills(&painted_svg);
            assert!(
                painted.iter().all(|f| *f != "none" && !f.starts_with("rgba(0,0,0,0")),
                "{label}: without a clear the mark text must paint; fills={painted:?}"
            );
            if matches!(mark, Mark::Text) {
                // Positive control for the channel: text paints the palette,
                // so the rows of the two groups differ.
                assert_ne!(painted[0], painted[1], "text must paint the color channel; fills={painted:?}");
            }

            for with_color in [true, false] {
                let (spec_c, batch) = text_clear_spec(mark.clone(), cleared_kwargs(), with_color);
                let cleared_svg = render_to_svg(&spec_c, &batch);
                let cleared = mark_text_fills(&cleared_svg);
                assert!(
                    cleared.iter().all(|f| *f == "none"),
                    "{label} (color channel: {with_color}): fill=\"none\" must paint no glyphs, \
                     beating any color channel; fills={cleared:?}"
                );
                assert!(
                    !cleared_svg.contains("rgba(0,0,0,0.000)"),
                    "{label} (color channel: {with_color}): a cleared text paint must never \
                     serialize a zero-alpha color; svg={cleared_svg:?}"
                );
                if with_color {
                    assert_ne!(
                        painted_svg, cleared_svg,
                        "{label}: the cleared render must differ from the uncleared one"
                    );
                }
            }
        }
    }

    /// The negative control for the text family: `fill="#00000000"` resolves
    /// the same bytes as a clear but was never cleared, so both marks keep
    /// serializing `rgba(0,0,0,0.000)` on their text.
    #[test]
    fn render_svg_text_explicit_zero_alpha_fill_serializes_by_value() {
        for mark in [Mark::Text, Mark::Label] {
            let (spec, batch) = text_clear_spec(
                mark.clone(),
                MarkKwargsSpec { fill: Some("#00000000".into()), ..Default::default() },
                false,
            );
            let svg = render_to_svg(&spec, &batch);
            let fills = mark_text_fills(&svg);
            assert!(
                fills.iter().all(|f| *f == "rgba(0,0,0,0.000)"),
                "{mark:?}: an explicit #00000000 fill must serialize by value; fills={fills:?}"
            );
        }
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            ..Default::default()
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

    /// Build a two-panel faceted bar chart over `n_cats` unique categories on
    /// x, with `resolve.x` set by `x_resolve`. Dense enough on a narrow
    /// viewport that the x-label cascade must degrade, so the rendered tick
    /// count is a live readout of the label-gap policy.
    fn dense_faceted_categorical(
        n_cats: usize,
        x_resolve: crate::layout::ResolveMode,
    ) -> (ChartSpec, RecordBatch) {
        let cats: Vec<String> = (0..n_cats).map(|i| format!("Cat{i:02}")).collect();
        let mut x_vals: Vec<String> = Vec::new();
        let mut y_vals: Vec<f64> = Vec::new();
        let mut panels: Vec<String> = Vec::new();
        for panel in ["A", "B"] {
            for (i, c) in cats.iter().enumerate() {
                x_vals.push(c.clone());
                y_vals.push(i as f64 + 1.0);
                panels.push(panel.to_string());
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("panel", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(x_vals)),
                Arc::new(Float64Array::from(y_vals)),
                Arc::new(StringArray::from(panels)),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "panel".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve {
                    x: x_resolve,
                    y: crate::layout::ResolveMode::Shared,
                },
            }),
            ..Default::default()
        };
        (spec, batch)
    }

    /// Count the rendered `CatNN` x tick labels in an SVG. Every category
    /// label is unique, so this is the visible-label count, not a node count.
    fn count_cat_labels(svg: &str) -> usize {
        (0..100)
            .filter(|i| svg.contains(&format!(">Cat{i:02}<")))
            .count()
    }

    // ── independent facet axes: the bands hold what the panels draw ────────

    use crate::layout::TextMetrics as _;
    use prepare::PreparedInputs;

    /// A two-panel bar chart, faceted by `panel` with `x_resolve` and
    /// `y_resolve`, where each panel has its own `per_panel` categories
    /// (`"A-category-00"` …, `"B-category-00{b_suffix}"` …) and its own y
    /// range (panel A 1..=n, panel B around 100,000), so the chart's own axes
    /// (built over the union) differ from the axes each independent panel
    /// draws.
    fn faceted_with_own_categories(
        per_panel: usize,
        b_suffix: &str,
        x_resolve: crate::layout::ResolveMode,
        y_resolve: crate::layout::ResolveMode,
    ) -> (ChartSpec, RecordBatch) {
        let mut x_vals: Vec<String> = Vec::new();
        let mut y_vals: Vec<f64> = Vec::new();
        let mut panels: Vec<String> = Vec::new();
        for (panel, base) in [("A", 0.0), ("B", 100_000.0)] {
            for i in 0..per_panel {
                let suffix = if panel == "B" { b_suffix } else { "" };
                x_vals.push(format!("{panel}-category-{i:02}{suffix}"));
                y_vals.push(base + 1.0 + i as f64 * 0.5);
                panels.push(panel.to_string());
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("panel", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(x_vals)),
                Arc::new(Float64Array::from(y_vals)),
                Arc::new(StringArray::from(panels)),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "panel".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve { x: x_resolve, y: y_resolve },
            }),
            ..Default::default()
        };
        (spec, batch)
    }

    /// What one rendered panel's axes draw: the x tick labels as
    /// `(text, angle, font size)` and the y tick labels' texts and font size,
    /// read off the scene (titles and strip text excluded).
    struct DrawnPanelLabels {
        x: Vec<(String, f64, f64)>,
        y: Vec<(String, f64)>,
    }

    fn drawn_panel_labels(panel: &ferrum_scene::Panel) -> DrawnPanelLabels {
        let mut drawn = DrawnPanelLabels { x: Vec::new(), y: Vec::new() };
        for node in &panel.axes {
            if let ferrum_scene::SceneNode::Text { content, style, .. } = node {
                if content.contains("-category-") {
                    drawn.x.push((content.clone(), style.angle, style.font_size));
                } else if content != "cat" && content != "val" {
                    drawn.y.push((content.clone(), style.font_size));
                }
            }
        }
        drawn
    }

    /// Lays `spec` out and builds its scene, returning the layout, the
    /// panels' drawn labels and the theme the chart rendered with.
    fn layout_and_drawn_labels(
        spec: &ChartSpec,
        batch: &RecordBatch,
    ) -> (crate::layout::LayoutResult, Vec<DrawnPanelLabels>, ThemeInputs, PreparedInputs) {
        let viewport = Viewport { width: 900.0, height: 400.0 };
        let chart_config = ChartConfig::default();
        let po = prepare_and_layout(spec, batch, &ThemeInputs::default(), viewport, &chart_config, None)
            .expect("layout should succeed");
        let mut warnings = Vec::new();
        let scene = scene_build::build_scene(
            spec,
            &po.prep,
            &po.layout,
            &po.applied.effective_theme,
            &config::RenderConfig::default(),
            &mut warnings,
            &chart_config,
            None,
        )
        .expect("scene should build");
        let drawn = scene.panels.iter().map(drawn_panel_labels).collect();
        (po.layout, drawn, po.applied.effective_theme, po.prep)
    }

    /// The x label band the panels' drawn labels need, read off the scene:
    /// a rotated label's full extent at its drawn angle, or the standoff
    /// plus one line.
    fn drawn_x_band(drawn: &[DrawnPanelLabels], theme: &ThemeInputs) -> f64 {
        let m = font::FontdueMetrics::new();
        let fs = theme.typography.label_font_size;
        let line_h = m.line_height(fs);
        let tick = theme.sizes.tick_size;
        drawn
            .iter()
            .map(|p| {
                let angle = p.x.first().map_or(0.0, |l| l.1);
                if angle == 0.0 {
                    return tick + 2.0 + line_h;
                }
                let w = p.x.iter().map(|(t, _, f)| m.measure_width(t, *f)).fold(0.0, f64::max);
                let (sin, cos) = angle.to_radians().sin_cos();
                tick + 2.0 + sin.abs() * (fs + w) + cos.abs() * line_h
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The x label band the layout reserved under the plot: the padded
    /// region's bottom less the plot region's, less the x title gutter.
    fn reserved_x_band(layout: &crate::layout::LayoutResult, prep: &PreparedInputs, theme: &ThemeInputs) -> f64 {
        let m = font::FontdueMetrics::new();
        let gutter = crate::layout::axis::compute_x_title_width(
            &prep.axes.x,
            theme.typography.title_font_size,
            theme.padding.axis_title_padding,
            &m,
            true,
        );
        let (padded, plot) = (layout.padded_region, layout.plot_region);
        (padded.y + padded.h) - (plot.y + plot.h) - gutter
    }

    /// The angle of the first drawn tick's label on the chart's own (union)
    /// x axis, laid out at the panels' width.
    fn chart_axis_presentation(layout: &crate::layout::LayoutResult, prep: &PreparedInputs, theme: &ThemeInputs) -> f64 {
        let m = font::FontdueMetrics::new();
        let rect = layout.panels[0].plot_area;
        let ctx = crate::layout::LabelBandCtx::for_axis(&prep.axes.x, theme, &m);
        let (axis, _) = crate::layout::axis::layout_x_axis(&prep.axes.x, rect, 0, 13.0, 4.0, &ctx);
        axis.ticks.iter().find(|t| !t.culled).map_or(0.0, |t| t.label_angle)
    }

    /// Independent x: the chart's own axis, over both panels' eight
    /// categories, rotates at the panel width, but each panel draws only its
    /// own four categories, flat. The reserved x band must be the flat band
    /// the panels draw, not the rotated one the chart's own axis would.
    #[test]
    fn independent_x_panels_drawn_flat_reserve_the_flat_band() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let (spec, batch) = faceted_with_own_categories(4, "", Independent, Shared);
        let (layout, drawn, theme, prep) = layout_and_drawn_labels(&spec, &batch);
        assert!(
            chart_axis_presentation(&layout, &prep, &theme) != 0.0,
            "fixture assumption: the chart's own x axis rotates at the panel width",
        );
        assert_eq!(drawn.len(), 2);
        for p in &drawn {
            assert_eq!(p.x.len(), 4, "each panel draws its own four categories");
            assert!(p.x.iter().all(|l| l.1 == 0.0), "fixture assumption: the panels draw flat ({:?})", p.x);
        }
        let reserved = reserved_x_band(&layout, &prep, &theme);
        let want = drawn_x_band(&drawn, &theme);
        assert!(
            (reserved - want).abs() < 1e-6,
            "the reserved x band ({reserved}) is the panels' drawn band ({want})",
        );
        for (i, p) in drawn.iter().enumerate() {
            let laid_out: Vec<(String, f64)> = layout
                .axes
                .iter()
                .filter(|a| a.panel_index == i && a.orient == crate::layout::AxisOrient::Bottom)
                .flat_map(|a| a.ticks.iter().filter(|t| !t.culled))
                .map(|t| (t.label.clone(), t.label_angle))
                .collect();
            let rendered: Vec<(String, f64)> = p.x.iter().map(|(t, a, _)| (t.clone(), *a)).collect();
            assert_eq!(laid_out, rendered, "panel {i}: the layout lays out the axis the panel draws");
        }
    }

    /// The control: with nine categories per panel the panels' own labels
    /// rotate too, and the band is the panels' drawn rotation, sized by the
    /// widest label any panel draws: panel B's longer names, not panel A's.
    #[test]
    fn independent_x_panels_drawn_rotated_reserve_their_rotated_band() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let (spec, batch) = faceted_with_own_categories(9, "-and-more", Independent, Shared);
        let (layout, drawn, theme, prep) = layout_and_drawn_labels(&spec, &batch);
        assert!(
            drawn.iter().all(|p| p.x.iter().all(|l| l.1 != 0.0)),
            "fixture assumption: the panels draw rotated labels",
        );
        let per_panel: Vec<f64> =
            drawn.iter().map(|p| drawn_x_band(std::slice::from_ref(p), &theme)).collect();
        assert!(per_panel[1] > per_panel[0], "fixture assumption: panel B needs the taller band ({per_panel:?})");
        let reserved = reserved_x_band(&layout, &prep, &theme);
        let want = drawn_x_band(&drawn, &theme);
        assert!(
            (reserved - want).abs() < 1e-6,
            "the reserved x band ({reserved}) is the panels' drawn rotated band ({want})",
        );
    }

    /// Independent y alone: the y band (left of the plot, less the y title
    /// gutter) is the widest y label any panel draws (panel B's
    /// `100,00x.x`, wider than any of the chart's own union labels).
    #[test]
    fn independent_y_panels_reserve_the_widest_drawn_y_label() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let (spec, batch) = faceted_with_own_categories(4, "", Shared, Independent);
        let (layout, drawn, theme, prep) = layout_and_drawn_labels(&spec, &batch);
        let m = font::FontdueMetrics::new();
        let widest = |labels: &mut dyn Iterator<Item = (&str, f64)>| {
            labels.map(|(t, f)| m.measure_width(t, f)).fold(0.0, f64::max)
        };
        let drawn_y_w = drawn
            .iter()
            .map(|p| widest(&mut p.y.iter().map(|(t, f)| (t.as_str(), *f))))
            .fold(0.0, f64::max);
        let fs = theme.typography.label_font_size;
        let chart_y_w = widest(&mut prep.axes.y.tick_labels.iter().map(|t| (t.as_str(), fs)));
        assert!(
            drawn_y_w > chart_y_w,
            "fixture assumption: a panel draws a y label wider than the chart's own ({drawn_y_w} vs {chart_y_w})",
        );
        let y_gutter = crate::layout::axis::compute_y_title_width(
            &prep.axes.y,
            theme.typography.title_font_size,
            theme.padding.axis_title_padding,
            &m,
            true,
        );
        let reserved_y = layout.plot_region.x - layout.padded_region.x - y_gutter;
        let want_y = theme.sizes.tick_size + 2.0 + drawn_y_w;
        assert!(
            (reserved_y - want_y).abs() < 1e-6,
            "the reserved y band ({reserved_y}) holds the widest y label a panel draws ({want_y})",
        );
    }

    /// An independent-x facet panel's ticks sit on the marks at the same
    /// values, as a shared panel's do: with x at 0, 5 and 10 in both panels,
    /// the ticks labeled 0 and 10 are at the circles for x = 0 and x = 10,
    /// including the scale's inset from the panel edge.
    #[test]
    fn independent_x_facet_ticks_sit_on_their_marks() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("p", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0, 0.0, 5.0, 10.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])),
                Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
            ],
        )
        .unwrap();
        for x_resolve in [Shared, Independent] {
            let spec = ChartSpec {
                mark: Mark::Point,
                encoding: Encoding {
                    x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                    y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                    ..Default::default()
                },
                facet: Some(crate::layout::FacetSpec {
                    field: "p".into(),
                    row: None,
                    title: None,
                    row_title: None,
                    sort: None,
                    row_sort: None,
                    mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                    spacing: None,
                    resolve: crate::layout::facet::FacetResolve { x: x_resolve, y: Shared },
                }),
                ..Default::default()
            };
            let po = prepare_and_layout(
                &spec,
                &batch,
                &ThemeInputs::default(),
                Viewport { width: 600.0, height: 300.0 },
                &ChartConfig::default(),
                None,
            )
            .expect("layout should succeed");
            let mut warnings = Vec::new();
            let scene = scene_build::build_scene(
                &spec,
                &po.prep,
                &po.layout,
                &po.applied.effective_theme,
                &config::RenderConfig::default(),
                &mut warnings,
                &ChartConfig::default(),
                None,
            )
            .expect("scene should build");
            for panel in &scene.panels {
                let (lo, hi) = (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w);
                let tick = |label: &str| {
                    panel.axes.iter().find_map(|n| match n {
                        ferrum_scene::SceneNode::Text { x, content, style, .. }
                            if content == label && style.angle == 0.0 && *x >= lo - 20.0 && *x <= hi + 20.0 =>
                        {
                            Some(*x)
                        }
                        _ => None,
                    })
                };
                let mark_xs: Vec<f64> = panel
                    .marks
                    .iter()
                    .flat_map(|b| b.nodes.iter())
                    .filter_map(|n| match n {
                        ferrum_scene::SceneNode::Circle { cx, .. } => Some(*cx),
                        _ => None,
                    })
                    .collect();
                let (min_x, max_x) = mark_xs
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                let (t0, t10) = (tick("0").expect("tick 0"), tick("10").expect("tick 10"));
                assert!(
                    (t0 - min_x).abs() < 1e-3 && (t10 - max_x).abs() < 1e-3,
                    "{x_resolve:?}: ticks 0 and 10 at ({t0}, {t10}), marks at ({min_x}, {max_x})",
                );
            }
        }
    }

    /// The y twin: an independent-y facet panel's ticks sit on the marks at
    /// the same values, as a shared panel's do. With y at 0, 5 and 10 in both
    /// panels, the ticks labeled 0 and 10 are level with the circles at
    /// y = 0 and y = 10 (the baseline sits a third of the font size below the
    /// tick).
    #[test]
    fn independent_y_facet_ticks_sit_on_their_marks() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("p", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0, 0.0, 5.0, 10.0])),
                Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
            ],
        )
        .unwrap();
        for y_resolve in [Shared, Independent] {
            let spec = ChartSpec {
                mark: Mark::Point,
                encoding: Encoding {
                    x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                    y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                    ..Default::default()
                },
                facet: Some(crate::layout::FacetSpec {
                    field: "p".into(),
                    row: None,
                    title: None,
                    row_title: None,
                    sort: None,
                    row_sort: None,
                    mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                    spacing: None,
                    resolve: crate::layout::facet::FacetResolve { x: Shared, y: y_resolve },
                }),
                ..Default::default()
            };
            let theme = ThemeInputs::default();
            let po = prepare_and_layout(
                &spec,
                &batch,
                &theme,
                Viewport { width: 600.0, height: 300.0 },
                &ChartConfig::default(),
                None,
            )
            .expect("layout should succeed");
            let mut warnings = Vec::new();
            let scene = scene_build::build_scene(
                &spec,
                &po.prep,
                &po.layout,
                &po.applied.effective_theme,
                &config::RenderConfig::default(),
                &mut warnings,
                &ChartConfig::default(),
                None,
            )
            .expect("scene should build");
            let baseline_drop = po.applied.effective_theme.typography.label_font_size / 3.0;
            for panel in &scene.panels {
                let tick = |label: &str| {
                    panel.axes.iter().find_map(|n| match n {
                        ferrum_scene::SceneNode::Text { y, content, style, .. }
                            if content == label && style.anchor == ferrum_scene::TextAnchor::End =>
                        {
                            Some(*y - baseline_drop)
                        }
                        _ => None,
                    })
                };
                let mark_ys: Vec<f64> = panel
                    .marks
                    .iter()
                    .flat_map(|b| b.nodes.iter())
                    .filter_map(|n| match n {
                        ferrum_scene::SceneNode::Circle { cy, .. } => Some(*cy),
                        _ => None,
                    })
                    .collect();
                let (top, bottom) = mark_ys
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &y| (a.min(y), b.max(y)));
                let (t0, t10) = (tick("0").expect("tick 0"), tick("10").expect("tick 10"));
                assert!(
                    (t0 - bottom).abs() < 1e-3 && (t10 - top).abs() < 1e-3,
                    "{y_resolve:?}: ticks 0 and 10 at ({t0}, {t10}), marks at ({bottom}, {top})",
                );
            }
        }
    }

    /// Which channel of [`facet_points`] is categorical, over an explicit
    /// `BandScale(range=[10, 110])`; the other stays numeric.
    #[derive(Clone, Copy, Debug)]
    enum BandedChannel {
        Neither,
        X,
        Y,
    }

    /// Points over `groups`, three rows each, faceted by `p` in a wrap of
    /// `ncols` columns. Both channels are titled (`x title`, `y title`) and
    /// numeric (0, 4.5, 9), except the `banded` channel, which places the
    /// categories `c0`, `c1`, `c2` over an explicit pixel range.
    fn facet_points(
        groups: &[&str],
        ncols: u32,
        resolve: crate::layout::facet::FacetResolve,
        banded: BandedChannel,
    ) -> (ChartSpec, RecordBatch) {
        let n = groups.len() * 3;
        let column = |categorical: bool| -> (DataType, arrow::array::ArrayRef) {
            if categorical {
                let cats: Vec<String> = (0..n).map(|i| format!("c{}", i % 3)).collect();
                (DataType::Utf8, Arc::new(StringArray::from(cats)))
            } else {
                let numbers: Vec<f64> = (0..n).map(|i| (i % 3) as f64 * 4.5).collect();
                (DataType::Float64, Arc::new(Float64Array::from(numbers)))
            }
        };
        let (x_banded, y_banded) = (matches!(banded, BandedChannel::X), matches!(banded, BandedChannel::Y));
        let ((x_type, x), (y_type, y)) = (column(x_banded), column(y_banded));
        let p: Vec<&str> = groups.iter().flat_map(|g| [*g; 3]).collect();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", x_type, false),
                Field::new("y", y_type, false),
                Field::new("p", DataType::Utf8, false),
            ])),
            vec![x, y, Arc::new(StringArray::from(p))],
        )
        .unwrap();
        let channel = |field: &str, banded: bool| EncodingSpec {
            field: field.into(),
            title: Some(format!("{field} title")),
            scale: banded.then(|| crate::spec::encoding::ScaleSpec::Band {
                domain: None,
                padding: crate::scale::discrete::BandPadding::default(),
                align: 0.5,
                range: Some(vec![10.0, 110.0]),
            }),
            ..Default::default()
        };
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(channel("x", x_banded)),
                y: Some(channel("y", y_banded)),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "p".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols },
                spacing: None,
                resolve,
            }),
            ..Default::default()
        };
        (spec, batch)
    }

    /// Lays `spec` out at `width`×`height` and builds its scene.
    fn facet_scene(spec: &ChartSpec, batch: &RecordBatch, width: f64, height: f64) -> (PipelineOutput, ferrum_scene::SceneGraph) {
        let chart_config = ChartConfig::default();
        let po = prepare_and_layout(spec, batch, &ThemeInputs::default(), Viewport { width, height }, &chart_config, None)
            .expect("layout should succeed");
        let mut warnings = Vec::new();
        let scene = scene_build::build_scene(
            spec,
            &po.prep,
            &po.layout,
            &po.applied.effective_theme,
            &config::RenderConfig::default(),
            &mut warnings,
            &chart_config,
            None,
        )
        .expect("scene should build");
        (po, scene)
    }

    /// How many times a scene panel's axes draw the text `content`.
    fn axis_text_count(panel: &ferrum_scene::Panel, content: &str) -> usize {
        panel
            .axes
            .iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Text { content: c, .. } if c == content))
            .count()
    }

    /// The pixel of every tick dash a scene panel's axes draw, ascending:
    /// the x of each vertical dash (`along_x`) or the y of each horizontal
    /// one. A dash's endpoints sit on its tick's position exactly; the axis
    /// domain lines, the only other axis lines, span the plot.
    fn tick_dashes(panel: &ferrum_scene::Panel, along_x: bool) -> Vec<f64> {
        let mut out: Vec<f64> = panel
            .axes
            .iter()
            .filter_map(|n| match *n {
                ferrum_scene::SceneNode::Line { x1, y1, x2, y2, .. } => {
                    let (at, across, along) = if along_x { (x1, x2, y2 - y1) } else { (y1, y2, x2 - x1) };
                    (at == across && along.abs() < 20.0).then_some(at)
                }
                _ => None,
            })
            .collect();
        out.sort_by(f64::total_cmp);
        out
    }

    /// The tick positions layout laid out for panel `panel_index`'s x axis
    /// (`along_x`) or y axis, ascending.
    fn laid_out_ticks(layout: &crate::layout::LayoutResult, panel_index: usize, along_x: bool) -> Vec<f64> {
        use crate::layout::AxisOrient::{Bottom, Left};
        let orient = if along_x { Bottom } else { Left };
        let mut out: Vec<f64> = layout
            .axes
            .iter()
            .filter(|a| a.panel_index == panel_index && a.orient == orient)
            .flat_map(|a| a.ticks.iter().map(|t| t.position))
            .collect();
        out.sort_by(f64::total_cmp);
        out
    }

    /// The distinct centers of a scene panel's points along x or y,
    /// ascending.
    fn point_centers(panel: &ferrum_scene::Panel, along_x: bool) -> Vec<f64> {
        let mut out: Vec<f64> = panel
            .marks
            .iter()
            .flat_map(|b| b.nodes.iter())
            .filter_map(|n| match *n {
                ferrum_scene::SceneNode::Circle { cx, cy, .. } => Some(if along_x { cx } else { cy }),
                _ => None,
            })
            .collect();
        out.sort_by(f64::total_cmp);
        out.dedup();
        out
    }

    /// An independent facet panel draws exactly the axes layout laid out for
    /// it, so an x title shows only on the bottom row and a y title only on
    /// the left column, as on shared facets: two of each on a 2×2 grid. In a
    /// wrap whose last row is short, the upper panel with no panel below it
    /// (B) draws no x title either. Every panel's tick dashes are the ticks
    /// layout laid out for it, at the same pixels.
    #[test]
    fn independent_facets_draw_the_axes_layout_lays_out() {
        use crate::layout::facet::FacetResolve;
        use crate::layout::ResolveMode::{Independent, Shared};
        let four = ["A", "B", "C", "D"];
        let cases: [(FacetResolve, &[&str], [usize; 2]); 5] = [
            (FacetResolve { x: Shared, y: Shared }, &four, [2, 2]),
            (FacetResolve { x: Independent, y: Shared }, &four, [2, 2]),
            (FacetResolve { x: Shared, y: Independent }, &four, [2, 2]),
            (FacetResolve { x: Independent, y: Independent }, &four, [2, 2]),
            (FacetResolve { x: Independent, y: Independent }, &four[..3], [1, 2]),
        ];
        for (resolve, groups, [x_titles, y_titles]) in cases {
            let label = format!("{resolve:?} over {} panels", groups.len());
            let (spec, batch) = facet_points(groups, 2, resolve, BandedChannel::Neither);
            let (po, scene) = facet_scene(&spec, &batch, 420.0, 360.0);
            assert_eq!(scene.panels.len(), groups.len(), "{label}: every panel renders");
            let titles = |t: &str| -> Vec<usize> { scene.panels.iter().map(|p| axis_text_count(p, t)).collect() };
            assert_eq!(titles("x title").iter().sum::<usize>(), x_titles, "{label}: x titles {:?}", titles("x title"));
            assert_eq!(titles("y title").iter().sum::<usize>(), y_titles, "{label}: y titles {:?}", titles("y title"));
            let bottom_row = po.layout.panels.iter().map(|p| p.row).max();
            for (i, panel) in scene.panels.iter().enumerate() {
                let p = &po.layout.panels[i];
                assert_eq!(axis_text_count(panel, "x title"), usize::from(Some(p.row) == bottom_row), "{label}: panel {i}");
                assert_eq!(axis_text_count(panel, "y title"), usize::from(p.col == 0), "{label}: panel {i}");
                for along_x in [true, false] {
                    assert_eq!(
                        tick_dashes(panel, along_x),
                        laid_out_ticks(&po.layout, i, along_x),
                        "{label}: panel {i} draws the {} ticks layout laid out",
                        if along_x { "x" } else { "y" },
                    );
                }
            }
        }
    }

    /// `route_panel_axes_and_grid` keeps two separate branches (independent
    /// vs. shared facet resolution), and the x/y write order is an artifact
    /// of that split, not a decision either branch makes on purpose: the
    /// independent branch emits the x axis's nodes before the y axis's;
    /// the shared branch walks the global per-panel axis list, which layout
    /// fills with the y axis before the x (`layout/mod.rs` pushes `y_axis`
    /// then `x_axis`), so a shared panel emits y before x. The bottom-left
    /// panel of a 2x2 grid draws both titles, so its node order pins which
    /// axis went first. The two asymmetric cases (only one channel
    /// independent) pin that each channel's own `resolve` flag decides the
    /// branch, not the other channel's. A future collapse of the two
    /// branches into one would change this order along with it; this test
    /// is then the one to rewrite, not preserve.
    #[test]
    fn an_independent_facet_panel_writes_x_before_y_a_shared_one_writes_y_first() {
        use crate::layout::facet::FacetResolve;
        use crate::layout::ResolveMode::{Independent, Shared};
        let four = ["A", "B", "C", "D"];
        let cases = [
            (FacetResolve { x: Shared, y: Shared }, false),
            (FacetResolve { x: Independent, y: Independent }, true),
            (FacetResolve { x: Independent, y: Shared }, true),
            (FacetResolve { x: Shared, y: Independent }, true),
        ];
        for (resolve, x_before_y) in cases {
            let (spec, batch) = facet_points(&four, 2, resolve.clone(), BandedChannel::Neither);
            let (po, scene) = facet_scene(&spec, &batch, 420.0, 360.0);
            let bottom_row = po.layout.panels.iter().map(|p| p.row).max();
            let panel_idx = po
                .layout
                .panels
                .iter()
                .position(|p| Some(p.row) == bottom_row && p.col == 0)
                .expect("a bottom-left panel exists");
            let panel = &scene.panels[panel_idx];
            let title_index = |content: &str| {
                panel
                    .axes
                    .iter()
                    .position(|n| matches!(n, ferrum_scene::SceneNode::Text { content: c, .. } if c == content))
                    .unwrap_or_else(|| panic!("{resolve:?}: no {content} node in {panel:?}"))
            };
            let (x_idx, y_idx) = (title_index("x title"), title_index("y title"));
            assert_eq!(
                x_idx < y_idx,
                x_before_y,
                "{resolve:?}: x title at {x_idx}, y title at {y_idx}"
            );
        }
    }

    /// An explicit `BandScale(range=[10, 110])` or `PointScale(range=[10,
    /// 110])` is chart-absolute, so each facet panel re-anchors it by its
    /// offset from the first panel, and the panel's tick dashes sit on its
    /// points bit for bit (f64 `==`): along x with two columns and along y
    /// with two rows, whether the channel is independent or shared.
    #[test]
    fn explicit_range_facet_ticks_sit_on_their_points_in_every_panel() {
        use crate::layout::facet::FacetResolve;
        use crate::layout::ResolveMode::{Independent, Shared};
        let point = crate::spec::encoding::ScaleSpec::Point {
            domain: None,
            padding: None,
            align: 0.5,
            reverse: false,
            range: Some(vec![10.0, 110.0]),
        };
        let cases = [("band", Independent), ("band", Shared), ("point", Independent), ("point", Shared)];
        for (scale, resolve) in cases {
            for (banded, ncols) in [(BandedChannel::X, 2), (BandedChannel::Y, 1)] {
                let along_x = matches!(banded, BandedChannel::X);
                let facet_resolve = if along_x {
                    FacetResolve { x: resolve, y: Shared }
                } else {
                    FacetResolve { x: Shared, y: resolve }
                };
                let (mut spec, batch) = facet_points(&["A", "B"], ncols, facet_resolve, banded);
                if scale == "point" {
                    let channel = if along_x { spec.encoding.x.as_mut() } else { spec.encoding.y.as_mut() };
                    if let Some(enc) = channel {
                        enc.scale = Some(point.clone());
                    }
                }
                let (po, scene) = facet_scene(&spec, &batch, 600.0, 600.0);
                let label = format!("{scale} {resolve:?} {banded:?}");
                let (first, second) = (po.layout.panels[0].plot_area, po.layout.panels[1].plot_area);
                let offset = if along_x { second.x - first.x } else { second.y - first.y };
                assert!(offset > 100.0, "{label}: fixture assumption: the second panel is offset ({offset})");
                assert_eq!(scene.panels.len(), 2);
                for (i, panel) in scene.panels.iter().enumerate() {
                    let points = point_centers(panel, along_x);
                    assert_eq!(points.len(), 3, "{label}: panel {i} draws three categories");
                    assert_eq!(tick_dashes(panel, along_x), points, "{label}: panel {i}'s ticks sit on its points");
                }
            }
        }
    }

    /// The secondary y axis of an `independent_y` layer re-anchors an
    /// explicit range onto each facet panel as its marks do: with the
    /// layer's categories over `BandScale(range=[10, 110])` and two facet
    /// rows, the secondary tick dashes every panel draws sit on that layer's
    /// own points bit for bit (f64 `==`).
    #[test]
    fn an_explicit_range_secondary_y_sits_on_its_points_in_every_facet_panel() {
        use crate::spec::layer::Layer;
        let (mut spec, batch) = facet_points(
            &["A", "B"],
            1,
            crate::layout::facet::FacetResolve::default(),
            BandedChannel::Neither,
        );
        let batch = {
            let cats: Vec<String> = (0..batch.num_rows()).map(|i| format!("c{}", i % 3)).collect();
            let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
            fields.push(Field::new("cat", DataType::Utf8, false));
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(StringArray::from(cats)));
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
        };
        let primary = Layer { mark: Mark::Point, ..Default::default() };
        let secondary = Layer {
            mark: Mark::Point,
            encoding: Encoding {
                y: Some(EncodingSpec {
                    field: "cat".into(),
                    scale: Some(crate::spec::encoding::ScaleSpec::Band {
                        domain: None,
                        padding: crate::scale::discrete::BandPadding::default(),
                        align: 0.5,
                        range: Some(vec![10.0, 110.0]),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            independent_y: true,
            ..Default::default()
        };
        spec.layers = Some(vec![primary, secondary]);
        let (po, scene) = facet_scene(&spec, &batch, 600.0, 600.0);
        assert_eq!(scene.panels.len(), 2);
        let offset = po.layout.panels[1].plot_area.y - po.layout.panels[0].plot_area.y;
        assert!(offset > 100.0, "fixture assumption: the second panel is offset ({offset})");
        for (i, panel) in scene.panels.iter().enumerate() {
            let mut secondary_points: Vec<f64> = panel
                .marks
                .iter()
                .filter(|b| b.y_slot == 1)
                .flat_map(|b| b.nodes.iter())
                .filter_map(|n| match *n {
                    ferrum_scene::SceneNode::Circle { cy, .. } => Some(cy),
                    _ => None,
                })
                .collect();
            secondary_points.sort_by(f64::total_cmp);
            secondary_points.dedup();
            let mut drawn_ticks: Vec<f64> = panel
                .axes
                .iter()
                .filter_map(|n| match n {
                    ferrum_scene::SceneNode::Group { attrs, children }
                        if attrs.iter().any(|(k, v)| k == "y_slot" && v == "1") =>
                    {
                        Some(children)
                    }
                    _ => None,
                })
                .flatten()
                .filter_map(|n| match *n {
                    ferrum_scene::SceneNode::Line { x1, y1, x2, y2, .. } if y1 == y2 && (x2 - x1).abs() < 20.0 => Some(y1),
                    _ => None,
                })
                .collect();
            drawn_ticks.sort_by(f64::total_cmp);
            assert_eq!(secondary_points.len(), 3, "panel {i}: the secondary layer draws three categories");
            assert_eq!(drawn_ticks, secondary_points, "panel {i}: the drawn secondary ticks sit on the secondary points");
        }
    }

    /// A hidden axis stays hidden on independent facets: with
    /// `axis(x=False)` on an independent x, no panel draws an x tick dash,
    /// label or title (every axis text sits left of the plot, on the y axis),
    /// and with `axis(y=False)` on an independent y, no panel draws a y one
    /// (every axis text sits under the plot).
    #[test]
    fn a_hidden_axis_stays_hidden_on_independent_facets() {
        use crate::layout::facet::FacetResolve;
        use crate::layout::ResolveMode::{Independent, Shared};
        for hide_x in [true, false] {
            let resolve = if hide_x {
                FacetResolve { x: Independent, y: Shared }
            } else {
                FacetResolve { x: Shared, y: Independent }
            };
            let (mut spec, batch) = facet_points(&["A", "B", "C", "D"], 2, resolve, BandedChannel::Neither);
            if hide_x {
                spec.axis_x = Some(false);
            } else {
                spec.axis_y = Some(false);
            }
            let (_, scene) = facet_scene(&spec, &batch, 420.0, 360.0);
            let hidden = if hide_x { "x" } else { "y" };
            assert_eq!(scene.panels.len(), 4);
            for (i, panel) in scene.panels.iter().enumerate() {
                let plot = panel.plot_area;
                assert!(tick_dashes(panel, hide_x).is_empty(), "{hidden} hidden: panel {i} draws no {hidden} tick dash");
                let texts: Vec<(f64, f64)> = panel
                    .axes
                    .iter()
                    .filter_map(|n| match *n {
                        ferrum_scene::SceneNode::Text { x, y, .. } => Some((x, y)),
                        _ => None,
                    })
                    .collect();
                assert!(!texts.is_empty(), "fixture assumption: panel {i} draws its other axis");
                for (x, y) in texts {
                    let on_other_axis = if hide_x { x < plot.x } else { y > plot.y + plot.h };
                    assert!(on_other_axis, "{hidden} hidden: panel {i} draws axis text at ({x}, {y}) off its other axis");
                }
            }
        }
    }

    /// A facet panel that layout collapses (too small to draw a plot) draws
    /// no axis, though it has rows, whether its channels are independent or
    /// shared: layout lays no axis out for it and the scene draws no panel
    /// for it.
    #[test]
    fn a_collapsed_facet_panel_draws_no_axis() {
        use crate::layout::facet::FacetResolve;
        use crate::layout::ResolveMode::{Independent, Shared};
        for resolve in [FacetResolve { x: Independent, y: Independent }, FacetResolve { x: Shared, y: Shared }] {
            let (spec, batch) = facet_points(&["A", "B", "C", "D"], 2, resolve.clone(), BandedChannel::Neither);
            let (po, scene) = facet_scene(&spec, &batch, 120.0, 90.0);
            let collapsed: Vec<usize> = po
                .layout
                .warnings
                .iter()
                .filter_map(|w| match w {
                    crate::layout::LayoutWarning::PanelCollapsed { panel_index } => Some(*panel_index),
                    _ => None,
                })
                .collect();
            assert!(!collapsed.is_empty(), "{resolve:?}: fixture assumption: a panel collapses");
            for &i in &collapsed {
                assert!(po.layout.axes.iter().all(|a| a.panel_index != i), "{resolve:?}: panel {i} lays out no axis");
                assert!(scene.panels.iter().all(|p| p.id != i), "{resolve:?}: panel {i} is not drawn");
            }
        }
    }

    /// An independent y whose panels share one y domain draws, in every
    /// panel, the axis the shared y draws: each panel's y input carries the
    /// chart's own tick labels and projection, in the same order. So under
    /// `label_overlap="parity"`, which keeps every other label counting from
    /// the first, each panel keeps the same values as the shared chart. The
    /// tick count is even, so a panel that reversed its labels would keep the
    /// other half.
    #[test]
    fn an_independent_y_over_one_domain_draws_the_shared_y_axis() {
        use crate::layout::facet::FacetResolve;
        use crate::layout::ResolveMode::{Independent, Shared};
        let kept_y_labels = |y: crate::layout::ResolveMode| {
            let (mut spec, batch) = facet_points(&["A", "B"], 2, FacetResolve { x: Shared, y }, BandedChannel::Neither);
            if let Some(enc) = spec.encoding.y.as_mut() {
                enc.axis = Some(Box::new(AxisStyleSpec { label_overlap: Some("parity".into()), ..Default::default() }));
            }
            let (po, scene) = facet_scene(&spec, &batch, 600.0, 200.0);
            let kept: Vec<Vec<String>> = scene
                .panels
                .iter()
                .map(|p| {
                    p.axes
                        .iter()
                        .filter_map(|n| match n {
                            ferrum_scene::SceneNode::Text { content, style, .. }
                                if style.anchor == ferrum_scene::TextAnchor::End =>
                            {
                                Some(content.clone())
                            }
                            _ => None,
                        })
                        .collect()
                })
                .collect();
            (po, kept)
        };
        let (shared, shared_kept) = kept_y_labels(Shared);
        let (independent, independent_kept) = kept_y_labels(Independent);
        let labels = &shared.prep.axes.y.tick_labels;
        assert_eq!(labels.len() % 2, 0, "fixture assumption: an even tick count, so a reversed order keeps others ({labels:?})");
        assert_eq!(independent.prep.axes.panels.len(), 2);
        for (i, panel) in independent.prep.axes.panels.iter().enumerate() {
            let own = panel.y.as_ref().expect("an independent y input per panel");
            assert_eq!(own.tick_labels, independent.prep.axes.y.tick_labels, "panel {i}: tick labels");
            assert_eq!(own.tick_projection, independent.prep.axes.y.tick_projection, "panel {i}: tick projection");
        }
        let every_other: Vec<String> = labels.iter().step_by(2).cloned().collect();
        assert_eq!(shared_kept[0], every_other, "fixture assumption: parity keeps every other label");
        assert_eq!(independent_kept, shared_kept);
    }

    /// A grid facet (row × column) with one empty combination under an
    /// independent x renders. The empty panel draws nothing, so it gets no
    /// axis inputs of its own and reserves nothing: the x band is the
    /// tallest one the drawn panels need.
    #[test]
    fn an_empty_grid_cell_under_an_independent_x_reserves_nothing() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let cells = [("A", "r1"), ("B", "r1"), ("A", "r2")];
        let (mut cats, mut vals, mut cols, mut rows) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (col, row) in cells {
            for i in 0..4 {
                cats.push(format!("{col}{row}-category-{i:02}"));
                vals.push(i as f64);
                cols.push(col);
                rows.push(row);
            }
        }
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cat", DataType::Utf8, false),
                Field::new("val", DataType::Float64, false),
                Field::new("panel", DataType::Utf8, false),
                Field::new("r", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vals)),
                Arc::new(StringArray::from(cols)),
                Arc::new(StringArray::from(rows)),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "panel".into(),
                row: Some("r".into()),
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Grid { nrows: 2, ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve { x: Independent, y: Shared },
            }),
            ..Default::default()
        };
        let (po, scene) = facet_scene(&spec, &batch, 900.0, 400.0);
        let groups = &po.prep.facet_groups;
        assert_eq!(groups.len(), 4, "fixture assumption: one panel per row × column pair");
        let empty = groups
            .iter()
            .position(|g| g.key.value == "B" && g.row_key.as_ref().is_some_and(|r| r.value == "r2"))
            .expect("the empty combination is a panel");
        for (i, inputs) in po.prep.axes.panels.iter().enumerate() {
            if i == empty {
                assert_eq!(*inputs, crate::layout::PanelAxisInputs::default(), "the empty panel has no inputs");
            } else {
                assert_eq!(inputs.x.as_ref().map(|x| x.tick_labels.len()), Some(4), "panel {i} has its own x");
            }
        }
        assert_eq!(scene.panels.len(), 3, "the empty panel draws nothing");
        let drawn: Vec<DrawnPanelLabels> = scene.panels.iter().map(drawn_panel_labels).collect();
        let theme = &po.applied.effective_theme;
        let (reserved, want) = (reserved_x_band(&po.layout, &po.prep, theme), drawn_x_band(&drawn, theme));
        assert!((reserved - want).abs() < 1e-6, "the reserved x band ({reserved}) is the drawn panels' ({want})");
    }

    /// A 1x2 facet grid keeps the first two of three groups and drops the
    /// third. Under an independent y the dropped group's labels never draw,
    /// so they reserve nothing: the third group's y values run to a billion
    /// (the widest labels), and the y band is the one the same chart draws
    /// from the first two groups alone.
    #[test]
    fn a_dropped_facet_panel_reserves_no_y_band() {
        use crate::layout::ResolveMode::{Independent, Shared};
        let batch = |groups: &[&str]| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
                Field::new("p", DataType::Utf8, false),
            ]));
            let (mut x, mut y, mut p) = (Vec::new(), Vec::new(), Vec::new());
            for g in groups {
                let top = if *g == "C" { 1.0e9 } else { 10.0 };
                for (i, v) in [0.0, top / 2.0, top].into_iter().enumerate() {
                    x.push(i as f64);
                    y.push(v);
                    p.push(*g);
                }
            }
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(x)),
                    Arc::new(Float64Array::from(y)),
                    Arc::new(StringArray::from(p)),
                ],
            )
            .unwrap()
        };
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "p".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Grid { nrows: 1, ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve { x: Shared, y: Independent },
            }),
            ..Default::default()
        };
        let lay_out = |groups: &[&str]| {
            prepare_and_layout(
                &spec,
                &batch(groups),
                &ThemeInputs::default(),
                Viewport { width: 600.0, height: 300.0 },
                &ChartConfig::default(),
                None,
            )
            .expect("layout should succeed")
            .layout
        };
        let three = lay_out(&["A", "B", "C"]);
        let two = lay_out(&["A", "B"]);
        assert_eq!(three.panels.len(), 2, "fixture assumption: the grid drops the third group");
        assert_eq!(three.plot_region, two.plot_region);
        assert_eq!(three.panels[0].plot_area, two.panels[0].plot_area);
    }

    /// The wrapped-label title repro: three categories `cat00_ww0`,
    /// `cat01_ww1`, `cat02_ww2` and a long x title, real fonts, on 120-720px
    /// wide charts 100, 160 and 250px tall. Where the labels draw flat, they
    /// wrap at `_` onto two lines on the narrower charts, and the title
    /// clears the last label line by exactly the axis title padding, with
    /// auto on and off: never drawn over the second line, and no blank
    /// strip under the labels.
    #[test]
    fn a_title_under_wrapped_category_labels_clears_every_line() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let cats: Vec<String> = (0..3).map(|i| format!("cat{i:02}_ww{i}")).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats.iter().map(String::as_str).collect::<Vec<_>>())),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(),
                    type_: Some(crate::spec::encoding::DataType::Nominal),
                    title: Some("A fairly long x axis title for the chart".into()),
                    ..Default::default()
                }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let m = font::FontdueMetrics::new();
        let (mut wrapped, mut single) = (0, 0);
        for (auto, height) in [true, false].into_iter().flat_map(|a| [100.0, 160.0, 250.0].map(move |h| (a, h))) {
            for width in (120..=720).step_by(40) {
                let mut theme = ThemeInputs::default();
                theme.padding.padding_auto = auto;
                let po = prepare_and_layout(
                    &spec,
                    &batch,
                    &theme,
                    Viewport { width: f64::from(width), height },
                    &ChartConfig::default(),
                    None,
                )
                .expect("layout should succeed");
                let t = &po.applied.effective_theme;
                let axis = po
                    .layout
                    .axes
                    .iter()
                    .find(|a| a.orient == crate::layout::AxisOrient::Bottom)
                    .expect("x axis");
                if axis.ticks.iter().all(|tk| tk.culled) || axis.ticks.iter().any(|tk| !tk.culled && tk.label_angle != 0.0) {
                    continue;
                }
                let lines = axis.ticks.iter().filter(|tk| !tk.culled).map(|tk| tk.label.split('\n').count()).max().unwrap_or(1);
                wrapped += usize::from(lines > 1);
                single += usize::from(lines == 1);
                let drawn_fs = axis
                    .ticks
                    .iter()
                    .find(|tk| !tk.culled)
                    .and_then(|tk| tk.label_font_size)
                    .unwrap_or(t.typography.label_font_size);
                let label_bottom = axis.axis_line.y
                    + t.sizes.tick_size
                    + 2.0
                    + m.line_height(t.typography.label_font_size)
                    + (lines - 1) as f64 * 1.2 * drawn_fs;
                let title = axis.title.as_ref().expect("x title");
                let title_top = title.anchor_y - m.line_height(t.typography.title_font_size) / 2.0;
                assert!(
                    (title_top - label_bottom - t.padding.axis_title_padding).abs() < 1e-9,
                    "auto {auto} at {width}px, {lines} line(s): the title sits {} past the labels, not {}",
                    title_top - label_bottom,
                    t.padding.axis_title_padding,
                );
            }
        }
        assert!(wrapped > 0 && single > 0, "the repro must draw wrapped and one-line labels ({wrapped}, {single})");
    }

    /// The wrapped lines the scene draws sit where the layout reserves them,
    /// on bottom and top axes: the `cat00_ww0` repro (three categories, a
    /// long title, real fonts) wraps its labels onto two lines, and the title
    /// baseline sits exactly as far past the last drawn line's baseline
    /// (beyond the axis) as on a one-line chart, plus the cascade's font
    /// shrink below a bottom axis (whose first line keeps the axis font's
    /// line height). On a top axis the lines are drawn as separate lines
    /// stacked away from the axis, not as one text.
    #[test]
    fn wrapped_label_lines_sit_where_the_layout_reserves_them() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let chart = |cats: Vec<String>, orient: Option<&str>| {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(cats.iter().map(String::as_str).collect::<Vec<_>>())),
                    Arc::new(Float64Array::from((0..cats.len()).map(|i| i as f64).collect::<Vec<_>>())),
                ],
            )
            .unwrap();
            let spec = ChartSpec {
                mark: Mark::Point,
                encoding: Encoding {
                    x: Some(EncodingSpec {
                        field: "x".into(),
                        type_: Some(crate::spec::encoding::DataType::Nominal),
                        title: Some("A fairly long x axis title for the chart".into()),
                        axis: orient.map(|o| Box::new(AxisStyleSpec { orient: Some(o.into()), ..Default::default() })),
                        ..Default::default()
                    }),
                    y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                    ..Default::default()
                },
                ..Default::default()
            };
            (spec, batch)
        };
        // (label baselines, title baseline) beyond the axis, from the scene.
        let drawn = |spec: &ChartSpec, batch: &RecordBatch, width: f64| {
            let theme = ThemeInputs::default();
            let po = prepare_and_layout(spec, batch, &theme, Viewport { width, height: 250.0 }, &ChartConfig::default(), None)
                .expect("layout should succeed");
            let mut warnings = Vec::new();
            let scene = scene_build::build_scene(
                spec,
                &po.prep,
                &po.layout,
                &po.applied.effective_theme,
                &config::RenderConfig::default(),
                &mut warnings,
                &ChartConfig::default(),
                None,
            )
            .expect("scene should build");
            let panel = &scene.panels[0];
            let mut labels = Vec::new();
            let mut title = None;
            for n in panel.axes.iter() {
                if let ferrum_scene::SceneNode::Text { y, content, style, .. } = n {
                    if content.starts_with("A fairly long") {
                        title = Some(*y);
                    } else if content.starts_with('c') || content.starts_with("ww") {
                        labels.push((content.clone(), *y, style.angle, style.font_size));
                    }
                }
            }
            (labels, title.expect("x title in the scene"))
        };
        let one_line = (0..3).map(|i| format!("c{i}")).collect::<Vec<_>>();
        let wrapped = (0..3).map(|i| format!("cat{i:02}_ww{i}")).collect::<Vec<_>>();
        for orient in [None, Some("top")] {
            let (spec, batch) = chart(one_line.clone(), orient);
            let (labels, title) = drawn(&spec, &batch, 400.0);
            let (reference, reference_fs) = ((title - labels[0].1).abs(), labels[0].3);
            let mut reached = 0;
            for width in (120..=320).step_by(20) {
                let (spec, batch) = chart(wrapped.clone(), orient);
                let (labels, title) = drawn(&spec, &batch, f64::from(width));
                if labels.is_empty() || labels.iter().any(|l| l.2 != 0.0) || labels.iter().any(|l| l.0.contains('\n')) {
                    assert!(!labels.iter().any(|l| l.0.contains('\n')), "{orient:?} at {width}px: a line break drawn inside one text");
                    continue;
                }
                let ys: Vec<f64> = labels.iter().map(|l| l.1).collect();
                let lines = ys.iter().map(|y| (y * 1000.0).round() as i64).collect::<std::collections::BTreeSet<_>>().len();
                if lines < 2 {
                    continue;
                }
                reached += 1;
                let last = if orient.is_some() { ys.iter().copied().fold(f64::INFINITY, f64::min) } else { ys.iter().copied().fold(f64::NEG_INFINITY, f64::max) };
                let want = if orient.is_some() { reference } else { reference + (reference_fs - labels[0].3) };
                assert!(
                    ((title - last).abs() - want).abs() < 1e-9,
                    "{orient:?} at {width}px, {lines} lines at font {}: the title sits {} past the last line, want {want}",
                    labels[0].3,
                    (title - last).abs(),
                );
            }
            assert!(reached > 0, "{orient:?}: the sweep must draw wrapped labels");
        }
    }


    /// A cramped 2×2 facet grid: panels A, B and C with `first`, `first + 1`
    /// and `first + 2` of their own seven-character categories on a shared
    /// (or independent) x axis, y independent, 780x220. Rotated, enough
    /// labels need a band that, held in the plot's bottom band and in the row
    /// gutter, leaves no panel height.
    fn cramped_facet_grid(x_resolve: crate::layout::ResolveMode, first: usize) -> (ChartSpec, RecordBatch) {
        let mut cats: Vec<String> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        let mut panels: Vec<String> = Vec::new();
        for (k, p) in ["A", "B", "C"].into_iter().enumerate() {
            for i in 0..(first + k) {
                cats.push(format!("{p}xxxx{i:02}"));
                vals.push(10f64.powi(2 * k as i32) * (1.0 + i as f64 * 0.37));
                panels.push(p.to_string());
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("c", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("p", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vals)),
                Arc::new(StringArray::from(panels)),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "c".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "p".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve {
                    x: x_resolve,
                    y: crate::layout::ResolveMode::Independent,
                },
            }),
            ..Default::default()
        };
        (spec, batch)
    }

    /// The cramped 2×2 grid draws a plot in every panel: the twelve shared
    /// labels (3, 4 and 5 per panel), and independent panels with 8, 9 and
    /// 10 labels of their own, would each need a band that leaves no panel
    /// height, so the label cascade culls and re-presents them within the
    /// band the panels leave room for, the independent panels' axes in the
    /// render as well as in layout. A control with 3, 4 and 5 independent
    /// labels per panel fits without a budget. Every case's bottom band and
    /// row gutter hold exactly the band the drawn labels need, so no label
    /// reaches the next row.
    #[test]
    fn a_cramped_facet_grid_keeps_a_plot_and_holds_its_drawn_labels() {
        use crate::layout::ResolveMode::{Independent, Shared};
        for (x_resolve, first, budgeted) in [(Shared, 3, true), (Independent, 8, true), (Independent, 3, false)] {
            let (spec, batch) = cramped_facet_grid(x_resolve, first);
            let viewport = Viewport { width: 780.0, height: 220.0 };
            let chart_config = ChartConfig::default();
            let po = prepare_and_layout(&spec, &batch, &ThemeInputs::default(), viewport, &chart_config, None)
                .expect("layout should succeed");
            let theme = po.applied.effective_theme.clone();
            let mut warnings = Vec::new();
            let scene = scene_build::build_scene(
                &spec,
                &po.prep,
                &po.layout,
                &theme,
                &config::RenderConfig::default(),
                &mut warnings,
                &chart_config,
                None,
            )
            .expect("scene should build");
            let layout = &po.layout;
            assert_eq!(layout.panels.len(), 3);
            for p in &layout.panels {
                assert!(
                    p.plot_area.h > crate::layout::MIN_PANEL_DIM,
                    "{x_resolve:?} {first}: every panel draws a plot ({:?})",
                    p.plot_area,
                );
            }
            assert_eq!(scene.panels.len(), 3, "{x_resolve:?}: every panel renders");
            let m = font::FontdueMetrics::new();
            let fs = theme.typography.label_font_size;
            let line_h = m.line_height(fs);
            let tick = theme.sizes.tick_size;
            let x_labels = |p: &ferrum_scene::Panel| -> Vec<(String, f64, f64)> {
                p.axes
                    .iter()
                    .filter_map(|n| match n {
                        ferrum_scene::SceneNode::Text { content, style, .. } if content.contains("xxxx") => {
                            Some((content.clone(), style.angle, style.font_size))
                        }
                        _ => None,
                    })
                    .collect()
            };
            let band_of = |labels: &[(String, f64, f64)]| -> f64 {
                let Some(first) = labels.first() else {
                    return tick;
                };
                if first.1 == 0.0 {
                    let lines = labels.iter().map(|l| l.0.split('\n').count()).max().unwrap_or(1);
                    return tick + 2.0 + line_h + (lines - 1) as f64 * 1.2 * first.2;
                }
                let w = labels.iter().map(|(t, _, f)| m.measure_width(t, *f)).fold(0.0, f64::max);
                let (sin, cos) = first.1.to_radians().sin_cos();
                tick + 2.0 + sin.abs() * (fs + w) + cos.abs() * line_h
            };
            let drawn = scene.panels.iter().map(|p| band_of(&x_labels(p))).fold(f64::NEG_INFINITY, f64::max);
            let drawn_count: usize = scene.panels.iter().map(|p| x_labels(p).len()).sum();
            let axis_count: usize = layout
                .axes
                .iter()
                .filter(|a| a.orient == crate::layout::AxisOrient::Bottom)
                .map(|a| a.ticks.len())
                .sum();
            if budgeted {
                assert!(drawn_count < axis_count, "{x_resolve:?}: the cascade culled labels to fit");
            } else {
                assert_eq!(drawn_count, axis_count, "each independent panel's own categories fit without a budget");
            }
            let gutter = crate::layout::axis::compute_x_title_width(
                &po.prep.axes.x,
                theme.typography.title_font_size,
                theme.padding.axis_title_padding,
                &m,
                true,
            );
            let (padded, plot) = (layout.padded_region, layout.plot_region);
            let reserved = (padded.y + padded.h) - (plot.y + plot.h) - gutter;
            assert!((reserved - drawn).abs() < 1e-6, "{x_resolve:?}: bottom band {reserved} vs drawn {drawn}");
            let top = layout.panels.iter().find(|p| p.row == 0).expect("row 0").plot_area;
            let below = layout.panels.iter().find(|p| p.row == 1).expect("row 1").plot_area;
            let strip = top.y - layout.plot_region.y;
            let row_gutter = below.y - strip - (top.y + top.h) - theme.padding.row_padding;
            assert!(
                (row_gutter - drawn).abs() < 1e-6,
                "{x_resolve:?}: the row gutter {row_gutter} holds the drawn band {drawn}",
            );
        }
    }

    /// Independent x and y together: the x band holds the panels' flat
    /// labels as above, and the y band (left of the plot, less the y title
    /// gutter) is the widest y label any panel draws (panel B's
    /// `100,00x.x`, wider than any of the chart's own union labels), not the
    /// chart's own y axis's widest.
    #[test]
    fn independent_x_and_y_panels_reserve_the_bands_they_draw() {
        use crate::layout::ResolveMode::Independent;
        let (spec, batch) = faceted_with_own_categories(4, "", Independent, Independent);
        let (layout, drawn, theme, prep) = layout_and_drawn_labels(&spec, &batch);
        let m = font::FontdueMetrics::new();
        let reserved_x = reserved_x_band(&layout, &prep, &theme);
        let want_x = drawn_x_band(&drawn, &theme);
        assert!((reserved_x - want_x).abs() < 1e-6, "x band {reserved_x} vs drawn {want_x}");
        let widest = |labels: &mut dyn Iterator<Item = (&str, f64)>| {
            labels.map(|(t, f)| m.measure_width(t, f)).fold(0.0, f64::max)
        };
        let drawn_y_w = drawn
            .iter()
            .map(|p| widest(&mut p.y.iter().map(|(t, f)| (t.as_str(), *f))))
            .fold(0.0, f64::max);
        let fs = theme.typography.label_font_size;
        let chart_y_w = widest(&mut prep.axes.y.tick_labels.iter().map(|t| (t.as_str(), fs)));
        assert!(
            drawn_y_w > chart_y_w,
            "fixture assumption: a panel draws a y label wider than the chart's own ({drawn_y_w} vs {chart_y_w})",
        );
        let y_gutter = crate::layout::axis::compute_y_title_width(
            &prep.axes.y,
            theme.typography.title_font_size,
            theme.padding.axis_title_padding,
            &m,
            true,
        );
        let reserved_y = layout.plot_region.x - layout.padded_region.x - y_gutter;
        let want_y = theme.sizes.tick_size + 2.0 + drawn_y_w;
        assert!(
            (reserved_y - want_y).abs() < 1e-6,
            "the reserved y band ({reserved_y}) holds the widest y label a panel draws ({want_y})",
        );
    }

    /// Render `spec`/`batch` under a theme whose label-gap policy is `policy`.
    fn render_with_label_gap(
        spec: &ChartSpec,
        batch: &RecordBatch,
        policy: crate::layout::LabelGapPolicy,
    ) -> String {
        let theme = ThemeInputs { label_gap: policy, ..ThemeInputs::default() };
        render_svg(
            spec,
            batch,
            &theme,
            Viewport { width: 900.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        )
        .unwrap()
        .bytes
    }

    /// `#146` / spec §4.3 — THE DEFECT. An independent-x facet axis must honor
    /// `Theme(cull_threshold=)` exactly as the shared-x path does.
    ///
    /// An independent-x facet panel's x axis is laid out per panel, and the
    /// per-panel layout used to pass the compile-time
    /// `DEFAULT_CULL_THRESHOLD` into `layout_x_axis` where every other call
    /// site passes the theme's own value. The theme key was therefore entirely
    /// INERT on that path: a dense axis rendered the same label count at every
    /// setting. This asserts the count MOVES, and in the same direction the
    /// shared-x path moves, which a hardcoded default cannot do.
    #[test]
    fn render_svg_independent_x_facet_honors_theme_cull_threshold() {
        use crate::layout::{LabelGapPolicy, ResolveMode};

        let (spec, batch) = dense_faceted_categorical(24, ResolveMode::Independent);
        let wide = render_with_label_gap(
            &spec,
            &batch,
            LabelGapPolicy::from_cull_threshold(60),
        );
        let none = render_with_label_gap(&spec, &batch, LabelGapPolicy::from_cull_threshold(0));
        let default = render_with_label_gap(&spec, &batch, LabelGapPolicy::default());

        let (n_wide, n_none, n_default) =
            (count_cat_labels(&wide), count_cat_labels(&none), count_cat_labels(&default));
        assert_eq!(
            n_none, 24,
            "cull_threshold=0 must keep every label on an independent-x facet; got {n_none}"
        );
        assert!(
            n_wide < n_default,
            "a 60px gap must drop MORE labels than the default on an independent-x \
             facet axis (pre-fix all three were equal); got wide={n_wide} default={n_default}"
        );
        assert!(
            n_default < n_none,
            "the default gap must drop labels an independent-x facet axis keeps at \
             cull_threshold=0; got default={n_default} none={n_none}"
        );
    }

    /// The independent-x fix reaches the same conclusion the shared-x path
    /// does: same data, same theme, same visible-label count on both facet
    /// resolutions. Pre-fix the independent side was pinned to the default
    /// whatever the theme said, so this held only at the default value.
    #[test]
    fn render_svg_independent_and_shared_facet_x_agree_on_label_gap() {
        use crate::layout::{LabelGapPolicy, ResolveMode};

        for gap in [0_u32, 8, 60] {
            let policy = LabelGapPolicy::from_cull_threshold(gap);
            let (ind_spec, ind_batch) = dense_faceted_categorical(24, ResolveMode::Independent);
            let (sh_spec, sh_batch) = dense_faceted_categorical(24, ResolveMode::Shared);
            let n_ind = count_cat_labels(&render_with_label_gap(&ind_spec, &ind_batch, policy));
            let n_sh = count_cat_labels(&render_with_label_gap(&sh_spec, &sh_batch, policy));
            assert_eq!(
                n_ind, n_sh,
                "independent-x and shared-x facets must cull alike at cull_threshold={gap}"
            );
        }
    }

    /// `#146` / spec §4.4, criterion 5 — the previously unspellable state.
    /// "Require a gap of N px AND never drop a label" must be observably
    /// distinct from BOTH legacy spellings on the same chart. Eight
    /// categories per panel fit flat with no gap, so the 60px gap is what
    /// drives the never-cull axis off flat (to -90, every label kept) and
    /// the may-cull one to dropping labels: the three draw different labels.
    #[test]
    fn render_svg_gap_without_culling_differs_from_both_legacy_spellings() {
        use crate::layout::{LabelGapPolicy, ResolveMode};

        let n = 8;
        let (spec, batch) = dense_faceted_categorical(n, ResolveMode::Independent);
        let no_gap = render_with_label_gap(&spec, &batch, LabelGapPolicy::from_cull_threshold(0));
        let gap_cull =
            render_with_label_gap(&spec, &batch, LabelGapPolicy::from_cull_threshold(60));
        let gap_no_cull = render_with_label_gap(
            &spec,
            &batch,
            LabelGapPolicy::from_cull_threshold(60).with_overrides(None, Some(false)),
        );
        let rotated_labels = |svg: &str| {
            svg.split("<text ")
                .filter(|t| t.contains("rotate(-90") && t.contains(">Cat"))
                .count()
        };
        assert_eq!(
            count_cat_labels(&gap_no_cull),
            n,
            "a policy that forbids culling must keep every label"
        );
        assert!(
            count_cat_labels(&gap_cull) < n,
            "the legacy gap-60 spelling must still drop labels"
        );
        assert!(
            rotated_labels(&no_gap) == 0 && rotated_labels(&gap_no_cull) == 2 * n,
            "the gap drives the never-cull axis off flat: no gap draws flat \
             labels, the gap rotates them"
        );
        assert_ne!(
            gap_no_cull, no_gap,
            "gap-60-never-cull must differ from no-gap-never-cull: the gap still \
             drives the flat/wrap/font-shrink/rotate stages"
        );
        assert_ne!(
            gap_no_cull, gap_cull,
            "gap-60-never-cull must differ from gap-60-may-cull"
        );
    }

    /// #133, end to end through the real per-panel independent-facet-axis
    /// surface (`scene_build::independent_panel_axis_inputs`'s `TickFormatMode::Immediate`
    /// path) — the third surface the issue's "both paths" phrasing did not
    /// name. A faceted chart with `resolve.x = "independent"` and a
    /// time-typed per-channel `x` format on a NON-temporal `x` column must
    /// refuse, not silently reparse the per-panel numeric tick labels as
    /// epoch-ms and render a nonsense 1970 date on every panel.
    #[test]
    fn render_svg_faceted_independent_x_refuses_time_format_on_non_temporal_column() {
        use crate::render::chart_config::AxisStyleSpec;

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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(),
                    type_: None,
                    axis: Some(Box::new(AxisStyleSpec {
                        label_format: Some("%Y-%m-%d".to_string()),
                        label_format_type: Some("time".to_string()),
                        ..Default::default()
                    })),
                    ..Default::default()
                }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve {
                    x: crate::layout::facet::ResolveMode::Independent,
                    y: crate::layout::facet::ResolveMode::Shared,
                },
            }),
            ..Default::default()
        };
        let err = render_svg(
            &spec,
            &batch,
            &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            RenderError::TimeFormatOnNonTemporalScale {
                channel: "x",
                surface: FormatSurface::Axis,
                format: Some("%Y-%m-%d".into()),
                coord_flipped: false,
            }
        );
    }

    /// F-L01-04 render-boundary proof: `FacetSpec.title` set must reach the
    /// rendered SVG as `"{title}: {value}"`, not just the `layout`-level
    /// `StripTitleLayout.text`. This is the walker-parity positive case the
    /// spec requires by test (design doc §7: "facet strip prefixes must
    /// render through the WASM path ... parity asserted by test") — SVG and
    /// WASM both consume `StripTitleLayout.text` through the same
    /// `SceneNode::Text` content, so proving it here is the shared-node half
    /// of that guarantee that doesn't require building the WASM crate.
    #[test]
    fn render_svg_faceted_with_title_prefixes_strip_text() {
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                title: Some("Species".into()),
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            ..Default::default()
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
        assert!(svg.contains("Species: a"), "expected 'Species: a' strip text in SVG:\n{svg}");
        assert!(svg.contains("Species: b"), "expected 'Species: b' strip text in SVG:\n{svg}");
        assert!(svg.contains("Species: c"), "expected 'Species: c' strip text in SVG:\n{svg}");
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

        let mut prep = prepare::prepare_render_inputs(&spec, &batch, &theme, viewport, &ChartConfig::default(), None).unwrap();
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
        let new_svg = svg_walk::walk_svg(&scene, cfg.embed_fonts).unwrap();

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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, legend, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
            ..Default::default()
        };
        let secondary = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y1".into(), ..Default::default() }),
                ..Default::default()
            },
            independent_y: true,
            ..Default::default()
        };
        let spec = ChartSpec {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                ..Default::default()
            },
            layers: Some(vec![primary, secondary]),
            ..Default::default()
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
    /// on `prep.axes.secondary_y[0].overrides.placement.min_band` — via the SAME
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
            output.prep.axes.secondary_y[0].overrides.placement.min_band,
            Some(123.0),
            "axis_y2.min_band must fill the secondary axis's own overrides"
        );
        assert_eq!(output.prep.axes.x.overrides.placement.min_band, None, "axis_y2 must not leak onto the primary x axis");
        assert_eq!(output.prep.axes.y.overrides.placement.min_band, None, "axis_y2 must not leak onto the primary y axis");
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
                .any(|w| matches!(w, RenderWarning::ConfigSurfaceNotPresent { section: ConfigSurface::AxisY2 })),
            "expected a ConfigSurfaceNotPresent{{section: \"axis_y2\"}} warning, got: {:?}",
            result.warnings
        );
    }

    // ── Batch E Task 9: MarkBatch clip escape (`mark_function(clip=False)`) ──

    /// A `Mark::Line` chart, with `mark_style.clip` set to `clip` (or unset
    /// when `None`). Reuses `scatter_3`'s 3-point x/y data — the geometry
    /// itself is irrelevant to these tests; only the SVG's clip-group
    /// PLACEMENT of the resulting `<path>` is under test.
    fn line_chart_with_clip(clip: Option<bool>) -> (ChartSpec, RecordBatch) {
        let (mut spec, batch) = scatter_3();
        spec.mark = Mark::Line;
        spec.mark_style = clip.map(|c| crate::spec::mark_style::MarkKwargsSpec {
            clip: Some(c),
            ..Default::default()
        });
        (spec, batch)
    }

    fn render_line_svg(clip: Option<bool>) -> String {
        let (spec, batch) = line_chart_with_clip(clip);
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default())
            .unwrap()
            .bytes
    }

    /// The panel's clip-group span: `(open_idx, close_idx)` bracketing the
    /// `<g clip-path="...">...</g>` this chart's single panel opens. Assumes
    /// no nested `<g>` inside (true for this fixture: no stroke_cap/join,
    /// normal blend, no per-node metadata wraps), so the first `</g>` after
    /// the open is its own matching close.
    fn clip_group_span(svg: &str) -> (usize, usize) {
        let open = svg.find("<g clip-path=").expect("clip group must be present");
        let close = svg[open..]
            .find("</g>")
            .map(|i| i + open)
            .expect("clip group must close");
        (open, close)
    }

    /// Default (`mark_style` unset, matching every mark before this field
    /// existed): the line's `<polyline>` (linear interpolate — the default —
    /// emits `Polyline`, not `Path`; see `render/marks/line.rs`) renders
    /// INSIDE the panel clip group.
    #[test]
    fn render_svg_clip_unset_line_renders_inside_panel_clip_group() {
        let svg = render_line_svg(None);
        let (open, close) = clip_group_span(&svg);
        let line_pos = svg.find("<polyline").expect("line polyline must be present");
        assert!(
            line_pos > open && line_pos < close,
            "default (clip unset) line must render INSIDE the panel clip group: \
             polyline at {line_pos}, clip group [{open}, {close})\n{svg}"
        );
    }

    /// `mark_function(clip=False)`'s wire shape (`mark_style.clip = Some(false)`):
    /// the line's `<polyline>` renders OUTSIDE (after) the panel clip group closes.
    #[test]
    fn render_svg_clip_false_line_renders_outside_panel_clip_group() {
        let svg = render_line_svg(Some(false));
        let (_, close) = clip_group_span(&svg);
        let line_pos = svg.find("<polyline").expect("line polyline must be present");
        assert!(
            line_pos > close,
            "clip=false line must render OUTSIDE (after) the panel clip group: \
             polyline at {line_pos}, clip group closes at {close}\n{svg}"
        );
    }

    /// Explicit `clip=True` must resolve identically to leaving `mark_style`
    /// unset — the byte-stability contract for the theme-default value.
    #[test]
    fn render_svg_explicit_clip_true_is_byte_identical_to_unset() {
        assert_eq!(render_line_svg(None), render_line_svg(Some(true)));
    }
}

#[cfg(test)]
mod png_tests {
    use super::*;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn render_png_produces_png_magic_bytes() {
        let spec = ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                title: None,
                row_title: None,
                sort: None,
                row_sort: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            ..Default::default()
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


