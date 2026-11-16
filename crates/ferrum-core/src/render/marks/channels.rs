//! Shared per-mark channel loaders (C9).
//!
//! Two patterns recurred verbatim across the mark builders and are unified here:
//!
//! - [`color_column_loader`] — the `(scale_kind, field) → (categorical, numeric)`
//!   split that `point` and `bar` both performed inline. The categorical string
//!   column is read for `Categorical` *and* scale-less charts; the numeric column
//!   for `Continuous` charts. This is byte-identical to `point`'s former inline
//!   `(color_values_str, color_values_f64)` block and to `bar`'s prior
//!   `load_color_columns` helper. `text`, `rect` (three draw paths) and `arc`
//!   (three draw paths) all route through it too as of the NF-A3 sweep
//!   completion (2026-09-02): the scale-less categorical branch those six sites
//!   used to omit is inert for them, because `resolve_fill_color` returns the
//!   constant fallback whenever no color scale resolved, so adopting the shared
//!   loader is byte-identical for them apart from the dtype fix it carries.
//!
//! - [`row_colors_from_scale`] — the same read for `rule`/`segment`, whose paint
//!   resolution consumes an already-looked-up `Color` rather than the raw
//!   category string. It is a thin adapter *over* `color_column_loader`'s
//!   fallible core, so there is exactly one `(scale → column)` decision in the
//!   crate; it only differs in being fallible, so the reader's typed error
//!   propagates instead of being discarded into a theme-colored render.
//!
//! - [`polar_channel_resolver`] — the theta→angle / radius→radial channel mapping
//!   under the Python polar remapping, shared by `arc` and `bar::build_polar`.
//!   `theta="x"` puts the angular channel on `x` and the radial channel on `y`;
//!   `theta="y"` mirrors. Byte-identical to both marks' prior inline `match`.
//!
//! `band_extent_or` used to live here too — the GH #39 phase-2 substitution
//! that let bar/box/heatmap/tick width formulas honor an explicit
//! `BandScale`/`PointScale` pixel range in place of the panel extent. It is
//! gone (F-L04-03, spec §4A, #67): those formulas no longer divide a pixel
//! extent by a batch-derived category count at all. They ask the resolved
//! scale for its drawn band directly
//! ([`ScaleKind::bandwidth`](crate::render::scale_resolve::ScaleKind)), which
//! is padding-aware and domain-counted, so the extent-vs-fallback choice this
//! helper existed to make no longer has a caller.

use crate::render::color::Color;
use crate::render::arrow_cast::col_as_identity_key_str;
use crate::render::draw::{binned_color_rows, binned_color_scale, col_as_f64, color_field, DrawCtx};
use ferrum_scene::numeric_index_dash;
use crate::render::scale_resolve::{dtype_reads_categorical, ColorInput, ColorScale, ResolvedScales, ScaleKind, StrokeDashScale};
use crate::render::series_keys::{ColorSplit, SeriesKey, SeriesKeys, SplitChannel, SplitSource};
use crate::render::RenderError;
use crate::spec::coord::PolarThetaChannel;
use crate::spec::encoding::Encoding;

/// The per-row color column a resolved [`ColorScale`] wants, already read.
///
/// One value rather than a `(categorical, numeric)` pair: [`ColorInput`]
/// partitions the scale variants, so exactly one representation is ever loaded,
/// and consumers that must act on the choice ([`row_colors_from_scale`]) match
/// the data instead of re-deriving the dispatch from the scale. A pair made
/// `(Some, Some)` representable, let the load and the lookup disagree without a
/// type error, and would have turned a future third [`ColorInput`] variant into
/// a silent "no color" rather than a non-exhaustive match.
pub(crate) enum ColorColumns {
    /// Category keys (`col_as_identity_key_str`), looked up via
    /// [`ColorScale::lookup`].
    Category(Vec<Option<String>>),
    /// Numeric values (`col_as_f64`), sampled via [`ColorScale::lookup_f64`].
    Numeric(Vec<Option<f64>>),
    /// No per-row color applies: no color field, or a field this batch does not
    /// carry.
    None,
}

/// The two-slot spelling of [`ColorColumns`]: `(categorical, numeric)`, at most
/// one `Some`.
///
/// The adapter shape [`color_column_loader`] hands the eight builders that feed
/// both slots straight to
/// [`resolve_fill_color`](crate::render::draw::resolve_fill_color) and index
/// whichever is populated. [`ColorColumns`] is the canonical form; this exists
/// because those consumers never dispatch on the choice, so a match would buy
/// them nothing.
pub(crate) type ColorColumnPair = (Option<Vec<Option<String>>>, Option<Vec<Option<f64>>>);

/// Load the per-row color-encoding columns for fill resolution (C9).
///
/// Mirrors `point`/`bar`: the categorical string column is read for
/// [`ColorInput::Category`] (and scale-less) charts; the numeric column for
/// [`ColorInput::Numeric`] ones. The numeric branch reads `col_as_f64` so the
/// caller can sample via `lookup_f64` without an `f64 → String → f64`
/// round-trip.
///
/// The categorical read is [`col_as_identity_key_str`], never `col_as_str`
/// (NF-A3, spec §4.4 — completed 2026-09-02). `col_as_str` returns `Err` for
/// every non-`Utf8` dtype, and the `.ok()` here swallows it, so an
/// `Int*`/`Float*`/`Bool` categorical color column used to yield `None` for
/// every row: each mark then fell back to the palette's first color while the
/// legend — built from the dtype-wide `distinct_values_in_order` — enumerated
/// every category. `col_as_identity_key_str` is a strict superset (identical
/// on every dtype the category reader takes, a `Timestamp` included, keyed by
/// its UTC ISO label) and stringifies exactly
/// the way `distinct_values_in_order` builds the domain, so the per-row key
/// matches a domain entry. Utf8 columns are byte-identical.
///
/// Returns the [`ColorColumnPair`] the eight consuming builders destructure
/// inline; [`ColorColumns`] is the same answer as one value, for the caller
/// that dispatches on it.
pub(crate) fn color_column_loader(ctx: &DrawCtx) -> ColorColumnPair {
    match color_columns(ctx).unwrap_or(ColorColumns::None) {
        ColorColumns::Category(v) => (Some(v), None),
        ColorColumns::Numeric(v) => (None, Some(v)),
        ColorColumns::None => (None, None),
    }
}

/// The fallible core of [`color_column_loader`]: the crate's single
/// `(color scale, color field) → which column to read` decision.
///
/// Split out so [`row_colors_from_scale`] can consume the same decision *and*
/// the typed error, rather than re-deriving the read with a policy of its own.
/// It returns [`ColorColumns`] rather than the pair so that decision is carried
/// as data: the consumer matches what was loaded instead of asking
/// [`ColorScale::input`] a second time and trusting the two answers to agree.
///
/// A column a numeric scale cannot read (one that reads categorical,
/// [`dtype_reads_categorical`]) loads nothing: on a layered chart it is a
/// later layer's color that the scale its establishing layer's numeric
/// column decided left out
/// ([`RenderWarning::LayerScaleExcluded`](crate::render::RenderWarning::LayerScaleExcluded),
/// already raised), so the layer draws its constant paint, on every mark
/// alike. The establishing column itself is numeric by construction, since
/// it decided the scale's kind.
fn color_columns(ctx: &DrawCtx) -> Result<ColorColumns, RenderError> {
    let field = color_field(ctx, ctx.spec);
    let input = ctx.scales.color.as_ref().map(ColorScale::input);
    Ok(match (input, field) {
        (Some(ColorInput::Category) | None, Some(f)) => {
            ColorColumns::Category(col_as_identity_key_str(ctx.batch, f)?)
        }
        (Some(ColorInput::Numeric), Some(f))
            if ctx.batch.column_by_name(f).is_some_and(|col| dtype_reads_categorical(col.data_type())) =>
        {
            ColorColumns::None
        }
        (Some(ColorInput::Numeric), Some(f)) => ColorColumns::Numeric(col_as_f64(ctx.batch, f)?),
        _ => ColorColumns::None,
    })
}

/// Per-row stroke [`Color`] resolved from the color encoding through the
/// resolved color scale, for the stroke-only marks that consume a resolved
/// color rather than the raw category string (`rule`, `segment`).
///
/// The counterpart to [`color_column_loader`] for marks whose paint resolution
/// consumes a resolved [`Color`] instead of feeding raw columns to
/// [`resolve_fill_color`](crate::render::draw::resolve_fill_color). It is that
/// loader's fallible core plus the per-row lookup, so the two readers cannot
/// disagree about which column a scale wants: the dispatch on
/// [`ColorScale::input`] here is the same one `resolve_fill_color` performs,
/// and the `Category`/`Numeric` treatment matches it row for row.
///
/// - [`ColorInput::Category`] → the [`col_as_identity_key_str`] column,
///   resolved via [`ColorScale::lookup`]. `Int*`/`Float*`/`Bool` category
///   columns resolve to their real colors instead of collapsing to the theme
///   stroke, and a `Timestamp` column keys by its UTC ISO label; `Utf8` is
///   byte-identical (NF-A3, spec §4.4).
/// - [`ColorInput::Numeric`] → the [`col_as_f64`] column, finite-filtered,
///   resolved via [`ColorScale::lookup_f64`]. Two things ride on reading the
///   numeric column here rather than stringifying and leaning on `lookup`'s
///   internal `value.parse()`. First, `Timestamp` is a supported *numeric*
///   dtype and not a supported *category* dtype, so a temporal color encoding
///   (which always resolves a numeric-keyed scale) renders a gradient on these
///   marks exactly as it does on `point`, instead of refusing the whole chart
///   on a read no numeric scale ever needed. Second, `float_as_ordinal_str`
///   renders a non-finite value as `"NaN"`, which `f64::from_str` accepts, so
///   the string path fed `NaN` into `normalize_continuous` and painted pure
///   black; the `is_finite` filter (the same one `resolve_fill_color` carries)
///   makes a non-finite row fall back to the constant mark-style paint, which
///   is what a gap should look like.
///
/// `Ok(None)` means "no per-row color applies", and callers fall back to their
/// constant mark-style stroke. Three shapes reach it, all structural rather
/// than failures: no color encoding, no resolved color scale, and — the third —
/// a color field this layer's batch does not carry at all. That last one is
/// ordinary in a layered chart: `Encoding::inherit_from` pushes the chart-level
/// `color` onto every layer, including ones drawing from a different
/// `data_source` (a `mark_qq` reference line reads the single-row `qq_line`
/// batch, which has none of the user's columns). Every other mark treats that
/// case the same way, so it is checked explicitly here rather than left to fall
/// out of a swallowed `UnknownColumn`. The producer-side quirk this compensates
/// for is owned by **#106** (`_Layer` needs first-class encoding-inheritance
/// control); when that lands, this guard is what should be re-examined.
///
/// A dtype the *selected* reader genuinely cannot key on is a different thing:
/// the typed [`RenderError::UnsupportedDtype`] it constructs is returned for the
/// caller to propagate with `?`, never discarded into a silently theme-colored
/// render (spec §4.4; `rule.rs`'s module doc states this invariant and
/// `rule_color_values` used to violate it). After the `ColorInput` dispatch the
/// only surface that reaches it is a category-keyed scale over a column
/// `col_as_identity_key_str` refuses (a `Decimal`, `Time` or nested column).
/// Scale resolution refuses a nominal such column first, for every mark alike
/// (`build_color_scale`'s categorical branch gates on
/// [`ensure_category_keyable`](crate::render::arrow_cast::ensure_category_keyable)
/// before it resolves a domain from any source), so this refusal is the
/// per-mark backstop for a column that scale did not see. A `Timestamp` column
/// is not in that class: it keys by its UTC ISO label, so a later layer's date
/// color under another layer's categorical scale draws on `rule`, `tick` and
/// `segment` as its label text does (a key that matches no domain entry
/// keeps the constant stroke).
pub(crate) fn row_colors_from_scale(
    ctx: &DrawCtx,
) -> Result<Option<Vec<Option<Color>>>, RenderError> {
    let (Some(field), Some(scale)) = (color_field(ctx, ctx.spec), ctx.scales.color.as_ref()) else {
        return Ok(None);
    };
    if ctx.batch.column_by_name(field).is_none() {
        return Ok(None);
    }
    Ok(match color_columns(ctx)? {
        ColorColumns::Category(vals) => Some(
            vals.into_iter().map(|v| v.as_deref().and_then(|s| scale.lookup(s))).collect(),
        ),
        ColorColumns::Numeric(vals) => Some(
            vals.into_iter()
                .map(|v| v.filter(|x| x.is_finite()).and_then(|x| scale.lookup_f64(x)))
                .collect(),
        ),
        ColorColumns::None => None,
    })
}

/// Per-row `stroke_dash` channel columns, loaded once per mark build (T12).
///
/// Mirrors [`color_column_loader`]'s categorical/numeric split, gated on
/// [`ResolvedScales::stroke_dash`] instead of the color scale's
/// [`ColorInput`]: a *categorical* field (scale resolved) loads the
/// identity-key string column, so [`resolve_row_stroke_dash`] can look
/// each row up in the [`StrokeDashScale`]; a *numeric* field (no scale
/// resolved — see that struct's "no third case" doc) loads the `f64` column,
/// unchanged from the pre-T12 palette-index read. At most one of the two is
/// `Some`.
pub(crate) struct DashColumns {
    pub(crate) categorical: Option<Vec<Option<String>>>,
    numeric: Option<Vec<Option<f64>>>,
}

/// Load [`DashColumns`] for `ctx`'s `stroke_dash` encoding, if any.
pub(crate) fn stroke_dash_column_loader(ctx: &DrawCtx) -> DashColumns {
    let field = ctx.spec.encoding.stroke_dash.as_ref().map(|e| e.field.as_str());
    let is_categorical = ctx.scales.stroke_dash.is_some();
    let categorical = match (is_categorical, field) {
        (true, Some(f)) => col_as_identity_key_str(ctx.batch, f).ok(),
        _ => None,
    };
    let numeric = match (is_categorical, field) {
        (false, Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    DashColumns { categorical, numeric }
}

/// Resolve row `i`'s dash pattern from [`DashColumns`], falling back to
/// `base` (the mark style's literal `stroke_dash`) when the row carries no
/// value of its own.
///
/// Categorical rows resolve through [`StrokeDashScale::dash_for`] directly:
/// a value outside the domain or mapped to the palette's solid slot both
/// mean "no dasharray" per that method's own contract, so a resolved
/// (possibly solid) categorical hit is assigned as-is, never re-rescued by
/// `base` — `base` only stands in for a row with NO categorical value at all
/// (a null category). Numeric rows read the value as a slot index through
/// [`numeric_index_dash`] (which owns the index space's clamp), falling back
/// to `base` for a null/non-finite value.
pub(crate) fn resolve_row_stroke_dash(
    cols: &DashColumns,
    scale: Option<&StrokeDashScale>,
    i: usize,
    base: Option<&[f64]>,
) -> Option<Vec<f64>> {
    if let (Some(values), Some(scale)) = (&cols.categorical, scale) {
        return match values.get(i).and_then(|o| o.as_deref()) {
            Some(v) => scale.dash_for(v),
            None => base.map(<[f64]>::to_vec),
        };
    }
    cols.numeric
        .as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| v.is_finite())
        .and_then(numeric_index_dash)
        .or_else(|| base.map(<[f64]>::to_vec))
}

/// The angular (`theta`) and radial (`radius`) field/scale assignment for a polar
/// coordinate system (C9), shared by `arc` and `bar::build_polar`.
///
/// Field references borrow from the [`Encoding`]; `radius_scale` borrows from the
/// [`ResolvedScales`]. `theta2`/`radius2` are the optional second extents bound
/// by annular arcs (`arc` reads them; `bar::build_polar` ignores them).
pub(crate) struct PolarChannels<'a> {
    pub theta_field: Option<&'a str>,
    pub theta2_field: Option<&'a str>,
    pub radius_field: Option<&'a str>,
    pub radius2_field: Option<&'a str>,
    pub radius_scale: &'a ScaleKind,
}

/// Resolve the polar channel assignment under the Python polar remapping (C9).
///
/// - `theta="x"` → theta=x, theta2=x2, radius=y, radius2=y2, radial scale = y
/// - `theta="y"` → theta=y, theta2=y2, radius=x, radius2=x2, radial scale = x
///
/// Byte-identical to the inline `match theta_ch { … }` blocks `arc::build` and
/// `bar::build_polar` previously carried.
pub(crate) fn polar_channel_resolver<'a>(
    theta_ch: PolarThetaChannel,
    enc: &'a Encoding,
    scales: &'a ResolvedScales,
) -> PolarChannels<'a> {
    let field = |e: &'a Option<crate::spec::encoding::EncodingSpec>| {
        e.as_ref().map(|s| s.field.as_str())
    };
    match theta_ch {
        PolarThetaChannel::X => PolarChannels {
            theta_field: field(&enc.x),
            theta2_field: field(&enc.x2),
            radius_field: field(&enc.y),
            radius2_field: field(&enc.y2),
            radius_scale: &scales.y,
        },
        PolarThetaChannel::Y => PolarChannels {
            theta_field: field(&enc.y),
            theta2_field: field(&enc.y2),
            radius_field: field(&enc.x),
            radius2_field: field(&enc.x2),
            radius_scale: &scales.x,
        },
    }
}

/// Partition row indices into color/detail groups for group-mark builders
/// (area, line, ribbon).
///
/// The partitioning algorithm is shared verbatim across all three builders;
/// only the *color-values loading* differs by caller and stays in each one.
/// All three (area, line, ribbon) load color via `col_as_identity_key_str`
/// so Int*/Float*/Bool color columns split into groups matching the legend
/// domain, not just Utf8 (NF-A3: `col_as_str` returns `Err` for any other
/// dtype, which used to collapse non-Utf8 color columns into one group on
/// line/ribbon) — **but only when the resolved color scale is category-keyed**
/// (`ColorScale::input() == ColorInput::Category`, or no scale at all).
/// `area` always force-resolves a `Categorical` scale regardless of column
/// dtype (scale_resolve/mod.rs), so its caller can load unconditionally; a
/// `Continuous`/`Discretizing` (numeric-keyed) color scale must NOT be grouped
/// by distinct value — line/ribbon gate their load on `ColorInput` for exactly
/// this reason (mirrors `color_column_loader` below), otherwise a quantitative
/// color column fragments into near-all singleton groups that this helper's
/// callers then drop via their `len() < 2` connectivity gates, silently
/// blanking the plot under a still-drawn colorbar legend. This helper
/// receives the already-loaded optional value slices and applies the common
/// 4-way split:
///
/// - Color only (scale present): one group per distinct color key; each group
///   retains its key for downstream color resolution.
/// - Detail only: one group per distinct detail value; the key is stripped to
///   `None` so callers fall through to their theme-default color.
/// - Both color and detail: one group per (color, detail) pair; the color key
///   is retained (detail subdivides each color group without its own legend).
/// - Neither: a single group over all row indices `0..n_rows`.
///
/// After the (color, detail) split above, each of `splits` subdivides every
/// resulting group further by its per-row category, in order. A categorical
/// `stroke_dash` is one: the series partition key extends
/// from (color, detail) to (color, detail, stroke_dash-field) so a
/// categorical `stroke_dash` encoding draws one polyline (or, for `ribbon`,
/// one closed-polygon band) per distinct style even within a single
/// color/detail group; `line` and `ribbon` pass `dash_cols.categorical`,
/// `area`, which does not consume the `stroke_dash` channel, does not. The
/// fields a statistic grouped the series by beyond the drawn channels
/// ([`grouping_series_values`]) follow, on all three. No splits is the
/// (color, detail) grouping alone.
///
/// # Arguments
/// - `color`: the caller's own color field and its optional per-row color
///   strings (loaded by the caller) — a binned color passes the caller's
///   raw column field name with its bin keys, so a dropped series names
///   the column the caller wrote, not its resolved bin edge.
/// - `detail`: the caller's own `detail` field and its optional per-row
///   detail strings.
/// - `splits`: each further split's own caller field and its per-row
///   category strings — a categorical `stroke_dash` field,
///   [`grouping_series_values`]'s fields.
/// - `has_color_scale`: whether a resolved color scale is present; gates the
///   color-only branch (matches `&ctx.scales.color` being `Some(_)`).
/// - `n_rows`: total row count, used for the fallback all-rows group.
///
/// # Returns
/// A [`SeriesPartition`]: `groups` is one entry per group (the
/// `Option<String>` is the color legend key for that group — `None` in the
/// detail-only and fallback cases; a subdivided group repeats its parent's
/// key), `fields` the caller fields of exactly the inputs this partition
/// actually read: color-only with a scale gives color; detail-only gives
/// detail; both give color then detail; color without a scale and the
/// neither branch give neither; every split's field follows, in order.
/// `fields` is [`crate::render::marks::curve::ShortRuns::finish`]'s
/// `naming_fields` — reading it from the same match that decides the
/// grouping means naming can never cite a field that did not actually
/// split the series.
pub(crate) fn build_color_detail_groups<'v>(
    color: Option<(&'v str, &'v Vec<Option<String>>)>,
    detail: Option<(&'v str, &'v Vec<Option<String>>)>,
    splits: impl IntoIterator<Item = (&'v str, &'v Vec<Option<String>>)>,
    has_color_scale: bool,
    n_rows: usize,
) -> SeriesPartition {
    let (groups, mut fields) = match (color, detail, has_color_scale) {
        // Color only: one group per color category; key retained for legend.
        (Some((cf, cv)), None, true) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in cv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            (groups, vec![cf.to_string()])
        }
        // Detail only: one group per detail value; key stripped so fill falls
        // to the mark-style default (no per-group color from the legend).
        (None, Some((df, dv)), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in dv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            (groups.into_iter().map(|(_, rows)| (None, rows)).collect(), vec![df.to_string()])
        }
        // Both color and detail: one group per (color, detail) pair.
        (Some((cf, cv)), Some((df, dv)), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for i in 0..n_rows {
                let composite = (cv[i].clone(), dv[i].clone());
                match groups.iter().position(|(_, rows)| {
                    rows.first()
                        .map(|&r| (cv[r].clone(), dv[r].clone()) == composite)
                        .unwrap_or(false)
                }) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((cv[i].clone(), vec![i])),
                }
            }
            (groups, vec![cf.to_string(), df.to_string()])
        }
        // No color, no detail (or color present but no scale): single group.
        _ => (vec![(None, (0..n_rows).collect())], Vec::new()),
    };
    let splits: Vec<(&str, &Vec<Option<String>>)> = splits.into_iter().collect();
    fields.extend(splits.iter().map(|(field, _)| field.to_string()));
    let groups = splits.into_iter().fold(groups, |groups, (_, values)| subdivide(groups, values));
    SeriesPartition { groups, fields }
}

/// [`build_color_detail_groups`]'s return: the series/ring partition every
/// `vertex_frame` and `build` reads (`groups`), and the caller's own fields
/// that partition actually split on (`fields`), read by
/// [`crate::render::marks::curve::ShortRuns::finish`] to name a dropped
/// run — one source for both, so a dropped series can never be named by a
/// field that did not split it.
pub(crate) struct SeriesPartition {
    pub(crate) groups: Vec<(Option<String>, Vec<usize>)>,
    pub(crate) fields: Vec<String>,
}

/// Subdivide each `(color_key, rows)` group by per-row category `values`
/// (a categorical stroke dash, a statistic's grouping field), preserving the
/// parent's color key on every subgroup. Rows within a parent group are
/// re-partitioned in first-encounter order of their category, so subgroup
/// ordering — and therefore the node/metadata alignment invariant (#6) —
/// stays deterministic.
fn subdivide(
    groups: Vec<(Option<String>, Vec<usize>)>,
    values: &[Option<String>],
) -> Vec<(Option<String>, Vec<usize>)> {
    let mut out = Vec::with_capacity(groups.len());
    for (key, rows) in groups {
        let mut sub: Vec<(Option<String>, Vec<usize>)> = Vec::new();
        for i in rows {
            let category = values.get(i).cloned().flatten();
            match sub.iter().position(|(seen, _)| seen == &category) {
                Some(p) => sub[p].1.push(i),
                None => sub.push((category, vec![i])),
            }
        }
        out.extend(sub.into_iter().map(|(_, sub_rows)| (key.clone(), sub_rows)));
    }
    out
}

/// Each field a series mark's statistic grouped its series by beyond the
/// channels the mark draws ([`GroupStyle::series`](crate::render::draw::GroupStyle)),
/// paired with each row's value of that field, for
/// [`build_color_detail_groups`]'s splits. A field the rows do not carry
/// contributes neither an entry nor a split.
pub(crate) fn grouping_series_values(ctx: &DrawCtx) -> Vec<(String, Vec<Option<String>>)> {
    ctx.mark_style
        .group
        .series
        .iter()
        .filter_map(|field| col_as_identity_key_str(ctx.batch, field).ok().map(|values| (field.clone(), values)))
        .collect()
}

/// Refuse a `detail` column no key reader takes, instead of letting its
/// unreadable read merge every row into one series with no warning.
///
/// The partitions read their keys with [`col_as_identity_key_str`] and take
/// `.ok()`, so a column that reader refuses (a `Decimal`, `Time`, binary or
/// nested column) would read as "no key". `detail` has no scale, so its column
/// is checked here, where the mark splits by it: `line`, `area`, `ribbon` and
/// `polygon`. A `stroke_dash` or `shape` column is checked where its scale is
/// built (the categorical-scale gate and the layered union's enumeration), as
/// colour's is, so no mark re-checks it. The refusal is the reader's
/// `UnsupportedDtype` re-voiced with the remedy, as the facet refusal is, and
/// the caller prefixes it with the layer and mark like every other mark
/// refusal. A column the batch does not carry is not refused: that is the
/// ordinary layered shape, and it keeps reading as "no key".
pub(crate) fn refuse_unreadable_detail(ctx: &DrawCtx) -> Result<(), RenderError> {
    if !ctx.spec.mark.splits_series_by_detail() {
        return Ok(());
    }
    let Some(field) = ctx.mark_style.group.detail.as_deref() else {
        return Ok(());
    };
    match (col_as_identity_key_str(ctx.batch, field), ctx.batch.column_by_name(field)) {
        (Err(RenderError::UnsupportedDtype { context: None, .. }), Some(col)) => Err(RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!(
                "{:?} (it cannot key a series; convert '{field}' to strings to split the mark by it)",
                col.data_type()
            ),
            context: None,
        }),
        _ => Ok(()),
    }
}

/// A per-row mark's own partition (`point`, `bar`, `tick`, `rule`,
/// `segment`, `text`, `rect`, `image`, `label`): the categorical channels
/// it draws, each row placed independently rather than joined into one
/// mark per group the way a connected mark's [`build_color_detail_groups`]
/// caller is. Read by
/// [`crate::render::draw::dispatch_mark_groups`] for every mark this
/// module's siblings (`line`, `area`, `ribbon`, `polygon`) do not
/// partition themselves.
///
/// `color` under [`line::series_groups`](crate::render::marks::line::series_groups)'s
/// own precedence: a binned color's own keys
/// ([`BinnedColorSeries`]) when the color is binned, else the category-keyed
/// gate [`color_column_loader`] applies — a quantitative, unbinned color is
/// not grouped, for the reason that sibling documents (it would fragment
/// the partition into near-singleton numeric groups). Then a categorical
/// `stroke_dash` and a categorical `shape` (`point`'s own legend channel),
/// in [`build_color_detail_groups`]'s splits order — the order the code
/// below applies, not color/shape/stroke_dash. A numeric `shape`/
/// `stroke_dash` (or neither bound) contributes no split, so the mark
/// still draws every row with no grouping at all. A binned partition is
/// ordered by bin, as `line`'s is. `detail` is a connected mark's own
/// grouping ([`GroupStyle::detail`](crate::render::draw::GroupStyle))
/// these marks never read, so it never appears here.
pub(crate) fn per_row_mark_groups(ctx: &DrawCtx) -> SeriesPartition {
    let n_rows = ctx.batch.num_rows();
    let keys = series_keys(ctx);
    let color = ColorSeries::of(ctx, &keys);
    let dash_cols = stroke_dash_column_loader(ctx);
    let dash = keys.partition_on(SplitChannel::StrokeDash).map(SeriesKey::field).zip(dash_cols.categorical.as_ref());
    let shape_field = keys.partition_on(SplitChannel::Shape).map(SeriesKey::field).filter(|_| ctx.scales.shape.is_some());
    let shape_values = shape_field.and_then(|f| col_as_identity_key_str(ctx.batch, f).ok());

    let mut partition = build_color_detail_groups(
        color.as_ref().map(ColorSeries::split),
        None,
        dash.into_iter().chain(shape_field.zip(shape_values.as_ref())),
        ctx.scales.color.is_some(),
        n_rows,
    );
    if let Some(color) = &color {
        color.order(&mut partition.groups);
    }
    partition
}

/// The series keys of the mark `ctx` draws, over its own rows, split as the
/// panel's resolved scales draw them ([`SplitSource::Resolved`]). Every
/// partition of a mark's series reads this, as the null-key drop and the
/// impute read prepare's same decision.
pub(in crate::render) fn series_keys(ctx: &DrawCtx) -> SeriesKeys {
    SeriesKeys::of(
        ctx.spec.mark,
        &ctx.spec.encoding,
        ctx.mark_style.group.detail.as_deref(),
        SplitSource::Resolved(ctx.scales),
        ctx.batch.schema().as_ref(),
    )
}

/// A mark's colour as a series key: the colour field and each row's key. A
/// category-keyed colour reads each row's value
/// ([`col_as_identity_key_str`], so Int*, Float*, Bool and date columns split
/// into per-category series matching the legend's domain, which `col_as_str`
/// would collapse into one); a binned colour reads its bin's key, one series
/// per bin in bin order ([`BinnedColorSeries`]); a continuous colour is no
/// key, since grouping by every distinct numeric value fragments a series
/// into single-point groups the builders then drop.
pub(in crate::render) struct ColorSeries {
    field: String,
    values: Vec<Option<String>>,
    binned: Option<BinnedColorSeries>,
}

impl ColorSeries {
    /// The colour key of `keys`, `None` when the colour splits nothing or its
    /// rows cannot be read as keys.
    pub(in crate::render) fn of(ctx: &DrawCtx, keys: &SeriesKeys) -> Option<Self> {
        let field = keys.partition_on(SplitChannel::Color)?.field();
        let (values, binned) = match keys.color_split() {
            ColorSplit::Bins => {
                let binned = BinnedColorSeries::of(ctx)?;
                (binned.keys().clone(), Some(binned))
            }
            ColorSplit::Categories => (col_as_identity_key_str(ctx.batch, field).ok()?, None),
            ColorSplit::Unsplit => return None,
        };
        Some(Self { field: field.to_string(), values, binned })
    }

    /// The colour field and its per-row keys, for [`build_color_detail_groups`].
    pub(in crate::render) fn split(&self) -> (&str, &Vec<Option<String>>) {
        (&self.field, &self.values)
    }

    /// Orders `groups` low bin first under a binned colour; any other
    /// partition keeps its order.
    pub(in crate::render) fn order(&self, groups: &mut [(Option<String>, Vec<usize>)]) {
        if let Some(binned) = &self.binned {
            binned.order(groups);
        }
    }
}

/// A mark's `detail` field and each row's key, when it splits series by one
/// ([`SeriesKeys::partition_on`]) over a column the rows read as keys.
pub(in crate::render) fn detail_series(ctx: &DrawCtx, keys: &SeriesKeys) -> Option<(String, Vec<Option<String>>)> {
    let field = keys.partition_on(SplitChannel::Detail)?.field();
    col_as_identity_key_str(ctx.batch, field).ok().map(|values| (field.to_string(), values))
}

/// The series a binned color (`Color(bin=...)`) splits a mark into: each
/// row's bin ([`binned_color_rows`]) and the key its series groups by, the
/// bin's lower edge, a value that bin's color resolves from through
/// [`ColorScale::lookup`]. The area, the line and the ribbon draw one series
/// per bin through it, so each series takes its bin's color and the bin
/// legend describes what is drawn.
pub(crate) struct BinnedColorSeries {
    bins: Vec<Option<usize>>,
    keys: Vec<Option<String>>,
}

impl BinnedColorSeries {
    /// `None` unless the mark's color is a binned color under the scale its
    /// bin rule cut ([`binned_color_scale`]), over a column the batch holds. A row with no finite
    /// value has no bin and no key.
    pub(crate) fn of(ctx: &DrawCtx) -> Option<Self> {
        let (binding, scale) = (ctx.spec.encoding.color.as_ref(), ctx.scales.color.as_ref());
        let buckets = binned_color_scale(binding, scale)?;
        let bins = binned_color_rows(binding, scale, ctx.batch)?;
        let keys = bins.iter().map(|bin| bin.map(|i| buckets.bounds()[i].to_string())).collect();
        Some(Self { bins, keys })
    }

    /// Each row's series key, for [`build_color_detail_groups`]'s color values.
    pub(crate) fn keys(&self) -> &Vec<Option<String>> {
        &self.keys
    }

    /// Orders `groups` low bin first, so the series draw (and a stack layers)
    /// in bin order; a series with no bin trails. Stable, so a detail or
    /// stroke-dash split keeps its order within a bin.
    pub(crate) fn order(&self, groups: &mut [(Option<String>, Vec<usize>)]) {
        groups.sort_by_key(|(_, rows)| rows.first().and_then(|&row| self.bins[row]).unwrap_or(usize::MAX));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;
    use crate::spec::encoding::EncodingSpec;

    fn enc_field(name: &str) -> Option<EncodingSpec> {
        Some(EncodingSpec { field: name.into(), ..Default::default() })
    }

    /// Distinguishable x/y scales: x domain [0,10], y domain [0,20], so a test
    /// can tell which axis `radius_scale` points at via `data_domain()`.
    fn distinguishable_scales() -> ResolvedScales {
        ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 10.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 20.0], vec![100.0, 0.0], false, false)),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
        }
    }

    /// C9 guard: under `theta="x"` the angular channel is x (theta2=x2), the
    /// radial channel is y (radius2=y2), and the radial scale is the y scale.
    /// A regression that swaps the axis mapping (e.g. radial scale ← x) fails.
    #[test]
    fn polar_resolver_theta_x_maps_radius_to_y() {
        let enc = Encoding {
            x: enc_field("angle"),
            x2: enc_field("angle2"),
            y: enc_field("value"),
            y2: enc_field("value2"),
            ..Default::default()
        };
        let scales = distinguishable_scales();
        let pc = polar_channel_resolver(PolarThetaChannel::X, &enc, &scales);
        assert_eq!(pc.theta_field, Some("angle"));
        assert_eq!(pc.theta2_field, Some("angle2"));
        assert_eq!(pc.radius_field, Some("value"));
        assert_eq!(pc.radius2_field, Some("value2"));
        // y scale has domain [0, 20].
        assert_eq!(pc.radius_scale.data_domain(), Some((0.0, 20.0)));
    }

    /// C9 guard: `theta="y"` mirrors — angular channel = y, radial channel = x,
    /// radial scale = the x scale.
    #[test]
    fn polar_resolver_theta_y_maps_radius_to_x() {
        let enc = Encoding {
            x: enc_field("value"),
            x2: enc_field("value2"),
            y: enc_field("angle"),
            y2: enc_field("angle2"),
            ..Default::default()
        };
        let scales = distinguishable_scales();
        let pc = polar_channel_resolver(PolarThetaChannel::Y, &enc, &scales);
        assert_eq!(pc.theta_field, Some("angle"));
        assert_eq!(pc.theta2_field, Some("angle2"));
        assert_eq!(pc.radius_field, Some("value"));
        assert_eq!(pc.radius2_field, Some("value2"));
        // x scale has domain [0, 10].
        assert_eq!(pc.radius_scale.data_domain(), Some((0.0, 10.0)));
    }

    /// C9 guard: absent paired channels resolve to `None` (the legacy pie path
    /// gates annular mode on theta2/radius2 being `Some`).
    #[test]
    fn polar_resolver_absent_paired_channels_are_none() {
        let enc = Encoding { x: enc_field("angle"), y: enc_field("value"), ..Default::default() };
        let scales = distinguishable_scales();
        let pc = polar_channel_resolver(PolarThetaChannel::X, &enc, &scales);
        assert_eq!(pc.theta2_field, None);
        assert_eq!(pc.radius2_field, None);
    }

    // ── color_column_loader (C9) ────────────────────────────────────────────

    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::color::{ContinuousScheme, NamedContinuous};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::ColorScale;
    use crate::spec::chart::ChartSpec;
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::borrow::Cow;
    use std::sync::Arc;

    fn color_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: enc_field("x"),
                y: enc_field("y"),
                color: enc_field("c"),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Batch with `c` as both a numeric and string column readable via the
    /// respective casts (numeric stored as Float64, stringified on demand).
    fn color_batch() -> arrow::record_batch::RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ]).unwrap()
    }

    fn cat_batch() -> arrow::record_batch::RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ]).unwrap()
    }

    fn scales_with_color(color: Option<ColorScale>) -> ResolvedScales {
        let mut s = distinguishable_scales();
        s.color = color;
        s
    }

    fn run_loader(
        spec: &ChartSpec,
        batch: &arrow::record_batch::RecordBatch,
        scales: &ResolvedScales,
    ) -> ColorColumnPair {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec, panel: &panel, theme: &theme, scales, batch, mark_style: &mark_style };
        color_column_loader(&ctx)
    }

    /// C9 guard: a `Continuous` color scale loads the numeric column only.
    #[test]
    fn color_loader_continuous_loads_numeric_only() {
        let spec = color_spec();
        let batch = color_batch();
        let scales = scales_with_color(Some(ColorScale::Continuous {
            domain: (1.0, 3.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        }));
        let (cat, num) = run_loader(&spec, &batch, &scales);
        assert!(cat.is_none(), "continuous scale must not load the categorical column");
        assert_eq!(num, Some(vec![Some(1.0), Some(2.0), Some(3.0)]));
    }

    /// C9 guard: a `Categorical` color scale loads the string column only.
    #[test]
    fn color_loader_categorical_loads_strings_only() {
        let spec = color_spec();
        let batch = cat_batch();
        let scales = scales_with_color(Some(ColorScale::Categorical {
            domain: vec!["a".into(), "b".into(), "c".into()],
            palette: Cow::Owned(vec![]),
        }));
        let (cat, num) = run_loader(&spec, &batch, &scales);
        assert_eq!(cat, Some(vec![Some("a".into()), Some("b".into()), Some("c".into())]));
        assert!(num.is_none(), "categorical scale must not load the numeric column");
    }

    /// C9 guard: the scale-less defensive branch (`color` field present, no scale
    /// resolved) loads the string column — point/bar both relied on this.
    #[test]
    fn color_loader_scaleless_field_loads_strings() {
        let spec = color_spec();
        let batch = cat_batch();
        let scales = scales_with_color(None);
        let (cat, num) = run_loader(&spec, &batch, &scales);
        assert_eq!(cat, Some(vec![Some("a".into()), Some("b".into()), Some("c".into())]));
        assert!(num.is_none(), "scale-less branch must not load the numeric column");
    }

    // ── build_color_detail_groups (FA-13) ────────────────────────────────────

    fn sv(s: &str) -> Option<String> { Some(s.to_string()) }

    /// FA-13 guard: color-only with a scale present partitions by color key and
    /// retains the key for downstream legend-based color resolution.
    #[test]
    fn groups_color_only_with_scale_partitions_by_color() {
        let cv: Vec<Option<String>> = vec![sv("A"), sv("B"), sv("A"), sv("B")];
        let groups = super::build_color_detail_groups(Some(("c", &cv)), None, Vec::new(), true, 4).groups;
        assert_eq!(groups.len(), 2, "two color keys → two groups");
        assert_eq!(groups[0].0, sv("A"));
        assert_eq!(groups[0].1, vec![0, 2]);
        assert_eq!(groups[1].0, sv("B"));
        assert_eq!(groups[1].1, vec![1, 3]);
    }

    /// FA-13 guard: color present but no scale → falls through to single group
    /// (the `_` fallback arm; mirrors what area/line do without a color legend).
    #[test]
    fn groups_color_only_without_scale_is_single_group() {
        let cv: Vec<Option<String>> = vec![sv("A"), sv("B"), sv("A")];
        let groups = super::build_color_detail_groups(Some(("c", &cv)), None, Vec::new(), false, 3).groups;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, None);
        assert_eq!(groups[0].1, vec![0, 1, 2]);
    }

    /// FA-13 guard: detail-only partitions by detail value and strips the key
    /// to None so callers use their theme-default color (no legend).
    #[test]
    fn groups_detail_only_strips_key_to_none() {
        let dv: Vec<Option<String>> = vec![sv("s0"), sv("s1"), sv("s0"), sv("s1")];
        let groups = super::build_color_detail_groups(None, Some(("d", &dv)), Vec::new(), false, 4).groups;
        assert_eq!(groups.len(), 2, "two detail values → two groups");
        // Keys are stripped to None in both groups.
        assert!(groups.iter().all(|(k, _)| k.is_none()), "detail-only keys must all be None");
        // Row assignment still correct.
        let rows0: Vec<usize> = groups[0].1.clone();
        let rows1: Vec<usize> = groups[1].1.clone();
        assert_eq!(rows0, vec![0, 2]);
        assert_eq!(rows1, vec![1, 3]);
    }

    /// FA-13 guard: color + detail partitions by the (color, detail) composite
    /// pair and retains the color key (not the detail key).
    #[test]
    fn groups_color_and_detail_partitions_by_pair() {
        // 4 rows: (A,x), (A,y), (B,x), (B,y)
        let cv: Vec<Option<String>> = vec![sv("A"), sv("A"), sv("B"), sv("B")];
        let dv: Vec<Option<String>> = vec![sv("x"), sv("y"), sv("x"), sv("y")];
        let groups = super::build_color_detail_groups(Some(("c", &cv)), Some(("d", &dv)), Vec::new(), true, 4).groups;
        assert_eq!(groups.len(), 4, "4 distinct (color, detail) pairs → 4 groups");
        // Color key is retained; each group has exactly one row.
        assert_eq!(groups[0], (sv("A"), vec![0]));
        assert_eq!(groups[1], (sv("A"), vec![1]));
        assert_eq!(groups[2], (sv("B"), vec![2]));
        assert_eq!(groups[3], (sv("B"), vec![3]));
    }

    /// FA-13 guard: no color, no detail → single group over all rows (the ribbon
    /// no-color case).
    #[test]
    fn groups_no_color_no_detail_is_single_group() {
        let groups = super::build_color_detail_groups(None, None, Vec::new(), false, 5).groups;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, None);
        assert_eq!(groups[0].1, vec![0, 1, 2, 3, 4]);
    }

    // ── dash_values subdivision (T12, spec §4.3) ────────────────────────────

    /// T12 guard: `dash_values` subdivides a color group further, retaining
    /// the color key on every subgroup — the (color, detail, dash) partition
    /// extension.
    #[test]
    fn groups_color_and_dash_subdivides_within_color_key() {
        // 4 rows, one color "A", two dash categories "solid"/"dashed" interleaved.
        let cv: Vec<Option<String>> = vec![sv("A"), sv("A"), sv("A"), sv("A")];
        let dv: Vec<Option<String>> = vec![sv("solid"), sv("dashed"), sv("solid"), sv("dashed")];
        let groups = super::build_color_detail_groups(Some(("c", &cv)), None, [("dash", &dv)], true, 4).groups;
        assert_eq!(groups.len(), 2, "one color key x two dash categories = 2 groups");
        assert_eq!(groups[0], (sv("A"), vec![0, 2]));
        assert_eq!(groups[1], (sv("A"), vec![1, 3]));
    }

    /// T12 guard: `dash_values = None` (numeric `stroke_dash` field, or none
    /// bound) leaves grouping byte-identical to the pre-T12 (color, detail)
    /// partition — no subdivision happens.
    #[test]
    fn groups_no_dash_values_is_byte_identical_to_pre_t12() {
        let cv: Vec<Option<String>> = vec![sv("A"), sv("B"), sv("A"), sv("B")];
        let groups = super::build_color_detail_groups(Some(("c", &cv)), None, Vec::new(), true, 4).groups;
        assert_eq!(groups.len(), 2, "two color keys → two groups, unaffected by absent dash_values");
        assert_eq!(groups[0], (sv("A"), vec![0, 2]));
        assert_eq!(groups[1], (sv("B"), vec![1, 3]));
    }

    /// T12 guard: dash-only (no color, no detail) partitions by dash category
    /// with the key stripped to `None` — mirrors the detail-only branch's
    /// "no legend key from this axis" contract.
    #[test]
    fn groups_dash_only_partitions_with_none_key() {
        let dv: Vec<Option<String>> = vec![sv("solid"), sv("dashed"), sv("solid")];
        let groups = super::build_color_detail_groups(None, None, [("dash", &dv)], false, 3).groups;
        assert_eq!(groups.len(), 2, "two dash categories → two groups");
        assert!(groups.iter().all(|(k, _)| k.is_none()));
        assert_eq!(groups[0].1, vec![0, 2]);
        assert_eq!(groups[1].1, vec![1]);
    }

    // ── SeriesPartition::fields (naming fields) ───────────────────────────────

    /// Guard: color-only with a scale names the color field.
    #[test]
    fn fields_color_only_with_scale_names_the_color_field() {
        let cv: Vec<Option<String>> = vec![sv("A"), sv("B")];
        let fields = super::build_color_detail_groups(Some(("c", &cv)), None, Vec::new(), true, 2).fields;
        assert_eq!(fields, vec!["c".to_string()]);
    }

    /// Guard: color present but no scale does not split the series, so it
    /// names no field — a field never named unless it actually partitioned
    /// a run.
    #[test]
    fn fields_color_without_a_scale_names_no_field() {
        let cv: Vec<Option<String>> = vec![sv("A"), sv("B")];
        let fields = super::build_color_detail_groups(Some(("c", &cv)), None, Vec::new(), false, 2).fields;
        assert_eq!(fields, Vec::<String>::new());
    }

    /// Guard: detail-only names the detail field.
    #[test]
    fn fields_detail_only_names_the_detail_field() {
        let dv: Vec<Option<String>> = vec![sv("x"), sv("y")];
        let fields = super::build_color_detail_groups(None, Some(("d", &dv)), Vec::new(), false, 2).fields;
        assert_eq!(fields, vec!["d".to_string()]);
    }

    /// Guard: color and detail together name color then detail, and
    /// each split's field follows, in the order the splits were given.
    #[test]
    fn fields_color_and_detail_name_both_then_each_split() {
        let cv: Vec<Option<String>> = vec![sv("A"), sv("B")];
        let dv: Vec<Option<String>> = vec![sv("x"), sv("y")];
        let dash: Vec<Option<String>> = vec![sv("solid"), sv("dashed")];
        let fields = super::build_color_detail_groups(Some(("c", &cv)), Some(("d", &dv)), [("dash", &dash)], true, 2).fields;
        assert_eq!(fields, vec!["c".to_string(), "d".to_string(), "dash".to_string()]);
    }

    // ── a temporal key is an identity key on every partition read ───────────

    /// Four rows of two instants (rows 0 and 1, then 2 and 3) in a `Timestamp`
    /// column `k`, beside `x` and `y`.
    fn timestamp_key_batch() -> arrow::record_batch::RecordBatch {
        use arrow::array::TimestampMillisecondArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("k", DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.0, 1.0])),
            Arc::new(TimestampMillisecondArray::from(vec![1_000i64, 1_000, 2_000, 2_000])),
        ]).unwrap()
    }

    fn rows_of(partition: &SeriesPartition) -> Vec<Vec<usize>> {
        partition.groups.iter().map(|(_, rows)| rows.clone()).collect()
    }

    /// A statistic's grouping series over a `Timestamp` column reads the
    /// column's label strings, so the series splits where two instants differ.
    #[test]
    fn grouping_series_values_read_a_timestamp_column() {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let spec = color_spec();
        let batch = timestamp_key_batch();
        let scales = scales_with_color(None);
        let mut mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        mark_style.group.series = vec!["k".to_string()];
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let values = grouping_series_values(&ctx);
        assert_eq!(values, vec![("k".to_string(), vec![sv("1970-01-01T00:00:01"), sv("1970-01-01T00:00:01"), sv("1970-01-01T00:00:02"), sv("1970-01-01T00:00:02")])]);
    }

    /// A per-row mark's `color` read splits on a `Timestamp` column under a
    /// categorical scale.
    #[test]
    fn per_row_groups_split_on_a_timestamp_color() {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mut spec = color_spec();
        spec.encoding.color = enc_field("k");
        let batch = timestamp_key_batch();
        let scales = scales_with_color(Some(ColorScale::Categorical {
            domain: vec!["1970-01-01T00:00:01".into(), "1970-01-01T00:00:02".into()],
            palette: Cow::Owned(vec![]),
        }));
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let partition = per_row_mark_groups(&ctx);
        assert_eq!(rows_of(&partition), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(partition.fields, vec!["k".to_string()]);
    }

    /// A per-row mark's `shape` read splits on a `Timestamp` column under a
    /// shape scale.
    #[test]
    fn per_row_groups_split_on_a_timestamp_shape() {
        use crate::render::scale_resolve::ShapeScale;
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mut spec = color_spec();
        spec.encoding.color = None;
        spec.encoding.shape = enc_field("k");
        let batch = timestamp_key_batch();
        let mut scales = scales_with_color(None);
        scales.shape = Some(ShapeScale { domain: vec!["1970-01-01T00:00:01".into(), "1970-01-01T00:00:02".into()], shapes: vec![] });
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let partition = per_row_mark_groups(&ctx);
        assert_eq!(rows_of(&partition), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(partition.fields, vec!["k".to_string()]);
    }

    /// A categorical `stroke_dash` read splits on a `Timestamp` column under a
    /// dash scale.
    #[test]
    fn stroke_dash_loader_reads_a_timestamp_column() {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mut spec = color_spec();
        spec.encoding.color = None;
        spec.encoding.stroke_dash = enc_field("k");
        let batch = timestamp_key_batch();
        let mut scales = scales_with_color(None);
        scales.stroke_dash = Some(StrokeDashScale { domain: vec!["1970-01-01T00:00:01".into(), "1970-01-01T00:00:02".into()], patterns: vec![vec![], vec![4.0, 4.0]] });
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        assert_eq!(
            stroke_dash_column_loader(&ctx).categorical,
            Some(vec![sv("1970-01-01T00:00:01"), sv("1970-01-01T00:00:01"), sv("1970-01-01T00:00:02"), sv("1970-01-01T00:00:02")]),
        );
    }

    /// Four rows with `x`, `y`, `x2` and `y2`, and a colour key `k` of two
    /// values (rows 0 and 1, then 2 and 3) as the given array.
    fn colored_batch(k: Arc<dyn arrow::array::Array>) -> arrow::record_batch::RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("k", k.data_type().clone(), false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![2.0, 3.0, 4.0, 5.0])),
            k,
        ]).unwrap()
    }

    /// The scene nodes `mark` draws over `batch` with `color='k'` under a
    /// categorical scale resolved elsewhere (a sibling layer's), through the
    /// one build dispatch every mark's refusal travels.
    fn build_under_sibling_scale(mark: Mark, batch: &arrow::record_batch::RecordBatch) -> Result<String, RenderError> {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mut encoding = Encoding { x: enc_field("x"), color: enc_field("k"), ..Default::default() };
        if matches!(mark, Mark::Point | Mark::Bar | Mark::Rect) {
            encoding.y = enc_field("y");
        }
        if matches!(mark, Mark::Rect) {
            encoding.x2 = enc_field("x2");
            encoding.y2 = enc_field("y2");
        }
        if matches!(mark, Mark::Segment) {
            encoding.y = enc_field("y");
            encoding.x2 = enc_field("x2");
            encoding.y2 = enc_field("y2");
        }
        let spec = ChartSpec { mark: mark.clone(), encoding, ..Default::default() };
        let scales = scales_with_color(Some(ColorScale::Categorical {
            domain: vec!["1970-01-01T00:00:01".into(), "1970-01-01T00:00:02".into()],
            palette: Cow::Owned(vec![Color::new(200, 0, 0, 255), Color::new(0, 0, 200, 255)]),
        }));
        let mark_style = resolve_mark_style(None, &theme, &mark).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch, mark_style: &mark_style };
        let built = crate::render::draw::dispatch_mark_build(&mark, &ctx)?;
        assert!(!built.nodes.is_empty(), "the mark must draw");
        Ok(format!("{:?}", built.nodes))
    }

    /// A date colour under another layer's categorical scale draws on `rule`,
    /// `tick` and `segment` exactly as the instants' label text does.
    #[test]
    fn a_timestamp_color_draws_on_rule_tick_and_segment_as_its_label_text_does() {
        use arrow::array::{StringArray, TimestampMillisecondArray};
        for mark in [Mark::Rule, Mark::Tick, Mark::Segment] {
            let date = colored_batch(Arc::new(TimestampMillisecondArray::from(vec![1_000i64, 1_000, 2_000, 2_000])));
            let twin = colored_batch(Arc::new(StringArray::from(vec!["1970-01-01T00:00:01", "1970-01-01T00:00:01", "1970-01-01T00:00:02", "1970-01-01T00:00:02"])));
            let drawn = build_under_sibling_scale(mark.clone(), &date);
            let expected = build_under_sibling_scale(mark.clone(), &twin);
            assert!(expected.is_ok(), "{mark:?}: the text twin draws: {expected:?}");
            assert_eq!(drawn.map_err(|e| e.to_string()), expected.map_err(|e| e.to_string()), "{mark:?}");
        }
    }

    // ── refuse_unreadable_detail ────────────────────────────────────────────

    /// `run` over a context for `mark` whose batch holds a column `d` of the
    /// given array, bound as `detail` where the mark takes one.
    fn run_with_detail<T>(mark: Mark, d: Arc<dyn arrow::array::Array>, run: impl Fn(&DrawCtx) -> T) -> T {
        use crate::spec::mark_style::MarkKwargsSpec;
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("d", d.data_type().clone(), false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            d,
        ]).unwrap();
        let spec = ChartSpec {
            mark: mark.clone(),
            encoding: Encoding { x: enc_field("x"), y: enc_field("y"), ..Default::default() },
            ..Default::default()
        };
        let scales = scales_with_color(None);
        let detail = mark.splits_series_by_detail().then(|| "d".to_string());
        let overrides = MarkKwargsSpec { detail, ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &mark).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        run(&ctx)
    }

    fn decimal_column() -> Arc<dyn arrow::array::Array> {
        Arc::new(arrow::array::Decimal128Array::from(vec![100_i128, 200]).with_precision_and_scale(10, 2).unwrap())
    }

    fn time_column() -> Arc<dyn arrow::array::Array> {
        Arc::new(arrow::array::Time64NanosecondArray::from(vec![1_i64, 2]))
    }

    /// A `detail` no key reader takes is refused with the reader's
    /// `UnsupportedDtype` re-voiced with the remedy, on the four marks that
    /// split by it.
    #[test]
    fn an_unreadable_detail_is_refused_on_the_marks_that_split_by_it() {
        let cases: [(Arc<dyn arrow::array::Array>, &str); 2] = [
            (decimal_column(), "Decimal128(10, 2)"),
            (time_column(), "Time64(Nanosecond)"),
        ];
        for mark in [Mark::Line, Mark::Area, Mark::Ribbon, Mark::Polygon] {
            for (column, dtype) in &cases {
                let refused = run_with_detail(mark.clone(), column.clone(), |ctx| refuse_unreadable_detail(ctx));
                assert_eq!(
                    refused,
                    Err(RenderError::UnsupportedDtype {
                        field: "d".into(),
                        dtype: format!("{dtype} (it cannot key a series; convert 'd' to strings to split the mark by it)"),
                        context: None,
                    }),
                    "{mark:?} {dtype}"
                );
            }
        }
    }

    /// A readable `detail` (a temporal or text column) is never refused.
    #[test]
    fn a_readable_detail_is_not_refused() {
        use arrow::array::TimestampMillisecondArray;
        for mark in [Mark::Line, Mark::Area, Mark::Ribbon, Mark::Polygon] {
            let date: Arc<dyn arrow::array::Array> = Arc::new(TimestampMillisecondArray::from(vec![1_000_i64, 2_000]));
            let text: Arc<dyn arrow::array::Array> = Arc::new(StringArray::from(vec!["a", "b"]));
            assert_eq!(run_with_detail(mark.clone(), date, |ctx| refuse_unreadable_detail(ctx)), Ok(()), "{mark:?}");
            assert_eq!(run_with_detail(mark, text, |ctx| refuse_unreadable_detail(ctx)), Ok(()));
        }
    }

    /// What a series mark names as not drawn when its `detail` is a
    /// `Timestamp` and one instant's run is too short: a first instant with
    /// three rows and a second (2024-01-02) with `short_rows`.
    fn dropped_under_a_temporal_detail(mark: Mark, short_rows: usize) -> Vec<String> {
        use arrow::array::TimestampMillisecondArray;
        use crate::spec::mark_style::MarkKwargsSpec;
        let n = 3 + short_rows;
        let day_one = 1_704_067_200_000_i64;
        let day_two = 1_704_153_600_000_i64;
        let keys: Vec<i64> = (0..n).map(|i| if i < 3 { day_one } else { day_two }).collect();
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("k", DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs.clone())),
            Arc::new(Float64Array::from(xs.iter().map(|x| x * 2.0 + (*x % 2.0)).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(xs.iter().map(|x| x * 2.0 + 5.0).collect::<Vec<_>>())),
            Arc::new(TimestampMillisecondArray::from(keys)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let spec = ChartSpec {
            mark: mark.clone(),
            encoding: Encoding {
                x: enc_field("x"),
                y: enc_field("y"),
                y2: matches!(mark, Mark::Ribbon).then(|| enc_field("y2")).flatten(),
                ..Default::default()
            },
            ..Default::default()
        };
        let (mut scales, _) = crate::render::scale_resolve::resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        scales.color = None;
        let overrides = MarkKwargsSpec { detail: Some("k".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &mark).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let built = crate::render::draw::dispatch_mark_build(&mark, &ctx).unwrap();
        assert_eq!(built.nodes.len(), 1, "{mark:?}: the three-row instant draws");
        built.dropped_series.expect("the short run is named").names().into_iter().map(String::from).collect()
    }

    /// A short series or ring under a temporal `detail` is named by its
    /// instant as an ISO date-time, in the caller's terms.
    #[test]
    fn a_short_run_under_a_temporal_detail_is_named_by_its_instant() {
        for mark in [Mark::Line, Mark::Area, Mark::Ribbon] {
            assert_eq!(dropped_under_a_temporal_detail(mark.clone(), 1), vec!["k=2024-01-02T00:00:00 (one point)"], "{mark:?}");
        }
        assert_eq!(dropped_under_a_temporal_detail(Mark::Polygon, 2), vec!["k=2024-01-02T00:00:00 (two vertices)"]);
    }

    /// The dispatch refuses an unreadable `detail` on the four marks that split
    /// by it, with the re-voiced error, and not on a mark that ignores `detail`.
    /// With the call removed, each of the four draws a merged series and returns
    /// `Ok`.
    #[test]
    fn the_build_dispatch_refuses_an_unreadable_detail_on_every_mark_that_splits_by_it() {
        let dispatch = |mark: Mark| {
            let m = mark.clone();
            run_with_detail(mark, decimal_column(), move |ctx| {
                crate::render::draw::dispatch_mark_build(&m, ctx).map(|_| ())
            })
        };
        for mark in [Mark::Line, Mark::Area, Mark::Ribbon, Mark::Polygon] {
            assert_eq!(
                dispatch(mark.clone()),
                Err(RenderError::UnsupportedDtype {
                    field: "d".into(),
                    dtype: "Decimal128(10, 2) (it cannot key a series; convert 'd' to strings to split the mark by it)".into(),
                    context: None,
                }),
                "{mark:?}"
            );
        }
        for mark in [Mark::Point, Mark::Bar, Mark::Rule, Mark::Tick, Mark::Rect, Mark::Text] {
            assert!(!matches!(dispatch(mark.clone()), Err(RenderError::UnsupportedDtype { .. })), "{mark:?} ignores detail");
        }
    }

    /// A `detail` column the batch does not carry reads as "no key": the reader
    /// errors, the guard does not refuse.
    #[test]
    fn a_missing_detail_column_is_not_refused() {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let batch = timestamp_key_batch();
        let mut spec = color_spec();
        spec.mark = Mark::Line;
        let scales = scales_with_color(None);
        let overrides = crate::spec::mark_style::MarkKwargsSpec { detail: Some("absent".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        assert_eq!(refuse_unreadable_detail(&ctx), Ok(()));
        assert!(col_as_identity_key_str(ctx.batch, "absent").is_err());
    }

    /// The per-row colour read every other mark shares keys a date colour by
    /// its label under a categorical scale a sibling layer establishes:
    /// each key maps through the scale, as the instants' label text does.
    #[test]
    fn a_timestamp_color_maps_each_key_through_the_scale_on_point_bar_and_rect() {
        use arrow::array::{StringArray, TimestampMillisecondArray};
        for mark in [Mark::Point, Mark::Bar, Mark::Rect] {
            let date = colored_batch(Arc::new(TimestampMillisecondArray::from(vec![1_000i64, 1_000, 2_000, 2_000])));
            let twin = colored_batch(Arc::new(StringArray::from(vec!["1970-01-01T00:00:01", "1970-01-01T00:00:01", "1970-01-01T00:00:02", "1970-01-01T00:00:02"])));
            let drawn = build_under_sibling_scale(mark.clone(), &date).map_err(|e| e.to_string());
            let expected = build_under_sibling_scale(mark.clone(), &twin).map_err(|e| e.to_string());
            assert!(expected.is_ok(), "{mark:?}: the text twin draws: {expected:?}");
            assert_eq!(drawn, expected, "{mark:?}");
        }
    }

    /// `arc` and every consumer of `color_column_loader` read the same
    /// per-row keys: a `Timestamp` column under a categorical scale loads as
    /// its label strings.
    #[test]
    fn the_color_loader_keys_a_timestamp_column_by_its_label() {
        use arrow::array::TimestampMillisecondArray;
        let batch = colored_batch(Arc::new(TimestampMillisecondArray::from(vec![1_000i64, 1_000, 2_000, 2_000])));
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mut spec = color_spec();
        spec.encoding.color = enc_field("k");
        let scales = scales_with_color(Some(ColorScale::Categorical {
            domain: vec!["1970-01-01T00:00:01".into(), "1970-01-01T00:00:02".into()],
            palette: Cow::Owned(vec![Color::new(200, 0, 0, 255), Color::new(0, 0, 200, 255)]),
        }));
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let (categories, numeric) = color_column_loader(&ctx);
        assert_eq!(categories, Some(vec![sv("1970-01-01T00:00:01"), sv("1970-01-01T00:00:01"), sv("1970-01-01T00:00:02"), sv("1970-01-01T00:00:02")]));
        assert!(numeric.is_none());
    }
}
