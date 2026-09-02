//! Shared per-mark channel loaders (C9).
//!
//! Two patterns recurred verbatim across the mark builders and are unified here:
//!
//! - [`color_column_loader`] — the `(scale_kind, field) → (categorical, numeric)`
//!   split that `point` and `bar` both performed inline. The categorical string
//!   column is read for `Categorical` *and* scale-less charts; the numeric column
//!   for `Continuous` charts. This is byte-identical to `point`'s former inline
//!   `(color_values_str, color_values_f64)` block and to `bar`'s prior
//!   `load_color_columns` helper. (`rect` deliberately does **not** use this: its
//!   three draw paths each load a different subset and omit the scale-less
//!   categorical branch, so unifying it would change `rect`'s behavior — see the
//!   note on that mark.)
//!
//! - [`polar_channel_resolver`] — the theta→angle / radius→radial channel mapping
//!   under the Python polar remapping, shared by `arc` and `bar::build_polar`.
//!   `theta="x"` puts the angular channel on `x` and the radial channel on `y`;
//!   `theta="y"` mirrors. Byte-identical to both marks' prior inline `match`.
//!
//! - [`band_extent_or`] — the band-geometry-unification consumer pattern (GH
//!   #39 phase 2): bar/box/heatmap/tick width and extent formulas divide by a
//!   panel-extent term (`panel.w` / `panel.h`) that must instead honor an
//!   explicit `BandScale`/`PointScale`/positional-`OrdinalScale` pixel range
//!   when the resolver recorded one. Ten call sites across `bar`, `rect`, and
//!   `tick` repeat this exact substitution, so it is unified here rather than
//!   left as ten copies of the same `.map(f64::abs).unwrap_or(..)` line.

use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_str, color_field, resolve_stroke_dash, DrawCtx};
use crate::render::scale_resolve::{ColorInput, ColorScale, ResolvedScales, ScaleKind, StrokeDashScale};
use crate::spec::coord::PolarThetaChannel;
use crate::spec::encoding::Encoding;

/// `(categorical_string_column, numeric_column)` for the color encoding.
pub(crate) type ColorColumns = (Option<Vec<Option<String>>>, Option<Vec<Option<f64>>>);

/// Band-pixel extent for a mark's width/size/extent formula (band-geometry
/// unification design §6 consumer contract): `scale`'s explicit range extent
/// when the resolver recorded one, otherwise `fallback` — the exact panel-
/// extent expression (`panel.w` / `panel.h`) each call site used before this
/// unification, so the no-range path stays byte-identical (§7 invariant).
///
/// `scale` must be the resolved scale on the *same axis* as `fallback`'s panel
/// dimension (x-axis scale with `panel.w`, y-axis scale with `panel.h`).
/// [`ScaleKind::explicit_band_extent`] returns `None` for every non-ordinal
/// scale, so passing a scale that carries no band range (including the dummy
/// unit scale synthesized for an absent axis) is safe: this always falls
/// through to `fallback`.
pub(crate) fn band_extent_or(scale: &ScaleKind, fallback: f64) -> f64 {
    scale.explicit_band_extent().map(f64::abs).unwrap_or(fallback)
}

/// Load the per-row color-encoding columns for fill resolution (C9).
///
/// Mirrors `point`/`bar`: the categorical string column is read for
/// [`ColorInput::Category`] (and scale-less) charts; the numeric column for
/// [`ColorInput::Numeric`] ones. The numeric branch reads `col_as_f64` so the
/// caller can sample via `lookup_f64` without an `f64 → String → f64`
/// round-trip.
pub(crate) fn color_column_loader(ctx: &DrawCtx) -> ColorColumns {
    let field = color_field(ctx, ctx.spec);
    let input = ctx.scales.color.as_ref().map(ColorScale::input);
    let categorical = match (input, field) {
        (Some(ColorInput::Category) | None, Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let numeric = match (input, field) {
        (Some(ColorInput::Numeric), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    (categorical, numeric)
}

/// Per-row `stroke_dash` channel columns, loaded once per mark build (T12).
///
/// Mirrors [`color_column_loader`]'s categorical/numeric split, gated on
/// [`ResolvedScales::stroke_dash`] instead of the color scale's
/// [`ColorInput`]: a *categorical* field (scale resolved) loads the
/// ordinal-category string column, so [`resolve_row_stroke_dash`] can look
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
        (true, Some(f)) => col_as_ordinal_category_str(ctx.batch, f).ok(),
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
/// (a null category). Numeric rows keep the pre-T12 contract byte-identically:
/// `resolve_stroke_dash` on a finite index, falling back to `base` for a
/// null/non-finite value.
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
        .and_then(resolve_stroke_dash)
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
/// All three (area, line, ribbon) load color via `col_as_ordinal_category_str`
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
/// After the (color, detail) split above, `dash_values` (T12, spec §4.3)
/// subdivides each resulting group further by per-row stroke-dash category —
/// the series partition key extends from (color, detail) to (color, detail,
/// stroke_dash-field) so a categorical `stroke_dash` encoding draws one
/// polyline (or, for `ribbon`, one closed-polygon band) per distinct style
/// even within a single color/detail group. `None` (a numeric `stroke_dash`
/// field, or none at all) skips this step entirely, which is the pre-T12
/// grouping, byte-identical. `line` and `ribbon` (T12 fix round, spec §4.3
/// amended 2026-09-01) both pass `dash_cols.categorical.as_ref()`, so a
/// categorical `stroke_dash` field subdivides both; `area`'s caller still
/// always passes `None` since it does not consume the `stroke_dash` channel
/// today.
///
/// # Arguments
/// - `color_values`: optional per-row color strings (loaded by the caller).
/// - `detail_values`: optional per-row detail strings (loaded by the caller).
/// - `dash_values`: optional per-row stroke-dash category strings (loaded by
///   the caller from [`DashColumns::categorical`](DashColumns), only when a
///   categorical `stroke_dash` scale resolved).
/// - `has_color_scale`: whether a resolved color scale is present; gates the
///   color-only branch (matches `&ctx.scales.color` being `Some(_)`).
/// - `n_rows`: total row count, used for the fallback all-rows group.
///
/// # Returns
/// `Vec<(Option<String>, Vec<usize>)>` — one entry per group. The `Option<String>`
/// is the color legend key for that group (`None` in the detail-only and
/// fallback cases); a dash-subdivided group repeats its parent's key.
pub(crate) fn build_color_detail_groups(
    color_values: Option<&Vec<Option<String>>>,
    detail_values: Option<&Vec<Option<String>>>,
    dash_values: Option<&Vec<Option<String>>>,
    has_color_scale: bool,
    n_rows: usize,
) -> Vec<(Option<String>, Vec<usize>)> {
    let groups = match (color_values, detail_values, has_color_scale) {
        // Color only: one group per color category; key retained for legend.
        (Some(cv), None, true) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in cv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups
        }
        // Detail only: one group per detail value; key stripped so fill falls
        // to the mark-style default (no per-group color from the legend).
        (None, Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in dv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups.into_iter().map(|(_, rows)| (None, rows)).collect()
        }
        // Both color and detail: one group per (color, detail) pair.
        (Some(cv), Some(dv), _) => {
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
            groups
        }
        // No color, no detail (or color present but no scale): single group.
        _ => vec![(None, (0..n_rows).collect())],
    };
    match dash_values {
        Some(dv) => subdivide_by_dash(groups, dv),
        None => groups,
    }
}

/// Subdivide each `(color_key, rows)` group by per-row stroke-dash category
/// (T12), preserving the parent's color key on every subgroup. Rows within a
/// parent group are re-partitioned in first-encounter order of their dash
/// category, so subgroup ordering — and therefore the node/metadata alignment
/// invariant (#6) — stays deterministic.
fn subdivide_by_dash(
    groups: Vec<(Option<String>, Vec<usize>)>,
    dash_values: &[Option<String>],
) -> Vec<(Option<String>, Vec<usize>)> {
    let mut out = Vec::with_capacity(groups.len());
    for (key, rows) in groups {
        let mut sub: Vec<(Option<String>, Vec<usize>)> = Vec::new();
        for i in rows {
            let dash_key = dash_values.get(i).cloned().flatten();
            match sub.iter().position(|(dk, _)| dk == &dash_key) {
                Some(p) => sub[p].1.push(i),
                None => sub.push((dash_key, vec![i])),
            }
        }
        out.extend(sub.into_iter().map(|(_, sub_rows)| (key.clone(), sub_rows)));
    }
    out
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
            x2: None,
            y2: None,
            y_slots: Default::default(),
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
    use crate::spec::data_ref::DataRef;
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::borrow::Cow;
    use std::sync::Arc;

    fn color_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: enc_field("x"),
                y: enc_field("y"),
                color: enc_field("c"),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
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
    ) -> ColorColumns {
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
        let groups = super::build_color_detail_groups(Some(&cv), None, None, true, 4);
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
        let groups = super::build_color_detail_groups(Some(&cv), None, None, false, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, None);
        assert_eq!(groups[0].1, vec![0, 1, 2]);
    }

    /// FA-13 guard: detail-only partitions by detail value and strips the key
    /// to None so callers use their theme-default color (no legend).
    #[test]
    fn groups_detail_only_strips_key_to_none() {
        let dv: Vec<Option<String>> = vec![sv("s0"), sv("s1"), sv("s0"), sv("s1")];
        let groups = super::build_color_detail_groups(None, Some(&dv), None, false, 4);
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
        let groups = super::build_color_detail_groups(Some(&cv), Some(&dv), None, true, 4);
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
        let groups = super::build_color_detail_groups(None, None, None, false, 5);
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
        let groups = super::build_color_detail_groups(Some(&cv), None, Some(&dv), true, 4);
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
        let groups = super::build_color_detail_groups(Some(&cv), None, None, true, 4);
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
        let groups = super::build_color_detail_groups(None, None, Some(&dv), false, 3);
        assert_eq!(groups.len(), 2, "two dash categories → two groups");
        assert!(groups.iter().all(|(k, _)| k.is_none()));
        assert_eq!(groups[0].1, vec![0, 2]);
        assert_eq!(groups[1].1, vec![1]);
    }
}
