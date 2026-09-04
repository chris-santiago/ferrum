//! Positional scale resolution: build x/y axis scales from encoding specs.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use crate::scale::linear::LinearScale;
use crate::scale::log::LogScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::pow::PowScale;
use crate::scale::symlog::SymlogScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::{RenderError, RenderWarning};

use super::domain::{
    apply_sort_to_domain, distinct_positional_categories_shared, locate_field,
    numeric_domain_union, SortContext,
};
use super::{
    column_min_max_f64, distinct_positional_categories, infer_spec_type, ScaleKind, SharedDomain,
};
use crate::transform::core::FINAL_OUTPUT_KEY;

/// The x/y field names bound at chart level, used to resolve data-aware sort
/// forms (channel shorthand `"-y"` etc.). Threaded into `build_axis_scale` so an
/// ordinal axis can sort its categories by the aggregate of the opposite
/// channel's field. Either side may be `None` (single-axis Tick/Rule marks).
#[derive(Clone, Copy, Default)]
pub(in crate::render) struct PositionalFields<'a> {
    pub(in crate::render) x: Option<&'a str>,
    pub(in crate::render) y: Option<&'a str>,
}

// The padding-inset constants and `inset_pixel_range` now live in
// `crate::layout::geometry` (the geometry layer) so that the layout engine can
// reproduce the *exact same* inset when projecting axis ticks. Re-export the
// helper here under its existing `pub(in crate::render)` path so render-side
// callers are unchanged.
use crate::layout::geometry::DEFAULT_SCALE_PADDING_FRAC;
pub(in crate::render) use crate::layout::geometry::inset_pixel_range;

/// Resolve the effective padding fraction.
///
/// Precedence:
///  - `Some(p)` → use `p` (including 0.0 to disable).
///  - `None && has_explicit_domain` → 0.0 (user-specified domain wins).
///  - `None && !has_explicit_domain` → `DEFAULT_SCALE_PADDING_FRAC`.
pub(in crate::render) fn resolve_padding_fraction(scale_padding: Option<f64>, has_explicit_domain: bool) -> f64 {
    if let Some(p) = scale_padding {
        return p;
    }
    if has_explicit_domain {
        return 0.0;
    }
    DEFAULT_SCALE_PADDING_FRAC
}

#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    positional_fields: PositionalFields<'_>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    pixel_range: (f64, f64),
    spec: &ChartSpec,
    // Faceted + this channel resolves `ResolveMode::Shared`: the auto-inferred
    // domain also unions the global all-panels batch (`FINAL_OUTPUT_KEY`) so a
    // faceted raw field's positional domain spans every panel (T4). `false` for
    // non-faceted charts and `Independent`-mode channels → byte-identical
    // per-panel behavior. Ignored on the explicit `enc.scale` bypass.
    include_final: bool,
    // Composite-shared domain for this channel (D4b). `Some` only for a
    // composite leaf whose channel shares a domain across the tree; it seeds this
    // leaf's extent (numeric) or category vector (ordinal) on the SAME auto path
    // facet-shared panels use, so composite-shared axes get the auto path's
    // padding/`nice`, never the explicit-scale bypass. `None` for every
    // standalone (flat/facet) render → byte-identical. Consulted only after the
    // explicit `enc.scale` bypass below, so a genuine user scale still wins.
    shared: Option<&SharedDomain>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<ScaleKind, RenderError> {
    let located = locate_field(&enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let dtype = infer_spec_type(enc, located.col.data_type());
    let pr = axis_pixel_range(channel, &dtype, pixel_range);

    if let Some(scale_spec) = &enc.scale {
        let sort_ctx = SortContext {
            category_field: &enc.field,
            batch: located.batch,
            x_field: positional_fields.x,
            y_field: positional_fields.y,
        };
        return build_from_scale_spec(scale_spec, enc, located.batch, pr, &sort_ctx, warnings);
    }

    let inset = inset_pixel_range(pr, resolve_padding_fraction(None, false));

    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = resolve_numeric_extent(
                shared, channel, enc, paired_enc, primary_batch, transform_outputs, spec, include_final,
            )?;
            Ok(ScaleKind::Linear(LinearScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            // Composite-shared ordinal (D4b): the seeded union vector arrives
            // ALREADY ordered (the resolve pass's order-preserving union, D2)
            // and is authoritative — local data-aware re-sorting is skipped,
            // because each leaf would compute the sort aggregate from its OWN
            // batch and different leaves could reorder the "shared" domain
            // differently (there is no cross-leaf merged batch to sort on,
            // unlike facet-shared's FINAL_OUTPUT_KEY). Falls back to this
            // leaf's own categories + normal sort when no composite sharing
            // applies.
            let mut domain = match shared {
                Some(SharedDomain::Ordinal(cats)) => {
                    return Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(
                        cats.clone(), vec![pr.0, pr.1], 0.0,
                    )));
                }
                _ => distinct_positional_categories_shared(
                    located.batch, &enc.field, transform_outputs, include_final,
                )?,
            };
            // For data-aware sort forms (`"-y"`, `"y"`, `{field, op, order}`),
            // `apply_sort_to_domain` computes a per-category aggregate from
            // `sort_ctx.batch`.  When the channel resolves Shared (`include_final`),
            // using the per-panel batch causes each panel to re-sort the shared
            // category vector by its OWN aggregate, placing marks under the wrong
            // tick.  Mirror `distinct_positional_categories_shared`: use the global
            // batch (`FINAL_OUTPUT_KEY`) so every panel's data-aware sort agrees
            // with the global/provisional axis order.  Fall back to the per-panel
            // batch when the global batch is absent (defensive, same as the
            // membership helper).  For Independent channels (`include_final == false`)
            // keep the per-panel batch so that escape hatch remains byte-identical.
            let sort_batch = if include_final {
                transform_outputs
                    .get(FINAL_OUTPUT_KEY)
                    .unwrap_or(located.batch)
            } else {
                located.batch
            };
            let sort_ctx = SortContext {
                category_field: &enc.field,
                batch: sort_batch,
                x_field: positional_fields.x,
                y_field: positional_fields.y,
            };
            apply_sort_to_domain(&mut domain, enc.sort.as_ref(), &sort_ctx, warnings);
            Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(
                domain, vec![pr.0, pr.1], 0.0,
            )))
        }
        SpecDataType::Temporal => {
            let (min, max) = resolve_numeric_extent(
                shared, channel, enc, paired_enc, primary_batch, transform_outputs, spec, include_final,
            )?;
            Ok(ScaleKind::Time(TimeScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
    }
}

/// Resolve a numeric/temporal axis extent: the composite shared domain (D4b) when
/// the leaf participates in composite sharing on this channel, otherwise the
/// leaf's own facet-union-aware auto-inferred extent. Consulted on the AUTO path
/// only (the explicit-scale bypass returns before reaching here), so the shared
/// extent receives the same padding/`nice` treatment as a facet-shared panel. A
/// non-numeric `shared` (never produced for a numeric leaf by the resolve pass)
/// falls through to the union, so this is a safe no-op when `shared` is `None`.
#[allow(clippy::too_many_arguments)]
fn resolve_numeric_extent(
    shared: Option<&SharedDomain>,
    channel: &str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
) -> Result<(f64, f64), RenderError> {
    if let Some(SharedDomain::Numeric { lo, hi }) = shared {
        return Ok((*lo, *hi));
    }
    numeric_domain_union(
        channel,
        &enc.field,
        paired_enc.map(|p| p.field.as_str()),
        primary_batch,
        transform_outputs,
        spec,
        include_final,
    )
}

/// Y-axis pixel range is inverted ONLY for quantitative/temporal scales.
pub(in crate::render) fn axis_pixel_range(channel: &str, dtype: &SpecDataType, pixel_range: (f64, f64)) -> (f64, f64) {
    if channel == "y" && !matches!(dtype, SpecDataType::Ordinal | SpecDataType::Nominal) {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    }
}

/// Build a ScaleKind from an explicit ScaleSpec, using the given pixel range.
pub(in crate::render) fn build_from_scale_spec(
    scale_spec: &crate::spec::encoding::ScaleSpec,
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    pr: (f64, f64),
    sort_ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<ScaleKind, RenderError> {
    use crate::spec::encoding::ScaleSpec;

    let col = batch
        .column_by_name(&enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;

    Ok(match scale_spec {
        ScaleSpec::Linear { common, nice, zero } => {
            let (mut d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            if *zero && d.len() == 2 {
                if d[0] > 0.0 { d[0] = 0.0; }
                if d[1] < 0.0 { d[1] = 0.0; }
            }
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Linear(LinearScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Log { base, common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Log(LogScale::new_internal(d, r, *base, common.clamp, *nice))
        }
        ScaleSpec::Time { common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Time(TimeScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Symlog { constant, common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Symlog(SymlogScale::new_internal(d, r, *constant, common.clamp, *nice))
        }
        ScaleSpec::Ordinal { domain, range, padding } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
            // Extract numeric pixel range from the typed polymorphic `range`.
            // String values (color ranges) are handled by `build_color_scale` —
            // fall back to the pixel-range default when the range has no numbers.
            let (pixel_range, explicit) = ordinal_pixel_range(range.as_deref(), pr);
            ScaleKind::Ordinal(
                OrdinalScale::new_internal(d, pixel_range, *padding)
                    .with_explicit_range(explicit),
            )
        }
        ScaleSpec::Pow { exponent, common } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Pow(PowScale::new_internal(d, r, *exponent, common.clamp))
        }
        ScaleSpec::Sqrt { common } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Pow(PowScale::new_internal(d, r, 0.5, common.clamp))
        }
        ScaleSpec::Utc { common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            let d = apply_domain_reverse(d, common.reverse);
            ScaleKind::Time(TimeScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Band { domain, padding, padding_inner, range, .. } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
            let effective_padding = padding_inner.unwrap_or(*padding);
            let (band_range, explicit) = band_point_pixel_range(range.as_deref(), pr);
            ScaleKind::Ordinal(
                OrdinalScale::new_internal(d, band_range, effective_padding)
                    .with_explicit_range(explicit),
            )
        }
        ScaleSpec::Point { domain, padding, range, reverse, .. } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
            // PointScale(reverse=True) reverses the resolved domain order (GH #65).
            // Domain reversal — not a pixel-range flip — so axis tick labels
            // (domain-order via tick_labels/uniform_center) and explicit-range band
            // centers follow the marks automatically. Within this resolver's
            // symmetric band model ((i + 0.5) * step centers), reversing the domain
            // is the same transform as mirroring each center about the range
            // midpoint — the reverse semantic of the pyclass facade
            // (scale/point.rs). Only the reverse *transform* is equivalent: the
            // facade's BASE positions use the true point formula
            // (extent/(n-1+2p), endpoints at the range edges) and do NOT match
            // this band model — that divergence is the #67 north star. Applied
            // after sort: "sort, then reverse", matching d3/Vega composition.
            if *reverse {
                d.reverse();
            }
            let (point_range, explicit) = band_point_pixel_range(range.as_deref(), pr);
            ScaleKind::Ordinal(
                OrdinalScale::new_internal(d, point_range, *padding)
                    .with_explicit_range(explicit),
            )
        }
        // Continuous-degrading variants: whether the spec's `domain` is a
        // positional extent or a discrete-binning artifact is classified by
        // ScaleSpec::positional_extent() (exhaustive — see issue #40).
        //
        // These discretizing/diverging positional scales degrade to
        // `ScaleKind::Linear` before minor-tick generation is reached, so
        // they receive linear-subdivided minor ticks (semantic corner,
        // archaeology R7).
        ScaleSpec::Sequential { .. }
        | ScaleSpec::Diverging { .. }
        | ScaleSpec::Quantize { .. }
        | ScaleSpec::BinOrdinal { .. }
        | ScaleSpec::Quantile { .. }
        | ScaleSpec::Threshold { .. } => {
            let (d, r) = resolve_continuous_domain_and_range(&scale_spec.positional_extent(), &None, None, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Linear(LinearScale::new_internal(d, r, false, false))
        }
    })
}

/// Resolve the pixel-coordinate range for a `ScaleSpec::Ordinal`, plus
/// whether that range counts as **explicit** for band-geometry purposes
/// (GH #39 phase 2): `true` only when at least 2 numeric entries were found,
/// mirroring `band_point_pixel_range`'s `>= 2`-entry gate so Band, Point, and
/// positional Ordinal scales are never special-cased against each other.
///
/// The typed `range` carries `Number` entries (pixel coordinates) and/or `Str`
/// entries (color strings, handled by `build_color_scale`). This helper pulls
/// the numbers for the positional resolver. When `range` is absent or carries
/// no numbers (i.e. an all-string color range), the plot-area extent
/// `[pr.0, pr.1]` is returned so the positional scale still functions. A
/// single numeric entry is passed through unchanged (pre-existing arithmetic,
/// untouched here) but does not count as explicit — `explicit_band_extent()`
/// requires 2 entries to compute a signed extent.
fn ordinal_pixel_range(
    range: Option<&[crate::scale::ordinal::OrdinalRangeValue]>,
    pr: (f64, f64),
) -> (Vec<f64>, bool) {
    match range {
        Some(values) => {
            let nums = crate::scale::ordinal::OrdinalRangeValue::numbers(values);
            if nums.is_empty() {
                (vec![pr.0, pr.1], false)
            } else {
                let explicit = nums.len() >= 2;
                (nums, explicit)
            }
        }
        None => (vec![pr.0, pr.1], false),
    }
}

/// Resolve the pixel-coordinate range for a `ScaleSpec::Band`/`ScaleSpec::Point`,
/// plus whether that range was explicitly supplied by the user (GH #39
/// phase 2): `true` only for a usable (>= 2-entry) numeric range; the
/// panel-extent fallback is never explicit.
///
/// Explicit numeric pixel range for band/point scales; falls back to the
/// panel extent when absent or degenerate (fewer than 2 entries).
fn band_point_pixel_range(range: Option<&[f64]>, pr: (f64, f64)) -> (Vec<f64>, bool) {
    match range {
        Some(r) if r.len() >= 2 => (vec![r[0], r[1]], true),
        _ => (vec![pr.0, pr.1], false),
    }
}

/// Resolve `(domain, range)` for a continuous ScaleSpec variant.
fn resolve_continuous_domain_and_range(
    domain: &Option<Vec<f64>>,
    range: &Option<Vec<f64>>,
    padding: Option<f64>,
    col: &dyn Array,
    field: &str,
    pr: (f64, f64),
) -> Result<(Vec<f64>, Vec<f64>), RenderError> {
    let d = match domain {
        Some(d) => d.clone(),
        None => {
            let (mn, mx) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
                field: field.to_string(),
                dtype: format!("{:?}", col.data_type()),
                context: Some("scale"),
            })?;
            vec![mn, mx]
        }
    };
    let r = if let Some(r) = range {
        r.clone()
    } else {
        let frac = resolve_padding_fraction(padding, domain.is_some());
        let inset = inset_pixel_range(pr, frac);
        vec![inset.0, inset.1]
    };
    Ok((d, r))
}

/// Domain-swap sugar for `ContinuousScaleCommon::reverse` (F-L04-07, spec
/// §4C): swaps the resolved domain pair. For an explicit `domain=[a, b]`
/// with `zero=false` (the only case where no other step reorders the pair),
/// this is exactly equivalent to writing `domain=[b, a]` by hand — see
/// `ContinuousScaleCommon::reverse`'s doc for the two cases where it is not
/// (auto-inferred domain's padding inset; `zero=true`'s degenerate-collapse
/// hazard covered next).
///
/// Called once per continuous variant arm in `build_from_scale_spec`, as the
/// LAST step of that arm's domain resolution — after
/// `resolve_continuous_domain_and_range` and after any per-variant domain
/// shaping (Linear's `zero` extension, immediately above in the `Linear`
/// arm). This ordering is load-bearing, and the reason is NOT that it
/// matches a hand-written descending domain — it does not: `domain=[100,
/// 0]` with `zero=true` collapses to `[0.0, 0.0]` today (pre-existing,
/// independent of `reverse`), through this exact same `Linear` arm, because
/// the `zero` block treats `d[0]` as the minimum (`if d[0] > 0.0 { d[0] =
/// 0.0; }`, `if d[1] < 0.0 { d[1] = 0.0; }`), an assumption only an
/// ASCENDING pair satisfies. The real reason to reverse LAST is that
/// reversing FIRST would hand that same ascending-pair assumption a
/// descending pair instead, silently corrupting the result — NOT
/// necessarily to `[0.0, 0.0]` (the exact wrong value depends on the
/// domain: an auto-inferred `[20, 80]` reversed-then-zeroed comes out
/// `[0.0, 20.0]`, not `[0.0, 0.0]`, and not the correct `[80.0, 0.0]`
/// either) — see
/// `linear_zero_true_reverse_true_yields_non_degenerate_descending_domain`
/// below, which pins the CORRECT ordering by failing (on the concrete
/// `[0.0, 20.0]` wrong value above) if the swap is hoisted above the `zero`
/// block.
///
/// Range orientation, the structural y-inversion predicate
/// (`axis_pixel_range`), and tick label/fraction pairing are untouched —
/// this only flips which value lands at each end of the domain pair.
/// Downstream (`apply_axis_domain_config`'s lo>hi normalize/restore, tick
/// generation, mark positioning) already tolerates a descending domain
/// (batch-B's domain-config cascade proved this), so no other call site
/// needs to change. A no-op on anything but a 2-element domain, mirroring
/// the `zero` guard in the Linear arm above.
fn apply_domain_reverse(mut d: Vec<f64>, reverse: bool) -> Vec<f64> {
    if reverse && d.len() == 2 {
        d.swap(0, 1);
    }
    d
}

/// Apply coord-level domain and padding overrides to the x/y scales.
pub(in crate::render) fn apply_coord_domain_overrides(
    spec: &ChartSpec,
    x: &mut ScaleKind,
    y: &mut ScaleKind,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
) {
    use crate::spec::coord::CoordKind;

    let (x_domain_override, y_domain_override, expand) = match &spec.coord {
        Some(CoordKind::Cartesian { x_domain, y_domain, expand, .. }) => {
            (*x_domain, *y_domain, *expand)
        }
        Some(CoordKind::Fixed { x_domain, y_domain, expand, .. }) => {
            (*x_domain, *y_domain, *expand)
        }
        _ => return,
    };

    if let Some((lo, hi)) = x_domain_override {
        let pr = if expand {
            inset_pixel_range(x_pixel_range, resolve_padding_fraction(None, false))
        } else {
            x_pixel_range
        };
        *x = ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi], vec![pr.0, pr.1], false, false,
        ));
    } else if !expand {
        if let ScaleKind::Linear(ref inner) = x {
            let d = inner.domain_pair().to_vec();
            *x = ScaleKind::Linear(LinearScale::new_internal(
                d, vec![x_pixel_range.0, x_pixel_range.1], false, false,
            ));
        }
    }

    if let Some((lo, hi)) = y_domain_override {
        let pr = if expand {
            inset_pixel_range(y_pixel_range, resolve_padding_fraction(None, false))
        } else {
            y_pixel_range
        };
        *y = ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi], vec![pr.1, pr.0], false, false,
        ));
    } else if !expand {
        if let ScaleKind::Linear(ref inner) = y {
            let d = inner.domain_pair().to_vec();
            *y = ScaleKind::Linear(LinearScale::new_internal(
                d, vec![y_pixel_range.1, y_pixel_range.0], false, false,
            ));
        }
    }
}

/// One axis's chart-level scale-domain configuration (`configure_axis(
/// domain_min=, domain_max=, nice=, zero=)` and its `axis_x`/`axis_y`/
/// `axis_y2` spellings), in the neutral value vocabulary this module speaks.
///
/// `ChartConfig` itself stays out of `scale_resolve` per the layering the
/// seam doc records (`seam.rs`: `render::mod` and `scene_build` call DOWN into
/// the scale engine, never the reverse), so the caller extracts this and the
/// engine never learns the config's shape.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(in crate::render) struct AxisDomainConfig {
    pub(in crate::render) min: Option<f64>,
    pub(in crate::render) max: Option<f64>,
    pub(in crate::render) nice: Option<bool>,
    pub(in crate::render) zero: Option<bool>,
}

impl AxisDomainConfig {
    /// Whether the caller asked for anything at all. `false` → applying this
    /// config is a guaranteed no-op, which is what keeps every chart that
    /// doesn't use these fields byte-identical.
    pub(in crate::render) fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The field names the caller set, for the wrong-surface warning.
    fn named_fields(&self) -> Vec<String> {
        [
            ("domain_min", self.min.is_some()),
            ("domain_max", self.max.is_some()),
            ("nice", self.nice.is_some()),
            ("zero", self.zero.is_some()),
        ]
        .into_iter()
        .filter(|(_, set)| *set)
        .map(|(name, _)| name.to_string())
        .collect()
    }
}

/// Apply one axis's chart-level scale-domain config to its resolved scale
/// (D3, spec §4.2).
///
/// Precedence, per the decision doc: an encoding-level `scale=` domain wins
/// ENTIRELY and silently — it is the more specific surface, so the documented
/// cascade already answers the question and a warning would be noise. That is
/// why `encoding_scale_explicit` is a parameter rather than something this fn
/// infers: only the caller can see the encoding.
///
/// On an ordinal/band axis the four fields describe nothing that exists (there
/// are no numeric bounds to clamp, round, or extend to zero), so they are
/// refused loudly instead: [`RenderWarning::ScaleDomainConfigOnOrdinalAxis`],
/// naming the axis and the fields. Wrong surface, not a cascade loss.
///
/// Order of operations on a continuous axis: `zero` first (extend the extent
/// to include 0), then `nice` (round outward to human bounds), then the
/// explicit `min`/`max` clamps — so an explicit bound is never re-rounded away
/// by `nice`, and `zero` cannot re-widen past an explicit bound.
///
/// `nice` delegates to [`ScaleKind::niced_domain`], which dispatches to THIS
/// scale kind's own `nice()` rather than re-deriving a kind-independent
/// rounding here — a log axis nices in log space (power-of-`base` rounding)
/// and a time axis nices calendar-aware, matching exactly what the
/// encoding-level `Scale(nice=True)` surface would produce on the same
/// domain. Rounding every kind with the same linear `nice_step` (the
/// pre-fix behavior) could drive a log axis's bound to 0 — refused by every
/// log-scale constructor, so `configure_axis(nice=True)` on a log axis used
/// to raise `InvalidScaleDomainConfig` instead of rendering.
///
/// `zero` and `nice` are ONE-DIRECTIONAL by design, and the `false` spelling
/// is a deliberate no-op rather than an oversight: both name an opt-in
/// widening of an already-resolved domain, and there is nothing on the other
/// side for `false` to switch off. Nothing in the auto path forces either —
/// `build_axis_scale` constructs every auto positional scale with
/// `nice: false`, and no mark or theme injects a zero baseline into the
/// domain (a bar chart's y domain reaches 0 because the post-Stack VALUES
/// start at 0, which `zero=false` must not and does not erase). So
/// `zero=false`/`nice=false` request the default and get it. Pinned as
/// byte-identical no-ops in `tests/test_axis_config_plumbing.py`; stated in
/// the manifest reasons and on the Python `AxisConfig` fields.
pub(in crate::render) fn apply_axis_domain_config(
    scale: &mut ScaleKind,
    cfg: &AxisDomainConfig,
    channel: &'static str,
    encoding_scale_explicit: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    if cfg.is_empty() || encoding_scale_explicit {
        return Ok(());
    }
    let Some((lo, hi)) = scale.data_domain() else {
        warnings.push(RenderWarning::ScaleDomainConfigOnOrdinalAxis {
            channel: channel.to_string(),
            fields: cfg.named_fields(),
        });
        return Ok(());
    };
    // A y axis carries its domain in display order (`[hi, lo]` for the
    // inverted pixel range is the RANGE, not the domain — the domain stays
    // ascending), but a reversed scale can still present `lo > hi`. Work in
    // ascending order and restore the caller's orientation at the end so a
    // reversed axis stays reversed.
    let reversed = lo > hi;
    let (mut d_lo, mut d_hi) = if reversed { (hi, lo) } else { (lo, hi) };
    if cfg.zero == Some(true) {
        d_lo = d_lo.min(0.0);
        d_hi = d_hi.max(0.0);
    }
    if cfg.nice == Some(true) {
        let (nlo, nhi) = scale.niced_domain(d_lo, d_hi);
        d_lo = nlo;
        d_hi = nhi;
    }
    if let Some(min) = cfg.min {
        d_lo = min;
    }
    if let Some(max) = cfg.max {
        d_hi = max;
    }
    // The computed domain must survive everything its own scale kind would
    // have refused at construction — a config-written domain is a USER-set
    // domain, so it meets the user-set contract, not the auto-inferred
    // fallback (spec §4.2). Two layers, both quoting
    // the constructors' own sentences so the words never drift:
    //
    //   1. KIND-INDEPENDENT: a degenerate pair, which every continuous
    //      constructor rejects via `core::validate_continuous_domain`.
    //      Checked on the COMPUTED result, so a zero-width domain produced by
    //      `zero`/`nice`/`min`/`max` in combination is caught too. Reversed
    //      (`lo > hi`) is NOT degenerate — an accepted reversed axis on the
    //      sibling surface too.
    //   2. KIND-SPECIFIC: whatever this scale kind constrains further, asked
    //      of the SCALE (`ScaleKind::validate_user_domain`) rather than
    //      re-derived here. That is what stops this site from growing a third
    //      hand-written check per kind, and what makes a future scale kind's
    //      own rule impossible to omit.
    if d_lo == d_hi {
        return Err(RenderError::InvalidScaleDomainConfig {
            channel: channel.to_string(),
            reason: crate::scale::core::DEGENERATE_DOMAIN_MESSAGE.to_string(),
        });
    }
    let (new_lo, new_hi) = if reversed { (d_hi, d_lo) } else { (d_lo, d_hi) };
    scale
        .validate_user_domain(new_lo, new_hi)
        .map_err(|reason| RenderError::InvalidScaleDomainConfig {
            channel: channel.to_string(),
            reason: reason.to_string(),
        })?;
    scale.set_data_domain(new_lo, new_hi);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{band_point_pixel_range, ordinal_pixel_range};
    use arrow::record_batch::RecordBatch;
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;
    use crate::scale::log::LogScale;
    use crate::scale::pow::PowScale;
    use crate::scale::symlog::SymlogScale;
    use crate::scale::time::TimeScale;

    // ── Batch B design review S4 (2026-09-03): `ScaleKind::niced_domain` ────
    //
    // Reviewer-prescribed property test: `configure_axis(nice=True)`'s
    // dispatch point (`ScaleKind::niced_domain`) must land on exactly the
    // domain each continuous kind's OWN `nice=True` construction produces —
    // the encoding-level `Scale(nice=True)` surface — for all five
    // continuous kinds. The "direct" side below constructs via
    // `<Kind>Scale::new_internal(..., nice: true)`, the exact crate-internal
    // constructor the Python-facing `#[new]` delegates to for its own `nice`
    // handling (verified by reading each — `if nice { d = d.nice(); }`),
    // rather than the private `#[new]` itself, which isn't reachable from
    // this module. Before the fix this property held only for
    // Linear/Pow/Symlog (which happen to share the same `nice_step`
    // rounding), by coincidence rather than by design; Log and Time
    // genuinely diverged.

    /// Linear: count-10 `nice_step` rounding.
    #[test]
    fn niced_domain_matches_linear_scale_nice_true() {
        let domain = vec![3.0, 847.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Linear(LinearScale::new_internal(
            domain.clone(), range.clone(), false, false,
        ));
        let direct = LinearScale::new_internal(domain.clone(), range, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Log: power-of-`base` rounding — the S4 bug's own kind. `(10, 1000)` is
    /// the reviewer-reproduced repro's domain shape (see
    /// `render::orchestration_tests::render_svg_log_axis_configure_axis_nice_true_renders_instead_of_refusing`),
    /// which the pre-fix `nice_step` rounding drove to a `0` lower bound.
    #[test]
    fn niced_domain_matches_log_scale_nice_true() {
        let domain = vec![10.0, 1000.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Log(LogScale::new_internal(
            domain.clone(), range.clone(), 10.0, false, false,
        ));
        let direct = LogScale::new_internal(domain.clone(), range, 10.0, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
        // And, concretely, neither bound is 0 — the shape of the bug this
        // property test exists to catch.
        assert_ne!(want_lo, 0.0);
    }

    /// Symlog: shares Linear's `nice_step` rounding on its (already-linear)
    /// domain representation.
    #[test]
    fn niced_domain_matches_symlog_scale_nice_true() {
        let domain = vec![3.0, 847.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Symlog(SymlogScale::new_internal(
            domain.clone(), range.clone(), 1.0, false, false,
        ));
        let direct = SymlogScale::new_internal(domain.clone(), range, 1.0, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Pow: shares Linear's `nice_step` rounding on its raw (untransformed)
    /// domain (verified by reading `PowScaleData::nice`, which is
    /// structurally identical to `LinearScaleData::nice`). `PowScale::new_internal`
    /// has no `nice` parameter (production never constructs an auto Pow scale
    /// with `nice: true` — `build_axis_scale` always passes `nice: false`),
    /// so the "direct" side here hand-derives the expected bounds via the
    /// same shared `nice_step` primitive `PowScaleData::nice` calls, rather
    /// than a from-scratch scale construction.
    #[test]
    fn niced_domain_matches_pow_scale_nice_true() {
        let domain = vec![3.0, 847.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Pow(PowScale::new_internal(domain.clone(), range, 0.5, false));
        let step = crate::scale::ticks::nice_step(domain[0], domain[1], 10);
        let want_lo = (domain[0] / step).floor() * step;
        let want_hi = (domain[1] / step).ceil() * step;
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Time: calendar-aware rounding (month/year boundaries via `chrono`),
    /// not a raw epoch-ms `nice_step` round — the other divergent kind the
    /// design review named alongside Log.
    #[test]
    fn niced_domain_matches_time_scale_nice_true() {
        // ~400 days apart, in epoch-ms.
        let domain = vec![1_700_000_000_000.0, 1_734_560_000_000.0];
        let range = vec![0.0, 500.0];
        let auto = ScaleKind::Time(TimeScale::new_internal(
            domain.clone(), range.clone(), false, false,
        ));
        let direct = TimeScale::new_internal(domain.clone(), range, false, true);
        let [want_lo, want_hi] = direct.domain_pair();
        assert_eq!(auto.niced_domain(domain[0], domain[1]), (want_lo, want_hi));
    }

    /// Issue #39: an explicit two-entry range is honored verbatim, and marked
    /// explicit for band-geometry consumers.
    #[test]
    fn band_point_pixel_range_honors_explicit_range() {
        let range = vec![40.0, 260.0];
        assert_eq!(
            band_point_pixel_range(Some(&range), (0.0, 500.0)),
            (vec![40.0, 260.0], true)
        );
    }

    /// Issue #39: an absent range falls back to the panel extent, and is not
    /// marked explicit.
    #[test]
    fn band_point_pixel_range_falls_back_when_absent() {
        assert_eq!(band_point_pixel_range(None, (0.0, 500.0)), (vec![0.0, 500.0], false));
    }

    /// Issue #39: a degenerate (fewer than 2 entries) range falls back to the
    /// panel extent rather than panicking on out-of-bounds indexing, and is
    /// not marked explicit.
    #[test]
    fn band_point_pixel_range_falls_back_when_too_short() {
        let range = vec![40.0];
        assert_eq!(
            band_point_pixel_range(Some(&range), (0.0, 500.0)),
            (vec![0.0, 500.0], false)
        );

        let empty: Vec<f64> = vec![];
        assert_eq!(
            band_point_pixel_range(Some(&empty), (0.0, 500.0)),
            (vec![0.0, 500.0], false)
        );
    }

    // ── explicitness contract (GH #39 phase 2, band-geometry unification) ──

    /// A Band scale built with an explicit two-entry range reports
    /// `explicit_band_extent()` as the signed extent `r1 - r0`.
    #[test]
    fn band_scale_explicit_range_reports_extent() {
        let (range, explicit) = band_point_pixel_range(Some(&[40.0, 260.0]), (0.0, 500.0));
        assert!(explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                0.0,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(scale.explicit_band_extent(), Some(220.0));
    }

    /// A reversed explicit range yields a negative signed extent.
    #[test]
    fn band_scale_explicit_reversed_range_reports_negative_extent() {
        let (range, explicit) = band_point_pixel_range(Some(&[260.0, 40.0]), (0.0, 500.0));
        assert!(explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                0.0,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(scale.explicit_band_extent(), Some(-220.0));
    }

    /// A Band scale falling back to the panel extent reports `None` even
    /// though its range is numerically identical to what an explicit range
    /// spanning the same pixels would be — explicitness is recorded at
    /// construction, not inferred from the numbers.
    #[test]
    fn band_scale_fallback_range_reports_no_extent() {
        let (range, explicit) = band_point_pixel_range(None, (0.0, 500.0));
        assert!(!explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                0.0,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(scale.explicit_band_extent(), None);
    }

    /// A positional Ordinal scale with an explicit >= 2-entry numeric range
    /// behaves identically to Band/Point (no special-casing).
    #[test]
    fn ordinal_positional_explicit_range_reports_extent() {
        let range_values = vec![
            crate::scale::ordinal::OrdinalRangeValue::Number(10.0),
            crate::scale::ordinal::OrdinalRangeValue::Number(210.0),
        ];
        let (range, explicit) = ordinal_pixel_range(Some(&range_values), (0.0, 500.0));
        assert!(explicit);
        let scale = ScaleKind::Ordinal(
            crate::scale::ordinal::OrdinalScale::new_internal(
                vec!["a".into(), "b".into()],
                range,
                0.0,
            )
            .with_explicit_range(explicit),
        );
        assert_eq!(scale.explicit_band_extent(), Some(200.0));
    }

    /// A non-ordinal (Linear) scale always reports `None` — the accessor is
    /// gated to ordinal positional scales only.
    #[test]
    fn linear_scale_never_reports_explicit_band_extent() {
        let scale = ScaleKind::Linear(crate::scale::linear::LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![40.0, 260.0],
            false,
            false,
        ));
        assert_eq!(scale.explicit_band_extent(), None);
    }

    // ── PointScale(reverse=True) domain reversal (GH #65) ──

    /// Builds a one-column string batch and a `ScaleSpec::Point` for it, then
    /// resolves via the real `build_from_scale_spec` arm — the same seam
    /// `build_axis_scale` calls when an encoding carries an explicit `scale`.
    fn resolve_point_scale(
        explicit_domain: Vec<String>,
        range: Option<Vec<f64>>,
        reverse: bool,
    ) -> ScaleKind {
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("cat", ArrowDataType::Utf8, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(explicit_domain.clone()))],
        )
        .unwrap();

        let scale_spec = crate::spec::encoding::ScaleSpec::Point {
            domain: Some(explicit_domain),
            padding: 0.0,
            align: 0.5,
            reverse,
            range,
        };
        let enc = crate::spec::encoding::EncodingSpec {
            field: "cat".into(),
            type_: None,
            ..Default::default()
        };
        let sort_ctx = super::SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: None,
            y_field: None,
        };
        let mut warnings = Vec::new();
        super::build_from_scale_spec(&scale_spec, &enc, &batch, (0.0, 500.0), &sort_ctx, &mut warnings)
            .unwrap()
    }

    /// GH #65: `reverse=True` reverses the resolved domain order, so the
    /// *first* category of the original domain lands on the *last* band
    /// center (and vice versa) — a domain-vector reversal, not a pixel-range
    /// flip.
    #[test]
    fn point_scale_reverse_true_reverses_domain_order() {
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let scale = resolve_point_scale(domain, None, true);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        // Panel-extent range (0.0, 500.0), 4 categories → step = 125.0,
        // centers at 62.5, 187.5, 312.5, 437.5. Non-reversed, "a" would sit
        // at the first center (62.5); reversed, it sits at the last (437.5).
        assert_eq!(ordinal.scale_internal("a"), Some(437.5));
        assert_eq!(ordinal.scale_internal("d"), Some(62.5));
    }

    /// GH #65: `reverse=True` composes with an explicit pixel range —
    /// centers for original domain `[a, b, c, d]` over range `[40, 260]`
    /// come out as `[232.5, 177.5, 122.5, 67.5]`.
    #[test]
    fn point_scale_reverse_true_composes_with_explicit_range() {
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let scale = resolve_point_scale(domain, Some(vec![40.0, 260.0]), true);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        assert_eq!(ordinal.scale_internal("a"), Some(232.5));
        assert_eq!(ordinal.scale_internal("b"), Some(177.5));
        assert_eq!(ordinal.scale_internal("c"), Some(122.5));
        assert_eq!(ordinal.scale_internal("d"), Some(67.5));
    }

    /// GH #65 regression: `reverse=False` (the pre-fix default) is
    /// unchanged — domain order is preserved and band centers ascend in
    /// original domain order.
    #[test]
    fn point_scale_reverse_false_preserves_domain_order() {
        let domain = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let scale = resolve_point_scale(domain, Some(vec![40.0, 260.0]), false);
        let ScaleKind::Ordinal(ordinal) = scale else { panic!("expected Ordinal scale") };

        assert_eq!(ordinal.scale_internal("a"), Some(67.5));
        assert_eq!(ordinal.scale_internal("b"), Some(122.5));
        assert_eq!(ordinal.scale_internal("c"), Some(177.5));
        assert_eq!(ordinal.scale_internal("d"), Some(232.5));
    }

    // ── explicitness gates: degenerate inputs (ported from
    // tests/bug_hunt_band_point_range.rs, R1) ────────────────────────────────

    /// A wire range with MORE than 2 entries truncates to the first two and
    /// still counts as explicit — matching the pyclass constructor's
    /// `[v[0], v[1]]`.
    #[test]
    fn band_point_gate_truncates_extra_entries_still_explicit() {
        let (r, explicit) = band_point_pixel_range(Some(&[40.0, 260.0, 500.0]), (0.0, 600.0));
        assert_eq!(r, vec![40.0, 260.0]);
        assert!(explicit);
    }

    /// Documents the validation gap the gate does NOT close: non-finite range
    /// entries pass straight through, still marked explicit. The
    /// constructor-level gap (no `validate_finite` for Band/Point ranges) is
    /// proven end-to-end by the Python tests; this pins that the resolver
    /// gate is not the layer catching it.
    #[test]
    fn band_point_gate_passes_non_finite_entries_through() {
        let (r, explicit) = band_point_pixel_range(Some(&[f64::NAN, 260.0]), (0.0, 600.0));
        assert!(r[0].is_nan(), "gate does not filter NaN (documented gap)");
        assert_eq!(r[1], 260.0);
        assert!(explicit, "a NaN-carrying 2-entry range is still treated as explicit");
    }

    /// Ordinal gate: a single NUMERIC entry is passed through unchanged but
    /// NOT marked explicit (`explicit_band_extent` needs 2 entries) — the
    /// documented asymmetry vs `band_point_pixel_range`, which falls back to
    /// the panel extent.
    #[test]
    fn ordinal_gate_single_number_passthrough_not_explicit() {
        use crate::scale::ordinal::OrdinalRangeValue;
        let (r, explicit) = ordinal_pixel_range(Some(&[OrdinalRangeValue::Number(40.0)]), (0.0, 600.0));
        assert_eq!(r, vec![40.0], "single numeric entry passes through unchanged");
        assert!(!explicit, "1-entry range must not count as explicit");
    }

    /// Ordinal gate: an all-string (color) range has no numbers → the panel
    /// extent fallback, not explicit.
    #[test]
    fn ordinal_gate_all_string_range_falls_back() {
        use crate::scale::ordinal::OrdinalRangeValue;
        let (r, explicit) = ordinal_pixel_range(
            Some(&[OrdinalRangeValue::Str("#ccc".into()), OrdinalRangeValue::Str("#e4572e".into())]),
            (0.0, 600.0),
        );
        assert_eq!(r, vec![0.0, 600.0]);
        assert!(!explicit);
    }

    /// Ordinal gate: mixed [num, str, num] extracts both numbers and IS
    /// explicit — the seam the Python mixed-range render test exercises
    /// end-to-end.
    #[test]
    fn ordinal_gate_mixed_range_extracts_numbers_explicit() {
        use crate::scale::ordinal::OrdinalRangeValue;
        let (r, explicit) = ordinal_pixel_range(
            Some(&[OrdinalRangeValue::Number(40.0), OrdinalRangeValue::Str("#ccc".into()), OrdinalRangeValue::Number(260.0)]),
            (0.0, 600.0),
        );
        assert_eq!(r, vec![40.0, 260.0]);
        assert!(explicit);
    }

    /// Ordinal gate: an empty range falls back to the panel extent, never
    /// explicit, never panicking on indexing. (`band_point_pixel_range`'s
    /// empty-range fallback is already pinned by
    /// `band_point_pixel_range_falls_back_when_too_short`.)
    #[test]
    fn ordinal_gate_empty_range_falls_back_no_panic() {
        let empty_vals: Vec<crate::scale::ordinal::OrdinalRangeValue> = Vec::new();
        assert_eq!(ordinal_pixel_range(Some(&empty_vals), (5.0, 95.0)), (vec![5.0, 95.0], false));
    }

    // ── Continuous `reverse` domain-swap (F-L04-07, batch-C task 1) ──
    //
    // `reverse` on `ContinuousScaleCommon` is domain-swap sugar (spec §4C):
    // after domain resolution, `reverse=true` swaps the resolved domain
    // pair, exactly equivalent to writing `domain=[hi, lo]` by hand. Every
    // test below resolves through the real `build_from_scale_spec` arm —
    // the same seam `build_axis_scale` calls for an encoding's explicit
    // `scale` — mirroring the `PointScale(reverse=True)` tests above rather
    // than hand-sequencing the swap.

    fn numeric_batch(field: &str, values: Vec<f64>) -> RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new(field, ArrowDataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn resolve_scale(
        spec: &crate::spec::encoding::ScaleSpec,
        field: &str,
        batch: &RecordBatch,
        pr: (f64, f64),
    ) -> ScaleKind {
        let enc = crate::spec::encoding::EncodingSpec {
            field: field.into(),
            type_: None,
            scale: Some(spec.clone()),
            ..Default::default()
        };
        let sort_ctx = super::SortContext {
            category_field: field,
            batch,
            x_field: None,
            y_field: None,
        };
        let mut warnings = Vec::new();
        super::build_from_scale_spec(spec, &enc, batch, pr, &sort_ctx, &mut warnings).unwrap()
    }

    /// A `ContinuousScaleCommon` with everything but `domain`/`range`/`reverse`
    /// at its wire default — the shape every one of these tests varies along
    /// exactly one axis at a time.
    fn reverse_common(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        reverse: bool,
    ) -> crate::spec::encoding::ContinuousScaleCommon {
        crate::spec::encoding::ContinuousScaleCommon {
            domain,
            range,
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse,
        }
    }

    /// F-L04-07: `reverse=true` on an auto-inferred domain swaps the
    /// resolved `[min, max]` pair to `[max, min]` — one test per continuous
    /// kind, so a future kind that forgets to wire `apply_domain_reverse`
    /// into its arm fails here instead of silently rendering ascending.
    #[test]
    fn linear_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear { common: reverse_common(None, None, true), nice: false, zero: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)));
    }

    #[test]
    fn log_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![1.0, 1000.0]);
        let spec = ScaleSpec::Log { base: 10.0, common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((1000.0, 1.0)));
    }

    #[test]
    fn time_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 86_400_000.0]);
        let spec = ScaleSpec::Time { common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((86_400_000.0, 0.0)));
    }

    #[test]
    fn symlog_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![-100.0, 100.0]);
        let spec = ScaleSpec::Symlog { constant: 1.0, common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, -100.0)));
    }

    #[test]
    fn pow_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Pow { exponent: 2.0, common: reverse_common(None, None, true) };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)));
    }

    #[test]
    fn sqrt_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Sqrt { common: reverse_common(None, None, true) };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)));
    }

    #[test]
    fn utc_reverse_true_swaps_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 86_400_000.0]);
        let spec = ScaleSpec::Utc { common: reverse_common(None, None, true), nice: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((86_400_000.0, 0.0)));
    }

    /// `reverse=false` (the wire default) leaves every kind's domain in
    /// ascending order — the regression guard alongside the seven flips
    /// above.
    #[test]
    fn linear_reverse_false_preserves_auto_inferred_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear { common: reverse_common(None, None, false), nice: false, zero: false };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((0.0, 100.0)));
    }

    /// Pins the ordering `apply_domain_reverse`'s doc calls load-bearing:
    /// the swap must run AFTER the `zero` block, not before. The `zero`
    /// block (`positional.rs:233-236`) assumes `d[0]` is the minimum — with
    /// an auto-inferred `[20, 80]` domain, `zero=true` extends it to
    /// `[0.0, 80.0]`, and reversing THAT (the implemented order) yields the
    /// correct `(80.0, 0.0)`. Reversing FIRST would hand `zero`'s
    /// ascending-pair assumption the already-swapped `[80, 20]` pair
    /// instead: `d[0]=80 > 0.0` zeros it, `d[1]=20 < 0.0` is false, so the
    /// wrong-order run corrupts the result to `(0.0, 20.0)` — a RED value I
    /// confirmed by hoisting `apply_domain_reverse` above the `zero` block
    /// in-place, running this test (it failed with exactly `Some((0.0,
    /// 20.0))` vs. the expected `Some((80.0, 0.0))`), and reverting the
    /// hoist back to its original position (see the task report's cycle-3
    /// section for the exact command/output).
    #[test]
    fn linear_zero_true_reverse_true_yields_non_degenerate_descending_domain() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![20.0, 80.0]);
        let spec = ScaleSpec::Linear { common: reverse_common(None, None, true), nice: false, zero: true };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(
            scale.data_domain(),
            Some((80.0, 0.0)),
            "zero=true, reverse=true must compose to the non-degenerate descending pair, not collapse to (0, 0)"
        );
    }

    // ── §4C composition rules ──

    /// Rule 1: `reverse=True` + explicit `domain=[a,b]` → the GIVEN pair is
    /// swapped (not the auto-inferred one — the column here carries values
    /// the explicit domain ignores entirely).
    #[test]
    fn linear_reverse_true_with_explicit_domain_swaps_the_given_pair() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![-999.0, 999.0]);
        let spec = ScaleSpec::Linear {
            common: reverse_common(Some(vec![10.0, 90.0]), None, true),
            nice: false,
            zero: false,
        };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((90.0, 10.0)));
    }

    /// Rule 2: `reverse=True` + explicit `range=[hi,lo]` → both apply as
    /// stated. The domain still swaps; the range passes through completely
    /// untouched by `reverse` (a reversed explicit range means what it
    /// says, independent of the domain-swap sugar).
    #[test]
    fn linear_reverse_true_composes_with_reversed_explicit_range() {
        use crate::spec::encoding::ScaleSpec;
        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let spec = ScaleSpec::Linear {
            common: reverse_common(None, Some(vec![500.0, 0.0]), true),
            nice: false,
            zero: false,
        };
        let scale = resolve_scale(&spec, "v", &batch, (0.0, 500.0));
        assert_eq!(scale.data_domain(), Some((100.0, 0.0)), "reverse still swaps the domain");
        let ScaleKind::Linear(linear) = &scale else { panic!("expected Linear scale") };
        assert_eq!(
            linear.range_pair(),
            [500.0, 0.0],
            "the explicit reversed range passes through unmodified by `reverse`"
        );
    }

    /// Rule 3: `coord_cartesian(x_domain=)` wins over the encoding scale's
    /// `reverse` — it replaces the domain wholesale and its own endpoint
    /// order is respected, regardless of what the encoding's scale asked
    /// for. Asserted through the real production sequence
    /// (`resolve_scales` → `build_axis_scale` → `apply_coord_domain_overrides`)
    /// rather than hand-calling the two functions in the test body, so a
    /// future reorder of that pipeline would be caught here.
    #[test]
    fn coord_cartesian_x_domain_override_wins_over_encoding_reverse() {
        use crate::spec::coord::CoordKind;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;

        let batch = {
            use arrow::array::Float64Array;
            use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", ArrowDataType::Float64, false),
                Field::new("y", ArrowDataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vec![0.0, 100.0])),
                    Arc::new(Float64Array::from(vec![0.0, 1.0])),
                ],
            )
            .unwrap()
        };
        let spec = crate::spec::chart::ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(),
                    type_: None,
                    scale: Some(ScaleSpec::Linear {
                        common: reverse_common(None, None, true),
                        nice: false,
                        zero: false,
                    }),
                    ..Default::default()
                }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(CoordKind::Cartesian {
                x_domain: Some((5.0, 20.0)),
                y_domain: None,
                expand: true,
                clip: true,
            }),
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
        let theme = crate::layout::ThemeInputs::default();
        let (scales, _) =
            crate::render::scale_resolve::resolve_scales(&spec, &batch, (0.0, 500.0), (0.0, 500.0), &theme).unwrap();
        // The override's own (5.0, 20.0) endpoint order wins outright — the
        // encoding scale's `reverse=true` never gets a chance to swap it,
        // because `apply_coord_domain_overrides` replaces the whole scale.
        assert_eq!(scales.x.data_domain(), Some((5.0, 20.0)));
    }

    // ── spec §9's discriminating y-axis assertion ──

    /// Resolves a `ScaleSpec::Linear` y-axis scale through the REAL
    /// `build_axis_scale("y", ...)` entry point (not `build_from_scale_spec`
    /// directly), so `axis_pixel_range`'s structural pixel-range inversion
    /// for the y channel is actually computed by production code and fed
    /// forward — a future reorder of that composition would fail this test,
    /// not just a hand-mirrored fixture.
    ///
    /// An explicit `domain=[0, 100]` (with no explicit `range`) keeps the
    /// pixel math exact: an explicit domain suppresses the default 5%
    /// padding inset (`resolve_padding_fraction`), so the un-padded plot
    /// range `(0.0, 400.0)` passed in comes back out of `axis_pixel_range`'s
    /// y-inversion as exactly `(400.0, 0.0)`, with no inset arithmetic to
    /// account for in the assertions below.
    fn build_y_axis_scale(reverse: bool) -> ScaleKind {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;
        use std::collections::HashMap;

        let batch = numeric_batch("v", vec![0.0, 100.0]);
        let enc = EncodingSpec {
            field: "v".into(),
            type_: None,
            scale: Some(ScaleSpec::Linear {
                common: reverse_common(Some(vec![0.0, 100.0]), None, reverse),
                nice: false,
                zero: false,
            }),
            ..Default::default()
        };
        // `spec` is unused on the explicit-`enc.scale` bypass `build_axis_scale`
        // takes here (it returns before ever consulting `spec`), so any valid
        // `ChartSpec` satisfies the parameter.
        let spec = crate::spec::chart::ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Default::default(),
            transforms: Vec::new(),
            facet: None,
            layers: None,
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
        let outputs: HashMap<String, RecordBatch> = HashMap::new();
        let mut warnings = Vec::new();
        super::build_axis_scale(
            "y",
            &enc,
            None,
            super::PositionalFields { x: None, y: Some("v") },
            &batch,
            &outputs,
            (0.0, 400.0),
            &spec,
            false,
            None,
            &mut warnings,
        )
        .unwrap()
    }

    /// `reverse=true` composes with the y-channel's structural pixel-range
    /// inversion (`axis_pixel_range`, computed inside `build_axis_scale`
    /// before `build_from_scale_spec` ever sees the range) without one
    /// mechanism clobbering the other: marks (`to_pixel_f64`) and tick
    /// labels (`tick_labels`, index-aligned with `tick_fractions`) flip
    /// together, not independently.
    #[test]
    fn linear_reverse_true_on_y_channel_flips_marks_and_labels_together() {
        let baseline = build_y_axis_scale(false);
        let reversed = build_y_axis_scale(true);

        // Marks: the minimum data value (0) sits at the BOTTOM (pixel 400)
        // on the default axis, and at the TOP (pixel 0) once reversed.
        assert_eq!(baseline.to_pixel_f64(0.0), Some(400.0));
        assert_eq!(reversed.to_pixel_f64(0.0), Some(0.0));
        assert_eq!(baseline.to_pixel_f64(100.0), Some(0.0));
        assert_eq!(reversed.to_pixel_f64(100.0), Some(400.0));

        // Labels: `tick_labels`/`tick_fractions` stay index-aligned on BOTH
        // scales, and the reversed scale's own tick VALUES (the same order
        // `tick_labels` uses) come out descending — the label order flipped
        // along with the marks, not independently of them.
        let base_values = baseline.tick_values_raw(5).unwrap();
        let rev_values = reversed.tick_values_raw(5).unwrap();
        assert_eq!(baseline.tick_labels(5).len(), baseline.tick_fractions(5).len());
        assert_eq!(reversed.tick_labels(5).len(), reversed.tick_fractions(5).len());
        assert!(!rev_values.is_empty());
        assert!(base_values.windows(2).all(|w| w[0] <= w[1]), "baseline ticks ascend: {base_values:?}");
        assert!(rev_values.windows(2).all(|w| w[0] >= w[1]), "reversed ticks descend: {rev_values:?}");

        // Same underlying tick set, reordered — `reverse` must not change
        // WHICH values get labeled, only the order they're labeled in.
        let mut base_sorted = base_values.clone();
        let mut rev_sorted = rev_values.clone();
        base_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(base_sorted, rev_sorted, "reverse reorders ticks, it does not pick a different set");
    }
}
