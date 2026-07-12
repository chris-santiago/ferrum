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
            ScaleKind::Linear(LinearScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Log { base, common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Log(LogScale::new_internal(d, r, *base, common.clamp, *nice))
        }
        ScaleSpec::Time { common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Time(TimeScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Symlog { constant, common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
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
            ScaleKind::Pow(PowScale::new_internal(d, r, *exponent, common.clamp))
        }
        ScaleSpec::Sqrt { common } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Pow(PowScale::new_internal(d, r, 0.5, common.clamp))
        }
        ScaleSpec::Utc { common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
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
        ScaleSpec::Point { domain, padding, range, .. } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_positional_categories(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref(), sort_ctx, warnings);
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

#[cfg(test)]
mod tests {
    use super::{band_point_pixel_range, ordinal_pixel_range};
    use crate::render::scale_resolve::ScaleKind;

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
}
