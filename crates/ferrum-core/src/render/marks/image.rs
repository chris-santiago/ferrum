//! Image mark renderer — two independent rendering paths:
//!
//! ## 1. URL-tile path (`mark_image` with a `url` encoding channel)
//!
//! Triggered when `ctx.spec.encoding.url.is_some()`.  Each row in the batch
//! becomes one image tile:
//!
//! - `x`, `y` columns (Float64) — required; rows with nulls are skipped.
//! - The field named by `encoding.url` (Utf8) — holds a `data:<mime>;base64,<payload>`
//!   URL.  Rows with nulls or malformed URLs are skipped silently.
//! - Optional `width`, `height` columns (Float64) for per-row pixel sizing.
//!   When absent, `mark_style.misc.width` / `mark_style.misc.height` are used, defaulting
//!   to 32.0 × 32.0 pixels.
//!
//! Each tile is center-anchored at `(px, py)` — consistent with `mark_point`.
//!
//! ## 2. Raster path (`mark_raster` composite desugaring)
//!
//! Triggered when `ctx.spec.encoding.url.is_none()`.  Reads the single-row
//! Raster transform output batch (columns: `x_min`, `x_max`, `y_min`, `y_max`,
//! `width` (UInt32), `height` (UInt32), `pixel_data` (Binary)).  Applies a
//! colormap and emits one PNG-encoded `SceneNode::Image` spanning the plot
//! extent.
//!
//! Colormap resolution priority (three-step):
//!   1. `mark_style.group.cmap` name (explicit kwarg on the mark)
//!   2. `theme.palette.sequential_scheme` (theme default)
//!   3. Viridis fallback

use arrow::array::{BinaryArray, Float64Array, UInt32Array};

use crate::render::scale_resolve::ColorScale;
use crate::render::draw::{col_as_f64, col_as_str, DrawCtx, MetadataColumns};
use crate::render::mark_nodes::MarkNodes;
use crate::render::rasterize::encode_png;

/// The Raster-transform batch columns holding the image's data extent on the
/// rendered x axis. The raster path draws the image over exactly this extent.
pub(crate) const RASTER_X_EXTENT: [&str; 2] = ["x_min", "x_max"];
/// The y-axis counterpart of [`RASTER_X_EXTENT`].
pub(crate) const RASTER_Y_EXTENT: [&str; 2] = ["y_min", "y_max"];

/// Which path an image mark whose effective `url` channel is `url` draws: the
/// tile path's url field (one tile per row), or `None` for the raster path
/// (one image over the Raster batch's extent).
pub(crate) fn url_tile_field(url: Option<&crate::spec::encoding::EncodingSpec>) -> Option<&str> {
    url.map(|enc| enc.field.as_str())
}

/// The extent columns an image layer draws on the rendered `channel` when it
/// takes the raster path ([`url_tile_field`] over its effective `url` is
/// `None`), or `None` for any other layer. Such a layer's `x`/`y` encoding
/// names the source columns the Raster transform binned, which its own
/// one-row batch does not carry, so these columns are what it contributes to
/// the axis domain.
pub(crate) fn raster_extent_columns(
    mark: crate::spec::mark::Mark,
    url: Option<&crate::spec::encoding::EncodingSpec>,
    channel: &str,
) -> Option<[&'static str; 2]> {
    if mark != crate::spec::mark::Mark::Image || url_tile_field(url).is_some() {
        return None;
    }
    match channel {
        "x" => Some(RASTER_X_EXTENT),
        "y" => Some(RASTER_Y_EXTENT),
        _ => None,
    }
}

/// The columns a layer of `mark` draws for its binding `field` on the
/// rendered `channel`: a raster image's extent columns on `x`/`y`
/// ([`raster_extent_columns`]) and, on `color`, the column naming the field
/// its cells hold ([`RASTER_VALUE_FIELD`], see [`image_draws_color`]); else
/// `field` itself. The one answer to "which column does this layer draw
/// here", read by the column gate (which judges each of them against the
/// layer's batch) and by an axis's typing read (`scale_resolve`'s
/// `locate_seed_field`, which types the axis from the first); the domain
/// union reads the extent pair through [`raster_extent_columns`] and a
/// raster's color domain its cells (`scale_resolve`'s typed raster cell
/// read, `AuxReadKind::RasterCells`). Published to
/// Python as [`drawn_positional_columns_py`], which the auto tooltip reads so
/// it lists a positional field only where a layer draws that field itself.
pub(crate) fn drawn_columns<'f>(
    mark: crate::spec::mark::Mark,
    url: Option<&crate::spec::encoding::EncodingSpec>,
    channel: &str,
    field: &'f str,
) -> Vec<&'f str> {
    if let Some(extent) = raster_extent_columns(mark, url, channel) {
        return extent.to_vec();
    }
    if channel == "color" && mark == crate::spec::mark::Mark::Image && url_tile_field(url).is_none() {
        return vec![RASTER_VALUE_FIELD];
    }
    vec![field]
}

pub(crate) use crate::transform::raster::{RASTER_VALUE_FIELD, RASTER_VALUE_SCALE};

/// The field a raster batch's cells hold ([`RASTER_VALUE_FIELD`]), or `None`
/// for a batch that is no Raster output.
pub(crate) fn raster_value_field(batch: &arrow::record_batch::RecordBatch) -> Option<&str> {
    let column = batch.column_by_name(RASTER_VALUE_FIELD)?.as_any().downcast_ref::<arrow::array::StringArray>()?;
    column.iter().flatten().next()
}

/// Whether a layer of `mark` draws `color` through a raster's cells: it is a
/// url-less image, and its own color binding, else the one it inherits from
/// `parent` (its authored chart encoding), names the field its cells hold
/// (`cells`, its Raster batch's [`raster_value_field`]). A raster image
/// draws its cells, so its color is the field they hold. The one answer
/// every reader asks: the image builder (of the layer's completed encoding,
/// its own parent), the reads of the chart color scale and of the cell scale
/// (`scale_resolve`'s `aux_reads` and `raster_cell_reads`), and `prepare`'s
/// drawn bindings and undrawn-channel warnings. A sibling layer's color
/// binding never enters it.
pub(crate) fn image_draws_color(
    mark: crate::spec::mark::Mark,
    encoding: &crate::spec::encoding::Encoding,
    parent: &crate::spec::encoding::Encoding,
    cells: Option<&str>,
) -> bool {
    let url = encoding.url.as_ref().or(parent.url.as_ref());
    mark == crate::spec::mark::Mark::Image
        && url_tile_field(url).is_none()
        && encoding.color.as_ref().or(parent.color.as_ref()).is_some_and(|color| cells == Some(color.field.as_str()))
}

/// The color domain a raster image's cells `(lo, hi)` (their finite min and
/// max) span as the image draws them: from zero when every cell is
/// non-negative (a count, a density, a non-negative mean), so the least
/// populated cell stays off the scheme's first color and a uniform raster
/// draws the scheme's top color; else `(lo, hi)` as they are, so a mean
/// spanning negatives keeps its full range. The zero anchor is the one bar
/// and linear scales take ([`crate::render::scale_resolve::extend_extent_to_zero`]).
/// Read by the image's own scale and by the chart's color domain over a
/// raster's cells (`scale_resolve`'s `aux_numeric_union`), so a colorbar
/// spans the domain the image draws. An all-zero raster is `(0, 0)`, whose
/// cells the scale maps to its midpoint; an empty one draws no cell.
pub(crate) fn raster_cell_domain((lo, hi): (f64, f64)) -> (f64, f64) {
    if lo >= 0.0 {
        crate::render::scale_resolve::extend_extent_to_zero(lo, hi)
    } else {
        (lo, hi)
    }
}

/// The finite extent `(min, max)` of a raster batch's cells (every row's
/// `pixel_data` decoded), or `None` for a batch that is no Raster output or
/// holds no finite cell: what a raster image's color domain reads
/// (`scale_resolve`'s raster cell reads), as its extent columns are what its
/// axes read.
pub(crate) fn raster_cell_extent(batch: &arrow::record_batch::RecordBatch) -> Option<(f64, f64)> {
    raster_cell_values(batch).fold(None, |extent: Option<(f64, f64)>, v| match extent {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

/// Every finite cell of a raster batch (every row's `pixel_data` decoded);
/// empty for a batch that is no Raster output. What a `Quantile` color scale
/// bound to a raster's cells samples, as it samples a column on any other
/// mark.
pub(crate) fn raster_cell_values(batch: &arrow::record_batch::RecordBatch) -> impl Iterator<Item = f64> + '_ {
    batch
        .column_by_name("pixel_data")
        .and_then(|column| column.as_any().downcast_ref::<BinaryArray>())
        .into_iter()
        .flat_map(|pixels| pixels.iter().flatten())
        .flat_map(|bytes| bytes.chunks_exact(8).map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]))))
        .filter(|v| v.is_finite())
}

/// The transform a raster batch's cells' color maps through, from the scale
/// its `value_scale` column names ([`RASTER_VALUE_SCALE`]): symlog with
/// constant 1 (`log1p` on non-negative cells) for `symlog`, else linear.
pub(crate) fn raster_value_transform(batch: &arrow::record_batch::RecordBatch) -> crate::render::scale_resolve::ColorTransform {
    use crate::render::scale_resolve::ColorTransform;
    let scale = batch
        .column_by_name(RASTER_VALUE_SCALE)
        .and_then(|column| column.as_any().downcast_ref::<arrow::array::StringArray>())
        .and_then(|column| column.iter().flatten().next());
    match scale {
        Some("symlog") => ColorTransform::Symlog { constant: 1.0 },
        _ => ColorTransform::Linear,
    }
}

/// `ferrum._core.drawn_positional_columns(mark, has_url, channel, field)`:
/// the columns a layer of `mark` draws for its binding `field` on the
/// `channel` ([`drawn_columns`]); `has_url` says
/// whether an image layer binds `url` (the url-tile path, which draws its
/// own fields). Python's per-layer auto tooltip lists a positional field
/// only when this answer is the field itself. Refuses an unknown mark.
/// Registered in `lib.rs`.
#[pyo3::pyfunction]
#[pyo3(name = "drawn_positional_columns")]
pub(crate) fn drawn_positional_columns_py(
    mark: &str,
    has_url: bool,
    channel: &str,
    field: &str,
) -> pyo3::PyResult<Vec<String>> {
    let mark = crate::spec::mark::parse_mark_arg(mark)?;
    let url = crate::spec::encoding::EncodingSpec::default();
    Ok(drawn_columns(mark, has_url.then_some(&url), channel, field).into_iter().map(str::to_string).collect())
}

pub fn build(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, crate::render::RenderError> {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::MarkBatchKind;

    // #118: both paths below are Cartesian-only, so a Polar/Geo coord leaves
    // this mark with nothing it can place — a genuine defect (the user
    // asked for images on a coord this mark cannot draw under), not a
    // documented degrade: nothing about `mark_image()`'s public API says
    // Polar/Geo is a legitimate, silent no-op.
    match &ctx.spec.coord {
        Some(crate::spec::coord::CoordKind::Polar { .. }) => {
            return Ok(MarkBuildResult::empty_because(
                MarkBatchKind::Image,
                "mark_image() does not support CoordPolar; both its url-tile and \
                 raster paths place images in Cartesian pixel space",
            ));
        }
        Some(crate::spec::coord::CoordKind::Geo { .. }) => {
            return Ok(MarkBuildResult::empty_because(
                MarkBatchKind::Image,
                "mark_image() does not support CoordGeo; both its url-tile and \
                 raster paths place images in Cartesian pixel space",
            ));
        }
        _ => {}
    }

    match url_tile_field(ctx.spec.encoding.url.as_ref()) {
        // ── URL-tile path ──────────────────────────────────────────────────
        Some(field) => Ok(build_url_tiles(ctx, field)),
        // ── Raster path ────────────────────────────────────────────────────
        None => build_raster(ctx),
    }
}

/// URL-tile path: one `SceneNode::Image` per row, positioned via x/y scales.
fn build_url_tiles(ctx: &DrawCtx, url_field: &str) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::{ImageData, MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Image);

    let batch = ctx.batch;
    let n = batch.num_rows();
    if n == 0 {
        return empty();
    }

    // Require x and y fields from the encoding.
    let x_field = match ctx.spec.encoding.x.as_ref() {
        Some(e) => e.field.as_str(),
        None => return empty(),
    };
    let y_field = match ctx.spec.encoding.y.as_ref() {
        Some(e) => e.field.as_str(),
        None => return empty(),
    };

    let xs = match col_as_f64(batch, x_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let ys = match col_as_f64(batch, y_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let urls = match col_as_str(batch, url_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };

    // Optional per-row width/height columns (Float64). If absent, fall back
    // to mark_style.misc.width / mark_style.misc.height, then to 32.0 pixels.
    let widths: Option<Vec<Option<f64>>> = col_as_f64(batch, "width").ok();
    let heights: Option<Vec<Option<f64>>> = col_as_f64(batch, "height").ok();
    let default_w = ctx.mark_style.misc.width.unwrap_or(32.0);
    let default_h = ctx.mark_style.misc.height.unwrap_or(32.0);

    let mut acc = MarkNodes::with_capacity(n);
    // #230-3: one exact `(px, py)` data anchor per tile — the scale-mapped
    // center position BEFORE the tile's own width/height is subtracted to
    // find its top-left corner below. `for_each_along` treats `Image` nodes
    // as a no-op (see its doc), so this anchor has no observable effect on
    // today's break rendering; it exists so an `Image` batch satisfies the
    // same `MoveGlyph` contract every other glyph batch does, rather than
    // being a silent exception a debug build has to special-case.
    let mut anchors: Vec<(f64, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        // Skip nulls in required fields.
        let x_val = match xs.get(i).and_then(|v| *v) {
            Some(v) => v,
            None => continue,
        };
        let y_val = match ys.get(i).and_then(|v| *v) {
            Some(v) => v,
            None => continue,
        };
        let url_val = match urls.get(i).and_then(|v| v.as_deref()) {
            Some(s) => s,
            None => continue,
        };

        // Must be a data URL: "data:<mime>;base64,<payload>"
        if !url_val.starts_with("data:") {
            continue;
        }

        // Map data coordinates to pixel space.
        let px = match ctx.scales.x.to_pixel_f64(x_val) {
            Some(p) => p,
            None => continue,
        };
        let py = match ctx.scales.y.to_pixel_f64(y_val) {
            Some(p) => p,
            None => continue,
        };

        // Per-row tile dimensions, falling back to mark_style / default.
        let tile_w = widths.as_ref()
            .and_then(|v| v.get(i).and_then(|v| *v))
            .unwrap_or(default_w);
        let tile_h = heights.as_ref()
            .and_then(|v| v.get(i).and_then(|v| *v))
            .unwrap_or(default_h);

        // Center-anchor: top-left corner of the <image> element.
        let img_x = px - tile_w / 2.0;
        let img_y = py - tile_h / 2.0;

        acc.push(SceneNode::Image {
            x: img_x,
            y: img_y,
            w: tile_w,
            h: tile_h,
            data: ImageData::Url { url: url_val.to_string() },
        }, i);
        anchors.push((px, py));
    }

    let (nodes, data_indices) = acc.finalize();
    let meta = MetadataColumns::from_ctx(ctx);

    MarkBuildResult::populated(MarkBatchKind::Image, nodes, data_indices, &meta)
        .with_glyph_anchors(anchors)
}

/// Raster path: single-row Raster transform output → one colormap-rendered PNG.
/// Refuses a raster whose cells hold a finite value that no color scale
/// reached: scale resolution gives every image a scale (the chart's color
/// scale or the shared cell scale) whenever a cell is finite, so the image
/// never silently draws nothing.
fn build_raster(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, crate::render::RenderError> {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::{ImageData, ImageMime, MarkBatchKind, SceneNode};

    let empty = || Ok(MarkBuildResult::empty(MarkBatchKind::Image));
    // #118: reached only when `build()`'s dispatch found no `url` encoding
    // (the URL-tile path's own presence/dtype guards are unreachable/
    // already-covered classes — see the doc above `build`) — so every
    // failure below means the batch is not a `url=`-carrying tile batch AND
    // not the single-row Raster-transform output `mark_raster()`'s own
    // desugar always produces. The only way a user reaches here is calling
    // `mark_image()` bare, with neither a `url` channel nor the internal
    // Raster transform behind it — a genuine defect (the user asked for an
    // image and named no source for one), not a documented degrade.
    let empty_no_data = || {
        Ok(MarkBuildResult::empty_because(
            MarkBatchKind::Image,
            "mark_image() requires either a `url` encoding channel or Raster-transform \
             data (produced by mark_raster()); this batch has neither",
        ))
    };

    let batch = ctx.batch;
    if batch.num_rows() != 1 {
        return empty_no_data();
    }

    let x_min = match batch
        .column_by_name(RASTER_X_EXTENT[0])
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };
    let x_max = match batch
        .column_by_name(RASTER_X_EXTENT[1])
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };
    let y_min = match batch
        .column_by_name(RASTER_Y_EXTENT[0])
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };
    let y_max = match batch
        .column_by_name(RASTER_Y_EXTENT[1])
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };
    let width = match batch
        .column_by_name("width")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };
    let height = match batch
        .column_by_name("height")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };
    let pixel_bytes = match batch
        .column_by_name("pixel_data")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
    {
        Some(a) => a.value(0),
        None => return empty_no_data(),
    };

    let n_cells = (width as usize) * (height as usize);
    if pixel_bytes.len() != n_cells * 8 {
        return empty_no_data();
    }

    // The cells map through one color scale, resolved with the chart's
    // scales: the chart's color scale when the image draws color (its binding
    // names the field the cells hold), whose colorbar describes them;
    // otherwise the one scale every raster drawing no color shares
    // (`ResolvedScales::raster_cells`), across facet panels too.
    let draws_color =
        image_draws_color(crate::spec::mark::Mark::Image, &ctx.spec.encoding, &ctx.spec.encoding, raster_value_field(batch));
    let scale = match draws_color {
        true => ctx.scales.color.as_ref(),
        false => ctx.scales.raster_cells.as_ref(),
    };
    // A categorical scale has no color for a cell value; scale resolution
    // refuses one over a raster's cells, so none reaches here.
    let scale = match scale {
        Some(ColorScale::Categorical { .. }) => {
            return Ok(MarkBuildResult::empty_because(
                MarkBatchKind::Image,
                "a raster image's cells draw through a continuous or discretizing color scale",
            ))
        }
        Some(scale) => scale,
        // No cell holds a finite value: every cell is transparent.
        None if raster_cell_values(batch).next().is_none() => return empty(),
        None => {
            return Err(crate::render::RenderError::ScaleResolutionFailed(
                "a raster image's cells hold finite values, but no color scale was resolved for them".to_string(),
            ))
        }
    };

    // Decode each cell and map it through the scale to RGBA8; an empty (NaN)
    // cell is transparent.
    let mut rgba: Vec<u8> = Vec::with_capacity(n_cells * 4);
    for chunk in pixel_bytes.chunks_exact(8) {
        let v = f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]));
        match scale.lookup_f64(v).filter(|_| v.is_finite()) {
            Some(c) => rgba.extend_from_slice(&[c.red, c.green, c.blue, c.alpha]),
            None => rgba.extend_from_slice(&[0, 0, 0, 0]),
        }
    }

    let png_bytes = encode_png(width, height, &rgba);

    // Map data extent to pixel space via scales.
    let svg_x = match ctx.scales.x.to_pixel_f64(x_min) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_x_far = match ctx.scales.x.to_pixel_f64(x_max) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_y_top = match ctx.scales.y.to_pixel_f64(y_max) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_y_bot = match ctx.scales.y.to_pixel_f64(y_min) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_w = (svg_x_far - svg_x).abs();
    let svg_h = (svg_y_bot - svg_y_top).abs();

    let meta = MetadataColumns::from_ctx(ctx);
    let node = SceneNode::Image {
        x: svg_x,
        y: svg_y_top,
        w: svg_w,
        h: svg_h,
        data: ImageData::Inline {
            bytes: png_bytes,
            mime: ImageMime::Png,
        },
    };
    // The anchor is the raster's drawn top-left corner, not its center: a
    // raster covers a data rect, not a point. It is inert because
    // `for_each_along` skips `Image` nodes; if that skip is lifted, move the
    // anchor to the extent's center first.
    Ok(MarkBuildResult::populated(MarkBatchKind::Image, vec![node], vec![0], &meta)
        .with_glyph_anchors(vec![(svg_x, svg_y_top)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::color::{ContinuousScheme, NamedContinuous};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{BinaryArray, Float64Array, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// Build an image-mark ChartSpec whose x/y encoding fields point at the
    /// Raster output's `x_min`/`y_min` columns. Explicit linear scale domains
    /// are supplied because the single-row Raster batch is degenerate for
    /// auto-domain inference.
    fn image_spec(x_domain: (f64, f64), y_domain: (f64, f64)) -> ChartSpec {
        let x_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![x_domain.0, x_domain.1]),
                range: None, clamp: false, padding: None, scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false, zero: false,
            system_injected: false,
        };
        let y_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![y_domain.0, y_domain.1]),
                range: None, clamp: false, padding: None, scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false, zero: false,
            system_injected: false,
        };
        ChartSpec {
            mark: Mark::Image,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x_min".into(), type_: None,
                    scale: Some(x_scale), ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y_min".into(), type_: None,
                    scale: Some(y_scale), ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Build an image-mark ChartSpec with x/y/url encodings for URL-tile tests.
    fn url_tile_spec(x_domain: (f64, f64), y_domain: (f64, f64)) -> ChartSpec {
        let x_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![x_domain.0, x_domain.1]),
                range: None, clamp: false, padding: None, scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false, zero: false,
            system_injected: false,
        };
        let y_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![y_domain.0, y_domain.1]),
                range: None, clamp: false, padding: None, scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false, zero: false,
            system_injected: false,
        };
        ChartSpec {
            mark: Mark::Image,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(), type_: None,
                    scale: Some(x_scale), ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y".into(), type_: None,
                    scale: Some(y_scale), ..Default::default()
                }),
                url: Some(EncodingSpec {
                    field: "url".into(), type_: None,
                    ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Build a single-row Raster output batch of `count` cells, as the
    /// Raster transform publishes it (its value field and linear value
    /// scale included).
    fn raster_batch(
        x_min: f64, x_max: f64, y_min: f64, y_max: f64,
        width: u32, height: u32, pixel_bytes: Vec<u8>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x_min", DataType::Float64, false),
            Field::new("x_max", DataType::Float64, false),
            Field::new("y_min", DataType::Float64, false),
            Field::new("y_max", DataType::Float64, false),
            Field::new("width", DataType::UInt32, false),
            Field::new("height", DataType::UInt32, false),
            Field::new("pixel_data", DataType::Binary, false),
            Field::new(RASTER_VALUE_FIELD, DataType::Utf8, false),
            Field::new(RASTER_VALUE_SCALE, DataType::Utf8, false),
        ]));
        let pixel_array = BinaryArray::from(vec![pixel_bytes.as_slice()]);
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![x_min])),
                Arc::new(Float64Array::from(vec![x_max])),
                Arc::new(Float64Array::from(vec![y_min])),
                Arc::new(Float64Array::from(vec![y_max])),
                Arc::new(UInt32Array::from(vec![width])),
                Arc::new(UInt32Array::from(vec![height])),
                Arc::new(pixel_array),
                Arc::new(StringArray::from(vec!["count"])),
                Arc::new(StringArray::from(vec!["linear"])),
            ],
        )
        .unwrap()
    }

    /// Build a multi-row URL-tile batch.
    fn url_tile_batch(xs: Vec<f64>, ys: Vec<f64>, urls: Vec<&str>) -> RecordBatch {
        let n = xs.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        let url_strs: Vec<&str> = urls;
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(url_strs)),
            ],
        )
        .unwrap()
    }

    fn unit_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    #[test]
    fn image_smoke_emits_data_url() {
        // 2x2 grid: 4 f64 values packed as 8-byte LE blobs (32 bytes total).
        let values: Vec<f64> = vec![0.0, 0.33, 0.67, 1.0];
        let mut pixel_data: Vec<u8> = Vec::with_capacity(values.len() * 8);
        for v in &values {
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        let spec = image_spec((0.0, 1.0), (0.0, 1.0));
        let batch = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).expect("builds");
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Image { .. })).count(), 1,
            "expected an Image node");
        // Verify PNG data is present.
        let has_png = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Image { data, .. } = n {
                matches!(data, ferrum_scene::ImageData::Inline { mime: ferrum_scene::ImageMime::Png, .. })
            } else { false }
        });
        assert!(has_png, "expected inline PNG image data");
    }

    /// A raster whose finite cells reach the builder with no color scale is
    /// refused with a typed error, never drawn as nothing; one whose cells
    /// are all empty draws nothing, its legitimate empty.
    #[test]
    fn a_raster_with_finite_cells_and_no_scale_is_refused() {
        let pack = |values: [f64; 4]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let spec = image_spec((0.0, 1.0), (0.0, 1.0));
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let finite = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pack([0.0, 1.0, f64::NAN, 2.0]));
        let (mut scales, _) = resolve_scales(&spec, &finite, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        scales.raster_cells = None;
        scales.color = None;
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &finite, mark_style: &mark_style };
        assert_eq!(
            super::build(&ctx).err().map(|e| e.to_string()).as_deref(),
            Some("scale resolution failed: a raster image's cells hold finite values, but no color scale was resolved for them")
        );
        let empty = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pack([f64::NAN; 4]));
        let ctx = DrawCtx { batch: &empty, ..ctx };
        assert!(super::build(&ctx).expect("draws nothing").nodes.is_empty());
    }

    /// #118 RED proof: both build paths place images in Cartesian pixel
    /// space, so a `CoordPolar` chart used to render nothing with no
    /// diagnostic — the same class the mechanism now covers for
    /// segment/rect/geoshape. The Cartesian control (same fixture, no coord
    /// override) renders and stays silent, proving the warning is keyed to
    /// the coord, not to this fixture generally.
    #[test]
    fn image_under_coord_polar_warns_named_empty() {
        use crate::spec::coord::CoordKind;
        use ferrum_scene::{PolarDirection, PolarThetaChannel};

        let values: Vec<f64> = vec![0.0, 0.33, 0.67, 1.0];
        let mut pixel_data: Vec<u8> = Vec::with_capacity(values.len() * 8);
        for v in &values {
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        let mut spec = image_spec((0.0, 1.0), (0.0, 1.0));
        spec.coord = Some(CoordKind::Polar {
            theta: PolarThetaChannel::X,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: None,
            pad_angle: 0.0,
        });
        let batch = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).expect("builds");
        assert!(result.nodes.is_empty(), "CoordPolar must not render an image");
        let reason = result.empty_reason.expect("CoordPolar must name why nothing was drawn");
        assert!(reason.0.contains("CoordPolar"), "got {:?}", reason.0);
    }

    /// The `CoordGeo` mirror of the above.
    #[test]
    fn image_under_coord_geo_warns_named_empty() {
        use crate::spec::coord::{CoordKind, GeoProjection};

        let pixel_data: Vec<u8> = 1.0_f64.to_le_bytes().to_vec();
        let mut spec = image_spec((0.0, 1.0), (0.0, 1.0));
        spec.coord = Some(CoordKind::Geo { projection: GeoProjection::Mercator });
        let batch = raster_batch(0.0, 1.0, 0.0, 1.0, 1, 1, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).expect("builds");
        assert!(result.nodes.is_empty(), "CoordGeo must not render an image");
        let reason = result.empty_reason.expect("CoordGeo must name why nothing was drawn");
        assert!(reason.0.contains("CoordGeo"), "got {:?}", reason.0);
    }

    /// Cartesian control for the two tests above: the SAME fixture, no coord
    /// override, must render and stay silent.
    #[test]
    fn image_under_default_cartesian_coord_renders_and_is_silent() {
        let values: Vec<f64> = vec![0.0, 0.33, 0.67, 1.0];
        let mut pixel_data: Vec<u8> = Vec::with_capacity(values.len() * 8);
        for v in &values {
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        let spec = image_spec((0.0, 1.0), (0.0, 1.0));
        let batch = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).expect("builds");
        assert_eq!(
            result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Image { .. })).count(),
            1,
            "the default Cartesian coord must still render"
        );
        assert!(result.empty_reason.is_none(), "a rendered image must carry no empty_reason");
    }

    #[test]
    fn image_byte_size_correctness() {
        // 8x8 grid: 64 f64 values packed as 8-byte LE blobs (512 bytes total).
        let n_cells = 8 * 8;
        let mut pixel_data: Vec<u8> = Vec::with_capacity(n_cells * 8);
        for i in 0..n_cells {
            let v = (i as f64) / ((n_cells - 1) as f64);
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(pixel_data.len(), 512);
        let spec = image_spec((0.0, 8.0), (0.0, 8.0));
        let batch = raster_batch(0.0, 8.0, 0.0, 8.0, 8, 8, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).expect("builds");
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Image { .. })).count(), 1,
            "expected exactly one Image node");
    }

    #[test]
    fn image_position_via_scales() {
        // Data extent x:[0,10], y:[0,10] → with linear scales mapping to [0,100]
        // pixel range, the image should cover the full plot area.
        // 1x1 grid: one f64 value (1.0 = fully saturated) as 8-byte LE blob.
        let pixel_data: Vec<u8> = 1.0_f64.to_le_bytes().to_vec();
        let spec = image_spec((0.0, 10.0), (0.0, 10.0));
        let batch = raster_batch(0.0, 10.0, 0.0, 10.0, 1, 1, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).expect("builds");
        // With scales mapping data [0,10] -> pixel [0,100], the image should span
        // x=0, w=100, h=100.
        let img = result.nodes.iter().find(|n| matches!(n, ferrum_scene::SceneNode::Image { .. }));
        assert!(img.is_some(), "expected Image node");
        if let Some(ferrum_scene::SceneNode::Image { x, y, w, h, .. }) = img {
            assert!((*x - 0.0).abs() < 1.0, "expected x near 0, got {x}");
            assert!((*y - 0.0).abs() < 1.0, "expected y near 0, got {y}");
            assert!((*w - 100.0).abs() < 1.0, "expected w near 100, got {w}");
            assert!((*h - 100.0).abs() < 1.0, "expected h near 100, got {h}");
        }
    }

    // ── URL-tile path tests ────────────────────────────────────────────────

    /// Minimal 1×1 red PNG as a base64 data URL.
    fn red_1x1_png_url() -> String {
        // Minimal valid 1x1 RGB PNG (no alpha).
        let png_bytes: Vec<u8> = vec![
            137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,2,0,0,0,
            144,119,83,222,0,0,0,12,73,68,65,84,8,215,99,248,207,192,0,0,0,2,0,1,
            227,33,188,51,0,0,0,0,73,69,78,68,174,66,96,130,
        ];
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        format!("data:image/png;base64,{b64}")
    }

    #[test]
    fn url_tile_emits_one_image_per_row() {
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let urls = vec![url.as_str(), url.as_str(), url.as_str()];
        let batch = url_tile_batch(
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 2.0],
            urls,
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");
        assert_eq!(
            result.nodes.len(), 3,
            "expected 3 Image nodes for 3 input rows, got {}", result.nodes.len()
        );
    }

    #[test]
    fn url_tile_uses_url_variant() {
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(
            vec![1.0],
            vec![2.0],
            vec![url.as_str()],
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");
        assert_eq!(result.nodes.len(), 1, "expected 1 Image node");
        let has_url_variant = result.nodes.iter().any(|n| {
            matches!(n, ferrum_scene::SceneNode::Image {
                data: ferrum_scene::ImageData::Url { .. }, ..
            })
        });
        assert!(has_url_variant, "expected ImageData::Url variant for URL-tile path");
    }

    #[test]
    fn url_tile_center_anchored() {
        // With x=3.0, domain [0,6] → pixel 50.0; default tile size 32×32
        // → top-left should be at pixel (50 - 16, 50 - 16) = (34, 34).
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(vec![3.0], vec![3.0], vec![url.as_str()]);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");
        assert_eq!(result.nodes.len(), 1);
        if let ferrum_scene::SceneNode::Image { x, y, w, h, .. } = &result.nodes[0] {
            // Center point is at pixel 50; tile half-size is 16.
            assert!((*x - 34.0).abs() < 1.0, "expected x≈34 (center-anchored), got {x}");
            assert!((*y - 34.0).abs() < 1.0, "expected y≈34 (center-anchored), got {y}");
            assert!((*w - 32.0).abs() < 1.0, "expected w=32 (default), got {w}");
            assert!((*h - 32.0).abs() < 1.0, "expected h=32 (default), got {h}");
        } else {
            panic!("expected Image node");
        }
    }

    #[test]
    fn url_tile_skips_non_data_url_rows() {
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 2.0],
            // Second row has a plain URL (not a data URL) → should be skipped.
            vec![url.as_str(), "https://example.com/img.png", url.as_str()],
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");
        assert_eq!(result.nodes.len(), 2, "plain URL row should be silently skipped");
    }

    // ── R3: metadata wiring tests ─────────────────────────────────────────

    /// Build a multi-row URL-tile batch that also carries a `tooltip` string column.
    fn url_tile_batch_with_tooltip(
        xs: Vec<f64>,
        ys: Vec<f64>,
        urls: Vec<&str>,
        tips: Vec<&str>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("url", DataType::Utf8, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(urls)),
                Arc::new(StringArray::from(tips)),
            ],
        )
        .unwrap()
    }

    /// URL-tile spec with tooltip encoding added.
    fn url_tile_spec_with_tooltip(x_domain: (f64, f64), y_domain: (f64, f64)) -> ChartSpec {
        let x_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![x_domain.0, x_domain.1]),
                range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false,
            },
            nice: false, zero: false,
            system_injected: false,
        };
        let y_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![y_domain.0, y_domain.1]),
                range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false,
            },
            nice: false, zero: false,
            system_injected: false,
        };
        ChartSpec {
            data: crate::spec::data_ref::DataRef::default(),
            mark: Mark::Image,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(), type_: None,
                    scale: Some(x_scale), ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y".into(), type_: None,
                    scale: Some(y_scale), ..Default::default()
                }),
                url: Some(EncodingSpec { field: "url".into(), type_: None, ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // ── Test: URL-tile tooltip reaches each emitted node ─────────────────

    #[test]
    fn r3_image_url_tile_tooltip_reaches_each_node() {
        let spec = url_tile_spec_with_tooltip((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch_with_tooltip(
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 2.0],
            vec![url.as_str(), url.as_str(), url.as_str()],
            vec!["tile-A", "tile-B", "tile-C"],
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");

        assert_eq!(result.nodes.len(), 3, "expected 3 Image nodes");

        // Fail-before: tooltips was always None; pass-after: Some with per-node values.
        let tooltips = result.tooltips.as_ref()
            .expect("tooltips must be Some when tooltip encoding is set");
        assert_eq!(tooltips.len(), 3, "tooltip vector must align with nodes");
        assert_eq!(tooltips[0].fields[0].value, "tile-A", "node 0 tooltip must be 'tile-A'");
        assert_eq!(tooltips[1].fields[0].value, "tile-B", "node 1 tooltip must be 'tile-B'");
        assert_eq!(tooltips[2].fields[0].value, "tile-C", "node 2 tooltip must be 'tile-C'");
    }

    // ── Test: backward compat — URL-tile without metadata encoding ────────

    #[test]
    fn r3_image_url_tile_no_metadata_encoding_metadata_is_none() {
        // Chart without tooltip/href/description — metadata fields must be None.
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(
            vec![1.0, 3.0],
            vec![2.0, 4.0],
            vec![url.as_str(), url.as_str()],
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");

        assert_eq!(result.nodes.len(), 2);
        assert!(result.tooltips.is_none(), "tooltips must be None when no tooltip encoding");
        assert!(result.hrefs.is_none(), "hrefs must be None when no href encoding");
        assert!(result.descriptions.is_none(), "descriptions must be None when no description encoding");
    }

    // ── Test: backward compat — raster path without metadata encoding ─────

    #[test]
    fn r3_image_raster_no_metadata_encoding_metadata_is_none() {
        let n_cells = 2 * 2;
        let mut pixel_data: Vec<u8> = Vec::with_capacity(n_cells * 8);
        for i in 0..n_cells {
            let v = i as f64 / (n_cells - 1) as f64;
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        let spec = image_spec((0.0, 1.0), (0.0, 1.0));
        let batch = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx).expect("builds");

        assert_eq!(result.nodes.len(), 1, "raster path must emit exactly one node");
        assert!(result.tooltips.is_none(), "raster without tooltip encoding: tooltips must be None");
        assert!(result.hrefs.is_none(), "raster without href encoding: hrefs must be None");
        assert!(result.descriptions.is_none(), "raster without description encoding: descriptions must be None");
    }

    /// The one drawn-column answer: a raster image draws `x`/`y` through its
    /// extent columns; a url-tile image, any other mark and any other
    /// channel draw the bound field.
    #[test]
    fn drawn_positional_columns_name_a_raster_images_extent_columns() {
        let url = EncodingSpec { field: "u".into(), ..Default::default() };
        assert_eq!(drawn_columns(Mark::Image, None, "x", "x"), RASTER_X_EXTENT);
        assert_eq!(drawn_columns(Mark::Image, None, "y", "y"), RASTER_Y_EXTENT);
        assert_eq!(drawn_columns(Mark::Image, None, "x2", "x2"), ["x2"]);
        assert_eq!(drawn_columns(Mark::Image, Some(&url), "x", "x"), ["x"]);
        assert_eq!(drawn_columns(Mark::Point, None, "x", "x"), ["x"]);
    }

    /// The published drawn-column answer: a raster image's `x`/`y` draw its
    /// extent columns and its `color` the column naming its cells' field; a
    /// url-tile image and every other mark draw their own field. Python's
    /// auto tooltip lists a positional field only when the answer is the
    /// field itself.
    #[test]
    fn the_published_drawn_columns_answer_matches_the_render() {
        assert_eq!(drawn_positional_columns_py("image", false, "x", "x").unwrap(), ["x_min", "x_max"]);
        assert_eq!(drawn_positional_columns_py("image", false, "color", "z").unwrap(), ["value_field"]);
        assert_eq!(drawn_positional_columns_py("image", true, "x", "x").unwrap(), ["x"]);
        assert_eq!(drawn_positional_columns_py("point", false, "x", "x").unwrap(), ["x"]);
        assert!(drawn_positional_columns_py("nope", false, "x", "x").is_err());
    }

    /// A raster image drawing no color maps its cells through a continuous
    /// scale over its own cell extent: the lowest drawn value takes the
    /// scheme's first color, the highest its last, an empty cell stays
    /// transparent. Rejects the private `[0, vmax]` normalization, under
    /// which a negative mean took the floor color with every value below
    /// zero.
    #[test]
    fn a_raster_image_maps_its_cells_through_one_scale_over_their_extent() {
        let cells = [-2.0, 6.0, f64::NAN, 2.0];
        let bytes: Vec<u8> = cells.iter().flat_map(|v| v.to_le_bytes()).collect();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x_min", DataType::Float64, false),
                Field::new("x_max", DataType::Float64, false),
                Field::new("y_min", DataType::Float64, false),
                Field::new("y_max", DataType::Float64, false),
                Field::new("width", DataType::UInt32, false),
                Field::new("height", DataType::UInt32, false),
                Field::new("pixel_data", DataType::Binary, false),
                Field::new(RASTER_VALUE_FIELD, DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(UInt32Array::from(vec![2])),
                Arc::new(UInt32Array::from(vec![2])),
                Arc::new(BinaryArray::from(vec![bytes.as_slice()])),
                Arc::new(StringArray::from(vec!["z"])),
            ],
        )
        .unwrap();
        assert_eq!(raster_value_field(&batch), Some("z"));
        assert_eq!(raster_cell_extent(&batch), Some((-2.0, 6.0)), "the finite cells' extent, NaN skipped");
        let scheme = ContinuousScheme::Named(NamedContinuous::Viridis);
        let scale = ColorScale::Continuous { domain: (-2.0, 6.0), scheme: scheme.clone(), midpoint: None, transform: crate::render::scale_resolve::ColorTransform::Linear };
        assert_eq!(scale.lookup_f64(-2.0), Some(scheme.sample(0.0)));
        assert_eq!(scale.lookup_f64(6.0), Some(scheme.sample(1.0)));
    }

    /// A raster's cell domain runs from zero when every cell is non-negative,
    /// so a count of 1 maps off the scheme's first color and a uniform raster
    /// draws its top color; a mean spanning negatives keeps its own
    /// `[min, max]`. An all-zero raster maps to the midpoint, dividing by
    /// nothing. Rejects the `[min, max]` mapping, under which count-1 cells
    /// took the scheme's first, near-background color.
    #[test]
    fn a_raster_cell_domain_runs_from_zero_when_every_cell_is_non_negative() {
        let scheme = ContinuousScheme::Named(NamedContinuous::Viridis);
        let through = |cells: (f64, f64), v: f64| {
            ColorScale::Continuous { domain: raster_cell_domain(cells), scheme: scheme.clone(), midpoint: None, transform: crate::render::scale_resolve::ColorTransform::Linear }
                .lookup_f64(v)
        };
        assert_eq!(raster_cell_domain((1.0, 5.0)), (0.0, 5.0));
        assert_eq!(through((1.0, 5.0), 1.0), Some(scheme.sample(0.2)), "a count of 1 is off t=0");
        assert_ne!(through((1.0, 5.0), 1.0), Some(scheme.sample(0.0)));
        assert_eq!(through((3.0, 3.0), 3.0), Some(scheme.sample(1.0)), "a uniform raster draws the top color");
        assert_eq!(raster_cell_domain((-2.0, 6.0)), (-2.0, 6.0), "a negative-spanning mean keeps its range");
        assert_eq!(through((-2.0, 6.0), -2.0), Some(scheme.sample(0.0)));
        assert_eq!(raster_cell_domain((0.0, 0.0)), (0.0, 0.0));
        assert_eq!(through((0.0, 0.0), 0.0), Some(scheme.sample(0.5)), "an all-zero raster draws, at the midpoint");
    }
}
