//! Phase 9 grid compositor — SVG row-major grid with row/col ratios and spacing.
//!
//! Used by JointChart, RepeatChart, ClusterMapChart. share_x/share_y groups
//! are accepted at the Python binding layer and ignored here; callers ensure
//! their child SVGs have aligned plot areas at construction time (e.g. via
//! Chart.properties). Coordinate-system rebinding within the SVG body is out
//! of scope for Phase 9.

use crate::render::compositor::{
    parse_svg_root, write_cell, write_scaled_cell, write_svg_open, CompositorError,
};
#[cfg(test)]
use crate::render::svg::fmt_f;

/// An owned representation of a parsed SVG cell, used so we can collect
/// dimension info in one pass and emit in a second pass without fighting the
/// borrow checker (ParsedSvg<'a> borrows from the input &str).
struct OwnedCell {
    width: f64,
    height: f64,
    body: String,
}

/// Compose a row-major grid of SVGs with ratio-weighted row/col sizing.
///
/// `cells[i*cols + j]` is row i, column j; `None` = empty (skipped) cell.
///
/// **Sizing rule (F20):** each col's allocated width is
/// `K_w * col_ratios[c]`, where `K_w` is chosen so the largest native
/// cell width in each col never exceeds its allocation. Same for rows.
/// When the callers pre-resize cells so that native dimensions match
/// the ratios exactly (Phase 9 JointChart/ClusterMapChart behavior pre-
/// F20b), this is byte-identical to the prior "max-per-row/col" algorithm.
/// When cells are at native size with non-matching ratios, the cell is
/// scaled into its slot via a nested `<svg viewBox preserveAspectRatio="none">`
/// wrapper. The current callers always satisfy the no-scaling case for
/// every cell (JointChart's marginals shrink the data axis but match
/// in the ratio dimension); even so we emit scaled wrappers when any
/// slot diverges from native to keep the contract symmetric.
///
/// `spacing` is in absolute SVG units between adjacent cells.
pub fn compose_svg_grid(
    cells: &[Option<String>],
    rows: usize,
    cols: usize,
    row_ratios: &[f64],
    col_ratios: &[f64],
    spacing: f64,
) -> Result<String, CompositorError> {
    if cells.len() != rows * cols {
        return Err(CompositorError::LengthMismatch {
            what: "cells",
            expected: rows * cols,
            got: cells.len(),
        });
    }
    if row_ratios.len() != rows {
        return Err(CompositorError::LengthMismatch {
            what: "row_ratios",
            expected: rows,
            got: row_ratios.len(),
        });
    }
    if col_ratios.len() != cols {
        return Err(CompositorError::LengthMismatch {
            what: "col_ratios",
            expected: cols,
            got: col_ratios.len(),
        });
    }
    let row_sum: f64 = row_ratios.iter().sum();
    let col_sum: f64 = col_ratios.iter().sum();
    if row_sum <= 0.0 || col_sum <= 0.0 {
        return Err(CompositorError::InvalidRatios);
    }
    // F24: an all-None cell list is conceptually empty — surface it as
    // EmptyInput (matching the 1D composers) rather than silently
    // emitting a 0×0 SVG.
    if cells.iter().all(Option::is_none) {
        return Err(CompositorError::EmptyInput);
    }

    // First pass: parse each non-None cell, record max native dimension per row/col.
    let mut native_col_w = vec![0.0_f64; cols];
    let mut native_row_h = vec![0.0_f64; rows];
    let mut owned: Vec<Option<OwnedCell>> = Vec::with_capacity(cells.len());

    for (idx, opt) in cells.iter().enumerate() {
        if let Some(svg) = opt {
            let p = parse_svg_root(svg)?;
            let r = idx / cols;
            let c = idx % cols;
            native_col_w[c] = native_col_w[c].max(p.width);
            native_row_h[r] = native_row_h[r].max(p.height);
            owned.push(Some(OwnedCell {
                width: p.width,
                height: p.height,
                body: p.body.to_owned(),
            }));
        } else {
            owned.push(None);
        }
    }

    // Compute scaling factors so the dominant-share row/col renders at its
    // native size, and smaller-share rows/cols get proportionally smaller
    // (and stretch-scaled) allocations.
    //
    //   col_widths[c] = K_w * col_ratios[c]
    //   K_w = MIN over c of (native_col_w[c] / col_ratios[c])
    //
    // Pick MIN rather than MAX: MAX would expand the dominant cell to make
    // room for an over-sized small-share cell at native; MIN keeps the
    // dominant cell at native and shrinks the smaller-share cell to fit
    // its allocated share. For JointChart this is the intent — the centre
    // panel is at native; marginals scale to their narrow strips. For
    // ClusterMapChart's pre-sized dendrograms K_w is uniform across cols
    // (intentional pre-resize), so MIN == MAX and the layout is
    // byte-identical to the prior algorithm.
    //
    // Cells that pre-resize cleanly (native matches slot for every row/col)
    // keep the lightweight `<g transform>` wrapper in the emit loop below.
    // Cells whose native dim differs from their slot get a nested
    // `<svg viewBox preserveAspectRatio="none">` wrapper that stretches
    // the body to fit (see compositor::write_scaled_cell).
    let k_w = col_ratios
        .iter()
        .zip(native_col_w.iter())
        .filter_map(|(r, nw)| if *r > 0.0 && *nw > 0.0 { Some(nw / r) } else { None })
        .fold(f64::INFINITY, f64::min);
    let k_h = row_ratios
        .iter()
        .zip(native_row_h.iter())
        .filter_map(|(r, nh)| if *r > 0.0 && *nh > 0.0 { Some(nh / r) } else { None })
        .fold(f64::INFINITY, f64::min);
    let k_w = if k_w.is_finite() { k_w } else { 0.0 };
    let k_h = if k_h.is_finite() { k_h } else { 0.0 };
    let col_widths: Vec<f64> = col_ratios.iter().map(|r| k_w * r).collect();
    let row_heights: Vec<f64> = row_ratios.iter().map(|r| k_h * r).collect();

    let total_w: f64 = col_widths.iter().sum::<f64>()
        + spacing * (cols.saturating_sub(1)) as f64;
    let total_h: f64 = row_heights.iter().sum::<f64>()
        + spacing * (rows.saturating_sub(1)) as f64;

    let capacity: usize = cells
        .iter()
        .filter_map(|c| c.as_ref().map(|s| s.len()))
        .sum::<usize>()
        + 256;
    let mut out = String::with_capacity(capacity);
    write_svg_open(&mut out, total_w, total_h);

    // Second pass: emit cells with translate-or-viewBox wrappers depending
    // on whether the cell's native size matches its allocated slot.
    let mut first_emitted = false;
    let mut y_offset = 0.0_f64;
    for r in 0..rows {
        let mut x_offset = 0.0_f64;
        for c in 0..cols {
            let idx = r * cols + c;
            if let Some(cell) = &owned[idx] {
                let is_first = !first_emitted;
                first_emitted = true;
                let slot_w = col_widths[c];
                let slot_h = row_heights[r];
                let near_eq = |a: f64, b: f64| (a - b).abs() < 1e-6;
                if near_eq(cell.width, slot_w) && near_eq(cell.height, slot_h) {
                    // No scaling needed — keep the lightweight <g translate> wrapper.
                    write_cell(&mut out, x_offset, y_offset, idx, &cell.body, is_first);
                } else {
                    // Slot != native: scale the cell to fit via nested <svg>
                    // with preserveAspectRatio="none". The inner content's
                    // viewBox preserves its coordinate system; only the outer
                    // bounding box stretches.
                    write_scaled_cell(
                        &mut out, x_offset, y_offset, slot_w, slot_h,
                        cell.width, cell.height, idx, &cell.body, is_first,
                    );
                }
            }
            x_offset += col_widths[c] + if c + 1 < cols { spacing } else { 0.0 };
        }
        y_offset += row_heights[r] + if r + 1 < rows { spacing } else { 0.0 };
    }
    out.push_str("</svg>");
    Ok(out)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn svg(w: f64, h: f64, fill: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}"><rect x="0" y="0" width="{}" height="{}" fill="{}" /></svg>"#,
            fmt_f(w),
            fmt_f(h),
            fmt_f(w),
            fmt_f(h),
            fmt_f(w),
            fmt_f(h),
            fill,
        )
    }

    #[test]
    fn compose_2x2_grid_with_ratios_and_spacing() {
        let a = svg(50.0, 50.0, "red");
        let b = svg(50.0, 50.0, "blue");
        let c = svg(50.0, 50.0, "green");
        let d = svg(50.0, 50.0, "yellow");
        let cells = vec![Some(a), Some(b), Some(c), Some(d)];
        let out =
            compose_svg_grid(&cells, 2, 2, &[1.0, 1.0], &[1.0, 1.0], 5.0).unwrap();
        assert!(out.contains(r#"width="105""#), "out: {out}");
        assert!(out.contains(r#"height="105""#), "out: {out}");
        assert!(out.contains(r#"transform="translate(0,0)""#), "out: {out}");
        assert!(out.contains(r#"translate(55,0)"#), "out: {out}");
        assert!(out.contains(r#"translate(0,55)"#), "out: {out}");
        assert!(out.contains(r#"translate(55,55)"#), "out: {out}");
    }

    #[test]
    fn compose_grid_with_none_cell_skips_empty_position() {
        let a = svg(40.0, 40.0, "red");
        let b = svg(40.0, 40.0, "blue");
        let cells = vec![Some(a), None, Some(b), None];
        let out =
            compose_svg_grid(&cells, 2, 2, &[1.0, 1.0], &[1.0, 1.0], 0.0).unwrap();
        assert!(out.contains(r#"translate(0,0)"#), "out: {out}");
        assert!(out.contains(r#"translate(0,40)"#), "out: {out}");
        assert!(!out.contains(r#"translate(40,0)"#), "out: {out}");
    }

    #[test]
    fn compose_grid_size_mismatch_errors() {
        let cells: Vec<Option<String>> = vec![None];
        let err =
            compose_svg_grid(&cells, 2, 2, &[1.0, 1.0], &[1.0, 1.0], 0.0).unwrap_err();
        assert!(matches!(
            err,
            CompositorError::LengthMismatch { what: "cells", expected: 4, got: 1 }
        ), "got: {err:?}");
    }

    #[test]
    fn compose_grid_row_ratios_arity_errors() {
        let cells: Vec<Option<String>> = vec![None, None, None, None];
        let err = compose_svg_grid(&cells, 2, 2, &[1.0], &[1.0, 1.0], 0.0).unwrap_err();
        assert!(matches!(err, CompositorError::LengthMismatch { what: "row_ratios", .. }));
    }

    #[test]
    fn compose_grid_zero_ratios_errors() {
        let a = svg(10.0, 10.0, "red");
        let cells = vec![Some(a), None, None, None];
        let err = compose_svg_grid(&cells, 2, 2, &[0.0, 0.0], &[1.0, 1.0], 0.0).unwrap_err();
        assert!(matches!(err, CompositorError::InvalidRatios));
    }

    #[test]
    fn compose_grid_all_none_errors() {
        // F24: all-None grid surfaces as EmptyInput (parallel to 1D composers).
        let cells: Vec<Option<String>> = vec![None, None, None, None];
        let err = compose_svg_grid(&cells, 2, 2, &[1.0, 1.0], &[1.0, 1.0], 0.0).unwrap_err();
        assert!(matches!(err, CompositorError::EmptyInput));
    }
}
