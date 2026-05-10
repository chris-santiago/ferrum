//! Phase 9 grid compositor — SVG row-major grid with row/col ratios and spacing.
//!
//! Used by JointChart, RepeatChart, ClusterMapChart. share_x/share_y groups
//! are accepted at the Python binding layer and ignored here; callers ensure
//! their child SVGs have aligned plot areas at construction time (e.g. via
//! Chart.properties). Coordinate-system rebinding within the SVG body is out
//! of scope for Phase 9.

use crate::render::compositor::{parse_svg_root, strip_font_defs, CompositorError};
use crate::render::svg::fmt_f;

/// An owned representation of a parsed SVG cell, used so we can collect
/// dimension info in one pass and emit in a second pass without fighting the
/// borrow checker (ParsedSvg<'a> borrows from the input &str).
struct OwnedCell {
    width: f64,
    height: f64,
    body: String,
}

/// Compose a row-major grid of SVGs.
///
/// `cells[i*cols + j]` is row i, column j; `None` = empty (skipped) cell.
/// `row_ratios` and `col_ratios` are accepted for forward-compatibility but
/// are not used to scale cells in Phase 9 — instead each row/col takes the
/// maximum size of its non-empty cells.
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
        return Err(CompositorError::EmptyInput);
    }
    if row_ratios.len() != rows || col_ratios.len() != cols {
        return Err(CompositorError::EmptyInput);
    }
    let row_sum: f64 = row_ratios.iter().sum();
    let col_sum: f64 = col_ratios.iter().sum();
    if row_sum <= 0.0 || col_sum <= 0.0 {
        return Err(CompositorError::EmptyInput);
    }

    // First pass: parse each non-None cell, record max dimensions per row/col.
    let mut col_widths = vec![0.0_f64; cols];
    let mut row_heights = vec![0.0_f64; rows];
    let mut owned: Vec<Option<OwnedCell>> = Vec::with_capacity(cells.len());

    for (idx, opt) in cells.iter().enumerate() {
        if let Some(svg) = opt {
            let p = parse_svg_root(svg)?;
            let r = idx / cols;
            let c = idx % cols;
            col_widths[c] = col_widths[c].max(p.width);
            row_heights[r] = row_heights[r].max(p.height);
            owned.push(Some(OwnedCell {
                width: p.width,
                height: p.height,
                body: p.body.to_owned(),
            }));
        } else {
            owned.push(None);
        }
    }

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
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        fmt_f(total_w),
        fmt_f(total_h),
        fmt_f(total_w),
        fmt_f(total_h),
    ));

    // Second pass: emit cells with translate transforms.
    let mut first_emitted = false;
    let mut y_offset = 0.0_f64;
    for r in 0..rows {
        let mut x_offset = 0.0_f64;
        for c in 0..cols {
            let idx = r * cols + c;
            if let Some(cell) = &owned[idx] {
                out.push_str(&format!(
                    r#"<g transform="translate({},{})">"#,
                    fmt_f(x_offset),
                    fmt_f(y_offset),
                ));
                if !first_emitted {
                    out.push_str(&cell.body);
                    first_emitted = true;
                } else {
                    let stripped = strip_font_defs(&cell.body);
                    out.push_str(&stripped);
                }
                out.push_str("</g>");
                let _ = (cell.width, cell.height); // suppress dead-code warning
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
        assert!(matches!(err, CompositorError::EmptyInput));
    }
}
