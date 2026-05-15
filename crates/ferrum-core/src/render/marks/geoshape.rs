//! mark_geoshape — GeoJSON polygon rendering.
//!
//! Reads the `__geometry__` string column from the RecordBatch (added by
//! `_coerce.py`), deserializes each GeoJSON geometry, projects coordinates
//! using the CoordGeo projection, and emits Polygon scene nodes.

use ferrum_scene::{FillStroke, MarkBatchKind, SceneNode};
use geojson::{Geometry, Value as GeoValue};

use crate::render::arrow_cast::col_as_str;
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_color, DrawCtx, MarkBuildResult};
use crate::spec::coord::CoordKind as SpecCoord;

const GEOMETRY_COL: &str = "__geometry__";

pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let projection = match &ctx.spec.coord {
        Some(SpecCoord::Geo { projection }) => *projection,
        _ => return MarkBuildResult::empty(MarkBatchKind::Polygon),
    };

    let Ok(geom_strs) = col_as_str(ctx.batch, GEOMETRY_COL) else {
        return MarkBuildResult::empty(MarkBatchKind::Polygon);
    };

    let fill = with_opacity(ctx.mark_style.fill, ctx.mark_style.opacity);
    let stroke = ctx.mark_style.stroke.map(|s| with_opacity(s, ctx.mark_style.opacity));

    // Two-pass: collect all projected rings to find bounding box for scaling.
    let pa = &ctx.panel.plot_area;
    let mut all_rings: Vec<(usize, Vec<[f64; 2]>)> = Vec::new();

    // Collect min/max in normalized projection space.
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (i, s_opt) in geom_strs.iter().enumerate() {
        let s = match s_opt { Some(s) => s, None => continue };
        let geom: Geometry = match s.parse() {
            Ok(g) => g,
            Err(_) => continue,
        };
        let rings = extract_exterior_rings(&geom);
        for ring in rings {
            let projected: Vec<[f64; 2]> = ring.iter().map(|&[lon, lat]| {
                let (x, y) = crate::projection::forward(projection, lon, lat);
                [x, y]
            }).filter(|[x, y]| x.is_finite() && y.is_finite()).collect();
            if projected.is_empty() { continue; }
            for [x, y] in &projected {
                if *x < min_x { min_x = *x; }
                if *x > max_x { max_x = *x; }
                if *y < min_y { min_y = *y; }
                if *y > max_y { max_y = *y; }
            }
            all_rings.push((i, projected));
        }
    }

    if all_rings.is_empty() || !min_x.is_finite() {
        return MarkBuildResult::empty(MarkBatchKind::Polygon);
    }

    // Scale projected coords to pixel space, preserving aspect ratio.
    let data_w = (max_x - min_x).max(1e-9);
    let data_h = (max_y - min_y).max(1e-9);
    let scale_x = pa.w / data_w;
    let scale_y = pa.h / data_h;
    let scale = scale_x.min(scale_y);
    let offset_x = pa.x + (pa.w - data_w * scale) / 2.0;
    // y is flipped: larger lat → smaller pixel y.
    let offset_y = pa.y + pa.h - (pa.h - data_h * scale) / 2.0;

    let fill_style = FillStroke {
        fill: Some(to_scene_color(fill)),
        stroke: stroke.map(to_scene_color),
        stroke_width: ctx.mark_style.stroke_width,
        opacity: ctx.mark_style.opacity,
        stroke_dash: None,
    };

    let mut nodes: Vec<SceneNode> = Vec::with_capacity(all_rings.len());
    let mut data_indices: Vec<usize> = Vec::with_capacity(all_rings.len());

    for (i, ring) in all_rings {
        let pixels: Vec<[f64; 2]> = ring.iter().map(|&[px, py]| {
            let sx = offset_x + (px - min_x) * scale;
            let sy = offset_y - (py - min_y) * scale;
            [sx, sy]
        }).collect();
        nodes.push(SceneNode::Polygon { points: pixels, style: fill_style.clone() });
        data_indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Polygon,
        nodes,
        data_indices: Some(data_indices),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}

/// Extract exterior rings from a GeoJSON geometry (exterior ring only; no holes).
fn extract_exterior_rings(geom: &Geometry) -> Vec<Vec<[f64; 2]>> {
    match &geom.value {
        GeoValue::Polygon(rings) => {
            rings.first().map(|r| vec![ring_to_array(r)]).unwrap_or_default()
        }
        GeoValue::MultiPolygon(polys) => {
            polys.iter().filter_map(|p| p.first().map(|r| ring_to_array(r))).collect()
        }
        GeoValue::GeometryCollection(geoms) => {
            geoms.iter().flat_map(extract_exterior_rings).collect()
        }
        _ => Vec::new(),
    }
}

fn ring_to_array(ring: &[Vec<f64>]) -> Vec<[f64; 2]> {
    ring.iter().filter_map(|c| {
        if c.len() >= 2 { Some([c[0], c[1]]) } else { None }
    }).collect()
}
