//! mark_geoshape — GeoJSON geometry rendering.
//!
//! Reads the `__geometry__` string column from the RecordBatch (added by
//! `_coerce.py`), deserializes each GeoJSON geometry, projects coordinates
//! using the CoordGeo projection, and emits scene nodes:
//!
//! - `Polygon` / `MultiPolygon` → `SceneNode::Polygon` with exterior + hole rings.
//!   The SVG renderer emits `fill-rule="evenodd"` so holes cut through the fill.
//! - `Point` / `MultiPoint` → `SceneNode::Circle` (radius `POINT_RADIUS` px).
//! - `LineString` / `MultiLineString` → `SceneNode::Polyline` (no fill).
//! - `GeometryCollection` → recursively handled.

use ferrum_scene::{FillStroke, MarkBatchKind, SceneNode};
use geojson::{Geometry, Value as GeoValue};

use crate::render::arrow_cast::col_as_str;
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_color, to_scene_stroke, DrawCtx, MarkBuildResult};
use crate::spec::coord::CoordKind as SpecCoord;

const GEOMETRY_COL: &str = "__geometry__";
/// Default radius for Point / MultiPoint glyphs (screen pixels).
const POINT_RADIUS: f64 = 3.0;

// ── Intermediate geometry collected from a single data row ───────────

/// All projected geometry from a single data row, in raw projection space
/// (not yet scaled to pixels). Grouped by emission type.
struct RowGeometry {
    /// GeoJSON data-row index (for `data_indices`).
    row_idx: usize,
    /// Polygon ring sets: each entry is `[exterior_ring, hole_ring*, …]`.
    polygons: Vec<Vec<Vec<[f64; 2]>>>,
    /// Single projected coordinates for Point/MultiPoint glyphs.
    points: Vec<[f64; 2]>,
    /// Sequences of coordinates for LineString/MultiLineString.
    lines: Vec<Vec<[f64; 2]>>,
}

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

    let pa = &ctx.panel.plot_area;

    // ── Pass 1: project all geometries; collect bounding box ──────────

    let mut rows: Vec<RowGeometry> = Vec::new();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let project = |lon: f64, lat: f64| -> Option<[f64; 2]> {
        let (x, y) = crate::projection::forward(projection, lon, lat);
        if x.is_finite() && y.is_finite() { Some([x, y]) } else { None }
    };

    let project_ring = |ring: &[Vec<f64>]| -> Vec<[f64; 2]> {
        ring.iter()
            .filter_map(|c| if c.len() >= 2 { project(c[0], c[1]) } else { None })
            .collect()
    };

    for (i, s_opt) in geom_strs.iter().enumerate() {
        let s = match s_opt { Some(s) => s, None => continue };
        let geom: Geometry = match s.parse() {
            Ok(g) => g,
            Err(_) => continue,
        };

        let mut row = RowGeometry { row_idx: i, polygons: Vec::new(), points: Vec::new(), lines: Vec::new() };
        collect_geometry(&geom, &project_ring, &project, &mut row);

        // Extend global bounding box from all projected coords in this row.
        for poly_rings in &row.polygons {
            for ring in poly_rings {
                for &[x, y] in ring {
                    if x < min_x { min_x = x; }
                    if x > max_x { max_x = x; }
                    if y < min_y { min_y = y; }
                    if y > max_y { max_y = y; }
                }
            }
        }
        for &[x, y] in &row.points {
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }
        for line in &row.lines {
            for &[x, y] in line {
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }

        rows.push(row);
    }

    if rows.is_empty() || !min_x.is_finite() {
        return MarkBuildResult::empty(MarkBatchKind::Polygon);
    }

    // ── Scaling: project-space → pixel-space, preserving aspect ratio ─

    let data_w = (max_x - min_x).max(1e-9);
    let data_h = (max_y - min_y).max(1e-9);
    let scale_x = pa.w / data_w;
    let scale_y = pa.h / data_h;
    let scale = scale_x.min(scale_y);
    let offset_x = pa.x + (pa.w - data_w * scale) / 2.0;
    // y is flipped: larger latitude → smaller pixel y.
    let offset_y = pa.y + pa.h - (pa.h - data_h * scale) / 2.0;

    let to_pixel = |coord: [f64; 2]| -> [f64; 2] {
        [
            offset_x + (coord[0] - min_x) * scale,
            offset_y - (coord[1] - min_y) * scale,
        ]
    };

    // ── Style setup ───────────────────────────────────────────────────

    let fill_style = FillStroke {
        fill: Some(to_scene_color(fill)),
        stroke: stroke.map(to_scene_color),
        stroke_width: ctx.mark_style.stroke_width,
        opacity: ctx.mark_style.opacity,
        stroke_dash: None,
        stroke_opacity: 1.0,
        fill_opacity: 1.0,
        angle: 0.0,
    };

    // Stroke color for lines: prefer explicit stroke, fall back to fill.
    let stroke_color = stroke.unwrap_or(fill);
    let stroke_style = to_scene_stroke(
        stroke_color,
        ctx.mark_style.stroke_width.max(1.0),
        ctx.mark_style.opacity,
        None,
        None,
        None,
    );

    // ── Pass 2: emit scene nodes ──────────────────────────────────────

    let mut nodes: Vec<SceneNode> = Vec::new();
    let mut data_indices: Vec<usize> = Vec::new();

    for row in rows {
        for poly_rings in row.polygons {
            if poly_rings.is_empty() { continue; }
            let pixel_rings: Vec<Vec<[f64; 2]>> = poly_rings
                .iter()
                .filter(|r| !r.is_empty())
                .map(|r| r.iter().map(|&c| to_pixel(c)).collect())
                .collect();
            if pixel_rings.is_empty() { continue; }
            nodes.push(SceneNode::Polygon { rings: pixel_rings, style: fill_style.clone() });
            data_indices.push(row.row_idx);
        }

        for pt in row.points {
            let [cx, cy] = to_pixel(pt);
            nodes.push(SceneNode::Circle { cx, cy, r: POINT_RADIUS, style: fill_style.clone() });
            data_indices.push(row.row_idx);
        }

        for line in row.lines {
            if line.len() < 2 { continue; }
            let pixel_pts: Vec<(f64, f64)> = line.iter().map(|&c| {
                let [x, y] = to_pixel(c);
                (x, y)
            }).collect();
            nodes.push(SceneNode::Polyline { points: pixel_pts, style: stroke_style.clone() });
            data_indices.push(row.row_idx);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
    use crate::scale::linear::LinearScale;
    use crate::spec::chart::ChartSpec;
    use crate::spec::coord::{CoordKind, GeoProjection};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Minimal geoshape spec with Mercator projection and no encodings.
    fn geo_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Geoshape,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(CoordKind::Geo { projection: GeoProjection::Equirectangular }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    /// 100×100 plot panel.
    fn rect_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Dummy resolved scales — geoshape ignores them entirely, but DrawCtx requires one.
    fn dummy_scales() -> ResolvedScales {
        ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0],
                vec![0.0, 100.0],
                false,
                false,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0],
                vec![100.0, 0.0],
                false,
                false,
            )),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            x2: None,
            y2: None,
        }
    }

    /// Build a single-row RecordBatch whose `__geometry__` column holds one GeoJSON string.
    fn batch_with_geometry(geojson: &str) -> arrow::record_batch::RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("__geometry__", DataType::Utf8, true),
        ]));
        arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some(geojson)]))],
        )
        .unwrap()
    }

    // ── Test 1: Point geometry emits at least one Circle ─────────────────

    #[test]
    fn point_geometry_emits_circle() {
        let geojson = r#"{"type":"Point","coordinates":[-87.65,41.85]}"#;
        let spec = geo_spec();
        let batch = batch_with_geometry(geojson);
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = dummy_scales();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Geoshape);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = build(&ctx);
        let circle_count = result
            .nodes
            .iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Circle { .. }))
            .count();
        assert!(
            circle_count >= 1,
            "Point geometry must emit at least one Circle, got {circle_count}"
        );
    }

    // ── Test 2: LineString geometry emits at least one Polyline ──────────

    #[test]
    fn linestring_geometry_emits_polyline() {
        let geojson =
            r#"{"type":"LineString","coordinates":[[-87.0,41.0],[-88.0,42.0],[-89.0,43.0]]}"#;
        let spec = geo_spec();
        let batch = batch_with_geometry(geojson);
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = dummy_scales();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Geoshape);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = build(&ctx);
        let polyline_count = result
            .nodes
            .iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Polyline { .. }))
            .count();
        assert!(
            polyline_count >= 1,
            "LineString geometry must emit at least one Polyline, got {polyline_count}"
        );
    }

    // ── Test 3: Simple polygon (no holes) produces exactly one ring ───────

    #[test]
    fn polygon_no_holes_has_one_ring() {
        // A square in lon/lat space — no interior rings.
        let geojson = r#"{"type":"Polygon","coordinates":[[[-87.0,41.0],[-86.0,41.0],[-86.0,42.0],[-87.0,42.0],[-87.0,41.0]]]}"#;
        let spec = geo_spec();
        let batch = batch_with_geometry(geojson);
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = dummy_scales();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Geoshape);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = build(&ctx);
        // Find the first Polygon node and check its ring count.
        let ring_count = result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Polygon { rings, .. } = n {
                Some(rings.len())
            } else {
                None
            }
        });
        assert_eq!(
            ring_count,
            Some(1),
            "Simple polygon (no holes) must produce exactly 1 ring, got {ring_count:?}"
        );
    }

    // ── Test 4: Polygon with one hole produces two rings ─────────────────
    //
    // This is the regression guard: the old code called `rings.first()` and
    // discarded all interior rings, producing `rings.len() == 1` even when a
    // hole was present. After the fix the exterior ring and the hole ring are
    // both preserved, so `rings.len() == 2`.

    #[test]
    fn polygon_with_hole_has_two_rings() {
        // Exterior: large square. Interior (hole): small square inside it.
        let geojson = r#"{
            "type": "Polygon",
            "coordinates": [
                [[-90.0,40.0],[-80.0,40.0],[-80.0,50.0],[-90.0,50.0],[-90.0,40.0]],
                [[-87.0,43.0],[-83.0,43.0],[-83.0,47.0],[-87.0,47.0],[-87.0,43.0]]
            ]
        }"#;
        let spec = geo_spec();
        let batch = batch_with_geometry(geojson);
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = dummy_scales();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Geoshape);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = build(&ctx);
        let ring_count = result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Polygon { rings, .. } = n {
                Some(rings.len())
            } else {
                None
            }
        });
        assert_eq!(
            ring_count,
            Some(2),
            "Polygon with one hole must produce 2 rings (exterior + hole), got {ring_count:?}. \
             This fails if the renderer reverts to `rings.first()` and discards interior rings."
        );
    }
}

// ── Geometry collection helpers ───────────────────────────────────────

/// Recursively collect all geometry from a GeoJSON `Geometry` into `row`.
/// All coordinates are returned in projection space (not yet pixel-scaled).
fn collect_geometry(
    geom: &Geometry,
    project_ring: &impl Fn(&[Vec<f64>]) -> Vec<[f64; 2]>,
    project: &impl Fn(f64, f64) -> Option<[f64; 2]>,
    row: &mut RowGeometry,
) {
    match &geom.value {
        GeoValue::Polygon(rings) => {
            // rings[0] = exterior, rings[1..] = holes.
            let poly_rings: Vec<Vec<[f64; 2]>> = rings
                .iter()
                .map(|r| project_ring(r))
                .filter(|r| !r.is_empty())
                .collect();
            if !poly_rings.is_empty() {
                row.polygons.push(poly_rings);
            }
        }
        GeoValue::MultiPolygon(polys) => {
            for rings in polys {
                let poly_rings: Vec<Vec<[f64; 2]>> = rings
                    .iter()
                    .map(|r| project_ring(r))
                    .filter(|r| !r.is_empty())
                    .collect();
                if !poly_rings.is_empty() {
                    row.polygons.push(poly_rings);
                }
            }
        }
        GeoValue::Point(pos) => {
            if pos.len() >= 2 {
                if let Some(pt) = project(pos[0], pos[1]) {
                    row.points.push(pt);
                }
            }
        }
        GeoValue::MultiPoint(positions) => {
            for pos in positions {
                if pos.len() >= 2 {
                    if let Some(pt) = project(pos[0], pos[1]) {
                        row.points.push(pt);
                    }
                }
            }
        }
        GeoValue::LineString(coords) => {
            let line: Vec<[f64; 2]> = coords
                .iter()
                .filter_map(|c| if c.len() >= 2 { project(c[0], c[1]) } else { None })
                .collect();
            if line.len() >= 2 {
                row.lines.push(line);
            }
        }
        GeoValue::MultiLineString(lines) => {
            for coords in lines {
                let line: Vec<[f64; 2]> = coords
                    .iter()
                    .filter_map(|c| if c.len() >= 2 { project(c[0], c[1]) } else { None })
                    .collect();
                if line.len() >= 2 {
                    row.lines.push(line);
                }
            }
        }
        GeoValue::GeometryCollection(geoms) => {
            for g in geoms {
                collect_geometry(g, project_ring, project, row);
            }
        }
    }
}
