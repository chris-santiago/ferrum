use ferrum_scene::*;
use lyon::tessellation::VertexBuffers;

use crate::tessellate::{self, MeshVertex};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleInstance {
    pub center: [f32; 2],
    pub radius: f32,
    pub fill_color: [f32; 4],
    pub stroke_color: [f32; 4],
    pub stroke_width: f32,
    pub opacity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectInstance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub corner_radius: f32,
    pub fill_color: [f32; 4],
    pub stroke_color: [f32; 4],
    pub stroke_width: f32,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct SceneData {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
    pub mesh_buffers: VertexBuffers<MeshVertex, u32>,
    pub text_elements: Vec<TextElementData>,
    pub image_quads: Vec<ImageQuad>,
    pub background: Option<[f32; 4]>,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone)]
pub struct TextElementData {
    pub x: f64,
    pub y: f64,
    pub content: String,
    pub style: TextStyle,
}

#[derive(Clone)]
pub struct ImageQuad {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub data: Vec<u8>,
    pub img_width: u32,
    pub img_height: u32,
}

pub fn load_scene(scene: &SceneGraph) -> SceneData {
    let mut circles = Vec::new();
    let mut rects = Vec::new();
    let mut mesh = VertexBuffers::new();
    let mut images = Vec::new();
    let mut texts = Vec::new();

    let background = scene.background.as_ref().map(|c| {
        [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        ]
    });

    collect_nodes(&scene.title, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);

    for panel in &scene.panels {
        collect_nodes(&panel.grid, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);

        for batch in &panel.marks {
            collect_nodes(
                &batch.nodes,
                &mut circles, &mut rects, &mut mesh, &mut texts, &mut images,
                batch.stroke_cap, batch.stroke_join,
            );
        }

        collect_nodes(&panel.axes, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
        collect_nodes(&panel.strip_title, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
        collect_nodes(&panel.annotations, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
    }

    collect_nodes(&scene.legend, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
    collect_nodes(&scene.decorations, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);

    SceneData {
        circle_instances: circles,
        rect_instances: rects,
        mesh_buffers: mesh,
        text_elements: texts,
        image_quads: images,
        background,
        width: scene.width as f32,
        height: scene.height as f32,
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_nodes(
    nodes: &[SceneNode],
    circles: &mut Vec<CircleInstance>,
    rects: &mut Vec<RectInstance>,
    mesh: &mut VertexBuffers<MeshVertex, u32>,
    texts: &mut Vec<TextElementData>,
    images: &mut Vec<ImageQuad>,
    batch_cap: Option<StrokeCap>,
    batch_join: Option<StrokeJoin>,
) {
    for node in nodes {
        match node {
            SceneNode::Circle { cx, cy, r, style } => {
                circles.push(CircleInstance {
                    center: [*cx as f32, *cy as f32],
                    radius: *r as f32,
                    fill_color: opt_color_to_f32(style.fill.as_ref(), style.opacity),
                    stroke_color: opt_color_to_f32(style.stroke.as_ref(), style.opacity),
                    stroke_width: style.stroke_width as f32,
                    opacity: style.opacity as f32,
                });
            }
            SceneNode::Rect { x, y, w, h, style, corner_radius } => {
                rects.push(RectInstance {
                    position: [*x as f32, *y as f32],
                    size: [*w as f32, *h as f32],
                    corner_radius: *corner_radius as f32,
                    fill_color: opt_color_to_f32(style.fill.as_ref(), style.opacity),
                    stroke_color: opt_color_to_f32(style.stroke.as_ref(), style.opacity),
                    stroke_width: style.stroke_width as f32,
                    opacity: style.opacity as f32,
                });
            }
            SceneNode::Line { x1, y1, x2, y2, style } => {
                let mut s = style.clone();
                if s.stroke_cap.is_none() { s.stroke_cap = batch_cap; }
                if s.stroke_join.is_none() { s.stroke_join = batch_join; }
                tessellate::tessellate_line(*x1, *y1, *x2, *y2, &s, mesh);
            }
            SceneNode::Path { commands, style, closed } => {
                tessellate::tessellate_path(commands, style, *closed, batch_cap, batch_join, mesh);
            }
            SceneNode::Polyline { points, style } => {
                let mut s = style.clone();
                if s.stroke_cap.is_none() { s.stroke_cap = batch_cap; }
                if s.stroke_join.is_none() { s.stroke_join = batch_join; }
                tessellate::tessellate_polyline(points, &s, mesh);
            }
            SceneNode::Polygon { rings, style } => {
                tessellate::tessellate_polygon(rings, style, mesh);
            }
            SceneNode::Text { x, y, content, style } => {
                texts.push(TextElementData {
                    x: *x,
                    y: *y,
                    content: content.clone(),
                    style: style.clone(),
                });
            }
            SceneNode::Image { x, y, w, h, data } => {
                if let ImageData::Inline { bytes, .. } = data {
                    if let Some(quad) = decode_image_quad(*x, *y, *w, *h, bytes) {
                        images.push(quad);
                    }
                }
            }
            SceneNode::Group { children, .. } => {
                collect_nodes(children, circles, rects, mesh, texts, images, batch_cap, batch_join);
            }
            SceneNode::Raw { .. } => {
                web_sys::console::warn_1(
                    &"ferrum: Raw SVG node skipped in WASM renderer".into(),
                );
            }
        }
    }
}

fn decode_image_quad(x: f64, y: f64, w: f64, h: f64, png_bytes: &[u8]) -> Option<ImageQuad> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(buf.len() / 3 * 4);
            for chunk in buf.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgba = Vec::with_capacity(buf.len() / 2 * 4);
            for chunk in buf.chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            rgba
        }
        png::ColorType::Grayscale => {
            let mut rgba = Vec::with_capacity(buf.len() * 4);
            for &g in &buf {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
            rgba
        }
        _ => return None,
    };

    Some(ImageQuad {
        x: x as f32,
        y: y as f32,
        w: w as f32,
        h: h as f32,
        data: rgba,
        img_width: info.width,
        img_height: info.height,
    })
}

fn opt_color_to_f32(color: Option<&Color>, opacity: f64) -> [f32; 4] {
    match color {
        Some(c) => [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            (c.a as f32 / 255.0) * opacity as f32,
        ],
        None => [0.0, 0.0, 0.0, 0.0],
    }
}
