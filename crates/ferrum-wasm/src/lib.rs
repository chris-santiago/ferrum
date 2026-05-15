#![deny(clippy::unwrap_used)]

// Pure-logic modules: compile on all targets (host tests use these).
pub mod conditional;
pub mod error;
pub mod hit_test;
pub mod scene_load;
pub mod selection_state;
pub mod tessellate;
pub mod transition;
pub mod zoom_pan;

// GPU / wasm-bindgen modules: only compile when targeting wasm32.
#[cfg(target_arch = "wasm32")]
pub mod gpu;
#[cfg(target_arch = "wasm32")]
pub mod pipelines;
#[cfg(target_arch = "wasm32")]
pub mod render;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

#[cfg(target_arch = "wasm32")]
use crate::conditional::resolve_conditionals;
#[cfg(target_arch = "wasm32")]
use crate::error::WasmRenderError;
#[cfg(target_arch = "wasm32")]
use crate::gpu::GpuContext;
#[cfg(target_arch = "wasm32")]
use crate::pipelines::RenderPipelines;
#[cfg(target_arch = "wasm32")]
use crate::render::{GpuBuffers, Uniforms};
#[cfg(target_arch = "wasm32")]
use crate::scene_load::SceneData;
#[cfg(target_arch = "wasm32")]
use crate::selection_state::InteractionState;
#[cfg(target_arch = "wasm32")]
use crate::transition::{ease_in_out_cubic, lerp_circles, lerp_rects};
#[cfg(target_arch = "wasm32")]
use crate::zoom_pan::{ScaleMode, ZoomPanState};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmRenderer {
    gpu: GpuContext,
    pipelines: RenderPipelines,
    loaded: Option<LoadedScene>,
    transition: Option<ActiveTransition>,
    selections: Vec<ferrum_scene::SelectionSpec>,
    interaction_state: InteractionState,
    zoom: ZoomPanState,
    interaction: ferrum_scene::InteractionConfig,
}

#[cfg(target_arch = "wasm32")]
struct ActiveTransition {
    old_data: SceneData,
    new_data: SceneData,
    new_scene: ferrum_scene::SceneGraph,
}

#[cfg(target_arch = "wasm32")]
struct LoadedScene {
    data: SceneData,
    buffers: GpuBuffers,
    scene: ferrum_scene::SceneGraph,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmRenderer {
    #[wasm_bindgen(js_name = "create")]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmRenderer, JsValue> {
        console_error_panic_hook::set_once();
        let gpu = gpu::init_gpu(canvas).await.map_err(JsValue::from)?;
        let pipelines = RenderPipelines::new(&gpu.device, gpu.format);
        Ok(WasmRenderer {
            gpu,
            pipelines,
            loaded: None,
            transition: None,
            selections: Vec::new(),
            interaction_state: InteractionState::new(&[]),
            zoom: ZoomPanState::new(0, &ferrum_scene::InteractionConfig::default()),
            interaction: ferrum_scene::InteractionConfig::default(),
        })
    }

    #[wasm_bindgen(js_name = "loadScene")]
    pub fn load_scene(&mut self, scene_json: &str) -> Result<String, JsValue> {
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(scene_json)
            .map_err(|e| JsValue::from(WasmRenderError::SceneDeserialization(e.to_string())))?;

        let data = scene_load::load_scene(&scene);
        let text_json = build_text_json(&data);
        let buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &data);
        let clear_color = data.background;

        self.selections = scene.selections.clone();
        self.interaction_state = InteractionState::new(&self.selections);
        self.interaction = scene.interaction.clone();
        self.zoom = ZoomPanState::new(scene.panels.len(), &self.interaction);
        self.loaded = Some(LoadedScene { data, buffers, scene });

        if let Some(ref loaded) = self.loaded {
            render::render_frame(&self.gpu, &self.pipelines, &loaded.buffers, clear_color)
                .map_err(JsValue::from)?;
        }

        Ok(text_json)
    }

    #[wasm_bindgen(js_name = "renderFrame")]
    pub fn render_frame_js(&self) -> Result<(), JsValue> {
        if let Some(ref loaded) = self.loaded {
            render::render_frame(
                &self.gpu,
                &self.pipelines,
                &loaded.buffers,
                loaded.data.background,
            )
            .map_err(JsValue::from)?;
        }
        Ok(())
    }

    /// Begin a GPU-interpolated transition from the currently loaded scene to a
    /// new scene JSON string.
    ///
    /// Call ``tick_transition(t)`` (t ∈ [0, 1]) from a requestAnimationFrame loop
    /// to drive the animation.  ``start_transition`` does not start the loop —
    /// the JavaScript caller owns the timing.
    ///
    /// Returns `Ok(())` immediately (no-op) if no scene is currently loaded.
    #[wasm_bindgen(js_name = "startTransition")]
    pub fn start_transition(
        &mut self,
        new_scene_json: &str,
    ) -> Result<(), JsValue> {
        let Some(loaded) = &self.loaded else {
            return Ok(());
        };
        let old_data = loaded.data.clone();
        let new_scene: ferrum_scene::SceneGraph = serde_json::from_str(new_scene_json)
            .map_err(|e| JsValue::from(WasmRenderError::SceneDeserialization(e.to_string())))?;
        let new_data = scene_load::load_scene(&new_scene);
        self.transition = Some(ActiveTransition { old_data, new_data, new_scene });
        Ok(())
    }

    /// Advance the transition to fractional progress ``t`` ∈ [0, 1].
    ///
    /// Applies eased interpolation and re-renders the GPU frame.
    /// When ``t >= 1.0`` the transition state is cleared and the new scene
    /// is committed as the loaded scene.
    #[wasm_bindgen(js_name = "tickTransition")]
    pub fn tick_transition(&mut self, t: f32) -> Result<(), JsValue> {
        let t_eased = ease_in_out_cubic(t.clamp(0.0, 1.0));
        if let Some(ref tr) = self.transition {
            let lerped_circles = lerp_circles(&tr.old_data.circle_instances, &tr.new_data.circle_instances, t_eased);
            let lerped_rects = lerp_rects(&tr.old_data.rect_instances, &tr.new_data.rect_instances, t_eased);
            let lerped_data = SceneData {
                circle_instances: lerped_circles,
                rect_instances: lerped_rects,
                mesh_buffers: tr.new_data.mesh_buffers.clone(),
                text_elements: tr.new_data.text_elements.clone(),
                image_quads: tr.new_data.image_quads.clone(),
                background: tr.new_data.background,
                width: tr.new_data.width,
                height: tr.new_data.height,
            };
            let buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &lerped_data);
            render::render_frame(&self.gpu, &self.pipelines, &buffers, lerped_data.background)
                .map_err(JsValue::from)?;
            if t >= 1.0 {
                let final_data = tr.new_data.clone();
                let final_buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &final_data);
                let final_scene = tr.new_scene.clone();
                self.transition = None;
                self.loaded = Some(LoadedScene { data: final_data, buffers: final_buffers, scene: final_scene });
            }
        }
        Ok(())
    }

    /// Hit-test a click at canvas pixel (x, y), update selection state, apply
    /// conditional encodings (dim non-selected marks), re-render frame, and
    /// return the new selection state as a JSON string.
    ///
    /// The returned JSON is a map of `selection_name → {field_name: field_value}`.
    /// The JS caller should forward this to `model.set('selection_state', ...)`.
    #[wasm_bindgen(js_name = "handleClick")]
    pub fn handle_click(&mut self, x: f32, y: f32) -> Result<String, JsValue> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Ok("{}".to_string());
        };

        // Update selection state via Rust hit-test (authoritative — operates on actual scene data).
        self.interaction_state.handle_click(
            &loaded.scene.panels,
            &self.selections,
            x as f64,
            y as f64,
            &self.zoom,
        );

        // Apply conditional encodings to produce dimmed/highlighted instance colors.
        let conditionals = &loaded.scene.interaction.conditionals;
        let updates = resolve_conditionals(
            &loaded.scene.panels,
            conditionals,
            &self.interaction_state.selections,
            &loaded.data.circle_instances,
            &loaded.data.rect_instances,
        );

        // Rebuild GPU buffers with updated colors and re-render.
        let updated_data = SceneData {
            circle_instances: updates.circle_instances,
            rect_instances: updates.rect_instances,
            mesh_buffers: loaded.data.mesh_buffers.clone(),
            text_elements: loaded.data.text_elements.clone(),
            image_quads: loaded.data.image_quads.clone(),
            background: loaded.data.background,
            width: loaded.data.width,
            height: loaded.data.height,
        };
        let new_buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &updated_data);
        render::render_frame(&self.gpu, &self.pipelines, &new_buffers, updated_data.background)
            .map_err(JsValue::from)?;
        loaded.buffers = new_buffers;

        // Serialize current selection state for Python sync.
        let state_json = self.interaction_state.to_json();
        Ok(state_json)
    }

    /// Apply a wheel-zoom event on the given panel and re-render via GPU affine transform.
    ///
    /// Returns updated text-element JSON (tick labels at new positions) so the JS
    /// overlay can reposition axis labels without a Python round-trip.
    #[wasm_bindgen(js_name = "onWheel")]
    pub fn on_wheel(&mut self, panel_id: u32, delta_y: f32, cx: f32, cy: f32) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        let coord = loaded.scene.panels.get(panel_id as usize).map(|p| &p.coord);
        // Polar and Geo panels have no affine-transform-compatible coordinate space.
        if matches!(coord, Some(ferrum_scene::CoordKind::Polar { .. })) {
            return Ok("[]".to_string());
        }
        let scale_mode = match coord {
            Some(ferrum_scene::CoordKind::Fixed { .. }) => ScaleMode::Uniform,
            _ => ScaleMode::Independent,
        };
        self.zoom.on_wheel(panel_id as usize, delta_y as f64, cx as f64, cy as f64, scale_mode);
        self.upload_transform_and_render(panel_id as usize)
    }

    /// Apply a pan delta on the given panel and re-render via GPU affine transform.
    ///
    /// Returns updated text-element JSON.
    #[wasm_bindgen(js_name = "onPan")]
    pub fn on_pan(&mut self, panel_id: u32, dx: f32, dy: f32) -> Result<String, JsValue> {
        let Some(_loaded) = &self.loaded else { return Ok("[]".to_string()); };
        self.zoom.on_pan(panel_id as usize, dx as f64, dy as f64);
        self.upload_transform_and_render(panel_id as usize)
    }

    /// Reset zoom/pan to identity for the given panel and re-render.
    ///
    /// Returns text-element JSON with tick labels at their original positions.
    #[wasm_bindgen(js_name = "resetZoom")]
    pub fn reset_zoom(&mut self, panel_id: u32) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        self.zoom.reset(panel_id as usize);
        let uniforms = Uniforms::identity(loaded.data.width, loaded.data.height);
        loaded.buffers.upload_uniforms(&self.gpu, &uniforms);
        render::render_frame(&self.gpu, &self.pipelines, &loaded.buffers, loaded.data.background)
            .map_err(JsValue::from)?;
        Ok(build_text_json(&loaded.data))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.config.width = width.max(1);
        self.gpu.config.height = height.max(1);
        self.gpu
            .surface
            .configure(&self.gpu.device, &self.gpu.config);
        let _ = self.render_frame_js();
    }
}

/// Upload the per-panel affine transform uniform and re-render, then return zoomed text JSON.
#[cfg(target_arch = "wasm32")]
impl WasmRenderer {
    fn upload_transform_and_render(&mut self, panel_id: usize) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        let transform = self.zoom.transforms.get(panel_id)
            .copied()
            .unwrap_or_else(crate::zoom_pan::Affine2::identity);
        let uniforms = Uniforms {
            canvas_w: loaded.data.width,
            canvas_h: loaded.data.height,
            _canvas_pad: [0.0; 2],
            sx: transform.sx as f32,
            sy: transform.sy as f32,
            tx: transform.tx as f32,
            ty: transform.ty as f32,
            clip_x: 0.0,
            clip_y: 0.0,
            clip_w: loaded.data.width,
            clip_h: loaded.data.height,
        };
        loaded.buffers.upload_uniforms(&self.gpu, &uniforms);
        render::render_frame(&self.gpu, &self.pipelines, &loaded.buffers, loaded.data.background)
            .map_err(JsValue::from)?;
        let text_json = build_zoomed_text_json(&loaded.data.text_elements, &self.interaction, panel_id, &transform);
        Ok(text_json)
    }
}

#[cfg(target_arch = "wasm32")]
fn build_text_json(data: &SceneData) -> String {
    let elements: Vec<serde_json::Value> = data
        .text_elements
        .iter()
        .map(|t| {
            serde_json::json!({
                "x": t.x,
                "y": t.y,
                "content": t.content,
                "fontSize": t.style.font_size,
                "fontWeight": match &t.style.font_weight {
                    ferrum_scene::FontWeight::Normal => "normal".to_string(),
                    ferrum_scene::FontWeight::Bold => "bold".to_string(),
                    ferrum_scene::FontWeight::Custom(s) => s.clone(),
                },
                "fontFamily": t.style.font_family,
                "anchor": match t.style.anchor {
                    ferrum_scene::TextAnchor::Start => "start",
                    ferrum_scene::TextAnchor::Middle => "center",
                    ferrum_scene::TextAnchor::End => "end",
                },
                "baseline": match &t.style.baseline {
                    ferrum_scene::TextBaseline::Top => "top".to_string(),
                    ferrum_scene::TextBaseline::Middle => "middle".to_string(),
                    ferrum_scene::TextBaseline::Bottom => "bottom".to_string(),
                    ferrum_scene::TextBaseline::Alphabetic => "alphabetic".to_string(),
                    ferrum_scene::TextBaseline::Custom(s) => s.clone(),
                },
                "angle": t.style.angle,
                "color": format!("rgba({},{},{},{})",
                    t.style.color.r, t.style.color.g, t.style.color.b,
                    t.style.opacity),
            })
        })
        .collect();
    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

/// Build text-element JSON for a zoomed panel.
///
/// Axis tick labels are identified by clustering text elements that share the
/// same y coordinate (x-axis row) or same x coordinate (y-axis column) and
/// whose content appears in the known tick-label set.  Each identified tick
/// label is repositioned by applying the affine zoom transform to its varying
/// coordinate (x for x-axis ticks, y for y-axis ticks).  All other text
/// (chart title, axis title, legend) is emitted at its original position.
///
/// This replaces the old pixel-match approach, which compared `tick_data`
/// scale-function outputs against axis text positions.  Those values differ
/// because the axis layout uses uniform band centering (`plot_area.x +
/// (i+0.5)*slot_w`) while `tick_data` uses the actual scale function — a
/// systematically different mapping that never matches.
#[cfg(target_arch = "wasm32")]
fn build_zoomed_text_json(
    all_text: &[crate::scene_load::TextElementData],
    interaction: &ferrum_scene::InteractionConfig,
    panel_id: usize,
    transform: &crate::zoom_pan::Affine2,
) -> String {
    use std::collections::{HashMap, HashSet};

    let Some(ptl) = interaction.tick_levels.iter().find(|p| p.panel_id == panel_id) else {
        return build_text_json_from(all_text);
    };

    // Union of tick label strings across all zoom levels.
    let x_tick_labels: HashSet<&str> = ptl.x_levels.iter()
        .flat_map(|lvl| lvl.ticks.iter().map(|t| t.label.as_str()))
        .collect();
    let y_tick_labels: HashSet<&str> = ptl.y_levels.iter()
        .flat_map(|lvl| lvl.ticks.iter().map(|t| t.label.as_str()))
        .collect();

    // --- Identify x-axis tick row ------------------------------------------
    // All x-axis tick labels share the same y coordinate.  Find the most
    // common rounded-y among elements whose content is a known x-tick label.
    let mut x_y_freq: HashMap<i64, usize> = HashMap::new();
    for te in all_text.iter().filter(|te| x_tick_labels.contains(te.content.as_str())) {
        *x_y_freq.entry((te.y * 10.0) as i64).or_insert(0) += 1;
    }
    let x_axis_y: Option<f64> = x_y_freq.into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k as f64 / 10.0);

    // --- Identify y-axis tick column ----------------------------------------
    // All y-axis tick labels share the same x coordinate.
    let mut y_x_freq: HashMap<i64, usize> = HashMap::new();
    for te in all_text.iter().filter(|te| y_tick_labels.contains(te.content.as_str())) {
        *y_x_freq.entry((te.x * 10.0) as i64).or_insert(0) += 1;
    }
    let y_axis_x: Option<f64> = y_x_freq.into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k as f64 / 10.0);

    // 1 px tolerance covers label_font_size / 3.0 baseline offset and rounding.
    const COORD_TOL: f64 = 1.5;

    let is_x_tick = |te: &crate::scene_load::TextElementData| {
        x_tick_labels.contains(te.content.as_str())
            && x_axis_y.map(|ay| (te.y - ay).abs() < COORD_TOL).unwrap_or(false)
    };
    let is_y_tick = |te: &crate::scene_load::TextElementData| {
        y_tick_labels.contains(te.content.as_str())
            && y_axis_x.map(|ax| (te.x - ax).abs() < COORD_TOL).unwrap_or(false)
    };

    let mut elements: Vec<serde_json::Value> = Vec::new();
    for te in all_text {
        if is_x_tick(te) {
            // Apply sx + tx to the x coordinate; y stays at the axis level.
            let new_x = te.x * transform.sx + transform.tx;
            elements.push(tick_label_json(new_x, te.y, &te.content, "center", Some(&te.style)));
        } else if is_y_tick(te) {
            // Apply sy + ty to the y coordinate; x stays at the axis level.
            let new_y = te.y * transform.sy + transform.ty;
            elements.push(tick_label_json(te.x, new_y, &te.content, "end", Some(&te.style)));
        } else {
            elements.push(text_element_to_json(te));
        }
    }

    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(target_arch = "wasm32")]
fn build_text_json_from(all_text: &[crate::scene_load::TextElementData]) -> String {
    let elements: Vec<serde_json::Value> = all_text.iter().map(text_element_to_json).collect();
    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(target_arch = "wasm32")]
fn text_element_to_json(t: &crate::scene_load::TextElementData) -> serde_json::Value {
    serde_json::json!({
        "x": t.x,
        "y": t.y,
        "content": t.content,
        "fontSize": t.style.font_size,
        "fontWeight": match &t.style.font_weight {
            ferrum_scene::FontWeight::Normal => "normal".to_string(),
            ferrum_scene::FontWeight::Bold => "bold".to_string(),
            ferrum_scene::FontWeight::Custom(s) => s.clone(),
        },
        "fontFamily": t.style.font_family,
        "anchor": match t.style.anchor {
            ferrum_scene::TextAnchor::Start => "start",
            ferrum_scene::TextAnchor::Middle => "center",
            ferrum_scene::TextAnchor::End => "end",
        },
        "baseline": match &t.style.baseline {
            ferrum_scene::TextBaseline::Top => "top".to_string(),
            ferrum_scene::TextBaseline::Middle => "middle".to_string(),
            ferrum_scene::TextBaseline::Bottom => "bottom".to_string(),
            ferrum_scene::TextBaseline::Alphabetic => "alphabetic".to_string(),
            ferrum_scene::TextBaseline::Custom(s) => s.clone(),
        },
        "angle": t.style.angle,
        "color": format!("rgba({},{},{},{})",
            t.style.color.r, t.style.color.g, t.style.color.b,
            t.style.opacity),
    })
}

#[cfg(target_arch = "wasm32")]
fn tick_label_json(
    x: f64, y: f64, label: &str, anchor: &str,
    style: Option<&ferrum_scene::TextStyle>,
) -> serde_json::Value {
    let (font_size, font_weight, font_family, baseline, color) = match style {
        Some(s) => (
            s.font_size,
            match &s.font_weight {
                ferrum_scene::FontWeight::Normal => "normal".to_string(),
                ferrum_scene::FontWeight::Bold => "bold".to_string(),
                ferrum_scene::FontWeight::Custom(v) => v.clone(),
            },
            s.font_family.clone(),
            match &s.baseline {
                ferrum_scene::TextBaseline::Top => "top".to_string(),
                ferrum_scene::TextBaseline::Middle => "middle".to_string(),
                ferrum_scene::TextBaseline::Bottom => "bottom".to_string(),
                ferrum_scene::TextBaseline::Alphabetic => "alphabetic".to_string(),
                ferrum_scene::TextBaseline::Custom(v) => v.clone(),
            },
            format!("rgba({},{},{},{})", s.color.r, s.color.g, s.color.b, s.opacity),
        ),
        None => (11.0, "normal".to_string(), "sans-serif".to_string(),
                 "alphabetic".to_string(), "rgba(51,51,51,1)".to_string()),
    };
    serde_json::json!({
        "x": x, "y": y, "content": label,
        "fontSize": font_size, "fontWeight": font_weight,
        "fontFamily": font_family, "anchor": anchor,
        "baseline": baseline, "angle": 0.0, "color": color,
    })
}
