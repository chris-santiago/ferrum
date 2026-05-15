#![deny(clippy::unwrap_used)]

pub mod conditional;
pub mod error;
pub mod gpu;
pub mod hit_test;
pub mod pipelines;
pub mod render;
pub mod scene_load;
pub mod selection_state;
pub mod tessellate;
pub mod transition;
pub mod zoom_pan;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::error::WasmRenderError;
use crate::gpu::GpuContext;
use crate::pipelines::RenderPipelines;
use crate::render::GpuBuffers;
use crate::scene_load::SceneData;
use crate::transition::{ease_in_out_cubic, lerp_circles, lerp_rects};

#[wasm_bindgen]
pub struct WasmRenderer {
    gpu: GpuContext,
    pipelines: RenderPipelines,
    loaded: Option<LoadedScene>,
    transition: Option<ActiveTransition>,
}

struct ActiveTransition {
    old_data: SceneData,
    new_data: SceneData,
}

struct LoadedScene {
    data: SceneData,
    buffers: GpuBuffers,
}

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

        self.loaded = Some(LoadedScene { data, buffers });

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
        self.transition = Some(ActiveTransition { old_data, new_data });
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
                self.transition = None;
                let final_buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &final_data);
                self.loaded = Some(LoadedScene { data: final_data, buffers: final_buffers });
            }
        }
        Ok(())
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
