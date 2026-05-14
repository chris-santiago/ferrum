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

#[wasm_bindgen]
pub struct WasmRenderer {
    gpu: GpuContext,
    pipelines: RenderPipelines,
    loaded: Option<LoadedScene>,
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
