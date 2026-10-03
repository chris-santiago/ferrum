#![allow(clippy::enum_variant_names)]

use wasm_bindgen::JsValue;

#[derive(Debug)]
pub enum WasmRenderError {
    GpuInit(String),
    ContextLost,
    SceneDeserialization(String),
    TextureUpload(String),
    ShaderCompilation(String),
    /// A tapered stroke whose `widths` the shared taper validation refuses.
    TaperedStroke(ferrum_scene::TaperError),
}

impl std::fmt::Display for WasmRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuInit(msg) => write!(f, "GPU init failed: {msg}"),
            Self::ContextLost => write!(f, "GPU context lost"),
            Self::SceneDeserialization(msg) => write!(f, "scene deserialization failed: {msg}"),
            Self::TextureUpload(msg) => write!(f, "texture upload failed: {msg}"),
            Self::ShaderCompilation(msg) => write!(f, "shader compilation failed: {msg}"),
            Self::TaperedStroke(e) => write!(f, "mark_line: the tapered line cannot be drawn: {e}"),
        }
    }
}

impl From<ferrum_scene::TaperError> for WasmRenderError {
    fn from(e: ferrum_scene::TaperError) -> Self {
        Self::TaperedStroke(e)
    }
}

impl From<WasmRenderError> for JsValue {
    fn from(e: WasmRenderError) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
}
