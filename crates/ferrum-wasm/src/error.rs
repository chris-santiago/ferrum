#![allow(clippy::enum_variant_names)]

use wasm_bindgen::JsValue;

#[derive(Debug)]
pub enum WasmRenderError {
    GpuInit(String),
    ContextLost,
    SceneDeserialization(String),
    TextureUpload(String),
    ShaderCompilation(String),
}

impl std::fmt::Display for WasmRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuInit(msg) => write!(f, "GPU init failed: {msg}"),
            Self::ContextLost => write!(f, "GPU context lost"),
            Self::SceneDeserialization(msg) => write!(f, "scene deserialization failed: {msg}"),
            Self::TextureUpload(msg) => write!(f, "texture upload failed: {msg}"),
            Self::ShaderCompilation(msg) => write!(f, "shader compilation failed: {msg}"),
        }
    }
}

impl From<WasmRenderError> for JsValue {
    fn from(e: WasmRenderError) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
}
